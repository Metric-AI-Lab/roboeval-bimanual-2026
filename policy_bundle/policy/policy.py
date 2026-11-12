"""RoboEval CoRL 2026 submission policy (copied to submission/policy/policy.py).

One multi-task chunk policy for all 8 families, conditioned on the family index from
``reset(episode)``. Inference options live in checkpoint/infer.json:

  mode            "ensemble": query every step, ACT-style temporal ensembling
                  "chunk": execute ``exec_horizon`` actions per query
  ensemble_k      ensembling weight decay (w_i = exp(-k * i), i = 0 is the oldest)
  smooth_alpha    previous executed arm action's EMA weight in [0, 1); 0 disables
                  smoothing, grippers untouched, history resets each episode
  exec_horizon    actions executed per query in "chunk" mode
  flow_steps      Euler steps of the flow sampler (default: the checkpoint's)
  n_samples, agg  draw n chunks per query, combine by "mean" or "medoid" (the sample closest to the
                  others; avoids averaging two modes)
  grasp_pause     hold both arms still (zero EE deltas) for this many steps after a gripper closes, so the
                  grasp settles before the lift; 0 disables
  ensemble        extra checkpoint names (same cameras and image size); each query draws n_samples chunks from
                  every model and combines them all with "agg" (a multi-model ensemble)
  speed           multiply the arm EE deltas (not grippers) by this factor; >1 follows the same path in fewer steps
  per_family      {family: {overrides of any key above, incl. "ckpt"}}: per-task specialists
  hybrid          {"kind": "pot", "pose": "pose_pot.pt"}: pose hybrid instead of the learned policy; the pose
                  regressor reads the first observation (and every frame for moving books/rod/flaps) and an observation-only
                  controller (rdb/hybrid/*_controller.py) acts from then on; an optional
                  "controller" dict supplies fixed controller parameters
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from rdb.common import CAMERAS, FAMILIES, resize_image
from rdb.model import ChunkPolicy, PolicyConfig, GRIP_DIMS

DEFAULT_INFER = {"mode": "ensemble", "ensemble_k": 0.01, "exec_horizon": 8, "ckpt": "policy.pt",
                 "flow_steps": None, "n_samples": 1, "agg": "mean", "smooth_alpha": 0.0, "grasp_pause": 0, "speed": 1.0,
                 "ensemble": [], "hybrid": None}


class ACTRunner(torch.nn.Module):
    """An original-ACT checkpoint (rdb.act.train_act) behind ChunkPolicy's sample() interface, so the same
    chunk / temporal-ensembling code runs it. ACT is deterministic at inference (latent z = 0)."""

    def __init__(self, ck: Dict[str, Any]):
        super().__init__()
        from types import SimpleNamespace
        from rdb.act.vendor.modeling_act import ACT, ACTConfig
        c = {**ck["cfg"], "pretrained_backbone_weights": None, "cameras": tuple(ck["cfg"]["cameras"])}
        self.net = ACT(ACTConfig(**c))
        self.net.load_state_dict(ck["state_dict"])
        for k, v in ck["stats"].items():
            self.register_buffer(k, torch.tensor(v, dtype=torch.float32))
        self.register_buffer("im_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1))
        self.rel = ck.get("target") == "rel"   # trained on the arm path relative to the current pose
        self.cfg = SimpleNamespace(n_cams=len(c["cameras"]), img_size=ck["img_size"], flow_steps=None)

    @torch.no_grad()
    def sample(self, imgs, state, task):
        img = (imgs.float() / 255 - self.im_mean) / self.im_std
        st = (state.float() - self.state_mean) / self.state_std
        with torch.autocast(imgs.device.type, enabled=False):
            a = self.net({"observation.state": st, "observation.images": list(img.unbind(1))})[0]
        if not self.rel:
            return a * self.action_std + self.action_mean
        rel = a * self.rel_std + self.rel_mean
        arm = rel[..., :12]
        return torch.cat([torch.cat([arm[..., :1, :], arm[..., 1:, :] - arm[..., :-1, :]], -2), rel[..., 12:]], -1)


class RDBPolicy:
    def __init__(self, checkpoint_dir: Path, metadata: Dict[str, Any]):
        self.dir = Path(checkpoint_dir)
        infer_path = self.dir / "infer.json"
        self.base = {**DEFAULT_INFER, **(json.loads(infer_path.read_text()) if infer_path.exists() else {})}
        self.per_family = self.base.pop("per_family", {}) or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, Any] = {}
        for o in (self.base, *self.per_family.values()):
            for name in (o.get("ckpt", self.base["ckpt"]), *o.get("ensemble", [])):
                self._load(name)
        self.poses = {}
        for o in (self.base, *self.per_family.values()):
            if o.get("hybrid"):
                from rdb.hybrid.pose_infer import PoseEstimator
                name = o["hybrid"]["pose"]
                self.poses.setdefault(name, PoseEstimator(self.dir / name, self.device))
        self._select(self.base)

    def _load(self, name: str):
        if name not in self.models:
            ck = torch.load(self.dir / name, map_location="cpu", weights_only=False)
            if ck.get("kind") == "act":
                model = ACTRunner(ck)
                cfg = model.cfg
            else:
                cfg = PolicyConfig(**{**ck["cfg"], "pretrained_backbone": False})
                model = ChunkPolicy(cfg)
                model.load_state_dict(ck["state_dict"])
            self.models[name] = (model.to(self.device).eval(), cfg)
        return self.models[name]

    def _select(self, infer: Dict[str, Any]) -> None:
        alpha = float(infer["smooth_alpha"])
        if not 0 <= alpha < 1:
            raise ValueError("smooth_alpha must be in [0, 1)")
        self.infer = infer
        self.model, cfg = self._load(infer["ckpt"])
        self.extra = [self._load(n)[0] for n in infer["ensemble"]]
        self.cfg = cfg
        self.cams = CAMERAS[: cfg.n_cams]

    def reset(self, episode) -> None:
        self._select({**self.base, **self.per_family.get(episode.family, {})})
        self.task = torch.tensor([FAMILIES.index(episode.family)], device=self.device)
        self.t = 0
        self.queue = []          # chunk mode: pending actions
        self.preds = {}          # ensemble mode: step -> list of predicted actions (oldest first)
        self.previous_action = np.zeros(14, dtype=np.float32)  # first executed no-op
        self.prev_grip = np.zeros(2, dtype=np.float32)
        self.pause = 0           # steps left with the arms held still
        self.controller = None   # pose hybrid: built from the first observation

    def _hybrid_act(self, obs) -> np.ndarray:
        h = self.infer["hybrid"]
        if self.controller is None:
            pose = self.poses[h["pose"]](obs)
            if h["kind"] == "pot":
                from rdb.hybrid.pot_v13b import PotController
                x, y = pose["pot_xyz"][:2]
                self.controller = PotController((x, y, pose["pot_yaw"]), **h.get("controller", {}))
            elif h["kind"] == "valve":
                from rdb.hybrid.valve_controller import ValveController
                self.controller = ValveController([pose["hubA_xyz"], pose["hubB_xyz"]], **h.get("controller", {}))
            elif h["kind"] == "blocks":
                from rdb.hybrid.v13_blocks.blocks_controller import BlocksController
                self.controller = BlocksController([(pose[f"blk{c}_xyz"], pose[f"blk{c}_yaw"]) for c in "AB"],
                                                   **h.get("controller", {}))
            elif h["kind"] == "pick":
                from rdb.hybrid.pick_v13 import PickBookController
                self.controller = PickBookController(self._book(pose), **h.get("controller", {}))
            elif h["kind"] == "shelf":
                from rdb.hybrid.shelf_v13 import ShelfBookController
                self.shelf_pose = np.r_[pose["shelf_xyz"], pose["shelf_quat_wxyz"]]   # the shelf is fixed: reset estimate
                self.controller = ShelfBookController(self._book(pose), self.shelf_pose, **h.get("controller", {}))
            elif h["kind"] == "box":
                from rdb.hybrid.box_v13 import BoxController
                self.controller = BoxController((*pose["box_xyz"], pose["box_yaw"]), **h.get("controller", {}))
            elif h["kind"] == "handover":
                from rdb.hybrid.handover_controller import HandoverController
                self.controller = HandoverController(self._rod(pose), **h.get("controller", {}))
            else:
                raise ValueError(f"unknown hybrid kind {h['kind']}")
            if h["kind"] == "box":
                return np.asarray(self.controller.act(obs, self._flaps(pose)), dtype=np.float32)
            if h["kind"] in ("pick", "shelf"):
                return np.asarray(self.controller.act(obs, *self._book_args(pose)), dtype=np.float32)
            if h["kind"] == "handover":
                return np.asarray(self.controller.act(obs, self._rod(pose)), dtype=np.float32)
        elif h["kind"] in ("pick", "shelf"):   # the book moves: estimate its pose on every frame
            return np.asarray(self.controller.act(obs, *self._book_args(self.poses[h["pose"]](obs))), dtype=np.float32)
        elif h["kind"] == "box":                # the flaps move: estimate their angles on every frame
            return np.asarray(self.controller.act(obs, self._flaps(self.poses[h["pose"]](obs))), dtype=np.float32)
        elif h["kind"] == "handover":           # the rod moves, including while held
            return np.asarray(self.controller.act(obs, self._rod(self.poses[h["pose"]](obs))), dtype=np.float32)
        return np.asarray(self.controller.act(obs), dtype=np.float32)

    @staticmethod
    def _flaps(pose):
        return [float(np.ravel(pose[f"flap{i}_rem"])[0]) for i in (0, 1)]

    @staticmethod
    def _book(pose) -> np.ndarray:
        return np.r_[pose["book_xyz"], pose["book_quat_wxyz"]]

    @staticmethod
    def _rod(pose) -> np.ndarray:
        return np.r_[pose["rod_xyz"], pose["rod_quat_wxyz"]]

    def _book_args(self, pose):
        return (self._book(pose),) if self.infer["hybrid"]["kind"] == "pick" else (self._book(pose), self.shelf_pose)

    def _obs_tensors(self, obs: Dict[str, np.ndarray]):
        imgs = np.stack([resize_image(np.ascontiguousarray(obs[f"rgb_{c}"].transpose(1, 2, 0)), self.cfg.img_size)
                         .transpose(2, 0, 1) for c in self.cams])
        state = np.concatenate([obs["proprioception"], obs["proprioception_grippers"]]).astype(np.float32)
        return (torch.from_numpy(imgs)[None].to(self.device),
                torch.from_numpy(state)[None].to(self.device))

    def _predict(self, obs) -> np.ndarray:
        imgs, state = self._obs_tensors(obs)
        k = int(self.infer["n_samples"])
        steps = self.infer["flow_steps"]
        out = []
        for model in (self.model, *self.extra):
            saved = model.cfg.flow_steps
            if steps:
                model.cfg.flow_steps = int(steps)
            try:
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    out.append(model.sample(imgs.expand(k, *imgs.shape[1:]), state.expand(k, -1), self.task.expand(k)))
            finally:
                model.cfg.flow_steps = saved
        c = torch.cat(out).float().cpu().numpy()               # [k * models, T, 14]
        k = len(c)
        if k == 1:
            return c[0]
        if self.infer["agg"] == "medoid":
            d = ((c[:, None] - c[None]) ** 2).sum((2, 3))
            return c[int(d.sum(1).argmin())]
        return c.mean(0)

    def act(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        if self.infer["hybrid"]:
            return self._hybrid_act(observation)
        # Demonstrations start with a no-op (grippers open); every stored observation
        # follows an action, so the first query is made after this step.
        if self.t == 0:
            self.t += 1
            return np.zeros(14, dtype=np.float32)
        if self.infer["mode"] == "chunk":
            if not self.queue:
                self.queue = list(self._predict(observation)[: self.infer["exec_horizon"]])
            action = self.queue.pop(0)
        else:
            chunk = self._predict(observation)
            for i, a in enumerate(chunk):
                self.preds.setdefault(self.t + i, []).append(a)
            cands = np.stack(self.preds.pop(self.t))
            w = np.exp(-self.infer["ensemble_k"] * np.arange(len(cands)))
            action = (cands * w[:, None]).sum(0) / w.sum()
        action = np.asarray(action, dtype=np.float32).copy()
        alpha = float(self.infer["smooth_alpha"])
        if alpha:
            action[:12] = (1 - alpha) * action[:12] + alpha * self.previous_action[:12]
            self.previous_action = action.copy()
        action[:12] *= float(self.infer["speed"])   # EE deltas: >1 moves along the same path faster
        action[list(GRIP_DIMS)] = (action[list(GRIP_DIMS)] > 0.5).astype(np.float32)
        grip = action[list(GRIP_DIMS)]
        if (grip > self.prev_grip).any():
            self.pause = int(self.infer["grasp_pause"])
        self.prev_grip = grip.copy()
        if self.pause:
            action[:12] = 0
            self.pause -= 1
        self.t += 1
        return action


def make_policy(checkpoint_dir: Path, metadata: Dict[str, Any]) -> RDBPolicy:
    return RDBPolicy(checkpoint_dir, metadata)
