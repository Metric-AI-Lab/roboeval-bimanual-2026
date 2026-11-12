"""Generated-episode storage shared by all data generators (scripted expert, policy rollouts, ...).

One compressed .npz per episode: data/gen/<source>/<Variation>/ep_<seed>.npz, e.g.
data/gen/rollout_v1/StackTwoBlocksPositionAndOrientation/ep_200007.npz

Keys (T = number of control steps):
  rgb_head, rgb_left_wrist, rgb_right_wrist   uint8   [T, S, S, 3]  (rdb.common.resize_image, S=128)
  proprioception                              float32 [T, 36]
  proprioception_grippers                     float32 [T, 2]
  actions                                     float32 [T, 14]       dataset units (EE delta, grippers 0/1);
                                                                    the label to train on
  executed (optional)                         float32 [T, 14]       actions actually sent, when they differ
                                                                    from the labels (scripted expert + DART noise)
  meta                                        0-d str, JSON: source, family, variation, seed, success, ...

Convention: obs[t] is what the agent saw when it chose actions[t] (obs[0] = reset observation,
actions[0] = the no-op, zeros with grippers open, that every episode starts with, exactly like
the human demos and policy_bundle/policy/policy.py).

The released dataset stores the observation AFTER each action instead: dataset frame f holds
(obs after action f, action f), and rdb.data trains obs(frame f) -> actions[f+1:]. So a converter
must SHIFT: dataset frame f = (generated obs[f+1], generated actions[f]) for f = 0 .. T-2, i.e.
drop obs[0] (the policy never queries it) and the last action's missing next observation.
`dataset_frames` does exactly that; after the shift rdb.data's rule becomes obs[t] -> actions[t:],
the same as at deployment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from rdb.common import CAMERAS, resize_image

IMG_SIZE = 128

# The interface in policy_bundle/metadata.yaml: EE delta, 20 Hz, 3 RGB cams at 256.
INTERFACE = {
    "action_mode": {"absolute": False, "ee": True},
    "control_frequency": 20,
    "observation": {"proprioception": True,
                    "cameras": [{"name": c, "resolution": [256, 256], "rgb": True, "depth": False} for c in CAMERAS]},
}


def episode_path(root: Path, source: str, variation: str, seed: int) -> Path:
    return Path(root) / source / variation / f"ep_{seed}.npz"


def variant_name(family: str, variation: str) -> str:
    """('StackTwoBlocks', 'Static') -> 'StackTwoBlocks'; otherwise family + variation (dataset folder names)."""
    return family if variation == "Static" else family + variation


def make_env(family: str, variation: str, render: bool = True):
    """The challenge env exactly as the evaluator builds it (cameras off for fast dry runs). Variations
    the challenge doesn't evaluate (Static, Position, Orientation) are built the same way from their
    class next to the evaluated one."""
    import importlib
    from unittest import mock
    from roboeval.challenge import spec
    from roboeval.challenge.submission import Interface, make_env as _make
    d = dict(INTERFACE, observation=dict(INTERFACE["observation"], cameras=INTERFACE["observation"]["cameras"] if render else []))
    iface = Interface.from_dict(d)
    if variation in spec.FAMILIES_BY_KEY[family].variations:
        return _make(family, variation, iface)
    module = spec.FAMILIES_BY_KEY[family].variations["PositionAndOrientation"].split(":")[0]
    cls = getattr(importlib.import_module(module), variant_name(family, variation))
    with mock.patch.object(spec, "env_class", lambda f, v: cls):
        return _make(family, variation, iface)


def obs_frame(obs: Dict[str, np.ndarray], size: int = IMG_SIZE) -> Dict[str, np.ndarray]:
    """One env observation (rgb_<cam> as CHW uint8) -> the stored per-step arrays (cameras optional)."""
    out = {f"rgb_{c}": resize_image(np.ascontiguousarray(obs[f"rgb_{c}"].transpose(1, 2, 0)), size)
           for c in CAMERAS if f"rgb_{c}" in obs}
    out["proprioception"] = np.asarray(obs["proprioception"], np.float32).copy()
    out["proprioception_grippers"] = np.asarray(obs["proprioception_grippers"], np.float32).copy()
    return out


def save_episode(path: Path, frames: list, actions, meta: Dict[str, Any], executed=None) -> None:
    """frames: list of obs_frame() dicts, one per action. Written atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
    arrays["actions"] = np.asarray(actions, np.float32)
    assert len(arrays["actions"]) == len(frames)
    if executed is not None:
        arrays["executed"] = np.asarray(executed, np.float32)
        assert arrays["executed"].shape == arrays["actions"].shape
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def load_episode(path: Path) -> Dict[str, Any]:
    with np.load(path) as z:
        ep = {k: z[k] for k in z.files if k != "meta"}
        ep["meta"] = json.loads(str(z["meta"]))
    return ep


class EpisodeRecorder:
    """Call `add(obs, label, executed)` before every env.step(executed), with the obs the action was chosen from."""

    def __init__(self, size: int = IMG_SIZE):
        self.size = size
        self.frames: List[Dict[str, np.ndarray]] = []
        self.actions: List[np.ndarray] = []
        self.executed: List[np.ndarray] = []

    def add(self, obs: Dict[str, np.ndarray], label: np.ndarray, executed: np.ndarray | None = None) -> None:
        self.frames.append(obs_frame(obs, self.size))
        self.actions.append(np.asarray(label, np.float32))
        self.executed.append(np.asarray(label if executed is None else executed, np.float32))

    def __len__(self) -> int:
        return len(self.actions)

    def save(self, path: Path, meta: Dict[str, Any]) -> None:
        same = all(np.array_equal(a, e) for a, e in zip(self.actions, self.executed))
        save_episode(path, self.frames, self.actions, meta, executed=None if same else self.executed)


def dataset_frames(ep: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Episode -> arrays in the released dataset's alignment (see module docstring), rdb.prep layout:
    img [T-1, n_cams, S, S, 3], state [T-1, 36], grip [T-1, 2], action [T-1, 14]."""
    return {
        "img": np.stack([ep[f"rgb_{c}"][1:] for c in CAMERAS], 1),
        "state": ep["proprioception"][1:],
        "grip": ep["proprioception_grippers"][1:],
        "action": ep["actions"][:-1],
    }
