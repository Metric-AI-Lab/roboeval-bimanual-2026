"""PackBox per-frame pose data for the pose hybrid (privileged labels, training data only).

  MUJOCO_GL=egl python -m rdb.hybrid.box_pose_data --first-seed 2263000 --count 500 --out data/hybrid/box_traj/a

Runs rdb.gen.scripted_pack_box_h.PackBoxHumanExpert with DART noise (default 0.25, its own default) and saves every
2nd observation (three CHW 256x256 cameras, proprioception) with labels taken at the observation instant:
`box_xyz`, `box_yaw` (packing-box body) and `flap0_rem`, `flap1_rem` = q − lower limit of the flap that the left
(0) / right (1) arm folds (the teacher's assignment: left arm → flap with the larger hinge y at reset).
`--policy ctl` instead drives our observation-only rdb.hybrid.box_controller.BoxController with flap grasps (grasp
length drawn from 24–48 steps) on noisy privileged inputs (box xy 4 mm, yaw 0.03 rad, flap angles 0.04 rad), so the
frames include the grippers holding the flaps, which the teacher never does. `--pause` adds random per-arm pause windows
so one flap closes while the other stays open (seeds 2267000–2267999).
One .npz per episode (atomic write). Seeds 2263000-2264999.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from rdb.gen.episode_io import make_env
from rdb.gen.scripted_common import yaw_of
from rdb.gen.scripted_pack_box_h import PackBoxExpert, PackBoxHumanExpert

CAMERAS = ("head", "left_wrist", "right_wrist")


def frame(obs, env, expert, seed, step):
    out = {f"rgb_{c}": np.asarray(obs[f"rgb_{c}"]).copy() for c in CAMERAS}
    for k in ("proprioception", "proprioception_grippers"):
        out[k] = np.asarray(obs[k], np.float32).copy()
    body = env.packing_box.body
    out["box_xyz"] = np.asarray(body.get_position(), np.float32).copy()
    out["box_yaw"] = np.float32(yaw_of(body.get_quaternion()))
    for i in (0, 1):
        _, _, q, lo, _ = PackBoxExpert.flap(expert, i)
        out[f"flap{i}_rem"] = np.float32(q - lo)
    out["seed"], out["step"] = np.int64(seed), np.int64(step)
    return out


def episode(env, seed, noise, max_steps, stride):
    """As rdb.gen.scripted_common.run_episode: reset, the no-op, then the teacher with DART."""
    obs, _ = env.reset(seed=seed)
    obs, _, term, trunc, _ = env.step(np.zeros(14, np.float32))
    expert = PackBoxHumanExpert(env, np.random.default_rng(seed + 7), noise)
    frames, steps, status = [], 1, "budget"
    while steps < max_steps:
        if (steps - 1) % stride == 0:
            frames.append(frame(obs, env, expert, seed, steps))
        obs, _, term, trunc, _ = env.step(expert.perturb(expert.act()))
        steps += 1
        if env.success:
            status = "success"
            frames.append(frame(obs, env, expert, seed, steps))
            break
        if term or trunc:
            status = "fail"
            break
    return frames, status, steps


def paused_controller(pauses):
    """BoxController whose arm i holds its commanded pose during the step windows pauses[i] = (start, end), so one
    flap moves while the other waits (decorrelates the two flap angles in the data)."""
    from rdb.hybrid.box_controller import BoxController

    class Paused(BoxController):
        def step(self):
            held = [i for i in (0, 1) if pauses[i] and pauses[i][0] <= self.tick < pauses[i][1]]
            if not held:
                return super().step()
            now = self.cmd_pads()
            keep = [(now[i].copy(), self.yaw[i], self.tilt[i], self.segs[i], self.k[i], self.grip[i]) for i in held]
            a = super().step()
            ee, pads = self.cmd_pose(), self.cmd_pads()
            for i, (p, y, t, seg, k, g) in zip(held, keep):
                self.segs[i], self.k[i], self.grip[i] = seg, k, g
                if seg is not None:
                    seg.k = max(0, seg.k - 1)
                self._pads = pads
                self._write(a, ee, i, p, y, t)
                a[12 + i] = g
            return a
    return Paused


def episode_ctl(env, seed, max_steps, stride, pause=False):
    """BoxController with grasps on noisy privileged inputs (labels stay exact)."""
    from rdb.hybrid.box_controller import BoxController
    rng = np.random.default_rng(seed + 11)
    if pause:
        pauses = [None, None]
        for i in ([0, 1] if rng.random() < .3 else [int(rng.integers(2))]):
            start = int(rng.integers(30, 130))
            pauses[i] = (start, start + int(rng.integers(20, 90)))
        BoxController = paused_controller(pauses)                       # noqa: N806
    obs, _ = env.reset(seed=seed)
    obs, _, term, trunc, _ = env.step(np.zeros(14, np.float32))
    expert = PackBoxHumanExpert(env, np.random.default_rng(0), 0.)      # sim geometry reader only
    b = env.packing_box.body
    p = np.asarray(b.get_position(), float) + np.r_[rng.normal(0, .004, 2), 0.]
    ctl = BoxController((*p, yaw_of(b.get_quaternion()) + rng.normal(0, .03)), grasp=True,
                        grasp_steps=int(rng.integers(24, 49)))
    frames, steps, status = [], 1, "budget"
    while steps < max_steps:
        if (steps - 1) % stride == 0:
            frames.append(frame(obs, env, expert, seed, steps))
        rem = [PackBoxExpert.flap(expert, i)[2] - PackBoxExpert.flap(expert, i)[3] + rng.normal(0, .04) for i in (0, 1)]
        obs, _, term, trunc, _ = env.step(ctl.act(obs, rem))
        steps += 1
        if env.success:
            status = "success"
            frames.append(frame(obs, env, expert, seed, steps))
            break
        if term or trunc:
            status = "fail"
            break
    return frames, status, steps


def main():
    from roboeval.challenge import spec
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-seed", type=int, default=2_263_000)
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--noise", type=float, default=.25)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", choices=("teacher", "ctl"), default="teacher")
    ap.add_argument("--pause", action="store_true", help="ctl: random per-arm pause windows (seeds 2267000-2267999)")
    args = ap.parse_args()
    lo, hi = (2_267_000, 2_268_000) if args.pause else (2_263_000, 2_265_000)
    if not (lo <= args.first_seed and args.first_seed + args.count <= hi):
        ap.error(f"seeds must stay in {lo}-{hi - 1}")
    args.out.mkdir(parents=True, exist_ok=True)
    env = make_env("PackBox", "PositionAndOrientation", render=True)
    max_steps = spec.max_steps("PackBox", 20)
    t0 = time.monotonic()
    try:
        for seed in range(args.first_seed, args.first_seed + args.count):
            path = args.out / f"ep_{seed}.npz"
            if path.exists():
                continue
            frames, status, steps = (episode_ctl(env, seed, max_steps, args.stride, args.pause) if args.policy == "ctl" else
                                     episode(env, seed, args.noise, max_steps, args.stride))
            arrays = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
            tmp = args.out / f"ep_{seed}.tmp.npz"
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, path)
            print(json.dumps(dict(seed=seed, status=status, steps=steps, frames=len(frames),
                                  elapsed_s=round(time.monotonic() - t0, 1))), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
