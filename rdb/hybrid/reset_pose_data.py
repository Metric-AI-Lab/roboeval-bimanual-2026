"""Collect reset-frame object-pose labels for pose-hybrid vision training (privileged labels, data only).

  MUJOCO_GL=egl python -m rdb.hybrid.reset_pose_data --task blocks --count 5000

Saves the unmodified env.reset observation (three CHW 256x256 RGB cameras, proprioception) and per-object labels
`<name>_xyz` (world position) and `<name>_yaw` (world yaw) in 500-reset .npz shards under data/hybrid/<task>/.
No actions are taken. Seeds: blocks 2230000-2239999, valve 2242000-2249999, pick 2216000-2217999, shelf
2218000-2219999, box 2261000-2262999.
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

CAMERAS = ("head", "left_wrist", "right_wrist")
def valve_labels(env):
    """Hub-top grasp points of env.valves (the valve teacher's hub_top), static for the episode."""
    import numpy as np
    from rdb.hybrid.verify_valve_controller import hub_tops
    tops, _ = hub_tops(env)
    return {f"hub{c}_xyz": np.asarray(t, np.float32) for c, t in zip("AB", tops)}  # names read by policy_bundle/policy/policy.py


def book_labels(env):
    """Same convention as rdb.hybrid.book_pose_data: book (and shelf assembly) body pose, world frame, wxyz."""
    out = {"book_xyz": np.asarray(env.books[0].body.get_position(), np.float32).copy(),
           "book_quat_wxyz": np.asarray(env.books[0].body.get_quaternion(), np.float32).copy()}
    if hasattr(env, "book_shelf"):
        out["shelf_xyz"] = np.asarray(env.book_shelf.body.get_position(), np.float32).copy()
        out["shelf_quat_wxyz"] = np.asarray(env.book_shelf.body.get_quaternion(), np.float32).copy()
    return out


TASKS = {  # task: (family, seed range, objects(env) -> {name: prop} or None, extra labels(env) -> {key: array})
    "blocks": ("StackTwoBlocks", (2_230_000, 2_240_000), lambda env: {f"blk{c}": b for c, b in zip("AB", env.blocks)},
               None),
    "valve": ("RotateValve", (2_242_000, 2_250_000), lambda env: {}, valve_labels),
    "box": ("PackBox", (2_261_000, 2_263_000), lambda env: {"box": env.packing_box}, None),
    "pick": ("PickSingleBookFromTable", (2_216_000, 2_218_000), lambda env: {}, book_labels),
    "shelf": ("StackSingleBookShelf", (2_218_000, 2_220_000), lambda env: {}, book_labels),
}


def record(obs, env, objects, seed: int, extra=None) -> dict:
    frame = {f"rgb_{c}": np.asarray(obs[f"rgb_{c}"]).copy() for c in CAMERAS}
    for key in ("proprioception", "proprioception_grippers"):
        frame[key] = np.asarray(obs[key], np.float32).copy()
    for name, prop in objects(env).items():
        frame[f"{name}_xyz"] = np.asarray(prop.body.get_position(), np.float32).copy()
        frame[f"{name}_yaw"] = np.float32(yaw_of(prop.body.get_quaternion()))
    if extra is not None:
        frame.update(extra(env))
    frame["seed"] = np.int64(seed)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--first-seed", type=int)
    ap.add_argument("--count", type=int, default=5_000)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    family, (lo, hi), objects, extra = TASKS[args.task]
    first = lo if args.first_seed is None else args.first_seed
    if not (lo <= first and first + args.count <= hi):
        ap.error(f"seeds must stay in {lo}-{hi - 1}")
    out = args.out or Path("data/hybrid") / args.task
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(family, "PositionAndOrientation", render=True)
    t0, rows = time.monotonic(), []
    try:
        for seed in range(first, first + args.count):
            obs, _ = env.reset(seed=seed)
            rows.append(record(obs, env, objects, seed, extra))
            if len(rows) >= args.shard_size or seed == first + args.count - 1:
                path = out / f"pose_{rows[0]['seed']}_{seed}.npz"
                arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **arrays)
                os.replace(tmp, path)
                print(json.dumps(dict(shard=str(path), n=len(rows), elapsed_wall_s=round(time.monotonic() - t0, 1))),
                      flush=True)
                rows.clear()
    finally:
        env.close()


if __name__ == "__main__":
    main()
