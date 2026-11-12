"""Collect reset-frame LiftPot pose labels for vision training (privileged data only).

Run with MUJOCO_GL=egl (one render process per GPU).
The saved observations are the *unmodified* arrays returned by env.reset: three
CHW 256x256 RGB cameras plus both proprioception fields. No actions are taken.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from rdb.gen.episode_io import make_env
from rdb.gen.scripted_common import yaw_of

CAMERAS = ("head", "left_wrist", "right_wrist")
FIRST_SEED = 2_201_000
LAST_SEED = 2_209_999


def handle_spheres(env) -> list[int]:
    """Find four pot-handle sphere geoms once; use their live positions as labels."""
    model = env._mojo.physics.model._model
    name = env.kitchenpot.body.mjcf.full_identifier
    out = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_SPHERE or not model.geom_contype[geom]:
            continue
        body = model.geom_bodyid[geom]
        while body > 0:
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if body_name.startswith(name):
                out.append(geom)
                break
            body = model.body_parentid[body]
    if len(out) != 4:
        raise RuntimeError(f"expected four pot handle spheres, found {out}")
    return out


def record(obs, env, sphere_ids, seed: int) -> dict:
    frame = {}
    for cam in CAMERAS:
        key = f"rgb_{cam}"
        rgb = np.asarray(obs[key])
        if rgb.shape != (3, 256, 256) or rgb.dtype != np.uint8:
            raise ValueError(f"{key}: expected uint8 CHW 256x256, got {rgb.shape} {rgb.dtype}")
        frame[key] = rgb.copy()
    for key in ("proprioception", "proprioception_grippers"):
        frame[key] = np.asarray(obs[key], np.float32).copy()
    pos = np.asarray(env.kitchenpot.body.get_position(), np.float32).copy()
    frame["pot_xyz"] = pos
    frame["pot_yaw"] = np.float32(yaw_of(env.kitchenpot.body.get_quaternion()))
    xyz = env._mojo.physics.data._data.geom_xpos[sphere_ids].copy()
    order = np.argsort(xyz[:, 1])
    frame["handle_centres"] = np.stack((xyz[order[2:]].mean(0), xyz[order[:2]].mean(0))).astype(np.float32)
    frame["seed"] = np.int64(seed)
    return frame


def save_shard(out: Path, rows: list[dict], start: int, end: int) -> Path:
    path = out / f"pose_{start}_{end}.npz"
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    arrays["meta"] = np.array(json.dumps(dict(family="LiftPot", variation="PositionAndOrientation",
                                               observation="env.reset raw CHW 256x256 RGB",
                                               first_seed=start, last_seed=end)))
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-seed", type=int, default=FIRST_SEED)
    ap.add_argument("--count", type=int, default=5_000)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--out", type=Path, default=Path("data/hybrid/pot"))
    args = ap.parse_args()
    if not (FIRST_SEED <= args.first_seed <= LAST_SEED and
            1 <= args.count <= LAST_SEED + 1 - args.first_seed and args.shard_size >= 1):
        ap.error("requested seeds must stay in 2201000–2209999")
    args.out.mkdir(parents=True, exist_ok=True)

    env = make_env("LiftPot", "PositionAndOrientation", render=True)
    t0 = time.monotonic()
    rows = []
    sphere_ids = None
    try:
        for seed in range(args.first_seed, args.first_seed + args.count):
            obs, _ = env.reset(seed=seed)
            if sphere_ids is None:
                sphere_ids = handle_spheres(env)
                print(f"head view: {obs['rgb_head'].shape} {obs['rgb_head'].dtype}; "
                      "saved preview_0.png through preview_3.png", flush=True)
            row = record(obs, env, sphere_ids, seed)
            if seed < args.first_seed + 4:
                Image.fromarray(row["rgb_head"].transpose(1, 2, 0)).save(
                    args.out / f"preview_{seed - args.first_seed}.png")
            rows.append(row)
            if len(rows) >= args.shard_size or seed == args.first_seed + args.count - 1:
                start = int(rows[0]["seed"])
                path = save_shard(args.out, rows, start, seed)
                print(json.dumps(dict(shard=str(path), n=len(rows),
                                      elapsed_wall_s=round(time.monotonic() - t0, 1))), flush=True)
                rows.clear()
    finally:
        env.close()


if __name__ == "__main__":
    main()
