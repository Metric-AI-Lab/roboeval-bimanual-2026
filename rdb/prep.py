"""Convert roboeval_ee_delta_20hz_scaled (LeRobot v3, PNG-in-parquet) into flat
memmapped arrays for fast random-access training.

Per variant <out>/<Variant>/:
  img.npy      uint8  [N, n_cams, S, S, 3]   (resized with rdb.common.resize_image)
  state.npy    float32 [N, 36]
  grip.npy     float32 [N, 2]
  action.npy   float32 [N, 14]
  episode.npy  int32   [N]   (episode_index within the variant)
  frame.npy    int32   [N]
  meta.json    {n_frames, n_episodes, cams, size, instructions}

Usage:
  python -m rdb.prep --src ../datasets/roboeval_ee_delta_20hz_scaled --out data/s128 --size 128 --procs 8
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from rdb.common import CAMERAS, resize_image

DATASET_CAM_KEY = {c: f"cam_{c}" for c in CAMERAS}


def _alloc(vdir: Path, n: int, n_cams: int, size: int) -> None:
    specs = {
        "img": (np.uint8, (n, n_cams, size, size, 3)),
        "state": (np.float32, (n, 36)),
        "grip": (np.float32, (n, 2)),
        "action": (np.float32, (n, 14)),
        "episode": (np.int32, (n,)),
        "frame": (np.int32, (n,)),
    }
    for name, (dt, shape) in specs.items():
        np.lib.format.open_memmap(vdir / f"{name}.npy", mode="w+", dtype=dt, shape=shape).flush()


def _fill(job) -> int:
    path, vdir, cams, size = job
    vdir = Path(vdir)
    mm = {k: np.load(vdir / f"{k}.npy", mmap_mode="r+")
          for k in ("img", "state", "grip", "action", "episode", "frame")}
    pf = pq.ParquetFile(path)
    cols = ["index", "episode_index", "frame_index", "state", "state_gripper", "actions"] + [
        DATASET_CAM_KEY[c] for c in cams]
    rows = 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)
        idx = t.column("index").to_numpy()
        mm["episode"][idx] = t.column("episode_index").to_numpy()
        mm["frame"][idx] = t.column("frame_index").to_numpy()
        for src, dst in (("state", "state"), ("state_gripper", "grip"), ("actions", "action")):
            mm[dst][idx] = np.stack(t.column(src).to_numpy(zero_copy_only=False))
        for ci, c in enumerate(cams):
            col = t.column(DATASET_CAM_KEY[c]).to_pylist()
            for j, cell in zip(idx, col):
                hwc = np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
                mm["img"][j, ci] = resize_image(hwc, size)
        rows += len(idx)
    for m in mm.values():
        m.flush()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--cams", default=",".join(CAMERAS))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--variants", default="", help="comma list; default all present")
    args = ap.parse_args()

    cams = args.cams.split(",")
    src, out = Path(args.src), Path(args.out)
    variants = args.variants.split(",") if args.variants else sorted(
        p.name for p in src.iterdir() if (p / "meta" / "info.json").exists())
    jobs = []
    for v in variants:
        vdir = out / v
        if (vdir / "meta.json").exists():
            print(f"skip {v} (done)")
            continue
        info = json.loads((src / v / "meta" / "info.json").read_text())
        files = sorted(glob.glob(str(src / v / "data" / "*" / "*.parquet")))
        vdir.mkdir(parents=True, exist_ok=True)
        _alloc(vdir, info["total_frames"], len(cams), args.size)
        tasks = pq.read_table(src / v / "meta" / "tasks.parquet").to_pandas()
        instr = [str(x) for x in (tasks.index if "task" not in tasks.columns else tasks["task"])]
        jobs.append((v, vdir, info, files, instr))

    flat = [(f, str(vdir), cams, args.size) for v, vdir, info, files, instr in jobs for f in files]
    print(f"{len(jobs)} variants, {len(flat)} parquet files")
    with Pool(args.procs) as pool:
        done = 0
        for n in pool.imap_unordered(_fill, flat):
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(flat)} files", flush=True)
    for v, vdir, info, files, instr in jobs:
        ep = np.load(vdir / "episode.npy", mmap_mode="r")
        (vdir / "meta.json").write_text(json.dumps({
            "variant": v, "n_frames": int(info["total_frames"]), "n_episodes": int(ep.max()) + 1,
            "cams": cams, "size": args.size, "instructions": instr}, indent=1))
        print(f"{v}: {info['total_frames']} frames, {int(ep.max()) + 1} episodes")


if __name__ == "__main__":
    main()
