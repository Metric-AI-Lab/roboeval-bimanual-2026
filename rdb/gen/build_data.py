"""Build a training data root = rdb.prep output + generated episodes (rdb.gen.episode_io format).

  python -m rdb.gen.build_data --base data/s128 --gen data/gen --sources scripted --out data/s128_v2

Variants with generated episodes get a new directory holding the human frames followed by the
generated ones (episode ids continue after the human ones; frames shifted by
episode_io.dataset_frames, so rdb.data's obs[f] -> actions[f+1:] rule holds for both). All other
variants are symlinked. --max_ratio caps generated frames per variant at that multiple of the
human frames (successful episodes only, in seed order). meta.json records what was added.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rdb.gen.episode_io import dataset_frames, load_episode

KEYS = ("img", "state", "grip", "action", "episode", "frame")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--sources", required=True, help="comma list of data/gen/<source> dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_ratio", type=float, default=1.0)
    ap.add_argument("--drop_human", action="store_true",
                    help="variants with generated episodes keep ONLY those (no human frames); budget still uses human count")
    a = ap.parse_args()
    base, gen, out = Path(a.base).resolve(), Path(a.gen), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for vdir in sorted(p for p in base.iterdir() if (p / "meta.json").exists()):
        files = sorted(f for s in a.sources.split(",") for f in (gen / s / vdir.name).glob("ep_*.npz"))
        dst = out / vdir.name
        if not files:
            if not dst.exists():
                dst.symlink_to(vdir, target_is_directory=True)
            continue
        meta = json.loads((vdir / "meta.json").read_text())
        src = {k: np.load(vdir / f"{k}.npy", mmap_mode="r") for k in KEYS}
        n0, ep0 = len(src["episode"]), int(src["episode"].max()) + 1
        eps, n_gen, budget = [], 0, int(a.max_ratio * n0)
        if a.drop_human:
            n0, ep0 = 0, 0
        for f in files:
            ep = load_episode(f)
            if not ep["meta"].get("success", True):
                continue
            d = dataset_frames(ep)
            if n_gen + len(d["action"]) > budget:
                break
            eps.append(d)
            n_gen += len(d["action"])
        n = n0 + n_gen
        dst.mkdir(exist_ok=True)
        mm = {}
        for k in KEYS:
            mm[k] = np.lib.format.open_memmap(dst / f"{k}.npy", mode="w+", dtype=src[k].dtype,
                                              shape=(n,) + src[k].shape[1:])
            if n0:
                mm[k][:n0] = src[k]
        i = n0
        for j, d in enumerate(eps):
            t = len(d["action"])
            for k in ("img", "state", "grip", "action"):
                mm[k][i:i + t] = d[k]
            mm["episode"][i:i + t] = ep0 + j
            mm["frame"][i:i + t] = np.arange(t)
            i += t
        for m in mm.values():
            m.flush()
        meta.update(n_frames=n, n_episodes=ep0 + len(eps), n_human_frames=n0,
                    generated={"sources": a.sources, "episodes": len(eps), "frames": n_gen,
                               "available": len(files), "max_ratio": a.max_ratio})
        (dst / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{vdir.name}: {n0} human + {n_gen} generated frames ({len(eps)}/{len(files)} episodes)")


if __name__ == "__main__":
    main()
