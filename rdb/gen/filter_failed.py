"""Data root without the human demonstrations that the official reference marks as failures.

  python -m rdb.gen.filter_failed --base data/s128 --out data/s128_clean --families StackSingleBookShelf,PickSingleBookFromTable

RoboEval/roboeval/challenge/human_reference_episodes.json labels the evaluated (PositionAndOrientation) demos; entry i is
our episode i (episode lengths match for all 8 families). Labelled variants are rewritten without the failed episodes;
every other variant is symlinked unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = "PositionAndOrientation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/s128")
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", required=True, help="comma list of families whose evaluated demos are filtered")
    a = ap.parse_args()
    base, out = Path(a.base).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    refs = json.loads((ROOT / "RoboEval/roboeval/challenge/human_reference_episodes.json").read_text())
    fams = a.families.split(",")
    for vd in sorted(p for p in base.iterdir() if (p / "meta.json").exists()):
        dst = out / vd.name
        fam = next((f for f in fams if vd.name == f + EVAL), None)
        if fam is None:
            if not dst.exists():
                dst.symlink_to(vd)
            continue
        ep = np.load(vd / "episode.npy")
        lens = np.bincount(ep)
        assert [e["dataset_steps"] for e in refs[fam]] == lens.tolist(), f"{vd.name}: labels do not match episodes"
        good = np.array([e["success"] for e in refs[fam]])
        keep = good[ep]
        dst.mkdir(exist_ok=True)
        for f in ("action.npy", "frame.npy", "grip.npy", "state.npy", "img.npy"):
            arr = np.load(vd / f, mmap_mode="r")
            np.save(dst / f, np.asarray(arr[keep]))
        new_id = np.cumsum(good) - 1                      # renumber kept episodes 0..k-1
        np.save(dst / "episode.npy", new_id[ep[keep]])
        meta = json.loads((vd / "meta.json").read_text())
        meta.update(n_frames=int(keep.sum()), n_episodes=int(good.sum()),
                    filtered=f"dropped {int((~good).sum())} reference-failed demos")
        (dst / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{vd.name}: kept {int(good.sum())}/{len(good)} episodes, {int(keep.sum())}/{len(ep)} frames")


if __name__ == "__main__":
    main()
