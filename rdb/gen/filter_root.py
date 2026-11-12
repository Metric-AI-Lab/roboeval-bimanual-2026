"""Copy a data root keeping only human episodes whose gripper closes on one arm (and, optionally, reference successes).

The shelf demos mix two grasp styles (2026-10-19): the right arm closes in 56/73
evaluated-variation episodes, the left in 17. The policy averages the two modes and closes late, high or not at all.
This keeps one mode so the policy learns one consistent grasp.

  python -m rdb.gen.filter_root --base data/s128 --out data/s128_shelf_right \
      --variants StackSingleBookShelf,StackSingleBookShelfPosition,StackSingleBookShelfPositionAndOrientation \
      --arm right --drop_failed StackSingleBookShelf

Other variants are symlinked. --drop_failed uses RoboEval's human_reference_episodes.json success labels, which
cover the PositionAndOrientation variant (list index = episode id).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import roboeval

from rdb.gen.build_data import KEYS

REFS = Path(roboeval.__file__).parent / "challenge" / "human_reference_episodes.json"
GRIP = {"left": 12, "right": 13}      # action dims of the two gripper commands


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", required=True, help="comma list of variants to filter")
    ap.add_argument("--arm", choices=tuple(GRIP), required=True)
    ap.add_argument("--drop_failed", default="", help="family whose reference-failed episodes are dropped")
    a = ap.parse_args()
    base, out = Path(a.base).resolve(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keep_v = set(a.variants.split(","))
    failed = set()
    if a.drop_failed:
        refs = json.loads(REFS.read_text())
        failed = {i for i, r in enumerate(refs[a.drop_failed]) if not r["success"]}
    for vdir in sorted(p for p in base.iterdir() if (p / "meta.json").exists()):
        dst = out / vdir.name
        if vdir.name not in keep_v:
            if not dst.exists():
                dst.symlink_to(vdir, target_is_directory=True)
            continue
        src = {k: np.load(vdir / f"{k}.npy", mmap_mode="r") for k in KEYS}
        ep, act = np.asarray(src["episode"]), src["action"]
        other = GRIP["left" if a.arm == "right" else "right"]
        drop_f = a.drop_failed and vdir.name == a.drop_failed + "PositionAndOrientation"
        keep = [i for i in np.unique(ep)
                if (act[ep == i, GRIP[a.arm]] > .5).any() and not (act[ep == i, other] > .5).any()
                and not (drop_f and int(i) in failed)]
        mask = np.isin(ep, keep)
        n = int(mask.sum())
        dst.mkdir(exist_ok=True)
        for k in KEYS:
            mm = np.lib.format.open_memmap(dst / f"{k}.npy", mode="w+", dtype=src[k].dtype, shape=(n,) + src[k].shape[1:])
            if k == "episode":
                mm[:] = np.searchsorted(np.asarray(keep), ep[mask])     # renumber 0..len(keep)-1
            elif k == "img":
                idx, j = np.flatnonzero(mask), 0
                for s in range(0, len(idx), 2048):
                    mm[j:j + len(idx[s:s + 2048])] = src[k][idx[s:s + 2048]]
                    j += len(idx[s:s + 2048])
            else:
                mm[:] = src[k][mask]
            mm.flush()
        meta = json.loads((vdir / "meta.json").read_text())
        meta.update(n_frames=n, n_episodes=len(keep),
                    filtered={"arm": a.arm, "kept": len(keep), "of": int(len(np.unique(ep))), "drop_failed": bool(drop_f)})
        (dst / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{vdir.name}: kept {len(keep)}/{len(np.unique(ep))} episodes, {n} frames")


if __name__ == "__main__":
    main()
