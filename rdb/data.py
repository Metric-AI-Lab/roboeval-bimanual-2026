"""Random-access dataset over the memmapped arrays written by rdb.prep."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from rdb.common import FAMILIES, family_of

EVAL_SUFFIX = "PositionAndOrientation"


class Variant:
    def __init__(self, vdir: Path):
        self.name = vdir.name
        self.meta = json.loads((vdir / "meta.json").read_text())
        self.family = family_of(self.name)
        self.task = FAMILIES.index(self.family)
        self.img = np.load(vdir / "img.npy", mmap_mode="r")
        self.state = np.load(vdir / "state.npy")
        self.grip = np.load(vdir / "grip.npy")
        self.action = np.load(vdir / "action.npy")
        ep = np.load(vdir / "episode.npy")
        # frames of one episode are contiguous; end index (exclusive) per frame
        bounds = np.flatnonzero(np.diff(ep)) + 1
        ends = np.append(bounds, len(ep))
        self.ep_end = np.repeat(ends, np.diff(np.insert(ends, 0, 0))).astype(np.int64)
        self.episode = ep
        self.n = len(ep)


def load_variants(root: Path, names: List[str] | None = None) -> List[Variant]:
    dirs = sorted(p for p in root.iterdir() if (p / "meta.json").exists())
    if names:
        dirs = [d for d in dirs if d.name in names]
    return [Variant(d) for d in dirs]


def split_episodes(variants: List[Variant], val_frac: float, seed: int = 0) -> Dict[str, set]:
    """Hold out a fraction of episodes per variant for a validation loss."""
    rng = np.random.default_rng(seed)
    held = {}
    for v in variants:
        n_ep = int(v.episode.max()) + 1
        k = int(round(n_ep * val_frac))
        held[v.name] = set(rng.choice(n_ep, size=k, replace=False).tolist()) if k else set()
    return held


class ChunkDataset(Dataset):
    """Item i -> (variant, frame) via a flat index; weighting is done by the sampler."""

    def __init__(self, variants: List[Variant], chunk: int, held: Dict[str, set], train: bool):
        self.variants = variants
        self.chunk = chunk
        index = []
        for vi, v in enumerate(variants):
            keep = np.array([(e in held[v.name]) != train for e in v.episode])
            keep &= np.arange(v.n) + 1 < v.ep_end   # last frame has no next action
            frames = np.flatnonzero(keep)
            index.append(np.stack([np.full_like(frames, vi), frames], 1))
        self.index = np.concatenate(index)

    def frame_weights(self, pr_weight: float) -> np.ndarray:
        """Equal mass per family; within a family the evaluated PR variant gets pr_weight."""
        vi = self.index[:, 0]
        counts = np.bincount(vi, minlength=len(self.variants)).astype(np.float64)
        w_var = np.zeros(len(self.variants))
        for fam in FAMILIES:
            members = [i for i, v in enumerate(self.variants) if v.family == fam and counts[i] > 0]
            if not members:
                continue
            pr = [i for i in members if self.variants[i].name.endswith(EVAL_SUFFIX)]
            rest = [i for i in members if i not in pr]
            if pr and rest:
                for i in pr:
                    w_var[i] = pr_weight / len(pr)
                tot = counts[rest].sum()
                for i in rest:
                    w_var[i] = (1 - pr_weight) * counts[i] / tot
            else:
                for i in members:
                    w_var[i] = counts[i] / counts[members].sum()
        w_var /= max(1, sum(1 for f in FAMILIES if any(v.family == f for v in self.variants)))
        return w_var[vi] / counts[vi]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        # Stored frame f is the observation *after* action f was applied (checked by
        # open-loop replay), so the actions that follow observation f start at f + 1.
        vi, f = self.index[i]
        v = self.variants[vi]
        end = v.ep_end[f]
        idx = np.arange(f + 1, f + 1 + self.chunk)
        mask = idx < end
        idx = np.minimum(idx, end - 1)
        img = np.ascontiguousarray(v.img[f].transpose(0, 3, 1, 2))  # [n,3,H,W]
        return {
            "img": torch.from_numpy(img),
            "state": torch.from_numpy(np.concatenate([v.state[f], v.grip[f]])),
            "task": torch.tensor(v.task, dtype=torch.long),
            "action": torch.from_numpy(v.action[idx]),
            "mask": torch.from_numpy(mask),
        }


def norm_stats(variants: List[Variant]) -> Dict[str, list]:
    s = np.concatenate([np.concatenate([v.state, v.grip], 1) for v in variants])
    a = np.concatenate([v.action for v in variants])
    s_std = s.std(0)
    s_std[s_std < 1e-4] = 1.0
    a_mean, a_std = a.mean(0), a.std(0)
    a_std[a_std < 1e-6] = 1.0
    a_mean[12:], a_std[12:] = 0.5, 0.5   # binary gripper commands -> [-1, 1]
    return {"state_mean": s.mean(0).tolist(), "state_std": s_std.tolist(),
            "action_mean": a_mean.tolist(), "action_std": a_std.tolist()}
