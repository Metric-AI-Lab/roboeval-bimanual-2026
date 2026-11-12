"""Train an object-pose regressor for the pose-hybrid policies (cameras + proprioception -> object state).

  python -m rdb.hybrid.train_pose --data data/hybrid/pot --labels pot_xyz pot_yaw handle_centres --out runs/hybrid/pot_pose

Shards are .npz files with one array per camera (`rgb_head`, `rgb_left_wrist`, `rgb_right_wrist`; NHWC or NCHW uint8),
`proprio` (N, D) (or `proprioception` + `proprioception_grippers`) and the label arrays named by --labels (each (N,) or (N, k)). Angles whose name ends in `yaw` are
regressed as (sin, cos). `<name>_quat_wxyz:axisy` regresses only the sign-free direction of the local y axis (as the
6 entries of u u^T) for objects like the handover rod whose quaternion is ambiguous. One shared ResNet18 encodes each camera; features + proprio go through an MLP. The best
checkpoint (validation loss) is saved with its normalisation stats; per-label validation errors are logged.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rdb.hybrid.pose_infer import PoseNet, axis_from_outer
from rdb.train import color_jitter, random_shift

CAMS = ("rgb_head", "rgb_left_wrist", "rgb_right_wrist")


def fold(k: str) -> int:
    return int(k.split(":")[1]) if ":" in k else 1


def quat_axis_y(q: np.ndarray) -> np.ndarray:
    """World direction of the local y axis for (N, 4) wxyz quaternions."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], 1)


def axis_outer(u: np.ndarray) -> np.ndarray:
    """Sign-free axis code: the 6 distinct entries of u u^T."""
    x, y, z = u.T
    return np.stack([x * x, y * y, z * z, x * y, x * z, y * z], 1).astype(np.float32)


def load(roots, labels, cams, size):
    imgs, prop, seeds, ys = [], [], [], {k: [] for k in labels}
    files = sorted({f for r in roots for d in sorted(glob.glob(r)) for f in Path(d).rglob("*.npz") if ".tmp" not in f.name})
    for f in files:
        z = np.load(f)
        ims = []
        for c in cams:
            a = z[c]
            if a.shape[-1] == 3:
                a = a.transpose(0, 3, 1, 2)
            t = torch.from_numpy(np.ascontiguousarray(a))
            if t.shape[-1] != size:
                t = nn.functional.interpolate(t.float(), size=(size, size), mode="area").round().to(torch.uint8)
            ims.append(t)
        imgs.append(torch.stack(ims, 1))
        p = z["proprio"] if "proprio" in z else np.concatenate([z["proprioception"], z["proprioception_grippers"]], 1)
        prop.append(p.astype(np.float32))
        seeds.append(z["seed"].reshape(-1) if "seed" in z else np.arange(len(p)))
        for k in labels:
            a = z[k.split(":")[0]]
            ys[k].append(a.astype(np.float32).reshape(len(a), -1))
    y_parts, spec = [], []
    for k in labels:
        v = np.concatenate(ys[k])
        if k.split(":")[0].endswith("yaw"):  # "<name>_yaw:4" = 90-degree symmetric object: regress 4 * yaw
            f = fold(k)
            v = np.concatenate([np.sin(f * v), np.cos(f * v)], 1)
        if k.endswith("quat_wxyz"):  # q and -q are the same rotation
            v = v * np.where(v[:, :1] < 0, -1, 1)
        if k.endswith(":axisy"):  # only the long axis matters, up to sign: regress u u^T
            v = axis_outer(quat_axis_y(v))
        spec.append((k, v.shape[1]))
        y_parts.append(v)
    out = torch.empty((sum(len(t) for t in imgs), *imgs[0].shape[1:]), dtype=torch.uint8)
    i = 0
    while imgs:                                  # copy and free file by file: peak memory ~1x the image set, not 2x
        t = imgs.pop(0)
        out[i:i + len(t)] = t
        i += len(t)
    return (out, torch.from_numpy(np.concatenate(prop)), torch.from_numpy(np.concatenate(y_parts, 1)), spec,
            np.concatenate(seeds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="shard roots (globs allowed), searched recursively")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--cams", nargs="+", default=list(CAMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.15)
    ap.add_argument("--init", help="start from this pose checkpoint's weights (same cams/labels; fine-tuning)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if dev == "cuda":   # GB10 shares system memory: reserve CUDA memory before the image set and page cache fill it
        torch.empty(int(4e9), dtype=torch.uint8, device=dev)    # freed into PyTorch's cache, kept for the run
    img, prop, y, spec, seed = load(args.data, args.labels, args.cams, args.size)
    n = len(y)
    # validation = whole episodes (seeds): frames of one episode are near-duplicates
    useeds = np.unique(seed)
    rng = np.random.default_rng(0)
    val_seeds = rng.choice(useeds, max(1, int(len(useeds) * args.val_frac)), replace=False)
    is_val = np.isin(seed, val_seeds)
    va, tr = torch.from_numpy(np.flatnonzero(is_val)), torch.from_numpy(np.flatnonzero(~is_val))
    nv = len(va)
    pm, ps = prop[tr].mean(0), prop[tr].std(0) + 1e-6
    ym, ys = y[tr].mean(0), y[tr].std(0) + 1e-6
    print(f"{n} samples ({len(tr)} train / {nv} val), labels {spec}, images {tuple(img.shape)}", flush=True)

    net = PoseNet(len(args.cams), prop.shape[1], y.shape[1]).to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["state"])
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    pm_d, ps_d, ym_d, ys_d = (t.to(dev) for t in (pm, ps, ym, ys))

    def batch(idx, aug):
        im = img[idx].to(dev, non_blocking=True)
        if aug:  # no shift by default: a translated image means a different object position
            im = color_jitter(random_shift(im, args.shift) if args.shift else im, args.jitter)
        return im.float() / 255, (prop[idx].to(dev) - pm_d) / ps_d, (y[idx].to(dev) - ym_d) / ys_d

    @torch.no_grad()
    def evaluate():
        net.eval()
        outs, tgts, loss = [], [], 0.0
        for i in range(0, nv, 256):
            im, p, t = batch(va[i:i + 256], False)
            o = net(im, p)
            loss += nn.functional.smooth_l1_loss(o, t, reduction="sum").item()
            outs.append((o * ys_d + ym_d).cpu())
            tgts.append((t * ys_d + ym_d).cpu())
        net.train()
        o, t = torch.cat(outs), torch.cat(tgts)
        rep, j = {}, 0
        for k, d in spec:
            if k.split(":")[0].endswith("yaw"):  # mean abs angle error in degrees (modulo the symmetry)
                da = (torch.atan2(o[:, j], o[:, j + 1]) - torch.atan2(t[:, j], t[:, j + 1])) / fold(k)
                rep[k.split(":")[0] + "_deg"] = round(float(torch.rad2deg(torch.atan2(da.sin(), da.cos()).abs()).mean()), 3)
            elif k.endswith(":axisy"):  # mean angle between predicted and true axis, modulo sign
                ua, ub = axis_from_outer(o[:, j:j + 6].numpy()), axis_from_outer(t[:, j:j + 6].numpy())
                rep[k.split(":")[0] + "_axis_deg"] = round(float(np.degrees(np.arccos(np.clip(np.abs((ua * ub).sum(1)), 0, 1))).mean()), 3)
            elif k.endswith("quat_wxyz"):  # mean rotation angle error in degrees
                q = nn.functional.normalize(o[:, j:j + 4], dim=1)
                dot = (q * t[:, j:j + 4]).sum(1).abs().clamp(max=1)
                rep[k.split(":")[0] + "_deg"] = round(float(torch.rad2deg(2 * torch.acos(dot)).mean()), 3)
            else:  # mean Euclidean error (metres for positions)
                rep[k] = round(float((o[:, j:j + d] - t[:, j:j + d]).norm(dim=1).mean()), 5)
            j += d
        return loss / (nv * y.shape[1]), rep

    best, log = float("inf"), open(out / "log.jsonl", "a")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = tr[torch.randint(len(tr), (args.bs,))]
        im, p, t = batch(idx, True)
        loss = nn.functional.smooth_l1_loss(net(im, p), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == args.steps:
            vl, rep = evaluate()
            rec = dict(step=step, train=round(loss.item(), 5), val=round(vl, 5), mae=rep, sec=round(time.time() - t0))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if vl < best:
                best = vl
                torch.save(dict(state=net.state_dict(), cams=args.cams, size=args.size, spec=spec, prop_mean=pm,
                                prop_std=ps, y_mean=ym, y_std=ys, step=step), out / "pose_best.pt")


if __name__ == "__main__":
    main()
