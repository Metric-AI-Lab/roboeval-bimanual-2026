"""Train the multi-task chunk policy.

  python -m rdb.train --data data/s128 --out runs/v1 --steps 100000
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
import time
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Sampler

from rdb.data import ChunkDataset, load_variants, norm_stats, split_episodes
from rdb.model import ChunkPolicy, PolicyConfig


def _worker_ignore_sigterm(_):
    # `timeout` signals the whole process group; workers must survive it so the main process
    # can finish its step, save ckpt_last.pt and exit cleanly.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


class InfiniteWeightedSampler(Sampler):
    """Endless weighted sampling with replacement, drawn in blocks (WeightedRandomSampler
    materialises all num_samples indices up front)."""

    def __init__(self, weights: np.ndarray, block: int = 1 << 16, seed: int = 0):
        self.p = np.asarray(weights, dtype=np.float64) / np.sum(weights)
        self.block = block
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            yield from self.rng.choice(len(self.p), size=self.block, p=self.p).tolist()


def random_shift(img: torch.Tensor, pad: int) -> torch.Tensor:
    """DrQ-style random translation, per image. img uint8 [B,n,3,H,W] on GPU."""
    b, n, c, h, w = img.shape
    x = img.flatten(0, 1).float()
    x = F.pad(x, (pad,) * 4, mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    ar = torch.linspace(-1 + eps, 1 - eps, h + 2 * pad, device=x.device)[:h]
    base = torch.stack(torch.meshgrid(ar, ar, indexing="ij")[::-1], -1)[None].expand(b * n, -1, -1, -1)
    shift = torch.randint(0, 2 * pad + 1, (b * n, 1, 1, 2), device=x.device).float() * 2.0 / (h + 2 * pad)
    x = F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)
    return x.clamp_(0, 255).to(torch.uint8).view(b, n, c, h, w)


def color_jitter(img: torch.Tensor, strength: float) -> torch.Tensor:
    """Per-image brightness/contrast jitter (cheap, on GPU)."""
    b, n = img.shape[:2]
    x = img.float()
    br = 1 + (torch.rand(b, n, 1, 1, 1, device=x.device) * 2 - 1) * strength
    ct = 1 + (torch.rand(b, n, 1, 1, 1, device=x.device) * 2 - 1) * strength
    m = x.mean(dim=(2, 3, 4), keepdim=True)
    x = ((x - m) * ct + m) * br
    return x.clamp_(0, 255).to(torch.uint8)


def init_wandb(args, out: Path, n_params: int):
    """Mirror log.jsonl to wandb. The run id lives in the run dir, so auto-resumed jobs keep
    appending to the same wandb run. Any wandb failure only disables logging."""
    if not args.wandb:
        return None
    env = Path(__file__).resolve().parents[1] / ".env"   # WANDB_API_KEY=... (git-ignored)
    if env.exists():
        for line in env.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    try:
        import wandb
        id_file = out / "wandb_id"
        if not id_file.exists():
            id_file.write_text(uuid.uuid4().hex[:12])
        # registry name, so a resume or backfill keeps the new name
        reg = (out / "wandb_name").read_text().strip() if (out / "wandb_name").exists() else ""
        run = wandb.init(project=args.wandb, group=reg.split("/")[0] if reg else (args.wandb_group or None),
                         name=reg or out.name,
                         id=id_file.read_text().strip(), resume="allow", dir=str(out),
                         config={**vars(args), "n_params": n_params})
        # "step" is our own x-axis: after a resume, steps already logged past the last
        # checkpoint are simply re-logged instead of being dropped as non-monotonic
        run.define_metric("step")
        run.define_metric("*", step_metric="step")
        return run
    except Exception as e:
        print(f"wandb disabled: {e!r}", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="", help="comma list; default all")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone_lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--pr_weight", type=float, default=0.5)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save_every", type=int, default=10_000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--val_every", type=int, default=2_000)
    ap.add_argument("--shift", type=int, default=6)
    ap.add_argument("--jitter", type=float, default=0.15)
    ap.add_argument("--resume", default="")
    ap.add_argument("--init", default="", help="policy_*.pt to start from (weights + its cfg/norm stats; fresh optimizer)")
    ap.add_argument("--wandb", default="", help="wandb project; \"\" = off")
    ap.add_argument("--wandb_group", default="", help="wandb group, e.g. the sweep name")
    ap.add_argument("--save_minutes", type=float, default=20.0, help="also save 'last' this often")
    # model
    ap.add_argument("--head", default="flow")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--dec_layers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    variants = load_variants(Path(args.data), args.variants.split(",") if args.variants else None)
    print("variants:", [f"{v.name}:{v.n}" for v in variants], flush=True)
    held = split_episodes(variants, args.val_frac)
    train_ds = ChunkDataset(variants, args.chunk, held, train=True)
    val_ds = ChunkDataset(variants, args.chunk, held, train=False)
    sampler = InfiniteWeightedSampler(train_ds.frame_weights(args.pr_weight))
    loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4,
                        worker_init_fn=_worker_ignore_sigterm)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True,
                            worker_init_fn=_worker_ignore_sigterm)

    init = torch.load(args.init, map_location="cpu", weights_only=False) if args.init else None
    if init:   # fine-tune: keep the parent's architecture and normalisation so its weights stay valid
        cfg = PolicyConfig(**{**init["cfg"], "pretrained_backbone": False})
        args.chunk = cfg.chunk
        size = variants[0].img.shape[2]
        if size != cfg.img_size:   # new image size: resample the camera-token position grid
            assert cfg.encoder == "resnet18" and size % 32 == 0, "resize only supported for resnet18"
            pos = init["state_dict"]["cams.pos"]                       # [1, n_cams, g*g, d]
            g, ng = cfg.img_size // 32, size // 32
            grid = pos[0].view(cfg.n_cams, g, g, -1).permute(0, 3, 1, 2).float()
            grid = F.interpolate(grid, size=(ng, ng), mode="bicubic", align_corners=False)
            init["state_dict"]["cams.pos"] = grid.permute(0, 2, 3, 1).reshape(1, cfg.n_cams, ng * ng, -1)
            print(f"init: image size {cfg.img_size} -> {size}, position grid {g}x{g} -> {ng}x{ng}", flush=True)
            cfg.img_size = size
    else:
        stats = norm_stats(variants)
        cfg = PolicyConfig(n_cams=variants[0].img.shape[1], img_size=variants[0].img.shape[2], chunk=args.chunk,
                           d_model=args.d_model, n_heads=args.n_heads, enc_layers=args.enc_layers,
                           dec_layers=args.dec_layers, head=args.head, **stats)
    model = ChunkPolicy(cfg).to(dev)
    if init:
        model.load_state_dict(init["state_dict"])
        print("init from", args.init, "step", init.get("step"), flush=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params / 1e6:.1f}M, train frames {len(train_ds)}, val frames {len(val_ds)}", flush=True)

    bb = [p for n, p in model.named_parameters() if n.startswith("cams.trunk")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("cams.trunk")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.backbone_lr}, {"params": rest, "lr": args.lr}],
                            weight_decay=args.wd, betas=(0.9, 0.95))
    base_lrs = [args.backbone_lr, args.lr]
    step = 0
    if not args.resume and (out / "ckpt_last.pt").exists():
        args.resume = str(out / "ckpt_last.pt")   # auto-resume (time-limited queues)
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print("resumed at", step, flush=True)

    def lr_at(s: int) -> float:
        if s < args.warmup:
            return s / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p))) * 0.95 + 0.05

    def to_dev(batch):
        return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}

    def save(tag: str):
        # write-then-rename so a job killed at its time limit never leaves a torn checkpoint
        for name, obj in ((f"ckpt_{tag}.pt", {"model": model.state_dict(), "ema": ema.state_dict(),
                                               "opt": opt.state_dict(), "step": step, "cfg": cfg.to_dict()}),
                          (f"policy_{tag}.pt", {"cfg": cfg.to_dict(), "state_dict": ema.state_dict(), "step": step})):
            torch.save(obj, out / (name + ".tmp"))
            (out / (name + ".tmp")).replace(out / name)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))  # save + exit cleanly

    log = open(out / "log.jsonl", "a")
    wb = init_wandb(args, out, n_params)
    t0, it = time.time(), iter(loader)
    last_save = time.time()
    model.train()
    while step < args.steps:
        batch = to_dev(next(it))
        batch["img"] = color_jitter(random_shift(batch["img"], args.shift), args.jitter)
        f = lr_at(step)
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss = model.loss(batch)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        with torch.no_grad():
            d = min(args.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        if step % args.log_every == 0:
            rec = {"step": step, "loss": loss.item(), "gn": gn.item(), "lr": args.lr * f,
                   "sps": args.log_every / (time.time() - t0)}
            t0 = time.time()
            if step % args.val_every == 0 and len(val_ds):
                vl = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    for i, vb in enumerate(val_loader):
                        if i >= 20:
                            break
                        vl.append(ema.loss(to_dev(vb))["loss"].item())
                rec["val_loss"] = sum(vl) / len(vl)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if wb:
                wb.log(rec)
        if step % args.save_every == 0 or step == args.steps:
            save(str(step)); save("last"); last_save = time.time()
        elif time.time() - last_save > args.save_minutes * 60:
            save("last"); last_save = time.time()
        if stop["flag"]:
            save("last")
            print(f"SIGTERM: saved at step {step}, exiting", flush=True)
            if wb:
                wb.finish(exit_code=0)
            return
    (out / "DONE").touch()
    if wb:
        wb.finish()
    print("done", flush=True)


if __name__ == "__main__":
    main()
    # persistent DataLoader workers / pin-memory threads can keep the process alive after "done"
    # everything is saved by now, so exit hard.
    sys.stdout.flush()
    os._exit(0)
