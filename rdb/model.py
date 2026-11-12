"""Multi-task chunked action policy: camera tokens + proprio + task token ->
transformer encoder -> action-chunk decoder (flow matching or L1 regression).

Self-contained (torch + torchvision only) so it ships inside policy_bundle/policy/
and runs unchanged on CUDA and ROCm.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torchvision

STATE_DIM = 36 + 2  # joint qpos+qvel (both arms) + gripper state
ACTION_DIM = 14     # [l xyz, l rotvec, r xyz, r rotvec, l grip, r grip]
GRIP_DIMS = (12, 13)


@dataclass
class PolicyConfig:
    n_cams: int = 3
    img_size: int = 128
    n_tasks: int = 8
    chunk: int = 16
    d_model: int = 384
    n_heads: int = 6
    enc_layers: int = 4
    dec_layers: int = 4
    dropout: float = 0.1
    head: str = "flow"            # "flow" | "l1"
    flow_steps: int = 10
    pretrained_backbone: bool = True
    encoder: str = "resnet18"
    enc_pool: int = 0       # unused; kept so saved checkpoint configs load
    # normalisation stats, filled from data before training
    state_mean: list = field(default_factory=lambda: [0.0] * STATE_DIM)
    state_std: list = field(default_factory=lambda: [1.0] * STATE_DIM)
    action_mean: list = field(default_factory=lambda: [0.0] * ACTION_DIM)
    action_std: list = field(default_factory=lambda: [1.0] * ACTION_DIM)

    def to_dict(self) -> dict:
        return asdict(self)


class CameraEncoder(nn.Module):
    """Shared camera trunk -> d_model tokens with learned per-camera positions."""

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.encoder = cfg.encoder
        if cfg.encoder == "resnet18":
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained_backbone else None
            # Frozen BN (as in ACT/DETR): pretrained statistics, identical in train and eval
            r = torchvision.models.resnet18(weights=weights, norm_layer=torchvision.ops.FrozenBatchNorm2d)
            self.trunk = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
            self.proj = nn.Conv2d(512, cfg.d_model, 1)
            n_tok = (cfg.img_size // 32) ** 2
        else:
            raise ValueError(f"Unknown camera encoder: {cfg.encoder}")
        self.pos = nn.Parameter(torch.randn(1, cfg.n_cams, n_tok, cfg.d_model) * 0.02)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: uint8 [B, n_cams, 3, H, W] -> [B, n_cams*n_tok, d]."""
        b, n = imgs.shape[:2]
        x = imgs.flatten(0, 1).float().div_(255.0)
        x = (x - self.mean) / self.std
        x = self.proj(self.trunk(x))                      # [B*n, d, h, w]
        x = x.flatten(2).transpose(1, 2)                  # [B*n, h*w, d]
        x = x.view(b, n, -1, x.shape[-1]) + self.pos
        return x.flatten(1, 2)


def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * 1000.0 * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class ChunkPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.cams = CameraEncoder(cfg)
        self.state_in = nn.Sequential(nn.Linear(STATE_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.task_emb = nn.Embedding(cfg.n_tasks, d)
        self.type_emb = nn.Parameter(torch.randn(3, d) * 0.02)  # cam / state / task
        enc_layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, batch_first=True,
                                               norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.enc_layers)
        dec_layer = nn.TransformerDecoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, batch_first=True,
                                               norm_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.dec_layers)
        self.query_pos = nn.Parameter(torch.randn(1, cfg.chunk, d) * 0.02)
        if cfg.head == "flow":
            self.act_in = nn.Linear(ACTION_DIM, d)
            self.time_in = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.act_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ACTION_DIM))
        for name in ("state_mean", "state_std", "action_mean", "action_std"):
            self.register_buffer(name, torch.tensor(getattr(cfg, name), dtype=torch.float32))

    # ---- normalisation -------------------------------------------------
    def norm_state(self, s):
        return (s - self.state_mean) / self.state_std

    def norm_action(self, a):
        return (a - self.action_mean) / self.action_std

    def denorm_action(self, a):
        return a * self.action_std + self.action_mean

    # ---- network -------------------------------------------------------
    def encode(self, imgs, state, task):
        cam = self.cams(imgs) + self.type_emb[0]
        st = self.state_in(self.norm_state(state))[:, None] + self.type_emb[1]
        tk = self.task_emb(task)[:, None] + self.type_emb[2]
        return self.encoder(torch.cat([tk, st, cam], dim=1))

    def decode(self, memory, noisy=None, t=None):
        b = memory.shape[0]
        q = self.query_pos.expand(b, -1, -1)
        if self.cfg.head == "flow":
            q = q + self.act_in(noisy) + self.time_in(sinusoidal(t, self.cfg.d_model))[:, None]
        return self.act_out(self.decoder(q, memory))

    def loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """batch: img uint8 [B,n,3,H,W], state [B,38], task [B], action [B,T,14], mask [B,T]"""
        memory = self.encode(batch["img"], batch["state"], batch["task"])
        target = self.norm_action(batch["action"])
        mask = batch["mask"].float()[..., None]
        if self.cfg.head == "flow":
            b = target.shape[0]
            # beta-ish time sampling that emphasises the noisy end, as in pi0
            t = torch.distributions.Beta(1.5, 1.0).sample((b,)).to(target.device) * 0.999 + 0.001
            noise = torch.randn_like(target)
            x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * target
            v = self.decode(memory, x_t, t)
            err = (v - (noise - target)) ** 2
        else:
            pred = self.decode(memory)
            err = (pred - target).abs()
        loss = (err * mask).sum() / (mask.sum() * ACTION_DIM).clamp(min=1)
        return {"loss": loss}

    @torch.no_grad()
    def sample(self, imgs, state, task, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Returns de-normalised action chunk [B, T, 14]."""
        memory = self.encode(imgs, state, task)
        b = memory.shape[0]
        if self.cfg.head == "flow":
            x = torch.randn(b, self.cfg.chunk, ACTION_DIM, device=memory.device, generator=generator)
            n = self.cfg.flow_steps
            for i in range(n):
                t = torch.full((b,), 1.0 - i / n, device=memory.device)
                x = x - self.decode(memory, x, t) / n
        else:
            x = self.decode(memory)
        return self.denorm_action(x)
