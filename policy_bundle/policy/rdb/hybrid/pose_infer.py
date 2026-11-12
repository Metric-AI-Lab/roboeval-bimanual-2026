"""Run a pose regressor trained by rdb.hybrid.train_pose on one standard observation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torchvision

from rdb.common import resize_image


class PoseNet(nn.Module):
    def __init__(self, n_cams, d_prop, d_out, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        r.fc = nn.Identity()
        self.enc, self.n_cams = r, n_cams
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.head = nn.Sequential(nn.Linear(512 * n_cams + d_prop, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(),
                                  nn.Linear(256, d_out))

    def forward(self, img, prop):  # img: (B, C, 3, H, W) float in [0, 1]; prop already normalised
        b = img.shape[0]
        f = self.enc((img.flatten(0, 1) - self.mean) / self.std).view(b, -1)
        return self.head(torch.cat([f, prop], 1))


def axis_from_outer(m: np.ndarray) -> np.ndarray:
    """Unit axes (N, 3) from predicted u u^T codes (N, 6): principal eigenvector, sign arbitrary."""
    a = np.empty((len(m), 3, 3))
    a[:, 0, 0], a[:, 1, 1], a[:, 2, 2] = m[:, 0], m[:, 1], m[:, 2]
    a[:, 0, 1] = a[:, 1, 0] = m[:, 3]
    a[:, 0, 2] = a[:, 2, 0] = m[:, 4]
    a[:, 1, 2] = a[:, 2, 1] = m[:, 5]
    return np.linalg.eigh(a)[1][:, :, -1]


def quat_y_to(u: np.ndarray) -> np.ndarray:
    """A wxyz quaternion whose local y axis points along u (shortest arc from world y)."""
    u = u / np.linalg.norm(u)
    if u[1] < 0:            # the sign is free: pick the hemisphere around +y so the arc is well defined
        u = -u
    w = 1.0 + u[1]
    q = np.array([w, u[2], 0.0, -u[0]])   # y x u = (u_z, 0, -u_x)
    return q / np.linalg.norm(q)


class PoseEstimator:
    def __init__(self, path: Path, device: torch.device):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.cams, self.size, self.spec, self.device = ck["cams"], ck["size"], ck["spec"], device
        self.pm, self.ps, self.ym, self.ys = (ck[k].to(device) for k in ("prop_mean", "prop_std", "y_mean", "y_std"))
        self.net = PoseNet(len(self.cams), len(ck["prop_mean"]), len(ck["y_mean"]), pretrained=False)
        self.net.load_state_dict(ck["state"])
        self.net.to(device).eval()

    @torch.no_grad()
    def __call__(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        imgs = np.stack([resize_image(np.ascontiguousarray(obs[c if c in obs else f"rgb_{c}"].transpose(1, 2, 0)), self.size).transpose(2, 0, 1)
                         for c in self.cams])
        img = torch.from_numpy(imgs)[None].to(self.device).float() / 255
        prop = torch.from_numpy(np.concatenate([obs["proprioception"], obs["proprioception_grippers"]])
                                .astype(np.float32))[None].to(self.device)
        y = (self.net(img, (prop - self.pm) / self.ps) * self.ys + self.ym)[0].cpu().numpy()
        out, j = {}, 0
        for k, d in self.spec:
            name, _, f = k.partition(":")
            if name.endswith("yaw"):  # "<name>_yaw:4": angle regressed as 4 * yaw
                out[name] = float(np.arctan2(y[j], y[j + 1])) / (int(f) if f else 1)
            elif name.endswith("quat_wxyz") and f == "axisy":   # sign-free long axis -> a quaternion with that y axis
                out[name] = quat_y_to(axis_from_outer(y[None, j:j + d])[0])
            elif name.endswith("quat_wxyz"):
                out[k] = y[j:j + d] / np.linalg.norm(y[j:j + d])
            else:
                out[name] = y[j:j + d]
            j += d
        return out
