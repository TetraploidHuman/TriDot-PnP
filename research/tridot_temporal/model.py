"""Shared conv block, soft-argmax, and Gaussian heatmap targets."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, dw: bool = False) -> None:
        super().__init__()
        if dw:
            self.block = nn.Sequential(
                nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
                nn.BatchNorm2d(cin),
                nn.ReLU(inplace=True),
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def soft_argmax_2d(heat: torch.Tensor) -> torch.Tensor:
    """heat: B,K,H,W logits → B,K,2 (x,y) in [0,1]."""
    b, k, h, w = heat.shape
    prob = F.softmax((heat * 4.0).view(b, k, -1), dim=-1)
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, h, device=heat.device, dtype=heat.dtype),
        torch.linspace(0, 1, w, device=heat.device, dtype=heat.dtype),
        indexing="ij",
    )
    xs = xs.reshape(1, 1, -1)
    ys = ys.reshape(1, 1, -1)
    x = (prob * xs).sum(dim=-1)
    y = (prob * ys).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


def render_gaussian_heatmaps(
    coords: torch.Tensor,
    vis: torch.Tensor,
    height: int,
    width: int,
    sigma: float = 1.5,
) -> torch.Tensor:
    """coords ...×3×2 in [0,1] → matching heatmaps."""
    leading = coords.shape[:-2]
    k = coords.shape[-2]
    c = coords.reshape(-1, k, 2)
    v = vis.reshape(-1, k)
    n = c.shape[0]
    device, dtype = coords.device, coords.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    heat = torch.zeros(n, k, height, width, device=device, dtype=dtype)
    for i in range(k):
        cx = c[:, i, 0] * (width - 1)
        cy = c[:, i, 1] * (height - 1)
        g = torch.exp(
            -((xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2)
            / (2 * sigma ** 2)
        )
        g = g * (v[:, i] > 0.4).to(dtype)[:, None, None]
        heat[:, i] = g
    return heat.reshape(*leading, k, height, width)
