"""Tiny dot-vs-clutter classifier on classic front-end patches.

The classic v3 front-end has a measured 16% front-end miss / 21% selection
error split under ``phone_far``; the wrong triples are mostly coloured
distractor blocks.  A 15x15 patch of the colour-difference map contains enough
shape/contrast information to tell a small round LED dot from a block edge, and
a 3-layer CNN on those patches costs well under 1 ms per candidate on CPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = 15


class DotClassifier(nn.Module):
    def __init__(self, in_ch: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).flatten(1)).squeeze(-1)


class DotScorer:
    """Thin inference wrapper used by the classic detector."""

    def __init__(self, path: str | Path, in_ch: int = 1, device: str = "cpu") -> None:
        self.device = torch.device(device)
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model = DotClassifier(in_ch=int(ckpt.get("in_ch", in_ch)))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.threshold = float(ckpt.get("threshold", 0.5))
        self.meta = ckpt.get("meta", {})

    @torch.no_grad()
    def predict(self, patches: np.ndarray) -> np.ndarray:
        if len(patches) == 0:
            return np.zeros(0, dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(patches, dtype=np.float32))
        if x.ndim == 3:
            x = x.unsqueeze(1)
        logits = self.model(x.to(self.device)).detach().cpu().numpy()
        return 1.0 / (1.0 + np.exp(-logits))


def patch_from_map_scaled(m: np.ndarray, u: float, v: float,
                          expected_radius_px: float) -> np.ndarray:
    """Scale-normalised crop: window sized by the expected dot radius.

    Mirrors ``research/scripts/build_dot_dataset._patch_scaled`` so training and
    inference see the same canonical dot size (round 11 fix for the small-dot
    regression).
    """
    import cv2

    if expected_radius_px <= 1e-3:
        return patch_from_map(m, u, v)
    scale = float(np.clip(expected_radius_px / 3.0, 0.45, 2.5))
    win = max(5, int(round(PATCH * scale)))
    half = win // 2
    h, w = m.shape
    ui, vi = int(round(u)), int(round(v))
    x0, y0 = max(0, ui - half), max(0, vi - half)
    x1, y1 = min(w, ui - half + win), min(h, vi - half + win)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((PATCH, PATCH), dtype=np.float32)
    crop = m[y0:y1, x0:x1]
    interp = cv2.INTER_AREA if crop.shape[0] > PATCH else cv2.INTER_LINEAR
    out = cv2.resize(crop.astype(np.float32), (PATCH, PATCH), interpolation=interp)
    return out / 255.0


def patch_from_map(m: np.ndarray, u: float, v: float) -> np.ndarray:
    h, w = m.shape
    half = PATCH // 2
    ui, vi = int(round(u)), int(round(v))
    x0, y0 = ui - half, vi - half
    out = np.zeros((PATCH, PATCH), dtype=np.float32)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(w, x0 + PATCH), min(h, y0 + PATCH)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = m[ya:yb, xa:xb]
    return out / 255.0
