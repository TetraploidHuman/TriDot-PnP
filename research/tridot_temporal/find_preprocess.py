"""Dense photometric maps for the finder. No blob extraction."""

from __future__ import annotations

import cv2
import numpy as np

_TARGETS_DEG = {
    "red": (0.0, 0.55, 0.95),
    "green": (120.0, 0.55, 0.95),
    "blue": (225.0, 0.55, 0.95),
}


def _hsv_sim_map(h180: np.ndarray, s: np.ndarray, v: np.ndarray, target_deg) -> np.ndarray:
    th, ts, tv = target_deg
    h360 = h180.astype(np.float32) * 2.0
    dh = np.abs(h360 - th)
    dh = np.minimum(dh, 360.0 - dh) / 180.0
    ds = np.abs(s.astype(np.float32) / 255.0 - ts)
    dv = np.abs(v.astype(np.float32) / 255.0 - tv)
    gate = ((s >= 40) & (v >= 25)).astype(np.float32)
    d = 1.0 * dh * dh + 0.6 * ds * ds + 0.3 * dv * dv
    return np.exp(-8.0 * d).astype(np.float32) * gate


def build_find_input(bgr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """CHW float32, C=7: RGB, R/G/B affinity, luma boost."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blur = cv2.GaussianBlur(v, (0, 0), 1.2)
    dog = np.clip((v.astype(np.float32) - blur.astype(np.float32)) / 255.0 + 0.5, 0, 1)
    luma = v.astype(np.float32) / 255.0
    x = np.zeros((7, out_h, out_w), dtype=np.float32)
    x[0:3] = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    x[3] = _hsv_sim_map(h, s, v, _TARGETS_DEG["red"])
    x[4] = _hsv_sim_map(h, s, v, _TARGETS_DEG["green"])
    x[5] = _hsv_sim_map(h, s, v, _TARGETS_DEG["blue"])
    x[6] = np.clip(luma * dog, 0, 1)
    return x


FIND_IN_CH = 7
