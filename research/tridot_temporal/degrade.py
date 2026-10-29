"""Realistic capture degradations for far / small-dot evaluation.

The synthetic renderer is ideal: no sensor noise, no chroma subsampling, perfect
exposure, no ISP.  Real phone video degrades small coloured dots precisely in
those dimensions, which is why a detector that looks fine on the synthetic
"far" factor can fail on a real target at 5 m.  These degradations are applied
*at eval time* to existing frames, so no dataset regeneration is needed and the
same ground truth still applies (degradation is photometric only).

Order of operations mirrors a real pipeline:

    optical (defocus / motion blur)
      -> sensor (downscale, exposure, shot+read noise, chroma subsampling)
        -> ISP (denoise, sharpen, JPEG)

Usage::

    from research.tridot_temporal import degrade
    degrade.set_active(degrade.preset("phone_far"))
    ...
    degrade.set_active(None)          # back to clean frames

or on the command line::

    eval_v2.py --degrade phone_far ...
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class DegradeConfig:
    """One capture pipeline profile; all knobs are no-ops at their defaults."""

    name: str = "clean"

    # optical
    blur_sigma: float = 0.0

    # sensor
    downscale: float = 1.0        # >1: simulate a lower sensor resolution (down+up)
    exposure_gain: float = 1.0
    exposure_offset: float = 0.0
    read_noise: float = 0.0       # gaussian DN (0..255 scale)
    shot_noise: float = 0.0       # sqrt(signal) coefficient (DN)
    chroma: str = "444"           # "444" | "420"

    # ISP
    denoise: float = 0.0          # 0..1: blend toward a median-filtered copy
    sharpen: float = 0.0           # unsharp-mask amount
    jpeg_quality: int = 100        # 100 = off

    def is_identity(self) -> bool:
        return (self.blur_sigma == 0.0 and self.downscale == 1.0
                and self.exposure_gain == 1.0 and self.exposure_offset == 0.0
                and self.read_noise == 0.0 and self.shot_noise == 0.0
                and self.chroma == "444" and self.denoise == 0.0
                and self.sharpen == 0.0 and self.jpeg_quality >= 100)


PRESETS = {
    "clean": DegradeConfig(name="clean"),
    # decent phone, native resolution, mild processing: what the synthetic set
    # more or less looks like today
    "phone_easy": DegradeConfig(
        name="phone_easy", chroma="420", exposure_gain=0.9, read_noise=1.5,
        shot_noise=0.6, denoise=0.25, sharpen=0.35, jpeg_quality=92),
    # long range on a phone: dot covers 1-3 px, chroma is half resolution, the
    # ISP removes single-pixel highlights and the JPEG kills the rest
    "phone_far": DegradeConfig(
        name="phone_far", downscale=2.0, chroma="420", exposure_gain=0.72,
        read_noise=2.5, shot_noise=1.0, denoise=0.55, sharpen=0.6,
        jpeg_quality=85),
    # worst case: dark scene, aggressive NR, heavy compression
    "phone_harsh": DegradeConfig(
        name="phone_harsh", downscale=2.0, chroma="420", exposure_gain=0.55,
        read_noise=4.0, shot_noise=1.6, denoise=0.85, sharpen=0.8,
        jpeg_quality=75),
}


def preset(name: str) -> DegradeConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown degrade preset {name!r}; have {sorted(PRESETS)}")
    return PRESETS[name]


# --------------------------------------------------------------------------- #
# individual stages
# --------------------------------------------------------------------------- #
def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 255.0).astype(np.uint8)


def exposure(frame: np.ndarray, gain: float, offset: float) -> np.ndarray:
    if gain == 1.0 and offset == 0.0:
        return frame
    return _clip(frame.astype(np.float32) * gain + offset)


def defocus(frame: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return frame
    return cv2.GaussianBlur(frame, (0, 0), sigma)


def downscale_upscale(frame: np.ndarray, factor: float) -> np.ndarray:
    """Simulate a lower-resolution sensor: area-average down, bilinear back up."""
    if factor <= 1.0:
        return frame
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(2, int(round(w / factor))),
                               max(2, int(round(h / factor)))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def add_noise(frame: np.ndarray, read_noise: float, shot_noise: float,
              seed: Optional[int] = None) -> np.ndarray:
    if read_noise <= 0 and shot_noise <= 0:
        return frame
    rng = np.random.default_rng(seed)
    f = frame.astype(np.float32)
    if shot_noise > 0:
        f = f + rng.normal(0.0, 1.0, f.shape).astype(np.float32) * \
            np.sqrt(np.maximum(f, 1.0)) * shot_noise
    if read_noise > 0:
        f = f + rng.normal(0.0, read_noise, f.shape).astype(np.float32)
    return _clip(f)


def chroma_subsample(frame: np.ndarray, mode: str = "420") -> np.ndarray:
    """Round-trip through 4:2:0 YUV (half-resolution chroma, full-res luma)."""
    if mode != "420":
        return frame
    h, w = frame.shape[:2]
    # cv2's I420 conversion needs even dimensions
    pad_h, pad_w = h % 2, w % 2
    f = frame
    if pad_h or pad_w:
        f = cv2.copyMakeBorder(f, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    yuv = cv2.cvtColor(f, cv2.COLOR_BGR2YUV_I420)
    back = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return back[:h, :w]


def denoise(frame: np.ndarray, strength: float) -> np.ndarray:
    """Blend toward a 3x3 median: what strong temporal/spatial NR does to a 1px dot."""
    if strength <= 0:
        return frame
    med = cv2.medianBlur(frame, 3)
    return _clip(frame.astype(np.float32) * (1.0 - strength)
                 + med.astype(np.float32) * strength)


def sharpen(frame: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return frame
    blur = cv2.GaussianBlur(frame, (0, 0), 1.0)
    return _clip(frame.astype(np.float32) * (1.0 + amount)
                 - blur.astype(np.float32) * amount)


def jpeg(frame: np.ndarray, quality: int) -> np.ndarray:
    if quality >= 100:
        return frame
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return frame
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def apply(frame: np.ndarray, cfg: DegradeConfig, seed: Optional[int] = None) -> np.ndarray:
    """Apply one capture pipeline profile (idempotent at the defaults)."""
    if cfg.is_identity():
        return frame
    out = defocus(frame, cfg.blur_sigma)
    out = downscale_upscale(out, cfg.downscale)
    out = exposure(out, cfg.exposure_gain, cfg.exposure_offset)
    out = add_noise(out, cfg.read_noise, cfg.shot_noise, seed=seed)
    out = chroma_subsample(out, cfg.chroma)
    out = denoise(out, cfg.denoise)
    out = sharpen(out, cfg.sharpen)
    out = jpeg(out, cfg.jpeg_quality)
    return out


# --------------------------------------------------------------------------- #
# global active profile (used by eval_v2's video reader)
# --------------------------------------------------------------------------- #
_ACTIVE: Optional[DegradeConfig] = None


def set_active(cfg: Optional[DegradeConfig]) -> None:
    global _ACTIVE
    _ACTIVE = cfg


def get_active() -> Optional[DegradeConfig]:
    return _ACTIVE


def describe(cfg: Optional[DegradeConfig]) -> Optional[dict]:
    return None if cfg is None else asdict(cfg)


def apply_to_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    cfg = _ACTIVE
    if cfg is None or cfg.is_identity():
        return frames
    # deterministic per frame index, so repeated runs are comparable
    return [apply(fr, cfg, seed=i) for i, fr in enumerate(frames)]
