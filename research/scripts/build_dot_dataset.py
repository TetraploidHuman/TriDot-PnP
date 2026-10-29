"""Build a dot-vs-clutter patch dataset from the classic v3 candidate lists.

Each sample is a 15x15 patch of the colour-difference map at a candidate
position; the label is "is this candidate within 4 px of the matching GT dot".
Degradations are applied per sequence (clean / phone_easy / phone_far /
phone_harsh) so the classifier sees the far/dim domain as well.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from research.tridot_temporal import degrade
from research.tridot_temporal import eval_v2 as ev
from research.tridot_temporal.classic_v3 import ClassicV3Detector
import cv2

DOT_RADIUS_RATIO = 0.09  # dot apparent radius / frame edge length, measured on the synth set

COLORS = ("red", "green", "blue")
PATCH = 15
HALF = PATCH // 2
PRESETS = ("clean", "phone_easy", "phone_far", "phone_harsh")


def _patch_scaled(m: np.ndarray, u: float, v: float, expected_radius_px: float) -> np.ndarray:
    """15x15 crop whose *content* is normalised to a canonical dot radius.

    Training patches are taken with a window of ``15 * scale`` original pixels
    and resized back to 15x15, so a 1.5 px far dot and a 10 px near dot both
    appear at roughly the same size.  ``scale`` is derived from the expected
    apparent dot radius (from GT depth), clamped so extreme scales stay sane.
    """
    if expected_radius_px <= 1e-3:
        return _patch(m, u, v)
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


def _patch(m: np.ndarray, u: float, v: float) -> np.ndarray:
    h, w = m.shape
    ui, vi = int(round(u)), int(round(v))
    x0, y0 = ui - HALF, vi - HALF
    out = np.zeros((PATCH, PATCH), dtype=np.float32)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(w, x0 + PATCH), min(h, y0 + PATCH)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = m[ya:yb, xa:xb]
    return out / 255.0


def build(dataset: Path, subsets, limit_seq, max_frames, out: Path, seed: int = 0):
    rng = random.Random(seed)
    pairs = []
    for sub in subsets:
        pairs += ev._discover_pairs(dataset, "synthetic", subset=sub)
    if limit_seq:
        rng.shuffle(pairs)
        pairs = pairs[:limit_seq]
    xs, ys, meta = [], [], []
    det = ClassicV3Detector(frontend="union", contrast_floor=12.0)
    n_frames = 0
    for video, ann_path in pairs:
        ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        cam = ev._camera(ann)
        fx, fy, cx, cy = (ev.camera_fx(cam), ev.camera_fy(cam),
                          ev.camera_cx(cam), ev.camera_cy(cam))
        degrade.set_active(degrade.preset(rng.choice(PRESETS)))
        frames = ev._read_video(video)
        det.reset()
        step = max(1, len(frames) // max(1, max_frames // max(1, len(pairs))))
        for fi in range(0, min(len(frames), len(ann["frames"])), step):
            fr = ann["frames"][fi]
            gt = np.asarray(fr["keypoints"], dtype=float).reshape(3, 3)
            tz = float(fr["pose_gt"]["translation_mm"][2])
            edge_px = fx * float(ann["info"].get("edge_length_mm", 100.0)) / max(tz, 1.0)
            r_exp = DOT_RADIUS_RATIO * edge_px
            bgr = frames[fi]
            maps = det.candidate_maps(bgr)
            cands = det.candidates_from_maps(maps, bgr)
            pos, neg = [], []
            for i, c in enumerate(COLORS):
                lst = cands.get(c) or []
                gx, gy, gv = gt[i]
                for b in lst:
                    d = float(np.hypot(b.u - gx, b.v - gy))
                    p = _patch_scaled(maps[c], b.u, b.v, r_exp)
                    if d <= 4.0 and gv >= 1:
                        pos.append(p)
                    elif d >= 12.0:
                        neg.append(p)
            if len(pos) == 0 and len(neg) == 0:
                continue
            for p in pos:
                xs.append(p); ys.append(1.0); meta.append(1)
            rng.shuffle(neg)
            for p in neg[:max(3, 2 * len(pos))]:
                xs.append(p); ys.append(0.0); meta.append(0)
            n_frames += 1
    degrade.set_active(None)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=np.stack(xs).astype(np.float32),
                        y=np.asarray(ys, dtype=np.float32),
                        meta=np.asarray(meta, dtype=np.int64))
    print(f"wrote {out}: {len(xs)} samples ({int(sum(ys))} positive) "
          f"from {len(pairs)} seqs / {n_frames} frames", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("datasets/TriDotPose"))
    ap.add_argument("--subsets", default="train,hard_train")
    ap.add_argument("--limit-seq", type=int, default=40)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.dataset, [s for s in a.subsets.split(",") if s], a.limit_seq,
          a.max_frames, a.out, a.seed)


if __name__ == "__main__":
    main()
