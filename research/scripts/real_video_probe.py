"""Probe the one real video we have (no labels): which degradation preset is
realistic, and does the contrast front-end find the dots there?

Usage::

    PYTHONPATH=. research/.venv/bin/python -m research.scripts.real_video_probe
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from research.tridot_temporal import degrade
from research.tridot_temporal.classic_v2 import ClassicV2Detector, color_weight_maps
from research.tridot_temporal.classic_v3 import ClassicV3Detector
from research.baseline_b0.pnp import PoseSolver

ROOT = Path(__file__).resolve().parents[2]
REAL_VIDEO = ROOT / "datasets/TriDotPose/real/videos/smoke_real_desk.mp4"
SYNTH_DIR = ROOT / "datasets/TriDotPose/synthetic/videos"
OUT = ROOT / "research/results/real_probe"
FX, FY, CX, CY = 1108.5, 1108.5, 639.5, 359.5


def read_frames(path: Path, limit: int | None = None):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(bgr)
            if limit and len(frames) >= limit:
                break
    finally:
        cap.release()
    return frames


def frame_stats(bgr: np.ndarray) -> dict:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F).var()
    maps = color_weight_maps(bgr)
    chroma = float(max(np.percentile(maps[c], 99.5) for c in maps))
    return {
        "mean_v": float(v.mean()),
        "gray_std": float(gray.std()),
        "laplacian_var": float(lap),
        "sat_ratio": float((v >= 250).mean()),
        "chroma_p995": chroma,
    }


def collect_points(frames, factory):
    det = factory()
    out = []
    for bgr in frames:
        d = det.detect(bgr, FX, FY, CX, CY, PoseSolver())
        pts = dict(getattr(d, "points", {}) or {}) if getattr(d, "ok", False) else {}
        out.append(pts if len(pts) == 3 else None)
    return out


def detect_rate(frames, factory, label: str, reference=None) -> dict:
    det = factory()
    ok = 0
    edges = []
    lat = []
    for bgr in frames:
        import time
        t0 = time.perf_counter()
        d = det.detect(bgr, FX, FY, CX, CY, PoseSolver())
        lat.append((time.perf_counter() - t0) * 1000.0)
        if getattr(d, "ok", False) and len(getattr(d, "points", {}) or {}) == 3:
            ok += 1
            pts = [d.points[c] for c in ("red", "green", "blue")]
            e = [math.hypot(pts[i][0] - pts[(i + 1) % 3][0], pts[i][1] - pts[(i + 1) % 3][1])
                 for i in range(3)]
            edges.append(float(np.mean(e)))
    n = max(len(frames), 1)
    drift, edge_err = [], []
    if reference is not None:
        det = factory()
        for bgr, ref in zip(frames, reference):
            d = det.detect(bgr, FX, FY, CX, CY, PoseSolver())
            pts = dict(getattr(d, "points", {}) or {}) if getattr(d, "ok", False) else {}
            if len(pts) != 3 or ref is None:
                continue
            for c in ("red", "green", "blue"):
                drift.append(math.hypot(pts[c][0] - ref[c][0], pts[c][1] - ref[c][1]))
            cur = [pts[c] for c in ("red", "green", "blue")]
            refp = [ref[c] for c in ("red", "green", "blue")]
            cur_e = float(np.mean([math.hypot(cur[i][0] - cur[(i + 1) % 3][0],
                                              cur[i][1] - cur[(i + 1) % 3][1]) for i in range(3)]))
            ref_e = float(np.mean([math.hypot(refp[i][0] - refp[(i + 1) % 3][0],
                                              refp[i][1] - refp[(i + 1) % 3][1]) for i in range(3)]))
            edge_err.append(abs(cur_e - ref_e))
    return {
        "method": label,
        "n_frames": len(frames),
        "detected_3_spots": ok,
        "det_rate": ok / n,
        "edge_px_med": float(np.median(edges)) if edges else None,
        "latency_ms_med": float(np.median(lat)) if lat else None,
        "point_drift_px_med": float(np.median(drift)) if drift else None,
        "edge_err_px_med": float(np.median(edge_err)) if edge_err else None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"video": str(REAL_VIDEO.name)}

    real = read_frames(REAL_VIDEO, limit=60)
    report["real_frames"] = len(real)
    report["real_stats"] = _mean_stats(real)

    # same-length synthetic reference frames (clean), so the comparison is fair
    syn_clean = read_frames(SYNTH_DIR / "syn_room_0200.mp4", limit=len(real))
    report["synth_clean_stats"] = _mean_stats(syn_clean)

    # what does the real video look like after the presets (for calibration only)
    report["real_plus_preset"] = {}
    for preset in ("phone_easy", "phone_far", "phone_harsh"):
        cfg = degrade.preset(preset)
        report["real_plus_preset"][preset] = _mean_stats(
            [degrade.apply(f, cfg, seed=i) for i, f in enumerate(real)])

    # detection on the real frames (no labels, but "found all three" is a signal)
    # real content + simulated capture: does the contrast front-end hold up?
    reference = collect_points(real, ClassicV3Detector)
    report["detection"] = [
        detect_rate(real, ClassicV2Detector, "classic2", reference),
        detect_rate(real, ClassicV3Detector, "classic3", reference),
    ]
    report["detection_degraded"] = []
    for preset in ("phone_easy", "phone_far", "phone_harsh"):
        cfg = degrade.preset(preset)
        frames = [degrade.apply(f, cfg, seed=i) for i, f in enumerate(real)]
        report["detection_degraded"].append(
            detect_rate(frames, ClassicV2Detector, f"classic2+{preset}", reference))
        report["detection_degraded"].append(
            detect_rate(frames, ClassicV3Detector, f"classic3+{preset}", reference))
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("detection", "detection_degraded")}, indent=2))
    for row in report["detection"] + report["detection_degraded"]:
        print(row)
    print(f"wrote {OUT / 'summary.json'}")


def _mean_stats(frames) -> dict:
    keys = ("mean_v", "gray_std", "laplacian_var", "sat_ratio", "chroma_p995")
    stats = [frame_stats(f) for f in frames]
    return {k: round(float(np.mean([s[k] for s in stats])), 2) for k in keys}


if __name__ == "__main__":
    main()
