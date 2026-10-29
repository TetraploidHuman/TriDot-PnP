#!/usr/bin/env python
"""Diagnose fixed vs tracked(smoothed) vs disamb(raw output) on a few sequences.

Keeps the detector fixed (B1 once per frame) so only the solver differs, then
prints per-variant error stats plus the tracker's accept/coast behaviour.

Usage:
    research/.venv/bin/python research/scripts/temporal_diag.py \
        --subset test --seqs syn_blur_0026 syn_noise_0028 syn_mixed_0033
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.baseline_b1.detector import B1Detector  # noqa: E402
from research.tridot_temporal.temporal import (  # noqa: E402
    AppTrackedSolver,
    rotation_angle_deg,
)
from research.tridot_temporal.app_pnp import (  # noqa: E402
    COLOR_ORDER,
    AppFixedSolver,
)

VID = ROOT / "datasets" / "TriDotPose" / "synthetic" / "videos"
ANN = ROOT / "datasets" / "TriDotPose" / "synthetic" / "annotations"
SPLITS = ROOT / "datasets" / "TriDotPose" / "meta" / "splits.json"


def t_err_stats(errs: list[float]) -> dict:
    a = np.asarray([e for e in errs if e is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            "within_50mm": float(np.mean(a <= 50.0))}


def run_seq(seq: str, variants: dict[str, object], edge_mm: float):
    video = VID / f"{seq}.mp4"
    ann = json.loads((ANN / f"{seq}.json").read_text())
    cam = ann["camera"]
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    cap = cv2.VideoCapture(str(video))
    det = B1Detector(grid_size=32)
    frames = {v: [] for v in variants}
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        fr = ann["frames"][idx] if idx < len(ann["frames"]) else None
        idx += 1
        if fr is None:
            break
        r = det.detect(bgr)
        if not r.ok:
            continue
        gt = np.asarray(fr["keypoints"], dtype=float).reshape(3, 3)
        gt_t = np.asarray(fr["pose_gt"]["translation_mm"], dtype=float)
        gt_r = np.asarray(fr["pose_gt"]["rotation_matrix"], dtype=float)
        for name, solver in variants.items():
            out = solver.solve(r.points, edge_mm, ann["info"]["width"],
                               ann["info"]["height"], t_s=idx / 30.0) \
                if name != "fixed" else solver.solve(r.points, edge_mm,
                                                     ann["info"]["width"],
                                                     ann["info"]["height"])
            if not getattr(out, "ok", False) or out.translation_mm is None:
                continue
            te = float(np.linalg.norm(np.asarray(out.translation_mm) - gt_t))
            re = rotation_angle_deg(out.rotation, gt_r)
            frames[name].append({"t_err": te, "rot_err": re,
                                 "accepted": getattr(out, "accepted", True),
                                 "coasting": getattr(out, "coasting", False)})
    cap.release()
    return frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="test")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "results" / "temporal_ab")
    a = ap.parse_args(argv)
    widths = {"test": 1280, "hard_test": 1280}
    edge = 100.0
    for seq in a.seqs:
        ann = json.loads((ANN / f"{seq}.json").read_text())
        variants = {
            "fixed": AppFixedSolver(),
            "tracked": AppTrackedSolver(smooth=True),
            "disamb": AppTrackedSolver(smooth=False),
        }
        rows = run_seq(seq, variants, edge)
        print(f"\n== {seq}  factor={ann['info']['factor']} frames={len(rows['fixed'])}")
        for name, frs in rows.items():
            if not frs:
                continue
            print(f"  {name:8s} t_err med {np.median([f['t_err'] for f in frs]):7.1f} "
                  f"p90 {np.percentile([f['t_err'] for f in frs], 90):7.1f}  "
                  f"rot med {np.median([f['rot_err'] for f in frs]):6.1f}  "
                  f"acc {sum(f['accepted'] for f in frs)}/{len(frs)} "
                  f"coast {sum(f['coasting'] for f in frs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
