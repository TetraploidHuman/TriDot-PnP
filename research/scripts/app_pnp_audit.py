#!/usr/bin/env python3
"""Audit the phone app's PnP math with *perfect* detections.

Feeds the ground-truth pixel coordinates into three pose paths and reports the
error each one makes when the detector is not part of the problem:

* ``app``    — the shipped Kotlin solver, ported 1:1 in
  :mod:`research.tridot_temporal.app_pnp` (multiplicative depth update);
* ``fixed``  — the same geometry, solved with a damped Newton step;
* ``b0``     — OpenCV ``solvePnP`` as used by the research baseline (reference).

If ``app`` is off by a factor of two on noiseless GT points, then no amount of
detector improvement can fix the app's DISTANCE readout — that is the point of
this audit.

Usage::

    research/.venv/bin/python research/scripts/app_pnp_audit.py
    research/.venv/bin/python research/scripts/app_pnp_audit.py --subsets test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.tridot_temporal.app_pnp import (
    app_solve_depths,
    distance_from_points,
    estimate_focal_length,
    fixed_solve_depths,
    solve_pose_from_depths,
)

DATASET = ROOT / "datasets" / "TriDotPose"
OUT = ROOT / "research" / "results" / "app_pnp_audit"


def _stats(xs: np.ndarray) -> dict:
    xs = np.asarray([x for x in xs if np.isfinite(x)], dtype=float)
    if xs.size == 0:
        return {"n": 0}
    return {
        "n": int(xs.size),
        "mean": float(xs.mean()),
        "median": float(np.median(xs)),
        "p90": float(np.percentile(xs, 90)),
        "max": float(xs.max()),
    }


def _b0_reference(uv, edge, cam):
    from research.pyc_import import install
    install()
    from research.baseline_b0.pnp import PoseSolver
    from research.tridot_temporal.eval_find import B0Detector, PoseSolver as _PS

    pts = {c: (float(uv[i, 0]), float(uv[i, 1])) for i, c in enumerate(("red", "green", "blue"))}
    solver = _PS()
    r = solver.solve(pts, edge, cam["fx"], cam["fy"], cam["cx"], cam["cy"],
                     np.array(cam.get("distortion", [0, 0, 0, 0, 0])))
    if not r.ok:
        return None
    t = np.asarray(r.translation_mm, dtype=float)
    return {"t": t, "dist": float(np.linalg.norm(t))}


def audit_sequence(ann_path: Path) -> list[dict]:
    ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    cam = ann["camera"]
    edge = float(ann["info"]["edge_length_mm"])
    w, h = int(ann["info"]["width"]), int(ann["info"]["height"])
    fx = estimate_focal_length(w, 60.0)  # app default FOV; GT is 60.0 anyway
    rows = []
    for fi, fr in enumerate(ann["frames"]):
        kp = fr["keypoints"]
        vis = [kp[0], kp[1], kp[2], kp[3], kp[4], kp[5], kp[6], kp[7], kp[8]]
        # only frames where the app would have all three dots
        if not (vis[2] >= 2 and vis[5] >= 2 and vis[8] >= 2):
            continue
        uv = np.array([[kp[0], kp[1]], [kp[3], kp[4]], [kp[6], kp[7]]])
        gt_t = np.asarray(fr["pose_gt"]["translation_mm"], dtype=float)
        gt_dist = float(np.linalg.norm(gt_t))
        row = {"seq": ann["info"]["seq_id"], "frame": fi, "gt_dist_mm": gt_dist}
        r_app = app_solve_depths(uv, edge, fx, fx, cam["cx"], cam["cy"])
        P, _, t_app = solve_pose_from_depths(uv, r_app.depths_mm, edge, fx, fx,
                                             cam["cx"], cam["cy"])
        r_fix = fixed_solve_depths(uv, edge, fx, fx, cam["cx"], cam["cy"])
        P2, _, t_fix = solve_pose_from_depths(uv, r_fix.depths_mm, edge, fx, fx,
                                              cam["cx"], cam["cy"])
        row.update({
            "app": {
                "dist": distance_from_points(P),
                "t_err": float(np.linalg.norm(t_app - gt_t)),
                "converged": r_app.converged,
                "iters": r_app.iterations,
                "edge_err_mm": r_app.max_abs_edge_err_mm,
            },
            "fixed": {
                "dist": distance_from_points(P2),
                "t_err": float(np.linalg.norm(t_fix - gt_t)),
                "converged": r_fix.converged,
                "iters": r_fix.iterations,
                "edge_err_mm": r_fix.max_abs_edge_err_mm,
            },
        })
        ref = _b0_reference(uv, edge, cam)
        if ref:
            row["b0"] = {"dist": ref["dist"],
                         "t_err": float(np.linalg.norm(ref["t"] - gt_t))}
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for key in ("app", "fixed", "b0"):
        have = [r for r in rows if key in r]
        if not have:
            continue
        dd = np.array([abs(r[key]["dist"] / r["gt_dist_mm"] - 1.0) for r in have])
        te = np.array([r[key]["t_err"] for r in have])
        out[key] = {
            "n": len(have),
            "dist_rel_err": _stats(dd),
            "dist_rel_err_le_5pct": float((dd <= 0.05).mean()),
            "dist_rel_err_le_10pct": float((dd <= 0.10).mean()),
            "t_err_mm": _stats(te),
            "posed_ok": float((te < 50.0).mean()),
        }
        if "converged" in have[0][key]:
            out[key]["converged_rate"] = float(np.mean([r[key]["converged"] for r in have]))
            out[key]["iters_median"] = float(np.median([r[key]["iters"] for r in have]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="test,hard_test")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    splits = json.loads((DATASET / "meta" / "splits.json").read_text(encoding="utf-8"))
    report = {}
    for subset in [s.strip() for s in args.subsets.split(",") if s.strip()]:
        rows, per_seq = [], {}
        for seq in splits[subset]:
            ann_path = DATASET / "synthetic" / "annotations" / f"{seq}.json"
            if not ann_path.exists():
                continue
            srows = audit_sequence(ann_path)
            rows.extend(srows)
            per_seq[seq] = summarize(srows)
        report[subset] = {"summary": summarize(rows), "per_sequence": per_seq,
                          "n_frames": len(rows)}
        s = report[subset]["summary"]
        print(f"[{subset}] frames={len(rows)}")
        for key in ("app", "fixed", "b0"):
            if key not in s:
                continue
            v = s[key]
            print(f"  {key:6s} dist_rel_err median={v['dist_rel_err']['median']:.3f} "
                  f"p90={v['dist_rel_err']['p90']:.3f} <=5%={v['dist_rel_err_le_5pct']:.3f} "
                  f"t_err median={v['t_err_mm']['median']:.1f}mm "
                  f"pose<50mm={v['posed_ok']:.3f}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
