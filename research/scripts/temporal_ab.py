"""A/B the phone-app PnP front end with and without temporal tracking.

Runs the *same* B1 detections through two solvers frame by frame:

* ``fixed``   - robust depth solver only (no tracking, no root selection)
* ``tracked`` - robust depth solver + P3P root disambiguation + alpha-beta filter

Reports the metrics that motivated the tracker: pose (rotation) flips,
rotation error vs GT, translation error/jitter, and latency.

Usage::

    research/.venv/bin/python -m research.scripts.temporal_ab --subset test
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.pyc_import import install  # noqa: E402
from research.tridot_temporal.app_pnp import AppFixedSolver  # noqa: E402
from research.tridot_temporal.temporal import AppTrackedSolver, rotation_angle_deg  # noqa: E402

DEFAULT_DATASET = ROOT / "datasets" / "TriDotPose"
ANN = DEFAULT_DATASET / "synthetic" / "annotations"
VID = DEFAULT_DATASET / "synthetic" / "videos"


def _sequence_pairs(subset: str) -> list[tuple[Path, Path]]:
    splits = json.loads((DEFAULT_DATASET / "meta" / "splits.json").read_text())
    out = []
    for seq in splits[subset]:
        ann = ANN / f"{seq}.json"
        vid = VID / f"{seq}.mp4"
        if ann.exists() and vid.exists():
            out.append((vid, ann))
    return out


def _read_video(path: Path) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


DEFAULT_TRACKER = dict(alpha=0.85, gate_abs_mm=60.0, gate_rel=0.08, smooth=True)


def run(subset: str, *, grid_size: int = 32, limit_seq: int | None = None,
        detect_stride: int = 1, tracker: dict | None = None) -> dict:
    install()
    from research.baseline_b1.detector import B1Detector

    pairs = _sequence_pairs(subset)
    if limit_seq:
        pairs = pairs[:limit_seq]
    tracker_kwargs = dict(DEFAULT_TRACKER)
    if tracker:
        tracker_kwargs.update(tracker)
    rows = {"fixed": [], "tracked": [], "disamb": []}
    per_seq = {}
    for si, (vid, ann_path) in enumerate(pairs):
        ann = json.loads(ann_path.read_text())
        info = ann["info"]
        fps = float(info.get("fps", 30) or 30)
        edge = float(info["edge_length_mm"])
        frames = _read_video(vid)
        n_frames = min(len(frames), len(ann["frames"]))
        frames = frames[:n_frames]
        det = B1Detector(grid_size=grid_size)
        solvers = {
            "fixed": AppFixedSolver(),
            "tracked": AppTrackedSolver(**tracker_kwargs),
            "disamb": AppTrackedSolver(**{**tracker_kwargs, "smooth": False}),
        }
        prev_rot = {k: None for k in solvers}
        prev_t = {k: None for k in solvers}
        seq_rows = {k: [] for k in solvers}
        for fi in range(n_frames):
            fr = ann["frames"][fi]
            d = det.detect(frames[fi])
            pts = dict(d.points) if getattr(d, "ok", False) else {}
            gt_t = np.array(fr["pose_gt"]["translation_mm"], dtype=float)
            gt_R = np.array(fr["pose_gt"]["rotation_matrix"], dtype=float)
            vis_ok = bool(fr.get("all_visible"))
            for name, solver in solvers.items():
                t0 = time.perf_counter()
                res = None
                if len(pts) >= 3:
                    kw = {"t_s": fi / fps} if name != "fixed" else {}
                    res = solver.solve({c: (float(pts[c][0]), float(pts[c][1]))
                                        for c in ("red", "green", "blue")},
                                       edge, int(info["width"]), int(info["height"]), **kw)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if res is None or not getattr(res, "ok", False):
                    seq_rows[name].append(None)
                    continue
                t = np.asarray(res.translation_mm, dtype=float)
                R = np.asarray(res.rotation, dtype=float)
                t_err = float(np.linalg.norm(t - gt_t)) if vis_ok else None
                rot_err = rotation_angle_deg(R, gt_R) if vis_ok else None
                jump = (rotation_angle_deg(R, prev_rot[name])
                        if prev_rot[name] is not None else None)
                jitter = (np.linalg.norm(t - prev_t[name]) if prev_t[name] is not None else None)
                prev_rot[name] = R
                prev_t[name] = t
                seq_rows[name].append({
                    "frame": fi, "factor": info.get("factor", "?"),
                    "t_err_mm": t_err, "rot_err_deg": rot_err,
                    "rot_jump_deg": jump, "step_mm": jitter,
                    "dist_mm": float(np.linalg.norm(t)),
                    "gt_dist_mm": float(np.linalg.norm(gt_t)),
                    "latency_ms": dt_ms, "vis_ok": vis_ok,
                })
        per_seq[info["seq_id"]] = seq_rows
        for name in solvers:
            rows[name].extend([r for r in seq_rows[name] if r is not None])
        print(f"[{subset}] {si+1}/{len(pairs)} {info['seq_id']} frames={n_frames}", flush=True)

    total_frames = sum(1 for v in per_seq.values() for _ in v.get("fixed", []))
    summary = {"subset": subset, "n_seq": len(pairs), "n_frames": total_frames,
               "tracker": tracker_kwargs,
               "methods": {k: _aggregate(v, fps=30.0, n_total=total_frames)
                           for k, v in rows.items()},
               "per_factor": _by_factor(rows)}
    return summary


def _aggregate(rows: list[dict], fps: float, n_total: int = 0) -> dict:
    def _arr(key):
        return np.array([r[key] for r in rows if r.get(key) is not None], dtype=float)

    t_err = _arr("t_err_mm")
    rot_err = _arr("rot_err_deg")
    jumps = _arr("rot_jump_deg")
    steps = _arr("step_mm")
    lat = _arr("latency_ms")
    dist_rel = _arr("dist_mm") / np.maximum(_arr("gt_dist_mm"), 1e-6) - 1.0
    return {
        "n_frames": len(rows),
        "pose_rate": float(len(rows) / n_total) if n_total else None,
        "t_err_mm": {"n": int(t_err.size),
                     "median": float(np.median(t_err)) if t_err.size else None,
                     "p90": float(np.percentile(t_err, 90)) if t_err.size else None,
                     "within_50mm": float(np.mean(t_err <= 50.0)) if t_err.size else None},
        "rot_err_deg": {"median": float(np.median(rot_err)) if rot_err.size else None,
                        "p90": float(np.percentile(rot_err, 90)) if rot_err.size else None,
                        "within_20deg": float(np.mean(rot_err <= 20.0)) if rot_err.size else None},
        "rot_flips_gt60": {"count": int(np.sum(jumps > 60.0)),
                           "rate": float(np.mean(jumps > 60.0)) if jumps.size else None,
                           "frames": int(jumps.size)},
        "step_mm_s": {"median": float(np.median(steps)) * fps if steps.size else None,
                      "p90": float(np.percentile(steps, 90)) * fps if steps.size else None},
        "dist_rel_err": {"median": float(np.median(dist_rel)) if dist_rel.size else None,
                         "p90": float(np.percentile(dist_rel, 90)) if dist_rel.size else None},
        "latency_ms": {"median": float(np.median(lat)) if lat.size else None,
                       "est_fps": float(1000.0 / np.median(lat)) if lat.size else None},
    }


def _by_factor(rows: dict[str, list[dict]]) -> dict:
    out = {}
    for name, rs in rows.items():
        by = {}
        for r in rs:
            by.setdefault(r.get("factor", "?"), []).append(r)
        out[name] = {f: {"t_err_med": float(np.median([x["t_err_mm"] for x in xs
                                                       if x["t_err_mm"] is not None]))
                         if any(x["t_err_mm"] is not None for x in xs) else None,
                         "flips": int(sum(1 for x in xs
                                          if x.get("rot_jump_deg") is not None
                                          and x["rot_jump_deg"] > 60.0)),
                         "frames": len(xs)} for f, xs in by.items()}
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="test")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--limit-seq", type=int, default=None)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "research" / "results" / "temporal_ab")
    a = ap.parse_args(argv)
    summary = run(a.subset, grid_size=a.grid, limit_seq=a.limit_seq)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.subset}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["methods"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
