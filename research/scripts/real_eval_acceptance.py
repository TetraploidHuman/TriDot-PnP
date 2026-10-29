"""Pre-registered acceptance check for the real-device capture protocol.

`research/docs/real_device_protocol.md` fixes the thresholds *before* data is
collected; this script turns a capture session into a pass/fail table so the
thresholds cannot be renegotiated after seeing the numbers.

Modes
-----
* **scored** (default): every sequence in ``splits.json["real_test"]`` that has
  an annotation file under ``datasets/TriDotPose/real/annotations`` is evaluated
  with the classic-v3 union front-end (+ optional dot classifier) and 3-point
  PnP.  Frames are bucketed by GT distance 1-5 m / 5-8 m / >=8 m and checked
  against the pre-registered thresholds.
* **detection-only** (``--video <mp4>``): no annotations yet - measures
  three-point detection rate and frame-to-frame point drift, which is enough to
  catch a broken capture (wrong FOV, lens smear, AE hunting) before labelling.

Usage
-----
    python -m research.scripts.real_eval_acceptance                 # scored
    python -m research.scripts.real_eval_acceptance \\
        --video datasets/TriDotPose/real/videos/smoke_real_desk.mp4  # pre-flight
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.tridot_temporal import eval_v2 as ev
from research.tridot_temporal.classic_v3 import ClassicV3Detector
from research.tridot_temporal.pnp2 import rotation_angle_deg  # noqa: F401  (parity of imports)

BANDS = ((1.0, 5.0), (5.0, 8.0), (8.0, 1e9))
BAND_NAMES = ("1-5 m", "5-8 m", ">=8 m", "<1 m (ignored)")
THRESHOLDS = {
    "1-5 m": dict(min_det10=0.80, max_dist_rel_err=0.05, max_t_err_mm=30.0),
    "5-8 m": dict(min_det10=0.50, max_dist_rel_err=None, max_t_err_mm=None,
                  forbid_bad_distance=0.20),
    ">=8 m": dict(min_det10=None, max_dist_rel_err=None, max_t_err_mm=None),
}


def _load_splits(dataset: Path) -> dict:
    return json.loads((dataset / "meta" / "splits.json").read_text(encoding="utf-8"))


def _seq_paths(dataset: Path, seq: str, subset: str = "real_test"):
    """real_test -> real/{videos,annotations}; 其余 subset -> synthetic/ 下同名目录。"""
    if subset == "real_test":
        ann = dataset / "real" / "annotations" / f"{seq}.json"
        video = dataset / "real" / "videos" / f"{seq}.mp4"
    else:
        ann = dataset / "synthetic" / "annotations" / f"{seq}.json"
        video = dataset / "synthetic" / "videos" / f"{seq}.mp4"
    return (video, ann) if video.exists() and ann.exists() else None


def _band(depth_mm: float) -> str:
    d = depth_mm / 1000.0
    if d < 1.0:
        return BAND_NAMES[-1]
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= d < hi:
            return BAND_NAMES[i]
    return BAND_NAMES[-2]


def eval_scored(dataset: Path, dot_cls: Path | None, dot_scale: bool, limit: int | None,
                subset: str = "real_test", frontend: str = "v3", degrade_name: str = "none") -> dict:
    splits = _load_splits(dataset)
    seqs = list(splits.get(subset) or [])
    if limit:
        seqs = seqs[:limit]
    rows = []
    if degrade_name and degrade_name != "none":
        from research.tridot_temporal import degrade as _deg
        _deg.set_active(_deg.preset(degrade_name))
    if frontend == "v2":
        from research.tridot_temporal.classic_v2 import ClassicV2Detector
        det = ClassicV2Detector()
    else:
        det = ClassicV3Detector(frontend="union", contrast_floor=12.0,
                                dot_cls_path=str(dot_cls) if dot_cls else None,
                                dot_cls_scale=dot_scale)
    solver = ev.PoseSolver()
    for seq in seqs:
        paths = _seq_paths(dataset, seq, subset)
        if paths is None:
            continue
        video, ann_path = paths
        ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        cam = ev._camera(ann)
        fx, fy, cx, cy = (ev.camera_fx(cam), ev.camera_fy(cam),
                          ev.camera_cx(cam), ev.camera_cy(cam))
        edge_mm = float(ann["info"].get("edge_length_mm", 100.0))
        frames = ev._read_video(video)
        det.reset()
        for fi in range(min(len(frames), len(ann["frames"]))):
            fr = ann["frames"][fi]
            gt = np.asarray(fr["keypoints"], dtype=float).reshape(3, 3)
            if int(fr.get("all_visible", 0)) == 0:
                continue
            gt_t = np.asarray(fr["pose_gt"]["translation_mm"], dtype=float)
            depth = float(np.linalg.norm(gt_t))
            res = det.detect(frames[fi], fx=fx, fy=fy, cx=cx, cy=cy, solver=solver)
            if not res.ok:
                rows.append(dict(seq=seq, frame=fi, band=_band(depth), det10=0,
                                 dist_rel_err=None, t_err_mm=None, reason=res.reason))
                continue
            pts = res.points
            errs = [float(np.hypot(pts[c][0] - gt[i][0], pts[c][1] - gt[i][1]))
                    for i, c in enumerate(("red", "green", "blue"))]
            ok = all(e <= 10.0 for e in errs)
            solved = solver.solve(pts, edge_mm, fx, fy, cx, cy,
                                  np.array(cam.get("distortion", [0, 0, 0, 0, 0])))
            if solved.ok:
                t_err = float(np.linalg.norm(np.asarray(solved.translation_mm) - gt_t))
                dist_pred = float(np.mean([np.linalg.norm(np.asarray(solved.translation_mm))]))
                dist_gt = float(np.linalg.norm(gt_t))
                rel = abs(dist_pred - dist_gt) / max(dist_gt, 1e-6)
            else:
                t_err, rel = None, None
                solver.reset()
            rows.append(dict(seq=seq, frame=fi, band=_band(depth), det10=int(ok),
                             dist_rel_err=rel, t_err_mm=t_err, reason=res.reason))
    return {"rows": rows, "table": _table(rows)}


def _table(rows: list[dict]) -> list[dict]:
    out = []
    for band in BAND_NAMES:
        rs = [r for r in rows if r["band"] == band]
        if not rs:
            out.append(dict(band=band, n=0, det10=None, dist_rel_median=None,
                            t_err_median=None, t_err_p90=None, bad_distance=None,
                            verdict="NO DATA"))
            continue
        det10 = float(np.mean([r["det10"] for r in rs]))
        t_errs = [r["t_err_mm"] for r in rs if r["t_err_mm"] is not None]
        rels = [r["dist_rel_err"] for r in rs if r["dist_rel_err"] is not None]
        bad = [r for r in rs if r["det10"] == 1 and r["dist_rel_err"] is not None
               and r["dist_rel_err"] > 0.20]
        if band.startswith("<1 m"):
            out.append(dict(band=band, n=len(rs), det10=det10,
                            dist_rel_median=float(np.median(rels)) if rels else None,
                            t_err_median=float(np.median(t_errs)) if t_errs else None,
                            t_err_p90=float(np.percentile(t_errs, 90)) if t_errs else None,
                            bad_distance=None, verdict="IGNORED"))
            continue
        th = THRESHOLDS[band]
        verdict = "PASS"
        if th["min_det10"] is not None and det10 < th["min_det10"]:
            verdict = "FAIL"
        if th["max_dist_rel_err"] is not None and rels and float(np.median(rels)) > th["max_dist_rel_err"]:
            verdict = "FAIL"
        if th["max_t_err_mm"] is not None and t_errs and float(np.median(t_errs)) > th["max_t_err_mm"]:
            verdict = "FAIL"
        if th.get("forbid_bad_distance") is not None and bad:
            verdict = "FAIL"
        if band.startswith("<1 m"):
            out.append(dict(band=band, n=len(rs), det10=det10,
                            dist_rel_median=float(np.median(rels)) if rels else None,
                            t_err_median=float(np.median(t_errs)) if t_errs else None,
                            t_err_p90=float(np.percentile(t_errs, 90)) if t_errs else None,
                            bad_distance=None, verdict="IGNORED"))
            continue
        out.append(dict(
            band=band, n=len(rs), det10=det10,
            dist_rel_median=float(np.median(rels)) if rels else None,
            t_err_median=float(np.median(t_errs)) if t_errs else None,
            t_err_p90=float(np.percentile(t_errs, 90)) if t_errs else None,
            bad_distance=len(bad), verdict=verdict))
    return out


def eval_detection_only(video: Path, dot_cls: Path | None, dot_scale: bool) -> dict:
    """Pre-flight on an unlabelled clip: detection rate + frame-to-frame drift."""
    det = ClassicV3Detector(frontend="union", contrast_floor=12.0,
                            dot_cls_path=str(dot_cls) if dot_cls else None,
                            dot_cls_scale=dot_scale)
    solver = ev.PoseSolver()
    frames = ev._read_video(video)
    prev = None
    detected = 0
    drifts = []
    for i, bgr in enumerate(frames):
        h, w = bgr.shape[:2]
        # unknown intrinsics on an unlabelled clip: assume the 60-deg FOV of the app
        fx = w / (2.0 * np.tan(np.deg2rad(60.0) / 2.0))
        res = det.detect(bgr, fx=fx, fy=fx, cx=w / 2.0, cy=h / 2.0, solver=solver)
        if not res.ok:
            prev = None
            continue
        detected += 1
        pts = res.points
        if prev is not None:
            drifts.append(float(np.mean([np.hypot(pts[c][0] - prev[c][0], pts[c][1] - prev[c][1])
                                         for c in ("red", "green", "blue")])))
        prev = {c: pts[c] for c in ("red", "green", "blue")}
    return {"video": video.name, "frames": len(frames), "detected": detected,
            "detect_rate": detected / max(len(frames), 1),
            "drift_median_px": float(np.median(drifts)) if drifts else None,
            "drift_p95_px": float(np.percentile(drifts, 95)) if drifts else None}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("datasets/TriDotPose"))
    ap.add_argument("--dot-cls", type=Path, default=None)
    ap.add_argument("--dot-cls-scale", action="store_true")
    ap.add_argument("--limit-seq", type=int, default=None)
    ap.add_argument("--subset", default="real_test",
                    help="splits.json 的 key；非 real_test 时读 synthetic/{videos,annotations}（用于 dry-run）")
    ap.add_argument("--frontend", choices=["v3", "v2"], default="v3")
    ap.add_argument("--degrade", default="none",
                    help="dry-run 时对帧叠加退化（clean/phone_easy/phone_far/phone_harsh）")
    ap.add_argument("--video", type=Path, default=None, help="detection-only pre-flight clip")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.video is not None:
        result = eval_detection_only(a.video, a.dot_cls, a.dot_cls_scale)
        print(f"pre-flight {result['video']}: detected {result['detected']}/{result['frames']} "
              f"({result['detect_rate']:.2%}), drift median {result['drift_median_px']} px, "
              f"p95 {result['drift_p95_px']} px")
    else:
        result = eval_scored(a.dataset, a.dot_cls, a.dot_cls_scale, a.limit_seq,
                             subset=a.subset, frontend=a.frontend, degrade_name=a.degrade)
        hdr = f"{'band':<8}{'n':>6}{'det@10':>9}{'dist_rel':>10}{'t_med':>9}{'t_p90':>9}{'bad':>6}  verdict"
        print(hdr)
        for r in result["table"]:
            def fmt(v, nd=2):
                return "-" if v is None else f"{v:.{nd}f}"
            print(f"{r['band']:<8}{r['n']:>6}{fmt(r['det10'],3):>9}{fmt(r['dist_rel_median'],3):>10}"
                  f"{fmt(r['t_err_median'],1):>9}{fmt(r['t_err_p90'],1):>9}{r['bad_distance'] if r['bad_distance'] is not None else '-':>6}  {r['verdict']}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
