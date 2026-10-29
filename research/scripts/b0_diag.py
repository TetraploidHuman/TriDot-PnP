"""Diagnose where the classic B0 pipeline loses accuracy.

For every frame we ask three questions:

1. *Is there a candidate near the GT point at all?*  (front-end / detection)
2. *If every colour has a candidate near GT, does B0 pick the right triple?*
   (association / triangle selection)
3. *Would the GT point pass B0's fixed HSV thresholds?*  (threshold brittleness)

The answers are aggregated per split and per factor so that improvements can be
prioritised with numbers instead of guesses.

Usage:
    python -m research.scripts.b0_diag --subset test,hard_test,holdout_hard \
        --out research/results/classic_diag
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from research.baseline_b0.detector import B0Detector, COLOR_ORDER, HSV_RANGES
from research.tridot_temporal.eval_v2 import _discover_pairs, _read_video

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets" / "TriDotPose"
NEAR_PX = 10.0          # "a candidate near the GT point"
FAR_PX = 40.0           # used only for reporting the tail


def _gt(frame):
    k = frame["keypoints"]
    return [[float(k[3 * i]), float(k[3 * i + 1]), float(k[3 * i + 2])]
            for i in range(3)]


def _passes_hsv(color: str, hsv) -> bool:
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
    for lo, hi in HSV_RANGES[color]:
        if lo[0] <= h <= hi[0] and s >= lo[1] and v >= lo[2]:
            return True
    return False


def _gt_hsv(bgr, x, y, r=4):
    h, w = bgr.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return None
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    flat = patch.reshape(-1, 3)
    # brightest pixel is the LED core
    idx = int(np.argmax(flat.sum(axis=1)))
    hsv = cv2.cvtColor(flat[idx:idx + 1].reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
    return hsv


def diag_sequence(video, ann_path, det: B0Detector):
    ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    frames = _read_video(video)
    n = min(len(frames), len(ann["frames"]))
    factor = ann["info"].get("factor", "?")
    rows = []
    for fi in range(n):
        fr = ann["frames"][fi]
        gt = _gt(fr)
        bgr = frames[fi]
        t0 = time.perf_counter()
        cands = det.propose(bgr)  # dict colour -> list[BlobCand]
        propose_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        d = det.detect(bgr)
        detect_ms = (time.perf_counter() - t1) * 1000.0
        row = {"seq_id": ann["info"]["seq_id"], "factor": factor,
               "frame": fi, "propose_ms": propose_ms, "detect_ms": detect_ms,
               "all_visible": bool(fr.get("all_visible")), "gt": gt,
               "n_cands": {}, "near": {}, "min_dist": {}, "pass_hsv": {},
               "gt_hsv": {}, "chosen": dict(getattr(d, "points", {}) or {}),
               "ok": bool(getattr(d, "ok", False))}
        for ci, color in enumerate(COLOR_ORDER):
            gx, gy, vis = gt[ci]
            lst = cands.get(color) or []
            row["n_cands"][color] = len(lst)
            if vis < 1.5:
                row["min_dist"][color] = None
                row["near"][color] = None
                row["pass_hsv"][color] = None
                row["gt_hsv"][color] = None
                continue
            dists = [float(np.hypot(c.u - gx, c.v - gy)) for c in lst]
            row["min_dist"][color] = min(dists) if dists else None
            row["near"][color] = (min(dists) <= NEAR_PX) if dists else False
            hsv = _gt_hsv(bgr, gx, gy)
            row["gt_hsv"][color] = None if hsv is None else [int(hsv[0]), int(hsv[1]), int(hsv[2])]
            row["pass_hsv"][color] = None if hsv is None else _passes_hsv(color, hsv)
        # det10 on all-visible frames, exactly like metrics.det10_legacy
        if fr.get("all_visible"):
            errs = []
            for ci, color in enumerate(COLOR_ORDER):
                if color in row["chosen"]:
                    px = row["chosen"][color]
                    errs.append(float(np.hypot(px[0] - gt[ci][0], px[1] - gt[ci][1])))
            row["det10"] = bool(len(errs) == 3 and max(errs) <= 10.0)
            row["orc10"] = bool(all(row["near"][c] is True for c in COLOR_ORDER))
        else:
            row["det10"] = None
            row["orc10"] = None
        rows.append(row)
    return rows


def summarize(rows, key="all"):
    out = {"n_frames": len(rows),
           "n_all_visible": sum(1 for r in rows if r["all_visible"])}
    vis = [r for r in rows if r["all_visible"]]
    out["det10_legacy"] = (sum(1 for r in vis if r["det10"]) / len(vis)) if vis else None
    out["oracle_det10"] = (sum(1 for r in vis if r["orc10"]) / len(vis)) if vis else None
    # failure split (all-visible frames only)
    miss = [r for r in vis if not r["orc10"]]
    assoc = [r for r in vis if r["orc10"] and not r["det10"]]
    out["detect_miss_frames"] = len(miss)
    out["assoc_wrong_frames"] = len(assoc)
    out["detect_miss_share"] = (len(miss) / len(vis)) if vis else None
    out["assoc_wrong_share"] = (len(assoc) / len(vis)) if vis else None
    # per colour: candidate counts / near rate / HSV pass rate
    per_color = {}
    for color in COLOR_ORDER:
        ns = [r["n_cands"][color] for r in vis]
        near = [r["near"][color] for r in vis if r["near"][color] is not None]
        phsv = [r["pass_hsv"][color] for r in vis if r["pass_hsv"][color] is not None]
        dists = [r["min_dist"][color] for r in vis if r["min_dist"][color] is not None]
        per_color[color] = {
            "mean_cands": float(np.mean(ns)) if ns else None,
            "p90_cands": float(np.percentile(ns, 90)) if ns else None,
            "near_rate": float(np.mean(near)) if near else None,
            "hsv_pass_rate": float(np.mean(phsv)) if phsv else None,
            "min_dist_median_px": float(np.median(dists)) if dists else None,
            "min_dist_p90_px": float(np.percentile(dists, 90)) if dists else None,
        }
    out["per_color"] = per_color
    # among missed frames, which colours are missing and why
    miss_colors = defaultdict(int)
    miss_empty = defaultdict(int)   # no candidate at all
    miss_far = defaultdict(int)     # candidates exist but none within 10 px
    miss_hsv_fail = defaultdict(int)
    for r in miss:
        for color in COLOR_ORDER:
            if r["near"][color] is False:
                miss_colors[color] += 1
                if r["n_cands"][color] == 0:
                    miss_empty[color] += 1
                else:
                    miss_far[color] += 1
                if r["pass_hsv"][color] is False:
                    miss_hsv_fail[color] += 1
    out["miss_by_color"] = dict(miss_colors)
    out["miss_empty"] = dict(miss_empty)
    out["miss_far"] = dict(miss_far)
    out["miss_hsv_fail"] = dict(miss_hsv_fail)
    # latency
    out["propose_ms_median"] = float(np.median([r["propose_ms"] for r in rows]))
    out["detect_ms_median"] = float(np.median([r["detect_ms"] for r in rows]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--subset", default="test,hard_test",
                    help="comma list of splits.json keys")
    ap.add_argument("--limit-seq", type=int, default=None)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "research" / "results" / "classic_diag")
    args = ap.parse_args(argv)

    det = B0Detector()
    args.out.mkdir(parents=True, exist_ok=True)
    all_summary = {}
    for subset in [s.strip() for s in args.subset.split(",") if s.strip()]:
        pairs = _discover_pairs(args.dataset, "synthetic", subset=subset)
        if args.limit_seq:
            pairs = pairs[: args.limit_seq]
        if not pairs:
            print(f"[diag] no sequences for subset={subset}")
            continue
        rows = []
        for si, (video, ann_path) in enumerate(pairs):
            ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
            rows.extend(diag_sequence(video, ann_path, det))
            print(f"[diag] {subset} {si + 1}/{len(pairs)} "
                  f"{ann['info']['seq_id']} frames={len(rows)}", flush=True)
        by_factor = defaultdict(list)
        for r in rows:
            by_factor[r["factor"]].append(r)
        summary = {"all": summarize(rows),
                   "by_factor": {k: summarize(v) for k, v in sorted(by_factor.items())}}
        (args.out / f"{subset}_summary.json").write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
        all_summary[subset] = summary
        print(f"[diag] wrote {args.out / (subset + '_summary.json')}", flush=True)

    # markdown table
    lines = ["# B0 failure diagnosis", ""]
    for subset, summary in all_summary.items():
        lines.append(f"## {subset}")
        lines.append("")
        lines.append("| factor | all-vis frames | det@10 | oracle det@10 | detect-miss | assoc-wrong |")
        lines.append("|---|---|---|---|---|---|")
        for name, s in [("ALL", summary["all"])] + list(summary["by_factor"].items()):
            if s["n_all_visible"] == 0:
                continue
            lines.append(
                f"| {name} | {s['n_all_visible']} | "
                f"{(s['det10_legacy'] or 0):.3f} | {(s['oracle_det10'] or 0):.3f} | "
                f"{(s['detect_miss_share'] or 0):.3f} | {(s['assoc_wrong_share'] or 0):.3f} |")
        lines.append("")
        s = summary["all"]
        lines.append("Per-colour candidate stats (all-visible frames):")
        lines.append("")
        lines.append("| colour | mean cands | p90 cands | near@10px | HSV pass | min-dist med (px) |")
        lines.append("|---|---|---|---|---|---|")
        for color, pc in s["per_color"].items():
            lines.append(
                f"| {color} | {pc['mean_cands']:.1f} | {pc['p90_cands']:.0f} | "
                f"{pc['near_rate']:.3f} | {pc['hsv_pass_rate']:.3f} | "
                f"{pc['min_dist_median_px']:.1f} |")
        lines.append("")
        lines.append(f"missed-frame colours: {s['miss_by_color']} "
                     f"(empty candidate: {s['miss_empty']}, far only: {s['miss_far']}, "
                     f"HSV-fail: {s['miss_hsv_fail']})")
        lines.append("")
        lines.append(f"latency: propose {s['propose_ms_median']:.1f} ms, "
                     f"detect {s['detect_ms_median']:.1f} ms")
        lines.append("")
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[diag] wrote {args.out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
