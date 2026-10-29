"""模拟端侧「三点自洽门」的真实先验版本（无 GT）：B1 主路径 + 上一帧已接受视边长。

对照三种策略（hard_test，三点全可见帧，det@10 all）：
  b1            主路径原始结果
  gate_gt       自洽门用 GT 深度换算的视边长（上界，round 19b 的数字）
  gate_prior    自洽门用「上一帧已接受结果的视边长」（端侧真实可得）
不自洽时用 classic3 union + dot-cls 重找；重找结果也需自洽才采纳，否则保留 B1。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import research.pyc_import as pyc
pyc.install()

from research.baseline_b1.detector import B1Detector
from research.tridot_temporal import degrade
from research.tridot_temporal.classic_v3 import ClassicV3Detector
from research.tridot_temporal.dataset import COLOR_ORDER, _discover_pairs
from research.tridot_temporal.eval_v2 import _camera, _gt, _read_video


def subset_ok(points: dict) -> bool:
    return len(points) >= 3


def edge_stats(points: dict) -> tuple[float, float]:
    p = np.array([points[c] for c in COLOR_ORDER], dtype=float)
    e = [float(np.linalg.norm(p[0] - p[1])), float(np.linalg.norm(p[1] - p[2])),
         float(np.linalg.norm(p[2] - p[0]))]
    mean = sum(e) / 3.0
    spread = (max(e) - min(e)) / mean if mean > 0 else 9.0
    return mean, spread


def plausible(points: dict, expected_edge: float) -> bool:
    if len(points) < 3:
        return False
    mean, spread = edge_stats(points)
    if mean < 6 or spread > 0.5:
        return False
    if expected_edge > 1:
        ratio = mean / expected_edge
        if ratio < 0.55 or ratio > 1.45:
            return False
    return True


def det_hit(points: dict, gt) -> bool:
    if len(points) < 3:
        return False
    for c, (gx, gy, vis) in zip(COLOR_ORDER, gt):
        if vis < 1:
            continue
        if c not in points:
            return False
        if math.hypot(points[c][0] - gx, points[c][1] - gy) > 10:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-seq", type=int, default=6)
    ap.add_argument("--presets", default="clean,phone_far")
    ap.add_argument("--out", default="research/results/classic_cv3/triple_gate_tracker_ab.json")
    args = ap.parse_args()

    pairs = _discover_pairs(Path("datasets/TriDotPose"), "synthetic", subset="hard_test")[: args.limit_seq]
    b1 = B1Detector(grid_size=32)
    v3 = ClassicV3Detector(frontend="union", contrast_floor=12.0,
                           dot_cls_path="research/results/dot_cls/dot_cls_scaled.pt",
                           dot_cls_scale=True)
    results: dict[str, dict] = {}
    for preset in args.presets.split(","):
        degrade.set_active(degrade.preset(preset))
        st = {"n": 0, "b1": 0, "gate_gt": 0, "gate_prior": 0, "trigger": 0, "rescue": 0}
        for vid, ann_path in pairs:
            ann = json.loads(Path(ann_path).read_text())
            cam = _camera(ann)
            frames = _read_video(Path(vid))
            v3.reset()
            prior_edge = 0.0
            for fr in ann["frames"]:
                fi = int(fr.get("frame_index", 0))
                if fi >= len(frames):
                    break
                gt = _gt(fr)
                if sum(1 for p in gt if p[2] >= 2) != 3:
                    continue
                st["n"] += 1
                depth = float(np.linalg.norm(np.asarray(fr["pose_gt"]["translation_mm"], dtype=float)))
                edge_gt = cam["fx"] * float(ann["info"].get("edge_length_mm", 100.0)) / max(depth, 1e-6)
                r = b1.detect(frames[fi])
                pts = dict(r.points) if r.ok else {}
                st["b1"] += int(det_hit(pts, gt))
                if plausible(pts, edge_gt):
                    st["gate_gt"] += int(det_hit(pts, gt))
                else:
                    st["trigger"] += 1
                    o = v3.detect(frames[fi], cam["fx"], cam["fy"], cam["cx"], cam["cy"], None)
                    cand = o.points if (o is not None and o.ok) else {}
                    if plausible(cand, edge_gt) and det_hit(cand, gt):
                        st["gate_gt"] += 1
                    elif plausible(pts, edge_gt):
                        st["gate_gt"] += int(det_hit(pts, gt))
                # 真实先验版本（不看 GT）
                if plausible(pts, prior_edge):
                    st["gate_prior"] += int(det_hit(pts, gt))
                    mean, _ = edge_stats(pts)
                    prior_edge = mean
                else:
                    o = v3.detect(frames[fi], cam["fx"], cam["fy"], cam["cx"], cam["cy"], None)
                    cand = o.points if (o is not None and o.ok) else {}
                    if plausible(cand, prior_edge):
                        st["gate_prior"] += int(det_hit(cand, gt))
                        st["rescue"] += 1
                        prior_edge = edge_stats(cand)[0]
                    elif pts:
                        st["gate_prior"] += int(det_hit(pts, gt))
                        if plausible(pts, 0.0):
                            prior_edge = edge_stats(pts)[0]
        results[preset] = {
            "n": st["n"], "trigger": st["trigger"], "rescue": st["rescue"],
            "det10_b1": round(st["b1"] / max(1, st["n"]), 4),
            "det10_gate_gt": round(st["gate_gt"] / max(1, st["n"]), 4),
            "det10_gate_prior": round(st["gate_prior"] / max(1, st["n"]), 4),
        }
        print(preset, results[preset], flush=True)
        degrade.set_active(None)
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
