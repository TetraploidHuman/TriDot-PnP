"""A/B: tiny-dot ROI upsampling (C) 对「匹配上的候选点」定位精度的影响。

按 GT 视边长分桶统计最近候选的像素误差（不跑 PnP），比较
  classic3 union + dot-cls，roi_upsample=True / False。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from research.tridot_temporal import degrade
from research.tridot_temporal.classic_v3 import ClassicV3Detector
from research.tridot_temporal.eval_v2 import _camera, _gt, _read_video
from research.tridot_temporal.dataset import COLOR_ORDER as COLOR_ORDER_
from research.tridot_temporal.dataset import _discover_pairs

BUCKETS = [(0, 20), (20, 40), (40, 80), (80, 160), (160, 1e9)]


def bucket_of(edge_px: float) -> str:
    for lo, hi in BUCKETS:
        if lo <= edge_px < hi:
            return f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/TriDotPose")
    ap.add_argument("--subset", default="hard_test")
    ap.add_argument("--limit-seq", type=int, default=12)
    ap.add_argument("--presets", default="clean,phone_far")
    ap.add_argument("--dot-cls", default="research/results/dot_cls/dot_cls_scaled.pt")
    ap.add_argument("--out", default="research/results/dot_cls/subpixel_ab.json")
    args = ap.parse_args()

    pairs = _discover_pairs(args.dataset, "synthetic", subset=args.subset)
    pairs = pairs[: args.limit_seq]
    rows = {}
    for preset in args.presets.split(","):
        degrade.set_active(degrade.preset(preset))
        for upsample in (True, False):
            det = ClassicV3Detector(frontend="union", contrast_floor=12.0,
                                    dot_cls_path=args.dot_cls, dot_cls_w=0.5,
                                    dot_cls_scale=True, roi_upsample=upsample)
            errs: dict[str, list[float]] = {}
            n_frames = 0
            for vid_path, ann_path in pairs:
                ann = json.loads(Path(ann_path).read_text())
                cam = _camera(ann)
                frames = _read_video(Path(vid_path))
                det.reset()
                for fr in ann["frames"]:
                    fi = int(fr.get("frame_index", 0))
                    if fi >= len(frames):
                        break
                    n_frames += 1
                    gt = _gt(fr)
                    depth = float(np.linalg.norm(np.asarray(fr["pose_gt"]["translation_mm"], dtype=float)))
                    edge_px = cam["fx"] * float(ann["info"].get("edge_length_mm", 100.0)) / max(depth, 1e-6)
                    out = det.detect(frames[fi], cam["fx"], cam["fy"], cam["cx"], cam["cy"], None)
                    if out is None or not out.ok:
                        continue
                    b = bucket_of(edge_px)
                    for color, (gx, gy, vis) in zip(COLOR_ORDER_, gt):
                        if vis < 1:
                            continue
                        best = None
                        for c in (out.candidates.get(color) or []):
                            e = math.hypot(c.u - gx, c.v - gy)
                            if best is None or e < best:
                                best = e
                        if best is not None:
                            errs.setdefault(b, []).append(best)
            key = f"{preset}|upsample={upsample}"
            rows[key] = {
                "frames": n_frames,
                "buckets": {
                    b: {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}
                    for b, v in sorted(errs.items())
                },
            }
            print(key, {b: (d["n"], round(d["median"], 3)) for b, d in rows[key]["buckets"].items()}, flush=True)
        degrade.set_active(None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
