"""对比度前端的输入分辨率 × 精度 × 耗时。

对每帧把图像缩到 maxDim 侧（模拟端侧小输入），在缩图上跑 classic3 union+dot-cls，
坐标换回原分辨率后与 GT 比较；统计 det@10 all-visible、逐点误差、单帧耗时。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from research.tridot_temporal import degrade
from research.tridot_temporal.classic_v3 import ClassicV3Detector
from research.tridot_temporal.eval_v2 import _camera, _discover_pairs, _gt, _read_video

COLORS = ("red", "green", "blue")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="hard_test")
    ap.add_argument("--limit-seq", type=int, default=6)
    ap.add_argument("--presets", default="clean,phone_far")
    ap.add_argument("--sizes", default="640,512,448")
    ap.add_argument("--out", default="research/results/frontend_cost/summary.json")
    args = ap.parse_args()

    pairs = _discover_pairs(Path("datasets/TriDotPose"), "synthetic", subset=args.subset)[: args.limit_seq]
    sizes = [int(s) for s in args.sizes.split(",")]
    det = ClassicV3Detector(
        frontend="union",
        contrast_floor=12.0,
        dot_cls_path="research/results/dot_cls/dot_cls_scaled.pt",
        dot_cls_scale=True,
    )
    out: dict = {"sizes": sizes, "presets": {}}
    for preset in args.presets.split(","):
        degrade.set_active(degrade.preset(preset))
        row: dict = {}
        for size in sizes:
            n = hit = 0
            times: list[float] = []
            perr: list[float] = []
            for vid, ann_path in pairs:
                ann = json.loads(Path(ann_path).read_text())
                cam = _camera(ann)
                frames = _read_video(Path(vid))
                det.reset()
                for fr in ann["frames"]:
                    fi = int(fr.get("frame_index", 0))
                    if fi >= len(frames):
                        break
                    gt = _gt(fr)
                    if not all(p[2] >= 2 for p in gt):
                        continue
                    n += 1
                    bgr = frames[fi]
                    h, w = bgr.shape[:2]
                    s = min(1.0, size / float(max(h, w)))
                    small = bgr if s >= 0.999 else cv2.resize(
                        bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
                    t0 = time.perf_counter()
                    o = det.detect(small, cam["fx"] * s, cam["fy"] * s, cam["cx"] * s, cam["cy"] * s, None)
                    times.append(time.perf_counter() - t0)
                    if o is None or not o.ok:
                        continue
                    pts = {c: (o.points[c][0] / s, o.points[c][1] / s) for c in o.points}
                    if len(pts) < 3:
                        continue
                    errs = [float(np.hypot(pts[c][0] - gt[i][0], pts[c][1] - gt[i][1])) for i, c in enumerate(COLORS)]
                    perr.extend(errs)
                    if max(errs) <= 10.0:
                        hit += 1
            row[str(size)] = {
                "frames": n,
                "det10_all": hit / n if n else 0.0,
                "median_frame_ms": float(np.median(times) * 1000.0) if times else None,
                "p90_frame_ms": float(np.percentile(times, 90) * 1000.0) if times else None,
                "point_err_median_px": float(np.median(perr)) if perr else None,
            }
            print(preset, size, row[str(size)], flush=True)
        out["presets"][preset] = row

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", p)


if __name__ == "__main__":
    main()
