"""真实视频上的『主线 vs 对比度前端』稳定性对照（无 GT）。

real/videos 只有一条近景真机视频、没有标注，因此这里不比精度，只比：
  1. 检出率（三点是否齐）
  2. 帧间点漂移（同一检测器相邻帧预测点的中位跳变）
  3. 视边长稳定性（相对标准差）
  4. 两种检测器的一致性（同一帧三点距离中位）
距离用「下采样再还原」模拟：scale=1.0 近景、0.7 中距、0.5 远距（点变小、边缘变短）。
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

from research.baseline_b1.detector import B1Detector
from research.tridot_temporal.classic_v3 import ClassicV3Detector
from research.tridot_temporal import degrade as dg
from research.tridot_temporal.eval_v2 import _read_video  # noqa: F401  (keeps import surface)


def norm_edge(pts):
    e = [float(np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
         for i, j in ((0, 1), (1, 2), (2, 0))]
    return float(np.mean(e))


def run(frames, det, scale):
    """返回每帧的三点（原分辨率坐标）或 None。"""
    out = []
    for fr in frames:
        h, w = fr.shape[:2]
        if scale != 1.0:
            sm = cv2.resize(fr, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=cv2.INTER_AREA)
        else:
            sm = fr
        res = det(sm) if det.__name__ == "b1" else det(sm)
        if res is None:
            out.append(None)
            continue
        pts = [[float(p[0]) / scale, float(p[1]) / scale] for p in res]
        out.append(pts)
    return out


def b1_runner(grid=32):
    det = B1Detector(grid_size=grid)

    def run_b1(bgr):
        r = det.detect(bgr)
        if not r.ok:
            return None
        return [r.points[c] for c in ("red", "green", "blue")]

    run_b1.__name__ = "b1"
    return run_b1


def v3_runner(ckpt):
    det = ClassicV3Detector(frontend="union", contrast_floor=12.0,
                            dot_cls_path=ckpt, dot_cls_scale=True, temporal_gate=0.6)

    def run_v3(bgr):
        h, w = bgr.shape[:2]
        fx = w / (2.0 * np.tan(np.deg2rad(60.0) / 2.0))
        r = det.detect(bgr, fx, fx, w / 2.0, h / 2.0)
        if not r.ok or len(r.points) < 3:
            return None
        return [r.points[c] for c in ("red", "green", "blue")]

    run_v3.__name__ = "v3"
    return run_v3


def stability(tracks):
    drift, edges, jumps = [], [], 0
    prev = None
    for pts in tracks:
        if pts is None:
            prev = None
            continue
        edges.append(norm_edge(pts))
        if prev is not None:
            d = max(float(np.hypot(pts[i][0] - prev[i][0], pts[i][1] - prev[i][1])) for i in range(3))
            drift.append(d)
            if d > 20:
                jumps += 1
        prev = pts
    n = len(tracks)
    ok = sum(1 for p in tracks if p is not None)
    return {
        "det_rate": round(ok / max(1, n), 4),
        "drift_med_px": round(statistics.median(drift), 3) if drift else None,
        "drift_p95_px": round(sorted(drift)[int(0.95 * (len(drift) - 1))], 3) if drift else None,
        "edge_rel_std": round(float(np.std(edges) / np.mean(edges)), 4) if edges else None,
        "edge_med_px": round(float(np.mean(edges)), 2) if edges else None,
        "jumps_gt20px": jumps,
        "n_frames": n,
    }


def agreement(a, b):
    ds = []
    for pa, pb in zip(a, b):
        if pa is None or pb is None:
            continue
        ds.append(float(np.median([np.hypot(pa[i][0] - pb[i][0], pa[i][1] - pb[i][1]) for i in range(3)])))
    return {"n": len(ds), "med_px": round(statistics.median(ds), 2) if ds else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="datasets/TriDotPose/real/videos/smoke_real_desk.mp4")
    ap.add_argument("--dot-cls", default="research/results/dot_cls/dot_cls_scaled.pt")
    ap.add_argument("--scales", default="1.0,0.7,0.5")
    ap.add_argument("--preset", default="clean")
    ap.add_argument("--out", default="research/results/real_probe/chain_probe.json")
    args = ap.parse_args()

    cfg = dg.preset(args.preset)
    dg.set_active(cfg)
    frames = dg.apply_to_frames(_read_video(Path(args.video)).copy()) if cfg else _read_video(Path(args.video))
    b1, v3 = b1_runner(), v3_runner(args.dot_cls)
    scales = [float(s) for s in args.scales.split(",")]

    out = {"video": args.video, "preset": args.preset, "scales": {}}
    for s in scales:
        tb = run(frames, b1, s)
        tv = run(frames, v3, s)
        out["scales"][f"{s:g}"] = {
            "b1": stability(tb),
            "v3": stability(tv),
            "agreement_b1_v3": agreement(tb, tv),
        }
        print(f"scale={s:g} b1={out['scales'][f'{s:g}']['b1']} v3={out['scales'][f'{s:g}']['v3']} "
              f"agree={out['scales'][f'{s:g}']['agreement_b1_v3']}", flush=True)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
