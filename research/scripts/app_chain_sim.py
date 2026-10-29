"""端侧全链路 Python 模拟：B1 主路径 → 三点自洽门 → 远距辅助 → 时序追踪 PnP。

目的：在真机数据到来前，用「app 主路径端口（B1）+ app 追踪器（AppTrackedSolver）」
估计 App 在四档退化下的端到端表现（det@10 all / observable / pose coverage / t_err）。

与真实 App 的差异（如实记录）：
- 未模拟概率矩阵/ROI 启发式与 refineSpotsOnOriginal 的多轮质心（B1 端口自带一次 refine）；
- 未模拟 1920x1080 高分辨率分析（E 的收益在 §4.9/§4.10 单独量化）；
- 远距辅助用 classic3 union + dot-cls，而非经 validator 的 PnP 校验版本。
"""
from __future__ import annotations

import argparse
import dataclasses
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
from research.tridot_temporal.temporal import AppTrackedSolver


def edge_stats(points: dict) -> tuple[float, float]:
    p = np.array([points[c] for c in COLOR_ORDER], dtype=float)
    e = [float(np.linalg.norm(p[0] - p[1])), float(np.linalg.norm(p[1] - p[2])),
         float(np.linalg.norm(p[2] - p[0]))]
    mean = sum(e) / 3.0
    return mean, ((max(e) - min(e)) / mean if mean > 0 else 9.0)


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


def point_errs(points: dict, gt) -> list[float] | None:
    out = []
    for c, (gx, gy, vis) in zip(COLOR_ORDER, gt):
        if vis < 1:
            continue
        if c not in points:
            return None
        out.append(math.hypot(points[c][0] - gx, points[c][1] - gy))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-seq", type=int, default=6)
    ap.add_argument("--presets", default="clean,phone_easy,phone_far,phone_harsh")
    ap.add_argument("--detector", choices=["b1", "v3"], default="b1",
                    help="b1: app 主路径端口；v3: 用 classic3 union+dot-cls 直接替代主路径（上界）")
    ap.add_argument("--far-period", type=int, default=1,
                    help="far_primary 下每 N 帧跑一次对比度前端（主线不可信时仍立即跑）")
    ap.add_argument("--strategy", choices=["main", "gate", "far_primary"], default="gate",
                    help="main: 只跑主线；gate: 主线不全/不自洽才用对比度前端（旧语义）；"
                         "far_primary: 对比度前端可信就用它（round 22 的 App 默认）")
    ap.add_argument("--degrade-override", default="",
                    help="k=v,... 覆盖退化 preset（用于采集开关 A/B）")
    ap.add_argument("--out", default="research/results/app_chain/app_chain_sim.json")
    args = ap.parse_args()

    pairs = _discover_pairs(Path("datasets/TriDotPose"), "synthetic", subset="hard_test")[: args.limit_seq]
    b1 = B1Detector(grid_size=32)
    v3 = ClassicV3Detector(frontend="union", contrast_floor=12.0,
                           dot_cls_path="research/results/dot_cls/dot_cls_scaled.pt",
                           dot_cls_scale=True)
    out: dict[str, dict] = {}
    for preset in args.presets.split(","):
        _preset = degrade.preset(preset)
        if args.degrade_override:
            kw = {}
            for item in args.degrade_override.split(","):
                if not item.strip():
                    continue
                k, v = item.split("=", 1)
                cur = getattr(_preset, k)
                kw[k] = type(cur)(v) if cur is not None else float(v)
            _preset = dataclasses.replace(_preset, **kw)
        degrade.set_active(_preset)
        solver = AppTrackedSolver(fov_deg=60.0)
        n = n_vis = hit10 = hit10_all = hit_obs = n_obs = 0
        terr: list[float] = []
        terr_clean: list[float] = []
        trigger = rescue = 0
        prior_edge = 0.0
        for vid, ann_path in pairs:
            ann = json.loads(Path(ann_path).read_text())
            cam = _camera(ann)
            frames = _read_video(Path(vid))
            v3.reset()
            solver.reset()
            edge_mm = float(ann["info"].get("edge_length_mm", 100.0))
            prior_edge = 0.0
            for fr in ann["frames"]:
                fi = int(fr.get("frame_index", 0))
                if fi >= len(frames):
                    break
                gt = _gt(fr)
                all_vis = all(p[2] >= 2 for p in gt)
                n += 1
                if all_vis:
                    n_vis += 1
                def run_v3() -> dict:
                    o = v3.detect(frames[fi], cam["fx"], cam["fy"], cam["cx"], cam["cy"], None)
                    return dict(o.points) if (o is not None and o.ok and len(o.points) == 3) else {}

                if args.detector == "v3":
                    pts = run_v3()
                else:
                    r = b1.detect(frames[fi])
                    main_pts = dict(r.points) if r.ok else {}
                    far_pts = {}
                    far_ok = False
                    if args.strategy == "far_primary":
                        main_ok = len(main_pts) == 3 and plausible(main_pts, prior_edge)
                        run_far = args.far_period <= 1 or (fi % args.far_period == 0) or not main_ok
                        far_pts = run_v3() if run_far else {}
                        far_ok = bool(far_pts) and plausible(far_pts, prior_edge)
                        trigger += 1 if run_far else 0
                        pts = far_pts if far_ok else main_pts
                        if far_ok and not main_ok:
                            rescue += 1
                    else:
                        pts = main_pts
                        if args.strategy == "gate" and not (len(main_pts) == 3 and plausible(main_pts, prior_edge)):
                            trigger += 1
                            far_pts = run_v3()
                            if far_pts and plausible(far_pts, prior_edge):
                                pts = far_pts
                                rescue += 1
                errs = point_errs(pts, gt)
                all_obs = sum(1 for p in gt if p[2] >= 1) == 3
                if all_obs:
                    n_obs += 1
                    if errs is not None and len(errs) == 3 and max(errs) <= 10:
                        hit_obs += 1
                if all_vis and errs is not None and len(errs) == 3 and max(errs) <= 10:
                    hit10_all += 1
                if len(pts) == 3:
                    prior_edge = edge_stats(pts)[0]
                    res = solver.solve(pts, edge_mm, cam["cx"] * 2, cam["cy"] * 2, t_s=fi / 30.0)
                    if getattr(res, "ok", False) and res.translation_mm is not None:
                        t_gt = np.asarray(fr["pose_gt"]["translation_mm"], dtype=float)
                        e = float(np.linalg.norm(np.asarray(res.translation_mm) - t_gt))
                        terr.append(e)
                        if all_vis:
                            terr_clean.append(e)
        def stats(v: list[float]) -> dict:
            if not v:
                return {"n": 0}
            a = np.array(v)
            return {"n": len(a), "median": round(float(np.median(a)), 2),
                    "p90": round(float(np.percentile(a, 90)), 2),
                    "within50": round(float((a <= 50).mean()), 3)}
        out[preset] = {
            "frames": n,
            "det10_all_visible": round(hit10_all / max(1, n_vis), 4),
            "det10_observable": round(hit_obs / max(1, n_obs), 4),
            "n_all_visible": n_vis, "n_observable": n_obs,
            "pose_coverage": round(len(terr) / max(1, n), 4),
            "trigger": trigger, "rescue": rescue,
            "t_err_all": stats(terr),
            "t_err_clean": stats(terr_clean),
        }
        print(preset, json.dumps(out[preset], ensure_ascii=False), flush=True)
        degrade.set_active(None)
    if args.detector == "v3":
        args.out = args.out.replace(".json", "_v3.json")
    elif args.strategy != "gate":
        suffix = f"_{args.strategy}" + (f"_k{args.far_period}" if args.far_period > 1 else "")
        args.out = args.out.replace(".json", f"{suffix}.json")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
