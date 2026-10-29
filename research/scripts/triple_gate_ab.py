"""A/B: 端侧主线（B1 端口）三点结果的「几何自洽门」是否值得开。

对每个三点全可见帧：
  1) B1 检测（app 主路径端口，grid 32）；
  2) 用等边性 + 视边长（这里用 GT 深度换算的 expectedEdge，比端侧"上一帧预测"更乐观）判定是否自洽；
  3) 不自洽时用 classic3 union + dot-cls 重找，命中则替代。
输出 det@10 all：B1 原始 vs 加门后。

注意：端侧实际用上一帧预测的视边长（会更松/更抖），本脚本用 GT 值，属于上界证据。
"""
import json, math
from pathlib import Path
import numpy as np
from research.tridot_temporal.eval_v2 import _camera, _gt, _read_video
from research.tridot_temporal.dataset import _discover_pairs, COLOR_ORDER
from research.tridot_temporal.classic_v3 import ClassicV3Detector
import research.pyc_import as pyc; pyc.install()
from research.baseline_b1.detector import B1Detector

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--limit-seq", type=int, default=6)
ap.add_argument("--out", default="research/results/classic_cv3/triple_gate_ab.json")
args = ap.parse_args()
pairs = _discover_pairs(Path("datasets/TriDotPose"), "synthetic", subset="hard_test")[: args.limit_seq]
b1 = B1Detector(grid_size=32)
v3 = ClassicV3Detector(frontend="union", contrast_floor=12.0,
                       dot_cls_path="research/results/dot_cls/dot_cls_scaled.pt",
                       dot_cls_scale=True)

def errs(points, gt):
    out=[]
    for c,(gx,gy,vis) in zip(COLOR_ORDER,gt):
        if vis<1: continue
        if c not in points: return None
        out.append(math.hypot(points[c][0]-gx, points[c][1]-gy))
    return out

stats={"b1":0,"gated":0,"n":0,"triggered":0}
for vid,annp in pairs:
    ann=json.loads(Path(annp).read_text()); cam=_camera(ann); frames=_read_video(Path(vid)); v3.reset()
    for fr in ann["frames"]:
        fi=int(fr.get("frame_index",0))
        if fi>=len(frames): break
        gt=_gt(fr)
        if sum(1 for p in gt if p[2]>=2)!=3: continue
        stats["n"]+=1
        depth=float(np.linalg.norm(np.asarray(fr["pose_gt"]["translation_mm"],dtype=float)))
        exp=cam["fx"]*float(ann["info"].get("edge_length_mm",100.0))/max(depth,1e-6)
        r=b1.detect(frames[fi])
        pts=dict(r.points) if r.ok else {}
        e=errs(pts,gt); hit1 = e is not None and max(e)<=10
        stats["b1"]+= int(hit1)
        # gate
        plausible=False
        if len(pts)==3:
            p=np.array([pts[c] for c in COLOR_ORDER],dtype=float)
            el=[np.linalg.norm(p[0]-p[1]),np.linalg.norm(p[1]-p[2]),np.linalg.norm(p[2]-p[0])]
            mean=sum(el)/3; spread=(max(el)-min(el))/mean if mean>0 else 9; ratio=mean/exp
            plausible = mean>=6 and spread<=0.5 and 0.55<=ratio<=1.45
        if not plausible:
            stats["triggered"]+=1
            o=v3.detect(frames[fi],cam["fx"],cam["fy"],cam["cx"],cam["cy"],None)
            if o is not None and o.ok:
                cand=o.points if isinstance(o.points, dict) else None
                if cand:
                    e2=errs(cand,gt)
                    if e2 is not None and max(e2)<=10: stats["gated"]+=1; continue
            if hit1: stats["gated"]+=1
        else:
            if hit1: stats["gated"]+=1
res = {"n": stats["n"], "triggered": stats["triggered"], "trigger_share": round(stats["triggered"]/stats["n"],4),
       "det10_b1": round(stats["b1"]/stats["n"],4), "det10_gated": round(stats["gated"]/stats["n"],4),
       "note": "expectedEdgePx 用 GT 深度换算（上界）；端侧用上一帧预测"}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=1))
print(res)
