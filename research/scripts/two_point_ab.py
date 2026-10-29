"""Evaluate two-point + motion-prior PnP on frames with exactly two visible dots.

Same frames / detector / prior chain for every row:

* ``current``    -- shipped 3-point pipeline (on two-visible frames it still emits a
  pose, but from a bogus third colour candidate);
* ``completion`` -- project the missing dot from the prior, then 3-point PnP;
* ``two_point``  -- 2 observed pixels (top-1 candidates) + prior, joint 6-DOF solve;
* ``near``       -- observed pixel = candidate nearest to the prediction (gated);
* ``checked``    -- ``near`` + the missing dot's predicted pixel must not carry that
  colour's evidence (it is supposed to be invisible);
* ``oracle``     -- ground-truth pixels for the two visible dots (estimation bound).

Both a *static* prior (last known pose) and a *velocity* prior (constant velocity
and angular velocity extrapolation) are evaluated, because the shipped tracker is
an alpha-beta filter and keeps a velocity.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from research.baseline_b0.pnp import PoseSolver
from research.tridot_temporal import eval_v2 as ev
from research.tridot_temporal.app_pnp import object_points_mm
from research.tridot_temporal.classic_v3 import ClassicV3Detector
from research.tridot_temporal.pnp2 import (
    exp_so3, log_so3, rotation_angle_deg, solve_pose_2point,
    solve_pose_2point_plane,
)

COLORS = ("red", "green", "blue")
MISSING_EVIDENCE_MAX = 12.0


def _project(R, t, P, fx, fy, cx, cy) -> Optional[np.ndarray]:
    p = R @ P + t
    if p[2] <= 1.0:
        return None
    return np.array([fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy])


def _evidence(m: np.ndarray, u: float, v: float, radius: int = 5) -> float:
    h, w = m.shape
    x0, x1 = max(0, int(round(u)) - radius), min(w, int(round(u)) + radius + 1)
    y0, y1 = max(0, int(round(v)) - radius), min(h, int(round(v)) + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(m[y0:y1, x0:x1].mean())


def _pnp3(pts: Dict[str, tuple], solver: PoseSolver, edge: float,
          fx: float, fy: float, cx: float, cy: float):
    pose = solver.solve(pts, edge, fx, fy, cx, cy)
    if not getattr(pose, "ok", False):
        return None, None
    return (np.asarray(pose.translation_mm, dtype=float),
            np.asarray(pose.rotation, dtype=float))


def _pose_stats(res, gt_t: np.ndarray, gt_R: np.ndarray) -> dict:
    if not res.ok:
        return {"ok": False, "reason": res.reason}
    return {"ok": True,
            "terr": float(np.linalg.norm(res.translation_mm - gt_t)),
            "roterr": rotation_angle_deg(res.rotation, gt_R),
            "reproj": float(res.reproj_px)}


def _prior_at(fi: int, hist: List[Tuple[int, np.ndarray, np.ndarray]],
              velocity: bool) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not hist:
        return None
    f2, R2, t2 = hist[-1]
    if not velocity or len(hist) < 2 or fi <= f2:
        return R2, t2
    f1, R1, t1 = hist[-2]
    dt = max(1, f2 - f1)
    gap = fi - f2
    t_pred = t2 + (t2 - t1) / dt * gap
    w = log_so3(R1.T @ R2) / dt
    R_pred = R2 @ exp_so3(w * gap)
    return R_pred, t_pred


def _attempt(Rp, tp, vis2, obj, gt, gt_t, gt_R, cands, maps,
             fx, fy, cx, cy, edge, solver, max_rot_jump, gate_frac=0.45) -> dict:
    gate = max(12.0, gate_frac * fx * edge / max(1.0, float(np.linalg.norm(tp))))
    top_uv: List[List[float]] = []
    near_uv: List[List[float]] = []
    det_errs: Dict[str, float] = {}
    for i in vis2:
        c = COLORS[i]
        lst = cands.get(c) or []
        pred = _project(Rp, tp, obj[i], fx, fy, cx, cy)
        if not lst:
            continue
        top = lst[0]
        det_errs[c] = float(np.hypot(top.u - gt[i][0], top.v - gt[i][1]))
        top_uv.append([float(top.u), float(top.v)])
        if pred is None:
            continue
        near = min(lst, key=lambda b: np.hypot(b.u - pred[0], b.v - pred[1]))
        if np.hypot(near.u - pred[0], near.v - pred[1]) <= gate:
            near_uv.append([float(near.u), float(near.v)])
    out: dict = {"det_errs": det_errs,
                 "near_detect": len(near_uv) == 2,
                 "gate_px": gate}
    if len(top_uv) == 2:
        res_plane = solve_pose_2point_plane(top_uv, obj[vis2], fx, fy, cx, cy)
        out["plane"] = _pose_stats(res_plane, gt_t, gt_R)
        res = solve_pose_2point(top_uv, obj[vis2], fx, fy, cx, cy, Rp, tp,
                                max_prior_jump_deg=max_rot_jump)
        out["two_point"] = _pose_stats(res, gt_t, gt_R)
    if len(near_uv) == 2:
        res_n = solve_pose_2point(near_uv, obj[vis2], fx, fy, cx, cy, Rp, tp,
                                  max_prior_jump_deg=max_rot_jump)
        out["near"] = _pose_stats(res_n, gt_t, gt_R)
        if res_n.ok:
            missing = [i for i in range(3) if i not in vis2][0]
            q = _project(res_n.rotation, res_n.translation_mm, obj[missing],
                         fx, fy, cx, cy)
            if q is not None:
                evd = _evidence(maps[COLORS[missing]], q[0], q[1])
                out["missing_evidence"] = evd
                out["checked"] = dict(out["near"])
                out["checked"]["passes"] = evd <= MISSING_EVIDENCE_MAX
    res_o = solve_pose_2point([gt[vis2[0]][:2], gt[vis2[1]][:2]], obj[vis2],
                              fx, fy, cx, cy, Rp, tp,
                              max_prior_jump_deg=max_rot_jump)
    out["oracle"] = _pose_stats(res_o, gt_t, gt_R)
    return out


def run(dataset: Path, subset: str, limit_seq: Optional[int], out: Path,
        max_rot_jump: float = 25.0, factors: Optional[List[str]] = None,
        velocity: bool = True) -> None:
    pairs = ev._discover_pairs(dataset, "synthetic", subset=subset)
    rows: List[dict] = []
    t_start = time.time()
    n_seq = 0
    for si, (video, ann_path) in enumerate(pairs[:limit_seq] if limit_seq else pairs):
        ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        factor = str(ann["info"].get("factor", ""))
        if factors and factor not in factors:
            continue
        n_seq += 1
        cam, edge = ev._camera(ann), float(ann["info"]["edge_length_mm"])
        fx, fy = ev.camera_fx(cam), ev.camera_fy(cam)
        cx, cy = ev.camera_cx(cam), ev.camera_cy(cam)
        obj = object_points_mm(edge)
        det = ClassicV3Detector(frontend="union", contrast_floor=12.0)
        solver = PoseSolver()
        frames = ev._read_video(video)
        n = min(len(frames), len(ann["frames"]))
        hist: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for fi in range(n):
            fr = ann["frames"][fi]
            gt = np.asarray(fr["keypoints"], dtype=float).reshape(3, 3)
            vis = [int(g[2]) for g in gt]
            vis2 = [i for i in range(3) if vis[i] == 2]
            frame = frames[fi]
            d = det.detect(frame, fx, fy, cx, cy)
            cur_t = cur_R = None
            if getattr(d, "ok", False) and len(getattr(d, "points", {})) >= 3:
                cur_t, cur_R = _pnp3(
                    {c: (float(d.points[c][0]), float(d.points[c][1])) for c in COLORS},
                    solver, edge, fx, fy, cx, cy)
            if len(vis2) == 3 and cur_t is not None:
                hist.append((fi, cur_R, cur_t))
                if len(hist) > 6:
                    hist.pop(0)
            if len(vis2) != 2:
                continue
            gt_t = np.asarray(fr["pose_gt"]["translation_mm"], dtype=float)
            gt_R = np.asarray(fr["pose_gt"]["rotation_matrix"], dtype=float)
            row: dict = dict(seq_id=ann["info"]["seq_id"], factor=factor, frame=fi,
                             vis=vis,
                             cur_ok=cur_t is not None,
                             cur_terr=(float(np.linalg.norm(cur_t - gt_t))
                                       if cur_t is not None else None),
                             cur_roterr=(rotation_angle_deg(cur_R, gt_R)
                                         if cur_R is not None else None),
                             prior_age=(fi - hist[-1][0]) if hist else None,
                             n_prior=len(hist))
            if getattr(d, "ok", False):
                tp = [[float(d.points[COLORS[i]][0]), float(d.points[COLORS[i]][1])]
                      for i in vis2 if COLORS[i] in d.points]
                if len(tp) == 2:
                    rp = solve_pose_2point_plane(np.array(tp), obj[vis2], fx, fy, cx, cy)
                    row["plane"] = _pose_stats(rp, gt_t, gt_R)
            maps = det.candidate_maps(frame)
            cands = det.candidates_from_maps(maps, frame)
            # completion baseline shares the static prior
            static = _prior_at(fi, hist, velocity=False)
            if static is not None:
                st = _attempt(static[0], static[1], vis2, obj, gt, gt_t, gt_R,
                              cands, maps, fx, fy, cx, cy, edge, solver,
                              max_rot_jump)
                row["static"] = st
                # completion: missing dot from the static prior
                missing = [i for i in range(3) if i not in vis2][0]
                q = _project(static[0], static[1], obj[missing], fx, fy, cx, cy)
                top = {COLORS[i]: cands[COLORS[i]][0] for i in vis2
                       if cands.get(COLORS[i])}
                if q is not None and len(top) == 2:
                    pts = {c: (float(b.u), float(b.v)) for c, b in top.items()}
                    pts[COLORS[missing]] = (float(q[0]), float(q[1]))
                    comp_t, comp_R = _pnp3(pts, solver, edge, fx, fy, cx, cy)
                    if comp_t is not None:
                        row["completion"] = {
                            "ok": True,
                            "terr": float(np.linalg.norm(comp_t - gt_t)),
                            "roterr": rotation_angle_deg(comp_R, gt_R)}
                if velocity:
                    pred = _prior_at(fi, hist, velocity=True)
                    if pred is not None:
                        row["velocity"] = _attempt(pred[0], pred[1], vis2, obj, gt,
                                                   gt_t, gt_R, cands, maps,
                                                   fx, fy, cx, cy, edge, solver,
                                                   max_rot_jump)
            rows.append(row)
        print(f"[two_point] {n_seq} {ann['info']['seq_id']} ({factor}) frames={n} "
              f"rows={len(rows)} t={time.time() - t_start:.0f}s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    out.write_text(json.dumps({"subset": subset, "n_rows": len(rows),
                               "factors": factors, "max_rot_jump": max_rot_jump,
                               "summary": summary, "rows": rows}, indent=1),
                   encoding="utf-8")
    print(json.dumps(summary, indent=1))


def _stat(vals: List[float]) -> dict:
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=float)
    return {"n": int(a.size), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            "within50": float(np.mean(a <= 50.0))}


def _method_stats(rows: List[dict], group: Optional[str], key: str) -> dict:
    n = len(rows)
    if group is None:
        having = [r for r in rows if key in r]
    else:
        having = [r for r in rows if group in r and key in r[group]]
    def get(r):
        return r[key] if group is None else r[group][key]
    ok = [r for r in having if get(r).get("ok")]
    return {"coverage": len(ok) / max(1, n),
            "t_err": _stat([get(r)["terr"] for r in ok]),
            "rot_err": _stat([get(r)["roterr"] for r in ok])}


def _checked_stats(rows: List[dict], group: Optional[str]) -> dict:
    n = len(rows)
    def get(r):
        return (r.get(group) or {}).get("checked") if group else r.get("checked")
    having = [r for r in rows if get(r)]
    passed = [r for r in having if get(r).get("passes")]
    return {"pass_rate": len(passed) / max(1, n),
            "coverage": len(passed) / max(1, n),
            "t_err": _stat([get(r)["terr"] for r in passed]),
            "rot_err": _stat([get(r)["roterr"] for r in passed])}


def summarize(rows: List[dict]) -> dict:
    n = len(rows)
    out: Dict[str, object] = {"n_frames_2visible": n,
                              "prior_available": sum(1 for r in rows if r.get("static")) / max(1, n)}
    out["prior_age"] = _stat([r["prior_age"] for r in rows if r["prior_age"] is not None])
    cur_ok = [r for r in rows if r["cur_ok"]]
    out["current"] = {"coverage": len(cur_ok) / max(1, n),
                      "t_err": _stat([r["cur_terr"] for r in cur_ok]),
                      "rot_err": _stat([r["cur_roterr"] for r in cur_ok])}
    out["static"] = {m: _method_stats(rows, "static", m)
                     for m in ("two_point", "near", "oracle")}
    out["static"]["checked"] = _checked_stats(rows, "static")
    out["completion"] = _method_stats(rows, None, "completion")
    out["plane"] = _method_stats(rows, None, "plane")
    if any("velocity" in r for r in rows):
        out["velocity"] = {m: _method_stats(rows, "velocity", m)
                           for m in ("two_point", "near", "oracle")}
        out["velocity"]["checked"] = _checked_stats(rows, "velocity")
    errs = [e for r in rows for e in (r.get("static", {}).get("det_errs") or {}).values()]
    out["detection_px_err"] = _stat(errs)
    if errs:
        out["near_detect_rate"] = float(np.mean(
            [r["static"]["near_detect"] for r in rows if r.get("static")]))
    by_factor: Dict[str, dict] = {}
    for r in rows:
        by_factor.setdefault(r["factor"] or "unknown", []).append(r)
    out["by_factor"] = {}
    for f, rs in sorted(by_factor.items()):
        d = {"n": len(rs),
             "prior_available": sum(1 for r in rs if r.get("static")) / len(rs),
             "cur_cov": sum(1 for r in rs if r["cur_ok"]) / len(rs),
             "static_checked_cov": _checked_stats(rs, "static")["coverage"],
             "static_checked_t": _checked_stats(rs, "static")["t_err"]}
        if any("velocity" in r for r in rs):
            d["vel_checked_cov"] = _checked_stats(rs, "velocity")["coverage"]
            d["vel_checked_t"] = _checked_stats(rs, "velocity")["t_err"]
        out["by_factor"][f] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/TriDotPose")
    ap.add_argument("--subset", default="hard_test")
    ap.add_argument("--limit-seq", type=int, default=None)
    ap.add_argument("--factors", default=None, help="comma separated factor filter")
    ap.add_argument("--max-rot-jump", type=float, default=25.0)
    ap.add_argument("--no-velocity", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    factors = [f.strip() for f in args.factors.split(",")] if args.factors else None
    tag = f"{args.subset}" + (f"_limit{args.limit_seq}" if args.limit_seq else "")
    if factors:
        tag += "_" + "-".join(factors)
    out = Path(args.out) if args.out else Path(f"research/results/two_point/{tag}.json")
    run(Path(args.dataset), args.subset, args.limit_seq, out, args.max_rot_jump,
        factors, velocity=not args.no_velocity)


if __name__ == "__main__":
    main()
