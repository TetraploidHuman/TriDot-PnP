"""Eval learned finder vs B0. Finder NEVER uses B0 propose/detect."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from research.tridot_temporal.dataset import COLOR_ORDER, _discover_pairs
from research.tridot_temporal.find_geom import (
    GeomTripleNet,
    associate_velocity,
    decode_find_geom,
    upsample_heat,
)
from research.tridot_temporal.find_model import decode_heat_local, load_find_checkpoint
from research.tridot_temporal.find_preprocess import build_find_input

try:
    from research.baseline_b0.detector import B0Detector
except Exception:
    B0Detector = None
try:
    from research.baseline_b0.pnp import PoseSolver
except Exception:
    PoseSolver = None
try:
    from research.tridot_temporal.train_research import _extract_patches
    from research.tridot_temporal.triple_rank import CropRefineNet
except Exception:
    _extract_patches = None
    CropRefineNet = None


class _NoPose:
    def solve(self, *args, **kwargs):
        class R:
            ok = False
            translation_mm = None
        return R()

    def reset(self) -> None:
        return None


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Row:
    method: str
    seq_id: str
    factor: str
    frame_index: int
    detect_ok: bool
    detect_success: bool
    detect_success_20: bool
    detect_success_40: bool
    pixel_rmse: Optional[float]
    pose_ok: bool
    trans_err_mm: Optional[float]
    latency_ms: float
    gt_all_visible: bool
    labeled_s10: bool = False


def _gt(frame):
    k = frame["keypoints"]
    return {n: (float(k[3 * i]), float(k[3 * i + 1]), int(k[3 * i + 2])) for i, n in enumerate(COLOR_ORDER)}


def _point_errs(points, gt, min_vis: int):
    errs = []
    for n in COLOR_ORDER:
        if gt[n][2] < min_vis:
            continue
        if n not in points:
            return None
        errs.append(np.hypot(points[n][0] - gt[n][0], points[n][1] - gt[n][1]))
    return errs or None


def _metrics(points, gt):
    """Score fully visible LEDs only (vis==2)."""
    errs = _point_errs(points, gt, min_vis=2)
    if not errs:
        return None, False, False, False
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    return rmse, all(e <= 10 for e in errs), all(e <= 20 for e in errs), all(e <= 40 for e in errs)


def _labeled_hit(points, gt, thr: float = 10.0) -> bool:
    """Score every annotated LED, including partially occluded (vis>=1)."""
    errs = _point_errs(points, gt, min_vis=1)
    if not errs:
        return False
    return all(e <= thr for e in errs)


def _pose(points, fr, cam, edge, solver):
    if not fr.get("all_visible", False) or len(points) < 3:
        return False, None
    pose = solver.solve(
        points, edge, cam["fx"], cam["fy"], cam["cx"], cam["cy"],
        np.array(cam.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64),
    )
    if not pose.ok:
        solver.reset()
        return False, None
    t_gt = np.array(fr["pose_gt"]["translation_mm"], dtype=np.float64)
    return True, float(np.linalg.norm(pose.translation_mm - t_gt))


def eval_b0(video, ann) -> List[Row]:
    if B0Detector is None or PoseSolver is None:
        raise RuntimeError("baseline_b0 is unavailable in this checkout")
    det, solver = B0Detector(), PoseSolver()
    cam, edge = ann["camera"], float(ann["info"]["edge_length_mm"])
    cap = cv2.VideoCapture(str(video))
    rows, fi = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok or fi >= len(ann["frames"]):
            break
        fr = ann["frames"][fi]
        gt = _gt(fr)
        t0 = time.perf_counter()
        d = det.detect(bgr)
        lat = (time.perf_counter() - t0) * 1000
        pts = d.points if d.ok else {}
        rmse, s10, s20, s40 = _metrics(pts, gt) if d.ok else (None, False, False, False)
        pok, te = _pose(pts, fr, cam, edge, solver) if d.ok else (False, None)
        rows.append(Row("B0", ann["info"]["seq_id"], ann["info"].get("factor", "?"), fi,
                        bool(d.ok), bool(s10), bool(s20), bool(s40), rmse, pok, te, lat,
                        bool(fr.get("all_visible", False)),
                        _labeled_hit(pts, gt) if d.ok else False))
        fi += 1
    cap.release()
    return rows


def eval_hybrid_b0_find(
    video,
    ann,
    model,
    device,
    geom: Optional[GeomTripleNet] = None,
    name: str = "Hybrid_B0_FindNet",
    top_k: int = 5,
    upsample: int = 1,
    temporal_w: float = 3.0,
    sharpen: float = 1.0,
    **extra,
) -> List[Row]:
    """B0 when it finds a triangle; otherwise fall back to FindNet heatmap geom."""
    if B0Detector is None or PoseSolver is None:
        raise RuntimeError("baseline_b0 is unavailable in this checkout")
    find_rows = eval_find(
        video, ann, model, device, None, "FindNet_geom", decode="geom",
        geom=geom, top_k=top_k, upsample=upsample, temporal_w=temporal_w, sharpen=sharpen,
        **extra,
    )
    det, solver = B0Detector(), PoseSolver()
    cam, edge = ann["camera"], float(ann["info"]["edge_length_mm"])
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    out: List[Row] = []
    for fi, fr in enumerate(ann["frames"][: len(frames)]):
        gt = _gt(fr)
        t0 = time.perf_counter()
        d = det.detect(frames[fi])
        if d.ok:
            lat = (time.perf_counter() - t0) * 1000
            pts = d.points
            rmse, s10, s20, s40 = _metrics(pts, gt)
            pok, te = _pose(pts, fr, cam, edge, solver)
            out.append(Row(
                name, ann["info"]["seq_id"], ann["info"].get("factor", "?"), fi,
                True, bool(s10), bool(s20), bool(s40), rmse, pok, te, lat,
                bool(fr.get("all_visible", False)),
                _labeled_hit(pts, gt),
            ))
        else:
            r = find_rows[fi]
            out.append(Row(
                name, r.seq_id, r.factor, r.frame_index, r.detect_ok,
                r.detect_success, r.detect_success_20, r.detect_success_40,
                r.pixel_rmse, r.pose_ok, r.trans_err_mm, r.latency_ms, r.gt_all_visible,
                r.labeled_s10,
            ))
    return out


def eval_find(
    video,
    ann,
    model,
    device,
    refiner=None,
    name="FindNet",
    decode: str = "local",
    geom: Optional[GeomTripleNet] = None,
    top_k: int = 5,
    upsample: int = 1,
    temporal_w: float = 3.0,
    sharpen: float = 1.0,
    vel_w: float = 0.0,
    heat_prior_w: float = 0.0,
    heat_prior_sigma: float = 0.04,
    repeat_frame: bool = False,
    temporal_conf_gate: bool = False,
    temporal_conf_thr: float = 0.05,
    coast_margin_thr: float = -1.0,
    coast_snap: float = 0.08,
    geom_gap_gate: bool = False,
    geom_gap_thr: float = 1.0,
    geom_gap_sharp: float = 4.0,
    jump_rescue: bool = False,
    jump_thr: float = 0.08,
    jump_logit_slack: float = 1.0,
    jump_max_dist: float = 0.05,
    repeak_on_gap: bool = False,
    repeak_gap_thr: float = 1.0,
    repeak_prior_w: float = 1.5,
    repeak_prior_sigma: float = 0.035,
    repeak_evidence: bool = False,
    repeak_evidence_thr: float = 0.0,
    repeak_evidence_win: int = 5,
    occ_coast_max: int = 0,
    occ_coast_jump: float = 0.06,
    inject_expect: bool = False,
    inject_heat_frac: float = 0.45,
    inject_score_floor: float = 0.0,
    heat_hold: bool = False,
    heat_hold_jump: float = 0.06,
    heat_hold_frac: float = 0.45,
    heat_hold_snap: float = 0.04,
) -> List[Row]:
    """Points come ONLY from model heatmaps (+ optional refine / learned geom on those peaks)."""
    solver = PoseSolver() if PoseSolver is not None else _NoPose()
    cam, edge = ann["camera"], float(ann["info"]["edge_length_mm"])
    w, h = int(ann["info"]["width"]), int(ann["info"]["height"])
    ih, iw = model.cfg.img_h, model.cfg.img_w
    t_len = model.cfg.seq_len
    cap = cv2.VideoCapture(str(video))
    frames_bgr, frames_x = [], []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames_bgr.append(bgr)
        frames_x.append(build_find_input(bgr, ih, iw))
    cap.release()

    rows = []
    model.eval()
    if refiner is not None:
        refiner.eval()
    if geom is not None:
        geom.eval()
    prev_xy: Optional[torch.Tensor] = None
    prev2_xy: Optional[torch.Tensor] = None
    prev_occ: Optional[np.ndarray] = None
    prev2_occ: Optional[np.ndarray] = None
    coast_age = np.zeros(3, dtype=np.int32)
    vel_coast = torch.zeros(1, 3, dtype=torch.long, device=device)
    use_temp = (
        temporal_w > 0 or heat_prior_w > 0 or coast_margin_thr >= 0
        or jump_rescue or repeak_on_gap or inject_expect or heat_hold
        or decode in ("peak", "vel", "geom_writeback")
    )
    for fi, fr in enumerate(ann["frames"][: len(frames_x)]):
        gt = _gt(fr)
        if repeat_frame:
            clip = [frames_x[fi]] * t_len
        else:
            start = max(0, fi - t_len + 1)
            clip = frames_x[start : fi + 1]
            if len(clip) < t_len:
                clip = [clip[0]] * (t_len - len(clip)) + clip
        x = torch.from_numpy(np.stack(clip)[None]).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            coords, vis, heat, coords0 = model(x, return_heat=True)
            hm = heat[:, -1]
            prev_arg = prev_xy if use_temp else None
            geom_kw = dict(
                top_k=top_k, upsample=upsample, temporal_w=temporal_w, sharpen=sharpen,
                prev_xy=prev_arg,
                prev2_xy=prev2_xy if use_temp else None,
                vel_w=vel_w, heat_prior_w=heat_prior_w, heat_prior_sigma=heat_prior_sigma,
                temporal_conf_gate=temporal_conf_gate,
                temporal_conf_thr=temporal_conf_thr,
                coast_margin_thr=coast_margin_thr,
                coast_snap=coast_snap,
                geom_gap_gate=geom_gap_gate,
                geom_gap_thr=geom_gap_thr,
                geom_gap_sharp=geom_gap_sharp,
                jump_rescue=jump_rescue,
                jump_thr=jump_thr,
                jump_logit_slack=jump_logit_slack,
                jump_max_dist=jump_max_dist,
                repeak_on_gap=repeak_on_gap,
                repeak_gap_thr=repeak_gap_thr,
                repeak_prior_w=repeak_prior_w,
                repeak_prior_sigma=repeak_prior_sigma,
                repeak_evidence=repeak_evidence,
                repeak_evidence_thr=repeak_evidence_thr,
                repeak_evidence_win=repeak_evidence_win,
                inject_expect=inject_expect,
                inject_heat_frac=inject_heat_frac,
                inject_score_floor=inject_score_floor,
                heat_hold=heat_hold,
                heat_hold_jump=heat_hold_jump,
                heat_hold_frac=heat_hold_frac,
                heat_hold_snap=heat_hold_snap,
            )
            if decode == "local":
                hm_u = upsample_heat(hm, upsample)
                win = 9 * max(upsample, 1)
                if win % 2 == 0:
                    win += 1
                c = decode_heat_local(hm_u, win=win, temp=14.0)[0].cpu().numpy()
            elif decode == "coords":
                # transformer-refined last-frame coords (training target path)
                c = coords[0, -1].cpu().numpy()
            elif decode == "coords0":
                # soft-argmax on heat only — no temporal residual
                c = coords0[0, -1].cpu().numpy()
            elif decode == "peak":
                # PeakContextNet / residual coords path — uses Transformer selection
                c = coords[0, -1].cpu().numpy()
                prev2_xy = prev_xy
                prev_xy = coords[:, -1].detach()
            elif decode == "vel":
                c_t, vel_coast = associate_velocity(
                    hm, prev_xy, prev2_xy, coast_age=vel_coast,
                    top_k=top_k, nms=5, snap=float(jump_max_dist or 0.05),
                    heat_frac=float(heat_hold_frac or 0.45),
                    max_coast=max(1, int(occ_coast_max) if occ_coast_max else 2),
                    vel_w=float(vel_w) if vel_w > 0 else 1.0,
                )
                c = c_t[0].cpu().numpy()
                prev2_xy = prev_xy
                prev_xy = c_t.detach()
            elif decode == "geom_writeback":
                # heat already wrote track expect when heat_writeback=True in the ckpt
                assert geom is not None
                c_t = decode_find_geom(hm, geom=geom, force_geom=True, **geom_kw)
                c = c_t[0].cpu().numpy()
                prev2_xy = prev_xy
                prev_xy = c_t.detach()
            elif decode == "analytic":
                c_t = decode_find_geom(hm, geom=None, force_geom=True, **geom_kw)
                c = c_t[0].cpu().numpy()
                prev2_xy = prev_xy
                prev_xy = c_t.detach()
            elif decode == "geom":
                assert geom is not None
                c_t = decode_find_geom(hm, geom=geom, force_geom=True, **geom_kw)
                c = c_t[0].cpu().numpy()
                prev2_xy = prev_xy
                prev_xy = c_t.detach()
            elif decode == "geom_coords_prior":
                # inject transformer coords into heat before TopK (no FindNet FT)
                assert geom is not None
                from research.tridot_temporal.find_geom import apply_temporal_heat_prior
                hm_p = apply_temporal_heat_prior(
                    hm, coords[:, -1], strength=float(heat_prior_w or 0.35),
                    sigma=float(heat_prior_sigma),
                )
                c_t = decode_find_geom(hm_p, geom=geom, force_geom=True, **geom_kw)
                c = c_t[0].cpu().numpy()
                prev2_xy = prev_xy
                prev_xy = c_t.detach()
            elif decode == "geom_fast":
                assert geom is not None
                c_t = decode_find_geom(hm, geom=geom, force_geom=False, **geom_kw)
                c = c_t[0].cpu().numpy()
                prev2_xy = prev_xy
                prev_xy = c_t.detach()
            else:
                raise ValueError(decode)
            # Occ-coast (replace jumped dets with velocity) is disabled by default:
            # diagnostics show it false-fires on ~13% of clear frames and is NOT
            # closer to GT than geom on cut/vis1. If enabled, freeze the track
            # while coasting so coasted coords never become the next expect.
            if occ_coast_max > 0 and prev_occ is not None:
                expect = prev_occ.copy()
                if prev2_occ is not None:
                    expect = np.clip(prev_occ + (prev_occ - prev2_occ), 0.0, 1.0)
                trusted = c.copy()
                for ci in range(3):
                    jumped = float(np.linalg.norm(c[ci] - expect[ci])) > occ_coast_jump
                    if jumped and coast_age[ci] < occ_coast_max:
                        c[ci] = expect[ci]
                        coast_age[ci] += 1
                        trusted[ci] = prev_occ[ci]  # keep last confident track point
                    else:
                        coast_age[ci] = 0
                prev_xy = torch.from_numpy(trusted[None].astype(np.float32)).to(device)
                prev2_occ = prev_occ
                prev_occ = trusted
            else:
                prev2_occ = prev_occ
                prev_occ = c.copy()
            v = torch.sigmoid(vis[0, -1]).cpu().numpy()
        pts = {n: (float(c[i, 0] * w), float(c[i, 1] * h)) for i, n in enumerate(COLOR_ORDER)}
        if refiner is not None and _extract_patches is not None:
            coarse = np.array([pts[n] for n in COLOR_ORDER], dtype=np.float32)
            patches = _extract_patches(frames_bgr[fi], coarse, refiner.patch)
            with torch.no_grad():
                delta = refiner(torch.from_numpy(patches[None]).to(device))[0].cpu().numpy()
            refined = np.clip(coarse / np.array([w, h], np.float32) + delta, 0, 1)
            pts = {n: (float(refined[i, 0] * w), float(refined[i, 1] * h)) for i, n in enumerate(COLOR_ORDER)}
        lat = (time.perf_counter() - t0) * 1000
        ok = bool((v >= 0.35).all())
        rmse, s10, s20, s40 = _metrics(pts, gt)
        pok, te = _pose(pts, fr, cam, edge, solver) if ok else (False, None)
        rows.append(Row(name, ann["info"]["seq_id"], ann["info"].get("factor", "?"), fi,
                        ok, bool(s10), bool(s20), bool(s40), rmse, pok, te, lat,
                        bool(fr.get("all_visible", False)),
                        _labeled_hit(pts, gt)))
    return rows


def _mean_flag(rows: List[Row], attr: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([getattr(r, attr) for r in rows]))


def _summarize(rows: List[Row]) -> dict:
    denom = [r for r in rows if r.gt_all_visible] or rows
    cut = [r for r in rows if not r.gt_all_visible]
    pix = [r.pixel_rmse for r in rows if r.pixel_rmse is not None]
    pose = [r for r in rows if r.pose_ok and r.trans_err_mm is not None]
    lat = [r.latency_ms for r in rows]
    by_f = {}
    for f in sorted({r.factor for r in rows}):
        rs = [r for r in rows if r.factor == f]
        d = [r for r in rs if r.gt_all_visible] or rs
        c = [r for r in rs if not r.gt_all_visible]
        p = [r.pixel_rmse for r in rs if r.pixel_rmse is not None]
        by_f[f] = {
            "det@10": float(np.mean([r.detect_success for r in d])) if d else 0.0,
            "det@10_all": _mean_flag(rs, "labeled_s10"),
            "det@10_cut": _mean_flag(c, "labeled_s10") if c else float("nan"),
            "pix": float(np.mean(p)) if p else float("nan"),
            "n_cut": len(c),
        }
    return {
        "det@10": float(np.mean([r.detect_success for r in denom])) if denom else 0.0,
        "det@20": float(np.mean([r.detect_success_20 for r in denom])) if denom else 0.0,
        "det@40": float(np.mean([r.detect_success_40 for r in denom])) if denom else 0.0,
        "det@10_all": _mean_flag(rows, "labeled_s10"),
        "det@10_cut": _mean_flag(cut, "labeled_s10") if cut else float("nan"),
        "pix_rmse": float(np.mean(pix)) if pix else float("nan"),
        "t_err": float(np.mean([r.trans_err_mm for r in pose])) if pose else float("nan"),
        "fps": float(1000.0 / np.mean(lat)) if lat else 0.0,
        "by_factor": by_f,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "TriDotPose")
    ap.add_argument("--find-ckpt", type=Path, required=True)
    ap.add_argument("--refine-ckpt", type=Path, default=None,
                    help="optional CropRefine; applied on MODEL peaks only (not B0)")
    ap.add_argument("--geom-ckpt", type=Path, default=None,
                    help="optional GeomTripleNet on heatmap TopK peaks")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--upsample", type=int, default=1)
    ap.add_argument("--temporal-w", type=float, default=3.0,
                    help="temporal triple bias; v3.6 best uses 3.0 with up1")
    ap.add_argument("--vel-w", type=float, default=0.0,
                    help="constant-velocity mix into geom temporal target")
    ap.add_argument("--heat-prior-w", type=float, default=0.0,
                    help="Gaussian bump on heat at previous coords before peak extract")
    ap.add_argument("--heat-prior-sigma", type=float, default=0.04)
    ap.add_argument("--repeat-frame", action="store_true",
                    help="ablate net temporal: repeat current frame for the whole clip")
    ap.add_argument("--temporal-conf-gate", action="store_true", default=False,
                    help="scale temporal_w by peak-margin confidence (inference-only)")
    ap.add_argument("--temporal-conf-thr", type=float, default=0.05,
                    help="margin midpoint for temporal conf gate")
    ap.add_argument("--coast-margin-thr", type=float, default=-1.0,
                    help="legacy peak-margin coast (keep <0; broken on this set)")
    ap.add_argument("--coast-snap", type=float, default=0.08,
                    help="max dist to snap coasted xy onto a TopK peak")
    ap.add_argument("--geom-gap-gate", action="store_true", default=False,
                    help="scale temporal_w by geom best-2nd logit gap (preferred traj gate)")
    ap.add_argument("--geom-gap-thr", type=float, default=1.0,
                    help="gap midpoint: below → use traj; above → trust geom")
    ap.add_argument("--geom-gap-sharp", type=float, default=4.0,
                    help="sigmoid sharpness for geom-gap gate")
    ap.add_argument("--jump-rescue", action="store_true", default=False,
                    help="if geom jumps off track, pick near-track competitive triple")
    ap.add_argument("--jump-thr", type=float, default=0.08,
                    help="min dist(geom, expect) to trigger jump rescue")
    ap.add_argument("--jump-logit-slack", type=float, default=1.0,
                    help="max geom-logit drop allowed for rescued triple")
    ap.add_argument("--jump-max-dist", type=float, default=0.05,
                    help="max dist(triple, expect) for rescue candidates")
    ap.add_argument("--repeak-on-gap", action="store_true", default=False,
                    help="if geom gap low, bump heat at track expect and re-extract peaks")
    ap.add_argument("--repeak-gap-thr", type=float, default=1.0)
    ap.add_argument("--repeak-prior-w", type=float, default=1.5)
    ap.add_argument("--repeak-prior-sigma", type=float, default=0.035)
    ap.add_argument("--repeak-evidence", action="store_true", default=False,
                    help="only repeak colors with local heat evidence at track expect")
    ap.add_argument("--repeak-evidence-thr", type=float, default=0.0)
    ap.add_argument("--repeak-evidence-win", type=int, default=5)
    ap.add_argument("--occ-coast-max", type=int, default=0,
                    help="per-color frames to follow velocity when the detection jumps off the track")
    ap.add_argument("--inject-expect", action="store_true", default=False,
                    help="add velocity-expect as an extra peak when local heat supports it")
    ap.add_argument("--inject-heat-frac", type=float, default=0.45,
                    help="min heat(expect)/top-peak to inject the extra candidate")
    ap.add_argument("--inject-score-floor", type=float, default=0.0,
                    help="raise injected peak score to this fraction of the top peak")
    ap.add_argument("--heat-hold", action="store_true", default=False,
                    help="if a color jumps off-track but expect still has heat, snap back")
    ap.add_argument("--heat-hold-jump", type=float, default=0.06)
    ap.add_argument("--heat-hold-frac", type=float, default=0.45)
    ap.add_argument("--heat-hold-snap", type=float, default=0.04,
                    help="max dist to snap hold target onto a real TopK peak")
    ap.add_argument("--occ-coast-jump", type=float, default=0.06,
                    help="normalized jump above which a color is treated as interrupted")
    ap.add_argument("--sharpen", type=float, default=1.0,
                    help="heat logit scale before upsample; 1.0 = off")
    ap.add_argument("--modes", type=str, default="all",
                    help="comma list: b0,local,coords,coords0,peak,vel,analytic,geom,"
                         "geom_writeback,geom_coords_prior,geom_fast,refine,hybrid or all")
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "results" / "find_v3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--subset", default="test",
        help="splits.json key: train|val|test|hard_test (realistic stress set)",
    )
    args = ap.parse_args(argv)
    if args.modes == "all":
        default_modes = ["local", "coords", "peak", "vel", "analytic", "geom"]
        if B0Detector is not None:
            default_modes = ["b0"] + default_modes + ["geom_fast", "refine", "hybrid"]
        modes = set(default_modes)
    else:
        modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_find_checkpoint(args.find_ckpt, device)
    refiner = None
    if args.refine_ckpt and Path(args.refine_ckpt).exists() and CropRefineNet is not None:
        rk = torch.load(args.refine_ckpt, map_location=device, weights_only=False)
        refiner = CropRefineNet().to(device)
        refiner.load_state_dict(rk["model"])
        refiner.eval()
    geom = None
    if args.geom_ckpt and Path(args.geom_ckpt).exists():
        gk = torch.load(args.geom_ckpt, map_location=device, weights_only=False)
        geom = GeomTripleNet().to(device)
        geom.load_state_dict(gk["model"])
        geom.eval()
        if "top_k" in gk:
            args.top_k = int(gk["top_k"])
        if "upsample" in gk:
            args.upsample = int(gk["upsample"])

    pairs = _discover_pairs(args.dataset, "synthetic", subset=args.subset)
    print(f"[eval] subset={args.subset} sequences={len(pairs)}", flush=True)
    if not pairs:
        raise SystemExit(f"no sequences for subset={args.subset}")
    all_rows: List[Row] = []
    common = dict(
        top_k=args.top_k, upsample=args.upsample,
        temporal_w=args.temporal_w, sharpen=args.sharpen,
        vel_w=args.vel_w, heat_prior_w=args.heat_prior_w,
        heat_prior_sigma=args.heat_prior_sigma, repeat_frame=args.repeat_frame,
        temporal_conf_gate=args.temporal_conf_gate,
        temporal_conf_thr=args.temporal_conf_thr,
        coast_margin_thr=args.coast_margin_thr,
        coast_snap=args.coast_snap,
        geom_gap_gate=args.geom_gap_gate,
        geom_gap_thr=args.geom_gap_thr,
        geom_gap_sharp=args.geom_gap_sharp,
        jump_rescue=args.jump_rescue,
        jump_thr=args.jump_thr,
        jump_logit_slack=args.jump_logit_slack,
        jump_max_dist=args.jump_max_dist,
        repeak_on_gap=args.repeak_on_gap,
        repeak_gap_thr=args.repeak_gap_thr,
        repeak_prior_w=args.repeak_prior_w,
        repeak_prior_sigma=args.repeak_prior_sigma,
        repeak_evidence=args.repeak_evidence,
        repeak_evidence_thr=args.repeak_evidence_thr,
        repeak_evidence_win=args.repeak_evidence_win,
        occ_coast_max=args.occ_coast_max,
        occ_coast_jump=args.occ_coast_jump,
        inject_expect=args.inject_expect,
        inject_heat_frac=args.inject_heat_frac,
        inject_score_floor=args.inject_score_floor,
        heat_hold=args.heat_hold,
        heat_hold_jump=args.heat_hold_jump,
        heat_hold_frac=args.heat_hold_frac,
        heat_hold_snap=args.heat_hold_snap,
    )
    for vid, ann_p in pairs:
        ann = json.loads(ann_p.read_text(encoding="utf-8"))
        print(f"== {ann_p.stem}", flush=True)
        if "b0" in modes:
            all_rows.extend(eval_b0(vid, ann))
        if "local" in modes:
            all_rows.extend(eval_find(
                vid, ann, model, device, None, "FindNet_local", decode="local", **common
            ))
        if "coords" in modes:
            all_rows.extend(eval_find(
                vid, ann, model, device, None, "FindNet_coords", decode="coords", **common
            ))
        if "coords0" in modes:
            all_rows.extend(eval_find(
                vid, ann, model, device, None, "FindNet_coords0", decode="coords0", **common
            ))
        if "peak" in modes:
            all_rows.extend(eval_find(
                vid, ann, model, device, None, "FindNet_peak", decode="peak", **common
            ))
        if "vel" in modes:
            all_rows.extend(eval_find(
                vid, ann, model, device, None, "FindNet_vel", decode="vel", **common
            ))
        if "geom_writeback" in modes and geom is not None:
            all_rows.extend(eval_find(
                vid, ann, model, device, None, "FindNet_geom_writeback",
                decode="geom_writeback", geom=geom, **common,
            ))
        if "analytic" in modes:
            all_rows.extend(eval_find(
                vid, ann, model, device, None, "FindNet_analytic", decode="analytic", **common
            ))
        if "geom" in modes and geom is not None:
            all_rows.extend(eval_find(
                vid, ann, model, device, None, "FindNet_geom", decode="geom",
                geom=geom, **common,
            ))
        if "geom_coords_prior" in modes and geom is not None:
            all_rows.extend(eval_find(
                vid, ann, model, device, None, "FindNet_geom_coords_prior",
                decode="geom_coords_prior", geom=geom, **common,
            ))
        if "geom_fast" in modes and geom is not None:
            all_rows.extend(eval_find(
                vid, ann, model, device, None, "FindNet_geom_fast", decode="geom_fast",
                geom=geom, **common,
            ))
        if "refine" in modes and refiner is not None and geom is not None:
            all_rows.extend(eval_find(
                vid, ann, model, device, refiner, "FindNet_geom_Refine",
                decode="geom", geom=geom, **common,
            ))
        if "hybrid" in modes and geom is not None:
            all_rows.extend(eval_hybrid_b0_find(
                vid, ann, model, device, geom=geom, name="Hybrid_B0_FindNet",
                **common,
            ))

    summary = {m: _summarize([r for r in all_rows if r.method == m]) for m in sorted({r.method for r in all_rows})}
    for m, s in summary.items():
        print(
            f"{m}: det@10={s['det@10']:.3f} det@10_all={s['det@10_all']:.3f} "
            f"det@10_cut={s['det@10_cut']:.3f} pix={s['pix_rmse']:.2f}",
            flush=True,
        )

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "per_frame.json").write_text(json.dumps([asdict(r) for r in all_rows], indent=2), encoding="utf-8")
    methods = list(summary.keys())
    factors = sorted({f for s in summary.values() for f in s["by_factor"]})
    lines = [
        "# FindNet：纯学习找点 vs B0（无经典 proposal）",
        "",
        f"- 时间：{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "- 允许：颜色亲和/亮度等**稠密预处理**作为网络输入通道",
        "- 禁止：经典连通域斑点找点；候选峰来自**模型热图**；三角几何可用解析分或可学习 GeomNet",
        "",
        "| method | det@10 | det@10_all | det@10_cut | det@20 | det@40 | pix_rmse | t_err | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in methods:
        s = summary[m]
        lines.append(
            f"| {m} | {s['det@10']:.3f} | {s['det@10_all']:.3f} | {s['det@10_cut']:.3f} | "
            f"{s['det@20']:.3f} | {s['det@40']:.3f} | "
            f"{s['pix_rmse']:.2f} | {s['t_err']:.1f} | {s['fps']:.1f} |"
        )
    lines += [
        "",
        "det@10 只统计三点都可见的帧。det@10_all 统计全部帧，遮挡点（vis>=1）也要在 10px 内。"
        "det@10_cut 只统计发生中断的帧。",
        "",
        "## 分因子 det@10",
        "",
        "| factor | " + " | ".join(methods) + " |",
    ]
    lines.append("|---|" + "---:|" * len(methods))
    for f in factors:
        lines.append("| " + f + " | " + " | ".join(f"{summary[m]['by_factor'][f]['det@10']:.3f}" for m in methods) + " |")
    for title, key in (("分因子 det@10_all（含中断）", "det@10_all"), ("分因子 det@10_cut（仅中断帧）", "det@10_cut")):
        lines += ["", f"## {title}", "", "| factor | " + " | ".join(methods) + " |"]
        lines.append("|---|" + "---:|" * len(methods))
        for f in factors:
            cells = []
            for m in methods:
                v = summary[m]["by_factor"][f][key]
                cells.append("nan" if v != v else f"{v:.3f}")
            lines.append("| " + f + " | " + " | ".join(cells) + " |")
    lines += ["", "## 分因子 pix", "", "| factor | " + " | ".join(methods) + " |"]
    lines.append("|---|" + "---:|" * len(methods))
    for f in factors:
        cells = [("nan" if (v := summary[m]["by_factor"][f]["pix"]) != v else f"{v:.1f}") for m in methods]
        lines.append("| " + f + " | " + " | ".join(cells) + " |")
    report = "\n".join(lines) + "\n"
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    (ROOT / "research" / "phase5_find_v3_report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
