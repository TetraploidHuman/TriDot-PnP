"""Honest, reproducible evaluation harness (v2) for TriDotPose methods.

Why another evaluator?  ``eval_find.py`` grew into a decoder-tuning workbench:
its ``det@10`` denominator is "frames where all three GT dots are visible",
``pix_rmse``/``t_err`` are conditional on success, and dozens of hand-tuned
offline switches change what is actually measured.  Its numbers therefore
cannot answer "is this method better than classic CV in the field?".

This harness keeps the *model* and *decoder* fixed and changes the *reporting*:

* fixed, explicit metrics from :mod:`research.tridot_temporal.metrics`
  (``det@10`` over all frames, per-point recall, occlusion behaviour, pose
  coverage, distance bias, jitter, recovery, latency, per-factor breakdown,
  sequence-bootstrap CIs);
* one code path per method with no evaluation-time gates, so a number cannot
  come from "if the heat looks bad, fall back to the previous frame";
* full provenance stored next to the numbers: command line, checkpoint hashes,
  split-file hash, library versions, seeds, timestamps.

Usage::

    # classic baseline, both subsets
    python -m research.tridot_temporal.eval_v2 --methods b0 --subset test \
        --out research/results/honest/b0_test

    # learned finder, no temporal heuristics
    python -m research.tridot_temporal.eval_v2 --methods find_geom --subset hard_test \
        --find-ckpt research/results/find_v3_12_data/best.pt \
        --geom-ckpt research/results/find_geom_v34/best.pt \
        --temporal-w 0 --out research/results/honest/find_hard

The learned path deliberately defaults to ``--temporal-w 0`` (pure per-frame
inference).  Pass ``--temporal-w 3`` to reproduce the historical temporal heat
prior, and report it as such.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from research.pyc_import import install as _install_pyc_import

_install_pyc_import()

from research.baseline_b0.detector import B0Detector  # noqa: E402
from research.baseline_b0.pnp import PoseSolver  # noqa: E402
from research.tridot_temporal.dataset import COLOR_ORDER, _discover_pairs  # noqa: E402
from research.tridot_temporal.find_geom import GeomTripleNet, decode_find_geom  # noqa: E402
from research.tridot_temporal.find_model import load_find_checkpoint  # noqa: E402
from research.tridot_temporal.find_offset import (  # noqa: E402
    decode_offset, refine_with_offset,
)
from research.tridot_temporal.app_pnp import AppFixedSolver  # noqa: E402
from research.tridot_temporal.find_preprocess import build_find_input  # noqa: E402
from research.tridot_temporal.completion import TemporalCompleter  # noqa: E402
from research.tridot_temporal import degrade  # noqa: E402
from research.tridot_temporal.classic_v2 import ClassicV2Detector  # noqa: E402
from research.tridot_temporal.classic_v3 import ClassicV3Detector  # noqa: E402
from research.tridot_temporal.metrics import (  # noqa: E402
    FrameRecord, det10_all, format_report, summarize, t_err_stats,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets" / "TriDotPose"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sha256(path: Optional[Path]) -> Optional[str]:
    if path is None or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gt(frame) -> List[List[float]]:
    k = frame["keypoints"]
    return [[float(k[3 * i]), float(k[3 * i + 1]), float(k[3 * i + 2])]
            for i in range(3)]


def _camera(ann) -> dict:
    return dict(ann["camera"])


def _solve_pose(points: Dict[str, tuple], fr, cam: dict, edge: float,
                solver: PoseSolver):
    """PnP on whatever three points we have; returns (ok, translation, reproj)."""
    if len(points) < 3:
        return False, None, None
    pose = solver.solve(
        {c: (float(points[c][0]), float(points[c][1])) for c in COLOR_ORDER},
        edge, camera_fx(cam), camera_fy(cam), camera_cx(cam), camera_cy(cam),
        np.array(cam.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64),
    )
    if not getattr(pose, "ok", False):
        solver.reset()
        return False, None, None
    return True, [float(x) for x in pose.translation_mm], getattr(pose, "reproj_err", None)


def camera_fx(cam: dict) -> float:
    return float(cam["fx"])


def camera_fy(cam: dict) -> float:
    return float(cam["fy"])


def camera_cx(cam: dict) -> float:
    return float(cam["cx"])


def camera_cy(cam: dict) -> float:
    return float(cam["cy"])


def _record(method: str, ann, fi: int, fr, pred: Dict[str, tuple],
            solver: PoseSolver, cam: dict, edge: float,
            latency_ms: float, extra: Optional[dict] = None) -> FrameRecord:
    gt = _gt(fr)
    pose_ok, pred_t, _ = _solve_pose(pred, fr, cam, edge, solver)
    return FrameRecord(
        method=method,
        seq_id=str(ann["info"]["seq_id"]),
        factor=str(ann["info"].get("factor", "?")),
        frame_index=fi,
        gt=gt,
        pred=[list(pred[c]) if c in pred else None for c in COLOR_ORDER],
        pose_ok=bool(pose_ok),
        pred_t_mm=pred_t,
        gt_t_mm=[float(x) for x in fr["pose_gt"]["translation_mm"]],
        latency_ms=float(latency_ms),
        fps=float(ann["info"].get("fps", 30) or 30),
        extra=dict(extra or {}),
    )


def _read_video(path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    # optional realistic-capture degradations (see degrade.py); no-op by default
    return degrade.apply_to_frames(frames)


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #
def evaluate_b0(pairs, limit_seq: Optional[int] = None) -> List[FrameRecord]:
    records: List[FrameRecord] = []
    for si, (video, ann_path) in enumerate(pairs[:limit_seq] if limit_seq else pairs):
        ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        cam, edge = _camera(ann), float(ann["info"]["edge_length_mm"])
        det, solver = B0Detector(), PoseSolver()
        frames = _read_video(video)
        n = min(len(frames), len(ann["frames"]))
        for fi in range(n):
            t0 = time.perf_counter()
            d = det.detect(frames[fi])
            pred = dict(d.points) if getattr(d, "ok", False) else {}
            detect_ms = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            rec = _record("b0", ann, fi, ann["frames"][fi], pred, solver, cam, edge,
                          0.0, {"detect_ms": detect_ms,
                                "combo_score": float(getattr(d, "combo_score", 0.0))})
            rec.latency_ms = detect_ms + (time.perf_counter() - t1) * 1000.0
            records.append(rec)
        print(f"[b0] {si + 1}/{len(pairs)} {ann['info']['seq_id']} frames={n}",
              flush=True)
    return records


def evaluate_classic(pairs, limit_seq: Optional[int] = None,
                     detector_factory=None,
                     method: str = "classic2") -> List[FrameRecord]:
    """Colour-difference detector + temporal/geometric triple selection."""
    if detector_factory is None:
        detector_factory = ClassicV2Detector
    records: List[FrameRecord] = []
    for si, (video, ann_path) in enumerate(pairs[:limit_seq] if limit_seq else pairs):
        ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        cam, edge = _camera(ann), float(ann["info"]["edge_length_mm"])
        det, solver = detector_factory(), PoseSolver()
        frames = _read_video(video)
        n = min(len(frames), len(ann["frames"]))
        fx, fy = camera_fx(cam), camera_fy(cam)
        cx, cy = camera_cx(cam), camera_cy(cam)
        for fi in range(n):
            t0 = time.perf_counter()
            d = det.detect(frames[fi], fx, fy, cx, cy, solver)
            detect_ms = (time.perf_counter() - t0) * 1000.0
            pred = dict(d.points) if getattr(d, "ok", False) else {}
            t1 = time.perf_counter()
            rec = _record(method, ann, fi, ann["frames"][fi], pred, solver,
                          cam, edge, 0.0,
                          {"detect_ms": detect_ms,
                           "combo_score": float(getattr(d, "combo_score", 0.0)),
                           "reason": str(getattr(d, "reason", ""))})
            rec.latency_ms = detect_ms + (time.perf_counter() - t1) * 1000.0
            records.append(rec)
        print(f"[{method}] {si + 1}/{len(pairs)} {ann['info']['seq_id']} "
              f"frames={n}", flush=True)
    return records


def evaluate_classic2(pairs, limit_seq: Optional[int] = None) -> List[FrameRecord]:
    return evaluate_classic(pairs, limit_seq, ClassicV2Detector, "classic2")


class _FindGeom:
    def __init__(self, find_ckpt: Path, geom_ckpt: Optional[Path], device: str,
                 temporal_w: float, top_k: Optional[int], upsample: Optional[int],
                 use_offset: bool = False):
        self.device = torch.device(device)
        # False / "top1" (offset head only) / "geom" (geom triple + offset refine)
        self.use_offset = use_offset
        self.model = load_find_checkpoint(find_ckpt, self.device)
        self.model.eval()
        self.geom = None
        self.temporal_w = float(temporal_w)
        self.top_k = int(top_k if top_k is not None else 5)
        self.upsample = int(upsample if upsample is not None else 1)
        if geom_ckpt is not None and Path(geom_ckpt).is_file():
            gk = torch.load(geom_ckpt, map_location=self.device, weights_only=False)
            self.geom = GeomTripleNet().to(self.device)
            self.geom.load_state_dict(gk["model"])
            self.geom.eval()
            if top_k is None and "top_k" in gk:
                self.top_k = int(gk["top_k"])
            if upsample is None and "upsample" in gk:
                self.upsample = int(gk["upsample"])

    def reset(self) -> None:
        self.prev_xy = None

    def predict(self, x: torch.Tensor):
        """Return normalized coords (3,2) + diagnostics for the last frame."""
        with torch.no_grad():
            _coords, vis, heat, _coords0 = self.model(x, return_heat=True)
            hm = heat[:, -1]
            if self.use_offset == "geom":
                off = getattr(self.model, "last_off", None)
                if off is None:
                    raise RuntimeError("checkpoint has no offset head")
                c_t = decode_find_geom(
                    hm, geom=self.geom, force_geom=True,
                    top_k=self.top_k, upsample=self.upsample,
                    temporal_w=0.0, prev_xy=None,
                )
                c_t = refine_with_offset(c_t, off[:, -1])
                c = c_t[0].cpu().numpy()
                v = torch.sigmoid(vis[0, -1]).cpu().numpy()
                return c, v, float(hm.max().item())
            if self.use_offset:
                off = getattr(self.model, "last_off", None)
                if off is None:
                    raise RuntimeError("checkpoint has no offset head")
                c_t = decode_offset(hm, off[:, -1], top_k=1, nms=5)
                if self.temporal_w > 0:
                    self.prev_xy = c_t.detach()
                c = c_t[0].cpu().numpy()
                v = torch.sigmoid(vis[0, -1]).cpu().numpy()
                return c, v, float(hm.max().item())
            prev = self.prev_xy if self.temporal_w > 0 else None
            c_t = decode_find_geom(
                hm, geom=self.geom, force_geom=True,
                top_k=self.top_k, upsample=self.upsample,
                temporal_w=self.temporal_w, prev_xy=prev,
            )
        if self.temporal_w > 0:
            self.prev_xy = c_t.detach()
        c = c_t[0].cpu().numpy()
        v = torch.sigmoid(vis[0, -1]).cpu().numpy()
        return c, v, float(hm.max().item())


def _evaluate_app_family(pairs, limit_seq, fov_deg: float, grid_size: int,
                         method: str, solver) -> List[FrameRecord]:
    """Shared body for the phone-app-style rows (B1 detector + some PnP)."""
    from research.pyc_import import install
    install()
    from research.baseline_b1.detector import B1Detector

    records: List[FrameRecord] = []
    for si, (video, ann_path) in enumerate(pairs[:limit_seq] if limit_seq else pairs):
        ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        edge = float(ann["info"]["edge_length_mm"])
        det = B1Detector(grid_size=grid_size)
        if hasattr(solver, "reset"):
            solver.reset()
        frames = _read_video(video)
        n = min(len(frames), len(ann["frames"]))
        for fi in range(n):
            fr = ann["frames"][fi]
            t0 = time.perf_counter()
            d = det.detect(frames[fi])
            detect_ms = (time.perf_counter() - t0) * 1000.0
            pred = dict(d.points) if getattr(d, "ok", False) else {}
            t1 = time.perf_counter()
            pose_ok, pred_t = False, None
            if len(pred) >= 3:
                res = solver.solve(
                    {c: (float(pred[c][0]), float(pred[c][1])) for c in COLOR_ORDER},
                    edge, int(ann["info"]["width"]), int(ann["info"]["height"]),
                )
                if getattr(res, "ok", False):
                    pose_ok = True
                    pred_t = [float(x) for x in res.translation_mm]
            rec = FrameRecord(
                method=method,
                seq_id=str(ann["info"]["seq_id"]),
                factor=str(ann["info"].get("factor", "?")),
                frame_index=fi,
                gt=_gt(fr),
                pred=[list(pred[c]) if c in pred else None for c in COLOR_ORDER],
                pose_ok=bool(pose_ok),
                pred_t_mm=pred_t,
                gt_t_mm=[float(x) for x in fr["pose_gt"]["translation_mm"]],
                latency_ms=detect_ms + (time.perf_counter() - t1) * 1000.0,
                fps=float(ann["info"].get("fps", 30) or 30),
                extra={"detect_ms": detect_ms,
                       "combo_score": float(getattr(d, "combo_score", 0.0)),
                       "fov_deg": fov_deg, "grid_size": grid_size},
            )
            records.append(rec)
        print(f"[{method}] {si + 1}/{len(pairs)} {ann['info']['seq_id']} frames={n}",
              flush=True)
    return records


def evaluate_app_fixed(pairs, limit_seq: Optional[int] = None, fov_deg: float = 60.0,
                       grid_size: int = 32) -> List[FrameRecord]:
    """App detector (B1 port) + the *fixed* three-point depth solver.

    Same detector as :func:`evaluate_b1`, corrected PnP, so the difference
    between the two rows isolates how much of the phone's error comes from the
    solver itself (see :mod:`research.tridot_temporal.app_pnp`).
    """
    return _evaluate_app_family(pairs, limit_seq, fov_deg, grid_size,
                                method="app_fixed",
                                solver=AppFixedSolver(fov_deg=fov_deg))


def evaluate_b1(pairs, limit_seq: Optional[int] = None, fov_deg: float = 60.0,
                grid_size: int = 32) -> List[FrameRecord]:
    """Evaluate the B1 port of the Android app (soft HSV grid + iterative PnP).

    ``grid_size`` defaults to 32 to match the app's 32x24 probability matrix;
    the port's own default (256) costs ~1.3 s/frame on CPU and is not what the
    phone runs.  NOTE: the port's PnP reproduces the app's own depth solver,
    which this repository now knows to be numerically divergent -- see
    ``app_pnp.py`` / ``research/results/app_pnp_audit``.  Prefer
    :func:`evaluate_app_fixed` when asking "how good could the app be".
    """
    from research.baseline_b1.pnp import B1PoseSolver

    return _evaluate_app_family(pairs, limit_seq, fov_deg, grid_size,
                                method="b1",
                                solver=B1PoseSolver(fov_deg=fov_deg,
                                                    use_gt_intrinsics=False))


def _snap_to_color_centroid(bgr, pred: dict, radius: int) -> dict:
    """Pull each predicted dot onto the local colour-centroid at full resolution.

    The learned head localises on a 448x256 grid, i.e. ~2.9 px of original
    resolution per input pixel, and its output is further quantised to a 112x64
    heat map.  The *classic* detector's strength is sub-pixel localisation of a
    bright saturated dot.  This refinement combines the two: use the learned
    head to decide *where* to look (recall), then re-localise with the colour
    image (precision).  ``radius`` is in original pixels.
    """
    if radius <= 0:
        return pred
    img = bgr.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    weights = {
        "red": np.clip(r - np.maximum(g, b), 0.0, None),
        "green": np.clip(g - np.maximum(r, b), 0.0, None),
        "blue": np.clip(b - np.maximum(r, g), 0.0, None),
    }
    h, w = bgr.shape[:2]
    out = {}
    for name, (u, v) in pred.items():
        x0, x1 = int(max(0, round(u) - radius)), int(min(w, round(u) + radius + 1))
        y0, y1 = int(max(0, round(v) - radius)), int(min(h, round(v) + radius + 1))
        wgt = weights[name][y0:y1, x0:x1]
        tot = float(wgt.sum())
        if x1 <= x0 or y1 <= y0 or tot <= 1e-6:
            out[name] = (float(u), float(v))
            continue
        cols = np.arange(x0, x1, dtype=np.float32)
        rows = np.arange(y0, y1, dtype=np.float32)
        out[name] = (float((wgt.sum(axis=0) * cols).sum() / tot),
                     float((wgt.sum(axis=1) * rows).sum() / tot))
    return out


_EVIDENCE_THR = 12.0


def _color_weight_maps(bgr: np.ndarray) -> dict:
    """Per-pixel colour evidence (same definition as the app `snap` weighting)."""
    f = bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    return {
        "red": np.clip(r - np.maximum(g, b), 0.0, None),
        "green": np.clip(g - np.maximum(r, b), 0.0, None),
        "blue": np.clip(b - np.maximum(r, g), 0.0, None),
    }


def _evidence_at(wmap: np.ndarray, xy, radius: int = 5) -> float:
    """Mean colour evidence in a (2r+1)^2 window around `xy`."""
    u, v = int(round(float(xy[0]))), int(round(float(xy[1])))
    h, w = wmap.shape[:2]
    y0, y1 = max(0, v - radius), min(h, v + radius + 1)
    x0, x1 = max(0, u - radius), min(w, u + radius + 1)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    return float(wmap[y0:y1, x0:x1].mean())


def evaluate_find_geom(pairs, find_ckpt: Path, geom_ckpt: Optional[Path],
                       device: str, temporal_w: float,
                       top_k: Optional[int], upsample: Optional[int],
                       limit_seq: Optional[int] = None,
                       snap_radius: int = 0,
                       use_offset: bool = False,
                       method_label: str = "find_geom",
                       img_size: Optional[Tuple[int, int]] = None,
                       complete: bool = False,
                       complete_mode: str = "vis",
                       evidence_thr: float = _EVIDENCE_THR) -> List[FrameRecord]:
    method = _FindGeom(find_ckpt, geom_ckpt, device, temporal_w, top_k, upsample,
                       use_offset=use_offset)
    ih, iw = img_size or (method.model.cfg.img_h, method.model.cfg.img_w)
    t_len = method.model.cfg.seq_len
    records: List[FrameRecord] = []
    for si, (video, ann_path) in enumerate(pairs[:limit_seq] if limit_seq else pairs):
        ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        cam, edge = _camera(ann), float(ann["info"]["edge_length_mm"])
        w, h = int(ann["info"]["width"]), int(ann["info"]["height"])
        solver = PoseSolver()
        completer = TemporalCompleter()
        method.reset()
        frames = _read_video(video)
        n = min(len(frames), len(ann["frames"]))
        xs = [build_find_input(fr, ih, iw) for fr in frames[:n]]
        for fi in range(n):
            start = max(0, fi - t_len + 1)
            clip = xs[start: fi + 1]
            if len(clip) < t_len:
                clip = [clip[0]] * (t_len - len(clip)) + clip
            x = torch.from_numpy(np.stack(clip)[None]).to(method.device)
            t0 = time.perf_counter()
            c, v, heat_peak = method.predict(x)
            detect_ms = (time.perf_counter() - t0) * 1000.0
            pred = {name: (float(c[i, 0] * w), float(c[i, 1] * h))
                    for i, name in enumerate(COLOR_ORDER)}
            if snap_radius:
                pred = _snap_to_color_centroid(frames[fi], pred, snap_radius)
            vis_list = [float(x) for x in np.asarray(v).reshape(-1)]
            completed: List[str] = []
            if complete:
                missing = None
                if complete_mode == "evidence":
                    maps = _color_weight_maps(frames[fi])
                    missing = [i for i, name in enumerate(COLOR_ORDER)
                               if _evidence_at(maps[name], pred[name]) < evidence_thr]
                out_c = completer.complete(pred, vis_list, cam, edge, w, h, fi,
                                           missing=missing)
                pred = out_c.points
                completed = out_c.completed
            t1 = time.perf_counter()
            label = f"{method_label}(w{method.temporal_w:g}"
            if snap_radius:
                label += f"+snap{snap_radius}"
            if complete:
                label += "+pred"
            label += ")"
            rec = _record(label, ann, fi,
                          ann["frames"][fi], pred, solver, cam, edge, 0.0,
                          {"detect_ms": detect_ms, "heat_peak": heat_peak,
                           "vis_min": float(v.min()),
                           "vis": [round(x, 3) for x in vis_list],
                           "completed": completed,
                           "top_k": method.top_k, "upsample": method.upsample,
                           "snap_radius": snap_radius})
            rec.latency_ms = detect_ms + (time.perf_counter() - t1) * 1000.0
            records.append(rec)
        print(f"[{method_label}] {si + 1}/{len(pairs)} {ann['info']['seq_id']} frames={n}",
              flush=True)
    return records


def _point_recall_10(records: List[FrameRecord]) -> Optional[float]:
    """Per-point recall@10 on a record subset (visible GT points only)."""
    num = 0
    den = 0
    for r in records:
        for i, name in enumerate(COLOR_ORDER):
            gt = r.gt[i] if len(r.gt) > i else None
            if gt is None or gt[2] < 0.5:
                continue
            den += 1
            if i < len(r.pred) and np.hypot(r.pred[i][0] - gt[0],
                                            r.pred[i][1] - gt[1]) <= 10.0:
                num += 1
    return num / den if den else None


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _provenance(args, method_cfg: dict, pairs) -> dict:
    srv = ROOT / "datasets" / "TriDotPose" / "meta" / "splits.json"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "dataset_root": str(Path(args.dataset).resolve()),
        "subset": args.subset,
        "n_sequences": len(pairs),
        "splits_sha256": _sha256(srv),
        "find_ckpt": str(Path(args.find_ckpt).resolve()) if args.find_ckpt else None,
        "find_ckpt_sha256": _sha256(Path(args.find_ckpt)) if args.find_ckpt else None,
        "geom_ckpt": str(Path(args.geom_ckpt).resolve()) if args.geom_ckpt else None,
        "geom_ckpt_sha256": _sha256(Path(args.geom_ckpt)) if args.geom_ckpt else None,
        "device": args.device,
        "seed": args.seed,
        "method_cfg": method_cfg,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--subset", default="test",
                    help="splits.json key, e.g. test|hard_test (comma-separated ok)")
    ap.add_argument("--dot-cls", default="",
                    help="dot-vs-clutter classifier ckpt for the classic3_dotcls method")
    ap.add_argument("--methods", default="b0",
                    help="comma list: b0,b1,app_fixed,classic2,classic3,find_geom,find_offset,find_offset_geom")
    ap.add_argument("--find-ckpt", type=Path, default=None)
    ap.add_argument("--geom-ckpt", type=Path, default=None)
    ap.add_argument("--temporal-w", type=float, default=0.0,
                    help="temporal heat prior strength for find_geom (0 = off)")
    ap.add_argument("--b1-fov", type=float, default=60.0,
                    help="B1 focal guess in degrees (app default: 60)")
    ap.add_argument("--b1-grid", type=int, default=32,
                    help="B1 soft-HSV grid size per side (app matrix is 32x24)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="override checkpoint top_k")
    ap.add_argument("--upsample", type=int, default=None,
                    help="override checkpoint upsample")
    ap.add_argument("--find-img-size", default=None,
                    help="override network input size as HxW, e.g. 384x640")
    ap.add_argument("--complete-mode", choices=("vis", "evidence"), default="vis",
                    help="how to decide which corners were lost: vis head or "
                         "per-pixel colour evidence (default: vis)")
    ap.add_argument("--evidence-thr", type=float, default=_EVIDENCE_THR,
                    help="colour-evidence threshold for --complete-mode evidence")
    ap.add_argument("--complete", action="store_true",
                    help="predict corners the visibility head marks as missing "
                         "(temporal completion; only for find_geom / "
                         "find_offset_geom)")
    ap.add_argument("--snap-radius", type=int, default=0,
                    help="refine learned dot positions to the local colour "
                         "centroid at full resolution (0 = off)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--factors", default=None,
                    help="comma list of annotation factors to keep, e.g. "
                         "occlude_hand,hardmix (filter applied after subset)")
    ap.add_argument("--limit-seq", type=int, default=None,
                    help="only the first N sequences (smoke test)")
    ap.add_argument("--degrade", default="",
                    help="apply a realistic capture degradation to every frame: "
                         + ", ".join(sorted(degrade.PRESETS)))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "results" / "honest")
    args = ap.parse_args(argv)
    if args.degrade:
        degrade.set_active(degrade.preset(args.degrade))

    img_size = None
    if args.find_img_size:
        hh, ww = args.find_img_size.lower().split("x")
        img_size = (int(hh), int(ww))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(False)

    pairs = _discover_pairs(args.dataset, "synthetic", subset=args.subset)
    if args.factors:
        wanted = {f.strip() for f in args.factors.split(",") if f.strip()}
        pairs = [p for p in pairs
                 if json.loads(Path(p[1]).read_text(encoding="utf-8"))
                 ["info"].get("factor") in wanted]
    if args.limit_seq:
        pairs = pairs[: args.limit_seq]
    if not pairs:
        raise SystemExit(f"no sequences for subset={args.subset}")
    print(f"[eval_v2] subset={args.subset} sequences={len(pairs)} "
          f"methods={args.methods}", flush=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = {}
    method_cfg = {}

    for method in methods:
        if method == "b0":
            cfg = {"top_k": 5, "roi": True}
            records = evaluate_b0(pairs, args.limit_seq)
        elif method in ("classic2", "classic3", "classic3_dotcls"):
            cfg = {"top_k": 6, "temporal": True, "detector": method}
            if method in ("classic3", "classic3_dotcls"):
                cfg.update({"contrast": True, "scale_adaptive": True,
                            "roi_upsample": True})
                if method == "classic3_dotcls":
                    cls_kw = dict(dot_cls_path=args.dot_cls or None,
                                  dot_cls_scale=True)
                    cfg["dot_cls"] = args.dot_cls
                    factory = lambda: ClassicV3Detector(**cls_kw)
                    records = evaluate_classic(pairs, args.limit_seq, factory, method)
                else:
                    records = evaluate_classic(pairs, args.limit_seq,
                                               ClassicV3Detector, "classic3")
            else:
                records = evaluate_classic2(pairs, args.limit_seq)
        elif method == "b1":
            cfg = {"fov_deg": args.b1_fov, "grid_size": args.b1_grid,
                   "note": "app-side pipeline (B1Detector + B1PoseSolver)"}
            records = evaluate_b1(pairs, args.limit_seq, args.b1_fov, args.b1_grid)
        elif method == "app_fixed":
            cfg = {"fov_deg": args.b1_fov, "grid_size": args.b1_grid,
                   "note": "app detector (B1) + fixed depth solver"}
            records = evaluate_app_fixed(pairs, args.limit_seq, args.b1_fov,
                                         args.b1_grid)
        elif method == "find_geom":
            if args.find_ckpt is None:
                raise SystemExit("--find-ckpt is required for method=find_geom")
            cfg = {"find_ckpt": str(args.find_ckpt),
                   "geom_ckpt": str(args.geom_ckpt) if args.geom_ckpt else None,
                   "temporal_w": args.temporal_w,
                   "top_k": args.top_k, "upsample": args.upsample,
                   "snap_radius": args.snap_radius,
                   "complete": args.complete,
                   "complete_mode": args.complete_mode,
                   "evidence_thr": args.evidence_thr,
                   "img_size": list(img_size) if img_size else None}
            records = evaluate_find_geom(
                pairs, args.find_ckpt, args.geom_ckpt, args.device,
                args.temporal_w, args.top_k, args.upsample, args.limit_seq,
                args.snap_radius, img_size=img_size, complete=args.complete,
                complete_mode=args.complete_mode, evidence_thr=args.evidence_thr)
        elif method == "find_offset_geom":
            if args.find_ckpt is None or args.geom_ckpt is None:
                raise SystemExit("--find-ckpt and --geom-ckpt are required for "
                                 "method=find_offset_geom")
            cfg = {"find_ckpt": str(args.find_ckpt),
                   "geom_ckpt": str(args.geom_ckpt),
                   "decoder": "geom triple + offset refine",
                   "snap_radius": args.snap_radius,
                   "complete": args.complete,
                   "complete_mode": args.complete_mode,
                   "evidence_thr": args.evidence_thr,
                   "img_size": list(img_size) if img_size else None}
            records = evaluate_find_geom(
                pairs, args.find_ckpt, args.geom_ckpt, args.device, 0.0,
                args.top_k, args.upsample, args.limit_seq, args.snap_radius,
                use_offset="geom", method_label="find_offset_geom",
                img_size=img_size, complete=args.complete,
                complete_mode=args.complete_mode, evidence_thr=args.evidence_thr)
        elif method == "find_offset":
            if args.find_ckpt is None:
                raise SystemExit("--find-ckpt is required for method=find_offset")
            cfg = {"find_ckpt": str(args.find_ckpt), "decoder": "offset_head",
                   "snap_radius": args.snap_radius,
                   "img_size": list(img_size) if img_size else None}
            records = evaluate_find_geom(
                pairs, args.find_ckpt, None, args.device, 0.0,
                None, 1, args.limit_seq, args.snap_radius,
                use_offset=True, method_label="find_offset", img_size=img_size)
        else:
            raise SystemExit(f"unknown method {method}")
        method_cfg[method] = cfg
        rec_dir = out / method.replace("(", "_").replace(")", "").replace("=", "")
        rec_dir.mkdir(parents=True, exist_ok=True)
        (rec_dir / "per_frame.json").write_text(
            json.dumps([dataclasses.asdict(r) for r in records], indent=1),
            encoding="utf-8")
        sm = summarize(records, n_boot=args.n_boot, seed=args.seed)
        comp = [r for r in records if r.extra.get("completed")]
        if comp:
            sm["completion"] = {
                "n_frames": len(comp),
                "n_points": sum(len(r.extra["completed"]) for r in comp),
                "by_color": {c: sum(1 for r in comp
                                    if c in r.extra["completed"])
                             for c in COLOR_ORDER},
                "det10_all": det10_all(comp),
                "point_recall": _point_recall_10(comp),
                "t_err_clean": t_err_stats(comp, clean=True)["median_mm"],
            }
        else:
            sm["completion"] = {"n_frames": 0}
        (rec_dir / "summary.json").write_text(
            json.dumps(sm, indent=2), encoding="utf-8")
        results[method] = sm
        print(f"[eval_v2] {method}: det10_all={sm['det']['det10_all']} "
              f"legacy={sm['det']['det10_legacy']} "
              f"t_med={sm['pose']['t_err_clean']['median_mm']}", flush=True)

    prov = _provenance(args, method_cfg, pairs)
    prov["elapsed_s"] = round(time.time() - started, 1)
    (out / "summary.json").write_text(
        json.dumps({"provenance": prov, "results": results}, indent=2),
        encoding="utf-8")

    lines = [f"# Honest eval — subset `{args.subset}`", "",
             "Definitions are in `research/tridot_temporal/metrics.py`: "
             "`det@10 (all frames)` counts occluded frames as failures; "
             "`t_err` is always reported next to `pose coverage`.", ""]
    for method, sm in results.items():
        lines.append(format_report(sm))
        lines.append("")
    lines += ["## Provenance", "", "```json",
              json.dumps(prov, indent=2), "```", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval_v2] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
