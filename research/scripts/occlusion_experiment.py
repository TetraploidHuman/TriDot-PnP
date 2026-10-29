"""Controlled occlusion experiment for the temporal completer.

Question
--------
On the synthetic splits, ~30% of frames do not have all three dots visible, and
the honest ``det@10`` metric counts every one of them as a failure because a
3-point PnP cannot run.  The app needs *something* to do when one dot is
covered by a hand for a few frames.

Protocol
--------
Real occlusion in ``hard_test`` cannot be used to isolate the mechanism: the
``occlude_hand`` sequences never show all three dots, so a tracker has no frame
to initialise from.  Instead this experiment takes *easy* sequences (dots
always visible) and **masks one dot for a run of frames** at its GT position --
a synthetic, reproducible occlusion with exact ground truth.  Two rows are
compared, both using the same learned detections and the same PnP:

* ``no_completion`` — what the pipeline does today: the masked dot is still
  "decoded" from the heat map, so its pixel is wrong and the pose follows it;
* ``completion`` — the same detections, but a corner whose colour window has no
  evidence is replaced by a constant-velocity projection of the tracked pose.

The trigger is evidence-based (colour weight inside the search window), not
ground-truth based: it is the same information the app has.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

from research.tridot_temporal.completion import TemporalCompleter  # noqa: E402
from research.tridot_temporal.eval_v2 import (  # noqa: E402
    _camera, _discover_pairs, _read_video, _FindGeom,
)
from research.tridot_temporal.find_preprocess import build_find_input  # noqa: E402
from research.baseline_b0.pnp import PoseSolver  # noqa: E402

COLORS = ("red", "green", "blue")


# --------------------------------------------------------------------------- #
# masking & evidence
# --------------------------------------------------------------------------- #
def mask_dot(bgr: np.ndarray, u: float, v: float, radius: int = 14) -> np.ndarray:
    """Paint the dot at ``(u, v)`` with the median of a surrounding ring."""
    out = bgr.copy()
    h, w = out.shape[:2]
    x0, x1 = int(max(0, round(u) - radius)), int(min(w, round(u) + radius + 1))
    y0, y1 = int(max(0, round(v) - radius)), int(min(h, round(v) + radius + 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return out
    ring = np.concatenate([
        out[max(0, y0 - 3):y0, x0:x1].reshape(-1, 3),
        out[y1:min(h, y1 + 3), x0:x1].reshape(-1, 3),
        out[y0:y1, max(0, x0 - 3):x0].reshape(-1, 3),
        out[y0:y1, x1:min(w, x1 + 3)].reshape(-1, 3),
    ])
    bg = np.median(ring.astype(np.float32), axis=0)
    out[y0:y1, x0:x1] = bg.astype(np.uint8)
    return out


def color_weights(bgr: np.ndarray) -> Dict[str, np.ndarray]:
    img = bgr.astype(np.float32)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return {
        "red": np.clip(r - np.maximum(g, b), 0.0, None),
        "green": np.clip(g - np.maximum(r, b), 0.0, None),
        "blue": np.clip(b - np.maximum(r, g), 0.0, None),
    }


def color_evidence(bgr: np.ndarray, points: Dict[str, Tuple[float, float]],
                   radius: int = 10) -> Dict[str, float]:
    """Mean colour weight inside each point's search window (0 = no evidence)."""
    weights = color_weights(bgr)
    h, w = bgr.shape[:2]
    out = {}
    for name, (u, v) in points.items():
        x0, x1 = int(max(0, round(u) - radius)), int(min(w, round(u) + radius + 1))
        y0, y1 = int(max(0, round(v) - radius)), int(min(h, round(v) + radius + 1))
        win = weights[name][y0:y1, x0:x1]
        out[name] = float(win.mean()) if win.size else 0.0
    return out


def snap(bgr: np.ndarray, points: Dict[str, Tuple[float, float]],
         radius: int = 14) -> Dict[str, Tuple[float, float]]:
    weights = color_weights(bgr)
    h, w = bgr.shape[:2]
    out = {}
    for name, (u, v) in points.items():
        x0, x1 = int(max(0, round(u) - radius)), int(min(w, round(u) + radius + 1))
        y0, y1 = int(max(0, round(v) - radius)), int(min(h, round(v) + radius + 1))
        wgt = weights[name][y0:y1, x0:x1]
        tot = float(wgt.sum())
        if tot <= 1e-6:
            out[name] = (float(u), float(v))
            continue
        cols = np.arange(x0, x1, dtype=np.float32)
        rows = np.arange(y0, y1, dtype=np.float32)
        out[name] = (float((wgt.sum(axis=0) * cols).sum() / tot),
                     float((wgt.sum(axis=1) * rows).sum() / tot))
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run_sequence(pairs, method, use_completion: bool, mask_color: str, max_gap: int,
                 mask_start: int, mask_end: int, evidence_thr: float,
                 snap_radius: int):
    completer = TemporalCompleter(max_gap=max_gap)
    rows = []
    for video, ann_path in pairs:
        ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        cam, edge = _camera(ann), float(ann["info"]["edge_length_mm"])
        w, h = int(ann["info"]["width"]), int(ann["info"]["height"])
        frames = _read_video(video)
        # The "camera" frames: during the window the chosen dot is painted away,
        # so the detector genuinely cannot see it (preprocessing included).
        seen = []
        for fi, fr in enumerate(frames):
            if mask_start <= fi < mask_end:
                idx = COLORS.index(mask_color)
                gx = float(ann["frames"][fi]["keypoints"][3 * idx])
                gy = float(ann["frames"][fi]["keypoints"][3 * idx + 1])
                fr = mask_dot(fr, gx, gy, 14)
            seen.append(fr)
        xs = [build_find_input(fr, method.model.cfg.img_h, method.model.cfg.img_w)
              for fr in seen]
        tlen = method.model.cfg.seq_len
        solver = PoseSolver()
        method.reset()
        for fi in range(len(frames)):
            start = max(0, fi - tlen + 1)
            clip = xs[start:fi + 1]
            if len(clip) < tlen:
                clip = [clip[0]] * (tlen - len(clip)) + clip
            x = torch.from_numpy(np.stack(clip)[None]).to(method.device)

            frame = seen[fi]
            gt = [(float(ann["frames"][fi]["keypoints"][3 * i]),
                   float(ann["frames"][fi]["keypoints"][3 * i + 1])) for i in range(3)]
            c, v, _peak = method.predict(x)
            pred = {name: (float(c[i][0]) * w, float(c[i][1]) * h)
                    for i, name in enumerate(COLORS)}
            pred = snap(frame, pred, snap_radius)

            missing: List[int] = []
            ev = color_evidence(frame, pred, radius=10)
            if use_completion:
                missing = [COLORS.index(n) for n in COLORS if ev[n] < evidence_thr]
                out_c = completer.complete(pred, v, cam, edge, w, h, fi,
                                           missing=missing)
                pred = out_c.points
                filled = list(out_c.completed)
                reason = out_c.reason
            else:
                # keep the tracker warm so both rows share the same state
                completer.complete(pred, v, cam, edge, w, h, fi, missing=[])
                reason = "off"
                filled = []

            pose_ok, t_err = False, None
            pose = solver.solve(
                {n: (float(pred[n][0]), float(pred[n][1])) for n in COLORS},
                edge, cam["fx"], cam["fy"], cam["cx"], cam["cy"],
                np.array(cam.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64))
            if getattr(pose, "ok", False):
                pose_ok = True
                t_err = float(np.linalg.norm(
                    np.asarray(pose.translation_mm, dtype=np.float64)
                    - np.asarray(ann["frames"][fi]["pose_gt"]["translation_mm"],
                                 dtype=np.float64)))
            else:
                solver.reset()
            errs = [float(np.hypot(pred[n][0] - gt[i][0], pred[n][1] - gt[i][1]))
                    for i, n in enumerate(COLORS)]
            rows.append({
                "seq_id": ann["info"]["seq_id"], "frame": fi,
                "masked": bool(mask_start <= fi < mask_end),
                "errs": errs, "det10": all(e <= 10.0 for e in errs),
                "pose_ok": pose_ok, "t_err_mm": t_err,
                "missing": missing, "filled": filled, "reason": reason,
                "evidence": {k: round(v, 2) for k, v in ev.items()},
            })
    return rows


def summarize_rows(rows, label: str) -> dict:
    masked = [r for r in rows if r["masked"]]
    clean = [r for r in rows if not r["masked"]]
    def stat(rs):
        mask_color = "blue"
        if not rs:
            return {}
        te = [r["t_err_mm"] for r in rs if r["t_err_mm"] is not None]
        return {
            "n": len(rs),
            "det10_all": sum(1 for r in rs if r["det10"]) / len(rs),
            "pose_coverage": sum(1 for r in rs if r["pose_ok"]) / len(rs),
            "t_err_median_mm": float(np.median(te)) if te else None,
            "t_err_p90_mm": float(np.percentile(te, 90)) if te else None,
            "pose_within_50mm": (sum(1 for v in te if v <= 50.0) / len(rs))
            if rs else None,
            "blue_err_median_px": float(np.median([r["errs"][2] for r in rs])),
            "n_completed": sum(len(r["missing"]) for r in rs),
            "evidence_blue_median": float(np.median([r["evidence"][mask_color] for r in rs]))
            if rs else None,
        }
    return {"label": label, "all": stat(rows), "masked": stat(masked),
            "unmasked": stat(clean)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "TriDotPose")
    ap.add_argument("--subset", default="test")
    ap.add_argument("--limit-seq", type=int, default=2)
    ap.add_argument("--find-ckpt", type=Path,
                    default=ROOT / "research" / "results" / "find_v3_25_offset" / "best.pt")
    ap.add_argument("--geom-ckpt", type=Path,
                    default=ROOT / "research" / "results" / "find_geom_v34" / "best.pt")
    ap.add_argument("--snap-radius", type=int, default=14)
    ap.add_argument("--mask-color", default="blue")
    ap.add_argument("--mask-start", type=int, default=40)
    ap.add_argument("--mask-end", type=int, default=48)
    ap.add_argument("--evidence-thr", type=float, default=12.0)
    ap.add_argument("--max-gap", type=int, default=12,
                    help="max frames the completer may bridge (state is only updated "
                         "by fully observed frames)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "research" / "results" / "completion_exp")
    args = ap.parse_args(argv)

    pairs = _discover_pairs(args.dataset, "synthetic", subset=args.subset)
    pairs = pairs[: args.limit_seq]
    method = _FindGeom(args.find_ckpt, args.geom_ckpt, "cpu", 0.0, 5, 1,
                       use_offset="geom")

    out = {"argv": [str(x) for x in argv] if argv else None,
           "mask": {"color": args.mask_color, "start": args.mask_start,
                    "end": args.mask_end},
           "evidence_thr": args.evidence_thr, "max_gap": args.max_gap,
           "rows": {}}
    for label, use_completion in (("no_completion", False), ("completion", True)):
        rows = run_sequence(pairs, method, use_completion, args.mask_color,
                            args.max_gap, args.mask_start, args.mask_end,
                            args.evidence_thr, args.snap_radius)
        out["rows"][label] = summarize_rows(rows, label)
        out["rows"][label]["per_seq"] = sorted({r["seq_id"] for r in rows})
        out.setdefault("per_frame", {})[label] = rows
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(out, indent=2),
                                           encoding="utf-8")

    lines = [f"# Occlusion completion experiment — subset `{args.subset}`", "",
             f"Mask `{args.mask_color}` frames {args.mask_start}..{args.mask_end} "
             f"at GT position; evidence threshold {args.evidence_thr}; "
             f"max gap {args.max_gap} frames.", "",
             "| config | window | det@10 | pose cov | t_err med (mm) | blue err med (px) | completed pts |",
             "|---|---|---|---|---|---|---|"]
    for label, body in out["rows"].items():
        for win in ("masked", "unmasked", "all"):
            s = body[win]
            if not s:
                continue
            lines.append(
                f"| {label} | {win} | {s['det10_all']:.3f} | {s['pose_coverage']:.3f} | "
                f"{s['t_err_median_mm']} | {s['blue_err_median_px']:.1f} | "
                f"{s['n_completed']} |")
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[occlusion_exp] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
