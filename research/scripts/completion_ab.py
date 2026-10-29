"""Compare two eval_v2 runs (e.g. no completion vs evidence-triggered completion).

Both runs must use the same subset and the same method label. Statistics are
reported separately for frames where at least one GT corner is not fully
visible (vis < 2) -- this is where completion is allowed to act -- and for the
fully visible frames (it must not hurt them).

Usage:
    python -m research.scripts.completion_ab \
        --a /tmp/nat_none --b /tmp/nat_evidence --method find_offset_geom
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _load(out: Path, method: str) -> List[dict]:
    p = out / method / "per_frame.json"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _t_err(rec: dict) -> Optional[float]:
    if not rec.get("pose_ok"):
        return None
    pt, gt = rec.get("pred_t_mm"), rec.get("gt_t_mm")
    if pt is None or gt is None:
        return None
    return float(np.linalg.norm(np.asarray(pt) - np.asarray(gt)))


def _point_err(rec: dict, color: int) -> Optional[float]:
    gt = rec["gt"][color]
    if gt[2] < 2:
        return None
    pred = rec["pred"][color]
    return float(np.hypot(pred[0] - gt[0], pred[1] - gt[1]))


def _stats(recs: List[dict], label: str) -> dict:
    t = [v for v in (_t_err(r) for r in recs) if v is not None]
    pt: List[float] = []
    for r in recs:
        errs = [_point_err(r, c) for c in range(3)]
        errs = [e for e in errs if e is not None]
        if errs:
            pt.append(float(np.mean(errs)))
    comp = [r for r in recs if r.get("extra", {}).get("completed")]
    return {
        "label": label,
        "frames": len(recs),
        "pose_ok": sum(1 for r in recs if r.get("pose_ok")),
        "pose_cov": sum(1 for r in recs if r.get("pose_ok")) / len(recs) if recs else 0.0,
        "t_err_n": len(t),
        "t_err_median": float(np.median(t)) if t else None,
        "t_err_p90": float(np.percentile(t, 90)) if t else None,
        "t_err_within50": sum(1 for v in t if v <= 50.0) / len(t) if t else None,
        "pix_err_mean_med": float(np.median(pt)) if pt else None,
        "n_completed_frames": len(comp),
        "n_completed_points": sum(len(r["extra"]["completed"]) for r in comp),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--method", default="find_offset_geom")
    ap.add_argument("--label-a", default="baseline")
    ap.add_argument("--label-b", default="completion")
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args(argv)

    a, b = _load(args.a, args.method), _load(args.b, args.method)
    key = lambda r: (r["seq_id"], r["frame_index"])  # noqa: E731
    bmap = {key(r): r for r in b}
    pairs = [(r, bmap[key(r)]) for r in a if key(r) in bmap]

    def n_missing(rec: dict) -> int:
        return sum(1 for g in rec["gt"] if g[2] < 2)

    occl = [(x, y) for x, y in pairs if n_missing(x) >= 1]
    full = [(x, y) for x, y in pairs if n_missing(x) == 0]
    zeros = [(x, y) for x, y in pairs if any(g[2] == 0 for g in x["gt"])]
    one = [(x, y) for x, y in pairs if n_missing(x) == 1]

    rows = []
    for name, subset in (("all", pairs), ("occluded(>=1 missing)", occl),
                         ("1 missing", one), ("visible(0 missing)", full),
                         ("has vis==0", zeros)):
        rows.append(_stats([x for x, _ in subset], f"{args.label_a} / {name}"))
        rows.append(_stats([y for _, y in subset], f"{args.label_b} / {name}"))
    # frames where completion actually fired (intersect by stream b's extra)
    fired = [(x, y) for x, y in pairs
             if y.get("extra", {}).get("completed")]
    if fired:
        rows.append(_stats([x for x, _ in fired], f"{args.label_a} / fired"))
        rows.append(_stats([y for _, y in fired], f"{args.label_b} / fired"))
    lines = ["| config | frames | pose cov | t_err n | t_err med (mm) | t_err p90 (mm) | <=50mm | pix err med (px) | completed pts |",
             "|---|---|---|---|---|---|---|---|---|"]
    def fmt(v, nd=2):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    for r in rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["label"], r["frames"], fmt(r["pose_cov"], 3), r["t_err_n"],
            fmt(r["t_err_median"]), fmt(r["t_err_p90"]), fmt(r["t_err_within50"]),
            fmt(r["pix_err_mean_med"]), r["n_completed_points"]))
    md = "\n".join(lines)
    print(md)
    if args.md:
        args.md.write_text(md + "\n", encoding="utf-8")
        print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
