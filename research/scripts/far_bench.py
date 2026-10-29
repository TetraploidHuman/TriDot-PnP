"""Far-distance / small-dot benchmark: det@10 bucketed by apparent dot size.

The honest evaluator aggregates over a whole subset, which hides the size
dependence that matters for far targets: a 3 px dot and a 20 px dot are not the
same problem.  This script reruns any evaluator method and reports det@10_all
(plus pose translation error) per *apparent edge-length* bucket, optionally with
realistic capture degradations (see ``degrade.py``).

Usage::

    python -m research.scripts.far_bench --subset hard_test --limit-seq 8 \
        --degrade phone_far --methods b0,classic2,classic3 \
        --out research/results/far_bench/phone_far
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from research.tridot_temporal import degrade
from research.tridot_temporal.eval_v2 import (
    DEFAULT_DATASET, _camera, _discover_pairs, _gt, evaluate_b0, evaluate_classic,
)
from research.tridot_temporal.classic_v2 import ClassicV2Detector
from research.tridot_temporal.classic_v3 import ClassicV3Detector
from research.tridot_temporal.dataset import COLOR_ORDER

BUCKETS = [(0, 20), (20, 40), (40, 80), (80, 10_000)]


def _edge_px(gt: np.ndarray) -> float:
    e = [math.hypot(gt[i][0] - gt[(i + 1) % 3][0], gt[i][1] - gt[(i + 1) % 3][1])
         for i in range(3)]
    return float(np.mean(e))


def _bucket(edge: float) -> str:
    for lo, hi in BUCKETS:
        if lo <= edge < hi:
            return f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
    return "?"


def _frame_hit(rec) -> Tuple[bool, Optional[float], float]:
    """det@10_all semantics + pose error for one frame; also GT apparent edge."""
    edge = _edge_px(rec.gt)
    if len(rec.pred) < 3 or any(p is None for p in rec.pred):
        return False, None, edge
    ok = True
    for i, c in enumerate(COLOR_ORDER):
        gx, gy, vis = rec.gt[i]
        px, py = rec.pred[i]
        if vis >= 1:
            if math.hypot(px - gx, py - gy) > 10.0:
                ok = False
                break
        # vis == 0: a prediction is only tolerated if it is close to the GT spot
        elif math.hypot(px - gx, py - gy) > 10.0:
            ok = False
            break
    return ok, rec.pred_t_mm, edge


def summarize(records, label: str) -> Dict[str, object]:
    per_bucket: Dict[str, List[float]] = {f"{lo}-{hi}" if hi < 10_000 else f"{lo}+": []
                                          for lo, hi in BUCKETS}
    per_bucket_err: Dict[str, List[float]] = {k: [] for k in per_bucket}
    per_factor: Dict[str, List[bool]] = {}
    for rec in records:
        hit, _, edge = _frame_hit(rec)
        b = _bucket(edge)
        per_bucket[b].append(1.0 if hit else 0.0)
        if hit and rec.pred_t_mm is not None and rec.gt_t_mm is not None:
            d = np.linalg.norm(np.asarray(rec.pred_t_mm, dtype=float)
                               - np.asarray(rec.gt_t_mm, dtype=float))
            per_bucket_err[b].append(float(d))
        per_factor.setdefault(rec.factor, []).append(hit)
    out = {"label": label,
           "n_frames": len(records),
           "det10_all": float(np.mean([_frame_hit(r)[0] for r in records])) if records else 0.0}
    out["buckets"] = {
        b: {"n": len(v), "det10_all": (float(np.mean(v)) if v else None),
            "t_err_med": (float(np.median(per_bucket_err[b]))
                          if per_bucket_err[b] else None)}
        for b, v in per_bucket.items()}
    out["factors"] = {f: {"n": len(v), "det10_all": float(np.mean(v))}
                      for f, v in sorted(per_factor.items())}
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="hard_test")
    ap.add_argument("--limit-seq", type=int, default=None)
    ap.add_argument("--degrade", default="")
    ap.add_argument("--degrade-override", default="",
                    help="覆盖 preset 的字段，如 'downscale=1,chroma=444,denoise=0'")
    ap.add_argument("--methods", default="b0,classic2,classic3")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--dot-cls-scale", action="store_true",
                    help="scale-normalise classifier patches by the expected dot radius")
    ap.add_argument("--dot-cls", default="",
                    help="optional dot-vs-clutter patch classifier ckpt for v3 configs")
    ap.add_argument("--v3-configs", default="contrast",
                    help="comma list of v3 decoder configs to run, e.g. "
                         "'contrast' or 'contrast:12:union' (name:floor:frontend)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.degrade:
        cfg = degrade.preset(args.degrade)
        if args.degrade_override:
            import dataclasses
            overrides = {}
            for kv in args.degrade_override.split(","):
                key, value = kv.split("=", 1)
                cur = getattr(cfg, key.strip())
                overrides[key.strip()] = type(cur)(value)
            cfg = dataclasses.replace(cfg, **overrides)
            print(f"[far_bench] degrade override: {overrides}", flush=True)
        degrade.set_active(cfg)
    else:
        degrade.set_active(None)
    pairs = _discover_pairs(args.dataset, "synthetic", subset=args.subset)
    if args.limit_seq:
        pairs = pairs[:args.limit_seq]

    factories = {
        "b0": (lambda: evaluate_b0(pairs)),
        "classic2": (lambda: evaluate_classic(pairs, None, ClassicV2Detector, "classic2")),
    }
    for spec in [x.strip() for x in args.v3_configs.split(",") if x.strip()]:
        parts = spec.split(":")
        name, floor, frontend = parts[0], "12", "contrast"
        if len(parts) > 1 and parts[1]:
            floor = parts[1]
        if len(parts) > 2 and parts[2]:
            frontend = parts[2]
        kwargs = dict(contrast_floor=float(floor), frontend=frontend)
        if args.dot_cls:
            kwargs["dot_cls_path"] = args.dot_cls
            if args.dot_cls_scale:
                kwargs["dot_cls_scale"] = True
        label = f"classic3_{name}"
        factories[label] = (
            lambda kw=kwargs, lb=label: evaluate_classic(
                pairs, None, lambda _kw=kw: ClassicV3Detector(**_kw), lb))
    report = {"subset": args.subset, "degrade": args.degrade or None,
              "n_sequences": len(pairs), "methods": {}}
    for m in [x.strip() for x in args.methods.split(",") if x.strip()]:
        if m not in factories:
            raise SystemExit(f"unknown method {m}; have {sorted(factories)}")
        print(f"[far_bench] running {m} ...", flush=True)
        records = factories[m]()
        report["methods"][m] = summarize(records, m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n== {args.subset} degrade={args.degrade or 'clean'} "
          f"({len(pairs)} seqs) ==")
    header = f"{'bucket':>10s}" + "".join(f"{m:>12s}" for m in report["methods"])
    print(header)
    print("-" * len(header))
    for lo, hi in BUCKETS:
        b = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
        row = f"{b:>10s}"
        for m, sm in report["methods"].items():
            d = sm["buckets"][b]
            row += f"{(d['det10_all'] if d['det10_all'] is not None else float('nan')):>7.3f}"
            row += f" n={d['n']:<3d}"
        print(row)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
