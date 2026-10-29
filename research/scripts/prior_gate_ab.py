"""Candidate-level temporal gating A/B (round 7).

The threshold study (REPORT §4.3) showed the bottleneck is candidate quality,
not the threshold.  This script compares the union front-end with and without
``prior_gate`` (prune candidates farther than ``prior_gate_frac * edge_px``
from the pose-predicted pixel, with a fallback to the unpruned list).

Usage::

    PYTHONPATH=. python -m research.scripts.prior_gate_ab --limit-seq 6
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional

from research.scripts.far_bench import summarize
from research.tridot_temporal import degrade
from research.tridot_temporal import eval_v2 as ev
from research.tridot_temporal.classic_v3 import ClassicV3Detector


def _configs() -> Dict[str, object]:
    pf = degrade.preset("phone_far")
    return {
        "clean": None,
        "phone_far": pf,
        "highres_chroma": dataclasses.replace(pf, downscale=1.0, chroma="444"),
        "all_on": dataclasses.replace(pf, downscale=1.0, chroma="444",
                                      denoise=0.0, jpeg_quality=100),
    }


def _variants():
    return {
        "v3_union": lambda: ClassicV3Detector(frontend="union",
                                              contrast_floor=12.0),
        "v3_union_gate": lambda: ClassicV3Detector(frontend="union",
                                                   contrast_floor=12.0,
                                                   prior_gate=True),
        "v3_gate_main": lambda: ClassicV3Detector(frontend="union",
                                                  contrast_floor=12.0,
                                                  prior_gate=True,
                                                  prior_gate_quick=False),
        "v3_gate_wide": lambda: ClassicV3Detector(frontend="union",
                                                  contrast_floor=12.0,
                                                  prior_gate=True,
                                                  prior_gate_frac=0.8),
        "v3_gate_dual": lambda: ClassicV3Detector(frontend="union",
                                                  contrast_floor=12.0,
                                                  prior_gate=True,
                                                  prior_gate_mode="dual"),
    }


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="hard_test")
    ap.add_argument("--limit-seq", type=int, default=6)
    ap.add_argument("--dataset", type=Path, default=ev.DEFAULT_DATASET)
    ap.add_argument("--configs", default="")
    ap.add_argument("--out", type=Path,
                    default=Path("research/results/prior_gate_ab/summary.json"))
    args = ap.parse_args(argv)

    pairs = ev._discover_pairs(args.dataset, "synthetic", subset=args.subset)
    if args.limit_seq:
        pairs = pairs[:args.limit_seq]

    cfg_all = _configs()
    if args.configs:
        keep = {k.strip() for k in args.configs.split(",") if k.strip()}
        cfg_all = {k: v for k, v in cfg_all.items() if k in keep}
    variants = _variants()

    cache: Dict[str, list] = {}
    orig_read = ev._read_video

    def cached_read(path, *a, **kw):
        key = str(path)
        if key not in cache:
            cache[key] = orig_read(path, *a, **kw)
        return cache[key]

    ev._read_video = cached_read
    report = {"subset": args.subset, "n_sequences": len(pairs), "configs": {}}
    try:
        for name, cfg in cfg_all.items():
            cache.clear()
            degrade.set_active(cfg)
            report["configs"][name] = {"methods": {}}
            for vname, factory in variants.items():
                print(f"[prior_gate_ab] {name} / {vname} ...", flush=True)
                records = ev.evaluate_classic(pairs, None, factory, vname)
                report["configs"][name]["methods"][vname] = summarize(records, vname)
            cache.clear()
    finally:
        ev._read_video = orig_read
        degrade.set_active(None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n== det@10 all (rows=config, cols=variant) ==")
    print(f"{'config':>18s}" + "".join(f"{m:>16s}" for m in variants))
    for name, blk in report["configs"].items():
        row = f"{name:>18s}"
        for m in variants:
            row += f"{blk['methods'][m]['det10_all']:>16.3f}"
        print(row)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
