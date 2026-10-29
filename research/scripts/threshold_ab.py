"""Does a noise-robust threshold make capture-side improvements pay off?

The capture-knob A/B (``research/results/capture_knobs``) showed the paradox:
removing sensor degradations (higher resolution, no JPEG, no ISP denoise) made
det@10 *worse*.  Hypothesis: the front-end threshold is coupled to global
content statistics (``global MAD`` and ``p99.5`` inflate on sharper frames).

This script re-runs the same knob configs with alternate threshold modes:

* ``percentile`` - ``max(floor, k*global_mad, rel*p99.5)`` (current default)
* ``tile``       - ``max(floor, k*p25(tile_mad))`` (quiet-tile noise floor)
* ``tile_rel``   - tile floor + frame bright tail

Usage::

    PYTHONPATH=. python -m research.scripts.threshold_ab --limit-seq 6
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
from research.tridot_temporal.classic_v2 import ClassicV2Detector
from research.tridot_temporal.classic_v3 import ClassicV3Detector


def _configs() -> Dict[str, object]:
    pf = degrade.preset("phone_far")
    return {
        "clean": None,
        "phone_far": pf,
        "highres": dataclasses.replace(pf, downscale=1.0),
        "all_on": dataclasses.replace(pf, downscale=1.0, chroma="444",
                                      denoise=0.0, jpeg_quality=100),
    }


def _modes():
    return {
        "v2_absolute": lambda: ClassicV2Detector(),
        "v3_percentile": lambda: ClassicV3Detector(threshold_mode="percentile"),
        "v3_tile": lambda: ClassicV3Detector(threshold_mode="tile"),
        "v3_tile_rel": lambda: ClassicV3Detector(threshold_mode="tile_rel"),
        "v3_snr": lambda: ClassicV3Detector(threshold_mode="snr", k_sigma=6.0),
    }


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="hard_test")
    ap.add_argument("--limit-seq", type=int, default=6)
    ap.add_argument("--dataset", type=Path, default=ev.DEFAULT_DATASET)
    ap.add_argument("--out", type=Path,
                    default=Path("research/results/threshold_ab/summary.json"))
    ap.add_argument("--configs", default="",
                    help="comma separated config subset (default: all)")
    ap.add_argument("--modes", default="",
                    help="comma separated mode subset (default: all)")
    args = ap.parse_args(argv)

    pairs = ev._discover_pairs(args.dataset, "synthetic", subset=args.subset)
    if args.limit_seq:
        pairs = pairs[:args.limit_seq]

    # cache degraded frames per config so 4 threshold modes reuse one decode
    cache: Dict[str, list] = {}
    orig_read = ev._read_video

    def cached_read(path, *a, **kw):
        key = str(path)
        if key not in cache:
            cache[key] = orig_read(path, *a, **kw)
        return cache[key]

    ev._read_video = cached_read
    cfg_all = _configs()
    mode_all = _modes()
    if args.configs:
        keep = {k.strip() for k in args.configs.split(",") if k.strip()}
        cfg_all = {k: v for k, v in cfg_all.items() if k in keep}
    if args.modes:
        keep = {k.strip() for k in args.modes.split(",") if k.strip()}
        mode_all = {k: v for k, v in mode_all.items() if k in keep}
    report = {"subset": args.subset, "n_sequences": len(pairs), "configs": {}}
    try:
        for name, cfg in cfg_all.items():
            cache.clear()
            degrade.set_active(cfg)
            report["configs"][name] = {"degrade": name if cfg else None,
                                       "methods": {}}
            for mname, factory in mode_all.items():
                print(f"[threshold_ab] {name} / {mname} ...", flush=True)
                records = ev.evaluate_classic(pairs, None, factory, mname)
                report["configs"][name]["methods"][mname] = summarize(records, mname)
            # free the frames before the next config
            cache.clear()
    finally:
        ev._read_video = orig_read
        degrade.set_active(None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n== det@10 all (rows=config, cols=threshold mode) ==")
    modes = list(mode_all)
    print(f"{'config':>12s}" + "".join(f"{m:>16s}" for m in modes))
    for name, blk in report["configs"].items():
        row = f"{name:>12s}"
        for m in modes:
            row += f"{blk['methods'][m]['det10_all']:>16.3f}"
        print(row)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
