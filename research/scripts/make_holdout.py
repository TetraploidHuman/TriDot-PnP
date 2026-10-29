"""Generate a fresh, never-tuned holdout set with the recovered tridot_synth.

Why: the existing `test` / `hard_test` splits have been used for model and
checkpoint selection dozens of times (see research/results/*/REPORT.md).  Any
number measured on them is optimistic.  The synthetic generator (v1.1.0) is
still available through the bytecode loader, so we can draw *new* sequences
from the same distributions with untouched seeds.

Sets produced
-------------
holdout_easy : 15 sequences, base presets (blur/clean/distractor/light/mixed/
               noise/occ) - the same distribution as the `test` split.
holdout_hard : 40 sequences, realistic presets (clutter/crosscolor/desk/far/
               handheld/hardmix/occlude_hand/room/screen) - the same
               distribution as the `hard_test` split.

Sequences are written into ``datasets/TriDotPose/synthetic/{videos,annotations}``
and registered as new keys in ``meta/splits.json`` (backed up first).  Existing
splits are not modified.

Usage::

    research/.venv/bin/python -m research.scripts.make_holdout [--out-root ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from research.pyc_import import install as _install_pyc_import  # noqa: E402

_install_pyc_import()

from research.tridot_synth.generate import (  # noqa: E402
    build_full_specs,
    build_realistic_specs,
    generate_sequence,
)

SEED_OFFSET = 500_000  # far away from every seed used by the shipped dataset


def _pick_easy_specs(n: int = 15) -> list:
    """Round-robin over factors so the holdout matches the base split mix."""
    specs = build_full_specs()
    by_factor: dict[str, list] = {}
    for spec in specs:
        by_factor.setdefault(spec.factor, []).append(spec)
    factors = sorted(by_factor)
    picked = []
    i = 0
    while len(picked) < n:
        progressed = False
        for factor in factors:
            bucket = by_factor[factor]
            if i < len(bucket) and len(picked) < n:
                picked.append(bucket[i])
                progressed = True
        if not progressed:
            break
        i += 1
    return picked


def _clone(specs, prefix: str) -> list:
    out = []
    for i, spec in enumerate(specs, start=1):
        out.append(dataclasses.replace(
            spec,
            seq_id=f"{prefix}_{i:04d}",
            seed=int(spec.seed) + SEED_OFFSET,
        ))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "TriDotPose")
    ap.add_argument("--skip-easy", action="store_true")
    ap.add_argument("--skip-hard", action="store_true")
    args = ap.parse_args(argv)

    dataset = args.dataset
    synthetic = dataset / "synthetic"
    (synthetic / "videos").mkdir(parents=True, exist_ok=True)
    (synthetic / "annotations").mkdir(parents=True, exist_ok=True)

    sets = {}
    if not args.skip_easy:
        sets["holdout_easy"] = _clone(_pick_easy_specs(15), "syn_hold_easy")
    if not args.skip_hard:
        sets["holdout_hard"] = _clone(build_realistic_specs(), "syn_hold_hard")

    t0 = time.time()
    done = 0
    for name, specs in sets.items():
        for spec in specs:
            info = generate_sequence(spec, dataset, write_examples_copy=False)
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (sum(len(v) for v in sets.values()) - done)
            print(f"[make_holdout] {done} {spec.seq_id} factor={spec.factor} "
                  f"frames={info.get('num_frames')} elapsed={elapsed:.0f}s "
                  f"eta={eta:.0f}s", flush=True)

    splits_path = dataset / "meta" / "splits.json"
    backup = splits_path.with_suffix(".json.bak_holdout")
    if not backup.exists():
        shutil.copy2(splits_path, backup)
        print(f"[make_holdout] backed up {splits_path.name} -> {backup.name}", flush=True)
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    for name, specs in sets.items():
        splits[name] = [s.seq_id for s in specs]
    splits_path.write_text(json.dumps(splits, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    info_path = dataset / "meta" / "holdout_info.json"
    info_path.write_text(json.dumps({
        "generator": "tridot_synth v1.1.0 (recovered from __pycache__)",
        "seed_offset": SEED_OFFSET,
        "sets": {
            name: [{"seq_id": s.seq_id, "factor": s.factor, "seed": s.seed}
                   for s in specs]
            for name, specs in sets.items()
        },
        "note": "fresh sequences; never used for training or checkpoint selection",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[make_holdout] splits updated: "
          + ", ".join(f"{k}={len(v)}" for k, v in sets.items()), flush=True)
    print(f"[make_holdout] total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
