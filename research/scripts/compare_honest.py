#!/usr/bin/env python3
"""Render the honest-evaluation summaries as a markdown comparison table.

Reads every ``research/results/honest/*/summary.json`` (one per method/subset
run produced by ``eval_v2.py``) and prints:

* the headline table (all-frames det@10, legacy det@10, point recall, pose
  coverage, translation error on clean frames, jitter, latency);
* per-factor detail for the hard test subset (where classic CV falls apart);
* for every row a reminder of *which* code path produced it, because the whole
  point of this harness is that the number is attributable.

Usage::

    research/.venv/bin/python research/scripts/compare_honest.py
    research/.venv/bin/python research/scripts/compare_honest.py --md out.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HONEST = ROOT / "research" / "results" / "honest"


def load() -> dict:
    rows = {}
    for p in sorted(HONEST.glob("*/summary.json")):
        s = json.loads(p.read_text())
        for key, r in s.get("results", {}).items():
            sub = s["provenance"]["subset"]
            label = key
            cfg = s["provenance"].get("method_cfg", {}).get(key, {})
            if key == "find_geom" and cfg.get("temporal_w") is not None:
                label = f"find_geom(w{int(cfg['temporal_w'])})"
            rows[(sub, label)] = (p.parent.name, s["provenance"], r)
    return rows


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--hard-factor", action="store_true", help="per-factor table")
    args = ap.parse_args()
    rows = load()
    order_subsets = ["test", "hard_test"]
    order_methods = ["b0", "b1", "app_fixed", "find_geom(w0)", "find_geom(w3)"]

    lines = []
    lines.append("# Honest evaluation — TriDotPose\n")
    lines.append("Produced by `research/tridot_temporal/eval_v2.py`; metrics defined in")
    lines.append("`research/tridot_temporal/metrics.py`. Every number is attributable to a")
    lines.append("checkpoint hash + split hash recorded in the per-run `summary.json`.\n")
    lines.append("Definitions that differ from the historical `eval_find.py` reports:\n")
    lines.append("- **det@10 (all frames)**: denominator is *every* frame; an occluded frame counts")
    lines.append("  as a miss because the product must output three points. The historical number")
    lines.append("  only counted frames where all three GT dots were visible (``legacy`` column).")
    lines.append("- **pose coverage**: fraction of frames where PnP returned a pose.")
    lines.append("- **t_err (clean)**: translation error on frames with three GT-visible dots and a")
    lines.append("  successful PnP; median / p90 reported. Mean is dominated by PnP flip outliers.")
    lines.append("- **jitter**: median |Δt|×fps between consecutive frames that produced a pose.")
    lines.append("- **point recall@10**: per-dot recall over GT-visible dots (main three).")
    lines.append("- **b1**: Python port of the Android app detector (HSV grid + probability")
    lines.append("  matrix) with the *shipped* iterative depth solver, i.e. the app as released.")
    lines.append("- **app_fixed**: same app detector, but driving the fixed Newton/LM depth solver —")
    lines.append("  what the app scores once only the PnP math is corrected.")
    lines.append("")

    for sub in order_subsets:
        keys = [(k, v) for k, v in rows.items() if k[0] == sub]
        if not keys:
            continue
        lines.append(f"## {sub}\n")
        lines.append("| method | det@10 all | det@10 legacy | recall@10 (pt) | pose cov | t_err med | t_err p90 | jitter mm/s | latency ms | est FPS |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for name in order_methods:
            hit = [v for k, v in keys if k[1] == name]
            if not hit:
                continue
            _, prov, r = hit[0]
            det = r["det"]
            pt = r["point"]
            po = r["pose"]
            la = r["latency"]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    name,
                    fmt(det["det10_all"]),
                    fmt(det["det10_legacy"]),
                    fmt(pt["recall@10"]),
                    fmt(po["coverage"]),
                    fmt(po["t_err_clean"]["median_mm"], 1),
                    fmt(po["t_err_clean"]["p90_mm"], 1),
                    fmt(po["jitter_mm_per_s"], 0),
                    fmt(la["median_ms"], 0),
                    fmt(la["fps_est"], 1),
                )
            )
        lines.append("")
        if args.hard_factor or sub == "hard_test":
            names = {}
            for k, v in keys:
                names.setdefault(k[1], v[2])
            factors = sorted({f for r in names.values() for f in r["by_factor"]})
            hdr = ["factor"] + [n for n in order_methods if n in names]
            lines.append("| " + " | ".join(hdr) + " |")
            lines.append("|" + "---|" * len(hdr))
            for f in factors:
                cells = [f]
                for n in [x for x in order_methods if x in names]:
                    bf = names[n]["by_factor"].get(f)
                    cells.append("-" if not bf else fmt(bf["det10_all"], 3))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    lines.append("## Provenance\n")
    seen = set()
    for (sub, name), (dirname, prov, r) in sorted(rows.items()):
        tag = prov.get("find_ckpt_sha256", "")
        key = (name, sub)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- `{dirname}`: subset=`{sub}` method=`{name}` "
                     f"find_ckpt={prov.get('find_ckpt', '-')} "
                     f"({tag[:12] if tag else '-'}) geom_ckpt={prov.get('geom_ckpt', '-')} "
                     f"splits={prov['splits_sha256'][:12]} seed={prov['seed']} "
                     f"torch={prov['versions']['torch']}")
    text = "\n".join(lines) + "\n"
    if args.md:
        args.md.write_text(text)
        print(f"wrote {args.md}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
