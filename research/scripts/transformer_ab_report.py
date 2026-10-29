"""Render a cross-subset / cross-method comparison from eval_v2 outputs.

Reads ``<root>/<dir>/summary.json`` files written by ``eval_v2.py`` and emits a
markdown table (stdout or ``--md`` file) plus a JSON bundle.

Example::

    research/.venv/bin/python -m research.scripts.transformer_ab_report \
        --root research/results/honest \
        --md research/results/transformer_ab/REPORT.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CORE_KEYS = (
    ("det", "det10_all"),
    ("det", "det10_legacy"),
    ("point", "recall@10"),
    ("pose", "coverage"),
    ("pose", "t_err_clean"),
    ("latency", "median_ms"),
)


def _g(node, *path, default=None):
    cur = node
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _fmt(v, digits=2, pct=False):
    if v is None:
        return "—"
    if isinstance(v, float):
        if pct:
            return f"{v * 100:.1f}%"
        return f"{v:.{digits}f}"
    return str(v)


def load(root: Path):
    rows = []
    for summary in sorted(root.glob("*/summary.json")):
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        prov = data.get("provenance", {})
        subset = prov.get("subset", summary.parent.name)
        for method, res in data.get("results", {}).items():
            det = res.get("det", {})
            pose = res.get("pose", {})
            point = res.get("point", {})
            lat = res.get("latency", {})
            t_clean = pose.get("t_err_clean") or {}
            t_all = pose.get("t_err_all") or {}
            rows.append({
                "experiment": summary.parent.name,
                "subset": subset,
                "method": method,
                "n_frames": res.get("n_frames"),
                "det10_all": det.get("det10_all"),
                "det10_legacy": det.get("det10_legacy"),
                "recall10": point.get("recall@10"),
                "recall20": point.get("recall@20"),
                "pose_cov": pose.get("coverage"),
                "t_med_clean": t_clean.get("median_mm"),
                "t_p90_clean": t_clean.get("p90_mm"),
                "t_med_all": t_all.get("median_mm"),
                "within50": t_clean.get("within_50mm"),
                "jitter": pose.get("jitter_mm_s"),
                "lat_ms": lat.get("median_ms"),
                "fps": lat.get("fps_est"),
            })
    return rows


ORDER = ["b0", "find_geom(w0)", "find_geom(w0+snap14)",
         "find_offset", "find_offset(w0+snap14)"]


def _sort_key(row):
    method = row["method"]
    base = method.split("(")[0]
    try:
        idx = ORDER.index(base)
    except ValueError:
        idx = len(ORDER)
    return (row["subset"], idx, method)


def render(rows, title: str) -> str:
    lines = [f"# {title}", ""]
    subsets = []
    for row in rows:
        if row["subset"] not in subsets:
            subsets.append(row["subset"])
    for subset in subsets:
        group = sorted([r for r in rows if r["subset"] == subset], key=_sort_key)
        lines.append(f"## {subset}")
        lines.append("")
        lines.append("| method | det@10 all | det@10 legacy | recall@10 | pose cov "
                     "| t_err med (mm) | t_err p90 (mm) | ≤50mm | jitter (mm/s) "
                     "| latency (ms) | fps |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in group:
            lines.append(
                f"| {r['method']} | {_fmt(r['det10_all'], pct=True)} "
                f"| {_fmt(r['det10_legacy'], pct=True)} "
                f"| {_fmt(r['recall10'], pct=True)} "
                f"| {_fmt(r['pose_cov'], pct=True)} "
                f"| {_fmt(r['t_med_clean'], 1)} | {_fmt(r['t_p90_clean'], 1)} "
                f"| {_fmt(r['within50'], pct=True)} "
                f"| {_fmt(r['jitter'], 0)} | {_fmt(r['lat_ms'], 1)} "
                f"| {_fmt(r['fps'], 1)} |")
        lines.append("")
    lines.append("## provenance")
    lines.append("")
    seen = set()
    for r in rows:
        if r["experiment"] in seen:
            continue
        seen.add(r["experiment"])
        lines.append(f"- `{r['experiment']}` — subset `{r['subset']}`")
    lines.append("")
    lines.append("Columns: **det@10 all** = frames with all three LEDs within 10px "
                 "over *all* frames (occluded frames count as failures, so this is "
                 "the honest number); **det@10 legacy** = same but over fully "
                 "visible frames only (the old, optimistic metric); **recall@10** = "
                 "per-LED recall on visible LEDs; **t_err** = |t_pred − t_gt| for "
                 "frames where the pose solved and all points were visible.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT / "research" / "results" / "honest")
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--title", default="TriDot 检测器/解算器对比（诚实口径）")
    ap.add_argument("--only", default=None,
                    help="comma list of experiment dir names to include")
    args = ap.parse_args(argv)

    rows = load(args.root)
    if args.only:
        keep = {s.strip() for s in args.only.split(",") if s.strip()}
        rows = [r for r in rows if r["experiment"] in keep]
    md = render(rows, args.title)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md, encoding="utf-8")
        print(f"wrote {args.md}")
    else:
        print(md)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
