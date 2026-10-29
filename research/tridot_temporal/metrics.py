"""Honest evaluation metrics for TriDotPose keypoint/pose pipelines.

The legacy ``eval_find.py`` reports ``det@10`` with a denominator of *only the
frames where all three GT dots are visible*.  On ``hard_test`` that discards
~30% of frames (and on ``occlude_hand`` it discards every frame, where the
per-factor fallback then silently scores a 2-point subtask).  ``pix_rmse`` and
``t_err`` are likewise conditional on the model having succeeded, so a method
that fails often is only penalised once.

This module defines denominators explicitly and computes them on richer
per-frame records.  Everything here is pure (no torch/cv2) so it can be unit
tested and re-used to re-score stored ``per_frame.json`` outputs.

Terminology (``COLOR_ORDER = red, green, blue``; GT visibility ``vis`` is
0 = not visible, 1 = partially visible / clipped, 2 = fully visible):

``det10_legacy``
    Frames whose three GT points are all visible; success = a prediction for
    each colour within 10 px.  Reproduces the old number; kept only for
    comparison.

``det10_all``
    Denominator = **all** frames.  Success requires all three GT points to be
    fully visible, *and* three predictions within 10 px.  Any occluded frame is
    a failure.  This is the number to quote.

``det10_visible``
    Denominator = frames where at least one point is partially visible but not
    all fully visible (vis in {1,2} on at least one, not all 2): measures the
    "partially occluded" regime honestly.

``point_recall@t``
    Denominator = GT points with ``vis == 2`` (i.e. individual visible dots).
    Numerator = prediction exists and error <= t.  This is *not* conditional on
    the other two dots.

``occluded_guess@t``
    Denominator = GT points with ``vis < 2`` for which the method still emitted
    a prediction.  Numerator = error <= t.  Measures whether coasting/tracking
    guesses the hidden dot or hallucinates.  Reported separately and never
    counted as a detection success.

``pose_coverage``
    Fraction of all frames with a valid 6-DoF pose (``pose_ok``).

``t_err*``
    Translation error in mm over frames with a pose; ``all`` uses every frame
    with a pose (including occluded frames), ``clean`` restricts to fully
    visible GT frames (comparable to the legacy number).  Report median/p90,
    not only the mean, and always next to ``pose_coverage``.

``jitter_mm_per_s``
    Median frame-to-frame change of the predicted camera position, over
    consecutive frames that both produced a pose, converted to mm/s with the
    sequence FPS.  A static camera should show ~0; it exposes solver jitter
    that accuracy metrics hide.

``recovery_frames``
    Median number of frames needed to re-acquire a colour after a run of >= 3
    consecutive misses.  ``None`` when no such run is followed by a hit.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

COLOR_ORDER = ("red", "green", "blue")
DEFAULT_THRESHOLDS = (2.0, 5.0, 10.0, 20.0)


@dataclass
class FrameRecord:
    """One evaluated frame for one method.

    ``pred`` is aligned with :data:`COLOR_ORDER`; an entry is ``None`` when the
    method did not emit a point for that colour.  ``gt`` entries are
    ``[u, v, vis]`` exactly as stored in the dataset annotation.
    """

    method: str
    seq_id: str
    factor: str
    frame_index: int
    gt: Sequence[Sequence[float]]
    pred: Sequence[Optional[Sequence[float]]]
    pose_ok: bool = False
    pred_t_mm: Optional[Sequence[float]] = None
    gt_t_mm: Optional[Sequence[float]] = None
    latency_ms: float = 0.0
    fps: float = 30.0
    extra: dict = field(default_factory=dict)

    # -- derived helpers ---------------------------------------------------
    @property
    def all_visible(self) -> bool:
        return all(float(g[2]) >= 2.0 for g in self.gt)

    @property
    def any_partial(self) -> bool:
        return any(0.0 < float(g[2]) < 2.0 for g in self.gt)

    def errors(self) -> list[Optional[float]]:
        """Per-colour pixel error, ``None`` where no prediction exists."""
        out: list[Optional[float]] = []
        for k, name in enumerate(COLOR_ORDER):
            p = self.pred[k]
            if p is None:
                out.append(None)
            else:
                out.append(math.hypot(float(p[0]) - float(self.gt[k][0]),
                                      float(p[1]) - float(self.gt[k][1])))
        return out

    def n_pred(self) -> int:
        return sum(1 for p in self.pred if p is not None)


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    s = sorted(float(v) for v in values)
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - pos) + s[hi] * (pos - lo)


def _det_ok(rec: FrameRecord, thr: float) -> bool:
    """All three colours predicted and within ``thr`` px (regardless of vis)."""
    errs = rec.errors()
    return all(e is not None and e <= thr for e in errs)


def det10_legacy(records: Sequence[FrameRecord], thr: float = 10.0) -> Optional[float]:
    denom = [r for r in records if r.all_visible]
    if not denom:
        return None
    return sum(1 for r in denom if _det_ok(r, thr)) / len(denom)


def det10_all(records: Sequence[FrameRecord], thr: float = 10.0) -> Optional[float]:
    if not records:
        return None
    return sum(1 for r in records if r.all_visible and _det_ok(r, thr)) / len(records)


def det10_observable(records: Sequence[FrameRecord],
                     thr: float = 10.0) -> Optional[float]:
    """Honest denominator: frames where all three dots are at least annotated
    as present (``vis >= 1``).

    Fully occluded dots (``vis == 0``) are a sensor-observability limit, not a
    detector failure; ``det10_all`` charges them to the method while
    ``det10_legacy`` drops every partially occluded frame.  This sits in
    between and is the number to quote when comparing against the hard factors
    (``hardmix`` / ``occlude_hand``).
    """
    denom = [r for r in records if all(float(g[2]) >= 1.0 for g in r.gt)]
    if not denom:
        return None
    return sum(1 for r in denom if _det_ok(r, thr)) / len(denom)


def observable_frames(records: Sequence[FrameRecord]) -> int:
    return sum(1 for r in records if all(float(g[2]) >= 1.0 for g in r.gt))


def det10_visible(records: Sequence[FrameRecord],
                  thr: float = 10.0) -> Optional[float]:
    """Frames that are not fully visible (partial-occlusion regime)."""
    denom = [r for r in records if not r.all_visible]
    if not denom:
        return None
    return sum(1 for r in denom if _det_ok(r, thr)) / len(denom)


def point_recall(records: Sequence[FrameRecord], thr: float = 10.0) -> Optional[float]:
    """Recall over individually visible GT points (denominator = vis==2 dots)."""
    hits = total = 0
    for r in records:
        for k in range(3):
            if float(r.gt[k][2]) < 2.0:
                continue
            total += 1
            e = r.errors()[k]
            if e is not None and e <= thr:
                hits += 1
    return hits / total if total else None


def per_color_recall(records: Sequence[FrameRecord],
                     thr: float = 10.0) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for k, name in enumerate(COLOR_ORDER):
        hits = total = 0
        for r in records:
            if float(r.gt[k][2]) < 2.0:
                continue
            total += 1
            e = r.errors()[k]
            if e is not None and e <= thr:
                hits += 1
        out[name] = hits / total if total else None
    return out


def occluded_guess(records: Sequence[FrameRecord],
                   thr: float = 10.0) -> Optional[float]:
    """Among invisible GT points that still got a prediction, how close was it?"""
    hits = total = 0
    for r in records:
        for k in range(3):
            if float(r.gt[k][2]) >= 2.0:
                continue
            e = r.errors()[k]
            if e is None:
                continue
            total += 1
            hits += int(e <= thr)
    return hits / total if total else None


def pose_coverage(records: Sequence[FrameRecord]) -> Optional[float]:
    return _mean(1.0 if r.pose_ok else 0.0 for r in records)


def _t_errors(records: Sequence[FrameRecord], clean: bool) -> list[float]:
    vals = []
    for r in records:
        if not r.pose_ok or r.pred_t_mm is None or r.gt_t_mm is None:
            continue
        if clean and not r.all_visible:
            continue
        vals.append(math.dist([float(x) for x in r.pred_t_mm],
                              [float(x) for x in r.gt_t_mm]))
    return vals


def t_err_stats(records: Sequence[FrameRecord], clean: bool = False) -> dict:
    vals = _t_errors(records, clean)
    return {
        "n": len(vals),
        "mean_mm": _mean(vals),
        "median_mm": _percentile(vals, 50),
        "p90_mm": _percentile(vals, 90),
        "within_50mm": (sum(1 for v in vals if v <= 50.0) / len(vals)) if vals else None,
    }


def distance_bias(records: Sequence[FrameRecord]) -> Optional[float]:
    diffs = []
    for r in records:
        if not r.pose_ok or r.pred_t_mm is None or r.gt_t_mm is None:
            continue
        pn = math.sqrt(sum(float(x) ** 2 for x in r.pred_t_mm))
        gn = math.sqrt(sum(float(x) ** 2 for x in r.gt_t_mm))
        diffs.append(pn - gn)
    return _percentile(diffs, 50)


def jitter(records: Sequence[FrameRecord]) -> Optional[float]:
    """Median |Δposition| between consecutive posed frames, in mm/s."""
    by_seq: dict[str, list[FrameRecord]] = {}
    for r in records:
        by_seq.setdefault(r.seq_id, []).append(r)
    deltas = []
    for seq, rows in by_seq.items():
        rows = sorted(rows, key=lambda r: r.frame_index)
        fps = rows[0].fps if rows and rows[0].fps else 30.0
        prev = None
        for r in rows:
            if not r.pose_ok or r.pred_t_mm is None:
                continue
            cur = [float(x) for x in r.pred_t_mm]
            if prev is not None:
                deltas.append(math.dist(cur, prev) * fps)
            prev = cur
    return _percentile(deltas, 50)


def recovery_frames(records: Sequence[FrameRecord], thr: float = 10.0,
                    min_gap: int = 3, max_lookahead: int = 120) -> Optional[float]:
    """Median frames to re-acquire a colour after a run of >= ``min_gap`` misses."""
    by_seq_color: dict[tuple[str, int], list[FrameRecord]] = {}
    for r in records:
        for k in range(3):
            by_seq_color.setdefault((r.seq_id, k), []).append(r)
    waits = []
    for (_, k), rows in by_seq_color.items():
        rows = sorted(rows, key=lambda r: r.frame_index)
        miss_run = 0
        pending: Optional[int] = None
        for r in rows:
            e = r.errors()[k]
            hit = e is not None and e <= thr
            if hit:
                if miss_run >= min_gap and pending is not None:
                    waits.append(r.frame_index - pending)
                miss_run = 0
                pending = None
            else:
                if miss_run == 0:
                    pending = r.frame_index
                miss_run += 1
                if miss_run >= min_gap and pending is not None and \
                        r.frame_index - pending > max_lookahead:
                    miss_run = 0
                    pending = None
    return _percentile(waits, 50)


def latency_stats(records: Sequence[FrameRecord]) -> dict:
    vals = [r.latency_ms for r in records if r.latency_ms]
    return {
        "median_ms": _percentile(vals, 50),
        "p90_ms": _percentile(vals, 90),
        "fps_est": (1000.0 / _percentile(vals, 50)) if vals and _percentile(vals, 50) else None,
    }


def _weighted(values: dict[str, float], weights: dict[str, int]) -> Optional[float]:
    tot = sum(weights[s] for s in values)
    if not tot:
        return None
    return sum(v * weights[s] for s, v in values.items()) / tot


def bootstrap_ci(by_seq: dict[str, Optional[float]], weights: dict[str, int],
                 n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> Optional[tuple[float, float]]:
    """Percentile bootstrap CI over sequences (sequence is the sampling unit)."""
    seqs = [s for s, v in by_seq.items() if v is not None]
    if len(seqs) < 2:
        return None
    rng = random.Random(seed)
    ests = []
    for _ in range(n_boot):
        sample = [rng.choice(seqs) for _ in seqs]
        vals = {s: by_seq[s] for s in sample}
        wts = {s: weights.get(s, 1) for s in sample}
        v = _weighted(vals, wts)
        if v is not None:
            ests.append(v)
    if not ests:
        return None
    ests.sort()
    lo = ests[int(alpha / 2 * (len(ests) - 1))]
    hi = ests[int((1 - alpha / 2) * (len(ests) - 1))]
    return (lo, hi)


def _frames_by_seq(records: Sequence[FrameRecord]) -> dict[str, list[FrameRecord]]:
    out: dict[str, list[FrameRecord]] = {}
    for r in records:
        out.setdefault(r.seq_id, []).append(r)
    return out


def _metric_by_seq(records: Sequence[FrameRecord],
                   fn: Callable[[Sequence[FrameRecord]], Optional[float]]
                   ) -> tuple[dict[str, Optional[float]], dict[str, int]]:
    by_seq = _frames_by_seq(records)
    vals = {s: fn(rows) for s, rows in by_seq.items()}
    weights = {s: len(rows) for s, rows in by_seq.items()}
    return vals, weights


def _grouped(records: Sequence[FrameRecord], key: str) -> dict[str, list[FrameRecord]]:
    out: dict[str, list[FrameRecord]] = {}
    for r in records:
        out.setdefault(getattr(r, key), []).append(r)
    return out


def summarize(records: Sequence[FrameRecord], n_boot: int = 1000,
              seed: int = 0) -> dict:
    """Full honest summary for one method over one subset."""
    if not records:
        return {"n_frames": 0, "n_seqs": 0}
    method = records[0].method
    thr_all = DEFAULT_THRESHOLDS

    det_by_seq, det_w = _metric_by_seq(records, det10_all)
    t_by_seq, t_w = _metric_by_seq(
        records, lambda rows: t_err_stats(rows, clean=True)["median_mm"])

    by_factor = {}
    for factor, rows in sorted(_grouped(records, "factor").items()):
        by_factor[factor] = {
            "n_frames": len(rows),
            "det10_all": det10_all(rows),
            "det10_legacy": det10_legacy(rows),
            "det10_observable": det10_observable(rows),
            "point_recall@10": point_recall(rows, 10.0),
            "pose_coverage": pose_coverage(rows),
            "t_err_median_mm": t_err_stats(rows, clean=True)["median_mm"],
        }

    summary = {
        "method": method,
        "n_frames": len(records),
        "n_seqs": len(_frames_by_seq(records)),
        "n_all_visible_frames": sum(1 for r in records if r.all_visible),
        "n_vis2_points": sum(1 for r in records for g in r.gt if float(g[2]) >= 2.0),
        "det": {
            "det10_all": det10_all(records),
            "det10_all_ci95": bootstrap_ci(det_by_seq, det_w, n_boot=n_boot, seed=seed),
            "det10_legacy": det10_legacy(records),
            "det10_observable": det10_observable(records),
            "n_observable_frames": observable_frames(records),
            "det10_visible_partial": det10_visible(records),
        },
        "point": {
            f"recall@{int(t)}": point_recall(records, t) for t in thr_all
        } | {
            "per_color_recall@10": per_color_recall(records, 10.0),
            "occluded_guess@10": occluded_guess(records, 10.0),
        },
        "pose": {
            "coverage": pose_coverage(records),
            "t_err_all": t_err_stats(records, clean=False),
            "t_err_clean": t_err_stats(records, clean=True),
            "t_err_clean_ci95_median": bootstrap_ci(
                t_by_seq, t_w, n_boot=n_boot, seed=seed + 1),
            "distance_bias_mm": distance_bias(records),
            "jitter_mm_per_s": jitter(records),
            "recovery_frames": recovery_frames(records),
        },
        "latency": latency_stats(records),
        "by_factor": by_factor,
    }
    return summary


def format_report(summary: dict) -> str:
    """Markdown table rendering of :func:`summarize` output."""
    det = summary["det"]
    pose = summary["pose"]
    ci = det.get("det10_all_ci95")
    ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"
    tci = pose.get("t_err_clean_ci95_median")
    tci_s = f"[{tci[0]:.1f}, {tci[1]:.1f}]mm" if tci else "n/a"
    lines = [
        f"### {summary['method']}",
        "",
        f"frames={summary['n_frames']} seqs={summary['n_seqs']} "
        f"all_visible={summary['n_all_visible_frames']} "
        f"visible_points={summary['n_vis2_points']}",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| det@10 (all frames) | {_fmt(det['det10_all'])}  CI95 {ci_s} |",
        f"| det@10 (legacy, visible only) | {_fmt(det['det10_legacy'])} |",
        f"| det@10 (observable, all dots present) | {_fmt(det['det10_observable'])} "
        f"({det['n_observable_frames']} frames) |",
        f"| det@10 (partial-vis frames) | {_fmt(det['det10_visible_partial'])} |",
        f"| point recall@5 | {_fmt(summary['point']['recall@5'])} |",
        f"| point recall@10 | {_fmt(summary['point']['recall@10'])} |",
        f"| point recall@20 | {_fmt(summary['point']['recall@20'])} |",
        f"| per-colour recall@10 | " +
        ", ".join(f"{k} {_fmt(v)}" for k, v in
                  summary["point"]["per_color_recall@10"].items()) + " |",
        f"| occluded guess@10 | {_fmt(summary['point']['occluded_guess@10'])} |",
        f"| pose coverage | {_fmt(pose['coverage'])} |",
        f"| t_err median (clean) | {_fmt(pose['t_err_clean']['median_mm'], 1)} mm "
        f"CI95 {tci_s} |",
        f"| t_err p90 (clean) | {_fmt(pose['t_err_clean']['p90_mm'], 1)} mm |",
        f"| t_err median (all posed) | {_fmt(pose['t_err_all']['median_mm'], 1)} mm |",
        f"| distance bias (median) | {_fmt(pose['distance_bias_mm'], 1)} mm |",
        f"| jitter | {_fmt(pose['jitter_mm_per_s'], 1)} mm/s |",
        f"| recovery | {_fmt(pose['recovery_frames'], 1)} frames |",
        f"| latency median | {_fmt(summary['latency']['median_ms'], 1)} ms |",
        "",
        "| factor | frames | det@10 all | legacy | observable | recall@10 | pose cov | t_err med |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for factor, row in summary["by_factor"].items():
        lines.append(
            f"| {factor} | {row['n_frames']} | {_fmt(row['det10_all'])} | "
            f"{_fmt(row['det10_legacy'])} | {_fmt(row.get('det10_observable'))} | "
            f"{_fmt(row['point_recall@10'])} | "
            f"{_fmt(row['pose_coverage'])} | "
            f"{_fmt(row['t_err_median_mm'], 1)} |")
    return "\n".join(lines)


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)
