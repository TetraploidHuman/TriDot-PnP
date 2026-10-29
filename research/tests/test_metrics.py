"""Unit tests for the honest evaluation metrics (no dataset / torch needed).

Run with::

    research/.venv/bin/python -m unittest discover -s research/tests -v
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.tridot_temporal.metrics import (  # noqa: E402
    FrameRecord,
    bootstrap_ci,
    det10_all,
    det10_legacy,
    det10_visible,
    distance_bias,
    jitter,
    occluded_guess,
    per_color_recall,
    point_recall,
    pose_coverage,
    recovery_frames,
    summarize,
    t_err_stats,
)


def rec(seq="s1", fi=0, gt=None, pred=None, pose_ok=False, pred_t=None,
        gt_t=(0.0, 0.0, 1000.0), factor="clean", fps=30.0):
    gt = gt or [[0.0, 0.0, 2.0], [100.0, 0.0, 2.0], [0.0, 100.0, 2.0]]
    return FrameRecord(
        method="m", seq_id=seq, factor=factor, frame_index=fi,
        gt=gt, pred=pred or [None, None, None], pose_ok=pose_ok,
        pred_t_mm=pred_t, gt_t_mm=list(gt_t), latency_ms=10.0, fps=fps,
    )


class DenominatorTests(unittest.TestCase):
    def setUp(self):
        # frame 0: all visible, perfect detection
        self.f0 = rec(fi=0, pred=[[0, 0], [100, 0], [0, 100]])
        # frame 1: all visible, blue missed
        self.f1 = rec(fi=1, gt=[[0, 0, 2], [100, 0, 2], [0, 100, 2]],
                      pred=[[0, 0], [100, 0], None])
        # frame 2: red partially visible (vis=1), all three found
        self.f2 = rec(fi=2, gt=[[0, 0, 1], [100, 0, 2], [0, 100, 2]],
                      pred=[[0, 0], [100, 0], [0, 100]])
        self.records = [self.f0, self.f1, self.f2]

    def test_legacy_denominator_ignores_occluded_frames(self):
        self.assertAlmostEqual(det10_legacy(self.records), 0.5)

    def test_all_frames_denominator_counts_occlusion_as_failure(self):
        # only frame 0 qualifies: frame 1 misses blue, frame 2 is not all-visible
        self.assertAlmostEqual(det10_all(self.records), 1 / 3)

    def test_partial_regime(self):
        self.assertAlmostEqual(det10_visible(self.records), 1.0)

    def test_point_recall_uses_visible_points(self):
        # visible points: f0 3, f1 3, f2 2 = 8; hits: f0 3, f1 2, f2 2 = 7
        self.assertAlmostEqual(point_recall(self.records, 10.0), 7 / 8)

    def test_per_color_recall(self):
        out = per_color_recall(self.records, 10.0)
        self.assertAlmostEqual(out["red"], 1.0)      # 2/2 (f2 partial excluded)
        self.assertAlmostEqual(out["green"], 1.0)
        self.assertAlmostEqual(out["blue"], 2 / 3)   # missed in f1

    def test_occluded_guess_measures_coasting(self):
        # only one invisible GT point appears (f2 red at distance 0 -> hit)
        self.assertAlmostEqual(occluded_guess(self.records, 10.0), 1.0)

    def test_empty(self):
        self.assertIsNone(det10_all([]))
        self.assertIsNone(point_recall([], 10.0))


class PoseTests(unittest.TestCase):
    def test_coverage_and_clean_filter(self):
        allv = rec(fi=0, pose_ok=True, pred_t=(0, 0, 1010), gt_t=(0, 0, 1000))
        occl = rec(fi=1, gt=[[0, 0, 1], [100, 0, 2], [0, 100, 2]],
                   pose_ok=True, pred_t=(0, 0, 1200), gt_t=(0, 0, 1000))
        fail = rec(fi=2, pose_ok=False)
        rows = [allv, occl, fail]
        self.assertAlmostEqual(pose_coverage(rows), 2 / 3)
        clean = t_err_stats(rows, clean=True)
        self.assertEqual(clean["n"], 1)
        self.assertAlmostEqual(clean["median_mm"], 10.0)
        every = t_err_stats(rows, clean=False)
        self.assertEqual(every["n"], 2)
        self.assertAlmostEqual(every["median_mm"], 105.0)

    def test_distance_bias(self):
        rows = [rec(pose_ok=True, pred_t=(0, 0, 1100), gt_t=(0, 0, 1000)),
                rec(pose_ok=True, pred_t=(0, 0, 900), gt_t=(0, 0, 1000))]
        self.assertAlmostEqual(distance_bias(rows), 0.0)

    def test_jitter_zero_for_static_camera(self):
        rows = [rec(fi=i, pose_ok=True, pred_t=(0, 0, 1000)) for i in range(5)]
        self.assertAlmostEqual(jitter(rows), 0.0)

    def test_jitter_units_mm_per_s(self):
        rows = [rec(fi=0, pose_ok=True, pred_t=(0, 0, 1000)),
                rec(fi=1, pose_ok=True, pred_t=(0, 0, 1010))]
        self.assertAlmostEqual(jitter(rows), 10.0 * 30.0)


class RecoveryTests(unittest.TestCase):
    def test_recovery_after_run_of_misses(self):
        good = [[0, 0], [100, 0], [0, 100]]
        rows = [rec(fi=i, pred=good) for i in range(3)]
        rows += [rec(fi=i, pred=[[0, 0], [100, 0], None]) for i in (3, 4, 5)]
        rows += [rec(fi=i, pred=good) for i in (6, 7)]
        self.assertAlmostEqual(recovery_frames(rows), 3.0)

    def test_no_recovery_when_never_recaptured(self):
        good = [[0, 0], [100, 0], [0, 100]]
        rows = [rec(fi=i, pred=good) for i in range(2)]
        rows += [rec(fi=i, pred=[[0, 0], [100, 0], None]) for i in range(2, 8)]
        self.assertIsNone(recovery_frames(rows))


class BootstrapTests(unittest.TestCase):
    def test_deterministic_and_bracketing(self):
        by_seq = {"a": 0.5, "b": 0.7, "c": 0.9, "d": 0.3}
        w = {s: 100 for s in by_seq}
        ci1 = bootstrap_ci(by_seq, w, n_boot=200, seed=7)
        ci2 = bootstrap_ci(by_seq, w, n_boot=200, seed=7)
        self.assertEqual(ci1, ci2)
        self.assertLessEqual(ci1[0], 0.6)
        self.assertGreaterEqual(ci1[1], 0.6)

    def test_needs_two_sequences(self):
        self.assertIsNone(bootstrap_ci({"a": 0.5}, {"a": 10}))


class SummaryTests(unittest.TestCase):
    def test_summary_shape_and_report(self):
        rows = [rec(fi=i, pred=[[0, 0], [100, 0], [0, 100]], pose_ok=True,
                    pred_t=(0, 0, 1000)) for i in range(4)]
        sm = summarize(rows, n_boot=50)
        self.assertEqual(sm["n_frames"], 4)
        self.assertAlmostEqual(sm["det"]["det10_all"], 1.0)
        self.assertAlmostEqual(sm["det"]["det10_legacy"], 1.0)
        self.assertIn("clean", sm["by_factor"])
        self.assertEqual(summarize([])["n_frames"], 0)

    def test_roundtrip_asdict(self):
        import dataclasses
        import json
        r = rec(fi=3)
        d = dataclasses.asdict(r)
        r2 = FrameRecord(**d)
        self.assertEqual(r2.errors(), r.errors())


if __name__ == "__main__":
    unittest.main()
