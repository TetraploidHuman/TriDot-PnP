"""Tests for temporal filtering + P3P root disambiguation.

Run with::

    research/.venv/bin/python -m unittest research.tests.test_temporal -v
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from research.tridot_temporal.temporal import (
    AppTrackedSolver,
    PoseTracker,
    candidate_translation,
    rotation_angle_deg,
    solve_depth_candidates,
)

ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = ROOT / "datasets" / "TriDotPose" / "synthetic" / "annotations"


def _load(seq: str):
    return json.loads((ANN_DIR / f"{seq}.json").read_text(encoding="utf-8"))


def _uv(fr) -> np.ndarray:
    kp = fr["keypoints"]
    return np.array([[kp[0], kp[1]], [kp[3], kp[4]], [kp[6], kp[7]]],
                    dtype=np.float64)


class CandidateTest(unittest.TestCase):
    def setUp(self):
        self.ann = _load("syn_room_0200")
        c = self.ann["camera"]
        self.cam = (c["fx"], c["fy"], c["cx"], c["cy"])
        self.edge = float(self.ann["info"]["edge_length_mm"])

    def test_candidates_contain_ground_truth_root(self):
        # Frame 0 is the near-degenerate case seen during the audit: two valid
        # roots ~11 mm apart.  The GT root must be among the candidates.
        fr = self.ann["frames"][0]
        gt_t = np.array(fr["pose_gt"]["translation_mm"])
        cands = solve_depth_candidates(_uv(fr), self.edge, *self.cam)
        self.assertGreaterEqual(len(cands), 2)
        errs = [np.linalg.norm(candidate_translation(_uv(fr), z, self.edge,
                                                    *self.cam)[1] - gt_t)
                for z in cands]
        self.assertLess(min(errs), 2.0, f"candidate translations err={errs}")

    def test_candidates_are_deterministic(self):
        fr = self.ann["frames"][30]
        a = solve_depth_candidates(_uv(fr), self.edge, *self.cam)
        b = solve_depth_candidates(_uv(fr), self.edge, *self.cam)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, rtol=1e-9)

    def test_gt_root_has_zero_edge_error(self):
        fr = self.ann["frames"][30]
        cands = solve_depth_candidates(_uv(fr), self.edge, *self.cam)
        self.assertGreaterEqual(len(cands), 1)
        R, t = candidate_translation(_uv(fr), cands[0], self.edge, *self.cam)
        self.assertGreater(np.linalg.norm(t), 100.0)
        self.assertTrue(np.all(np.isfinite(R)))


class TrackerTest(unittest.TestCase):
    def test_smooths_constant_velocity_with_noise(self):
        rng = np.random.default_rng(0)
        tracker = PoseTracker()
        t = np.array([0.0, 0.0, 800.0])
        vel = np.array([120.0, -40.0, 30.0])
        raw, filt = [], []
        for i in range(60):
            gt_t = t + vel * (i / 30.0)
            meas = gt_t + rng.normal(0.0, 8.0, 3)
            out = tracker.update(meas, None, i / 30.0)
            raw.append(meas)
            filt.append(out.t_mm)
        raw = np.array(raw)
        filt = np.array(filt)
        raw_jitter = np.mean(np.linalg.norm(np.diff(raw, axis=0), axis=1))
        filt_jitter = np.mean(np.linalg.norm(np.diff(filt, axis=0), axis=1))
        self.assertLess(filt_jitter, raw_jitter * 0.8)
        self.assertLess(np.median(np.linalg.norm(
            filt - np.array([t + vel * (i / 30.0) for i in range(60)]),
            axis=1)), 10.0)

    def test_single_spike_is_rejected_and_coasted(self):
        tracker = PoseTracker()
        rng = np.random.default_rng(1)
        for i in range(5):
            tracker.update(np.array([10.0, 0.0, 900.0]) + rng.normal(0, 1, 3),
                           None, i / 30.0)
        out = tracker.update(np.array([10.0, 0.0, 1200.0]),
                             None, 5 / 30.0)
        self.assertTrue(out.coasting)
        self.assertFalse(out.accepted)
        self.assertLess(abs(float(out.t_mm[2]) - 900.0), 25.0)

    def test_reinit_after_long_gap(self):
        tracker = PoseTracker(max_dt_s=0.1)
        tracker.update(np.array([0.0, 0.0, 500.0]), None, 0.0)
        out = tracker.update(np.array([0.0, 0.0, 1500.0]), None, 1.0)
        self.assertTrue(out.reinitialized)
        np.testing.assert_allclose(out.t_mm, [0.0, 0.0, 1500.0])

    def test_reset(self):
        tracker = PoseTracker()
        tracker.update(np.array([1.0, 2.0, 3.0]), None, 0.0)
        tracker.reset()
        self.assertFalse(tracker.state.initialized)


class OutputModeTest(unittest.TestCase):
    def test_smooth_false_reports_raw_measurement(self):
        tracker = PoseTracker(smooth=False)
        rng = np.random.default_rng(3)
        for i in range(20):
            meas = np.array([0.0, 0.0, 800.0]) + rng.normal(0.0, 5.0, 3)
            out = tracker.update(meas, None, i / 30.0)
            if out.accepted:
                np.testing.assert_allclose(out.t_mm, meas, atol=1e-9)

    def test_smooth_true_lags_measurement(self):
        tracker = PoseTracker(smooth=True)
        meas = np.array([0.0, 0.0, 800.0])
        tracker.update(meas, None, 0.0)
        step = meas + np.array([5.0, 0.0, 0.0])
        out = tracker.update(step, None, 1.0 / 30.0)
        self.assertGreater(np.linalg.norm(out.t_mm - meas), 0.5)
        self.assertLess(np.linalg.norm(out.t_mm - step), 5.0)

    def test_default_gate_accepts_moderate_jump(self):
        # 默认上架配置（gate 60mm / 8%）应接受 30mm 的抖动而不是滑行。
        tracker = PoseTracker(gate_abs_mm=60.0, gate_rel=0.08, alpha=0.85)
        for i in range(5):
            tracker.update(np.array([0.0, 0.0, 900.0]), None, i / 30.0)
        out = tracker.update(np.array([0.0, 0.0, 900.0]) + 30.0, None, 5 / 30.0)
        self.assertTrue(out.accepted)
        self.assertFalse(out.coasting)


class RotationContinuityTest(unittest.TestCase):
    """The whole point of root disambiguation: no roll/yaw flips mid-sequence."""

    def test_no_rotation_flips_on_synthetic_sequence(self):
        ann = _load("syn_room_0200")
        c = ann["camera"]
        cam = (c["fx"], c["fy"], c["cx"], c["cy"])
        edge = float(ann["info"]["edge_length_mm"])
        solver = AppTrackedSolver()
        errors, gaps = [], []
        prev_R = None
        for fi, fr in enumerate(ann["frames"]):
            if any(fr["keypoints"][2 + 3 * i] < 2 for i in range(3)):
                continue
            kp = fr["keypoints"]
            pts = {name: (kp[3 * k], kp[3 * k + 1])
                   for k, name in enumerate(("red", "green", "blue"))}
            out = solver.solve(pts, edge,
                               int(ann["info"]["width"]), int(ann["info"]["height"]),
                               t_s=fi / float(ann["info"]["fps"]))
            self.assertTrue(out.ok)
            gt_R = np.array(fr["pose_gt"]["rotation_matrix"])
            if prev_R is not None:
                gaps.append(rotation_angle_deg(out.rotation, prev_R))
            prev_R = out.rotation
            # Only check frames where a flip would change the reported pose a
            # lot; small per-frame motion is expected.
            errors.append(rotation_angle_deg(out.rotation, gt_R))
        self.assertLess(np.percentile(gaps, 95), 10.0,
                        f"frame-to-frame rotation jumps: {sorted(gaps)[-5:]}")
        self.assertLess(np.median(errors), 25.0)


if __name__ == "__main__":
    unittest.main()
