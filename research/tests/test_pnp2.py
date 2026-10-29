"""Unit tests for the two-point PnP + temporal-prior solver."""
from __future__ import annotations

import unittest

import numpy as np

from research.tridot_temporal.app_pnp import estimate_focal_length, object_points_mm
from research.tridot_temporal.pnp2 import exp_so3, rotation_angle_deg, solve_pose_2point

EDGE = 100.0
FX = FY = estimate_focal_length(1280, 60.0)
CX, CY = 639.5, 359.5


def _project(R, t, P):
    p = R @ P + t
    return np.array([FX * p[0] / p[2] + CX, FY * p[1] / p[2] + CY])


def _scene(pitch_deg=8.0, yaw_deg=-6.0, t=None):
    R = exp_so3(np.radians([pitch_deg, 0.0, yaw_deg]))
    if t is None:
        t = np.array([12.0, -18.0, 1210.0])
    obj = object_points_mm(EDGE)
    uv = np.array([_project(R, t, obj[i]) for i in range(3)])
    return R, t, obj, uv


class TwoPointSolverTest(unittest.TestCase):
    def test_exact_prior_recovers_pose(self):
        R, t, obj, uv = _scene()
        res = solve_pose_2point(uv[:2], obj[:2], FX, FY, CX, CY, R, t.copy())
        self.assertTrue(res.ok, res.reason)
        self.assertLess(float(np.linalg.norm(res.translation_mm - t)), 2.0)
        self.assertLess(rotation_angle_deg(res.rotation, R), 1.0)
        self.assertLess(res.reproj_px, 1.0)

    def test_noisy_prior_still_recovers(self):
        R, t, obj, uv = _scene()
        R_prior = exp_so3(np.radians([2.0, 1.0, -3.0])) @ R
        t_prior = t + np.array([15.0, -10.0, 40.0])
        res = solve_pose_2point(uv[:2], obj[:2], FX, FY, CX, CY, R_prior, t_prior)
        self.assertTrue(res.ok, res.reason)
        self.assertLess(float(np.linalg.norm(res.translation_mm - t)), 8.0)
        self.assertLess(rotation_angle_deg(res.rotation, R), 4.0)

    def test_bad_prior_is_rejected(self):
        R, t, obj, uv = _scene()
        res = solve_pose_2point(uv[:2], obj[:2], FX, FY, CX, CY, np.eye(3),
                               np.array([0.0, 0.0, 600.0]))
        self.assertFalse(res.ok)
        self.assertIn(res.reason, {"translation-jump", "rotation-jump",
                                   "reproj-too-large"})
        self.assertIsNone(res.translation_mm)

    def test_requires_exactly_two_points(self):
        R, t, obj, uv = _scene()
        res = solve_pose_2point(uv[:1], obj[:1], FX, FY, CX, CY, R, t.copy())
        self.assertFalse(res.ok)
        self.assertEqual(res.reason, "need-exactly-2-points")

    def test_observed_edge_is_explained(self):
        R, t, obj, uv = _scene()
        res = solve_pose_2point(uv[:2], obj[:2], FX, FY, CX, CY, R, t.copy())
        self.assertTrue(res.ok, res.reason)
        p = [res.rotation @ obj[i] + res.translation_mm for i in range(2)]
        proj = [np.array([FX * q[0] / q[2] + CX, FY * q[1] / q[2] + CY]) for q in p]
        gt_edge_px = float(np.linalg.norm(uv[0] - uv[1]))
        got_edge_px = float(np.linalg.norm(proj[0] - proj[1]))
        self.assertAlmostEqual(gt_edge_px, got_edge_px, delta=0.5)


if __name__ == "__main__":
    unittest.main()
