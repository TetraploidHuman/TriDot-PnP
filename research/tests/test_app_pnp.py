"""Tests for the ported app PnP math (``research.tridot_temporal.app_pnp``).

The point of these tests is not that the app solver "works" - it is that we can
*prove* it does not.  ``app_solve_depths`` reproduces the Kotlin update rule;
on noiseless ground-truth points it diverges and lands on the 100 mm depth
floor.  ``fixed_solve_depths`` solves the same constraint equations with Newton
and recovers the ground-truth ranges.  If somebody later "fixes" the app solver,
the ``test_app_solver_documented_failure`` assertion should be updated together
with ``research/results/app_pnp_audit``.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from research.tridot_temporal.app_pnp import (
    app_solve_depths,
    distance_from_points,
    estimate_focal_length,
    fixed_solve_depths,
    object_points_mm,
    solve_pose_from_depths,
)

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "datasets" / "TriDotPose" / "synthetic" / "annotations" / "syn_room_0200.json"


def _load():
    ann = json.loads(ANN.read_text(encoding="utf-8"))
    fr = ann["frames"][0]
    kp = fr["keypoints"]
    uv = np.array([[kp[0], kp[1]], [kp[3], kp[4]], [kp[6], kp[7]]], dtype=float)
    gt_t = np.asarray(fr["pose_gt"]["translation_mm"], dtype=float)
    return ann, fr, uv, gt_t


class FocalTest(unittest.TestCase):
    def test_matches_dataset_camera(self):
        ann, _, _, _ = _load()
        fx = estimate_focal_length(int(ann["info"]["width"]), 60.0)
        self.assertAlmostEqual(fx, ann["camera"]["fx"], places=1)

    def test_object_points_are_equilateral(self):
        p = object_points_mm(100.0)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            self.assertAlmostEqual(np.linalg.norm(p[i] - p[j]), 100.0, places=6)


class AppSolverIsBrokenTest(unittest.TestCase):
    """Documents the shipped solver's numerical failure."""

    @classmethod
    def setUpClass(cls):
        cls.ann, cls.fr, cls.uv, cls.gt_t = _load()
        cls.fx = estimate_focal_length(int(cls.ann["info"]["width"]), 60.0)

    def test_app_solver_does_not_converge(self):
        r = app_solve_depths(self.uv, 100.0, self.fx, self.fx,
                             self.ann["camera"]["cx"], self.ann["camera"]["cy"])
        self.assertFalse(r.converged)
        self.assertGreater(r.max_abs_edge_err_mm, 50.0)

    def test_app_solver_distance_is_off_by_factor(self):
        r = app_solve_depths(self.uv, 100.0, self.fx, self.fx,
                             self.ann["camera"]["cx"], self.ann["camera"]["cy"])
        P, _, _ = solve_pose_from_depths(self.uv, r.depths_mm, 100.0,
                                         self.fx, self.fx,
                                         self.ann["camera"]["cx"], self.ann["camera"]["cy"])
        gt_dist = np.linalg.norm(self.gt_t)
        rel = distance_from_points(P) / gt_dist
        self.assertLess(rel, 0.8)  # app under-reports the distance


class FixedSolverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ann, cls.fr, cls.uv, cls.gt_t = _load()
        cls.fx = estimate_focal_length(int(cls.ann["info"]["width"]), 60.0)

    def test_fixed_solver_converges(self):
        r = fixed_solve_depths(self.uv, 100.0, self.fx, self.fx,
                               self.ann["camera"]["cx"], self.ann["camera"]["cy"])
        self.assertTrue(r.converged)
        self.assertLess(r.max_abs_edge_err_mm, 0.1)

    def test_fixed_solver_returns_a_valid_root(self):
        """Frame 0 is one of the ~10 % frames where 3-point PnP is ambiguous.

        Three rays + three known edge lengths admit *two* depth solutions; the
        reprojection residual is zero for both, so a solver that does not use
        temporal continuity may land on the mirrored one.  What we can require
        is that the returned root actually satisfies the rigid-triangle
        constraints (and is therefore a legitimate PnP answer).
        """
        r = fixed_solve_depths(self.uv, 100.0, self.fx, self.fx,
                               self.ann["camera"]["cx"], self.ann["camera"]["cy"])
        self.assertTrue(r.converged)
        self.assertLess(r.max_abs_edge_err_mm, 0.1)
        P, _, t = solve_pose_from_depths(self.uv, r.depths_mm, 100.0,
                                         self.fx, self.fx,
                                         self.ann["camera"]["cx"], self.ann["camera"]["cy"])
        # distance is unambiguous even when the rotation root is not
        gt_dist = float(np.linalg.norm(self.gt_t))
        self.assertLess(abs(distance_from_points(P) / gt_dist - 1.0), 0.02)

    def test_temporal_seed_picks_the_consistent_root(self):
        """Seeding with the previous frame's depths follows a moving target."""
        from research.tridot_temporal.app_pnp import AppFixedSolver
        solver = AppFixedSolver(fov_deg=60.0)
        ann = json.loads(ANN.read_text(encoding="utf-8"))
        h, w = ann["info"]["height"], ann["info"]["width"]
        worst = 0.0
        for fr in ann["frames"][:10]:
            kp = fr["keypoints"]
            if any(kp[2 + 3 * i] < 2 for i in range(3)):
                continue
            pts = {"red": (kp[0], kp[1]), "green": (kp[3], kp[4]), "blue": (kp[6], kp[7])}
            res = solver.solve(pts, 100.0, w, h)
            self.assertTrue(res.ok)
            worst = max(worst, float(np.linalg.norm(
                np.asarray(res.translation_mm) - np.asarray(fr["pose_gt"]["translation_mm"]))))
        self.assertLess(worst, 50.0)

    def test_fixed_solver_distance_accuracy(self):
        r = fixed_solve_depths(self.uv, 100.0, self.fx, self.fx,
                               self.ann["camera"]["cx"], self.ann["camera"]["cy"])
        P, _, _ = solve_pose_from_depths(self.uv, r.depths_mm, 100.0,
                                         self.fx, self.fx,
                                         self.ann["camera"]["cx"], self.ann["camera"]["cy"])
        gt_dist = np.linalg.norm(self.gt_t)
        self.assertLess(abs(distance_from_points(P) / gt_dist - 1.0), 0.02)

    def test_fixed_solver_handles_degenerate_rays(self):
        uv = np.array([[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]])
        r = fixed_solve_depths(uv, 100.0, 1108.5, 1108.5, 640.0, 360.0)
        self.assertTrue(np.all(np.isfinite(r.depths_mm)))
        self.assertTrue(np.all(r.depths_mm >= 100.0))


if __name__ == "__main__":
    unittest.main()
