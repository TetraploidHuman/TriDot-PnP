"""Unit tests for the classic-CV v2 detector (colour-difference + triple search)."""
from __future__ import annotations

import unittest

import numpy as np

from research.baseline_b0.pnp import PoseSolver
from research.tridot_temporal.classic_v2 import (COLOR_ORDER, ClassicV2Detector,
                                                 color_weight_maps)

W, H = 640, 480
FX = FY = 800.0
CX, CY = W / 2.0, H / 2.0
EDGE_MM = 100.0
DEPTH_MM = 700.0
BGR = {"red": (0.0, 0.0, 255.0), "green": (0.0, 255.0, 0.0),
       "blue": (255.0, 0.0, 0.0)}


def _object_points(edge_mm=EDGE_MM):
    return np.array([[0.0, 0.0, 0.0],
                     [edge_mm, 0.0, 0.0],
                     [edge_mm / 2.0, edge_mm * np.sqrt(3.0) / 2.0, 0.0]])


def _project(R, t, edge_mm=EDGE_MM):
    obj = _object_points(edge_mm)
    cam = (R @ obj.T).T + t[None, :]
    return {c: (FX * cam[i, 0] / cam[i, 2] + CX,
                FY * cam[i, 1] / cam[i, 2] + CY)
            for i, c in enumerate(COLOR_ORDER)}, cam


def _render(points, sigma=2.2, peak=230.0, extra=()):
    img = np.zeros((H, W, 3), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    dots = [(points[c], COLOR_ORDER[i]) for i, c in enumerate(COLOR_ORDER)]
    dots += list(extra)
    for (u, v), color in dots:
        g = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2.0 * sigma ** 2))
        for ch, cval in enumerate(BGR[color]):
            img[..., ch] += peak * g * (cval / 255.0)
    return np.clip(img, 0, 255).astype(np.uint8)


class ColourMapsTest(unittest.TestCase):
    def test_pure_colour_maps(self):
        img = np.zeros((8, 8, 3), np.uint8)
        img[..., 2] = 200
        m = color_weight_maps(img)
        self.assertGreater(m["red"].max(), 150)
        self.assertEqual(m["green"].max(), 0)
        self.assertEqual(m["blue"].max(), 0)

    def test_grey_has_no_colour(self):
        img = np.full((8, 8, 3), 120, np.uint8)
        m = color_weight_maps(img)
        for c in COLOR_ORDER:
            self.assertEqual(m[c].max(), 0)


class DetectTest(unittest.TestCase):
    def test_detects_fronto_parallel_triangle(self):
        pts, _ = _project(np.eye(3), np.array([0.0, 0.0, DEPTH_MM]))
        det = ClassicV2Detector()
        res = det.detect(_render(pts), FX, FY, CX, CY)
        self.assertTrue(res.ok, res.reason)
        for c in COLOR_ORDER:
            self.assertLess(np.hypot(res.points[c][0] - pts[c][0],
                                     res.points[c][1] - pts[c][1]), 1.5, c)

    def test_detects_rotated_target(self):
        th = np.deg2rad(35)
        R = np.array([[np.cos(th), -np.sin(th), 0.0],
                      [np.sin(th), np.cos(th), 0.0],
                      [0.0, 0.0, 1.0]])
        pts, _ = _project(R, np.array([0.0, 0.0, DEPTH_MM]))
        det = ClassicV2Detector()
        res = det.detect(_render(pts), FX, FY, CX, CY)
        self.assertTrue(res.ok, res.reason)
        for c in COLOR_ORDER:
            self.assertLess(np.hypot(res.points[c][0] - pts[c][0],
                                     res.points[c][1] - pts[c][1]), 2.0, c)

    def test_ignores_far_distractor_of_same_colour(self):
        pts, _ = _project(np.eye(3), np.array([0.0, 0.0, DEPTH_MM]))
        img = _render(pts, extra=[((120.0, 120.0), "red")])
        det = ClassicV2Detector()
        res = det.detect(img, FX, FY, CX, CY)
        self.assertTrue(res.ok, res.reason)
        # the chosen red blob must be the target one
        self.assertLess(np.hypot(res.points["red"][0] - pts["red"][0],
                                 res.points["red"][1] - pts["red"][1]), 2.0)

    def test_ignores_size_outlier(self):
        pts, _ = _project(np.eye(3), np.array([0.0, 0.0, DEPTH_MM]))
        # a huge red patch is a different object, not the LED
        img = _render(pts, extra=[((150.0, 330.0), "red")])
        yy, xx = np.mgrid[0:H, 0:W]
        big = (((xx - 150) ** 2 + (yy - 330) ** 2) < 22 ** 2).astype(np.float32)
        img[..., 2] = np.clip(img[..., 2].astype(np.float32) + 150.0 * big, 0, 255).astype(np.uint8)
        det = ClassicV2Detector()
        res = det.detect(img, FX, FY, CX, CY)
        self.assertTrue(res.ok, res.reason)
        self.assertLess(np.hypot(res.points["red"][0] - pts["red"][0],
                                 res.points["red"][1] - pts["red"][1]), 2.0)

    def test_empty_frame_returns_not_ok(self):
        det = ClassicV2Detector()
        res = det.detect(np.zeros((H, W, 3), np.uint8), FX, FY, CX, CY)
        self.assertFalse(res.ok)


class TemporalTest(unittest.TestCase):
    def test_tracks_moving_target(self):
        det = ClassicV2Detector()
        solver = PoseSolver()
        th = np.deg2rad(10)
        R = np.array([[np.cos(th), -np.sin(th), 0.0],
                      [np.sin(th), np.cos(th), 0.0],
                      [0.0, 0.0, 1.0]])
        pts_true = []
        for k in range(6):
            t = np.array([25.0 * k, -10.0 * k, DEPTH_MM])
            pts, _ = _project(R, t)
            pts_true.append(pts)
            res = det.detect(_render(pts), FX, FY, CX, CY, solver)
            self.assertTrue(res.ok, f"frame {k}")
        for c in COLOR_ORDER:
            self.assertLess(np.hypot(res.points[c][0] - pts_true[-1][c][0],
                                     res.points[c][1] - pts_true[-1][c][1]),
                            3.0, c)

    def test_reset_clears_state(self):
        det = ClassicV2Detector()
        solver = PoseSolver()
        pts, _ = _project(np.eye(3), np.array([0.0, 0.0, DEPTH_MM]))
        img = _render(pts)
        for _ in range(3):
            det.detect(img, FX, FY, CX, CY, solver)
        self.assertIsNotNone(det.t)
        det.reset()
        self.assertIsNone(det.t)
        self.assertIsNone(det.R)


if __name__ == "__main__":
    unittest.main()
