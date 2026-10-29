"""Tests for temporal completion of occluded triangle corners."""

from __future__ import annotations

import unittest

import numpy as np

from research.tridot_temporal.app_pnp import object_points_mm
from research.tridot_temporal.completion import TemporalCompleter

CAM = {"fx": 1108.5, "fy": 1108.5, "cx": 639.5, "cy": 359.5}
W, H = 1280, 720
EDGE = 100.0


def project(rot: np.ndarray, t: np.ndarray, edge: float = EDGE):
    obj = object_points_mm(edge)
    cam = obj @ rot.T + np.asarray(t, dtype=np.float64)
    u = CAM["fx"] * cam[:, 0] / cam[:, 2] + CAM["cx"]
    v = CAM["fy"] * cam[:, 1] / cam[:, 2] + CAM["cy"]
    return {name: (float(u[i]), float(v[i]))
            for i, name in enumerate(("red", "green", "blue"))}, cam


def rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0.0, np.sin(a)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(a), 0.0, np.cos(a)]])


class StaticCompletionTest(unittest.TestCase):
    def test_fills_hidden_corner_near_truth(self):
        rot = np.eye(3)
        t = np.array([0.0, 0.0, 600.0])
        truth, _ = project(rot, t)
        comp = TemporalCompleter(max_gap=8)
        for frame in range(6):
            out = comp.complete(truth, [1.0, 1.0, 1.0], CAM, EDGE, W, H, frame)
            self.assertEqual(out.completed, [])
        hidden = dict(truth)
        del hidden["blue"]  # caller passes only observed points? no: full dict required
        hidden = dict(truth)
        for frame in range(6, 8):
            out = comp.complete(truth, [1.0, 1.0, 0.0], CAM, EDGE, W, H, frame)
            self.assertEqual(out.completed, ["blue"])
            pred = np.array(out.points["blue"])
            err = np.hypot(pred[0] - truth["blue"][0], pred[1] - truth["blue"][1])
            self.assertLess(err, 2.0, f"completion error {err:.2f}px at frame {frame}")

    def test_all_visible_is_untouched(self):
        truth, _ = project(np.eye(3), np.array([0.0, 0.0, 600.0]))
        comp = TemporalCompleter()
        out = comp.complete(truth, [1.0, 1.0, 1.0], CAM, EDGE, W, H, 0)
        self.assertEqual(out.points, truth)
        self.assertEqual(out.completed, [])
        self.assertEqual(out.reason, "observed")


class MotionTest(unittest.TestCase):
    def test_velocity_prediction_beats_static_pose(self):
        rot = np.eye(3)
        comp = TemporalCompleter(max_gap=8)
        states = []
        for frame in range(6):
            t = np.array([20.0 * frame, 0.0, 600.0])
            pts, _ = project(rot, t)
            comp.complete(pts, [1.0, 1.0, 1.0], CAM, EDGE, W, H, frame)
            states.append(t)
        frame = 7
        t_true = np.array([20.0 * frame, 0.0, 600.0])
        pts, _ = project(rot, t_true)
        out = comp.complete(pts, [1.0, 1.0, 0.0], CAM, EDGE, W, H, frame)
        self.assertEqual(out.completed, ["blue"])
        pred = np.array(out.points["blue"])

        # naive alternative: reuse the last pose verbatim (no velocity)
        naive_obj = object_points_mm(EDGE)[2]
        naive_cam = naive_obj + states[-1]
        nu = CAM["fx"] * naive_cam[0] / naive_cam[2] + CAM["cx"]
        nv = CAM["fy"] * naive_cam[1] / naive_cam[2] + CAM["cy"]
        err_pred = np.hypot(pred[0] - pts["blue"][0], pred[1] - pts["blue"][1])
        err_naive = np.hypot(nu - pts["blue"][0], nv - pts["blue"][1])
        self.assertLess(err_pred, err_naive)
        self.assertLess(err_pred, 12.0)


class GuardTest(unittest.TestCase):
    def test_gap_limit_stops_prediction(self):
        rot = np.eye(3)
        t = np.array([0.0, 0.0, 600.0])
        truth, _ = project(rot, t)
        comp = TemporalCompleter(max_gap=3)
        comp.complete(truth, [1.0, 1.0, 1.0], CAM, EDGE, W, H, 0)
        out = comp.complete(truth, [1.0, 1.0, 0.0], CAM, EDGE, W, H, 10)
        self.assertEqual(out.completed, [])
        self.assertEqual(out.reason, "stale")

    def test_no_state_means_no_completion(self):
        truth, _ = project(np.eye(3), np.array([0.0, 0.0, 600.0]))
        comp = TemporalCompleter()
        out = comp.complete(truth, [1.0, 1.0, 0.0], CAM, EDGE, W, H, 0)
        self.assertEqual(out.completed, [])
        self.assertEqual(out.reason, "no-state")

    def test_two_missing_not_completed(self):
        truth, _ = project(np.eye(3), np.array([0.0, 0.0, 600.0]))
        comp = TemporalCompleter(max_missing=1)
        comp.complete(truth, [1.0, 1.0, 1.0], CAM, EDGE, W, H, 0)
        out = comp.complete(truth, [1.0, 0.0, 0.0], CAM, EDGE, W, H, 1)
        self.assertEqual(out.completed, [])
        self.assertEqual(out.reason, "too-many-missing")

    def test_reset_clears_state(self):
        truth, _ = project(np.eye(3), np.array([0.0, 0.0, 600.0]))
        comp = TemporalCompleter()
        comp.complete(truth, [1.0, 1.0, 1.0], CAM, EDGE, W, H, 0)
        self.assertTrue(comp.initialized)
        comp.reset()
        self.assertFalse(comp.initialized)
        out = comp.complete(truth, [1.0, 1.0, 0.0], CAM, EDGE, W, H, 1)
        self.assertEqual(out.reason, "no-state")

    def test_poor_visibility_of_observed_point_is_not_promoted(self):
        """A corner the visibility head is unsure about must not update state."""
        truth, _ = project(np.eye(3), np.array([0.0, 0.0, 600.0]))
        comp = TemporalCompleter()
        comp.complete(truth, [1.0, 1.0, 0.2], CAM, EDGE, W, H, 0)
        self.assertEqual(comp.n_updates, 0)


class OrientationTest(unittest.TestCase):
    def test_rotated_target_completion(self):
        rot = rot_y(25.0)
        t = np.array([10.0, -5.0, 700.0])
        truth, cam = project(rot, t)
        comp = TemporalCompleter(max_gap=8)
        for frame in range(4):
            comp.complete(truth, [1.0, 1.0, 1.0], CAM, EDGE, W, H, frame)
        out = comp.complete(truth, [1.0, 1.0, 0.0], CAM, EDGE, W, H, 4)
        self.assertEqual(out.completed, ["blue"])
        pred = np.array(out.points["blue"])
        err = np.hypot(pred[0] - truth["blue"][0], pred[1] - truth["blue"][1])
        self.assertLess(err, 2.0)


if __name__ == "__main__":
    unittest.main()


class EvidenceHelpersTest(unittest.TestCase):
    """eval_v2 的「颜色证据」触发器：只测纯函数，不跑网络。"""

    def _image(self, draw: bool):
        import cv2
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        if draw:
            # BGR 里的红点
            cv2.circle(img, (32, 32), 5, (0, 0, 255), -1)
        return img

    def test_evidence_high_on_dot_low_off_dot(self):
        from research.tridot_temporal import eval_v2
        maps = eval_v2._color_weight_maps(self._image(draw=True))
        self.assertGreater(eval_v2._evidence_at(maps["red"], (32, 32)), 12.0)
        self.assertLess(eval_v2._evidence_at(maps["red"], (8, 8)), 1.0)

    def test_evidence_low_when_dot_masked(self):
        from research.tridot_temporal import eval_v2
        maps = eval_v2._color_weight_maps(self._image(draw=False))
        self.assertLess(eval_v2._evidence_at(maps["red"], (32, 32)), 1.0)

    def test_evidence_clips_at_image_border(self):
        from research.tridot_temporal import eval_v2
        maps = eval_v2._color_weight_maps(self._image(draw=True))
        # 靠近边界也不应抛异常
        self.assertGreaterEqual(eval_v2._evidence_at(maps["blue"], (0, 0)), 0.0)


if __name__ == "__main__":
    unittest.main()
