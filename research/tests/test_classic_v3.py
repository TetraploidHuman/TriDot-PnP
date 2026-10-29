"""Unit tests for the classic-CV v3 front-end (contrast + scale adaptation)."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from research.baseline_b0.pnp import PoseSolver
from research.tests.test_classic_v2 import (
    BGR, CX, CY, DEPTH_MM, EDGE_MM, FX, FY, H, W, _object_points, _project, _render,
)
from research.tridot_temporal import degrade
from research.tridot_temporal.classic_v2 import ClassicV2Detector
from research.tridot_temporal.classic_v3 import ClassicV3Detector, _upsampled_centroid


def _errors(det, img, gt_uv):
    d = det.detect(img, FX, FY, CX, CY, PoseSolver())
    if not getattr(d, "ok", False):
        return None
    return {c: float(np.hypot(d.points[c][0] - gt_uv[c][0],
                              d.points[c][1] - gt_uv[c][1]))
            for c in ("red", "green", "blue")}


class ContrastFrontEndTest(unittest.TestCase):
    def setUp(self):
        R = np.eye(3)
        t = np.array([0.0, 0.0, DEPTH_MM])
        self.uv, self.cam = _project(R, t)

    def test_clean_frame_is_accurate(self):
        img = _render(self.uv)
        errs = _errors(ClassicV3Detector(), img, self.uv)
        self.assertIsNotNone(errs)
        self.assertLess(max(errs.values()), 1.5)

    def test_glare_gradient_does_not_blind_the_detector(self):
        # a bright, low-frequency background must not raise the threshold
        img = _render(self.uv).astype(np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        glare = 120.0 * (0.35 + 0.65 * (xx / W))      # warm-ish gradient
        img[..., 2] += glare                          # push the red channel up
        img[..., 1] += glare * 0.8
        img[..., 0] += glare * 0.6
        det = ClassicV3Detector()
        errs = _errors(det, np.clip(img, 0, 255).astype(np.uint8), self.uv)
        self.assertIsNotNone(errs)
        self.assertLess(max(errs.values()), 2.0, msg=str(errs))

    def test_scale_adaptive_kernel_follows_the_predicted_edge(self):
        det = ClassicV3Detector()
        img = _render(self.uv)
        det.detect(img, FX, FY, CX, CY, PoseSolver())
        self.assertIsNotNone(det.edge_px)
        self.assertAlmostEqual(det.expected_radius_px(), 0.09 * det.edge_px,
                               places=5)
        # kernel should stay inside the supported range
        for k in det._bg_ksizes():
            self.assertGreaterEqual(k, 3)
            self.assertLessEqual(k, det.max_bg_ksize)


class ColourFallbackTest(unittest.TestCase):
    """D: a dot whose colour signature was erased must still be recovered."""

    def test_desaturated_dot_is_recovered(self):
        R = np.eye(3)
        t = np.array([0.0, 0.0, DEPTH_MM])
        uv, _ = _project(R, t)
        # render red+green only; the blue dot is a white blob (chroma gone)
        img = _render({"red": uv["red"], "green": uv["green"],
                       "blue": (W + 50.0, H + 50.0)}).astype(np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        ub, vb = uv["blue"]
        g = np.exp(-((xx - ub) ** 2 + (yy - vb) ** 2) / (2.0 * 2.2 ** 2))
        for c in range(3):
            img[..., c] += 200.0 * g
        img = np.clip(img, 0, 255).astype(np.uint8)
        from research.tridot_temporal.classic_v2 import ClassicV2Detector
        self.assertFalse(ClassicV2Detector().detect(img, FX, FY, CX, CY,
                                                    PoseSolver()).ok)
        self.assertFalse(ClassicV3Detector(colorless=False)
                         .detect(img, FX, FY, CX, CY, PoseSolver()).ok)
        d = ClassicV3Detector().detect(img, FX, FY, CX, CY, PoseSolver())
        self.assertTrue(d.ok)
        self.assertLess(abs(d.points["blue"][0] - ub), 2.0)


class DegradedFarDotTest(unittest.TestCase):
    """The whole point of v3: tiny dots survive a phone-like pipeline."""

    def setUp(self):
        R = np.eye(3)
        # ~5 m with a 60 deg-ish FOV: edge ~22 px, dot radius ~1.5-2 px
        t = np.array([40.0, 20.0, 5000.0])
        self.uv, _ = _project(R, t)

    def _degraded(self):
        det = degrade.preset("phone_far")
        img = _render(self.uv, sigma=1.1, peak=250.0)
        return degrade.apply(img, det, seed=0)

    def test_tiny_degraded_dots_are_found(self):
        img = self._degraded()
        errs = _errors(ClassicV3Detector(contrast_floor=10.0), img, self.uv)
        self.assertIsNotNone(errs, "v3 found no triple on degraded far dots")
        self.assertLess(max(errs.values()), 3.0, msg=str(errs))

    def test_roi_upsample_refines_one_pixel_blobs(self):
        # a single-pixel dot: integer centroid is off by up to 0.5 px, the
        # x4 upsample must get closer than that
        tru_u, tru_v = 200.37, 150.62
        m = np.zeros((H, W), np.uint8)
        m[int(tru_v), int(tru_u)] = 255
        m[int(tru_v) + 1, int(tru_u)] = 180
        m[int(tru_v), int(tru_u) + 1] = 140
        u, v = _upsampled_centroid(None, m, int(tru_u), int(tru_v),
                                   radius=3.0, factor=4)
        self.assertLess(abs(u - tru_u), 0.5)
        self.assertLess(abs(v - tru_v), 0.5)

    def test_extract_returns_small_dot_candidates(self):
        from research.tridot_temporal.classic_v3 import _upsampled_centroid  # noqa: F401
        img = _render(self.uv, sigma=0.9, peak=255.0)
        det = ClassicV3Detector(roi_upsample=True)
        maps = __import__("research.tridot_temporal.classic_v2",
                          fromlist=["color_weight_maps"]).color_weight_maps(img)
        cands = det._extract(maps["red"], "red")
        self.assertTrue(cands)
        self.assertLess(abs(cands[0].u - self.uv["red"][0]), 3.0)


class FrontendVariantTest(unittest.TestCase):
    """The extra front-end switches added for the far/dim-dot work."""

    def setUp(self):
        self.uv = {"red": (300.0, 250.0), "green": (350.0, 250.0),
                   "blue": (325.0, 293.0)}
        self.img = _render(self.uv, sigma=2.0, peak=250.0)

    def test_raw_response_mode_matches_v2(self):
        from research.tridot_temporal.classic_v2 import ClassicV2Detector
        v2 = ClassicV2Detector()
        v3 = ClassicV3Detector(response_mode="raw")
        bgr = self.img
        d2 = v2.detect(bgr, 800.0, 800.0, 320.0, 240.0)
        d3 = v3.detect(bgr, 800.0, 800.0, 320.0, 240.0)
        self.assertEqual(d2.ok, d3.ok)
        for c in ("red", "green", "blue"):
            self.assertLess(abs(d2.points[c][0] - d3.points[c][0]), 1.5)
            self.assertLess(abs(d2.points[c][1] - d3.points[c][1]), 1.5)

    def test_union_frontend_is_superset_of_contrast(self):
        base = ClassicV3Detector(frontend="contrast").detect(
            self.img, 800.0, 800.0, 320.0, 240.0)
        union = ClassicV3Detector(frontend="union").detect(
            self.img, 800.0, 800.0, 320.0, 240.0)
        for c in ("red", "green", "blue"):
            self.assertGreaterEqual(len(union.candidates.get(c) or []),
                                    len(base.candidates.get(c) or []))

    def test_rel_frac_does_not_break_clean_dots(self):
        det = ClassicV3Detector(contrast_floor=12.0, rel_frac=0.18)
        res = det.detect(self.img, 800.0, 800.0, 320.0, 240.0)
        self.assertTrue(res.ok)
        self.assertLess(abs(res.points["red"][0] - self.uv["red"][0]), 2.0)


if __name__ == "__main__":
    unittest.main()


class ThresholdModeTest(unittest.TestCase):
    """Round 6: alternate threshold modes are documented negative results.

    On real degraded frames the effective threshold sits on the constant
    ``contrast_floor`` (measured MAD = 0, p99.5 * rel_frac < floor), so
    ``tile`` and ``tile_rel`` must be *behaviourally identical* to
    ``percentile`` -- if that ever stops being true the report's §4.3 needs a
    re-check.
    """

    def setUp(self):
        R = np.eye(3)
        t = np.array([0.0, 0.0, DEPTH_MM])
        self.uv, self.cam = _project(R, t)
        self.img = _render(self.uv)

    def test_tile_matches_percentile_when_floor_dominates(self):
        base = ClassicV3Detector(threshold_mode="percentile")
        tile = ClassicV3Detector(threshold_mode="tile")
        a = _errors(base, self.img, self.uv)
        b = _errors(tile, self.img, self.uv)
        self.assertIsNotNone(a)
        self.assertEqual(a.keys(), b.keys())
        for c in a:
            self.assertAlmostEqual(a[c], b[c], places=4)

    def test_tile_noise_sigma_is_zero_on_flat_response(self):
        det = ClassicV3Detector()
        flat = np.zeros((128, 128), dtype=np.float32)
        self.assertEqual(det._tile_noise_sigma(flat), 0.0)

    def test_tile_noise_sigma_ignores_a_bright_block(self):
        det = ClassicV3Detector(tile_px=16)
        resp = np.zeros((160, 160), dtype=np.float32)
        resp[16:48, 16:48] = 200.0        # one bright structured block
        # a quiet-tile estimator stays near zero; a global one would not
        self.assertLess(det._tile_noise_sigma(resp), 5.0)

    def test_snr_mode_is_not_the_default(self):
        self.assertEqual(ClassicV3Detector().threshold_mode, "percentile")


class PriorGateTest(unittest.TestCase):
    """Round 7: candidate-level temporal gating (the real bottleneck is
    candidate quality, per REPORT §4.3)."""

    def setUp(self):
        R = np.eye(3)
        t = np.array([0.0, 0.0, DEPTH_MM])
        self.uv, _ = _project(R, t)
        self._uv = self.uv

    def _det(self, gated):
        d = ClassicV3Detector(frontend="union", contrast_floor=12.0,
                              prior_gate=gated)
        # seed the tracker with the true pose of this frame
        d.R = np.eye(3)
        d.t = np.array([0.0, 0.0, DEPTH_MM])
        d.prev_frame = 0
        d.frame = 0
        return d

    def test_gate_prunes_distant_candidates(self):
        # a much brighter red distractor 120 px right of the true red dot
        extra = [((self.uv["red"][0] + 120.0, self.uv["red"][1]), "red")]
        img = _render(self.uv, extra=extra)
        calls = []
        orig = ClassicV2Detector._best_triple

        def spy(_self, cands, pred_uv=None, pred_edge=None, accept=2.2):
            calls.append(tuple(len(cands.get(c, []))
                               for c in ("red", "green", "blue")))
            return orig(_self, cands, pred_uv, pred_edge, accept)

        gated = self._det(True)
        with mock.patch.object(ClassicV2Detector, "_best_triple", spy):
            gated.detect(img, FX, FY, CX, CY, PoseSolver())
        self.assertTrue(calls)
        # gate = 0.55 * edge_px ≈ 63 px < 120 px, so the distractor must have
        # been pruned before at least one triple search
        self.assertLess(min(c[0] for c in calls), 2)

    def test_gated_detection_still_accurate(self):
        img = _render(self.uv)
        errs = _errors(self._det(True), img, self.uv)
        self.assertIsNotNone(errs)
        self.assertLess(max(errs.values()), 1.5)

    def test_stale_pose_falls_back_to_full_list(self):
        img = _render(self.uv)
        d = self._det(True)
        d.t = np.array([0.0, 0.0, 4.0 * DEPTH_MM])  # wildly wrong prediction
        res = d.detect(img, FX, FY, CX, CY, PoseSolver())
        self.assertTrue(res.ok)
        base = _errors(self._det(False), img, self.uv)
        self.assertIsNotNone(base)
        gated = {c: float(np.hypot(res.points[c][0] - self.uv[c][0],
                                   res.points[c][1] - self.uv[c][1]))
                 for c in ("red", "green", "blue")}
        self.assertLess(max(gated.values()), 1.5)


class Round9KnobsTest(unittest.TestCase):
    """Shape-free matched filter and circularity reward are opt-in only."""

    def setUp(self):
        self.uv = {"red": (300.0, 250.0), "green": (350.0, 250.0),
                   "blue": (325.0, 293.0)}
        self.img = _render(self.uv, sigma=2.0, peak=250.0)

    def test_defaults_keep_previous_behaviour(self):
        det = ClassicV3Detector()
        self.assertEqual(det.frontend, "union")
        self.assertEqual(det.circ_w, 0.0)

    def test_union_mf_still_solves_a_clean_triangle(self):
        det = ClassicV3Detector(frontend="union_mf")
        res = det.detect(self.img, 800.0, 800.0, 320.0, 240.0)
        self.assertTrue(res.ok)
        for c in ("red", "green", "blue"):
            self.assertLess(abs(res.points[c][0] - self.uv[c][0]), 2.0)
            self.assertLess(abs(res.points[c][1] - self.uv[c][1]), 2.0)

    def test_circ_w_only_shifts_the_triple_cost(self):
        det0 = ClassicV3Detector()
        det1 = ClassicV3Detector(circ_w=0.4)
        maps = det0.candidate_maps(self.img)
        cands = det0.candidates_from_maps(maps, self.img)
        tri = {c: cands[c][0] for c in ("red", "green", "blue")
               if cands.get(c)}
        if len(tri) < 3:
            self.skipTest("no candidates in the synthetic frame")
        c0 = det0._score_triple(tri, None, None)
        c1 = det1._score_triple(tri, None, None)
        circ = (tri["red"].circularity + tri["green"].circularity
                + tri["blue"].circularity) / 3.0
        self.assertAlmostEqual(c1, c0 - 0.4 * circ, places=6)
