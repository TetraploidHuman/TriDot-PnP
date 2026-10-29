"""Unit tests for the dot-vs-clutter patch classifier and its v3 hook."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from research.tridot_temporal.classic_v2 import Blob
from research.tridot_temporal.classic_v3 import ClassicV3Detector
from research.tridot_temporal.dot_cls import (DotClassifier, DotScorer, patch_from_map,
                                              patch_from_map_scaled)


class _StubScorer:
    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=np.float32)

    def predict(self, patches):
        n = len(patches)
        out = np.zeros(n, dtype=np.float32)
        out[: min(n, len(self._probs))] = self._probs[: min(n, len(self._probs))]
        return out


class PatchTest(unittest.TestCase):
    def test_patch_shape_and_scale(self):
        m = np.full((40, 40), 255, dtype=np.uint8)
        p = patch_from_map(m, 20.0, 20.0)
        self.assertEqual(p.shape, (15, 15))
        self.assertAlmostEqual(float(p.mean()), 1.0, places=5)

    def test_patch_pads_at_border(self):
        m = np.full((40, 40), 200, dtype=np.uint8)
        p = patch_from_map(m, 0.0, 0.0)
        self.assertEqual(p.shape, (15, 15))
        # only the bottom-right part of the patch is inside the image
        self.assertEqual(float(p[:7, :7].sum()), 0.0)
        self.assertAlmostEqual(float(p[7:, 7:].mean()), 200.0 / 255.0, places=5)


class ScaledPatchTest(unittest.TestCase):
    def test_scaled_shape_and_uniform_content(self):
        m = np.full((60, 60), 120, dtype=np.uint8)
        p = patch_from_map_scaled(m, 30.0, 30.0, expected_radius_px=6.0)
        self.assertEqual(p.shape, (15, 15))
        self.assertAlmostEqual(float(p.mean()), 120.0 / 255.0, places=5)

    def test_scaled_falls_back_for_tiny_radius(self):
        m = np.arange(900, dtype=np.uint8).reshape(30, 30)
        a = patch_from_map_scaled(m, 15.0, 15.0, expected_radius_px=0.0)
        b = patch_from_map(m, 15.0, 15.0)
        np.testing.assert_allclose(a, b)

    def test_scaled_smaller_window_for_tiny_dot(self):
        m = np.zeros((80, 80), dtype=np.uint8)
        m[39:41, 39:41] = 255  # 2x2 dot
        tiny = patch_from_map_scaled(m, 39.5, 39.5, expected_radius_px=1.0)
        big = patch_from_map_scaled(m, 39.5, 39.5, expected_radius_px=9.0)
        # a 2px dot fills a much larger fraction of the canonical patch
        self.assertGreater(float((tiny > 0.3).mean()), float((big > 0.3).mean()))


class ModelTest(unittest.TestCase):
    def test_forward_shape_and_determinism(self):
        model = DotClassifier(in_ch=1).eval()
        x = torch.rand(4, 1, 15, 15)
        with torch.no_grad():
            a = model(x)
            b = model(x)
        self.assertEqual(tuple(a.shape), (4,))
        self.assertTrue(torch.allclose(a, b))

    def test_scorer_roundtrip(self):
        model = DotClassifier(in_ch=1)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.pt"
            torch.save({"state_dict": model.state_dict(), "in_ch": 1,
                        "threshold": 0.5, "meta": {"toy": True}}, path)
            scorer = DotScorer(path)
            probs = scorer.predict(np.random.rand(3, 15, 15).astype(np.float32))
        self.assertEqual(probs.shape, (3,))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))


class RerankTest(unittest.TestCase):
    def _blobs(self):
        return [Blob(color="red", u=10.0 * i, v=5.0, area=2, peak=50.0,
                     score=0.9 - 0.1 * i, aspect=1.0, circularity=1.0)
                for i in range(3)]

    def test_rerank_mixes_probability_into_score(self):
        det = ClassicV3Detector()
        kept = self._blobs()
        det._dot_scorer = _StubScorer([0.1, 0.9, 0.4])
        out = det._rerank_with_dot_cls(np.zeros((20, 40), np.uint8), kept)
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out[0].score, 0.5 * 0.8 + 0.5 * 0.9, places=6)
        self.assertAlmostEqual(out[1].score, 0.5 * 0.7 + 0.5 * 0.4, places=6)
        # sorted by the blended score, best (0.85) first
        self.assertGreaterEqual(out[0].score, out[1].score)
        self.assertGreaterEqual(out[1].score, out[2].score)

    def test_floor_drops_candidates(self):
        det = ClassicV3Detector(dot_cls_floor=0.5)
        det._dot_scorer = _StubScorer([0.1, 0.9, 0.4])
        out = det._rerank_with_dot_cls(np.zeros((20, 40), np.uint8), self._blobs())
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].score, 0.5 * 0.8 + 0.5 * 0.9, places=6)

    def test_scaled_mode_uses_expected_radius(self):
        det = ClassicV3Detector(dot_cls_scale=True)
        det._dot_scorer = _StubScorer([0.2, 0.8, 0.5])
        out = det._rerank_with_dot_cls(np.full((40, 60), 90, np.uint8), self._blobs())
        self.assertEqual(len(out), 3)
        self.assertAlmostEqual(out[0].score, 0.5 * 0.8 + 0.5 * 0.8, places=6)

    def test_no_scorer_returns_input_order(self):
        det = ClassicV3Detector()
        kept = self._blobs()
        out = det._rerank_with_dot_cls(np.zeros((20, 40), np.uint8), kept)
        self.assertEqual([b.u for b in out], [b.u for b in kept])


if __name__ == "__main__":
    unittest.main()
