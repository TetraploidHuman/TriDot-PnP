"""Unit tests for the realistic-capture degradation presets (plan F)."""
from __future__ import annotations

import unittest

import numpy as np

from research.tridot_temporal import degrade


def _dot_image(u=320.0, v=240.0, sigma=2.2, peak=250.0, size=(480, 640)):
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2.0 * sigma ** 2))
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 0.05 * peak
    img[..., 1] = 0.10 * peak
    img[..., 2] = peak * g + 0.15 * peak
    return np.clip(img, 0, 255).astype(np.uint8)


def _red_peak(img, u=320, v=240, r=6):
    c = img[v - r:v + r + 1, u - r:u + r + 1, 2].astype(np.int32)
    return int(c.max())


class PresetTest(unittest.TestCase):
    def test_clean_is_identity(self):
        cfg = degrade.preset("clean")
        self.assertTrue(cfg.is_identity())
        img = _dot_image()
        out = degrade.apply(img, cfg, seed=0)
        np.testing.assert_array_equal(out, img)

    def test_unknown_preset_raises(self):
        with self.assertRaises(KeyError):
            degrade.preset("nope")

    def test_phone_far_fades_the_dot(self):
        img = _dot_image()
        out = degrade.apply(img, degrade.preset("phone_far"), seed=0)
        self.assertLess(_red_peak(out), _red_peak(img))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, img.shape)


class StageTest(unittest.TestCase):
    def test_noise_is_deterministic_per_seed(self):
        img = _dot_image()
        a = degrade.apply(img, degrade.preset("phone_harsh"), seed=3)
        b = degrade.apply(img, degrade.preset("phone_harsh"), seed=3)
        np.testing.assert_array_equal(a, b)

    def test_chroma_roundtrip_keeps_luma(self):
        # grey gradient: 4:2:0 must not change the luminance plane materially
        h, w = 120, 160
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.stack([xx * 255 // w, xx * 255 // w, xx * 255 // w], axis=-1).astype(np.uint8)
        out = degrade.chroma_subsample(img, "420")
        # chroma is constant here, so the roundtrip should be (almost) lossless
        self.assertLessEqual(int(np.abs(out.astype(int) - img.astype(int)).max()), 1)

    def test_downscale_removes_detail(self):
        img = _dot_image(sigma=0.8, peak=255.0)
        out = degrade.downscale_upscale(img, 2.0)
        peak_before = _red_peak(img)
        peak_after = _red_peak(out)
        self.assertLess(peak_after, peak_before)
        self.assertEqual(out.shape, img.shape)

    def test_denoise_blends_towards_median(self):
        img = np.zeros((20, 20, 3), np.uint8)
        img[10, 10] = 255
        out = degrade.denoise(img, 1.0)
        # a single-pixel spike is exactly what median filtering removes
        self.assertEqual(int(out[10, 10].max()), 0)

    def test_jpeg_quality_100_is_identity(self):
        img = _dot_image()
        np.testing.assert_array_equal(degrade.jpeg(img, 100), img)

    def test_apply_to_frames_uses_active_profile(self):
        frames = [_dot_image(), _dot_image()]
        degrade.set_active(None)
        self.assertIs(degrade.apply_to_frames(frames)[0], frames[0])
        degrade.set_active(degrade.preset("phone_far"))
        try:
            out = degrade.apply_to_frames(frames)
            self.assertEqual(len(out), 2)
            self.assertLess(_red_peak(out[0]), _red_peak(frames[0]))
            # same frame index -> same noise (deterministic repeat)
            out2 = degrade.apply_to_frames(frames)
            np.testing.assert_array_equal(out[0], out2[0])
        finally:
            degrade.set_active(None)


if __name__ == "__main__":
    unittest.main()
