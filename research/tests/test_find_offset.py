"""Unit tests for the offset-regression head used by ``find_offset``.

The head is trained to predict, for each colour channel, the sub-cell position
of the LED inside its heat-map cell.  These tests pin down the conventions that
train and eval share:

* ``coords * (size - 1) = cell + frac`` with ``cell = floor(...)``
* a zero offset (sigmoid(0) = 0.5) decodes to the *cell centre*
* ``offset_loss`` is zero when the head predicts the GT fraction exactly and
  grows with the fraction error
* ``decode_offset`` picks the peak of the right colour channel and returns
  coordinates in the ``[0, 1]`` normalised frame used everywhere else
"""

from __future__ import annotations

import unittest

import torch

from research.tridot_temporal.find_offset import (
    _cell_and_frac,
    _sample_offset,
    decode_offset,
    offset_loss,
)


def _gt(b: int = 2, t: int = 2):
    torch.manual_seed(0)
    return torch.rand(b, t, 3, 2)


def _perfect_off(gt: torch.Tensor, h: int, w: int, fill: float = 0.5) -> torch.Tensor:
    """Offset map that carries exactly the GT fraction at the GT cell."""
    b, t, _, _ = gt.shape
    cx, cy, fx, fy = _cell_and_frac(gt, h, w)
    off = torch.full((b, t, 3, 2, h, w), fill)
    for bi in range(b):
        for ti in range(t):
            for c in range(3):
                off[bi, ti, c, 0, cy[bi, ti, c], cx[bi, ti, c]] = fx[bi, ti, c]
                off[bi, ti, c, 1, cy[bi, ti, c], cx[bi, ti, c]] = fy[bi, ti, c]
    return off


class CellAndFracTest(unittest.TestCase):
    def test_decomposition(self):
        coords = torch.tensor([[[0.25, 0.0], [0.75, 0.5], [1.0, 1.0]]])[:, None]
        cx, cy, fx, fy = _cell_and_frac(coords, h=5, w=9)
        self.assertEqual(tuple(cx.shape), (1, 1, 3))
        # gx = 0.25 * 8 = 2.0 -> cell 2, frac 0 (x of the first LED)
        self.assertEqual(cx[0, 0, 0].item(), 2)
        self.assertAlmostEqual(fx[0, 0, 0].item(), 0.0, places=6)
        # gy = 0.5 * 4 = 2.0 (y of the second LED)
        self.assertEqual(cy[0, 0, 1].item(), 2)
        self.assertAlmostEqual(fy[0, 0, 1].item(), 0.0, places=6)

    def test_frac_in_unit_interval(self):
        coords = torch.rand(4, 3, 2)
        cx, cy, fx, fy = _cell_and_frac(coords, h=64, w=112)
        self.assertTrue(bool((fx >= 0).all() and (fx < 1).all()))
        self.assertTrue(bool((fy >= 0).all() and (fy < 1).all()))
        self.assertTrue(bool((cx >= 0).all() and (cx <= 111).all()))
        self.assertTrue(bool((cy >= 0).all() and (cy <= 63).all()))


class SampleOffsetTest(unittest.TestCase):
    def test_gathers_the_gt_cell(self):
        h, w = 4, 6
        off = torch.arange(2 * 1 * 3 * 2 * h * w, dtype=torch.float32)
        off = off.view(2, 1, 3, 2, h, w)
        cx = torch.tensor([[[0, 1, 2]], [[3, 4, 5]]])
        cy = torch.tensor([[[1, 2, 3]], [[0, 1, 2]]])
        out = _sample_offset(off, cx, cy)
        self.assertEqual(tuple(out.shape), (2, 1, 3, 2))
        for bi in range(2):
            for ci in range(3):
                self.assertEqual(
                    float(out[bi, 0, ci, 0]),
                    float(off[bi, 0, ci, 0, cy[bi, 0, ci], cx[bi, 0, ci]]))


class OffsetLossTest(unittest.TestCase):
    def test_zero_at_gt_fraction(self):
        h, w = 4, 6
        gt = _gt()
        off = _perfect_off(gt, h, w)
        vis = torch.ones(2, 2, 3)
        self.assertAlmostEqual(offset_loss(off, gt, vis).item(), 0.0, places=6)

    def test_grows_with_error(self):
        h, w = 4, 6
        gt = _gt(1, 1)
        off = _perfect_off(gt, h, w)
        vis = torch.ones(1, 1, 3)
        base = offset_loss(off, gt, vis).item()
        off2 = (off + 0.1).clamp(0, 1)
        self.assertGreater(offset_loss(off2, gt, vis).item(), base + 0.004)

    def test_ignores_invisible(self):
        gt = _gt(1, 1)
        off = torch.full((1, 1, 3, 2, 4, 6), 0.9)
        vis = torch.zeros(1, 1, 3)
        self.assertAlmostEqual(offset_loss(off, gt, vis).item(), 0.0, places=6)

    def test_partial_visibility_is_supervised(self):
        h, w = 4, 6
        gt = _gt(1, 1)
        off_ok = _perfect_off(gt, h, w)
        off_bad = torch.full_like(off_ok, 0.5)
        off_bad_ok = _perfect_off(gt, h, w)
        vis = torch.full((1, 1, 3), 0.5)
        good = offset_loss(off_ok, gt, vis).item()
        # a half-visible LED is still supervised (min_vis=0.5)
        self.assertLess(good, offset_loss(off_bad, gt, vis).item())


class DecodeOffsetTest(unittest.TestCase):
    def _heat(self, h=4, w=6):
        heat = torch.full((1, 3, h, w), -5.0)
        heat[0, 0, 1, 2] = 5.0
        heat[0, 1, 2, 3] = 5.0
        heat[0, 2, 3, 4] = 5.0
        return heat

    def test_neutral_offset_lands_on_cell_centre(self):
        h, w = 4, 6
        coords = decode_offset(self._heat(h, w),
                               torch.full((1, 3, 2, h, w), 0.5))
        self.assertEqual(tuple(coords.shape), (1, 3, 2))
        self.assertAlmostEqual(coords[0, 0, 0].item(), (2 + 0.5) / (w - 1), places=5)
        self.assertAlmostEqual(coords[0, 0, 1].item(), (1 + 0.5) / (h - 1), places=5)

    def test_offset_moves_within_cell(self):
        h, w = 4, 6
        heat = self._heat(h, w)
        lo = decode_offset(heat, torch.full((1, 3, 2, h, w), 0.0))
        hi = decode_offset(heat, torch.full((1, 3, 2, h, w), 0.999))
        self.assertAlmostEqual(lo[0, 0, 0].item(), 2.0 / (w - 1), places=5)
        self.assertAlmostEqual(hi[0, 0, 0].item(), 2.999 / (w - 1), places=5)

    def test_output_in_unit_range(self):
        heat = torch.randn(2, 3, 8, 12)
        off = torch.rand(2, 3, 2, 8, 12)
        coords = decode_offset(heat, off)
        self.assertTrue(bool((coords >= 0).all() and (coords <= 1).all()))


class HeadGradientTest(unittest.TestCase):
    def test_only_offset_head_gets_gradient(self):
        from research.tridot_temporal.find_model import FindConfig, TriDotFindNet

        model = TriDotFindNet(FindConfig(offset_head=True, feat_ch=48, d_model=64,
                                   nlayers=1, peak_assoc=False))
        # emulate the fine-tuning setup: backbone frozen, only head trains
        for param in model.encoder.parameters():
            param.requires_grad_(False)
        x = torch.rand(1, 5, 7, 64, 112)
        model(x, return_heat=True)
        off = model.last_off
        self.assertIsNotNone(off)
        self.assertEqual(tuple(off.shape)[:4], (1, 5, 3, 2))
        off.square().mean().backward()
        self.assertIsNotNone(model.offset_head.weight.grad)
        self.assertGreater(float(model.offset_head.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(model.encoder.net[0].block[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
