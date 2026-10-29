"""Round 15: observable-frame denominator.

`det10_all` charges fully hidden dots to the detector; `det10_legacy` drops every
partially occluded frame.  `det10_observable` keeps frames whose three dots are
at least annotated as present (vis >= 1) -- the number to quote on hard factors.
"""
import unittest

from research.tridot_temporal import metrics
from research.tridot_temporal.metrics import FrameRecord


def rec(vis, pred_err=0.0, seq="s", frame=0):
    gt = [[10.0, 10.0, vis[0]], [30.0, 10.0, vis[1]], [20.0, 40.0, vis[2]]]
    pred = [[10.0 + pred_err, 10.0], [30.0 + pred_err, 10.0], [20.0 + pred_err, 40.0]]
    return FrameRecord(method="m", seq_id=seq, factor="f", frame_index=frame,
                       gt=gt, pred=pred)


class ObservableMetricTest(unittest.TestCase):
    def test_hidden_dot_excluded_from_denominator(self):
        rows = [rec([2, 2, 2]), rec([2, 2, 0])]
        self.assertEqual(metrics.det10_observable(rows), 1.0)
        self.assertEqual(metrics.observable_frames(rows), 1)
        # all-frame denominator still charges the hidden frame
        self.assertAlmostEqual(metrics.det10_all(rows), 0.5)

    def test_partial_occlusion_counts_when_dots_present(self):
        rows = [rec([1, 1, 1]), rec([1, 1, 1], pred_err=20.0)]
        self.assertEqual(metrics.observable_frames(rows), 2)
        self.assertAlmostEqual(metrics.det10_observable(rows), 0.5)
        # legacy only looks at fully visible frames -> None here
        self.assertIsNone(metrics.det10_legacy(rows))

    def test_summary_carries_observable_fields(self):
        rows = [rec([2, 2, 2]), rec([1, 2, 2]), rec([0, 2, 2])]
        sm = metrics.summarize(rows, n_boot=20)
        self.assertIn("det10_observable", sm["det"])
        self.assertEqual(sm["det"]["n_observable_frames"], 2)
        text = metrics.format_report(sm)
        self.assertIn("observable", text)


if __name__ == "__main__":
    unittest.main()
