"""Unit tests for the pre-registered real-device acceptance table.

The thresholds come from ``research/docs/real_device_protocol.md`` and are
deliberately hard-coded in the script *before* any real data exists; these tests
pin that contract so a future edit cannot quietly loosen it.
"""
import unittest

from research.scripts import real_eval_acceptance as ra


def row(band, det10, rel=None, t=None):
    return {"band": band, "det10": det10, "dist_rel_err": rel, "t_err_mm": t}


class BandTest(unittest.TestCase):
    def test_distance_bands(self):
        self.assertEqual(ra._band(1000.0), "1-5 m")
        self.assertEqual(ra._band(4999.0), "1-5 m")
        self.assertEqual(ra._band(5000.0), "5-8 m")
        self.assertEqual(ra._band(12000.0), ">=8 m")


class AcceptanceTableTest(unittest.TestCase):
    def test_good_capture_passes_every_band(self):
        rows = [row("1-5 m", 1, 0.02, 10.0) for _ in range(20)] + \
               [row("5-8 m", 1, 0.08, 40.0) for _ in range(10)] + \
               [row(">=8 m", 0, None, None) for _ in range(5)]
        table = ra._table(rows)
        verdicts = {r["band"]: r["verdict"] for r in table}
        self.assertEqual(verdicts["1-5 m"], "PASS")
        self.assertEqual(verdicts["5-8 m"], "PASS")
        self.assertEqual(verdicts[">=8 m"], "PASS")  # recorded, not gated

    def test_detection_below_threshold_fails(self):
        rows = [row("1-5 m", 0, None, None) for _ in range(10)]
        self.assertEqual(ra._table(rows)[0]["verdict"], "FAIL")

    def test_distance_error_above_threshold_fails(self):
        rows = [row("1-5 m", 1, 0.10, 10.0) for _ in range(10)]
        self.assertEqual(ra._table(rows)[0]["verdict"], "FAIL")

    def test_any_bad_distance_at_mid_range_fails(self):
        rows = [row("5-8 m", 1, 0.05, 40.0) for _ in range(9)] + [row("5-8 m", 1, 0.35, 90.0)]
        self.assertEqual(ra._table(rows)[1]["verdict"], "FAIL")

    def test_no_data_band_is_not_a_pass(self):
        table = ra._table([row("1-5 m", 1, 0.01, 5.0)])
        self.assertEqual(table[1]["verdict"], "NO DATA")


if __name__ == "__main__":
    unittest.main()


class DryRunBandTest(unittest.TestCase):
    """dry-run 支持：<1 m 不入判定、subset 路径切换。"""

    def test_band_boundaries(self):
        from research.scripts.real_eval_acceptance import _band
        self.assertEqual(_band(500.0), "<1 m (ignored)")
        self.assertEqual(_band(1200.0), "1-5 m")
        self.assertEqual(_band(6500.0), "5-8 m")
        self.assertEqual(_band(9000.0), ">=8 m")

    def test_ignored_band_has_no_verdict(self):
        from research.scripts.real_eval_acceptance import _table
        rows = [dict(seq="s", frame=0, band="<1 m (ignored)", det10=1,
                     dist_rel_err=0.01, t_err_mm=5.0, reason="")]
        table = _table(rows)
        ignored = [t for t in table if t["band"].startswith("<1 m")][0]
        self.assertEqual(ignored["verdict"], "IGNORED")
        self.assertIsNone(ignored["bad_distance"])

    def test_synthetic_pair_paths(self):
        from research.scripts.real_eval_acceptance import _seq_paths
        from pathlib import Path
        root = Path("datasets/TriDotPose")
        got = _seq_paths(root, "syn_hold_hard_0001", "holdout_hard")
        self.assertIsNotNone(got)
        self.assertIn("synthetic", str(got[0]))
