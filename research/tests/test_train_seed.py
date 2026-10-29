"""Tests for training reproducibility helpers in train_find.py."""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from research.tridot_temporal.train_find import (  # noqa: E402
    _dataset_fingerprint,
    _seed_everything,
    _seed_worker,
)


class SeedingTests(unittest.TestCase):
    def test_seed_everything_repeats_numpy_and_random(self):
        _seed_everything(11)
        first = (np.random.rand(4).tolist(), random.random(), random.getrandbits(32))
        _seed_everything(11)
        second = (np.random.rand(4).tolist(), random.random(), random.getrandbits(32))
        self.assertEqual(first, second)

    def test_seed_worker_repeats(self):
        _seed_worker(0)
        first = np.random.rand(3).tolist()
        _seed_worker(0)
        self.assertEqual(first, np.random.rand(3).tolist())

    def test_worker_seeds_differ(self):
        _seed_worker(0)
        a = np.random.rand(2).tolist()
        _seed_worker(1)
        b = np.random.rand(2).tolist()
        self.assertNotEqual(a, b)


class FingerprintTests(unittest.TestCase):
    def test_missing_files_are_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = _dataset_fingerprint(Path(tmp))
            self.assertTrue(fp)
            self.assertTrue(all(v["sha256"] is None for v in fp.values()))

    def test_hash_changes_with_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            (root / "synthetic").mkdir()
            (root / "meta" / "splits.json").write_text("{}", encoding="utf-8")
            fp1 = _dataset_fingerprint(root)
            (root / "meta" / "splits.json").write_text('{"a": 1}', encoding="utf-8")
            fp2 = _dataset_fingerprint(root)
            self.assertNotEqual(fp1["meta/splits.json"]["sha256"],
                                fp2["meta/splits.json"]["sha256"])


if __name__ == "__main__":
    unittest.main()
