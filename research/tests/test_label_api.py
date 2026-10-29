"""End-to-end test for the recovered label tool's HTTP API (research/label_tool).

The UI (static/index.html + app.js) is the only part of that package that had to
be rewritten; the server and its export path are recovered bytecode.  This test
pins down the API contract the UI relies on, and checks that ``export`` really
writes an annotation file the training pipeline can read.

It starts the server on a spare port, uses the real smoke video, then restores
every file it touched (annotation copy, video copy, splits.json).

Run:  research/.venv/bin/python -m unittest research.tests.test_label_api -v
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "TriDotPose"
VIDEO = DATASET / "real" / "videos" / "smoke_real_desk.mp4"
SPLITS = DATASET / "meta" / "splits.json"
REAL_ANN = DATASET / "real" / "annotations" / "smoke_real_desk.json"
SYNTH_ANN = DATASET / "synthetic" / "annotations" / "smoke_real_desk.json"
SYNTH_VIDEO = DATASET / "synthetic" / "videos" / "smoke_real_desk.mp4"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _request(method: str, url: str, payload: dict | None = None, timeout: float = 20.0):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:  # pragma: no cover - surfaced as failure
        return exc.code, json.loads(exc.read() or b"null")


@unittest.skipUnless(VIDEO.is_file(), f"missing test video: {VIDEO}")
class LabelToolApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.port = _free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls._backup_dir = Path(tempfile.mkdtemp(prefix="label_backup_"))
        cls._backups = []
        for path in (SPLITS, REAL_ANN, SYNTH_ANN, SYNTH_VIDEO):
            if path.exists():
                dst = cls._backup_dir / path.name
                shutil.copy2(path, dst)
                cls._backups.append((path, dst))
        cls.proc = subprocess.Popen(
            [sys.executable, "-m", "research.label_tool", "--port", str(cls.port)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                status, body = _request("GET", f"{cls.base}/api/health")
                if status == 200 and body and body.get("ok"):
                    return
            except OSError:
                pass
            time.sleep(0.2)
        cls.proc.kill()
        raise RuntimeError("label_tool server did not become healthy")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            cls.proc.kill()
        for path, dst in cls._backups:
            shutil.copy2(dst, path)
        for path in (REAL_ANN, SYNTH_ANN, SYNTH_VIDEO):
            if not any(path == p for p, _ in cls._backups) and path.exists():
                path.unlink()
        # restore splits exactly: strip a real_train entry that this test added
        if SPLITS.is_file():
            original = next((dst for p, dst in cls._backups if p == SPLITS), None)
            if original is not None:
                shutil.copy2(original, SPLITS)
        shutil.rmtree(cls._backup_dir, ignore_errors=True)

    def _new_session(self) -> str:
        status, body = _request(
            "POST", f"{self.base}/api/session", {"video": "smoke_real_desk.mp4", "stride": 10}
        )
        self.assertEqual(status, 200)
        return body["id"]

    def test_health_and_videos(self) -> None:
        status, body = _request("GET", f"{self.base}/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertIn("find_enabled", body)

        status, body = _request("GET", f"{self.base}/api/videos")
        self.assertEqual(status, 200)
        self.assertIn("smoke_real_desk.mp4", body["videos"])

    def test_session_frames_and_actions(self) -> None:
        sid = self._new_session()
        base = f"{self.base}/api/session/{sid}"

        status, summary = _request("GET", f"{base}/summary")
        self.assertEqual(status, 200)
        n_frames = summary["n_frames"]
        self.assertGreaterEqual(n_frames, 2)
        self.assertIn("counts", summary)

        status, frame = _request("GET", f"{base}/frame/0")
        self.assertEqual(status, 200)
        self.assertEqual(frame["index"], 0)
        self.assertIn("pred", frame)
        self.assertIn("status", frame)

        # frame image endpoint the UI uses for the canvas
        with urllib.request.urlopen(f"{base}/frame/0.jpg", timeout=20) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get_content_type(), "image/jpeg")
            self.assertGreater(len(resp.read()), 1000)

        # navigation
        status, frame = _request("POST", f"{base}/action", {"action": "goto", "index": 1})
        self.assertEqual(status, 200)
        self.assertEqual(frame["index"], 1)

        # manual points must be complete
        status, err = _request(
            "POST", f"{base}/action", {"action": "set_points", "index": 1, "points": {"red": [1, 2]}}
        )
        self.assertNotEqual(status, 200)
        self.assertIn("error", err or {})

        pts = {"red": [100.0, 120.0], "green": [220.0, 140.0], "blue": [160.0, 240.0]}
        status, frame = _request(
            "POST", f"{base}/action", {"action": "set_points", "index": 1, "points": pts}
        )
        self.assertEqual(status, 200)
        self.assertTrue(frame["manual"])
        for color in ("red", "green", "blue"):
            self.assertAlmostEqual(frame["manual"][color][0], pts[color][0], places=2)

        status, frame = _request("POST", f"{base}/action", {"action": "accept", "index": 1})
        self.assertEqual(status, 200)
        self.assertEqual(frame["status"], "accepted")

        status, frame = _request("POST", f"{base}/action", {"action": "reject", "index": 2})
        self.assertEqual(status, 200)
        self.assertEqual(frame["status"], "rejected")

        status, err = _request("POST", f"{base}/action", {"action": "nope", "index": 0})
        self.assertNotEqual(status, 200)

    def test_export_writes_trainable_annotation(self) -> None:
        sid = self._new_session()
        base = f"{self.base}/api/session/{sid}"
        status, frame = _request("GET", f"{base}/frame/0")
        self.assertTrue(frame["pred"]["ok"], "B0 should detect the smoke video")
        status, _ = _request("POST", f"{base}/action", {"action": "accept", "index": 0})
        self.assertEqual(status, 200)

        status, res = _request("POST", f"{base}/export")
        self.assertEqual(status, 200)
        self.assertEqual(res["n_accepted"], 1)
        self.assertEqual(res["split"], "real_train")

        sparse = Path(res["annotation_sparse"])
        compact = Path(res["annotation_train"])
        self.assertTrue(sparse.is_file())
        self.assertTrue(compact.is_file())

        data = json.loads(compact.read_text())
        self.assertEqual(data["info"]["source"], "real")
        self.assertEqual(data["info"]["seq_id"], "smoke_real_desk")
        self.assertEqual(len(data["frames"]), 1)
        kp = data["frames"][0]["keypoints"]
        self.assertEqual(len(kp), 9)
        self.assertTrue(data["frames"][0]["all_visible"])
        self.assertEqual(len(data["camera"]["distortion"]), 5)

        splits = json.loads(SPLITS.read_text())
        self.assertIn("smoke_real_desk", splits.get("real_train", []))


if __name__ == "__main__":
    unittest.main()
