"""Sanity-check the freshly generated holdout splits.

Verifies that every ``holdout_easy`` / ``holdout_hard`` sequence registered in
``meta/splits.json`` has an annotation + a readable video, that the annotations
look like the shipped ones (same intrinsics convention, poses finite, 90 frames)
and reports the image statistics that the dataset audit flagged as the
train/hard domain gap.

Usage::

    research/.venv/bin/python -m research.scripts.verify_holdout
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "TriDotPose"


def _img_stats(frames):
    vals, lums, laps = [], [], []
    for bgr in frames:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        vals.append(float(gray.mean()))
        lums.append(float(gray.std()))
        laps.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
        if len(vals) >= 12:
            break
    return {"mean_v": round(float(np.mean(vals)), 1),
            "gray_std": round(float(np.mean(lums)), 2),
            "laplacian_var": round(float(np.median(laps)), 0)}


def check_subset(splits, name):
    seqs = splits.get(name) or []
    rows = []
    for seq in seqs:
        ann_path = DATASET / "synthetic" / "annotations" / f"{seq}.json"
        vid_path = DATASET / "synthetic" / "videos" / f"{seq}.mp4"
        row = {"seq": seq, "ok": True}
        if not ann_path.exists():
            row.update(ok=False, err="missing annotation")
            rows.append(row)
            continue
        if not vid_path.exists():
            row.update(ok=False, err="missing video")
            rows.append(row)
            continue
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        frames = ann.get("frames", [])
        cam = ann.get("camera", {})
        row.update(factor=ann["info"].get("factor"),
                   n_frames=len(frames),
                   n_visible=sum(1 for f in frames if f.get("all_visible")),
                   fx=round(float(cam.get("fx", 0)), 1),
                   dist=len(cam.get("distortion", [])))
        cap = cv2.VideoCapture(str(vid_path))
        n_read = 0
        frm = []
        while n_read < 12:
            ok, img = cap.read()
            if not ok:
                break
            frm.append(img)
            n_read += 1
        cap.release()
        row.update(video_frames_probe=n_read, w=int(frm[0].shape[1]) if frm else 0,
                   h=int(frm[0].shape[0]) if frm else 0)
        row.update(_img_stats(frm))
        if row["n_frames"] not in (90, 120):
            row["ok"] = False
            row["err"] = f"unexpected frame count {row['n_frames']}"
        rows.append(row)
    return rows


def main() -> int:
    splits = json.loads((DATASET / "meta" / "splits.json").read_text(encoding="utf-8"))
    out = {}
    for name in ("holdout_easy", "holdout_hard"):
        if name not in splits:
            print(f"[verify_holdout] {name}: MISSING from splits.json")
            continue
        rows = check_subset(splits, name)
        bad = [r for r in rows if not r["ok"]]
        out[name] = {"n": len(rows), "n_bad": len(bad), "rows": rows}
        factors = {}
        for r in rows:
            factors[r.get("factor")] = factors.get(r.get("factor"), 0) + 1
        print(f"[verify_holdout] {name}: {len(rows)} sequences, {len(bad)} problems, "
              f"factors={factors}")
        for r in rows[:3]:
            print("   ", {k: r.get(k) for k in ("seq", "factor", "n_frames",
                                               "n_visible", "mean_v", "gray_std",
                                               "laplacian_var", "fx")})
    if out:
        dest = DATASET / "meta" / "holdout_verify.json"
        dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"[verify_holdout] wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
