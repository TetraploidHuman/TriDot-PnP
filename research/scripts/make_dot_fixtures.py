"""Export a handful of real candidate patches + reference probabilities.

The Kotlin forward pass is checked against these fixtures (bit-level parity is
impossible with float32 rounding, but 1e-3 absolute is plenty for a sigmoid
score), and ONNX Runtime gives a second independent reference.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from research.tridot_temporal import degrade
from research.tridot_temporal import eval_v2 as ev
from research.tridot_temporal.classic_v3 import ClassicV3Detector
from research.tridot_temporal.dot_cls import DotClassifier, patch_from_map_scaled

COLORS = ("red", "green", "blue")
PRESETS = ("clean", "phone_far")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("datasets/TriDotPose"))
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    ckpt = torch.load(str(a.ckpt), map_location="cpu", weights_only=False)
    model = DotClassifier(in_ch=1)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    pairs = ev._discover_pairs(a.dataset, "synthetic", subset="hard_test")
    rng.shuffle(pairs)
    det = ClassicV3Detector(frontend="union", contrast_floor=12.0)
    samples = []
    for video, ann_path in pairs:
        if len(samples) >= a.n:
            break
        ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        cam = ev._camera(ann)
        fx = ev.camera_fx(cam)
        degrade.set_active(degrade.preset(rng.choice(PRESETS)))
        frames = ev._read_video(video)
        det.reset()
        for fi in (10, 40, 70):
            if len(samples) >= a.n:
                break
            fr = ann["frames"][fi]
            gt = np.asarray(fr["keypoints"], dtype=float).reshape(3, 3)
            tz = float(fr["pose_gt"]["translation_mm"][2])
            edge_px = fx * 100.0 / max(tz, 1.0)
            r_exp = 0.09 * edge_px
            maps = det.candidate_maps(frames[fi])
            cands = det.candidates_from_maps(maps, frames[fi])
            for i, c in enumerate(COLORS):
                for b in (cands.get(c) or [])[:4]:
                    if len(samples) >= a.n:
                        break
                    p = patch_from_map_scaled(maps[c], b.u, b.v, r_exp)
                    with torch.no_grad():
                        logit = model(torch.from_numpy(p)[None, None]).item()
                    prob = 1.0 / (1.0 + np.exp(-logit))
                    onnx_prob = float(1.0 / (1.0 + np.exp(-sess.run(None, {"patch": p[None, None].astype(np.float32)})[0][0])))
                    d = float(np.hypot(b.u - gt[i][0], b.v - gt[i][1]))
                    samples.append({"seq": Path(video).stem, "frame": fi, "color": c,
                                    "u": b.u, "v": b.v, "dist_gt": d,
                                    "patch": [float(x) for x in p.reshape(-1)],
                                    "prob": prob, "onnx_prob": onnx_prob})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(samples, indent=1), encoding="utf-8")
    max_diff = max(abs(s["prob"] - s["onnx_prob"]) for s in samples)
    print(f"wrote {len(samples)} fixtures, max |torch-onnx| = {max_diff:.3e}, "
          f"pos={sum(1 for s in samples if s['dist_gt'] <= 4)}, "
          f"neg={sum(1 for s in samples if s['dist_gt'] >= 12)}")


if __name__ == "__main__":
    main()
