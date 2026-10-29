#!/usr/bin/env python3
"""Compare v3.12 vs v3.19b heat on easy blur/noise: GT logit and TopK membership."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from research.tridot_temporal.dataset import _discover_pairs
from research.tridot_temporal.find_geom import extract_heatmap_peaks, upsample_heat
from research.tridot_temporal.find_model import load_find_checkpoint
from research.tridot_temporal.find_preprocess import build_find_input


def stats(ckpt: Path, factors: set[str]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_find_checkpoint(ckpt, device)
    model.eval()
    pairs = _discover_pairs(Path("datasets/TriDotPose"), "synthetic", subset="test")
    rows = {f: [] for f in factors}
    with torch.no_grad():
        for vid, ann_p in pairs:
            ann = json.loads(ann_p.read_text(encoding="utf-8"))
            fac = ann["info"].get("factor", "?")
            if fac not in factors:
                continue
            w, h = int(ann["info"]["width"]), int(ann["info"]["height"])
            ih, iw = model.cfg.img_h, model.cfg.img_w
            t_len = model.cfg.seq_len
            cap = cv2.VideoCapture(str(vid))
            xs = []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                xs.append(build_find_input(bgr, ih, iw))
            cap.release()
            for fi, fr in enumerate(ann["frames"][: len(xs)]):
                start = max(0, fi - t_len + 1)
                clip = xs[start : fi + 1]
                if len(clip) < t_len:
                    clip = [clip[0]] * (t_len - len(clip)) + clip
                x = torch.from_numpy(np.stack(clip)[None]).to(device)
                _, _, heat, _ = model(x, return_heat=True)
                hm = heat[0, -1]  # 3,H,W
                peaks = extract_heatmap_peaks(upsample_heat(heat[:, -1], 1), top_k=5, nms=5)
                k = fr["keypoints"]
                Hh, Ww = hm.shape[-2:]
                for ci, (kx, ky) in enumerate(((k[0], k[1]), (k[3], k[4]), (k[6], k[7]))):
                    px = int(np.clip(round((kx / w) * (Ww - 1)), 0, Ww - 1))
                    py = int(np.clip(round((ky / h) * (Hh - 1)), 0, Hh - 1))
                    y0, y1 = max(0, py - 2), min(Hh, py + 3)
                    x0, x1 = max(0, px - 2), min(Ww, px + 3)
                    gt_logit = float(hm[ci, y0:y1, x0:x1].amax().item())
                    valid = peaks[0, ci, :, 2] >= 0
                    in_topk = False
                    if bool(valid.any()):
                        pts = peaks[0, ci, valid, :2].cpu().numpy()
                        gt = np.array([kx / w, ky / h], np.float32)
                        d = np.linalg.norm((pts - gt) * np.array([w, h], np.float32), axis=-1)
                        in_topk = bool(d.min() <= 10)
                    rows[fac].append((gt_logit, in_topk))
    print(f"ckpt={ckpt.name if False else ckpt}")
    for fac, items in rows.items():
        if not items:
            continue
        logits = np.array([a for a, _ in items])
        rec = np.mean([b for _, b in items])
        print(
            f"  {fac:8} n={len(items):4d}  heat@GT mean={logits.mean():6.3f} "
            f"p10={np.percentile(logits,10):6.3f}  topk@10={rec:.3f}"
        )


def main() -> None:
    fac = {"blur", "noise", "clean", "light"}
    stats(Path("research/results/find_v3_12_data/best.pt"), fac)
    stats(Path("research/results/find_v3_19b/best.pt"), fac)


if __name__ == "__main__":
    main()
