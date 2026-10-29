#!/usr/bin/env python3
"""Can we snap jumped colors to a TopK peak near velocity expect?"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from research.tridot_temporal.dataset import _discover_pairs
from research.tridot_temporal.find_geom import (
    GeomTripleNet,
    decode_find_geom,
    extract_heatmap_peaks,
    upsample_heat,
)
from research.tridot_temporal.find_model import load_find_checkpoint
from research.tridot_temporal.find_preprocess import build_find_input


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_find_checkpoint(Path("research/results/find_v3_22/best.pt"), device)
    model.eval()
    gk = torch.load(Path("research/results/find_geom_v322/best.pt"), map_location=device, weights_only=False)
    geom = GeomTripleNet().to(device)
    geom.load_state_dict(gk["model"])
    geom.eval()

    pairs = _discover_pairs(Path("datasets/TriDotPose"), "synthetic", subset="hard_test")
    # (tag, d_geom, d_snap, d_coast, has_peak, improved)
    rows = []

    with torch.no_grad():
        for vid, ann_p in pairs:
            ann = json.loads(ann_p.read_text(encoding="utf-8"))
            fac = ann["info"].get("factor", "?")
            w, h = int(ann["info"]["width"]), int(ann["info"]["height"])
            scale = np.array([w, h], np.float32)
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
            prev = prev2 = None
            for fi, fr in enumerate(ann["frames"][: len(xs)]):
                cut = not fr.get("all_visible", True)
                start = max(0, fi - t_len + 1)
                clip = xs[start : fi + 1]
                if len(clip) < t_len:
                    clip = [clip[0]] * (t_len - len(clip)) + clip
                x = torch.from_numpy(np.stack(clip)[None]).to(device)
                _, _, heat, _ = model(x, return_heat=True)
                hm = heat[:, -1]
                prev_t = None if prev is None else torch.from_numpy(prev[None]).to(device)
                prev2_t = None if prev2 is None else torch.from_numpy(prev2[None]).to(device)
                c = decode_find_geom(
                    hm, geom=geom, force_geom=True, top_k=5, upsample=1,
                    prev_xy=prev_t, prev2_xy=prev2_t, temporal_w=10.0, vel_w=1.0,
                )[0].cpu().numpy()
                peaks = extract_heatmap_peaks(upsample_heat(hm, 1), top_k=5, nms=5)[0].cpu().numpy()
                k = fr["keypoints"]
                gt = np.array(
                    [[k[0] / w, k[1] / h], [k[3] / w, k[4] / h], [k[6] / w, k[7] / h]],
                    np.float32,
                )
                gvis = np.array([int(k[2]), int(k[5]), int(k[8])], np.int32)
                if prev is not None:
                    expect = prev.copy()
                    if prev2 is not None:
                        expect = np.clip(prev + (prev - prev2), 0, 1)
                    for ci in range(3):
                        jump = float(np.linalg.norm(c[ci] - expect[ci]))
                        if jump <= 0.06:
                            continue
                        pk = peaks[ci]
                        valid = pk[:, 2] >= 0.25
                        d_geom = float(np.linalg.norm((c[ci] - gt[ci]) * scale))
                        d_coast = float(np.linalg.norm((expect[ci] - gt[ci]) * scale))
                        has = False
                        d_snap = d_geom
                        if valid.any():
                            d_exp = np.linalg.norm((pk[valid, :2] - expect[ci]) * scale, axis=1)
                            # nearest peak to expect within 40px
                            j = int(np.argmin(d_exp))
                            if d_exp[j] <= 40.0:
                                has = True
                                xy = pk[valid][j, :2]
                                d_snap = float(np.linalg.norm((xy - gt[ci]) * scale))
                        tag = f"{fac}|{'cut' if cut else 'full'}|vis{gvis[ci]}"
                        rows.append((tag, d_geom, d_snap, d_coast, has, d_snap + 1 < d_geom))
                prev2 = prev
                prev = c

    print(f"{'bucket':32} {'n':>5} {'has%':>6} {'imp%':>6} {'geomD':>7} {'snapD':>7} {'coastD':>7}")
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
        parts = r[0].split("|")
        by[f"ALL|{parts[1]}|{parts[2]}"].append(r)
    for key in sorted(by):
        rs = by[key]
        if len(rs) < 15:
            continue
        print(
            f"{key:32} {len(rs):5d} {100*np.mean([r[4] for r in rs]):5.1f}% "
            f"{100*np.mean([r[5] for r in rs]):5.1f}% "
            f"{np.mean([r[1] for r in rs]):7.1f} {np.mean([r[2] for r in rs]):7.1f} "
            f"{np.mean([r[3] for r in rs]):7.1f}"
        )


if __name__ == "__main__":
    main()
