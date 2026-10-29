#!/usr/bin/env python3
"""Why did short coast fail? Jump size vs GT occlusion vs peak score."""
from __future__ import annotations

import json
from collections import defaultdict
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
    find = Path("research/results/find_v3_22/best.pt")
    geom_p = Path("research/results/find_geom_v322/best.pt")
    model = load_find_checkpoint(find, device)
    model.eval()
    gk = torch.load(geom_p, map_location=device, weights_only=False)
    geom = GeomTripleNet().to(device)
    geom.load_state_dict(gk["model"])
    geom.eval()

    pairs = _discover_pairs(Path("datasets/TriDotPose"), "synthetic", subset="hard_test")
    # rows: (tag, jump, peak, vis_prob, d_det, d_coast, coast_better)
    rows = []

    with torch.no_grad():
        for vid, ann_p in pairs:
            ann = json.loads(ann_p.read_text(encoding="utf-8"))
            fac = ann["info"].get("factor", "?")
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
            prev = prev2 = None
            for fi, fr in enumerate(ann["frames"][: len(xs)]):
                start = max(0, fi - t_len + 1)
                clip = xs[start : fi + 1]
                if len(clip) < t_len:
                    clip = [clip[0]] * (t_len - len(clip)) + clip
                x = torch.from_numpy(np.stack(clip)[None]).to(device)
                _, vis, heat, _ = model(x, return_heat=True)
                hm = heat[:, -1]
                vprob = torch.sigmoid(vis[0, -1]).cpu().numpy()
                prev_t = None if prev is None else torch.from_numpy(prev[None]).to(device)
                prev2_t = None if prev2 is None else torch.from_numpy(prev2[None]).to(device)
                c = decode_find_geom(
                    hm, geom=geom, force_geom=True, top_k=5, upsample=1,
                    prev_xy=prev_t, prev2_xy=prev2_t, temporal_w=10.0, vel_w=1.0,
                )[0].cpu().numpy()
                peaks = extract_heatmap_peaks(upsample_heat(hm, 1), top_k=5, nms=5)[0]
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
                        pscore = float(peaks[ci, 0, 2].item()) if float(peaks[ci, 0, 2]) >= 0 else -1.0
                        d_det = float(np.linalg.norm((c[ci] - gt[ci]) * np.array([w, h])))
                        d_coast = float(np.linalg.norm((expect[ci] - gt[ci]) * np.array([w, h])))
                        if gvis[ci] == 2:
                            tag = "vis2"
                        elif gvis[ci] == 1:
                            tag = "vis1"
                        else:
                            tag = "vis0"
                        rows.append(
                            (fac, tag, jump, pscore, float(vprob[ci]), d_det, d_coast, d_coast + 1.0 < d_det)
                        )
                prev2 = prev
                prev = c

    print(f"{'bucket':28} {'n':>6} {'jump50':>7} {'jump90':>7} {'peak50':>7} {'vis50':>6} "
          f"{'cb%':>6} {'detE':>7} {'coastE':>7}")
    by = defaultdict(list)
    for r in rows:
        by[(r[0], r[1])].append(r)
        by[("ALL", r[1])].append(r)

    for key in sorted(by.keys()):
        rs = by[key]
        if len(rs) < 30:
            continue
        j = np.array([r[2] for r in rs])
        p = np.array([r[3] for r in rs])
        vv = np.array([r[4] for r in rs])
        cb = 100 * np.mean([r[7] for r in rs])
        de = np.mean([r[5] for r in rs])
        ce = np.mean([r[6] for r in rs])
        name = f"{key[0]}|{key[1]}"
        print(
            f"{name:28} {len(rs):6d} {np.percentile(j,50):7.4f} {np.percentile(j,90):7.4f} "
            f"{np.percentile(p,50):7.3f} {np.percentile(vv,50):6.3f} {cb:6.1f} {de:7.1f} {ce:7.1f}"
        )

    print("\nGATE: jump>thr & peak<ps  (want high coast_better on vis0/1, low fire on vis2)")
    for thr in [0.04, 0.06, 0.08, 0.10, 0.12]:
        for ps in [0.25, 0.40, 0.55, 0.70, 9.0]:
            sel = [r for r in rows if r[2] > thr and r[3] < ps]
            if len(sel) < 20:
                continue
            for tag in ("vis2", "vis1", "vis0"):
                st = [r for r in sel if r[1] == tag]
                if not st:
                    continue
                cb = 100 * np.mean([r[7] for r in st])
                print(
                    f"  j>{thr:.2f} p<{ps:4.1f} {tag}: n={len(st):5d} cb={cb:5.1f}% "
                    f"det={np.mean([r[5] for r in st]):5.1f} coast={np.mean([r[6] for r in st]):5.1f}"
                )

    print("\nGATE: jump>thr & vis_prob<vv")
    for thr in [0.06, 0.08, 0.10]:
        for vv in [0.3, 0.5, 0.7]:
            sel = [r for r in rows if r[2] > thr and r[4] < vv]
            if len(sel) < 20:
                continue
            for tag in ("vis2", "vis1", "vis0"):
                st = [r for r in sel if r[1] == tag]
                if not st:
                    continue
                cb = 100 * np.mean([r[7] for r in st])
                print(
                    f"  j>{thr:.2f} v<{vv:.1f} {tag}: n={len(st):5d} cb={cb:5.1f}% "
                    f"det={np.mean([r[5] for r in st]):5.1f} coast={np.mean([r[6] for r in st]):5.1f}"
                )

    # naive coast fire rate at jump>0.06 (no peak gate) — why scores collapsed
    print("\nFIRE RATE jump-only (no peak gate):")
    for thr in [0.04, 0.06, 0.08, 0.10]:
        for tag in ("vis2", "vis1", "vis0"):
            base = [r for r in rows if r[1] == tag]
            fire = [r for r in base if r[2] > thr]
            if not base:
                continue
            cb = 100 * np.mean([r[7] for r in fire]) if fire else 0
            print(f"  j>{thr:.2f} {tag}: fire={100*len(fire)/len(base):5.1f}% "
                  f"({len(fire)}/{len(base)}) among-fired coast_better={cb:5.1f}%")


if __name__ == "__main__":
    main()
