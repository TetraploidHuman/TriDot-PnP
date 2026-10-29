#!/usr/bin/env python3
"""Would injecting velocity-expect as an extra peak candidate fix cut vis1?"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from research.tridot_temporal.dataset import _discover_pairs
from research.tridot_temporal.find_geom import (
    GeomTripleNet,
    decode_find_geom,
    extract_heatmap_peaks,
    upsample_heat,
)
from research.tridot_temporal.find_model import load_find_checkpoint
from research.tridot_temporal.find_preprocess import build_find_input


def sample_heat(hm: torch.Tensor, xy: np.ndarray) -> float:
    """hm: H,W logits → sigmoid at normalized xy."""
    x = torch.tensor(
        [[[[xy[0] * 2 - 1, xy[1] * 2 - 1]]]], device=hm.device, dtype=hm.dtype
    )
    return float(F.grid_sample(torch.sigmoid(hm[None, None]), x, align_corners=True).item())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_find_checkpoint(Path("research/results/find_v3_22/best.pt"), device)
    model.eval()
    gk = torch.load(Path("research/results/find_geom_v322/best.pt"), map_location=device, weights_only=False)
    geom = GeomTripleNet().to(device)
    geom.load_state_dict(gk["model"])
    geom.eval()

    pairs = _discover_pairs(Path("datasets/TriDotPose"), "synthetic", subset="hard_test")
    rows = []  # tag, d_geom, d_oracle_inject, heat_exp, heat_gt, inject_wins

    with torch.no_grad():
        for vid, ann_p in pairs:
            ann = json.loads(ann_p.read_text(encoding="utf-8"))
            fac = ann["info"].get("factor", "?")
            if fac not in ("hardmix", "occlude_hand", "handheld", "screen"):
                continue
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
                        d_geom = float(np.linalg.norm((c[ci] - gt[ci]) * scale))
                        he = sample_heat(hm[0, ci], expect[ci])
                        hg = sample_heat(hm[0, ci], gt[ci])
                        # oracle: among TopK + expect, nearest to GT
                        pk = peaks[ci]
                        valid = pk[:, 2] >= 0
                        cands = [c[ci]]
                        if valid.any():
                            cands.extend(list(pk[valid, :2]))
                        cands.append(expect[ci])
                        ds = [float(np.linalg.norm((np.asarray(p) - gt[ci]) * scale)) for p in cands]
                        d_oracle = min(ds)
                        # gated inject: only if heat_expect >= frac * top1
                        pmax = float(pk[0, 2]) if float(pk[0, 2]) >= 0 else 0.0
                        gate = he >= 0.5 * max(pmax, 1e-3)
                        d_gated = d_geom
                        if gate:
                            d_gated = min(d_geom, float(np.linalg.norm((expect[ci] - gt[ci]) * scale)))
                            # also nearest peak to expect if closer
                            if valid.any():
                                de = np.linalg.norm((pk[valid, :2] - expect[ci]) * scale, axis=1)
                                j = int(np.argmin(de))
                                if de[j] <= 30:
                                    d_gated = min(
                                        d_gated,
                                        float(np.linalg.norm((pk[valid][j, :2] - gt[ci]) * scale)),
                                    )
                        tag = f"{fac}|{'cut' if cut else 'full'}|vis{gvis[ci]}"
                        rows.append((tag, d_geom, d_oracle, d_gated, he, hg, gate, d_gated + 1 < d_geom))
                prev2 = prev
                prev = c

    print(f"{'bucket':32} {'n':>5} {'geomD':>7} {'orcD':>7} {'gatD':>7} {'he':>5} {'hg':>5} "
          f"{'gate%':>6} {'imp%':>6} {'@10g':>6} {'@10o':>6} {'@10t':>6}")
    by = defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
        p = r[0].split("|")
        by[f"ALL|{p[1]}|{p[2]}"].append(r)
    for key in sorted(by):
        rs = by[key]
        if len(rs) < 30:
            continue
        g = np.array([r[1] for r in rs])
        o = np.array([r[2] for r in rs])
        t = np.array([r[3] for r in rs])
        print(
            f"{key:32} {len(rs):5d} {g.mean():7.1f} {o.mean():7.1f} {t.mean():7.1f} "
            f"{np.mean([r[4] for r in rs]):5.2f} {np.mean([r[5] for r in rs]):5.2f} "
            f"{100*np.mean([r[6] for r in rs]):5.1f}% {100*np.mean([r[7] for r in rs]):5.1f}% "
            f"{100*np.mean(g<=10):5.1f}% {100*np.mean(o<=10):5.1f}% {100*np.mean(t<=10):5.1f}%"
        )


if __name__ == "__main__":
    main()
