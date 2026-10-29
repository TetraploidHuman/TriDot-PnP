#!/usr/bin/env python3
"""On interrupt frames: is GT in TopK? heat@GT? best-peak vs geom vs coast."""
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


def heat_at(hm: torch.Tensor, xy: np.ndarray) -> float:
    """hm: H,W logits; xy normalized. bilinear sample of sigmoid."""
    if hm.ndim == 2:
        hm = hm[None, None]
    elif hm.ndim == 3:
        hm = hm[None]
    x = torch.tensor([[[[xy[0] * 2 - 1, xy[1] * 2 - 1]]]], device=hm.device, dtype=hm.dtype)
    v = F.grid_sample(torch.sigmoid(hm), x, align_corners=True)
    return float(v.item())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_find_checkpoint(Path("research/results/find_v3_22/best.pt"), device)
    model.eval()
    gk = torch.load(Path("research/results/find_geom_v322/best.pt"), map_location=device, weights_only=False)
    geom = GeomTripleNet().to(device)
    geom.load_state_dict(gk["model"])
    geom.eval()

    pairs = _discover_pairs(Path("datasets/TriDotPose"), "synthetic", subset="hard_test")
    stats = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for vid, ann_p in pairs:
            ann = json.loads(ann_p.read_text(encoding="utf-8"))
            fac = ann["info"].get("factor", "?")
            if fac not in ("hardmix", "occlude_hand", "handheld"):
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
            prev = prev2 = None
            for fi, fr in enumerate(ann["frames"][: len(xs)]):
                if fr.get("all_visible", True):
                    tag_frame = "full"
                else:
                    tag_frame = "cut"
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
                peaks = extract_heatmap_peaks(upsample_heat(hm, 1), top_k=5, nms=5)[0]  # 3,K,3
                k = fr["keypoints"]
                gt = np.array(
                    [[k[0] / w, k[1] / h], [k[3] / w, k[4] / h], [k[6] / w, k[7] / h]],
                    np.float32,
                )
                gvis = np.array([int(k[2]), int(k[5]), int(k[8])], np.int32)
                expect = None
                if prev is not None:
                    expect = prev.copy()
                    if prev2 is not None:
                        expect = np.clip(prev + (prev - prev2), 0, 1)

                for ci in range(3):
                    # nearest valid peak to GT
                    pk = peaks[ci].cpu().numpy()
                    valid = pk[:, 2] >= 0
                    if valid.any():
                        d = np.linalg.norm((pk[valid, :2] - gt[ci]) * np.array([w, h]), axis=1)
                        best_d = float(d.min())
                        in_topk10 = best_d <= 10.0
                        in_topk20 = best_d <= 20.0
                    else:
                        best_d, in_topk10, in_topk20 = 1e9, False, False
                    d_geom = float(np.linalg.norm((c[ci] - gt[ci]) * np.array([w, h])))
                    hat = heat_at(hm[0, ci], gt[ci])
                    pmax = float(pk[0, 2]) if float(pk[0, 2]) >= 0 else 0.0
                    d_coast = (
                        float(np.linalg.norm((expect[ci] - gt[ci]) * np.array([w, h])))
                        if expect is not None else None
                    )
                    # oracle: pick among geom / best-peak / coast
                    cands = [("geom", d_geom), ("bestpk", best_d)]
                    if d_coast is not None:
                        cands.append(("coast", d_coast))
                    oracle = min(cands, key=lambda t: t[1])[0]

                    key = f"{fac}|{tag_frame}|vis{gvis[ci]}"
                    stats[key]["n"].append(1)
                    stats[key]["in10"].append(float(in_topk10))
                    stats[key]["in20"].append(float(in_topk20))
                    stats[key]["best_d"].append(best_d if best_d < 1e8 else np.nan)
                    stats[key]["geom_d"].append(d_geom)
                    stats[key]["heat_gt"].append(hat)
                    stats[key]["pmax"].append(pmax)
                    stats[key]["oracle"].append(oracle)
                    if d_coast is not None:
                        stats[key]["coast_d"].append(d_coast)

                prev2 = prev
                prev = c

    print(f"{'bucket':32} {'n':>5} {'topk@10':>8} {'topk@20':>8} {'bestD':>7} {'geomD':>7} "
          f"{'coastD':>7} {'heatGT':>7} {'pmax':>6}  oracle%")
    for key in sorted(stats.keys()):
        s = stats[key]
        n = len(s["n"])
        if n < 20:
            continue
        oc = s["oracle"]
        from collections import Counter
        cnt = Counter(oc)
        ostr = " ".join(f"{k}={100*v/n:.0f}" for k, v in sorted(cnt.items()))
        cd = np.nanmean(s["coast_d"]) if s["coast_d"] else float("nan")
        print(
            f"{key:32} {n:5d} {100*np.mean(s['in10']):7.1f}% {100*np.mean(s['in20']):7.1f}% "
            f"{np.nanmean(s['best_d']):7.1f} {np.mean(s['geom_d']):7.1f} {cd:7.1f} "
            f"{np.mean(s['heat_gt']):7.3f} {np.mean(s['pmax']):6.3f}  {ostr}"
        )


if __name__ == "__main__":
    main()
