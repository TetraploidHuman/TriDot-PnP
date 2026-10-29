#!/usr/bin/env python3
"""Check whether eval sets actually contain interrupt/occlusion events."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path("datasets/TriDotPose")
    for cand in [root / "meta" / "splits.json", root / "splits.json"]:
        if cand.exists():
            splits = json.loads(cand.read_text())
            print("splits", cand)
            break
    else:
        raise SystemExit("no splits")
    ann_dir = root / "synthetic" / "annotations"

    def analyze(split: str) -> None:
        ids = splits.get(split, [])
        by_fac: Counter = Counter()
        vis_stats = defaultdict(list)
        motion = defaultdict(list)
        occ_frames = defaultdict(list)
        # consecutive invisible run length
        run_lens = defaultdict(list)
        n_seq = 0
        for sid in ids:
            stem = sid if isinstance(sid, str) else str(sid)
            p = ann_dir / f"{stem}.json"
            if not p.exists():
                hits = list(ann_dir.glob(f"{stem}.json"))
                if not hits:
                    continue
                p = hits[0]
            ann = json.loads(p.read_text())
            fac = ann["info"].get("factor", "?")
            by_fac[fac] += 1
            n_seq += 1
            w = float(ann["info"]["width"])
            h = float(ann["info"]["height"])
            frames = ann["frames"]
            invis = 0
            total = 0
            coords = []
            run = 0
            for fr in frames:
                k = fr["keypoints"]
                row = []
                frame_invis = False
                for j in range(3):
                    u, v, vv = float(k[3 * j]), float(k[3 * j + 1]), int(k[3 * j + 2])
                    row.append([u / w, v / h])
                    total += 1
                    if vv != 2:
                        invis += 1
                        frame_invis = True
                coords.append(row)
                occ_frames[fac].append(1 if frame_invis else 0)
                if frame_invis:
                    run += 1
                else:
                    if run:
                        run_lens[fac].append(run)
                        run = 0
            if run:
                run_lens[fac].append(run)
            vis_stats[fac].append(invis / max(total, 1))
            c = np.array(coords, np.float32)
            if len(c) >= 2:
                d = np.linalg.norm(c[1:] - c[:-1], axis=-1).mean()
                motion[fac].append(float(d))
        print(f"\n## {split} sequences={n_seq}")
        print(f"{'factor':14} {'n':>3} {'invis_pct':>9} {'occ_fr_pct':>10} {'avg_run':>8} {'mot':>8}")
        for fac in sorted(by_fac):
            inv = 100 * float(np.mean(vis_stats[fac])) if vis_stats[fac] else 0.0
            of = 100 * float(np.mean(occ_frames[fac])) if occ_frames[fac] else 0.0
            ar = float(np.mean(run_lens[fac])) if run_lens[fac] else 0.0
            mot = float(np.mean(motion[fac])) if motion[fac] else 0.0
            print(f"{fac:14} {by_fac[fac]:3d} {inv:9.2f} {of:10.2f} {ar:8.1f} {mot:8.4f}")
        # global: any occlusion at all?
        any_occ = sum(1 for fac in by_fac for x in occ_frames[fac] if x)
        tot_fr = sum(len(occ_frames[fac]) for fac in by_fac)
        print(f"  frames_with_any_invis={any_occ}/{tot_fr} ({100*any_occ/max(tot_fr,1):.2f}%)")

    for s in ["test", "hard_test", "hard_train", "train"]:
        analyze(s)


if __name__ == "__main__":
    main()
