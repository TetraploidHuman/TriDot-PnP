"""Clip dataset with dense find-preprocess inputs (no blob labels as input)."""

from __future__ import annotations

import json
import os
import random
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from research.tridot_temporal.dataset import ClipSample, _discover_pairs
from research.tridot_temporal.find_preprocess import build_find_input


class FindClipDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        subset: str = "train",
        seq_len: int = 5,
        img_h: int = 256,
        img_w: int = 448,
        augment: bool = True,
        max_clips_per_seq: int = 16,
        cache_videos: bool = True,
        disk_cache: bool = True,
        lru_max: int = 256,
        frame_drop_p: float = 0.0,
    ) -> None:
        self.seq_len = seq_len
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment
        self.frame_drop_p = float(frame_drop_p)
        self.clips: List[ClipSample] = []
        self.anns: Dict[str, dict] = {}
        self._cache: Dict[str, np.ndarray] = {}  # sid -> N,C,H,W float32 (optional RAM)
        # sid -> (NpzFile keep-alive, memmap array); slice-only reads, no full copy
        self._mmaps: "OrderedDict[str, Tuple[object, np.ndarray]]" = OrderedDict()
        self._lru: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._lru_max = int(lru_max)
        self._disk_cache = (
            Path(dataset_root) / "synthetic" / "find_cache" / f"{img_h}x{img_w}"
            if disk_cache else None
        )
        if self._disk_cache is not None:
            self._disk_cache.mkdir(parents=True, exist_ok=True)
        factor_w = {
            "clean": 1, "light": 1, "blur": 1, "noise": 1,
            "distractor": 2, "occ": 2, "mixed": 2,
            # realistic / hard factors — oversample v3.12 weak tails (no traj FT)
            "room": 2, "desk": 2, "clutter": 10, "screen": 14,
            # v3.19: push TopK recall on hardmix/occlude via heavier oversample
            "handheld": 8, "occlude_hand": 18, "crosscolor": 4,
            "far": 3, "hardmix": 24,
            # real capture (from label_tool export)
            "real": 4,
        }
        pairs = _discover_pairs(Path(dataset_root), "synthetic", subset=subset)
        pending_ram: List[Tuple[str, Path, int]] = []
        skipped = 0
        for vid, ann_path in pairs:
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            n = int(ann["info"]["num_frames"])
            if n < seq_len:
                continue
            # skip corrupt / unreadable mp4 (e.g. killed mid-generate)
            if self._npz_path(ann_path.stem) is None:
                cap = cv2.VideoCapture(str(vid))
                ok, _ = cap.read()
                cap.release()
                if not ok:
                    skipped += 1
                    continue
            self.anns[ann_path.stem] = ann
            if cache_videos:
                pending_ram.append((ann_path.stem, vid, n))
            step = max(1, (n - seq_len) // max(1, max_clips_per_seq - 1))
            starts = list(range(0, n - seq_len + 1, step))[:max_clips_per_seq]
            reps = factor_w.get(ann["info"].get("factor", "clean"), 1)
            for _ in range(reps):
                for s in starts:
                    self.clips.append(ClipSample(vid, ann_path, s, seq_len))
        if cache_videos and pending_ram:
            # Parallel RAM materialize only when disk npz already exists (read-only).
            # Missing npz are built sequentially to avoid concurrent writers.
            missing = []
            ready = []
            for stem, vid, n in pending_ram:
                if self._npz_path(stem) is not None:
                    ready.append((stem, vid, n))
                else:
                    missing.append((stem, vid, n))
            for stem, vid, n in missing:
                try:
                    self._cache[stem] = self._load_array(vid, n, stem)
                except Exception as exc:
                    print(f"[find] skip cache {stem}: {exc}", flush=True)
                    skipped += 1
                    self.clips = [c for c in self.clips if c.ann_path.stem != stem]
                    self.anns.pop(stem, None)

            def _one(item: Tuple[str, Path, int]) -> Tuple[str, np.ndarray]:
                stem, vid, n = item
                return stem, self._load_array(vid, n, stem)

            if ready:
                with ThreadPoolExecutor(max_workers=min(8, len(ready))) as pool:
                    for stem, arr in pool.map(_one, ready):
                        self._cache[stem] = arr
        if skipped:
            print(f"[find] skipped_unreadable={skipped}", flush=True)
        if not self.clips:
            raise RuntimeError("no find clips")

    def _npz_path(self, stem: str) -> Optional[Path]:
        if self._disk_cache is None:
            return None
        p = self._disk_cache / f"{stem}.npz"
        return p if p.exists() else None

    def _open_mmap(self, stem: str) -> np.ndarray:
        if stem in self._mmaps:
            self._mmaps.move_to_end(stem)
            return self._mmaps[stem][1]
        npz = self._npz_path(stem)
        if npz is None:
            raise FileNotFoundError(stem)
        # Keep NpzFile alive; closing it would invalidate the memmap.
        z = np.load(npz, mmap_mode="r")
        arr = z["x"]
        self._mmaps[stem] = (z, arr)
        while len(self._mmaps) > self._lru_max:
            old_stem, (old_z, _) = self._mmaps.popitem(last=False)
            try:
                old_z.close()
            except Exception:
                pass
            del old_stem
        return arr

    def _load_array(self, vid: Path, n: int, stem: Optional[str] = None) -> np.ndarray:
        """Full-sequence float32 array (used only for --cache-videos RAM preload)."""
        stem = stem or vid.stem
        npz = self._npz_path(stem)
        if npz is not None:
            try:
                with np.load(npz, mmap_mode="r") as z:
                    return np.asarray(z["x"], dtype=np.float32)
            except (ValueError, OSError):
                with np.load(npz) as z:
                    return z["x"].astype(np.float32)
        cap = cv2.VideoCapture(str(vid))
        frames = []
        for _ in range(n):
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(build_find_input(bgr, self.img_h, self.img_w))
        cap.release()
        if not frames:
            raise RuntimeError(f"unreadable/empty video: {vid}")
        while len(frames) < n:
            frames.append(frames[-1].copy())
        arr = np.stack(frames, 0).astype(np.float32)
        if self._disk_cache is not None and frames:
            # np.savez appends .npz unless the path already ends with .npz
            tmp = (
                self._disk_cache
                / f"{stem}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
            )
            np.savez(tmp, x=arr.astype(np.float16))
            os.replace(tmp, self._disk_cache / f"{stem}.npz")
        return arr

    def _clip_frames(self, sid: str, vid: Path, n: int, start: int) -> np.ndarray:
        """Return [T,C,H,W] float32 for one clip — prefer mmap slice (no full video copy)."""
        if sid in self._cache:
            return np.asarray(self._cache[sid][start : start + self.seq_len], dtype=np.float32)
        npz = self._npz_path(sid)
        if npz is not None:
            try:
                mm = self._open_mmap(sid)
                return np.asarray(mm[start : start + self.seq_len], dtype=np.float32)
            except (ValueError, OSError, FileNotFoundError):
                pass
        if sid in self._lru:
            self._lru.move_to_end(sid)
            return np.asarray(self._lru[sid][start : start + self.seq_len], dtype=np.float32)
        arr = self._load_array(vid, n, sid)
        self._lru[sid] = arr
        self._lru.move_to_end(sid)
        while len(self._lru) > self._lru_max:
            self._lru.popitem(last=False)
        return np.asarray(arr[start : start + self.seq_len], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.clips)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if not self.augment:
            return x
        # work in-place on a writable copy
        x = np.array(x, dtype=np.float32, copy=True)
        if random.random() < 0.5:
            g = random.uniform(0.7, 1.3)
            x *= g
        if random.random() < 0.3:
            x += random.uniform(-0.1, 0.1)
        if random.random() < 0.45:
            _, _, h, w = x.shape
            rh, rw = random.randint(h // 10, h // 3), random.randint(w // 10, w // 3)
            y0, x0 = random.randint(0, h - rh), random.randint(0, w - rw)
            x[:, :, y0 : y0 + rh, x0 : x0 + rw] = 0
        if random.random() < 0.35:
            _, _, h, w = x.shape
            thick = random.randint(max(2, h // 40), max(3, h // 12))
            if random.random() < 0.5:
                y0 = random.randint(0, max(0, h - thick))
                x[:, :, y0 : y0 + thick, :] = 0
            else:
                x0 = random.randint(0, max(0, w - thick))
                x[:, :, :, x0 : x0 + thick] = 0
        if random.random() < 0.25:
            noise = np.random.uniform(0.0, 0.35, size=x[:, 3:6].shape).astype(np.float32)
            x[:, 3:6] = np.clip(x[:, 3:6] + noise, 0, 1)
        np.clip(x, 0, 1, out=x)
        return x

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clip = self.clips[idx]
        ann = self.anns[clip.ann_path.stem]
        sid = clip.ann_path.stem
        ow = float(ann["info"]["width"])
        oh = float(ann["info"]["height"])
        n = int(ann["info"]["num_frames"])
        x = self._clip_frames(sid, clip.video_path, n, clip.start)
        coords, vis = [], []
        for i in range(self.seq_len):
            fi = clip.start + i
            k = ann["frames"][fi]["keypoints"]
            c_row, v_row = [], []
            for j in range(3):
                u, v, vv = float(k[3 * j]), float(k[3 * j + 1]), int(k[3 * j + 2])
                c_row.append([u / ow, v / oh])
                v_row.append(1.0 if vv == 2 else (0.5 if vv == 1 else 0.0))
            coords.append(c_row)
            vis.append(v_row)
        x = self._augment(x)
        if self.augment and self.frame_drop_p > 0 and random.random() < self.frame_drop_p:
            x[-1].fill(0.0)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x)),
            "coords": torch.tensor(coords, dtype=torch.float32),
            "vis": torch.tensor(vis, dtype=torch.float32),
        }
