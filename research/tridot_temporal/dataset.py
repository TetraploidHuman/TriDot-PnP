"""Sequence discovery shared by the finder dataset and eval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

COLOR_ORDER = ("red", "green", "blue")


@dataclass
class ClipSample:
    video_path: Path
    ann_path: Path
    start: int
    seq_len: int


def _discover_pairs(
    dataset_root: Path,
    split: str,
    subset: Optional[str] = None,
) -> List[Tuple[Path, Path]]:
    """split: examples | synthetic. subset filters via meta/splits.json."""
    root = Path(dataset_root)
    if split == "examples":
        ann_dir = root / "examples" / "annotations"
        vid_dir = root / "examples" / "videos"
    else:
        ann_dir = root / "synthetic" / "annotations"
        vid_dir = root / "synthetic" / "videos"

    allow = None
    if subset:
        splits_path = root / "meta" / "splits.json"
        if splits_path.exists():
            splits = json.loads(splits_path.read_text(encoding="utf-8"))
            keys = [k.strip() for k in str(subset).split(",") if k.strip()]
            allow = set()
            for key in keys:
                allow.update(splits.get(key, []))

    pairs = []
    if not ann_dir.exists():
        return pairs
    for ann in sorted(ann_dir.glob("*.json")):
        if allow is not None and ann.stem not in allow:
            continue
        vid = vid_dir / f"{ann.stem}.mp4"
        if vid.exists():
            pairs.append((vid, ann))
    return pairs
