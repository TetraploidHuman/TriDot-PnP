"""Sub-pixel offset head for the learned point finder.

Why this exists
---------------
The v3.x detector predicts 3 heatmaps at stride 4 (112x64 for a 448x256 input,
i.e. 11.4 original pixels per cell) and has **no offset / regression head**:
sub-pixel accuracy comes from a hand-written 9x9 softmax centroid at decode
time. That caps localization right at the det@10 decision boundary and makes
the metric bimodal (0.04 or 0.88). This module adds the standard fix used by
keypoint detectors: one 2-D offset vector per color per heat cell, predicting
where inside its own cell the peak lies.

Conventions
-----------
* heat / offset maps are at the same resolution HxW (stride `heatmap_stride`).
* normalized coordinates use the same convention as `soft_argmax_2d` /
  `render_gaussian_heatmaps`: index i in [0, H-1] maps to i/(H-1).
* offset maps store the fractional position inside the cell in (0, 1)
  (sigmoid output), so a refined normalized coordinate is
  ``(cell_index + offset) / (size - 1)``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from research.tridot_temporal.find_geom import extract_heatmap_peaks


def _cell_and_frac(
    coords: torch.Tensor, h: int, w: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """coords ...×2 in [0,1] → (cx, cy, frac_x, frac_y) with cx/cy cell indices."""
    gx = coords[..., 0] * (w - 1)
    gy = coords[..., 1] * (h - 1)
    cx = gx.floor().clamp(0, w - 1)
    cy = gy.floor().clamp(0, h - 1)
    fx = (gx - cx).clamp(0.0, 1.0)
    fy = (gy - cy).clamp(0.0, 1.0)
    return cx.long(), cy.long(), fx, fy


def _sample_offset(
    off_map: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
) -> torch.Tensor:
    """off_map B,T,3,2,H,W; cx/cy B,T,3 → B,T,3,2."""
    b, t, c, _, h, w = off_map.shape
    p = off_map.permute(0, 1, 2, 4, 5, 3).reshape(b, t, c, h * w, 2)
    idx = (cy * w + cx).view(b, t, c, 1, 1).expand(b, t, c, 1, 2)
    return torch.gather(p, 3, idx).squeeze(3)


def offset_loss(
    off_map: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_vis: torch.Tensor,
    min_vis: float = 0.5,
) -> torch.Tensor:
    """L1 on the per-cell fractional position at the GT cell.

    off_map: B,T,3,2,H,W in (0,1); gt_coords: B,T,3,2 in [0,1].
    Only LEDs with vis >= min_vis (fully or partially visible) are supervised.
    """
    b, t, c, _, h, w = off_map.shape
    cx, cy, fx, fy = _cell_and_frac(gt_coords, h, w)
    pred = _sample_offset(off_map, cx, cy)  # B,T,3,2
    target = torch.stack([fx, fy], dim=-1)
    valid = (gt_vis >= min_vis).unsqueeze(-1).to(off_map.dtype)  # B,T,3,1
    err = F.smooth_l1_loss(pred, target, reduction="none") * valid
    return err.sum() / valid.sum().clamp_min(1.0)


@torch.no_grad()
def decode_offset(
    heat: torch.Tensor,
    off_map: torch.Tensor,
    top_k: int = 1,
    nms: int = 5,
) -> torch.Tensor:
    """heat B,3,H,W logits + off_map B,3,2,H,W → coords B,3,2 in [0,1].

    Picks the strongest NMS peak per color, then snaps it to the predicted
    sub-cell offset. When every peak belongs to the same cell neighborhood the
    expected error drops from ~half a cell (~5.7 px at 720p) to ~1-2 px.
    """
    b, c, h, w = heat.shape
    peaks = extract_heatmap_peaks(heat, top_k=max(top_k, 1), nms=nms)  # B,3,K,3
    xy = peaks[:, :, 0, :2]  # B,3,2 (normalized)
    cx, cy, _, _ = _cell_and_frac(xy, h, w)  # B,3
    off = _sample_offset(off_map.unsqueeze(1), cx.unsqueeze(1), cy.unsqueeze(1))[:, 0]
    refined = (torch.stack([cx, cy], dim=-1).to(heat.dtype) + off) / torch.tensor(
        [w - 1, h - 1], device=heat.device, dtype=heat.dtype,
    ).view(1, 1, 2)
    bad = peaks[:, :, 0, 2] < 0  # no valid peak in that color
    if bool(bad.any().item()):
        refined = torch.where(bad.unsqueeze(-1), torch.full_like(refined, 0.5), refined)
    return refined.clamp(0.0, 1.0)


@torch.no_grad()
def refine_with_offset(coords: torch.Tensor, off_map: torch.Tensor) -> torch.Tensor:
    """Snap already-decoded coordinates to their learned sub-cell offset.

    ``coords`` (B,3,2) in [0,1] come from a *robust* peak/triple selection such
    as ``decode_find_geom``; this only refines the position inside the cell that
    was already chosen, so a wrong peak can never be made worse by the head.
    ``off_map`` is (B,3,2,H,W), the same layout as ``decode_offset``.
    """
    b, c, _, h, w = off_map.shape
    cx, cy, _, _ = _cell_and_frac(coords, h, w)
    off = _sample_offset(off_map.unsqueeze(1), cx.unsqueeze(1), cy.unsqueeze(1))[:, 0]
    refined = (torch.stack([cx, cy], dim=-1).to(coords.dtype) + off) / torch.tensor(
        [w - 1, h - 1], device=coords.device, dtype=coords.dtype,
    ).view(1, 1, 2)
    return refined.clamp(0.0, 1.0)
