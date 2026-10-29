"""Training losses for temporal tricolor detector."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .model import render_gaussian_heatmaps


def equilateral_penalty(coords: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    mask = (vis > 0.9).all(dim=-1)
    if mask.sum() < 1:
        return coords.new_zeros(())
    p0, p1, p2 = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :]
    d01 = torch.linalg.norm(p0 - p1, dim=-1)
    d12 = torch.linalg.norm(p1 - p2, dim=-1)
    d20 = torch.linalg.norm(p2 - p0, dim=-1)
    mean = (d01 + d12 + d20) / 3.0 + 1e-6
    pen = ((d01 - mean).abs() + (d12 - mean).abs() + (d20 - mean).abs()) / mean
    return pen[mask].mean()


def peak_margin_loss(
    pred_heat: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_vis: torch.Tensor,
    margin: float = 0.5,
    exclude_r: int = 3,
) -> torch.Tensor:
    """Push heat at GT above the strongest false peak (TopK recall surrogate).

    pred_heat: B,T,3,H,W logits; gt_coords: B,T,3,2 in [0,1].
    For each visible color: relu(max_false + margin - heat_gt).
    """
    b, t, c, h, w = pred_heat.shape
    device, dtype = pred_heat.device, pred_heat.dtype
    gx = (gt_coords[..., 0] * (w - 1)).round().long().clamp(0, w - 1)
    gy = (gt_coords[..., 1] * (h - 1)).round().long().clamp(0, h - 1)
    # gather heat at GT
    flat = pred_heat.view(b, t, c, -1)
    gidx = (gy * w + gx).clamp(0, h * w - 1)
    heat_gt = flat.gather(-1, gidx.unsqueeze(-1)).squeeze(-1)  # B,T,3

    # mask out a disk around GT so max is over distractors
    yy = torch.arange(h, device=device).view(1, 1, 1, h, 1)
    xx = torch.arange(w, device=device).view(1, 1, 1, 1, w)
    cy = gy.view(b, t, c, 1, 1)
    cx = gx.view(b, t, c, 1, 1)
    near = (yy - cy).abs() <= exclude_r
    near = near & ((xx - cx).abs() <= exclude_r)
    masked = pred_heat.masked_fill(near, -1e4)
    false_max = masked.amax(dim=(-2, -1))  # B,T,3

    # Include partial occlusion (dataset vis=0.5). Old `> 0.5` only trained
    # fully-visible LEDs, so hardmix/occlude TopK never got a peak-margin push.
    vis = (gt_vis >= 0.5).to(dtype)
    pen = F.relu(false_max + margin - heat_gt) * vis
    denom = vis.sum().clamp_min(1.0)
    return pen.sum() / denom


def peak_candidate_loss(
    pick_logits: torch.Tensor,
    cand_xy: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_vis: torch.Tensor,
    thr: float = 0.06,
) -> torch.Tensor:
    """CE on the candidate nearest to GT. pick_logits/cand_xy: B,T,3,K."""
    dist = torch.linalg.norm(cand_xy - gt_coords.unsqueeze(-2), dim=-1)
    dist = dist.masked_fill(pick_logits <= -1e3, 1e3)
    target = dist.argmin(dim=-1)
    ok = (dist.min(dim=-1).values <= thr) & (gt_vis >= 0.5)
    if not bool(ok.any().item()):
        return pick_logits.new_zeros(())
    return F.cross_entropy(pick_logits[ok], target[ok])


def tridot_loss(
    pred_coords: torch.Tensor,
    pred_vis: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_vis: torch.Tensor,
    pred_heat: Optional[torch.Tensor] = None,
    pred_coords0: Optional[torch.Tensor] = None,
    lambda_xy: float = 5.0,
    lambda_xy0: float = 0.0,
    lambda_v: float = 0.5,
    lambda_eq: float = 0.02,
    lambda_hm: float = 2.0,
    lambda_sep: float = 0.0,
    lambda_temp: float = 0.0,
    lambda_peak: float = 0.0,
    peak_margin: float = 0.5,
    peak_exclude: int = 3,
    last_frame_boost: float = 2.0,
    hm_sigma: float = 1.5,
) -> Dict[str, torch.Tensor]:
    b, t, _, _ = pred_coords.shape
    # Partial LEDs (vis=0.5) keep half xy/temp weight. Full weight on them
    # pulled easy and hard down in v3.23 without raising TopK recall.
    w = gt_vis.clone()
    w = torch.where(gt_vis > 0.9, torch.ones_like(w), w)
    w = torch.where(gt_vis < 0.1, torch.zeros_like(w), w)
    frame_w = torch.ones(t, device=pred_coords.device, dtype=pred_coords.dtype)
    frame_w[-1] = last_frame_boost
    w = w * frame_w.view(1, t, 1)

    xy = F.smooth_l1_loss(pred_coords, gt_coords, reduction="none").sum(dim=-1)
    denom = w.sum().clamp_min(1.0)
    loss_xy = (xy * w).sum() / denom

    loss_xy0 = pred_coords.new_zeros(())
    if pred_coords0 is not None and lambda_xy0 > 0:
        xy0 = F.smooth_l1_loss(pred_coords0, gt_coords, reduction="none").sum(dim=-1)
        loss_xy0 = (xy0 * w).sum() / denom

    target_v = (gt_vis >= 0.5).float()
    loss_v = F.binary_cross_entropy_with_logits(pred_vis, target_v, reduction="mean")
    loss_eq = equilateral_penalty(pred_coords, gt_vis)

    loss_hm = pred_coords.new_zeros(())
    loss_peak = pred_coords.new_zeros(())
    if pred_heat is not None:
        _, _, _, hh, ww = pred_heat.shape
        gt_hm = render_gaussian_heatmaps(gt_coords, gt_vis, hh, ww, sigma=hm_sigma)
        pred_logp = F.log_softmax(pred_heat.flatten(-2) * 4.0, dim=-1)
        gt_flat = gt_hm.flatten(-2)
        gt_sum = gt_flat.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        gt_prob = (gt_flat / gt_sum)
        mass = gt_flat.sum(dim=-1) > 0.1  # B,T,3
        if mass.any():
            # KL(gt || pred) on valid color maps — sharper than MSE on probs
            loss_hm = F.kl_div(pred_logp[mass], gt_prob[mass], reduction="batchmean")
        else:
            loss_hm = pred_heat.new_zeros(())
        if lambda_peak > 0:
            loss_peak = peak_margin_loss(
                pred_heat, gt_coords, gt_vis, margin=peak_margin, exclude_r=peak_exclude,
            )

    # discourage collapsed peaks: predicted pairwise distances should match GT
    loss_sep = pred_coords.new_zeros(())
    if lambda_sep > 0:
        src = pred_coords0 if pred_coords0 is not None else pred_coords
        mask = (gt_vis > 0.9).all(dim=-1)  # B,T
        if mask.any():
            def _edges(c):
                d01 = torch.linalg.norm(c[..., 0, :] - c[..., 1, :], dim=-1)
                d12 = torch.linalg.norm(c[..., 1, :] - c[..., 2, :], dim=-1)
                d20 = torch.linalg.norm(c[..., 2, :] - c[..., 0, :], dim=-1)
                return torch.stack([d01, d12, d20], dim=-1)
            loss_sep = F.smooth_l1_loss(_edges(src)[mask], _edges(gt_coords)[mask])

    loss_temp = pred_coords.new_zeros(())
    if lambda_temp > 0 and t >= 2:
        d_pred = pred_coords[:, 1:] - pred_coords[:, :-1]
        d_gt = gt_coords[:, 1:] - gt_coords[:, :-1]
        w_pair = torch.minimum(w[:, 1:], w[:, :-1])
        if w_pair.sum() > 0:
            loss_temp = (F.smooth_l1_loss(d_pred, d_gt, reduction="none").sum(dim=-1) * w_pair).sum() / w_pair.sum().clamp_min(1.0)

    total = (
        lambda_xy * loss_xy
        + lambda_xy0 * loss_xy0
        + lambda_v * loss_v
        + lambda_eq * loss_eq
        + lambda_hm * loss_hm
        + lambda_sep * loss_sep
        + lambda_temp * loss_temp
        + lambda_peak * loss_peak
    )
    return {
        "loss": total,
        "loss_xy": loss_xy.detach(),
        "loss_xy0": loss_xy0.detach() if torch.is_tensor(loss_xy0) else loss_xy0,
        "loss_v": loss_v.detach(),
        "loss_eq": loss_eq.detach() if torch.is_tensor(loss_eq) else loss_eq,
        "loss_hm": loss_hm.detach() if torch.is_tensor(loss_hm) else loss_hm,
        "loss_sep": loss_sep.detach() if torch.is_tensor(loss_sep) else loss_sep,
        "loss_temp": loss_temp.detach() if torch.is_tensor(loss_temp) else loss_temp,
        "loss_peak": loss_peak.detach() if torch.is_tensor(loss_peak) else loss_peak,
    }
