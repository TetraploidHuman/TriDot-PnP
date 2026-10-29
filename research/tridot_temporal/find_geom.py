"""Multi-peak decode from FindNet heatmaps + learned equilateral triangle scoring.

Peaks come ONLY from model heatmaps (NMS on logits) — not classical blobs.
Geometry: a small MLP learns to score (R,G,B) triples using coords + edge/shape features.

v3.2: fully vectorized peak/triple ops + optional heat upsample for sub-pixel +
adaptive fast path (top1 when geometry is already clean).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def upsample_heat(heat: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Upsample heat logits for finer peak localization (bicubic)."""
    if scale <= 1:
        return heat
    return F.interpolate(heat, scale_factor=scale, mode="bicubic", align_corners=True)


@torch.no_grad()
def extract_heatmap_peaks(
    heat: torch.Tensor,
    top_k: int = 5,
    nms: int = 5,
) -> torch.Tensor:
    """
    Vectorized local-max NMS + top-k (v3.3 path; matches geom_v32 training).
    heat: B,3,H,W logits → peaks B,3,K,3 with (x,y in [0,1], score in [0,1])
    Invalid slots have score = -1.
    """
    b, c, h, w = heat.shape
    device, dtype = heat.device, heat.dtype
    ksz = nms if nms % 2 == 1 else nms + 1
    pad = ksz // 2
    # pad explicitly then pool without padding to avoid size mismatches on CUDA
    hm = F.pad(heat, (pad, pad, pad, pad), mode="constant", value=-1e4)
    pooled = F.max_pool2d(hm, kernel_size=ksz, stride=1, padding=0)
    is_peak = heat >= pooled - 1e-5
    scored = heat.masked_fill(~is_peak, -1e4)
    flat = scored.reshape(b, c, -1)
    k_eff = min(top_k, h * w)
    vals, idxs = flat.topk(k_eff, dim=-1)
    if k_eff < top_k:
        pad_v = torch.full((b, c, top_k - k_eff), -1e4, device=device, dtype=dtype)
        pad_i = torch.zeros((b, c, top_k - k_eff), device=device, dtype=idxs.dtype)
        vals = torch.cat([vals, pad_v], dim=-1)
        idxs = torch.cat([idxs, pad_i], dim=-1)
    py = torch.div(idxs, w, rounding_mode="floor")
    px = idxs - py * w
    out = torch.empty(b, c, top_k, 3, device=device, dtype=dtype)
    out[..., 0] = px.float() / max(w - 1, 1)
    out[..., 1] = py.float() / max(h - 1, 1)
    out[..., 2] = torch.sigmoid(vals)
    bad = vals < -20
    out[..., 2] = torch.where(bad, torch.full_like(out[..., 2], -1.0), out[..., 2])
    return out


def _triple_features(peaks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized. peaks: B,3,K,3 → feats B,M,15 / idx B,M,3 / valid B,M
    """
    b, _, k, _ = peaks.shape
    device = peaks.device
    ir = torch.arange(k, device=device)
    ig = torch.arange(k, device=device)
    ib = torch.arange(k, device=device)
    ir, ig, ib = torch.meshgrid(ir, ig, ib, indexing="ij")
    ir = ir.reshape(-1)
    ig = ig.reshape(-1)
    ib = ib.reshape(-1)
    m = ir.numel()
    pr = peaks[:, 0, ir, :]  # B,M,3
    pg = peaks[:, 1, ig, :]
    pb = peaks[:, 2, ib, :]
    e01 = torch.linalg.norm(pr[..., :2] - pg[..., :2], dim=-1)
    e12 = torch.linalg.norm(pg[..., :2] - pb[..., :2], dim=-1)
    e20 = torch.linalg.norm(pb[..., :2] - pr[..., :2], dim=-1)
    mean = (e01 + e12 + e20) / 3.0 + 1e-6
    shape = (torch.abs(e01 - mean) + torch.abs(e12 - mean) + torch.abs(e20 - mean)) / mean
    area = 0.5 * torch.abs(
        (pg[..., 0] - pr[..., 0]) * (pb[..., 1] - pr[..., 1])
        - (pb[..., 0] - pr[..., 0]) * (pg[..., 1] - pr[..., 1])
    )
    prop = pr[..., 2] + pg[..., 2] + pb[..., 2]
    feats = torch.stack(
        [
            pr[..., 0], pr[..., 1], pr[..., 2],
            pg[..., 0], pg[..., 1], pg[..., 2],
            pb[..., 0], pb[..., 1], pb[..., 2],
            e01, e12, e20, shape, area, prop,
        ],
        dim=-1,
    )
    idxs = torch.stack([ir, ig, ib], dim=-1).unsqueeze(0).expand(b, m, 3)
    valid = (pr[..., 2] >= 0) & (pg[..., 2] >= 0) & (pb[..., 2] >= 0)
    return feats, idxs, valid


class GeomTripleNet(nn.Module):
    """Learns which (R,G,B) heatmap-peak triple is the LED triangle."""

    def __init__(self, in_dim: int = 15, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats).squeeze(-1)

    def score_peaks(self, peaks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats, idxs, valid = _triple_features(peaks)
        logits = self.forward(feats)
        logits = logits.masked_fill(~valid, -1e9)
        return logits, idxs, valid


@torch.no_grad()
def refine_peak_local(
    heat: torch.Tensor,
    xy: torch.Tensor,
    win: int = 9,
    temp: float = 12.0,
) -> torch.Tensor:
    """Sub-pixel refine. heat: B,3,H,W or 3,H,W ; xy: B,3,2 or 3,2."""
    single = heat.dim() == 3
    if single:
        heat = heat.unsqueeze(0)
        xy = xy.unsqueeze(0)
    b, _, h, w = heat.shape
    half = win // 2
    out = xy.clamp(0, 1).clone()
    for bi in range(b):
        for ci in range(3):
            px = int(torch.round(out[bi, ci, 0] * (w - 1)).clamp(0, w - 1).item())
            py = int(torch.round(out[bi, ci, 1] * (h - 1)).clamp(0, h - 1).item())
            y0, y1 = max(0, py - half), min(h, py + half + 1)
            x0, x1 = max(0, px - half), min(w, px + half + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            patch = heat[bi, ci, y0:y1, x0:x1] * temp
            prob = torch.softmax(patch.reshape(-1), dim=0).view(y1 - y0, x1 - x0)
            yy = torch.arange(y0, y1, device=heat.device, dtype=heat.dtype)
            xx = torch.arange(x0, x1, device=heat.device, dtype=heat.dtype)
            ys, xs = torch.meshgrid(yy, xx, indexing="ij")
            out[bi, ci, 0] = (prob * xs).sum() / max(w - 1, 1)
            out[bi, ci, 1] = (prob * ys).sum() / max(h - 1, 1)
    return out[0] if single else out


def _gather_triple_coords(
    peaks: torch.Tensor,
    idxs: torch.Tensor,
    best: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """peaks B,3,K,3 ; idxs B,M,3 ; best B → coords B,3,2"""
    b, _, k, _ = peaks.shape
    out = torch.zeros(b, 3, 2, device=peaks.device, dtype=peaks.dtype)
    for bi in range(b):
        mi = int(best[bi].clamp(0, idxs.shape[1] - 1).item())
        if valid is not None and not bool(valid[bi, mi].item()):
            out[bi] = peaks[bi, :, 0, :2].clamp(0, 1)
            continue
        ir = int(idxs[bi, mi, 0].clamp(0, k - 1).item())
        ig = int(idxs[bi, mi, 1].clamp(0, k - 1).item())
        ib = int(idxs[bi, mi, 2].clamp(0, k - 1).item())
        out[bi, 0] = peaks[bi, 0, ir, :2]
        out[bi, 1] = peaks[bi, 1, ig, :2]
        out[bi, 2] = peaks[bi, 2, ib, :2]
    return out.clamp(0, 1)


@torch.no_grad()
def select_triangle_analytic(
    peaks: torch.Tensor,
    heat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    feats, idxs, valid = _triple_features(peaks)
    shape = feats[..., 12]
    prop = feats[..., 14]
    area = feats[..., 13]
    score = prop - 0.8 * shape + 0.15 * area.clamp(0, 0.05) * 20
    score = score.masked_fill(~valid, -1e9)
    score = score.masked_fill(area < 1e-4, -1e9)
    best = score.argmax(dim=-1)
    out = _gather_triple_coords(peaks, idxs, best, valid=valid)
    if heat is not None:
        out = refine_peak_local(heat, out)
    return out


@torch.no_grad()
def apply_temporal_heat_prior(
    heat: torch.Tensor,
    prev_xy: torch.Tensor,
    strength: float = 2.0,
    sigma: float = 0.04,
) -> torch.Tensor:
    """Add Gaussian bumps at prev_xy on heat logits. heat: B,3,H,W; prev_xy: B,3,2."""
    if strength <= 0:
        return heat
    b, c, h, w = heat.shape
    device, dtype = heat.device, heat.dtype
    yy = torch.linspace(0, 1, h, device=device, dtype=dtype).view(1, 1, h, 1)
    xx = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w)
    out = heat.clone()
    sig2 = 2.0 * (sigma ** 2)
    for ki in range(c):
        px = prev_xy[:, ki, 0].view(b, 1, 1, 1)
        py = prev_xy[:, ki, 1].view(b, 1, 1, 1)
        g = torch.exp(-((xx - px) ** 2 + (yy - py) ** 2) / sig2)
        out[:, ki : ki + 1] = out[:, ki : ki + 1] + strength * g
    return out


@torch.no_grad()
def geom_triple_gap(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Best−2nd geom logit among valid triples. logits/valid: B,M → gap B."""
    b, m = logits.shape
    gaps = torch.zeros(b, device=logits.device, dtype=logits.dtype)
    for bi in range(b):
        v = valid[bi]
        if int(v.sum().item()) <= 1:
            gaps[bi] = 99.0
            continue
        vals = logits[bi].masked_fill(~v, -1e9)
        top2 = torch.topk(vals, k=2).values
        gaps[bi] = top2[0] - top2[1]
    return gaps


@torch.no_grad()
def select_triangle_learned(
    geom: GeomTripleNet,
    peaks: torch.Tensor,
    heat: Optional[torch.Tensor] = None,
    prev_xy: Optional[torch.Tensor] = None,
    temporal_w: float = 3.0,
    prev2_xy: Optional[torch.Tensor] = None,
    vel_w: float = 1.0,
    geom_gap_gate: bool = False,
    geom_gap_thr: float = 1.0,
    geom_gap_sharp: float = 4.0,
    jump_rescue: bool = False,
    jump_thr: float = 0.08,
    jump_logit_slack: float = 1.0,
    jump_max_dist: float = 0.05,
) -> torch.Tensor:
    """prev_xy: B,3,2 previous-frame coords — bias toward temporally consistent triples.

    If prev2_xy is set, also bias toward constant-velocity prediction
    expect = prev + (prev - prev2).

    geom_gap_gate: scale temporal_w by how ambiguous the geom ranking is
      (high best−2nd logit gap → trust geom; low gap → use trajectory).
      This replaces the broken peak-score margin gate.

    jump_rescue: if pure-geom best jumps far from the track, pick a near-track
      triple whose geom logit is within jump_logit_slack of the best (still a
      real peak triple — never blind coast).
    """
    logits, idxs, valid = geom.score_peaks(peaks)
    base = logits
    gap = geom_triple_gap(base, valid)  # B

    expect = None
    dist = None
    if prev_xy is not None and (temporal_w > 0 or jump_rescue):
        ir, ig, ib = idxs[0, :, 0], idxs[0, :, 1], idxs[0, :, 2]
        pred = torch.stack(
            [peaks[:, 0, ir, :2], peaks[:, 1, ig, :2], peaks[:, 2, ib, :2]], dim=2
        )  # B,M,3,2
        expect = prev_xy
        if prev2_xy is not None and vel_w > 0:
            expect = prev_xy + vel_w * (prev_xy - prev2_xy)
        expect = expect.clamp(0, 1)
        dist = torch.linalg.norm(pred - expect.unsqueeze(1), dim=-1).mean(dim=-1)  # B,M

    scored = base
    if prev_xy is not None and temporal_w > 0 and dist is not None:
        if geom_gap_gate:
            # soft gate in [0,1]: gap << thr → ~1; gap >> thr → ~0
            gate = torch.sigmoid((geom_gap_thr - gap) * geom_gap_sharp).unsqueeze(-1)
            scored = base - (temporal_w * gate) * dist
        else:
            scored = base - temporal_w * dist
        scored = scored.masked_fill(~valid, -1e9)

    best = scored.argmax(dim=-1)

    # Jump rescue: confident-wrong geom that left the track; stay on a real triple.
    if jump_rescue and prev_xy is not None and expect is not None and dist is not None:
        for bi in range(peaks.shape[0]):
            gi = int(base[bi].masked_fill(~valid[bi], -1e9).argmax().item())
            if not bool(valid[bi, gi].item()):
                continue
            d_geom = float(dist[bi, gi].item())
            if d_geom < jump_thr:
                continue
            best_logit = float(base[bi, gi].item())
            # candidates: near track and not much worse than geom-best
            ok = valid[bi] & (dist[bi] <= jump_max_dist) & (base[bi] >= best_logit - jump_logit_slack)
            if not bool(ok.any().item()):
                continue
            # prefer highest geom logit among near-track (not pure nearest)
            cand = base[bi].clone()
            cand = cand.masked_fill(~ok, -1e9)
            best[bi] = int(cand.argmax().item())

    out = _gather_triple_coords(peaks, idxs, best, valid=valid)
    none = ~valid.any(dim=-1)
    if none.any():
        out[none] = peaks[none][:, :, 0, :2].clamp(0, 1)
    if heat is not None:
        out = refine_peak_local(heat, out)
    return out


@torch.no_grad()
def top1_shape_and_margin(peaks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """shape of top1 triple + min per-color (score0 - score1) margin. B"""
    p0 = peaks[:, :, 0, :]  # B,3,3
    pr, pg, pb = p0[:, 0], p0[:, 1], p0[:, 2]
    e01 = torch.linalg.norm(pr[:, :2] - pg[:, :2], dim=-1)
    e12 = torch.linalg.norm(pg[:, :2] - pb[:, :2], dim=-1)
    e20 = torch.linalg.norm(pb[:, :2] - pr[:, :2], dim=-1)
    mean = (e01 + e12 + e20) / 3.0 + 1e-6
    shape = (torch.abs(e01 - mean) + torch.abs(e12 - mean) + torch.abs(e20 - mean)) / mean
    # margin: how dominant is peak0 vs peak1
    s0 = peaks[:, :, 0, 2].clamp_min(0)
    s1 = peaks[:, :, 1, 2].clone()
    s1 = torch.where(s1 < 0, torch.zeros_like(s1), s1)
    margin = (s0 - s1).min(dim=-1).values
    return shape, margin


@torch.no_grad()
def snap_to_nearest_peak(
    peaks: torch.Tensor,
    xy: torch.Tensor,
    max_dist: float = 0.08,
) -> torch.Tensor:
    """For each color, snap xy to nearest TopK peak if within max_dist. peaks B,3,K,3; xy B,3,2."""
    b, c, k, _ = peaks.shape
    out = xy.clone()
    for bi in range(b):
        for ci in range(c):
            valid = peaks[bi, ci, :, 2] >= 0
            if not bool(valid.any().item()):
                continue
            pts = peaks[bi, ci, valid, :2]
            d = torch.linalg.norm(pts - xy[bi, ci].unsqueeze(0), dim=-1)
            j = int(d.argmin().item())
            if float(d[j].item()) <= max_dist:
                out[bi, ci] = pts[j]
    return out


@torch.no_grad()
def append_expect_peaks(
    peaks: torch.Tensor,
    heat: torch.Tensor,
    expect: torch.Tensor,
    heat_frac: float = 0.45,
    min_sep: float = 0.02,
    score_floor: float = 0.0,
) -> torch.Tensor:
    """Append constant-velocity expect as one extra TopK slot per color.

    Only when local heat is at least ``heat_frac`` of that color's top peak
    and expect is not already a peak. Score is sigmoid(heat) at expect so
    GeomTripleNet can pick it; empty occlusions stay out.
    peaks: B,3,K,3  heat: B,3,H,W logits  expect: B,3,2 → B,3,K+1,3
    """
    b, c, k, _ = peaks.shape
    _, _, h, w = heat.shape
    extra = torch.full((b, c, 1, 3), -1.0, device=peaks.device, dtype=peaks.dtype)
    for bi in range(b):
        for ci in range(c):
            xy = expect[bi, ci].clamp(0, 1)
            valid = peaks[bi, ci, :, 2] >= 0
            if bool(valid.any().item()):
                d = torch.linalg.norm(peaks[bi, ci, valid, :2] - xy, dim=-1)
                if float(d.min().item()) <= min_sep:
                    continue
                top = float(peaks[bi, ci, 0, 2].clamp_min(0).item())
            else:
                top = 0.0
            px = int(torch.round(xy[0] * (w - 1)).clamp(0, w - 1).item())
            py = int(torch.round(xy[1] * (h - 1)).clamp(0, h - 1).item())
            score = float(torch.sigmoid(heat[bi, ci, py, px]).item())
            if score < heat_frac * max(top, 1e-3):
                continue
            if score_floor > 0:
                score = max(score, score_floor * top)
            extra[bi, ci, 0, 0] = xy[0]
            extra[bi, ci, 0, 1] = xy[1]
            extra[bi, ci, 0, 2] = score
    return torch.cat([peaks, extra], dim=2)


@torch.no_grad()
def associate_velocity(
    heat: torch.Tensor,
    prev_xy: Optional[torch.Tensor],
    prev2_xy: Optional[torch.Tensor],
    coast_age: Optional[torch.Tensor] = None,
    top_k: int = 5,
    nms: int = 5,
    snap: float = 0.05,
    heat_frac: float = 0.45,
    max_coast: int = 2,
    vel_w: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Online constant-velocity association on heatmap peaks.

    heat: B,3,H,W. For each color, take the hottest Top-K peak that lies near
    expect = prev + vel and still has score >= heat_frac * top. If none does
    and the expected pixel itself has enough heat, hold that point for at most
    max_coast frames. Otherwise keep the current top peak.
    """
    peaks = extract_heatmap_peaks(heat, top_k=top_k, nms=nms)
    b, _, k, _ = peaks.shape
    device = heat.device
    out = peaks[:, :, 0, :2].clamp(0, 1).clone()
    if coast_age is None:
        coast_age = torch.zeros(b, 3, device=device, dtype=torch.long)
    else:
        coast_age = coast_age.clone()
    if prev_xy is None:
        return out, coast_age

    expect = prev_xy
    if prev2_xy is not None and vel_w > 0:
        expect = (prev_xy + vel_w * (prev_xy - prev2_xy)).clamp(0, 1)
    _, _, hh, ww = heat.shape
    for bi in range(b):
        for ci in range(3):
            top_score = float(peaks[bi, ci, 0, 2].clamp_min(0).item())
            valid = peaks[bi, ci, :, 2] >= 0
            picked = False
            if bool(valid.any().item()) and top_score > 0:
                pts = peaks[bi, ci, valid, :2]
                scores = peaks[bi, ci, valid, 2]
                dist = torch.linalg.norm(pts - expect[bi, ci], dim=-1)
                near = dist <= snap
                good = near & (scores >= heat_frac * top_score)
                if bool(good.any().item()):
                    # hottest peak on the track, not the globally hottest distractor
                    choice = scores.clone()
                    choice = choice.masked_fill(~good, -1.0)
                    j = int(choice.argmax().item())
                    out[bi, ci] = pts[j]
                    coast_age[bi, ci] = 0
                    picked = True
            if picked:
                continue
            xy = expect[bi, ci].clamp(0, 1)
            px = int(torch.round(xy[0] * (ww - 1)).clamp(0, ww - 1).item())
            py = int(torch.round(xy[1] * (hh - 1)).clamp(0, hh - 1).item())
            local = float(torch.sigmoid(heat[bi, ci, py, px]).item())
            age = int(coast_age[bi, ci].item())
            if local >= heat_frac * max(top_score, 1e-3) and age < max_coast:
                out[bi, ci] = xy
                coast_age[bi, ci] = age + 1
            else:
                coast_age[bi, ci] = 0
    return out, coast_age


@torch.no_grad()
def decode_find_geom(
    heat: torch.Tensor,
    geom: Optional[GeomTripleNet] = None,
    top_k: int = 5,
    upsample: int = 1,
    nms: int = 5,
    fast_shape_thr: float = 0.12,
    fast_margin_thr: float = 0.08,
    force_geom: bool = False,
    prev_xy: Optional[torch.Tensor] = None,
    temporal_w: float = 3.0,
    sharpen: float = 1.0,
    prev2_xy: Optional[torch.Tensor] = None,
    vel_w: float = 1.0,
    heat_prior_w: float = 0.0,
    heat_prior_sigma: float = 0.04,
    temporal_conf_gate: bool = False,
    temporal_conf_thr: float = 0.05,
    coast_margin_thr: float = -1.0,
    coast_snap: float = 0.08,
    geom_gap_gate: bool = False,
    geom_gap_thr: float = 1.0,
    geom_gap_sharp: float = 4.0,
    jump_rescue: bool = False,
    jump_thr: float = 0.08,
    jump_logit_slack: float = 1.0,
    jump_max_dist: float = 0.05,
    repeak_on_gap: bool = False,
    repeak_gap_thr: float = 1.0,
    repeak_prior_w: float = 1.5,
    repeak_prior_sigma: float = 0.035,
    repeak_evidence: bool = False,
    repeak_evidence_thr: float = 0.0,
    repeak_evidence_win: int = 5,
    inject_expect: bool = False,
    inject_heat_frac: float = 0.45,
    inject_score_floor: float = 0.0,
    heat_hold: bool = False,
    heat_hold_jump: float = 0.06,
    heat_hold_frac: float = 0.45,
    heat_hold_snap: float = 0.04,
) -> torch.Tensor:
    """
    Full decode: upsample → peaks → (fast top1 | geom/analytic) → local refine.
    heat: B,3,H,W → coords B,3,2 in [0,1]

    Temporal (no retrain):
      geom_gap_gate — scale temporal_w by geom best−2nd logit gap (preferred)
      jump_rescue — if geom jumps off-track, pick near-track competitive triple
      repeak_on_gap — if first-pass geom gap is low, bump heat at velocity
        expect and re-extract TopK (pulls true LEDs back into candidates),
        then re-run geom+temporal.
      repeak_evidence — only bump colors where local heat at expect already
        exceeds repeak_evidence_thr (don't invent peaks on empty occlusion).
      temporal_conf_gate — legacy peak-score margin (avoid; prefer geom_gap_gate)
      coast_margin_thr — legacy blind coast (keep <0)
    """
    hm0 = upsample_heat(heat * sharpen, upsample)
    # unconditional prior (legacy); repeak path applies prior only when ambiguous
    hm = hm0
    if prev_xy is not None and heat_prior_w > 0 and not repeak_on_gap:
        hm = apply_temporal_heat_prior(hm, prev_xy, strength=heat_prior_w, sigma=heat_prior_sigma)
    nms_u = nms * max(upsample, 1)
    if nms_u % 2 == 0:
        nms_u += 1
    peaks = extract_heatmap_peaks(hm, top_k=top_k, nms=nms_u)
    if inject_expect and prev_xy is not None:
        expect_xy = prev_xy
        if prev2_xy is not None and vel_w > 0:
            expect_xy = (prev_xy + vel_w * (prev_xy - prev2_xy)).clamp(0, 1)
        peaks = append_expect_peaks(
            peaks, hm, expect_xy,
            heat_frac=inject_heat_frac, score_floor=inject_score_floor,
        )
    shape, margin = top1_shape_and_margin(peaks)
    use_fast = (~force_geom) & (shape < fast_shape_thr) & (margin > fast_margin_thr)
    top1_ok = (peaks[:, :, 0, 2] >= 0).all(dim=-1)
    use_fast = use_fast & top1_ok
    out = torch.zeros(heat.shape[0], 3, 2, device=heat.device, dtype=heat.dtype)
    for bi in range(heat.shape[0]):
        prev_b = None if prev_xy is None else prev_xy[bi : bi + 1]
        prev2_b = None if prev2_xy is None else prev2_xy[bi : bi + 1]
        if (
            coast_margin_thr >= 0
            and prev_b is not None
            and float(margin[bi].item()) < float(coast_margin_thr)
        ):
            expect = prev_b.clone()
            if prev2_b is not None and vel_w > 0:
                expect = prev_b + vel_w * (prev_b - prev2_b)
            expect = expect.clamp(0, 1)
            if coast_snap > 0:
                expect = snap_to_nearest_peak(peaks[bi : bi + 1], expect, max_dist=coast_snap)
            out[bi] = expect[0]
            continue
        tw = float(temporal_w)
        if (not geom_gap_gate) and temporal_conf_gate and prev_b is not None and tw > 0:
            conf = torch.sigmoid((margin[bi] - temporal_conf_thr) * 20.0).item()
            tw = tw * (1.0 - conf)

        peaks_b = peaks[bi : bi + 1]
        hm_b = hm[bi : bi + 1]
        did_repeak = False

        # Two-pass: only when geom ranking is ambiguous AND we have a track.
        if repeak_on_gap and prev_b is not None and geom is not None:
            logits0, _, valid0 = geom.score_peaks(peaks_b)
            gap0 = float(geom_triple_gap(logits0, valid0)[0].item())
            if gap0 < repeak_gap_thr:
                expect = prev_b.clone()
                if prev2_b is not None and vel_w > 0:
                    expect = prev_b + vel_w * (prev_b - prev2_b)
                expect = expect.clamp(0, 1)
                hm_src = hm0[bi : bi + 1]
                if repeak_evidence:
                    # per-color: only boost channels with local heat evidence
                    _, _, hh, ww = hm_src.shape
                    half = max(1, repeak_evidence_win // 2)
                    mask = torch.zeros(1, 3, 1, 1, device=hm_src.device, dtype=hm_src.dtype)
                    for ci in range(3):
                        px = int(torch.round(expect[0, ci, 0] * (ww - 1)).clamp(0, ww - 1).item())
                        py = int(torch.round(expect[0, ci, 1] * (hh - 1)).clamp(0, hh - 1).item())
                        y0, y1 = max(0, py - half), min(hh, py + half + 1)
                        x0, x1 = max(0, px - half), min(ww, px + half + 1)
                        local = float(hm_src[0, ci, y0:y1, x0:x1].amax().item())
                        if local >= repeak_evidence_thr:
                            mask[0, ci, 0, 0] = 1.0
                    if float(mask.sum().item()) < 1.0:
                        pass  # no evidence → skip repeak
                    else:
                        # build prior then zero out colors without evidence
                        hm_rp = apply_temporal_heat_prior(
                            hm_src, expect,
                            strength=repeak_prior_w, sigma=repeak_prior_sigma,
                        )
                        # restore non-evidence channels to original
                        hm_rp = torch.where(mask.bool().expand_as(hm_rp), hm_rp, hm_src)
                        peaks_b = extract_heatmap_peaks(hm_rp, top_k=top_k, nms=nms_u)
                        hm_b = hm_rp
                        did_repeak = True
                else:
                    hm_rp = apply_temporal_heat_prior(
                        hm_src, expect,
                        strength=repeak_prior_w, sigma=repeak_prior_sigma,
                    )
                    peaks_b = extract_heatmap_peaks(hm_rp, top_k=top_k, nms=nms_u)
                    hm_b = hm_rp
                    did_repeak = True

        if bool(use_fast[bi].item()) and not did_repeak:
            out[bi] = refine_peak_local(hm[bi], peaks[bi, :, 0, :2].clamp(0, 1))
        elif geom is not None:
            out[bi] = select_triangle_learned(
                geom, peaks_b, heat=hm_b,
                prev_xy=prev_b, temporal_w=tw,
                prev2_xy=prev2_b, vel_w=vel_w,
                geom_gap_gate=geom_gap_gate,
                geom_gap_thr=geom_gap_thr,
                geom_gap_sharp=geom_gap_sharp,
                jump_rescue=jump_rescue,
                jump_thr=jump_thr,
                jump_logit_slack=jump_logit_slack,
                jump_max_dist=jump_max_dist,
            )[0]
        else:
            out[bi] = select_triangle_analytic(peaks_b, heat=hm_b)[0]

        # Heat-gated hold: if geom jumped off-track, snap back ONLY onto a
        # real TopK peak near the velocity expect. Never write raw coast coords
        # (that collapsed scores when diffuse heat lit empty track points).
        if heat_hold and prev_b is not None:
            expect = prev_b.clone()
            if prev2_b is not None and vel_w > 0:
                expect = (prev_b + vel_w * (prev_b - prev2_b)).clamp(0, 1)
            for ci in range(3):
                jump = float(torch.linalg.norm(out[bi, ci] - expect[0, ci]).item())
                if jump <= heat_hold_jump:
                    continue
                valid = peaks_b[0, ci, :, 2] >= 0
                if not bool(valid.any().item()):
                    continue
                pts = peaks_b[0, ci, valid, :2]
                scores = peaks_b[0, ci, valid, 2]
                d = torch.linalg.norm(pts - expect[0, ci], dim=-1)
                j = int(d.argmin().item())
                if float(d[j].item()) > heat_hold_snap:
                    continue
                top = float(peaks_b[0, ci, 0, 2].clamp_min(0).item())
                if float(scores[j].item()) < heat_hold_frac * max(top, 1e-3):
                    continue
                out[bi, ci] = pts[j]
    return out


def best_triple_label(
    peaks: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_vis: torch.Tensor,
    thr: float = 0.04,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Label best triple index by mean L2 to GT (normalized)."""
    feats, idxs, valid = _triple_features(peaks)
    b, m, _ = feats.shape
    # vectorized distance to GT for all triples
    ir, ig, ib = idxs[0, :, 0], idxs[0, :, 1], idxs[0, :, 2]
    # rebuild pred coords B,M,3,2
    pred = torch.stack(
        [
            peaks[:, 0, ir, :2],
            peaks[:, 1, ig, :2],
            peaks[:, 2, ib, :2],
        ],
        dim=2,
    )  # B,M,3,2
    gt = gt_coords.unsqueeze(1).expand_as(pred)
    err = torch.linalg.norm(pred - gt, dim=-1).mean(dim=-1)  # B,M
    err = err.masked_fill(~valid, 1e9)
    best_e, best_i = err.min(dim=-1)
    vis_ok = (gt_vis > 0.9).all(dim=-1)
    ok = vis_ok & (best_e <= thr)
    return best_i, ok
