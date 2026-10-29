"""Learned point finder: per-frame heatmaps, optional peak-context association."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.tridot_temporal.find_geom import extract_heatmap_peaks
from research.tridot_temporal.find_preprocess import FIND_IN_CH
from research.tridot_temporal.model import ConvBNAct, soft_argmax_2d

Variant = Literal["transformer", "gru"]


@torch.no_grad()
def decode_heat_local(heat: torch.Tensor, win: int = 9, temp: float = 8.0) -> torch.Tensor:
    """Hard argmax then local-window soft-argmax. heat B,K,H,W → B,K,2 in [0,1]."""
    b, k, h, w = heat.shape
    peak = heat.view(b, k, -1).argmax(dim=-1)
    py = torch.div(peak, w, rounding_mode="floor")
    px = peak - py * w
    half = win // 2
    coords = torch.zeros(b, k, 2, device=heat.device, dtype=heat.dtype)
    for bi in range(b):
        for ki in range(k):
            y0 = int(max(0, int(py[bi, ki]) - half))
            x0 = int(max(0, int(px[bi, ki]) - half))
            y1 = int(min(h, int(py[bi, ki]) + half + 1))
            x1 = int(min(w, int(px[bi, ki]) + half + 1))
            patch = heat[bi, ki, y0:y1, x0:x1] * temp
            prob = torch.softmax(patch.reshape(-1), dim=0).view(y1 - y0, x1 - x0)
            yy = torch.arange(y0, y1, device=heat.device, dtype=heat.dtype)
            xx = torch.arange(x0, x1, device=heat.device, dtype=heat.dtype)
            ys, xs = torch.meshgrid(yy, xx, indexing="ij")
            coords[bi, ki, 0] = (prob * xs).sum() / max(w - 1, 1)
            coords[bi, ki, 1] = (prob * ys).sum() / max(h - 1, 1)
    return coords


def sample_heat_at(heat: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """heat B,3,H,W logits, xy B,3,2 in [0,1] → B,3."""
    b = heat.shape[0]
    vals = []
    for ci in range(3):
        grid = xy[:, ci].view(b, 1, 1, 2) * 2.0 - 1.0
        v = F.grid_sample(
            heat[:, ci : ci + 1], grid, mode="bilinear", align_corners=True, padding_mode="border",
        )
        vals.append(v.view(b))
    return torch.stack(vals, dim=-1)


def gated_heat_bump(
    heat: torch.Tensor,
    xy: torch.Tensor,
    strength: float,
    sigma: float,
    logit_slack: float = 1.5,
) -> torch.Tensor:
    """Add a bump at xy only where that location is already a strong response."""
    if strength <= 0:
        return heat
    b, c, h, w = heat.shape
    yy = torch.linspace(0, 1, h, device=heat.device, dtype=heat.dtype).view(1, 1, h, 1)
    xx = torch.linspace(0, 1, w, device=heat.device, dtype=heat.dtype).view(1, 1, 1, w)
    px = xy[:, :, 0].view(b, c, 1, 1)
    py = xy[:, :, 1].view(b, c, 1, 1)
    sig2 = 2.0 * (sigma ** 2) + 1e-8
    gauss = torch.exp(-((xx - px) ** 2 + (yy - py) ** 2) / sig2)
    local = sample_heat_at(heat, xy)
    top = heat.amax(dim=(-2, -1))
    gate = (local >= top - logit_slack).to(heat.dtype).view(b, c, 1, 1)
    return heat + float(strength) * gauss * gate


class FindEncoder(nn.Module):
    def __init__(self, in_ch: int = FIND_IN_CH, out_ch: int = 72, stride: int = 4) -> None:
        super().__init__()
        self.stride = stride
        if stride != 4:
            raise ValueError("FindEncoder supports stride=4 only")
        self.net = nn.Sequential(
            ConvBNAct(in_ch, 32, stride=2, dw=False),
            ConvBNAct(32, 48, stride=2, dw=True),
            ConvBNAct(48, out_ch, stride=1, dw=True),
            ConvBNAct(out_ch, out_ch, stride=1, dw=True),
            ConvBNAct(out_ch, out_ch, stride=1, dw=True),
            ConvBNAct(out_ch, out_ch, stride=1, dw=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalHeatFuse(nn.Module):
    """Conv fusion of the current heatmap with the previous fused map."""

    def __init__(self, ch: int = 3, conf_gate: bool = True) -> None:
        super().__init__()
        self.conf_gate = bool(conf_gate)
        self.net = nn.Sequential(
            nn.Conv2d(ch * 2, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, ch, 1),
        )
        self.logit_gate = nn.Parameter(torch.tensor(-2.0))
        self.conf_scale = nn.Parameter(torch.tensor(0.35))
        self.conf_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, heat: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = heat.shape
        g = torch.sigmoid(self.logit_gate)
        outs = []
        prev = heat[:, 0]
        for i in range(t):
            curr = heat[:, i]
            if i == 0:
                fused = curr
            else:
                delta = self.net(torch.cat([curr, prev], dim=1))
                if self.conf_gate:
                    sharp = curr.amax(dim=(-2, -1), keepdim=True)
                    conf = torch.sigmoid(self.conf_scale * (sharp - self.conf_bias))
                    fused = curr + (g * (1.0 - conf)) * delta
                else:
                    fused = curr + g * delta
            outs.append(fused)
            prev = fused
        return torch.stack(outs, dim=1)


class PeakContextNet(nn.Module):
    """Pick one heatmap peak per color using a causal Transformer over peak tokens.

    Tokens are the Top-K peaks, not a soft-argmax point. A coast slot is added
    only when constant-velocity expectation still has heat.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        nlayers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        top_k: int = 5,
        seq_len: int = 5,
    ) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.seq_len = int(seq_len)
        self.in_proj = nn.Linear(3, d_model)
        self.time_emb = nn.Embedding(self.seq_len, d_model)
        self.slot_emb = nn.Embedding(self.top_k, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.score = nn.Linear(d_model, 1)
        self.dist_scale = nn.Parameter(torch.tensor(2.0))
        # coast logit = w * heat_at_expect + b; b<0 keeps empty tracks off
        self.coast_w = nn.Parameter(torch.tensor(1.5))
        self.coast_b = nn.Parameter(torch.tensor(-2.5))

    def forward(
        self, peaks: torch.Tensor, heat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        peaks: B,T,3,K,3 (x, y, score). heat: B,T,3,H,W logits.
        returns coords B,T,3,2, logits B,T,3,K+1, candidates B,T,3,K+1,2.
        """
        b, t, c, k, _ = peaks.shape
        if k != self.top_k:
            raise ValueError(f"peak K={k} != configured {self.top_k}")
        device = peaks.device
        xy = peaks[..., :2]
        score = peaks[..., 2]
        valid = score >= 0
        feat = torch.cat([xy, score.unsqueeze(-1).clamp(0, 1)], dim=-1)
        tok = self.in_proj(feat)
        time_i = torch.arange(t, device=device).clamp(max=self.seq_len - 1)
        slot_i = torch.arange(k, device=device)
        tok = tok + self.time_emb(time_i).view(1, t, 1, 1, -1)
        tok = tok + self.slot_emb(slot_i).view(1, 1, 1, k, -1)

        logits_out = xy.new_full((b, t, c, k + 1), -1e4)
        cand_out = xy.new_zeros(b, t, c, k + 1, 2)
        coords = xy.new_zeros(b, t, c, 2)
        memory: list[torch.Tensor] = []
        prev: Optional[torch.Tensor] = None
        prev2: Optional[torch.Tensor] = None

        for ti in range(t):
            cur = tok[:, ti]  # B,3,K,D
            memory.append(cur)
            mem = torch.stack(memory, dim=2)  # B,3,L,K,D with L=ti+1
            mem = mem.reshape(b, c, (ti + 1) * k, -1).reshape(b * c, (ti + 1) * k, -1)
            frames = torch.arange(ti + 1, device=device).repeat_interleave(k)
            # True blocks attention. A token may see its own frame and the past.
            causal = frames.unsqueeze(0) > frames.unsqueeze(1)
            # Same-frame is allowed; the comparison above is strict future-only.
            hidden = self.temporal(mem, mask=causal)
            cur_h = hidden[:, -k:].reshape(b, c, k, -1)
            logits = self.score(cur_h).squeeze(-1)
            logits = logits.masked_fill(~valid[:, ti], -1e4)

            cand_xy = xy[:, ti]
            if prev is not None:
                expect = prev if prev2 is None else (prev + (prev - prev2)).clamp(0, 1)
                dist = torch.linalg.norm(cand_xy - expect.unsqueeze(2), dim=-1)
                logits = logits - self.dist_scale.clamp(0, 8) * dist
                evidence = sample_heat_at(heat[:, ti], expect)
                coast = self.coast_w * evidence + self.coast_b
                # Empty / wiped frames: keep the track (frame_drop training).
                no_peak = ~valid[:, ti].any(dim=-1)
                coast = torch.where(no_peak, torch.full_like(coast, 8.0), coast)
                # Weak but nonzero heat at expect can still coast; pure noise cannot.
                coast = torch.where(
                    (evidence > -2.0) | no_peak, coast, torch.full_like(coast, -1e4),
                )
                logits = torch.cat([logits, coast.unsqueeze(-1)], dim=-1)
                cand_xy = torch.cat([cand_xy, expect.unsqueeze(2)], dim=2)
            else:
                logits = torch.cat([logits, logits.new_full((b, c, 1), -1e4)], dim=-1)
                pad = xy.new_zeros(b, c, 1, 2)
                cand_xy = torch.cat([cand_xy, pad], dim=2)

            finite = torch.isfinite(logits)
            logits = logits.masked_fill(~finite, -1e4)
            empty = ~finite.any(dim=-1, keepdim=True)
            fallback = torch.zeros_like(logits)
            fallback[..., 0] = 0.0
            fallback[..., 1:] = -1e4
            logits = torch.where(empty, fallback, logits)

            probs = torch.softmax(logits, dim=-1)
            if self.training:
                hard = F.one_hot(logits.argmax(dim=-1), logits.shape[-1]).to(probs.dtype)
                mix = hard + probs - probs.detach()
            else:
                mix = F.one_hot(logits.argmax(dim=-1), logits.shape[-1]).to(probs.dtype)
            chosen = (mix.unsqueeze(-1) * cand_xy.detach()).sum(dim=2)
            coords[:, ti] = chosen
            logits_out[:, ti] = logits
            cand_out[:, ti] = cand_xy
            prev2 = prev.detach() if prev is not None else None
            prev = chosen.detach()

        return coords.clamp(0, 1), logits_out, cand_out


@dataclass
class FindConfig:
    variant: Variant = "transformer"
    seq_len: int = 5
    d_model: int = 128
    nhead: int = 4
    nlayers: int = 2
    ffn_dim: int = 256
    dropout: float = 0.1
    img_h: int = 256
    img_w: int = 448
    feat_ch: int = 72
    in_ch: int = FIND_IN_CH
    heatmap_stride: int = 4
    heat_fuse: bool = False
    conf_gate: bool = True
    refine_scale: float = 0.06
    peak_assoc: bool = False
    peak_k: int = 5
    heat_writeback: bool = False
    writeback_w: float = 1.5
    writeback_sigma: float = 0.03
    offset_head: bool = False


class TriDotFindNet(nn.Module):
    """
    Per-frame heatmap finder.
    peak_assoc: causal Transformer over Top-K peaks selects the tracked peak.
    heat_writeback: bump the returned heatmap at the velocity expectation
    when that location already has heat, so geom decoding sees the track.
    """

    def __init__(self, cfg: Optional[FindConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or FindConfig()
        d = self.cfg.d_model
        self.encoder = FindEncoder(self.cfg.in_ch, self.cfg.feat_ch, self.cfg.heatmap_stride)
        self.heat_head = nn.Conv2d(self.cfg.feat_ch, 3, 1)
        self.aff_to_heat = nn.Conv2d(3, 3, 1, bias=False)
        with torch.no_grad():
            self.aff_to_heat.weight.zero_()
            for i in range(3):
                self.aff_to_heat.weight[i, i, 0, 0] = 3.0
        self.offset_head = (
            nn.Conv2d(self.cfg.feat_ch, 6, 1) if self.cfg.offset_head else None
        )
        if self.offset_head is not None:
            # Start as "cell center" (sigmoid(0)=0.5) and learn the residual.
            nn.init.zeros_(self.offset_head.weight)
            nn.init.zeros_(self.offset_head.bias)
        self.heat_fuse = (
            CausalHeatFuse(3, conf_gate=self.cfg.conf_gate) if self.cfg.heat_fuse else None
        )
        self.vis_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(self.cfg.feat_ch, 3),
        )
        self.token_proj = nn.Linear(6 + 3 + self.cfg.feat_ch, d)
        self.pos = nn.Embedding(self.cfg.seq_len, d)
        self.variant = self.cfg.variant
        if self.variant == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=self.cfg.nhead, dim_feedforward=self.cfg.ffn_dim,
                dropout=self.cfg.dropout, batch_first=True, activation="gelu", norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=self.cfg.nlayers)
        else:
            self.temporal = nn.GRU(d, d, num_layers=1, batch_first=True)
        self.refine_coord = nn.Sequential(nn.Linear(d, 64), nn.ReLU(inplace=True), nn.Linear(64, 6))
        self.refine_vis = nn.Linear(d, 3)
        self.refine_scale = float(self.cfg.refine_scale)
        self.peak_ctx = (
            PeakContextNet(
                d_model=d, nhead=self.cfg.nhead, nlayers=self.cfg.nlayers,
                ffn_dim=self.cfg.ffn_dim, dropout=self.cfg.dropout,
                top_k=self.cfg.peak_k, seq_len=self.cfg.seq_len,
            )
            if self.cfg.peak_assoc else None
        )
        self.last_pick: Optional[torch.Tensor] = None
        self.last_cand: Optional[torch.Tensor] = None
        self.last_off: Optional[torch.Tensor] = None

    def _writeback(self, heat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        out = heat.clone()
        t = heat.shape[1]
        for ti in range(1, t):
            prev = coords[:, ti - 1]
            if ti >= 2:
                expect = (prev + (prev - coords[:, ti - 2])).clamp(0, 1)
            else:
                expect = prev
            out[:, ti] = gated_heat_bump(
                out[:, ti], expect,
                strength=float(self.cfg.writeback_w),
                sigma=float(self.cfg.writeback_sigma),
            )
        return out

    def forward(self, x: torch.Tensor, return_heat: bool = False):
        b, t, c, h, w = x.shape
        x_bt = x.reshape(b * t, c, h, w)
        feat = self.encoder(x_bt)
        aff = F.avg_pool2d(x_bt[:, 3:6], kernel_size=self.cfg.heatmap_stride)
        heat = self.heat_head(feat) + self.aff_to_heat(aff)
        hh, ww = heat.shape[-2:]
        if self.offset_head is not None:
            # B*T,6,h,w → B,T,3,2,h,w ; values in (0,1) = position inside the cell
            self.last_off = torch.sigmoid(self.offset_head(feat)).view(b, t, 3, 2, hh, ww)
        else:
            self.last_off = None
        heat_bt = heat.view(b, t, 3, hh, ww)
        if self.heat_fuse is not None:
            heat_bt = self.heat_fuse(heat_bt)
        coords0 = soft_argmax_2d(heat_bt.reshape(b * t, 3, hh, ww)).view(b, t, 3, 2)
        vis0 = self.vis_head(feat).view(b, t, 3)
        self.last_pick = None
        self.last_cand = None

        if self.peak_ctx is not None:
            flat = heat_bt.reshape(b * t, 3, hh, ww)
            peaks = extract_heatmap_peaks(flat, top_k=self.cfg.peak_k, nms=5)
            peaks = peaks.view(b, t, 3, self.cfg.peak_k, 3)
            coords, self.last_pick, self.last_cand = self.peak_ctx(peaks, heat_bt)
            vis = vis0
        else:
            gap = F.adaptive_avg_pool2d(feat, 1).flatten(1).view(b, t, -1)
            token = torch.cat([coords0.reshape(b, t, 6), vis0, gap], dim=-1)
            feats = self.token_proj(token)
            idx = torch.arange(t, device=x.device).clamp(max=self.cfg.seq_len - 1)
            feats = feats + self.pos(idx).unsqueeze(0)
            if self.variant == "transformer":
                mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
                h_t = self.temporal(feats, mask=mask)
            else:
                h_t, _ = self.temporal(feats)
            dcoord = self.refine_coord(h_t).view(b, t, 3, 2)
            coords = torch.clamp(coords0 + self.refine_scale * torch.tanh(dcoord), 0.0, 1.0)
            vis = vis0 + self.refine_vis(h_t)

        heat_out = heat_bt
        if self.cfg.heat_writeback:
            heat_out = self._writeback(heat_bt, coords.detach())
        if return_heat:
            return coords, vis, heat_out, coords0
        return coords, vis

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_find_net(**kwargs) -> TriDotFindNet:
    fields = set(FindConfig.__dataclass_fields__)
    cfg = FindConfig(**{k: v for k, v in kwargs.items() if k in fields})
    return TriDotFindNet(cfg)


def load_find_checkpoint(path, device) -> TriDotFindNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    raw = ck.get("cfg") or {}
    if not isinstance(raw, dict):
        raw = vars(raw)
    fields = set(FindConfig.__dataclass_fields__)
    cfg = FindConfig(**{k: v for k, v in raw.items() if k in fields})
    model = TriDotFindNet(cfg).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"[find] load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()
    return model
