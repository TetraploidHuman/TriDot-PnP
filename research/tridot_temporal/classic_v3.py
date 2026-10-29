"""Classic CV v3: contrast-normalised, scale-adaptive dot front-end.

Motivated by the far-distance failure analysis (see
``research/results/classic_cv2/REPORT.md``): v2 gates candidates on *absolute*
brightness (``peak_v``/``peak`` thresholds and a "0.22 * p99.5" cut on the
colour-difference map), which a real phone pipeline breaks from both ends:

* chroma subsampling + ISP denoise shrink a 2-3 px coloured dot so its
  absolute peak collapses (and its colour difference with it);
* background glare / warm textures raise the per-frame percentile, pushing the
  threshold above the dot at the same time.

v3 replaces the absolute gate with *local contrast* (top-hat at the expected
dot scale, thresholded against robust noise), which is invariant to exposure,
white balance and background level.  The remaining changes:

* **A** local contrast instead of absolute peak: ``resp = max_k tophat_k(m)``,
  ``thr = max(floor, k_sigma * 1.4826 * median|resp|)``; no brightness gate.
* **B** scale adaptive: when a temporal pose is available the expected dot
  radius in pixels is ``r_px = ratio * edge_px`` (``edge_px = fx * edge / z``),
  so the top-hat kernel and the centroid window follow distance.  Without a
  pose the response is the pixelwise max over a small multi-scale pyramid.
* **C** tiny-dot subpixel: 1-3 px blobs are refined by upsampling the local
  colour-weight patch (x4, cubic) and taking its weighted centroid.
* **D** colour assignment fallback is left to the caller (see ``colorless``).

The triple scoring / temporal selection / pose bookkeeping are inherited from
:class:`ClassicV2Detector` unchanged, so v2 and v3 are directly comparable.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from research.tridot_temporal.classic_v2 import (
    COLOR_ORDER, Blob, ClassicV2Detector, DetectResult, color_weight_maps, _subpixel,
)


def _odd(k: float) -> int:
    k = int(round(k))
    if k % 2 == 0:
        k += 1
    return max(3, k)


def _upsampled_centroid(bgr: np.ndarray, m: np.ndarray, u: float, v: float,
                        radius: float, factor: int = 4) -> Tuple[float, float]:
    """Subpixel centroid of a tiny blob on an x-factor upsampled patch."""
    h, w = m.shape
    r = max(2.0, radius)
    x0, x1 = int(math.floor(u - r)), int(math.ceil(u + r)) + 1
    y0, y1 = int(math.floor(v - r)), int(math.ceil(v + r)) + 1
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return u, v
    patch = m[y0:y1, x0:x1].astype(np.float32)
    if float(patch.max()) <= 0.0:
        return u, v
    up = cv2.resize(patch, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    up = np.clip(up, 0.0, None)
    if up.sum() <= 1e-6:
        return u, v
    yy, xx = np.mgrid[0:up.shape[0], 0:up.shape[1]]
    cu = float((xx * up).sum() / up.sum())
    cv_ = float((yy * up).sum() / up.sum())
    return x0 + cu / factor, y0 + cv_ / factor


class ClassicV3Detector(ClassicV2Detector):
    """v2 with contrast-based detection + scale adaptation (+ tiny-dot subpixel)."""

    def __init__(self, *args, contrast: bool = True, k_sigma: float = 3.5,
                 contrast_floor: float = 12.0, dot_radius_ratio: float = 0.09,
                 scales: Tuple[int, ...] = (3, 7, 15, 31),
                 max_bg_ksize: int = 31, roi_upsample: bool = True,
                 roi_factor: int = 4, merge_frac: float = 0.45,
                 upsample_mode: str = "tiny",
                 frontend: str = "union", colorless: bool = True,
                 max_colorless: int = 3, response_mode: str = "contrast",
                 rel_frac: float = 0.18, threshold_mode: str = "percentile",
                 tile_px: int = 32, tile_percentile: float = 25.0,
                 prior_gate: bool = False, prior_gate_frac: float = 0.55,
                 prior_gate_quick: bool = True,
                 prior_gate_mode: str = "prune",
                 mf_floor: float = 6.0, mf_k: float = 3.0,
                 mf_rel: float = 0.10, mf_max: int = 6, circ_w: float = 0.0,
                 dot_cls_path: Optional[str] = None, dot_cls_w: float = 0.5,
                 dot_cls_floor: float = 0.0, dot_cls_scale: bool = False,
                 dot_cls_gate_area: int = 0,
                 **kwargs):
        # v2's tophat stage is replaced by the contrast response below
        kwargs.setdefault("tophat", False)
        super().__init__(*args, **kwargs)
        self.contrast = contrast
        self.k_sigma = k_sigma
        self.contrast_floor = contrast_floor
        self.dot_radius_ratio = dot_radius_ratio
        self.scales = tuple(scales)
        self.max_bg_ksize = max_bg_ksize
        self.roi_upsample = roi_upsample
        self.roi_factor = roi_factor
        # "tiny": 只对 <=3px 像素的连通域做 4x 上采样质心（历史行为）
        # "all":  对每个候选的峰值窗口都做 4x 上采样质心（C 的完整形态，成本 ~2x 质心）
        # "off":  只用窗口质心
        self.upsample_mode = upsample_mode
        self.merge_frac = merge_frac
        # "contrast": only the local-contrast front-end
        # "union": contrast candidates + the v2 absolute-threshold candidates;
        #          the geometric/photometric triple score picks the winner, so a
        #          frame where one front-end is blind still has a chance
        self.frontend = frontend
        # D: colour assignment fallback.  When chroma subsampling / ISP denoise
        # erase a dot's colour signature the per-colour map goes dark; the dot
        # may still be visible as a bright spot.  If a colour has no candidate
        # we take bright spots from the max-over-colours map and let the
        # geometric triple score pick one, with a weak colour-plausibility veto.
        self.colorless = colorless
        self.max_colorless = max_colorless
        # "contrast": always use the top-hat response (degraded-friendly)
        # "raw": always use the colour-difference map (clean-scene behaviour)
        # "adaptive": raw on dark low-clutter frames, contrast when the map has
        #             a bright / noisy background.  This is a per-frame SNR
        #             decision (like v2's adaptive top-hat), not a scene prior.
        self.response_mode = response_mode
        # relative floor w.r.t. the frame's own bright tail: keeps mid-bright
        # clutter from flooding the candidate list on clean frames.
        self.rel_frac = rel_frac
        # "percentile": thr = max(floor, k_sigma*global MAD, rel_frac*p99.5)
        #   -> the global MAD tracks *content* sharpness, so a sharper capture
        #      (e.g. no sensor downscale) inflates the threshold and the
        #      detector loses dots.  This is the coupling that made the
        #      capture-side "improvements" look harmful in the knob A/B.
        # "tile": estimate the noise floor from the *quietest* image tiles
        #   (p25 of per-tile MAD) so structured clutter/edges do not raise it.
        # "tile_rel": tile noise for the absolute floor plus the frame's own
        #   bright tail as a relative guard.
        self.threshold_mode = threshold_mode
        self.tile_px = int(tile_px)
        self.tile_percentile = float(tile_percentile)
        # Round 7: candidate-level temporal gating.  When a tracked pose exists,
        # a candidate more than ``prior_gate_frac * edge_px`` away from the
        # pose-predicted pixel is almost certainly clutter (the rigid object
        # cannot teleport).  Prune those *before* the exhaustive triple search,
        # with a fallback to the unpruned list so a stale prediction cannot
        # cause a hard miss.  This targets what the threshold study showed to be
        # the real bottleneck: candidate quality, not the threshold.
        self.prior_gate = bool(prior_gate)
        self.prior_gate_frac = float(prior_gate_frac)
        # pruning on the cheap "quick" pass can accept a wrong near-prediction
        # triple early; experiments run both settings
        self.prior_gate_quick = bool(prior_gate_quick)
        # "prune": search only the pruned lists (fall back if no triple)
        # "dual": search pruned *and* full, keep the cheaper triple - the
        #         temporal term already rewards being near the prediction, so a
        #         pruned-only search can lock onto a plausible-but-wrong triple
        self.prior_gate_mode = str(prior_gate_mode)
        # Round 9: matched-filter (disc) candidates; shape/occlusion robust.
        self.mf_floor = float(mf_floor)
        self.mf_k = float(mf_k)
        self.mf_rel = float(mf_rel)
        self.mf_max = int(mf_max)
        # Round 9: reward disc-like blobs (LED dots) over rectangular
        # distractors in the triple score.  Circularity lives on Blob and was
        # only used in the per-candidate score before.
        self.circ_w = float(circ_w)
        # Round 10: optional tiny CNN that re-scores candidates on a 15x15
        # colour-map patch (dot vs distractor block).  Lazily loaded so the
        # default path stays free of torch.
        self.dot_cls_path = dot_cls_path
        self.dot_cls_w = float(dot_cls_w)
        self.dot_cls_floor = float(dot_cls_floor)
        self.dot_cls_scale = bool(dot_cls_scale)
        self.dot_cls_gate_area = int(dot_cls_gate_area)
        self._dot_scorer = None
        self._maps: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------ scale
    def expected_radius_px(self) -> Optional[float]:
        """Expected dot radius in pixels from the last tracked edge length."""
        if self.edge_px is None or self.edge_px <= 0:
            return None
        return max(1.0, float(self.dot_radius_ratio) * float(self.edge_px))

    def _bg_ksizes(self) -> Tuple[int, ...]:
        r = self.expected_radius_px()
        if r is not None:
            # the top-hat kernel must be a little larger than the dot halo
            k = _odd(2.0 * (2.0 * r) + 1.0)
            return (min(k, self.max_bg_ksize),)
        return tuple(min(_odd(k), self.max_bg_ksize) for k in self.scales)

    def _merge_radius(self) -> float:
        r = self.expected_radius_px()
        if r is None:
            return 20.0
        return float(np.clip(self.merge_frac * 3.0 * r, 6.0, 20.0))

    # --------------------------------------------------------- D: colour fallback
    def _colour_plausible(self, color: str, u: float, v: float) -> bool:
        """Veto a spot that is clearly dominated by another colour."""
        if self._maps is None:
            return True
        h, w = self._maps[color].shape
        r = 3
        x0, x1 = max(0, int(u) - r), min(w, int(u) + r + 1)
        y0, y1 = max(0, int(v) - r), min(h, int(v) + r + 1)
        if x1 <= x0 or y1 <= y0:
            return False
        own = float(self._maps[color][y0:y1, x0:x1].max())
        for other, m in self._maps.items():
            if other == color:
                continue
            oth = float(m[y0:y1, x0:x1].max())
            if oth >= 40.0 and oth >= 1.6 * own:
                return False
        return True

    def _colourless_candidates(self, bgr: np.ndarray) -> List[Blob]:
        # brightness/chroma magnitude: survives when the colour *difference*
        # is gone (chroma subsampling, desaturation, white-clipped core)
        if bgr is None:
            return []
        luma = bgr.max(axis=2).astype(np.uint8)
        return self._extract(luma, "colourless")

    def _after_candidates(self, cands, maps, bgr) -> None:
        self._maps = maps
        if not self.colorless:
            return
        missing = [c for c in COLOR_ORDER if not cands[c]]
        if not missing:
            return
        extra = self._colourless_candidates(bgr)[:self.max_colorless]
        if not extra:
            return
        for c in missing:
            picked: List[Blob] = []
            for b in extra:
                if not self._colour_plausible(c, b.u, b.v):
                    continue
                picked.append(Blob(color=c, u=b.u, v=b.v, area=b.area,
                                   peak=b.peak, score=0.9 * b.score,
                                   aspect=b.aspect, circularity=b.circularity))
                if len(picked) >= self.top_k:
                    break
            if picked:
                cands[c] = picked

    # ------------------------------------------------------------------ selection
    def _best_triple(self, cands, pred_uv=None, pred_edge=None, accept=2.2):
        if not (self.prior_gate and pred_uv is not None
                and (self.prior_gate_quick or accept >= 1.9)):
            return super()._best_triple(cands, pred_uv, pred_edge, accept)
        gate = max(10.0, self.prior_gate_frac * float(pred_edge or 40.0))
        pruned = {}
        for c in COLOR_ORDER:
            keep = []
            px, py = pred_uv[c]
            for b in cands.get(c, []):
                if math.hypot(b.u - px, b.v - py) <= gate:
                    keep.append(b)
            pruned[c] = keep
        if all(pruned.get(c) for c in COLOR_ORDER):
            chosen_p = super()._best_triple(pruned, pred_uv, pred_edge, accept)
            if chosen_p is not None:
                if self.prior_gate_mode != "dual":
                    return chosen_p
                chosen_f = super()._best_triple(cands, pred_uv, pred_edge, accept)
                if chosen_f is None:
                    return chosen_p
                cp = self._score_triple(chosen_p, pred_uv, pred_edge)
                cf = self._score_triple(chosen_f, pred_uv, pred_edge)
                return chosen_p if cp <= cf else chosen_f
        # stale / wrong prediction: fall back to the full candidate lists
        return super()._best_triple(cands, pred_uv, pred_edge, accept)

    # ------------------------------------------------------------------ front-end
    def _contrast_response(self, m: np.ndarray) -> np.ndarray:
        best: Optional[np.ndarray] = None
        for k in self._bg_ksizes():
            kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            resp = cv2.morphologyEx(m, cv2.MORPH_TOPHAT, kern)
            if best is None:
                best = resp
            else:
                np.maximum(best, resp, out=best)
        if best is None:
            best = np.zeros_like(m)
        return best

    def _noise_sigma(self, resp: np.ndarray) -> float:
        sub = resp[::4, ::4].astype(np.float32)
        if sub.size == 0:
            return 0.0
        med = float(np.median(sub))
        mad = float(np.median(np.abs(sub - med)))
        return 1.4826 * mad

    def _tile_noise_sigma(self, resp: np.ndarray) -> float:
        """Noise floor from the quietest tiles (robust to structured clutter)."""
        h, w = resp.shape
        tp = max(8, self.tile_px)
        mads: List[float] = []
        for y0 in range(0, h, tp):
            for x0 in range(0, w, tp):
                tile = resp[y0:y0 + tp, x0:x0 + tp]
                if tile.size < 16:
                    continue
                med = float(np.median(tile))
                mads.append(1.4826 * float(np.median(np.abs(tile - med))))
        if not mads:
            return 0.0
        return float(np.percentile(np.asarray(mads), self.tile_percentile))

    def _local_noise_sigma(self, resp: np.ndarray) -> np.ndarray:
        """Per-pixel noise estimate (SNR thresholding).

        ``residual = resp - median_blur(resp)`` removes the smooth part of the
        response (which for a top-hat map is mostly the local mean); its local
        RMS is an estimate of the noise level at each pixel.  Thresholding
        ``resp / sigma_local`` is scale free: a sensor-noise-heavy capture does
        not lose the dot, it just raises the bar uniformly.
        """
        med = cv2.medianBlur(resp.astype(np.float32), 5)
        resid = resp.astype(np.float32) - med
        var = cv2.GaussianBlur(resid * resid, (0, 0), 3.0)
        return np.sqrt(np.maximum(var, 0.0)) + 1.0

    def _use_contrast(self, m: np.ndarray) -> bool:
        if not self.contrast:
            return False
        if self.response_mode == "contrast":
            return True
        if self.response_mode == "raw":
            return False
        sub = m[::4, ::4]
        p995 = float(np.percentile(sub, 99.5)) if sub.size else 0.0
        p50 = float(np.percentile(sub, 50)) if sub.size else 0.0
        return p995 > 15.0 or p50 > 2.0

    def _ensure_dot_scorer(self):
        if self._dot_scorer is None and self.dot_cls_path:
            from research.tridot_temporal.dot_cls import DotScorer
            self._dot_scorer = DotScorer(self.dot_cls_path)
        return self._dot_scorer

    def _rerank_with_dot_cls(self, m: np.ndarray, kept: List[Blob]) -> List[Blob]:
        scorer = self._ensure_dot_scorer()
        if scorer is None or not kept:
            return kept
        if self.dot_cls_scale:
            from research.tridot_temporal.dot_cls import patch_from_map_scaled
            r_exp = self.expected_radius_px() or 0.0
            patches = np.stack([
                patch_from_map_scaled(m, b.u, b.v,
                                      r_exp if r_exp > 0 else float(np.sqrt(max(b.area, 1) / np.pi)))
                for b in kept])
        else:
            from research.tridot_temporal.dot_cls import patch_from_map
            patches = np.stack([patch_from_map(m, b.u, b.v) for b in kept])
        probs = scorer.predict(patches)
        out: List[Blob] = []
        for b, p in zip(kept, probs):
            p = float(p)
            if p < self.dot_cls_floor:
                continue
            if self.dot_cls_gate_area > 0 and b.area < self.dot_cls_gate_area:
                # tiny candidates: the 15x15 content is mostly noise, keep the
                # classic score instead of trusting the CNN
                out.append(b)
                continue
            b.score = (1.0 - self.dot_cls_w) * float(b.score) + self.dot_cls_w * p
            out.append(b)
        out.sort(key=lambda b: -b.score)
        return out

    def _score_triple(self, tri, pred, edge_px):
        cost = super()._score_triple(tri, pred, edge_px)
        if cost >= 1e8 or self.circ_w <= 0.0:
            return cost
        circ = (float(tri["red"].circularity) + float(tri["green"].circularity)
                + float(tri["blue"].circularity)) / 3.0
        return cost - self.circ_w * circ

    def _mf_candidates(self, m: np.ndarray, radius: float) -> List[Blob]:
        """Shape-free matched-filter peaks (partial dots, 1-2 px dots)."""
        h, w = m.shape
        mf = m.astype(np.float32)
        mp = float(mf.max())
        if mp <= 0.0:
            return []
        scale = 1.0 / mp
        r = float(np.clip(radius, 1.0, 12.0))
        sig = max(0.6, r * 0.6)
        resp = cv2.GaussianBlur(mf * scale, (0, 0), sig) * mp
        sub = resp[::4, ::4]
        med = float(np.median(sub))
        mad = float(np.median(np.abs(sub - med)))
        sigma = 1.4826 * mad
        p995 = float(np.percentile(sub, 99.5))
        thr = max(self.mf_floor, self.mf_k * sigma, self.mf_rel * p995)
        dil = cv2.dilate(resp, np.ones((3, 3), np.uint8))
        mask = ((resp >= thr) & (resp >= dil)).astype(np.uint8)
        if int(mask.sum()) < 1:
            return []
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return []
        out: List[Blob] = []
        order = np.argsort(-resp[labels > 0].reshape(-1)) if False else None
        peaks = []
        for i in range(1, n):
            x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
            bwj = int(stats[i, cv2.CC_STAT_WIDTH]); bhj = int(stats[i, cv2.CC_STAT_HEIGHT])
            peak = float(resp[y:y + bhj, x:x + bwj].max())
            peaks.append((peak, i))
        peaks.sort(reverse=True)
        merge_r = self._merge_radius()
        kept: List[Blob] = []
        for peak, i in peaks[: 4 * self.mf_max]:
            cx0, cy0 = float(cents[i][0]), float(cents[i][1])
            rad = int(np.clip(1.6 * r, 2, 14))
            u, v = _subpixel(None, m, cx0, cy0, radius=rad)
            if any(np.hypot(u - k2.u, v - k2.v) <= max(4.0, merge_r) for k2 in kept):
                continue
            area = int(stats[i, cv2.CC_STAT_AREA])
            kept.append(Blob(color="", u=u, v=v, area=max(area, 1), peak=peak,
                             score=0.60 * min(peak / 255.0, 1.0) + 0.40 * area,
                             aspect=1.0, circularity=1.0))
            if len(kept) >= self.mf_max:
                break
        return kept

    def _extract(self, m: np.ndarray, color: str,
                 bgr: Optional[np.ndarray] = None) -> List[Blob]:
        h, w = m.shape
        use_contrast = self._use_contrast(m)
        resp = self._contrast_response(m) if use_contrast else m
        sub = m[::4, ::4]
        p995 = float(np.percentile(sub, 99.5)) if sub.size else 0.0
        if use_contrast:
            if self.threshold_mode == "tile":
                sigma = self._tile_noise_sigma(resp)
                thr = max(self.contrast_floor, self.k_sigma * sigma)
            elif self.threshold_mode == "tile_rel":
                sigma = self._tile_noise_sigma(resp)
                thr = max(self.contrast_floor, self.k_sigma * sigma,
                          self.rel_frac * p995)
            elif self.threshold_mode == "snr":
                sigma = 0.0
                thr = self.contrast_floor
            else:  # "percentile"
                sigma = self._noise_sigma(resp)
                thr = max(self.contrast_floor, self.k_sigma * sigma,
                          self.rel_frac * p995)
        else:
            thr = max(self.thr_floor, 0.22 * p995)
        if use_contrast and self.threshold_mode == "snr":
            z = resp / self._local_noise_sigma(resp)
            mask = (z >= self.k_sigma).astype(np.uint8)
        else:
            mask = (resp >= thr).astype(np.uint8)
        if int(mask.sum()) < 1:
            return []
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return []
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
        bw = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
        bh = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
        max_area = int(self.max_area_frac * h * w)
        aspect = np.maximum(bw, bh) / np.maximum(1.0, np.minimum(bw, bh))
        valid = (areas >= 1) & (areas <= max_area) & (aspect <= self.max_aspect)
        idx = np.nonzero(valid)[0]
        if idx.size == 0:
            return []
        # rank by contrast peak (NOT by area: far dots are small but sharp)
        peaks = np.zeros(idx.size, dtype=np.float32)
        for j, i in enumerate(idx):
            x = int(stats[i + 1, cv2.CC_STAT_LEFT])
            y = int(stats[i + 1, cv2.CC_STAT_TOP])
            bwj = int(stats[i + 1, cv2.CC_STAT_WIDTH])
            bhj = int(stats[i + 1, cv2.CC_STAT_HEIGHT])
            peaks[j] = float(resp[y:y + bhj, x:x + bwj].max())
        keep_n = min(32, idx.size)
        order = idx[np.argsort(-peaks)][:keep_n]
        circ = 4.0 * np.pi * areas[order] / np.maximum(
            1e-6, (2.0 * (bw[order] + bh[order])) ** 2)
        circ = np.minimum(circ, 1.0)
        r_expected = self.expected_radius_px()
        out: List[Blob] = []
        for k, i in enumerate(order):
            cx0, cy0 = float(cents[i + 1][0]), float(cents[i + 1][1])
            area = int(areas[i])
            want_up = (self.upsample_mode == "all") or (
                self.upsample_mode == "tiny" and self.roi_upsample and area <= 3)
            if want_up:
                u, v = _upsampled_centroid(bgr, m, cx0, cy0,
                                           max(2.0, (r_expected or 2.0) * 1.5),
                                           factor=self.roi_factor)
            else:
                if r_expected is not None:
                    rad = int(np.clip(2.2 * r_expected, 3, 20))
                else:
                    rad = int(max(6.0, min(20.0, 1.6 * max(bw[i], bh[i]) / 2.0 + 3)))
                u, v = _subpixel(None, m, cx0, cy0, radius=rad)
            x0, x1 = max(0, int(u) - 2), min(w, int(u) + 3)
            y0, y1 = max(0, int(v) - 2), min(h, int(v) + 3)
            peak = float(m[y0:y1, x0:x1].max()) if y1 > y0 and x1 > x0 else 0.0
            peak_v = float(self._v[max(0, int(v) - 4):int(v) + 5,
                                   max(0, int(u) - 4):int(u) + 5].max()) \
                if self._v is not None else 0.0
            score = 0.62 * (peak / 255.0) + 0.30 * float(circ[k]) + \
                0.08 * float(min(peak_v / 255.0, 1.0))
            out.append(Blob(color=color, u=u, v=v, area=area, peak=peak,
                            score=score, aspect=float(aspect[i]),
                            circularity=float(circ[k])))
        out.sort(key=lambda b: b.score, reverse=True)
        merge_r = self._merge_radius()
        kept: List[Blob] = []
        for b in out:
            if all(np.hypot(b.u - k2.u, b.v - k2.v) > merge_r for k2 in kept):
                kept.append(b)
                if len(kept) >= self.top_k:
                    break
        if self.frontend == "fallback":
            # only when the contrast front-end found nothing: the absolute
            # threshold front-end still sees big/bright blobs that a top-hat
            # might have flattened
            if not out:
                for b in super()._extract(m, color, bgr):
                    if all(np.hypot(b.u - k2.u, b.v - k2.v) > max(6.0, merge_r)
                           for k2 in kept):
                        kept.append(b)
                        if len(kept) >= self.top_k:
                            break
        if self.frontend == "union_mf":
            r_exp = self.expected_radius_px() or 0.0
            for b in self._mf_candidates(m, max(1.0, r_exp if r_exp else 2.0)):
                if all(np.hypot(b.u - k2.u, b.v - k2.v) > max(6.0, merge_r)
                       for k2 in kept):
                    kept.append(b)
                    if len(kept) >= 2 * self.top_k:
                        break
            for b in kept:
                b.color = color
        if self.frontend in ("union", "union_mf"):
            # absolute-threshold candidates (v2 front-end) as a second
            # hypothesis set; scores are not compared across front-ends, both
            # lists are handed to the triple scorer
            abs_cands = super()._extract(m, color, bgr)
            for b in abs_cands:
                if all(np.hypot(b.u - k2.u, b.v - k2.v) > max(6.0, merge_r)
                       for k2 in kept):
                    kept.append(b)
                    if len(kept) >= 2 * self.top_k:
                        break
        return self._rerank_with_dot_cls(m, kept)
