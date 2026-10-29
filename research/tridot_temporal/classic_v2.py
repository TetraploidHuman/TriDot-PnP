"""Classic-CV detector v2: colour-difference front-end + temporal/geometric triple selection.

Why this exists
---------------
The B0 diagnostic (`research/scripts/b0_diag.py`) showed that on `hard_test`:
  * 53% of all-visible frames lose at least one colour at the *front-end*
    (fixed HSV ranges, distractor blobs, bright/screen backgrounds), and
  * 22% of frames find all three colours but pick the wrong triple
    (clutter 49%, desk 37%, crosscolor 26%).
Only 24.5% succeed, while an "oracle association" would reach 46.9%.

This module replaces the two brittle stages:
  1. front-end: per-pixel colour *difference* maps (r - max(g, b), ...), adaptive
     threshold, connected components with a shape/area filter and a weighted
     sub-pixel centroid.  No fixed hue/saturation/value windows.
  2. selection: exhaustive triple search scored by photometry, equilateral /
     edge-length consistency, *size consistency* (same LED seen at the same
     distance) and, when available, the **temporal prediction** from the last
     accepted pose (constant-velocity translation, constant rotation).

The output mirrors `research.baseline_b0.detector.DetectResult` closely enough
for `eval_v2.evaluate_classic2` to consume it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

COLOR_ORDER = ("red", "green", "blue")


@dataclass
class Blob:
    color: str
    u: float
    v: float
    area: int
    peak: float          # peak of the colour-difference response
    score: float         # combined photometric score
    aspect: float        # bbox aspect ratio (>= 1)
    circularity: float


@dataclass
class DetectResult:
    ok: bool
    points: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    combo_score: float = 0.0
    reason: str = ""
    candidates: Dict[str, List[Blob]] = field(default_factory=dict)


def color_weight_maps(bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Signed colour-difference maps (uint8, 0..255)."""
    b = bgr[..., 0]
    g = bgr[..., 1]
    r = bgr[..., 2]
    out = {}
    for name, ch, others in (("red", r, (g, b)), ("green", g, (r, b)),
                             ("blue", b, (r, g))):
        d = ch.astype(np.int16) - np.maximum(others[0], others[1])
        out[name] = np.clip(d, 0, 255).astype(np.uint8)
    return out


def _subpixel(bgr: np.ndarray, m: np.ndarray, u: float, v: float,
              radius: int = 9) -> Tuple[float, float]:
    h, w = m.shape
    x0, x1 = max(0, int(u) - radius), min(w, int(u) + radius + 1)
    y0, y1 = max(0, int(v) - radius), min(h, int(v) + radius + 1)
    win = m[y0:y1, x0:x1]
    if win.size == 0 or win.sum() <= 1e-6:
        return u, v
    yy, xx = np.mgrid[y0:y1, x0:x1]
    wgt = win.astype(np.float64)
    return (float((xx * wgt).sum() / wgt.sum()),
            float((yy * wgt).sum() / wgt.sum()))


class ClassicV2Detector:
    """Colour-difference detector with temporal triple selection."""

    def __init__(self, top_k: int = 6, thr_floor: float = 8.0,
                 min_area: int = 3, max_area_frac: float = 0.04,
                 max_aspect: float = 6.0, temporal_gate: float = 0.6,
                 edge_mm: float = 100.0, union_b0: bool = True,
                 tophat: bool = True, tophat_ksize: int = 31):
        self.top_k = top_k
        self.thr_floor = thr_floor
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.max_aspect = max_aspect
        self.temporal_gate = temporal_gate   # in units of the predicted edge (px)
        self.edge_mm = edge_mm
        self.union_b0 = union_b0
        self.tophat = tophat
        self.tophat_ksize = tophat_ksize
        self._b0 = None
        self._v: Optional[np.ndarray] = None
        # temporal state
        self.R: Optional[np.ndarray] = None   # target -> camera (3x3)
        self.t: Optional[np.ndarray] = None   # mm
        self.t_prev: Optional[np.ndarray] = None
        self.frame: int = 0
        self.prev_frame: Optional[int] = None
        self.edge_px: Optional[float] = None

    # ------------------------------------------------------------------ utils
    def reset(self) -> None:
        self.R = self.t = self.t_prev = None
        self.frame = 0
        self.prev_frame = None
        self.edge_px = None

    def _blob_peak(self, m: np.ndarray, u: float, v: float) -> Tuple[float, float]:
        """(colour-difference peak, luminance peak) in a small window."""
        h, w = m.shape
        x0, x1 = max(0, int(u) - 4), min(w, int(u) + 5)
        y0, y1 = max(0, int(v) - 4), min(h, int(v) + 5)
        return float(m[y0:y1, x0:x1].max()), float(self._v[y0:y1, x0:x1].max())

    def _merge_b0_candidates(self, bgr: np.ndarray,
                             maps: Dict[str, np.ndarray],
                             cands: Dict[str, List[Blob]]) -> None:
        """Union with the B0 HSV front-end.

        B0's fixed HSV windows are surprisingly good on washed-out indoor
        scenes (room/light) where the colour-difference map is swamped by a
        warm, low-saturation background.  Keeping both front-ends and letting
        the triple search arbitrate is strictly more robust than either.
        """
        if self._b0 is None:
            from research.baseline_b0.detector import B0Detector
            self._b0 = B0Detector(top_k=self.top_k * 2)
        else:
            self._b0.reset()
        # half resolution is plenty for *seeding* association; the sub-pixel
        # position then comes from the full-resolution colour map
        small = cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        prop = self._b0.propose(small, top_k=self.top_k * 2)
        for c in COLOR_ORDER:
            existing = cands[c]
            for cand in prop.get(c, []):
                u, v = float(cand.u) * 2.0, float(cand.v) * 2.0
                if any(np.hypot(u - b.u, v - b.v) <= 8.0 for b in existing):
                    continue
                m = maps[c]
                rad = int(max(8.0, min(24.0, math.sqrt(max(1.0, cand.area)) * 2.4 + 4)))
                u, v = _subpixel(None, m, u, v, radius=rad)
                peak_cd, peak_v = self._blob_peak(m, u, v)
                circ = float(min(max(cand.circularity, 0.0), 1.0))
                existing.append(Blob(
                    color=c, u=u, v=v, area=int(max(1, cand.area)) * 4,
                    peak=peak_cd,
                    score=0.34 * peak_cd / 255.0 + 0.32 * peak_v / 255.0
                          + 0.18 * circ + 0.16 * float(min(max(cand.score, 0.0), 1.0)),
                    aspect=1.0, circularity=circ))
            existing.sort(key=lambda b: b.score, reverse=True)
            del existing[self.top_k:]

    def _extract(self, m: np.ndarray, color: str, bgr: Optional[np.ndarray] = None) -> List[Blob]:
        h, w = m.shape
        sub = m[::4, ::4]
        thr = max(self.thr_floor, 0.22 * float(np.percentile(sub, 99.5)))
        mask = (m >= thr).astype(np.uint8)
        if int(mask.sum()) < self.min_area:
            return []
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((2, 2), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        if n <= 1:
            return []
        # vectorised filtering: colour-cast frames can produce >100k tiny
        # components, so never loop over all of them in Python
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
        bw = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
        bh = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
        max_area = int(self.max_area_frac * h * w)
        aspect = np.maximum(bw, bh) / np.maximum(1.0, np.minimum(bw, bh))
        valid = (areas >= self.min_area) & (areas <= max_area) & (aspect <= self.max_aspect)
        idx = np.nonzero(valid)[0]
        if idx.size == 0:
            return []
        # rank by area (cheap proxy: the LED halo is one of the largest blobs)
        order = idx[np.argsort(-areas[idx])][:24]
        circ = 4.0 * np.pi * areas[order] / np.maximum(1e-6, (2.0 * (bw[order] + bh[order])) ** 2)
        circ = np.minimum(circ, 1.0)
        out: List[Blob] = []
        for k, i in enumerate(order):
            cx0, cy0 = float(cents[i + 1][0]), float(cents[i + 1][1])
            rad = int(max(6.0, min(20.0, 1.6 * max(bw[i], bh[i]) / 2.0 + 3)))
            u, v = _subpixel(None, m, cx0, cy0, radius=rad)
            x0, x1 = max(0, int(u) - 3), min(w, int(u) + 4)
            y0, y1 = max(0, int(v) - 3), min(h, int(v) + 4)
            peak = float(m[y0:y1, x0:x1].max()) if y1 > y0 and x1 > x0 else 0.0
            peak_v = float(self._v[max(0, int(v) - 4):int(v) + 5,
                                   max(0, int(u) - 4):int(u) + 5].max()) if self._v is not None else 0.0
            out.append(Blob(color=color, u=u, v=v, area=int(areas[i]),
                            peak=peak,
                            score=0.34 * peak / 255.0 + 0.32 * peak_v / 255.0
                                  + 0.18 * float(circ[k]) + 0.16 * (0.5 * float(circ[k]) + 0.5 * peak / 255.0),
                            aspect=float(aspect[i]),
                            circularity=float(circ[k])))
        # halo fragments of the same LED land on (almost) the same centre:
        # merge duplicates before ranking so a weak fragment never wins.
        out.sort(key=lambda b: b.score, reverse=True)
        kept: List[Blob] = []
        for b in out:
            if all(np.hypot(b.u - k.u, b.v - k.v) > 20.0 for k in kept):
                kept.append(b)
                if len(kept) >= self.top_k:
                    break
        return kept

    def _after_candidates(self, cands: Dict[str, List[Blob]],
                          maps: Dict[str, np.ndarray],
                          bgr: Optional[np.ndarray]) -> None:
        """Hook for subclasses (v3 uses it for colour assignment fallback)."""
        return None

    def _predict_pixels(self, fx: float, fy: float, cx: float, cy: float
                        ) -> Optional[Tuple[Dict[str, Tuple[float, float]], float]]:
        if self.R is None or self.t is None or self.prev_frame is None:
            return None
        gap = max(1, self.frame - self.prev_frame)
        if self.t_prev is not None:
            t_pred = self.t + (self.t - self.t_prev) / max(1, gap) * gap
        else:
            t_pred = self.t
        e = self.edge_mm
        obj = np.array([[0.0, 0.0, 0.0], [e, 0.0, 0.0],
                        [e / 2.0, e * np.sqrt(3.0) / 2.0, 0.0]])
        cam = (self.R @ obj.T).T + t_pred[None, :]
        if np.any(cam[:, 2] <= 1.0):
            return None
        uv = {}
        for i, color in enumerate(COLOR_ORDER):
            uv[color] = (fx * cam[i, 0] / cam[i, 2] + cx,
                         fy * cam[i, 1] / cam[i, 2] + cy)
        depth = float(cam[:, 2].mean())
        edge_px = float(fx * e / max(1.0, depth))
        return uv, edge_px

    def _score_triple(self, tri: Dict[str, Blob],
                      pred: Optional[Dict[str, Tuple[float, float]]],
                      edge_px: Optional[float]) -> float:
        ra, ga, ba = tri["red"], tri["green"], tri["blue"]
        e1 = math.hypot(ra.u - ga.u, ra.v - ga.v)
        e2 = math.hypot(ga.u - ba.u, ga.v - ba.v)
        e3 = math.hypot(ba.u - ra.u, ba.v - ra.v)
        mean_e = (e1 + e2 + e3) / 3.0
        if mean_e < 6.0:
            return 1e9
        emax = max(e1, e2, e3)
        emin = min(e1, e2, e3)
        shape = (emax - emin) / mean_e
        a_avg = (max(1, ra.area) + max(1, ga.area) + max(1, ba.area)) / 3.0
        size_pen = (abs(math.log(max(1, ra.area) / a_avg))
                    + abs(math.log(max(1, ga.area) / a_avg))
                    + abs(math.log(max(1, ba.area) / a_avg))) / 3.0
        phot = (ra.score + ga.score + ba.score) / 3.0
        cost = shape + 0.35 * size_pen - 2.4 * phot
        if edge_px:
            cost += 0.5 * abs(math.log(max(1.0, mean_e) / max(1.0, edge_px)))
        if pred is not None:
            gate = max(20.0, self.temporal_gate * (edge_px or mean_e))
            d = (min(3.0, math.hypot(ra.u - pred["red"][0], ra.v - pred["red"][1]) / gate)
                 + min(3.0, math.hypot(ga.u - pred["green"][0], ga.v - pred["green"][1]) / gate)
                 + min(3.0, math.hypot(ba.u - pred["blue"][0], ba.v - pred["blue"][1]) / gate))
            cost += 0.8 * (d / 3.0)
        return cost

    def _best_triple(self, cands: Dict[str, List[Blob]],
                     pred_uv: Optional[Dict[str, Tuple[float, float]]],
                     pred_edge: Optional[float],
                     accept: float) -> Optional[Dict[str, Blob]]:
        best: Optional[Dict[str, Blob]] = None
        for prior in ((pred_uv, pred_edge), (None, None)):
            best_cost, best_tri = 1e9, None
            for a in cands["red"]:
                for b in cands["green"]:
                    for c in cands["blue"]:
                        tri = {"red": a, "green": b, "blue": c}
                        cost = self._score_triple(tri, prior[0], prior[1])
                        if cost < best_cost:
                            best_cost, best_tri = cost, tri
            if best_tri is not None and best_cost <= accept:
                return best_tri
            if best is None:
                best = None
        return None

    # ------------------------------------------------------------------ detect
    def candidate_maps(self, bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """Per-colour difference maps with the optional adaptive top-hat."""
        maps = color_weight_maps(bgr)
        self._v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        if self.tophat:
            kern = cv2.getStructuringElement(cv2.MORPH_RECT,
                                             (self.tophat_ksize, self.tophat_ksize))
            # adaptive: only suppress the background of maps that actually have
            # a bright background (dark synthetic scenes need no top-hat and
            # lose a little accuracy when it is applied unconditionally)
            for c, m in list(maps.items()):
                sub = m[::4, ::4]
                if (float(np.percentile(sub, 99.5)) > 15.0
                        or float(np.percentile(sub, 50)) > 2.0):
                    maps[c] = cv2.morphologyEx(m, cv2.MORPH_TOPHAT, kern)
        return maps

    def candidates_from_maps(self, maps: Dict[str, np.ndarray],
                             bgr: np.ndarray) -> Dict[str, List[Blob]]:
        cands = {c: self._extract(maps[c], c) for c in COLOR_ORDER}
        self._after_candidates(cands, maps, bgr)
        return cands

    def propose(self, bgr: np.ndarray) -> Dict[str, List[Blob]]:
        """Per-colour candidate lists without selecting a triangle or a pose."""
        return self.candidates_from_maps(self.candidate_maps(bgr), bgr)

    def detect(self, bgr: np.ndarray, fx: float, fy: float, cx: float, cy: float,
               solver=None) -> DetectResult:
        h, w = bgr.shape[:2]
        maps = self.candidate_maps(bgr)
        cands = self.candidates_from_maps(maps, bgr)
        missing = [c for c in COLOR_ORDER if not cands[c]]
        note = "" if not missing else f"no-candidate:{','.join(missing)}"
        self.frame += 1

        pred = self._predict_pixels(fx, fy, cx, cy)
        pred_uv = pred[0] if pred else None
        pred_edge = pred[1] if pred else None

        # try the cheap colour-difference front-end alone first; the B0 HSV
        # union is only needed when that is missing a colour or unconvincing
        quick = None if missing else self._best_triple(cands, pred_uv, pred_edge, 1.4)
        if quick is not None:
            chosen = quick
        elif self.union_b0:
            self._merge_b0_candidates(bgr, maps, cands)
            missing = [c for c in COLOR_ORDER if not cands[c]]
            note = note if missing else ""

        # 1) fast path: temporal gate, one candidate per colour nearest to prediction
        chosen: Optional[Dict[str, Blob]] = None
        if False and pred_uv is not None:  # gated fast path disabled (see notes)
            near: Dict[str, Blob] = {}
            gate = max(18.0, self.temporal_gate * pred_edge)
            for c in COLOR_ORDER:
                if not cands[c]:
                    continue
                best = min(cands[c],
                           key=lambda b: np.hypot(b.u - pred_uv[c][0],
                                                  b.v - pred_uv[c][1]))
                if np.hypot(best.u - pred_uv[c][0],
                            best.v - pred_uv[c][1]) <= gate:
                    near[c] = best
            if len(near) == 3:
                # even a temporally plausible triple must look like the target
                if self._score_triple(near, None, pred_edge) <= 1.9:
                    chosen = near

        # 2) exhaustive triple search (photometry + geometry + size [+ temporal])
        if chosen is None and not missing:
            chosen = self._best_triple(cands, pred_uv, pred_edge, 2.2)
            if chosen is None:
                note = note or "low-combo-score"

        if chosen is None:
            self.prev_frame = None  # break the velocity chain
            return DetectResult(ok=False, reason=note or "no-triple",
                                candidates=cands)

        pts = {c: (chosen[c].u, chosen[c].v) for c in COLOR_ORDER}
        scores = {c: float(chosen[c].score) for c in COLOR_ORDER}
        combo = float(np.mean(list(scores.values())))
        # update temporal state through the caller-provided solver (optional)
        if solver is not None:
            pose = solver.solve(pts, self.edge_mm, fx, fy, cx, cy,
                                np.zeros(5))
            if getattr(pose, "ok", False):
                R = np.asarray(pose.rotation, dtype=float).reshape(3, 3)
                t = np.asarray(pose.tvec, dtype=float).reshape(3)
                if self.t is not None:
                    gap = max(1, self.frame - (self.prev_frame or self.frame))
                    if self.t_prev is not None:
                        t_pred = self.t + (self.t - self.t_prev) / gap * gap
                    else:
                        t_pred = self.t
                    depth = float(np.linalg.norm(self.t))
                    if (np.linalg.norm(t - t_pred) > max(150.0, 0.25 * depth)
                            and np.linalg.norm(t - t_pred) > 0.25 * depth):
                        # implausible jump -> drop temporal state, re-acquire next frame
                        self.R = self.t = self.t_prev = None
                        self.prev_frame = None
                        return DetectResult(ok=True, points=pts, scores=scores,
                                            combo_score=combo, reason="pose-jump",
                                            candidates=cands)
                self.t_prev = None if self.t is None else self.t.copy()
                self.R, self.t = R, t
                self.prev_frame = self.frame
                e = []
                for i in range(3):
                    a, b = COLOR_ORDER[i], COLOR_ORDER[(i + 1) % 3]
                    e.append(float(np.hypot(pts[a][0] - pts[b][0],
                                            pts[a][1] - pts[b][1])))
                self.edge_px = float(np.mean(e))
            else:
                self.prev_frame = None
        return DetectResult(ok=True, points=pts, scores=scores,
                            combo_score=combo, reason=note, candidates=cands)
