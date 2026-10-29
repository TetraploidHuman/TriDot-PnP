"""Temporal completion of triangle corners the detector cannot see.

Why
---
30% of the frames in the hard/realistic splits do **not** have all three dots
visible (hands, desks, distractor clutter).  Every metric and every downstream
3-point-PnP path treats such a frame as a total failure, even though the app
could keep reporting a pose for a few frames by remembering where the missing
corner must be: the target is rigid and metric, so "where is the blue dot?"
is fully answered by "where was the target one frame ago, and how fast is it
moving?".

This module is the smallest honest version of that idea:

* state is updated **only on frames where all three dots were detected with
  high visibility**, so the predictor never feeds on its own output;
* the missing corner is projected with a constant-velocity prediction of the
  target's translation (``t + v * gap``) using the last good rotation;
* a hard ``max_gap`` stops the prediction after a few frames, so a stale pose
  cannot coast forever;
* nothing here changes the *observed* corners.

Everything the caller learns about the completion is returned explicitly
(which colours were filled in, whether the frame had state), so evaluation can
report completion accuracy separately instead of folding it into one number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .app_pnp import (COLOR_ORDER, fixed_solve_depths, object_points_mm,
                      solve_pose_from_depths)

Point = Tuple[float, float]


@dataclass
class CompletionOutput:
    """Points to use for PnP/metrics, plus what had to be invented."""

    points: Dict[str, Point]
    completed: List[str]
    reason: str = "ok"


class TemporalCompleter:
    """Predict missing triangle corners from the last fully-observed pose.

    Parameters
    ----------
    vis_thr:
        A corner with predicted visibility below this is treated as missing
        (the detector's visibility head, not the GT, decides).
    max_gap:
        Frames since the last fully-observed pose after which no prediction is
        made (a stale pose is worse than an honest failure).
    max_missing:
        Do not "complete" more than this many corners; completing two of three
        is no longer an observation.
    edge_margin:
        Predicted pixels must land inside the image by this many pixels.
    """

    def __init__(self, vis_thr: float = 0.35, max_gap: int = 8,
                 max_missing: int = 1, edge_margin: float = 4.0,
                 min_depth_mm: float = 80.0) -> None:
        self.vis_thr = float(vis_thr)
        self.max_gap = int(max_gap)
        self.max_missing = int(max_missing)
        self.edge_margin = float(edge_margin)
        self.min_depth_mm = float(min_depth_mm)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._r: Optional[np.ndarray] = None
        self._t: Optional[np.ndarray] = None
        self._t_prev: Optional[np.ndarray] = None
        self._frame: Optional[int] = None
        self._prev_frame: Optional[int] = None
        self.n_updates = 0

    @property
    def initialized(self) -> bool:
        return self._r is not None

    # ------------------------------------------------------------------ #
    def _update_state(self, points: Dict[str, Point], cam: dict,
                      edge_mm: float, frame_index: int) -> bool:
        """Try to advance the pose from a fully-observed frame."""
        uv = np.array([points[name] for name in COLOR_ORDER], dtype=np.float64)
        res = fixed_solve_depths(uv, edge_mm, cam["fx"], cam["fy"],
                                 cam["cx"], cam["cy"])
        if not res.converged or res.max_abs_edge_err_mm > 2.0:
            return False
        _cam_pts, rot, t = solve_pose_from_depths(
            uv, res.depths_mm, edge_mm, cam["fx"], cam["fy"], cam["cx"], cam["cy"])
        if float(np.min(res.depths_mm)) < self.min_depth_mm:
            return False
        # keep the previous velocity pair: (prev_frame, t_prev) -> (frame, t)
        if self._frame is not None:
            self._t_prev = None if self._t is None else self._t.copy()
            self._prev_frame = int(self._frame)
        self._r = rot
        self._t = np.asarray(t, dtype=np.float64)
        self._frame = int(frame_index)
        self.n_updates += 1
        return True

    # ------------------------------------------------------------------ #
    def complete(self, points: Dict[str, Point], vis, cam: dict,
                 edge_mm: float, image_w: int, image_h: int,
                 frame_index: int,
                 missing: Optional[List[int]] = None) -> CompletionOutput:
        """Predict the missing corners of this frame.

        ``missing`` lets the caller override the visibility-based decision with
        its own evidence (e.g. "the colour window around blue is empty"), which
        is what the app can actually observe; ``vis`` stays useful for the
        plain visibility-head path and for tests.
        """
        vis = [float(x) for x in np.asarray(vis).reshape(-1)]
        if len(vis) != len(COLOR_ORDER):
            raise ValueError(f"need {len(COLOR_ORDER)} visibility values, got {len(vis)}")
        if missing is None:
            missing = [i for i, v in enumerate(vis) if v < self.vis_thr]
        else:
            missing = sorted({int(i) for i in missing})

        if not missing:
            self._update_state(points, cam, edge_mm, frame_index)
            return CompletionOutput(dict(points), [], "observed")

        if not self.initialized or self._frame is None:
            return CompletionOutput(dict(points), [], "no-state")
        gap_obs = int(frame_index) - int(self._frame)
        if gap_obs > self.max_gap:
            return CompletionOutput(dict(points), [], "stale")
        if len(missing) > self.max_missing:
            return CompletionOutput(dict(points), [], "too-many-missing")

        # constant-velocity prediction of the translation, in mm / frame
        t_pred = np.array(self._t, dtype=np.float64)
        if self._t_prev is not None and self._prev_frame is not None:
            span = max(1, int(self._frame) - int(self._prev_frame))
            vel = (self._t - self._t_prev) / float(span)
            t_pred = self._t + vel * float(max(0, gap_obs))

        obj = object_points_mm(edge_mm)
        out = dict(points)
        filled: List[str] = []
        for i in missing:
            world = obj[i]
            cam_pt = self._r @ world + t_pred
            if cam_pt[2] <= self.min_depth_mm:
                continue
            u = cam["fx"] * cam_pt[0] / cam_pt[2] + cam["cx"]
            v = cam["fy"] * cam_pt[1] / cam_pt[2] + cam["cy"]
            if not (self.edge_margin <= u <= image_w - self.edge_margin
                    and self.edge_margin <= v <= image_h - self.edge_margin):
                continue
            out[COLOR_ORDER[i]] = (float(u), float(v))
            filled.append(COLOR_ORDER[i])

        if not filled:
            return CompletionOutput(dict(points), [], "offscreen")
        return CompletionOutput(out, filled, "completed")
