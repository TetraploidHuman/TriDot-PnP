"""Temporal filtering + three-point PnP root disambiguation for the phone app.

The app runs at ~30 fps on a moving target, so two cheap wins are available
before touching the detector:

1. **Root disambiguation.**  Three rays through the three dots plus the known
   equilateral edge lengths admit more than one depth solution (classic P3P
   ambiguity).  When no temporal prior exists the solver may land on the
   mirrored root, which barely changes the *distance* but flips the *pose*.
   ``solve_depth_candidates`` enumerates the valid roots with a multi-start
   damped Newton solve; the caller picks the root consistent with the previous
   track.
2. **Temporal smoothing.**  An alpha-beta (steady-state Kalman) filter on the
   target translation, with a physically motivated innovation gate: a
   measurement that jumps more than ``gate`` mm is treated as an outlier and
   the filter coasts on the constant-velocity prediction for up to
   ``max_coast`` frames.

Both pieces are pure geometry + arithmetic so the Kotlin port
(``PnPTemporalTracker.kt``) can mirror them line by line.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from research.tridot_temporal.app_pnp import (
    COLOR_ORDER,
    estimate_focal_length,
    fixed_solve_depths,
    solve_pose_from_depths,
)


# --------------------------------------------------------------------------- #
# root candidates
# --------------------------------------------------------------------------- #
def solve_depth_candidates(uv: np.ndarray, edge_mm: float, fx: float, fy: float,
                           cx: float, cy: float, *,
                           max_iters: int = 50, conv_mm: float = 0.05,
                           min_depth_mm: float = 100.0,
                           rel_dedup: float = 0.02) -> list[np.ndarray]:
    """All distinct valid depth solutions found from several initial guesses.

    Returns depth vectors sorted by mean depth (ascending).  The mirrored P3P
    root shows up as a second, geometrically valid solution: same edge lengths,
    different camera-frame pose.
    """
    uv = np.asarray(uv, dtype=np.float64)
    edge_len = float(edge_mm)
    scale = edge_len / 100.0

    # Image-derived scale: depth roughly edge / angular size.
    xn = (uv[:, 0] - cx) / fx
    yn = (uv[:, 1] - cy) / fy
    base = np.stack([xn, yn, np.ones(3)], axis=1)
    d1 = float(np.linalg.norm(base[0] - base[1]))
    d2 = float(np.linalg.norm(base[1] - base[2]))
    d3 = float(np.linalg.norm(base[2] - base[0]))
    typical = edge_len / max((d1 + d2 + d3) / 3.0, 1e-6)
    typical *= scale  # scale is 1 for a 100 mm edge; keeps units honest for others

    seeds = [
        typical * np.ones(3),
        typical * np.array([0.7, 0.85, 1.0]),
        typical * np.array([1.0, 0.85, 0.7]),
        typical * 2.0 * np.ones(3),
        typical * 0.4 * np.ones(3),
    ]
    out: list[np.ndarray] = []
    for seed in seeds:
        res = fixed_solve_depths(uv, edge_mm, fx, fy, cx, cy,
                                 max_iters=max_iters, conv_mm=conv_mm,
                                 min_depth_mm=min_depth_mm,
                                 init_depths_mm=seed)
        if not res.converged:
            continue
        z = res.depths_mm
        if not np.all(np.isfinite(z)) or z.min() <= 0:
            continue
        if any(np.max(np.abs(z - y) / np.maximum(y, 1e-6)) < rel_dedup for y in out):
            continue
        out.append(z.copy())
    out.sort(key=lambda z: float(z.mean()))
    return out


def candidate_translation(uv: np.ndarray, depths_mm: np.ndarray, edge_mm: float,
                          fx: float, fy: float, cx: float, cy: float):
    _, R, t = solve_pose_from_depths(uv, depths_mm, edge_mm, fx, fy, cx, cy)
    return R, t


# --------------------------------------------------------------------------- #
# alpha-beta tracker
# --------------------------------------------------------------------------- #
@dataclass
class TrackState:
    initialized: bool = False
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rot: np.ndarray | None = None
    last_t_s: float | None = None
    coast_count: int = 0
    accepted: int = 0
    rejected: int = 0


@dataclass
class TrackOutput:
    t_mm: np.ndarray
    rotation: np.ndarray | None
    distance_mm: float
    accepted: bool
    coasting: bool
    reinitialized: bool


class PoseTracker:
    """Alpha-beta filter on the target translation, gate on the innovation."""

    def __init__(self, alpha: float = 0.45, beta: float = 0.12,
                 gate_abs_mm: float = 25.0, gate_rel: float = 0.04,
                 max_coast: int = 5, max_dt_s: float = 0.25,
                 max_speed_mm_s: float = 4000.0,
                 rot_gate_deg: float = 30.0, smooth: bool = True) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gate_abs_mm = float(gate_abs_mm)
        self.gate_rel = float(gate_rel)
        self.max_coast = int(max_coast)
        self.max_dt_s = float(max_dt_s)
        self.max_speed_mm_s = float(max_speed_mm_s)
        self.rot_gate_deg = float(rot_gate_deg)
        # smooth=False keeps the internal alpha-beta state (prediction/velocity)
        # but reports the raw measurement: root disambiguation without lag.
        self.smooth = bool(smooth)
        self.state = TrackState()

    # -- lifecycle ------------------------------------------------------- #
    def reset(self) -> None:
        self.state = TrackState()

    def _reinit(self, t_meas: np.ndarray, rot_meas, t_s: float) -> TrackOutput:
        s = self.state
        s.initialized = True
        s.pos = np.asarray(t_meas, dtype=np.float64).copy()
        s.vel = np.zeros(3)
        s.rot = None if rot_meas is None else np.asarray(rot_meas, dtype=np.float64).copy()
        s.last_t_s = float(t_s)
        s.coast_count = 0
        return TrackOutput(s.pos.copy(), s.rot, float(np.linalg.norm(s.pos)),
                           accepted=True, coasting=False, reinitialized=True)

    def update(self, t_meas, rot_meas, t_s: float) -> TrackOutput:
        s = self.state
        t_meas = np.asarray(t_meas, dtype=np.float64).reshape(3)
        t_s = float(t_s)

        if not s.initialized or s.last_t_s is None:
            return self._reinit(t_meas, rot_meas, t_s)

        dt = t_s - s.last_t_s
        if dt <= 0.0 or dt > self.max_dt_s:
            self.state.accepted = s.accepted
            self.state.rejected = s.rejected
            return self._reinit(t_meas, rot_meas, t_s)

        pred = s.pos + s.vel * dt
        inno = t_meas - pred
        inno_norm = float(np.linalg.norm(inno))
        distance = float(np.linalg.norm(pred))
        gate = max(self.gate_abs_mm, self.gate_rel * distance)

        # A 2-fold planar-target ambiguity means the mirrored root can win a
        # single frame's candidate ranking.  Reject a measurement whose pose
        # branch disagrees with the locked one by more than the rotation gate;
        # only a sustained disagreement (``max_coast`` frames) may relock.
        rot_mismatch_deg = 0.0
        if rot_meas is not None and s.rot is not None:
            rot_mismatch_deg = rotation_angle_deg(
                np.asarray(rot_meas, dtype=np.float64), s.rot)

        s.last_t_s = t_s
        if inno_norm > gate or rot_mismatch_deg > self.rot_gate_deg:
            # Outlier: coast on the prediction, decay the velocity.
            s.coast_count += 1
            s.rejected += 1
            s.pos = pred
            s.vel = s.vel * 0.5
            if s.coast_count > self.max_coast:
                return self._reinit(t_meas, rot_meas, t_s)
            return TrackOutput(s.pos.copy(), s.rot, float(np.linalg.norm(s.pos)),
                               accepted=False, coasting=True, reinitialized=False)

        # Accepted measurement: alpha-beta update.
        s.coast_count = 0
        s.accepted += 1
        smoothed = pred + self.alpha * inno
        s.pos = smoothed
        s.vel = s.vel + (self.beta / dt) * inno
        speed = float(np.linalg.norm(s.vel))
        if speed > self.max_speed_mm_s:
            s.vel *= self.max_speed_mm_s / speed
        if rot_meas is not None:
            s.rot = np.asarray(rot_meas, dtype=np.float64).copy()
        out_t = smoothed if self.smooth else t_meas
        return TrackOutput(out_t.copy(), s.rot, float(np.linalg.norm(out_t)),
                           accepted=True, coasting=False, reinitialized=False)


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices, in degrees."""
    c = (np.trace(np.asarray(r_a).T @ np.asarray(r_b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


class AppTrackedSolver:
    """App camera model + root disambiguation + temporal filter (single object).

    This is the reference implementation of what the Kotlin
    ``PnPTemporalTracker`` does: each frame the detector supplies three pixels;
    we enumerate the valid depth roots, pick the one closest to the tracker's
    constant-velocity prediction, then feed it through the alpha-beta filter.
    """

    def __init__(self, fov_deg: float = 60.0, **tracker_kwargs) -> None:
        self.fov_deg = float(fov_deg)
        self.tracker = PoseTracker(**tracker_kwargs)
        self._last_candidates = 0

    def reset(self) -> None:
        self.tracker.reset()

    def solve(self, points: dict, edge_mm: float, image_w: int, image_h: int,
              t_s: float | None = None):
        from types import SimpleNamespace

        w, h = int(image_w), int(image_h)
        fx = estimate_focal_length(w, self.fov_deg)
        fy = fx
        cx, cy = w / 2.0, h / 2.0
        uv = np.array([[points[c][0], points[c][1]] for c in COLOR_ORDER],
                      dtype=np.float64)
        cands = solve_depth_candidates(uv, edge_mm, fx, fy, cx, cy)
        self._last_candidates = len(cands)
        if not cands:
            return SimpleNamespace(ok=False, translation_mm=None, rotation=None,
                                   distance_mm=None, snap=False, reason="no_root")

        if t_s is None:
            t_s = 0.0 if self.tracker.state.last_t_s is None else \
                self.tracker.state.last_t_s + 1.0 / 30.0
        s = self.tracker.state
        if s.initialized and s.last_t_s is not None:
            dt = max(float(t_s) - s.last_t_s, 1e-3)
            pred_t = s.pos + s.vel * dt
            scored = []
            for z in cands:
                R_c, t_c = candidate_translation(uv, z, edge_mm, fx, fy, cx, cy)
                d_t = float(np.linalg.norm(t_c - pred_t))
                d_r = (rotation_angle_deg(R_c, s.rot)
                       if s.rot is not None else 0.0)
                scored.append((d_r, d_t, z))
            # Pose continuity first: the two roots' translations differ by only
            # a few mm (below the noise floor) but their rotations differ by
            # tens of degrees, so the rotation is the discriminating signal.
            best = min(scored, key=lambda x: (x[0], x[1]))[2]
        else:
            # No temporal prior yet: three coplanar dots admit a genuine 2-fold
            # pose ambiguity (front/back mirror), so pixels alone cannot tell
            # the branches apart.  Use the physical prior that the board faces
            # the camera: pick the root whose rotation is closest to identity
            # (board plane roughly parallel to the image plane, board up axis
            # roughly camera up).  The tracker then locks onto that branch.
            identity = np.eye(3)
            best = min(cands, key=lambda z: rotation_angle_deg(
                candidate_translation(uv, z, edge_mm, fx, fy, cx, cy)[0], identity))

        R, t = candidate_translation(uv, best, edge_mm, fx, fy, cx, cy)
        tr = self.tracker.update(t, R, float(t_s))
        return SimpleNamespace(ok=True, translation_mm=tr.t_mm, rotation=tr.rotation,
                               distance_mm=tr.distance_mm, snap=tr.accepted,
                               coasting=tr.coasting, candidates=len(cands),
                               reason="tracked")
