"""Audit + fixed version of the phone app's three-point PnP solver.

Why this module exists
----------------------
The shipped Android app solves three-point PnP itself
(``composeApp/src/androidMain/kotlin/org/example/tridotpnp/PnPDistanceCalculator.kt``)
instead of calling OpenCV.  Its ``solve3PointDepths`` estimates one depth per
dot by multiplicative update::

    z_i *= 1 + (d_target / d_current - 1) * learning_rate

That update is an undamped fixed-point iteration and, as this module shows, it
*diverges* on perfectly noiseless synthetic geometry: the pairwise 3-D
distances move away from the known 100 mm edge until every depth is clamped at
the 100 mm floor.  The app's DISTANCE readout is the mean norm of the three
recovered 3-D points, so a depth error of 2x shows up directly as a distance
error of 2x - regardless of how good the detector is.

``app_solve_depths`` is a line-faithful port of the Kotlin loop (the "current
app" reference); ``fixed_solve_depths`` replaces the update with a damped
Newton solve of the same three equations.  The comparison is the point: same
camera model, same inputs, only the algorithm differs.

Geometry
--------
Pinhole ray for pixel ``(u, v)`` is ``d = ((u-cx)/fx, (v-cy)/fy, 1)``.  With
per-dot perpendicular depth ``z_i`` the camera-frame point is ``X_i = z_i * d_i``
(exact, no small-angle approximation) and the constraints are
``||z_i d_i - z_j d_j|| = edge`` for the three known sides of the equilateral
target.  The Kotlin caller rebuilds points as ``(u*z, v*z, z)``, so the solver
must keep returning *depths*, not Euclidean ranges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

COLOR_ORDER = ("red", "green", "blue")


# --------------------------------------------------------------------------- #
# app-side helpers (faithful ports of the Kotlin code)
# --------------------------------------------------------------------------- #
def estimate_focal_length(image_width: int, fov_deg: float = 60.0) -> float:
    """Kotlin ``estimateFocalLength``: f = w / (2 tan(fov/2))."""
    fov = float(np.clip(fov_deg, 10.0, 170.0))
    return image_width / (2.0 * np.tan(np.radians(fov) / 2.0))


@dataclass
class DepthSolveResult:
    depths_mm: np.ndarray
    converged: bool
    iterations: int
    max_abs_edge_err_mm: float


def _edge_dists(xn: np.ndarray, yn: np.ndarray, depths: np.ndarray) -> np.ndarray:
    P = depths[:, None] * np.stack([xn, yn, np.ones(3)], axis=1)
    return np.array([
        np.linalg.norm(P[1] - P[0]),
        np.linalg.norm(P[2] - P[1]),
        np.linalg.norm(P[0] - P[2]),
    ])


def app_solve_depths(uv: np.ndarray, edge_mm: float, fx: float, fy: float,
                     cx: float, cy: float, *,
                     max_iters: int = 50, learning_rate: float = 0.5,
                     conv_mm: float = 0.1, min_depth_mm: float = 100.0,
                     init_fallback_mm: float = 1000.0) -> DepthSolveResult:
    """Line-faithful port of ``PnPDistanceCalculator.solve3PointDepths``.

    ``uv`` is ``(3, 2)`` pixels in COLOR_ORDER.  Mirrors the Kotlin update
    order (red/green/blue, learning-rate blend, floor clamp) so outputs can be
    compared against the real app.
    """
    uv = np.asarray(uv, dtype=np.float64)
    xn = (uv[:, 0] - cx) / fx
    yn = (uv[:, 1] - cy) / fy
    d_img = [
        float(np.hypot(xn[1] - xn[0], yn[1] - yn[0])),  # 0: p1-p2
        float(np.hypot(xn[2] - xn[1], yn[2] - yn[1])),  # 1: p2-p3
        float(np.hypot(xn[0] - xn[2], yn[0] - yn[2])),  # 2: p3-p1
    ]
    floor = max(min_depth_mm, 1.0)

    def _init(da: float, db: float) -> float:
        if da > 0.001 and db > 0.001:
            return float(np.sqrt((edge_mm / da) * (edge_mm / db)))
        return max(init_fallback_mm, 1.0)

    # Kotlin: z1 <- d12,d31 ; z2 <- d12,d23 ; z3 <- d23,d31
    z = np.array([_init(d_img[0], d_img[2]),
                  _init(d_img[0], d_img[1]),
                  _init(d_img[1], d_img[2])], dtype=np.float64)
    lr = float(np.clip(learning_rate, 0.001, 1.0))
    converged, it = False, 0
    for it in range(max(int(max_iters), 1)):
        d_cur = _edge_dists(xn, yn, z)
        if np.abs(d_cur - edge_mm).max() < max(conv_mm, 1e-4):
            converged = True
            break
        # Kotlin order: z1(d12,d31), z2(d12,d23), z3(d23,d31)
        for zi, pairs in ((0, (0, 2)), (1, (0, 1)), (2, (1, 2))):
            for k in pairs:
                if d_cur[k] > 0.001:
                    z[zi] *= 1.0 + (edge_mm / d_cur[k] - 1.0) * lr
        z = np.maximum(z, floor)
    d_final = _edge_dists(xn, yn, z)
    return DepthSolveResult(z, converged, it,
                            float(np.abs(d_final - edge_mm).max()))


def fixed_solve_depths(uv: np.ndarray, edge_mm: float, fx: float, fy: float,
                       cx: float, cy: float, *,
                       max_iters: int = 50, conv_mm: float = 0.05,
                       min_depth_mm: float = 100.0,
                       init_fallback_mm: float = 1000.0,
                       damping: float = 1e-6,
                       init_depths_mm=None) -> DepthSolveResult:
    """Damped Newton solve of the same three edge constraints (the fix).

    Mirrors the Kotlin replacement in ``PnPDistanceCalculator.solve3PointDepths``:
    analytic Jacobian, Levenberg damping, step clamped to half the current
    depth, depths kept above ``min_depth_mm``.
    """
    uv = np.asarray(uv, dtype=np.float64)
    xn = (uv[:, 0] - cx) / fx
    yn = (uv[:, 1] - cy) / fy
    d_img = [
        float(np.hypot(xn[1] - xn[0], yn[1] - yn[0])),
        float(np.hypot(xn[2] - xn[1], yn[2] - yn[1])),
        float(np.hypot(xn[0] - xn[2], yn[0] - yn[2])),
    ]
    floor = max(min_depth_mm, 1.0)

    def _init(da: float, db: float) -> float:
        if da > 0.001 and db > 0.001:
            return float(np.sqrt((edge_mm / da) * (edge_mm / db)))
        return max(init_fallback_mm, floor)

    if init_depths_mm is not None:
        z = np.maximum(np.asarray(init_depths_mm, dtype=np.float64).copy(), floor)
    else:
        z = np.array([_init(d_img[0], d_img[2]),
                      _init(d_img[0], d_img[1]),
                      _init(d_img[1], d_img[2])], dtype=np.float64)
    basis = np.stack([xn, yn, np.ones(3)], axis=1)  # exact ray directions (z = 1)
    edge_i = np.array([0, 1, 2])
    edge_j = np.array([1, 2, 0])
    lam = float(damping)
    converged, it = False, 0
    for it in range(max(int(max_iters), 1)):
        P = z[:, None] * basis
        diff = P[edge_j] - P[edge_i]
        dist = np.linalg.norm(diff, axis=1).clip(min=1e-3)
        nrm = diff / dist[:, None]
        res = dist - edge_mm
        if np.abs(res).max() < max(conv_mm, 1e-4):
            converged = True
            break
        J = np.zeros((3, 3))
        for e in range(3):
            i, j = edge_i[e], edge_j[e]
            J[e, i] = -float(nrm[e] @ basis[i])
            J[e, j] = float(nrm[e] @ basis[j])
        A = J.T @ J + lam * np.eye(3)
        g = -J.T @ res
        try:
            delta = np.linalg.solve(A, g)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            break
        step = np.clip(delta, -0.5 * z, 0.5 * z)
        z = np.maximum(z + step, floor)
        if not np.all(np.isfinite(z)):  # pragma: no cover - defensive
            raise FloatingPointError("depth solver produced non-finite depths")
        lam = max(lam * 0.5, 1e-9)
    d_final = _edge_dists(xn, yn, z)
    return DepthSolveResult(z, converged, it,
                            float(np.abs(d_final - edge_mm).max()))


# --------------------------------------------------------------------------- #
# pose from the recovered 3-D points (both pipelines share this step)
# --------------------------------------------------------------------------- #
def kabsch_rt(obj_pts: np.ndarray, cam_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid transform mapping object points into the camera frame."""
    oc = obj_pts.mean(axis=0)
    cc = cam_pts.mean(axis=0)
    H = (obj_pts - oc).T @ (cam_pts - cc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cc - R @ oc
    return R, t


def object_points_mm(edge_mm: float) -> np.ndarray:
    """Equilateral triangle used by both the app and the synthetic dataset."""
    h = edge_mm * np.sqrt(3.0) / 2.0
    return np.array([[0.0, 0.0, 0.0],
                     [edge_mm, 0.0, 0.0],
                     [edge_mm / 2.0, h, 0.0]])


def solve_pose_from_depths(uv: np.ndarray, depths_mm: np.ndarray, edge_mm: float,
                           fx: float, fy: float, cx: float,
                           cy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera-frame points, rotation and translation, exactly as the app builds them."""
    xn = (np.asarray(uv, dtype=np.float64)[:, 0] - cx) / fx
    yn = (np.asarray(uv, dtype=np.float64)[:, 1] - cy) / fy
    cam_pts = np.asarray(depths_mm, dtype=np.float64)[:, None] * np.stack(
        [xn, yn, np.ones(3)], axis=1)
    R, t = kabsch_rt(object_points_mm(edge_mm), cam_pts)
    return cam_pts, R, t


def distance_from_points(cam_pts: np.ndarray) -> float:
    """App's DISTANCE readout: mean Euclidean norm of the three dots."""
    return float(np.mean(np.linalg.norm(cam_pts, axis=1)))


class AppFixedSolver:
    """Drop-in replacement for ``B1PoseSolver`` using the fixed depth solver.

    Mirrors the app's camera model (principal point = image centre, focal from
    FOV) but solves the constraints properly.  Used by
    ``eval_v2 --methods app_fixed`` to answer "what would the phone report if
    its PnP were fixed, keeping the app's own detector?".
    """

    def __init__(self, fov_deg: float = 60.0) -> None:
        self.fov_deg = float(fov_deg)
        self._last_ok = False
        self._last_depths = None  # temporal seed, mirrors Kotlin lastDepthsMm

    def reset(self) -> None:
        self._last_ok = False
        self._last_depths = None

    def solve(self, points: dict, edge_mm: float, image_w: int, image_h: int,
              fx=None, fy=None, cx=None, cy=None):
        """Mirror ``B1PoseSolver.solve``: FOV-estimated focal, image-centre principal point."""
        from types import SimpleNamespace

        fx = estimate_focal_length(int(image_w), self.fov_deg)
        fy = fx
        cx = int(image_w) / 2.0
        cy = int(image_h) / 2.0
        uv = np.array([[points[c][0], points[c][1]] for c in COLOR_ORDER],
                      dtype=np.float64)
        res = fixed_solve_depths(uv, edge_mm, fx, fy, cx, cy,
                                 init_depths_mm=self._last_depths)
        self._last_depths = res.depths_mm.copy() if res.converged else None
        P, R, t = solve_pose_from_depths(uv, res.depths_mm, edge_mm,
                                         fx, fy, cx, cy)
        self._last_ok = bool(res.converged)
        return SimpleNamespace(
            ok=True,
            translation_mm=t,
            rotation=R,
            ranges_mm=res.depths_mm,
            distance_mm=distance_from_points(P),
            edge_err_mm=res.max_abs_edge_err_mm,
            reason="fixed" if res.converged else "not_converged",
        )


__all__ = [
    "COLOR_ORDER",
    "AppFixedSolver",
    "DepthSolveResult",
    "app_solve_depths",
    "distance_from_points",
    "estimate_focal_length",
    "fixed_solve_depths",
    "kabsch_rt",
    "object_points_mm",
    "solve_pose_from_depths",
]
