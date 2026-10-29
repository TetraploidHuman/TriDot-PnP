"""Two-point PnP with a temporal motion prior.

When only two of the three target dots are visible (partial occlusion, dot off
screen, colour invisible), the classic 3-point PnP has no solution.  The rigid
target still gives a strong constraint though: the two observed image points pin
down 4 of the 6 pose DOF, and the *previous* pose (rigid-body motion prior) fills
the remaining 2.  This module solves that joint problem with a damped
Gauss-Newton over a 6-vector (rotation-vector around the prior, translation),
using numeric Jacobians (the residual is tiny and cheap, and scipy is not
available in this venv).

The result is deliberately reported with its cost and reprojection error so the
caller can gate it much harder than a 3-point solve: a two-point pose is a
*guess* informed by motion, not a measurement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TwoPointResult:
    ok: bool
    rotation: Optional[np.ndarray]
    translation_mm: Optional[np.ndarray]
    reproj_px: float
    rotation_from_prior_deg: float
    translation_from_prior_mm: float
    cost: float
    iterations: int
    reason: str


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def exp_so3(omega: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    K = _skew(k)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))



def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix (robust near 0 and pi)."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    theta = math.acos(cos)
    if theta < 1e-8:
        return np.array([(R[2, 1] - R[1, 2]) / 2.0,
                         (R[0, 2] - R[2, 0]) / 2.0,
                         (R[1, 0] - R[0, 1]) / 2.0])
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / (2.0 * math.sin(theta))
    return axis * theta

def _project(R: np.ndarray, t: np.ndarray, P: np.ndarray,
             fx: float, fy: float, cx: float, cy: float) -> Optional[np.ndarray]:
    p = R @ P + t
    if p[2] <= 1.0:
        return None
    return np.array([fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy])


def _residuals(x: np.ndarray, object_pts: np.ndarray, uv: np.ndarray,
               fx: float, fy: float, cx: float, cy: float,
               R_prior: np.ndarray, t_prior: np.ndarray,
               sigma_px: float, sigma_t_mm: float, sigma_rot_rad: float) -> np.ndarray:
    R = R_prior @ exp_so3(x[:3])
    t = t_prior + x[3:]
    res: List[float] = []
    for P, z in zip(object_pts, uv):
        p = R @ P + t
        if p[2] <= 1.0:
            res.extend([1e3, 1e3])
            continue
        u = fx * p[0] / p[2] + cx
        v = fy * p[1] / p[2] + cy
        res.extend([(u - z[0]) / max(1e-6, sigma_px),
                    (v - z[1]) / max(1e-6, sigma_px)])
    res.extend(((t - t_prior) / max(1e-6, sigma_t_mm)).tolist())
    res.extend((x[:3] / max(1e-6, sigma_rot_rad)).tolist())
    return np.asarray(res, dtype=float)


def solve_pose_2point(
    uv_pts: Sequence[Sequence[float]],
    object_pts_mm: Sequence[Sequence[float]],
    fx: float, fy: float, cx: float, cy: float,
    R_prior: np.ndarray, t_prior: np.ndarray,
    *,
    sigma_px: float = 1.5,
    sigma_t_mm: float = 300.0,
    sigma_rot_rad: float = 0.20,
    max_iters: int = 20,
    max_reproj_px: float = 3.0,
    max_prior_jump_deg: float = 25.0,
    max_prior_jump_mm: Optional[float] = None,
) -> TwoPointResult:
    """Solve a two-point pose, regularised by the previous frame's pose."""
    uv = np.asarray(uv_pts, dtype=float)
    obj = np.asarray(object_pts_mm, dtype=float)
    R_prior = np.asarray(R_prior, dtype=float).reshape(3, 3)
    t_prior = np.asarray(t_prior, dtype=float).reshape(3)
    if uv.shape != (2, 2) or obj.shape != (2, 3):
        return TwoPointResult(False, None, None, math.inf, math.inf, math.inf,
                              math.inf, 0, "need-exactly-2-points")

    x = np.zeros(6)
    lam = 1e-3
    cost = math.inf
    for it in range(max_iters):
        r = _residuals(x, obj, uv, fx, fy, cx, cy, R_prior, t_prior,
                       sigma_px, sigma_t_mm, sigma_rot_rad)
        cost = float(r @ r)
        # numeric Jacobian (6 columns; residual length is small)
        J = np.empty((r.size, 6))
        h = 1e-6
        for j in range(6):
            dx = np.zeros(6)
            dx[j] = h
            rp = _residuals(x + dx, obj, uv, fx, fy, cx, cy, R_prior, t_prior,
                            sigma_px, sigma_t_mm, sigma_rot_rad)
            J[:, j] = (rp - r) / h
        g = J.T @ r
        A = J.T @ J
        diag = np.diag(A).copy() + 1e-9
        improved = False
        for _ in range(6):
            try:
                step = -np.linalg.solve(A + lam * np.diag(diag), g)
            except np.linalg.LinAlgError:
                break
            # cap each block separately: rotation in rad, translation in mm
            scale = 1.0
            dw = float(np.linalg.norm(step[:3]))
            dt = float(np.linalg.norm(step[3:]))
            if dw > 0.3:
                scale = min(scale, 0.3 / dw)
            if dt > 500.0:
                scale = min(scale, 500.0 / dt)
            step *= scale
            x_new = x + step
            r_new = _residuals(x_new, obj, uv, fx, fy, cx, cy, R_prior, t_prior,
                               sigma_px, sigma_t_mm, sigma_rot_rad)
            if float(r_new @ r_new) <= cost:
                x = x_new
                lam = max(1e-7, lam * 0.5)
                improved = True
                break
            lam = min(1e7, lam * 4.0)
        if not improved:
            break

    R = R_prior @ exp_so3(x[:3])
    t = t_prior + x[3:]
    # observed reprojection error (exclude prior terms)
    errs: List[float] = []
    for P, z in zip(obj, uv):
        q = _project(R, t, P, fx, fy, cx, cy)
        if q is None:
            errs.append(math.inf)
        else:
            errs.append(float(np.hypot(q[0] - z[0], q[1] - z[1])))
    reproj = float(np.sqrt(np.mean(np.square(errs)))) if errs else math.inf
    d_rot = rotation_angle_deg(R, R_prior)
    d_t = float(np.linalg.norm(t - t_prior))
    depth = float(np.linalg.norm(t))
    jump_lim = max(250.0, 0.35 * depth) if max_prior_jump_mm is None else float(max_prior_jump_mm)

    if not np.isfinite(reproj) or reproj > max_reproj_px:
        return TwoPointResult(False, None, None, reproj, d_rot, d_t, cost, it + 1,
                              "reproj-too-large")
    if d_rot > max_prior_jump_deg:
        return TwoPointResult(False, None, None, reproj, d_rot, d_t, cost, it + 1,
                              "rotation-jump")
    if d_t > jump_lim:
        return TwoPointResult(False, None, None, reproj, d_rot, d_t, cost, it + 1,
                              "translation-jump")
    return TwoPointResult(True, R, t, reproj, d_rot, d_t, cost, it + 1, "ok")


def _residuals_plane(x: np.ndarray, object_pts: np.ndarray, uv: np.ndarray,
                     fx: float, fy: float, cx: float, cy: float,
                     normal_prior: np.ndarray, sigma_px: float,
                     sigma_normal: float) -> np.ndarray:
    R = exp_so3(x[:3])
    t = x[3:]
    res: List[float] = []
    for P, z in zip(object_pts, uv):
        p = R @ P + t
        if p[2] <= 1.0:
            res.extend([1e3, 1e3])
            continue
        res.extend([(fx * p[0] / p[2] + cx - z[0]) / max(1e-6, sigma_px),
                    (fy * p[1] / p[2] + cy - z[1]) / max(1e-6, sigma_px)])
    n_obj = np.array([0.0, 0.0, 1.0])
    n_pred = R @ n_obj
    res.extend(((n_pred - normal_prior) / max(1e-6, sigma_normal)).tolist())
    return np.asarray(res, dtype=float)


def solve_pose_2point_plane(
    uv_pts: Sequence[Sequence[float]],
    object_pts_mm: Sequence[Sequence[float]],
    fx: float, fy: float, cx: float, cy: float,
    *,
    normal_prior: Sequence[float] = (0.0, 0.0, 1.0),
    sigma_px: float = 1.5,
    sigma_normal: float = 0.15,
    max_iters: int = 25,
    max_reproj_px: float = 3.0,
    normal_gate_deg: float = 25.0,
) -> TwoPointResult:
    """Two-point pose from the *plane-normal* prior alone (no motion prior).

    The target plane is assumed to face the camera (its object +Z axis maps onto
    the camera +Z axis, up to ``normal_gate_deg``).  Together with the two rays
    this determines all six DOF, which lets a tracker bootstrap from frames that
    never show all three dots (e.g. ``occlude_hand``).
    """
    uv = np.asarray(uv_pts, dtype=float)
    obj = np.asarray(object_pts_mm, dtype=float)
    normal_prior = np.asarray(normal_prior, dtype=float)
    normal_prior = normal_prior / max(1e-9, float(np.linalg.norm(normal_prior)))
    if uv.shape != (2, 2) or obj.shape != (2, 3):
        return TwoPointResult(False, None, None, math.inf, math.inf, math.inf,
                              math.inf, 0, "need-exactly-2-points")

    # initialise: identity rotation, depth from the observed edge scale
    edge_px = float(np.linalg.norm(uv[0] - uv[1]))
    edge_mm = float(np.linalg.norm(obj[0] - obj[1]))
    if edge_px < 6.0:
        return TwoPointResult(False, None, None, math.inf, math.inf, math.inf,
                              math.inf, 0, "edge-too-small")
    depth = fx * edge_mm / edge_px
    u_mid = float(uv[:, 0].mean())
    v_mid = float(uv[:, 1].mean())
    t0 = np.array([(u_mid - cx) * depth / fx, (v_mid - cy) * depth / fy, depth])
    x = np.concatenate([np.zeros(3), t0])
    lam = 1e-3
    cost = math.inf
    for it in range(max_iters):
        r = _residuals_plane(x, obj, uv, fx, fy, cx, cy, normal_prior,
                             sigma_px, sigma_normal)
        cost = float(r @ r)
        J = np.empty((r.size, 6))
        h = 1e-6
        for j in range(6):
            dx = np.zeros(6)
            dx[j] = h
            rp = _residuals_plane(x + dx, obj, uv, fx, fy, cx, cy, normal_prior,
                                  sigma_px, sigma_normal)
            J[:, j] = (rp - r) / h
        g = J.T @ r
        A = J.T @ J
        diag = np.diag(A).copy() + 1e-9
        improved = False
        for _ in range(6):
            try:
                step = -np.linalg.solve(A + lam * np.diag(diag), g)
            except np.linalg.LinAlgError:
                break
            scale = 1.0
            dw = float(np.linalg.norm(step[:3]))
            dt = float(np.linalg.norm(step[3:]))
            if dw > 0.3:
                scale = min(scale, 0.3 / dw)
            if dt > 500.0:
                scale = min(scale, 500.0 / dt)
            step *= scale
            x_new = x + step
            r_new = _residuals_plane(x_new, obj, uv, fx, fy, cx, cy, normal_prior,
                                     sigma_px, sigma_normal)
            if float(r_new @ r_new) <= cost:
                x = x_new
                lam = max(1e-7, lam * 0.5)
                improved = True
                break
            lam = min(1e7, lam * 4.0)
        if not improved:
            break

    R = exp_so3(x[:3])
    t = x[3:]
    errs = []
    for P, z in zip(obj, uv):
        q = _project(R, t, P, fx, fy, cx, cy)
        errs.append(math.inf if q is None else float(np.hypot(q[0] - z[0], q[1] - z[1])))
    reproj = float(np.sqrt(np.mean(np.square(errs)))) if errs else math.inf
    n_pred = R @ np.array([0.0, 0.0, 1.0])
    normal_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(n_pred @ normal_prior)))))
    if not np.isfinite(reproj) or reproj > max_reproj_px:
        return TwoPointResult(False, None, None, reproj, normal_deg, float(np.linalg.norm(t)),
                              cost, it + 1, "reproj-too-large")
    if normal_deg > normal_gate_deg:
        return TwoPointResult(False, None, None, reproj, normal_deg, float(np.linalg.norm(t)),
                              cost, it + 1, "normal-jump")
    return TwoPointResult(True, R, t, reproj, normal_deg, 0.0, cost, it + 1, "ok")
