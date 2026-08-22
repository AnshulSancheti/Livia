"""Pose quality vs mocap 2D. Evaluation only — does not train MediaPipe."""

from __future__ import annotations

import numpy as np

from .constants import MP_TO_MOCAP, ABDUCTION_REQUIRED, VISIBILITY_MIN


def umeyama_rmse(src: np.ndarray, dst: np.ndarray) -> float:
    """RMSE after 2D similarity (scale + rotation + translation). src/dst (N,2)."""
    if src.shape[0] < 3:
        return float("nan")
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    var_s = (xs**2).sum() / src.shape[0]
    if var_s < 1e-12:
        return float("nan")
    cov = xd.T @ xs / src.shape[0]
    u, _, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    scale = np.trace(r.T @ cov) / var_s
    t = mu_d - scale * (r @ mu_s)
    aligned = (scale * (r @ src.T)).T + t
    return float(np.sqrt(((aligned - dst) ** 2).sum(axis=1).mean()))


def frame_error(
    mp_xy_norm: np.ndarray,
    vis: np.ndarray,
    mocap_xy: np.ndarray,
    width: int,
    height: int,
) -> float:
    src = []
    dst = []
    for mp_i, moc_i in MP_TO_MOCAP.items():
        if vis[mp_i] < VISIBILITY_MIN:
            continue
        if np.isnan(mp_xy_norm[mp_i, 0]) or np.isnan(mocap_xy[moc_i, 0]):
            continue
        src.append([mp_xy_norm[mp_i, 0] * width, mp_xy_norm[mp_i, 1] * height])
        dst.append(mocap_xy[moc_i].tolist())
    if len(src) < 4:
        return float("nan")
    return umeyama_rmse(np.asarray(src), np.asarray(dst))


def required_visible_frac(vis: np.ndarray, detected: np.ndarray, first: int, last: int, required) -> float:
    idxs = list(required)
    ok = 0
    n = 0
    for i in range(first, min(last + 1, len(vis))):
        n += 1
        if detected[i] and float(np.mean(vis[i, idxs])) >= VISIBILITY_MIN:
            ok += 1
    return ok / n if n else float("nan")


def video_pose_metrics(payload: dict, mocap: np.ndarray, required=ABDUCTION_REQUIRED) -> dict:
    xyz = payload["xyz"]
    vis = payload["vis"]
    detected = payload["detected"]
    w = int(payload["width"])
    h = int(payload["height"])
    t = min(len(xyz), len(mocap))
    errors = []
    for i in range(t):
        if not detected[i]:
            continue
        err = frame_error(xyz[i, :, :2], vis[i], mocap[i], w, h)
        if not np.isnan(err):
            errors.append(err)
    diag = float(np.hypot(w, h))
    mean_err = float(np.mean(errors)) if errors else float("nan")
    return {
        "n_frames": int(payload["n_frames"] if "n_frames" in payload else len(xyz)),
        "detect_frac": float(detected[:t].mean()) if t else 0.0,
        "n_error_frames": len(errors),
        "rmse_px": mean_err,
        "rmse_frac_diag": mean_err / diag if errors else float("nan"),
        "required_visible_all_frames": required_visible_frac(vis, detected, 0, t - 1, required),
    }
