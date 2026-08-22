"""Joint angles. Front-view coronal metrics use image (x, y) only. z is not metres."""

from __future__ import annotations

import math

from . import constants as C


def _vec(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def xy2(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """Drop image-z. Use for all front-view coronal scalars."""
    return (p[0], p[1], 0.0)


def midpoint(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, 0.0)


def as_2d(lms: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    return [xy2(p) for p in lms]


def angle_deg(
    origin: tuple[float, float, float],
    p: tuple[float, float, float],
    q: tuple[float, float, float],
) -> float:
    """Interior angle at origin between P and Q, degrees."""
    u = _vec(origin, p)
    v = _vec(origin, q)
    nu, nv = _norm(u), _norm(v)
    if nu < 1e-8 or nv < 1e-8:
        return float("nan")
    c = max(-1.0, min(1.0, _dot(u, v) / (nu * nv)))
    return math.degrees(math.acos(c))


def elbow_angle(lms: list[tuple[float, float, float]], right: bool = True) -> float:
    """Angle at elbow (shoulder–elbow–wrist). Straight ≈ 180°, flexion decreases."""
    s = lms[C.RIGHT_SHOULDER if right else C.LEFT_SHOULDER]
    e = lms[C.RIGHT_ELBOW if right else C.LEFT_ELBOW]
    w = lms[C.RIGHT_WRIST if right else C.LEFT_WRIST]
    return angle_deg(e, s, w)


def abduction_angle(lms: list[tuple[float, float, float]], right: bool = True) -> float:
    """Humerus vs thorax vertical in the image plane: angle at shoulder (hip, elbow).

    Arm hanging ≈ 0°, 90° abduction ≈ 90°, overhead ≈ 180°. Callers must pass 2D points.
    """
    s = lms[C.RIGHT_SHOULDER if right else C.LEFT_SHOULDER]
    h = lms[C.RIGHT_HIP if right else C.LEFT_HIP]
    e = lms[C.RIGHT_ELBOW if right else C.LEFT_ELBOW]
    return angle_deg(s, h, e)


def knee_angle(lms: list[tuple[float, float, float]], right: bool = True) -> float:
    h = lms[C.RIGHT_HIP if right else C.LEFT_HIP]
    k = lms[C.RIGHT_KNEE if right else C.LEFT_KNEE]
    a = lms[C.RIGHT_ANKLE if right else C.LEFT_ANKLE]
    return angle_deg(k, h, a)


def mid_shoulder(lms: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return midpoint(lms[C.LEFT_SHOULDER], lms[C.RIGHT_SHOULDER])


def mid_hip(lms: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return midpoint(lms[C.LEFT_HIP], lms[C.RIGHT_HIP])


def torso_length(lms: list[tuple[float, float, float]]) -> float:
    return _norm(_vec(mid_hip(lms), mid_shoulder(lms)))


def trunk_lean_signed(lms: list[tuple[float, float, float]]) -> float:
    """Degrees from image vertical using mid-hip → mid-shoulder.

    Sign: positive if mid-shoulder is to the subject's right in the image (larger x).
    """
    ms = mid_shoulder(lms)
    mh = mid_hip(lms)
    torso = _vec(mh, ms)
    nt = _norm(torso)
    if nt < 1e-8:
        return float("nan")
    up = (0.0, -1.0, 0.0)
    c = max(-1.0, min(1.0, _dot(torso, up) / nt))
    mag = math.degrees(math.acos(c))
    sign = 1.0 if (ms[0] - mh[0]) >= 0 else -1.0
    return sign * mag


def trunk_vs_vertical(lms: list[tuple[float, float, float]], right: bool = True) -> float:
    """Degrees from vertical using one hip→shoulder (squat / side-ish). Prefer trunk_lean_signed for Ex1."""
    s = lms[C.RIGHT_SHOULDER if right else C.LEFT_SHOULDER]
    h = lms[C.RIGHT_HIP if right else C.LEFT_HIP]
    torso = _vec(h, s)
    nt = _norm(torso)
    if nt < 1e-8:
        return float("nan")
    up = (0.0, -1.0, 0.0)
    c = max(-1.0, min(1.0, _dot(torso, up) / nt))
    return math.degrees(math.acos(c))


def trunk_lean_abs(lms: list[tuple[float, float, float]]) -> float:
    v = trunk_lean_signed(lms)
    return abs(v) if v == v else float("nan")


def shrug_gap_norm(lms: list[tuple[float, float, float]], right: bool = True) -> float:
    """Vertical ear–shoulder gap / torso length. y down: hike shrinks the gap."""
    e = lms[C.RIGHT_EAR if right else C.LEFT_EAR]
    s = lms[C.RIGHT_SHOULDER if right else C.LEFT_SHOULDER]
    tl = torso_length(lms)
    if tl < 1e-6:
        return float("nan")
    return (s[1] - e[1]) / tl


def ear_shoulder_dist(lms: list[tuple[float, float, float]], right: bool = True) -> float:
    """Deprecated 3D Euclidean; prefer shrug_gap_norm on 2D points."""
    e = lms[C.RIGHT_EAR if right else C.LEFT_EAR]
    s = lms[C.RIGHT_SHOULDER if right else C.LEFT_SHOULDER]
    return _norm(_vec(s, e))


def shoulder_width_over_torso(lms: list[tuple[float, float, float]]) -> float:
    w = abs(lms[C.LEFT_SHOULDER][0] - lms[C.RIGHT_SHOULDER][0])
    tl = torso_length(lms)
    if tl < 1e-6:
        return float("nan")
    return w / tl


def shoulder_y_split(lms: list[tuple[float, float, float]]) -> float:
    return abs(lms[C.LEFT_SHOULDER][1] - lms[C.RIGHT_SHOULDER][1])


def humerus_coronal_dev_world(world: list[tuple[float, float, float]], right: bool = True) -> float:
    """Best-effort: angle between humerus and the world YZ (coronal-ish) plane.

    World coords: metres, hip origin. Disabled if world is missing/NaN.
    Returns degrees of forward/back deviation (0 = in frontal plane).
    """
    s = world[C.RIGHT_SHOULDER if right else C.LEFT_SHOULDER]
    e = world[C.RIGHT_ELBOW if right else C.LEFT_ELBOW]
    if any(math.isnan(c) for c in s + e):
        return float("nan")
    hx, hy, hz = e[0] - s[0], e[1] - s[1], e[2] - s[2]
    # Frontal plane ≈ YZ if X is camera-forward in GHUM world; MediaPipe world X is often lateral.
    # Use deviation of the humerus out of the plane spanned by torso (hip-shoulder) and world vertical.
    # Safer 2D proxy: |world x of elbow - shoulder| relative to humerus length (lateral vs depth).
    mag = math.sqrt(hx * hx + hy * hy + hz * hz)
    if mag < 1e-8:
        return float("nan")
    # Component along camera-typical depth is z in image; in world, use the smaller of |hx|,|hz|
    # as "out of coronal" is not uniquely defined. Disable unless clearly large.
    depthish = abs(hz)
    return math.degrees(math.asin(min(1.0, depthish / mag)))
