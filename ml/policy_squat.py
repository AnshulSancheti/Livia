"""Deterministic squat policy (front view).

This is NOT sit-to-stand. REHAB24-6 Ex6 is squats (front/half-profile).
Sit-to-stand is a later product exercise and is not scored here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from . import constants as C
from . import features as F
from .one_euro import LandmarkSmoother

# Documented product gap — do not report STS metrics from this module.
SIT_TO_STAND_VALIDATED = False
SIT_TO_STAND_NOTE = (
    "Sit-to-stand is not in REHAB24-6. Ex6 is squats filmed front-on. "
    "Do not claim STS accuracy from this test bench."
)


class State(Enum):
    STANDING = "STANDING"
    LOWERING = "LOWERING"
    BOTTOM = "BOTTOM"
    RISING = "RISING"


@dataclass
class FrameEvent:
    frame: int
    state: str
    knee_deg: float
    trunk_deg: float
    paused: bool
    trunk_flex: bool
    counted: bool = False


@dataclass
class PolicyResult:
    events: list[FrameEvent]
    rep_count: int
    trunk_flex_frames: int
    visible_required_frac: float
    paused_frac: float


def _xyz_tuple(row: np.ndarray) -> list[tuple[float, float, float]]:
    return [(float(p[0]), float(p[1]), float(p[2])) for p in row]


def run_squat(xyz: np.ndarray, vis: np.ndarray, detected: np.ndarray) -> PolicyResult:
    smoother = LandmarkSmoother()
    state = State.STANDING
    dwell = 0
    reps = 0
    events: list[FrameEvent] = []
    vis_ok = 0
    paused_n = 0
    trunk_n = 0
    t = 0.0
    dt = 1.0 / C.FPS

    for i in range(len(xyz)):
        t += dt
        sm = smoother.apply(t, _xyz_tuple(xyz[i]))
        req = C.SQUAT_REQUIRED
        mean_vis = float(np.mean(vis[i, list(req)])) if detected[i] else 0.0
        paused = (not detected[i]) or mean_vis < C.VISIBILITY_MIN or any(
            np.isnan(sm[j][0]) for j in req
        )
        if not paused:
            vis_ok += 1
        else:
            paused_n += 1

        sm2 = [(p[0], p[1], 0.0) for p in sm]
        knee = F.knee_angle(sm2, right=True) if not paused else float("nan")
        trunk = F.trunk_vs_vertical(sm2, right=True) if not paused else float("nan")
        trunk_flex = (not paused) and (not np.isnan(trunk)) and trunk > C.TRUNK_FLEX_DEG
        if trunk_flex:
            trunk_n += 1

        counted = False
        if paused:
            dwell = 0
        else:
            if state == State.STANDING:
                if knee < C.SQUAT_STAND_LEAVE:
                    dwell += 1
                    if dwell >= C.SQUAT_MIN_DWELL_FRAMES:
                        state = State.LOWERING
                        dwell = 0
                else:
                    dwell = 0
            elif state == State.LOWERING:
                if knee < C.SQUAT_BOTTOM_ENTER:
                    dwell += 1
                    if dwell >= C.SQUAT_MIN_DWELL_FRAMES:
                        state = State.BOTTOM
                        dwell = 0
                elif knee > C.SQUAT_STAND_ENTER:
                    state = State.STANDING
                    dwell = 0
            elif state == State.BOTTOM:
                if knee > C.SQUAT_BOTTOM_LEAVE:
                    dwell += 1
                    if dwell >= C.SQUAT_MIN_DWELL_FRAMES:
                        state = State.RISING
                        dwell = 0
                else:
                    dwell = 0
            elif state == State.RISING:
                if knee > C.SQUAT_STAND_ENTER:
                    dwell += 1
                    if dwell >= C.SQUAT_MIN_DWELL_FRAMES:
                        state = State.STANDING
                        dwell = 0
                        reps += 1
                        counted = True
                elif knee < C.SQUAT_BOTTOM_ENTER:
                    state = State.BOTTOM
                    dwell = 0

        events.append(
            FrameEvent(
                frame=i,
                state=state.value,
                knee_deg=knee,
                trunk_deg=trunk,
                paused=paused,
                trunk_flex=trunk_flex,
                counted=counted,
            )
        )

    n = max(len(xyz), 1)
    return PolicyResult(
        events=events,
        rep_count=reps,
        trunk_flex_frames=trunk_n,
        visible_required_frac=vis_ok / n,
        paused_frac=paused_n / n,
    )


def cues_on_interval(result: PolicyResult, first: int, last: int) -> dict[str, bool]:
    slice_ev = [e for e in result.events if first <= e.frame <= last]
    return {
        "trunk_flex": any(e.trunk_flex for e in slice_ev),
        "counted_inside": sum(1 for e in slice_ev if e.counted),
    }
