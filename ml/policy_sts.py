"""Deterministic five-times sit-to-stand prototype for a side-view clip.

This is a prototype exercise policy, not a clinically validated assessment.
It selects the more visible body side, smooths landmarks, and counts complete
SEATED -> RISING -> STANDING cycles with hysteresis and dwell.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from . import constants as C
from . import features as F
from .one_euro import LandmarkSmoother


class State(Enum):
    CALIBRATING = "CALIBRATING"
    SEATED = "SEATED"
    RISING = "RISING"
    STANDING = "STANDING"
    LOWERING = "LOWERING"


@dataclass
class FrameEvent:
    frame: int
    state: str
    side: str
    knee_deg: float
    trunk_deg: float
    paused: bool
    excessive_trunk_flex: bool
    fast_descent: bool
    counted: bool = False
    rep_duration_sec: float | None = None
    cue_text: str | None = None


@dataclass
class PolicyResult:
    events: list[FrameEvent]
    rep_count: int
    side: str
    visible_required_frac: float
    paused_frac: float
    five_rep_time_sec: float | None
    rep_durations_sec: list[float]


def _xyz_tuple(row: np.ndarray) -> list[tuple[float, float, float]]:
    return [(float(p[0]), float(p[1]), float(p[2])) for p in row]


def _required(right: bool) -> tuple[int, int, int, int]:
    if right:
        return (C.RIGHT_SHOULDER, C.RIGHT_HIP, C.RIGHT_KNEE, C.RIGHT_ANKLE)
    return (C.LEFT_SHOULDER, C.LEFT_HIP, C.LEFT_KNEE, C.LEFT_ANKLE)


def choose_visible_side(vis: np.ndarray, detected: np.ndarray) -> bool:
    """Return True for right, False for left using detected-frame visibility."""
    valid = np.asarray(detected, dtype=bool)
    if not np.any(valid):
        return True
    left = float(np.mean(vis[valid][:, list(_required(False))]))
    right = float(np.mean(vis[valid][:, list(_required(True))]))
    return right >= left


def run_sts(xyz: np.ndarray, vis: np.ndarray, detected: np.ndarray) -> PolicyResult:
    right = choose_visible_side(vis, detected)
    side = "right" if right else "left"
    req = _required(right)
    smoother = LandmarkSmoother()
    state = State.CALIBRATING
    dwell = 0
    reps = 0
    visible_n = 0
    paused_n = 0
    rise_start: int | None = None
    lowering_start: int | None = None
    first_rise_start: int | None = None
    fifth_stand: int | None = None
    trunk_run = 0
    rep_durations: list[float] = []
    events: list[FrameEvent] = []

    for i in range(len(xyz)):
        sm = smoother.apply((i + 1) / C.FPS, _xyz_tuple(xyz[i]))
        mean_vis = float(np.mean(vis[i, list(req)])) if detected[i] else 0.0
        paused = (not detected[i]) or mean_vis < C.VISIBILITY_MIN or any(
            np.isnan(sm[j][0]) for j in req
        )
        if paused:
            paused_n += 1
        else:
            visible_n += 1

        sm2 = F.as_2d(sm)
        knee = F.knee_angle(sm2, right) if not paused else float("nan")
        trunk = F.trunk_vs_vertical(sm2, right) if not paused else float("nan")
        counted = False
        fast_descent = False
        rep_duration: float | None = None

        trunk_raw = (
            not paused
            and state == State.RISING
            and trunk == trunk
            and trunk > C.STS_TRUNK_FLEX_DEG
        )
        trunk_run = trunk_run + 1 if trunk_raw else 0
        excessive_trunk = trunk_run >= C.STS_FORM_DWELL_FRAMES

        if paused:
            dwell = 0
        elif state == State.CALIBRATING:
            if knee >= C.STS_STAND_ENTER:
                state = State.STANDING
            elif knee <= C.STS_SEATED_ENTER:
                state = State.SEATED
        elif state == State.SEATED:
            if knee > C.STS_SEATED_LEAVE:
                dwell += 1
                if dwell >= C.STS_MIN_DWELL_FRAMES:
                    state = State.RISING
                    dwell = 0
                    rise_start = i - C.STS_MIN_DWELL_FRAMES + 1
                    if first_rise_start is None:
                        first_rise_start = rise_start
            else:
                dwell = 0
        elif state == State.RISING:
            if knee >= C.STS_STAND_ENTER:
                dwell += 1
                if dwell >= C.STS_MIN_DWELL_FRAMES:
                    state = State.STANDING
                    dwell = 0
                    reps += 1
                    counted = True
                    if rise_start is not None:
                        rep_duration = (i - rise_start + 1) / C.FPS
                        rep_durations.append(rep_duration)
                    if reps == 5:
                        fifth_stand = i
            elif knee <= C.STS_SEATED_ENTER:
                state = State.SEATED
                dwell = 0
        elif state == State.STANDING:
            if knee < C.STS_STAND_LEAVE:
                dwell += 1
                if dwell >= C.STS_MIN_DWELL_FRAMES:
                    state = State.LOWERING
                    dwell = 0
                    lowering_start = i - C.STS_MIN_DWELL_FRAMES + 1
            else:
                dwell = 0
        elif state == State.LOWERING:
            if knee <= C.STS_SEATED_ENTER:
                dwell += 1
                if dwell >= C.STS_MIN_DWELL_FRAMES:
                    state = State.SEATED
                    dwell = 0
                    if lowering_start is not None:
                        descent_sec = (i - lowering_start + 1) / C.FPS
                        fast_descent = descent_sec < C.STS_FAST_DESCENT_SEC
            elif knee >= C.STS_STAND_ENTER:
                state = State.STANDING
                dwell = 0

        cue = None
        if excessive_trunk:
            cue = "Keep your chest more upright"
        elif fast_descent:
            cue = "Lower with control"

        events.append(
            FrameEvent(
                frame=i,
                state=state.value,
                side=side,
                knee_deg=knee,
                trunk_deg=trunk,
                paused=paused,
                excessive_trunk_flex=excessive_trunk,
                fast_descent=fast_descent,
                counted=counted,
                rep_duration_sec=rep_duration,
                cue_text=cue,
            )
        )

    five_time = None
    if first_rise_start is not None and fifth_stand is not None:
        five_time = (fifth_stand - first_rise_start + 1) / C.FPS
    n = max(len(xyz), 1)
    return PolicyResult(
        events=events,
        rep_count=reps,
        side=side,
        visible_required_frac=visible_n / n,
        paused_frac=paused_n / n,
        five_rep_time_sec=five_time,
        rep_durations_sec=rep_durations,
    )
