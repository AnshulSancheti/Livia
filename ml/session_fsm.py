"""Session layer above the abduction rep SM. Landmarks only — no extra net."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from . import constants as C


@dataclass
class SessionSnapshot:
    state: str
    allow_count: bool
    flags: list[str]


class SessionTracker:
    def __init__(self) -> None:
        self._lost = 0
        self._idle = 0
        self._no_cycle = 0
        self._abd_r: deque[float] = deque(maxlen=int(C.FPS * 2))
        self._abd_l: deque[float] = deque(maxlen=int(C.FPS * 2))
        self._speeds: deque[float] = deque(maxlen=int(C.FPS * 2))
        self._ankle: deque[float] = deque(maxlen=int(C.FPS * 2))
        self._last_cycle_age = 0
        self._pose_ok_frames = 0
        self.state = "NO_PERSON"

    def note_cycle(self) -> None:
        self._last_cycle_age = 0
        self._no_cycle = 0
        self._idle = 0

    def update(
        self,
        *,
        detected: bool,
        vis_ok: bool,
        abd_r: float,
        abd_l: float,
        limb_speed: float,
        ankle_speed: float,
        width_over_torso: float,
        shoulder_y_split: float,
        in_bottom: bool,
        counted_this_frame: bool,
    ) -> SessionSnapshot:
        if counted_this_frame:
            self.note_cycle()

        self._last_cycle_age += 1
        if not detected or not vis_ok:
            self._lost += 1
        else:
            self._lost = 0
            self._pose_ok_frames += 1
            if abd_r == abd_r:
                self._abd_r.append(abd_r)
            if abd_l == abd_l:
                self._abd_l.append(abd_l)
            self._speeds.append(limb_speed)
            self._ankle.append(ankle_speed)

        flags: list[str] = []
        allow = False

        if self._lost >= C.NO_PERSON_FRAMES:
            self.state = "NO_PERSON"
            flags.append("NO_PERSON")
            return SessionSnapshot(self.state, False, flags)

        if width_over_torso == width_over_torso and (
            width_over_torso < C.WRONG_VIEW_WIDTH_RATIO or shoulder_y_split > C.WRONG_VIEW_Y_SPLIT
        ):
            self.state = "WRONG_VIEW"
            flags.append("WRONG_VIEW")
            return SessionSnapshot(self.state, False, flags)

        if len(self._abd_r) >= 20 and len(self._abd_l) >= 20:
            vr = float(np.var(self._abd_r))
            vl = float(np.var(self._abd_l))
            if vl > C.WRONG_SIDE_VAR_RATIO * max(vr, 1e-6) and vl > C.WRONG_SIDE_VAR_MIN:
                self.state = "WRONG_SIDE"
                flags.append("WRONG_SIDE")
                return SessionSnapshot(self.state, False, flags)

        mean_speed = float(sum(self._speeds) / max(len(self._speeds), 1))
        mean_ankle = float(sum(self._ankle) / max(len(self._ankle), 1))

        if in_bottom and mean_speed < C.IDLE_SPEED:
            self._idle += 1
        else:
            self._idle = 0

        if self._idle >= C.IDLE_FRAMES:
            self.state = "IDLE"
            flags.append("IDLE")
            return SessionSnapshot(self.state, False, flags)

        cycle_stale = self._last_cycle_age > C.UNKNOWN_NO_CYCLE_FRAMES
        other = mean_speed > C.UNKNOWN_SPEED and mean_ankle > C.UNKNOWN_ANKLE
        if (
            cycle_stale
            and mean_speed > C.UNKNOWN_SPEED
            and self._pose_ok_frames > C.UNKNOWN_NO_CYCLE_FRAMES
        ):
            self._no_cycle += 1
        else:
            self._no_cycle = 0

        if self._no_cycle >= 15 and other:
            self.state = "UNKNOWN_ACTIVITY"
            flags.append("UNKNOWN_ACTIVITY")
            return SessionSnapshot(self.state, False, flags)

        self.state = "EXERCISING"
        allow = True
        return SessionSnapshot(self.state, allow, flags)
