"""Heuristic helpers: dwell, cooldown, stacked flags. Not a trained classifier."""

from __future__ import annotations

from dataclasses import dataclass, field


# Speak / HUD priority (high index first in SPEAK_ORDER).
SPEAK_ORDER = (
    "NO_PERSON",
    "WRONG_VIEW",
    "WRONG_SIDE",
    "UNKNOWN_ACTIVITY",
    "IDLE",
    "INCOMPLETE_ROM",
    "BENT_ELBOW",
    "SHRUG",
    "TRUNK_LEAN",
    "TOO_FAST",
    "NO_HOLD",
    "PLANE_FWD",
)

HUD_COPY = {
    "NO_PERSON": "Step into frame",
    "WRONG_VIEW": "Face the camera",
    "WRONG_SIDE": "Use the right arm",
    "UNKNOWN_ACTIVITY": "Not the prescribed exercise",
    "IDLE": "Continue your set",
    "INCOMPLETE_ROM": "Lift higher",
    "BENT_ELBOW": "Keep the arm straight",
    "SHRUG": "Don't shrug",
    "TRUNK_LEAN": "Stand tall, don't lean",
    "TOO_FAST": "Slow the movement",
    "NO_HOLD": "Pause at the top",
    "PLANE_FWD": "Keep the arm in line with your body",
    "MULTI": "More than one form issue",
}

FORM_FLAGS = frozenset(
    {"INCOMPLETE_ROM", "BENT_ELBOW", "SHRUG", "TRUNK_LEAN", "TOO_FAST", "NO_HOLD", "PLANE_FWD"}
)


@dataclass
class DwellFlag:
    need: int
    _run: int = 0
    latched: bool = False

    def update(self, raw: bool) -> bool:
        if raw:
            self._run += 1
            if self._run >= self.need:
                self.latched = True
        else:
            self._run = 0
        return self.latched

    def reset(self) -> None:
        self._run = 0
        self.latched = False


@dataclass
class Cooldown:
    frames: int
    _left: int = 0

    def tick(self) -> None:
        if self._left > 0:
            self._left -= 1

    def ready(self) -> bool:
        return self._left <= 0

    def fire(self) -> None:
        self._left = self.frames


@dataclass
class SpeakGate:
    """TTS eligibility: second episode of the same id, or continuous hold >= need frames.

    HUD/latch is independent. MULTI is ignored.
    """

    need: int
    _run: dict[str, int] = field(default_factory=dict)
    _on: set[str] = field(default_factory=set)
    _episodes: dict[str, int] = field(default_factory=dict)

    def update(self, active_ids: list[str] | set[str]) -> set[str]:
        active = {i for i in active_ids if i != "MULTI"}
        eligible: set[str] = set()
        for fid in list(self._on):
            if fid not in active:
                self._episodes[fid] = self._episodes.get(fid, 0) + 1
                self._run[fid] = 0
                self._on.discard(fid)
        for fid in active:
            rising = fid not in self._on
            self._on.add(fid)
            self._run[fid] = self._run.get(fid, 0) + 1
            if self._run[fid] >= self.need:
                eligible.add(fid)
            if rising and self._episodes.get(fid, 0) >= 1:
                eligible.add(fid)
        return eligible


def stack_flags(active: list[str]) -> list[str]:
    """Unique flags in SPEAK_ORDER. Caller may append MULTI for HUD."""
    seen = set(active)
    return [f for f in SPEAK_ORDER if f in seen]


def speak_id(ordered: list[str]) -> str | None:
    return ordered[0] if ordered else None
