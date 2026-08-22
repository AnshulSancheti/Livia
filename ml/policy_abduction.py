"""Deterministic right-arm abduction policy (front view). Not a neural net."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import constants as C
from . import features as F
from .heuristics import Cooldown, DwellFlag, FORM_FLAGS, HUD_COPY, SPEAK_ORDER, SpeakGate, stack_flags, speak_id
from .one_euro import LandmarkSmoother
from .session_fsm import SessionTracker


class State(Enum):
    BOTTOM = "BOTTOM"
    ASCENT = "ASCENT"
    TOP = "TOP"
    DESCENT = "DESCENT"


@dataclass
class FrameEvent:
    frame: int
    state: str
    session: str
    abduction_deg: float
    elbow_deg: float
    trunk_lean_deg: float
    shrug_gap: float
    paused: bool
    allow_count: bool
    bent_elbow: bool
    shrug: bool
    trunk_lean: bool
    incomplete_rom: bool
    too_fast: bool
    no_hold: bool
    plane_fwd: bool
    flags: list[str] = field(default_factory=list)
    counted: bool = False
    cue: str | None = None
    cue_text: str | None = None


@dataclass
class PolicyResult:
    events: list[FrameEvent]
    rep_count: int
    bent_elbow_frames: int
    shrug_frames: int
    visible_required_frac: float
    paused_frac: float
    flag_hist: list[dict] = field(default_factory=list)
    cue_on_reps: dict = field(default_factory=dict)


def _xyz_tuple(row: np.ndarray) -> list[tuple[float, float, float]]:
    return [(float(p[0]), float(p[1]), float(p[2])) for p in row]


def _speed(prev: list[tuple[float, float, float]] | None, cur: list[tuple[float, float, float]], idx: tuple[int, ...]) -> float:
    if prev is None:
        return 0.0
    acc = 0.0
    n = 0
    for i in idx:
        dx = cur[i][0] - prev[i][0]
        dy = cur[i][1] - prev[i][1]
        if dx == dx and dy == dy:
            acc += (dx * dx + dy * dy) ** 0.5
            n += 1
    return acc / max(n, 1)


class AbductionEngine:
    """Frame-by-frame policy for viewer + bench."""

    def __init__(self) -> None:
        self.smoother = LandmarkSmoother()
        self.session = SessionTracker()
        self.state = State.BOTTOM
        self.dwell = 0
        self.hold = 0
        self.reps = 0
        self.shrug_baseline: float | None = None
        self.rom_cal: float | None = None
        self.peak_abd = 0.0
        self.ascent_frames = 0
        self.descent_frames = 0
        self.top_frames = 0
        self.bent_dwell = DwellFlag(C.ELBOW_DWELL_FRAMES)
        self.shrug_dwell = DwellFlag(C.SHRUG_DWELL_FRAMES)
        self.lean_dwell = DwellFlag(C.TRUNK_DWELL_FRAMES)
        self.plane_dwell = DwellFlag(C.PLANE_DWELL_FRAMES)
        self.cooldown = Cooldown(C.CUE_COOLDOWN_FRAMES)
        self.speak_gate = SpeakGate(C.TTS_HOLD_FRAMES)
        self.prev_sm: list[tuple[float, float, float]] | None = None
        self.t = 0.0
        self.frame = -1
        self.flag_hist: deque[dict] = deque(maxlen=C.HIST_REPS)
        self.vis_ok = 0
        self.paused_n = 0
        self.bent_n = 0
        self.shrug_n = 0

    def step(
        self,
        xyz_row,
        vis_row,
        detected: bool,
        world_row=None,
    ) -> FrameEvent:
        self.frame += 1
        self.t += 1.0 / C.FPS
        self.cooldown.tick()
        sm = F.as_2d(self.smoother.apply(self.t, _xyz_tuple(np.asarray(xyz_row))))
        req = C.ABDUCTION_REQUIRED
        vis_arr = np.asarray(vis_row, dtype=np.float64)
        mean_vis = float(np.mean(vis_arr[list(req)])) if detected else 0.0
        pose_ok = bool(detected) and mean_vis >= C.VISIBILITY_MIN and not any(
            np.isnan(sm[j][0]) for j in req
        )
        paused = not pose_ok
        if pose_ok:
            self.vis_ok += 1
        else:
            self.paused_n += 1

        abd = F.abduction_angle(sm, True) if pose_ok else float("nan")
        abd_l = F.abduction_angle(sm, False) if pose_ok else float("nan")
        elb = F.elbow_angle(sm, True) if pose_ok else float("nan")
        lean = F.trunk_lean_signed(sm) if pose_ok else float("nan")
        gap = F.shrug_gap_norm(sm, True) if pose_ok else float("nan")
        w_over = F.shoulder_width_over_torso(sm) if pose_ok else float("nan")
        ysplit = F.shoulder_y_split(sm) if pose_ok else float("nan")
        limb = _speed(self.prev_sm, sm, (C.RIGHT_WRIST, C.RIGHT_ELBOW, C.LEFT_WRIST, C.LEFT_ELBOW))
        ankle = _speed(self.prev_sm, sm, (C.LEFT_ANKLE, C.RIGHT_ANKLE))
        if pose_ok:
            self.prev_sm = sm
        else:
            self.prev_sm = None

        plane = float("nan")
        if pose_ok and world_row is not None:
            wr = _xyz_tuple(np.asarray(world_row))
            if not np.isnan(wr[C.RIGHT_SHOULDER][0]):
                plane = F.humerus_coronal_dev_world(wr, True)

        raised = pose_ok and (not np.isnan(abd)) and abd >= C.RAISED_ABD_DEG

        if pose_ok and self.state == State.BOTTOM and not np.isnan(gap):
            if self.shrug_baseline is None:
                self.shrug_baseline = gap
            else:
                self.shrug_baseline = 0.98 * self.shrug_baseline + 0.02 * gap

        bent_raw = raised and (not np.isnan(elb)) and elb < C.ELBOW_BENT_DEG
        shrug_raw = (
            raised
            and self.shrug_baseline is not None
            and not np.isnan(gap)
            and gap < self.shrug_baseline * (1.0 - C.SHRUG_DROP_FRAC)
        )
        lean_raw = raised and (not np.isnan(lean)) and abs(lean) > C.TRUNK_LEAN_DEG
        plane_raw = raised and (not np.isnan(plane)) and plane > C.PLANE_FWD_DEG
        if not raised:
            self.bent_dwell._run = 0
            self.shrug_dwell._run = 0
            self.lean_dwell._run = 0
            self.plane_dwell._run = 0
        bent = self.bent_dwell.update(bent_raw) if raised else self.bent_dwell.latched
        shrug = self.shrug_dwell.update(shrug_raw) if raised else self.shrug_dwell.latched
        trunk = self.lean_dwell.update(lean_raw) if raised else self.lean_dwell.latched
        plane_f = self.plane_dwell.update(plane_raw) if raised else self.plane_dwell.latched
        if bent:
            self.bent_n += 1
        if shrug:
            self.shrug_n += 1

        counted = False
        incomplete = False
        too_fast = False
        no_hold = False
        if paused:
            self.dwell = 0
            self.hold = 0
        else:
            if not np.isnan(abd):
                self.peak_abd = max(self.peak_abd, abd)
            if self.state == State.BOTTOM:
                if abd > C.ABDUCTION_BOTTOM_LEAVE:
                    self.dwell += 1
                    if self.dwell >= C.ABDUCTION_MIN_DWELL_FRAMES:
                        self.state = State.ASCENT
                        self.dwell = 0
                        self.peak_abd = abd
                        self.ascent_frames = 0
                        self.descent_frames = 0
                        self.top_frames = 0
                        self.bent_dwell.reset()
                        self.shrug_dwell.reset()
                        self.lean_dwell.reset()
                        self.plane_dwell.reset()
                else:
                    self.dwell = 0
            elif self.state == State.ASCENT:
                self.ascent_frames += 1
                if abd >= C.ABDUCTION_TOP_ENTER:
                    self.hold += 1
                    if self.hold >= C.ABDUCTION_HOLD_FRAMES:
                        self.state = State.TOP
                        self.dwell = 0
                        self.hold = 0
                elif abd < C.ABDUCTION_BOTTOM_ENTER:
                    self.state = State.BOTTOM
                    self.dwell = 0
                    self.hold = 0
                    self.bent_dwell.reset()
                    self.shrug_dwell.reset()
                    self.lean_dwell.reset()
                    self.plane_dwell.reset()
            elif self.state == State.TOP:
                self.top_frames += 1
                if abd < C.ABDUCTION_TOP_LEAVE:
                    self.dwell += 1
                    if self.dwell >= C.ABDUCTION_MIN_DWELL_FRAMES:
                        self.state = State.DESCENT
                        self.dwell = 0
                else:
                    self.dwell = 0
            elif self.state == State.DESCENT:
                self.descent_frames += 1
                if abd < C.ABDUCTION_BOTTOM_ENTER:
                    self.dwell += 1
                    if self.dwell >= C.ABDUCTION_MIN_DWELL_FRAMES:
                        floor = C.ROM_ABS_FLOOR_DEG
                        if self.rom_cal is not None:
                            floor = max(floor, C.ROM_FRAC_OF_CAL * self.rom_cal)
                        incomplete = self.peak_abd < floor
                        if self.rom_cal is None:
                            self.rom_cal = self.peak_abd
                        else:
                            self.rom_cal = 0.8 * self.rom_cal + 0.2 * self.peak_abd
                        too_fast = (
                            self.ascent_frames / C.FPS < C.TOO_FAST_SEC
                            or self.descent_frames / C.FPS < C.TOO_FAST_SEC
                        )
                        no_hold = self.top_frames < C.NO_HOLD_MIN_FRAMES
                        self.state = State.BOTTOM
                        self.dwell = 0
                        counted = True
                        self.peak_abd = 0.0
                elif abd >= C.ABDUCTION_TOP_ENTER:
                    self.state = State.TOP
                    self.dwell = 0

        snap = self.session.update(
            detected=bool(detected),
            vis_ok=pose_ok,
            abd_r=abd if abd == abd else 0.0,
            abd_l=abd_l if abd_l == abd_l else 0.0,
            limb_speed=limb,
            ankle_speed=ankle,
            width_over_torso=w_over,
            shoulder_y_split=ysplit if ysplit == ysplit else 0.0,
            in_bottom=self.state == State.BOTTOM,
            counted_this_frame=counted,
        )
        if not snap.allow_count:
            counted = False
            incomplete = False
            too_fast = False
            no_hold = False
        elif counted:
            self.reps += 1

        form = []
        if incomplete:
            form.append("INCOMPLETE_ROM")
        if bent:
            form.append("BENT_ELBOW")
        if shrug:
            form.append("SHRUG")
        if trunk:
            form.append("TRUNK_LEAN")
        if too_fast:
            form.append("TOO_FAST")
        if no_hold:
            form.append("NO_HOLD")
        if plane_f:
            form.append("PLANE_FWD")
        flags = stack_flags(snap.flags + form)
        if sum(1 for f in flags if f in FORM_FLAGS) >= 2:
            flags = flags + (["MULTI"] if "MULTI" not in flags else [])

        if counted:
            self.flag_hist.append({f: (f in flags) for f in FORM_FLAGS})
            self.bent_dwell.reset()
            self.shrug_dwell.reset()
            self.lean_dwell.reset()
            self.plane_dwell.reset()

        cue = None
        eligible = self.speak_gate.update(flags)
        if self.cooldown.ready() and eligible:
            cue = speak_id([f for f in SPEAK_ORDER if f in eligible])
            if cue:
                self.cooldown.fire()

        return FrameEvent(
            frame=self.frame,
            state=self.state.value,
            session=snap.state,
            abduction_deg=abd,
            elbow_deg=elb,
            trunk_lean_deg=lean if lean == lean else float("nan"),
            shrug_gap=gap if gap == gap else float("nan"),
            paused=paused or not snap.allow_count,
            allow_count=snap.allow_count,
            bent_elbow=bent,
            shrug=shrug,
            trunk_lean=trunk,
            incomplete_rom=incomplete,
            too_fast=too_fast,
            no_hold=no_hold,
            plane_fwd=plane_f,
            flags=flags,
            counted=counted,
            cue=cue,
            cue_text=HUD_COPY.get(cue) if cue else None,
        )


def run_abduction(
    xyz: np.ndarray,
    vis: np.ndarray,
    detected: np.ndarray,
    world_xyz: np.ndarray | None = None,
) -> PolicyResult:
    eng = AbductionEngine()
    events: list[FrameEvent] = []
    for i in range(len(xyz)):
        w = world_xyz[i] if world_xyz is not None else None
        events.append(eng.step(xyz[i], vis[i], bool(detected[i]), w))
    n = max(len(xyz), 1)
    return PolicyResult(
        events=events,
        rep_count=eng.reps,
        bent_elbow_frames=eng.bent_n,
        shrug_frames=eng.shrug_n,
        visible_required_frac=eng.vis_ok / n,
        paused_frac=eng.paused_n / n,
        flag_hist=list(eng.flag_hist),
    )


def cues_on_interval(result: PolicyResult, first: int, last: int) -> dict[str, bool]:
    slice_ev = [e for e in result.events if first <= e.frame <= last]
    keys = (
        "bent_elbow",
        "shrug",
        "trunk_lean",
        "incomplete_rom",
        "too_fast",
        "no_hold",
        "plane_fwd",
    )
    out = {k: any(getattr(e, k) for e in slice_ev) for k in keys}
    out["counted_inside"] = sum(1 for e in slice_ev if e.counted)
    return out
