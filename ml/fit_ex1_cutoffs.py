"""Recompute Ex1 front 2D percentiles from cached lite npz. Evaluation only."""

from __future__ import annotations

import json

import numpy as np

from . import constants as C
from . import features as F
from .dataset import camera17_mp4, front_c17_videos, reps_for_video
from .one_euro import LandmarkSmoother
from .pose import download_models, load_or_infer


def _xyz_tuple(row) -> list[tuple[float, float, float]]:
    return [(float(p[0]), float(p[1]), float(p[2])) for p in row]


def main() -> None:
    models = download_models()
    vids = front_c17_videos(1)
    raised_elb_c, raised_elb_i = [], []
    peak_c, peak_i = [], []
    lean_c, lean_i = [], []
    gap_drop_c, gap_drop_i = [], []

    for vid in vids:
        path = camera17_mp4("Ex1", vid)
        payload = load_or_infer(path, vid, "Ex1", "lite", models["lite"])
        xyz, vis, det = payload["xyz"], payload["vis"], payload["detected"]
        smth = LandmarkSmoother()
        t = 0.0
        dt = 1.0 / C.FPS
        abd = np.full(len(xyz), np.nan)
        elb = np.full(len(xyz), np.nan)
        lean = np.full(len(xyz), np.nan)
        gap = np.full(len(xyz), np.nan)
        for i in range(len(xyz)):
            t += dt
            sm = F.as_2d(smth.apply(t, _xyz_tuple(xyz[i])))
            if not det[i]:
                continue
            abd[i] = F.abduction_angle(sm, True)
            elb[i] = F.elbow_angle(sm, True)
            lean[i] = F.trunk_lean_signed(sm)
            gap[i] = F.shrug_gap_norm(sm, True)

        for r in reps_for_video(vid, 1):
            sl = slice(r.first_frame, r.last_frame + 1)
            a, e, ln, g = abd[sl], elb[sl], lean[sl], gap[sl]
            raised = a >= C.RAISED_ABD_DEG
            if not np.any(raised):
                continue
            e_min = float(np.nanmin(e[raised]))
            peak = float(np.nanmax(a))
            lean_max = float(np.nanmax(np.abs(ln[raised])))
            rest = g[~raised]
            rest_m = float(np.nanmedian(rest)) if np.any(~raised) else float(np.nanmedian(g))
            hike = float(np.nanmin(g[raised]))
            drop = (rest_m - hike) / rest_m if rest_m and rest_m == rest_m and rest_m > 1e-6 else float("nan")
            if r.correctness == 1:
                raised_elb_c.append(e_min)
                peak_c.append(peak)
                lean_c.append(lean_max)
                gap_drop_c.append(drop)
            else:
                raised_elb_i.append(e_min)
                peak_i.append(peak)
                lean_i.append(lean_max)
                gap_drop_i.append(drop)

    def pct(xs, q):
        xs = [x for x in xs if x == x]
        return float(np.percentile(xs, q)) if xs else float("nan")

    out = {
        "n_correct": len(raised_elb_c),
        "n_incorrect": len(raised_elb_i),
        "elbow_min_raised": {
            "correct_p10": pct(raised_elb_c, 10),
            "correct_p50": pct(raised_elb_c, 50),
            "incorrect_p10": pct(raised_elb_i, 10),
            "incorrect_p50": pct(raised_elb_i, 50),
        },
        "peak_abd": {
            "correct_p10": pct(peak_c, 10),
            "correct_p50": pct(peak_c, 50),
            "incorrect_p10": pct(peak_i, 10),
            "incorrect_p50": pct(peak_i, 50),
        },
        "abs_trunk_lean_raised": {
            "correct_p90": pct(lean_c, 90),
            "correct_p50": pct(lean_c, 50),
            "incorrect_p90": pct(lean_i, 90),
        },
        "shrug_gap_drop_frac": {
            "correct_p90": pct(gap_drop_c, 90),
            "incorrect_p50": pct(gap_drop_i, 50),
            "incorrect_p90": pct(gap_drop_i, 90),
        },
        "suggested": {},
    }
    e_cut = (out["elbow_min_raised"]["correct_p10"] + out["elbow_min_raised"]["incorrect_p10"]) / 2
    rom_cut = (out["peak_abd"]["correct_p10"] + out["peak_abd"]["incorrect_p10"]) / 2
    lean_cut = out["abs_trunk_lean_raised"]["correct_p90"]
    shrug_cut = out["shrug_gap_drop_frac"]["correct_p90"]
    out["suggested"] = {
        "ELBOW_BENT_DEG": round(e_cut, 1),
        "ROM_ABS_FLOOR_DEG": round(rom_cut, 1),
        "TRUNK_LEAN_DEG": round(lean_cut, 1),
        "SHRUG_DROP_FRAC": round(shrug_cut, 3),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
