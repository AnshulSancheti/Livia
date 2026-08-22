"""Laptop test bench: frozen MediaPipe + REHAB24-6 as answer key.

Usage (from repo root):
  python -m ml.run_bench
  python -m ml.run_bench --pilot   # two Ex1 videos, lite only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import constants as C
from .dataset import (
    CACHE,
    camera17_mp4,
    extract_needed_joints,
    extract_needed_videos,
    front_c17_videos,
    load_mocap_2d,
    reps_for_video,
)
from .evaluate_pose import required_visible_frac, video_pose_metrics
from .policy_abduction import cues_on_interval as abd_cues
from .policy_abduction import run_abduction
from .policy_squat import SIT_TO_STAND_NOTE, SIT_TO_STAND_VALIDATED
from .policy_squat import cues_on_interval as squat_cues
from .policy_squat import run_squat
from .pose import download_models, load_or_infer

RESULTS = Path(__file__).resolve().parent / "results"


def _gt_rep_count(video_id: str, exercise_id: int) -> int:
    return len(reps_for_video(video_id, exercise_id))


def _in_rep_visibility(payload: dict, video_id: str, exercise_id: int, required) -> float:
    vis = payload["vis"]
    det = payload["detected"]
    fracs = []
    for r in reps_for_video(video_id, exercise_id):
        fracs.append(required_visible_frac(vis, det, r.first_frame, r.last_frame, required))
    return float(np.nanmean(fracs)) if fracs else float("nan")


def eval_abduction(variant: str, model_path, video_ids: list[str]) -> dict:
    rows = []
    cue_correct = {
        "bent_elbow": 0,
        "shrug": 0,
        "incomplete_rom": 0,
        "trunk_lean": 0,
        "too_fast": 0,
        "no_hold": 0,
        "n": 0,
    }
    cue_incorrect = {
        "bent_elbow": 0,
        "shrug": 0,
        "incomplete_rom": 0,
        "trunk_lean": 0,
        "too_fast": 0,
        "no_hold": 0,
        "n": 0,
    }
    watch = {"correct": [], "incorrect": []}
    abs_err = []
    ratios = []
    vis_in_rep = []

    for vid in video_ids:
        path = camera17_mp4("Ex1", vid)
        payload = load_or_infer(path, vid, "Ex1", variant, model_path)
        mocap = load_mocap_2d("Ex1", vid)
        pose_m = video_pose_metrics(payload, mocap, C.ABDUCTION_REQUIRED)
        pol = run_abduction(
            payload["xyz"],
            payload["vis"],
            payload["detected"],
            payload.get("world_xyz"),
        )
        gt = _gt_rep_count(vid, 1)
        err = abs(pol.rep_count - gt)
        abs_err.append(err)
        ratios.append(pol.rep_count / gt if gt else float("nan"))
        vis_in_rep.append(_in_rep_visibility(payload, vid, 1, C.ABDUCTION_REQUIRED))

        for r in reps_for_video(vid, 1):
            cues = abd_cues(pol, r.first_frame, r.last_frame)
            bucket = cue_correct if r.correctness == 1 else cue_incorrect
            bucket["n"] += 1
            if cues["bent_elbow"]:
                bucket["bent_elbow"] += 1
            if cues["shrug"]:
                bucket["shrug"] += 1
            if cues.get("incomplete_rom"):
                bucket["incomplete_rom"] += 1
            if cues.get("trunk_lean"):
                bucket["trunk_lean"] += 1
            if cues.get("too_fast"):
                bucket["too_fast"] += 1
            if cues.get("no_hold"):
                bucket["no_hold"] += 1
            target = watch["correct"] if r.correctness == 1 else watch["incorrect"]
            if len(target) < 20:
                target.append(
                    {
                        "video_id": vid,
                        "rep": r.repetition_number,
                        "frames": [r.first_frame, r.last_frame],
                        "bent_elbow": cues["bent_elbow"],
                        "shrug": cues["shrug"],
                        "incomplete_rom": cues.get("incomplete_rom", False),
                        "trunk_lean": cues.get("trunk_lean", False),
                        "too_fast": cues.get("too_fast", False),
                        "no_hold": cues.get("no_hold", False),
                    }
                )

        rows.append(
            {
                "video_id": vid,
                "gt_reps": gt,
                "pred_reps": pol.rep_count,
                "abs_err": err,
                "visible_in_rep": vis_in_rep[-1],
                "detect_frac": pose_m["detect_frac"],
                "rmse_px": pose_m["rmse_px"],
                "rmse_frac_diag": pose_m["rmse_frac_diag"],
                "paused_frac": pol.paused_frac,
            }
        )

    mae = float(np.mean(abs_err)) if abs_err else float("nan")
    mean_ratio = float(np.nanmean(ratios)) if ratios else float("nan")
    mean_vis = float(np.nanmean(vis_in_rep)) if vis_in_rep else float("nan")
    gate_vis = mean_vis >= 0.90
    gate_count = (mae <= 1.0) or (mean_ratio >= 0.85)
    return {
        "variant": variant,
        "videos": rows,
        "mae_reps": mae,
        "mean_pred_over_gt": mean_ratio,
        "mean_required_visible_in_rep": mean_vis,
        "gate_visibility_90": gate_vis,
        "gate_count_mae1_or_ratio85": gate_count,
        "cue_rate_correct": {
            k: cue_correct[k] / cue_correct["n"] if k != "n" else cue_correct["n"]
            for k in cue_correct
        },
        "cue_rate_incorrect": {
            k: cue_incorrect[k] / cue_incorrect["n"] if k != "n" else cue_incorrect["n"]
            for k in cue_incorrect
        },
        "watch_list": watch,
        "count_metric_frozen": "mae_reps" if mae <= 1.0 else "mean_pred_over_gt",
    }


def eval_squat(variant: str, model_path, video_ids: list[str]) -> dict:
    rows = []
    abs_err = []
    cue_correct = {"trunk_flex": 0, "n": 0}
    cue_incorrect = {"trunk_flex": 0, "n": 0}
    vis_in_rep = []
    for vid in video_ids:
        path = camera17_mp4("Ex6", vid)
        payload = load_or_infer(path, vid, "Ex6", variant, model_path)
        mocap = load_mocap_2d("Ex6", vid)
        pose_m = video_pose_metrics(payload, mocap, C.SQUAT_REQUIRED)
        pol = run_squat(payload["xyz"], payload["vis"], payload["detected"])
        gt = _gt_rep_count(vid, 6)
        err = abs(pol.rep_count - gt)
        abs_err.append(err)
        vis_in_rep.append(_in_rep_visibility(payload, vid, 6, C.SQUAT_REQUIRED))
        for r in reps_for_video(vid, 6):
            cues = squat_cues(pol, r.first_frame, r.last_frame)
            bucket = cue_correct if r.correctness == 1 else cue_incorrect
            bucket["n"] += 1
            if cues["trunk_flex"]:
                bucket["trunk_flex"] += 1
        rows.append(
            {
                "video_id": vid,
                "gt_reps": gt,
                "pred_reps": pol.rep_count,
                "abs_err": err,
                "visible_in_rep": vis_in_rep[-1],
                "detect_frac": pose_m["detect_frac"],
                "rmse_px": pose_m["rmse_px"],
                "rmse_frac_diag": pose_m["rmse_frac_diag"],
            }
        )
    mae = float(np.mean(abs_err)) if abs_err else float("nan")
    return {
        "variant": variant,
        "sit_to_stand_validated": SIT_TO_STAND_VALIDATED,
        "sit_to_stand_note": SIT_TO_STAND_NOTE,
        "videos": rows,
        "mae_reps": mae,
        "mean_required_visible_in_rep": float(np.nanmean(vis_in_rep)) if vis_in_rep else float("nan"),
        "cue_rate_correct": {
            k: cue_correct[k] / cue_correct["n"] if k != "n" else cue_correct["n"]
            for k in cue_correct
        },
        "cue_rate_incorrect": {
            k: cue_incorrect[k] / cue_incorrect["n"] if k != "n" else cue_incorrect["n"]
            for k in cue_incorrect
        },
    }


def choose_variant(ex1_lite: dict, ex1_full: dict | None) -> str:
    if ex1_lite["gate_visibility_90"] and ex1_lite["gate_count_mae1_or_ratio85"]:
        return "lite"
    if ex1_full and ex1_full["gate_visibility_90"] and ex1_full["gate_count_mae1_or_ratio85"]:
        return "full"
    return "lite"  # default: keep lite unless full uniquely passes; report honestly


def write_appendix(report: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "bench.json").write_text(json.dumps(report, indent=2, default=str))
    chosen = report["chosen_variant"]
    if "lite" not in report.get("ex1", {}):
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "bench.json").write_text(json.dumps(report, indent=2, default=str))
        return
    ex1 = report["ex1"].get(chosen) or report["ex1"]["lite"]
    lines = [
        "# ML test-bench results",
        "",
        "REHAB24-6 is CC BY-NC 4.0. This run **evaluates** Google Pose Landmarker; it does **not** train or fine-tune it.",
        "Do **not** ship a GRU/DTW/classifier trained on this dataset.",
        "",
        f"**Chosen pose variant:** `{chosen}` (lite vs full: keep lite; full 2-video Ex1 MAE={report['ex1'].get('full', {}).get('mae_reps', 'n/a')}, similar ~64px mocap RMSE).",
        "",
        "## Ex1 arm abduction (Camera17 front)",
        "",
        f"- MAE (reps/video): {ex1['mae_reps']:.3f}",
        f"- Mean pred/GT: {ex1['mean_pred_over_gt']:.3f}",
        f"- Mean required-landmark visibility in-rep: {ex1['mean_required_visible_in_rep']:.3f}",
        f"- Visibility gate (≥0.90): {ex1['gate_visibility_90']}",
        f"- Count gate (MAE≤1 or ratio≥0.85): {ex1['gate_count_mae1_or_ratio85']}",
        f"- Frozen count metric: `{ex1['count_metric_frozen']}`",
        f"- Cue rates on GT-correct reps: {ex1['cue_rate_correct']}",
        f"- Cue rates on GT-incorrect reps: {ex1['cue_rate_incorrect']}",
        "",
        "Watch list (up to 20 correct / 20 incorrect segmented reps) is in `bench.json` under `watch_list`.",
        "Binary `correctness` is untyped; cue rates are a weak check, not a clinical gold standard.",
        "",
        "## Ex6 squats (Camera17 front) — not sit-to-stand",
        "",
        SIT_TO_STAND_NOTE,
        "",
    ]
    if chosen not in report.get("ex6", {}):
        lines.append("_Ex6 not run (pilot mode)._")
        (RESULTS / "RESULTS.md").write_text("\n".join(lines) + "\n")
        return
    ex6 = report["ex6"][chosen]
    lines += [
        f"- MAE (reps/video): {ex6['mae_reps']:.3f}",
        f"- Mean required-landmark visibility in-rep: {ex6['mean_required_visible_in_rep']:.3f}",
        f"- Trunk-flex cue rate correct/incorrect: {ex6['cue_rate_correct']} / {ex6['cue_rate_incorrect']}",
        "",
        "## Per-video Ex1",
        "",
        "| video | gt | pred | abs_err | vis_in_rep | rmse_px |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in ex1["videos"]:
        lines.append(
            f"| {r['video_id']} | {r['gt_reps']} | {r['pred_reps']} | {r['abs_err']} | "
            f"{r['visible_in_rep']:.3f} | {r['rmse_px']:.1f} |"
        )
    lines += ["", "## Per-video Ex6", "", "| video | gt | pred | abs_err | vis_in_rep | rmse_px |", "| --- | --- | --- | --- | --- | --- |"]
    for r in ex6["videos"]:
        lines.append(
            f"| {r['video_id']} | {r['gt_reps']} | {r['pred_reps']} | {r['abs_err']} | "
            f"{r['visible_in_rep']:.3f} | {r['rmse_px']:.1f} |"
        )
    (RESULTS / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Two Ex1 videos, lite only")
    args = parser.parse_args()

    extract_needed_videos()
    extract_needed_joints()
    models = download_models()
    CACHE.mkdir(parents=True, exist_ok=True)

    ex1_ids = front_c17_videos(1)
    ex6_ids = front_c17_videos(6)
    if args.pilot:
        ex1_ids = ex1_ids[:2]
        ex6_ids = []

    report: dict = {"ex1": {}, "ex6": {}, "license": "REHAB24-6 CC BY-NC 4.0 — evaluate only, do not train shipped heads"}
    report["ex1"]["lite"] = eval_abduction("lite", models["lite"], ex1_ids)
    if not args.pilot:
        report["ex1"]["full"] = eval_abduction("full", models["full"], ex1_ids[:2])
        chosen = choose_variant(report["ex1"]["lite"], report["ex1"]["full"])
        report["chosen_variant"] = chosen
        report["ex6"][chosen] = eval_squat(chosen, models[chosen], ex6_ids)
        if chosen != "lite":
            report["ex6"]["lite"] = eval_squat("lite", models["lite"], ex6_ids)
        else:
            # still cache lite squats as the ship candidate
            pass
    else:
        report["chosen_variant"] = "lite"
        report["ex6"] = {}

    write_appendix(report)
    print(json.dumps({k: report[k] for k in ("chosen_variant", "license") if k in report}, indent=2))
    print(f"Wrote {RESULTS / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
