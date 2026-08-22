"""Watch pose mapping: dataset video playback or laptop webcam.

Dataset (uses cached landmarks if present, otherwise infers):
  python -m ml.view --video PM_023

Webcam (same lite .task, VIDEO timestamps — close to LIVE_STREAM):
  python -m ml.view --camera

Keys: q quit, space pause (video only).
"""

from __future__ import annotations

import argparse
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .dataset import camera17_mp4, extract_needed_videos
from .heuristics import HUD_COPY
from .policy_abduction import AbductionEngine, run_abduction
from .pose import download_models, load_or_infer

# BlazePose 33 topology (same as MediaPipe pose connections).
POSE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
)
RIGHT_ARM_EDGES = ((12, 14), (14, 16))


def _pt(lm, w: int, h: int) -> tuple[int, int] | None:
    if np.isnan(lm[0]) or np.isnan(lm[1]):
        return None
    return int(lm[0] * w), int(lm[1] * h)


def draw_pose(frame, xyz_row, vis_row, vis_min: float = 0.5, arm_bgr=(0, 200, 255)) -> None:
    h, w = frame.shape[:2]
    pts = []
    for i in range(33):
        p = _pt(xyz_row[i], w, h)
        pts.append(p)
        if p is None or vis_row[i] < vis_min:
            continue
        cv2.circle(frame, p, 4, (0, 255, 0), -1, cv2.LINE_AA)
    for a, b in POSE_EDGES:
        pa, pb = pts[a], pts[b]
        if pa is None or pb is None:
            continue
        if vis_row[a] < vis_min or vis_row[b] < vis_min:
            continue
        color = arm_bgr if (a, b) in RIGHT_ARM_EDGES or (b, a) in RIGHT_ARM_EDGES else (0, 200, 255)
        cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)


def _arm_color(ev) -> tuple[int, int, int]:
    if ev is None:
        return (0, 200, 255)
    if ev.bent_elbow:
        return (0, 0, 255)
    if ev.shrug or ev.trunk_lean:
        return (0, 140, 255)
    if ev.incomplete_rom or ev.too_fast:
        return (0, 200, 255)
    return (0, 200, 255)


def _ex1_hud(ev, counted: int, extra: str = "") -> list[str]:
    abd = ev.abduction_deg if ev.abduction_deg == ev.abduction_deg else float("nan")
    elb = ev.elbow_deg if ev.elbow_deg == ev.elbow_deg else float("nan")
    lines = [
        extra,
        f"reps {counted}  {ev.session}/{ev.state}  abd {abd:.0f}  elb {elb:.0f}",
    ]
    stack = [HUD_COPY[f] for f in ev.flags if f in HUD_COPY]
    if stack:
        lines.append(" | ".join(stack[:4]))
    if ev.cue_text:
        lines.append(f"speak: {ev.cue_text}")
    return [ln for ln in lines if ln]


def _hud(frame, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


def _landmarker(model_path, mode):
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=mode,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _result_row(result):
    nan = [(np.nan, np.nan, np.nan)] * 33
    if not result.pose_landmarks:
        return nan, [0.0] * 33, False, nan
    lm = result.pose_landmarks[0]
    xyz = [(p.x, p.y, p.z) for p in lm]
    vis = [getattr(p, "visibility", 1.0) for p in lm]
    if result.pose_world_landmarks:
        wlm = result.pose_world_landmarks[0]
        world = [(p.x, p.y, p.z) for p in wlm]
    else:
        world = nan
    return xyz, vis, True, world


def play_dataset_video(video_id: str, exercise: str = "Ex1") -> None:
    extract_needed_videos()
    models = download_models()
    path = camera17_mp4(exercise, video_id)
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run extract or pick another id.")
    payload = load_or_infer(path, video_id, exercise, "lite", models["lite"])
    xyz, vis, det = payload["xyz"], payload["vis"], payload["detected"]
    world = payload["world_xyz"] if "world_xyz" in payload else None
    pol = run_abduction(xyz, vis, det, world) if exercise == "Ex1" else None
    counted = 0
    cap = cv2.VideoCapture(str(path))
    paused = False
    idx = 0
    win = f"Livia pose — {exercise}/{video_id} (q quit, space pause)"
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            if idx < len(xyz):
                ev = pol.events[idx] if pol is not None else None
                draw_pose(frame, xyz[idx], vis[idx], arm_bgr=_arm_color(ev))
                if pol is not None:
                    if ev.counted:
                        counted += 1
                    lines = _ex1_hud(ev, counted, f"{exercise} {video_id}  frame {idx}")
                else:
                    lines = [f"{exercise} {video_id}  frame {idx}"]
                _hud(frame, lines)
            cv2.imshow(win, frame)
            idx += 1
        key = cv2.waitKey(33 if not paused else 50) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
    cap.release()
    cv2.destroyAllWindows()


def play_camera() -> None:
    models = download_models()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera 0.")
    win = "Livia pose — webcam (q quit)"
    idx = 0
    t0 = time.time()
    counted = 0
    eng = AbductionEngine()
    with _landmarker(models["lite"], vision.RunningMode.VIDEO) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
            ts = int((time.time() - t0) * 1000)
            result = landmarker.detect_for_video(mp_image, ts)
            xyz, vis, found, world = _result_row(result)
            ev = eng.step(xyz, vis, found, world)
            if ev.counted:
                counted += 1
            if found:
                draw_pose(frame, xyz, vis, arm_bgr=_arm_color(ev))
            fps = idx / max(time.time() - t0, 1e-6)
            lines = _ex1_hud(ev, counted, f"webcam  lite.task  {fps:.0f} fps")
            _hud(frame, lines)
            cv2.imshow(win, frame)
            idx += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="Live or dataset pose overlay")
    p.add_argument("--camera", action="store_true", help="Laptop webcam")
    p.add_argument("--video", default="PM_023", help="REHAB24-6 video_id")
    p.add_argument("--ex", default="Ex1", choices=("Ex1", "Ex6"))
    args = p.parse_args()
    if args.camera:
        play_camera()
    else:
        play_dataset_video(args.video, args.ex)


if __name__ == "__main__":
    main()
