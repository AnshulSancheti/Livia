"""Frozen MediaPipe Pose Landmarker (VIDEO mode). Does not train or fine-tune."""

from __future__ import annotations

import ssl
import subprocess
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from . import constants as C
from .dataset import CACHE, MODELS

_FPS = C.FPS


def download_models() -> dict[str, Path]:
    MODELS.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, url in (("lite", C.LITE_URL), ("full", C.FULL_URL)):
        dest = MODELS / f"pose_landmarker_{key}.task"
        if not dest.exists():
            print(f"Downloading {dest.name} …")
            try:
                subprocess.run(["curl", "-fsSL", "-o", str(dest), url], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(url, context=ctx) as resp:
                    dest.write_bytes(resp.read())
        out[key] = dest
    return out


def infer_video(video_path: Path, model_path: Path) -> dict:
    """Return dict with xyz (T,33,3), optional world_xyz, vis (T,33), detected (T,), width, height."""
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    xyz_rows = []
    world_rows = []
    vis_rows = []
    detected = []
    idx = 0
    nan33 = [(np.nan, np.nan, np.nan)] * 33
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgb)
            ts = int(idx * 1000.0 / _FPS)
            result = landmarker.detect_for_video(mp_image, ts)
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                xyz_rows.append([(p.x, p.y, p.z) for p in lm])
                vis_rows.append([getattr(p, "visibility", 1.0) for p in lm])
                detected.append(1)
                if result.pose_world_landmarks:
                    wlm = result.pose_world_landmarks[0]
                    world_rows.append([(p.x, p.y, p.z) for p in wlm])
                else:
                    world_rows.append(nan33)
            else:
                xyz_rows.append(nan33)
                world_rows.append(nan33)
                vis_rows.append([0.0] * 33)
                detected.append(0)
            idx += 1
    cap.release()
    return {
        "xyz": np.asarray(xyz_rows, dtype=np.float32),
        "world_xyz": np.asarray(world_rows, dtype=np.float32),
        "vis": np.asarray(vis_rows, dtype=np.float32),
        "detected": np.asarray(detected, dtype=np.uint8),
        "width": w,
        "height": h,
        "n_frames": idx,
    }


def cache_path(video_id: str, exercise: str, variant: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{exercise}_{video_id}_c17_{variant}.npz"


def load_or_infer(video_path: Path, video_id: str, exercise: str, variant: str, model_path: Path) -> dict:
    path = cache_path(video_id, exercise, variant)
    if path.exists():
        data = np.load(path)
        return {k: data[k] for k in data.files}
    print(f"Infer {variant} {exercise}/{video_id} …")
    payload = infer_video(video_path, model_path)
    np.savez_compressed(path, **payload)
    return payload
