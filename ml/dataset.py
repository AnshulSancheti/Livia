"""Load REHAB24-6 labels and zip members. Dataset is evaluation-only (CC BY-NC)."""

from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "ml" / "cache"
MODELS = ROOT / "ml" / "models"
VIDEOS_DIR = DATA / "extracted" / "videos"
JOINTS_DIR = DATA / "extracted" / "joints_2d"


@dataclass(frozen=True)
class Rep:
    video_id: str
    repetition_number: int
    exercise_id: int
    person_id: int
    first_frame: int
    last_frame: int
    cam17_orientation: str
    mocap_erroneous: int
    exercise_subtype: str
    extra_person_in_cam17: int
    extra_person_in_cam18: int
    correctness: int


def load_reps() -> list[Rep]:
    path = DATA / "Segmentation.csv"
    out: list[Rep] = []
    with path.open() as f:
        for row in csv.DictReader(f, delimiter=";"):
            out.append(
                Rep(
                    video_id=row["video_id"],
                    repetition_number=int(row["repetition_number"]),
                    exercise_id=int(row["exercise_id"]),
                    person_id=int(row["person_id"]),
                    first_frame=int(row["first_frame"]),
                    last_frame=int(row["last_frame"]),
                    cam17_orientation=row["cam17_orientation"],
                    mocap_erroneous=int(row["mocap_erroneous"]),
                    exercise_subtype=row.get("exercise_subtype") or "",
                    extra_person_in_cam17=int(row["extra_person_in_cam17"]),
                    extra_person_in_cam18=int(row["extra_person_in_cam18"]),
                    correctness=int(row["correctness"]),
                )
            )
    return out


def front_c17_videos(exercise_id: int) -> list[str]:
    vids = []
    seen = set()
    for r in load_reps():
        if (
            r.exercise_id == exercise_id
            and r.cam17_orientation == "front"
            and r.mocap_erroneous == 0
            and r.video_id not in seen
        ):
            seen.add(r.video_id)
            vids.append(r.video_id)
    return sorted(vids)


def reps_for_video(video_id: str, exercise_id: int) -> list[Rep]:
    return [
        r
        for r in load_reps()
        if r.video_id == video_id
        and r.exercise_id == exercise_id
        and r.cam17_orientation == "front"
        and r.mocap_erroneous == 0
    ]


def camera17_mp4(exercise: str, video_id: str) -> Path:
    return VIDEOS_DIR / exercise / f"{video_id}-Camera17-30fps.mp4"


def mocap_c17_npy(exercise: str, video_id: str) -> Path:
    return JOINTS_DIR / exercise / f"{video_id}-c17-30fps.npy"


def extract_needed_videos() -> None:
    zpath = DATA / "videos.zip"
    wanted_suffix = "-Camera17-30fps.mp4"
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.endswith(wanted_suffix):
                continue
            if not (name.startswith("Ex1/") or name.startswith("Ex6/")):
                continue
            dest = VIDEOS_DIR / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                continue
            dest.write_bytes(z.read(name))


def extract_needed_joints() -> None:
    zpath = DATA / "2d_joints.zip"
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.endswith("-c17-30fps.npy"):
                continue
            if not (name.startswith("Ex1/") or name.startswith("Ex6/")):
                continue
            dest = JOINTS_DIR / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                continue
            dest.write_bytes(z.read(name))


def load_mocap_2d(exercise: str, video_id: str) -> np.ndarray:
    """(T, 26, 2) pixel coordinates."""
    return np.load(mocap_c17_npy(exercise, video_id))
