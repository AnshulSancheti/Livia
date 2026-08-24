"""Render a side-view sit-to-stand prototype overlay on an arbitrary video.

Usage:
  python -m ml.render_sts input.mp4 output.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import constants as C
from .policy_sts import run_sts
from .pose import infer_video
from .view import POSE_EDGES


def _point(row, index: int, width: int, height: int) -> tuple[int, int] | None:
    x, y = row[index][:2]
    if np.isnan(x) or np.isnan(y):
        return None
    return int(x * width), int(y * height)


def _draw_pose(frame, row, visibility, right: bool) -> None:
    height, width = frame.shape[:2]
    points = [_point(row, i, width, height) for i in range(33)]
    selected = (
        {C.RIGHT_SHOULDER, C.RIGHT_HIP, C.RIGHT_KNEE, C.RIGHT_ANKLE}
        if right
        else {C.LEFT_SHOULDER, C.LEFT_HIP, C.LEFT_KNEE, C.LEFT_ANKLE}
    )
    for a, b in POSE_EDGES:
        if points[a] is None or points[b] is None:
            continue
        if visibility[a] < 0.5 or visibility[b] < 0.5:
            continue
        selected_edge = a in selected and b in selected
        color = (80, 240, 140) if selected_edge else (0, 200, 255)
        thickness = 4 if selected_edge else 2
        cv2.line(frame, points[a], points[b], color, thickness, cv2.LINE_AA)
    for i, point in enumerate(points):
        if point is None or visibility[i] < 0.5:
            continue
        color = (80, 240, 140) if i in selected else (0, 255, 0)
        cv2.circle(frame, point, 5 if i in selected else 3, color, -1, cv2.LINE_AA)


def _put(frame, text: str, point, scale: float, color, thickness: int = 2) -> None:
    cv2.putText(frame, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render(input_path: Path, output_path: Path, model_path: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="livia-sts-") as work_raw:
        work = Path(work_raw)
        normalized = work / "input-30fps.mp4"
        raw_output = work / "annotated-raw.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(input_path), "-vf", "fps=30", "-an",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                str(normalized),
            ],
            check=True,
        )
        payload = infer_video(normalized, model_path)
        result = run_sts(payload["xyz"], payload["vis"], payload["detected"])

        cap = cv2.VideoCapture(str(normalized))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), C.FPS, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("Could not open annotated video writer")

        count = 0
        feedback = "Calibrating visible side"
        feedback_until = int(C.FPS * 2)
        first_rise = next((e.frame for e in result.events if e.state == "RISING"), None)
        fifth_frame = next(
            (e.frame for e in result.events if e.counted and sum(x.counted for x in result.events[: e.frame + 1]) == 5),
            None,
        )
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok or index >= len(result.events):
                break
            event = result.events[index]
            if payload["detected"][index]:
                _draw_pose(frame, payload["xyz"][index], payload["vis"][index], result.side == "right")
            if event.counted:
                count += 1
                if count == 5 and result.five_rep_time_sec is not None:
                    feedback = f"Five-rep time: {result.five_rep_time_sec:.1f} seconds"
                else:
                    feedback = f"Rep {count} complete"
                feedback_until = index + int(C.FPS * 2.5)
            elif event.cue_text:
                feedback = event.cue_text
                feedback_until = index + int(C.FPS * 2)
            elif index > feedback_until:
                feedback = f"Tracking {result.side} hip, knee and ankle"

            if first_rise is None or index < first_rise:
                timer = 0.0
            elif fifth_frame is not None and index > fifth_frame and result.five_rep_time_sec is not None:
                timer = result.five_rep_time_sec
            else:
                timer = (index - first_rise + 1) / C.FPS

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 168), (12, 18, 28), -1)
            cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
            knee = event.knee_deg if event.knee_deg == event.knee_deg else 0.0
            _put(frame, "LIVIA  |  SIT-TO-STAND", (16, 28), 0.58, (255, 255, 255))
            _put(frame, f"REPS  {count}", (16, 66), 0.84, (92, 230, 140))
            _put(frame, f"{event.state.title()}  {knee:.0f} deg", (190, 66), 0.64, (255, 255, 255))
            _put(frame, f"5-REP TIMER  {timer:04.1f}s", (16, 101), 0.54, (180, 210, 235))
            _put(frame, feedback, (16, 132), 0.48, (255, 255, 255))
            _put(frame, "PROTOTYPE - NOT CLINICALLY VALIDATED", (16, 158), 0.38, (90, 190, 255), 1)

            progress = index / max(len(result.events) - 1, 1)
            cv2.rectangle(frame, (16, height - 24), (width - 16, height - 17), (50, 58, 68), -1)
            cv2.rectangle(
                frame,
                (16, height - 24),
                (16 + int((width - 32) * progress), height - 17),
                (92, 230, 140),
                -1,
            )
            _put(frame, f"{index / C.FPS:04.1f}s", (16, height - 34), 0.43, (255, 255, 255), 1)
            writer.write(frame)
            index += 1

        cap.release()
        writer.release()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(raw_output), "-i", str(input_path),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                "-shortest", "-movflags", "+faststart", str(output_path),
            ],
            check=True,
        )

    counted = [e for e in result.events if e.counted]
    return {
        "frames": len(result.events),
        "selected_side": result.side,
        "rep_count": result.rep_count,
        "count_times_seconds": [round(e.frame / C.FPS, 2) for e in counted],
        "five_rep_time_seconds": round(result.five_rep_time_sec, 2) if result.five_rep_time_sec else None,
        "required_landmarks_visible_percent": round(100 * result.visible_required_frac, 1),
        "paused_percent": round(100 * result.paused_frac, 1),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Livia sit-to-stand prototype overlay")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=Path("ml/models/pose_landmarker_lite.task"))
    args = parser.parse_args()
    print(json.dumps(render(args.input, args.output, args.model), indent=2))


if __name__ == "__main__":
    main()
