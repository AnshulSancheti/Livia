# ML-model — final plan (laptop test bench)

Status: **locked**. Does not change [`livia-project-plan.md`](livia-project-plan.md). Flutter / Android / TTS / export are out of scope until this workstream has numbers.

---

## Locked decisions

- **Laptop only.** No Flutter, no Android, no live camera.
- **Frozen pretrained pose.** Google `pose_landmarker_lite.task` (~5.5 MB). Same file the phone will ship. **Do not fine-tune. Do not train a new pose net.**
- **REHAB24-6 is a test bench**, not a training set. Nothing in `data/` updates MediaPipe weights.
- **Policy is code, not a net:** One-Euro → joint angles → hysteretic rep / deviation rules. Python now; Dart later must match the same formulae and constants.
- **Exercises for this workstream:** Ex1 arm abduction (primary), Ex6 squats (secondary). Sit-to-stand is **not** claimed as validated here.
- **License:** REHAB24-6 is CC BY-NC 4.0. Using it to *evaluate* a Google model is fine. Shipping a classifier *trained on this set* is not.

---

## Pipeline (laptop = phone maths, dataset = camera)

```text
Phone later:  Camera frames  →  MediaPipe LIVE_STREAM  →  policy  →  TTS
Laptop now:   REHAB24-6 mp4  →  MediaPipe VIDEO        →  same policy  →  metrics
```

Same: `.task` weights, 33 landmarks + world coords, filter, angles, rules.  
Different: OpenCV + `RunningMode.VIDEO` and timestamps `frame_index * 1000 / 30` instead of CameraX + `LIVE_STREAM`.

### Roles of each artifact

- **Pretrained MediaPipe** — only neural net; frozen.
- **RGB videos** (`data/videos.zip`) — stand-in patient. Run MediaPipe on Camera17 when `cam17_orientation=front`.
- **`data/Segmentation.csv`** — answer key: rep `[first_frame, last_frame]`, exercise_id, person_id, binary `correctness`. Score **rep counts**. Weakly check whether deviation rules fire more on `correctness=0`.
- **`data/2d_joints.zip` (30 fps)** — mocap 2D GT. Score pose after mapping overlapping joints (shoulder/elbow/wrist/hip/knee/ankle). Not training labels.
- Filter rows: `mocap_erroneous=0`; prefer extra-person flags 0–1.

---

## Exercises (dataset-backed)

| ID | Movement | View for tests | Why |
| --- | --- | --- | --- |
| Ex1 | Arm abduction (right arm) | Camera17, front | Matches product front-view abduction policy |
| Ex6 | Squats | Camera17, front | Closest lower-limb pattern in this set |
| — | Sit-to-stand | *not in REHAB24-6* | Product later; do not report STS accuracy from this data |

Out of scope for v1 policy validation: Ex2 Arm VW, Ex3 push-ups, Ex4 leg abduction, Ex5 lunge.

Ex1 Camera18 (true side when c17 is front) is an optional **negative** check: abduction policy should not trust a side view.

---

## What we will implement (when execution starts)

Python under something like `ml/` (not the Flutter app):

1. Download `pose_landmarker_lite.task` (and keep `full` on disk for a bake-off).
2. Stream or extract Ex1/Ex6 front Camera17 videos; run MediaPipe VIDEO; **cache landmarks** (npy/parquet) so videos are not re-inferred every run.
3. Pose report vs mocap: detection rate, visibility of required landmarks, 2D error after a similarity transform on mapped joints. **Switch to `full` only if lite misses gates.**
4. Policy v0 for abduction: BOTTOM/ASCENT/TOP/DESCENT with hysteresis; bent elbow; shrug proxy (ear–shoulder). Untrimmed **rep count vs CSV**. Hand-watch ~20 correct and ~20 incorrect clips for cue sanity (`correctness` is untyped).
5. Policy v0 for squats: knee/hip/trunk features; same counting protocol. Document that this is squat, not STS.
6. Write a short results appendix (numbers + lite vs full choice). Constants that survive become the Dart policy later.

**Not in this workstream:** Flutter, TTS, GRU/LLM, training on REHAB24-6, unzipping all 184k frames unless needed.

---

## Gates (must pass before Android)

- Required landmarks visible on ≥90% of in-rep frames (front Ex1), after a 2-video pilot to confirm the metric is sane.
- Untrimmed front Ex1: mean abs error ≤ 1 rep per video **or** counted/GT ≥ 85% — freeze one metric after the first full Ex1 pass.
- Lite vs full: keep lite unless those gates fail.
- Cues: bent-elbow and shrug fire on watched incorrect clips and stay mostly quiet on watched correct clips.

---

## Explicitly rejected

- Fine-tune BlazePose / Pose Landmarker on this dataset (unsupported + NC license + mocap-suit domain).
- Train pose from 3D markers.
- On-device or laptop LLM in the live loop.
- Shipping any head trained on REHAB24-6.
