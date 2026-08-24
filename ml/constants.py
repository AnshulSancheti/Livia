"""Shared constants. Keep identical when porting the policy to Dart."""

FPS = 30.0

# MediaPipe Pose Landmarker indices (BlazePose 33).
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# REHAB24-6 OptiTrack 26-joint indices (see data/joints_names.txt).
# Mixamo-style: Arm=elbow, ForeArm=wrist, UpLeg=hip, Leg=knee, Foot=ankle.
MOCAP = {
    "hips": 0,
    "left_shoulder": 6,
    "left_elbow": 7,
    "left_wrist": 8,
    "right_shoulder": 11,
    "right_elbow": 12,
    "right_wrist": 13,
    "left_hip": 16,
    "left_knee": 17,
    "left_ankle": 18,
    "right_hip": 21,
    "right_knee": 22,
    "right_ankle": 23,
}

# Overlapping joints for 2D pose error (MP index -> mocap index).
MP_TO_MOCAP = {
    LEFT_SHOULDER: MOCAP["left_shoulder"],
    LEFT_ELBOW: MOCAP["left_elbow"],
    LEFT_WRIST: MOCAP["left_wrist"],
    RIGHT_SHOULDER: MOCAP["right_shoulder"],
    RIGHT_ELBOW: MOCAP["right_elbow"],
    RIGHT_WRIST: MOCAP["right_wrist"],
    LEFT_HIP: MOCAP["left_hip"],
    LEFT_KNEE: MOCAP["left_knee"],
    LEFT_ANKLE: MOCAP["left_ankle"],
    RIGHT_HIP: MOCAP["right_hip"],
    RIGHT_KNEE: MOCAP["right_knee"],
    RIGHT_ANKLE: MOCAP["right_ankle"],
}

ABDUCTION_REQUIRED = (
    RIGHT_SHOULDER,
    RIGHT_ELBOW,
    RIGHT_WRIST,
    RIGHT_HIP,
    RIGHT_EAR,
)
SQUAT_REQUIRED = (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_SHOULDER)

# One-Euro (mincutoff/beta typical for noisy landmarks).
ONE_EURO_MINCUTOFF = 1.0
ONE_EURO_BETA = 0.007
ONE_EURO_DCUTOFF = 1.0

# Abduction policy (right arm, front view). Coronal scalars are 2D (x,y).
# Allow a small tracking margin at the bottom. A visually lowered arm can sit
# around 25-28 degrees in portrait phone footage; requiring <25 for the full
# dwell window can merge two otherwise complete repetitions.
ABDUCTION_BOTTOM_ENTER = 30.0
ABDUCTION_BOTTOM_LEAVE = 35.0
ABDUCTION_TOP_ENTER = 80.0
ABDUCTION_TOP_LEAVE = 70.0
ABDUCTION_MIN_DWELL_FRAMES = 3
ABDUCTION_HOLD_FRAMES = 4
ELBOW_STRAIGHT_DEG = 160.0  # textbook; too strict for MediaPipe 2D — not used for cues
# Raised-phase 2D elbow min p10: correct 138.4°, incorrect 124.9° (cache, 2026-08-22).
ELBOW_BENT_DEG = 132.0
ELBOW_DWELL_FRAMES = 8
# Peak 2D abd p10: correct 118.8°, incorrect 92.9°.
ROM_ABS_FLOOR_DEG = 106.0
ROM_FRAC_OF_CAL = 0.75
# Vertical gap drop / rest: correct p90≈0.39, incorrect p90≈0.41 — weak. Stricter than p90.
SHRUG_DROP_FRAC = 0.50
SHRUG_DWELL_FRAMES = 8
# Mid-torso vs vertical: correct p90≈1.6°, incorrect p90≈1.8° — no split. 8° = visible lean.
TRUNK_LEAN_DEG = 8.0
TRUNK_DWELL_FRAMES = 8
PLANE_FWD_DEG = 28.0
PLANE_DWELL_FRAMES = 8
TOO_FAST_SEC = 0.35
NO_HOLD_MIN_FRAMES = 3  # hysteresis floor; raise when a hold is prescribed
CUE_COOLDOWN_FRAMES = 45
TTS_HOLD_SEC = 0.5
TTS_HOLD_FRAMES = 15  # 0.5s at 30 fps
RAISED_ABD_DEG = 50.0

# Session FSM (landmarks only).
NO_PERSON_FRAMES = 15  # 0.5 s
WRONG_VIEW_WIDTH_RATIO = 0.32
WRONG_VIEW_Y_SPLIT = 0.085
WRONG_SIDE_VAR_RATIO = 2.2
WRONG_SIDE_VAR_MIN = 80.0
IDLE_SPEED = 0.006
IDLE_FRAMES = 240  # 8 s
UNKNOWN_NO_CYCLE_FRAMES = 180  # 6 s
UNKNOWN_SPEED = 0.018
UNKNOWN_ANKLE = 0.012
HIST_REPS = 5

# Squat policy (front view). Standing ~170°, bottom ~90°.
SQUAT_STAND_ENTER = 160.0
SQUAT_STAND_LEAVE = 150.0
SQUAT_BOTTOM_ENTER = 110.0
SQUAT_BOTTOM_LEAVE = 120.0
SQUAT_MIN_DWELL_FRAMES = 3
TRUNK_FLEX_DEG = 45.0
# Image-normalized hip drop (y down). Front-view squat proxy, not STS.
HIP_DROP_LEAVE_STAND = 0.04
HIP_DROP_ENTER_BOTTOM = 0.10
HIP_DROP_LEAVE_BOTTOM = 0.07
HIP_DROP_ENTER_STAND = 0.03

# Five-times sit-to-stand prototype (side view). This policy is intentionally
# separate from the REHAB24-6 front-view squat policy and is not dataset- or
# clinically validated. The visible side is selected during clip calibration.
STS_SEATED_ENTER = 110.0
STS_SEATED_LEAVE = 120.0
STS_STAND_ENTER = 155.0
STS_STAND_LEAVE = 145.0
STS_MIN_DWELL_FRAMES = 3
STS_TRUNK_FLEX_DEG = 55.0
STS_FORM_DWELL_FRAMES = 8
STS_FAST_DESCENT_SEC = 0.35

VISIBILITY_MIN = 0.6
LITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
FULL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
