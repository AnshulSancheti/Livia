# ml/ — laptop test bench (no Flutter)

Frozen **MediaPipe Pose Landmarker** + deterministic policies, scored on REHAB24-6.

```bash
cd /Users/ariyan/Projects/Livia
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt
python -m ml.run_bench          # full Ex1 lite + 2-video full bake-off + Ex6
python -m ml.run_bench --pilot  # two Ex1 videos, lite only
```

## What this is / is not

- **Is:** evaluate Google’s pretrained `.task` on our RGB videos; tune **rule thresholds** if needed.
- **Is not:** fine-tuning MediaPipe, training a pose net, or training a shipped GRU/DTW/quality head.

REHAB24-6 is **CC BY-NC 4.0**. Using it as an answer key is allowed here. **Do not ship weights trained on this set.**

Sit-to-stand is **not** in this dataset. Ex6 is **squats**. See `policy_squat.py` (`SIT_TO_STAND_VALIDATED = False`).

Outputs: `ml/cache/*.npz` (landmarks), `ml/models/*.task`, `ml/results/RESULTS.md`, `ml/results/bench.json`.

## Watch it move

Same lite model. A window opens on your machine (run this in **Terminal.app**, not headless).

```bash
cd /Users/ariyan/Projects/Livia
source .venv/bin/activate

# Dataset video with skeleton + abduction HUD (cached landmarks — smooth playback)
python -m ml.view --video PM_023

# Another clip / squats (skeleton only on Ex6)
python -m ml.view --video PM_022 --ex Ex6

# Your laptop camera, real time
python -m ml.view --camera

# Render the prototype side-view sit-to-stand overlay on a phone video
python -m ml.render_sts input.mp4 output.mp4
```

`q` quits. Space pauses dataset playback. Camera uses the same `.task` file with increasing timestamps (VIDEO mode), which is the laptop stand-in for phone `LIVE_STREAM`.

The sit-to-stand renderer is a prototype policy and is not clinically or
dataset validated. It is separate from the REHAB24-6 front-view squat policy.
