# ML test-bench results

REHAB24-6 is CC BY-NC 4.0. This run **evaluates** Google Pose Landmarker; it does **not** train or fine-tune it.
Do **not** ship a GRU/DTW/classifier trained on this dataset.

**Chosen pose variant:** `lite` (lite vs full: keep lite; full 2-video Ex1 MAE=1.0, similar ~64px mocap RMSE).

## Ex1 arm abduction (Camera17 front)

- MAE (reps/video): 1.000
- Mean pred/GT: 1.067
- Mean required-landmark visibility in-rep: 1.000
- Visibility gate (≥0.90): True
- Count gate (MAE≤1 or ratio≥0.85): True
- Frozen count metric: `mae_reps`
- Cue rates on GT-correct reps: {'bent_elbow': 0.0, 'shrug': 0.0, 'incomplete_rom': 0.0425531914893617, 'trunk_lean': 0.0, 'too_fast': 0.1276595744680851, 'no_hold': 0.0, 'n': 47}
- Cue rates on GT-incorrect reps: {'bent_elbow': 0.14634146341463414, 'shrug': 0.0, 'incomplete_rom': 0.0975609756097561, 'trunk_lean': 0.0, 'too_fast': 0.3170731707317073, 'no_hold': 0.0, 'n': 41}

Watch list (up to 20 correct / 20 incorrect segmented reps) is in `bench.json` under `watch_list`.
Binary `correctness` is untyped; cue rates are a weak check, not a clinical gold standard.

## Ex6 squats (Camera17 front) — not sit-to-stand

Sit-to-stand is not in REHAB24-6. Ex6 is squats filmed front-on. Do not claim STS accuracy from this test bench.

- MAE (reps/video): 6.556
- Mean required-landmark visibility in-rep: 1.000
- Trunk-flex cue rate correct/incorrect: {'trunk_flex': 0.0, 'n': 72} / {'trunk_flex': 0.0, 'n': 26}

## Per-video Ex1

| video | gt | pred | abs_err | vis_in_rep | rmse_px |
| --- | --- | --- | --- | --- | --- |
| PM_000 | 5 | 6 | 1 | 1.000 | 64.2 |
| PM_001 | 10 | 11 | 1 | 1.000 | 63.5 |
| PM_012 | 10 | 11 | 1 | 1.000 | 61.2 |
| PM_023 | 10 | 10 | 0 | 1.000 | 65.9 |
| PM_032 | 11 | 12 | 1 | 1.000 | 48.5 |
| PM_039 | 11 | 12 | 1 | 1.000 | 55.2 |
| PM_109 | 10 | 11 | 1 | 1.000 | 47.7 |
| PM_114 | 11 | 9 | 2 | 1.000 | 59.8 |
| PM_122 | 10 | 11 | 1 | 1.000 | 49.2 |

## Per-video Ex6

| video | gt | pred | abs_err | vis_in_rep | rmse_px |
| --- | --- | --- | --- | --- | --- |
| PM_008 | 17 | 3 | 14 | 1.000 | 56.4 |
| PM_022 | 10 | 4 | 6 | 1.000 | 55.2 |
| PM_029 | 10 | 4 | 6 | 1.000 | 54.3 |
| PM_038 | 10 | 5 | 5 | 1.000 | 53.5 |
| PM_043 | 10 | 5 | 5 | 1.000 | 51.9 |
| PM_105 | 10 | 10 | 0 | 1.000 | 57.7 |
| PM_113 | 10 | 12 | 2 | 1.000 | 49.8 |
| PM_118 | 11 | 0 | 11 | 1.000 | 56.4 |
| PM_126 | 10 | 0 | 10 | 1.000 | 47.6 |
