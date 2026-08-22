# Heuristic posture deviations (not a trained model)

A heuristic policy is a **named feature + a phase + a dwell + a cutoff**. It is not a neural net.

**Scope:** Ex1 right-arm abduction, **front Camera17** only. Overlay: [`view.py`](view.py). Canonical product still says therapist-configured monitoring, not diagnosis.

CSV `correctness` is **binary and untyped**. Heuristics will never equal that column. Evaluation is (a) rep counts vs `Segmentation.csv`, (b) cue rates on GT-correct vs GT-incorrect as a **weak** check, (c) hand-watch clips in the viewer.

## Maths (image vs world)

MediaPipe **image** landmarks: `x,y` in `[0,1]`, origin **top-left**, **y down**. Image `z` is a rough depth scaled like `x`, **not metres**. **World** landmarks: metres, origin roughly hip midpoint — used only for optional `PLANE_FWD`.

Interior angle at O between P and Q:

`acos(clamp(dot(P-O, Q-O) / (|P-O||Q-O|), -1, 1))` in [`features.py`](features.py).

| Scalar | Definition | Notes |
| --- | --- | --- |
| Elbow | Angle at elbow (shoulder, wrist) | Extension ≈ 180°. Flexion decreases it. **2D `(x,y,0)`**. |
| Abduction (front) | Angle at shoulder (hip, elbow) | Hang ≈ 0°, 90° abd ≈ 90°, overhead ≈ 180°. **2D only** (image-z must not mix in). |
| Trunk lean | Mid-shoulder vs mid-hip vs image up `(0,-1)` | Sign = mid-shoulder x − mid-hip x. Not one shrugged shoulder. |
| Shrug | `(shoulder_y - ear_y) / torso_length` | y down → hike **shrinks** the gap. Rest baseline at BOTTOM. |
| Plane (optional) | World humerus vs frontal-ish | Disabled if `world_xyz` missing or NaN. Do not use world z as height. |

All front coronal scalars go through `as_2d()`. Do not treat image z as metres.

## Recipe

1. Name the error in language a physio would use; only errors **visible from the front**.
2. Compute that scalar (table above).
3. Gate on phase (raised = `abd ≥ 50°`, or on-count for ROM / speed / hold).
4. Dwell ~8 frames (`DwellFlag`). Latch **independently** per cycle so two mistakes stay stacked.
5. Session layer **above** BOTTOM/ASCENT/TOP/DESCENT ([`session_fsm.py`](session_fsm.py)): pause counting for setup / idle / unknown activity.
6. Speak only the highest-priority id (`SPEAK_ORDER`) after **a second episode of the same error or a continuous hold ≥ 0.5s** (`SpeakGate`); HUD still lists the stack immediately. Cooldown ~1.5s.
7. Cutoffs from **correct-rep tails** on cached lite Ex1 front (`python -m ml.fit_ex1_cutoffs`). If tails do not separate, pick a visible-error threshold and say so.

## Session states

| ID | Scalar | HUD |
| --- | --- | --- |
| `NO_PERSON` | required vis `< 0.6` for `> 0.5s` | Step into frame |
| `WRONG_VIEW` | shoulder width / torso `< 0.32` or y-split `> 0.085` | Face the camera |
| `WRONG_SIDE` | `var(left abd) ≫ var(right abd)` over 2s | Use the right arm |
| `IDLE` | low wrist/elbow speed, BOTTOM, 8s | Continue your set |
| `UNKNOWN_ACTIVITY` | high motion, no abd cycle ~6s, ankle energy | Not the prescribed exercise |

Lost pose mid-rep: pause; do not count a ghost rep. Idle / unknown **cannot** be scored from CSV — demonstrate in the viewer.

## Typed Ex1 deviations

Priority: setup → unknown → incomplete ROM → bent elbow → shrug → trunk lean → too fast → plane.

| ID | Scalar | When | HUD |
| --- | --- | --- | --- |
| `INCOMPLETE_ROM` | peak 2D abd this cycle | on count | Lift higher |
| `BENT_ELBOW` | 2D elbow | raised, 8 frames | Keep the arm straight |
| `SHRUG` | norm. vertical gap vs rest | raised | Don't shrug |
| `TRUNK_LEAN` | mid-torso vs vertical | raised | Stand tall, don't lean |
| `TOO_FAST` | time in ASCENT or DESCENT `< 0.35s` | on count | Slow the movement |
| `NO_HOLD` | frames in TOP `< 6` | on count | Pause at the top |
| `PLANE_FWD` | world humerus out of plane | raised, if world stable | Keep the arm in line with your body |
| `MULTI` | two+ form flags | HUD stack; TTS still one | (see stack) |

**Cannot detect** from this camera / skeleton (do not fake): internal rotation, pain, breath-hold, scapular winging detail, load, goniometer-accurate degrees.

## Measured cutoffs (Ex1 front lite cache, 47 correct / 41 incorrect GT reps)

| Cue | Correct tail | Incorrect tail | Constant |
| --- | --- | --- | --- |
| Raised min elbow | p10 **138.4°** | p10 **124.9°** | `ELBOW_BENT_DEG = 132` |
| Peak 2D abduction | p10 **118.8°** | p10 **92.9°** | `ROM_ABS_FLOOR_DEG = 106` (+ 75% of running peak) |
| Shrug gap drop / rest | p90 **0.39** | p90 **0.41** | `SHRUG_DROP_FRAC = 0.50` (weak split; high bar so it is rare on correct) |
| Abs mid-torso lean | p90 **1.6°** | p90 **1.8°** | `TRUNK_LEAN_DEG = 8` (no split on this set; 8° = obvious lean) |

2D unification did **not** move elbow/ROM tails vs the earlier mixed-z abduction pass. ROM floor moved 100° → 106° to sit between p10s.

Success on form flags = **rarer on GT-correct than GT-incorrect**, not high F1 vs untyped labels.

Lite Ex1 front after this catalog (9 videos): MAE reps **1.00**, pred/GT **1.07**. Cue rates correct vs incorrect: bent elbow **0% / 15%**, incomplete ROM **4% / 10%**, too fast **13% / 32%**, shrug **0% / 0%** at `SHRUG_DROP_FRAC=0.50` (silent on this set), trunk lean **0% / 0%** at 8°, no-hold **0% / 0%** at hysteresis floor.

## What not to do

- Do not train a classifier on REHAB24-6 for “something else” (or any shipped head). Unknown activity is energy + no abduction cycle. See [`NO_SHIP_CLASSIFIER.md`](NO_SHIP_CLASSIFIER.md).
- Do not flag every frame the inequality is true.
- Do not mix views (front abduction vs side STS). Sit-to-stand is not in this dataset.
