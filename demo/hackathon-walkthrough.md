# Livia hackathon demo walkthrough

## Deliverable

A 90–120 second landscape video showing two patient exercises, the live pose
model, a post-session overview, and the information a physiotherapist would see
before a visit.

The working pose footage should be labelled **Live prototype**. Patient and
physiotherapist interface screens may be labelled **Product mockup** until those
screens exist in the application.

## Demo story and narration

### 0:00–0:08 — The problem

**Visual:** Livia title, followed by a patient preparing to exercise at home.

**Narration:**

> A physiotherapist can demonstrate an exercise in the clinic, but they cannot
> watch every repetition a patient does at home. Livia helps bridge that gap.

### 0:08–0:20 — What Livia is

**Visual:** A simple prescribed-session overview: two exercises, target reps,
and a start button.

**Narration:**

> Livia is a home-exercise companion configured by the physiotherapist. It does
> not diagnose or change the treatment plan. It guides the prescribed session,
> counts repetitions, and records a concise summary.

### 0:20–0:48 — Exercise one: shoulder abduction

**Visual:** Annotated shoulder-abduction footage. Start near the first ascent,
show the counter increasing, then include one useful form cue.

**Narration:**

> Here, the pose model tracks shoulder abduction on the device. The exercise
> policy follows the movement through ascent, top position, descent, and return.
> It counts completed repetitions and gives short, specific feedback when the
> movement is rushed or the range is incomplete.

### 0:48–1:13 — Exercise two: five sit-to-stands

**Visual:** Side-view sit-to-stand footage with the hip, knee, ankle, and
shoulder highlighted. Show the live rep count and elapsed time.

**Narration:**

> The same pipeline can support a different exercise policy. In a five-times
> sit-to-stand session, Livia tracks the seated and standing positions, counts
> each complete cycle, and records the total time as a trend for the clinician.

**Accuracy label:** Sit-to-stand is a prototype policy and has not yet been
clinically validated.

### 1:13–1:30 — Patient session overview

**Visual:** Session-complete card containing:

- Shoulder abduction: 9 of 9 repetitions
- Sit-to-stand: 5 of 5 repetitions and total time
- Form cues by exercise
- Visibility or tracking pauses
- Patient-entered pain and exertion scores

**Narration:**

> After the session, the patient can see what they completed and add pain or
> exertion feedback. The summary stores derived measurements, not the raw video.

### 1:30–1:48 — Physiotherapist visit view

**Visual:** A visit-detail mockup showing adherence, prescribed versus completed
repetitions, range-of-motion trend, form flags, pain, exertion, and the patient's
note.

**Narration:**

> Before the next visit, the physiotherapist sees the details that matter:
> adherence, completed repetitions, movement trends, recurring form flags, and
> the patient's own feedback. The physiotherapist remains in control of every
> prescription and progression decision.

### 1:48–2:00 — Validation and close

**Visual:** “5 physiotherapist conversations” followed by the Livia closing
card.

**Narration:**

> We have already spoken with five physiotherapists. They told us this kind of
> visibility could be useful for both clinicians and patients. Our next step is
> to test Livia with a clinical design partner.

## Sit-to-stand recording needed

Record one continuous video containing exactly five comfortable repetitions.

- Use a stable landscape recording if possible; 1080p at 30 or 60 fps is fine.
- Use a firm chair without wheels. Do not perform the exercise if it is unsafe
  or uncomfortable for you.
- Put the camera approximately 2.5–3 metres away at hip height.
- Film from your right side so the right shoulder, hip, knee, and ankle remain
  visible.
- Keep the entire body, feet, and chair in frame throughout.
- Sit still for two seconds, complete five normal sit-to-stands, then remain
  seated for two seconds at the end.
- Use good, even lighting and avoid loose clothing that hides the hip or knee.

## Final production checklist

- Working model footage and UI mockups are labelled accurately.
- No claim of diagnosis, autonomous prescription, or clinical validation.
- On-screen text remains readable on a phone and in a judging-room projection.
- Voiceover is conversational and does not merely read every label on screen.
- Captions are included.
- Export at 1920×1080, H.264, with a target duration under two minutes.
