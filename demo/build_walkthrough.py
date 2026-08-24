"""Build the caption-led Livia hackathon walkthrough from two model demos."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT, FPS = 1920, 1080, 30
FONT = Path("/usr/share/fonts/bitstream-vera-sans-fonts/Vera.ttf")
FONT_BOLD = Path("/usr/share/fonts/bitstream-vera-sans-fonts/VeraBd.ttf")
GREEN = "#5CE68C"
INK = "#F7FAFC"
MUTED = "#AAB9C8"
PANEL = "#141F2D"
BG = "#0B111B"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT), size)


def base_card(label: str, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        ratio = y / HEIGHT
        draw.line((0, y, WIDTH, y), fill=(11 + int(8 * ratio), 17 + int(12 * ratio), 27 + int(15 * ratio)))
    label_font = font(25, True)
    label_text = label.upper()
    label_box = draw.textbbox((0, 0), label_text, font=label_font)
    label_width = label_box[2] - label_box[0]
    draw.rounded_rectangle((88, 78, 136 + label_width, 128), radius=25, fill=GREEN)
    draw.text((111, 88), label_text, font=label_font, fill="#07120C")
    draw.text((88, 172), title, font=font(72, True), fill=INK)
    draw.rectangle((88, 278, 270, 286), fill=GREEN)
    return image, draw


def save_text_card(path: Path, label: str, title: str, lines: list[str], footer: str = "") -> None:
    image, draw = base_card(label, title)
    y = 356
    for line in lines:
        draw.ellipse((96, y + 12, 114, y + 30), fill=GREEN)
        draw.text((142, y), line, font=font(39), fill=INK)
        y += 92
    if footer:
        draw.text((88, 972), footer, font=font(27), fill=MUTED)
    image.save(path)


def save_session_card(path: Path) -> None:
    image, draw = base_card("Product mockup", "Today's prescribed session")
    rows = [
        ("01", "Shoulder abduction", "Front view", "9 reps"),
        ("02", "Sit-to-stand", "Side view", "5 reps"),
    ]
    y = 350
    for number, exercise, view, target in rows:
        draw.rounded_rectangle((88, y, 1832, y + 180), radius=30, fill=PANEL, outline="#26394C", width=3)
        draw.text((128, y + 42), number, font=font(48, True), fill=GREEN)
        draw.text((250, y + 34), exercise, font=font(45, True), fill=INK)
        draw.text((250, y + 96), view, font=font(30), fill=MUTED)
        draw.text((1570, y + 57), target, font=font(37, True), fill=INK)
        y += 212
    draw.rounded_rectangle((1410, 875, 1832, 975), radius=28, fill=GREEN)
    draw.text((1501, 902), "START SESSION", font=font(29, True), fill="#07120C")
    draw.text((88, 920), "The physiotherapist sets the plan. Livia follows it.", font=font(31), fill=MUTED)
    image.save(path)


def save_summary_card(path: Path) -> None:
    image, draw = base_card("Live session data", "Session complete")
    metrics = [
        ("Shoulder abduction", "9 / 9", "completed"),
        ("Sit-to-stand", "6 / 5", "completed"),
        ("First five stands", "22.2 s", "prototype trend"),
        ("Tracking pauses", "0", "both exercises"),
    ]
    x_positions = (88, 970)
    for index, (name, value, note) in enumerate(metrics):
        x = x_positions[index % 2]
        y = 350 + (index // 2) * 250
        draw.rounded_rectangle((x, y, x + 800, y + 200), radius=30, fill=PANEL, outline="#26394C", width=3)
        draw.text((x + 38, y + 30), name, font=font(31), fill=MUTED)
        draw.text((x + 38, y + 82), value, font=font(58, True), fill=GREEN)
        draw.text((x + 285, y + 109), note, font=font(27), fill=INK)
    draw.text((88, 910), "Next: the patient can add pain, exertion and a short note.", font=font(33), fill=INK)
    draw.text((88, 968), "Derived measurements are retained; raw video is not part of the session summary.", font=font(27), fill=MUTED)
    image.save(path)


def save_physio_card(path: Path) -> None:
    image, draw = base_card("Product mockup • demo data", "Physiotherapist visit report")

    draw.rounded_rectangle((88, 345, 1832, 510), radius=28, fill=PANEL, outline="#26394C", width=3)
    draw.text((128, 375), "ADHERENCE", font=font(24, True), fill=GREEN)
    draw.text((128, 423), "Shoulder", font=font(30), fill=MUTED)
    draw.text((325, 413), "9 / 9", font=font(46, True), fill=INK)
    draw.text((740, 423), "Sit-to-stand", font=font(30), fill=MUTED)
    draw.text((1028, 413), "6 / 5", font=font(46, True), fill=INK)
    draw.text((1425, 423), "2 / 2 completed", font=font(31, True), fill=GREEN)

    draw.rounded_rectangle((88, 540, 1832, 755), radius=28, fill=PANEL, outline="#26394C", width=3)
    draw.text((128, 570), "MOVEMENT DETAILS TO REVIEW", font=font(24, True), fill=GREEN)
    movement = [
        ("Shoulder peak", "113.6°"),
        ("Slow movement", "5 reps"),
        ("Incomplete range", "3 reps"),
        ("First five stands", "22.2 s"),
    ]
    x = 128
    for heading, value in movement:
        draw.text((x, 626), heading, font=font(25), fill=MUTED)
        draw.text((x, 672), value, font=font(39, True), fill=INK)
        x += 420

    draw.rounded_rectangle((88, 785, 1832, 930), radius=28, fill="#10271B", outline="#285A3D", width=3)
    draw.text((128, 815), "PATIENT CONTEXT", font=font(24, True), fill=GREEN)
    draw.text((128, 862), "Pain and exertion were not entered in this demo.", font=font(31), fill=INK)
    draw.text((1080, 862), "The physio decides whether the plan changes.", font=font(29, True), fill=GREEN)
    draw.text((88, 977), "A concise pre-visit view — not a diagnosis or autonomous recommendation.", font=font(27), fill=MUTED)
    image.save(path)


def save_focus_card(source: Path, path: Path, box: tuple[int, int, int, int], prompt: str) -> None:
    """Dim the rest of a card and add a cursor plus an ordered reading prompt."""
    image = Image.open(source).convert("RGBA")
    dim = Image.new("RGBA", image.size, (0, 0, 0, 105))
    dim_draw = ImageDraw.Draw(dim)
    dim_draw.rounded_rectangle(box, radius=34, fill=(0, 0, 0, 0))
    image = Image.alpha_composite(image, dim)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(box, radius=34, outline=GREEN, width=9)

    prompt_font = font(24, True)
    prompt_box = draw.textbbox((0, 0), prompt, font=prompt_font)
    prompt_width = prompt_box[2] - prompt_box[0]
    prompt_y = max(298, box[1] - 45)
    draw.rounded_rectangle((box[0] + 28, prompt_y, box[0] + prompt_width + 76, prompt_y + 42), radius=21, fill=GREEN)
    draw.text((box[0] + 52, prompt_y + 7), prompt, font=prompt_font, fill="#07120C")

    cursor_x = min(box[2] - 48, WIDTH - 70)
    cursor_y = box[1] + 54
    draw.ellipse((cursor_x - 35, cursor_y - 35, cursor_x + 35, cursor_y + 35), outline=GREEN, width=7)
    cursor = [
        (cursor_x - 15, cursor_y - 24),
        (cursor_x + 27, cursor_y + 17),
        (cursor_x + 8, cursor_y + 19),
        (cursor_x + 20, cursor_y + 43),
        (cursor_x + 6, cursor_y + 50),
        (cursor_x - 7, cursor_y + 24),
        (cursor_x - 20, cursor_y + 37),
    ]
    draw.polygon(cursor, fill="white", outline="#07120C")
    image.convert("RGB").save(path)


def save_close_card(path: Path) -> None:
    image, draw = base_card("Early validation", "Built around real clinical workflow")
    draw.text((88, 370), "5", font=font(170, True), fill=GREEN)
    draw.text((305, 415), "physiotherapist conversations", font=font(50, True), fill=INK)
    draw.text((305, 486), "They said this visibility could help clinicians and patients.", font=font(34), fill=MUTED)
    draw.rounded_rectangle((88, 650, 1832, 840), radius=34, fill=PANEL, outline="#26394C", width=3)
    draw.text((138, 692), "NEXT STEP", font=font(26, True), fill=GREEN)
    draw.text((138, 742), "Test Livia with a clinical design partner.", font=font(48, True), fill=INK)
    draw.text((88, 965), "LIVIA  •  The plan stays with the physio. The support continues at home.", font=font(29), fill=MUTED)
    image.save(path)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def encode_card(
    image: Path,
    output: Path,
    duration: float,
    *,
    fade_in: bool = True,
    fade_out: bool = True,
) -> None:
    filters = []
    if fade_in:
        filters.append("fade=t=in:st=0:d=0.45")
    if fade_out:
        filters.append(f"fade=t=out:st={max(duration - 0.45, 0)}:d=0.45")
    filters.append("format=yuv420p")
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-i", str(image),
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-t", str(duration),
        "-vf", ",".join(filters),
        "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-shortest", str(output),
    ])


def encode_demo_clip(source: Path, output: Path, start: float, duration: float, exercise: str, detail: str) -> None:
    fade_out = max(duration - 0.45, 0)
    filter_graph = (
        "[0:v]split[bg][fg];"
        "[bg]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,"
        "boxblur=25:2,eq=brightness=-0.55[bg];"
        "[fg]scale=-2:1000[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
        f"drawtext=fontfile={FONT_BOLD}:text='LIVE PROTOTYPE':x=80:y=170:fontsize=28:fontcolor=0x5CE68C,"
        f"drawtext=fontfile={FONT_BOLD}:text='{exercise}':x=80:y=225:fontsize=46:fontcolor=white,"
        f"drawtext=fontfile={FONT}:text='{detail}':x=80:y=292:fontsize=28:fontcolor=0xAAB9C8,"
        "drawtext=fontfile=" + str(FONT) + ":text='Pose landmarks':x=1390:y=225:fontsize=29:fontcolor=white,"
        "drawtext=fontfile=" + str(FONT) + ":text='Joint features':x=1390:y=272:fontsize=29:fontcolor=white,"
        "drawtext=fontfile=" + str(FONT) + ":text='Rule-based policy':x=1390:y=319:fontsize=29:fontcolor=white,"
        "drawtext=fontfile=" + str(FONT) + ":text='No diagnosis':x=1390:y=390:fontsize=25:fontcolor=0xAAB9C8,"
        f"fade=t=in:st=0:d=0.45,fade=t=out:st={fade_out}:d=0.45,format=yuv420p[v]"
    )
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-t", str(duration), "-i", str(source),
        "-filter_complex", filter_graph, "-map", "[v]", "-map", "0:a:0?",
        "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", str(output),
    ])


def build(shoulder: Path, sts: Path, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="livia-walkthrough-") as work_raw:
        work = Path(work_raw)
        cards = {
            "title": work / "title.png",
            "problem": work / "problem.png",
            "session": work / "session.png",
            "summary": work / "summary.png",
            "physio": work / "physio.png",
            "close": work / "close.png",
        }
        save_text_card(
            cards["title"], "Hackathon demo", "Livia",
            ["Home exercise, with visibility between visits.", "Two working movement prototypes. One clinician-controlled workflow."],
            "Caption-led first cut  •  Working model footage + clearly labelled product mockups",
        )
        save_text_card(
            cards["problem"], "The gap", "What happens between visits?",
            ["Patients remember the exercise.", "Physiotherapists rarely see how it was performed.", "Livia turns each home session into useful visit context."],
            "Guidance and monitoring only — no diagnosis or autonomous progression",
        )
        save_session_card(cards["session"])
        save_summary_card(cards["summary"])
        save_physio_card(cards["physio"])
        save_close_card(cards["close"])

        focus_specs = {
            "session": [
                ((72, 335, 1848, 545), "1  READ THE FIRST EXERCISE"),
                ((72, 547, 1848, 757), "2  READ THE SECOND EXERCISE"),
                ((1385, 850, 1855, 1000), "3  START THE PRESCRIBED SESSION"),
            ],
            "summary": [
                ((72, 335, 1848, 565), "1  COMPLETED REPS"),
                ((72, 585, 1848, 815), "2  TIMING AND TRACKING"),
                ((72, 875, 1848, 1018), "3  PATIENT FEEDBACK COMES NEXT"),
            ],
            "physio": [
                ((72, 330, 1848, 525), "1  CHECK ADHERENCE"),
                ((72, 525, 1848, 770), "2  REVIEW MOVEMENT DETAILS"),
                ((72, 770, 1848, 945), "3  ADD PATIENT CONTEXT"),
            ],
            "close": [
                ((72, 330, 1848, 585), "1  EARLY CLINICAL SIGNAL"),
                ((72, 630, 1848, 860), "2  THE NEXT STEP"),
            ],
        }
        focused: dict[str, list[Path]] = {}
        for key, specs in focus_specs.items():
            focused[key] = []
            for index, (box, prompt) in enumerate(specs):
                focus_path = work / f"{key}-focus-{index}.png"
                save_focus_card(cards[key], focus_path, box, prompt)
                focused[key].append(focus_path)

        segment_index = 0
        segments: list[Path] = []

        def next_segment() -> Path:
            nonlocal segment_index
            path = work / f"segment-{segment_index:02d}.mp4"
            segments.append(path)
            segment_index += 1
            return path

        encode_card(cards["title"], next_segment(), 5)
        encode_card(cards["problem"], next_segment(), 7)
        for index, image in enumerate(focused["session"]):
            encode_card(image, next_segment(), (3, 3, 2)[index], fade_in=index == 0, fade_out=index == 2)
        encode_demo_clip(shoulder, next_segment(), 3.6, 25.5, "SHOULDER ABDUCTION", "Front view  •  reps + bounded cues")
        encode_demo_clip(sts, next_segment(), 3.3, 29.6, "SIT-TO-STAND", "Side view  •  reps + first-five timing")
        for index, image in enumerate(focused["summary"]):
            encode_card(image, next_segment(), 5, fade_in=index == 0, fade_out=index == 2)
        for index, image in enumerate(focused["physio"]):
            encode_card(image, next_segment(), 5, fade_in=index == 0, fade_out=index == 2)
        for index, image in enumerate(focused["close"]):
            encode_card(image, next_segment(), 4, fade_in=index == 0, fade_out=index == 1)

        concat_file = work / "segments.txt"
        concat_file.write_text("".join(f"file '{segment}'\n" for segment in segments))
        output.parent.mkdir(parents=True, exist_ok=True)
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", "-movflags", "+faststart", str(output),
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Livia hackathon walkthrough")
    parser.add_argument("--shoulder", type=Path, required=True)
    parser.add_argument("--sts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.shoulder, args.sts, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
