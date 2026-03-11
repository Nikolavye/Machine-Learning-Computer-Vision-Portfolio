"""
Evaluate Solution C on first 100 frames (static per-frame classification, no tracking).

For each frame:
  1. YOLO detection (person class)
  2. Pitch filter + grass check
  3. Color-based team classification (red / white / referee)
  4. Save annotated frame to frames_result_c/

Output: per-frame counts + summary statistics.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from solution import (
    TEAM_COLORS,
    choose_model_path,
    classify_by_strict_colors,
    get_device,
    get_team_by_color,
    is_on_pitch,
)

CASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = CASE_DIR / "sample.mp4"
OUTPUT_DIR = CASE_DIR / "frames_result_c"
OUTPUT_CSV = CASE_DIR / "eval_c_100.csv"

NUM_FRAMES = 100


def main():
    device = get_device()
    model_path = choose_model_path()
    print(f"Loading model: {model_path} (device: {device})")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {frame_w}x{frame_h} @ {fps:.1f}fps, {total_frames} total frames")
    print(f"Evaluating first {NUM_FRAMES} frames...\n")

    OUTPUT_DIR.mkdir(exist_ok=True)
    for f in OUTPUT_DIR.iterdir():
        f.unlink()

    csv_rows = []

    for frame_idx in range(NUM_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, verbose=False, conf=0.10, iou=0.5, classes=[0], device=device)
        r = results[0]

        counts = {"red": 0, "white": 0, "goalkeeper_yellow": 0, "goalkeeper_blue": 0}
        annotated = frame.copy()

        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                if not is_on_pitch(box, frame):
                    continue

                upper_bgr, vote_override = classify_by_strict_colors(frame, box)
                if upper_bgr is None:
                    continue

                if vote_override is not None:
                    team = vote_override
                else:
                    team = get_team_by_color(upper_bgr[0], upper_bgr[1], upper_bgr[2])
                    if team is None:
                        continue
                counts[team] += 1

                x1, y1, x2, y2 = map(int, box)
                color = TEAM_COLORS.get(team, (128, 128, 128))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                txt = f"{team} {conf:.2f}"
                cv2.putText(annotated, txt, (x1, max(18, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        # Draw overlay counts
        y = 30
        for name in ["red", "white", "goalkeeper_yellow", "goalkeeper_blue"]:
            text = f"{name}: {counts[name]}"
            c = TEAM_COLORS[name]
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(annotated, (10, y - th - 6), (10 + tw + 10, y + 6), (0, 0, 0), -1)
            cv2.putText(annotated, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
            y += 34

        cv2.imwrite(str(OUTPUT_DIR / f"frame_{frame_idx + 1:03d}.jpg"), annotated)

        csv_rows.append({
            "frame_id": frame_idx,
            "red": counts["red"],
            "white": counts["white"],
            "goalkeeper_yellow": counts["goalkeeper_yellow"],
            "goalkeeper_blue": counts["goalkeeper_blue"],
            "total": sum(counts.values()),
        })

        if frame_idx % 10 == 0:
            print(f"  frame {frame_idx:3d}: red={counts['red']}, white={counts['white']}, "
                  f"gk_y={counts['goalkeeper_yellow']}, gk_b={counts['goalkeeper_blue']}")

    cap.release()

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "red", "white",
                                                "goalkeeper_yellow", "goalkeeper_blue", "total"])
        writer.writeheader()
        writer.writerows(csv_rows)

    reds = [r["red"] for r in csv_rows]
    whites = [r["white"] for r in csv_rows]
    gk_y = [r["goalkeeper_yellow"] for r in csv_rows]
    gk_b = [r["goalkeeper_blue"] for r in csv_rows]
    totals = [r["total"] for r in csv_rows]

    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY — First {len(csv_rows)} frames")
    print(f"{'='*60}")
    print(f"  Red team        — avg: {np.mean(reds):.1f}, min: {min(reds)}, max: {max(reds)}")
    print(f"  White team      — avg: {np.mean(whites):.1f}, min: {min(whites)}, max: {max(whites)}")
    print(f"  GK Yellow-Green — avg: {np.mean(gk_y):.1f}, min: {min(gk_y)}, max: {max(gk_y)}")
    print(f"  GK Blue         — avg: {np.mean(gk_b):.1f}, min: {min(gk_b)}, max: {max(gk_b)}")
    print(f"  Total dets      — avg: {np.mean(totals):.1f}, min: {min(totals)}, max: {max(totals)}")
    print(f"\n  Frames with 0 red:    {sum(1 for r in reds if r == 0)}")
    print(f"  Frames with 0 white:  {sum(1 for w in whites if w == 0)}")
    print(f"  Frames with GK_Y:     {sum(1 for g in gk_y if g > 0)}")
    print(f"  Frames with GK_B:     {sum(1 for g in gk_b if g > 0)}")
    print(f"\n  Annotated frames saved to: {OUTPUT_DIR}/")
    print(f"  Per-frame CSV saved to:    {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
