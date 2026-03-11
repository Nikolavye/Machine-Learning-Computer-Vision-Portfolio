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
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Reuse Solution C's helpers ──
CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parent
VIDEO_PATH = CASE_DIR / "sample.mp4"
OUTPUT_DIR = CASE_DIR / "frames_result_c"
OUTPUT_CSV = CASE_DIR / "eval_c_100.csv"

NUM_FRAMES = 100

MODELS_DIR = CASE_DIR / "models"

TEAM_COLORS = {
    "red": (0, 0, 255),
    "white": (255, 255, 255),
    "referee": (0, 255, 255),
    "goalkeeper_yellow": (0, 200, 200),
    "goalkeeper_blue": (255, 100, 0),
}


def choose_model_path():
    candidates = [
        MODELS_DIR / "yolo26x.pt",
        MODELS_DIR / "yolo26l.pt",
        MODELS_DIR / "yolo26m.pt",
        MODELS_DIR / "yolo26s.pt",
        MODELS_DIR / "yolo26n.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "yolo26n.pt"


def is_on_pitch(box, frame):
    frame_h, frame_w = frame.shape[:2] # frame_h is the height of the frame, frame_w is the width of the frame
    x1, y1, x2, y2 = box # x1, y1 is the top-left corner of the box, x2, y2 is the bottom-right corner of the box
    w = x2 - x1 # w is the width of the box
    h = y2 - y1 # h is the height of the box
    cx = (x1 + x2) / 2.0 # cx is the x-coordinate of the center of the box
    cy = (y1 + y2) / 2.0 # cy is the y-coordinate of the center of the box

    if cy < frame_h * 0.25 or cy > frame_h * 0.98:
        return False
    if h < frame_h * 0.045 or h > frame_h * 0.60:
        return False
    if w < 8 or w > frame_w * 0.25:
        return False
    aspect = h / max(w, 1.0)
    if aspect < 1.0 or aspect > 6.5:
        return False
    # Require feet to be low enough on screen to be on the pitch.
    # Distant players can have y2 as low as ~0.36. The grass check below
    # prevents spectator false positives even with this relaxed threshold.
    if y2 < frame_h * 0.36:
        return False
    if cx < frame_w * 0.01 or cx > frame_w * 0.99:
        return False

    # Grass check below feet
    ix1, iy1, ix2, iy2 = map(int, [x1, y1, x2, y2])
    foot_y1 = max(0, min(frame_h - 1, iy2))
    foot_y2 = max(foot_y1 + 1, min(frame_h, iy2 + max(5, int(frame_h * 0.03))))
    pad = max(2, int((ix2 - ix1) * 0.1))
    foot_x1 = max(0, ix1 + pad)
    foot_x2 = max(foot_x1 + 1, min(frame_w, ix2 - pad))
    foot_patch = frame[foot_y1:foot_y2, foot_x1:foot_x2]
    if foot_patch.size == 0:
        return False
    foot_hsv = cv2.cvtColor(foot_patch, cv2.COLOR_BGR2HSV)
    green_mask = (
        (foot_hsv[:, :, 0] >= 30)
        & (foot_hsv[:, :, 0] <= 95)
        & (foot_hsv[:, :, 1] >= 35)
        & (foot_hsv[:, :, 2] >= 25)
    )
    if float(np.mean(green_mask)) < 0.15:
        return False
    return True


def classify_by_strict_colors(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    w = x2 - x1

    # Narrow the crop horizontally (central 60%) to exclude background
    # (e.g. advertising boards behind the player).
    pad_x = int(w * 0.2)
    cx1 = max(x1 + pad_x, 0)
    cx2 = max(x2 - pad_x, cx1 + 1)

    upper_crop = frame[max(y1, 0):max(y1 + int(h * 0.4), y1 + 1), cx1:cx2]
    lower_crop = frame[max(y1 + int(h * 0.4), 0):max(y1 + int(h * 0.7), y1 + 1), cx1:cx2]

    if upper_crop.size == 0 or lower_crop.size == 0:
        return None, None

    def get_fg_pixels(crop):
        """Get non-green foreground pixels."""
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_fg = cv2.bitwise_not(cv2.inRange(hsv_crop, lower_green, upper_green))
        return crop[mask_fg > 0]

    def get_dominant_fg_color(crop):
        fg_pixels = get_fg_pixels(crop)
        if len(fg_pixels) < 5:
            return np.array([0, 0, 0])
        return np.median(fg_pixels, axis=0)

    def pixel_vote_team(crop):
        """Count white-like vs red-like pixels for robust classification.
        Returns (white_ratio, red_ratio, yellow_ratio) of foreground pixels."""
        fg_pixels = get_fg_pixels(crop)
        if len(fg_pixels) < 10:
            return 0, 0, 0, 0
        fg_hsv = cv2.cvtColor(fg_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        n = len(fg_pixels)
        # White: low saturation, high value
        white_mask = (fg_hsv[:, 1] < 50) & (fg_hsv[:, 2] > 140)
        # Red: red hue range, moderate saturation
        red_mask = ((fg_hsv[:, 0] < 12) | (fg_hsv[:, 0] > 168)) & (fg_hsv[:, 1] > 50) & (fg_hsv[:, 2] > 50)
        # Yellow (referee): yellow hue, high saturation
        yellow_mask = (fg_hsv[:, 0] >= 20) & (fg_hsv[:, 0] <= 38) & (fg_hsv[:, 1] >= 100) & (fg_hsv[:, 2] > 100)
        # Blue: blue hue, moderate saturation
        blue_mask = (fg_hsv[:, 0] >= 90) & (fg_hsv[:, 0] <= 130) & (fg_hsv[:, 1] >= 50) & (fg_hsv[:, 2] > 40)
        return float(np.sum(white_mask)) / n, float(np.sum(red_mask)) / n, float(np.sum(yellow_mask)) / n, float(np.sum(blue_mask)) / n

    upper_bgr = get_dominant_fg_color(upper_crop)
    lower_bgr = get_dominant_fg_color(lower_crop)

    # Pixel voting: count actual white vs red vs yellow vs blue pixels in upper body
    white_r, red_r, yellow_r, blue_r = pixel_vote_team(upper_crop)

    ideal_red = np.array([50, 50, 200])
    ideal_white = np.array([240, 240, 240])
    ideal_yellow = np.array([21, 169, 172])
    ideal_blue = np.array([130, 70, 50])  # Blue goalkeeper BGR
    ideal_black = np.array([30, 30, 30])

    sr = np.linalg.norm(upper_bgr - ideal_red) * 0.6 + np.linalg.norm(lower_bgr - ideal_red) * 0.4
    sw = np.linalg.norm(upper_bgr - ideal_white) * 0.6 + np.linalg.norm(lower_bgr - ideal_white) * 0.4
    sref = np.linalg.norm(upper_bgr - ideal_yellow) * 0.8 + np.linalg.norm(lower_bgr - ideal_black) * 0.2
    sblue = np.linalg.norm(upper_bgr - ideal_blue) * 0.7 + np.linalg.norm(lower_bgr - ideal_blue) * 0.3

    # Pixel vote override
    vote_override = None
    white_r_lower, red_r_lower, _, blue_r_lower = pixel_vote_team(lower_crop)

    # ── Blue goalkeeper gate ──
    ub_hsv = cv2.cvtColor(np.uint8([[upper_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    ub_h, ub_s, ub_v = int(ub_hsv[0]), int(ub_hsv[1]), int(ub_hsv[2])

    if blue_r > 0.25 or (90 <= ub_h <= 130 and ub_s >= 50 and ub_v > 40):
        vote_override = "goalkeeper_blue"
    elif white_r > 0.12 and white_r_lower > 0.15:
        vote_override = "white"
    elif red_r > 0.50 and white_r < 0.08 and white_r_lower < 0.15:
        vote_override = "red"
    elif yellow_r > 0.30:
        vote_override = "goalkeeper_yellow"

    # ── Referee / yellow-green goalkeeper gate ──
    is_yellow_hue = 20 <= ub_h <= 38
    is_high_sat = ub_s >= 120
    raw_yellow_dist = np.linalg.norm(upper_bgr - ideal_yellow)
    is_very_close_to_yellow = raw_yellow_dist < 50

    if not ((is_yellow_hue and is_high_sat) or is_very_close_to_yellow):
        sref += 1000

    if yellow_r > 0.30:
        sref = min(sref, 50)

    # Blue goalkeeper: require actual blue hue
    if not (90 <= ub_h <= 130 and ub_s >= 50):
        sblue += 1000

    feature = np.concatenate([upper_bgr, lower_bgr])
    min_score = min(sr, sw, sref, sblue)
    if min_score > 200:
        return None, None, None
    return feature, upper_bgr, vote_override


def get_team_by_color(b, g, r):
    ideal_white = np.array([240, 240, 240])
    ideal_red = np.array([50, 50, 200])
    ideal_yellow = np.array([21, 169, 172])
    ideal_blue = np.array([130, 70, 50])
    color = np.array([b, g, r])

    d_white = np.linalg.norm(color - ideal_white)
    d_red = np.linalg.norm(color - ideal_red)
    d_referee = np.linalg.norm(color - ideal_yellow)
    d_blue = np.linalg.norm(color - ideal_blue)

    ub_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]

    # ── Blue goalkeeper: blue hue + saturation ──
    if 90 <= ub_hsv[0] <= 130 and ub_hsv[1] >= 50 and ub_hsv[2] > 40:
        return "goalkeeper_blue"

    # ── Yellow-green goalkeeper / referee gate ──
    if ub_hsv[1] < 120 and d_referee > 50:
        d_referee += 1000

    # If it passes referee gate, classify as goalkeeper_yellow (not generic referee)
    if d_referee < d_white and d_referee < d_red and d_referee < 1000:
        return "goalkeeper_yellow"

    # Dark clothing gate
    if ub_hsv[2] < 80:
        return None

    # Red team requires actual redness
    if d_red < d_white:
        if r < 80 or (r < b or r < g):
            return None

    if d_white < d_red:
        return "white"
    return "red"


def main():
    model_path = choose_model_path()
    print(f"Loading model: {model_path}")
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
    # Clear existing files
    for f in OUTPUT_DIR.iterdir():
        f.unlink()

    csv_rows = []

    for frame_idx in range(NUM_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break

        # Static detection (no tracking)
        results = model(frame, verbose=False, conf=0.10, iou=0.5, classes=[0], device="mps")
        r = results[0]

        counts = {"red": 0, "white": 0, "goalkeeper_yellow": 0, "goalkeeper_blue": 0}
        annotated = frame.copy()

        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                if not is_on_pitch(box, frame):
                    continue

                strict_feat, upper_bgr, vote_override = classify_by_strict_colors(frame, box)
                if strict_feat is None:
                    continue

                if vote_override is not None:
                    team = vote_override
                else:
                    team = get_team_by_color(upper_bgr[0], upper_bgr[1], upper_bgr[2])
                    if team is None:
                        continue  # rejected as spectator (dark clothing)
                counts[team] += 1

                # Draw bounding box
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

        # Save frame
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
            print(f"  frame {frame_idx:3d}: red={counts['red']}, white={counts['white']}, gk_y={counts['goalkeeper_yellow']}, gk_b={counts['goalkeeper_blue']}")

    cap.release()

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "red", "white", "goalkeeper_yellow", "goalkeeper_blue", "total"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Summary statistics
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
