"""
Real-Time Sports Player Tracking Pipeline
(YOLOv8 + BoT-SORT + Track-Level Consensus Voting)

Design goals:
1) Stable counts for moving camera footage.
2) Use track-level majority voting (not per-frame hard labels).
3) Smooth final counts over a short temporal window.
4) Detect both goalkeepers (yellow-green and blue).

Outputs:
- sports_player_tracking/output_c.mp4
- sports_player_tracking/counts_c.csv
"""

import csv
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Paths
ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = CASE_DIR / "sample.mp4"
OUTPUT_VIDEO = CASE_DIR / "output_c.mp4"
OUTPUT_CSV = CASE_DIR / "counts_c.csv"

# Tracker config
TRACKER_CFG = "botsort.yaml"

# Stabilization config
VOTE_WINDOW = 20
MIN_VOTES_FOR_STABLE_ID = 3
SMOOTH_WINDOW = 10

# Visualization
TEAM_COLORS = {
    "red": (0, 0, 255),
    "white": (255, 255, 255),
    "goalkeeper_yellow": (0, 200, 200),
    "goalkeeper_blue": (255, 100, 0),
}

MODELS_DIR = CASE_DIR / "models"


def choose_model_path() -> str:
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


def is_on_pitch(box, frame: np.ndarray) -> bool:
    """Filter out spectators / obvious false positives."""
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    if cy < frame_h * 0.25 or cy > frame_h * 0.98:
        return False
    if h < frame_h * 0.045 or h > frame_h * 0.60:
        return False
    if w < 8 or w > frame_w * 0.25:
        return False
    aspect = h / max(w, 1.0)
    if aspect < 1.0 or aspect > 6.5:
        return False
    if y2 < frame_h * 0.36:
        return False
    if cx < frame_w * 0.01 or cx > frame_w * 0.99:
        return False

    # Grass check below feet.
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
    """Classify a detection into red / white / goalkeeper_yellow / goalkeeper_blue.

    Returns (feature, upper_bgr, vote_override) or (None, None, None) if rejected.
    """
    x1, y1, x2, y2 = map(int, box)
    h = y2 - y1
    w = x2 - x1

    # Narrow crop horizontally (central 60%) to exclude background
    pad_x = int(w * 0.2)
    cx1 = max(x1 + pad_x, 0)
    cx2 = max(x2 - pad_x, cx1 + 1)

    upper_crop = frame[max(y1, 0):max(y1 + int(h * 0.4), y1 + 1), cx1:cx2]
    lower_crop = frame[max(y1 + int(h * 0.4), 0):max(y1 + int(h * 0.7), y1 + 1), cx1:cx2]

    if upper_crop.size == 0 or lower_crop.size == 0:
        return None, None, None

    def get_fg_pixels(crop):
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask_fg = cv2.bitwise_not(
            cv2.inRange(hsv_crop, np.array([35, 40, 40]), np.array([85, 255, 255]))
        )
        return crop[mask_fg > 0]

    def get_dominant_fg_color(crop):
        fg_pixels = get_fg_pixels(crop)
        if len(fg_pixels) < 5:
            return np.array([0, 0, 0])
        return np.median(fg_pixels, axis=0)

    def pixel_vote_team(crop):
        fg_pixels = get_fg_pixels(crop)
        if len(fg_pixels) < 10:
            return 0, 0, 0, 0
        fg_hsv = cv2.cvtColor(fg_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        n = len(fg_pixels)
        white_mask = (fg_hsv[:, 1] < 50) & (fg_hsv[:, 2] > 140)
        red_mask = ((fg_hsv[:, 0] < 12) | (fg_hsv[:, 0] > 168)) & (fg_hsv[:, 1] > 50) & (fg_hsv[:, 2] > 50)
        yellow_mask = (fg_hsv[:, 0] >= 20) & (fg_hsv[:, 0] <= 38) & (fg_hsv[:, 1] >= 100) & (fg_hsv[:, 2] > 100)
        blue_mask = (fg_hsv[:, 0] >= 90) & (fg_hsv[:, 0] <= 130) & (fg_hsv[:, 1] >= 50) & (fg_hsv[:, 2] > 40)
        return float(np.sum(white_mask)) / n, float(np.sum(red_mask)) / n, float(np.sum(yellow_mask)) / n, float(np.sum(blue_mask)) / n

    upper_bgr = get_dominant_fg_color(upper_crop)
    lower_bgr = get_dominant_fg_color(lower_crop)

    white_r, red_r, yellow_r, blue_r = pixel_vote_team(upper_crop)

    ideal_red = np.array([50, 50, 200])
    ideal_white = np.array([240, 240, 240])
    ideal_yellow = np.array([21, 169, 172])
    ideal_blue = np.array([130, 70, 50])
    ideal_black = np.array([30, 30, 30])

    sr = np.linalg.norm(upper_bgr - ideal_red) * 0.6 + np.linalg.norm(lower_bgr - ideal_red) * 0.4
    sw = np.linalg.norm(upper_bgr - ideal_white) * 0.6 + np.linalg.norm(lower_bgr - ideal_white) * 0.4
    sref = np.linalg.norm(upper_bgr - ideal_yellow) * 0.8 + np.linalg.norm(lower_bgr - ideal_black) * 0.2
    sblue = np.linalg.norm(upper_bgr - ideal_blue) * 0.7 + np.linalg.norm(lower_bgr - ideal_blue) * 0.3

    # Pixel vote override for robust classification
    vote_override = None
    white_r_lower, red_r_lower, _, blue_r_lower = pixel_vote_team(lower_crop)

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

    # Yellow/referee gate
    is_yellow_hue = 20 <= ub_h <= 38
    is_high_sat = ub_s >= 120
    raw_yellow_dist = np.linalg.norm(upper_bgr - ideal_yellow)
    is_very_close_to_yellow = raw_yellow_dist < 50

    if not ((is_yellow_hue and is_high_sat) or is_very_close_to_yellow):
        sref += 1000

    if yellow_r > 0.30:
        sref = min(sref, 50)

    # Blue goalkeeper gate
    if not (90 <= ub_h <= 130 and ub_s >= 50):
        sblue += 1000

    feature = np.concatenate([upper_bgr, lower_bgr])
    min_score = min(sr, sw, sref, sblue)
    if min_score > 200:
        return None, None, None

    return feature, upper_bgr, vote_override


def get_team_by_color(b, g, r):
    """Fallback classification when pixel voting has no override."""
    ideal_white = np.array([240, 240, 240])
    ideal_red = np.array([50, 50, 200])
    ideal_yellow = np.array([21, 169, 172])
    color = np.array([b, g, r])

    d_white = np.linalg.norm(color - ideal_white)
    d_red = np.linalg.norm(color - ideal_red)
    d_referee = np.linalg.norm(color - ideal_yellow)

    ub_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]

    # Blue goalkeeper
    if 90 <= ub_hsv[0] <= 130 and ub_hsv[1] >= 50 and ub_hsv[2] > 40:
        return "goalkeeper_blue"

    # Yellow-green goalkeeper
    if ub_hsv[1] < 120 and d_referee > 50:
        d_referee += 1000
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


def majority_vote(votes: deque) -> str:
    c = Counter(votes)
    return c.most_common(1)[0][0]


def draw_overlay(frame, smooth_red, smooth_white, gk_y, gk_b):
    lines = [
        ("Red Team", smooth_red, TEAM_COLORS["red"]),
        ("White Team", smooth_white, TEAM_COLORS["white"]),
        ("GK Yellow", gk_y, TEAM_COLORS["goalkeeper_yellow"]),
        ("GK Blue", gk_b, TEAM_COLORS["goalkeeper_blue"]),
    ]
    y = 30
    for name, value, color in lines:
        text = f"{name}: {value}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (10, y - th - 6), (10 + tw + 10, y + 6), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 34


def main():
    model_path = choose_model_path()
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {frame_w}x{frame_h} @ {fps:.1f}fps, {total_frames} frames")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    track_votes = defaultdict(lambda: deque(maxlen=VOTE_WINDOW))
    track_centers = defaultdict(lambda: deque(maxlen=20))
    static_track_ids = set()
    smooth_red_buf = deque(maxlen=SMOOTH_WINDOW)
    smooth_white_buf = deque(maxlen=SMOOTH_WINDOW)

    csv_rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            frame,
            persist=True,
            conf=0.10,
            iou=0.5,
            tracker=TRACKER_CFG,
            classes=[0],
            verbose=False,
            device="mps",
        )
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
            confs = np.empty((0,), dtype=np.float32)
            track_ids = np.empty((0,), dtype=np.int64)
        else:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            if r.boxes.id is None:
                track_ids = -np.arange(1, len(boxes) + 1, dtype=np.int64)
            else:
                track_ids = r.boxes.id.cpu().numpy().astype(np.int64)

        detections = []
        for box, conf, tid in zip(boxes, confs, track_ids):
            if not is_on_pitch(box, frame):
                continue

            # Suppress static high-position tracks (audience/boards).
            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)
            hist = track_centers[int(tid)]
            hist.append((cx, cy))
            if len(hist) >= 12:
                arr = np.array(hist, dtype=np.float32)
                drift = float(np.max(np.linalg.norm(arr - arr[0], axis=1)))
                if drift < 8.0 and cy < frame_h * 0.68:
                    static_track_ids.add(int(tid))
            if int(tid) in static_track_ids:
                continue

            # Classify by color
            strict_feat, upper_bgr, vote_override = classify_by_strict_colors(frame, box)
            if strict_feat is None:
                continue

            if vote_override is not None:
                team = vote_override
            else:
                team = get_team_by_color(upper_bgr[0], upper_bgr[1], upper_bgr[2])
                if team is None:
                    continue

            detections.append((int(tid), box, float(conf), team))

        frame_tracks_by_team = {"red": set(), "white": set(), "goalkeeper_yellow": set(), "goalkeeper_blue": set()}

        for tid, box, conf, instant_team in detections:
            votes = track_votes[tid]
            votes.append(instant_team)
            if len(votes) >= MIN_VOTES_FOR_STABLE_ID:
                team = majority_vote(votes)
            else:
                team = instant_team

            if team in frame_tracks_by_team:
                frame_tracks_by_team[team].add(tid)

            # Draw detection
            x1, y1, x2, y2 = map(int, box)
            color = TEAM_COLORS.get(team, (180, 180, 180))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            txt = f"{team} id:{tid} {conf:.2f}"
            cv2.putText(frame, txt, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        raw_red = len(frame_tracks_by_team["red"])
        raw_white = len(frame_tracks_by_team["white"])
        raw_gk_y = len(frame_tracks_by_team["goalkeeper_yellow"])
        raw_gk_b = len(frame_tracks_by_team["goalkeeper_blue"])

        smooth_red_buf.append(raw_red)
        smooth_white_buf.append(raw_white)

        smooth_red = int(np.median(smooth_red_buf)) if smooth_red_buf else 0
        smooth_white = int(np.median(smooth_white_buf)) if smooth_white_buf else 0

        draw_overlay(frame, smooth_red, smooth_white, raw_gk_y, raw_gk_b)

        writer.write(frame)

        csv_rows.append(
            {
                "frame_id": frame_idx,
                "raw_red": raw_red,
                "raw_white": raw_white,
                "smooth_red": smooth_red,
                "smooth_white": smooth_white,
                "gk_yellow": raw_gk_y,
                "gk_blue": raw_gk_b,
                "total_detections": len(detections),
            }
        )

        if frame_idx % 50 == 0:
            print(
                f"frame={frame_idx:4d} "
                f"raw=({raw_red},{raw_white}) "
                f"smooth=({smooth_red},{smooth_white}) "
                f"gk=({raw_gk_y},{raw_gk_b}) "
                f"dets={len(detections)}"
            )

        frame_idx += 1

    cap.release()
    writer.release()

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "raw_red",
                "raw_white",
                "smooth_red",
                "smooth_white",
                "gk_yellow",
                "gk_blue",
                "total_detections",
            ],
        )
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    print(f"\nDone.")
    print(f"  Video: {OUTPUT_VIDEO}")
    print(f"  CSV:   {OUTPUT_CSV}")
    if csv_rows:
        reds = [r["smooth_red"] for r in csv_rows]
        whites = [r["smooth_white"] for r in csv_rows]
        print(f"  Avg smooth red:   {float(np.mean(reds)):.2f}")
        print(f"  Avg smooth white: {float(np.mean(whites)):.2f}")


if __name__ == "__main__":
    main()
