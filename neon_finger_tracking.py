"""
Finger Tracking Shapes — High Performance Edition
===================================================
Real-time hand tracking with smooth, lightweight line drawing.

Controls:
    [1] Cyan    [2] Magenta   [3] Green   [4] Gold   [5] Rainbow
    [M] Cycle shape mode    [T] Toggle trail    [C] Clear trail
    [G] Toggle glow (on/off for extra FPS)
    [Q / ESC] Quit

Requirements:
    pip install opencv-python mediapipe numpy

Model file (auto-downloaded on first run):
    hand_landmarker.task
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import time
import math
import os
import sys
import argparse
import threading
import urllib.request
from collections import deque


# ─────────────────────────── CONSTANTS ───────────────────────────

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# BGR color presets
COLORS = {
    "white":   (255, 255, 255),
    "cyan":    (255, 255, 0),
    "magenta": (255, 0, 255),
    "green":   (0, 255, 128),
    "gold":    (0, 215, 255),
}
COLOR_NAMES = list(COLORS.keys())

# All fingertip landmark indices
# Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
FINGERTIP_IDS = [4, 8, 12, 16, 20]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# PIP (proximal interphalangeal) joint indices — used for extended detection
# For index/middle/ring/pinky: PIP joints
# For thumb: IP joint (landmark 3)
FINGER_PIP_IDS = [3, 6, 10, 14, 18]
# Index MCP (landmark 5) — used as reference for thumb extended check
INDEX_MCP_ID = 5

# Shape modes
SHAPE_MODES = ["web", "line", "triangle", "rectangle", "diamond"]

# Fill effect modes
FILL_MODES = ["none", "bw", "heatmap", "halftone"]


# ─────────────────────────── MODEL DOWNLOAD ───────────────────────────

def ensure_model_exists():
    """Download the hand_landmarker.task model if it doesn't exist."""
    if os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH)
        if size > 1_000_000:
            return
        print(f"[!] Model file seems too small ({size} bytes). Re-downloading...")

    print("[*] Downloading hand_landmarker.task model (~8 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[OK] Model downloaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        print(f"   Please download manually from:\n   {MODEL_URL}")
        print(f"   and place it at: {MODEL_PATH}")
        raise SystemExit(1)


# ─────────────────────────── THREADED CAMERA ───────────────────────────

class ThreadedCamera:
    """
    Reads frames in a background thread so the main loop always gets
    the latest frame, eliminating buffer delay from network streams.
    """

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ret = False
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()

    def _grab_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()


# ─────────────────────────── ARROW ANIMATION ───────────────────────────

class ArrowAnimation:
    def __init__(self):
        self.arrows = []

    def shoot(self, start_pos, direction_vector):
        speed = 45.0
        mag = math.hypot(*direction_vector)
        if mag == 0: return
        vx = (direction_vector[0]/mag) * speed
        vy = (direction_vector[1]/mag) * speed
        self.arrows.append({'pos': list(start_pos), 'velocity': [vx, vy], 'age': 0})

    def update_and_draw(self, canvas, color):
        for arr in self.arrows[:]:
            arr['pos'][0] += arr['velocity'][0]
            arr['pos'][1] += arr['velocity'][1]
            arr['age'] += 1

            x, y = int(arr['pos'][0]), int(arr['pos'][1])
            vx, vy = arr['velocity']

            # Arrow tail
            tx = int(x - vx * 1.5)
            ty = int(y - vy * 1.5)

            # Draw arrow
            cv2.arrowedLine(canvas, (tx, ty), (x, y), color, 4, tipLength=0.3)
            # Inner white glow
            cv2.arrowedLine(canvas, (tx, ty), (x, y), (255, 255, 255), 2, tipLength=0.3)

            if arr['age'] > 30:
                self.arrows.remove(arr)


# ─────────────────────────── LIGHTWEIGHT DRAWING ───────────────────────────

def draw_line(canvas, pt1, pt2, color, thickness=2):
    """Draw a simple anti-aliased line. Fast."""
    cv2.line(canvas, _int_pt(pt1), _int_pt(pt2), color, thickness, cv2.LINE_AA)


def draw_glow_line(canvas, pt1, pt2, color, thickness=2):
    """Draw a line with a lightweight 2-layer glow (core + soft outer)."""
    p1, p2 = _int_pt(pt1), _int_pt(pt2)
    # Soft outer glow (semi-transparent, thicker)
    glow_color = tuple(c // 3 for c in color)
    cv2.line(canvas, p1, p2, glow_color, thickness + 8, cv2.LINE_AA)
    # Colored core
    cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
    # Bright center
    cv2.line(canvas, p1, p2, (255, 255, 255), max(1, thickness // 2), cv2.LINE_AA)


def _int_pt(pt):
    """Convert a point to integer tuple."""
    return (int(pt[0]), int(pt[1]))


# ─────────────────────────── TRAIL SYSTEM ───────────────────────────

class TrailSystem:
    """Lightweight trail/afterglow effect for finger positions."""

    def __init__(self, max_length=40):
        self.trails = {}  # key -> deque of points
        self.enabled = True

    def add_point(self, key, point, max_length=40):
        if point is None:
            return
        if key not in self.trails:
            self.trails[key] = deque(maxlen=max_length)
        self.trails[key].append(point)

    def clear(self):
        self.trails.clear()

    def draw(self, canvas, color):
        if not self.enabled:
            return
        for trail in self.trails.values():
            n = len(trail)
            if n < 2:
                continue
            for i in range(1, n):
                alpha = i / n
                fade_color = tuple(int(c * alpha) for c in color)
                thickness = max(1, int(2 * alpha))
                cv2.line(canvas, _int_pt(trail[i - 1]), _int_pt(trail[i]),
                         fade_color, thickness, cv2.LINE_AA)


# ─────────────────────────── UTILITY ───────────────────────────

def get_rainbow_color(t):
    """Smoothly cycling rainbow color (BGR)."""
    r = int(127.5 * (1 + math.sin(t)))
    g = int(127.5 * (1 + math.sin(t + 2.094)))
    b = int(127.5 * (1 + math.sin(t + 4.189)))
    return (b, g, r)


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_finger_extended(hand_landmarks, finger_index):
    """
    Check if a finger is extended (pointing up/out) or curled.

    For index/middle/ring/pinky: extended if fingertip Y < PIP joint Y
        (fingertip is above the PIP joint on screen)
    For thumb: extended if tip is farther from index MCP than IP joint is
        (works regardless of hand orientation)
    """
    tip = hand_landmarks[FINGERTIP_IDS[finger_index]]
    pip = hand_landmarks[FINGER_PIP_IDS[finger_index]]

    if finger_index == 0:  # Thumb — use distance-based check
        index_mcp = hand_landmarks[INDEX_MCP_ID]
        tip_dist = math.sqrt((tip.x - index_mcp.x) ** 2 + (tip.y - index_mcp.y) ** 2)
        pip_dist = math.sqrt((pip.x - index_mcp.x) ** 2 + (pip.y - index_mcp.y) ** 2)
        return tip_dist > pip_dist
    else:  # Other fingers — tip above PIP joint
        return tip.y < pip.y


def extract_all_fingertips(hand_landmarks, w, h):
    """Extract fingertip positions for EXTENDED fingers only.
    Returns dict: {"thumb": (x,y), ...} — only includes fingers that are up.
    """
    tips = {}
    for i, (idx, name) in enumerate(zip(FINGERTIP_IDS, FINGER_NAMES)):
        if is_finger_extended(hand_landmarks, i):
            lm = hand_landmarks[idx]
            tips[name] = (lm.x * w, lm.y * h)
    return tips


# ─────────────────────────── SHAPE DRAWING ───────────────────────────

def _sort_hands_by_position(hand_data):
    """
    Sort hands by X position (leftmost hand first).
    Returns (hand_left_tips, hand_right_tips) regardless of MediaPipe labels.
    """
    hands = list(hand_data.values())
    if len(hands) < 2:
        return hands[0], hands[0]
    # Sort by average X of available fingertips
    def avg_x(tips):
        if not tips:
            return 0
        return sum(p[0] for p in tips.values()) / len(tips)
    hands.sort(key=avg_x)
    return hands[0], hands[1]


def _apply_fill_effect(canvas, points, mode):
    """Apply visual effects inside the polygon defined by points."""
    if mode == "none" or len(points) < 3:
        return

    pts = np.array([_int_pt(p) for p in points], dtype=np.int32)
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    if mode == "bw":
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        effect_layer = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    elif mode == "heatmap":
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        effect_layer = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    elif mode == "halftone":
        # Pixelated halftone effect
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (canvas.shape[1]//8, canvas.shape[0]//8), interpolation=cv2.INTER_LINEAR)
        large = cv2.resize(small, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST)
        _, thresh = cv2.threshold(large, 127, 255, cv2.THRESH_BINARY)
        effect_layer = np.zeros_like(canvas)
        effect_layer[thresh == 0] = (80, 40, 10)   # Dark blue tint for shadows
        effect_layer[thresh == 255] = (255, 255, 255) # White for highlights

    # Blend effect layer only where mask is 255
    canvas[mask == 255] = effect_layer[mask == 255]


def draw_shape(canvas, hand_data, color, mode, draw_fn, fill_mode="none"):
    """
    Draw shapes connecting two hands.
    Only draws to extended (open) fingers.
    """
    left, right = _sort_hands_by_position(hand_data)

    if mode == "line":
        # Connect index fingers if both extended
        if "index" in left and "index" in right:
            draw_fn(canvas, left["index"], right["index"], color, 3)

    elif mode == "triangle":
        li = left.get("index")
        ri = right.get("index")
        lt = left.get("thumb")
        rt = right.get("thumb")
        if li and ri and (lt or rt):
            tmid = midpoint(lt or li, rt or ri)
            pts = [li, ri, tmid]
            _apply_fill_effect(canvas, pts, fill_mode)
            _draw_polygon(canvas, pts, color, draw_fn)

    elif mode == "rectangle":
        pts = [left.get("index"), right.get("index"),
               right.get("thumb"), left.get("thumb")]
        pts = [p for p in pts if p]  # filter out None
        if len(pts) >= 3:
            hull = cv2.convexHull(np.array([_int_pt(p) for p in pts], dtype=np.int32))
            hull_pts = [tuple(p[0]) for p in hull]
            _apply_fill_effect(canvas, hull_pts, fill_mode)
            _draw_polygon(canvas, pts, color, draw_fn)

    elif mode == "diamond":
        li = left.get("index")
        ri = right.get("index")
        if li and ri:
            cx, cy = midpoint(li, ri)
            dx = abs(ri[0] - li[0]) / 2
            dy = abs(ri[1] - li[1]) / 2 + 60
            pts = [(cx, cy - dy), (cx + dx, cy), (cx, cy + dy), (cx - dx, cy)]
            _apply_fill_effect(canvas, pts, fill_mode)
            _draw_polygon(canvas, pts, color, draw_fn)

    elif mode == "web":
        # Chain extended fingertips as one continuous line:
        left_chain = [left[n] for n in reversed(FINGER_NAMES) if n in left]
        right_chain = [right[n] for n in FINGER_NAMES if n in right]
        full_chain = left_chain + right_chain
        
        if len(full_chain) >= 3:
            hull = cv2.convexHull(np.array([_int_pt(p) for p in full_chain], dtype=np.int32))
            hull_pts = [tuple(p[0]) for p in hull]
            _apply_fill_effect(canvas, hull_pts, fill_mode)

        for i in range(len(full_chain) - 1):
            draw_fn(canvas, full_chain[i], full_chain[i + 1], color, 2)

        # Also connect matching extended fingers across hands
        for name in FINGER_NAMES:
            if name in left and name in right:
                draw_fn(canvas, left[name], right[name], color, 1)


def _draw_polygon(canvas, points, color, draw_fn):
    """Draw a closed polygon using the given line drawing function."""
    n = len(points)
    for i in range(n):
        draw_fn(canvas, points[i], points[(i + 1) % n], color, 3)


def draw_single_hand(canvas, tips, color, draw_fn):
    """Draw effects for a single hand — connect extended fingertips in a chain."""
    # Only chain the fingers that are actually extended
    active = [tips[n] for n in FINGER_NAMES if n in tips]
    for i in range(len(active) - 1):
        draw_fn(canvas, active[i], active[i + 1], color, 2)


# ─────────────────────────── HUD ───────────────────────────

def draw_hud(frame, fps, color_name, shape_mode, trail_on, glow_on, hands_count, fill_mode):
    """Minimal HUD overlay."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 28
    texts = [
        (f"FPS: {fps:.0f}", 10),
        (f"Color: {color_name.upper()}", 100),
        (f"Shape: {shape_mode.upper()}", 260),
        (f"Fill: {fill_mode.upper()}", 420),
        (f"Trail: {'ON' if trail_on else 'OFF'}", 560),
        (f"Glow: {'ON' if glow_on else 'OFF'}", 680),
        (f"Hands: {hands_count}", w - 110),
    ]
    for text, x in texts:
        cv2.putText(frame, text, (x, y), font, 0.5, (0, 240, 200), 1, cv2.LINE_AA)


# ─────────────────────────── MAIN ───────────────────────────

def main():
    # ─── Parse arguments ───
    parser = argparse.ArgumentParser(description="Finger Tracking Shapes")
    parser.add_argument(
        "--source", "-s",
        default="0",
        help=(
            "Camera source. Options:\n"
            "  0, 1, 2...         = local webcam index (default: 0)\n"
            "  http://IP:PORT/video = IP Webcam stream URL\n"
            "\n"
            "For IP Webcam (Android):\n"
            "  1. Install 'IP Webcam' from Play Store\n"
            "  2. Open app, tap 'Start server'\n"
            "  3. Use the URL shown, e.g.: http://192.168.1.5:8080/video"
        ),
    )
    args = parser.parse_args()

    ensure_model_exists()

    # ─── MediaPipe HandLandmarker (async LIVE_STREAM) ───
    latest_result = {"data": None}

    def _on_result(result, output_image, timestamp_ms):
        latest_result["data"] = result

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=_on_result,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # ─── Camera init ───
    cap = None
    source = args.source

    # Check if source is a URL (IP Webcam) or local camera index
    if source.startswith("http"):
        print(f"[*] Connecting to IP Webcam: {source}")
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            ret_test, _ = cap.read()
            if ret_test:
                print("[OK] IP Webcam connected!")
            else:
                cap.release()
                cap = None
                print("[!] Connected but cannot read frames.")
        else:
            print("[!] Cannot connect to IP Webcam.")
            print("    Make sure:")
            print("    1. HP dan PC di WiFi yang sama")
            print("    2. App IP Webcam sudah 'Start server'")
            print("    3. URL benar (cek di app IP Webcam)")
            cap = None
    else:
        # Local camera
        cam_index = int(source)
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF,  "MSMF"),
            (cv2.CAP_ANY,   "Default"),
        ]
        for backend_id, backend_name in backends:
            print(f"[*] Trying camera: {backend_name}...")
            test_cap = cv2.VideoCapture(cam_index, backend_id)
            if test_cap.isOpened():
                for _ in range(5):
                    test_cap.read()
                    time.sleep(0.03)
                ret_test, _ = test_cap.read()
                if ret_test:
                    cap = test_cap
                    print(f"[OK] Camera opened with {backend_name}.")
                    break
                else:
                    test_cap.release()
            else:
                test_cap.release()

    if cap is None:
        print("[ERROR] Cannot open camera. Check connection and retry.")
        landmarker.close()
        return

    # Only set resolution for local cameras (IP Webcam handles its own)
    if not source.startswith("http"):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Wrap in ThreadedCamera for IP Webcam to eliminate buffer delay
    is_ip_cam = source.startswith("http")
    if is_ip_cam:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam = ThreadedCamera(cap)
        print("[*] Threaded frame grabber started (anti-delay)")
        print("")
        print("    TIPS kurangi delay di app IP Webcam:")
        print("    - Video preferences > Resolution: 640x480")
        print("    - Video preferences > Quality: 50%")
        print("    - Video preferences > FPS limit: 30")
        print("")
    else:
        cam = cap  # local cam, use directly

    # ─── State ───
    trail = TrailSystem(max_length=40)
    arrow_anim = ArrowAnimation()
    color_idx = 0
    rainbow = False
    shape_idx = 0
    fill_idx = 0
    use_glow = False  # Start with glow OFF for max FPS
    
    # Auto-cycle timers
    prev_time = time.time()
    last_fill_switch = time.time()
    
    # Bow and arrow state
    bow_drawn = False
    bow_hand = None
    pull_hand = None

    fps = 0.0
    ts_ms = 0
    fail_count = 0

    print("\n" + "=" * 58)
    print("  FINGER TRACKING SHAPES -- HIGH PERFORMANCE")
    print("=" * 58)
    print("  [1-4] Color  [5] Rainbow  [M] Shape mode")
    print("  [F] FillFX   [T] Trail    [C] Clear    [G] Toggle glow")
    print("  [Q/ESC] Quit")
    print("=" * 58 + "\n")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            fail_count += 1
            if fail_count > 30:
                print("[!] Too many frame failures. Exiting.")
                break
            time.sleep(0.005)
            continue
        fail_count = 0

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ─── Send frame to MediaPipe (async) ───
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms += 33
        try:
            landmarker.detect_async(mp_image, ts_ms)
        except Exception:
            pass

        # ─── FPS ───
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else 0.0

        # ─── Color ───
        if rainbow:
            color = get_rainbow_color(now * 2.5)
            cname = "rainbow"
        else:
            cname = COLOR_NAMES[color_idx]
            color = COLORS[cname]

        # Pick draw function based on glow toggle
        draw_fn = draw_glow_line if use_glow else draw_line

        # Auto cycle fill every 4 seconds
        current_time = time.time()
        if current_time - last_fill_switch > 4.0:
            fill_idx = (fill_idx + 1) % len(FILL_MODES)
            last_fill_switch = current_time

        # ─── Process hands ───
        hands_count = 0
        hand_data = {}  # label -> {all extended fingertip positions}
        raw_hands = {}  # label -> {raw key points for gestures}

        result = latest_result["data"]
        if result and result.hand_landmarks and result.handedness:
            hands_count = len(result.hand_landmarks)

            for i, (hand_lm, handedness) in enumerate(
                zip(result.hand_landmarks, result.handedness)
            ):
                label = f"hand_{i}"  # Use index, not Left/Right (unreliable)
                tips = extract_all_fingertips(hand_lm, w, h)
                hand_data[label] = tips

                # Store raw landmarks for gesture detection regardless of extended state
                raw_hands[label] = {
                    "thumb": (hand_lm[4].x * w, hand_lm[4].y * h),
                    "index": (hand_lm[8].x * w, hand_lm[8].y * h),
                    "middle": (hand_lm[12].x * w, hand_lm[12].y * h),
                }

                # Trail: track index fingertip (only if extended)
                index_tip = tips.get("index")
                if index_tip:
                    trail.add_point(f"{label}_index", index_tip)

        # ─── Bow and Arrow Gesture Logic ───
        if hands_count == 2:
            labels = list(raw_hands.keys())
            h1, h2 = raw_hands[labels[0]], raw_hands[labels[1]]
            
            # Distance between thumb and index for pinch detection
            pinch_dist1 = distance(h1["thumb"], h1["index"])
            pinch_dist2 = distance(h2["thumb"], h2["index"])
            
            is_pinched1 = pinch_dist1 < 40
            is_pinched2 = pinch_dist2 < 40
            hands_dist = distance(h1["middle"], h2["middle"])

            if hands_dist > 150:  # Hands must be far apart to draw a bow
                if is_pinched1 and not is_pinched2:
                    bow_drawn = True
                    pull_hand = h1
                    bow_hand = h2
                elif is_pinched2 and not is_pinched1:
                    bow_drawn = True
                    pull_hand = h2
                    bow_hand = h1
                else:
                    if bow_drawn:
                        # Released the pinch! Shoot arrow
                        direction = (bow_hand["middle"][0] - pull_hand["index"][0], 
                                     bow_hand["middle"][1] - pull_hand["index"][1])
                        arrow_anim.shoot(bow_hand["middle"], direction)
                    bow_drawn = False
            else:
                bow_drawn = False
        else:
            bow_drawn = False

        if bow_drawn:
            # Draw the glowing bow string
            bh_center = bow_hand["middle"]
            ph_pinch = pull_hand["index"]
            # String lines
            cv2.line(frame, _int_pt(bow_hand["thumb"]), _int_pt(ph_pinch), (255, 255, 255), 2)
            cv2.line(frame, _int_pt(bow_hand["index"]), _int_pt(ph_pinch), (255, 255, 255), 2)
            # Arrow loaded in bow
            cv2.line(frame, _int_pt(bh_center), _int_pt(ph_pinch), color, 4)

        # ─── Draw shape ───
        shape_mode = SHAPE_MODES[shape_idx]
        fill_mode = FILL_MODES[fill_idx]

        if hands_count >= 2:
            draw_shape(frame, hand_data, color, shape_mode, draw_fn, fill_mode)
        elif hands_count == 1:
            label = list(hand_data.keys())[0]
            draw_single_hand(frame, hand_data[label], color, draw_fn)

        # ─── Animations ───
        trail.draw(frame, color)
        arrow_anim.update_and_draw(frame, color)

        # ─── HUD ───
        draw_hud(frame, fps, cname, shape_mode, trail.enabled, use_glow, hands_count, fill_mode)

        # ─── Show ───
        cv2.imshow("Finger Tracking", frame)

        # ─── Input ───
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('1'):
            color_idx, rainbow = 0, False
        elif key == ord('2'):
            color_idx, rainbow = 1, False
        elif key == ord('3'):
            color_idx, rainbow = 2, False
        elif key == ord('4'):
            color_idx, rainbow = 3, False
        elif key == ord('5'):
            rainbow = True
        elif key in (ord('t'), ord('T')):
            trail.enabled = not trail.enabled
        elif key in (ord('c'), ord('C')):
            trail.clear()
        elif key in (ord('g'), ord('G')):
            use_glow = not use_glow
            print(f"  > Glow: {'ON' if use_glow else 'OFF'}")
        elif key in (ord('m'), ord('M')):
            shape_idx = (shape_idx + 1) % len(SHAPE_MODES)
            print(f"  > Shape: {SHAPE_MODES[shape_idx].upper()}")
        elif key in (ord('f'), ord('F')):
            fill_idx = (fill_idx + 1) % len(FILL_MODES)
            print(f"  > Fill Effect: {FILL_MODES[fill_idx].upper()}")

    cam.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("\n[OK] Session ended.\n")


if __name__ == "__main__":
    main()
