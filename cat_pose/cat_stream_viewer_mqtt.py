"""
ESP32-CAM + YOLO pose + SG90 MQTT tracker

Purpose:
- Open a single VIDEO_PATH (file or IP stream)
- Run YOLO pose detection and draw bbox + skeleton
- Publish LEFT / RIGHT / STOP / START commands over MQTT
- Keep the same debounce / stable-detect logic as the serial version

Dependencies:
- pip install paho-mqtt
"""

import time
from pathlib import Path

import cv2
import numpy as np
import paho.mqtt.client as mqtt

import sys
sys.path.insert(0, str(Path(__file__).parent))

from detectors.keypoint_detector import KeypointDetector

# --- Simple skeleton definition ---
_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 2),
    (0, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (3, 8), (8, 9),
    (5, 10), (10, 11), (5, 12), (12, 13),(5, 14), (14, 15), (15, 16)
]


def draw_skeleton(frame, kpts, kpt_conf, conf_thresh=0.35):
    if kpts is None or kpt_conf is None:
        return

    try:
        kp_list = list(kpts)
        conf_list = list(kpt_conf)
    except Exception:
        return

    h, w = frame.shape[:2]
    line_w = max(1, int(round(min(w, h) / 360)))

    for a, b in _SKELETON_EDGES:
        if a >= len(kp_list) or b >= len(kp_list):
            continue
        try:
            ca = float(conf_list[a])
            cb = float(conf_list[b])
        except Exception:
            continue
        if ca <= conf_thresh or cb <= conf_thresh:
            continue
        pa = kp_list[a]
        pb = kp_list[b]
        if not (np.isfinite(pa[0]) and np.isfinite(pa[1]) and np.isfinite(pb[0]) and np.isfinite(pb[1])):
            continue
        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (200, 200, 255), line_w, cv2.LINE_AA)

    r_outer = max(2, int(3 * line_w))
    r_inner = max(1, int(2 * line_w))
    for i, p in enumerate(kp_list):
        if i >= len(conf_list):
            break
        try:
            conf = float(conf_list[i])
        except Exception:
            continue
        if conf <= conf_thresh:
            continue
        if not (np.isfinite(p[0]) and np.isfinite(p[1])):
            continue
        cx, cy = int(p[0]), int(p[1])
        cv2.circle(frame, (cx, cy), r_outer, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r_inner, (120, 220, 255), -1, cv2.LINE_AA)


# ---------------------------
# Stock overlay utilities
# ---------------------------
# Enable overlay of a small stock-price sparkline on the video frame.
STOCK_OVERLAY_ENABLED = True
STOCK_BUFFER_LEN = 200
STOCK_TEST_MODE = False  # If True, synthesize prices for demo

# persistent buffer for prices (most recent last)
stock_prices = []


# ---------------------------
# Color palette derived from user mapping
# ---------------------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

COLOR_HEAD = (255, 255, 0)
COLOR_BODY = (0, 255, 0)
COLOR_LIMB = (255, 0, 0)
COLOR_TAIL = (255, 0, 255)
COLOR_HIP_TAIL = (0, 165, 255)
COLOR_KPT = (0, 0, 255)

# 四隻腳的顏色（左前、右前、左後、右後）
COLOR_LEFT_FRONT = (255, 0, 255)
COLOR_RIGHT_FRONT = (0, 255, 255)
COLOR_LEFT_HIND = (255, 165, 0)
COLOR_RIGHT_HIND = (0, 255, 0)

# Map 17 keypoints to colors (index order matches YOLO-Pose v11 mapping)
COLOR_BY_KP = [
    COLOR_HEAD,       # 0 Nose
    COLOR_HEAD,       # 1 Left_Ear
    COLOR_HEAD,       # 2 Right_Ear
    COLOR_BODY,       # 3 Chest
    COLOR_BODY,       # 4 Mid_Back
    COLOR_BODY,       # 5 Hip
    COLOR_LEFT_FRONT, # 6 LF_Elbow
    COLOR_LEFT_FRONT, # 7 LF_Paw
    COLOR_RIGHT_FRONT,# 8 RF_Elbow
    COLOR_RIGHT_FRONT,# 9 RF_Paw
    COLOR_LEFT_HIND,  # 10 LH_Knee
    COLOR_LEFT_HIND,  # 11 LH_Paw
    COLOR_RIGHT_HIND, # 12 RH_Knee
    COLOR_RIGHT_HIND, # 13 RH_Paw
    COLOR_HIP_TAIL,   # 14 Tail_Root
    COLOR_TAIL,       # 15 Tail_Mid
    COLOR_TAIL,       # 16 Tail_Tip
]


def add_stock_price(price):
    """Append a new price to the rolling buffer."""
    try:
        v = float(price)
    except Exception:
        return
    stock_prices.append(v)
    if len(stock_prices) > STOCK_BUFFER_LEN:
        del stock_prices[0:(len(stock_prices) - STOCK_BUFFER_LEN)]


def draw_stock_overlay(frame, prices, area_frac=0.22):
    """Draw a colored sparkline of `prices` at the bottom of `frame`.
    Colors: green for rising segment, red for falling. Keeps line connected.
    """
    if not STOCK_OVERLAY_ENABLED or prices is None or len(prices) < 2:
        return frame

    h, w = frame.shape[:2]
    area_h = max(24, int(round(h * area_frac)))
    top = h - area_h

    # background panel
    cv2.rectangle(frame, (0, top), (w, h), (18, 18, 26), -1)

    arr = np.asarray(prices, dtype=np.float64)
    minv = float(np.min(arr))
    maxv = float(np.max(arr))
    if maxv - minv < 1e-6:
        maxv = minv + 1.0

    # map prices to x,y coordinates across panel width
    n = len(arr)
    xs = np.linspace(4, w - 4, n)
    ys = top + area_h - 4 - (arr - minv) / (maxv - minv) * max(1, area_h - 8)

    # draw segment-by-segment with color assigned from COLOR_BY_KP mapping
    lw = max(1, int(round(min(w, h) / 360)))
    for i in range(n - 1):
        x0, y0 = int(xs[i]), int(ys[i])
        x1, y1 = int(xs[i + 1]), int(ys[i + 1])
        # map position to one of 17 keypoint color bins for consistent palette
        kp_idx = int(i * 17 / max(1, n))
        kp_idx = min(max(kp_idx, 0), 16)
        color = COLOR_BY_KP[kp_idx]
        cv2.line(frame, (x0, y0), (x1, y1), color, lw, cv2.LINE_AA)

    # draw latest value text using keypoint/color highlight
    latest = arr[-1]
    txt = f"{latest:.2f}"
    txt_w = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
    txt_x = w - 10 - txt_w
    txt_y = top + 18
    cv2.putText(frame, txt, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_KPT, 2, cv2.LINE_AA)
    return frame


# =========================================================
# MQTT Config
# =========================================================

MQTT_BROKER = "10.63.178.34"
MQTT_PORT = 1883
MQTT_KEEPALIVE = 60
MQTT_CLIENT_ID = "cat-tracker-py"
MQTT_CMD_TOPIC = "cat/servo/cmd"
MQTT_STATUS_TOPIC = "cat/servo/status"
MQTT_QOS = 0

# =========================================================
# YOLO / Video Config
# =========================================================

VIDEO_PATH = "http://10.63.178.42:81/stream"
YOLO_MODEL_PATH = r"C:\AI_Project\cat_pose\v11s_72.pt"
INFERENCE_DEVICE = "cuda"
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.5
WINDOW_NAME = "Cat Tracker MQTT"
DISPLAY_SIZE = (1280, 720)
# EMA smoothing alpha for keypoint smoothing (0<alpha<=1). Larger -> more responsive, smaller -> smoother
EMA_ALPHA = 0.6

# =========================================================
# Follow / tracking control
# =========================================================

CENTER_TOLERANCE = 80
TRACK_COMMAND_INTERVAL = 0.04
STOP_REPEAT_INTERVAL = 0.25
STATE_SWITCH_COOLDOWN = 0.15
DETECT_REQUIRED = 3
LOST_REQUIRED = 5
BBOX_EDGE_MARGIN = 12
PARTIAL_BBOX_RATIO = 0.65

# =========================================================
# Detector / MQTT setup
# =========================================================

mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID, clean_session=True)


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")
        client.publish(MQTT_STATUS_TOPIC, "PY_ONLINE", qos=MQTT_QOS, retain=False)
    else:
        print(f"MQTT connect failed rc={rc}")


mqtt_client.on_connect = on_connect
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
mqtt_client.loop_start()


def publish_cmd(cmd_text):
    try:
        mqtt_client.publish(MQTT_CMD_TOPIC, cmd_text, qos=MQTT_QOS, retain=False)
    except Exception as exc:
        print("MQTT publish failed:", exc)


def infer_tracking_cmd_from_bbox(frame_width, bbox):
    """Infer servo direction from bbox geometry.

    If the bbox touches an edge and is relatively narrow, treat it as a partial
    detection and keep moving toward the visible side. Otherwise, fall back to
    center-based tracking.
    """
    x1, y1, x2, y2 = map(float, bbox)
    bbox_width = max(1.0, x2 - x1)
    bbox_ratio = bbox_width / max(1.0, float(frame_width))

    touches_left = x1 <= BBOX_EDGE_MARGIN
    touches_right = x2 >= (frame_width - BBOX_EDGE_MARGIN)

    if bbox_ratio < PARTIAL_BBOX_RATIO:
        if touches_left and not touches_right:
            return "LEFT"
        if touches_right and not touches_left:
            return "RIGHT"

    cat_center_x = (x1 + x2) / 2.0
    frame_center_x = frame_width / 2.0
    error = cat_center_x - frame_center_x

    if error < -CENTER_TOLERANCE:
        return "LEFT"
    if error > CENTER_TOLERANCE:
        return "RIGHT"
    return "STOP"


def reset_runtime_state():
    """Reset per-stream tracking state after a video stream reconnect."""
    global cat_detected, tracking_cmd, last_send_time, detected_count, lost_count, ema_kpts
    cat_detected = False
    tracking_cmd = None
    last_send_time = 0.0
    detected_count = 0
    lost_count = 0
    ema_kpts = None


def reopen_stream():
    """Reopen the configured video stream."""
    global cap
    try:
        cap.release()
    except Exception:
        pass
    # reuse same normalized source as initial open
    try:
        src = video_source
    except NameError:
        src = VIDEO_PATH
    cap = cv2.VideoCapture(src)
    return cap.isOpened()


detector = KeypointDetector(
    YOLO_MODEL_PATH,
    device=INFERENCE_DEVICE,
    imgsz=YOLO_IMGSZ,
    conf_thres=YOLO_CONF_THRESHOLD,
)


def get_video_source(path):
    """Normalize VIDEO_PATH into an int index or a string URL.

    - If `path` is an int, return it.
    - If `path` is a numeric string like '0', return int(0).
    - Otherwise return the original string (URL/filepath).
    """
    if isinstance(path, int):
        return path
    if isinstance(path, str):
        s = path.strip()
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return path
        return path
    return path

# store normalized source so reconnects use same type (int index vs URL)
video_source = get_video_source(VIDEO_PATH)
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Cannot Open Stream")
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    raise SystemExit(1)

# Create a resizable window and set initial display size
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
try:
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])
except Exception:
    pass

# =========================================================
# State
# =========================================================

cat_detected = False
tracking_cmd = None
last_send_time = 0.0
frame_count = 0
start_time = time.time()

detected_count = 0
lost_count = 0

# EMA state for smoothed keypoints
ema_kpts = None


# =========================================================
# Main loop
# =========================================================

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot Read Frame, trying to reopen stream...")
        reset_runtime_state()
        if reopen_stream():
            print("Stream reopened")
            continue
        print("Stream reopen failed")
        break

    frame_count += 1
    kpts, kpt_conf, bbox, bbox_conf = detector.detect(frame)

    detected_now = bbox is not None
    current_time = time.time()

    if detected_now:
        detected_count += 1
        lost_count = 0
    else:
        lost_count += 1
        detected_count = 0

    stable_detected = detected_count >= DETECT_REQUIRED
    stable_lost = lost_count >= LOST_REQUIRED

    if stable_lost and cat_detected and (current_time - last_send_time) >= STATE_SWITCH_COOLDOWN:
        print("CAT LOST (stable)")
        publish_cmd("START")
        cat_detected = False
        tracking_cmd = None

    if stable_detected and bbox is not None:
        desired_cmd = infer_tracking_cmd_from_bbox(frame.shape[1], bbox)

        if not cat_detected and (current_time - last_send_time) >= STATE_SWITCH_COOLDOWN:
            print("CAT DETECTED (stable)")
            cat_detected = True
            tracking_cmd = None

        if cat_detected:
            should_repeat = desired_cmd in ("LEFT", "RIGHT") and (
                tracking_cmd != desired_cmd or (current_time - last_send_time) >= TRACK_COMMAND_INTERVAL
            )
            should_hold = desired_cmd == "STOP" and (
                tracking_cmd != "STOP" or (current_time - last_send_time) >= STOP_REPEAT_INTERVAL
            )

            if should_repeat or should_hold:
                publish_cmd(desired_cmd)
                tracking_cmd = desired_cmd
                last_send_time = current_time

    try:
        if kpts is not None and kpt_conf is not None:
            kpts_arr = np.asarray(kpts, dtype=np.float64)
            # convert to pixel coords if normalized
            if np.max(kpts_arr) <= 1.0:
                curr_kpts_pix = np.asarray([(float(p[0]) * frame.shape[1], float(p[1]) * frame.shape[0]) for p in kpts_arr], dtype=np.float64)
            else:
                curr_kpts_pix = np.asarray([(float(p[0]), float(p[1])) for p in kpts_arr], dtype=np.float64)

            # initialize or apply EMA smoothing
            if curr_kpts_pix.size > 0:
                if ema_kpts is None:
                    ema_kpts = curr_kpts_pix.copy()
                else:
                    try:
                        ema_kpts = EMA_ALPHA * curr_kpts_pix + (1.0 - EMA_ALPHA) * ema_kpts
                    except Exception:
                        ema_kpts = curr_kpts_pix.copy()

            # prepare list of tuples for drawing (use EMA-smoothed keypoints if available)
            if ema_kpts is not None:
                kpts_pix = [(float(p[0]), float(p[1])) for p in ema_kpts]
            else:
                kpts_pix = [(float(p[0]), float(p[1])) for p in curr_kpts_pix]

            draw_skeleton(frame, kpts_pix, kpt_conf, conf_thresh=0.35)
    except Exception:
        pass

    # --- stock overlay: optional test data generator and draw ---
    if STOCK_TEST_MODE:
        # synthesize a price stream for demo: smooth random walk
        if len(stock_prices) == 0:
            add_stock_price(100.0 + float(np.random.randn()) * 0.1)
        else:
            last = stock_prices[-1]
            new = last + float(np.random.randn()) * 0.3 + 0.2 * np.sin(frame_count * 0.02)
            add_stock_price(new)

    # draw overlay if enabled
    if STOCK_OVERLAY_ENABLED:
        try:
            draw_stock_overlay(frame, stock_prices)
        except Exception:
            pass

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "CAT", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0.0
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    # Resize frame for display to make stream larger on screen
    if DISPLAY_SIZE is not None:
        try:
            disp_frame = cv2.resize(frame, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
        except Exception:
            disp_frame = frame
    else:
        disp_frame = frame
    cv2.imshow(WINDOW_NAME, disp_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
mqtt_client.publish(MQTT_STATUS_TOPIC, "PY_OFFLINE", qos=MQTT_QOS, retain=False)
mqtt_client.loop_stop()
mqtt_client.disconnect()
