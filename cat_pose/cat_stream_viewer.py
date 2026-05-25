import time
from pathlib import Path
import cv2
import numpy as np
import serial

import sys
sys.path.insert(0, str(Path(__file__).parent))

from detectors.keypoint_detector import KeypointDetector

# --- Simple skeleton definition ---
_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 2),
    (0, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (3, 8), (8, 9),
    (5, 10), (10, 11), (5, 12), (12, 13),
]


def draw_skeleton(frame, kpts, kpt_conf, conf_thresh=0.35):
    """Draw skeleton lines and keypoints onto `frame`.
    `kpts` is expected as an iterable of (x,y) pixel coordinates or None.
    `kpt_conf` is an iterable of confidences parallel to kpts.
    """
    if kpts is None or kpt_conf is None:
        return

    # Ensure we can index
    try:
        kp_list = list(kpts)
        conf_list = list(kpt_conf)
    except Exception:
        return

    h, w = frame.shape[:2]
    line_w = max(1, int(round(min(w, h) / 360)))

    # draw edges
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

    # draw keypoints
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

# =========================================================
# ESP32 Serial
# =========================================================000

SERIAL_PORT = "COM5"     # ← 改成你的ESP32 COM
BAUD_RATE = 9600

ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

# =========================================================
# YOLO Config
# =========================================================

VIDEO_PATH = "http://192.168.4.1:81/stream"

YOLO_MODEL_PATH = r"C:\AI_Project\cat_pose\v11s_72.pt"

INFERENCE_DEVICE = "cuda"

YOLO_IMGSZ = 640

YOLO_CONF_THRESHOLD = 0.25

WINDOW_NAME = "Cat Tracker"

DISPLAY_SIZE = (1280, 720)

# =========================================================
# 新增區塊：Follow / tracking control
# =========================================================

CENTER_TOLERANCE = 80
TRACK_COMMAND_INTERVAL = 0.04  # seconds between repeated LEFT/RIGHT commands
STOP_REPEAT_INTERVAL = 0.25     # resend STOP occasionally so Arduino stays in hold mode
STATE_SWITCH_COOLDOWN = 0.15     # minimum gap for STOP/START state transitions

# =========================================================
# 初始化 Detector
# =========================================================

detector = KeypointDetector(
    YOLO_MODEL_PATH,
    device=INFERENCE_DEVICE,
    imgsz=YOLO_IMGSZ,
    conf_thres=YOLO_CONF_THRESHOLD
)

# =========================================================
# 開啟串流
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("Cannot Open Stream")

    exit()

# =========================================================
# 狀態控制
# =========================================================

cat_detected = False
tracking_cmd = None

last_send_time = 0

frame_count = 0

start_time = time.time()

# Debounce / stability thresholds to avoid rapid toggling
DETECT_REQUIRED = 3   # require this many consecutive detections to consider "detected"
LOST_REQUIRED = 5     # require this many consecutive non-detections to consider "lost"

detected_count = 0
lost_count = 0


def send_serial_cmd(cmd_text):
    """Send one serial command safely."""
    global last_send_time
    try:
        ser.write(cmd_text.encode("ascii") + b"\n")
        last_send_time = time.time()
    except Exception as e:
        print("Serial write failed:", e)

# =========================================================
# 主迴圈
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Cannot Read Frame")

        break

    frame_count += 1

    # =====================================================
    # YOLO Detect
    # =====================================================

    kpts, kpt_conf, bbox, bbox_conf = detector.detect(frame)

    # =====================================================
    # 是否偵測到貓（含去彈跳邏輯）
    # =====================================================
    detected_now = bbox is not None
    current_time = time.time()

    # update consecutive counters
    if detected_now:
        detected_count += 1
        lost_count = 0
    else:
        lost_count += 1
        detected_count = 0

    stable_detected = detected_count >= DETECT_REQUIRED
    stable_lost = lost_count >= LOST_REQUIRED

    # send START when lost becomes stable
    if stable_lost and cat_detected and (current_time - last_send_time) >= STATE_SWITCH_COOLDOWN:
        print("CAT LOST (stable)")
        send_serial_cmd("START")
        cat_detected = False
        tracking_cmd = None

    # send tracking commands when detection becomes stable
    if stable_detected:
        if bbox is not None:
            x1, y1, x2, y2 = map(float, bbox)
            cat_center_x = (x1 + x2) / 2.0
            frame_center_x = frame.shape[1] / 2.0
            error = cat_center_x - frame_center_x

            if error < -CENTER_TOLERANCE:
                desired_cmd = "LEFT"
            elif error > CENTER_TOLERANCE:
                desired_cmd = "RIGHT"
            else:
                desired_cmd = "STOP"

            if not cat_detected and (current_time - last_send_time) >= STATE_SWITCH_COOLDOWN:
                print("CAT DETECTED (stable)")
                cat_detected = True
                tracking_cmd = None

            if cat_detected:
                # 只有進入中心死區才停止，否則持續微調方向
                should_repeat = desired_cmd in ("LEFT", "RIGHT") and (
                    tracking_cmd != desired_cmd or (current_time - last_send_time) >= TRACK_COMMAND_INTERVAL
                )
                should_hold = desired_cmd == "STOP" and (
                    tracking_cmd != "STOP" or (current_time - last_send_time) >= STOP_REPEAT_INTERVAL
                )

                if should_repeat or should_hold:
                    send_serial_cmd(desired_cmd)
                    tracking_cmd = desired_cmd

    # =====================================================
    # 畫 Bounding Box
    # =====================================================
    # =====================================================
    # 畫 Skeleton（若有）
    # 將可能為 normalized 的關鍵點換算為像素座標再繪製
    try:
        if kpts is not None and kpt_conf is not None:
            kpts_arr = np.asarray(kpts, dtype=np.float64)
            if np.max(kpts_arr) <= 1.0:
                kpts_pix = [(float(p[0]) * frame.shape[1], float(p[1]) * frame.shape[0]) for p in kpts_arr]
            else:
                kpts_pix = [(float(p[0]), float(p[1])) for p in kpts_arr]
            draw_skeleton(frame, kpts_pix, kpt_conf, conf_thresh=0.35)
    except Exception:
        pass

    if bbox is not None:

        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "CAT",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # =====================================================
    # FPS
    # =====================================================

    elapsed = time.time() - start_time

    fps = frame_count / elapsed

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # =====================================================
    # 顯示畫面
    # =====================================================

    cv2.imshow(WINDOW_NAME, frame)

    # =====================================================
    # 離開
    # =====================================================

    key = cv2.waitKey(1)

    if key == ord('q'):

        break

# =========================================================
# Cleanup
# =========================================================

cap.release()

cv2.destroyAllWindows()

ser.close()