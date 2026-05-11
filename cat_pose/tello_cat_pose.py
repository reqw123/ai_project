import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|buffer_size;102400"
from djitellopy import Tello
from ultralytics import YOLO
import cv2
import numpy as np
import time
import threading
import queue
import keyboard  # pip install keyboard

# ==================== 基本設定 ====================
SPEED = 60          # 飛行速度 (0~100)
IMGSZ = 640
CONF_THRES = 0.50
KP_CONF_THRES = 0.5
DEVIATION_THRES = 0.60
TOTAL_KPTS = 17

# ==================== 跳幀設定 ====================
FRAME_STEP = 5

# ==================== 裁切設定 ====================
CROP_MODE = None
CROP_OPTIONS = [None, "1:1", "4:3"]
CROP_OFFSET_Y = 0.0
CROP_OFFSET_X = 0.0
CROP_OFFSET_STEP = 0.05

# ==================== 方向鍵 keycode（waitKeyEx） ====================
KEY_UP    = 2490368
KEY_DOWN  = 2621440
KEY_LEFT  = 2424832
KEY_RIGHT = 2555904

# ==================== 模型清單 ====================
MODEL_LIST = [
    r"C:\cat_pose\yolo11n.pt",
    r"C:\cat_pose\v11s_10.pt",
    r"C:\cat_pose\v11s_15.pt"
]

# ==================== 截圖存放資料夾（FIX #8：指定明確路徑）====================
SNAPSHOT_DIR = r"C:\cat_pose\snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ==================== 顏色 ====================
GREEN = (0, 255, 0)
RED   = (0, 0, 255)
BLUE  = (255, 0, 0)

COLOR_HEAD = (255, 255, 0)
COLOR_BODY = (0, 255, 0)
COLOR_TAIL = (255, 0, 255)
COLOR_KPT  = (0, 0, 255)

COLOR_LEFT_FRONT  = (255, 0, 255)
COLOR_RIGHT_FRONT = (0, 255, 255)
COLOR_LEFT_HIND   = (255, 165, 0)
COLOR_RIGHT_HIND  = (0, 255, 0)

# ==================== 骨架連結 ====================
HEAD_LINKS  = [(0,1),(0,2),(1,2)]
BODY_LINKS  = [(0,3),(3,4),(4,5)]
TAIL_LINKS  = [(5,14),(14,15),(15,16)]

# ==================== 文字繪製（白字+黑邊） ====================
def draw_text(frame, text, x, y, scale=0.7, thickness=2):
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness, cv2.LINE_AA)

# ==================== 視覺化 ====================
def draw_vertex(frame, x, y, idx, conf, size=5):
    if idx in [6, 7]:
        point_color = COLOR_LEFT_FRONT
    elif idx in [8, 9]:
        point_color = COLOR_RIGHT_FRONT
    elif idx in [10, 11]:
        point_color = COLOR_LEFT_HIND
    elif idx in [12, 13]:
        point_color = COLOR_RIGHT_HIND
    else:
        point_color = COLOR_KPT

    cv2.line(frame, (x-size, y), (x+size, y), point_color, 2)
    cv2.line(frame, (x, y-size), (x, y+size), point_color, 2)

    cv2.putText(frame, f"{idx}", (x+6, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, point_color, 2)

    text = f"{conf:.2f}"
    cv2.putText(frame, text, (x+6, y+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
    cv2.putText(frame, text, (x+6, y+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1)

def draw_links(frame, kpts, conf, links, color):
    for a, b in links:
        if conf[a] > KP_CONF_THRES and conf[b] > KP_CONF_THRES:
            cv2.line(frame,
                     tuple(kpts[a].astype(int)),
                     tuple(kpts[b].astype(int)),
                     color, 2)

def draw_skeleton(frame, kpts, conf):
    draw_links(frame, kpts, conf, HEAD_LINKS, COLOR_HEAD)
    draw_links(frame, kpts, conf, BODY_LINKS, COLOR_BODY)
    draw_links(frame, kpts, conf, [(3,6), (6,7)], COLOR_LEFT_FRONT)
    draw_links(frame, kpts, conf, [(3,8), (8,9)], COLOR_RIGHT_FRONT)
    draw_links(frame, kpts, conf, [(5,10), (10,11)], COLOR_LEFT_HIND)
    draw_links(frame, kpts, conf, [(5,12), (12,13)], COLOR_RIGHT_HIND)
    draw_links(frame, kpts, conf, TAIL_LINKS, COLOR_TAIL)

    for i, (x, y) in enumerate(kpts):
        if conf[i] > KP_CONF_THRES:
            draw_vertex(frame, int(x), int(y), i, conf[i])
    return frame

# ==================== 姿態異常 ====================
def pose_deviation(prev_kpts, curr_kpts, kpt_conf):
    if prev_kpts is None:
        return False, None

    body_scale = np.linalg.norm(curr_kpts[3] - curr_kpts[5])
    if body_scale < 1e-3:
        return False, None

    diffs = np.linalg.norm(curr_kpts - prev_kpts, axis=1)
    valid = kpt_conf > KP_CONF_THRES

    if not np.any(valid):
        return False, None

    norm_diffs = diffs[valid] / body_scale
    abnormal = np.max(norm_diffs) > DEVIATION_THRES
    idx = np.where(valid)[0][np.argmax(norm_diffs)]
    return abnormal, idx

def draw_status_light(frame, abnormal):
    h, w = frame.shape[:2]
    color = RED if abnormal else GREEN
    text = "NO" if abnormal else "OK"
    cx, cy = w - 25, 25
    cv2.circle(frame, (cx, cy), 10, color, -1)
    draw_text(frame, text, cx - 70, cy + 8, scale=0.8, thickness=2)

# ==================== H.264 損壞幀偵測（FIX #6：加入連續幀確認）====================
_corrupt_count = 0          # 連續損壞幀計數
CORRUPT_CONFIRM = 2         # 需連續 N 幀才判定為真正損壞

def is_frame_corrupted(frame, stripe_thresh=60.0):
    """
    偵測 H.264 UDP 丟包造成的水平色條 artifacts。
    雙重判斷：
      1. 下半幀 std < 5  → 畫面幾乎單色（macroblock 全填黑/灰）
      2. 相鄰列均值差 > stripe_thresh → 水平彩色條紋
    FIX #6：兩個條件必須同時滿足，降低快速運動場景的誤判率。
    """
    h, w = frame.shape[:2]
    bottom = frame[h // 2:, :, :].astype(np.float32)
    std_val = float(np.std(bottom))
    row_means = bottom.mean(axis=(1, 2))
    row_diffs = float(np.abs(np.diff(row_means)).max())

    # 嚴重單色損壞（單獨判斷即可）
    if std_val < 5:
        return True
    # 條紋損壞：需同時滿足條紋 + 整體標準差偏低（排除正常快速運動）
    if row_diffs > stripe_thresh and std_val < 30:
        return True
    return False

# ==================== 中央裁切（含偏移）====================
def crop_center(frame, mode, offset_y=0.0, offset_x=0.0):
    if mode is None:
        return frame, 0, 0
    h, w = frame.shape[:2]

    if mode == "1:1":
        crop_w, crop_h = min(w, h), min(w, h)
    elif mode == "4:3":
        target_w = int(h * 4 / 3)
        if target_w <= w:
            crop_w, crop_h = target_w, h
        else:
            crop_w, crop_h = w, int(w * 3 / 4)
    else:
        return frame, 0, 0

    cx = w // 2 + int(offset_x * w)
    cy = h // 2 + int(offset_y * h)
    x1 = max(0, min(cx - crop_w // 2, w - crop_w))
    y1 = max(0, min(cy - crop_h // 2, h - crop_h))
    return frame[y1:y1 + crop_h, x1:x1 + crop_w], x1, y1

# ==================== 顯示視窗縮放比例 ====================
window_scale = 1.0
SCALE_MIN = 0.2
SCALE_MAX = 2.0
SCALE_STEP = 0.1
# FIX #4：YOLO 推論固定使用此最大解析度，與 window_scale 無關
MAX_W, MAX_H = 640, 640

# ==================== 模型載入 ====================
model_index = 0

# FIX #2：用 list 包裝 model，避免全域變數賦值的 race condition
# worker thread 透過 model_holder[0] 存取，切換時持 _model_lock 再替換
model_holder = [YOLO(MODEL_LIST[model_index])]
model_holder[0].to("cuda")
print(f"[INFO] Loaded model: {MODEL_LIST[model_index]}")

# ==================== Tello 連線 ====================
tello = Tello()
tello.connect()
print(f"[INFO] Battery: {tello.get_battery()}%")
tello.set_video_resolution(Tello.RESOLUTION_480P)
tello.set_video_fps(Tello.FPS_15)
tello.set_video_bitrate(Tello.BITRATE_4MBPS)
tello.streamon()
time.sleep(3)

frame_read = None
for _attempt in range(5):
    try:
        frame_read = tello.get_frame_read(with_queue=True, max_queue_len=1)
        print("[INFO] Tello stream started")
        break
    except Exception as e:
        print(f"[WARN] get_frame_read attempt {_attempt+1} failed: {e}, retrying...")
        time.sleep(3)
if frame_read is None:
    print("[ERROR] Cannot open video stream. Check Tello WiFi connection.")
    try:
        tello.streamoff()
    except Exception:
        pass
    try:
        tello.end()
    except Exception:
        pass
    exit(1)

prev_kpts     = None
# FIX #7：分別記錄「正常幀」的時間，避免損壞幀跳過後 FPS 暴跌
prev_time     = time.time()
_frame_debug_printed = False

# FIX #3 + FIX #1：
#   _last_good_frame 改為存「已繪製骨架的完整顯示幀」
#   drain buffer 改用 _drain_corrupt_frames() 實際讀取並丟棄
_last_display_frame = None   # 最後一張完整繪製完成的顯示幀

in_air = False

# ==================== FIX #3：drain corrupt frames ====================
def _drain_corrupt_frames(n=3):
    """
    真正消耗 frame_read 的最新幀 n 次，
    讓 djitellopy 背景 thread 有機會填入新的解碼幀。
    """
    for _ in range(n):
        _ = frame_read.frame   # 觸發屬性讀取（背景 thread 會持續更新）
        time.sleep(0.02)       # 等待背景 thread 更新下一幀（~15fps → 67ms/幀）

# ==================== YOLO Worker Thread ====================
# FIX #4：YOLO queue 只接收「縮放到 MAX_W/MAX_H、但未套用 window_scale」的幀
# FIX #2：使用 model_holder[0] 存取模型，切換時持鎖替換 list 元素

_yolo_queue       = queue.Queue(maxsize=1)
_yolo_result      = None
_yolo_result_lock = threading.Lock()
_yolo_stop        = threading.Event()
_model_lock       = threading.Lock()

def _yolo_worker():
    global _yolo_result
    local_count = 0
    while not _yolo_stop.is_set():
        try:
            frame = _yolo_queue.get(block=True, timeout=0.1)
        except queue.Empty:
            continue
        local_count += 1
        if local_count % FRAME_STEP != 0:
            continue
        try:
            with _model_lock:
                # FIX #2：透過 model_holder[0] 存取，確保讀到最新模型
                res = model_holder[0].predict(frame, imgsz=IMGSZ, conf=CONF_THRES,
                                              half=True, verbose=False)[0]
            with _yolo_result_lock:
                _yolo_result = res
        except Exception as e:
            print(f"[WARN] YOLO error: {e}")

_yolo_thread = threading.Thread(target=_yolo_worker, daemon=True)
_yolo_thread.start()
print("[INFO] YOLO worker thread started")

# ==================== RC Control Thread (20Hz) ====================
_rc_lock  = threading.Lock()
_rc_vals  = [0, 0, 0, 0]
_rc_stop  = threading.Event()

def _rc_worker():
    interval = 1 / 20
    while not _rc_stop.is_set():
        t0 = time.time()
        if in_air:
            with _rc_lock:
                lr_, fb_, ud_, yaw_ = _rc_vals
            try:
                tello.send_rc_control(lr_, fb_, ud_, yaw_)
            except Exception:
                pass
        elapsed = time.time() - t0
        sleep_t = interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

_rc_thread = threading.Thread(target=_rc_worker, daemon=True)
_rc_thread.start()
print("[INFO] RC control thread started (20Hz)")

battery = tello.get_battery()
battery_update_time = time.time()
BATTERY_INTERVAL = 3.0

print("\n================ HOTKEYS ================")
print("--- 飛行控制 ---")
print("T     = 起飛")
print("L     = 降落")
print("W/S   = 前進 / 後退")
print("A/D   = 左平移 / 右平移")
print("R/F   = 上升 / 下降")
print("Q/E   = 左旋轉 / 右旋轉")
print("--- 視覺設定 ---")
print("Model: 1 / 2 / 3")
print("SPACE = Snapshot (save current frame)")
print("P     = Cycle crop mode (OFF / 1:1 / 4:3)")
print("↑↓←→ = Move crop window (when crop ON)")
print("G     = Reset crop offset to center")
print("+ / - = Zoom in/out window")
print("ESC   = Quit (land if flying)")
print("=========================================\n")

while True:
    # ===== 從 Tello 取得畫面 =====
    frame = frame_read.frame
    if frame is None:
        time.sleep(0.01)
        continue
    frame = frame.copy()

    if frame.shape[0] < 100 or frame.shape[1] < 100:
        time.sleep(0.01)
        continue

    if not _frame_debug_printed:
        print(f"[DEBUG] frame.shape={frame.shape}, dtype={frame.dtype}")
        _frame_debug_printed = True

    if frame.ndim == 3 and frame.shape[2] == 4:
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # ===== H.264 損壞幀偵測（FIX #6：改用雙條件判斷）=====
    _corrupted = is_frame_corrupted(display_frame)

    # FIX #6：連續幀確認，單幀誤判不觸發
    if _corrupted:
        _corrupt_count += 1
    else:
        _corrupt_count = 0

    if _corrupt_count >= CORRUPT_CONFIRM:
        # FIX #3：改用真正能推進 buffer 的 drain 函式
        _drain_corrupt_frames(3)
        _corrupt_count = 0  # 重置計數

        # FIX #1：顯示最後一張「已完整繪製」的顯示幀
        if _last_display_frame is not None:
            cv2.imshow("Tello Cat Pose", _last_display_frame)
            cv2.waitKeyEx(1)
        continue

    # ===== 正常幀處理 =====

    # FIX #7：只在正常幀計算 FPS，損壞幀跳過不影響計時
    now = time.time()
    fps = 1 / max(now - prev_time, 1e-6)
    prev_time = now

    # FIX #4：先縮放到 MAX_W/MAX_H（YOLO 用），再另外套 window_scale（顯示用）
    h0, w0 = display_frame.shape[:2]
    auto_scale = min(MAX_W / w0, MAX_H / h0, 1.0)
    if auto_scale < 1.0:
        yolo_frame = cv2.resize(display_frame,
                                (int(w0 * auto_scale), int(h0 * auto_scale)),
                                interpolation=cv2.INTER_AREA)
    else:
        yolo_frame = display_frame.copy()

    # 裁切（YOLO 用，不含 window_scale）
    if CROP_MODE is not None:
        yolo_frame, _, _ = crop_center(yolo_frame, CROP_MODE,
                                       CROP_OFFSET_Y, CROP_OFFSET_X)

    # 送 YOLO（不含 window_scale 的乾淨幀）
    try:
        _yolo_queue.put_nowait(yolo_frame.copy())
    except queue.Full:
        pass

    # 顯示幀：在 yolo_frame 基礎上再套 window_scale
    if abs(window_scale - 1.0) > 1e-3:
        h1, w1 = yolo_frame.shape[:2]
        display_frame = cv2.resize(yolo_frame,
                                   (int(w1 * window_scale), int(h1 * window_scale)),
                                   interpolation=cv2.INTER_AREA)
    else:
        display_frame = yolo_frame.copy()

    # ===== 從 YOLO worker thread 取得最新結果 =====
    with _yolo_result_lock:
        result = _yolo_result

    active_kpts = 0
    abnormal = False

    if result is not None:

        # ========== BBox ==========
        # FIX #4：YOLO 座標基於 yolo_frame（未含 window_scale），
        # 顯示時需乘以 window_scale 對齊畫面
        ws = window_scale

        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = (int(x1*ws), int(y1*ws),
                                   int(x2*ws), int(y2*ws))
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model_holder[0].names.get(cls_id, f"id_{cls_id}")
                label = f"{cls_name} {conf:.2f}"

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), BLUE, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), BLUE, -1)
                cv2.putText(display_frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ========== Pose ==========
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            kpts_raw = result.keypoints.xy[0].cpu().numpy()
            kpt_conf = result.keypoints.conf[0].cpu().numpy()

            # FIX #4：keypoint 座標乘以 window_scale，對齊顯示幀
            kpts = kpts_raw * ws

            active_kpts = int(np.sum(kpt_conf > KP_CONF_THRES))
            abnormal, _ = pose_deviation(prev_kpts, kpts_raw, kpt_conf)
            prev_kpts = kpts_raw.copy()   # 以原始座標比較，不受 window_scale 影響

            display_frame = draw_skeleton(display_frame, kpts, kpt_conf)
            draw_status_light(display_frame, abnormal)

    # ========== 更新電量 ==========
    now_t = time.time()
    if now_t - battery_update_time >= BATTERY_INTERVAL:
        try:
            battery = tello.get_battery()
        except Exception:
            pass
        battery_update_time = now_t

    if battery >= 50:
        bat_color = (0, 200, 0)
    elif battery >= 20:
        bat_color = (0, 165, 255)
    else:
        bat_color = (0, 0, 255)

    # ========== 顯示資訊 ==========
    info_lines = [
        f"Tello Cat Pose | M:{model_index+1}/{len(MODEL_LIST)}",
        f"FPS:{fps:.1f} | KPts:{active_kpts}/{TOTAL_KPTS} | Crop:{CROP_MODE or 'OFF'} ({CROP_OFFSET_X:+.2f},{CROP_OFFSET_Y:+.2f})"
    ]

    y = 20
    for line in info_lines:
        draw_text(display_frame, line, 15, y, scale=0.5, thickness=1)
        y += 18

    h_f, w_f = display_frame.shape[:2]
    bat_text = f"BAT: {battery}%"
    (btw, bth), _ = cv2.getTextSize(bat_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    bx, by = w_f - btw - 14, bth + 8
    cv2.rectangle(display_frame, (bx - 6, by - bth - 4), (bx + btw + 6, by + 4), (0, 0, 0), -1)
    cv2.rectangle(display_frame, (bx - 6, by - bth - 4), (bx + btw + 6, by + 4), bat_color, 2)
    cv2.putText(display_frame, bat_text, (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, bat_color, 2, cv2.LINE_AA)

    cv2.imshow("Tello Cat Pose", display_frame)

    # FIX #1：在顯示後才存 _last_display_frame（含骨架、BBox、UI 全部繪製完成）
    _last_display_frame = display_frame.copy()

    # ====== RC 控制 ======
    new_lr = new_fb = new_ud = new_yaw = 0
    if keyboard.is_pressed('w'):   new_fb =  SPEED
    if keyboard.is_pressed('s'):   new_fb = -SPEED
    if keyboard.is_pressed('a'):   new_lr = -SPEED
    if keyboard.is_pressed('d'):   new_lr =  SPEED
    if keyboard.is_pressed('r'):   new_ud =  SPEED
    if keyboard.is_pressed('f'):   new_ud = -SPEED
    if keyboard.is_pressed('q'):   new_yaw = -SPEED
    if keyboard.is_pressed('e'):   new_yaw =  SPEED
    with _rc_lock:
        _rc_vals[:] = [new_lr, new_fb, new_ud, new_yaw]

    raw_key = cv2.waitKeyEx(1)
    key = raw_key & 0xFF

    # ====== 起飛 / 降落 ======
    if key == ord('t'):
        if not in_air:
            try:
                tello.takeoff()
                in_air = True
                print('[INFO] Takeoff OK')
            except Exception as e:
                print(f'[WARN] Takeoff failed: {e}')

    if key == ord('l'):
        if in_air:
            try:
                tello.land()
                in_air = False
                print('[INFO] Land OK')
            except Exception as e:
                print(f'[WARN] Land failed: {e}')

    # ====== Quit ======
    if key == 27:  # ESC
        break

    # ====== 裁切模式 ======
    if key == ord("p"):
        crop_idx = CROP_OPTIONS.index(CROP_MODE)
        CROP_MODE = CROP_OPTIONS[(crop_idx + 1) % len(CROP_OPTIONS)]
        prev_kpts = None
        print(f"[INFO] Crop mode: {CROP_MODE or 'OFF'}")

    # ====== 裁切偏移 ======
    if raw_key == KEY_UP:
        CROP_OFFSET_Y = max(-0.45, CROP_OFFSET_Y - CROP_OFFSET_STEP)
    if raw_key == KEY_DOWN:
        CROP_OFFSET_Y = min(0.45, CROP_OFFSET_Y + CROP_OFFSET_STEP)
    if raw_key == KEY_LEFT:
        CROP_OFFSET_X = max(-0.45, CROP_OFFSET_X - CROP_OFFSET_STEP)
    if raw_key == KEY_RIGHT:
        CROP_OFFSET_X = min(0.45, CROP_OFFSET_X + CROP_OFFSET_STEP)
    if key == ord("g"):
        CROP_OFFSET_Y = 0.0
        CROP_OFFSET_X = 0.0
        print("[INFO] Crop offset reset to center")

    # ====== 放大/縮小視窗 ======
    if key in (ord("+"), ord("=")):
        window_scale = min(SCALE_MAX, window_scale + SCALE_STEP)
        print(f"[INFO] Window scale: {window_scale:.2f}")
    if key == ord("-"):
        window_scale = max(SCALE_MIN, window_scale - SCALE_STEP)
        print(f"[INFO] Window scale: {window_scale:.2f}")

    # ====== 截圖（FIX #8：存到指定資料夾）======
    if key == 32:  # SPACE
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(SNAPSHOT_DIR, f"tello_snapshot_{ts}.jpg")
        cv2.imwrite(fname, display_frame)
        print(f"[INFO] Snapshot saved: {fname}")

    # ====== 切換模型（FIX #2：改為替換 model_holder[0]）======
    if key in (ord("1"), ord("2"), ord("3")):
        new_index = int(chr(key)) - 1
        if new_index < len(MODEL_LIST):
            model_index = new_index
            with _model_lock:
                # FIX #2：替換 list 元素而非全域變數，避免 race condition
                new_model = YOLO(MODEL_LIST[model_index])
                new_model.to("cuda")
                model_holder[0] = new_model
            prev_kpts = None
            print(f"[INFO] Loaded model: {MODEL_LIST[model_index]}")

# ====== 結束 ======
_yolo_stop.set()
_rc_stop.set()
_yolo_thread.join(timeout=2)
_rc_thread.join(timeout=2)
if in_air:
    try:
        tello.land()
    except Exception as e:
        print(f"[WARN] Land on exit failed: {e}")
try:
    tello.streamoff()
except Exception:
    pass
try:
    tello.end()
except Exception:
    pass
cv2.destroyAllWindows()
print("[INFO] Tello disconnected. Bye!")