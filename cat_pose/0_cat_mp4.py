from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

# ==================== 基本設定 ====================
IMGSZ = 640
CONF_THRES = 0.50
KP_CONF_THRES = 0
DEVIATION_THRES = 0.60
TOTAL_KPTS = 17

# ==================== 跳幀設定 ====================
FRAME_STEP = 5   # A / D 每次跳幾幀（可用 Z/X 動態調整）

# ==================== 模型時基對齊（避免高 fps 影片時間尺度不一致） ====================
TARGET_MODEL_FPS = 30.0

# ==================== 快速預覽設定 ====================
# True: 先在 display 畫完 overlay，再縮放作為預覽（避免重畫）
# False: 先縮放再推論與繪圖（舊行為）
FAST_PREVIEW_OVERLAY = True

# ==================== 裁切設定 ====================
# None  = 不裁切（原始比例直接推論）
# "1:1" = 中央裁成正方形（強烈建議用於 9:16 直式影片）
# "4:3" = 中央裁成 4:3（適合一般橫式但想去除側邊黑邊）
CROP_MODE = None          # ← 預設關閉，按 P 可即時切換
CROP_OPTIONS = [None, "1:1", "4:3"]
# 0.0 = 畫面正中央；負值=往上/左移；正值=往下/右移
CROP_OFFSET_Y = 0.0   # ↑ 上移，↓ 下移
CROP_OFFSET_X = 0.0   # ← 左移，→ 右移
CROP_OFFSET_STEP = 0.05   # 每按一次移動 5%

# ==================== 方向鍵 keycode（waitKeyEx） ====================
KEY_UP    = 2490368
KEY_DOWN  = 2621440
KEY_LEFT  = 2424832
KEY_RIGHT = 2555904

# ==================== 模型清單（3個） ====================
MODEL_LIST = [
    r"C:\cat_pose\tiktok.pt",
    r"C:\cat_pose\v11s_66.pt",
    r"C:\cat_pose\worker4.pt"
]

# ==================== 影片清單（可放「影片檔」或「資料夾」） ====================
# - 放影片檔：直接加入播放清單
# - 放資料夾：自動掃描資料夾內所有影片
VIDEO_SOURCES = [
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\lick",  # 6
    r"C:\Users\homec\OneDrive\圖片\貓咪\2月18日.mp4",  # 7
    r"C:\Users\homec\OneDrive\圖片\貓咪\2月18日(1).mp4",  # 8
    r"C:\Users\homec\OneDrive\圖片\貓咪\VID20251128115657.mp4",  # 9
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\walk\walk_1.mp4",  # 0
    r"C:\Users\homec\OneDrive\圖片\貓咪\VID20260218194049.mp4",  # e
    r"C:\Users\homec\OneDrive\圖片\貓咪\VID20251201155614.mp4",  # f
    r"C:\Users\homec\OneDrive\圖片\貓咪\VID20251201155344.mp4",  # c
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\摳圖影片集\0_Cat_Feline_1920x1080 (1).mp4",  # v
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\摳圖影片集\0_Cat_Metronome_1920x1080.mp4",  # b
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\shake\shake_1.mp4",  # n
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\shake\shake_2.mp4",  # m
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\shake\shake_3.mp4",  # j
    r"C:\cat_pose\模型測試影片\0_Cat_Ginger_Cat_1280x720.mp4",  # k
    r"C:\Users\homec\Videos\Captures\(305) 橘猫的4脚被贴上胶带，瞬间开启震动模式，走路都像踩滑板鞋！ - YouTube - Google Chrome 2026-02-25 01-20-25.mp4",  # l
]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv", ".webm"}
VIDEO_SCAN_RECURSIVE = False  # True 會掃描子資料夾


def build_video_list(sources, recursive=False):
    """把來源路徑展開成最終影片清單：支援單檔與資料夾。"""
    videos = []
    seen = set()

    def add_video(path):
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized in seen:
            return
        seen.add(normalized)
        videos.append(path)

    for src in sources:
        if os.path.isfile(src):
            ext = os.path.splitext(src)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                add_video(src)
            else:
                print(f"[WARN] Skip non-video file: {src}")
            continue

        if os.path.isdir(src):
            if recursive:
                for root, _, files in os.walk(src):
                    for name in sorted(files):
                        full = os.path.join(root, name)
                        ext = os.path.splitext(full)[1].lower()
                        if ext in VIDEO_EXTENSIONS:
                            add_video(full)
            else:
                for name in sorted(os.listdir(src)):
                    full = os.path.join(src, name)
                    if not os.path.isfile(full):
                        continue
                    ext = os.path.splitext(full)[1].lower()
                    if ext in VIDEO_EXTENSIONS:
                        add_video(full)
            continue

        print(f"[WARN] Source not found: {src}")

    return videos


VIDEO_LIST = build_video_list(VIDEO_SOURCES, recursive=VIDEO_SCAN_RECURSIVE)
if not VIDEO_LIST:
    raise RuntimeError("No valid videos found. Please check VIDEO_SOURCES.")

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
COLOR_HIP_TAIL    = (0, 165, 255)   # hip(5) → tail-base(14)，與尾巴其餘段區隔

# True = 顯示關鍵點旁的索引與信心值；False = 隱藏文字標籤
SHOW_KPT_LABELS = True

# ==================== 骨架連結 ====================
HEAD_LINKS  = [(0,1),(0,2),(1,2)]
BODY_LINKS  = [(0,3),(3,4),(4,5)]
TAIL_LINKS  = [(5,14),(14,15),(15,16)]

# ==================== 文字繪製（白字+黑邊） ====================
def draw_text(frame, text, x, y, scale=0.7, thickness=2):
    # 黑色描邊
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    # 白色主字
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

    if SHOW_KPT_LABELS:
        # index
        cv2.putText(frame, f"{idx}", (x+6, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    point_color, 2)

        # confidence (黑邊 + 紅字)
        text = f"{conf:.2f}"
        cv2.putText(frame, text, (x+6, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 0, 0), 3)
        cv2.putText(frame, text, (x+6, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 0, 255), 1)

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

    draw_links(frame, kpts, conf, [(5, 14)],       COLOR_HIP_TAIL)  # hip → tail-base
    draw_links(frame, kpts, conf, [(14,15),(15,16)], COLOR_TAIL)     # tail-base → tip

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

# ==================== 模型載入 ====================
def load_model(path):
    m = YOLO(path)
    m.to("cuda")
    print(f"[INFO] Loaded model: {path}")
    return m

# ==================== 影片載入 ====================
def load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {path}")
        return None
    print(f"[INFO] Loaded video: {path}")
    return cap

def get_model_frame_step(cap):
    """依來源影片 fps 自動計算模型取樣步長，使實際時序接近 TARGET_MODEL_FPS。"""
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap is not None else 0.0
    if src_fps <= 0:
        src_fps = TARGET_MODEL_FPS
    step = 1
    if src_fps > TARGET_MODEL_FPS:
        step = max(1, int(round(src_fps / TARGET_MODEL_FPS)))
    eff_fps = src_fps / step
    return src_fps, step, eff_fps

def read_frame_with_step(cap, frame_step=1):
    """讀取一張推論幀，若 frame_step > 1 則額外跳過若干來源幀。"""
    ret, frame = cap.read()
    if not ret:
        return False, None

    # 只保留一張送模型，其餘幀用 grab 跳過以對齊模型時基
    for _ in range(max(0, frame_step - 1)):
        if not cap.grab():
            break

    return True, prescale_frame(frame)

# ==================== 跳轉指定幀 ====================
MAX_W, MAX_H = 1080, 720  # 全域最大顯示尺寸

def prescale_frame(frame):
    """讀入後立即縮放，避免高解析度影片佔用過多記憶體與推論時間"""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return frame

# ==================== 中央裁切（含偏移）====================
def crop_center(frame, mode, offset_y=0.0, offset_x=0.0):
    """將 frame 從指定中心裁成目標比例，回傳 (裁後影像, x1, y1)。
    x1/y1 是裁切區域左上角在原圖的座標，用於把推論結果對應回原圖。
    mode: None | '1:1' | '4:3'
    offset_y: -0.5（往上）~ +0.5（往下），0 = 正中央
    offset_x: -0.5（往左）~ +0.5（往右），0 = 正中央
    """
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

    # 計算中心點（含偏移）
    cx = w // 2 + int(offset_x * w)
    cy = h // 2 + int(offset_y * h)

    # 計算裁切座標並夾緊邊界
    x1 = max(0, min(cx - crop_w // 2, w - crop_w))
    y1 = max(0, min(cy - crop_h // 2, h - crop_h))

    return frame[y1:y1 + crop_h, x1:x1 + crop_w], x1, y1

def jump_to_frame(cap, target_frame):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = max(0, min(target_frame, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        return None
    return prescale_frame(frame)

# ==================== 主程式 ====================
model_index = 0

video_index = 0

# ========== 顯示視窗縮放比例 ===========
window_scale = 1.0  # 預設為1.0 (100%)
SCALE_MIN = 0.2
SCALE_MAX = 2.0
SCALE_STEP = 0.1

model = load_model(MODEL_LIST[model_index])
cap = load_video(VIDEO_LIST[video_index])
source_fps, model_frame_step, effective_input_fps = get_model_frame_step(cap)

play_mode = True   # True = 正常播放, False = 逐幀模式(暫停)
show_overlay = True  # True = 顯示推論結果（骨架/BBox）；False = 純淨畫面
show_info    = True  # True = 顯示 FPS/Frame 等資訊文字；False = 完全隱藏
prev_kpts = None
prev_time = time.time()
frame = None

print("\n================ HOTKEYS ================")

print("Model: 1 / 2 / 3")
print("Video: 6 / 7 / 8 / 9 / 0 / e / f / c / v / b / n / m / j / k / l")
print("SPACE = Play/Pause")
print("A = jump backward (STEP)  [pause mode]")
print("D = jump forward  (STEP)  [pause mode]")
print("[ / ] = Previous / Next video")
print("Z = STEP -1")
print("X = STEP +1")
print("T = jump to frame number")
print("R = restart video")
print("O = Toggle overlay (pose/bbox ON/OFF)")
print("I = Toggle info (FPS/Frame/KPts ON/OFF)")
print("U = Toggle keypoint labels (index/conf ON/OFF)")
print("Q = quit")
print("+ / - = Zoom in/out window")
print("P        = Cycle crop mode (OFF / 1:1 / 4:3)")
print("↑↓←→  = Move crop window (when crop ON)")
print("G        = Reset crop offset to center")
print(f"Timing align: source>{TARGET_MODEL_FPS:.0f}fps 時自動降採樣到 ~{TARGET_MODEL_FPS:.0f}fps")
print(f"Fast preview overlay: {'ON' if FAST_PREVIEW_OVERLAY else 'OFF'}")
print(f"Resolved videos: {len(VIDEO_LIST)}")
print("=========================================\n")

print(f"[INFO] Source FPS={source_fps:.2f}, Model frame_step={model_frame_step}, Effective input FPS={effective_input_fps:.2f}")

while True:
    if cap is None:
        break

    if play_mode:
        ret, frame = read_frame_with_step(cap, model_frame_step)
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            prev_kpts = None
            continue
    else:
        if frame is None:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = prescale_frame(frame)


    # --- 自動縮放到 1080x720 以內 ---
    h0, w0 = frame.shape[:2]
    max_w, max_h = 1080, 720
    auto_scale = min(max_w / w0, max_h / h0, 1.0)
    display_frame = frame.copy()
    if auto_scale < 1.0:
        display_frame = cv2.resize(display_frame, (int(w0 * auto_scale), int(h0 * auto_scale)), interpolation=cv2.INTER_AREA)

    # --- 舊行為：先縮放再推論（Fast preview 關閉時） ---
    if (not FAST_PREVIEW_OVERLAY) and abs(window_scale - 1.0) > 1e-3:
        h1, w1 = display_frame.shape[:2]
        new_w, new_h = int(w1 * window_scale), int(h1 * window_scale)
        display_frame = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # --- 裁切：display_frame 即為裁切後畫面，YOLO 推論與顯示使用同一張 ---
    if CROP_MODE is not None:
        display_frame, _, _ = crop_center(display_frame, CROP_MODE, CROP_OFFSET_Y, CROP_OFFSET_X)

    # ===== 推論 =====
    result = model.predict(
        display_frame,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        half=True,
        verbose=False
    )[0]

    # FPS
    now = time.time()
    fps = 1 / max(now - prev_time, 1e-6)
    prev_time = now

    active_kpts = 0
    abnormal = False

    if show_overlay:
        # ========== BBox ==========
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names.get(cls_id, f"id_{cls_id}")
                label = f"{cls_name} {conf:.2f}"

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), BLUE, 2)

                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), BLUE, -1)

                cv2.putText(display_frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ========== Pose ==========
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            kpts = result.keypoints.xy[0].cpu().numpy()
            kpt_conf = result.keypoints.conf[0].cpu().numpy()

            active_kpts = np.sum(kpt_conf > KP_CONF_THRES)
            abnormal, _ = pose_deviation(prev_kpts, kpts, kpt_conf)
            prev_kpts = kpts.copy()

            display_frame = draw_skeleton(display_frame, kpts, kpt_conf)
            if show_info:
                draw_status_light(display_frame, abnormal)

    # ========== 顯示資訊 ==========
    mode_text = "PLAY" if play_mode else "FRAME"
    current_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 合併資訊到更少的行，使用更小的字體和間距
    if show_info:
        info_lines = [
            f"{mode_text} | M:{model_index+1}/{len(MODEL_LIST)} | V:{video_index+1}/{len(VIDEO_LIST)}",
            f"Frame:{current_frame_id}/{total_frames} | Step:{FRAME_STEP} | FStep:{model_frame_step}",
            f"FPS:{fps:.1f} | KPts:{active_kpts}/{TOTAL_KPTS} | Crop:{CROP_MODE or 'OFF'} ({CROP_OFFSET_X:+.2f},{CROP_OFFSET_Y:+.2f})"
        ]
        info_lines.append(f"SrcFPS:{source_fps:.1f} -> InFPS:{effective_input_fps:.1f} | FastPreview:{'ON' if FAST_PREVIEW_OVERLAY else 'OFF'}")
        y = 20
        for line in info_lines:
            draw_text(display_frame, line, 15, y, scale=0.5, thickness=1)
            y += 18

    preview_frame = display_frame
    if FAST_PREVIEW_OVERLAY and abs(window_scale - 1.0) > 1e-3:
        h1, w1 = preview_frame.shape[:2]
        new_w, new_h = int(w1 * window_scale), int(h1 * window_scale)
        preview_frame = cv2.resize(preview_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imshow("Cat Pose Pro", preview_frame)

    delay = 1 if play_mode else 0
    raw_key = cv2.waitKeyEx(delay)
    key = raw_key & 0xFF


    # ====== Toggle overlay（O）骨架/BBox ======
    if key == ord("o"):
        show_overlay = not show_overlay
        print(f"[INFO] Overlay: {'ON' if show_overlay else 'OFF'}")

    # ====== Toggle info（I）FPS/Frame 資訊文字 ======
    if key == ord("i"):
        show_info = not show_info
        print(f"[INFO] Info display: {'ON' if show_info else 'OFF'}")

    # ====== Toggle keypoint labels（U）索引/信心值 ======
    if key == ord("u"):
        SHOW_KPT_LABELS = not SHOW_KPT_LABELS
        print(f"[INFO] Keypoint labels: {'ON' if SHOW_KPT_LABELS else 'OFF'}")

    # ====== Quit ======
    if key == ord("q"):
        break

    # ====== 切換裁切模式（P）======
    if key == ord("p"):
        crop_idx = CROP_OPTIONS.index(CROP_MODE)
        CROP_MODE = CROP_OPTIONS[(crop_idx + 1) % len(CROP_OPTIONS)]
        prev_kpts = None
        print(f"[INFO] Crop mode: {CROP_MODE or 'OFF'}")

    # ====== 裁切偏移（方向鍵，G = 重置）======
    if raw_key == KEY_UP:
        CROP_OFFSET_Y = max(-0.45, CROP_OFFSET_Y - CROP_OFFSET_STEP)
        print(f"[INFO] Crop offset Y: {CROP_OFFSET_Y:+.2f}  (往上移)")
    if raw_key == KEY_DOWN:
        CROP_OFFSET_Y = min(0.45, CROP_OFFSET_Y + CROP_OFFSET_STEP)
        print(f"[INFO] Crop offset Y: {CROP_OFFSET_Y:+.2f}  (往下移)")
    if raw_key == KEY_LEFT:
        CROP_OFFSET_X = max(-0.45, CROP_OFFSET_X - CROP_OFFSET_STEP)
        print(f"[INFO] Crop offset X: {CROP_OFFSET_X:+.2f}  (往左移)")
    if raw_key == KEY_RIGHT:
        CROP_OFFSET_X = min(0.45, CROP_OFFSET_X + CROP_OFFSET_STEP)
        print(f"[INFO] Crop offset X: {CROP_OFFSET_X:+.2f}  (往右移)")
    if key == ord("g"):
        CROP_OFFSET_Y = 0.0
        CROP_OFFSET_X = 0.0
        print("[INFO] Crop offset reset to center")

    # ====== 放大/縮小視窗 ======
    if key == ord("+") or key == ord("="):
        window_scale = min(SCALE_MAX, window_scale + SCALE_STEP)
        print(f"[INFO] Window scale: {window_scale:.2f}")
    if key == ord("-"):
        window_scale = max(SCALE_MIN, window_scale - SCALE_STEP)
        print(f"[INFO] Window scale: {window_scale:.2f}")

    # ====== 播放/逐幀模式切換 ======
    if key == 32:  # SPACE
        play_mode = not play_mode
        prev_kpts = None
        print(f"[INFO] MODE switched to: {'PLAY' if play_mode else 'FRAME'}")

    # ====== STEP 調整 ======
    if key == ord("z"):
        FRAME_STEP = max(1, FRAME_STEP - 1)
        print(f"[INFO] STEP = {FRAME_STEP}")

    if key == ord("x"):
        FRAME_STEP = min(200, FRAME_STEP + 1)
        print(f"[INFO] STEP = {FRAME_STEP}")

    # ====== 跳轉指定 Frame ======
    if key == ord("t"):
        play_mode = False
        try:
            user_input = input("\n[INPUT] Jump to frame number: ")
            target = int(user_input)
            frame = jump_to_frame(cap, target)
            prev_kpts = None
            print(f"[INFO] Jumped to frame: {target}")
        except:
            print("[ERROR] Invalid frame number")

    # ====== A / D 快速跳幀（逐幀模式才有效）======
    if not play_mode:
        if key == ord("a"):
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # 目前顯示幀（read後pos+1，需-1）
            target = pos - FRAME_STEP
            frame = jump_to_frame(cap, target)
            prev_kpts = None

        elif key == ord("d"):
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # 目前顯示幀（read後pos+1，需-1）
            target = pos + FRAME_STEP
            frame = jump_to_frame(cap, target)
            prev_kpts = None

    # ====== 重新播放影片 ======
    if key == ord("r"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_kpts = None
        frame = None
        print("[INFO] Restart video")

    # ====== 切換模型（1~3）======
    if key in [ord("1"), ord("2"), ord("3")]:
        new_index = int(chr(key)) - 1
        if new_index < len(MODEL_LIST):
            model_index = new_index
            model = load_model(MODEL_LIST[model_index])
            prev_kpts = None

    # ====== 切換影片（快速鍵對應清單）======
    video_key_map = {
        ord("6"): 0,
        ord("7"): 1,
        ord("8"): 2,
        ord("9"): 3,
        ord("0"): 4,
        ord("e"): 5,
        ord("f"): 6,
        ord("c"): 7,
        ord("v"): 8,
        ord("b"): 9,
        ord("n"): 10,
        ord("m"): 11,
        ord("j"): 12,
        ord("k"): 13,
        ord("l"): 14
    }
    if key in video_key_map:
        video_index = video_key_map[key]
        cap.release()
        cap = load_video(VIDEO_LIST[video_index])
        source_fps, model_frame_step, effective_input_fps = get_model_frame_step(cap)
        prev_kpts = None
        frame = None
        print(f"[INFO] Source FPS={source_fps:.2f}, Model frame_step={model_frame_step}, Effective input FPS={effective_input_fps:.2f}")
        print(f"[INFO] Switched video to: {VIDEO_LIST[video_index]}")

    # ====== 切換上一部 / 下一部影片（支援任意清單長度）======
    if key in [ord("["), ord("{")]:
        if len(VIDEO_LIST) > 0:
            video_index = (video_index - 1) % len(VIDEO_LIST)
            cap.release()
            cap = load_video(VIDEO_LIST[video_index])
            source_fps, model_frame_step, effective_input_fps = get_model_frame_step(cap)
            prev_kpts = None
            frame = None
            print(f"[INFO] Source FPS={source_fps:.2f}, Model frame_step={model_frame_step}, Effective input FPS={effective_input_fps:.2f}")
            print(f"[INFO] Switched video to: {VIDEO_LIST[video_index]}")

    if key in [ord("]"), ord("}")]:
        if len(VIDEO_LIST) > 0:
            video_index = (video_index + 1) % len(VIDEO_LIST)
            cap.release()
            cap = load_video(VIDEO_LIST[video_index])
            source_fps, model_frame_step, effective_input_fps = get_model_frame_step(cap)
            prev_kpts = None
            frame = None
            print(f"[INFO] Source FPS={source_fps:.2f}, Model frame_step={model_frame_step}, Effective input FPS={effective_input_fps:.2f}")
            print(f"[INFO] Switched video to: {VIDEO_LIST[video_index]}")

# ====== 結束 ======
if cap is not None:
    cap.release()

cv2.destroyAllWindows()
