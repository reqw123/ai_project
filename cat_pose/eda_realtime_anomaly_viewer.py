"""
Cat Pose Analysis - Fast English Version
High FPS optimized with trigger keypoint detection
"""
from ultralytics import YOLO
import cv2
import numpy as np
import time
import csv
import math

# ==================== Configuration ====================
MODEL_PATH = r"C:\ai_project\cat_pose\v11s_128.pt"
VIDEO_PATH = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\1_貓咪姿勢影片分類\模型專用\walk\walk_12.mp4"
IMGSZ = 640
CONF_THRES = 0.50
KP_CONF_THRES = 0.98
TOTAL_KPTS = 17

# Abnormality detection parameters
EMA_ALPHA = 0.7
MILD_THRES = 4   # 輕度異常（黃色）
SEVERE_THRES = 6 # 重度異常（紅色）
MIN_BODY_SCALE = 1e-3
STABILITY_K = 4.0

# ==================== Colors ====================
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

COLOR_HEAD = (255, 255, 0)
COLOR_BODY = (0, 255, 0)
COLOR_LIMB = (255, 0, 0)
COLOR_TAIL = (255, 0, 255)

# Four legs colors (Left Front, Right Front, Left Hind, Right Hind)
COLOR_LEFT_FRONT  = (255, 0, 255)   # Magenta
COLOR_RIGHT_FRONT = (0, 255, 255)   # Cyan
COLOR_LEFT_HIND   = (255, 165, 0)   # Orange
COLOR_RIGHT_HIND  = (0, 255, 0)     # Green

# ==================== Skeleton Links ====================
HEAD_LINKS = [(0,1), (0,2), (1,2)]
BODY_LINKS = [(0,3), (3,4), (4,5)]
FRONT_LIMBS = [(3,6), (6,7), (3,8), (8,9)]
HIND_LIMBS = [(5,10), (10,11), (5,12), (12,13)]
TAIL_LINKS = [(5,14), (14,15), (15,16)]

# ==================== Fast Drawing Functions ====================
def draw_links_fast(frame, kpts, conf, links, color):
    """Fast link drawing without checks"""
    for a, b in links:
        if conf[a] > KP_CONF_THRES and conf[b] > KP_CONF_THRES:
            cv2.line(frame,
                     (int(kpts[a][0]), int(kpts[a][1])),
                     (int(kpts[b][0]), int(kpts[b][1])),
                     color, 2)

def draw_skeleton_fast(frame, kpts, conf):
    """Optimized skeleton drawing"""
    draw_links_fast(frame, kpts, conf, HEAD_LINKS, COLOR_HEAD)
    draw_links_fast(frame, kpts, conf, BODY_LINKS, COLOR_BODY)
    
    # Four legs with different colors
    draw_links_fast(frame, kpts, conf, [(3,6), (6,7)], COLOR_LEFT_FRONT)   # Left front
    draw_links_fast(frame, kpts, conf, [(3,8), (8,9)], COLOR_RIGHT_FRONT)  # Right front
    draw_links_fast(frame, kpts, conf, [(5,10), (10,11)], COLOR_LEFT_HIND)   # Left hind
    draw_links_fast(frame, kpts, conf, [(5,12), (12,13)], COLOR_RIGHT_HIND)  # Right hind
    
    draw_links_fast(frame, kpts, conf, TAIL_LINKS, COLOR_TAIL)
    
    # Draw keypoints with corresponding colors
    for i, (x, y) in enumerate(kpts):
        if conf[i] > KP_CONF_THRES:
            # Choose color based on keypoint index
            if i in [6, 7]:  # Left front leg
                color = COLOR_LEFT_FRONT
            elif i in [8, 9]:  # Right front leg
                color = COLOR_RIGHT_FRONT
            elif i in [10, 11]:  # Left hind leg
                color = COLOR_LEFT_HIND
            elif i in [12, 13]:  # Right hind leg
                color = COLOR_RIGHT_HIND
            else:
                color = (0, 0, 255)  # Red for other keypoints
            
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)


# 多級異常警報（版面通知）

# 右上角異常旗標與特效



# 根據異常等級畫yolo pose偵測到的bbox外框，並在右下角畫紅綠燈圓圈
def draw_status_alert(frame, frame_score, mild_thres, severe_thres, result, scale_draw=1.0):
    h, w = frame.shape[:2]
    # 判斷狀態顏色
    if frame_score > severe_thres:
        color = (0, 0, 255)   # 紅色
    elif frame_score > mild_thres:
        color = (0, 255, 255) # 黃色
    else:
        color = (0, 200, 0)   # 綠色
    # 針對所有yolo pose偵測到的bbox都畫外框（需縮放）
    if hasattr(result, 'boxes') and result.boxes is not None and hasattr(result.boxes, 'xyxy'):
        for box_tensor in result.boxes.xyxy:
            box = box_tensor.cpu().numpy()
            x1, y1, x2, y2 = (box * scale_draw).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 6)
    # 畫右下角紅綠燈圓圈
    cx, cy = w - 60, h - 60
    r = 28
    cv2.circle(frame, (cx, cy), r, (60,60,60), -1)
    cv2.circle(frame, (cx, cy), r-6, color, -1)
    cv2.circle(frame, (cx, cy), r, (255,255,255), 3)

def draw_trigger_marker(frame, kpts, idx, score):
    """Highlight trigger keypoint"""
    if idx is None:
        return
    
    x, y = int(kpts[idx][0]), int(kpts[idx][1])
    cv2.circle(frame, (x, y), 12, YELLOW, 2)
    cv2.circle(frame, (x, y), 4, YELLOW, -1)
    
    # Simple label
    label = f"KPT{idx} {score:.3f}"
    cv2.putText(frame, label, (x + 15, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2)

def draw_progress_bar(frame, value, y_pos, label, max_val=1.0):
    """Generic progress bar (bottom left)"""
    h, w = frame.shape[:2]
    bar_w = 200
    bar_h = 14
    x0 = 15
    y0 = h - y_pos
    
    # Background
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (40, 40, 40), -1)
    
    # Color based on value
    if value > 0.6:
        color = RED
    elif value > 0.3:
        color = YELLOW
    else:
        color = GREEN
    
    fill_w = int(bar_w * min(value / max_val, 1.0))
    cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), color, -1)
    
    # Label
    cv2.putText(frame, f"{label}: {value:.2f}", (x0, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

def compute_body_scale(kpts):
    """Body scale: chest(3) to hip(5)"""
    return float(np.linalg.norm(kpts[3] - kpts[5]))

# ==================== Main ====================
print("="*70)
print("Cat Pose Analysis - Fast Version")
print("="*70)

# 檢查檔案是否存在
import os
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found: {MODEL_PATH}")
    exit(1)
if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: Video file not found: {VIDEO_PATH}")
    exit(1)

print(f"Model: {MODEL_PATH}")
print(f"Video: {VIDEO_PATH}")
print(f"Mild Threshold: {MILD_THRES}")
print(f"Severe Threshold: {SEVERE_THRES}")
print("="*70)

model = YOLO(MODEL_PATH)
model.to("cuda")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total video frames: {total_frames}")

prev_kpts = None
prev_time = time.time()
ema_norm_disp = np.zeros(TOTAL_KPTS, dtype=np.float32)

# CSV logging
csv_path = "abnormal_events_fast.csv"
csv_file = open(csv_path, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow(["frame", "trigger_kpt", "trigger_conf", "trigger_score", 
                 "frame_score", "stability"])


frame_idx = 0
fps_display = 0
fps_update_time = time.time()

# 影片縮放比例
user_scale = 1.0
user_scale_min = 0.2
user_scale_max = 2.5
user_scale_step = 0.1

print("\n[RUNNING] Press 'q' to quit, '+' to zoom in, '-' to zoom out\n")

while True:
    ret, frame = cap.read()
    if not ret:
        # 影片結束，自動重播
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        continue

    # --- 自動縮放顯示用 frame ---
    disp_frame = frame.copy()
    h0, w0 = disp_frame.shape[:2]
    # 先根據最大視窗自動縮放，再乘以user_scale
    max_w, max_h = 1280, 720
    scale = min(max_w / w0, max_h / h0, 1.0) * user_scale
    if scale != 1.0:
        disp_frame = cv2.resize(disp_frame, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    # YOLO inference (optimized)（仍用原始frame推論）
    result = model.predict(
        frame,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        half=True,
        verbose=False,
        device=0
    )[0]
    
    # FPS calculation
    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now
    
    # Update display FPS every 10 frames
    if frame_idx % 10 == 0:
        fps_display = fps
        fps_update_time = now
    
    active_kpts = 0
    abnormal = False
    trigger_idx = None
    trigger_score = 0.0
    trigger_conf = 0.0
    frame_score = 0.0
    stability = 1.0
    alert_level = 0  # 0: normal, 1: mild, 2: severe
    prev_accel = np.zeros(TOTAL_KPTS)
    
    # ========== Pose Analysis ==========
    if result.keypoints is not None and len(result.keypoints.xy) > 0:
        kpts = result.keypoints.xy[0].cpu().numpy()
        kpt_conf = result.keypoints.conf[0].cpu().numpy()

        # 關鍵點與繪圖都要根據縮放後的 disp_frame 調整座標
        h_disp, w_disp = disp_frame.shape[:2]
        scale_draw = w_disp / w0
        kpts_disp = kpts * scale_draw

        active_kpts = int(np.sum(kpt_conf > KP_CONF_THRES))

        # Abnormality detection（對齊EDA2增強版）
        body_scale = compute_body_scale(kpts)
        accel = np.zeros(TOTAL_KPTS)
        if body_scale > MIN_BODY_SCALE and prev_kpts is not None:
            diffs = kpts - prev_kpts
            dists = np.linalg.norm(diffs, axis=1)
            norm_disp = dists / body_scale

            ema_norm_disp = EMA_ALPHA * ema_norm_disp + (1.0 - EMA_ALPHA) * norm_disp
            # 加速度
            if 'prev_prev_kpts' in locals() and prev_prev_kpts is not None:
                prev_diffs = prev_kpts - prev_prev_kpts
                prev_dists = np.linalg.norm(prev_diffs, axis=1)
                accel = dists - prev_dists
            else:
                accel = np.zeros(TOTAL_KPTS)

            ACCEL_WEIGHT = 0.3
            motion_score = ema_norm_disp + ACCEL_WEIGHT * np.abs(accel)
            score = kpt_conf * motion_score

            valid = kpt_conf > KP_CONF_THRES
            if np.any(valid):
                valid_indices = np.where(valid)[0]
                best_local = valid_indices[np.argmax(score[valid])]
                trigger_idx = int(best_local)
                trigger_score = float(score[trigger_idx])
                trigger_conf = float(kpt_conf[trigger_idx])
                frame_score = float(np.max(score[valid]))
                # 多級警報
                if frame_score > SEVERE_THRES:
                    alert_level = 2
                elif frame_score > MILD_THRES:
                    alert_level = 1
                else:
                    alert_level = 0
                abnormal = alert_level > 0
                stability = math.exp(-STABILITY_K * frame_score)

            prev_prev_kpts = prev_kpts.copy() if 'prev_kpts' in locals() and prev_kpts is not None else None

        prev_kpts = kpts.copy()

        # Draw skeleton
        draw_skeleton_fast(disp_frame, kpts_disp, kpt_conf)
        draw_status_alert(disp_frame, frame_score, MILD_THRES, SEVERE_THRES, result, scale_draw)
        # Highlight trigger point
        if trigger_idx is not None:
            draw_trigger_marker(disp_frame, kpts_disp, trigger_idx, trigger_score)
        # Progress bars
        draw_progress_bar(disp_frame, frame_score, 50, "Abnormal")
        draw_progress_bar(disp_frame, stability, 25, "Stability")
        # Log abnormal events
        if alert_level > 0 and trigger_idx is not None:
            writer.writerow([
                frame_idx,
                trigger_idx,
                f"{trigger_conf:.4f}",
                f"{trigger_score:.6f}",
                f"{frame_score:.6f}",
                f"{stability:.6f}"
            ])
    
    # ========== Info Display (Top Left) ==========
    info = [
        f"Frame: {frame_idx}",
        f"FPS: {fps_display:.1f}",
        f"Kpts: {active_kpts}/{TOTAL_KPTS}",
        f"Mild Thres: {MILD_THRES}",
        f"Severe Thres: {SEVERE_THRES}"
    ]

    for i, txt in enumerate(info):
        cv2.putText(disp_frame, txt, (15, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 3)
        cv2.putText(disp_frame, txt, (15, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    cv2.imshow("Cat Pose - Fast Analysis", disp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        user_scale = min(user_scale + user_scale_step, user_scale_max)
        print(f"[Zoom] user_scale: {user_scale:.2f}")
    elif key == ord('-') or key == ord('_'):
        user_scale = max(user_scale - user_scale_step, user_scale_min)
        print(f"[Zoom] user_scale: {user_scale:.2f}")
    frame_idx += 1

cap.release()
csv_file.close()
cv2.destroyAllWindows()

print("\n" + "="*70)
print(f"Analysis completed!")
print(f"Total frames: {frame_idx}")
print(f"CSV saved: {csv_path}")
print("="*70)
