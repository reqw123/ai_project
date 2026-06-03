#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cat Pose Video Inference & Save Tool
- 逐幀瀏覽或自動播放，按 S 儲存原始影像作為訓練資料
- 重複幀保護：同一幀不會重複儲存
"""

import os
import re
import cv2
from ultralytics import YOLO
from pathlib import Path

# ==================== 設定 ====================
MODEL_PATH       = r"C:\ai_project\cat_pose\v11s_88.pt"
VIDEO_DIR        = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\scratch"
OUTPUT_DIR       = r"C:\AI_Project\cat_pose\cat7"
IMG_NAME_FORMAT  = "scratch11-{}.png"
TARGET_MODEL_FPS = 30.0

VIDEO_EXTS         = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpeg', '.mpg'}
KP_CONF_THRES      = 0.5
MAX_DISPLAY_WIDTH  = 1920
MAX_DISPLAY_HEIGHT = 1080

# ---- 骨架視覺 ----
_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 2),                                    # 頭
    (0, 3), (3, 4), (4, 5),                                    # 軀幹中軸
    (3, 6), (6, 7), (3, 8), (8, 9),                           # 前肢
    (5, 10), (10, 11), (5, 12), (12, 13),                     # 後肢
    (5, 14), (14, 15), (15, 16),                               # 尾
]
_KP_COLORS = [
    (255,  80,  80), (255, 160,  40), (255, 160,  40),
    (255, 255,  60), (200, 255,  60), (100, 255, 100),
    ( 60, 200, 255), ( 60, 120, 255), ( 60, 200, 255), ( 60, 120, 255),
    (180,  80, 255), (120,  40, 255), (180,  80, 255), (120,  40, 255),
    ( 80, 220, 180), ( 60, 180, 140), ( 40, 140, 100),
]
_EDGE_COLORS = [
    (255, 120,  60), (255, 120,  60), (255, 120,  60),
    (220, 220,  60), (200, 220,  60), (160, 220,  60),
    (102,  85, 255), (102,  85, 255), (255,  68, 204), (255,  68, 204),
    (255, 170,  34), (255, 170,  34), (  0, 153, 255), (  0, 153, 255),
    ( 80, 200, 160), ( 60, 170, 130), ( 40, 140, 100),
]

# ==================== 輸出目錄 & 接續命名 ====================
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

prefix  = IMG_NAME_FORMAT.split('{')[0]
pattern = re.compile(rf"{re.escape(prefix)}(\d+)\.(jpg|png)$")
max_idx = 0
for _f in output_dir.iterdir():
    if _f.is_file():
        _m = pattern.match(_f.name)
        if _m and int(_m.group(1)) > max_idx:
            max_idx = int(_m.group(1))
save_idx = max_idx + 1

# ==================== 影片清單 ====================
def _natural_key(p):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', os.path.basename(p).lower())]

def get_all_videos(folder):
    videos = [
        os.path.join(r, f)
        for r, _, files in os.walk(folder)
        for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    ]
    videos.sort(key=_natural_key)
    return videos

video_list = get_all_videos(VIDEO_DIR)
if not video_list:
    print(f"[Error] No videos found in {VIDEO_DIR}")
    exit(1)
video_idx  = 0
VIDEO_PATH = video_list[0]

# ==================== 載入模型 ====================
print("[Loading] Model...")
try:
    model = YOLO(MODEL_PATH)
    model.to("cuda")
    USE_HALF = True
    print("[OK] Model loaded (GPU)")
except Exception as _e:
    print(f"[GPU failed] {_e}")
    try:
        model = YOLO(MODEL_PATH)
        USE_HALF = False
        print("[OK] Model loaded (CPU)")
    except Exception as _e2:
        print(f"[Fatal] {_e2}")
        exit(1)

# ==================== 執行時狀態 ====================
cap                  = None
fps                  = 0.0
total_frames         = 0
width                = 0
height               = 0
auto_infer_interval  = 1
skeleton_ov          = 1.0

frame_idx            = 0
step_mode            = True
frame_step           = 1
last_result          = None
last_infer_frame_idx = -1
cached_frame         = None
last_frame_auto      = None
last_auto_frame_idx  = -1   # auto mode 下 last_frame_auto 對應的 frame_idx

saved_log              = {}  # {video_path: [img_name, ...]}
saved_frames_per_video = {}  # {video_path: set(frame_idx)} — 重複幀保護

# ==================== 輔助函式 ====================
def open_video(idx):
    global cap, frame_idx, fps, total_frames, width, height, VIDEO_PATH
    global auto_infer_interval, skeleton_ov
    VIDEO_PATH = video_list[idx]
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[Error] Cannot open: {VIDEO_PATH}")
        exit(1)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps      = fps if fps > 1 else TARGET_MODEL_FPS
    auto_infer_interval = max(1, round(src_fps / TARGET_MODEL_FPS))
    skeleton_ov  = max(0.6, height / 720.0)
    frame_idx    = 0
    print(f"[Video] {VIDEO_PATH}  {width}x{height}  {fps:.2f}fps  {total_frames}f"
          f"  | infer every {auto_infer_interval} frame(s)")

def get_display_size(w, h):
    scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0)
    return int(w * scale), int(h * scale), scale

def draw_styled_skeleton(frame, kpts, kpt_conf, ov, conf_thresh=KP_CONF_THRES):
    lw      = max(2, int(2.5 * ov))
    r_outer = max(4, int(5.0 * ov))
    r_inner = max(3, int(3.5 * ov))
    for ei, (a, b) in enumerate(_SKELETON_EDGES):
        if a >= len(kpts) or b >= len(kpts):
            continue
        if kpt_conf[a] > conf_thresh and kpt_conf[b] > conf_thresh:
            cv2.line(frame,
                     (int(kpts[a][0]), int(kpts[a][1])),
                     (int(kpts[b][0]), int(kpts[b][1])),
                     _EDGE_COLORS[ei], lw, cv2.LINE_AA)
    for i in range(min(17, len(kpts))):
        if kpt_conf[i] <= conf_thresh:
            continue
        cx, cy = int(kpts[i][0]), int(kpts[i][1])
        cv2.circle(frame, (cx, cy), r_outer, (0, 0, 0),     -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r_inner, _KP_COLORS[i], -1, cv2.LINE_AA)

def draw_result(frame, result):
    disp = frame.copy()
    if result and result.boxes is not None and len(result.boxes.xyxy) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            label = f"{model.names.get(int(box.cls[0]), '?')} {float(box.conf[0]):.2f}"
            cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(disp, (x1, y1 - th - 8), (x1 + tw + 4, y1), (255, 0, 0), -1)
            cv2.putText(disp, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    if result and result.keypoints is not None and len(result.keypoints.xy) > 0:
        draw_styled_skeleton(
            disp,
            result.keypoints.xy[0].cpu().numpy(),
            result.keypoints.conf[0].cpu().numpy(),
            skeleton_ov,
        )
    return disp

def _put_text(img, text, pos, fs, color, thickness):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, fs, color,     thickness,     cv2.LINE_AA)

def draw_hud(img, lines_top, lines_bottom, scale):
    """半透明頂部 / 底部 HUD 欄，保留原始影像不被文字直接遮蓋。"""
    h, w = img.shape[:2]
    lh    = int(28 * scale)
    pad   = int(6  * scale)
    top_h = lh * len(lines_top)    + pad
    bot_h = lh * len(lines_bottom) + pad
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0),          (w, top_h),     (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - bot_h),  (w, h),         (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, img, 0.50, 0, img)
    fs = max(0.35, 0.58 * scale)
    th = max(1, int(1.5 * scale))
    for i, line in enumerate(lines_top):
        y = pad + (i + 1) * lh - int(6 * scale)
        _put_text(img, line, (int(10 * scale), y), fs, (240, 240, 240), th)
    for i, line in enumerate(lines_bottom):
        y = h - bot_h + pad + (i + 1) * lh - int(6 * scale)
        _put_text(img, line, (int(10 * scale), y), fs, (190, 190, 190), max(1, th - 1))

def do_save(raw_frame, src_frame_idx):
    """儲存原始影像。同一影片同一幀只儲存一次。"""
    global save_idx
    vpath     = VIDEO_PATH
    saved_set = saved_frames_per_video.setdefault(vpath, set())
    if src_frame_idx in saved_set:
        print(f"[Skip] frame {src_frame_idx} 已儲存過，略過。")
        return False
    img_name = Path(IMG_NAME_FORMAT.format(save_idx)).with_suffix('.png')
    cv2.imwrite(str(output_dir / img_name), raw_frame)
    print(f"[Saved] {img_name}  ({width}x{height})  frame={src_frame_idx}")
    saved_log.setdefault(vpath, []).append(str(img_name))
    saved_set.add(src_frame_idx)
    save_idx += 1
    return True

def apply_video_switch(initial_key):
    global video_idx, last_result, last_frame_auto, last_infer_frame_idx, last_auto_frame_idx
    delta, key = 0, initial_key
    while True:
        if   key == ord('1'): delta -= 1
        elif key == ord('2'): delta += 1
        else: break
        key = cv2.waitKey(30) & 0xFF
    if delta == 0:
        return False
    new_idx = max(0, min(len(video_list) - 1, video_idx + delta))
    if new_idx == video_idx:
        return False
    video_idx = new_idx
    open_video(video_idx)
    cv2.resizeWindow(WIN_NAME, *get_display_size(width, height)[:2])
    print_mode()
    last_result = last_frame_auto = None
    last_infer_frame_idx = last_auto_frame_idx = -1
    return True

def print_mode():
    mode = "Step" if step_mode else "Auto"
    print(f"\n[Mode={mode}]  S=Save  1/2=Prev/NextVideo  Space=Toggle  Q=Quit")
    if step_mode:
        print(f"  D=NextFrame  A=PrevFrame  Z=+Step  X=-Step  (step={frame_step})")

def write_log():
    log_path = output_dir / "saved_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        for vpath, imgs in saved_log.items():
            f.write(f"{vpath}\n")
            for img in imgs:
                f.write(f"  {img}\n")
            f.write("\n")
    print(f"[Log] {log_path}  (total saved: {save_idx - 1})")

# ==================== 主視窗 ====================
WIN_NAME = "Cat Pose Inference"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_KEEPRATIO)
print_mode()
open_video(video_idx)
cv2.resizeWindow(WIN_NAME, *get_display_size(width, height)[:2])

# ==================== 主迴圈 ====================
while True:
    if not cap.isOpened():
        break

    # ---- Step 模式 ----
    if step_mode:
        outer_break = False
        while True:
            if frame_idx != last_infer_frame_idx:
                # random access 在 h264/h265 影片較慢；大量標注建議先用 ffmpeg 轉 image sequence
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    frame_idx = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                cached_frame         = frame.copy()
                last_result          = model.predict(frame, imgsz=640, conf=0.5,
                                                      half=USE_HALF, verbose=False)[0]
                last_infer_frame_idx = frame_idx
            else:
                frame = cached_frame

            disp = draw_result(frame, last_result)
            dw, dh, _ = get_display_size(width, height)
            disp = cv2.resize(disp, (dw, dh), interpolation=cv2.INTER_AREA)
            sc = min(dh / 720.0, 1.0)

            saved_set  = saved_frames_per_video.get(VIDEO_PATH, set())
            saved_mark = "  [SAVED]" if frame_idx in saved_set else ""
            draw_hud(disp,
                     [f"[{video_idx+1}/{len(video_list)}] {os.path.basename(VIDEO_PATH)}",
                      f"Frame {frame_idx+1}/{total_frames}{saved_mark}   Saved:{save_idx-1}   Step:{frame_step}"],
                     ["S=Save  D=Next  A=Prev  Z=+Step  X=-Step  1/2=Video  Space=Auto  Q=Quit"],
                     sc)
            cv2.imshow(WIN_NAME, disp)
            key = cv2.waitKey(0) & 0xFF

            if   key in (ord('d'), ord('D')): frame_idx = min(frame_idx + frame_step, total_frames - 1)
            elif key in (ord('a'), ord('A')): frame_idx = max(frame_idx - frame_step, 0)
            elif key in (ord('z'), ord('Z')): frame_step += 1;              print_mode()
            elif key in (ord('x'), ord('X')): frame_step = max(1, frame_step - 1); print_mode()
            elif key in (ord('s'), ord('S')): do_save(cached_frame, frame_idx)
            elif key in (ord('1'), ord('2')): apply_video_switch(key); outer_break = True; break
            elif key == 32:                   step_mode = not step_mode; print_mode(); break
            elif key in (ord('q'), ord('Q')):
                cap.release()
                cv2.destroyAllWindows()
                write_log()
                print(f"\n[Done] {save_idx-1} frames saved  →  {OUTPUT_DIR}")
                exit(0)
        if outer_break:
            continue

    # ---- Auto 模式 ----
    else:
        key = cv2.waitKey(1) & 0xFF
        if   key in (ord('1'), ord('2')): apply_video_switch(key); continue
        elif key in (ord('q'), ord('Q')): break
        elif key in (ord('s'), ord('S')):
            if last_frame_auto is not None:
                do_save(last_frame_auto, last_auto_frame_idx)
        elif key == 32: step_mode = not step_mode; print_mode(); continue

        ret, frame = cap.read()
        if not ret:
            if video_idx < len(video_list) - 1:
                video_idx += 1
                open_video(video_idx)
                cv2.resizeWindow(WIN_NAME, *get_display_size(width, height)[:2])
                print_mode()
                last_result = last_frame_auto = None
                last_auto_frame_idx = -1
                continue
            else:
                break

        last_auto_frame_idx = frame_idx   # 記在 increment 之前，do_save 用此值做重複偵測
        last_frame_auto     = frame
        if frame_idx % auto_infer_interval == 0:
            last_result = model.predict(frame, imgsz=640, conf=0.5,
                                         half=USE_HALF, verbose=False)[0]

        disp = draw_result(frame, last_result)
        dw, dh, _ = get_display_size(width, height)
        disp = cv2.resize(disp, (dw, dh), interpolation=cv2.INTER_AREA)
        sc = min(dh / 720.0, 1.0)
        draw_hud(disp,
                 [f"[{video_idx+1}/{len(video_list)}] {os.path.basename(VIDEO_PATH)}",
                  f"Frame {frame_idx+1}/{total_frames}   Saved:{save_idx-1}"],
                 ["S=Save  1/2=Video  Space=Step  Q=Quit"],
                 sc)
        cv2.imshow(WIN_NAME, disp)
        frame_idx += 1

# ==================== 結束 ====================
cap.release()
cv2.destroyAllWindows()
write_log()
print(f"\n[Done] {save_idx-1} frames saved  →  {OUTPUT_DIR}")
