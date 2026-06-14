 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cat Pose Video Inference & Save Tool
- 單純推論影片
- 按 S 鍵儲存當前影像到指定資料夾
- 空白鍵切換 Step/Auto 模式
- 影像命名: walk{i}.png
"""

import os
import re
import cv2
from ultralytics import YOLO
from pathlib import Path

# ==================== 設定 ====================
MODEL_PATH = r"C:\ai_project\cat_pose\v11s_96.pt"
VIDEO_DIR = r"C:\Users\homec\Downloads\shake_ai"  # 讀取資料夾下所有影片
OUTPUT_DIR = r"C:/cat_pose/cat37"
IMG_NAME_FORMAT = "lick15-{}.png"
TARGET_MODEL_FPS = 30.0

# 支援影片副檔名
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpeg', '.mpg'}

# ==================== 資料夾自動接續命名 ====================
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# 根據 IMG_NAME_FORMAT 前綴搜尋現有檔案
prefix = IMG_NAME_FORMAT.split('{')[0]
pattern = re.compile(rf"{re.escape(prefix)}(\d+)\.(jpg|png)$")
max_idx = 0
for file in output_dir.iterdir():
    if file.is_file():
        m = pattern.match(file.name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
save_idx = max_idx + 1

# 取得所有影片清單（自然排序：screen_1, screen_2, ..., screen_10 順序正確）
def _natural_key(path):
    parts = re.split(r'(\d+)', os.path.basename(path).lower())
    return [int(p) if p.isdigit() else p for p in parts]

def get_all_videos(folder):
    videos = []
    for root, _, files in os.walk(folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in VIDEO_EXTS:
                videos.append(os.path.join(root, file))
    videos.sort(key=_natural_key)
    return videos

video_list = get_all_videos(VIDEO_DIR)
if not video_list:
    print(f"[Error] No videos found in {VIDEO_DIR}")
    exit(1)
video_idx = 0
VIDEO_PATH = video_list[video_idx]

# ==================== 載入模型 ====================
print("[Loading] Model...")
try:
    model = YOLO(MODEL_PATH)
    model.to("cuda")
    print("[OK] Model loaded (GPU)!")
    USE_HALF = True
except Exception as e:
    print(f"[Failed] GPU loading failed: {e}")
    try:
        model = YOLO(MODEL_PATH)
        print("[OK] Model loaded (CPU)!")
        USE_HALF = False
    except Exception as e2:
        print(f"[Failed] Model loading failed: {e2}")
        exit(1)

# ========== 儲存日誌 ========== 
saved_log = {}  # 影片路徑: [儲存過的圖片檔名]

# ==================== 開啟影片 ====================
cap = None
fps = 0.0
total_frames = 0
width = 0
height = 0
auto_infer_interval = 1
skeleton_ov = 1.0



def open_video(idx):
    global cap, frame_idx, fps, total_frames, width, height, VIDEO_PATH
    global auto_infer_interval, skeleton_ov
    VIDEO_PATH = video_list[idx]
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {VIDEO_PATH}")
        exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = fps if fps > 1 else TARGET_MODEL_FPS
    auto_infer_interval = max(1, int(round(source_fps / TARGET_MODEL_FPS)))
    model_input_fps = source_fps / auto_infer_interval
    skeleton_ov = max(0.6, height / 720.0)
    frame_idx = 0
    print(
        f"[Info] Video: {VIDEO_PATH}  {width}x{height}  {fps:.2f} FPS  {total_frames} frames"
        f"  | model_fps≈{model_input_fps:.2f} (target={TARGET_MODEL_FPS:.0f}, step={auto_infer_interval})"
    )

frame_idx = 0
# save_idx 已於上方自動取得
step_mode = True  # 預設逐幀
last_result = None  # 保存最後一次推論結果，避免閃爍
last_infer_frame_idx = -1  # 記錄上次推論的 frame_idx，避免重複推論
cached_frame = None  # 快取當前 frame（step 模式用）
last_frame_auto = None  # auto 模式保存最後一幀原始畫質（供 S 鍵儲存）
frame_step = 1  # 每次移動的幀數（Z 增加，X 減少）




# ==================== 顯示窗口限制 ====================
MAX_DISPLAY_WIDTH = 1920
MAX_DISPLAY_HEIGHT = 1080

def get_display_size(orig_width, orig_height):
    """計算顯示窗口尺寸，保持寬高比不變，高度/寬度一律限縮在 1080/1920 以內"""
    scale_w = MAX_DISPLAY_WIDTH / orig_width
    scale_h = MAX_DISPLAY_HEIGHT / orig_height
    scale = min(scale_w, scale_h, 1.0)  # 只縮不放

    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    return new_width, new_height, scale

# ==================== 建立固定顯示視窗 ====================
WIN_NAME = "Cat Pose Inference"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_KEEPRATIO)

def resize_window_to_video():
    """根據當前影片解析度設定視窗顯示大小（不超過 1920x1080）"""
    dw, dh, _ = get_display_size(width, height)
    cv2.resizeWindow(WIN_NAME, dw, dh)

def apply_video_switch(initial_key):
    """
    連續換片：先以 waitKey(30) 收集所有排隊中的 1/2 按鍵，
    再一次完成切換，避免 imshow/open_video/resizeWindow 的
    Windows 訊息幫浦中途吃掉後續按鍵。
    回傳 True 代表確實切換了影片。
    """
    global video_idx, frame_idx, last_result, last_frame_auto, last_infer_frame_idx
    delta = 0
    key = initial_key
    while True:
        if key == ord('1'):
            delta -= 1
        elif key == ord('2'):
            delta += 1
        else:
            break
        key = cv2.waitKey(30) & 0xFF  # 非阻塞，30ms 內無新鍵則停止收集
    if delta == 0:
        return False
    new_idx = max(0, min(len(video_list) - 1, video_idx + delta))
    if new_idx == video_idx:
        return False
    video_idx = new_idx
    open_video(video_idx)
    resize_window_to_video()
    print_mode()
    frame_idx = 0
    last_result = None
    last_infer_frame_idx = -1
    last_frame_auto = None
    return True

def print_mode():
    mode = "Step" if step_mode else "Auto"
    print(
        f"\n[操作說明] S=儲存影像  Z=增加步長  X=減少步長 (當前步長={frame_step})  "
        f"1=上一部影片 2=下一部影片  Space=切換模式({mode})  Q=離開\n"
        f"[Step模式] D=下一幀  A=上一幀\n"
        f"[Auto模式] 目標推論FPS={TARGET_MODEL_FPS:.0f}，每 {auto_infer_interval} 幀推論一次\n"
    )

print_mode()
open_video(video_idx)
resize_window_to_video()

# ==================== 骨架顏色與連結（與 cat_mp4.py 一致） ====================
KP_CONF_THRES = 0.5
BLUE = (255, 0, 0)

_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 2),
    (0, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (3, 8), (8, 9),
    (5, 10), (10, 11), (5, 12), (12, 13),
    (5, 14), (14, 15), (15, 16),
]

_KP_COLORS = [
    (255, 80, 80), (255, 160, 40), (255, 160, 40),
    (255, 255, 60), (200, 255, 60), (100, 255, 100),
    (60, 200, 255), (60, 120, 255), (60, 200, 255), (60, 120, 255),
    (180, 80, 255), (120, 40, 255), (180, 80, 255), (120, 40, 255),
    (80, 220, 180), (60, 180, 140), (40, 140, 100),
]

_EDGE_COLORS = [
    (255, 120, 60), (255, 120, 60), (255, 120, 60),
    (220, 220, 60), (200, 220, 60), (160, 220, 60),
    (102, 85, 255), (102, 85, 255), (255, 68, 204), (255, 68, 204),
    (255, 170, 34), (255, 170, 34), (0, 153, 255), (0, 153, 255),
    (80, 200, 160), (60, 170, 130), (40, 140, 100),
]


def draw_styled_skeleton(frame, kpts, kpt_conf, ov, conf_thresh=KP_CONF_THRES):
    """套用 video.py 的骨架視覺風格。"""
    line_w = max(1, int(2 * ov))
    r_outer = max(3, int(4 * ov))
    r_inner = max(2, int(3 * ov))

    for ei, (a, b) in enumerate(_SKELETON_EDGES):
        if a >= len(kpts) or b >= len(kpts):
            continue
        if float(kpt_conf[a]) > conf_thresh and float(kpt_conf[b]) > conf_thresh:
            pa = (int(kpts[a][0]), int(kpts[a][1]))
            pb = (int(kpts[b][0]), int(kpts[b][1]))
            col = _EDGE_COLORS[ei] if ei < len(_EDGE_COLORS) else (180, 180, 180)
            cv2.line(frame, pa, pb, col, line_w, cv2.LINE_AA)

    for i in range(min(17, len(kpts))):
        if float(kpt_conf[i]) <= conf_thresh:
            continue
        cx, cy = int(kpts[i][0]), int(kpts[i][1])
        col = _KP_COLORS[i] if i < len(_KP_COLORS) else (200, 200, 200)
        cv2.circle(frame, (cx, cy), r_outer, (0, 0, 0), -1)
        cv2.circle(frame, (cx, cy), r_inner, col, -1)

# ==================== 绘制函数 ====================
def draw_result(frame, result):
    disp_frame = frame.copy()

    # BBox：所有偵測到的貓
    if result and result.boxes is not None and len(result.boxes.xyxy) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names.get(cls_id, f"id_{cls_id}")
            label = f"{cls_name} {conf:.2f}"

            cv2.rectangle(disp_frame, (x1, y1), (x2, y2), BLUE, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(disp_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), BLUE, -1)
            cv2.putText(disp_frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 骨架：第一隻貓
    if result and result.keypoints is not None and len(result.keypoints.xy) > 0:
        kpts     = result.keypoints.xy[0].cpu().numpy()
        kpt_conf = result.keypoints.conf[0].cpu().numpy()
        draw_styled_skeleton(disp_frame, kpts, kpt_conf, ov=skeleton_ov)

    return disp_frame

def draw_text(img, text, pos, scale, font_scale=0.8, thickness=2):
    """自适应文字绘制"""
    font_scale = font_scale * scale
    thickness = max(1, int(thickness * scale))
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)

while True:
    if not cap.isOpened():
        break

    if step_mode:
        # 逐幀模式下，僅在按鍵時才換幀，畫面每次都即時更新
        step_break_outer = False
        while True:
            # 只有當 frame_idx 改變時才需要重新讀取和推論
            if frame_idx != last_infer_frame_idx:
                # 注意：cap.set() random access 在 h264/h265 影片很慢（需重新解碼 GOP）
                # 如果需要大量標註，建議先用 ffmpeg 將影片轉為 image sequence
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    frame_idx = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                cached_frame = frame.copy()
                last_result = model.predict(frame, imgsz=640, conf=0.5, half=USE_HALF, verbose=False)[0]
                last_infer_frame_idx = frame_idx
            else:
                # frame_idx 未改變，直接使用快取的 frame 和 result
                frame = cached_frame
            
            disp_frame = draw_result(frame, last_result)
            
            # 縮放顯示畫面至 1920x1080 以內（保存時使用 cached_frame 原始畫質）
            display_width, display_height, _ = get_display_size(width, height)
            disp_frame = cv2.resize(disp_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
            
            # UI 文字縮放因子：以 720p 為基準，上限 1.0 避免文字過大
            scale = min(display_height / 720.0, 1.0)
            video_info = f"Video: [{video_idx+1}/{len(video_list)}] {os.path.basename(VIDEO_PATH)}"
            draw_text(disp_frame, video_info, (int(20*scale), int(30*scale)), scale, 0.8, 2)
            draw_text(disp_frame, f"Frame: {frame_idx+1}/{total_frames}", (int(20*scale), int(60*scale)), scale, 1, 2)
            draw_text(disp_frame, f"Saved: {save_idx-1}", (int(20*scale), int(95*scale)), scale, 0.8, 2)
            draw_text(disp_frame, f"Step: {frame_step}", (int(20*scale), int(125*scale)), scale, 0.8, 2)
            mode_str = "Step" if step_mode else "Auto"
            infer_text = f"Infer: target {TARGET_MODEL_FPS:.0f} FPS / every {auto_infer_interval} frame(s)"
            help_text = f"Mode: {mode_str}  S=Save  D=Next  A=Prev  Z=+Step  X=-Step  1=PrevVideo 2=NextVideo  Space=Switch  Q=Quit"
            margin = int(24 * scale)
            y_pos = disp_frame.shape[0] - margin
            draw_text(disp_frame, infer_text, (int(20*scale), y_pos - int(28*scale)), scale, 0.6, 2)
            draw_text(disp_frame, help_text, (int(20*scale), y_pos), scale, 0.6, 2)
            cv2.imshow(WIN_NAME, disp_frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('d') or key == ord('D'):
                frame_idx = min(frame_idx + frame_step, total_frames - 1)
            elif key == ord('a') or key == ord('A'):
                frame_idx = max(frame_idx - frame_step, 0)
            elif key == ord('z') or key == ord('Z'):
                frame_step += 1
                print(f"[Step] 每次移動步長: {frame_step}")
            elif key == ord('x') or key == ord('X'):
                frame_step = max(1, frame_step - 1)
                print(f"[Step] 每次移動步長: {frame_step}")
            elif key == ord('1') or key == ord('2'):
                apply_video_switch(key)
                step_break_outer = True
                break
            elif key == ord('q') or key == ord('Q'):
                print("[Exit] Quit.")
                cap.release()
                cv2.destroyAllWindows()
                log_path = output_dir / "saved_log.txt"
                with open(log_path, "w", encoding="utf-8") as f:
                    for vpath, imgs in saved_log.items():
                        f.write(f"{vpath}\n")
                        for img in imgs:
                            f.write(f"    {img}\n")
                        f.write("\n")
                print(f"\n[Complete] 共儲存 {save_idx-1} 張影像於: {OUTPUT_DIR}")
                print(f"[Log] 已產生日誌: {log_path}")
                exit(0)
            elif key == ord('s') or key == ord('S'):
                img_name = IMG_NAME_FORMAT.format(save_idx)
                img_name_png = Path(img_name).with_suffix('.png')
                save_path = output_dir / img_name_png
                # 保存原始分辨率的 frame，不受显示缩放影响
                cv2.imwrite(str(save_path), cached_frame)
                print(f"[Saved] {img_name_png} ({width}x{height})")
                vpath = VIDEO_PATH
                if vpath not in saved_log:
                    saved_log[vpath] = []
                saved_log[vpath].append(str(img_name_png))
                save_idx += 1
            elif key == 32:  # Space
                step_mode = not step_mode
                print_mode()
                break
            # 其他按鍵不動，繼續顯示當前 frame
        if step_break_outer:
            continue

    else:
        # 自動模式：先偵測按鍵（捕捉上一幀推論期間累積的按鍵事件），再處理畫面
        # cv2.imshow 在 Windows 上有時會消耗按鍵事件，因此在 imshow 之前先 poll 一次
        key = cv2.waitKey(1) & 0xFF

        # ---- 按鍵處理 ----
        if key == ord('1') or key == ord('2'):
            apply_video_switch(key)
            continue
        elif key == ord('q') or key == ord('Q'):
            print("[Exit] Quit.")
            break
        elif key == ord('s') or key == ord('S'):
            if last_frame_auto is not None:
                img_name = IMG_NAME_FORMAT.format(save_idx)
                img_name_png = Path(img_name).with_suffix('.png')
                save_path = output_dir / img_name_png
                cv2.imwrite(str(save_path), last_frame_auto)
                print(f"[Saved] {img_name_png} ({width}x{height})")
                vpath = VIDEO_PATH
                if vpath not in saved_log:
                    saved_log[vpath] = []
                saved_log[vpath].append(str(img_name_png))
                save_idx += 1
        elif key == ord('z') or key == ord('Z'):
            frame_step += 1
            print(f"[Step] 每次移動步長: {frame_step}")
        elif key == ord('x') or key == ord('X'):
            frame_step = max(1, frame_step - 1)
            print(f"[Step] 每次移動步長: {frame_step}")
        elif key == 32:  # Space
            step_mode = not step_mode
            print_mode()
            continue

        # ---- 讀取與推論 ----
        ret, frame = cap.read()
        if not ret:
            # 自動切換到下一部影片
            if video_idx < len(video_list) - 1:
                video_idx += 1
                open_video(video_idx)
                resize_window_to_video()
                print_mode()
                frame_idx = 0
                last_result = None
                last_frame_auto = None
                continue
            else:
                # 已經是最後一部影片，結束
                break
        last_frame_auto = frame  # 保留原始畫質供 S 鍵儲存
        do_infer = (frame_idx % auto_infer_interval == 0)
        if do_infer:
            # 使用 640 與 step mode 一致，避免漏檢小目標
            last_result = model.predict(frame, imgsz=640, conf=0.5, half=USE_HALF, verbose=False)[0]
        # 使用最后一次推论结果绘制，避免闪烁
        disp_frame = draw_result(frame, last_result)

        # 縮放顯示畫面至 1920x1080 以內（保存時使用 last_frame_auto 原始畫質）
        display_width, display_height, _ = get_display_size(width, height)
        disp_frame = cv2.resize(disp_frame, (display_width, display_height), interpolation=cv2.INTER_AREA)

        # UI 文字縮放因子：以 720p 為基準，上限 1.0 避免文字過大
        scale = min(display_height / 720.0, 1.0)
        video_info = f"Video: [{video_idx+1}/{len(video_list)}] {os.path.basename(VIDEO_PATH)}"
        draw_text(disp_frame, video_info, (int(20*scale), int(30*scale)), scale, 0.8, 2)
        draw_text(disp_frame, f"Frame: {frame_idx+1}/{total_frames}", (int(20*scale), int(60*scale)), scale, 1, 2)
        draw_text(disp_frame, f"Saved: {save_idx-1}", (int(20*scale), int(95*scale)), scale, 0.8, 2)
        draw_text(disp_frame, f"Step: {frame_step}", (int(20*scale), int(125*scale)), scale, 0.8, 2)
        mode_str = "Step" if step_mode else "Auto"
        infer_text = f"Infer: target {TARGET_MODEL_FPS:.0f} FPS / every {auto_infer_interval} frame(s)"
        help_text = f"Mode: {mode_str}  S=Save  Z=+Step  X=-Step  1=PrevVideo 2=NextVideo  Space=Switch  Q=Quit"
        margin = int(24 * scale)
        y_pos = disp_frame.shape[0] - margin
        draw_text(disp_frame, infer_text, (int(20*scale), y_pos - int(28*scale)), scale, 0.6, 2)
        draw_text(disp_frame, help_text, (int(20*scale), y_pos), scale, 0.6, 2)
        cv2.imshow(WIN_NAME, disp_frame)
        frame_idx += 1


cap.release()
cv2.destroyAllWindows()

# 輸出日誌
log_path = output_dir / "saved_log.txt"
with open(log_path, "w", encoding="utf-8") as f:
    for vpath, imgs in saved_log.items():
        f.write(f"{vpath}\n")
        for img in imgs:
            f.write(f"    {img}\n")
        f.write("\n")
print(f"\n[Complete] 共儲存 {save_idx-1} 張影像於: {OUTPUT_DIR}")
print(f"[Log] 已產生日誌: {log_path}")
