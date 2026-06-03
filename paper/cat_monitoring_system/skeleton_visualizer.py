"""貓咪骨架視覺化腳本。

功能：
1) 從 VIDEO_LIST 與 VIDEO_PATH 載入影片或串流
2) 進行關鍵點偵測與骨架繪製
3) 提供播放、暫停、上一部、下一部、重置與縮放控制

此版本只保留骨架偵測與視覺化，不包含接觸區域、行為統計或 CSV 匯出。
"""

import sys
from pathlib import Path
from collections import deque

try:
    import msvcrt
except ImportError:
    msvcrt = None

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from detectors.keypoint_detector import KeypointDetector
from models.stgcn_model import (
    interpolate_missing,
    flip_normalize,
    orientation_normalize,
    normalize_skeleton_coords,
)

# ===== Velocity (overlay) settings =====
SEQUENCE_LENGTH = 16
VELOCITY_CONF_THRESHOLD = 0.1
VELOCITY_PANEL_TOP_K = 3

# 17 關鍵點名稱（YOLO-Pose v11 cat skeleton）
KEYPOINT_NAMES = [
    "Nose", "Left_Ear", "Right_Ear", "Chest", "Mid_Back", "Hip",
    "LF_Elbow", "LF_Paw", "RF_Elbow", "RF_Paw", "LH_Knee", "LH_Paw",
    "RH_Knee", "RH_Paw", "Tail_Root", "Tail_Mid", "Tail_Tip",
]

# which joints to highlight for velocity overlay (LeftEar, RightEar, Nose, Chest, LH paw, RH paw)
VELOCITY_HIGHLIGHT_JOINTS = [1, 2, 0, 3, 11, 13]


def _blend_color(start_color, end_color, t):
    t = float(np.clip(t, 0.0, 1.0))
    start = np.asarray(start_color, dtype=np.float32)
    end = np.asarray(end_color, dtype=np.float32)
    color = start + (end - start) * t
    return tuple(int(np.clip(v, 0, 255)) for v in color)


def _velocity_heat_color(speed, max_speed):
    # color ramp: slow=green -> mid=cyan -> fast=red
    slow = (60, 220, 80)
    mid = (0, 220, 255)
    fast = (60, 80, 255)
    if not np.isfinite(speed) or max_speed <= 1e-6:
        return slow
    ratio = float(np.clip(speed / max_speed, 0.0, 1.0))
    if ratio < 0.5:
        return _blend_color(slow, mid, ratio / 0.5)
    return _blend_color(mid, fast, (ratio - 0.5) / 0.5)


def compute_velocity_overlay(seq_array, conf_arr, conf_threshold=VELOCITY_CONF_THRESHOLD):
    """Compute velocity overlay summary from sequence arrays.
    seq_array: (T, J, 2)  conf_arr: (T, J)
    returns dict with 'joint_scores', 'overall_mean', 'recent_mean', 'peak_speed', 'top_entries'
    """
    if seq_array is None or conf_arr is None:
        return None

    seq = interpolate_missing(seq_array, conf_arr, threshold=conf_threshold)
    seq = flip_normalize(seq)
    seq = orientation_normalize(seq)
    seq = normalize_skeleton_coords(seq)

    velocity = np.zeros_like(seq)
    velocity[1:] = seq[1:] - seq[:-1]
    speed_map = np.linalg.norm(velocity, axis=-1)  # T x J

    valid_mask = conf_arr > conf_threshold
    joint_scores = np.zeros(speed_map.shape[1], dtype=np.float32)
    for j in range(speed_map.shape[1]):
        valid = valid_mask[:, j]
        if np.any(valid):
            joint_scores[j] = float(np.mean(speed_map[valid, j]))
        else:
            joint_scores[j] = 0.0

    if np.any(valid_mask):
        overall_mean = float(np.mean(speed_map[valid_mask]))
        masked = np.where(valid_mask, speed_map, -1.0)
        peak_flat = int(np.argmax(masked))
        peak_frame_idx, peak_joint_idx = np.unravel_index(peak_flat, speed_map.shape)
        peak_speed = float(speed_map[peak_frame_idx, peak_joint_idx])
    else:
        overall_mean = float(np.mean(speed_map))
        peak_flat = int(np.argmax(speed_map))
        peak_frame_idx, peak_joint_idx = np.unravel_index(peak_flat, speed_map.shape)
        peak_speed = float(speed_map[peak_frame_idx, peak_joint_idx])

    recent_len = min(3, speed_map.shape[0])
    recent_slice = speed_map[-recent_len:]
    recent_valid = valid_mask[-recent_len:]
    if np.any(recent_valid):
        recent_mean = float(np.mean(recent_slice[recent_valid]))
    else:
        recent_mean = float(np.mean(recent_slice))

    top_indices = np.argsort(joint_scores)[::-1][:VELOCITY_PANEL_TOP_K]
    top_entries = [
        {"joint_idx": int(idx), "joint_name": KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else str(idx), "score": float(joint_scores[idx])}
        for idx in top_indices
    ]

    return {
        "joint_scores": joint_scores,
        "overall_mean": overall_mean,
        "recent_mean": recent_mean,
        "peak_speed": peak_speed,
        "peak_joint_idx": int(peak_joint_idx),
        "peak_joint_name": KEYPOINT_NAMES[int(peak_joint_idx)] if int(peak_joint_idx) < len(KEYPOINT_NAMES) else str(peak_joint_idx),
        "recent_speed": speed_map[-1],
        "top_entries": top_entries,
    }


def draw_velocity_overlay_panel(frame, ovl, keypoint_names=None):
    if ovl is None:
        return frame
    h, w = frame.shape[:2]
    ui_scale = max(0.9, min(1.6, np.hypot(w, h) / 1500.0))
    panel_w = int(260 * ui_scale)
    panel_h = int(120 * ui_scale)
    pad = int(8 * ui_scale)
    x0 = w - panel_w - pad
    # shift panel down to avoid overlapping top-right HUDs (time/source/norm box)
    y0 = pad + int(44 * ui_scale)
    # draw border only (no opaque black background)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (120, 120, 120), 1, cv2.LINE_AA)
    # draw texts without opaque background; increase spacing to avoid overlap
    tx = x0 + int(8 * ui_scale)
    ty = y0 + int(18 * ui_scale)
    cv2.putText(frame, "Velocity (px/frame)", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ui_scale, (220, 220, 220), 1, cv2.LINE_AA)
    ty += int(20 * ui_scale)
    cv2.putText(frame, f"mean: {ovl['overall_mean']:.3f}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ui_scale, (200, 200, 200), 1, cv2.LINE_AA)
    ty += int(18 * ui_scale)
    cv2.putText(frame, f"peak: {ovl['peak_speed']:.3f}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ui_scale, (200, 200, 200), 1, cv2.LINE_AA)
    ty += int(18 * ui_scale)

    # Draw horizontal bars for the selected highlight joints for clearer visibility
    max_score = max(float(ovl.get('peak_speed', 1e-6)), 1e-6)
    bar_x = tx
    bar_w = int(panel_w * 0.6)
    for i, jid in enumerate(VELOCITY_HIGHLIGHT_JOINTS):
        if jid >= len(keypoint_names or []):
            name = str(jid)
        else:
            name = keypoint_names[jid] if keypoint_names is not None else KEYPOINT_NAMES[jid]
        score = float(ovl['joint_scores'][jid]) if jid < len(ovl['joint_scores']) else 0.0
        ratio = float(np.clip(score / max_score, 0.0, 1.0))
        color = _velocity_heat_color(score, max_score)
        row_y = ty + i * int(16 * ui_scale)
        # label
        cv2.putText(frame, f"{name}", (bar_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * ui_scale, (230, 230, 230), 1, cv2.LINE_AA)
        # bar background (subtle)
        bx0 = x0 + int(panel_w * 0.28)
        by0 = row_y - int(10 * ui_scale)
        bx1 = bx0 + bar_w
        by1 = by0 + int(10 * ui_scale)
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), (50, 50, 50), -1)
        # filled portion
        fx1 = bx0 + int(round(bar_w * ratio))
        if fx1 > bx0:
            cv2.rectangle(frame, (bx0, by0), (fx1, by1), color, -1)
        # numeric value
        cv2.putText(frame, f"{score:.3f}", (bx1 + int(6 * ui_scale), row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * ui_scale, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


WHITE = (255, 255, 255)
BLACK = (100, 50, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

COLOR_HEAD = (25, 25, 0)
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

# ==================== 骨架連結 ====================
# 頭部連線：0-1, 0-2, 1-2
HEAD_LINKS = [(0, 1), (0, 2), (1, 2)]

# 身體連線：0-3, 3-4, 4-5
BODY_LINKS = [(0, 3), (3, 4), (4, 5)]

# 前肢：3-6, 6-7, 3-8, 8-9
FRONT_LIMBS = [(3, 6), (6, 7), (3, 8), (8, 9)]

# 後肢：5-10, 10-11, 5-12, 12-13
HIND_LIMBS = [(5, 10), (10, 11), (5, 12), (12, 13)]

# 尾巴接身體：5-14
HIP_TAIL_LINK = [(5, 14)]

# 尾巴末端：14-15, 15-16
TAIL_LINKS = [(14, 15), (15, 16)]

# ===== 骨架視覺樣式 =====
SKELETON_EDGES = HEAD_LINKS + BODY_LINKS + FRONT_LIMBS + HIND_LIMBS + HIP_TAIL_LINK + TAIL_LINKS

SKELETON_KP_COLORS = [
    COLOR_HEAD,
    COLOR_HEAD,
    COLOR_HEAD,
    COLOR_BODY,
    COLOR_BODY,
    COLOR_BODY,
    COLOR_LEFT_FRONT,
    COLOR_LEFT_FRONT,
    COLOR_RIGHT_FRONT,
    COLOR_RIGHT_FRONT,
    COLOR_LEFT_HIND,
    COLOR_LEFT_HIND,
    COLOR_RIGHT_HIND,
    COLOR_RIGHT_HIND,
    COLOR_HIP_TAIL,
    COLOR_TAIL,
    COLOR_TAIL,
]

SKELETON_EDGE_COLORS = [
    COLOR_HEAD,
    COLOR_HEAD,
    COLOR_HEAD,
    COLOR_BODY,
    COLOR_BODY,
    COLOR_BODY,
    COLOR_LIMB,
    COLOR_LIMB,
    COLOR_LIMB,
    COLOR_LIMB,
    COLOR_LIMB,
    COLOR_LIMB,
    COLOR_LIMB,
    COLOR_LIMB,
    COLOR_HIP_TAIL,
    COLOR_TAIL,
    COLOR_TAIL,
]


# ===== 可直接修改的預設參數 =====
VIDEO_PATH = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\scratch"
VIDEO_LIST = [
    r"C:\Users\homec\Downloads\OneDrive_1_2026-5-21",
    r"C:\Users\homec\Downloads\OneDrive_2_2026-5-21",
    r"C:\Users\homec\Downloads\OneDrive_3_2026-5-21",
    r"C:\Users\homec\Downloads\OneDrive_4_2026-5-21",
]

MAX_VIDEOS = 40
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
YOLO_MODEL_PATH = r"C:\AI_Project\cat_pose\v11s_86.pt"

INFERENCE_DEVICE = "cuda"
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.8
TARGET_MODEL_FPS = 30.0
ENABLE_FPS_DOWNSAMPLE = True
EMA_ALPHA = 1.0

DISPLAY_WINDOW = True
WINDOW_NAME = "Cat Skeleton Visualizer"
DISPLAY_SIZE = (1080, 720)
LOOP_PLAYBACK = True

DRAW_KP_CONF_THRESHOLD = 0.5

SCRATCH_NOSE_IDX = 0
SCRATCH_CHEST_IDX = 3
SCRATCH_HIP_IDX = 5
SCRATCH_FRONT_PAW_IDXS = (7, 9)
SCRATCH_HIND_PAW_IDXS = (11, 13)
SCRATCH_DISTANCE_THRESHOLD_NORM = 0.22
SCRATCH_DISTANCE_CONF_THRESHOLD = 0.5

WINDOW_SCALE_STEP = 0.10
WINDOW_SCALE_MIN = 0.50
WINDOW_SCALE_MAX = 2.00


def _is_url(path_str):
    lowered = str(path_str).lower()
    return lowered.startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://"))


def collect_video_paths(max_videos=MAX_VIDEOS):
    """收集 VIDEO_LIST 內的影片檔或串流 URL。"""
    videos = []
    seen = set()

    def _add_file(path_obj: Path):
        resolved = str(path_obj.resolve())
        if resolved not in seen and path_obj.is_file() and path_obj.suffix.lower() in VIDEO_EXTENSIONS:
            seen.add(resolved)
            videos.append(path_obj)

    for entry in VIDEO_LIST:
        if _is_url(entry):
            if entry not in seen:
                seen.add(entry)
                videos.append(entry)
            if len(videos) >= max_videos:
                return videos
            continue

        path_obj = Path(entry)
        if not path_obj.exists():
            print(f"⚠️  找不到路徑，略過: {path_obj}")
            continue
        if path_obj.is_file():
            _add_file(path_obj)
        elif path_obj.is_dir():
            print(f"⚠️  VIDEO_LIST 僅接受檔案，資料夾請改填 VIDEO_PATH，已略過: {path_obj}")

        if len(videos) >= max_videos:
            return videos

    return videos


def collect_video_paths_from_folder(folder_path, max_videos=MAX_VIDEOS, existing_paths=None):
    """由 VIDEO_PATH 遞迴收集影片，或直接使用串流 URL / 攝影機索引。"""
    videos = []
    seen = set(existing_paths or [])

    if isinstance(folder_path, int):
        videos.append(folder_path)
        return videos

    if isinstance(folder_path, str) and folder_path.strip().isdigit():
        try:
            videos.append(int(folder_path.strip()))
            return videos
        except Exception:
            pass

    if _is_url(folder_path):
        if folder_path not in seen:
            seen.add(folder_path)
            videos.append(folder_path)
        return videos

    path_obj = Path(folder_path)
    if not path_obj.exists():
        print(f"⚠️  VIDEO_PATH 不存在: {path_obj}")
        return videos

    if path_obj.is_file():
        resolved = str(path_obj.resolve())
        if path_obj.suffix.lower() in VIDEO_EXTENSIONS and resolved not in seen:
            videos.append(path_obj)
        return videos

    for item in sorted(path_obj.rglob("*")):
        if not item.is_file() or item.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        resolved = str(item.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        videos.append(item)
        if len(videos) >= max_videos:
            break

    return videos


def draw_styled_skeleton(frame, kpts, kpt_conf, bbox, bbox_conf, sx, sy, ov, conf_thresh=DRAW_KP_CONF_THRESHOLD):
    """依目前視覺風格繪製 bbox、骨架與關鍵點。"""
    line_w = max(1, int(2 * ov))
    r_outer = max(3, int(4 * ov))
    r_inner = max(2, int(3 * ov))

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        bx1 = int(x1 * sx)
        by1 = int(y1 * sy)
        bx2 = int(x2 * sx)
        by2 = int(y2 * sy)

        outer_w = max(2, int(4 * ov))
        inner_w = max(1, int(2 * ov))
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), BLACK, outer_w, cv2.LINE_AA)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_HEAD, inner_w, cv2.LINE_AA)

        conf_val = None
        if bbox_conf is not None:
            try:
                conf_val = float(bbox_conf)
            except (TypeError, ValueError):
                conf_val = None

        if conf_val is not None and np.isfinite(conf_val):
            label = f"conf:{conf_val:.2f}"
            fs = 0.45 * ov
            th = max(1, int(1 * ov))
            text_shadow = max(2, th + 1)
            tx = bx1
            ty = by1 - max(6, int(8 * ov))
            if ty < max(16, int(18 * ov)):
                ty = by1 + max(18, int(20 * ov))

            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, BLACK, text_shadow, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, COLOR_HEAD, th, cv2.LINE_AA)

    for edge_idx, (a, b) in enumerate(SKELETON_EDGES):
        if a >= len(kpts) or b >= len(kpts):
            continue
        if float(kpt_conf[a]) > conf_thresh and float(kpt_conf[b]) > conf_thresh:
            pa = (int(kpts[a][0] * sx), int(kpts[a][1] * sy))
            pb = (int(kpts[b][0] * sx), int(kpts[b][1] * sy))
            col = SKELETON_EDGE_COLORS[edge_idx] if edge_idx < len(SKELETON_EDGE_COLORS) else GREEN
            cv2.line(frame, pa, pb, col, line_w, cv2.LINE_AA)

    for i in range(min(17, len(kpts))):
        conf_val = float(kpt_conf[i])
        if conf_val <= conf_thresh:
            continue
        cx, cy = int(kpts[i][0] * sx), int(kpts[i][1] * sy)
        col = SKELETON_KP_COLORS[i] if i < len(SKELETON_KP_COLORS) else COLOR_KPT
        cv2.circle(frame, (cx, cy), r_outer, BLACK, -1)
        cv2.circle(frame, (cx, cy), r_inner, col, -1)


def compute_normalized_hind_paw_distances(kpts, kpt_conf):
    """回傳鼻子到左右後腿腳尖的正規化距離（距離 / 身體長度）。"""
    if kpts is None or kpt_conf is None:
        return float("nan"), float("nan"), -1, float("nan"), float("nan")

    if len(kpts) <= SCRATCH_HIP_IDX:
        return float("nan"), float("nan"), -1, float("nan"), float("nan")

    nose_conf = float(kpt_conf[SCRATCH_NOSE_IDX])
    chest_conf = float(kpt_conf[SCRATCH_CHEST_IDX])
    hip_conf = float(kpt_conf[SCRATCH_HIP_IDX])
    if nose_conf < SCRATCH_DISTANCE_CONF_THRESHOLD:
        return float("nan"), float("nan"), -1, float("nan"), float("nan")
    if chest_conf < SCRATCH_DISTANCE_CONF_THRESHOLD or hip_conf < SCRATCH_DISTANCE_CONF_THRESHOLD:
        return float("nan"), float("nan"), -1, float("nan"), float("nan")

    nose = np.asarray(kpts[SCRATCH_NOSE_IDX], dtype=np.float64)
    chest = np.asarray(kpts[SCRATCH_CHEST_IDX], dtype=np.float64)
    hip = np.asarray(kpts[SCRATCH_HIP_IDX], dtype=np.float64)
    body_len = float(np.linalg.norm(hip - chest))
    if body_len < 1e-6:
        return float("nan"), float("nan"), -1, float("nan"), float("nan")

    hind_norms = {
        11: float("nan"),
        13: float("nan"),
    }
    best_norm = float("inf")
    best_idx = -1
    best_raw = float("nan")

    for paw_idx in SCRATCH_HIND_PAW_IDXS:
        if paw_idx >= len(kpts):
            continue
        if float(kpt_conf[paw_idx]) < SCRATCH_DISTANCE_CONF_THRESHOLD:
            continue
        paw = np.asarray(kpts[paw_idx], dtype=np.float64)
        raw_dist = float(np.linalg.norm(paw - nose))
        norm_dist = raw_dist / body_len
        hind_norms[paw_idx] = float(norm_dist)
        if norm_dist < best_norm:
            best_norm = norm_dist
            best_idx = paw_idx
            best_raw = raw_dist

    if best_idx < 0:
        return hind_norms[11], hind_norms[13], -1, float("nan"), body_len

    return hind_norms[11], hind_norms[13], int(best_idx), float(best_raw), float(body_len)


def main():
    video_paths = collect_video_paths(MAX_VIDEOS)
    existing = set()
    for item in video_paths:
        try:
            existing.add(str(item.resolve()) if isinstance(item, Path) else str(item))
        except Exception:
            existing.add(str(item))

    remaining = max(0, MAX_VIDEOS - len(video_paths))
    if remaining > 0:
        video_paths.extend(collect_video_paths_from_folder(VIDEO_PATH, remaining, existing_paths=existing))

    if not video_paths:
        print("❌ 找不到可用影片（VIDEO_LIST 與 VIDEO_PATH 都無效）")
        return

    print("=" * 60)
    print("貓咪骨架視覺化")
    print("=" * 60)
    print(f"共載入 {len(video_paths)} 支來源：")
    for index, video_path in enumerate(video_paths, start=1):
        print(f"  [{index}] {video_path}")
    print("控制: q=離開, space=播放/暫停, a=上一幀, d=下一幀, 2=下一部, 1=上一部, r=重置本片, +=放大, -=縮小")

    detector = KeypointDetector(
        YOLO_MODEL_PATH,
        device=INFERENCE_DEVICE,
        imgsz=YOLO_IMGSZ,
        conf_thres=YOLO_CONF_THRESHOLD,
    )

    display_w, display_h = DISPLAY_SIZE if DISPLAY_SIZE is not None else (1080, 720)
    window_scale = 1.0
    base_win_w, base_win_h = display_w, display_h
    current_video_idx = 0
    stop_all = False
    switch_delta = 0
    is_paused = False

    def _ui_scale():
        return max(0.6, base_win_h / 720.0)

    def _apply_window_scale():
        if not DISPLAY_WINDOW:
            return
        width = max(320, int(base_win_w * window_scale))
        height = max(240, int(base_win_h * window_scale))
        cv2.resizeWindow(WINDOW_NAME, width, height)

    def _handle_key(key, cap_obj=None, in_pause_loop=False):
        nonlocal window_scale, switch_delta, stop_all, is_paused, raw_frame_idx, ema_kpts, keypoints_buffer

        def _seek_display_frame(direction):
            nonlocal raw_frame_idx, ema_kpts, keypoints_buffer, is_paused
            if cap_obj is None:
                return "noop"
            current_pos = int(cap_obj.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            target_pos = max(0, current_pos - 1 + direction * frame_step)
            cap_obj.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
            raw_frame_idx = max(0, target_pos - 1)
            ema_kpts = None
            keypoints_buffer.clear()
            is_paused = True
            return "seek"

        if key == ord("q"):
            print("\n使用者中斷：q")
            stop_all = True
            is_paused = False
            return "break"
        if key == ord("+") or key == ord("="):
            window_scale = min(WINDOW_SCALE_MAX, window_scale + WINDOW_SCALE_STEP)
            _apply_window_scale()
            return "handled"
        if key == ord("-") or key == ord("_"):
            window_scale = max(WINDOW_SCALE_MIN, window_scale - WINDOW_SCALE_STEP)
            _apply_window_scale()
            return "handled"
        if key == ord("a") and is_paused:
            return _seek_display_frame(-1)
        if key == ord("d") and is_paused:
            return _seek_display_frame(1)
        if key == ord("2"):
            switch_delta = 1
            is_paused = False
            return "break"
        if key == ord("1"):
            switch_delta = -1
            is_paused = False
            return "break"
        if key == ord("r"):
            if cap_obj is not None:
                cap_obj.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return "handled"
        if key == ord(" "):
            is_paused = not is_paused
            return "handled"
        return "noop"

    if DISPLAY_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        _apply_window_scale()

    while not stop_all:
        video_path = video_paths[current_video_idx]
        is_stream = _is_url(str(video_path))

        if isinstance(video_path, int):
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(str(video_path))

        if is_stream:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print(f"❌ 無法開啟來源，跳過: {video_path}")
            current_video_idx = (current_video_idx + 1) % len(video_paths)
            continue

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 1:
            source_fps = TARGET_MODEL_FPS

        frame_step = 1
        if ENABLE_FPS_DOWNSAMPLE and source_fps > TARGET_MODEL_FPS + 1e-6:
            frame_step = max(1, int(round(source_fps / TARGET_MODEL_FPS)))
        model_input_fps = source_fps / frame_step

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if is_stream and total_frames <= 0:
            total_frames = 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if DISPLAY_SIZE is not None and width > 0 and height > 0:
            scale = min(display_w / float(width), display_h / float(height))
            scale = max(scale, 1e-6)
            render_w = max(1, int(round(width * scale)))
            render_h = max(1, int(round(height * scale)))
        else:
            render_w = display_w
            render_h = display_h

        base_win_w, base_win_h = render_w, render_h
        _apply_window_scale()

        print("-" * 60)
        if isinstance(video_path, int):
            print(f"目前來源 [{current_video_idx + 1}] [Camera {video_path}]")
        elif is_stream:
            print(f"目前來源 [{current_video_idx + 1}] [Stream] {video_path}")
        else:
            try:
                print(f"目前來源 [{current_video_idx + 1}] {video_path.name}")
            except Exception:
                print(f"目前來源 [{current_video_idx + 1}] {video_path}")

        if total_frames > 0:
            print(f"解析度: {width}x{height}, source_fps={source_fps:.2f}, model_fps={model_input_fps:.2f}, total={total_frames}")
        else:
            print(f"解析度: {width}x{height}, source_fps={source_fps:.2f}, model_fps={model_input_fps:.2f}, 類型=串流/未知")

        switch_delta = 0
        raw_frame_idx = 0
        ema_kpts = None
        keypoints_buffer = deque(maxlen=SEQUENCE_LENGTH)

        while True:
            ret, frame = cap.read()
            if not ret:
                if LOOP_PLAYBACK:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    raw_frame_idx = 0
                    ema_kpts = None
                    continue
                break

            raw_frame_idx += 1
            if frame_step > 1 and ((raw_frame_idx - 1) % frame_step != 0):
                continue

            kpts, kpt_conf, bbox, bbox_conf = detector.detect(frame)

            if DISPLAY_WINDOW and DISPLAY_SIZE is not None:
                interpolation = cv2.INTER_CUBIC if (render_w > width or render_h > height) else cv2.INTER_AREA
                display = cv2.resize(frame, (render_w, render_h), interpolation=interpolation)
                sx = render_w / max(width, 1)
                sy = render_h / max(height, 1)
            else:
                display = frame.copy()
                sx = 1.0
                sy = 1.0

            if kpts is not None and kpt_conf is not None:
                if EMA_ALPHA >= 1.0 - 1e-9:
                    ema_kpts = kpts
                else:
                    if ema_kpts is None:
                        ema_kpts = kpts.copy()
                    else:
                        ema_kpts = EMA_ALPHA * kpts + (1.0 - EMA_ALPHA) * ema_kpts
                    kpts = ema_kpts

                draw_styled_skeleton(display, kpts, kpt_conf, bbox, bbox_conf, sx, sy, _ui_scale())

                # append smoothed keypoints into buffer for velocity overlay
                try:
                    keypoints_buffer.append((kpts.copy(), kpt_conf.copy()))
                except Exception:
                    # defensive: ensure types are numpy arrays
                    keypoints_buffer.append((np.asarray(kpts, dtype=np.float32), np.asarray(kpt_conf, dtype=np.float32)))

                # compute velocity overlay when buffer full
                velocity_ovl = None
                if len(keypoints_buffer) >= SEQUENCE_LENGTH:
                    kpts_arr = np.array([item[0] for item in keypoints_buffer])  # T x J x 2
                    conf_arr = np.array([item[1] for item in keypoints_buffer])  # T x J
                    velocity_ovl = compute_velocity_overlay(kpts_arr, conf_arr, conf_threshold=VELOCITY_CONF_THRESHOLD)

                # draw per-joint velocity heat rings on top of skeleton
                if velocity_ovl is not None:
                    recent = velocity_ovl.get('recent_speed')
                    max_speed = max(float(velocity_ovl.get('peak_speed', 1e-6)), 1e-6)
                    for j in VELOCITY_HIGHLIGHT_JOINTS:
                        if j >= len(kpts):
                            continue
                        if float(kpt_conf[j]) <= DRAW_KP_CONF_THRESHOLD:
                            continue
                        cx = int(kpts[j][0] * sx)
                        cy = int(kpts[j][1] * sy)
                        sp = float(recent[j]) if j < len(recent) else 0.0
                        col = _velocity_heat_color(sp, max_speed)
                        r = max(5, int(6 * _ui_scale()))
                        cv2.circle(display, (cx, cy), r, col, -1)
                        cv2.circle(display, (cx, cy), r, (240, 240, 240), 2, cv2.LINE_AA)

                    # draw info panel
                    draw_velocity_overlay_panel(display, velocity_ovl, KEYPOINT_NAMES)

            hind_left_norm_dist, hind_right_norm_dist, scratch_paw_idx, scratch_raw_dist, body_len = compute_normalized_hind_paw_distances(kpts, kpt_conf)
            scratch_hit = (
                (np.isfinite(hind_left_norm_dist) and hind_left_norm_dist < SCRATCH_DISTANCE_THRESHOLD_NORM)
                or (np.isfinite(hind_right_norm_dist) and hind_right_norm_dist < SCRATCH_DISTANCE_THRESHOLD_NORM)
            )

            ui_scale = _ui_scale()
            time_sec = raw_frame_idx / source_fps if source_fps > 0 else 0.0
            time_mm = int(time_sec // 60)
            time_ss = int(time_sec % 60)
            time_label = f"TIME {time_mm:02d}:{time_ss:02d}"
            time_fs = 0.80 * ui_scale
            time_th = max(1, int(2 * ui_scale))
            time_x = int(12 * ui_scale)
            time_y = int(22 * ui_scale)
            cv2.putText(display, time_label, (time_x, time_y), cv2.FONT_HERSHEY_SIMPLEX, time_fs, (0, 0, 0), time_th + 1, cv2.LINE_AA)
            cv2.putText(display, time_label, (time_x, time_y), cv2.FONT_HERSHEY_SIMPLEX, time_fs, (255, 250, 140), time_th, cv2.LINE_AA)

            source_label = f"SRC {current_video_idx + 1}/{len(video_paths)}"
            src_fs = 0.52 * ui_scale
            src_th = max(1, int(2 * ui_scale))
            src_y = int(48 * ui_scale)
            cv2.putText(display, source_label, (time_x, src_y), cv2.FONT_HERSHEY_SIMPLEX, src_fs, (0, 0, 0), src_th + 1, cv2.LINE_AA)
            cv2.putText(display, source_label, (time_x, src_y), cv2.FONT_HERSHEY_SIMPLEX, src_fs, (220, 220, 220), src_th, cv2.LINE_AA)

            hl_text = "HL_NORM: N/A"
            hr_text = "HR_NORM: N/A"
            if np.isfinite(hind_left_norm_dist):
                hl_text = f"HL_NORM: {hind_left_norm_dist:.3f}"
            if np.isfinite(hind_right_norm_dist):
                hr_text = f"HR_NORM: {hind_right_norm_dist:.3f}"

            norm_text = f"{hl_text}  {hr_text}"

            status_text = "scratch" if scratch_hit else ""
            status_color = (0, 0, 255) if scratch_hit else (140, 255, 140)
            norm_box_x = int(render_w - 12 * ui_scale)
            norm_box_y = int(18 * ui_scale)
            norm_fs = 0.56 * ui_scale
            norm_th = max(1, int(2 * ui_scale))
            norm_shadow = norm_th + 1
            norm_size = cv2.getTextSize(norm_text, cv2.FONT_HERSHEY_SIMPLEX, norm_fs, norm_th)[0]
            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.88 * ui_scale, max(2, int(2 * ui_scale)))[0] if status_text else (0, 0)
            box_w = max(norm_size[0], status_size[0]) + int(20 * ui_scale)
            box_h = norm_size[1] + (status_size[1] + int(8 * ui_scale) if status_text else 0) + int(18 * ui_scale)
            x2 = norm_box_x
            y1 = norm_box_y
            x1 = max(0, x2 - box_w)
            y2 = y1 + box_h
            # remove solid background; draw border only to avoid blocking the scene
            cv2.rectangle(display, (x1, y1), (x2, y2), status_color if scratch_hit else (80, 80, 80), 1, cv2.LINE_AA)
            text_x = x1 + int(10 * ui_scale)
            text_y = y1 + int(18 * ui_scale) + norm_size[1]
            cv2.putText(display, norm_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, norm_fs, (0, 0, 0), norm_shadow, cv2.LINE_AA)
            cv2.putText(display, norm_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, norm_fs, (255, 255, 255), norm_th, cv2.LINE_AA)
            if status_text:
                status_y = text_y + int(8 * ui_scale) + status_size[1]
                cv2.putText(display, status_text, (text_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.88 * ui_scale, (0, 0, 0), norm_shadow + 1, cv2.LINE_AA)
                cv2.putText(display, status_text, (text_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.88 * ui_scale, status_color, max(2, int(2 * ui_scale)), cv2.LINE_AA)

            if DISPLAY_WINDOW:
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                action = _handle_key(key, cap_obj=cap, in_pause_loop=False)
                if action == "break":
                    break
                if is_paused:
                    paused_frame = display.copy()
                    while is_paused:
                        pause_img = paused_frame.copy()
                        cv2.putText(
                            pause_img,
                            "PAUSED (Space:Play, A/D:Step)",
                            (int(12 * ui_scale), int(render_h - 18 * ui_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55 * ui_scale,
                            (0, 255, 255),
                            max(1, int(2 * ui_scale)),
                            cv2.LINE_AA,
                        )
                        cv2.imshow(WINDOW_NAME, pause_img)
                        key2 = cv2.waitKey(50) & 0xFF
                        pause_action = _handle_key(key2, cap_obj=cap, in_pause_loop=True)
                        if pause_action == "seek":
                            # keep paused, but let the outer loop fetch/render the new frame immediately
                            break
                        if pause_action == "break":
                            break
                        if pause_action == "handled" and not is_paused:
                            break
                    if stop_all or switch_delta != 0:
                        break
            else:
                if msvcrt is not None and msvcrt.kbhit():
                    key_raw = msvcrt.getch()
                    if key_raw in (b"q", b"Q"):
                        print("\n使用者中斷：q")
                        stop_all = True
                        break

        cap.release()

        if stop_all:
            break

        if switch_delta != 0:
            current_video_idx = (current_video_idx + switch_delta) % len(video_paths)
        else:
            current_video_idx = (current_video_idx + 1) % len(video_paths)
            if not LOOP_PLAYBACK and current_video_idx == 0:
                break

    if DISPLAY_WINDOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()