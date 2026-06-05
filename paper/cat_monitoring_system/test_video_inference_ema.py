"""
測試影片推論腳本（EMA 平滑版）- 使用指數移動平均對關鍵點座標平滑，提升穩定性
其餘功能與 test_video_inference.py 完全相同
"""
import sys
import os
import csv
import cv2
import numpy as np
import time
from pathlib import Path
from collections import deque
from collections import defaultdict
from typing import Iterable

# 加入系統路徑
# Ensure both the package folder and repository root are on sys.path so
# top-level modules like config.py can be imported when running this script
# from within the cat_monitoring_system folder.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from detectors.keypoint_detector import KeypointDetector
from detectors.behavior_classifier import BehaviorClassifier
from processors.visualizer import Visualizer
from models.stgcn_model import (
    interpolate_missing,
    flip_normalize,
    orientation_normalize,
    normalize_skeleton_coords,
    build_feature_tensor,
    get_in_channels_for_mode,
)
from utils.constants import (
    BEHAVIOR_CLASSES,
    BEHAVIOR_TEXT_MAP,
    BEHAVIOR_COLORS,
    LOW_CONF_ID,
    BLACK,
    COLOR_HEAD,
)
from utils.helpers import get_behavior_name
from config import BehaviorTrackingConfig as _BehaviorTrackingConfig

# ── 五個行為資料夾（按 z/x/c/v/b 切換）────────────────────────────────
_BASE = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存"
FOLDER_WALK    = rf"{_BASE}\walk"
FOLDER_LICK    = rf"{_BASE}\lick"
FOLDER_SCRATCH = rf"{_BASE}\scratch"
FOLDER_SHAKE   = rf"{_BASE}\shake"
FOLDER_STOP    = rf"{_BASE}\stop"

# 按鍵 → (資料夾路徑, 顯示名稱)
FOLDER_MAP = {
    'z': (FOLDER_WALK,    "WALK"),
    'x': (FOLDER_LICK,    "LICK"),
    'c': (FOLDER_SCRATCH, "SCRATCH"),
    'v': (FOLDER_SHAKE,   "SHAKE"),
    'b': (FOLDER_STOP,    "STOP"),
}
DEFAULT_FOLDER_KEY = 'z'   # 啟動時預設進入的資料夾

# VIDEO_PATHS 保留作備用（不使用 FOLDER_MAP 時可手動指定）
VIDEO_PATHS = []
YOLO_MODEL_PATH = r"C:\AI_Project\cat_pose\v11s_91.pt"
STGCN_MODEL_PATH = r"C:\Users\homec\Downloads\stgcn_best_022_xy_v_att_on.pth"
INFERENCE_DEVICE = 'cuda'
YOLO_IMGSZ = 640  # 與 YOLO 訓練尺寸一致
YOLO_CONF_THRESHOLD = 0.5
STGCN_NORMALIZE = True
SEQUENCE_LENGTH = 16
_raw_stgcn_mode = os.getenv("STGCN_FEATURE_MODE", "xy_v")
STGCN_FEATURE_MODE = str(_raw_stgcn_mode).strip().lower()
# Normalize legacy/variant feature-mode names to canonical nmes used by the STGCN module
# Canonical names: "xy_v", "xy_conf_v", "xy_conf_v_bone", "xy_conf_v_bone_bmotion"
_FEATURE_MODE_MAP = {
    "xyv": "xy_v",
    "xyv_conf": "xy_conf_v",
    "xyv_conf_bone": "xy_conf_v_bone",
    "xyv_conf_bone_bone_motion": "xy_conf_v_bone_bmotion",
    "xyv_conf_bone_bmotion": "xy_conf_v_bone_bmotion",
    # Some possible compact variants
    "xyvconf": "xy_conf_v",
    "xyvconfbone": "xy_conf_v_bone",
    "xyvconfbonebmotion": "xy_conf_v_bone_bmotion",
}
STGCN_FEATURE_MODE = _FEATURE_MODE_MAP.get(STGCN_FEATURE_MODE, STGCN_FEATURE_MODE)
# Use centralized config for behavior label confidence threshold
BEHAVIOR_MIN_CONFIDENCE = _BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD
TARGET_MODEL_FPS = 30.0  # 模型訓練/推論設計時基
ENABLE_FPS_DOWNSAMPLE = True  # 只要不是 30fps，就把模型時基統一到 30fps（高於則降採樣，低於則用 30fps 時基）
CLASSIFY_STRIDE = 2  # 每幾個處理幀做一次分類（1=每幀）
DISPLAY_WINDOW = True
WINDOW_NAME = "Cat Behavior Inference (EMA)"
DISPLAY_SIZE = (1080, 720)  # 視窗顯示解析度（寬, 高），設為 None 維持原始解析度
LOOP_PLAYBACK = True  # 是否循環播放
JITTER_CONF_THRESHOLD = 0.3  # 抖動統計只使用高於此信心值的關鍵點
REPORT_OUTPUT_PATH = r"C:\paper\output\inference_analysis_report_ema.csv"  # 最終 CSV 報告
RUN_MODE = 0  # 0: 啟動時選擇, 1: 只生成統計, 2: 只做視窗測試
JITTER_WARNING_THRESHOLD = 30.0  # 像素抖動警告閾值

# ===== 關鍵點顯示/統計門檻 =====
DRAW_KP_CONF_THRESHOLD = 0.25  # 畫骨架線段與關鍵點圓點用門檻（>此值才畫）
SHOW_PROBABILITY_BARS = False  # 關閉機率條可減少每幀繪圖負載

# ===== EMA 平滑設定 =====
# alpha 越大 → 越貼近原始偵測值（響應快、平滑少）
# alpha 越小 → 越平滑（延遲多、噪音少）
EMA_ALPHA = 1.0  # 須與 train_gcn.py 的 KP_EMA_ALPHA 保持一致

# 17 關鍵點名稱映射（根據 YOLO-Pose v11 cat skeleton）
KEYPOINT_NAMES = [
    "Nose",           # 0: 鼻尖
    "Left_Ear",       # 1: 左耳
    "Right_Ear",      # 2: 右耳
    "Chest",          # 3: 前胸
    "Mid_Back",       # 4: 中背
    "Hip",            # 5: 髖部
    "LF_Elbow",       # 6: 左前肢肘
    "LF_Paw",         # 7: 左前肢掌
    "RF_Elbow",       # 8: 右前肢肘
    "RF_Paw",         # 9: 右前肢掌
    "LH_Knee",        # 10: 左後肢膝
    "LH_Paw",         # 11: 左後肢掌
    "RH_Knee",        # 12: 右後肢膝
    "RH_Paw",         # 13: 右後肢掌
    "Tail_Root",      # 14: 尾根
    "Tail_Mid",       # 15: 尾中
    "Tail_Tip",       # 16: 尾尖
]

# 17 關鍵點中文名稱
KEYPOINT_NAMES_ZH = [
    "鼻子",              # 0: nose
    "左耳尖",           # 1: left_ear_tip
    "右耳尖",           # 2: right_ear_tip
    "胸口",             # 3: 前胸（前肢附著點）
    "中背",             # 4: 身體中背
    "臀部",             # 5: hip
    "左前腿肘部",       # 6: left_front_elbow
    "左前爪",           # 7: left_front_paw
    "右前腿肘部",       # 8: right_front_elbow
    "右前爪",           # 9: right_front_paw
    "左後腿膝部",       # 10: left_hind_knee
    "左後爪",           # 11: left_hind_paw
    "右後腿膝部",       # 12: right_hind_knee
    "右後爪",           # 13: right_hind_paw
    "尾巴根部",         # 14: tail_base
    "尾巴中段",         # 15: tail_mid
    "尾巴尖端",         # 16: tail_tip
]

# ===== test2.py 骨架視覺樣式 =====
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

BEHAVIOR_PANEL_LABELS = tuple(str(name).upper() for name in BEHAVIOR_CLASSES)

SUPPORTED_VIDEO_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"
}


def _is_stream_url(path_str: str) -> bool:
    """判斷是否為 IP 串流 URL。"""
    lowered = str(path_str).lower()
    return lowered.startswith(("http://", "https://", "rtsp://", "rtsps://", "rtmp://"))


def open_video_capture_with_retry(path, retries=5, delay=3):
    """嘗試多次開啟串流來源，若成功回傳 cv2.VideoCapture，否則回傳 None。"""
    for attempt in range(retries):
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            return cap
        try:
            cap.release()
        except Exception:
            pass
        print(f"⚠ 無法開啟串流或影片 {path} (嘗試 {attempt+1}/{retries})，{delay}秒後重試...")
        time.sleep(delay)
    return None


def resolve_video_paths(video_sources: Iterable[str]):
    """將來源清單展開成影片檔路徑；來源可為影片檔或資料夾。"""
    resolved = []
    seen = set()

    for src in video_sources:
        if _is_stream_url(src):
            key = str(src).strip().lower()
            if key not in seen:
                seen.add(key)
                resolved.append(str(src).strip())
            continue

        p = Path(src).expanduser()

        if p.is_file():
            if p.suffix.lower() in SUPPORTED_VIDEO_EXTS:
                key = str(p.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    resolved.append(str(p))
            else:
                print(f"⚠ 非支援影片副檔名，略過: {p}")
            continue

        if p.is_dir():
            matched = sorted(
                [
                    f for f in p.rglob("*")
                    if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTS
                ]
            )
            if not matched:
                print(f"⚠ 資料夾內未找到影片，略過: {p}")
            for f in matched:
                key = str(f.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    resolved.append(str(f))
            continue

        print(f"⚠ 路徑不存在，略過: {p}")

    return resolved


def compute_ui_scale(width, height, base_width=1920.0, base_height=1080.0):
    """依影像對角線估算 UI 縮放，讓不同解析度下 overlay 視覺一致。"""
    diag = float(np.hypot(max(1.0, float(width)), max(1.0, float(height))))
    base_diag = float(np.hypot(base_width, base_height))
    scale = diag / max(base_diag, 1.0)
    return float(np.clip(scale, 0.65, 2.4))


def scale_px(value, ui_scale, min_px=1):
    """將像素值依 UI 縮放後取整，並限制最小值。"""
    return max(int(min_px), int(round(float(value) * float(ui_scale))))


def resize_with_letterbox(image, target_size):
    """等比例縮放並裁切成目標尺寸（無黑邊）。"""
    target_w, target_h = target_size
    src_h, src_w = image.shape[:2]

    if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
        return cv2.resize(image, target_size), 1.0, 0, 0

    scale = max(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    crop_x = max(0, (new_w - target_w) // 2)
    crop_y = max(0, (new_h - target_h) // 2)
    cropped = resized[crop_y:crop_y + target_h, crop_x:crop_x + target_w]
    return cropped, scale, crop_x, crop_y


def scale_kpts_and_bbox_for_letterbox(kpts, bbox, scale, crop_x, crop_y):
    """將原圖座標映射到滿版裁切後的顯示座標。"""
    scaled_kpts = kpts * scale - np.array([crop_x, crop_y], dtype=np.float32)
    scaled_bbox = None
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        scaled_bbox = np.array([
            x1 * scale - crop_x,
            y1 * scale - crop_y,
            x2 * scale - crop_x,
            y2 * scale - crop_y,
        ], dtype=np.float32)
    return scaled_kpts, scaled_bbox


def draw_no_cat_overlay(frame, text="No cat detected"):
    """依畫面解析度自適應繪製無偵測提示文字。"""
    h, w = frame.shape[:2]
    ui_scale = compute_ui_scale(w, h)
    x = scale_px(12, ui_scale, min_px=8)
    y = scale_px(34, ui_scale, min_px=20)
    font_scale = 0.62 * ui_scale
    outline = scale_px(3, ui_scale, min_px=2)
    thickness = scale_px(1, ui_scale, min_px=1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    return frame


def draw_behavior_duration_panel(frame, elapsed_sec, behavior_duration_sec, behavior_current_confidences=None):
    """高對比行為信心值面板：顯示四類行為的當下信心值百分比。"""
    h, w = frame.shape[:2]
    ui_scale = compute_ui_scale(w, h) * 1.10

    left = scale_px(8, ui_scale, min_px=4)
    right = scale_px(8, ui_scale, min_px=4)
    bottom = scale_px(6, ui_scale, min_px=3)

    title = "ST-GCN Behavior Confidence"
    timer = f"TIMER {float(elapsed_sec):7.2f}s"
    title_fs = 0.60 * ui_scale
    meta_fs = 0.56 * ui_scale
    row_fs = 0.52 * ui_scale
    pct_fs = 0.46 * ui_scale
    text_th = scale_px(2, ui_scale, min_px=1)
    shadow_th = scale_px(2, ui_scale, min_px=2)

    row_h = scale_px(28, ui_scale, min_px=18)
    header_h = scale_px(42, ui_scale, min_px=26)
    row_count = len(BEHAVIOR_PANEL_LABELS)
    panel_h = header_h + row_h * row_count
    panel_top = max(scale_px(2, ui_scale, min_px=1), h - panel_h - bottom)

    tx = left
    ty = panel_top + scale_px(16, ui_scale, min_px=12)
    cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, title_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
    cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, title_fs, (255, 245, 180), text_th, cv2.LINE_AA)

    timer_y = ty + scale_px(16, ui_scale, min_px=10)
    cv2.putText(frame, timer, (tx, timer_y), cv2.FONT_HERSHEY_SIMPLEX, meta_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
    cv2.putText(frame, timer, (tx, timer_y), cv2.FONT_HERSHEY_SIMPLEX, meta_fs, (170, 250, 255), text_th, cv2.LINE_AA)

    label_w = 0
    for label in BEHAVIOR_PANEL_LABELS:
        tw, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, row_fs, text_th)[0]
        label_w = max(label_w, tw)

    conf_w = cv2.getTextSize("100.0%", cv2.FONT_HERSHEY_SIMPLEX, row_fs, text_th)[0][0]
    pct_w = cv2.getTextSize("100.0%", cv2.FONT_HERSHEY_SIMPLEX, pct_fs, text_th)[0][0]

    col_gap = scale_px(8, ui_scale, min_px=4)
    conf_x = tx + label_w + col_gap
    bar_x = conf_x + conf_w + col_gap
    bar_h = scale_px(12, ui_scale, min_px=8)
    bar_w = max(scale_px(60, ui_scale, min_px=48), w - right - pct_w - col_gap - bar_x)

    row_y0 = panel_top + header_h

    for bid, label in enumerate(BEHAVIOR_PANEL_LABELS):
        # 取得當下信心值
        if behavior_current_confidences is not None and bid < len(behavior_current_confidences):
            pct = float(np.clip(behavior_current_confidences[bid], 0.0, 1.0))  # 已是 0-1 範圍
        else:
            pct = 0.0
        
        color = BEHAVIOR_COLORS.get(bid, (130, 230, 255))

        line_top = row_y0 + bid * row_h
        baseline_y = line_top + scale_px(14, ui_scale, min_px=9)

        cv2.putText(frame, label, (tx, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, color, text_th, cv2.LINE_AA)

        conf_text = f"{pct * 100.0:5.1f}%"
        cv2.putText(frame, conf_text, (conf_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, conf_text, (conf_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (235, 235, 235), text_th, cv2.LINE_AA)

        bar_top = line_top + scale_px(1, ui_scale, min_px=0)
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_top + bar_h), (78, 78, 78), -1)
        fill_w = int(round(bar_w * float(np.clip(pct, 0.0, 1.0))))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_top), (bar_x + fill_w, bar_top + bar_h), color, -1)
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_top + bar_h), (120, 120, 120), scale_px(1, ui_scale, min_px=1))

        pct_text = f"{pct * 100.0:5.1f}%"
        pct_x = bar_x + bar_w + col_gap
        pct_y = baseline_y
        cv2.putText(frame, pct_text, (pct_x, pct_y), cv2.FONT_HERSHEY_SIMPLEX, pct_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, pct_text, (pct_x, pct_y), cv2.FONT_HERSHEY_SIMPLEX, pct_fs, (220, 220, 220), text_th, cv2.LINE_AA)

    return frame





def draw_test2_style_overlay(
    frame,
    kpts,
    kpt_conf,
    bbox,
    behavior_id,
    confidence,
    probs,
    visualizer,
    show_info=True,
    conf_thresh=DRAW_KP_CONF_THRESHOLD,
):
    """使用 test2.py 的骨架外觀，並沿用既有行為資訊 HUD。"""
    h, w = frame.shape[:2]
    ui_scale = compute_ui_scale(w, h)
    bbox_thickness = scale_px(1, ui_scale, min_px=1)
    edge_thickness = scale_px(2, ui_scale, min_px=1)
    kp_outer_radius = scale_px(4, ui_scale, min_px=2)
    kp_inner_radius = max(1, kp_outer_radius - 1)

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        outer_w = 4
        inner_w = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), BLACK, outer_w, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HEAD, inner_w, cv2.LINE_AA)

    for ei, (a, b) in enumerate(_SKELETON_EDGES):
        # 骨架線段：兩端關鍵點都要高於顯示門檻才畫
        if float(kpt_conf[a]) > conf_thresh and float(kpt_conf[b]) > conf_thresh:
            pa = (int(kpts[a][0]), int(kpts[a][1]))
            pb = (int(kpts[b][0]), int(kpts[b][1]))
            col = _EDGE_COLORS[ei] if ei < len(_EDGE_COLORS) else (180, 180, 180)
            cv2.line(frame, pa, pb, col, edge_thickness, cv2.LINE_AA)

    for i in range(min(17, len(kpts))):
        # 關鍵點圓點：該點信心高於顯示門檻才畫
        if float(kpt_conf[i]) > conf_thresh:
            cx, cy = int(kpts[i][0]), int(kpts[i][1])
            col = _KP_COLORS[i] if i < len(_KP_COLORS) else (200, 200, 200)
            cv2.circle(frame, (cx, cy), kp_outer_radius, (0, 0, 0), -1)
            cv2.circle(frame, (cx, cy), kp_inner_radius, col, -1)

    if not show_info:
        return frame

    is_display_normal = (behavior_id == LOW_CONF_ID) or (float(confidence) < BEHAVIOR_MIN_CONFIDENCE)
    if is_display_normal:
        visualizer.draw_prediction_on_frame(
            frame,
            'Normal',
            0.0,
            (200, 200, 200),
            show_confidence=True,
            emphasize_label=False,
            label_background=False,
            font_scale_override=0.86,
        )
        # Always render the probability bars when enabled to avoid missing
        # bars after replaying a video even if values are temporarily 0.
        if SHOW_PROBABILITY_BARS and probs is not None:
            pb = probs if (hasattr(probs, '__len__') and len(probs) == len(BEHAVIOR_CLASSES)) else np.zeros(len(BEHAVIOR_CLASSES), dtype=np.float32)
            visualizer.draw_probability_bars(frame, pb, BEHAVIOR_CLASSES)
    elif behavior_id is not None and confidence > 0:
        behavior_name = get_behavior_name(behavior_id, use_text=False, fallback=str(behavior_id), confidence=confidence)
        visualizer.draw_prediction_on_frame(
            frame,
            behavior_name,
            confidence,
            BEHAVIOR_COLORS.get(behavior_id, (255, 255, 255)),
            show_confidence=True,
            emphasize_label=False,
            label_background=False,
            font_scale_override=0.86,
        )
        if SHOW_PROBABILITY_BARS:
            visualizer.draw_probability_bars(frame, probs if probs is not None else np.zeros(4, dtype=np.float32), BEHAVIOR_CLASSES)

    return frame


def print_jitter_report(title, jitter_px, jitter_norm, valid_counts, pair_counts):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    all_px = [v for arr in jitter_px for v in arr]
    all_norm = [v for arr in jitter_norm for v in arr]
    total_valid = int(np.sum(valid_counts))
    total_pairs = int(np.sum(pair_counts))

    if not all_px:
        print("無足夠資料計算抖動（可能關鍵點信心不足或連續幀不足）")
        return

    print("[全域抖動指標]")
    print(f"  樣本數(像素): {len(all_px)}")
    print(f"  平均: {np.mean(all_px):.3f} px")
    print(f"  標準差: {np.std(all_px):.3f} px")
    print(f"  P95: {np.percentile(all_px, 95):.3f} px")
    print(f"  最大值: {np.max(all_px):.3f} px")
    print(f"  有效關鍵點數: {total_valid}")
    print(f"  連續可比較配對數: {total_pairs}")

    if all_norm:
        print(f"  正規化平均(除以bbox對角線): {np.mean(all_norm):.5f}")
        print(f"  正規化P95: {np.percentile(all_norm, 95):.5f}")

    print("\n[17關鍵點逐點統計]")
    print("  idx | valid | pairs | mean_px | std_px | p95_px | max_px | mean_norm")
    for i in range(17):
        if jitter_px[i]:
            mean_px = np.mean(jitter_px[i])
            std_px = np.std(jitter_px[i])
            p95_px = np.percentile(jitter_px[i], 95)
            max_px = np.max(jitter_px[i])
        else:
            mean_px = std_px = p95_px = max_px = 0.0

        mean_norm = np.mean(jitter_norm[i]) if jitter_norm[i] else 0.0

        print(
            f"  {i:>3d} | {int(valid_counts[i]):>5d} | {int(pair_counts[i]):>5d} | "
            f"{mean_px:>7.3f} | {std_px:>6.3f} | {p95_px:>6.3f} | {max_px:>6.3f} | {mean_norm:>9.5f}"
        )


def generate_report_file(report_path, recorded_video_stats):
    """輸出 CSV 統計摘要（每列一部影片）。"""
    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "video_idx",
        "video_path",
        "width",
        "height",
        "source_fps",
        "model_input_fps",
        "frame_step",
        "total_frames",
        "processed_frames",
        "frames_with_cat",
        "frames_without_cat",
        "pred_walk",
        "pred_lick",
        "pred_scratch",
        "pred_shake",
        "pred_stop",
        "duration_walk_sec",
        "duration_lick_sec",
        "duration_scratch_sec",
        "duration_shake_sec",
        "duration_stop_sec",
        "mean_confidence",
        "jitter_mean_px",
        "jitter_p95_px",
        "jitter_max_px",
    ]

    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for vid_idx in sorted(recorded_video_stats.keys()):
            s = recorded_video_stats[vid_idx]
            behavior_counts = np.asarray(s.get("behavior_counts", np.zeros(4, dtype=np.int64)), dtype=np.int64)
            behavior_duration_sec = np.asarray(s.get("behavior_duration_sec", np.zeros(4, dtype=np.float64)), dtype=np.float64)
            confidences = s.get("behavior_confidences", [])
            jp = s.get("jitter_px", [[] for _ in range(17)])
            all_jitter = [v for arr in jp for v in arr]

            writer.writerow([
                int(s.get("video_idx", vid_idx)),
                s.get("video_path", ""),
                int(s.get("width", 0)),
                int(s.get("height", 0)),
                float(s.get("fps", 0.0)),
                float(s.get("model_input_fps", 0.0)),
                int(s.get("frame_step", 1)),
                int(s.get("total_frames", 0)),
                int(s.get("processed_frames", 0)),
                int(s.get("frames_with_cat", 0)),
                int(s.get("frames_without_cat", 0)),
                int(behavior_counts[0]) if len(behavior_counts) > 0 else 0,
                int(behavior_counts[1]) if len(behavior_counts) > 1 else 0,
                int(behavior_counts[2]) if len(behavior_counts) > 2 else 0,
                int(behavior_counts[3]) if len(behavior_counts) > 3 else 0,
                int(behavior_counts[4]) if len(behavior_counts) > 4 else 0,
                float(behavior_duration_sec[0]) if len(behavior_duration_sec) > 0 else 0.0,
                float(behavior_duration_sec[1]) if len(behavior_duration_sec) > 1 else 0.0,
                float(behavior_duration_sec[2]) if len(behavior_duration_sec) > 2 else 0.0,
                float(behavior_duration_sec[3]) if len(behavior_duration_sec) > 3 else 0.0,
                float(behavior_duration_sec[4]) if len(behavior_duration_sec) > 4 else 0.0,
                float(np.mean(confidences)) if confidences else 0.0,
                float(np.mean(all_jitter)) if all_jitter else 0.0,
                float(np.percentile(all_jitter, 95)) if all_jitter else 0.0,
                float(np.max(all_jitter)) if all_jitter else 0.0,
            ])

    return out_path


def resolve_run_mode():
    if RUN_MODE in (1, 2):
        return RUN_MODE

    if not sys.stdin.isatty():
        print("\n偵測到非互動式輸入環境，預設使用模式 2（只測試模型效果，開視窗）")
        return 2

    print("\n請選擇執行模式:")
    print("  1) 只生成統計結果（不開視窗）")
    print("  2) 只測試模型效果（開視窗）")
    try:
        choice = input("輸入模式 (1/2, 預設=2): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n未輸入模式，預設使用模式 2（只測試模型效果，開視窗）")
        return 2

    if choice == "1":
        return 1
    return 2

def main():
    run_mode = resolve_run_mode()
    is_stats_mode = (run_mode == 1)
    is_test_mode = (run_mode == 2)

    # use a local mutable copy to avoid modifying module-level constant
    feature_mode = STGCN_FEATURE_MODE

    # 解析所有資料夾的影片清單（啟動時一次完成）
    folder_videos: dict = {}
    for fkey, (fpath, fname) in FOLDER_MAP.items():
        vids = resolve_video_paths([fpath])
        folder_videos[fkey] = vids
        print(f"  [{fkey}] {fname}: {len(vids)} 部影片  ({fpath})")

    # 若指定了 VIDEO_PATHS 就用那個；否則從 DEFAULT_FOLDER_KEY 資料夾開始
    if VIDEO_PATHS:
        video_paths = resolve_video_paths(VIDEO_PATHS)
        current_folder_key = DEFAULT_FOLDER_KEY
    else:
        current_folder_key = DEFAULT_FOLDER_KEY
        video_paths = folder_videos[current_folder_key]

    if not video_paths:
        print("❌ 找不到可用影片，請確認 FOLDER_MAP / VIDEO_PATHS 的路徑")
        return

    # 記住每個資料夾上次的播放位置（切回去時能續播）
    folder_positions: dict = {k: 0 for k in FOLDER_MAP}
    switch_folder_key: str = ""   # 非空時代表要切換資料夾

    display_window = DISPLAY_WINDOW and is_test_mode
    loop_playback = LOOP_PLAYBACK and is_test_mode

    print("="*60)
    print("影片推論測試（EMA 平滑版）")
    print("="*60)
    print(f"執行模式: {'模式1-統計分析' if is_stats_mode else '模式2-視窗測試'}")
    print(f"EMA Alpha: {EMA_ALPHA}")
    print(f"影片路徑 (展開後共 {len(video_paths)} 部):")
    for i, p in enumerate(video_paths):
        print(f"  [{i}] {p}")
    print(f"分類步長 CLASSIFY_STRIDE: {CLASSIFY_STRIDE}")
    print(f"序列長度: {SEQUENCE_LENGTH}")
    print("="*60)

    # 初始化偵測器
    print("\n初始化模型...")
    print(f"特徵模式: {feature_mode}")
    # 嘗試讀取 checkpoint 的 bn_input 通道數，若與目前 feature mode 不符，
    # 盡量自動將 feature mode 換成與 checkpoint 通道數相對應的 canonical 模式。
    in_channels = None
    try:
        ck_channel_map = {4: 'xy_v', 5: 'xy_conf_v', 7: 'xy_conf_v_bone', 9: 'xy_conf_v_bone_bmotion'}
        import torch
        if os.path.exists(STGCN_MODEL_PATH):
            try:
                ck = torch.load(STGCN_MODEL_PATH, map_location='cpu')
                state_dict = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
                if isinstance(state_dict, dict) and 'bn_input.weight' in state_dict:
                    ck_in_ch = int(state_dict['bn_input.weight'].shape[0])
                    try:
                        expected_ch = get_in_channels_for_mode(feature_mode)
                    except Exception:
                        expected_ch = None
                    if expected_ch is not None and ck_in_ch != expected_ch:
                        if ck_in_ch in ck_channel_map:
                            new_mode = ck_channel_map[ck_in_ch]
                            print(f"⚠ 模型檔案 {STGCN_MODEL_PATH} 的 bn_input channels={ck_in_ch}，與目前 feature_mode={feature_mode} 不符。")
                            print(f"  → 自動將 feature_mode 調整為 {new_mode} 以匹配 checkpoint。")
                            feature_mode = new_mode
                        else:
                            print(f"⚠ 模型檔案 {STGCN_MODEL_PATH} 的 bn_input channels={ck_in_ch}，無對應 canonical feature mode，將以該 channel 數為主。")
                    in_channels = ck_in_ch
            except Exception as e:
                print(f"⚠ 無法載入 checkpoint 以推斷通道數: {e}")
    except Exception:
        # torch 或其他步驟失敗時，退回到使用 get_in_channels_for_mode
        pass

    if in_channels is None:
        in_channels = get_in_channels_for_mode(feature_mode)
    
    keypoint_detector = KeypointDetector(
        YOLO_MODEL_PATH,
        device=INFERENCE_DEVICE,
        imgsz=YOLO_IMGSZ,
        conf_thres=YOLO_CONF_THRESHOLD,
    )
    behavior_classifier = BehaviorClassifier(
        STGCN_MODEL_PATH,
        device=INFERENCE_DEVICE,
        sequence_length=SEQUENCE_LENGTH,
        normalize=STGCN_NORMALIZE,
        feature_mode=feature_mode,
        in_channels=in_channels,
    )
    visualizer = Visualizer()

    # 統計累計（僅計入完整播放完成的影片）
    frame_count = 0
    predictions = []
    behavior_change_count = 0
    frames_with_cat = 0
    frames_without_cat = 0

    # 17點抖動統計（跨影片）
    global_jitter_px = [[] for _ in range(17)]
    global_jitter_norm = [[] for _ in range(17)]
    global_valid_counts = np.zeros(17, dtype=np.int64)
    global_pair_counts = np.zeros(17, dtype=np.int64)
    global_behavior_duration_sec = np.zeros(5, dtype=np.float64)

    # 每影片抖動統計
    per_video_stats = defaultdict(
        lambda: {
            "jitter_px": [[] for _ in range(17)],
            "jitter_norm": [[] for _ in range(17)],
            "valid_counts": np.zeros(17, dtype=np.int64),
            "pair_counts": np.zeros(17, dtype=np.int64),
        }
    )

    # 完整播完才會寫入的每影片最終統計
    recorded_video_stats = {}

    # 狀態控制
    paused = False
    stop_requested = False
    current_video_idx = 0
    show_overlay_info = True

    # 即時顯示狀態
    behavior_id = LOW_CONF_ID
    confidence = 0.0
    probs = np.zeros(5, dtype=np.float32)

    def reset_behavior_display_state():
        nonlocal behavior_id, confidence, probs
        behavior_id = LOW_CONF_ID
        confidence = 0.0
        probs = np.zeros(5, dtype=np.float32)

    def reset_video_runtime_state():
        nonlocal prev_kpts, prev_kpt_conf, ema_kpts
        nonlocal local_predictions, local_behavior_change_count, local_last_behavior
        nonlocal raw_frames_read, local_frames_processed, local_sampled_frames
        nonlocal local_frames_with_cat, local_frames_without_cat
        nonlocal local_jitter_px, local_jitter_norm, local_valid_counts, local_pair_counts
        nonlocal local_behavior_duration_sec, local_behavior_current_confidences

        keypoints_buffer.clear()
        prev_kpts = None
        prev_kpt_conf = None
        ema_kpts = None
        local_predictions = []
        local_behavior_change_count = 0
        local_last_behavior = None
        raw_frames_read = 0
        local_frames_processed = 0
        local_sampled_frames = 0
        local_frames_with_cat = 0
        local_frames_without_cat = 0
        local_jitter_px = [[] for _ in range(17)]
        local_jitter_norm = [[] for _ in range(17)]
        local_valid_counts = np.zeros(17, dtype=np.int64)
        local_pair_counts = np.zeros(17, dtype=np.int64)
        local_behavior_duration_sec = np.zeros(5, dtype=np.float64)
        local_behavior_current_confidences = np.zeros(5, dtype=np.float32)
        reset_behavior_display_state()

    if display_window:
        if DISPLAY_SIZE is not None:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    while not stop_requested:
        video_path = video_paths[current_video_idx]
        is_stream_url = _is_stream_url(video_path)
        if not is_stream_url and not Path(video_path).exists():
            print(f"❌ 影片不存在，跳過: {video_path}")
            if is_stats_mode:
                current_video_idx += 1
                if current_video_idx >= len(video_paths):
                    break
            else:
                current_video_idx = (current_video_idx + 1) % len(video_paths)
            continue

        if is_stream_url:
            cap = open_video_capture_with_retry(video_path, retries=5, delay=3)
            if cap is None or not cap.isOpened():
                print(f"❌ 無法開啟串流 {video_path}，請確認 URL 與網路連線，跳過")
                if is_stats_mode:
                    current_video_idx += 1
                    if current_video_idx >= len(video_paths):
                        break
                else:
                    current_video_idx = (current_video_idx + 1) % len(video_paths)
                continue
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ 無法開啟影片，跳過: {video_path}")
                if is_stats_mode:
                    current_video_idx += 1
                    if current_video_idx >= len(video_paths):
                        break
                else:
                    current_video_idx = (current_video_idx + 1) % len(video_paths)
                continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if is_stream_url and total_frames <= 0:
            total_frames = 0
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 1:
            source_fps = TARGET_MODEL_FPS
        frame_step = 1
        if ENABLE_FPS_DOWNSAMPLE and source_fps > TARGET_MODEL_FPS + 1e-6:
            frame_step = max(1, int(round(source_fps / TARGET_MODEL_FPS)))
        model_input_fps = TARGET_MODEL_FPS if ENABLE_FPS_DOWNSAMPLE else source_fps / frame_step

        if source_fps < TARGET_MODEL_FPS - 0.5:
            print(
                f"⚠ 來源影片 FPS={source_fps:.2f} 低於模型目標 {TARGET_MODEL_FPS:.2f}，"
                f"模型時基將統一視為 {TARGET_MODEL_FPS:.2f} fps"
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = (total_frames / source_fps) if source_fps > 0 else 0.0
        frame_dt = 1.0 / max(model_input_fps, 1e-6)

        print("\n" + "=" * 60)
        folder_name = FOLDER_MAP.get(current_folder_key, ("", "UNKNOWN"))[1]
        folder_hint = "  ".join(f"[{k}]{FOLDER_MAP[k][1]}" for k in FOLDER_MAP)
        print(f"資料夾: {folder_name} [{current_folder_key}]  |  {folder_hint}")
        print(f"目前影片 [{current_video_idx + 1}/{len(video_paths)}] {video_path}")
        print(f"影片資訊: {width}x{height}, source_fps={source_fps:.1f}, total={total_frames} 幀")
        print(f"模型輸入時基: {model_input_fps:.2f} fps (frame_step={frame_step})")
        print(f"時長: {duration:.1f} 秒")
        if is_test_mode:
            print("控制: q=退出  space=暫停  r=重置  1/2=上/下部  z/x/c/v/b=切換資料夾  i=資訊")
        if loop_playback:
            print("🔁 循環播放模式（當前影片播完會重播）")
        print("-" * 60)

        keypoints_buffer = deque(maxlen=SEQUENCE_LENGTH)
        local_loop_count = 0
        switch_delta = 0
        prev_kpts = None
        prev_kpt_conf = None
        first_pass_completed = False
        switched_before_first_pass_complete = False

        reset_behavior_display_state()

        # EMA 狀態：跨幀累積，切影片或貓消失時重置
        ema_kpts = None  # shape (17, 2)，儲存上一幀的 EMA 平滑座標

        # 本次影片臨時統計（只有完整第一輪才會被提交）
        local_predictions = []
        local_behavior_change_count = 0
        local_last_behavior = None
        raw_frames_read = 0
        local_frames_processed = 0
        local_sampled_frames = 0
        local_frames_with_cat = 0
        local_frames_without_cat = 0
        local_jitter_px = [[] for _ in range(17)]
        local_jitter_norm = [[] for _ in range(17)]
        local_valid_counts = np.zeros(17, dtype=np.int64)
        local_pair_counts = np.zeros(17, dtype=np.int64)
        local_behavior_duration_sec = np.zeros(5, dtype=np.float64)
        local_behavior_current_confidences = np.zeros(5, dtype=np.float32)

        while True:
            ret, frame = cap.read()
            if not ret:
                if is_stream_url:
                    print(f"⚠ 串流讀取失敗，嘗試重新連線: {video_path}")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = open_video_capture_with_retry(video_path, retries=3, delay=2)
                    if cap is not None and cap.isOpened():
                        print(f"✅ 串流已重連成功: {video_path}")
                        continue
                    print(f"❌ 串流無法重新開啟，跳過: {video_path}")
                    break
                # 影片播放完畢
                if loop_playback and not stop_requested:
                    if local_loop_count == 0:
                        first_pass_completed = True
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    reset_video_runtime_state()
                    local_loop_count += 1
                    print(f"\n🔁 影片 [{current_video_idx}] 循環播放第 {local_loop_count} 次...\n")
                    continue
                if local_loop_count == 0:
                    first_pass_completed = True
                break

            # 只記錄第一輪統計；後續循環僅供展示
            is_first_pass = (local_loop_count == 0)
            raw_frames_read += 1
            local_sampled_frames += 1

            if is_first_pass:
                local_frames_processed += 1

            frame_time_sec = raw_frames_read / source_fps if source_fps > 0 else 0.0

            # YOLO-Pose 偵測
            kpts, kpt_conf, bbox, _ = keypoint_detector.detect(frame)
            # velocity_overlay removed

            if kpts is not None:
                # ===== EMA 平滑：對 YOLO 偵測的原始座標做指數移動平均 =====
                # 初始化：第一幀直接使用原始值作為起始 EMA
                if ema_kpts is None:
                    ema_kpts = kpts.copy()
                else:
                    ema_kpts = EMA_ALPHA * kpts + (1.0 - EMA_ALPHA) * ema_kpts
                # 以下所有處理均使用平滑後的座標
                kpts = ema_kpts.copy()
                # ============================================================

                # kpts: (17, 2), kpt_conf: (17,)
                if is_first_pass:
                    local_frames_with_cat += 1

                # 統計有效關鍵點幀數
                valid_mask = (kpt_conf > JITTER_CONF_THRESHOLD)
                if is_first_pass:
                    local_valid_counts += valid_mask.astype(np.int64)

                # 計算 bbox 對角線供正規化抖動使用
                bbox_diag = None
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    w_box = max(1.0, float(x2 - x1))
                    h_box = max(1.0, float(y2 - y1))
                    bbox_diag = float(np.sqrt(w_box * w_box + h_box * h_box))

                # 計算逐點抖動（EMA 平滑後的座標，反映模型實際接收到的穩定度）
                if prev_kpts is not None and prev_kpt_conf is not None:
                    pair_mask = (kpt_conf > JITTER_CONF_THRESHOLD) & (prev_kpt_conf > JITTER_CONF_THRESHOLD)
                    for kp_idx in range(17):
                        if not pair_mask[kp_idx]:
                            continue

                        jitter_px = float(np.linalg.norm(kpts[kp_idx] - prev_kpts[kp_idx]))
                        if is_first_pass:
                            local_jitter_px[kp_idx].append(jitter_px)
                            local_pair_counts[kp_idx] += 1

                        if bbox_diag is not None and bbox_diag > 0:
                            jitter_norm = jitter_px / bbox_diag
                            if is_first_pass:
                                local_jitter_norm[kp_idx].append(jitter_norm)

                prev_kpts = kpts.copy()
                prev_kpt_conf = kpt_conf.copy()

                # 加入緩衝區
                keypoints_buffer.append((kpts, kpt_conf))

                # velocity overlay removed (erroneous edit)

                # 有足夠序列時做行為分類
                if len(keypoints_buffer) >= SEQUENCE_LENGTH and (local_sampled_frames % CLASSIFY_STRIDE == 0):
                    # 解包緩衝區
                    kpts_arr = np.array([item[0] for item in keypoints_buffer])  # (32, 17, 2)
                    conf_arr = np.array([item[1] for item in keypoints_buffer])  # (32, 17)

                    # 插值補全
                    seq_array = interpolate_missing(kpts_arr, conf_arr, threshold=0.1)
                    if STGCN_NORMALIZE:
                        seq_array = flip_normalize(seq_array)
                        seq_array = orientation_normalize(seq_array)
                        seq_array = normalize_skeleton_coords(seq_array)
                    seq_features = build_feature_tensor(seq_array, conf_arr, feature_mode)
                    pred_id, pred_conf, pred_probs = behavior_classifier.classify(seq_features, precomputed=True)
                    
                    if pred_id is None:
                        behavior_id = LOW_CONF_ID
                        confidence = 0.0
                        probs = np.zeros(5, dtype=np.float32)
                    else:
                        behavior_id = int(pred_id)
                        confidence = float(pred_conf)
                        probs = pred_probs.copy()

                    # 與主系統一致：低信心顯示「目前正常」
                    if confidence < BEHAVIOR_MIN_CONFIDENCE:
                        behavior_id_for_display = LOW_CONF_ID
                    else:
                        behavior_id_for_display = behavior_id

                    # 只統計高信心預測
                    if behavior_id_for_display != LOW_CONF_ID:
                        behavior_text = get_behavior_name(behavior_id, use_text=False, fallback=str(behavior_id), confidence=confidence)
                        if is_first_pass:
                            local_predictions.append({
                                'video_idx': current_video_idx,
                                'video_path': video_path,
                                'frame': local_frames_processed,
                                'time': frame_time_sec,
                                'behavior_id': behavior_id,
                                'behavior_name': BEHAVIOR_CLASSES[behavior_id],
                                'confidence': confidence,
                                'probs': probs.copy()
                            })
                        if local_last_behavior != behavior_id:
                            if local_last_behavior is not None and is_first_pass:
                                local_behavior_change_count += 1
                                probs_str = " ".join(
                                    f"{cls}:{probs[i]*100:4.1f}%"
                                    for i, cls in enumerate(BEHAVIOR_CLASSES)
                                    if i < len(probs)
                                )
                                print(f"影片[{current_video_idx}] 幀 {local_frames_processed:6d}: {behavior_text:6s} {confidence*100:5.1f}% [{probs_str}]")
                            local_last_behavior = behavior_id
                    else:
                        behavior_id = LOW_CONF_ID

                if is_first_pass and behavior_id != LOW_CONF_ID and 0 <= int(behavior_id) < 5 and float(confidence) >= BEHAVIOR_MIN_CONFIDENCE:
                    local_behavior_duration_sec[int(behavior_id)] += frame_dt
                    local_behavior_current_confidences[int(behavior_id)] = float(confidence)
            else:
                if is_first_pass:
                    local_frames_without_cat += 1
                prev_kpts = None
                prev_kpt_conf = None
                ema_kpts = None  # 貓消失時重置 EMA，避免下次出現時使用過時的平均值

            if display_window:
                if DISPLAY_SIZE is not None:
                    show_frame, preview_scale, preview_pad_x, preview_pad_y = resize_with_letterbox(frame, DISPLAY_SIZE)
                    if kpts is not None:
                        scaled_kpts, scaled_bbox = scale_kpts_and_bbox_for_letterbox(
                            kpts,
                            bbox,
                            preview_scale,
                            preview_pad_x,
                            preview_pad_y,
                        )
                        show_frame = draw_test2_style_overlay(
                            show_frame,
                            scaled_kpts,
                            kpt_conf,
                            scaled_bbox,
                            behavior_id,
                            confidence,
                            probs if len(probs) == 5 else np.zeros(5, dtype=np.float32),
                            visualizer,
                            show_info=show_overlay_info,
                        )
                    else:
                        draw_no_cat_overlay(show_frame)
                    if show_overlay_info:
                        draw_behavior_duration_panel(show_frame, frame_time_sec, local_behavior_duration_sec, local_behavior_current_confidences)
                else:
                    show_frame = frame.copy()
                    if kpts is not None:
                        show_frame = draw_test2_style_overlay(
                            show_frame,
                            kpts,
                            kpt_conf,
                            bbox,
                            behavior_id,
                            confidence,
                            probs if len(probs) == 5 else np.zeros(5, dtype=np.float32),
                            visualizer,
                            show_info=show_overlay_info,
                        )
                    else:
                        draw_no_cat_overlay(show_frame)
                    if show_overlay_info:
                        draw_behavior_duration_panel(show_frame, frame_time_sec, local_behavior_duration_sec, local_behavior_current_confidences)
                # 資料夾名稱 + 影片進度條（左上角）
                _fn  = FOLDER_MAP.get(current_folder_key, ("", "?"))[1]
                _nav = (f"[{current_folder_key.upper()}]{_fn}  "
                        f"{current_video_idx + 1}/{len(video_paths)}  "
                        f"| z WALK  x LICK  c SCRATCH  v SHAKE  b STOP")
                _h, _w = show_frame.shape[:2]
                _ui = compute_ui_scale(_w, _h)
                _fs = 0.42 * _ui
                _th = max(1, int(_ui))
                # Compute text size to ensure background rectangle fully covers label
                txt_size = cv2.getTextSize(_nav, cv2.FONT_HERSHEY_SIMPLEX, _fs, _th)[0]
                rect_h = max(int(txt_size[1] * 1.6), int(22 * _ui))
                # Draw a taller background rect so the label is never clipped
                cv2.rectangle(show_frame, (0, 0), (_w, rect_h), (10, 16, 30), -1)
                text_y = int(rect_h * 0.7)
                cv2.putText(show_frame, _nav, (6, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, _fs, (160, 210, 255), _th, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, show_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n使用者中斷：q")
                    stop_requested = True
                    break
                if key == ord('i'):
                    show_overlay_info = not show_overlay_info
                    print(f"\n資訊面板: {'顯示' if show_overlay_info else '隱藏'}")
                    continue
                if key == ord('2'):
                    switch_delta = 1
                    if not first_pass_completed:
                        switched_before_first_pass_complete = True
                    reset_video_runtime_state()
                    print("\n切換到下一部影片")
                    break
                if key == ord('1'):
                    switch_delta = -1
                    if not first_pass_completed:
                        switched_before_first_pass_complete = True
                    reset_video_runtime_state()
                    print("\n切換到上一部影片")
                    break
                # z/x/c/v/b 切換行為資料夾
                if chr(key & 0xFF) in FOLDER_MAP and chr(key & 0xFF) != current_folder_key:
                    switch_folder_key = chr(key & 0xFF)
                    if not first_pass_completed:
                        switched_before_first_pass_complete = True
                    reset_video_runtime_state()
                    print(f"\n切換資料夾 → {FOLDER_MAP[switch_folder_key][1]} [{switch_folder_key}]")
                    break
                if key == ord('r'):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    local_loop_count = 0
                    first_pass_completed = False
                    switched_before_first_pass_complete = False
                    reset_video_runtime_state()
                    print("\n↺ 已重置：回到影片開頭並清空偵測狀態")
                    continue
                if key == ord(' '):
                    paused = not paused
                    while paused:
                        k2 = cv2.waitKey(50) & 0xFF
                        if k2 == ord(' '):
                            paused = False
                        elif k2 == ord('q'):
                            paused = False
                            print("\n使用者中斷：q")
                            stop_requested = True
                            break
                        elif k2 == ord('i'):
                            show_overlay_info = not show_overlay_info
                            print(f"\n資訊面板: {'顯示' if show_overlay_info else '隱藏'}")
                        elif k2 == ord('2'):
                            paused = False
                            switch_delta = 1
                            if not first_pass_completed:
                                switched_before_first_pass_complete = True
                            reset_video_runtime_state()
                            print("\n切換到下一部影片")
                            break
                        elif k2 == ord('1'):
                            paused = False
                            switch_delta = -1
                            if not first_pass_completed:
                                switched_before_first_pass_complete = True
                            reset_video_runtime_state()
                            print("\n切換到上一部影片")
                            break
                        elif chr(k2 & 0xFF) in FOLDER_MAP and chr(k2 & 0xFF) != current_folder_key:
                            paused = False
                            switch_folder_key = chr(k2 & 0xFF)
                            if not first_pass_completed:
                                switched_before_first_pass_complete = True
                            reset_video_runtime_state()
                            print(f"\n切換資料夾 → {FOLDER_MAP[switch_folder_key][1]} [{switch_folder_key}]")
                            break
                        elif k2 == ord('r'):
                            paused = False
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            local_loop_count = 0
                            first_pass_completed = False
                            switched_before_first_pass_complete = False
                            reset_video_runtime_state()
                            print("\n↺ 已重置：回到影片開頭並清空偵測狀態")
                            break

            if stop_requested or switch_delta != 0:
                break

            # 降採樣時跳過後續 frame_step-1 幀，避免不必要的完整解碼
            if frame_step > 1:
                for _ in range(frame_step - 1):
                    if not cap.grab():
                        break
                    raw_frames_read += 1

            # 每 100 幀顯示進度（僅第一次循環顯示）
            if local_loop_count == 0 and local_frames_processed % 100 == 0:
                pct = (raw_frames_read / total_frames * 100) if total_frames > 0 else 0.0
                print(f"  影片[{current_video_idx}] 處理進度: {raw_frames_read}/{total_frames} ({pct:.1f}%)")

        cap.release()

        # 只有完整播放第一輪且非中途切換，才提交本影片統計
        if first_pass_completed and not switched_before_first_pass_complete:
            behavior_counts = np.zeros(5, dtype=np.int64)
            behavior_confidences = []
            for p in local_predictions:
                behavior_counts[p['behavior_id']] += 1
                behavior_confidences.append(p['confidence'])

            recorded_video_stats[current_video_idx] = {
                "video_idx": current_video_idx,
                "video_path": video_path,
                "width": width,
                "height": height,
                "fps": float(source_fps),
                "model_input_fps": float(model_input_fps),
                "frame_step": int(frame_step),
                "total_frames": int(total_frames),
                "processed_frames": int(local_frames_processed),
                "frames_with_cat": int(local_frames_with_cat),
                "frames_without_cat": int(local_frames_without_cat),
                "behavior_counts": behavior_counts,
                "behavior_confidences": behavior_confidences,
                "behavior_duration_sec": local_behavior_duration_sec.copy(),
                "jitter_px": local_jitter_px,
                "jitter_norm": local_jitter_norm,
                "valid_counts": local_valid_counts,
                "pair_counts": local_pair_counts,
            }

            # 合併到全域統計
            frame_count += local_frames_processed
            frames_with_cat += local_frames_with_cat
            frames_without_cat += local_frames_without_cat
            predictions.extend(local_predictions)
            behavior_change_count += local_behavior_change_count
            global_behavior_duration_sec += local_behavior_duration_sec

            for i in range(17):
                global_jitter_px[i].extend(local_jitter_px[i])
                global_jitter_norm[i].extend(local_jitter_norm[i])
            global_valid_counts += local_valid_counts
            global_pair_counts += local_pair_counts

            per_video_stats[current_video_idx] = {
                "jitter_px": local_jitter_px,
                "jitter_norm": local_jitter_norm,
                "valid_counts": local_valid_counts,
                "pair_counts": local_pair_counts,
            }
            print(f"✓ 影片[{current_video_idx}] 已完整播放，統計已記錄")
        else:
            if switched_before_first_pass_complete:
                print(f"⚠ 影片[{current_video_idx}] 中途切換，該影片統計不記錄")
            else:
                print(f"⚠ 影片[{current_video_idx}] 未完成第一輪播放，該影片統計不記錄")

        if stop_requested:
            break

        # 資料夾切換（z/x/c/v/b）：儲存目前位置後切換到新資料夾
        if switch_folder_key:
            folder_positions[current_folder_key] = current_video_idx
            current_folder_key = switch_folder_key
            switch_folder_key = ""
            video_paths = folder_videos[current_folder_key]
            current_video_idx = folder_positions.get(current_folder_key, 0)
            switch_delta = 0
            continue

        if is_stats_mode:
            current_video_idx += 1
            if current_video_idx >= len(video_paths):
                break
        else:
            if switch_delta != 0:
                current_video_idx = (current_video_idx + switch_delta) % len(video_paths)
            elif is_stream_url:
                # 串流不做循環播放，結束後維持在當前來源即可
                break
            elif not loop_playback:
                break

    if display_window:
        cv2.destroyAllWindows()

    if is_test_mode:
        print("\n模式2完成：視窗測試結束（未產生統計報告）")
        print("=" * 60)
        return

    print("-"*60)
    print(f"\n推論完成！共納入 {frame_count} 幀（僅完整播放影片）")
    print(f"\nYOLO 偵測統計:")
    if frame_count > 0:
        print(f"  偵測到貓咪: {frames_with_cat} 幀 ({frames_with_cat/frame_count*100:.1f}%)")
        print(f"  未偵測到: {frames_without_cat} 幀 ({frames_without_cat/frame_count*100:.1f}%)")
    else:
        print("  偵測到貓咪: 0 幀 (0.0%)")
        print("  未偵測到: 0 幀 (0.0%)")
    print(f"\n有效預測: {len(predictions)} 次")
    print(f"行為變化: {behavior_change_count} 次")

    # 抖動統計（全域）
    print_jitter_report(
        title=f"17關鍵點抖動統計（全域，EMA={EMA_ALPHA}，conf>{JITTER_CONF_THRESHOLD}）",
        jitter_px=global_jitter_px,
        jitter_norm=global_jitter_norm,
        valid_counts=global_valid_counts,
        pair_counts=global_pair_counts,
    )

    # 抖動統計（每影片）
    for vid_idx, stats in sorted(per_video_stats.items(), key=lambda x: x[0]):
        print_jitter_report(
            title=f"17關鍵點抖動統計（影片[{vid_idx}]，EMA={EMA_ALPHA}，conf>{JITTER_CONF_THRESHOLD}）",
            jitter_px=stats["jitter_px"],
            jitter_norm=stats["jitter_norm"],
            valid_counts=stats["valid_counts"],
            pair_counts=stats["pair_counts"],
        )

    # 產出文檔報告
    report_path = generate_report_file(REPORT_OUTPUT_PATH, recorded_video_stats)
    print(f"\n✓ 分析報告已輸出: {report_path}")

    # 統計分析
    if predictions:
        print("\n" + "="*60)
        print("統計分析")
        print("="*60)

        from collections import Counter
        behavior_counts = Counter([p['behavior_id'] for p in predictions])
        print("\n各行為出現次數:")
        for bid in range(5):
            count = behavior_counts.get(bid, 0)
            pct = count / len(predictions) * 100 if predictions else 0
            print(f"  {BEHAVIOR_TEXT_MAP[bid]:6s} ({BEHAVIOR_CLASSES[bid]:8s}): {count:4d} 次 ({pct:5.1f}%)")

        print("\n各行為持續時間（秒）:")
        for bid in range(5):
            print(f"  {BEHAVIOR_TEXT_MAP[bid]:6s} ({BEHAVIOR_CLASSES[bid]:8s}): {float(global_behavior_duration_sec[bid]):7.2f} s")

        avg_probs = np.mean([p['probs'] for p in predictions], axis=0)
        print("\n平均機率分布:")
        for i, (cls, prob) in enumerate(zip(BEHAVIOR_CLASSES, avg_probs)):
            print(f"  {BEHAVIOR_TEXT_MAP[i]:6s} ({cls:8s}): {prob*100:5.1f}%")

        confidences = [p['confidence'] for p in predictions]
        print(f"\n信心值統計:")
        print(f"  平均: {np.mean(confidences)*100:.1f}%")
        print(f"  最小: {np.min(confidences)*100:.1f}%")
        print(f"  最大: {np.max(confidences)*100:.1f}%")

        most_common_id = behavior_counts.most_common(1)[0][0]
        print(f"\n✓ 主要行為: {BEHAVIOR_TEXT_MAP[most_common_id]} ({BEHAVIOR_CLASSES[most_common_id]})")
        print(f"  出現比例: {behavior_counts[most_common_id]/len(predictions)*100:.1f}%")

        print("\n" + "="*60)
        print("結果分析")
        print("="*60)

        if most_common_id == 1:
            print("⚠ 主要預測為 scratch（搔抓），建議檢查:")
            print("  1. 影片內容是否包含抓癢/停頓等與 scratch 相似片段")
            print("  2. 是否有大量低信心窗被濾除，造成剩餘樣本偏向 scratch")
            print("  3. 重新檢視混淆矩陣與該影片逐幀機率曲線")
        elif most_common_id == 0:
            print("✓ 主要預測為 walk（走動），符合預期")
            print("  → 模型正確辨識出行走行為")
        else:
            print(f"預測為 {BEHAVIOR_TEXT_MAP[most_common_id]}，需檢查:")
            print("  1. 影片內容是否確實為此行為")
            print("  2. 模型訓練數據品質")
            print("  3. 正規化是否正確 (normalize=True)")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
