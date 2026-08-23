"""
雙模型同步比較腳本 - 同一支影片，左右兩個獨立視窗分別套用「模型 A」與「模型 B」
（各自獨立的 YOLO-Pose + ST-GCN 組合）做即時推論，方便肉眼比較兩組模型在同一段
動作上的骨架穩定度與行為判斷差異。

跟 1_run_video_inference.py 共用同一套 detectors/models/utils 模組（同樣的
sys.path 設定），但為了讓兩個模型的執行狀態（EMA/緩衝區/hysteresis）互不干擾，
把單一模型的推論流程封裝進 ModelPipeline，同一幀畫面分別餵給兩個獨立的
ModelPipeline 實例。

純視覺比較工具，不輸出統計報表／CSV——量化比較請用 eval_pose_compare.py
（YOLO）或 eval_gcn_compare.py（ST-GCN）。
"""
import sys
import os
import queue
import threading
from functools import lru_cache
from pathlib import Path
from collections import deque
from typing import Iterable, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    BEHAVIOR_COLORS,
    LOW_CONF_ID,
    BLACK,
    COLOR_HEAD,
)
from utils.helpers import get_behavior_name
from config import BehaviorTrackingConfig as _BehaviorTrackingConfig

# ═══════════════════════════════════════════════════════
#  使用者設定區
# ═══════════════════════════════════════════════════════
MODEL_A = dict(
    label="Model A",
    yolo_path=r"C:\ai_project\yolo_models\v11s_144.pt",
    stgcn_path=r"C:\ai_project\stgcn_models\run_124_xy_conf_v_bone_att_on\124_best_model.pth",
    feature_mode=None,   # None = 用下面 STGCN_FEATURE_MODE 的預設值
    ema_alpha=1.0,
    banner_color=(60, 60, 180),   # BGR，紅色系
)
MODEL_B = dict(
    label="Model B",
    yolo_path=r"C:\ai_project\yolo_models\v11s_147.pt",
    stgcn_path=r"C:\ai_project\stgcn_models\run_124_xy_conf_v_bone_att_on\124_best_model.pth",
    feature_mode=None,
    ema_alpha=1.0,
    banner_color=(60, 180, 60),   # BGR，綠色系
)

INFERENCE_DEVICE = "cuda"
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.5
DRAW_KP_CONF_THRESHOLD = 0.5
SEQUENCE_LENGTH = 16
CLASSIFY_STRIDE = 2
STGCN_NORMALIZE = True
STGCN_FEATURE_MODE = "xy"  # checkpoint 讀取失敗時的 fallback 值；正常情況下比較邏輯會依
                            # 各自 checkpoint 的 bn_input 通道數自動校正為實際的 feature mode

# ── 五個行為資料夾（按 z/x/c/v/b 切換，只有 FOLDER_TEST_MODE='all' 時才有作用）──
_BASE = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\1_貓咪姿勢影片分類\模型專用"
FOLDER_MAP = {
    'z': (rf"{_BASE}\walk",    "WALK"),
    'x': (rf"{_BASE}\lick",    "LICK"),
    'c': (rf"{_BASE}\scratch", "SCRATCH"),
    'v': (rf"{_BASE}\shake",   "SHAKE"),
    'b': (rf"{_BASE}\stop",    "STOP"),
}
DEFAULT_FOLDER_KEY = 'z'

# 'single' : 測試 SINGLE_FOLDER_PATH 指定的單一扁平資料夾
# 'all'    : 合併所有五個行為資料夾（可用 z/x/c/v/b 切換）
FOLDER_TEST_MODE = 'single'   # 'single' / 'all' / 'z' / 'x' / 'c' / 'v' / 'b'
SINGLE_FOLDER_PATH = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\白貓舔舐測試"
VIDEO_PATHS = []   # 手動指定影片清單則優先使用（忽略 FOLDER_TEST_MODE）

# 若設定 TEST_VIDEO_PATH 環境變數，優先使用該單一影片路徑（覆蓋 FOLDER_TEST_MODE / VIDEO_PATHS）
_env_test_video = os.getenv("TEST_VIDEO_PATH", "").strip()
if _env_test_video:
    VIDEO_PATHS = [_env_test_video]

TARGET_MODEL_FPS = 30.0
ENABLE_FPS_DOWNSAMPLE = True
LOOP_PLAYBACK = True

DISPLAY_SIZE = (860, 620)   # 每個視窗畫面大小（寬, 高）
WINDOW_GAP = 24              # 兩視窗之間的水平間距（螢幕像素）
WINDOW_POS = (60, 60)        # 左視窗（模型 A）的螢幕座標，右視窗自動排在其右側

BEHAVIOR_MIN_CONFIDENCE = _BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD
DISPLAY_HYSTERESIS_WINDOWS = _BehaviorTrackingConfig.DISPLAY_HYSTERESIS_WINDOWS
CAT_MISSING_TOLERANCE_FRAMES = _BehaviorTrackingConfig.CAT_MISSING_TOLERANCE_FRAMES

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"}
BEHAVIOR_PANEL_LABELS = tuple(str(name).upper() for name in BEHAVIOR_CLASSES)
# ═══════════════════════════════════════════════════════


# ─── test2.py 骨架視覺樣式（與 1_run_video_inference.py 相同定義）───────────
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


def _is_stream_url(path_str: str) -> bool:
    lowered = str(path_str).lower()
    return lowered.startswith(("http://", "https://", "rtsp://", "rtsps://", "rtmp://"))


def resolve_video_paths(video_sources: Iterable[str]):
    """將來源清單展開成影片檔路徑；來源可為影片檔或資料夾（與 1_run_video_inference.py 相同邏輯）。"""
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
            try:
                matched = sorted(
                    f for f in p.rglob("*")
                    if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTS
                )
            except Exception as e:
                print(f"⚠ 掃描資料夾出錯，已略過: {p} ({e})")
                matched = []
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


@lru_cache(maxsize=16)
def compute_ui_scale(width, height, base_width=1920.0, base_height=1080.0):
    diag = float(np.hypot(max(1.0, float(width)), max(1.0, float(height))))
    base_diag = float(np.hypot(base_width, base_height))
    scale = diag / max(base_diag, 1.0)
    return float(np.clip(scale, 0.65, 2.4))


def scale_px(value, ui_scale, min_px=1):
    return max(int(min_px), int(round(float(value) * float(ui_scale))))


def resize_with_letterbox(image, target_size):
    """等比例縮放至目標尺寸（保留完整畫面，四周補黑邊）。"""
    target_w, target_h = target_size
    src_h, src_w = image.shape[:2]
    if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
        return cv2.resize(image, target_size), 1.0, 0, 0

    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, -pad_x, -pad_y


def scale_kpts_and_bbox_for_letterbox(kpts, bbox, scale, crop_x, crop_y):
    scaled_kpts = kpts * scale - np.array([crop_x, crop_y], dtype=np.float32)
    scaled_bbox = None
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        scaled_bbox = np.array([
            x1 * scale - crop_x, y1 * scale - crop_y,
            x2 * scale - crop_x, y2 * scale - crop_y,
        ], dtype=np.float32)
    return scaled_kpts, scaled_bbox


def draw_no_cat_overlay(frame, text="No cat detected"):
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


def draw_test2_style_overlay(frame, kpts, kpt_conf, bbox, behavior_id, confidence, probs,
                              visualizer, show_info=True, conf_thresh=DRAW_KP_CONF_THRESHOLD,
                              bbox_conf=None):
    """畫骨架 + bbox + 行為預測標籤（與 1_run_video_inference.py 的同名函式相同）。"""
    h, w = frame.shape[:2]
    ui_scale = compute_ui_scale(w, h)
    kp_outer_radius = scale_px(4, ui_scale, min_px=2)
    kp_inner_radius = max(1, kp_outer_radius - 1)
    edge_thickness = scale_px(2, ui_scale, min_px=1)

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BLACK, 4, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HEAD, 2, cv2.LINE_AA)
        if bbox_conf is not None:
            label = f"{float(bbox_conf):.2f}"
            label_fs = 0.5 * ui_scale
            label_th = scale_px(1, ui_scale, min_px=1)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_fs, label_th)
            pad = scale_px(3, ui_scale, min_px=2)
            label_h = th + baseline + pad * 2
            label_w = tw + pad * 2
            fits_above = (y1 - label_h) >= 0
            rect_y1 = (y1 - label_h) if fits_above else y1
            rect_y2 = y1 if fits_above else (y1 + label_h)
            text_x = x1 + pad
            text_y = rect_y2 - pad - baseline
            cv2.rectangle(frame, (x1, rect_y1), (x1 + label_w, rect_y2), COLOR_HEAD, -1, cv2.LINE_AA)
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, label_fs, BLACK, label_th, cv2.LINE_AA)

    for ei, (a, b) in enumerate(_SKELETON_EDGES):
        if float(kpt_conf[a]) > conf_thresh and float(kpt_conf[b]) > conf_thresh:
            pa = (int(kpts[a][0]), int(kpts[a][1]))
            pb = (int(kpts[b][0]), int(kpts[b][1]))
            col = _EDGE_COLORS[ei] if ei < len(_EDGE_COLORS) else (180, 180, 180)
            cv2.line(frame, pa, pb, col, edge_thickness, cv2.LINE_AA)

    for i in range(min(17, len(kpts))):
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
            frame, 'LOW_CONF', float(confidence), (200, 200, 200),
            show_confidence=True, emphasize_label=False, label_background=False,
            font_scale_override=0.86,
        )
    elif behavior_id is not None and confidence > 0:
        behavior_name = get_behavior_name(behavior_id, use_text=False, fallback=str(behavior_id), confidence=confidence)
        visualizer.draw_prediction_on_frame(
            frame, behavior_name, confidence, BEHAVIOR_COLORS.get(behavior_id, (255, 255, 255)),
            show_confidence=True, emphasize_label=False, label_background=False,
            font_scale_override=0.86,
        )
    return frame


_PANEL_LAYOUT_CACHE: dict = {}


def draw_behavior_duration_panel(frame, elapsed_sec, behavior_duration_sec,
                                  behavior_current_confidences=None, behavior_occurrence_counts=None):
    """行為面板：每列顯示行為名稱、信心長條、累積持續秒數、發生次數（與主腳本相同）。"""
    h, w = frame.shape[:2]
    cache_key = (w, h)
    layout = _PANEL_LAYOUT_CACHE.get(cache_key)
    if layout is None:
        ui_scale = compute_ui_scale(w, h) * 1.10
        left = scale_px(8, ui_scale, min_px=4)
        right = scale_px(8, ui_scale, min_px=4)
        bottom = scale_px(6, ui_scale, min_px=3)
        title_fs = 0.60 * ui_scale
        meta_fs = 0.56 * ui_scale
        row_fs = 0.52 * ui_scale
        pct_fs = 0.46 * ui_scale
        text_th = scale_px(2, ui_scale, min_px=1)
        shadow_th = scale_px(2, ui_scale, min_px=2)
        row_h = scale_px(28, ui_scale, min_px=18)
        base_header_h = scale_px(42, ui_scale, min_px=26)
        header_extra_pad = scale_px(18, ui_scale, min_px=12)
        header_h = base_header_h + header_extra_pad
        row_count = len(BEHAVIOR_PANEL_LABELS)
        panel_h = header_h + row_h * row_count
        panel_top = max(scale_px(2, ui_scale, min_px=1), h - panel_h - bottom)
        tx = left
        ty = panel_top + scale_px(16, ui_scale, min_px=12)
        timer_y = ty + scale_px(16, ui_scale, min_px=10)
        label_w = 0
        for lbl in BEHAVIOR_PANEL_LABELS:
            tw, _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, row_fs, text_th)[0]
            label_w = max(label_w, tw)
        conf_w = cv2.getTextSize("100.0%", cv2.FONT_HERSHEY_SIMPLEX, row_fs, text_th)[0][0]
        col_gap = scale_px(8, ui_scale, min_px=4)
        conf_x = tx + label_w + col_gap
        bar_x = conf_x + conf_w + col_gap
        bar_h = scale_px(12, ui_scale, min_px=8)
        dur_w = cv2.getTextSize("999.9s", cv2.FONT_HERSHEY_SIMPLEX, pct_fs, text_th)[0][0]
        cnt_w = cv2.getTextSize("x99", cv2.FONT_HERSHEY_SIMPLEX, pct_fs, text_th)[0][0]
        available_space = max(0, w - right - dur_w - col_gap - cnt_w - col_gap - bar_x)
        max_bar_w = scale_px(180, ui_scale, min_px=80)
        min_bar_w = scale_px(50, ui_scale, min_px=40)
        bar_w = max(min_bar_w, min(available_space, max_bar_w))
        row_y0 = panel_top + header_h
        baseline_off = scale_px(14, ui_scale, min_px=9)
        bar_top_off = scale_px(1, ui_scale, min_px=0)
        dur_x = bar_x + bar_w + col_gap
        cnt_x = dur_x + dur_w + col_gap
        bar_border_th = scale_px(1, ui_scale, min_px=1)
        layout = dict(
            title_fs=title_fs, meta_fs=meta_fs, row_fs=row_fs, pct_fs=pct_fs,
            text_th=text_th, shadow_th=shadow_th, row_h=row_h, bar_h=bar_h,
            bar_w=bar_w, bar_border_th=bar_border_th, tx=tx, ty=ty, timer_y=timer_y,
            conf_x=conf_x, bar_x=bar_x, dur_x=dur_x, cnt_x=cnt_x, row_y0=row_y0,
            baseline_off=baseline_off, bar_top_off=bar_top_off,
        )
        _PANEL_LAYOUT_CACHE[cache_key] = layout

    title_fs, meta_fs, row_fs, pct_fs = layout['title_fs'], layout['meta_fs'], layout['row_fs'], layout['pct_fs']
    text_th, shadow_th = layout['text_th'], layout['shadow_th']
    row_h, bar_h, bar_w, bar_border_th = layout['row_h'], layout['bar_h'], layout['bar_w'], layout['bar_border_th']
    tx, ty, timer_y = layout['tx'], layout['ty'], layout['timer_y']
    conf_x, bar_x, dur_x, cnt_x = layout['conf_x'], layout['bar_x'], layout['dur_x'], layout['cnt_x']
    row_y0, baseline_off, bar_top_off = layout['row_y0'], layout['baseline_off'], layout['bar_top_off']

    title = "ST-GCN Behavior Confidence"
    cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, title_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
    cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, title_fs, (255, 245, 180), text_th, cv2.LINE_AA)

    timer = f"TIMER {float(elapsed_sec):7.2f}s"
    cv2.putText(frame, timer, (tx, timer_y), cv2.FONT_HERSHEY_SIMPLEX, meta_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
    cv2.putText(frame, timer, (tx, timer_y), cv2.FONT_HERSHEY_SIMPLEX, meta_fs, (170, 250, 255), text_th, cv2.LINE_AA)

    for bid, label in enumerate(BEHAVIOR_PANEL_LABELS):
        pct = float(np.clip(behavior_current_confidences[bid], 0.0, 1.0)) \
            if behavior_current_confidences is not None and bid < len(behavior_current_confidences) else 0.0
        color = BEHAVIOR_COLORS.get(bid, (130, 230, 255))
        line_top = row_y0 + bid * row_h
        baseline_y = line_top + baseline_off

        cv2.putText(frame, label, (tx, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, color, text_th, cv2.LINE_AA)

        conf_text = f"{pct * 100.0:5.1f}%"
        cv2.putText(frame, conf_text, (conf_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, conf_text, (conf_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (235, 235, 235), text_th, cv2.LINE_AA)

        bar_top = line_top + bar_top_off
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_top + bar_h), (78, 78, 78), -1)
        fill_w = int(round(bar_w * pct))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_top), (bar_x + fill_w, bar_top + bar_h), color, -1)
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_top + bar_h), (120, 120, 120), bar_border_th)

        dur_val = behavior_duration_sec[bid] if behavior_duration_sec is not None and bid < len(behavior_duration_sec) else 0.0
        dur_text = f"{dur_val:.1f}s"
        cv2.putText(frame, dur_text, (dur_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, pct_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, dur_text, (dur_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, pct_fs, (180, 255, 180), text_th, cv2.LINE_AA)

        occ_val = int(behavior_occurrence_counts[bid]) \
            if behavior_occurrence_counts is not None and bid < len(behavior_occurrence_counts) else 0
        cnt_text = f"x{occ_val}"
        cv2.putText(frame, cnt_text, (cnt_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, pct_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, cnt_text, (cnt_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, pct_fs, (255, 240, 160), text_th, cv2.LINE_AA)

    return frame


def draw_top_right_banner(frame, text, color):
    """右上角顯示模型標籤，避開左上角的行為預測標籤與底部的行為面板。"""
    h, w = frame.shape[:2]
    ui_scale = compute_ui_scale(w, h)
    fs = 0.62 * ui_scale
    th = scale_px(2, ui_scale, min_px=1)
    (tw, th_px), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = scale_px(8, ui_scale, min_px=5)
    x2 = w - scale_px(8, ui_scale, min_px=6)
    x1 = x2 - tw - pad * 2
    y1 = scale_px(6, ui_scale, min_px=4)
    y2 = y1 + th_px + baseline + pad * 2
    # 只對橫幅這一小塊 ROI 做半透明底色，不需要複製/混合整張畫面
    # （矩形範圍外 overlay 跟 frame 本來就相同，混合等於沒動，直接跳過即可）
    roi = frame[y1:y2, x1:x2]
    if roi.size > 0:
        plate = np.full_like(roi, color)
        cv2.addWeighted(roi, 0.45, plate, 0.55, 0, dst=roi)
    text_x = x1 + pad
    text_y = y2 - pad - baseline // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return frame


def _resolve_feature_mode_and_channels(stgcn_path: str, requested_feature_mode: str):
    """嘗試讀取 checkpoint 的 bn_input 通道數，若與目前 feature mode 不符，
    自動將 feature mode 換成與 checkpoint 通道數相對應的 canonical 模式
    （與 1_run_video_inference.py main() 內的邏輯相同）。"""
    feature_mode = requested_feature_mode
    in_channels = None
    ck_channel_map = {2: 'xy', 3: 'xy_conf', 5: 'xy_conf_v', 7: 'xy_conf_v_bone', 9: 'xy_conf_v_bone_bmotion'}
    try:
        import torch
        if os.path.exists(stgcn_path):
            try:
                ck = torch.load(stgcn_path, map_location='cpu', weights_only=True)
            except Exception:
                ck = torch.load(stgcn_path, map_location='cpu')
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
                        print(f"⚠ {Path(stgcn_path).name} 的 bn_input channels={ck_in_ch}，與 feature_mode={feature_mode} 不符 → 自動改用 {new_mode}")
                        feature_mode = new_mode
                    else:
                        print(f"⚠ {Path(stgcn_path).name} 的 bn_input channels={ck_in_ch}，無對應 canonical feature mode，將以該 channel 數為主。")
                in_channels = ck_in_ch
    except Exception as e:
        print(f"⚠ 無法從 checkpoint 推斷通道數（{stgcn_path}）：{e}")

    if in_channels is None:
        in_channels = get_in_channels_for_mode(feature_mode)
    return feature_mode, in_channels


class ModelPipeline:
    """封裝單一模型組合（YOLO-Pose + ST-GCN）的推論狀態與流程，讓雙模型比較時
    可以各自獨立維護 EMA 平滑、序列緩衝區、hysteresis 等狀態，互不干擾。"""

    def __init__(self, cfg: dict, device: str):
        self.label = cfg["label"]
        self.banner_color = cfg.get("banner_color", (80, 80, 80))
        self.ema_alpha = float(cfg.get("ema_alpha", 1.0))

        requested_mode = cfg.get("feature_mode") or STGCN_FEATURE_MODE
        self.feature_mode, in_channels = _resolve_feature_mode_and_channels(cfg["stgcn_path"], requested_mode)

        self.keypoint_detector = KeypointDetector(
            cfg["yolo_path"], device=device, imgsz=YOLO_IMGSZ, conf_thres=YOLO_CONF_THRESHOLD,
        )
        self.behavior_classifier = BehaviorClassifier(
            cfg["stgcn_path"], device=device, sequence_length=SEQUENCE_LENGTH,
            normalize=STGCN_NORMALIZE, feature_mode=self.feature_mode, in_channels=in_channels,
        )
        self.num_joints = getattr(self.behavior_classifier.model, 'num_joints', 17)
        self.num_classes = len(BEHAVIOR_CLASSES)

        self.reset_for_new_video()

    def reset_for_new_video(self):
        self.keypoints_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.bbox_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.ema_kpts = None
        self._cat_missing_streak = 0
        self._last_known_kpts = None
        self._last_known_kpt_conf = None
        self._last_known_bbox = None
        self._last_known_bbox_conf = None
        self.keypoint_detector.reset_track()

        self.behavior_id = LOW_CONF_ID
        self.confidence = 0.0
        self.probs = np.zeros(self.num_classes, dtype=np.float32)
        self.display_behavior_id = LOW_CONF_ID
        self.display_confidence = 0.0
        self.display_probs = np.zeros(self.num_classes, dtype=np.float32)
        self._hyst_candidate_id = LOW_CONF_ID
        self._hyst_streak = 0

        self.behavior_duration_sec = np.zeros(self.num_classes, dtype=np.float64)
        self.behavior_occurrence_counts = np.zeros(self.num_classes, dtype=np.int64)
        self._last_behavior_for_occurrence = LOW_CONF_ID

        self.sample_count = 0
        self.elapsed_sec = 0.0

        self.last_kpts = None
        self.last_kpt_conf = None
        self.last_bbox = None
        self.last_bbox_conf = None
        self.has_cat = False

    def _update_display_hysteresis(self, candidate_id, candidate_confidence, candidate_probs):
        threshold = DISPLAY_HYSTERESIS_WINDOWS.get(candidate_id, 1) if candidate_id != LOW_CONF_ID else 1
        if threshold <= 1 or candidate_id == LOW_CONF_ID:
            self.display_behavior_id = candidate_id
            self.display_confidence = candidate_confidence
            self.display_probs = candidate_probs
            self._hyst_candidate_id = LOW_CONF_ID
            self._hyst_streak = 0
            return
        if candidate_id == self._hyst_candidate_id:
            self._hyst_streak += 1
        else:
            self._hyst_candidate_id = candidate_id
            self._hyst_streak = 1
        if self._hyst_streak >= threshold:
            self.display_behavior_id = candidate_id
            self.display_confidence = candidate_confidence
            self.display_probs = candidate_probs

    def process_frame(self, frame: np.ndarray, frame_dt: float):
        """處理一幀原始影格：更新關鍵點/行為分類狀態，累積行為持續秒數。"""
        self.sample_count += 1
        self.elapsed_sec += frame_dt

        kpts, kpt_conf, bbox, bbox_conf = self.keypoint_detector.detect(frame)

        if kpts is not None:
            self._cat_missing_streak = 0
            # kpts/kpt_conf 是 KeypointDetector.detect() 每次呼叫新配置的陣列
            # （.cpu().numpy() 轉換），從未被原地修改，這裡直接持有參考即可，
            # 不需要額外複製一份。
            self._last_known_kpts = kpts
            self._last_known_kpt_conf = kpt_conf
            self._last_known_bbox = bbox
            self._last_known_bbox_conf = bbox_conf
        elif self._cat_missing_streak < CAT_MISSING_TOLERANCE_FRAMES and self._last_known_kpts is not None:
            self._cat_missing_streak += 1
            kpts, kpt_conf = self._last_known_kpts, self._last_known_kpt_conf
            bbox, bbox_conf = self._last_known_bbox, self._last_known_bbox_conf

        self.has_cat = kpts is not None

        if kpts is None:
            self.last_kpts = None
            self.last_kpt_conf = None
            self.last_bbox = None
            self.last_bbox_conf = None
            self.ema_kpts = None
            self._update_display_hysteresis(LOW_CONF_ID, 0.0, np.zeros(self.num_classes, dtype=np.float32))
            self._last_behavior_for_occurrence = LOW_CONF_ID
            return

        # EMA 平滑
        if self.ema_kpts is None:
            self.ema_kpts = kpts.copy()
        else:
            self.ema_kpts = self.ema_alpha * kpts + (1.0 - self.ema_alpha) * self.ema_kpts
        kpts = self.ema_kpts.copy()

        self.last_kpts = kpts
        self.last_kpt_conf = kpt_conf
        self.last_bbox = bbox
        self.last_bbox_conf = bbox_conf

        self.keypoints_buffer.append((kpts, kpt_conf))
        self.bbox_buffer.append(bbox if bbox is not None else np.full(4, np.nan))

        if len(self.keypoints_buffer) >= SEQUENCE_LENGTH and (self.sample_count % CLASSIFY_STRIDE == 0):
            kpts_arr = np.array([item[0] for item in self.keypoints_buffer])
            conf_arr = np.array([item[1] for item in self.keypoints_buffer])
            if self.num_joints < 17:
                kpts_arr = kpts_arr[:, :self.num_joints, :]
                conf_arr = conf_arr[:, :self.num_joints]

            seq_array = interpolate_missing(kpts_arr, conf_arr, threshold=0.0)
            if STGCN_NORMALIZE:
                seq_array = flip_normalize(seq_array)
                seq_array = orientation_normalize(seq_array)
                seq_array = normalize_skeleton_coords(seq_array)
            seq_features = build_feature_tensor(seq_array, conf_arr, self.feature_mode)
            pred_id, pred_conf, pred_probs = self.behavior_classifier.model.predict(seq_features, precomputed=True)

            if pred_id is None:
                self.behavior_id = LOW_CONF_ID
                self.confidence = 0.0
                self.probs = np.zeros(self.num_classes, dtype=np.float32)
            else:
                self.behavior_id = int(pred_id)
                self.confidence = float(pred_conf)
                _probs_padded = list(pred_probs) if pred_probs is not None else []
                while len(_probs_padded) < self.num_classes:
                    _probs_padded.append(0.0)
                self.probs = np.array(_probs_padded[:self.num_classes], dtype=np.float32)

            behavior_id_for_display = self.behavior_id if self.confidence >= BEHAVIOR_MIN_CONFIDENCE else LOW_CONF_ID
            candidate_confidence = self.confidence if behavior_id_for_display != LOW_CONF_ID else 0.0
            # self.probs 在下一次分類時會被整個重新賦值（rebind），不是原地修改，
            # 這裡不需要複製一份再傳進去。
            self._update_display_hysteresis(behavior_id_for_display, candidate_confidence, self.probs)

            if behavior_id_for_display != LOW_CONF_ID and behavior_id_for_display != self._last_behavior_for_occurrence:
                self.behavior_occurrence_counts[int(behavior_id_for_display)] += 1
            self._last_behavior_for_occurrence = behavior_id_for_display

        if self.display_behavior_id != LOW_CONF_ID:
            self.behavior_duration_sec[int(self.display_behavior_id)] += frame_dt


class _PipelineWorker:
    """背景執行緒：讓單一 ModelPipeline 的推論（YOLO+ST-GCN）跟主執行緒的畫面
    渲染/按鍵輪詢解耦，修正兩組模型依序同步推論造成的卡頓。

    設計採「跟不上就丟舊留新」（跟 processors/frame_processor.py 的
    _LatestFrameGrabber 同一種精神）：submit() 永遠只保留最新一幀待處理，
    worker 忙不過來時中間幀會被自然丟棄——這支腳本純視覺比較、不輸出統計，
    可以接受畫面偶爾跳過幾幀。_lock 保護 pipeline 內部狀態的讀寫，避免
    render 執行緒讀到「一半新一半舊」的欄位組合，也避免 reset_for_new_video()
    跟背景執行緒正在跑的 process_frame() 對 KeypointDetector 產生競態。
    """

    def __init__(self, pipeline: ModelPipeline):
        self.pipeline = pipeline
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, frame_dt: float) -> None:
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait((frame, frame_dt))
        except queue.Full:
            pass

    def _loop(self) -> None:
        while self._running:
            try:
                frame, frame_dt = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            with self._lock:
                self.pipeline.process_frame(frame, frame_dt)

    def process_now(self, frame: np.ndarray, frame_dt: float) -> None:
        """同步立即處理一幀，繞過 queue（給 a/d 單步用）。"""
        with self._lock:
            self.pipeline.process_frame(frame, frame_dt)

    def reset_for_new_video(self) -> None:
        with self._lock:
            self.pipeline.reset_for_new_video()
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass

    def snapshot(self) -> dict:
        """回傳目前 pipeline 顯示所需欄位的一致快照，供 render 執行緒使用。"""
        with self._lock:
            p = self.pipeline
            return dict(
                label=p.label, banner_color=p.banner_color,
                has_cat=p.has_cat, last_kpts=p.last_kpts, last_kpt_conf=p.last_kpt_conf,
                last_bbox=p.last_bbox, last_bbox_conf=p.last_bbox_conf,
                display_behavior_id=p.display_behavior_id, display_confidence=p.display_confidence,
                display_probs=p.display_probs, elapsed_sec=p.elapsed_sec,
                behavior_duration_sec=p.behavior_duration_sec,
                behavior_occurrence_counts=p.behavior_occurrence_counts,
            )

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)


def render_pipeline_frame(display_frame: np.ndarray, scale: float, crop_x: float, crop_y: float,
                           snap: dict, visualizer: Visualizer, nav_text: str) -> np.ndarray:
    """把單一模型 pipeline 目前的推論狀態（由 _PipelineWorker.snapshot() 取得的
    一致快照）疊繪到（已 letterbox 縮放好的）畫面上。

    display_frame/scale/crop_x/crop_y 由呼叫端算好、兩個視窗共用一次結果（見
    main() 呼叫處），避免對同一份 raw_frame 重複做兩次 letterbox resize。
    """
    if snap["has_cat"] and snap["last_kpts"] is not None:
        scaled_kpts, scaled_bbox = scale_kpts_and_bbox_for_letterbox(
            snap["last_kpts"], snap["last_bbox"], scale, crop_x, crop_y
        )
        draw_test2_style_overlay(
            display_frame, scaled_kpts, snap["last_kpt_conf"], scaled_bbox,
            snap["display_behavior_id"], snap["display_confidence"], snap["display_probs"],
            visualizer, show_info=True, bbox_conf=snap["last_bbox_conf"],
        )
    else:
        draw_no_cat_overlay(display_frame)

    draw_behavior_duration_panel(
        display_frame, snap["elapsed_sec"], snap["behavior_duration_sec"],
        behavior_current_confidences=snap["display_probs"],
        behavior_occurrence_counts=snap["behavior_occurrence_counts"],
    )
    draw_top_right_banner(display_frame, f"{snap['label']}  |  {nav_text}", snap["banner_color"])
    return display_frame


def main():
    print("=" * 60)
    print("雙模型同步比較（左右視窗）")
    print("=" * 60)

    folder_videos: dict = {}
    for fkey, (fpath, fname) in FOLDER_MAP.items():
        vids = resolve_video_paths([fpath])
        folder_videos[fkey] = vids
        print(f"  [{fkey}] {fname}: {len(vids)} 部影片  ({fpath})")

    folder_range: dict = {}
    if VIDEO_PATHS:
        video_paths = resolve_video_paths(VIDEO_PATHS)
        current_folder_key = DEFAULT_FOLDER_KEY
        is_all_mode = False
    elif FOLDER_TEST_MODE == 'all':
        video_paths = []
        _offset = 0
        for fkey in FOLDER_MAP:
            _n = len(folder_videos[fkey])
            folder_range[fkey] = (_offset, _offset + _n)
            video_paths.extend(folder_videos[fkey])
            _offset += _n
        current_folder_key = DEFAULT_FOLDER_KEY
        is_all_mode = True
        print(f"[FOLDER_TEST_MODE=all] 已合併全部 {len(video_paths)} 部影片")
    else:
        video_paths = resolve_video_paths([SINGLE_FOLDER_PATH])
        current_folder_key = DEFAULT_FOLDER_KEY
        is_all_mode = False
        print(f"[FOLDER_TEST_MODE=single] {SINGLE_FOLDER_PATH}  共 {len(video_paths)} 部影片")

    if not video_paths:
        print("❌ 找不到可用影片，請確認 FOLDER_MAP / VIDEO_PATHS / SINGLE_FOLDER_PATH 的路徑")
        return

    print(f"\n模型 A: {MODEL_A['label']}  ({Path(MODEL_A['yolo_path']).name} + {Path(MODEL_A['stgcn_path']).name})")
    print(f"模型 B: {MODEL_B['label']}  ({Path(MODEL_B['yolo_path']).name} + {Path(MODEL_B['stgcn_path']).name})")
    print("控制: q=退出  space=暫停  r=重置目前影片  1/2=上/下一部影片  a/d=後退/前進一幀（雙視窗同步）" +
          ("  z/x/c/v/b=切換資料夾" if is_all_mode else ""))
    print("=" * 60)

    print("\n初始化模型 A...")
    try:
        pipeline_a = ModelPipeline(MODEL_A, INFERENCE_DEVICE)
    except Exception as e:
        print(f"❌ 模型 A 載入失敗：{e}")
        return
    print("初始化模型 B...")
    try:
        pipeline_b = ModelPipeline(MODEL_B, INFERENCE_DEVICE)
    except Exception as e:
        print(f"❌ 模型 B 載入失敗：{e}")
        return
    worker_a = _PipelineWorker(pipeline_a)
    worker_b = _PipelineWorker(pipeline_b)
    visualizer = Visualizer()

    win_a = f"[A] {MODEL_A['label']}"
    win_b = f"[B] {MODEL_B['label']}"
    cv2.namedWindow(win_a, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_b, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_a, *DISPLAY_SIZE)
    cv2.resizeWindow(win_b, *DISPLAY_SIZE)
    cv2.moveWindow(win_a, WINDOW_POS[0], WINDOW_POS[1])
    cv2.moveWindow(win_b, WINDOW_POS[0] + DISPLAY_SIZE[0] + WINDOW_GAP, WINDOW_POS[1])

    current_video_idx = 0
    stop_requested = False

    try:
        while not stop_requested:
            if is_all_mode:
                for _fk, (_s, _e) in folder_range.items():
                    if _s <= current_video_idx < _e:
                        current_folder_key = _fk
                        break

            video_path = video_paths[current_video_idx]
            if not Path(video_path).exists():
                print(f"❌ 影片不存在，跳過: {video_path}")
                current_video_idx = (current_video_idx + 1) % len(video_paths)
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ 無法開啟影片，跳過: {video_path}")
                current_video_idx = (current_video_idx + 1) % len(video_paths)
                continue

            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if source_fps <= 1:
                source_fps = TARGET_MODEL_FPS
            frame_step = 1
            if ENABLE_FPS_DOWNSAMPLE and source_fps > TARGET_MODEL_FPS + 1e-6:
                frame_step = max(1, int(round(source_fps / TARGET_MODEL_FPS)))
            model_input_fps = source_fps / frame_step
            frame_dt = 1.0 / max(model_input_fps, 1e-6)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"\n目前影片 [{current_video_idx + 1}/{len(video_paths)}] {video_path}")
            print(f"模型輸入時基: {model_input_fps:.2f} fps (frame_step={frame_step})")

            worker_a.reset_for_new_video()
            worker_b.reset_for_new_video()

            paused = False
            switch_delta = 0
            switch_folder_key = ""
            last_raw_frame: Optional[np.ndarray] = None
            consecutive_read_failures = 0

            def _step_one_model_frame(direction: int) -> bool:
                """前進/後退一個「模型時基」幀（依 frame_step 降採樣間距），同時餵給
                pipeline_a/pipeline_b，兩個視窗一定同步在同一幀上。

                注意：這只是把該幀原始畫面重新跑一次 process_frame()，並不會把
                pipeline 內部的 EMA/序列緩衝區/hysteresis 狀態「倒帶」回那個時間點
                （沒有做完整重算），所以剛倒退/快轉那幾幀的行為判斷可能會有短暫
                不連續，純供肉眼比較骨架/框線用；要看正確的行為判斷請按 r 重置後
                重新從頭播放。
                """
                nonlocal last_raw_frame
                if direction < 0:
                    cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    target_pos = cur_pos - 2 * frame_step
                    if target_pos < 0:
                        return False
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
                ret, frame = cap.read()
                if not ret:
                    return False
                # frame 不需要為兩個 pipeline 各自複製一份：KeypointDetector.detect()
                # 只讀取傳入的畫面（YOLO 推論不會原地修改輸入），兩個 worker 各自在
                # 自己的背景執行緒裡處理，共用同一份 frame 安全無虞。
                worker_a.process_now(frame, frame_dt)
                worker_b.process_now(frame, frame_dt)
                last_raw_frame = frame
                if frame_step > 1:
                    for _ in range(frame_step - 1):
                        if not cap.grab():
                            break
                return True

            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        if LOOP_PLAYBACK:
                            if last_raw_frame is None:
                                # 這支影片從頭到尾一幀都沒成功讀過就一直跳回開頭重來，
                                # 代表影片疑似損毀而非正常播完，避免無 waitKey 的忙迴圈卡死
                                consecutive_read_failures += 1
                                if consecutive_read_failures >= 3:
                                    print(f"\n❌ 影片疑似損毀，連續 {consecutive_read_failures} 次讀取失敗，跳過: {video_path}")
                                    switch_delta = 1
                                    break
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            worker_a.reset_for_new_video()
                            worker_b.reset_for_new_video()
                            continue
                        # 不循環播放時，自然播完就自動前進到下一部，避免卡在同一支影片
                        switch_delta = 1
                        break

                    # frame 不需要為兩個 pipeline 各自複製一份，理由同 _step_one_model_frame()；
                    # submit() 是 fire-and-forget，worker 背景處理跟不上時會自動丟舊留新，
                    # 讓這裡的讀幀/渲染/按鍵輪詢不會被推論卡住
                    worker_a.submit(frame, frame_dt)
                    worker_b.submit(frame, frame_dt)
                    last_raw_frame = frame

                    # 降採樣時跳過後續 frame_step-1 幀，避免不必要的完整解碼
                    if frame_step > 1:
                        for _ in range(frame_step - 1):
                            if not cap.grab():
                                break

                    # 持續在終端機同一行更新目前幀數（不佔用捲動歷史，用 \r 覆寫）
                    _cur_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    print(f"\rframe {_cur_frame_pos}/{total_frames}", end="", flush=True)

                if last_raw_frame is None:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        stop_requested = True
                        break
                    continue

                folder_hint = f"[{FOLDER_MAP[current_folder_key][1]}] " if is_all_mode else ""
                nav_text = f"{folder_hint}{current_video_idx + 1}/{len(video_paths)}" + (" [暫停]" if paused else "")
                # letterbox resize 只算一次、兩個視窗共用同一份結果（内容相同，僅疊圖不同），
                # 避免同一張 raw_frame 被重複做兩次一樣的縮放運算
                _base_display, _scale, _crop_x, _crop_y = resize_with_letterbox(last_raw_frame, DISPLAY_SIZE)
                frame_a = render_pipeline_frame(_base_display.copy(), _scale, _crop_x, _crop_y, worker_a.snapshot(), visualizer, nav_text)
                frame_b = render_pipeline_frame(_base_display, _scale, _crop_x, _crop_y, worker_b.snapshot(), visualizer, nav_text)
                cv2.imshow(win_a, frame_a)
                cv2.imshow(win_b, frame_b)

                key = cv2.waitKey(1) & 0xFF
                if key == 255:
                    continue
                if key == ord('q'):
                    print("\n使用者中斷：q")
                    stop_requested = True
                    break
                if key == ord(' '):
                    paused = not paused
                    print("\n⏸ 已暫停" if paused else "\n▶ 繼續播放")
                    continue
                if key == ord('r'):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    worker_a.reset_for_new_video()
                    worker_b.reset_for_new_video()
                    paused = False
                    print("\n↺ 已重置：回到影片開頭並清空偵測狀態")
                    continue
                if key == ord('d'):
                    paused = True
                    if _step_one_model_frame(+1):
                        print(f"\n⏭ 前進一幀  frame_pos={int(cap.get(cv2.CAP_PROP_POS_FRAMES))}")
                    else:
                        print("\n⏭ 已到影片結尾，無法再前進")
                    continue
                if key == ord('a'):
                    paused = True
                    if _step_one_model_frame(-1):
                        print(f"\n⏮ 後退一幀  frame_pos={int(cap.get(cv2.CAP_PROP_POS_FRAMES))}")
                    else:
                        print("\n⏮ 已在影片開頭，無法再後退")
                    continue
                if key == ord('2'):
                    switch_delta = 1
                    break
                if key == ord('1'):
                    switch_delta = -1
                    break
                if is_all_mode and 0 <= key < 256 and chr(key) in FOLDER_MAP and chr(key) != current_folder_key:
                    fk = chr(key)
                    if folder_videos.get(fk):
                        switch_folder_key = fk
                        break
                    print(f"\n⚠ 資料夾 [{fk}] 沒有影片，忽略")

            cap.release()

            if stop_requested:
                break
            if switch_folder_key:
                current_video_idx = folder_range[switch_folder_key][0]
                print(f"\n切換資料夾 → {FOLDER_MAP[switch_folder_key][1]} [{switch_folder_key}]")
            else:
                current_video_idx = (current_video_idx + switch_delta) % len(video_paths)

    finally:
        worker_a.stop()
        worker_b.stop()
        cv2.destroyAllWindows()
    print("\n已結束雙模型比較。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ 已收到中斷（Ctrl+C），正在關閉視窗...")
        cv2.destroyAllWindows()
