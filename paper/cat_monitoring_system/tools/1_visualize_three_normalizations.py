"""
互動式骨架正規化視覺化 Demo——用來向教授展示三種正規化步驟
（flip_normalize / orientation_normalize / normalize_skeleton_coords）
各自的目的與視覺效果，可於播放中即時按鍵切換：
    f = flip_normalize（翻轉統一朝向）
    o = orientation_normalize（旋轉至身體朝上）
    n = normalize_skeleton_coords（置中＋依體型縮放）
    p = raw/正規化 overlay 切換（原始影像+原始骨架 vs 黑底+正規化骨架）
另含 EMA 關鍵點平滑（本檔預設 EMA_ALPHA=1.0，即關閉，不影響三步驟正規化的示範）。
"""
import sys
import os
import cv2
import numpy as np
import time
from functools import lru_cache
from pathlib import Path
from collections import deque
from typing import Iterable

# 加入系統路徑
# Ensure both the package folder and repository root are on sys.path so
# top-level modules like config.py can be imported when running this script
# from within the cat_monitoring_system folder.
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
    BEHAVIOR_TEXT_MAP,
    BEHAVIOR_COLORS,
    LOW_CONF_ID,
    BLACK,
    COLOR_HEAD,
)
from utils.helpers import get_behavior_name
from config import BehaviorTrackingConfig as _BehaviorTrackingConfig

# ── 五個行為資料夾（按 z/x/c/v/b 切換）────────────────────────────────
_BASE = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\1_貓咪姿勢影片分類\暫存"
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

# 測試資料夾模式
# 'single' : 測試 SINGLE_FOLDER_PATH 指定的單一扁平資料夾（影片直接放在該目錄，不分子資料夾）
# 'all'    : 測試所有五個行為資料夾（按 FOLDER_MAP 順序合併為一份播放清單）
FOLDER_TEST_MODE = 'single'
SINGLE_FOLDER_PATH = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\泛化測試"  # 'single' 模式使用的扁平資料夾

# VIDEO_PATHS 保留作備用（不使用 FOLDER_MAP 時可手動指定）
VIDEO_PATHS = []
YOLO_MODEL_PATH = r"C:\AI_Project\cat_pose\v11s_121.pt"
STGCN_MODEL_PATH = r"C:\Users\homec\Downloads\stgcn_results\run_121_xy_conf_v_bone_att_on\121_best_model.pth"
INFERENCE_DEVICE = 'cuda'
YOLO_IMGSZ = 640  # 與 YOLO 訓練尺寸一致
YOLO_CONF_THRESHOLD = 0.5
STGCN_NORMALIZE = True
SEQUENCE_LENGTH = 16
_raw_stgcn_mode = os.getenv("STGCN_FEATURE_MODE", "xy")
STGCN_FEATURE_MODE = str(_raw_stgcn_mode).strip().lower()
# Normalize legacy/variant feature-mode names to canonical names used by the STGCN module
# Canonical names: "xy", "xy_conf", "xy_conf_v", "xy_conf_v_bone", "xy_conf_v_bone_bmotion"
_FEATURE_MODE_MAP = {
    # compact / legacy variants → canonical
    "xyconf":                    "xy_conf",
    "xyv_conf":                  "xy_conf_v",
    "xyv_conf_bone":             "xy_conf_v_bone",
    "xyv_conf_bone_bone_motion": "xy_conf_v_bone_bmotion",
    "xyv_conf_bone_bmotion":     "xy_conf_v_bone_bmotion",
    "xyvconf":                   "xy_conf_v",
    "xyvconfbone":               "xy_conf_v_bone",
    "xyvconfbonebmotion":        "xy_conf_v_bone_bmotion",
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

# ===== 關鍵點顯示門檻 =====
DRAW_KP_CONF_THRESHOLD = 0.25  # 畫骨架線段與關鍵點圓點用門檻（>此值才畫）
SHOW_PROBABILITY_BARS = False  # 關閉機率條可減少每幀繪圖負載

# ===== EMA 平滑設定 =====
# alpha 越大 → 越貼近原始偵測值（響應快、平滑少）
# alpha 越小 → 越平滑（延遲多、噪音少）
EMA_ALPHA = 1.0  # 須與 train_gcn.py 的 KP_EMA_ALPHA 保持一致

# ===== 正規化消融開關（初始值；執行中可用 f/o/n 按鍵動態切換）=====
# 三個步驟均只在 STGCN_NORMALIZE=True 時生效
NORM_FLIP   = False  # f 鍵：flip_normalize       翻轉統一方向
NORM_ORIENT = False  # o 鍵：orientation_normalize 旋轉至 y 軸正向
NORM_COORD  = False  # n 鍵：normalize_skeleton_coords 中心化＋體型縮放
# ===== Overlay 模式（p 鍵切換）=====
# False = 黑底 + 正規化骨架（檢視正規化效果）
# True  = 原始影像 + 原始骨架（檢視 YOLO 偵測結果）
OVERLAY_RAW = False

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
            try:
                matched = sorted(
                    [
                        f for f in p.rglob("*")
                        if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTS
                    ]
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


_PANEL_LAYOUT_CACHE: dict = {}


def draw_behavior_duration_panel(frame, elapsed_sec, behavior_duration_sec, behavior_current_confidences=None, behavior_occurrence_counts=None):
    """行為面板：每列顯示行為名稱、信心長條（一個）、累積持續秒數、發生次數。"""
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
        # 長條後顯示持續秒數（e.g. "999.9s"）和次數（e.g. "99次"），各保留一個欄位
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

    title_fs      = layout['title_fs']
    meta_fs       = layout['meta_fs']
    row_fs        = layout['row_fs']
    pct_fs        = layout['pct_fs']
    text_th       = layout['text_th']
    shadow_th     = layout['shadow_th']
    row_h         = layout['row_h']
    bar_h         = layout['bar_h']
    bar_w         = layout['bar_w']
    bar_border_th = layout['bar_border_th']
    tx            = layout['tx']
    ty            = layout['ty']
    timer_y       = layout['timer_y']
    conf_x        = layout['conf_x']
    bar_x         = layout['bar_x']
    dur_x         = layout['dur_x']
    cnt_x         = layout['cnt_x']
    row_y0        = layout['row_y0']
    baseline_off  = layout['baseline_off']
    bar_top_off   = layout['bar_top_off']

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

        # 行為標籤
        cv2.putText(frame, label, (tx, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, color, text_th, cv2.LINE_AA)

        # 信心百分比（只顯示一次，在長條左側）
        conf_text = f"{pct * 100.0:5.1f}%"
        cv2.putText(frame, conf_text, (conf_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, conf_text, (conf_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (235, 235, 235), text_th, cv2.LINE_AA)

        # 信心長條
        bar_top = line_top + bar_top_off
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_top + bar_h), (78, 78, 78), -1)
        fill_w = int(round(bar_w * pct))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_top), (bar_x + fill_w, bar_top + bar_h), color, -1)
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_top + bar_h), (120, 120, 120), bar_border_th)

        # 持續秒數
        dur_val = behavior_duration_sec[bid] if behavior_duration_sec is not None and bid < len(behavior_duration_sec) else 0.0
        dur_text = f"{dur_val:.1f}s"
        cv2.putText(frame, dur_text, (dur_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, pct_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, dur_text, (dur_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, pct_fs, (180, 255, 180), text_th, cv2.LINE_AA)

        # 發生次數
        occ_val = int(behavior_occurrence_counts[bid]) \
            if behavior_occurrence_counts is not None and bid < len(behavior_occurrence_counts) else 0
        cnt_text = f"x{occ_val}"
        cv2.putText(frame, cnt_text, (cnt_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, pct_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, cnt_text, (cnt_x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, pct_fs, (255, 240, 160), text_th, cv2.LINE_AA)

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

    _n_kp = min(len(kpts), len(kpt_conf))
    for ei, (a, b) in enumerate(_SKELETON_EDGES):
        # 兩端關鍵點索引都要在陣列範圍內（14 關節模型會截斷尾巴三點 14-16）
        if a >= _n_kp or b >= _n_kp:
            continue
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
            'LOW_CONF',
            float(confidence),
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


# ===== 關鍵點標記 + 可收合面板（h 鍵整體開關，面板另外用滑鼠點擊收合/展開）=====
# 用來把「舔舐↔鼻子」「搔抓↔前後腳掌」這種關鍵點-行為相關性呈現出來：固定追蹤
# 鼻子＋左右耳＋四個腳掌共 7 個關鍵點，順序固定不隨數值大小重排（方便盯著同一列
# 看趨勢）；跟目前行為相關的用強調色，其餘用高對比中性色，避免像之前那樣用大面
# 積發光疊加（關節靠近時光暈會糊成一團、看起來太亮太雜）。
ALWAYS_TRACKED_JOINTS = [0, 1, 2, 7, 9, 11, 13]  # Nose, Left_Ear, Right_Ear, LF_Paw, RF_Paw, LH_Paw, RH_Paw
HIGHLIGHT_JOINTS = {
    1: [0],             # lick    → Nose
    2: [7, 9, 11, 13],  # scratch → LF_Paw, RF_Paw, LH_Paw, RH_Paw
}
HIGHLIGHT_COLOR = {
    1: (255, 255, 255),  # lick    → 白色（跟其他關節同一種白，不特別用黃色標示鼻子）
    2: (255, 150, 60),   # scratch → 藍色
}
# 黑底/暗色背景下要保持清楚可讀，中性色一律用高對比的近白色，不用中灰
NEUTRAL_MARKER_COLOR = (225, 225, 225)   # 目前行為用不到的關節環／文字
DIMMED_TEXT_COLOR    = (130, 130, 130)   # 使用者手動隱藏的列，維持可辨識但視覺上退居次要
PANEL_SCALE = 1.5  # 面板文字/圖示/關節環整體放大倍率


def _compute_per_joint_motion(seq_xy, conf_seq, conf_threshold=0.3):
    """算這個 window 裡每個關節各自的平均逐幀位移量（跟 0_train_gcn.py 同一套定義），
    用來驅動標記顏色與面板數值——哪個關節動得多，數值就越大。"""
    diffs = seq_xy[1:] - seq_xy[:-1]
    dist = np.linalg.norm(diffs, axis=-1)
    valid = (conf_seq[1:] > conf_threshold) & (conf_seq[:-1] > conf_threshold)
    out = np.full(dist.shape[1], np.nan, dtype=np.float64)
    for j in range(dist.shape[1]):
        if valid[:, j].any():
            out[j] = dist[valid[:, j], j].mean()
    return out


def draw_joint_highlight(frame, kpts, kpt_conf, joint_motion, behavior_id, row_visible, ui_scale,
                          conf_thresh=DRAW_KP_CONF_THRESHOLD):
    """在鼻子＋左右耳＋四個腳掌畫小而清楚的細環標記（固定小半徑，不做大面積發光疊加），
    跟目前行為相關的關節用強調色、粗一點；其餘用高對比中性色、細一點。使用者在
    左上角面板把某個關節隱藏時，這裡也不畫，維持「隱藏」的一致性。"""
    if joint_motion is None:
        return frame
    active = set(HIGHLIGHT_JOINTS.get(int(behavior_id), [])) if behavior_id is not None else set()
    color_active = HIGHLIGHT_COLOR.get(int(behavior_id), (255, 255, 255)) if behavior_id is not None else (255, 255, 255)

    for j in ALWAYS_TRACKED_JOINTS:
        if not row_visible.get(j, True):
            continue
        if j >= len(kpts) or j >= len(kpt_conf) or float(kpt_conf[j]) <= conf_thresh:
            continue
        cx, cy = int(kpts[j][0]), int(kpts[j][1])
        is_active = j in active
        color = color_active if is_active else NEUTRAL_MARKER_COLOR
        r = scale_px(9 if is_active else 6, ui_scale, min_px=4)
        thickness = max(1, int((2.4 if is_active else 1.6) * ui_scale))
        cv2.circle(frame, (cx, cy), r, color, thickness, cv2.LINE_AA)

    return frame


def draw_key_joint_panel(frame, panel_expanded, behavior_id, joint_motion, row_visible, ui_scale):
    """左上角可收合面板：固定順序列出鼻子＋左右耳＋四個腳掌共 7 個關鍵點的動作幅度
    數值（順序不隨數值大小重排，方便盯著同一列看趨勢），跟目前行為相關的用強調
    色、其餘用高對比中性色（黑底上避免用中灰，太糊）。收合時只顯示一個小按鈕
    「KEY JOINTS >」，點擊後展開成完整清單；展開後點「標題列」收合回去，點每一
    列右手邊的 +/- 圖示可以個別隱藏/顯示該關節（隱藏後只留關節名稱＋"+"，數值
    不顯示，讓面板可以只留下你想講解的關節）。

    回傳 (frame, header_rect, row_toggle_rects)：
    - header_rect：收合按鈕（收合時）或展開後標題列（展開時）的可點擊範圍，點擊切換收合/展開
    - row_toggle_rects：{joint_idx: (x0,y0,x1,y1)}，展開時每一列 +/- 圖示的可點擊範圍；收合時為空 dict
    供外層滑鼠回呼判斷點擊命中。"""
    color_active = HIGHLIGHT_COLOR.get(int(behavior_id), (230, 230, 230)) if behavior_id is not None else (230, 230, 230)
    active = set(HIGHLIGHT_JOINTS.get(int(behavior_id), [])) if behavior_id is not None else set()

    # PANEL_SCALE 統一放大字體/圖示/間距（面板定位錨點 x0/y0 不放大，只有內容變大）
    s_ui = ui_scale * PANEL_SCALE

    # 字體規格比照畫面左下角「ST-GCN Behavior Confidence」面板的作法：陰影（黑色
    # 描邊）跟文字本體粗細接近，不是陰影粗很多——陰影過粗會把彩色文字本體吃掉一
    # 圈，看起來就會糊；這裡陰影/本體粗細幾乎一樣，字才會像下面那個面板一樣清楚。
    text_th   = scale_px(2, s_ui, min_px=1)
    shadow_th = scale_px(2, s_ui, min_px=2)

    x0 = max(6, scale_px(10, ui_scale, min_px=8))
    y0 = max(6, scale_px(70, ui_scale, min_px=55))  # 留在既有左上角預測文字下方，避免重疊
    pad = max(6, int(8 * s_ui))
    header_fs = 0.58 * s_ui

    if not panel_expanded:
        header = "KEY JOINTS >"
        (tw, th_) = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, header_fs, shadow_th)[0]
        bw, bh = tw + pad * 2, th_ + pad * 2
        roi = frame[y0:y0 + bh, x0:x0 + bw]
        if roi.size > 0:
            dark = np.full_like(roi, 10)
            cv2.addWeighted(roi, 0.28, dark, 0.72, 0, roi)
            frame[y0:y0 + bh, x0:x0 + bw] = roi
        cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (210, 210, 210), 1)
        ty = y0 + pad + th_
        cv2.putText(frame, header, (x0 + pad, ty), cv2.FONT_HERSHEY_SIMPLEX, header_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, header, (x0 + pad, ty), cv2.FONT_HERSHEY_SIMPLEX, header_fs, (250, 250, 250), text_th, cv2.LINE_AA)
        return frame, (x0, y0, x0 + bw, y0 + bh), {}

    # ── 展開狀態：固定順序（不依動作幅度重排），每列右側有獨立的 +/- 顯示切換 ──
    header = "KEY JOINTS <"
    row_fs  = 0.52 * s_ui
    row_h   = max(22, int(30 * s_ui))
    icon_sz = max(18, int(22 * s_ui))
    gap     = max(8, int(10 * s_ui))

    (header_w, header_h) = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, header_fs, shadow_th)[0]

    def _full_row_text(j):
        # 最長的關節名（Right_Ear）是 9 個字元，<8s 會直接沒有留白貼在數字前面；
        # 統一 pad 到 10 再加一個空白，確保任何名稱後面至少有一格間距
        m = float(joint_motion[j]) if joint_motion is not None and j < len(joint_motion) and not np.isnan(joint_motion[j]) else 0.0
        return f"{KEYPOINT_NAMES[j]:<10s} {m:.4f}"

    # 寬度一律假設「全部列都顯示數值」去算，不受目前實際隱藏狀態影響——否則使用者
    # 隱藏掉當下最寬的那一列時，面板會突然變窄，版面跳來跳去很不穩定
    full_row_texts = [_full_row_text(j) for j in ALWAYS_TRACKED_JOINTS]
    row_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, row_fs, shadow_th)[0][0] for t in full_row_texts)
    content_w = max(header_w, row_w) + gap + icon_sz
    panel_w = pad * 2 + content_w
    content_top = y0 + pad + header_h + int(header_h * 0.75)
    panel_h = (content_top - y0) + row_h * len(ALWAYS_TRACKED_JOINTS) + pad

    h, w = frame.shape[:2]
    panel_w = min(panel_w, w - x0 - 4)
    panel_h = min(panel_h, h - y0 - 4)
    if panel_w <= 0 or panel_h <= 0:
        return frame, (x0, y0, x0, y0), {}

    roi = frame[y0:y0 + panel_h, x0:x0 + panel_w]
    if roi.size > 0:
        dark = np.full_like(roi, 10)
        cv2.addWeighted(roi, 0.24, dark, 0.76, 0, roi)
        frame[y0:y0 + panel_h, x0:x0 + panel_w] = roi
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (210, 210, 210), 1)

    tx = x0 + pad
    header_ty = y0 + pad + header_h
    header_rect = (x0, y0, x0 + panel_w, content_top)
    cv2.putText(frame, header, (tx, header_ty), cv2.FONT_HERSHEY_SIMPLEX, header_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
    cv2.putText(frame, header, (tx, header_ty), cv2.FONT_HERSHEY_SIMPLEX, header_fs, (250, 250, 250), text_th, cv2.LINE_AA)
    cv2.line(frame, (x0 + 2, content_top), (x0 + panel_w - 2, content_top), (90, 90, 90), 1)

    icon_x = x0 + panel_w - pad - icon_sz
    row_toggle_rects = {}
    for idx, j in enumerate(ALWAYS_TRACKED_JOINTS):
        row_top = content_top + idx * row_h
        text_y = row_top + row_h - max(6, int(7 * s_ui))
        visible = row_visible.get(j, True)
        is_active = j in active

        if visible:
            m = float(joint_motion[j]) if joint_motion is not None and j < len(joint_motion) and not np.isnan(joint_motion[j]) else 0.0
            row_text = f"{KEYPOINT_NAMES[j]:<10s} {m:.4f}"
            col = color_active if is_active else NEUTRAL_MARKER_COLOR
        else:
            row_text = f"{KEYPOINT_NAMES[j]}"
            col = DIMMED_TEXT_COLOR
        # 不管是否為目前行為的關鍵關節，文字/描邊粗細都用同一組數值——之前針對
        # active 列額外加粗，會讓那一列的字看起來比其他列大一圈、描邊位置像是偏移
        # （其實是筆畫變粗造成的視覺錯覺），改成統一粗細後就不會有這種不一致感，
        # active 與否單純用顏色（scratch 藍色 vs 中性灰）分辨就好。
        cv2.putText(frame, row_text, (tx, text_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, row_text, (tx, text_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, col, text_th, cv2.LINE_AA)

        # +/- 圖示：該列右手邊，點擊切換這個關節顯示/隱藏
        icon_y0 = row_top + max(0, (row_h - icon_sz) // 2)
        icon_rect = (icon_x, icon_y0, icon_x + icon_sz, icon_y0 + icon_sz)
        row_toggle_rects[j] = icon_rect
        icon_char = "-" if visible else "+"
        (icw, ich) = cv2.getTextSize(icon_char, cv2.FONT_HERSHEY_SIMPLEX, row_fs, shadow_th)[0]
        icon_tx = icon_x + max(0, (icon_sz - icw) // 2)
        icon_ty = icon_y0 + icon_sz - max(0, (icon_sz - ich) // 2)
        cv2.rectangle(frame, (icon_rect[0], icon_rect[1]), (icon_rect[2], icon_rect[3]), (210, 210, 210), 1)
        cv2.putText(frame, icon_char, (icon_tx, icon_ty), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, icon_char, (icon_tx, icon_ty), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (250, 250, 250), text_th, cv2.LINE_AA)

    return frame, header_rect, row_toggle_rects


def draw_key_joint_coords_panel(frame, anchor_rect, panel_expanded, row_visible, norm_kpts, ui_scale):
    """緊貼在 KEY JOINTS 面板右側的第二個可收合面板，顯示同一批 7 個關鍵點目前的
    正規化座標 (x, y)。跟 KEY JOINTS 面板一樣支援：收合成一個小按鈕「NORM COORDS >」、
    展開後每列右側有獨立的 +/- 顯示切換；面板寬度一律假設「全部列都顯示數值」去
    算，不受目前隱藏狀態影響，避免使用者隱藏某列時版面跟著跳動變形。

    anchor_rect 是 draw_key_joint_panel() 回傳的 header_rect，用它的右邊界＋間距
    當這個面板的錨點，跟著 KEY JOINTS 面板一起變動、對齊同一條水平線；跟
    draw_key_joint_panel 一樣，回傳 (frame, header_rect, row_toggle_rects) 供外層
    滑鼠回呼判斷點擊命中。"""
    s_ui = ui_scale * PANEL_SCALE
    text_th   = scale_px(2, s_ui, min_px=1)
    shadow_th = scale_px(2, s_ui, min_px=2)
    pad = max(6, int(8 * s_ui))
    header_fs = 0.58 * s_ui
    gap = max(10, int(14 * ui_scale))

    x0 = anchor_rect[2] + gap
    y0 = anchor_rect[1]
    h, w = frame.shape[:2]

    if not panel_expanded:
        header = "NORM COORDS >"
        (tw, th_) = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, header_fs, shadow_th)[0]
        bw, bh = tw + pad * 2, th_ + pad * 2
        if x0 + bw > w - 4 or y0 + bh > h - 4:
            return frame, (x0, y0, x0, y0), {}
        roi = frame[y0:y0 + bh, x0:x0 + bw]
        if roi.size > 0:
            dark = np.full_like(roi, 10)
            cv2.addWeighted(roi, 0.28, dark, 0.72, 0, roi)
            frame[y0:y0 + bh, x0:x0 + bw] = roi
        cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (210, 210, 210), 1)
        ty = y0 + pad + th_
        cv2.putText(frame, header, (x0 + pad, ty), cv2.FONT_HERSHEY_SIMPLEX, header_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, header, (x0 + pad, ty), cv2.FONT_HERSHEY_SIMPLEX, header_fs, (250, 250, 250), text_th, cv2.LINE_AA)
        return frame, (x0, y0, x0 + bw, y0 + bh), {}

    # ── 展開狀態：固定順序，每列右側有獨立的 +/- 顯示切換 ──
    header = "NORM COORDS <"
    row_fs  = 0.52 * s_ui
    row_h   = max(22, int(30 * s_ui))
    icon_sz = max(18, int(22 * s_ui))
    rgap    = max(8, int(10 * s_ui))

    (header_w, header_h) = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, header_fs, shadow_th)[0]

    def _full_row_text(j):
        if norm_kpts is not None and j < len(norm_kpts):
            x, y = float(norm_kpts[j][0]), float(norm_kpts[j][1])
        else:
            x, y = 0.0, 0.0
        return f"{KEYPOINT_NAMES[j]:<10s} ({x:+.2f},{y:+.2f})"

    # 寬度一律假設「全部列都顯示數值」去算，理由同 draw_key_joint_panel
    full_row_texts = [_full_row_text(j) for j in ALWAYS_TRACKED_JOINTS]
    row_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, row_fs, shadow_th)[0][0] for t in full_row_texts)
    content_w = max(header_w, row_w) + rgap + icon_sz
    panel_w = pad * 2 + content_w
    content_top = y0 + pad + header_h + int(header_h * 0.75)
    panel_h = (content_top - y0) + row_h * len(ALWAYS_TRACKED_JOINTS) + pad

    if x0 + panel_w > w - 4 or y0 + panel_h > h - 4:
        return frame, (x0, y0, x0, y0), {}

    roi = frame[y0:y0 + panel_h, x0:x0 + panel_w]
    if roi.size > 0:
        dark = np.full_like(roi, 10)
        cv2.addWeighted(roi, 0.24, dark, 0.76, 0, roi)
        frame[y0:y0 + panel_h, x0:x0 + panel_w] = roi
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (210, 210, 210), 1)

    tx = x0 + pad
    header_ty = y0 + pad + header_h
    header_rect = (x0, y0, x0 + panel_w, content_top)
    cv2.putText(frame, header, (tx, header_ty), cv2.FONT_HERSHEY_SIMPLEX, header_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
    cv2.putText(frame, header, (tx, header_ty), cv2.FONT_HERSHEY_SIMPLEX, header_fs, (250, 250, 250), text_th, cv2.LINE_AA)
    cv2.line(frame, (x0 + 2, content_top), (x0 + panel_w - 2, content_top), (90, 90, 90), 1)

    icon_x = x0 + panel_w - pad - icon_sz
    row_toggle_rects = {}
    for idx, j in enumerate(ALWAYS_TRACKED_JOINTS):
        row_top = content_top + idx * row_h
        text_y = row_top + row_h - max(6, int(7 * s_ui))
        visible = row_visible.get(j, True)

        if visible:
            row_text = _full_row_text(j)
            col = NEUTRAL_MARKER_COLOR
        else:
            row_text = f"{KEYPOINT_NAMES[j]}"
            col = DIMMED_TEXT_COLOR
        cv2.putText(frame, row_text, (tx, text_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, row_text, (tx, text_y), cv2.FONT_HERSHEY_SIMPLEX, row_fs, col, text_th, cv2.LINE_AA)

        # +/- 圖示：該列右手邊，點擊切換這個關節顯示/隱藏
        icon_y0 = row_top + max(0, (row_h - icon_sz) // 2)
        icon_rect = (icon_x, icon_y0, icon_x + icon_sz, icon_y0 + icon_sz)
        row_toggle_rects[j] = icon_rect
        icon_char = "-" if visible else "+"
        (icw, ich) = cv2.getTextSize(icon_char, cv2.FONT_HERSHEY_SIMPLEX, row_fs, shadow_th)[0]
        icon_tx = icon_x + max(0, (icon_sz - icw) // 2)
        icon_ty = icon_y0 + icon_sz - max(0, (icon_sz - ich) // 2)
        cv2.rectangle(frame, (icon_rect[0], icon_rect[1]), (icon_rect[2], icon_rect[3]), (210, 210, 210), 1)
        cv2.putText(frame, icon_char, (icon_tx, icon_ty), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
        cv2.putText(frame, icon_char, (icon_tx, icon_ty), cv2.FONT_HERSHEY_SIMPLEX, row_fs, (250, 250, 250), text_th, cv2.LINE_AA)

    return frame, header_rect, row_toggle_rects


# ===== Graph Lab（g 鍵）：互動畫節點/邊，教學用來解釋「圖 = 節點 + 邊」這個 GCN 的核心概念 =====
GRAPH_NODE_COLOR     = (80, 220, 120)   # 使用者自己畫的節點/邊：綠色
GRAPH_SELECTED_COLOR = (0, 220, 255)    # 等待連邊的「已選取」節點：黃色
REAL_GRAPH_COLOR     = (255, 120, 255)  # 疊上去對照用的「模型真實骨架圖」：洋紅色，跟綠色明顯區分
GRAPH_HIT_RADIUS_PX  = 14               # 點擊命中既有節點的判定半徑（原始像素，不隨 ui_scale 縮放，貼近滑鼠手感）


def draw_graph_lab(frame, nodes, edges, selected, show_real_graph, kpts, kpt_conf, ui_scale,
                    conf_thresh=DRAW_KP_CONF_THRESHOLD):
    """互動式圖形教學疊層。使用者點出來的節點/邊用綠色畫；開啟 [s] 後額外疊一份
    模型實際使用的骨架圖（洋紅色，17 個關鍵點固定連線）——兩者放在一起對照，
    直接示範「圖＝節點＋邊」，以及「你剛剛隨手畫的是任意圖，模型用的是固定
    結構的圖」這個差異，呼應 ST-GCN 名稱裡的 Graph Convolutional 概念。"""
    node_r = scale_px(6, ui_scale, min_px=4)
    edge_th = max(1, int(2 * ui_scale))

    if show_real_graph and kpts is not None and kpt_conf is not None:
        n_kp = min(len(kpts), len(kpt_conf))
        for a, b in _SKELETON_EDGES:
            if a >= n_kp or b >= n_kp:
                continue
            if float(kpt_conf[a]) > conf_thresh and float(kpt_conf[b]) > conf_thresh:
                pa = (int(kpts[a][0]), int(kpts[a][1]))
                pb = (int(kpts[b][0]), int(kpts[b][1]))
                cv2.line(frame, pa, pb, REAL_GRAPH_COLOR, edge_th, cv2.LINE_AA)
        for i in range(min(17, len(kpts))):
            if float(kpt_conf[i]) > conf_thresh:
                cx, cy = int(kpts[i][0]), int(kpts[i][1])
                cv2.circle(frame, (cx, cy), node_r, REAL_GRAPH_COLOR, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), node_r, (0, 0, 0), 1, cv2.LINE_AA)

    for (i, j) in edges:
        if i < len(nodes) and j < len(nodes):
            cv2.line(frame, nodes[i], nodes[j], GRAPH_NODE_COLOR, edge_th, cv2.LINE_AA)

    for idx, (x, y) in enumerate(nodes):
        is_selected = (idx == selected)
        color = GRAPH_SELECTED_COLOR if is_selected else GRAPH_NODE_COLOR
        r = node_r + (scale_px(3, ui_scale, min_px=2) if is_selected else 0)
        cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, (0, 0, 0), max(1, int(1.5 * ui_scale)), cv2.LINE_AA)
        label = str(idx)
        fs = 0.4 * ui_scale
        lth = max(1, int(ui_scale))
        cv2.putText(frame, label, (x + r + 3, y - r), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), lth + 2, cv2.LINE_AA)
        cv2.putText(frame, label, (x + r + 3, y - r), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), lth, cv2.LINE_AA)

    h, w = frame.shape[:2]
    hud1 = "GRAPH LAB [g]  click empty=add node   click 2 nodes=connect edge   right-click node=delete"
    hud2 = (f"[e]clear   [s]{'hide' if show_real_graph else 'show'} model's real graph (magenta)   "
            f"nodes={len(nodes)}  edges={len(edges)}")
    fs = 0.42 * ui_scale
    th = max(1, int(ui_scale))
    y1 = h - scale_px(56, ui_scale, min_px=40)
    y2 = h - scale_px(40, ui_scale, min_px=28)
    for txt, ypos in ((hud1, y1), (hud2, y2)):
        (tw, _th) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0]
        tx = max(6, (w - tw) // 2)
        cv2.putText(frame, txt, (tx, ypos), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(frame, txt, (tx, ypos), cv2.FONT_HERSHEY_SIMPLEX, fs, (140, 255, 180), th, cv2.LINE_AA)

    return frame


# ===== Sliding Window Lab（w 鍵）：把 SEQUENCE_LENGTH 滑動視窗切幀畫成即時動畫 =====
def draw_sliding_window_panel(frame, buffer_len, seq_length, classify_stride, just_classified, ui_scale):
    """畫一排格子代表目前緩衝區裡的 seq_length 幀（最右邊＝最新一幀）。緩衝區還沒
    填滿時，格子由右往左依序點亮；填滿後每次觸發分類（每 classify_stride 幀一次），
    整個面板外框會亮一下，直接把「用最近 N 幀當一個輸入樣本，每隔幾幀重新分類
    一次」這個滑動視窗概念變成看得到的動畫，不用只靠口頭解釋。"""
    s_ui = ui_scale * PANEL_SCALE
    cell_w = max(10, int(16 * s_ui))
    cell_h = max(18, int(28 * s_ui))
    cell_gap = max(1, int(2 * s_ui))
    pad = max(6, int(8 * s_ui))
    fs = 0.5 * s_ui
    label_fs = fs * 0.75
    text_th   = scale_px(2, s_ui, min_px=1)
    shadow_th = scale_px(2, s_ui, min_px=2)

    n = seq_length
    strip_w = n * cell_w + (n - 1) * cell_gap
    header = f"SLIDING WINDOW   seq_len={seq_length}  stride={classify_stride}  buffer={min(buffer_len, n)}/{n}"
    (header_w, header_h) = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, fs, shadow_th)[0]
    label_h = int(header_h * 0.8)
    panel_w = pad * 2 + max(strip_w, header_w)
    panel_h = pad * 3 + header_h + cell_h + label_h

    h, w = frame.shape[:2]
    x0 = max(6, (w - panel_w) // 2)
    y0 = max(6, h - panel_h - scale_px(70, ui_scale, min_px=55))

    roi = frame[y0:y0 + panel_h, x0:x0 + panel_w]
    if roi.size > 0:
        dark = np.full_like(roi, 10)
        cv2.addWeighted(roi, 0.26, dark, 0.74, 0, roi)
        frame[y0:y0 + panel_h, x0:x0 + panel_w] = roi
    border_col = (140, 255, 180) if just_classified else (210, 210, 210)
    border_th = max(1, int((3 if just_classified else 1) * ui_scale))
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), border_col, border_th)

    tx = x0 + pad
    ty = y0 + pad + header_h
    label_col = (140, 255, 180) if just_classified else (230, 230, 230)
    cv2.putText(frame, header, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
    cv2.putText(frame, header, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, label_col, text_th, cv2.LINE_AA)

    strip_x0 = x0 + (panel_w - strip_w) // 2
    strip_y0 = ty + pad
    filled = min(buffer_len, n)
    for i in range(n):
        cx0 = strip_x0 + i * (cell_w + cell_gap)
        is_filled = i >= (n - filled)   # 靠右對齊：緩衝區還沒填滿時，空格留在左邊
        is_newest = is_filled and i == n - 1
        col = (140, 255, 180) if is_newest else ((80, 200, 255) if is_filled else (45, 45, 45))
        cv2.rectangle(frame, (cx0, strip_y0), (cx0 + cell_w, strip_y0 + cell_h), col, -1)
        cv2.rectangle(frame, (cx0, strip_y0), (cx0 + cell_w, strip_y0 + cell_h), (15, 15, 15), 1)

    arrow_y = strip_y0 + cell_h + label_h
    cv2.putText(frame, "OLD", (strip_x0, arrow_y), cv2.FONT_HERSHEY_SIMPLEX, label_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
    cv2.putText(frame, "OLD", (strip_x0, arrow_y), cv2.FONT_HERSHEY_SIMPLEX, label_fs, (180, 180, 180), text_th, cv2.LINE_AA)
    _new_txt = "NEW"
    (nw, _nh) = cv2.getTextSize(_new_txt, cv2.FONT_HERSHEY_SIMPLEX, label_fs, shadow_th)[0]
    _new_x = strip_x0 + strip_w - nw
    cv2.putText(frame, _new_txt, (_new_x, arrow_y), cv2.FONT_HERSHEY_SIMPLEX, label_fs, (0, 0, 0), shadow_th, cv2.LINE_AA)
    cv2.putText(frame, _new_txt, (_new_x, arrow_y), cv2.FONT_HERSHEY_SIMPLEX, label_fs, (180, 180, 180), text_th, cv2.LINE_AA)

    return frame


def _norm_kpts_to_display(norm_kpts, frame_h, frame_w):
    """Map normalized skeleton coords (center=joint 4, unit=chest-hip dist) to pixel coords."""
    scale = min(frame_h, frame_w) / 6.0
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    disp = np.empty_like(norm_kpts, dtype=np.float32)
    disp[:, 0] = cx + norm_kpts[:, 0] * scale
    disp[:, 1] = cy + norm_kpts[:, 1] * scale
    return disp


# ===== 正規化座標面板（右下角）=====
_ALL_KP_NAMES = [
    "Nose", "LEar", "REar", "Chst", "MidB",
    "Hip ", "LFEl", "LFPw", "RFEl", "RFPw",
    "LHKn", "LHPw", "RHKn", "RHPw", "TRot",
    "TMid", "TTip",
]

def draw_norm_coords_panel(frame, norm_history):
    """右下角：顯示當前幀全部 17 個關鍵點的正規化座標（單欄即時更新）。"""
    if not norm_history:
        return frame

    fnum, kpts = norm_history[-1]   # 只取最新一筆
    h, w = frame.shape[:2]
    ui_scale = compute_ui_scale(w, h)
    fs    = max(0.28, 0.34 * ui_scale)
    th    = 1
    sh    = 2
    row_h = max(13, int(16 * ui_scale))
    pad   = max(4, int(6 * ui_scale))

    n_kp    = min(len(_ALL_KP_NAMES), len(kpts))  # 14 關節模型會截斷尾巴三點 14-16
    n_rows  = 1 + n_kp   # 標題列 + 關節列
    col_w   = cv2.getTextSize("10 LHKn(+0.00,+0.00)", cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0]
    panel_w = pad + col_w + pad
    panel_h = pad + n_rows * row_h + pad

    margin = max(6, int(8 * ui_scale))
    x0 = max(0, w - panel_w - margin)
    y0 = max(0, h - panel_h - margin)

    # 半透明深色背景
    roi = frame[y0:y0 + panel_h, x0:x0 + panel_w]
    if roi.size > 0:
        dark = np.full_like(roi, 18)
        cv2.addWeighted(roi, 0.30, dark, 0.70, 0, roi)
        frame[y0:y0 + panel_h, x0:x0 + panel_w] = roi
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (75, 75, 75), 1)

    tx0   = x0 + pad
    cur_y = y0 + pad + row_h

    # 標題列（含幀號）
    title = f"NORM COORDS  F:{fnum}"
    cv2.putText(frame, title, (tx0, cur_y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), sh, cv2.LINE_AA)
    cv2.putText(frame, title, (tx0, cur_y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 80), th, cv2.LINE_AA)
    cur_y += row_h

    # 關鍵點（單欄，數量依實際模型關節數而定）
    for jidx, jname in enumerate(_ALL_KP_NAMES[:n_kp]):
        xy  = kpts[jidx]
        row = f"{jidx:2d} {jname}({xy[0]:+.2f},{xy[1]:+.2f})"
        cv2.putText(frame, row, (tx0, cur_y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), sh, cv2.LINE_AA)
        cv2.putText(frame, row, (tx0, cur_y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (215, 215, 215), th, cv2.LINE_AA)
        cur_y += row_h

    return frame


def main():
    # use a local mutable copy to avoid modifying module-level constant
    feature_mode = STGCN_FEATURE_MODE

    # 解析所有資料夾的影片清單（啟動時一次完成）
    folder_videos: dict = {}
    for fkey, (fpath, fname) in FOLDER_MAP.items():
        vids = resolve_video_paths([fpath])
        folder_videos[fkey] = vids
        print(f"  [{fkey}] {fname}: {len(vids)} 部影片  ({fpath})")

    # 若指定了 VIDEO_PATHS 就用那個；否則依 FOLDER_TEST_MODE 決定播放清單
    if VIDEO_PATHS:
        video_paths = resolve_video_paths(VIDEO_PATHS)
        current_folder_key = DEFAULT_FOLDER_KEY
    elif FOLDER_TEST_MODE == 'all':
        # 所有行為子資料夾依 FOLDER_MAP 順序合併
        video_paths = []
        for fkey in FOLDER_MAP:
            video_paths.extend(folder_videos[fkey])
        current_folder_key = DEFAULT_FOLDER_KEY
        print(f"[FOLDER_TEST_MODE=all] 已合併全部 {len(video_paths)} 部影片")
    else:
        # 'single'：掃描 SINGLE_FOLDER_PATH 扁平資料夾，影片直接放在該目錄
        video_paths = resolve_video_paths([SINGLE_FOLDER_PATH])
        current_folder_key = DEFAULT_FOLDER_KEY
        print(f"[FOLDER_TEST_MODE=single] {SINGLE_FOLDER_PATH}  共 {len(video_paths)} 部影片")

    if not video_paths:
        print("❌ 找不到可用影片，請確認 FOLDER_MAP / VIDEO_PATHS 的路徑")
        return

    # 記住每個資料夾上次的播放位置（切回去時能續播）
    folder_positions: dict = {k: 0 for k in FOLDER_MAP}
    switch_folder_key: str = ""   # 非空時代表要切換資料夾

    display_window = DISPLAY_WINDOW
    loop_playback = LOOP_PLAYBACK

    print("="*60)
    print("骨架正規化視覺化 Demo")
    print("="*60)
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
        ck_channel_map = {2: 'xy', 3: 'xy_conf', 5: 'xy_conf_v', 7: 'xy_conf_v_bone', 9: 'xy_conf_v_bone_bmotion'}
        import torch
        if os.path.exists(STGCN_MODEL_PATH):
            try:
                try:
                    ck = torch.load(STGCN_MODEL_PATH, map_location='cpu', weights_only=True)
                except Exception:
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
    
    try:
        keypoint_detector = KeypointDetector(
            YOLO_MODEL_PATH,
            device=INFERENCE_DEVICE,
            imgsz=YOLO_IMGSZ,
            conf_thres=YOLO_CONF_THRESHOLD,
        )
    except Exception as e:
        print(f"❌ 無法載入 YOLO 模型（{YOLO_MODEL_PATH}）：{e}")
        return

    try:
        behavior_classifier = BehaviorClassifier(
            STGCN_MODEL_PATH,
            device=INFERENCE_DEVICE,
            sequence_length=SEQUENCE_LENGTH,
            normalize=STGCN_NORMALIZE,
            feature_mode=feature_mode,
            in_channels=in_channels,
        )
    except Exception as e:
        print(f"❌ 無法載入 ST-GCN 模型（{STGCN_MODEL_PATH}）：{e}")
        return
    visualizer = Visualizer()

    # 狀態控制
    paused = False
    stop_requested = False
    current_video_idx = 0
    show_overlay_info = True
    show_all_panels   = True   # u 鍵切換：關閉/開啟所有分析面板
    show_joint_highlight = True  # h 鍵切換：鼻子/腳掌關鍵點標記＋面板整體開關
    panel_expanded = False       # 左上角「KEY JOINTS」面板收合狀態，點擊標題列切換
    row_visible = {j: True for j in ALWAYS_TRACKED_JOINTS}  # 每個關節列自己的顯示/隱藏狀態
    _panel_button_rect = (0, 0, 0, 0)   # 每幀更新：收合按鈕／展開後標題列的可點擊範圍
    _row_toggle_rects = {}              # 每幀更新：{joint_idx: rect}，展開時每列 +/- 圖示的可點擊範圍

    coords_panel_expanded = False       # 右側「NORM COORDS」面板收合狀態，獨立於 KEY JOINTS
    coords_row_visible = {j: True for j in ALWAYS_TRACKED_JOINTS}  # 這個面板自己的每列顯示/隱藏狀態
    _coords_button_rect = (0, 0, 0, 0)  # 每幀更新：NORM COORDS 收合按鈕／標題列的可點擊範圍
    _coords_toggle_rects = {}           # 每幀更新：NORM COORDS 展開時每列 +/- 圖示的可點擊範圍

    # Graph Lab（g 鍵）狀態：使用者點出來的節點/邊、目前選取中（等待連邊）的節點、
    # 是否疊上模型真實骨架圖對照
    graph_mode = False
    graph_nodes = []      # [(x, y), ...] 螢幕像素座標
    graph_edges = []      # [(i, j), ...] graph_nodes 的索引配對
    graph_selected = None # 目前選取、等待點第二下連邊的節點索引；None＝沒有選取
    show_real_graph = False

    # Sliding Window Lab（w 鍵）狀態
    window_viz_mode = False

    def _hit_test_graph_node(x, y):
        for idx, (nx, ny) in enumerate(graph_nodes):
            if (nx - x) ** 2 + (ny - y) ** 2 <= GRAPH_HIT_RADIUS_PX ** 2:
                return idx
        return None

    def _on_key_panel_click(event, x, y, flags, param):
        nonlocal panel_expanded, coords_panel_expanded, graph_selected
        if event == cv2.EVENT_LBUTTONDOWN:
            # 先比對兩個面板「每一列」的 +/- 圖示（範圍較小、優先權較高），
            # 命中就只切換該列，不動整體收合狀態
            for j, rect in _row_toggle_rects.items():
                rx0, ry0, rx1, ry1 = rect
                if rx0 <= x <= rx1 and ry0 <= y <= ry1:
                    row_visible[j] = not row_visible[j]
                    return
            for j, rect in _coords_toggle_rects.items():
                rx0, ry0, rx1, ry1 = rect
                if rx0 <= x <= rx1 and ry0 <= y <= ry1:
                    coords_row_visible[j] = not coords_row_visible[j]
                    return
            bx0, by0, bx1, by1 = _panel_button_rect
            if bx0 <= x <= bx1 and by0 <= y <= by1:
                panel_expanded = not panel_expanded
                return
            cx0, cy0, cx1, cy1 = _coords_button_rect
            if cx0 <= x <= cx1 and cy0 <= y <= cy1:
                coords_panel_expanded = not coords_panel_expanded
                return
            # 都沒命中任何面板按鈕，且目前在 Graph Lab 模式 → 當成畫圖的點擊
            if graph_mode:
                hit = _hit_test_graph_node(x, y)
                if hit is None:
                    graph_nodes.append((x, y))
                    graph_selected = None
                elif graph_selected is None:
                    graph_selected = hit
                elif graph_selected == hit:
                    graph_selected = None
                else:
                    edge = (min(graph_selected, hit), max(graph_selected, hit))
                    if edge not in graph_edges:
                        graph_edges.append(edge)
                    graph_selected = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            if graph_mode:
                hit = _hit_test_graph_node(x, y)
                if hit is not None:
                    graph_nodes.pop(hit)
                    graph_edges[:] = [
                        (i - (i > hit), j - (j > hit))
                        for (i, j) in graph_edges if i != hit and j != hit
                    ]
                    if graph_selected == hit:
                        graph_selected = None
                    elif graph_selected is not None and graph_selected > hit:
                        graph_selected -= 1

    # 正規化消融 + overlay 模式（可於播放中即時切換）
    norm_state = {
        'flip':        NORM_FLIP,
        'orient':      NORM_ORIENT,
        'coord':       NORM_COORD,
        'overlay_raw': OVERLAY_RAW,
    }

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
        nonlocal ema_kpts, _last_norm_kpts, _last_norm_kconf
        nonlocal _last_raw_kpts, _last_raw_kconf, _last_prenorm_kpts, _last_joint_motion
        nonlocal local_last_behavior
        nonlocal raw_frames_read, local_frames_processed, local_sampled_frames
        nonlocal local_behavior_duration_sec, local_behavior_current_confidences
        nonlocal local_behavior_occurrence_counts, local_last_behavior_for_occurrence

        keypoints_buffer.clear()
        ema_kpts = None
        keypoint_detector.reset_track()  # 避免上一段影片鎖定的貓誤帶到新的一段
        _last_norm_kpts = None
        _last_norm_kconf = None
        _last_raw_kpts = None
        _last_raw_kconf = None
        _last_prenorm_kpts = None
        _last_joint_motion = None
        _norm_history.clear()
        local_last_behavior = None
        raw_frames_read = 0
        local_frames_processed = 0
        local_sampled_frames = 0
        local_behavior_duration_sec = np.zeros(5, dtype=np.float64)
        local_behavior_current_confidences = np.zeros(5, dtype=np.float32)
        local_behavior_occurrence_counts = np.zeros(5, dtype=np.int64)
        local_last_behavior_for_occurrence = LOW_CONF_ID
        reset_behavior_display_state()

    if display_window:
        if DISPLAY_SIZE is not None:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, _on_key_panel_click)

    while not stop_requested:
        video_path = video_paths[current_video_idx]
        is_stream_url = _is_stream_url(video_path)
        if not is_stream_url and not Path(video_path).exists():
            print(f"❌ 影片不存在，跳過: {video_path}")
            current_video_idx = (current_video_idx + 1) % len(video_paths)
            continue

        if is_stream_url:
            cap = open_video_capture_with_retry(video_path, retries=5, delay=3)
            if cap is None or not cap.isOpened():
                print(f"❌ 無法開啟串流 {video_path}，請確認 URL 與網路連線，跳過")
                current_video_idx = (current_video_idx + 1) % len(video_paths)
                continue
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ 無法開啟影片，跳過: {video_path}")
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
        print("控制: q=退出  space=暫停  r=重置  1/2=上/下部  z/x/c/v/b=切換資料夾  i=資訊  u=全部面板  h=關鍵點標記開關（點擊左上角 KEY JOINTS 展開/收合數值）")
        print("教學: g=Graph Lab（點畫布=節點/邊，e=清空，s=疊模型真實骨架圖對照）  w=Sliding Window Lab（滑動視窗切幀動畫）")
        print("消融: f=flip  o=orient  n=coord  p=raw/norm overlay")
        if loop_playback:
            print("🔁 循環播放模式（當前影片播完會重播）")
        print("-" * 60)

        keypoints_buffer = deque(maxlen=SEQUENCE_LENGTH)
        local_loop_count = 0
        switch_delta = 0
        first_pass_completed = False
        switched_before_first_pass_complete = False

        reset_behavior_display_state()

        # EMA 狀態：跨幀累積，切影片或貓消失時重置
        ema_kpts = None  # shape (17, 2)，儲存上一幀的 EMA 平滑座標
        keypoint_detector.reset_track()  # 新影片開始，避免延續上一支影片鎖定的貓
        _last_norm_kpts = None   # shape (V, 2)，最近一次正規化後的骨架座標（供顯示用）
        _last_norm_kconf = None  # shape (V,)，對應信心值
        _last_raw_kpts   = None   # shape (17, 2)，原始像素座標（overlay_raw 模式顯示用）
        _last_raw_kconf  = None   # shape (17,)
        _last_prenorm_kpts = None # shape (V, 2)，flip+orient 後、coord 正規化前的像素座標（n=OFF 時顯示用）
        _last_joint_motion = None # shape (V,)，最近一個 window 的逐關節動作幅度（供 h 高亮動畫用）
        _norm_history = deque(maxlen=1)  # (frame_num, norm_kpts) 供右下角座標面板使用；只保留最新一幀

        # 本次影片播放期間的即時 HUD 狀態
        local_last_behavior = None
        raw_frames_read = 0
        local_frames_processed = 0
        local_sampled_frames = 0
        local_behavior_duration_sec = np.zeros(5, dtype=np.float64)
        local_behavior_current_confidences = np.zeros(5, dtype=np.float32)
        local_behavior_occurrence_counts = np.zeros(5, dtype=np.int64)
        local_last_behavior_for_occurrence = LOW_CONF_ID

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
            just_classified = False  # 這一幀是否剛好觸發了一次分類，供 Sliding Window Lab 面板閃爍提示

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

                # 儲存原始像素座標供 overlay_raw 模式顯示
                _last_raw_kpts  = kpts.copy()
                _last_raw_kconf = kpt_conf.copy()

                # 加入緩衝區
                keypoints_buffer.append((kpts, kpt_conf))

                # 有足夠序列時做行為分類
                if len(keypoints_buffer) >= SEQUENCE_LENGTH and (local_sampled_frames % CLASSIFY_STRIDE == 0):
                    just_classified = True
                    kpts_arr = np.array([item[0] for item in keypoints_buffer])  # (T, 17, 2)
                    conf_arr = np.array([item[1] for item in keypoints_buffer])  # (T, 17)

                    # 插值補全（threshold=0.1 與訓練端一致）—— 在截斷關節數之前先做完，
                    # 尾巴三點（14-16）才能一起被插值、正規化，純視覺骨架才畫得出完整尾巴。
                    # flip/orient/coord 三個正規化函式都只讀 chest/mid_back/hip（索引 3/4/5），
                    # 跟陣列裡有沒有多帶尾巴三欄無關，所以這裡先對完整 17 點做，結果跟「先截斷
                    # 再正規化」對索引 0-13 完全一樣，不影響模型分類，只是多留了尾巴給顯示用。
                    seq_array = interpolate_missing(kpts_arr, conf_arr)
                    if STGCN_NORMALIZE:
                        # 正規化消融：各步驟獨立可關閉（f/o/n 鍵動態切換）
                        if norm_state['flip']:
                            seq_array = flip_normalize(seq_array)
                        if norm_state['orient']:
                            seq_array = orientation_normalize(seq_array)
                        # flip+orient 後、coord 前的像素座標：n=OFF 時用來在黑底上
                        # 顯示骨架的真實位置與大小，讓教授直觀看見正規化的效果
                        _last_prenorm_kpts = seq_array[-1].copy()
                        if norm_state['coord']:
                            seq_array = normalize_skeleton_coords(seq_array)
                    else:
                        _last_prenorm_kpts = seq_array[-1].copy()
                    _last_norm_kpts = seq_array[-1].copy()   # 17 點（含尾巴），純顯示用
                    _last_norm_kconf = conf_arr[-1].copy()   # 17 點
                    _norm_history.append((local_frames_processed, seq_array[-1].copy()))
                    _last_joint_motion = _compute_per_joint_motion(seq_array, conf_arr)

                    # 截斷至模型關節數（14 節點模型忽略尾巴三點 14-16）——只影響送進模型的
                    # 張量，不影響上面已經算好、給畫面顯示用的 17 點座標／動作幅度。
                    _model_nj = getattr(behavior_classifier.model, 'num_joints', 17)
                    if _model_nj < seq_array.shape[1]:
                        model_seq_array = seq_array[:, :_model_nj, :]
                        model_conf_arr  = conf_arr[:, :_model_nj]
                    else:
                        model_seq_array = seq_array
                        model_conf_arr  = conf_arr
                    seq_features = build_feature_tensor(model_seq_array, model_conf_arr, feature_mode)
                    # 外部已完成正規化，直接呼叫 model.predict(precomputed=True) 避免雙重正規化
                    pred_id, pred_conf, pred_probs = behavior_classifier.model.predict(seq_features, precomputed=True)

                    # ── 診斷：每 60 幀列印一次，確認輸入信心與模型輸出是否正常 ──
                    _dbg_every = 60
                    if local_sampled_frames % _dbg_every < CLASSIFY_STRIDE:
                        _cmean = float(model_conf_arr.mean())
                        _czero = int((model_conf_arr < 0.05).sum())
                        _ctot  = int(model_conf_arr.size)
                        _pstr  = ' '.join(f'{p*100:.1f}' for p in (pred_probs or []))
                        print(f"[STGCN DBG] frame={local_frames_processed:5d} "
                              f"kpt_conf mean={_cmean:.3f} zeros<0.05={_czero}/{_ctot} "
                              f"num_classes={len(pred_probs) if pred_probs else '?'} "
                              f"probs=[{_pstr}]%")
                    # ─────────────────────────────────────────────────────────

                    if pred_id is None:
                        behavior_id = LOW_CONF_ID
                        confidence = 0.0
                        probs = np.zeros(5, dtype=np.float32)
                    else:
                        behavior_id = int(pred_id)
                        confidence = float(pred_conf)
                        probs = pred_probs.copy()

                    # Update the per-class current confidences for the bottom panel
                    # so the UI shows the latest probabilities for all classes.
                    # Always update (not just first pass) so bars refresh on replay.
                    # 若模型為 4-class，把 probs 補齊到 5 元素，避免 STOP bar 永遠為 0
                    _probs_padded = list(probs) if probs is not None else [0.0]
                    while len(_probs_padded) < len(BEHAVIOR_PANEL_LABELS):
                        _probs_padded.append(0.0)
                    local_behavior_current_confidences = _probs_padded

                    # 與主系統一致：低信心顯示「目前正常」
                    if confidence < BEHAVIOR_MIN_CONFIDENCE:
                        behavior_id_for_display = LOW_CONF_ID
                    else:
                        behavior_id_for_display = behavior_id

                    # 計算行為發生次數：切換到「不同的有效行為」時計為一次新發生
                    # 貓消失或低信心 (LOW_CONF_ID) 期間會重置 local_last_behavior_for_occurrence，
                    # 所以中斷後再次出現同一行為也會被計為新的一次。
                    if is_first_pass:
                        if (behavior_id_for_display != LOW_CONF_ID
                                and behavior_id_for_display != local_last_behavior_for_occurrence):
                            local_behavior_occurrence_counts[int(behavior_id_for_display)] += 1
                        local_last_behavior_for_occurrence = behavior_id_for_display

                    # 行為切換時印一行即時 log，方便對照畫面講解
                    if behavior_id_for_display != LOW_CONF_ID:
                        behavior_text = get_behavior_name(behavior_id, use_text=False, fallback=str(behavior_id), confidence=confidence)
                        if local_last_behavior != behavior_id:
                            if local_last_behavior is not None and is_first_pass:
                                probs_str = " ".join(
                                    f"{cls}:{probs[i]*100:4.1f}%"
                                    for i, cls in enumerate(BEHAVIOR_CLASSES)
                                    if i < len(probs)
                                )
                                print(f"影片[{current_video_idx}] 幀 {local_frames_processed:6d}: {behavior_text:6s} {confidence*100:5.1f}% [{probs_str}]")
                            local_last_behavior = behavior_id
                    else:
                        behavior_id = LOW_CONF_ID

                if behavior_id != LOW_CONF_ID and 0 <= int(behavior_id) < 5 and float(confidence) >= BEHAVIOR_MIN_CONFIDENCE:
                    if is_first_pass:
                        local_behavior_duration_sec[int(behavior_id)] += frame_dt
                    local_behavior_current_confidences[int(behavior_id)] = float(confidence)
            else:
                ema_kpts = None  # 貓消失時重置 EMA，避免下次出現時使用過時的平均值
                _last_norm_kpts = None
                _last_norm_kconf = None
                _last_raw_kpts = None
                _last_raw_kconf = None
                _last_prenorm_kpts = None
                _last_joint_motion = None
                if is_first_pass:
                    local_last_behavior_for_occurrence = LOW_CONF_ID

            def render_frame():
                # 抽成函式是為了讓「暫停中」也能重畫（例如滑鼠點擊 KEY JOINTS 面板、或
                # 暫停時按 h/f/o/n/p 切換），否則暫停迴圈只 waitKey 不重畫，狀態改了畫面
                # 卻不會更新，使用者會覺得「點了沒反應」。
                nonlocal _panel_button_rect, _row_toggle_rects, _coords_button_rect, _coords_toggle_rects
                # ── overlay_raw 模式：原始影像＋原始骨架；False：黑底＋正規化骨架 ──
                _use_raw = norm_state['overlay_raw']
                if DISPLAY_SIZE is not None:
                    show_frame, _lb_scale, _lb_cx, _lb_cy = resize_with_letterbox(frame, DISPLAY_SIZE)
                    if not _use_raw:
                        show_frame[:] = 0   # 黑底
                    if _use_raw and _last_raw_kpts is not None:
                        _disp_kpts, _disp_bbox = scale_kpts_and_bbox_for_letterbox(
                            _last_raw_kpts, bbox, _lb_scale, _lb_cx, _lb_cy)
                        _disp_conf = _last_raw_kconf
                    elif not _use_raw and not norm_state['coord'] and _last_prenorm_kpts is not None:
                        # n=OFF：用 flip+orient 後的像素座標（隨貓移動，大小反映距離）
                        _disp_kpts, _disp_bbox = scale_kpts_and_bbox_for_letterbox(
                            _last_prenorm_kpts, None, _lb_scale, _lb_cx, _lb_cy)
                        _disp_conf = _last_norm_kconf
                        _disp_bbox = None
                    elif not _use_raw and _last_norm_kpts is not None:
                        _disp_kpts = _norm_kpts_to_display(_last_norm_kpts, show_frame.shape[0], show_frame.shape[1])
                        _disp_bbox = None
                        _disp_conf = _last_norm_kconf
                    else:
                        _disp_kpts = None
                        _disp_bbox = None
                        _disp_conf = None
                    if _disp_kpts is not None:
                        show_frame = draw_test2_style_overlay(
                            show_frame,
                            _disp_kpts,
                            _disp_conf,
                            _disp_bbox if _use_raw else None,
                            behavior_id,
                            confidence,
                            np.array((list(probs) + [0.0] * 5)[:5], dtype=np.float32),
                            visualizer,
                            show_info=(show_overlay_info and show_all_panels),
                        )
                        if show_joint_highlight:
                            _hl_ui_scale = compute_ui_scale(show_frame.shape[1], show_frame.shape[0])
                            show_frame = draw_joint_highlight(
                                show_frame, _disp_kpts, _disp_conf, _last_joint_motion,
                                behavior_id, row_visible, _hl_ui_scale,
                            )
                            show_frame, _panel_button_rect, _row_toggle_rects = draw_key_joint_panel(
                                show_frame, panel_expanded, behavior_id, _last_joint_motion, row_visible, _hl_ui_scale,
                            )
                            show_frame, _coords_button_rect, _coords_toggle_rects = draw_key_joint_coords_panel(
                                show_frame, _panel_button_rect, coords_panel_expanded, coords_row_visible,
                                _last_norm_kpts, _hl_ui_scale,
                            )
                    else:
                        draw_no_cat_overlay(show_frame)
                    if show_overlay_info and show_all_panels:
                        draw_behavior_duration_panel(show_frame, frame_time_sec, local_behavior_duration_sec, local_behavior_current_confidences, local_behavior_occurrence_counts)
                else:
                    _lb_scale, _lb_cx, _lb_cy = 1.0, 0, 0
                    show_frame = frame.copy() if _use_raw else np.zeros_like(frame)
                    if _use_raw and _last_raw_kpts is not None:
                        _disp_kpts = _last_raw_kpts
                        _disp_conf = _last_raw_kconf
                    elif not _use_raw and _last_norm_kpts is not None:
                        _disp_kpts = _norm_kpts_to_display(_last_norm_kpts, show_frame.shape[0], show_frame.shape[1])
                        _disp_conf = _last_norm_kconf
                    else:
                        _disp_kpts = None
                        _disp_conf = None
                    if _disp_kpts is not None:
                        show_frame = draw_test2_style_overlay(
                            show_frame,
                            _disp_kpts,
                            _disp_conf,
                            None,
                            behavior_id,
                            confidence,
                            np.array((list(probs) + [0.0] * 5)[:5], dtype=np.float32),
                            visualizer,
                            show_info=(show_overlay_info and show_all_panels),
                        )
                        if show_joint_highlight:
                            _hl_ui_scale = compute_ui_scale(show_frame.shape[1], show_frame.shape[0])
                            show_frame = draw_joint_highlight(
                                show_frame, _disp_kpts, _disp_conf, _last_joint_motion,
                                behavior_id, row_visible, _hl_ui_scale,
                            )
                            show_frame, _panel_button_rect, _row_toggle_rects = draw_key_joint_panel(
                                show_frame, panel_expanded, behavior_id, _last_joint_motion, row_visible, _hl_ui_scale,
                            )
                            show_frame, _coords_button_rect, _coords_toggle_rects = draw_key_joint_coords_panel(
                                show_frame, _panel_button_rect, coords_panel_expanded, coords_row_visible,
                                _last_norm_kpts, _hl_ui_scale,
                            )
                    else:
                        draw_no_cat_overlay(show_frame)
                    if show_overlay_info and show_all_panels:
                        draw_behavior_duration_panel(show_frame, frame_time_sec, local_behavior_duration_sec, local_behavior_current_confidences, local_behavior_occurrence_counts)
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
                # Place the nav panel at the top-right without a background to avoid
                # overlapping the prediction label at the top-left.
                text_w = txt_size[0]
                right_margin = scale_px(8, _ui, min_px=6)
                x_pos = max(6, _w - right_margin - text_w)
                text_y = int(max(rect_h * 0.7, scale_px(20, _ui, min_px=14)))
                cv2.putText(show_frame, _nav, (x_pos, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, _fs, (160, 210, 255), _th, cv2.LINE_AA)
                # ── 原始影片縮圖（nav 正下方右側）+ 影片名稱 ──
                _fh_r, _fw_r = frame.shape[:2]
                _thumb_w = max(80, _w // 4)
                _thumb_h = int(_thumb_w * _fh_r / _fw_r)
                _thumb = cv2.resize(frame, (_thumb_w, _thumb_h), interpolation=cv2.INTER_AREA)
                _tx_t = _w - _thumb_w - 4
                _ty_t = text_y + 6
                _t_bot = _ty_t + _thumb_h
                if _tx_t >= 0 and _t_bot <= _h:
                    show_frame[_ty_t:_t_bot, _tx_t:_tx_t + _thumb_w] = _thumb
                    cv2.rectangle(show_frame, (_tx_t - 1, _ty_t - 1),
                                  (_tx_t + _thumb_w + 1, _t_bot + 1), (110, 110, 110), 1)
                    cv2.putText(show_frame, "RAW", (_tx_t + 4, _ty_t + 13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 220, 60), 1, cv2.LINE_AA)
                _vname = Path(video_path).name
                _vname_disp = (_vname[:36] + '..') if len(_vname) > 38 else _vname
                _vn_y = _t_bot + 14
                if _tx_t >= 0 and _vn_y < _h:
                    cv2.putText(show_frame, _vname_disp, (_tx_t, _vn_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 200, 255), 1, cv2.LINE_AA)
                # 左下角：模式標示 + 正規化消融狀態
                _mode_label = "RAW OVERLAY" if norm_state['overlay_raw'] else "NORMALIZED"
                _mode_col   = (80, 200, 200) if norm_state['overlay_raw'] else (70, 100, 70)
                cv2.putText(show_frame, _mode_label, (6, _h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4 * _ui, _mode_col, 1, cv2.LINE_AA)
                # 正規化各步驟開關（僅在 STGCN_NORMALIZE=True 時有意義）
                _ns = norm_state
                _norm_hud = (
                    f"[f]flip:{'ON' if _ns['flip'] else 'OFF'}  "
                    f"[o]orient:{'ON' if _ns['orient'] else 'OFF'}  "
                    f"[n]coord:{'ON' if _ns['coord'] else 'OFF'}  "
                    f"[p]raw:{'ON' if _ns['overlay_raw'] else 'OFF'}  "
                    f"[h]highlight:{'ON' if show_joint_highlight else 'OFF'}"
                )
                _nhud_y = _h - 8 - max(13, int(16 * _ui))
                cv2.putText(show_frame, _norm_hud, (6, _nhud_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35 * _ui, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(show_frame, _norm_hud, (6, _nhud_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35 * _ui, (200, 200, 100), 1, cv2.LINE_AA)
                # 右下角正規化座標面板
                if show_all_panels:
                    draw_norm_coords_panel(show_frame, _norm_history)
                # Graph Lab / Sliding Window Lab 教學疊層
                _lab_ui_scale = compute_ui_scale(show_frame.shape[1], show_frame.shape[0])
                if graph_mode:
                    show_frame = draw_graph_lab(
                        show_frame, graph_nodes, graph_edges, graph_selected, show_real_graph,
                        _disp_kpts, _disp_conf, _lab_ui_scale,
                    )
                if window_viz_mode:
                    show_frame = draw_sliding_window_panel(
                        show_frame, len(keypoints_buffer), SEQUENCE_LENGTH, CLASSIFY_STRIDE,
                        just_classified, _lab_ui_scale,
                    )
                cv2.imshow(WINDOW_NAME, show_frame)

            if display_window:
                render_frame()

            if display_window:
                _raw = cv2.waitKey(1)
                key  = _raw & 0xFF if _raw >= 0 else 0xFF
                # 方向鍵 → / ← 作為 2 / 1 的替代（繞過中文輸入法截鍵問題）
                # Windows raw codes: right=2555904, left=2424832
                # Linux  raw codes: right=65363,   left=65361
                _go_next = (key == ord('2')) or (_raw in (2555904, 65363))
                _go_prev = (key == ord('1')) or (_raw in (2424832, 65361))

                if key == ord('q'):
                    print("\n使用者中斷：q")
                    stop_requested = True
                    break
                if key == ord('i'):
                    show_overlay_info = not show_overlay_info
                    print(f"\n資訊面板: {'顯示' if show_overlay_info else '隱藏'}")
                    continue
                if key == ord('u'):
                    show_all_panels = not show_all_panels
                    print(f"\n所有面板: {'顯示' if show_all_panels else '隱藏'}")
                    continue
                if key == ord('h'):
                    show_joint_highlight = not show_joint_highlight
                    print(f"\n關鍵點高亮動畫: {'顯示' if show_joint_highlight else '隱藏'}")
                    continue
                if key == ord('g'):
                    graph_mode = not graph_mode
                    print(f"\nGraph Lab: {'開啟' if graph_mode else '關閉'}"
                          f"（點空白處=新增節點  點兩個節點=連邊  右鍵節點=刪除  e=清空  s=顯示/隱藏模型真實骨架圖）")
                    continue
                if key == ord('e'):
                    graph_nodes.clear()
                    graph_edges.clear()
                    graph_selected = None
                    print("\nGraph Lab: 已清空節點與邊")
                    continue
                if key == ord('s'):
                    show_real_graph = not show_real_graph
                    print(f"\nGraph Lab 疊圖對照（模型真實骨架）: {'顯示' if show_real_graph else '隱藏'}")
                    continue
                if key == ord('w'):
                    window_viz_mode = not window_viz_mode
                    print(f"\nSliding Window Lab: {'開啟' if window_viz_mode else '關閉'}")
                    continue
                if key == ord('f'):
                    norm_state['flip'] = not norm_state['flip']
                    print(f"\n[消融] flip_normalize: {'ON' if norm_state['flip'] else 'OFF'}")
                    continue
                if key == ord('o'):
                    norm_state['orient'] = not norm_state['orient']
                    print(f"\n[消融] orientation_normalize: {'ON' if norm_state['orient'] else 'OFF'}")
                    continue
                if key == ord('n'):
                    norm_state['coord'] = not norm_state['coord']
                    print(f"\n[消融] normalize_skeleton_coords: {'ON' if norm_state['coord'] else 'OFF'}")
                    continue
                if key == ord('p'):
                    norm_state['overlay_raw'] = not norm_state['overlay_raw']
                    print(f"\n[Overlay] {'原始影像 + 原始骨架' if norm_state['overlay_raw'] else '黑底 + 正規化骨架'}")
                    continue
                if _go_next:
                    switch_delta = 1
                    if not first_pass_completed:
                        switched_before_first_pass_complete = True
                    reset_video_runtime_state()
                    print("\n切換到下一部影片")
                    break
                if _go_prev:
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
                        # 暫停中也要重畫：滑鼠點擊 KEY JOINTS 面板（展開/收合、+/- 顯示切換）
                        # 是透過 cv2.setMouseCallback 非同步觸發，waitKey 期間就會生效，但
                        # 沒有這行的話畫面不會重繪，使用者會看不到點擊結果。
                        render_frame()
                        _r2 = cv2.waitKey(50)
                        k2  = _r2 & 0xFF if _r2 >= 0 else 0xFF
                        _p_next = (k2 == ord('2')) or (_r2 in (2555904, 65363))
                        _p_prev = (k2 == ord('1')) or (_r2 in (2424832, 65361))
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
                        elif k2 == ord('u'):
                            show_all_panels = not show_all_panels
                            print(f"\n所有面板: {'顯示' if show_all_panels else '隱藏'}")
                        elif k2 == ord('h'):
                            show_joint_highlight = not show_joint_highlight
                            print(f"\n關鍵點高亮動畫: {'顯示' if show_joint_highlight else '隱藏'}")
                        elif k2 == ord('g'):
                            graph_mode = not graph_mode
                            print(f"\nGraph Lab: {'開啟' if graph_mode else '關閉'}")
                        elif k2 == ord('e'):
                            graph_nodes.clear()
                            graph_edges.clear()
                            graph_selected = None
                            print("\nGraph Lab: 已清空節點與邊")
                        elif k2 == ord('s'):
                            show_real_graph = not show_real_graph
                            print(f"\nGraph Lab 疊圖對照（模型真實骨架）: {'顯示' if show_real_graph else '隱藏'}")
                        elif k2 == ord('w'):
                            window_viz_mode = not window_viz_mode
                            print(f"\nSliding Window Lab: {'開啟' if window_viz_mode else '關閉'}")
                        elif k2 == ord('f'):
                            norm_state['flip'] = not norm_state['flip']
                            print(f"\n[消融] flip_normalize: {'ON' if norm_state['flip'] else 'OFF'}")
                        elif k2 == ord('o'):
                            norm_state['orient'] = not norm_state['orient']
                            print(f"\n[消融] orientation_normalize: {'ON' if norm_state['orient'] else 'OFF'}")
                        elif k2 == ord('n'):
                            norm_state['coord'] = not norm_state['coord']
                            print(f"\n[消融] normalize_skeleton_coords: {'ON' if norm_state['coord'] else 'OFF'}")
                        elif k2 == ord('p'):
                            norm_state['overlay_raw'] = not norm_state['overlay_raw']
                            print(f"\n[Overlay] {'原始影像 + 原始骨架' if norm_state['overlay_raw'] else '黑底 + 正規化骨架'}")
                        elif _p_next:
                            paused = False
                            switch_delta = 1
                            if not first_pass_completed:
                                switched_before_first_pass_complete = True
                            reset_video_runtime_state()
                            print("\n切換到下一部影片")
                            break
                        elif _p_prev:
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

        # 只有完整播放第一輪且非中途切換，才印出本影片行為摘要
        if first_pass_completed and not switched_before_first_pass_complete:
            print(f"✓ 影片[{current_video_idx}] 已完整播放")
            print(f"  ┌─{'─'*10}─┬─{'─'*6}─┬─{'─'*9}─┐")
            print(f"  │ {'行為':<10} │ {'次數':>6} │ {'持續(秒)':>9} │")
            print(f"  ├─{'─'*10}─┼─{'─'*6}─┼─{'─'*9}─┤")
            for _bid in range(5):
                _bname = BEHAVIOR_CLASSES[_bid] if _bid < len(BEHAVIOR_CLASSES) else str(_bid)
                _occ = int(local_behavior_occurrence_counts[_bid])
                _dur = local_behavior_duration_sec[_bid]
                print(f"  │ {_bname:<10} │ {_occ:>6} │ {_dur:>9.2f} │")
            print(f"  └─{'─'*10}─┴─{'─'*6}─┴─{'─'*9}─┘")
        else:
            if switched_before_first_pass_complete:
                print(f"⚠ 影片[{current_video_idx}] 中途切換，略過行為摘要")
            else:
                print(f"⚠ 影片[{current_video_idx}] 未完成第一輪播放，略過行為摘要")

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

        if switch_delta != 0:
            current_video_idx = (current_video_idx + switch_delta) % len(video_paths)
        elif is_stream_url:
            # 串流不做循環播放，結束後維持在當前來源即可
            break
        elif not loop_playback:
            break

    if display_window:
        cv2.destroyAllWindows()

    print("\nDemo 結束")
    print("=" * 60)


if __name__ == "__main__":
    main()
