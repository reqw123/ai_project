"""
Pose Jitter Analysis Script — 貓咪姿勢抖動分析工具
單一影片版：播完自動結束。按 [q] 可提前中止。

====================================================
分析模式說明 (ANALYSIS_MODE)
====================================================

Mode 1 — BBox 穩定性分析
  量化 Bounding Box 中心的逐幀位移大小。
  目的：判斷 BBox 是否為骨架整體抖動的根源。
  輸出：bbox_timeseries.csv、位移時序圖、位移統計圖

Mode 2 — 關鍵點穩定性分析
  量化全部 17 個關鍵點的逐幀絕對位移（像素）。
  目的：找出最不穩定的關鍵點，排名輸出。
  輸出：keypoint_timeseries.csv、keypoint_statistics.csv、
        全17點比較圖、Top5最不穩定點圖

Mode 3 — 關鍵點信心度分析
  分析每個關鍵點的 YOLO 信心分數隨時間變化，
  計算「信心 vs 位移」的 Pearson 相關性
  （r < -0.5 診斷為信心驅動抖動）。
  另含 BBox 偵測信心欄位（bbox_conf）。
  輸出：confidence_timeseries.csv（含 bbox_conf 欄）、
        confidence_statistics.csv（含 BBox 列）、
        confidence_displacement_correlation.csv、
        全17點信心圖、Top5低信心點圖

Mode 4 — BBox EMA 平滑分析
  對 BBox Center 套用 EMA 平滑（α=0.2~1.0），
  計算平滑前後的 BBox 位移與相對關鍵點位移改善幅度。
  目的：量化 BBox 漂移對骨架抖動的貢獻比例。
  輸出：bbox_smoothing_summary.csv、diagnosis.txt、
        BBox位移比較圖、中心點比較圖、Alpha掃描圖×3

Mode 5 — 關鍵點 EMA 平滑分析
  對全部 17 個關鍵點 x/y 各自套用 EMA 平滑，
  與 Mode 4 BBox 平滑做跨模式比較，自動診斷抖動根源。
  輸出：keypoint_smoothing_summary.csv、
        keypoint_improvement_detail.csv、
        overall_smoothing_summary.csv、diagnosis.txt、
        6 張圖（含跨模式 Before/After 比較圖）

Mode 6 — 全模式一鍵啟動
  依序執行 Mode 1~5，輸出所有分析結果至各自子資料夾。
  適合初次使用或需要完整報告的場合。
  輸出：等同 Mode 1+2+3+4+5 的所有檔案

Mode 7 — ROI Padding Impact on Keypoint Stability
  對每種 padding ratio（0.0~0.5），將 BBox 擴大後重新 crop 並執行
  YOLO Pose 推論，比較 Keypoint P95 Jitter 的改善幅度。
  目的：判斷 ROI 大小是否為抖動根源。
  輸出：padding_summary.csv、3 張圖、diagnosis.txt

====================================================
使用方式
====================================================
  1. 設定 VIDEO_PATH 為目標影片路徑
  2. 設定 ANALYSIS_MODE = 所需模式 (1~6)
  3. 執行腳本，影片播完後自動輸出結果
  4. 輸出根目錄：OUTPUT_DIR / {mode_subdir} / run_YYYYMMDD_HHMMSS/
"""
import sys
import os
import csv
import cv2
import numpy as np
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from collections import deque
from typing import Iterable

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from detectors.keypoint_detector import KeypointDetector
from detectors.behavior_classifier import BehaviorClassifier
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
    LOW_CONF_ID,
    BLACK,
    COLOR_HEAD,
)
from config import BehaviorTrackingConfig as _BehaviorTrackingConfig

# ╔══════════════════════════════════════════════════════════════════╗
# ║          使 用 者 設 定 區（每次執行前只需修改此區）             ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── 分析模式（1~6）──────────────────────────────────────────────────
#   1 = BBox 穩定性分析
#   2 = 關鍵點穩定性分析
#   3 = 關鍵點信心度分析（含 BBox 信心欄）
#   4 = BBox EMA 平滑分析
#   5 = 關鍵點 EMA 平滑分析（含跨模式比較）
#   6 = 全模式一鍵啟動（依序執行 Mode 1~5）
ANALYSIS_MODE = 6
# ── 影片路徑 ────────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\1_貓咪姿勢影片分類\暫存\stop\stop_22.mp4"

# ── 輸出根目錄 ──────────────────────────────────────────────────────
OUTPUT_DIR = Path(r"C:\ai_project\paper\output\jitter_analysis")

# ── Mode 4 / 5 / 6：EMA Alpha 設定 ─────────────────────────────────
#   *_EMA_ALPHA : 供單張比較圖使用的參考 alpha（須包含在 *_EMA_LIST 中）
#   *_EMA_LIST  : alpha 掃描清單（1.0 = 不平滑，作為基準線）
BBOX_EMA_ALPHA     = 0.2
BBOX_EMA_LIST      = [1.0, 0.8, 0.6, 0.4, 0.2]

KEYPOINT_EMA_ALPHA = 0.2
KEYPOINT_EMA_LIST  = [1.0, 0.8, 0.6, 0.4, 0.2]

# ── Mode 7：ROI Padding 設定 ────────────────────────────────────────
#   0.0 = 不填充（原始 bbox）；數值越大 ROI 越寬裕
ROI_PADDING_LIST   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# ── 排除關鍵點（不納入本次分析）──────────────────────────────────────
#   填入要排除的關鍵點 ID（整數集合），留空 = 全部納入
#   0=Nose       1=Left_Ear   2=Right_Ear  3=Chest      4=Mid_Back  5=Hip
#   6=LF_Elbow   7=LF_Paw     8=RF_Elbow   9=RF_Paw
#   10=LH_Knee  11=LH_Paw    12=RH_Knee   13=RH_Paw
#   14=Tail_Root 15=Tail_Mid  16=Tail_Tip
EXCLUDED_KEYPOINTS: set = set({14,15,16})   # 範例：{14, 15, 16} 排除尾巴三點

# ╔══════════════════════════════════════════════════════════════════╗
# ║                      以 下 無 需 修 改                           ║
# ╚══════════════════════════════════════════════════════════════════╝

_MODE_SUBDIR = {
    1: '1_bbox',
    2: '2_keypoint',
    3: '3_confidence',
    4: '4_bbox_smoothing',
    5: '5_keypoint_smoothing',
    6: '6_all_modes',
    7: '7_roi_padding',
}

# 由 EXCLUDED_KEYPOINTS 衍生，供各分析函式共用
_ACTIVE_KP_IDS: list = [i for i in range(17) if i not in EXCLUDED_KEYPOINTS]

YOLO_MODEL_PATH  = r"C:\AI_Project\cat_pose\v11s_121.pt"
STGCN_MODEL_PATH = r"C:\Users\homec\Downloads\stgcn_results\run_121_xy_conf_v_bone_att_on\121_best_model.pth"
INFERENCE_DEVICE = 'cuda'
YOLO_IMGSZ = 640
STGCN_NORMALIZE = True
SEQUENCE_LENGTH = 16
_raw_stgcn_mode = os.getenv("STGCN_FEATURE_MODE", "xy")
STGCN_FEATURE_MODE = str(_raw_stgcn_mode).strip().lower()
_FEATURE_MODE_MAP = {
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
BEHAVIOR_MIN_CONFIDENCE = _BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD
TARGET_MODEL_FPS = 30.0
ENABLE_FPS_DOWNSAMPLE = True
CLASSIFY_STRIDE = 2
WINDOW_NAME = "Pose Jitter Analysis"
DISPLAY_SIZE = (1080, 720)
EMA_ALPHA = 1.0

# ===== 信心值門檻設定（bbox conf / keypoint conf，集中管理）=====
YOLO_CONF_THRESHOLD = 0.5      # YOLO bbox 偵測信心門檻
DRAW_KP_CONF_THRESHOLD = 0.25  # 畫骨架線段與關鍵點圓點用門檻（>此值才畫）
JITTER_CONF_THRESHOLD = 0.3    # 抖動統計只使用高於此信心值的關鍵點

# 17 關鍵點名稱
KEYPOINT_NAMES = [
    "Nose", "Left_Ear", "Right_Ear", "Chest", "Mid_Back", "Hip",
    "LF_Elbow", "LF_Paw", "RF_Elbow", "RF_Paw",
    "LH_Knee", "LH_Paw", "RH_Knee", "RH_Paw",
    "Tail_Root", "Tail_Mid", "Tail_Tip",
]

# 折線圖末端標籤用的縮寫（≤7 字元）
_KP_SHORT_NAMES = [
    "Nose",    "L.Ear",  "R.Ear",  "Chest", "M.Back", "Hip",
    "LF.Elb",  "LF.Paw", "RF.Elb", "RF.Paw",
    "LH.Kne",  "LH.Paw", "RH.Kne", "RH.Paw",
    "T.Root",  "T.Mid",  "T.Tip",
]

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

SUPPORTED_VIDEO_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"
}


# ── 工具函數（沿用自 base 腳本）────────────────────────────────────────
def _is_stream_url(path_str: str) -> bool:
    lowered = str(path_str).lower()
    return lowered.startswith(("http://", "https://", "rtsp://", "rtsps://", "rtmp://"))


def open_video_capture_with_retry(path, retries=5, delay=3):
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
            continue
        if p.is_dir():
            try:
                matched = sorted([
                    f for f in p.rglob("*")
                    if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTS
                ])
            except Exception as e:
                print(f"⚠ 掃描資料夾出錯，已略過: {p} ({e})")
                matched = []
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
    return float(np.clip(diag / max(base_diag, 1.0), 0.65, 2.4))


def scale_px(value, ui_scale, min_px=1):
    return max(int(min_px), int(round(float(value) * float(ui_scale))))


def resize_with_letterbox(image, target_size):
    target_w, target_h = target_size
    src_h, src_w = image.shape[:2]
    if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
        return cv2.resize(image, target_size), 1.0, 0, 0
    scale = max(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    crop_x = max(0, (new_w - target_w) // 2)
    crop_y = max(0, (new_h - target_h) // 2)
    return resized[crop_y:crop_y + target_h, crop_x:crop_x + target_w], scale, crop_x, crop_y


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


def draw_skeleton_overlay(frame, kpts, kpt_conf, bbox, conf_thresh=DRAW_KP_CONF_THRESHOLD):
    """簡單骨架繪製（不含行為資訊 HUD）。"""
    ui_scale = compute_ui_scale(*frame.shape[:2][::-1])
    edge_thickness = scale_px(2, ui_scale, min_px=1)
    kp_r = scale_px(4, ui_scale, min_px=2)

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BLACK, 4, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HEAD, 2, cv2.LINE_AA)

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
            cv2.circle(frame, (cx, cy), kp_r, (0, 0, 0), -1)
            cv2.circle(frame, (cx, cy), max(1, kp_r - 1), col, -1)
    return frame


# ── 統計工具函數 ────────────────────────────────────────────────────
def _bbox_center(bbox):
    if bbox is None:
        return np.nan, np.nan
    x1, y1, x2, y2 = bbox
    return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0


def _finite(arr):
    return [v for v in arr if np.isfinite(v)]


def _stats(values):
    """回傳 (n, mean, std, p95, max) 或全 0。"""
    v = _finite(values)
    if not v:
        return 0, 0.0, 0.0, 0.0, 0.0
    a = np.array(v)
    return len(a), float(a.mean()), float(a.std()), float(np.percentile(a, 95)), float(a.max())


def _corr_aligned(xs, ys):
    """Pearson r with proper paired alignment (both must be finite)."""
    pairs = [(x, y) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 5:
        return np.nan
    xa = np.array([p[0] for p in pairs])
    ya = np.array([p[1] for p in pairs])
    if xa.std() < 1e-9 or ya.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(xa, ya)[0, 1])


# ── 輸出資料夾管理 ──────────────────────────────────────────────────
def create_run_output_dir() -> Path:
    subdir = _MODE_SUBDIR.get(ANALYSIS_MODE, 'unknown')
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / subdir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _print_output_summary(run_dir: Path, generated: list):
    sep = "=" * 52
    print("\n" + sep)
    print("Output Folder")
    print(sep)
    print(str(run_dir))
    print(sep)
    print("Generated Files:")
    for fname in generated:
        print(f"  - {fname}")
    print(sep)


# ── 統計計算函數 ────────────────────────────────────────────────────
def get_bbox_statistics(frame_records: list) -> dict:
    bbox_disp = [r['bbox_disp'] for r in frame_records]
    bbox_dx   = [r['bbox_dx']   for r in frame_records]
    bbox_dy   = [r['bbox_dy']   for r in frame_records]
    n, mean, std, p95, mx = _stats(bbox_disp)
    return {
        'n': n, 'mean': mean, 'std': std, 'p95': p95, 'max': mx,
        'bbox_disp': bbox_disp, 'bbox_dx': bbox_dx, 'bbox_dy': bbox_dy,
    }


def get_keypoint_statistics(frame_records: list) -> list:
    """Returns list of dicts, one per keypoint: {name, mean, std, p95, max}."""
    result = []
    for i in range(17):
        disp_vals = [r['kp_disp'][i] for r in frame_records]
        n, mean, std, p95, mx = _stats(disp_vals)
        result.append({
            'idx': i, 'name': KEYPOINT_NAMES[i],
            'n': n, 'mean': mean, 'std': std, 'p95': p95, 'max': mx,
            'disp': disp_vals,
        })
    return result


def get_confidence_statistics(frame_records: list) -> list:
    """Returns list of dicts, one per keypoint: {name, mean, std, min, p5}."""
    result = []
    for i in range(17):
        conf_vals = [r['kp_conf'][i].item() for r in frame_records]
        v = _finite(conf_vals)
        if v:
            a = np.array(v)
            result.append({
                'idx': i, 'name': KEYPOINT_NAMES[i],
                'n': len(a), 'mean': a.mean().item(), 'std': a.std().item(),
                'min': a.min().item(), 'p5': np.percentile(a, 5).item(),
                'conf': conf_vals,
            })
        else:
            result.append({
                'idx': i, 'name': KEYPOINT_NAMES[i],
                'n': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'p5': 0.0,
                'conf': conf_vals,
            })
    return result


# ── 各 mode 輸出函數 ────────────────────────────────────────────────
def save_mode1_results(frame_records: list, run_dir: Path):
    """MODE 1: BBox Stability Analysis."""
    generated = []

    # CSV
    csv_path = run_dir / "bbox_timeseries.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'bbox_cx', 'bbox_cy', 'bbox_dx', 'bbox_dy', 'bbox_disp'])
        for r in frame_records:
            writer.writerow([
                r['frame'],
                f"{r['bbox_cx']:.4f}" if np.isfinite(r['bbox_cx']) else '',
                f"{r['bbox_cy']:.4f}" if np.isfinite(r['bbox_cy']) else '',
                f"{r['bbox_dx']:.4f}" if np.isfinite(r['bbox_dx']) else '',
                f"{r['bbox_dy']:.4f}" if np.isfinite(r['bbox_dy']) else '',
                f"{r['bbox_disp']:.4f}" if np.isfinite(r['bbox_disp']) else '',
            ])
    generated.append(csv_path.name)

    # Terminal stats
    stats = get_bbox_statistics(frame_records)
    print("\n" + "=" * 55)
    print("MODE 1: BBox Stability Analysis")
    print("=" * 55)
    print(f"{'Metric':<16} {'Samples':>8} {'Mean':>8} {'Std':>8} {'P95':>8} {'Max':>8}")
    print("-" * 55)
    print(f"{'BBox Disp':<16} {stats['n']:>8} {stats['mean']:>8.3f} {stats['std']:>8.3f} {stats['p95']:>8.3f} {stats['max']:>8.3f}")
    if stats['p95'] > 10.0:
        print(f"  [SEVERE WARNING] P95={stats['p95']:.2f}px > 10px — BBox 嚴重不穩定")
    elif stats['p95'] > 5.0:
        print(f"  [WARNING] P95={stats['p95']:.2f}px > 5px — BBox 輕度不穩定")
    print("=" * 55)

    # Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        frames = [r['frame'] for r in frame_records]
        cx_vals   = [r['bbox_cx']   for r in frame_records]
        cy_vals   = [r['bbox_cy']   for r in frame_records]
        disp_vals = [r['bbox_disp'] for r in frame_records]

        # Plot 1: bbox_center_timeseries.png
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(frames, cx_vals, color='royalblue', linewidth=0.8, label='BBox Center X')
        axes[0].set_ylabel('X (px)'); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(frames, cy_vals, color='tomato', linewidth=0.8, label='BBox Center Y')
        axes[1].set_ylabel('Y (px)'); axes[1].set_xlabel('Frame'); axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.suptitle('BBox Center Time Series')
        plt.tight_layout()
        p = run_dir / "bbox_center_timeseries.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        generated.append(p.name)

        # Plot 2: bbox_displacement.png
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(frames, disp_vals, color='gray', linewidth=0.8, label='BBox Displacement')
        if stats['p95'] > 0:
            ax.axhline(stats['p95'], color='orange', linestyle='--', linewidth=1,
                       label=f"P95={stats['p95']:.2f}px")
        ax.set_xlabel('Frame'); ax.set_ylabel('Displacement (px)')
        ax.set_title('BBox Frame-over-Frame Displacement')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        p = run_dir / "bbox_displacement.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        generated.append(p.name)

    except ImportError:
        print("[WARN] matplotlib 未安裝，跳過繪圖")

    _print_output_summary(run_dir, generated)


def save_mode2_results(frame_records: list, run_dir: Path):
    """MODE 2: Keypoint Stability Analysis."""
    generated = []
    kp_stats = get_keypoint_statistics(frame_records)

    # CSV 1: keypoint_timeseries.csv
    csv_path = run_dir / "keypoint_timeseries.csv"
    header = ['frame']
    for i in range(17):
        header += [f'kp{i}_x', f'kp{i}_y', f'kp{i}_disp']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in frame_records:
            row = [r['frame']]
            for i in range(17):
                for val in (r['kp_x'][i], r['kp_y'][i], r['kp_disp'][i]):
                    row.append(f"{val:.4f}" if np.isfinite(val) else '')
            writer.writerow(row)
    generated.append(csv_path.name)

    # CSV 2: keypoint_statistics.csv
    stat_path = run_dir / "keypoint_statistics.csv"
    with open(stat_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Keypoint', 'Mean', 'Std', 'P95', 'Max'])
        for s in kp_stats:
            writer.writerow([
                s['name'],
                f"{s['mean']:.4f}", f"{s['std']:.4f}",
                f"{s['p95']:.4f}", f"{s['max']:.4f}",
            ])
    generated.append(stat_path.name)

    # Terminal ranking by P95
    active_kp_stats = [s for s in kp_stats if s['idx'] not in EXCLUDED_KEYPOINTS]
    ranked = sorted(active_kp_stats, key=lambda s: s['p95'], reverse=True)
    print("\n" + "=" * 65)
    print("MODE 2: Keypoint Stability Analysis")
    print("=" * 65)
    if EXCLUDED_KEYPOINTS:
        excl_names = [KEYPOINT_NAMES[i] for i in sorted(EXCLUDED_KEYPOINTS) if i < 17]
        print(f"  ※ 已排除: {excl_names}")
    print(f"\n{'Rank':<5} {'Keypoint':<16} {'N':>6} {'Mean':>8} {'Std':>8} {'P95':>8} {'Max':>8}")
    print("-" * 65)
    for rank, s in enumerate(ranked, 1):
        print(f"{rank:<5} {s['name']:<16} {s['n']:>6} {s['mean']:>8.3f} "
              f"{s['std']:>8.3f} {s['p95']:>8.3f} {s['max']:>8.3f}")
    stable   = min(active_kp_stats, key=lambda s: s['p95'])
    unstable = max(active_kp_stats, key=lambda s: s['p95'])
    print(f"\nMost Stable Keypoint  : {stable['name']} (P95={stable['p95']:.3f}px)")
    print(f"Most Unstable Keypoint: {unstable['name']} (P95={unstable['p95']:.3f}px)")
    print("=" * 65)

    # Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        frames = [r['frame'] for r in frame_records]
        cmap = plt.cm.get_cmap('tab20', 17)

        # Plot 1: keypoint_displacement_comparison.png (17 lines, index end-labels)
        fig, ax = plt.subplots(figsize=(14, 6))
        for i in range(17):
            if i in EXCLUDED_KEYPOINTS:
                continue
            disp_vals = [r['kp_disp'][i] for r in frame_records]
            ax.plot(frames, disp_vals, color=cmap(i), linewidth=0.7, alpha=0.8)
        # 末端索引標籤：直接貼齊最後一個有限值，不做 y 位移
        for i in _ACTIVE_KP_IDS:
            pairs = [(r['frame'], r['kp_disp'][i]) for r in frame_records]
            last_pair = next(((f, v) for f, v in reversed(pairs) if np.isfinite(v)), None)
            if last_pair is not None:
                ax.annotate(str(i), xy=last_pair,
                            xytext=(3, 0), textcoords='offset points',
                            fontsize=7, fontweight='bold', color=cmap(i),
                            va='center', clip_on=False)
        ax.set_xlabel('Frame'); ax.set_ylabel('Displacement (px)')
        ax.set_title('All Keypoints Frame-over-Frame Displacement')
        ax.grid(alpha=0.3)
        # 圖下方索引對照表（在 axes 外，不占圖表空間）
        _items = [f'{i}={_KP_SHORT_NAMES[i]}' for i in _ACTIVE_KP_IDS]
        _rows  = [_items[j:j+6] for j in range(0, len(_items), 6)]
        fig.text(0.5, 0.01, '\n'.join('   '.join(r) for r in _rows),
                 ha='center', va='bottom', fontsize=7.5, family='monospace',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='#f5f5f5',
                           edgecolor='#bbbbbb', alpha=0.9))
        fig.subplots_adjust(bottom=0.20, right=0.93)
        p = run_dir / "keypoint_displacement_comparison.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        generated.append(p.name)

        # Plot 2: top5_jitter_keypoints.png
        top5 = ranked[:5]
        colors5 = ['tomato', 'darkorange', 'gold', 'mediumseagreen', 'steelblue']
        fig, ax = plt.subplots(figsize=(12, 5))
        for s, col in zip(top5, colors5):
            disp_vals = [r['kp_disp'][s['idx']] for r in frame_records]
            ax.plot(frames, disp_vals, color=col, linewidth=0.9, alpha=0.85,
                    label=f"{s['name']} (P95={s['p95']:.2f})")
        ax.set_xlabel('Frame'); ax.set_ylabel('Displacement (px)')
        ax.set_title('Top 5 Most Unstable Keypoints')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout()
        p = run_dir / "top5_jitter_keypoints.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        generated.append(p.name)

    except ImportError:
        print("[WARN] matplotlib 未安裝，跳過繪圖")

    _print_output_summary(run_dir, generated)


def save_mode3_results(frame_records: list, run_dir: Path):
    """MODE 3: Keypoint Confidence Analysis."""
    generated = []
    conf_stats = get_confidence_statistics(frame_records)

    # BBox confidence stats (scalar per frame, stored alongside kp_conf)
    bbox_conf_vals = [r['bbox_conf'] for r in frame_records]
    _bc_v = _finite(bbox_conf_vals)
    if _bc_v:
        _bc_a = np.array(_bc_v)
        bbox_conf_stat = {
            'mean': _bc_a.mean().item(), 'std': _bc_a.std().item(),
            'min': _bc_a.min().item(), 'p5': np.percentile(_bc_a, 5).item(),
        }
    else:
        bbox_conf_stat = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'p5': 0.0}

    # CSV 1: confidence_timeseries.csv  (含 bbox_conf 欄)
    csv_path = run_dir / "confidence_timeseries.csv"
    header = ['frame'] + [f'kp{i}_conf' for i in range(17)] + ['bbox_conf']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in frame_records:
            row = (
                [r['frame']]
                + [f"{r['kp_conf'][i].item():.4f}" for i in range(17)]
                + [f"{r['bbox_conf']:.4f}" if np.isfinite(r['bbox_conf']) else '']
            )
            writer.writerow(row)
    generated.append(csv_path.name)

    # CSV 2: confidence_statistics.csv  (含 BBox_Conf 列)
    stat_path = run_dir / "confidence_statistics.csv"
    with open(stat_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Keypoint', 'Mean', 'Std', 'Min', 'P5'])
        for s in conf_stats:
            writer.writerow([
                s['name'],
                f"{s['mean']:.4f}", f"{s['std']:.4f}",
                f"{s['min']:.4f}", f"{s['p5']:.4f}",
            ])
        writer.writerow([
            'BBox_Conf',
            f"{bbox_conf_stat['mean']:.4f}", f"{bbox_conf_stat['std']:.4f}",
            f"{bbox_conf_stat['min']:.4f}", f"{bbox_conf_stat['p5']:.4f}",
        ])
    generated.append(stat_path.name)

    # CSV 3: confidence_displacement_correlation.csv
    corr_path = run_dir / "confidence_displacement_correlation.csv"
    corr_results = []
    for i in range(17):
        conf_vals = [r['kp_conf'][i].item() for r in frame_records]
        disp_vals = [r['kp_disp'][i] for r in frame_records]
        r_val = _corr_aligned(conf_vals, disp_vals)
        corr_results.append({'idx': i, 'name': KEYPOINT_NAMES[i], 'r': r_val})
    with open(corr_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Keypoint', 'Correlation'])
        for cr in corr_results:
            r_str = f"{cr['r']:.4f}" if np.isfinite(cr['r']) else ''
            writer.writerow([cr['name'], r_str])
    generated.append(corr_path.name)

    # Terminal output
    print("\n" + "=" * 60)
    print("MODE 3: Keypoint Confidence Analysis")
    print("=" * 60)
    print(f"\n{'Keypoint':<16} {'Conf_Mean':>10} {'Conf_P5':>10} "
          f"{'Corr(r)':>10}  Diagnosis")
    print("-" * 60)
    for i in range(17):
        if i in EXCLUDED_KEYPOINTS:
            continue
        cs = conf_stats[i]
        cr = corr_results[i]
        r_str = f"{cr['r']:+.3f}" if np.isfinite(cr['r']) else "   N/A"
        diag = "Likely Conf-Driven Jitter" if (np.isfinite(cr['r']) and cr['r'] < -0.5) else ""
        print(f"{cs['name']:<16} {cs['mean']:>10.3f} {cs['p5']:>10.3f} "
              f"{r_str:>10}  {diag}")
    print(f"\n{'BBox_Conf':<16} {bbox_conf_stat['mean']:>10.3f} {bbox_conf_stat['p5']:>10.3f} "
          f"{'':>10}  (BBox 偵測信心，非關鍵點)")
    print("=" * 60)

    # Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        frames = [r['frame'] for r in frame_records]
        cmap = plt.cm.get_cmap('tab20', 17)

        # Plot 1: confidence_comparison.png (17 lines, index end-labels)
        fig, ax = plt.subplots(figsize=(14, 6))
        for i in range(17):
            if i in EXCLUDED_KEYPOINTS:
                continue
            conf_vals = [r['kp_conf'][i].item() for r in frame_records]
            ax.plot(frames, conf_vals, color=cmap(i), linewidth=0.7, alpha=0.8)
        # 末端索引標籤：直接貼齊最後一個有限值，不做 y 位移
        for i in _ACTIVE_KP_IDS:
            pairs = [(r['frame'], r['kp_conf'][i].item()) for r in frame_records]
            last_pair = next(((f, v) for f, v in reversed(pairs) if np.isfinite(v)), None)
            if last_pair is not None:
                ax.annotate(str(i), xy=last_pair,
                            xytext=(3, 0), textcoords='offset points',
                            fontsize=7, fontweight='bold', color=cmap(i),
                            va='center', clip_on=False)
        ax.set_xlabel('Frame'); ax.set_ylabel('Confidence')
        ax.set_title('All Keypoints Confidence Over Time')
        ax.grid(alpha=0.3)
        # 圖下方索引對照表（在 axes 外，不占圖表空間）
        _items = [f'{i}={_KP_SHORT_NAMES[i]}' for i in _ACTIVE_KP_IDS]
        _rows  = [_items[j:j+6] for j in range(0, len(_items), 6)]
        fig.text(0.5, 0.01, '\n'.join('   '.join(r) for r in _rows),
                 ha='center', va='bottom', fontsize=7.5, family='monospace',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='#f5f5f5',
                           edgecolor='#bbbbbb', alpha=0.9))
        fig.subplots_adjust(bottom=0.20, right=0.93)
        p = run_dir / "confidence_comparison.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        generated.append(p.name)

        # Plot 2: top5_low_confidence.png
        ranked_conf = sorted([s for s in conf_stats if s['idx'] not in EXCLUDED_KEYPOINTS], key=lambda s: s['mean'])
        top5_low = ranked_conf[:5]
        colors5 = ['tomato', 'darkorange', 'gold', 'mediumseagreen', 'steelblue']
        fig, ax = plt.subplots(figsize=(12, 5))
        for s, col in zip(top5_low, colors5):
            conf_vals = [r['kp_conf'][s['idx']].item() for r in frame_records]
            ax.plot(frames, conf_vals, color=col, linewidth=0.9, alpha=0.85,
                    label=f"{s['name']} (mean={s['mean']:.3f})")
        ax.set_xlabel('Frame'); ax.set_ylabel('Confidence')
        ax.set_title('Top 5 Lowest Confidence Keypoints')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout()
        p = run_dir / "top5_low_confidence.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        generated.append(p.name)

    except ImportError:
        print("[WARN] matplotlib 未安裝，跳過繪圖")

    _print_output_summary(run_dir, generated)


# ── MODE 4 helpers ──────────────────────────────────────────────────
def _apply_bbox_ema(frame_records: list, alpha) -> list:
    """Apply EMA to raw bbox centers. Returns [(smooth_cx, smooth_cy)] per frame."""
    smooth_cx = smooth_cy = np.nan
    result = []
    for r in frame_records:
        raw_cx = r['bbox_cx']
        raw_cy = r['bbox_cy']
        if np.isfinite(raw_cx):
            if np.isfinite(smooth_cx):
                smooth_cx = alpha * raw_cx + (1.0 - alpha) * smooth_cx
                smooth_cy = alpha * raw_cy + (1.0 - alpha) * smooth_cy
            else:
                smooth_cx = raw_cx
                smooth_cy = raw_cy
        else:
            smooth_cx = smooth_cy = np.nan
        result.append((smooth_cx, smooth_cy))
    return result


def _compute_alpha_stats(frame_records: list, alpha) -> dict:
    """Compute raw and smoothed bbox + relative keypoint stats for one alpha value."""
    smooth_centers = _apply_bbox_ema(frame_records, alpha)

    # Smooth bbox displacement
    smooth_bbox_disp = []
    prev_sx = prev_sy = np.nan
    for sx, sy in smooth_centers:
        if np.isfinite(sx) and np.isfinite(prev_sx):
            d = np.sqrt((sx - prev_sx) ** 2 + (sy - prev_sy) ** 2)
            smooth_bbox_disp.append(d)
        else:
            smooth_bbox_disp.append(np.nan)
        if np.isfinite(sx):
            prev_sx, prev_sy = sx, sy
        else:
            prev_sx = prev_sy = np.nan

    raw_bbox_disp = [r['bbox_disp'] for r in frame_records]
    _, _, _, bbox_p95_raw,    _ = _stats(raw_bbox_disp)
    _, _, _, bbox_p95_smooth, _ = _stats(smooth_bbox_disp)
    bbox_improvement = (
        (bbox_p95_raw - bbox_p95_smooth) / bbox_p95_raw * 100.0
        if bbox_p95_raw > 1e-9 else 0.0
    )

    # Relative keypoint displacement vs raw and smoothed bbox center
    prev_raw_rel    = np.full((17, 2), np.nan)
    prev_smooth_rel = np.full((17, 2), np.nan)
    raw_rel_disp    = [[] for _ in range(17)]
    smooth_rel_disp = [[] for _ in range(17)]

    for r, (sx, sy) in zip(frame_records, smooth_centers):
        raw_cx = r['bbox_cx']
        raw_cy = r['bbox_cy']
        for i in range(17):
            kx = r['kp_x'][i]
            ky = r['kp_y'][i]
            if np.isfinite(kx) and np.isfinite(raw_cx):
                rrx = kx - raw_cx
                rry = ky - raw_cy
                if np.isfinite(prev_raw_rel[i, 0]):
                    d = np.sqrt((rrx - prev_raw_rel[i, 0]) ** 2 + (rry - prev_raw_rel[i, 1]) ** 2)
                    raw_rel_disp[i].append(d)
                else:
                    raw_rel_disp[i].append(np.nan)
                prev_raw_rel[i, 0] = rrx
                prev_raw_rel[i, 1] = rry
            else:
                raw_rel_disp[i].append(np.nan)
                prev_raw_rel[i] = np.nan

            if np.isfinite(kx) and np.isfinite(sx):
                srx = kx - sx
                sry = ky - sy
                if np.isfinite(prev_smooth_rel[i, 0]):
                    d = np.sqrt((srx - prev_smooth_rel[i, 0]) ** 2 + (sry - prev_smooth_rel[i, 1]) ** 2)
                    smooth_rel_disp[i].append(d)
                else:
                    smooth_rel_disp[i].append(np.nan)
                prev_smooth_rel[i, 0] = srx
                prev_smooth_rel[i, 1] = sry
            else:
                smooth_rel_disp[i].append(np.nan)
                prev_smooth_rel[i] = np.nan

    kp_p95_raw    = []
    kp_p95_smooth = []
    for i in range(17):
        _, _, _, p95r, _ = _stats(raw_rel_disp[i])
        _, _, _, p95s, _ = _stats(smooth_rel_disp[i])
        kp_p95_raw.append(p95r)
        kp_p95_smooth.append(p95s)

    valid_raw    = [v for v in kp_p95_raw    if v > 0]
    valid_smooth = [v for v in kp_p95_smooth if v > 0]
    mean_kp_p95_raw    = np.mean(valid_raw)    if valid_raw    else 0.0
    mean_kp_p95_smooth = np.mean(valid_smooth) if valid_smooth else 0.0
    kp_improvement = (
        (mean_kp_p95_raw - mean_kp_p95_smooth) / mean_kp_p95_raw * 100.0
        if mean_kp_p95_raw > 1e-9 else 0.0
    )

    return {
        'alpha': alpha,
        'bbox_p95_raw': bbox_p95_raw,
        'bbox_p95_smooth': bbox_p95_smooth,
        'bbox_improvement': bbox_improvement,
        'kp_p95_raw': kp_p95_raw,
        'kp_p95_smooth': kp_p95_smooth,
        'mean_kp_p95_raw': mean_kp_p95_raw,
        'mean_kp_p95_smooth': mean_kp_p95_smooth,
        'kp_improvement': kp_improvement,
        'smooth_bbox_disp': smooth_bbox_disp,
        'smooth_cx': [sc[0] for sc in smooth_centers],
        'smooth_cy': [sc[1] for sc in smooth_centers],
        'raw_rel_disp': raw_rel_disp,
        'smooth_rel_disp': smooth_rel_disp,
    }


def save_mode4_results(frame_records: list, run_dir: Path):
    """MODE 4: BBox Smoothing Analysis — quantifies jitter reduction across alpha values."""
    generated = []

    all_alpha_stats = [_compute_alpha_stats(frame_records, a) for a in BBOX_EMA_LIST]
    frames        = [r['frame']    for r in frame_records]
    raw_bbox_disp = [r['bbox_disp'] for r in frame_records]
    raw_cx_vals   = [r['bbox_cx']   for r in frame_records]
    raw_cy_vals   = [r['bbox_cy']   for r in frame_records]

    # Best alpha = lowest keypoint improvement when alpha < 1.0
    best = max(
        (s for s in all_alpha_stats if s['alpha'] < 1.0 - 1e-9),
        key=lambda s: s['kp_improvement'],
        default=all_alpha_stats[-1],
    )

    # CSV: bbox_smoothing_summary.csv
    nose_idx    = 0
    tailtip_idx = 16
    csv_path = run_dir / "bbox_smoothing_summary.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Alpha',
            'BBox_P95_Raw', 'BBox_P95_Smooth', 'BBox_Improvement_%',
            'Mean_Keypoint_P95_Raw', 'Mean_Keypoint_P95_Smooth', 'Keypoint_Improvement_%',
            'TailTip_P95_Raw', 'TailTip_P95_Smooth',
            'Nose_P95_Raw', 'Nose_P95_Smooth',
        ])
        for s in all_alpha_stats:
            writer.writerow([
                f"{s['alpha']:.1f}",
                f"{s['bbox_p95_raw']:.4f}",   f"{s['bbox_p95_smooth']:.4f}",
                f"{s['bbox_improvement']:.2f}",
                f"{s['mean_kp_p95_raw']:.4f}", f"{s['mean_kp_p95_smooth']:.4f}",
                f"{s['kp_improvement']:.2f}",
                f"{s['kp_p95_raw'][tailtip_idx]:.4f}", f"{s['kp_p95_smooth'][tailtip_idx]:.4f}",
                f"{s['kp_p95_raw'][nose_idx]:.4f}",    f"{s['kp_p95_smooth'][nose_idx]:.4f}",
            ])
    generated.append(csv_path.name)

    # Terminal output
    sep = "=" * 52
    print("\n" + sep)
    print("MODE 4: BBox Smoothing Analysis")
    print(sep)
    for s in all_alpha_stats:
        print(f"\nAlpha={s['alpha']:.1f}")
        if abs(s['alpha'] - 1.0) < 1e-9:
            print(f"  BBox P95:              {s['bbox_p95_raw']:>8.2f} px")
            print(f"  Relative Keypoint P95: {s['mean_kp_p95_raw']:>8.2f} px")
        else:
            print(f"  BBox P95:              {s['bbox_p95_smooth']:>8.2f} px")
            print(f"  Improvement:           {s['bbox_improvement']:>7.1f} %")
            print(f"  Relative Keypoint P95: {s['mean_kp_p95_smooth']:>8.2f} px")
            print(f"  Improvement:           {s['kp_improvement']:>7.1f} %")
        print("-" * 44)

    print(f"\nBest Alpha:           {best['alpha']:.1f}")
    print(f"BBox Improvement:     {best['bbox_improvement']:.1f} %")
    print(f"Keypoint Improvement: {best['kp_improvement']:.1f} %")
    print(sep)

    # diagnosis.txt
    txt_path = run_dir / "diagnosis.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Best Alpha:           {best['alpha']:.1f}\n")
        f.write(f"BBox Improvement:     {best['bbox_improvement']:.1f} %\n")
        f.write(f"Keypoint Improvement: {best['kp_improvement']:.1f} %\n")
        f.write(f"Recommended EMA Alpha:{best['alpha']:.1f}\n")
    generated.append(txt_path.name)

    # Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        ref = next(
            (s for s in all_alpha_stats if abs(s['alpha'] - BBOX_EMA_ALPHA) < 1e-9),
            best,
        )
        alphas = [s['alpha'] for s in all_alpha_stats]

        # Plot 1: bbox_smoothing_comparison.png
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(frames, raw_bbox_disp, color='royalblue', linewidth=0.8, alpha=0.85,
                label='Raw BBox Displacement (alpha=1.0)')
        ax.plot(frames, ref['smooth_bbox_disp'], color='tomato', linewidth=0.8, alpha=0.85,
                label=f'Smooth BBox Displacement (alpha={ref["alpha"]:.1f})')
        ax.set_xlabel('Frame'); ax.set_ylabel('Displacement (px)')
        ax.set_title('BBox Displacement: Raw vs Smoothed')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        p = run_dir / "bbox_smoothing_comparison.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        generated.append(p.name)

        # Plot 2: bbox_center_comparison.png
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(frames, raw_cx_vals, color='royalblue', linewidth=0.7, alpha=0.6,
                     label='Raw Center X')
        axes[0].plot(frames, ref['smooth_cx'], color='tomato', linewidth=0.9, alpha=0.9,
                     label=f'Smooth Center X (α={ref["alpha"]:.1f})')
        axes[0].set_ylabel('X (px)'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
        axes[1].plot(frames, raw_cy_vals, color='steelblue', linewidth=0.7, alpha=0.6,
                     label='Raw Center Y')
        axes[1].plot(frames, ref['smooth_cy'], color='darkorange', linewidth=0.9, alpha=0.9,
                     label=f'Smooth Center Y (α={ref["alpha"]:.1f})')
        axes[1].set_ylabel('Y (px)'); axes[1].set_xlabel('Frame')
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        fig.suptitle('BBox Center: Raw vs Smoothed')
        plt.tight_layout()
        p = run_dir / "bbox_center_comparison.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        generated.append(p.name)

        # Plot 3: keypoint_jitter_improvement.png
        mean_raw_p95    = [s['mean_kp_p95_raw']    for s in all_alpha_stats]
        mean_smooth_p95 = [s['mean_kp_p95_smooth']  for s in all_alpha_stats]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(alphas, mean_raw_p95,    'o--', color='royalblue', linewidth=1.2,
                label='Raw Relative Kp P95')
        ax.plot(alphas, mean_smooth_p95, 'o-',  color='tomato',    linewidth=1.5,
                label='Relative-to-Smoothed-BBox P95')
        ax.set_xlabel('Alpha'); ax.set_ylabel('Mean P95 Jitter (px)')
        ax.set_title('Keypoint Jitter vs Alpha')
        ax.legend(); ax.grid(alpha=0.3); ax.invert_xaxis()
        plt.tight_layout()
        p = run_dir / "keypoint_jitter_improvement.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        generated.append(p.name)

        # Plot 4: top5_keypoint_improvement.png — 5 keypoints with most improvement at best alpha
        kp_delta = [
            (i, KEYPOINT_NAMES[i], best['kp_p95_raw'][i] - best['kp_p95_smooth'][i])
            for i in range(17) if i not in EXCLUDED_KEYPOINTS
        ]
        top5_kp = sorted(kp_delta, key=lambda x: x[2], reverse=True)[:5]
        colors5 = ['tomato', 'darkorange', 'gold', 'mediumseagreen', 'steelblue']
        fig, axes2 = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
        for ax, (kp_idx, kp_name, delta), col in zip(axes2, top5_kp, colors5):
            raw_d  = best['raw_rel_disp'][kp_idx]
            smth_d = best['smooth_rel_disp'][kp_idx]
            ax.plot(frames, raw_d,  color='lightsteelblue', linewidth=0.7, label='Raw relative')
            ax.plot(frames, smth_d, color=col, linewidth=0.9, alpha=0.85,
                    label=f'Smooth relative α={best["alpha"]:.1f} (Δ={delta:.2f}px)')
            ax.set_ylabel(kp_name, fontsize=8); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        axes2[-1].set_xlabel('Frame')
        fig.suptitle(f'Top 5 Improved Keypoints — Relative Disp (Best Alpha={best["alpha"]:.1f})')
        plt.tight_layout()
        p = run_dir / "top5_keypoint_improvement.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        generated.append(p.name)

        # Plot 5: alpha_sweep_comparison.png
        bbox_impr = [s['bbox_improvement']  for s in all_alpha_stats]
        kp_impr   = [s['kp_improvement']    for s in all_alpha_stats]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(alphas, bbox_impr, 'o-', color='royalblue', linewidth=1.5,
                label='BBox Improvement %')
        ax.plot(alphas, kp_impr,   'o-', color='tomato',    linewidth=1.5,
                label='Keypoint Improvement %')
        ax.axhline(0,  color='gray',   linewidth=0.7, linestyle='--')
        ax.axhline(20, color='orange', linewidth=0.7, linestyle=':', label='20 % threshold')
        ax.set_xlabel('Alpha'); ax.set_ylabel('Improvement (%)')
        ax.set_title('Alpha Sweep: BBox and Keypoint Improvement')
        ax.legend(); ax.grid(alpha=0.3); ax.invert_xaxis()
        plt.tight_layout()
        p = run_dir / "alpha_sweep_comparison.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        generated.append(p.name)

    except ImportError:
        print("[WARN] matplotlib 未安裝，跳過繪圖")

    _print_output_summary(run_dir, generated)


# ── MODE 5 helpers ──────────────────────────────────────────────────
def _apply_keypoint_ema(frame_records: list, alpha) -> list:
    """Apply EMA to each of 17 keypoints' x,y independently.
    Returns list of np.ndarray(17,2) per frame; NaN where keypoint was invalid."""
    smooth_xy = np.full((17, 2), np.nan)
    result = []
    for r in frame_records:
        kp_x = r['kp_x']
        kp_y = r['kp_y']
        new_smooth = np.full((17, 2), np.nan)
        for i in range(17):
            if np.isfinite(kp_x[i]):
                if np.isfinite(smooth_xy[i, 0]):
                    new_smooth[i, 0] = alpha * kp_x[i] + (1.0 - alpha) * smooth_xy[i, 0]
                    new_smooth[i, 1] = alpha * kp_y[i] + (1.0 - alpha) * smooth_xy[i, 1]
                else:
                    new_smooth[i, 0] = kp_x[i]
                    new_smooth[i, 1] = kp_y[i]
        smooth_xy = new_smooth.copy()
        result.append(new_smooth.copy())
    return result


def _compute_kp_alpha_stats(frame_records: list, alpha) -> dict:
    """Compute raw vs EMA-smoothed keypoint displacement stats for one alpha."""
    smooth_kp_list = _apply_keypoint_ema(frame_records, alpha)

    raw_kp_p95 = []
    for i in range(17):
        disp_vals = [r['kp_disp'][i] for r in frame_records]
        _, _, _, p95, _ = _stats(disp_vals)
        raw_kp_p95.append(p95)

    smooth_kp_disp = [[] for _ in range(17)]
    prev_smooth = np.full((17, 2), np.nan)
    for smooth_xy in smooth_kp_list:
        for i in range(17):
            sx = smooth_xy[i, 0]
            sy = smooth_xy[i, 1]
            if np.isfinite(sx) and np.isfinite(prev_smooth[i, 0]):
                d = np.sqrt((sx - prev_smooth[i, 0]) ** 2 + (sy - prev_smooth[i, 1]) ** 2)
                smooth_kp_disp[i].append(d)
            else:
                smooth_kp_disp[i].append(np.nan)
            if np.isfinite(sx):
                prev_smooth[i, 0] = sx
                prev_smooth[i, 1] = sy
            else:
                prev_smooth[i] = np.nan

    smooth_kp_p95 = []
    for i in range(17):
        _, _, _, p95, _ = _stats(smooth_kp_disp[i])
        smooth_kp_p95.append(p95)

    improvements = [
        (raw_p - smo_p) / raw_p * 100.0 if raw_p > 1e-9 else 0.0
        for raw_p, smo_p in zip(raw_kp_p95, smooth_kp_p95)
    ]
    valid_raw    = [v for v in raw_kp_p95    if v > 0]
    valid_smooth = [v for v in smooth_kp_p95 if v > 0]
    mean_raw    = np.mean(valid_raw)    if valid_raw    else 0.0
    mean_smooth = np.mean(valid_smooth) if valid_smooth else 0.0
    mean_improvement = (mean_raw - mean_smooth) / mean_raw * 100.0 if mean_raw > 1e-9 else 0.0

    return {
        'alpha': alpha,
        'raw_kp_p95': raw_kp_p95,
        'smooth_kp_p95': smooth_kp_p95,
        'improvements': improvements,
        'mean_raw': mean_raw,
        'mean_smooth': mean_smooth,
        'mean_improvement': mean_improvement,
        'smooth_kp_disp': smooth_kp_disp,
    }


def _compute_bbox_ema_kp_p95(frame_records: list, alpha):
    """Mean absolute KP P95 after BBox-EMA position correction.

    Reconstructed position = raw_kp + (smooth_bbox_center - raw_bbox_center).
    This isolates how much KP jitter is driven by bbox drift.
    """
    smooth_centers = _apply_bbox_ema(frame_records, alpha)
    adj_disp = [[] for _ in range(17)]
    prev_adj = np.full((17, 2), np.nan)

    for r, (sx, sy) in zip(frame_records, smooth_centers):
        raw_cx = r['bbox_cx']
        raw_cy = r['bbox_cy']
        for i in range(17):
            kx = r['kp_x'][i]
            ky = r['kp_y'][i]
            if np.isfinite(kx) and np.isfinite(sx) and np.isfinite(raw_cx):
                ax_ = kx + (sx - raw_cx)
                ay_ = ky + (sy - raw_cy)
                if np.isfinite(prev_adj[i, 0]):
                    d = np.sqrt((ax_ - prev_adj[i, 0]) ** 2 + (ay_ - prev_adj[i, 1]) ** 2)
                    adj_disp[i].append(d)
                else:
                    adj_disp[i].append(np.nan)
                prev_adj[i, 0] = ax_
                prev_adj[i, 1] = ay_
            else:
                adj_disp[i].append(np.nan)
                prev_adj[i] = np.nan

    p95_list = []
    for i in range(17):
        _, _, _, p95, _ = _stats(adj_disp[i])
        p95_list.append(p95)
    valid = [v for v in p95_list if v > 0]
    return np.mean(valid).item() if valid else 0.0


def save_mode5_results(frame_records: list, run_dir: Path):
    """MODE 5: Keypoint Jitter — compare BBox EMA vs Keypoint EMA on the same KP metric."""
    generated = []

    # Baseline: raw absolute KP P95 (no smoothing)
    raw_kp_p95 = _compute_kp_alpha_stats(frame_records, 1.0)['mean_raw']

    # BBox EMA → absolute KP improvement for each alpha
    bbox_rows = []
    for a in BBOX_EMA_LIST:
        if abs(a - 1.0) < 1e-9:
            continue
        smoothed = _compute_bbox_ema_kp_p95(frame_records, a)
        imp = (raw_kp_p95 - smoothed) / raw_kp_p95 * 100.0 if raw_kp_p95 > 1e-9 else 0.0
        bbox_rows.append({'alpha': a, 'smoothed': smoothed, 'improvement': imp})

    # Keypoint EMA → absolute KP improvement for each alpha
    kp_rows = []
    for a in KEYPOINT_EMA_LIST:
        if abs(a - 1.0) < 1e-9:
            continue
        s = _compute_kp_alpha_stats(frame_records, a)
        imp = (raw_kp_p95 - s['mean_smooth']) / raw_kp_p95 * 100.0 if raw_kp_p95 > 1e-9 else 0.0
        kp_rows.append({'alpha': a, 'smoothed': s['mean_smooth'], 'improvement': imp})

    best_bbox_row = next(
        (r for r in bbox_rows if abs(r['alpha'] - BBOX_EMA_ALPHA) < 1e-9),
        max(bbox_rows, key=lambda r: r['improvement']) if bbox_rows
        else {'alpha': BBOX_EMA_ALPHA, 'smoothed': raw_kp_p95, 'improvement': 0.0},
    )
    best_kp_row = next(
        (r for r in kp_rows if abs(r['alpha'] - KEYPOINT_EMA_ALPHA) < 1e-9),
        max(kp_rows, key=lambda r: r['improvement']) if kp_rows
        else {'alpha': KEYPOINT_EMA_ALPHA, 'smoothed': raw_kp_p95, 'improvement': 0.0},
    )

    bbox_imp = best_bbox_row['improvement']
    kp_imp   = best_kp_row['improvement']

    # Diagnosis
    if kp_imp > bbox_imp:
        diag = "Most pose jitter originates from keypoint localization error rather than bounding box drift."
    elif bbox_imp > kp_imp:
        diag = "Most pose jitter originates from bounding box drift rather than keypoint localization error."
    else:
        diag = "Pose jitter has mixed sources: bounding box drift and keypoint localization error contribute equally."

    # CSV: smoothing_effect_summary.csv
    csv_path = run_dir / "smoothing_effect_summary.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Alpha', 'Raw_KP_P95', 'Smoothed_KP_P95', 'KP_Improvement_%'])
        writer.writerow([
            'BBox EMA',
            f"{best_bbox_row['alpha']:.1f}",
            f"{raw_kp_p95:.3f}",
            f"{best_bbox_row['smoothed']:.3f}",
            f"{bbox_imp:.1f}",
        ])
        writer.writerow([
            'Keypoint EMA',
            f"{best_kp_row['alpha']:.1f}",
            f"{raw_kp_p95:.3f}",
            f"{best_kp_row['smoothed']:.3f}",
            f"{kp_imp:.1f}",
        ])
    generated.append(csv_path.name)

    # diagnosis.txt
    txt_path = run_dir / "diagnosis.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Raw KP P95 (baseline):   {raw_kp_p95:.3f} px\n\n")
        f.write(f"BBox EMA (alpha={best_bbox_row['alpha']:.1f}):\n")
        f.write(f"  Smoothed KP P95:       {best_bbox_row['smoothed']:.3f} px\n")
        f.write(f"  KP Improvement:        {bbox_imp:.1f} %\n\n")
        f.write(f"Keypoint EMA (alpha={best_kp_row['alpha']:.1f}):\n")
        f.write(f"  Smoothed KP P95:       {best_kp_row['smoothed']:.3f} px\n")
        f.write(f"  KP Improvement:        {kp_imp:.1f} %\n\n")
        f.write(f"Diagnosis:\n{diag}\n")
    generated.append(txt_path.name)

    # Terminal output
    sep = "=" * 56
    print(f"\n{sep}")
    print("MODE 5: Keypoint Jitter — Smoothing Method Comparison")
    print(sep)
    print(f"  Raw KP P95 (baseline):     {raw_kp_p95:.3f} px")
    print(f"\n  BBox EMA   (best alpha={best_bbox_row['alpha']:.1f})")
    print(f"    Smoothed KP P95:         {best_bbox_row['smoothed']:.3f} px")
    print(f"    KP Improvement:          {bbox_imp:.1f} %")
    print(f"\n  Keypoint EMA (best alpha={best_kp_row['alpha']:.1f})")
    print(f"    Smoothed KP P95:         {best_kp_row['smoothed']:.3f} px")
    print(f"    KP Improvement:          {kp_imp:.1f} %")
    print(f"\n  → {diag}")
    print(sep)

    # Plot: smoothing_method_comparison.png
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        methods    = ['BBox EMA', 'Keypoint EMA']
        impr_vals  = [bbox_imp, kp_imp]
        bar_colors = ['royalblue', 'tomato']

        fig, ax = plt.subplots(figsize=(7, 6))
        bars = ax.bar(methods, impr_vals, color=bar_colors, alpha=0.85, width=0.45)

        y_max = max(impr_vals) if max(impr_vals) > 0 else 1.0
        for bar, val in zip(bars, impr_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold',
            )

        ax.set_ylabel('Keypoint Jitter Improvement (%)', fontsize=11)
        ax.set_title(
            f'Effect of Smoothing on Keypoint Jitter\n'
            f'(BBox EMA α={best_bbox_row["alpha"]:.1f}  vs  KP EMA α={best_kp_row["alpha"]:.1f})',
            fontsize=11,
        )
        ax.set_ylim(0, y_max * 1.25)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        p = run_dir / "smoothing_method_comparison.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        generated.append(p.name)

    except ImportError:
        print("[WARN] matplotlib 未安裝，跳過繪圖")

    _print_output_summary(run_dir, generated)


# ── MODE 7 helpers ──────────────────────────────────────────────────
def _pad_bbox(bbox, pad_ratio, frame_h, frame_w):
    """Expand bbox symmetrically by pad_ratio of each side length, clipped to frame."""
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    w = x2 - x1
    h = y2 - y1
    x1p = max(0.0, x1 - w * pad_ratio)
    y1p = max(0.0, y1 - h * pad_ratio)
    x2p = min(frame_w, x2 + w * pad_ratio)
    y2p = min(frame_h, y2 + h * pad_ratio)
    return x1p, y1p, x2p, y2p


def _detect_on_crop(frame, bbox, pad_ratio, detector):
    """Run YOLO pose on a padded crop; returns (kpts, kpt_conf) mapped to full-frame coords."""
    fh, fw = frame.shape[:2]
    x1p, y1p, x2p, y2p = _pad_bbox(bbox, pad_ratio, fh, fw)
    x1i, y1i, x2i, y2i = int(x1p), int(y1p), int(x2p), int(y2p)
    if x2i <= x1i or y2i <= y1i:
        return None, None
    crop = frame[y1i:y2i, x1i:x2i]
    results = detector.model.predict(
        crop, imgsz=detector.imgsz, conf=detector.conf_thres,
        quantize=16 if detector._use_half else None, verbose=False,
    )[0]
    if results.keypoints is None or len(results.keypoints.xy) == 0:
        return None, None
    kpts_crop = results.keypoints.xy[0].cpu().numpy()
    kpt_conf = (
        results.keypoints.conf[0].cpu().numpy()
        if results.keypoints.conf is not None
        else np.ones(kpts_crop.shape[0], dtype=np.float32)
    )
    kpts_full = kpts_crop.copy()
    kpts_full[:, 0] += x1i
    kpts_full[:, 1] += y1i
    return kpts_full, kpt_conf


def _mode7_kp_stats(records):
    """Returns (per_kp_p95_list, mean_p95) for one padding ratio's frame records."""
    p95_list = []
    for i in range(17):
        disp_vals = [r['kp_disp'][i] for r in records]
        _, _, _, p95, _ = _stats(disp_vals)
        p95_list.append(p95)
    valid = [v for v in p95_list if v > 0]
    mean_p95 = np.mean(valid).item() if valid else 0.0
    return p95_list, mean_p95


def save_mode7_results(padding_records: dict, run_dir: Path):
    """MODE 7: ROI Padding Impact on Keypoint Stability."""
    generated = []

    # Compute stats per padding ratio
    stats_rows = []
    for pad in ROI_PADDING_LIST:
        records = padding_records.get(pad, [])
        kp_p95_list, mean_p95 = _mode7_kp_stats(records)
        stats_rows.append({
            'pad': pad,
            'kp_p95_list': kp_p95_list,
            'mean_p95': mean_p95,
            'records': records,
        })

    baseline = next((s for s in stats_rows if abs(s['pad']) < 1e-9), stats_rows[0])
    raw_p95 = baseline['mean_p95']
    for s in stats_rows:
        s['improvement'] = (
            (raw_p95 - s['mean_p95']) / raw_p95 * 100.0
            if raw_p95 > 1e-9 else 0.0
        )

    non_zero = [s for s in stats_rows if s['pad'] > 1e-9]
    best = max(non_zero, key=lambda s: s['improvement']) if non_zero else baseline

    # ── Terminal output ──────────────────────────────────────────────
    sep = "=" * 50
    print(f"\n{sep}")
    print("ROI Padding Analysis")
    print(sep)
    for s in stats_rows:
        if abs(s['pad']) < 1e-9:
            print(f"\nPadding=0.0\nMean KP P95:\n{s['mean_p95']:.3f} px")
        else:
            print(f"\nPadding={s['pad']:.1f}\nMean KP P95:\n{s['mean_p95']:.3f} px"
                  f"\nImprovement:\n{s['improvement']:.1f} %")
    print(f"\nBest Padding:\n{best['pad']:.1f}")
    print(f"\nMaximum Improvement:\n{best['improvement']:.1f} %")
    print(sep)

    if best['improvement'] < 10.0:
        diag = (
            "ROI padding has limited influence on pose stability.\n"
            "The dominant source of jitter is likely keypoint localization error rather than ROI size."
        )
    else:
        diag = (
            "ROI size has a noticeable influence on pose stability.\n"
            "Further optimization of the detection region may improve keypoint stability."
        )
    print(f"\n{diag}")
    print(sep)

    # ── CSV: padding_summary.csv ────────────────────────────────────
    csv_path = run_dir / "padding_summary.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Padding', 'Mean_KP_P95', 'Improvement_%'])
        for s in stats_rows:
            writer.writerow([
                f"{s['pad']:.1f}",
                f"{s['mean_p95']:.3f}",
                f"{s['improvement']:.1f}",
            ])
    generated.append(csv_path.name)

    # ── diagnosis.txt ───────────────────────────────────────────────
    txt_path = run_dir / "diagnosis.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Raw KP P95 (padding=0.0): {raw_p95:.3f} px\n")
        f.write(f"Best Padding Ratio:        {best['pad']:.1f}\n")
        f.write(f"Best Smoothed KP P95:      {best['mean_p95']:.3f} px\n")
        f.write(f"Maximum Improvement:       {best['improvement']:.1f} %\n\n")
        f.write(f"Diagnosis:\n{diag}\n")
    generated.append(txt_path.name)

    # ── Plots ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        pads      = [s['pad']        for s in stats_rows]
        p95_vals  = [s['mean_p95']   for s in stats_rows]
        impr_vals = [s['improvement'] for s in stats_rows]

        # Plot 1: roi_padding_vs_kp_jitter.png
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(pads, p95_vals, 'o-', color='royalblue', linewidth=1.8, markersize=6)
        ax.axvline(best['pad'], color='tomato', linewidth=0.9, linestyle='--',
                   label=f'Best={best["pad"]:.1f}')
        ax.set_xlabel('Padding Ratio', fontsize=11)
        ax.set_ylabel('Mean Keypoint P95 (px)', fontsize=11)
        ax.set_title('ROI Padding vs Keypoint Jitter', fontsize=12)
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        p = run_dir / "roi_padding_vs_kp_jitter.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        generated.append(p.name)

        # Plot 2: roi_padding_improvement.png
        y_scale = max(impr_vals) if max(impr_vals) > 0 else 1.0
        bar_colors = ['seagreen' if v >= 0 else 'tomato' for v in impr_vals]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar([str(p) for p in pads], impr_vals, color=bar_colors, alpha=0.85, width=0.55)
        for bar, val in zip(bars, impr_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_scale * 0.02,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9,
            )
        ax.axhline(0,  color='gray',   linewidth=0.8, linestyle='--')
        ax.axhline(10, color='orange', linewidth=0.7, linestyle=':', label='10% threshold')
        ax.set_xlabel('Padding Ratio', fontsize=11)
        ax.set_ylabel('Keypoint Jitter Improvement (%)', fontsize=11)
        ax.set_title('ROI Padding: Improvement over No Padding', fontsize=12)
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        p = run_dir / "roi_padding_improvement.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        generated.append(p.name)

        # Plot 3: top5_keypoints_best_padding.png
        base_records = baseline['records']
        best_records = best['records']
        n = min(len(base_records), len(best_records))
        frames_b = [base_records[fi]['frame'] for fi in range(n)]

        kp_delta = []
        for i in range(17):
            if i in EXCLUDED_KEYPOINTS:
                continue
            bp95 = baseline['kp_p95_list'][i]
            bsp95 = best['kp_p95_list'][i]
            imp = (bp95 - bsp95) / bp95 * 100.0 if bp95 > 1e-9 else 0.0
            kp_delta.append((i, KEYPOINT_NAMES[i], imp))
        top5 = sorted(kp_delta, key=lambda x: x[2], reverse=True)[:5]

        colors5 = ['tomato', 'darkorange', 'gold', 'mediumseagreen', 'steelblue']
        fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
        for ax, (kp_idx, kp_name, imp_pct), col in zip(axes, top5, colors5):
            raw_d = [base_records[fi]['kp_disp'][kp_idx] for fi in range(n)]
            pad_d = [best_records[fi]['kp_disp'][kp_idx] for fi in range(n)]
            ax.plot(frames_b, raw_d, color='lightsteelblue', linewidth=0.7, label='Padding=0.0')
            ax.plot(frames_b, pad_d, color=col, linewidth=0.9, alpha=0.85,
                    label=f'Padding={best["pad"]:.1f} (Δ={imp_pct:.1f}%)')
            ax.set_ylabel(kp_name, fontsize=8); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        axes[-1].set_xlabel('Frame')
        fig.suptitle(f'Top 5 Improved Keypoints — Padding 0.0 vs {best["pad"]:.1f}')
        plt.tight_layout()
        p = run_dir / "top5_keypoints_best_padding.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        generated.append(p.name)

    except ImportError:
        print("[WARN] matplotlib 未安裝，跳過繪圖")

    _print_output_summary(run_dir, generated)


# ── MODE 6 helpers ──────────────────────────────────────────────────
def _create_mode_run_dir(mode: int, ts: str) -> Path:
    """Create a timestamped run directory for a specific analysis mode."""
    subdir = _MODE_SUBDIR.get(mode, 'unknown')
    run_dir = OUTPUT_DIR / subdir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_mode6_results(frame_records: list):
    """MODE 6: 一鍵依序執行 Mode 1~5，輸出所有分析結果。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sep = "=" * 60
    print("\n" + sep)
    print("MODE 6: 全模式分析 — 依序執行 Mode 1~5")
    print(sep)

    mode_funcs = [
        (1, save_mode1_results),
        (2, save_mode2_results),
        (3, save_mode3_results),
        (4, save_mode4_results),
        (5, save_mode5_results),
    ]
    for mode_num, func in mode_funcs:
        print(f"\n{'─' * 52}")
        print(f"  ▶ Mode {mode_num} / 5 : {_MODE_SUBDIR[mode_num]}")
        print(f"{'─' * 52}")
        run_dir = _create_mode_run_dir(mode_num, ts)
        func(frame_records, run_dir)

    print(f"\n{sep}")
    print("MODE 6 完成：所有模式輸出已儲存")
    print(sep)


# ── 主函數 ─────────────────────────────────────────────────────────
def main():
    feature_mode = STGCN_FEATURE_MODE

    if not Path(VIDEO_PATH).exists():
        print(f"❌ 影片不存在: {VIDEO_PATH}")
        return

    print("=" * 60)
    print(f"Pose Jitter Analysis  [MODE {ANALYSIS_MODE}: {_MODE_SUBDIR.get(ANALYSIS_MODE, '?')}]")
    print("=" * 60)
    print(f"影片: {VIDEO_PATH}")
    if EXCLUDED_KEYPOINTS:
        excl_names = [KEYPOINT_NAMES[i] for i in sorted(EXCLUDED_KEYPOINTS) if i < 17]
        print(f"排除關鍵點: {sorted(EXCLUDED_KEYPOINTS)} → {excl_names}")
    print("=" * 60)

    # 推斷模型通道數
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
                            feature_mode = ck_channel_map[ck_in_ch]
                    in_channels = ck_in_ch
            except Exception as e:
                print(f"⚠ 無法載入 checkpoint 以推斷通道數: {e}")
    except Exception:
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

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片: {VIDEO_PATH}")
        cv2.destroyAllWindows()
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 1:
        source_fps = TARGET_MODEL_FPS
    frame_step = 1
    if ENABLE_FPS_DOWNSAMPLE and source_fps > TARGET_MODEL_FPS + 1e-6:
        frame_step = max(1, int(round(source_fps / TARGET_MODEL_FPS)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS={source_fps:.1f}  total={total_frames}  frame_step={frame_step}")

    frame_records: list = []
    # MODE 7: per-padding ratio records & prev-KP state
    _m7_pad_records: dict = (
        {p: [] for p in ROI_PADDING_LIST} if ANALYSIS_MODE == 7 else {}
    )
    _m7_pad_prev_kp: dict = (
        {p: np.full((17, 2), np.nan) for p in ROI_PADDING_LIST} if ANALYSIS_MODE == 7 else {}
    )
    keypoints_buffer = deque(maxlen=SEQUENCE_LENGTH)
    ema_kpts = None
    prev_bbox_cx = prev_bbox_cy = np.nan
    prev_kp_xy = np.full((17, 2), np.nan)
    local_sampled = 0
    raw_frames_read = 0
    user_quit = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 播完自動結束

        raw_frames_read += 1
        if raw_frames_read % frame_step != 0:
            continue
        local_sampled += 1

        kpts, kpt_conf, bbox, bbox_conf_raw = keypoint_detector.detect(frame)
        bbox_conf = (
            np.float64(bbox_conf_raw)
            if bbox_conf_raw is not None
            else np.nan
        )
        # Save raw values for MODE 7 (before EMA modification)
        _m7_kpts_raw = kpts.copy() if kpts is not None else None
        _m7_kconf_raw = kpt_conf.copy() if kpt_conf is not None else None
        _m7_bbox_raw = bbox.copy() if bbox is not None else None

        if kpts is not None:
            if ema_kpts is None:
                ema_kpts = kpts.copy()
            else:
                ema_kpts = EMA_ALPHA * kpts + (1.0 - EMA_ALPHA) * ema_kpts
            kpts = ema_kpts.copy()

            keypoints_buffer.append((kpts, kpt_conf))

            # 行為分類（保留原始推論管線）
            if len(keypoints_buffer) >= SEQUENCE_LENGTH and (local_sampled % CLASSIFY_STRIDE == 0):
                kpts_arr = np.array([item[0] for item in keypoints_buffer])
                conf_arr = np.array([item[1] for item in keypoints_buffer])
                seq_array = interpolate_missing(kpts_arr, conf_arr, threshold=0.0)
                if STGCN_NORMALIZE:
                    seq_array = flip_normalize(seq_array)
                    seq_array = orientation_normalize(seq_array)
                    seq_array = normalize_skeleton_coords(seq_array)
                seq_features = build_feature_tensor(seq_array, conf_arr, feature_mode)
                behavior_classifier.classify(seq_features, precomputed=True)

            # BBox center displacement
            bbox_cx, bbox_cy = _bbox_center(bbox)
            if np.isfinite(prev_bbox_cx) and np.isfinite(bbox_cx):
                bbox_dx   = bbox_cx - prev_bbox_cx
                bbox_dy   = bbox_cy - prev_bbox_cy
                bbox_disp = np.sqrt(bbox_dx * bbox_dx + bbox_dy * bbox_dy).item()
            else:
                bbox_dx = bbox_dy = bbox_disp = np.nan

            # All 17 keypoints displacement
            kp_x    = np.full(17, np.nan)
            kp_y    = np.full(17, np.nan)
            kp_dx   = np.full(17, np.nan)
            kp_dy   = np.full(17, np.nan)
            kp_disp = np.full(17, np.nan)
            for i in range(17):
                if kpt_conf[i].item() >= JITTER_CONF_THRESHOLD:
                    kp_x[i] = kpts[i][0].item()
                    kp_y[i] = kpts[i][1].item()
                    if np.isfinite(prev_kp_xy[i, 0]):
                        dx = kp_x[i] - prev_kp_xy[i, 0]
                        dy = kp_y[i] - prev_kp_xy[i, 1]
                        kp_dx[i]   = dx
                        kp_dy[i]   = dy
                        kp_disp[i] = np.sqrt(dx * dx + dy * dy).item()

            # 將排除的關鍵點強制清除，避免混入分析結果
            for _excl in EXCLUDED_KEYPOINTS:
                if 0 <= _excl < 17:
                    kp_x[_excl] = kp_y[_excl] = np.nan
                    kp_dx[_excl] = kp_dy[_excl] = kp_disp[_excl] = np.nan
            stored_kpt_conf = kpt_conf.copy()
            for _excl in EXCLUDED_KEYPOINTS:
                if 0 <= _excl < 17:
                    stored_kpt_conf[_excl] = 0.0

            frame_records.append({
                'frame': local_sampled,
                'bbox_cx': bbox_cx, 'bbox_cy': bbox_cy,
                'bbox_dx': bbox_dx, 'bbox_dy': bbox_dy, 'bbox_disp': bbox_disp,
                'bbox_conf': bbox_conf,
                'kp_x': kp_x.copy(), 'kp_y': kp_y.copy(),
                'kp_dx': kp_dx.copy(), 'kp_dy': kp_dy.copy(),
                'kp_disp': kp_disp.copy(),
                'kp_conf': stored_kpt_conf,
            })

            prev_bbox_cx, prev_bbox_cy = bbox_cx, bbox_cy
            for i in range(17):
                if np.isfinite(kp_x[i]):
                    prev_kp_xy[i, 0] = kp_x[i]
                    prev_kp_xy[i, 1] = kp_y[i]
                else:
                    prev_kp_xy[i] = np.nan
        else:
            ema_kpts = None
            prev_bbox_cx = prev_bbox_cy = np.nan
            prev_kp_xy[:] = np.nan

        # ── MODE 7: per-padding-ratio re-detection ───────────────────
        if ANALYSIS_MODE == 7:
            for _pad in ROI_PADDING_LIST:
                if abs(_pad) < 1e-9:
                    _kp_p, _kc_p = _m7_kpts_raw, _m7_kconf_raw
                elif _m7_bbox_raw is not None:
                    _kp_p, _kc_p = _detect_on_crop(frame, _m7_bbox_raw, _pad, keypoint_detector)
                else:
                    _kp_p, _kc_p = None, None

                _kp_x_p   = np.full(17, np.nan)
                _kp_y_p   = np.full(17, np.nan)
                _kp_disp_p = np.full(17, np.nan)
                if _kp_p is not None and _kc_p is not None:
                    for _i in range(min(17, len(_kp_p))):
                        _cv = _kc_p[_i].item() if hasattr(_kc_p[_i], 'item') else _kc_p[_i]
                        if _cv >= JITTER_CONF_THRESHOLD:
                            _kx = _kp_p[_i, 0].item() if hasattr(_kp_p[_i, 0], 'item') else _kp_p[_i, 0]
                            _ky = _kp_p[_i, 1].item() if hasattr(_kp_p[_i, 1], 'item') else _kp_p[_i, 1]
                            _kp_x_p[_i] = _kx
                            _kp_y_p[_i] = _ky
                            if np.isfinite(_m7_pad_prev_kp[_pad][_i, 0]):
                                _dx = _kx - _m7_pad_prev_kp[_pad][_i, 0]
                                _dy = _ky - _m7_pad_prev_kp[_pad][_i, 1]
                                _kp_disp_p[_i] = np.sqrt(_dx * _dx + _dy * _dy).item()
                            _m7_pad_prev_kp[_pad][_i, 0] = _kx
                            _m7_pad_prev_kp[_pad][_i, 1] = _ky
                        else:
                            _m7_pad_prev_kp[_pad][_i] = np.nan
                else:
                    _m7_pad_prev_kp[_pad][:] = np.nan

                for _excl in EXCLUDED_KEYPOINTS:
                    if 0 <= _excl < 17:
                        _kp_x_p[_excl] = np.nan
                        _kp_y_p[_excl] = np.nan
                        _kp_disp_p[_excl] = np.nan
                        _m7_pad_prev_kp[_pad][_excl] = np.nan
                _m7_pad_records[_pad].append({
                    'frame':    local_sampled,
                    'kp_x':    _kp_x_p.copy(),
                    'kp_y':    _kp_y_p.copy(),
                    'kp_disp': _kp_disp_p.copy(),
                })

        # GUI 顯示（骨架 + 幀計數 HUD）
        show_frame, sc, cx_off, cy_off = resize_with_letterbox(frame.copy(), DISPLAY_SIZE)
        if kpts is not None:
            disp_kpts, disp_bbox = scale_kpts_and_bbox_for_letterbox(kpts, bbox, sc, cx_off, cy_off)
            draw_skeleton_overlay(show_frame, disp_kpts, kpt_conf, disp_bbox)

        h, w = show_frame.shape[:2]
        ui_s = compute_ui_scale(w, h)
        fs = 0.55 * ui_s
        th = scale_px(2, ui_s, min_px=1)
        info = f"Frame {local_sampled}  Records: {len(frame_records)}  [q] 提前結束"
        cv2.putText(show_frame, info, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 1, cv2.LINE_AA)
        cv2.putText(show_frame, info, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, fs, (220, 255, 220), th, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, show_frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            user_quit = True
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n已處理 {local_sampled} 幀，記錄 {len(frame_records)} 筆{'（提前結束）' if user_quit else ''}")

    if not frame_records:
        print("❌ 無任何記錄，請確認影片與 YOLO 模型路徑正確")
        return

    if ANALYSIS_MODE == 6:
        save_mode6_results(frame_records)
    elif ANALYSIS_MODE == 7:
        run_dir = create_run_output_dir()
        if not any(_m7_pad_records.get(p) for p in ROI_PADDING_LIST):
            print("❌ MODE 7: 無任何 padding 記錄，請確認影片與模型路徑")
        else:
            save_mode7_results(_m7_pad_records, run_dir)
    else:
        run_dir = create_run_output_dir()
        if ANALYSIS_MODE == 1:
            save_mode1_results(frame_records, run_dir)
        elif ANALYSIS_MODE == 2:
            save_mode2_results(frame_records, run_dir)
        elif ANALYSIS_MODE == 3:
            save_mode3_results(frame_records, run_dir)
        elif ANALYSIS_MODE == 4:
            save_mode4_results(frame_records, run_dir)
        elif ANALYSIS_MODE == 5:
            save_mode5_results(frame_records, run_dir)
        else:
            print(f"❌ 未知的 ANALYSIS_MODE: {ANALYSIS_MODE}（應為 1–7）")


if __name__ == '__main__':
    main()
