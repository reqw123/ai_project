"""
多影片左右耳距離監測腳本

功能：
1) 由 VIDEO_LIST 與 VIDEO_PATH 載入多支影片
2) 每幀偵測左右耳關鍵點（LeftEar=1, RightEar=2）
3) 計算兩點歐式距離（像素）並即時顯示於畫面
4) 將每幀距離與區域命中資訊寫入 CSV 檔案
"""

import concurrent.futures
import csv
import json
import math
import socket as _socket
import socketserver
import sys
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler
from pathlib import Path

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _requests = None
    _HAS_REQUESTS = False

try:
    import msvcrt
except ImportError:
    msvcrt = None

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from detectors.keypoint_detector import KeypointDetector
from utils.constants import (
    EAR_DISTANCE_EDGE_COLORS as _EDGE_COLORS,
    EAR_DISTANCE_KP_COLORS as _KP_COLORS,
    EAR_DISTANCE_SKELETON_EDGES as _SKELETON_EDGES,
)


# ===== 可直接修改的預設參數 =====
VIDEO_PATH = r"C:\Users\homec\Downloads\lick_ai" # 主要作為資料夾來源（會遞迴掃描影片）
VIDEO_LIST = [
    # 只放「單一影片檔案路徑」
   # r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\泛化測試\cat5.mp4",
    #r"C:\Users\homec\Downloads\OneDrive_2_2026-5-21", 
    #r"C:\Users\homec\Downloads\OneDrive_3_2026-5-21",
    #r"C:\Users\homec\Downloads\OneDrive_4_2026-5-21",
  #  r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\lick\2448166_Cat_Licking_1920x1080.mp4",
  #  r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\lick\2404508_Cat_Licking_1920x1080.mp4",#不要刪
  #  r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\lick\5878298_Fur_Baby_Cat_1920x1080.mp4",#不要刪
  # r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\lick\5878299_Fur_Baby_Cat_1920x1080.mp4"#不要刪
]

MAX_VIDEOS = 40  # 讀取上限：目前最多 20 部（原 10 部 + 額外 10 部）
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
YOLO_MODEL_PATH = r"C:\AI_Project\cat_pose\v11s_104.pt"

# 輸出 CSV 路徑（可直接在此處修改）
OUTPUT_CSV_PATH = r"C:\Users\homec\left_right_ear_distance.csv"
# Node-RED 即時推送（設為 None 或空字串可停用）
NODERED_URL = "http://127.0.0.1:1880/lick_zone_result"
INFERENCE_DEVICE = "cuda"
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.5
TARGET_MODEL_FPS = 30.0
ENABLE_FPS_DOWNSAMPLE = True
EMA_ALPHA = 1.0
DISPLAY_WINDOW = True
WINDOW_NAME = "LeftEar-RightEar Distance"
DISPLAY_SIZE = (1080, 720)
# 顯示模式: 1=本地 OpenCV 視窗  2=HTTP MJPEG 串流（不開視窗）
STREAM_MODE = 1
MJPEG_PORT = 5000
MJPEG_QUALITY = 75
NODERED_ONLINE_URL = "http://127.0.0.1:1880/lick_python_online"
_NEED_FRAME = DISPLAY_WINDOW or (STREAM_MODE == 2)
# ===== 關鍵點信心門檻（建議集中看這區） =====
EAR_CONF_THRESHOLD = 0.5  # 左右耳/胸/臀「幾何與耳距有效性」門檻（>此值才視為可用）
DRAW_KP_CONF_THRESHOLD = 0.5  # 骨架與關鍵點「顯示門檻」（>此值才畫；只影響畫面，不影響耳距有效性）
LIMB_CONF_THRESHOLD = 0.10  # 四肢區域建立門檻（膝與掌都需 > 此值）
LOOP_PLAYBACK = True
WRITE_LOOPED_PASSES_TO_CSV = False  # LOOP_PLAYBACK=True 時，是否把第 2 輪以後的資料也寫入 CSV


KP_LEFT_EAR   = 1
KP_RIGHT_EAR  = 2
KP_NOSE       = 0
KP_CHEST      = 3  # 前胸/胸口（前肢附著點）— 對應 stgcn chest_joint=3
KP_HIP        = 5  # 臀部（後肢附著點）— 對應 stgcn lower_body_joint=5
KP_FRONT_LEFT_KNEE = 6
KP_FRONT_LEFT_PAW = 7
KP_FRONT_RIGHT_KNEE = 8
KP_FRONT_RIGHT_PAW = 9
KP_HIND_LEFT_KNEE = 10
KP_HIND_LEFT_PAW = 11
KP_HIND_RIGHT_KNEE = 12
KP_HIND_RIGHT_PAW = 13

# ===== 面相偵測參數（可依資料再微調） =====
NOSE_CONF_THRESHOLD = 0.3  # 鼻子可用門檻（同時用於本體幾何與背向規則）
STATE_SMOOTH_WINDOW = 30

# ===== 舊版螢幕座標 fallback（通常不建議啟用） =====
ENABLE_LEGACY_SCREEN_FALLBACK = False
FACE_NORM_MIN = 0.35
FACE_NORM_MAX = 0.8
FRONT_LR_MARGIN_RATIO = 0.08
FRONT_LR_MARGIN_MIN_PX = 0.5

# ===== 貓咪本體座標朝向判定參數（優先使用） =====
# forward_norm = dot(nose - body_center, -body_axis_unit) / body_len
# lateral_norm = dot(nose - body_center, body_normal) / body_len
CAT_FRONT_FORWARD_MIN = 0.08
CAT_BACK_FORWARD_MIN = 0.10
CAT_LR_MARGIN = 0.06
CAT_LR_SIGN = 1.0  # 若左右與你的標註相反，改成 -1.0

# ===== 使用者規則（高優先級） =====
# 1) 頭朝鏡頭：nose-ears 夾角 > 45 且 n(=distance_norm) > 0.3
# 2) 頭背鏡頭：nose 信心值低，且 n > 3（這裡以像素距離 distance_px 實作）
FRONT_CAMERA_ANGLE_MIN_DEG = 45.0
FRONT_CAMERA_NORM_MIN = 0.30
BACK_CAMERA_NOSE_CONF_MAX = 0.5
BACK_CAMERA_DIST_MIN_PX = 3.0
BACK_VIEW_REQUIRE_LOW_NOSE_CONF = True

# ===== 正面視角守門（避免 body scale 退化時誤判） =====
FRONT_VIEW_GUARD_ENABLED = True
# 以畫面對角線正規化後的 body scale 門檻，避免受解析度影響
FRONT_VIEW_BODY_SCALE_NORM_MAX = 0.015
FRONT_VIEW_BODY_EAR_RATIO_MAX = 0.75

STATE_UNKNOWN = "UNKNOWN"
STATE_FRONT = "FACING_CAMERA"
STATE_FRONT_LEFT = "FRONT_LEFT"
STATE_FRONT_RIGHT = "FRONT_RIGHT"
STATE_BACK = "BACK_VIEW"
STATE_FRONT_VIEW = "FRONT_VIEW"
STATE_NO_CAT = "NO_CAT"

ELLIPSE_SAMPLE_COUNT = 40
_ELLIPSE_SAMPLE_ANGLES = np.linspace(0.0, 2.0 * np.pi, ELLIPSE_SAMPLE_COUNT, endpoint=False)
_ELLIPSE_COS = np.cos(_ELLIPSE_SAMPLE_ANGLES)
_ELLIPSE_SIN = np.sin(_ELLIPSE_SAMPLE_ANGLES)

WINDOW_SCALE_STEP = 0.10
WINDOW_SCALE_MIN = 0.50
WINDOW_SCALE_MAX = 2.00

UI_MODE_FULL = "FULL"
UI_MODE_DET_ONLY = "DET_ONLY"

# ===== 效能調校 =====
# True: 保留完整視覺特效；False: 減少每幀昂貴特效以提升流暢度
ENABLE_HEAVY_VISUAL_EFFECTS = False
_HEAVY_FX = bool(ENABLE_HEAVY_VISUAL_EFFECTS)   # 預計算，避免每幀 bool() 呼叫
_EMA_BYPASS = EMA_ALPHA >= 1.0 - 1e-9           # 預計算，避免每幀浮點比較

# ===== 頭部朝向身體區域判斷與可視化參數 =====
# 身體中心區域（橢圓）：寬=橫向軸（body_normal），高=縱向軸（body_axis）
BODY_REGION_ELLIPSE_WIDTH_RATIO = 0.65
BODY_REGION_ELLIPSE_HEIGHT_RATIO = 0.27
HEAD_RAY_LENGTH_RATIO = 1.60      # 頭部向量延伸射線長度（相對 body_axis）
HEAD_RAY_MIN_PX = 60.0

# ===== 鼻子接觸區域（梯形）參數：依身體尺度等比適配 =====
# 厚度以「公分」描述，再依 CAT_BODY_LENGTH_CM 轉換到像素；寬度以 body_len 比例描述。
CAT_BODY_LENGTH_CM = 40.0
NOSE_CONTACT_TRAPEZOID_THICKNESS_CM = 2.2
NOSE_CONTACT_TRAPEZOID_THICKNESS_SCALE = 1.0
NOSE_CONTACT_TRAPEZOID_TOP_WIDTH_RATIO = 0.10
NOSE_CONTACT_TRAPEZOID_BOTTOM_WIDTH_RATIO = 0.20
NOSE_CONTACT_TRAPEZOID_WIDTH_SCALE = 1.15
# 梯形與四肢區域的等效身體長度範圍，限縮遠近情境的尺寸差距：
#   MIN：貓較遠（body_len 小）時的下限，防止區域過小漏判
#   MAX：貓較近（body_len 大）時的上限，防止區域過大誤觸
# 所有接觸區域尺寸皆以 clamp(body_len, MIN, MAX) 計算，不再線性隨距離無限放大/縮小。
CONTACT_BODY_LEN_MIN_PX = 300.0
CONTACT_BODY_LEN_MAX_PX = 650.0

# ===== 四肢接觸區域參數：全部可調、依身體尺度等比適配 =====
LIMB_CONTACT_SCALE = 1.0
LIMB_PAW_CIRCLE_RADIUS_RATIO = 0.04
LIMB_STRIP_HALF_WIDTH_RATIO = 0.055
# 長條端點與圓邊界的間隙（0=貼齊圓邊界；>0 可留細縫提升視覺分離感）
LIMB_STRIP_EDGE_GAP_RATIO = 0.0

# ===== 舔舐時間量測（依接觸區域累積秒數） =====
# 說明：時間僅做「各部位舔舐時長」記錄，不作為舔舐成立判準。

LIMB_SEGMENTS = [
    ("LIMB_FL", KP_FRONT_LEFT_KNEE, KP_FRONT_LEFT_PAW),
    ("LIMB_FR", KP_FRONT_RIGHT_KNEE, KP_FRONT_RIGHT_PAW),
    ("LIMB_HL", KP_HIND_LEFT_KNEE, KP_HIND_LEFT_PAW),
    ("LIMB_HR", KP_HIND_RIGHT_KNEE, KP_HIND_RIGHT_PAW),
]

LIMB_LABEL_MAP = {
    "LIMB_FL": "FL",
    "LIMB_FR": "FR",
    "LIMB_HL": "HL",
    "LIMB_HR": "HR",
}

# ===== 以鼻子落點推估命中區域 =====
LICK_ZONE_NO_TARGET = "NO_TARGET"
LICK_ZONE_CENTER = "BODY_CENTER"
LICK_ZONE_FL = "FL"
LICK_ZONE_FR = "FR"
LICK_ZONE_HL = "HL"
LICK_ZONE_HR = "HR"

CSV_FIELDNAMES = [
    "video_idx",
    "video_path",
    "playback_pass",
    "frame_step",
    "source_fps",
    "model_input_fps",
    "frame",
    "processed_frame",
    "processed_frame_global",
    "time_sec",
    "left_ear_x",
    "left_ear_y",
    "right_ear_x",
    "right_ear_y",
    "distance_px",
    "distance_norm",
    "ray_end_x",
    "ray_end_y",
    "ray_norm_x",
    "ray_norm_y",
    "lick_zone",
    "lick_axis_score",
    "lick_lateral_score",
    "nearest_target_label",
    "nearest_target_t",
    "limb_hit_any",
    "limb_hit_labels",
    "limb_hit_fl_frame",
    "limb_hit_fr_frame",
    "limb_hit_hl_frame",
    "limb_hit_hr_frame",
    "limb_entry_count_fl",
    "limb_entry_count_fr",
    "limb_entry_count_hl",
    "limb_entry_count_hr",
    "target_entry_count",
    "lick_time_body_sec",
    "lick_time_fl_sec",
    "lick_time_fr_sec",
    "lick_time_hl_sec",
    "lick_time_hr_sec",
    "lick_sec_per_hit_body",
    "lick_sec_per_hit_fl",
    "lick_sec_per_hit_fr",
    "lick_sec_per_hit_hl",
    "lick_sec_per_hit_hr",
    "lick_pref_pct_body",
    "lick_pref_pct_fl",
    "lick_pref_pct_fr",
    "lick_pref_pct_hl",
    "lick_pref_pct_hr",
    "nose_detected",
    "gaze_forward_norm",
    "gaze_lateral_norm",
    "gaze_angle_deg",
    "face_state_cat",
    "face_state_raw",
    "face_state",
    "state_stability",
    "valid",
]

SUMMARY_FIELDNAMES = [
    "video_idx",
    "video_path",
    "source_fps",
    "model_input_fps",
    "processed_frames",
    "video_elapsed_sec",
    "total_lick_time_sec",
    "dominant_pref_zone",
    "dominant_pref_pct",
    "body_hits",
    "body_lick_time_sec",
    "body_pref_pct",
    "body_bout_count",
    "body_mean_bout_sec",
    "fl_hits",
    "fl_lick_time_sec",
    "fl_pref_pct",
    "fl_bout_count",
    "fl_mean_bout_sec",
    "fr_hits",
    "fr_lick_time_sec",
    "fr_pref_pct",
    "fr_bout_count",
    "fr_mean_bout_sec",
    "hl_hits",
    "hl_lick_time_sec",
    "hl_pref_pct",
    "hl_bout_count",
    "hl_mean_bout_sec",
    "hr_hits",
    "hr_lick_time_sec",
    "hr_pref_pct",
    "hr_bout_count",
    "hr_mean_bout_sec",
]


def infer_face_state(valid_norm, dist_norm, nose_conf, lxy, rxy, nose_xy, nose_ok):
    """依正規化耳距與 nose 信心值推論朝向狀態。"""
    nose_low_conf = nose_conf < NOSE_CONF_THRESHOLD

    ear_span_x = float("nan")
    dx = float("nan")
    if nose_ok and np.isfinite(nose_xy[0]) and np.isfinite(lxy[0]) and np.isfinite(rxy[0]):
        ear_mid_x = 0.5 * (lxy[0] + rxy[0])
        ear_span_x = abs(lxy[0] - rxy[0])
        dx = nose_xy[0] - ear_mid_x

    if not valid_norm:
        return STATE_BACK if nose_low_conf else STATE_UNKNOWN

    # 正面與背面可能共享相近耳距，先以區間分群，再由 nose 拆分
    if FACE_NORM_MIN <= dist_norm <= FACE_NORM_MAX:
        if nose_low_conf:
            return STATE_BACK
        if np.isfinite(dx) and np.isfinite(ear_span_x):
            lr_margin_front = max(FRONT_LR_MARGIN_MIN_PX, FRONT_LR_MARGIN_RATIO * ear_span_x)
            if dx < -lr_margin_front:
                return STATE_FRONT_LEFT
            if dx > lr_margin_front:
                return STATE_FRONT_RIGHT
        return STATE_FRONT

    # 過渡區間：未達正面且非明確背面時，回傳 UNKNOWN。
    if dist_norm < FACE_NORM_MIN:
        return STATE_UNKNOWN

    # 耳距偏大且 nose 低信心時，偏向背對
    if nose_low_conf and dist_norm > FACE_NORM_MAX:
        return STATE_BACK

    return STATE_UNKNOWN


def infer_face_state_cat_centric_metrics(target_geom, nose_ok):
    """
    以貓咪本體座標判定朝向，並回傳連續量。

    Returns:
        (state_cat, forward_norm, lateral_norm, gaze_angle_deg)
    """
    if target_geom is None or not nose_ok:
        return STATE_UNKNOWN, float("nan"), float("nan"), float("nan")

    nose = np.asarray(target_geom.get("nose"), dtype=np.float64)
    body_center = np.asarray(target_geom.get("body_center"), dtype=np.float64)
    body_axis_unit = np.asarray(target_geom.get("body_axis_unit"), dtype=np.float64)
    body_normal = np.asarray(target_geom.get("body_normal"), dtype=np.float64)
    body_len = float(target_geom.get("body_len", 0.0))

    if body_len < 1e-6 or not (np.all(np.isfinite(nose)) and np.all(np.isfinite(body_center))):
        return STATE_UNKNOWN, float("nan"), float("nan"), float("nan")

    rel = nose - body_center
    # 以前方（胸方向）為正，後方（臀方向）為負。
    forward_norm = float(np.dot(rel, -body_axis_unit) / body_len)
    # 左右以身體法向量定義，可用 CAT_LR_SIGN 調整對應。
    lateral_norm = float(np.dot(rel, body_normal) / body_len) * float(CAT_LR_SIGN)
    gaze_angle_deg = float(np.degrees(np.arctan2(lateral_norm, forward_norm)))

    if forward_norm >= CAT_FRONT_FORWARD_MIN:
        if lateral_norm <= -CAT_LR_MARGIN:
            return STATE_FRONT_LEFT, forward_norm, lateral_norm, gaze_angle_deg
        if lateral_norm >= CAT_LR_MARGIN:
            return STATE_FRONT_RIGHT, forward_norm, lateral_norm, gaze_angle_deg
        return STATE_FRONT, forward_norm, lateral_norm, gaze_angle_deg

    if forward_norm <= -CAT_BACK_FORWARD_MIN:
        return STATE_BACK, forward_norm, lateral_norm, gaze_angle_deg

    return STATE_UNKNOWN, forward_norm, lateral_norm, gaze_angle_deg


def infer_face_state_user_rules(head_ear_angle_deg, dist_norm, dist_px, nose_conf):
    """依使用者指定條件優先判定 FRONT/BACK。"""
    # BACK 優先：背向成立時鼻子信心通常偏低。
    if BACK_VIEW_REQUIRE_LOW_NOSE_CONF and nose_conf <= BACK_CAMERA_NOSE_CONF_MAX:
        if (not np.isfinite(dist_px)) or dist_px > BACK_CAMERA_DIST_MIN_PX:
            return STATE_BACK, True

    if np.isfinite(head_ear_angle_deg) and np.isfinite(dist_norm):
        if head_ear_angle_deg > FRONT_CAMERA_ANGLE_MIN_DEG and dist_norm > FRONT_CAMERA_NORM_MIN:
            return STATE_FRONT, True

    if (not BACK_VIEW_REQUIRE_LOW_NOSE_CONF) and nose_conf <= BACK_CAMERA_NOSE_CONF_MAX and np.isfinite(dist_px) and dist_px > BACK_CAMERA_DIST_MIN_PX:
        return STATE_BACK, True

    return STATE_UNKNOWN, False


def smooth_state(state_hist):
    """用最近 N 幀多數決，降低狀態跳動。"""
    if not state_hist:
        return STATE_UNKNOWN, 0.0
    c = Counter(state_hist)
    state, votes = c.most_common(1)[0]
    return state, votes / len(state_hist)


def _point_in_oriented_ellipse(point, center, axis_u, axis_v, radius_u, radius_v):
    rel = point - center
    u = float(np.dot(rel, axis_u))
    v = float(np.dot(rel, axis_v))
    ru2 = max(float(radius_u) * float(radius_u), 1e-9)
    rv2 = max(float(radius_v) * float(radius_v), 1e-9)
    return (u * u) / ru2 + (v * v) / rv2 <= (1.0 + 1e-9)


def _sample_ellipse_boundary(center, axis_u, axis_v, radius_u, radius_v, sample_count=ELLIPSE_SAMPLE_COUNT):
    center = np.asarray(center, dtype=np.float64)
    axis_u = np.asarray(axis_u, dtype=np.float64)
    axis_v = np.asarray(axis_v, dtype=np.float64)

    if sample_count == ELLIPSE_SAMPLE_COUNT:
        cos_t = _ELLIPSE_COS
        sin_t = _ELLIPSE_SIN
    else:
        angles = np.linspace(0.0, 2.0 * np.pi, int(sample_count), endpoint=False)
        cos_t = np.cos(angles)
        sin_t = np.sin(angles)

    pts = (
        center
        + np.outer(cos_t * float(radius_u), axis_u)
        + np.outer(sin_t * float(radius_v), axis_v)
    )
    return pts


def _distance_point_to_segment(point, seg_start, seg_end):
    """回傳點到線段最短距離。"""
    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(seg_start, dtype=np.float64)
    b = np.asarray(seg_end, dtype=np.float64)
    ab = b - a
    ab2 = ab[0] * ab[0] + ab[1] * ab[1]
    if ab2 < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    pa = p - a
    t = max(0.0, min(1.0, (pa[0] * ab[0] + pa[1] * ab[1]) / ab2))
    return math.hypot(pa[0] - t * ab[0], pa[1] - t * ab[1])


def _compute_strip_corners(p0, p1, half_width):
    """回傳長條區域四角點（順時針）。"""
    seg = p1 - p0
    seg_len = math.hypot(seg[0], seg[1])
    if seg_len < 1e-6:
        return None
    axis = seg / seg_len
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    off = normal * half_width
    return [p0 + off, p1 + off, p1 - off, p0 - off]


def _point_in_polygon(point, polygon):
    """以射線法判斷點是否在多邊形內（含邊界）。"""
    p = np.asarray(point, dtype=np.float64)
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return False

    inside = False
    x = float(p[0])
    y = float(p[1])
    n = poly.shape[0]
    for i in range(n):
        j = (i - 1) % n
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])

        # 邊界容差判定
        sx, sy = xj - xi, yj - yi
        rx, ry = x - xi, y - yi
        seg_norm = math.hypot(sx, sy)
        if seg_norm > 1e-9:
            area2 = abs(sx * ry - sy * rx)
            dotv = rx * sx + ry * sy
            if area2 / seg_norm <= 1e-6 and -1e-9 <= dotv <= sx * sx + sy * sy + 1e-9:
                return True

        dy = yj - yi
        if abs(dy) < 1e-12:
            dy = 1e-12
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / dy + xi)
        if intersects:
            inside = not inside

    return inside


def _segments_intersect(p1, p2, q1, q2):
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    def _orient(a, b, c):
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def _on_seg(a, b, c):
        return (
            min(float(a[0]), float(b[0])) - 1e-9 <= float(c[0]) <= max(float(a[0]), float(b[0])) + 1e-9
            and min(float(a[1]), float(b[1])) - 1e-9 <= float(c[1]) <= max(float(a[1]), float(b[1])) + 1e-9
        )

    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)

    if (o1 * o2 < 0.0) and (o3 * o4 < 0.0):
        return True

    if abs(o1) <= 1e-9 and _on_seg(p1, p2, q1):
        return True
    if abs(o2) <= 1e-9 and _on_seg(p1, p2, q2):
        return True
    if abs(o3) <= 1e-9 and _on_seg(q1, q2, p1):
        return True
    if abs(o4) <= 1e-9 and _on_seg(q1, q2, p2):
        return True
    return False


def _polygons_intersect(poly_a, poly_b):
    a = np.asarray(poly_a, dtype=np.float64)
    b = np.asarray(poly_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 3 or b.shape[0] < 3:
        return False

    for pa in a:
        if _point_in_polygon(pa, b):
            return True
    for pb in b:
        if _point_in_polygon(pb, a):
            return True

    na = a.shape[0]
    nb = b.shape[0]
    for i in range(na):
        a0 = a[i]
        a1 = a[(i + 1) % na]
        for j in range(nb):
            b0 = b[j]
            b1 = b[(j + 1) % nb]
            if _segments_intersect(a0, a1, b0, b1):
                return True
    return False


def _polygon_aabb(poly):
    p = np.asarray(poly, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 1:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(np.min(p[:, 0])),
        float(np.min(p[:, 1])),
        float(np.max(p[:, 0])),
        float(np.max(p[:, 1])),
    )


def _aabb_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _polygon_contacts_circle(poly_pts, center, radius):
    """多邊形與圓接觸判定（含內含、邊界相切）。"""
    poly = np.asarray(poly_pts, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return False

    c = np.asarray(center, dtype=np.float64)
    r = max(float(radius), 0.0)
    if r <= 0.0:
        return False

    poly_box = _polygon_aabb(poly)
    circ_box = (float(c[0] - r), float(c[1] - r), float(c[0] + r), float(c[1] + r))
    if not _aabb_overlap(poly_box, circ_box):
        return False

    if _point_in_polygon(c, poly):
        return True

    for p in poly:
        if math.hypot(p[0] - c[0], p[1] - c[1]) <= r + 1e-9:
            return True

    n = poly.shape[0]
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if _distance_point_to_segment(c, a, b) <= r + 1e-9:
            return True

    return False


def _polygon_contacts_oriented_ellipse(poly_pts, center, axis_u, axis_v, radius_u, radius_v):
    """多邊形與橢圓任一處接觸即命中（含相切）。"""
    poly = np.asarray(poly_pts, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return False

    center = np.asarray(center, dtype=np.float64)
    eu = np.asarray(axis_u, dtype=np.float64)
    ev = np.asarray(axis_v, dtype=np.float64)
    ru = float(radius_u)
    rv = float(radius_v)

    poly_box = _polygon_aabb(poly)
    ex = abs(float(eu[0])) * ru + abs(float(ev[0])) * rv
    ey = abs(float(eu[1])) * ru + abs(float(ev[1])) * rv
    ellipse_box = (float(center[0] - ex), float(center[1] - ey), float(center[0] + ex), float(center[1] + ey))
    if not _aabb_overlap(poly_box, ellipse_box):
        return False

    if _point_in_polygon(center, poly):
        return True

    for p in poly:
        if _point_in_oriented_ellipse(p, center, eu, ev, ru, rv):
            return True

    for p in _sample_ellipse_boundary(center, eu, ev, ru, rv):
        if _point_in_polygon(p, poly):
            return True

    return False


def _limb_zone_group(zone_label):
    """將關節區域標籤（例如 FL_KNEE）映射為肢別（FL/FR/HL/HR）。"""
    if not zone_label:
        return ""
    if zone_label in (LICK_ZONE_FL, LICK_ZONE_FR, LICK_ZONE_HL, LICK_ZONE_HR):
        return zone_label
    for prefix in (LICK_ZONE_FL, LICK_ZONE_FR, LICK_ZONE_HL, LICK_ZONE_HR):
        if str(zone_label).startswith(prefix + "_"):
            return prefix
    return ""


def _angle_between_vectors_deg(vec_a, vec_b):
    """回傳兩向量夾角（度），無法計算時回傳 NaN。"""
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    na = math.hypot(a[0], a[1])
    nb = math.hypot(b[0], b[1])
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    cosv = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    return math.degrees(math.acos(cosv))


def _safe_sec_per_hit(total_sec, hit_count):
    """回傳每次舔舐平均秒數；無次數時回傳 NaN。"""
    if hit_count <= 0:
        return float("nan")
    return total_sec / hit_count


def _safe_pref_pct(part_sec, total_sec):
    """回傳偏好占比（%）；總時長為 0 時回傳 NaN。"""
    if total_sec <= 1e-9:
        return float("nan")
    return 100.0 * part_sec / total_sec


def _post_nodered(row: dict) -> None:
    if not NODERED_URL or not _HAS_REQUESTS:
        return
    try:
        _requests.post(NODERED_URL, json=row, timeout=0.3)
    except Exception:
        pass


# ===== MJPEG 串流伺服器（STREAM_MODE=2 使用） =====
_mjpeg_lock = threading.Lock()
_mjpeg_latest: bytes = b""
_ctrl_lock = threading.Lock()
_ctrl_switch_delta = 0  # 由 Node-RED 按鈕透過 POST /video_control 設定
_nr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="nr_post")


def _mjpeg_push(frame_bgr) -> None:
    global _mjpeg_latest
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
    if ok:
        with _mjpeg_lock:
            _mjpeg_latest = buf.tobytes()


class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        if self.path != "/stream":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--jpgboundary")
        self.send_header("Cache-Control", "no-cache")
        self._send_cors()
        self.end_headers()
        try:
            while True:
                with _mjpeg_lock:
                    data = _mjpeg_latest
                if data:
                    header = (
                        b"--jpgboundary\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(data)}\r\n\r\n".encode()
                    )
                    self.wfile.write(header + data + b"\r\n")
                    self.wfile.flush()
                time.sleep(1 / 30)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_POST(self):
        global _ctrl_switch_delta
        if self.path == "/video_control":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = self.rfile.read(length)
                action = json.loads(body).get("action", "")
                with _ctrl_lock:
                    if action == "next":
                        _ctrl_switch_delta = 1
                    elif action == "prev":
                        _ctrl_switch_delta = -1
            except (json.JSONDecodeError, ValueError, OSError):
                pass
            self.send_response(200)
            self._send_cors()
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()


def _start_mjpeg_server() -> str:
    class _ReuseServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        daemon_threads = True

    srv = _ReuseServer(("0.0.0.0", MJPEG_PORT), _MJPEGHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ip = _socket.gethostbyname(_socket.gethostname())
    except Exception:
        ip = "127.0.0.1"
    print(f"[MJPEG] 串流伺服器啟動：http://{ip}:{MJPEG_PORT}/stream")
    return ip


def _safe_mean(total_value, count):
    """回傳平均值；count<=0 時回傳 NaN。"""
    if count <= 0:
        return float("nan")
    return total_value / count


def _zone_key_from_label(label):
    """將命中標籤映射為統計區域鍵：BODY/FL/FR/HL/HR/NO_TARGET。"""
    if label == LICK_ZONE_CENTER:
        return "BODY"
    group = _limb_zone_group(label)
    if group in ("FL", "FR", "HL", "HR"):
        return group
    return LICK_ZONE_NO_TARGET


def _canonical_contact_label(label):
    """將任意子區域標籤歸一化為嚴格計數標籤：BODY/FL/FR/HL/HR/NO_TARGET。"""
    if label == LICK_ZONE_CENTER:
        return LICK_ZONE_CENTER
    group = _limb_zone_group(label)
    if group in ("FL", "FR", "HL", "HR"):
        return group
    return LICK_ZONE_NO_TARGET


def compute_limb_joint_targets(kpts, kpt_conf, body_len):
    """建立四肢圓形區域（僅保留腳尖 PAW）。"""
    if kpts is None or kpt_conf is None:
        return []
    eff_len = max(CONTACT_BODY_LEN_MIN_PX, min(CONTACT_BODY_LEN_MAX_PX, body_len))

    paw_radius = max(
        1e-6,
        eff_len * LIMB_PAW_CIRCLE_RADIUS_RATIO * LIMB_CONTACT_SCALE,
    )
    limb_targets = []
    for label, knee_idx, paw_idx in LIMB_SEGMENTS:
        short_label = LIMB_LABEL_MAP.get(label, label)
        paw_ok = kpt_conf[paw_idx] > LIMB_CONF_THRESHOLD

        if paw_ok:
            paw = np.asarray(kpts[paw_idx], dtype=np.float64)
            limb_targets.append(
                {
                    "zone_label": f"{short_label}_PAW",
                    "group": short_label,
                    "joint": "PAW",
                    "center": paw,
                    "radius": paw_radius,
                }
            )

    return limb_targets


def compute_limb_strip_targets(kpts, kpt_conf, body_len):
    """建立四肢 PAW-KNEE 長條區域（僅腳尖端貼齊圓邊界）。"""
    if kpts is None or kpt_conf is None:
        return []
    eff_len = max(CONTACT_BODY_LEN_MIN_PX, min(CONTACT_BODY_LEN_MAX_PX, body_len))

    half_width = max(
        1e-6,
        eff_len * LIMB_STRIP_HALF_WIDTH_RATIO * LIMB_CONTACT_SCALE,
    )
    edge_gap = max(
        0.0,
        eff_len * LIMB_STRIP_EDGE_GAP_RATIO * LIMB_CONTACT_SCALE,
    )
    paw_radius = max(
        1e-6,
        eff_len * LIMB_PAW_CIRCLE_RADIUS_RATIO * LIMB_CONTACT_SCALE,
    )
    strip_targets = []
    for label, knee_idx, paw_idx in LIMB_SEGMENTS:
        short_label = LIMB_LABEL_MAP.get(label, label)
        knee_ok = kpt_conf[knee_idx] > LIMB_CONF_THRESHOLD
        paw_ok = kpt_conf[paw_idx] > LIMB_CONF_THRESHOLD
        if not (knee_ok and paw_ok):
            continue

        knee = np.asarray(kpts[knee_idx], dtype=np.float64)
        paw = np.asarray(kpts[paw_idx], dtype=np.float64)
        seg = paw - knee
        seg_len = math.hypot(seg[0], seg[1])
        if seg_len < 1e-6:
            continue

        axis = seg / seg_len
        normal = np.array([-axis[1], axis[0]], dtype=np.float64)

        # 僅保留腳尖圓形：長條從膝/肘關節開始，於腳尖圓邊界前結束。
        strip_start = knee + axis * edge_gap
        strip_end = paw - axis * (paw_radius + edge_gap)
        _se = strip_end - strip_start
        strip_len = math.hypot(_se[0], _se[1])
        if strip_len < 1e-3:
            continue

        corners = _compute_strip_corners(strip_start, strip_end, half_width)
        if corners is None:
            continue

        strip_targets.append(
            {
                "zone_label": f"{short_label}_STRIP",
                "group": short_label,
                "p0": strip_start,
                "p1": strip_end,
                "axis": axis,
                "normal": normal,
                "length": strip_len,
                "half_width": half_width,
                "corners": corners,
                "mid": 0.5 * (strip_start + strip_end),
            }
        )

    return strip_targets


def infer_nose_lick_zone(target_geom, nearest_label, nearest_hit):
    """以鼻子落點推估 lick_zone；僅在 BODY_CENTER 時回傳正規化座標分數。"""
    if target_geom is None or not nearest_hit:
        return LICK_ZONE_NO_TARGET, float("nan"), float("nan"), False

    if nearest_label == LICK_ZONE_CENTER:
        nose_pt = np.asarray(target_geom["nose"], dtype=np.float64)
        body_center = np.asarray(target_geom["body_center"], dtype=np.float64)
        body_normal = np.asarray(target_geom["body_normal"], dtype=np.float64)
        body_axis_unit = np.asarray(target_geom["body_axis_unit"], dtype=np.float64)
        rx = max(float(target_geom["region_rx"]), 1e-9)
        ry = max(float(target_geom["region_ry"]), 1e-9)
        rel = nose_pt - body_center
        u_norm = float(np.clip(np.dot(rel, body_normal) / rx, -1.0, 1.0))
        v_norm = float(np.clip(np.dot(rel, body_axis_unit) / ry, -1.0, 1.0))
        return LICK_ZONE_CENTER, v_norm, u_norm, True

    if _limb_zone_group(nearest_label) in (LICK_ZONE_FL, LICK_ZONE_FR, LICK_ZONE_HL, LICK_ZONE_HR):
        return nearest_label, float("nan"), float("nan"), True

    return LICK_ZONE_NO_TARGET, float("nan"), float("nan"), False


def infer_nearest_nose_region(target_geom):
    """
    以鼻子梯形接觸判斷命中區域；若同時命中多區域，取最近區域。

    嚴格規則：四肢的腳尖圓區與 KNEE-PAW 長條區視為同一肢體實體。
    只要命中該肢體任一子區，最終標籤一律歸一為 FL/FR/HL/HR。
    """
    if target_geom is None:
        return LICK_ZONE_NO_TARGET, float("nan"), False

    nose_pt = target_geom.get("nose", None)
    if nose_pt is None:
        return LICK_ZONE_NO_TARGET, float("nan"), False

    nose_trap_pts = np.asarray(target_geom.get("nose_contact_trapezoid", []), dtype=np.float64)
    if nose_trap_pts.ndim != 2 or nose_trap_pts.shape[0] != 4:
        return LICK_ZONE_NO_TARGET, float("nan"), False

    candidates = []

    in_body = _polygon_contacts_oriented_ellipse(
        nose_trap_pts,
        target_geom["body_center"],
        np.asarray(target_geom["body_normal"], dtype=np.float64),
        np.asarray(target_geom["body_axis_unit"], dtype=np.float64),
        float(target_geom["region_rx"]),
        float(target_geom["region_ry"]),
    )
    if in_body:
        body_center = np.asarray(target_geom["body_center"], dtype=np.float64)
        body_axis_unit = np.asarray(target_geom["body_axis_unit"], dtype=np.float64)
        body_len = float(target_geom["body_len"])
        chest = body_center - 0.5 * body_len * body_axis_unit
        hip = body_center + 0.5 * body_len * body_axis_unit
        d_body = _distance_point_to_segment(nose_pt, chest, hip)
        candidates.append((d_body, LICK_ZONE_CENTER))

    limb_group_min_dist = {
        "FL": float("inf"),
        "FR": float("inf"),
        "HL": float("inf"),
        "HR": float("inf"),
    }

    for limb in target_geom.get("limb_targets", []):
        center = np.asarray(limb["center"], dtype=np.float64)
        radius = float(limb["radius"])
        inside_limb = _polygon_contacts_circle(
            nose_trap_pts,
            center,
            radius,
        )
        if not inside_limb:
            continue

        group = str(limb.get("group", ""))
        if group not in limb_group_min_dist:
            continue
        d_limb = math.hypot(nose_pt[0] - center[0], nose_pt[1] - center[1])
        if d_limb < limb_group_min_dist[group]:
            limb_group_min_dist[group] = d_limb

    for strip in target_geom.get("limb_strip_targets", []):
        strip_corners = strip.get("corners")
        if strip_corners is None or len(strip_corners) != 4:
            continue
        inside_strip = _polygons_intersect(nose_trap_pts, np.asarray(strip_corners, dtype=np.float64))
        if not inside_strip:
            continue

        group = str(strip.get("group", ""))
        if group not in limb_group_min_dist:
            continue
        d_strip = _distance_point_to_segment(nose_pt, strip["p0"], strip["p1"])
        d_strip = float(d_strip)
        if d_strip < limb_group_min_dist[group]:
            limb_group_min_dist[group] = d_strip

    for group, d_group in limb_group_min_dist.items():
        if math.isfinite(d_group):
            candidates.append((d_group, group))

    if not candidates:
        return LICK_ZONE_NO_TARGET, float("nan"), False

    d_min, label_min = min(candidates, key=lambda x: x[0])
    return label_min, float(d_min), True


def compute_head_body_target_geometry(kpts, kpt_conf):
    """建立頭部/身體幾何資料與區域，供可視化與鼻子命中判定。"""
    if kpts is None or kpt_conf is None:
        return None

    left_ok = kpt_conf[KP_LEFT_EAR] > EAR_CONF_THRESHOLD
    right_ok = kpt_conf[KP_RIGHT_EAR] > EAR_CONF_THRESHOLD
    nose_ok = kpt_conf[KP_NOSE] >= NOSE_CONF_THRESHOLD
    chest_ok = kpt_conf[KP_CHEST] > EAR_CONF_THRESHOLD
    hip_ok = kpt_conf[KP_HIP] > EAR_CONF_THRESHOLD
    if not (nose_ok and chest_ok and hip_ok):
        return None

    nose = np.asarray(kpts[KP_NOSE], dtype=np.float64)
    chest = np.asarray(kpts[KP_CHEST], dtype=np.float64)
    hip = np.asarray(kpts[KP_HIP], dtype=np.float64)

    if left_ok and right_ok:
        left_ear = np.asarray(kpts[KP_LEFT_EAR], dtype=np.float64)
        right_ear = np.asarray(kpts[KP_RIGHT_EAR], dtype=np.float64)
        ear_center = 0.5 * (left_ear + right_ear)
        head_vec = nose - ear_center
        head_norm = math.hypot(head_vec[0], head_vec[1])
        if head_norm > 1e-6:
            head_dir = head_vec / head_norm
        else:
            head_vec = np.zeros(2, dtype=np.float64)
            head_dir = np.zeros(2, dtype=np.float64)
            head_norm = 0.0
    else:
        # 雙耳不完整時仍保留身體/四肢區域判斷，避免鼻子命中邏輯中斷。
        ear_center = nose.copy()
        head_vec = np.zeros(2, dtype=np.float64)
        head_dir = np.zeros(2, dtype=np.float64)
        head_norm = 0.0

    body_axis = hip - chest
    body_len = math.hypot(body_axis[0], body_axis[1])
    if body_len < 1e-6:
        return None
    body_axis_unit = body_axis / body_len
    body_normal = np.array([-body_axis_unit[1], body_axis_unit[0]], dtype=np.float64)

    body_center = 0.5 * (chest + hip)
    region_rx = max(1e-6, 0.5 * BODY_REGION_ELLIPSE_WIDTH_RATIO * body_len)
    region_ry = max(1e-6, 0.5 * BODY_REGION_ELLIPSE_HEIGHT_RATIO * body_len)

    ray_len = max(HEAD_RAY_MIN_PX, HEAD_RAY_LENGTH_RATIO * body_len)
    ray_end = ear_center + head_dir * ray_len

    limb_targets = compute_limb_joint_targets(kpts, kpt_conf, body_len)
    limb_strip_targets = compute_limb_strip_targets(kpts, kpt_conf, body_len)

    # 梯形尺寸以等效身體長度計算，clamp 到 [MIN, MAX] 以縮小遠近情境差距
    eff_len = max(CONTACT_BODY_LEN_MIN_PX, min(CONTACT_BODY_LEN_MAX_PX, body_len))
    pixels_per_cm = max(eff_len / max(CAT_BODY_LENGTH_CM, 1e-6), 1e-6)

    nose_trap_height = max(
        1e-6,
        NOSE_CONTACT_TRAPEZOID_THICKNESS_CM
        * NOSE_CONTACT_TRAPEZOID_THICKNESS_SCALE
        * pixels_per_cm,
    )
    trap_top_half = max(
        1e-6,
        0.5 * NOSE_CONTACT_TRAPEZOID_TOP_WIDTH_RATIO * NOSE_CONTACT_TRAPEZOID_WIDTH_SCALE * eff_len,
    )
    trap_bottom_half = max(
        1e-6,
        0.5 * NOSE_CONTACT_TRAPEZOID_BOTTOM_WIDTH_RATIO * NOSE_CONTACT_TRAPEZOID_WIDTH_SCALE * eff_len,
    )

    # 梯形橫軸（trap_perp）方向取決於左右耳信心是否均 > EAR_CONF_THRESHOLD：
    #   兩耳皆有效（left_ok and right_ok）→ 平行耳中線（right_ear - left_ear 方向）
    #   否則                               → fallback 身體法向量（body_normal）
    if left_ok and right_ok:
        ear_line = right_ear - left_ear
        ear_line_norm = math.hypot(ear_line[0], ear_line[1])
        if ear_line_norm > 1e-9:
            trap_perp = ear_line / ear_line_norm
        else:
            trap_perp = np.array([1.0, 0.0], dtype=np.float64)
    else:
        # 雙耳不完整時，回退到身體法向量作為梯形橫軸。
        trap_perp = body_normal.copy()
        trap_perp_norm = math.hypot(trap_perp[0], trap_perp[1])
        if trap_perp_norm < 1e-9:
            trap_perp = np.array([1.0, 0.0], dtype=np.float64)
        else:
            trap_perp = trap_perp / trap_perp_norm

    trap_dir = np.array([-trap_perp[1], trap_perp[0]], dtype=np.float64)
    # Prefer trap_dir pointing toward the body center
    if np.dot(trap_dir, body_center - nose) < 0.0:
        trap_dir = -trap_dir

    # Compute bottom center and ensure trapezoid is below the nose in image coords
    trap_center_bottom = nose + trap_dir * nose_trap_height
    # Image y increases downward; enforce bottom center y > nose y so trapezoid is drawn below nose
    if not np.isfinite(trap_center_bottom[1]) or trap_center_bottom[1] <= nose[1]:
        trap_dir = -trap_dir
        trap_center_bottom = nose + trap_dir * nose_trap_height
    nose_contact_trapezoid = np.asarray(
        [
            nose - trap_perp * trap_top_half,
            nose + trap_perp * trap_top_half,
            trap_center_bottom + trap_perp * trap_bottom_half,
            trap_center_bottom - trap_perp * trap_bottom_half,
        ],
        dtype=np.float64,
    )

    return {
        "ear_center": ear_center,
        "nose": nose,
        "head_vec": head_vec,
        "head_dir": head_dir,
        "head_norm": head_norm,
        "ray_end": ray_end,
        "body_center": body_center,
        "body_normal": body_normal,
        "body_len": body_len,
        "body_axis_unit": body_axis_unit,
        "region_rx": region_rx,
        "region_ry": region_ry,
        "limb_targets": limb_targets,
        "limb_strip_targets": limb_strip_targets,
        "nose_contact_trapezoid": nose_contact_trapezoid,
        "nose_contact_thickness_px": nose_trap_height,
    }


def draw_styled_skeleton(frame, kpts, kpt_conf, bbox, bbox_conf, sx, sy, ov, conf_thresh=DRAW_KP_CONF_THRESHOLD):
    """依 test2.py 風格繪製骨架、關鍵點與 bbox。"""
    line_w = max(1, int(2 * ov))
    r_outer = max(3, int(4 * ov))
    r_inner = max(2, int(3 * ov))

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        bx1 = int(x1 * sx)
        by1 = int(y1 * sy)
        bx2 = int(x2 * sx)
        by2 = int(y2 * sy)

        # 完整邊框：先畫黑色外層，再畫亮色內層
        outer_w = max(2, int(4 * ov))
        inner_w = max(1, int(2 * ov))
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), outer_w, cv2.LINE_AA)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), inner_w, cv2.LINE_AA)

        # 左上角顯示 bbox confidence
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

            # 舊式數值顯示：無背景框，僅文字與描邊
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), text_shadow, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), th, cv2.LINE_AA)

    for ei, (a, b) in enumerate(_SKELETON_EDGES):
        if a >= len(kpts) or b >= len(kpts):
            continue
        # 線段繪製條件：兩端關鍵點都需高於顯示門檻
        if float(kpt_conf[a]) > conf_thresh and float(kpt_conf[b]) > conf_thresh:
            pa = (int(kpts[a][0] * sx), int(kpts[a][1] * sy))
            pb = (int(kpts[b][0] * sx), int(kpts[b][1] * sy))
            col = _EDGE_COLORS[ei] if ei < len(_EDGE_COLORS) else (180, 180, 180)
            cv2.line(frame, pa, pb, col, line_w, cv2.LINE_AA)

    for i in range(min(17, len(kpts))):
        conf_val = float(kpt_conf[i])
        # 圓點繪製條件：該點信心值需高於顯示門檻
        if conf_val <= conf_thresh:
            continue
        cx, cy = int(kpts[i][0] * sx), int(kpts[i][1] * sy)
        col = _KP_COLORS[i] if i < len(_KP_COLORS) else (200, 200, 200)
        cv2.circle(frame, (cx, cy), r_outer, (0, 0, 0), -1)
        cv2.circle(frame, (cx, cy), r_inner, col, -1)


def _is_url(path_str):
    """檢查是否為 IP 攝影機流 URL（RTSP, HTTP, 等）"""
    path_lower = str(path_str).lower()
    return path_lower.startswith(('rtsp://', 'rtsps://', 'http://', 'https://', 'rtmp://'))


def collect_video_paths(max_videos=MAX_VIDEOS):
    """由 VIDEO_LIST 收集影片路徑或 IP 攝影機 URL（最多 max_videos 支）。
    VIDEO_LIST 接受單一影片檔案路徑或 IP 攝影機 URL；資料夾請改填在 VIDEO_PATH。
    """
    videos = []
    seen = set()

    def _add(f: Path):
        fp = str(f.resolve())
        if fp not in seen and f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
            seen.add(fp)
            videos.append(f)

    for entry in VIDEO_LIST:
        # 檢查是否為 URL（IP 攝影機）
        if _is_url(entry):
            if entry not in seen:
                seen.add(entry)
                videos.append(entry)  # 直接存儲 URL 字符串
                print(f"✓ 已加入 IP 攝影機: {entry}")
        else:
            p = Path(entry)
            if not p.exists():
                print(f"⚠️  找不到路徑，略過: {p}")
                continue
            if p.is_file():
                _add(p)
            elif p.is_dir():
                print(f"⚠️  VIDEO_LIST 僅接受檔案，資料夾請改填 VIDEO_PATH，已略過: {p}")
        if len(videos) >= max_videos:
            return videos
    return videos


def collect_video_paths_from_folder(folder_path, max_videos=MAX_VIDEOS, existing_paths=None):
    """由 VIDEO_PATH 指定資料夾遞迴收集影片，或直接使用 IP 攝影機 URL（最多補到 max_videos 支）。"""
    videos = []
    seen = set(existing_paths or [])

    # 檢查是否為 IP 攝影機 URL
    # support numeric camera index (0, 1, ...) passed as int or numeric string
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
            print(f"✓ 已加入 IP 攝影機: {folder_path}")
        return videos

    p = Path(folder_path)
    if not p.exists():
        print(f"⚠️  VIDEO_PATH 不存在: {p}")
        return videos

    if p.is_file():
        fp = str(p.resolve())
        if p.suffix.lower() in VIDEO_EXTENSIONS and fp not in seen:
            videos.append(p)
        else:
            print(f"⚠️  VIDEO_PATH 是檔案但非支援格式或已重複: {p}")
        return videos

    try:
        all_files = sorted(p.rglob("*"))
    except Exception as e:
        print(f"⚠️ 無法掃描部分子資料夾，已略過: {e}")
        all_files = []
    for f in all_files:
        if not f.is_file() or f.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        fp = str(f.resolve())
        if fp in seen:
            continue
        seen.add(fp)
        videos.append(f)
        if len(videos) >= max_videos:
            break
    return videos


def main():
    video_paths = collect_video_paths(MAX_VIDEOS)
    existing = set()
    for v in video_paths:
        try:
            if isinstance(v, Path):
                existing.add(str(v.resolve()))
            else:
                existing.add(str(v))
        except Exception:
            existing.add(str(v))
    remaining = max(0, MAX_VIDEOS - len(video_paths))
    if remaining > 0:
        video_paths.extend(collect_video_paths_from_folder(VIDEO_PATH, remaining, existing_paths=existing))

    if not video_paths:
        print("❌ 找不到可用影片（VIDEO_LIST 檔案清單與 VIDEO_PATH 資料夾都無效）")
        return

    print("=" * 60)
    print("左右耳距離監測（多影片）")
    print("=" * 60)
    print(f"共載入 {len(video_paths)} 支影片：")
    for i, vp in enumerate(video_paths, start=1):
        print(f"  [{i}] {vp}")
    print(f"輸出 CSV: {OUTPUT_CSV_PATH}")
    print("控制: q=離開, space=暫停/播放, i=資訊顯示, m=切換顯示模式(僅骨架/完整), +=放大, -=縮小, 2=下一部, 1=上一部, r=重置本片")

    try:
        detector = KeypointDetector(
            YOLO_MODEL_PATH,
            device=INFERENCE_DEVICE,
            imgsz=YOLO_IMGSZ,
            conf_thres=YOLO_CONF_THRESHOLD,
        )
    except Exception as e:
        print(f"❌ 無法載入 YOLO 模型（路徑：{YOLO_MODEL_PATH}）：{e}")
        return

    output_csv = Path(OUTPUT_CSV_PATH)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = output_csv.with_name(output_csv.stem + "_summary.csv")

    rows = []
    summary_rows = []
    show_overlay_info = True
    is_paused = False
    window_scale = 1.0
    current_video_idx = 0
    stop_all = False
    switch_delta = 0
    state_size_cache = {}
    target_size_cache = {}
    ui_mode = UI_MODE_FULL

    raw_frame_idx = 0
    processed_frame_idx = 0
    video_processed_total = 0
    playback_pass = 0
    ema_kpts = None
    state_history = deque(maxlen=STATE_SMOOTH_WINDOW)
    target_entry_count = 0
    limb_entry_count_fl = 0
    limb_entry_count_fr = 0
    limb_entry_count_hl = 0
    limb_entry_count_hr = 0
    prev_nose_region_label = LICK_ZONE_NO_TARGET
    lick_time_body_sec = 0.0
    lick_time_fl_sec = 0.0
    lick_time_fr_sec = 0.0
    lick_time_hl_sec = 0.0
    lick_time_hr_sec = 0.0
    bout_count_body = 0
    bout_count_fl = 0
    bout_count_fr = 0
    bout_count_hl = 0
    bout_count_hr = 0
    bout_time_body_sec = 0.0
    bout_time_fl_sec = 0.0
    bout_time_fr_sec = 0.0
    bout_time_hl_sec = 0.0
    bout_time_hr_sec = 0.0
    active_bout_zone = LICK_ZONE_NO_TARGET
    active_bout_sec = 0.0

    if DISPLAY_SIZE is not None:
        base_render_w, base_render_h = DISPLAY_SIZE
    else:
        base_render_w, base_render_h = 1080, 720
    render_w, render_h = base_render_w, base_render_h
    _ov = max(0.6, render_h / 720.0)
    base_win_w, base_win_h = render_w, render_h

    def _apply_window_scale():
        if not DISPLAY_WINDOW or STREAM_MODE != 1:
            return
        w = max(320, int(base_win_w * window_scale))
        h = max(240, int(base_win_h * window_scale))
        cv2.resizeWindow(WINDOW_NAME, w, h)

    def _reset_video_state(cap_obj=None, seek_start=False):
        nonlocal raw_frame_idx, processed_frame_idx, ema_kpts
        nonlocal target_entry_count, limb_entry_count_fl, limb_entry_count_fr, limb_entry_count_hl, limb_entry_count_hr
        nonlocal prev_nose_region_label, is_paused
        nonlocal lick_time_body_sec, lick_time_fl_sec, lick_time_fr_sec, lick_time_hl_sec, lick_time_hr_sec
        nonlocal bout_count_body, bout_count_fl, bout_count_fr, bout_count_hl, bout_count_hr
        nonlocal bout_time_body_sec, bout_time_fl_sec, bout_time_fr_sec, bout_time_hl_sec, bout_time_hr_sec
        nonlocal active_bout_zone, active_bout_sec

        if seek_start and cap_obj is not None:
            cap_obj.set(cv2.CAP_PROP_POS_FRAMES, 0)

        raw_frame_idx = 0
        processed_frame_idx = 0
        ema_kpts = None
        state_history.clear()
        target_entry_count = 0
        limb_entry_count_fl = 0
        limb_entry_count_fr = 0
        limb_entry_count_hl = 0
        limb_entry_count_hr = 0
        prev_nose_region_label = LICK_ZONE_NO_TARGET
        lick_time_body_sec = 0.0
        lick_time_fl_sec = 0.0
        lick_time_fr_sec = 0.0
        lick_time_hl_sec = 0.0
        lick_time_hr_sec = 0.0
        bout_count_body = 0
        bout_count_fl = 0
        bout_count_fr = 0
        bout_count_hl = 0
        bout_count_hr = 0
        bout_time_body_sec = 0.0
        bout_time_fl_sec = 0.0
        bout_time_fr_sec = 0.0
        bout_time_hl_sec = 0.0
        bout_time_hr_sec = 0.0
        active_bout_zone = LICK_ZONE_NO_TARGET
        active_bout_sec = 0.0
        is_paused = False

    def _increase_contact_counter(label):
        nonlocal target_entry_count, limb_entry_count_fl, limb_entry_count_fr, limb_entry_count_hl, limb_entry_count_hr
        group = _limb_zone_group(label)
        if label == LICK_ZONE_CENTER:
            target_entry_count += 1
        elif group == "FL":
            limb_entry_count_fl += 1
        elif group == "FR":
            limb_entry_count_fr += 1
        elif group == "HL":
            limb_entry_count_hl += 1
        elif group == "HR":
            limb_entry_count_hr += 1

    def _accumulate_lick_time(label, dt_sec):
        nonlocal lick_time_body_sec, lick_time_fl_sec, lick_time_fr_sec, lick_time_hl_sec, lick_time_hr_sec
        if dt_sec <= 0.0:
            return
        group = _limb_zone_group(label)
        if label == LICK_ZONE_CENTER:
            lick_time_body_sec += dt_sec
        elif group == "FL":
            lick_time_fl_sec += dt_sec
        elif group == "FR":
            lick_time_fr_sec += dt_sec
        elif group == "HL":
            lick_time_hl_sec += dt_sec
        elif group == "HR":
            lick_time_hr_sec += dt_sec

    def _close_active_bout():
        nonlocal bout_count_body, bout_count_fl, bout_count_fr, bout_count_hl, bout_count_hr
        nonlocal bout_time_body_sec, bout_time_fl_sec, bout_time_fr_sec, bout_time_hl_sec, bout_time_hr_sec
        nonlocal active_bout_zone, active_bout_sec

        if active_bout_zone == "BODY" and active_bout_sec > 0.0:
            bout_count_body += 1
            bout_time_body_sec += active_bout_sec
        elif active_bout_zone == "FL" and active_bout_sec > 0.0:
            bout_count_fl += 1
            bout_time_fl_sec += active_bout_sec
        elif active_bout_zone == "FR" and active_bout_sec > 0.0:
            bout_count_fr += 1
            bout_time_fr_sec += active_bout_sec
        elif active_bout_zone == "HL" and active_bout_sec > 0.0:
            bout_count_hl += 1
            bout_time_hl_sec += active_bout_sec
        elif active_bout_zone == "HR" and active_bout_sec > 0.0:
            bout_count_hr += 1
            bout_time_hr_sec += active_bout_sec

        active_bout_zone = LICK_ZONE_NO_TARGET
        active_bout_sec = 0.0

    def _handle_key(key, cap_obj, in_pause_loop=False):
        nonlocal show_overlay_info, window_scale, switch_delta, stop_all, is_paused, ui_mode

        if key == ord("q") or key == 3:  # 3 = Ctrl+C 被 OpenCV 視窗攔截
            print("\n使用者中斷：q / Ctrl+C")
            stop_all = True
            is_paused = False
            return "break"
        if key == ord("i"):
            show_overlay_info = not show_overlay_info
            print(f"\n資訊面板: {'顯示' if show_overlay_info else '隱藏'}")
            return "handled"
        if key == ord("m"):
            ui_mode = UI_MODE_DET_ONLY if ui_mode == UI_MODE_FULL else UI_MODE_FULL
            mode_text = "僅 YOLO 偵測框與骨架" if ui_mode == UI_MODE_DET_ONLY else "完整功能"
            print(f"\n顯示模式: {mode_text}")
            return "handled"
        if key == ord("+") or key == ord("="):
            window_scale = min(WINDOW_SCALE_MAX, window_scale + WINDOW_SCALE_STEP)
            _apply_window_scale()
            return "handled"
        if key == ord("-") or key == ord("_"):
            window_scale = max(WINDOW_SCALE_MIN, window_scale - WINDOW_SCALE_STEP)
            _apply_window_scale()
            return "handled"
        if key == ord("2"):
            switch_delta = 1
            is_paused = False
            return "break"
        if key == ord("1"):
            switch_delta = -1
            is_paused = False
            return "break"
        if key == ord("r"):
            _reset_video_state(cap_obj=cap_obj, seek_start=True)
            return "handled"
        if key == ord(" ") and not in_pause_loop:
            is_paused = True
            return "enter_pause"
        if key == ord(" ") and in_pause_loop:
            is_paused = False
            return "handled"

        return "noop"

    if DISPLAY_WINDOW and STREAM_MODE == 1:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        _apply_window_scale()

    if STREAM_MODE == 2:
        _mjpeg_ip = _start_mjpeg_server()
        if NODERED_ONLINE_URL and _HAS_REQUESTS:
            try:
                _requests.post(NODERED_ONLINE_URL, json={"ip": _mjpeg_ip}, timeout=2)
            except Exception:
                pass

    while not stop_all:
        video_path = video_paths[current_video_idx]
        is_ip_stream = _is_url(str(video_path))

        # if video_path is an int (camera index), pass it directly to VideoCapture
        if isinstance(video_path, int):
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(str(video_path))
        
        # 為 IP 攝影機流設置優化參數（減少延遲和緩衝）
        if is_ip_stream:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 減小緩衝大小
            cap.set(cv2.CAP_PROP_FPS, 30)  # 設置期望的 FPS
        
        if not cap.isOpened():
            print(f"❌ 無法開啟影片，跳過: {video_path}")
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
        # IP 流通常無法獲得總幀數，這時會回傳 0 或 -1
        if is_ip_stream and total_frames <= 0:
            total_frames = 0  # 標記為實時流
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_diag = math.hypot(max(width, 1), max(height, 1))

        # 依影片原始長寬比計算顯示尺寸，避免直式影片被拉伸。
        if DISPLAY_SIZE is not None and width > 0 and height > 0:
            scale = min(base_render_w / float(width), base_render_h / float(height))
            scale = max(scale, 1e-6)
            render_w = max(1, int(round(width * scale)))
            render_h = max(1, int(round(height * scale)))
        else:
            render_w = base_render_w
            render_h = base_render_h
        _ov = max(0.6, render_h / 720.0)
        base_win_w, base_win_h = render_w, render_h
        _apply_window_scale()

        print("-" * 60)
        if is_ip_stream:
            print(f"目前影片 [{current_video_idx + 1}] [IP 攝影機] {video_path}")
        else:
            # video_path may be a Path, a string filepath, or an int camera index
            if isinstance(video_path, int):
                print(f"目前影片 [{current_video_idx + 1}] [Camera {video_path}]")
            else:
                try:
                    print(f"目前影片 [{current_video_idx + 1}] {video_path.name}")
                except Exception:
                    print(f"目前影片 [{current_video_idx + 1}] {video_path}")
        
        if total_frames > 0:
            print(f"解析度: {width}x{height}, source_fps={source_fps:.2f}, model_fps={model_input_fps:.2f}, total={total_frames}")
        else:
            # IP 流顯示為實時流
            stream_type = "實時流" if is_ip_stream else "未知"
            print(f"解析度: {width}x{height}, source_fps={source_fps:.2f}, model_fps={model_input_fps:.2f}, 類型={stream_type}")

        switch_delta = 0
        playback_pass = 0
        video_processed_total = 0
        _reset_video_state()

        while True:
            # 跳幀優化：grab() 只移動讀取位置不解碼，比 read() 快數倍
            if frame_step > 1 and (raw_frame_idx % frame_step != 0):
                if cap.grab():
                    raw_frame_idx += 1
                    continue
                if LOOP_PLAYBACK:
                    playback_pass += 1
                    _reset_video_state(cap_obj=cap, seek_start=True)
                    continue
                break

            ret, frame = cap.read()
            if not ret:
                if LOOP_PLAYBACK:
                    playback_pass += 1
                    _reset_video_state(cap_obj=cap, seek_start=True)
                    continue
                break

            raw_frame_idx += 1

            processed_frame_idx += 1
            video_processed_total += 1
            time_sec = raw_frame_idx / source_fps if source_fps > 0 else 0.0

            kpts, kpt_conf, bbox, bbox_conf = detector.detect(frame)
            dist_px = float("nan")
            dist_norm = float("nan")
            valid = 0
            nose_ok = False
            nose_conf = 0.0
            head_ear_angle_deg = float("nan")
            gaze_forward_norm = float("nan")
            gaze_lateral_norm = float("nan")
            gaze_angle_deg = float("nan")
            face_state_cat = STATE_UNKNOWN
            lxy = (float("nan"), float("nan"))
            rxy = (float("nan"), float("nan"))
            nxy = (float("nan"), float("nan"))
            body_scale = float("nan")
            body_scale_norm = float("nan")
            body_ear_ratio = float("nan")
            front_view_guard = False
            front_view_guard_reason = ""

            if _NEED_FRAME and DISPLAY_SIZE is not None:
                interp = cv2.INTER_LINEAR if (render_w > width or render_h > height) else cv2.INTER_AREA
                display = cv2.resize(frame, (render_w, render_h), interpolation=interp)
                sx = render_w / max(width, 1)
                sy = render_h / max(height, 1)
            else:
                display = frame
                sx = 1.0
                sy = 1.0

            full_ui_mode = ui_mode == UI_MODE_FULL
            draw_ui = _NEED_FRAME and full_ui_mode
            draw_overlay_info = draw_ui and show_overlay_info
            heavy_fx = _HEAVY_FX

            if kpts is not None and kpt_conf is not None:
                if _EMA_BYPASS:
                    ema_kpts = kpts
                else:
                    if ema_kpts is None:
                        ema_kpts = kpts.copy()
                    else:
                        ema_kpts = EMA_ALPHA * kpts + (1.0 - EMA_ALPHA) * ema_kpts
                    kpts = ema_kpts

                if _NEED_FRAME:
                    draw_styled_skeleton(display, kpts, kpt_conf, bbox, bbox_conf, sx, sy, _ov)

                left_ok = kpt_conf[KP_LEFT_EAR] > EAR_CONF_THRESHOLD
                right_ok = kpt_conf[KP_RIGHT_EAR] > EAR_CONF_THRESHOLD
                nose_conf = float(kpt_conf[KP_NOSE])
                nose_ok = nose_conf >= NOSE_CONF_THRESHOLD

                if left_ok:
                    lxy = (float(kpts[KP_LEFT_EAR][0]), float(kpts[KP_LEFT_EAR][1]))
                if right_ok:
                    rxy = (float(kpts[KP_RIGHT_EAR][0]), float(kpts[KP_RIGHT_EAR][1]))
                if nose_ok:
                    nxy = (float(kpts[KP_NOSE][0]), float(kpts[KP_NOSE][1]))

                if left_ok and right_ok and nose_ok:
                    _kf = kpts.astype(np.float64, copy=False)
                    nose_pt_np  = _kf[KP_NOSE]
                    left_pt_np  = _kf[KP_LEFT_EAR]
                    right_pt_np = _kf[KP_RIGHT_EAR]
                    v_nose_left = left_pt_np - nose_pt_np
                    v_nose_right = right_pt_np - nose_pt_np
                    head_ear_angle_deg = _angle_between_vectors_deg(v_nose_left, v_nose_right)

                # 耳距有效條件：左右耳都高於 EAR_CONF_THRESHOLD 才計算距離
                if left_ok and right_ok:
                    dx = lxy[0] - rxy[0]
                    dy = lxy[1] - rxy[1]
                    dist_px = math.hypot(dx, dy)
                    valid = 1
                    if draw_ui:
                        cv2.line(
                            display,
                            (int(lxy[0] * sx), int(lxy[1] * sy)),
                            (int(rxy[0] * sx), int(rxy[1] * sy)),
                            (0, 200, 0),
                            max(1, int(2 * _ov)),
                            cv2.LINE_AA,
                        )

                # 正規化有效條件：胸與臀都高於 EAR_CONF_THRESHOLD 才計算 body scale
                chest_ok = kpt_conf[KP_CHEST] > EAR_CONF_THRESHOLD
                hip_ok = kpt_conf[KP_HIP] > EAR_CONF_THRESHOLD
                if chest_ok and hip_ok:
                    body_scale = math.hypot(
                        kpts[KP_CHEST][0] - kpts[KP_HIP][0],
                        kpts[KP_CHEST][1] - kpts[KP_HIP][1],
                    )
                    if frame_diag > 1e-6:
                        body_scale_norm = body_scale / frame_diag

                    if FRONT_VIEW_GUARD_ENABLED and left_ok and right_ok and np.isfinite(dist_px) and body_scale > 1e-6:
                        body_ear_ratio = body_scale / max(dist_px, 1e-6)
                        if np.isfinite(body_scale_norm) and body_scale_norm <= FRONT_VIEW_BODY_SCALE_NORM_MAX:
                            front_view_guard = True
                            front_view_guard_reason = "BODY_SCALE_NORM_SMALL"
                        elif body_ear_ratio <= FRONT_VIEW_BODY_EAR_RATIO_MAX:
                            front_view_guard = True
                            front_view_guard_reason = "BODY_EAR_RATIO_SMALL"
                if valid and chest_ok and hip_ok:
                    if body_scale > 1e-6:
                        dist_norm = dist_px / body_scale
            else:
                ema_kpts = None
                state_history.append(STATE_NO_CAT)

            # 僅在 DET_ONLY 模式下才需要此副本（用於暫停畫面）
            display_det_only = (
                display.copy()
                if (DISPLAY_WINDOW and STREAM_MODE == 1 and ui_mode == UI_MODE_DET_ONLY)
                else None
            )

            valid_norm = valid and np.isfinite(dist_norm)

            target_geom = None
            if not front_view_guard and full_ui_mode:
                target_geom = compute_head_body_target_geometry(kpts, kpt_conf)

            if kpts is not None and kpt_conf is not None:
                if front_view_guard:
                    if BACK_VIEW_REQUIRE_LOW_NOSE_CONF and nose_conf <= BACK_CAMERA_NOSE_CONF_MAX:
                        state_now = STATE_BACK
                        face_state_cat = STATE_BACK
                    else:
                        state_now = STATE_FRONT_VIEW
                        face_state_cat = STATE_FRONT_VIEW
                    state_history.append(state_now)
                    state_smoothed = state_now
                    state_stability = 1.0
                else:
                    face_state_cat, gaze_forward_norm, gaze_lateral_norm, gaze_angle_deg = infer_face_state_cat_centric_metrics(
                        target_geom, nose_ok
                    )
                    state_now, rule_applied = infer_face_state_user_rules(
                        head_ear_angle_deg=head_ear_angle_deg,
                        dist_norm=dist_norm,
                        dist_px=dist_px,
                        nose_conf=nose_conf,
                    )
                    if state_now == STATE_UNKNOWN:
                        # 低鼻信心規則啟用時，避免以高鼻信心的 cat-centric 結果落入 BACK。
                        if BACK_VIEW_REQUIRE_LOW_NOSE_CONF and face_state_cat == STATE_BACK and nose_conf > BACK_CAMERA_NOSE_CONF_MAX:
                            state_now = STATE_UNKNOWN
                        else:
                            state_now = face_state_cat
                        rule_applied = False
                    if state_now == STATE_UNKNOWN and ENABLE_LEGACY_SCREEN_FALLBACK:
                        # 後備：資料不足時回到舊版畫面視角判斷。
                        state_now = infer_face_state(valid_norm, dist_norm, nose_conf, lxy, rxy, nxy, nose_ok)
                    state_history.append(state_now)
                    # 命中使用者 FRONT/BACK 規則時直接採信，避免多數決延遲。
                    if rule_applied and state_now in (
                        STATE_FRONT,
                        STATE_FRONT_LEFT,
                        STATE_FRONT_RIGHT,
                        STATE_BACK,
                    ):
                        state_smoothed = state_now
                        state_stability = 1.0
                    else:
                        state_smoothed, state_stability = smooth_state(state_history)
            else:
                state_now = STATE_NO_CAT
                state_smoothed, state_stability = smooth_state(state_history)

            status_color = (235, 235, 235)
            if state_smoothed == STATE_FRONT:
                status_color = (40, 230, 40)
            elif state_smoothed == STATE_BACK:
                status_color = (30, 140, 255)
            elif state_smoothed == STATE_FRONT_VIEW:
                status_color = (255, 120, 60)
            elif state_smoothed in (
                STATE_FRONT_LEFT,
                STATE_FRONT_RIGHT,
            ):
                status_color = (0, 220, 220)

            state_label = state_smoothed
            state_fs = 0.95 * _ov
            state_th_main = max(2, int(3 * _ov))
            state_th_shadow = state_th_main + 2
            box_x2 = 0
            box_y2 = 0
            if draw_overlay_info:
                state_cache_key = (state_label, state_fs, state_th_main)
                state_size = state_size_cache.get(state_cache_key)
                if state_size is None:
                    state_size = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, state_fs, state_th_main)
                    state_size_cache[state_cache_key] = state_size
                (tw, th), _ = state_size
                pad = int(12 * _ov)
                box_h = th + int(18 * _ov)
                box_w = tw + int(24 * _ov)
                box_x2 = render_w - pad
                box_y1 = pad
                box_x1 = max(0, box_x2 - box_w)
                box_y2 = box_y1 + box_h
            state_text_pos = None
            target_status_text = None
            target_status_pos = None
            target_status_style = None
            ui_overlay = display.copy() if draw_overlay_info else None
            if draw_overlay_info:
                cv2.rectangle(ui_overlay, (box_x1, box_y1), (box_x2, box_y2), (18, 18, 18), -1)
                tx = box_x1 + int(12 * _ov)
                ty = box_y1 + th + int(6 * _ov)
                state_text_pos = (tx, ty)

            info_x = int(12 * _ov)
            line_y0 = int(34 * _ov)
            line_step = int(24 * _ov)
            info_fs = 0.50 * _ov
            th_main = max(1, int(2 * _ov))
            th_shadow = max(2, int(3 * _ov))

            norm_value = f"{dist_norm:.5f}" if valid_norm else "N/A"
            head_ear_angle_text = "NEA:N/A"
            lick_zone = LICK_ZONE_NO_TARGET
            lick_axis_score = float("nan")
            lick_lateral_score = float("nan")
            lick_score_text = "LZ:None"
            nearest_target_label = LICK_ZONE_NO_TARGET
            nearest_target_t = float("nan")
            nearest_target_hit = False
            ray_end = None
            ray_nx = float("nan")
            ray_ny = float("nan")
            limb_hit_labels = []
            limb_hit_fl = 0
            limb_hit_fr = 0
            limb_hit_hl = 0
            limb_hit_hr = 0

            # ===== 鼻子區域命中（唯一觸發來源）+ 可視化 =====
            if full_ui_mode and target_geom is not None:
                ear_c = target_geom["ear_center"]
                nose_p = target_geom["nose"]
                nose_trap_pts = np.asarray(target_geom.get("nose_contact_trapezoid", []), dtype=np.float64)
                trap_draw = None
                trap_pulse = 0.0
                ray_end = target_geom.get("ray_end", None)
                if ray_end is not None:
                    ray_nx = float(np.clip(ray_end[0] / max(float(width - 1), 1.0), 0.0, 1.0))
                    ray_ny = float(np.clip(ray_end[1] / max(float(height - 1), 1.0), 0.0, 1.0))
                body_c = target_geom["body_center"]
                region_rx = float(target_geom["region_rx"])
                region_ry = float(target_geom["region_ry"])

                # 目標判定以鼻子梯形與各區域交集為準；若同時命中多區域，取最近區域。
                nearest_target_label, nearest_target_t, nearest_target_hit = infer_nearest_nose_region(target_geom)
                lick_zone, lick_axis_score, lick_lateral_score, _ = infer_nose_lick_zone(
                    target_geom,
                    nearest_target_label,
                    nearest_target_hit,
                )
                if lick_zone == LICK_ZONE_CENTER and np.isfinite(lick_axis_score):
                    lick_score_text = f"LZ:{lick_zone} V:{lick_axis_score:+.2f} U:{lick_lateral_score:+.2f}"
                elif nearest_target_hit:
                    lick_score_text = f"LZ:{lick_zone}"

                if draw_ui:
                    nose_pt = (int(nose_p[0] * sx), int(nose_p[1] * sy))
                    # 方向向量僅作視覺輔助，命中仍只由鼻子梯形交集決定
                    if np.isfinite(head_ear_angle_deg):
                        head_ear_angle_text = f"NEA:{head_ear_angle_deg:.1f}deg"

                    # 保留鼻子方向箭頭視覺（不參與命中邏輯）
                    if ray_end is not None:
                        ear_pt = (int(ear_c[0] * sx), int(ear_c[1] * sy))
                        ray_pt = (int(ray_end[0] * sx), int(ray_end[1] * sy))
                        cv2.arrowedLine(display, ear_pt, nose_pt, (255, 230, 0), max(1, int(2 * _ov)), cv2.LINE_AA, tipLength=0.22)
                        cv2.arrowedLine(display, ear_pt, ray_pt, (255, 170, 0), max(1, int(1 * _ov)), cv2.LINE_AA, tipLength=0.08)

                    # 鼻子接觸梯形：右轉 90 度的接觸區域
                    if nose_trap_pts.ndim == 2 and nose_trap_pts.shape[0] == 4:
                        trap_draw = np.array([[int(p[0] * sx), int(p[1] * sy)] for p in nose_trap_pts], dtype=np.int32)
                        if heavy_fx:
                            trap_pulse = 0.5 + 0.5 * np.sin(processed_frame_idx * 0.28)

                    if np.isfinite(head_ear_angle_deg):
                        nea_text = f"{head_ear_angle_deg:.1f} deg"
                        nea_x = int(nose_p[0] * sx + 8 * _ov)
                        nea_y = int(nose_p[1] * sy - 10 * _ov)
                        if heavy_fx:
                            cv2.putText(display, nea_text, (nea_x, nea_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * _ov, (0, 0, 0), max(2, int(2 * _ov)), cv2.LINE_AA)
                        cv2.putText(display, nea_text, (nea_x, nea_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * _ov, (120, 255, 255), max(1, int(1 * _ov)), cv2.LINE_AA)

                    # 身體中心點
                    body_pt = (int(body_c[0] * sx), int(body_c[1] * sy))
                    cv2.circle(display, body_pt, max(3, int(4 * _ov)), (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(display, body_pt, max(2, int(2 * _ov)), (0, 180, 255), -1, cv2.LINE_AA)

                    # 局部身體區域（半透明橢圓）
                    region_center = body_pt
                    axes = (max(2, int(region_rx * sx)), max(2, int(region_ry * sy)))
                    axis = target_geom["body_axis_unit"]
                    angle_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))

                    region_overlay = display.copy()
                    body_is_nearest = nearest_target_hit and nearest_target_label == LICK_ZONE_CENTER
                    fill_color = (30, 210, 80) if body_is_nearest else (80, 120, 240)
                    cv2.ellipse(region_overlay, region_center, axes, angle_deg, 0, 360, fill_color, -1, cv2.LINE_AA)

                    limb_targets = target_geom.get("limb_targets", [])
                    limb_strip_targets = target_geom.get("limb_strip_targets", [])
                    strip_draw_data = []
                    for strip in limb_strip_targets:
                        corners = strip.get("corners")
                        if corners is None or len(corners) != 4:
                            continue
                        pts = np.array([[int(p[0] * sx), int(p[1] * sy)] for p in corners], dtype=np.int32)
                        zone_label = str(strip.get("zone_label", "LIMB_STRIP"))
                        zone_group = str(strip.get("group", ""))
                        is_nearest_strip = nearest_target_hit and nearest_target_label == zone_group
                        fill = (70, 180, 220) if is_nearest_strip else (95, 95, 95)
                        border = (40, 220, 255) if is_nearest_strip else (155, 155, 155)
                        cv2.fillConvexPoly(region_overlay, pts, fill, cv2.LINE_AA)
                        strip_draw_data.append((pts, border))

                    limb_draw_data = []
                    for limb in limb_targets:
                        center = np.asarray(limb.get("center"), dtype=np.float64)
                        radius = float(limb.get("radius", 0.0))
                        if not np.all(np.isfinite(center)) or radius <= 0.0:
                            continue

                        cx = int(center[0] * sx)
                        cy = int(center[1] * sy)
                        rr = max(1, int(round(radius * 0.5 * (sx + sy))))
                        zone_label = str(limb.get("zone_label", "LIMB"))
                        zone_group = str(limb.get("group", ""))
                        is_nearest_limb = nearest_target_hit and nearest_target_label == zone_group

                        fill = (40, 40, 255) if is_nearest_limb else (110, 110, 110)
                        border = (30, 240, 255) if is_nearest_limb else (180, 180, 180)

                        cv2.circle(region_overlay, (cx, cy), rr, fill, -1, cv2.LINE_AA)
                        limb_draw_data.append(((cx, cy), rr, border))

                    # 區域半透明圖層一次混合，避免每幀多次 copy/addWeighted。
                    cv2.addWeighted(region_overlay, 0.35, display, 0.76, 0, display)
                    cv2.ellipse(display, region_center, axes, angle_deg, 0, 360, (230, 230, 230), max(1, int(1 * _ov)), cv2.LINE_AA)
                    for pts, border in strip_draw_data:
                        cv2.polylines(display, [pts], True, border, max(1, int(1 * _ov)), cv2.LINE_AA)

                    for center_px, rr, border in limb_draw_data:
                        cx, cy = center_px
                        cv2.circle(display, (cx, cy), rr, border, max(1, int(1 * _ov)), cv2.LINE_AA)

                    if trap_draw is not None:
                        fill_final = (80, 255, 170) if nearest_target_hit else (60, 170, 255)
                        if heavy_fx:
                            trap_overlay_final = display.copy()
                            cv2.fillConvexPoly(trap_overlay_final, trap_draw, fill_final, cv2.LINE_AA)
                            alpha_final = 0.65 if nearest_target_hit else 0.36
                            cv2.addWeighted(trap_overlay_final, alpha_final, display, 1.0 - alpha_final, 0, display)
                        else:
                            cv2.fillConvexPoly(display, trap_draw, fill_final, cv2.LINE_AA)

                        glow_col = (40, 255, 220) if nearest_target_hit else (255, 210, 90)
                        pulse = trap_pulse if heavy_fx else 0.0
                        glow_thick = max(2, int((4.0 + 2.0 * pulse) * _ov))
                        edge_thick = max(1, int((2.0 + 1.0 * pulse) * _ov))
                        cv2.polylines(display, [trap_draw], True, glow_col, glow_thick, cv2.LINE_AA)
                        cv2.polylines(display, [trap_draw], True, (255, 255, 255), edge_thick, cv2.LINE_AA)

                        if heavy_fx:
                            for idx, pt in enumerate(trap_draw):
                                dot_r = max(2, int((3 + idx % 2) * _ov))
                                cv2.circle(display, (int(pt[0]), int(pt[1])), dot_r + 1, (0, 0, 0), -1, cv2.LINE_AA)
                                cv2.circle(display, (int(pt[0]), int(pt[1])), dot_r, (255, 255, 255), -1, cv2.LINE_AA)

                        zone_text = "NOSE CONTACT ZONE"
                        if nearest_target_hit:
                            zone_text = f"CONTACT -> {nearest_target_label}"
                        zx = int(nose_p[0] * sx + 14 * _ov)
                        zy = int(nose_p[1] * sy + 24 * _ov)
                        if heavy_fx:
                            cv2.putText(display, zone_text, (zx, zy), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * _ov, (0, 0, 0), max(2, int(2 * _ov)), cv2.LINE_AA)
                        cv2.putText(display, zone_text, (zx, zy), cv2.FONT_HERSHEY_SIMPLEX, 0.40 * _ov, glow_col, max(1, int(1 * _ov)), cv2.LINE_AA)

                        if heavy_fx and nearest_target_hit:
                            ripple_r = max(5, int((8 + 10 * trap_pulse) * _ov))
                            cv2.circle(display, nose_pt, ripple_r, (80, 255, 255), max(2, int(3 * _ov)), cv2.LINE_AA)
                            cv2.circle(display, nose_pt, max(2, int(2 * _ov)), (255, 255, 255), -1, cv2.LINE_AA)

                # limb_hit_* 也統一以 nose 最近區域為準（單一區域）
                nearest_group = _limb_zone_group(nearest_target_label)
                if nearest_target_hit and nearest_group in ("FL", "FR", "HL", "HR"):
                    limb_hit_labels = [nearest_target_label]
                    if nearest_group == "FL":
                        limb_hit_fl = 1
                    elif nearest_group == "FR":
                        limb_hit_fr = 1
                    elif nearest_group == "HL":
                        limb_hit_hl = 1
                    elif nearest_group == "HR":
                        limb_hit_hr = 1

            # 進入次數以 nose 實際命中區域為準；若同時命中多區域，只取最近區域。
            # 嚴格計數以肢體群組級標籤為準，避免同肢體子區切換造成重複計數。
            nose_nearest_label = _canonical_contact_label(nearest_target_label)
            if nearest_target_hit and nose_nearest_label != prev_nose_region_label:
                _increase_contact_counter(nose_nearest_label)

            if nearest_target_hit:
                frame_dt = 1.0 / max(model_input_fps, 1e-6)
                _accumulate_lick_time(nose_nearest_label, frame_dt)
                current_bout_zone = _zone_key_from_label(nose_nearest_label)
                if current_bout_zone == active_bout_zone and current_bout_zone != LICK_ZONE_NO_TARGET:
                    active_bout_sec += frame_dt
                else:
                    _close_active_bout()
                    if current_bout_zone != LICK_ZONE_NO_TARGET:
                        active_bout_zone = current_bout_zone
                        active_bout_sec = frame_dt
            else:
                _close_active_bout()

            prev_nose_region_label = nose_nearest_label if nearest_target_hit else LICK_ZONE_NO_TARGET

            # 右上角顯示統一目標狀態（身體中心與四肢同欄）
            if draw_overlay_info:
                target_status_label = f"NOSE_HIT:{nose_nearest_label if nearest_target_hit else LICK_ZONE_NO_TARGET}"
                lick_fs = 0.62 * _ov
                lick_th = max(1, int(2 * _ov))
                lick_shadow = lick_th + 2
                target_cache_key = (target_status_label, lick_fs, lick_th)
                target_size = target_size_cache.get(target_cache_key)
                if target_size is None:
                    target_size = cv2.getTextSize(target_status_label, cv2.FONT_HERSHEY_SIMPLEX, lick_fs, lick_th)
                    target_size_cache[target_cache_key] = target_size
                (lw, lh), _ = target_size
                lick_pad = int(10 * _ov)
                lick_gap = int(8 * _ov)
                lick_box_h = lh + int(14 * _ov)
                lick_box_w = lw + int(20 * _ov)
                lick_x2 = box_x2
                lick_y1 = box_y2 + lick_gap
                lick_x1 = max(0, lick_x2 - lick_box_w)
                lick_y2 = lick_y1 + lick_box_h

                cv2.rectangle(ui_overlay, (lick_x1, lick_y1), (lick_x2, lick_y2), (18, 18, 18), -1)

                ltx = lick_x1 + lick_pad
                lty = lick_y1 + lh + int(4 * _ov)
                lick_color = (140, 255, 140) if nearest_target_hit else (200, 220, 220)
                target_status_text = target_status_label
                target_status_pos = (ltx, lty)
                target_status_style = (lick_fs, lick_th, lick_shadow, lick_color)

            if ui_overlay is not None:
                cv2.addWeighted(ui_overlay, 0.55, display, 0.45, 0, display)
            if state_text_pos is not None:
                cv2.putText(display, state_label, state_text_pos, cv2.FONT_HERSHEY_SIMPLEX, state_fs, (0, 0, 0), state_th_shadow, cv2.LINE_AA)
                cv2.putText(display, state_label, state_text_pos, cv2.FONT_HERSHEY_SIMPLEX, state_fs, status_color, state_th_main, cv2.LINE_AA)
            if target_status_text is not None and target_status_pos is not None and target_status_style is not None:
                lick_fs, lick_th, lick_shadow, lick_color = target_status_style
                cv2.putText(display, target_status_text, target_status_pos, cv2.FONT_HERSHEY_SIMPLEX, lick_fs, (0, 0, 0), lick_shadow, cv2.LINE_AA)
                cv2.putText(display, target_status_text, target_status_pos, cv2.FONT_HERSHEY_SIMPLEX, lick_fs, lick_color, lick_th, cv2.LINE_AA)

            # 顯示當前影片時間（MM:SS）於畫面左上
            time_mm = int(time_sec // 60)
            time_ss = int(time_sec % 60)
            time_label = f"TIME {time_mm:02d}:{time_ss:02d}"
            time_fs = 0.80 * _ov
            time_th = max(1, int(2 * _ov))
            time_x = int(12 * _ov)
            time_y = int(20 * _ov)
            cv2.putText(display, time_label, (time_x, time_y), cv2.FONT_HERSHEY_SIMPLEX, time_fs, (0, 0, 0), time_th + 1, cv2.LINE_AA)
            cv2.putText(display, time_label, (time_x, time_y), cv2.FONT_HERSHEY_SIMPLEX, time_fs, (255, 250, 140), time_th, cv2.LINE_AA)

            if draw_overlay_info:
                info_lines = [
                    (f"N:{norm_value}", (40, 230, 40) if valid_norm else (0, 140, 255)),
                    (
                        f"VIEW_GUARD:{front_view_guard_reason} bs_norm:{body_scale_norm:.4f} be_ratio:{body_ear_ratio:.3f}"
                        if front_view_guard and np.isfinite(body_scale_norm)
                        else "VIEW_GUARD:OFF",
                        (255, 180, 120) if front_view_guard else (130, 130, 130),
                    ),
                    (
                        f"CAT f:{gaze_forward_norm:+.3f} l:{gaze_lateral_norm:+.3f} a:{gaze_angle_deg:+.1f}"
                        if np.isfinite(gaze_angle_deg)
                        else "CAT f:N/A l:N/A a:N/A",
                        (120, 255, 180),
                    ),
                    (f"CAT_SIGN:{CAT_LR_SIGN:+.1f}", (120, 255, 180)),
                    (head_ear_angle_text, (120, 255, 255)),
                    (lick_score_text, (140, 255, 140)),
                ]
                for i, (line_text, line_color) in enumerate(info_lines):
                    y = line_y0 + i * line_step
                    cv2.putText(display, line_text, (info_x, y), cv2.FONT_HERSHEY_SIMPLEX, info_fs, (0, 0, 0), th_shadow, cv2.LINE_AA)
                    cv2.putText(display, line_text, (info_x, y), cv2.FONT_HERSHEY_SIMPLEX, info_fs, line_color, th_main, cv2.LINE_AA)

            # 左下角統計面板：碼表 + 狀態 + 各區域次數/時長/sec-per-hit
            body_hits = int(target_entry_count)
            fl_hits = int(limb_entry_count_fl)
            fr_hits = int(limb_entry_count_fr)
            hl_hits = int(limb_entry_count_hl)
            hr_hits = int(limb_entry_count_hr)

            body_sph = _safe_sec_per_hit(lick_time_body_sec, body_hits)
            fl_sph = _safe_sec_per_hit(lick_time_fl_sec, fl_hits)
            fr_sph = _safe_sec_per_hit(lick_time_fr_sec, fr_hits)
            hl_sph = _safe_sec_per_hit(lick_time_hl_sec, hl_hits)
            hr_sph = _safe_sec_per_hit(lick_time_hr_sec, hr_hits)
            total_lick_time_sec = (
                lick_time_body_sec + lick_time_fl_sec + lick_time_fr_sec + lick_time_hl_sec + lick_time_hr_sec
            )
            body_pref_pct = _safe_pref_pct(lick_time_body_sec, total_lick_time_sec)
            fl_pref_pct = _safe_pref_pct(lick_time_fl_sec, total_lick_time_sec)
            fr_pref_pct = _safe_pref_pct(lick_time_fr_sec, total_lick_time_sec)
            hl_pref_pct = _safe_pref_pct(lick_time_hl_sec, total_lick_time_sec)
            hr_pref_pct = _safe_pref_pct(lick_time_hr_sec, total_lick_time_sec)

            if draw_overlay_info:
                timer_line = (f"TIMER {time_sec:7.2f}s", (255, 250, 120))
                panel_lines = [
                    (
                        f"BODY  C:{body_hits:3d}  T:{lick_time_body_sec:6.2f}s  PREF:{body_pref_pct:5.1f}%"
                        if np.isfinite(body_pref_pct)
                        else f"BODY  C:{body_hits:3d}  T:{lick_time_body_sec:6.2f}s  PREF:N/A",
                        (255, 255, 165),
                    ),
                    (
                        f"FL    C:{fl_hits:3d}  T:{lick_time_fl_sec:6.2f}s  PREF:{fl_pref_pct:5.1f}%"
                        if np.isfinite(fl_pref_pct)
                        else f"FL    C:{fl_hits:3d}  T:{lick_time_fl_sec:6.2f}s  PREF:N/A",
                        (130, 230, 255),
                    ),
                    (
                        f"FR    C:{fr_hits:3d}  T:{lick_time_fr_sec:6.2f}s  PREF:{fr_pref_pct:5.1f}%"
                        if np.isfinite(fr_pref_pct)
                        else f"FR    C:{fr_hits:3d}  T:{lick_time_fr_sec:6.2f}s  PREF:N/A",
                        (130, 230, 255),
                    ),
                    (
                        f"HL    C:{hl_hits:3d}  T:{lick_time_hl_sec:6.2f}s  PREF:{hl_pref_pct:5.1f}%"
                        if np.isfinite(hl_pref_pct)
                        else f"HL    C:{hl_hits:3d}  T:{lick_time_hl_sec:6.2f}s  PREF:N/A",
                        (130, 230, 255),
                    ),
                    (
                        f"HR    C:{hr_hits:3d}  T:{lick_time_hr_sec:6.2f}s  PREF:{hr_pref_pct:5.1f}%"
                        if np.isfinite(hr_pref_pct)
                        else f"HR    C:{hr_hits:3d}  T:{lick_time_hr_sec:6.2f}s  PREF:N/A",
                        (130, 230, 255),
                    ),
                ]

                panel_pad_x = int(3 * _ov)
                panel_pad_y = int(6 * _ov)
                panel_line_h = int(20 * _ov)
                panel_fs = 0.46 * _ov
                panel_th = max(1, int(1 * _ov))
                panel_shadow = max(2, int(2 * _ov))
                timer_fs = 0.60 * _ov
                timer_th = max(2, int(2 * _ov))
                timer_shadow = max(3, int(3 * _ov))
                lx = panel_pad_x
                total_lines = len(panel_lines) + 1
                ly = render_h - panel_pad_y - int((total_lines - 1) * panel_line_h)

                cv2.putText(display, timer_line[0], (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, timer_fs, (0, 0, 0), timer_shadow, cv2.LINE_AA)
                cv2.putText(display, timer_line[0], (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, timer_fs, timer_line[1], timer_th, cv2.LINE_AA)
                ly += panel_line_h

                for text_line, text_col in panel_lines:
                    cv2.putText(display, text_line, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, panel_fs, (0, 0, 0), panel_shadow, cv2.LINE_AA)
                    cv2.putText(display, text_line, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, panel_fs, text_col, panel_th, cv2.LINE_AA)
                    ly += panel_line_h

            if WRITE_LOOPED_PASSES_TO_CSV or playback_pass == 0:
                rows.append(
                    {
                        "video_idx": current_video_idx + 1,
                        "video_path": str(video_path),
                        "playback_pass": playback_pass,
                        "frame_step": frame_step,
                        "source_fps": round(source_fps, 6),
                        "model_input_fps": round(model_input_fps, 6),
                        "frame": raw_frame_idx,
                        "processed_frame": processed_frame_idx,
                        "processed_frame_global": video_processed_total,
                        "time_sec": round(time_sec, 6),
                        "left_ear_x": round(lxy[0], 4) if np.isfinite(lxy[0]) else "",
                        "left_ear_y": round(lxy[1], 4) if np.isfinite(lxy[1]) else "",
                        "right_ear_x": round(rxy[0], 4) if np.isfinite(rxy[0]) else "",
                        "right_ear_y": round(rxy[1], 4) if np.isfinite(rxy[1]) else "",
                        "distance_px": round(dist_px, 6) if np.isfinite(dist_px) else "",
                        "distance_norm": round(dist_norm, 8) if np.isfinite(dist_norm) else "",
                        "ray_end_x": round(ray_end[0], 4) if target_geom is not None else "",
                        "ray_end_y": round(ray_end[1], 4) if target_geom is not None else "",
                        "ray_norm_x": round(ray_nx, 6) if target_geom is not None else "",
                        "ray_norm_y": round(ray_ny, 6) if target_geom is not None else "",
                        "lick_zone": lick_zone,
                        "lick_axis_score": round(lick_axis_score, 6) if np.isfinite(lick_axis_score) else "",
                        "lick_lateral_score": round(lick_lateral_score, 6) if np.isfinite(lick_lateral_score) else "",
                        "nearest_target_label": nearest_target_label,
                        "nearest_target_t": round(nearest_target_t, 6) if np.isfinite(nearest_target_t) else "",
                        "limb_hit_any": int(bool(limb_hit_labels)),
                        "limb_hit_labels": ",".join(limb_hit_labels),
                        "limb_hit_fl_frame": limb_hit_fl,
                        "limb_hit_fr_frame": limb_hit_fr,
                        "limb_hit_hl_frame": limb_hit_hl,
                        "limb_hit_hr_frame": limb_hit_hr,
                        "limb_entry_count_fl": int(limb_entry_count_fl),
                        "limb_entry_count_fr": int(limb_entry_count_fr),
                        "limb_entry_count_hl": int(limb_entry_count_hl),
                        "limb_entry_count_hr": int(limb_entry_count_hr),
                        "target_entry_count": int(target_entry_count),
                        "lick_time_body_sec": round(lick_time_body_sec, 4),
                        "lick_time_fl_sec": round(lick_time_fl_sec, 4),
                        "lick_time_fr_sec": round(lick_time_fr_sec, 4),
                        "lick_time_hl_sec": round(lick_time_hl_sec, 4),
                        "lick_time_hr_sec": round(lick_time_hr_sec, 4),
                        "lick_sec_per_hit_body": round(body_sph, 6) if np.isfinite(body_sph) else "",
                        "lick_sec_per_hit_fl": round(fl_sph, 6) if np.isfinite(fl_sph) else "",
                        "lick_sec_per_hit_fr": round(fr_sph, 6) if np.isfinite(fr_sph) else "",
                        "lick_sec_per_hit_hl": round(hl_sph, 6) if np.isfinite(hl_sph) else "",
                        "lick_sec_per_hit_hr": round(hr_sph, 6) if np.isfinite(hr_sph) else "",
                        "lick_pref_pct_body": round(body_pref_pct, 6) if np.isfinite(body_pref_pct) else "",
                        "lick_pref_pct_fl": round(fl_pref_pct, 6) if np.isfinite(fl_pref_pct) else "",
                        "lick_pref_pct_fr": round(fr_pref_pct, 6) if np.isfinite(fr_pref_pct) else "",
                        "lick_pref_pct_hl": round(hl_pref_pct, 6) if np.isfinite(hl_pref_pct) else "",
                        "lick_pref_pct_hr": round(hr_pref_pct, 6) if np.isfinite(hr_pref_pct) else "",
                        "nose_detected": int(nose_ok),
                        "gaze_forward_norm": round(gaze_forward_norm, 4) if np.isfinite(gaze_forward_norm) else "",
                        "gaze_lateral_norm": round(gaze_lateral_norm, 4) if np.isfinite(gaze_lateral_norm) else "",
                        "gaze_angle_deg": round(gaze_angle_deg, 2) if np.isfinite(gaze_angle_deg) else "",
                        "face_state_cat": face_state_cat,
                        "face_state_raw": state_now,
                        "face_state": state_smoothed,
                        "state_stability": round(state_stability, 4),
                        "valid": valid,
                    }
                )
                _nr_executor.submit(_post_nodered, dict(rows[-1]))

            if STREAM_MODE == 2 and _NEED_FRAME:
                _mjpeg_push(display)
                global _ctrl_switch_delta
                with _ctrl_lock:
                    _remote_delta = _ctrl_switch_delta
                    if _remote_delta != 0:
                        _ctrl_switch_delta = 0
                if _remote_delta != 0:
                    switch_delta = _remote_delta
                    is_paused = False
                    break

            if not DISPLAY_WINDOW and msvcrt is not None and msvcrt.kbhit():
                key_raw = msvcrt.getch()
                if key_raw in (b"q", b"Q", b"\x03"):  # \x03 = Ctrl+C
                    print("\n使用者中斷：q / Ctrl+C")
                    stop_all = True
                    break

            if DISPLAY_WINDOW and STREAM_MODE == 1:
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                action = _handle_key(key, cap_obj=cap, in_pause_loop=False)
                if action == "break":
                    break
                if action == "enter_pause":
                    def _build_pause_frame():
                        if ui_mode == UI_MODE_DET_ONLY and display_det_only is not None:
                            source = display_det_only
                        else:
                            source = display

                        pause_img = source.copy()
                        cv2.putText(
                            pause_img,
                            "PAUSED (Space:Play)",
                            (int(12 * _ov), int(render_h - 18 * _ov)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55 * _ov,
                            (0, 255, 255),
                            max(1, int(2 * _ov)),
                            cv2.LINE_AA,
                        )
                        return pause_img

                    while is_paused:
                        pause_frame = _build_pause_frame()
                        cv2.imshow(WINDOW_NAME, pause_frame)
                        k2 = cv2.waitKey(50) & 0xFF
                        pause_action = _handle_key(k2, cap_obj=cap, in_pause_loop=True)
                        if pause_action == "break":
                            break
                    if stop_all or switch_delta != 0:
                        break

        cap.release()

        # 影片結束前收斂尚未關閉的 bout
        _close_active_bout()

        # 每影片摘要（一支影片一列）
        video_elapsed_sec = processed_frame_idx / max(model_input_fps, 1e-6)
        total_lick_time_sec_video = (
            lick_time_body_sec + lick_time_fl_sec + lick_time_fr_sec + lick_time_hl_sec + lick_time_hr_sec
        )
        body_pref_pct_video = _safe_pref_pct(lick_time_body_sec, total_lick_time_sec_video)
        fl_pref_pct_video = _safe_pref_pct(lick_time_fl_sec, total_lick_time_sec_video)
        fr_pref_pct_video = _safe_pref_pct(lick_time_fr_sec, total_lick_time_sec_video)
        hl_pref_pct_video = _safe_pref_pct(lick_time_hl_sec, total_lick_time_sec_video)
        hr_pref_pct_video = _safe_pref_pct(lick_time_hr_sec, total_lick_time_sec_video)

        pref_candidates = [
            ("BODY", body_pref_pct_video),
            ("FL", fl_pref_pct_video),
            ("FR", fr_pref_pct_video),
            ("HL", hl_pref_pct_video),
            ("HR", hr_pref_pct_video),
        ]
        pref_candidates_valid = [x for x in pref_candidates if np.isfinite(x[1])]
        if pref_candidates_valid:
            dominant_pref_zone, dominant_pref_pct = max(pref_candidates_valid, key=lambda x: x[1])
        else:
            dominant_pref_zone, dominant_pref_pct = LICK_ZONE_NO_TARGET, float("nan")

        body_mean_bout_sec = _safe_mean(bout_time_body_sec, bout_count_body)
        fl_mean_bout_sec = _safe_mean(bout_time_fl_sec, bout_count_fl)
        fr_mean_bout_sec = _safe_mean(bout_time_fr_sec, bout_count_fr)
        hl_mean_bout_sec = _safe_mean(bout_time_hl_sec, bout_count_hl)
        hr_mean_bout_sec = _safe_mean(bout_time_hr_sec, bout_count_hr)

        summary_rows.append(
            {
                "video_idx": current_video_idx + 1,
                "video_path": str(video_path),
                "source_fps": round(source_fps, 6),
                "model_input_fps": round(model_input_fps, 6),
                "processed_frames": int(processed_frame_idx),
                "video_elapsed_sec": round(video_elapsed_sec, 6),
                "total_lick_time_sec": round(total_lick_time_sec_video, 6),
                "dominant_pref_zone": dominant_pref_zone,
                "dominant_pref_pct": round(dominant_pref_pct, 6) if np.isfinite(dominant_pref_pct) else "",
                "body_hits": int(target_entry_count),
                "body_lick_time_sec": round(lick_time_body_sec, 6),
                "body_pref_pct": round(body_pref_pct_video, 6) if np.isfinite(body_pref_pct_video) else "",
                "body_bout_count": int(bout_count_body),
                "body_mean_bout_sec": round(body_mean_bout_sec, 6) if np.isfinite(body_mean_bout_sec) else "",
                "fl_hits": int(limb_entry_count_fl),
                "fl_lick_time_sec": round(lick_time_fl_sec, 6),
                "fl_pref_pct": round(fl_pref_pct_video, 6) if np.isfinite(fl_pref_pct_video) else "",
                "fl_bout_count": int(bout_count_fl),
                "fl_mean_bout_sec": round(fl_mean_bout_sec, 6) if np.isfinite(fl_mean_bout_sec) else "",
                "fr_hits": int(limb_entry_count_fr),
                "fr_lick_time_sec": round(lick_time_fr_sec, 6),
                "fr_pref_pct": round(fr_pref_pct_video, 6) if np.isfinite(fr_pref_pct_video) else "",
                "fr_bout_count": int(bout_count_fr),
                "fr_mean_bout_sec": round(fr_mean_bout_sec, 6) if np.isfinite(fr_mean_bout_sec) else "",
                "hl_hits": int(limb_entry_count_hl),
                "hl_lick_time_sec": round(lick_time_hl_sec, 6),
                "hl_pref_pct": round(hl_pref_pct_video, 6) if np.isfinite(hl_pref_pct_video) else "",
                "hl_bout_count": int(bout_count_hl),
                "hl_mean_bout_sec": round(hl_mean_bout_sec, 6) if np.isfinite(hl_mean_bout_sec) else "",
                "hr_hits": int(limb_entry_count_hr),
                "hr_lick_time_sec": round(lick_time_hr_sec, 6),
                "hr_pref_pct": round(hr_pref_pct_video, 6) if np.isfinite(hr_pref_pct_video) else "",
                "hr_bout_count": int(bout_count_hr),
                "hr_mean_bout_sec": round(hr_mean_bout_sec, 6) if np.isfinite(hr_mean_bout_sec) else "",
            }
        )

        if stop_all:
            break

        if switch_delta != 0:
            current_video_idx = (current_video_idx + switch_delta) % len(video_paths)
        else:
            current_video_idx = (current_video_idx + 1) % len(video_paths)
            if not LOOP_PLAYBACK and current_video_idx == 0:
                break

    if DISPLAY_WINDOW and STREAM_MODE == 1:
        cv2.destroyAllWindows()

    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        print(f"⚠️ CSV 寫入失敗，略過: {e}")

    try:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
            writer.writeheader()
            writer.writerows(summary_rows)
    except Exception as e:
        print(f"⚠️ Summary CSV 寫入失敗，略過: {e}")

    valid_count = sum(int(r["valid"]) for r in rows)
    print("\n" + "=" * 60)
    print(f"✅ 完成，共處理 {len(rows)} 個模型輸入幀")
    print(f"有效耳距幀數: {valid_count}")
    print(f"CSV 已輸出: {output_csv}")
    print(f"Summary 已輸出: {summary_csv}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中止] 使用者按下 Ctrl+C，程式結束。")
    finally:
        _nr_executor.shutdown(wait=False)
