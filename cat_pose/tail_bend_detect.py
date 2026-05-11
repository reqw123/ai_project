"""
tail_bend_detect.py
-------------------
專門判斷貓咪尾巴「彎曲 / 伸直」的視覺化腳本。

關鍵點索引（17-kpt 貓咪 pose 模型）：
  14 = tail_base
  15 = tail_mid
  16 = tail_tip

判斷邏輯：
    在 tail_mid(15) 計算 tail_base(14)→tail_mid(15)→tail_tip(16) 的夾角（曲率）。
    曲率 160 ~ 180    → STRAIGHT
    曲率 130 ~ 159    → SLIGHT_BEND
    曲率 <= 129       → BENT

操作說明：
  SPACE  = 播放 / 暫停
  A / D  = 逐幀後退 / 前進（暫停模式）
  Z / X  = STEP -1 / +1
  T      = 跳至指定幀號
  R      = 重播
  1/2/3  = 切換模型
  6~l    = 切換影片（同原腳本）
  Q      = 結束
"""

from ultralytics import YOLO
import cv2
import numpy as np
import time
import csv
from pathlib import Path

from collections import deque
_USE_PIL = False  # Disable PIL for performance

# ==================== 時間序列輸出設定 ====================
SAVE_TAIL_TS = True
TAIL_TS_DIR = Path(r"C:\cat_pose\output")
ENABLE_VIDEO_LOCK = False        # 單一旗標：是否啟用目標影片鎖定
LOG_ONLY_TARGET_VIDEO = ENABLE_VIDEO_LOCK
TARGET_VIDEO_KEY = "j"           # 目前指定要記錄的影片按鍵
TARGET_VIDEO_INDEX = 12          # key j 對應索引（與 VIDEO_KEY_MAP 一致）
AUTO_LOAD_TARGET_VIDEO = ENABLE_VIDEO_LOCK

# ==================== 基本設定 ====================
IMGSZ          = 640
CONF_THRES     = 0.50
KP_CONF_THRES  = 0.5      # 關鍵點信心度門檻

# 尾巴曲率分類門檻（度）
# 160 ~ 180 -> STRAIGHT
# 130 ~ 159 -> SLIGHT_BEND
# <= 129    -> BENT
CURV_STRAIGHT_MIN = 160.0
CURV_SLIGHT_MIN   = 130.0
# 尾巴上舉時的「伸直」專用門檻（實務上常低於 160）
CURV_UPRIGHT_STRAIGHT_MIN = 145.0

# ==================== 尾巴關鍵點索引 ====================
IDX_TAIL_BASE = 14
IDX_TAIL_MID  = 15
IDX_TAIL_TIP  = 16

# ==================== 跳幀設定 ====================
FRAME_STEP = 5
SHOW_TAIL_TRAJECTORY = False  # 是否顯示尾尖軌跡視覺跟隨

TRAJ_LEN = 40     # 軌跡保留幀數
SPEED_SMOOTH_N  = 6      # 速度平滑視窗
ANGVEL_SMOOTH_N = 6      # 角速度平滑視窗
SPEED_FAST_THRES = 8.0   # 歸一化速度 >= 此值視為「快速甩尾」

# 情緒三分類門檻（僅用 tail_base/mid/tip）
EMO_VEL_HAPPY_MAX = 10.0    # Happy 速度上限（% tail-length / frame）
EMO_VEL_ANGRY_MIN = 8.0     # 高速下限
EMO_OSC_HAPPY_MAX = 0.7     # 低頻上限（Hz）
EMO_OSC_ANGRY_MIN = 1.4     # 高頻下限（Hz）
EMO_HEIGHT_HAPPY_MIN = 150.0 # Happy 高度門檻：Hmed > 150px
EMO_UPANGLE_HAPPY_MIN = 55.0 # 尾巴上舉角（相對水平）中位數門檻（度）
EMO_WINDOW_SECONDS = 1.0    # 情緒分析窗口秒數（統計用）
EMO_CONFIRM_SECONDS = 1.0   # 任一情緒需連續滿足此秒數才確認（也作為最小資料秒數）


def get_emotion_color(emotion):
    return {
        "HAPPY": (0, 220, 255),
        "FOCUSED": (180, 180, 180),
        "ANGRY": (0, 80, 255),
        "ANALYZING": (150, 180, 200),
    }.get(emotion, (180, 180, 180))

# ==================== 模型清單 ====================
MODEL_LIST = [
    r"C:\cat_pose\white_edge.pt",
    r"C:\cat_pose\v11s_53.pt",
    r"C:\cat_pose\v11s_60.pt",
]

# ==================== 影片清單 ====================
# 來源可同時混用：單一影片檔 or 資料夾（會遞迴掃描影片）
VIDEO_SOURCES = [
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\walk",  # key: 6
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\摳圖影片集\1669342_Cat_Mammal_3840x2160.mp4",  # key: 7
    r"C:\cat_pose\模型測試影片\0_Cat_Tabby_1280x720.mp4",  # key: 8
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\screen5測試影片\screen_37.mp4",  # key: 9
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\walk\walk_1.mp4",  # key: 0
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\walk\5724242 (1).mp4",  # key: e
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\摳圖影片集\0_Cat_Ginger_Cat_1920x1080.mp4",  # key: f
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\摳圖影片集\0_Cat_Calico_1920x1080.mp4",  # key: c
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\摳圖影片集\0_Cat_Feline_1920x1080 (1).mp4",  # key: v
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\摳圖影片集\0_Cat_Metronome_1920x1080.mp4",  # key: b
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\shake\shake_1.mp4",  # key: n
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\shake\shake_2.mp4",  # key: m
    r"C:\Users\homec\Downloads\0_Cat_Yellow_3840x2160.mov",  # key: j
    r"C:\cat_pose\模型測試影片\0_Cat_Ginger_Cat_1280x720.mp4",  # key: k
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\摳圖影片集\6700140_Bicolor_Cat_Bicolor_1920x1080.mp4",  # key: l
]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm"}
VIDEO_SWITCH_KEY_ORDER = "67890efcvbnmjkl"


def build_video_list(sources):
    """將來源清單展開成實際可讀影片路徑（支援單檔/資料夾混用）。"""
    videos = []
    seen = set()

    for src in sources:
        p = Path(src).expanduser()
        if not p.exists():
            print(f"[WARN] Source not found: {p}")
            continue

        if p.is_file():
            if p.suffix.lower() in VIDEO_EXTENSIONS:
                key = str(p.resolve()).lower()
                if key not in seen:
                    videos.append(str(p))
                    seen.add(key)
            else:
                print(f"[WARN] Skip non-video file: {p}")
            continue

        if p.is_dir():
            for fp in sorted(p.rglob("*")):
                if not fp.is_file() or fp.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                key = str(fp.resolve()).lower()
                if key in seen:
                    continue
                videos.append(str(fp))
                seen.add(key)

    return videos


def build_video_key_map(video_count):
    """依可用影片數量動態建立快捷鍵對照。"""
    keys = [ord(ch) for ch in VIDEO_SWITCH_KEY_ORDER]
    usable = min(video_count, len(keys))
    return {keys[i]: i for i in range(usable)}

# ==================== 顏色 ====================
MAX_W, MAX_H   = 1280, 720
COLOR_STRAIGHT    = (0, 220, 0)       # 綠色 → 直
COLOR_SLIGHT_BEND = (0, 200, 255)     # 黃橘 → 微彎
COLOR_BENT        = (0, 80, 255)      # 橘紅 → 彎曲
COLOR_UNKNOWN     = (120, 120, 120)   # 灰色 → 無法判斷
COLOR_KPT      = (255, 255, 0)     # 一般關鍵點（青黃）
COLOR_ANGLE    = (255, 255, 255)   # 角度文字
COLOR_BBOX     = (255, 160, 0)     # Bounding box（亮橘）
COLOR_SKEL     = (80, 80, 80)      # 一般骨架（暗灰）

# ==================== 完整骨架連結 ====================
SKEL_LINKS = [
    (0,1),(0,2),(1,2),          # 頭部
    (0,3),(3,4),(4,5),          # 脊椎
    (3,6),(6,7),                # 左前腿
    (3,8),(8,9),                # 右前腿
    (5,10),(10,11),             # 左後腿
    (5,12),(12,13),             # 右後腿
    (5,14),(14,15),(15,16),     # 尾巴
]
# 尾巴連結（用於特別強調）
TAIL_LINKS = [(5,14),(14,15),(15,16)]

# ==================== 方向鍵 keycode ====================
KEY_UP    = 2490368
KEY_DOWN  = 2621440
KEY_LEFT  = 2424832
KEY_RIGHT = 2555904


# ======================================================
#  工具函式
# ======================================================

def draw_text_outlined(frame, text, x, y, scale=0.7, thickness=2,
                        fg=(255, 255, 255), bg=(0, 0, 0)):
    """白字 + 黑色描邊（純 ASCII / 英數）"""
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, bg, thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, fg, thickness, cv2.LINE_AA)


def draw_pil_text(frame, text, x, y, font, color_bgr):
    # Use only cv2 for English text for performance
    draw_text_outlined(frame, text, x, y+28, scale=1.1, thickness=3, fg=color_bgr)


def prescale_frame(frame):
    """讀入後立即縮放到 MAX_W x MAX_H 以內"""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame,
                           (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return frame


def jump_to_frame(cap, target):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = max(0, min(target, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, f = cap.read()
    return prescale_frame(f) if ret else None


def extract_tail_triplet(kpts, kpt_conf, frame_w, frame_h):
    """擷取 tail_base/mid/tip 原始與正規化座標；不足信心度回傳 None。"""
    need = [IDX_TAIL_BASE, IDX_TAIL_MID, IDX_TAIL_TIP]
    if any(kpt_conf[i] < KP_CONF_THRES for i in need):
        return None

    b = kpts[IDX_TAIL_BASE].astype(float)
    m = kpts[IDX_TAIL_MID].astype(float)
    t = kpts[IDX_TAIL_TIP].astype(float)

    fw = max(float(frame_w), 1.0)
    fh = max(float(frame_h), 1.0)
    return {
        "base_x": float(b[0]), "base_y": float(b[1]),
        "mid_x": float(m[0]), "mid_y": float(m[1]),
        "tip_x": float(t[0]), "tip_y": float(t[1]),
        "base_x_norm": float(b[0] / fw), "base_y_norm": float(b[1] / fh),
        "mid_x_norm": float(m[0] / fw), "mid_y_norm": float(m[1] / fh),
        "tip_x_norm": float(t[0] / fw), "tip_y_norm": float(t[1] / fh),
    }


# ======================================================
#  核心：尾巴角度計算
# ======================================================

def calc_tail_curvature(kpts, kpt_conf):
    """
    計算 tail_base(14) → tail_mid(15) → tail_tip(16) 夾角（度），即尾巴曲率。
    若任一點信心度不足，回傳 None。
    """
    for idx in (IDX_TAIL_BASE, IDX_TAIL_MID, IDX_TAIL_TIP):
        if kpt_conf[idx] < KP_CONF_THRES:
            return None
    p_base = kpts[IDX_TAIL_BASE]
    p_mid  = kpts[IDX_TAIL_MID]
    p_tip  = kpts[IDX_TAIL_TIP]
    v1 = p_base - p_mid   # mid → base
    v2 = p_tip  - p_mid   # mid → tip
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-3 or norm2 < 1e-3:
        return None
    cos_a = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))

def calc_tail_angle_body(kpts, kpt_conf):
    """
    計算 spine(5→3) → tail_base(14) 及 tail_base(14)→tail_tip(16) 的夾角（度）。
    若任一點信心度不足，回傳 None。
    """
    need = [3, 5, IDX_TAIL_BASE, IDX_TAIL_TIP]
    if any(kpt_conf[i] < KP_CONF_THRES for i in need):
        return None
    spine = kpts[IDX_TAIL_BASE] - kpts[5]  # hip(5)→tail_base(14)
    tail  = kpts[IDX_TAIL_TIP] - kpts[IDX_TAIL_BASE]  # tail_base→tail_tip
    norm1 = np.linalg.norm(spine)
    norm2 = np.linalg.norm(tail)
    if norm1 < 1e-3 or norm2 < 1e-3:
        return None
    cos_a = np.clip(np.dot(spine, tail) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))

def calc_tail_height(kpts, kpt_conf):
    """
    計算 tail_tip_y - tail_base_y（像素座標，y向下）。
    若任一點信心度不足，回傳 None。
    """
    if kpt_conf[IDX_TAIL_BASE] < KP_CONF_THRES or kpt_conf[IDX_TAIL_TIP] < KP_CONF_THRES:
        return None
    return float(kpts[IDX_TAIL_BASE][1] - kpts[IDX_TAIL_TIP][1])  # tip高於base為正


def calc_tail_up_angle(kpts, kpt_conf):
    """
    計算尾巴上舉角（相對水平）：
    0 度約為水平，90 度為幾乎垂直向上。
    """
    if kpt_conf[IDX_TAIL_BASE] < KP_CONF_THRES or kpt_conf[IDX_TAIL_TIP] < KP_CONF_THRES:
        return None
    dx = float(kpts[IDX_TAIL_TIP][0] - kpts[IDX_TAIL_BASE][0])
    dy_up = float(kpts[IDX_TAIL_BASE][1] - kpts[IDX_TAIL_TIP][1])
    return float(np.degrees(np.arctan2(dy_up, abs(dx) + 1e-6)))


def infer_cat_emotion_window(curv_hist, vel_hist, osc_hist, height_hist, upang_hist, fps):
    """
    使用時間窗口（多幀）推論情緒，降低瞬時噪聲影響。
    回傳 (emotion_str, color_bgr)
    """
    min_frames = max(1, int(max(fps, 1.0) * EMO_CONFIRM_SECONDS))

    # 資料不足時顯示判斷中
    if (
        len(vel_hist) < min_frames
        or len(osc_hist) < max(3, int(min_frames * 0.6))
        or len(height_hist) < max(3, int(min_frames * 0.6))
        or len(upang_hist) < max(3, int(min_frames * 0.6))
    ):
        return "ANALYZING", (150, 180, 200)

    curv = np.array(curv_hist, dtype=float) if len(curv_hist) else np.array([], dtype=float)
    vel = np.array(vel_hist, dtype=float)
    osc = np.array(osc_hist, dtype=float)
    hgt = np.array(height_hist, dtype=float)
    upa = np.array(upang_hist, dtype=float)

    med_curv = float(np.median(curv)) if curv.size else 0.0
    std_curv = float(np.std(curv)) if curv.size else 999.0
    med_vel = float(np.median(vel))
    p90_vel = float(np.percentile(vel, 90))
    med_osc = float(np.median(osc))
    p90_osc = float(np.percentile(osc, 90))
    med_hgt = float(np.median(hgt))
    med_upa = float(np.median(upa))

    high_vel_ratio = float(np.mean(vel >= (EMO_VEL_ANGRY_MIN + 1.5)))
    high_osc_ratio = float(np.mean(osc >= (EMO_OSC_ANGRY_MIN + 0.3)))

    # Angry / Irritated（嚴格）：必須速度與擺頻都持續偏高
    if (
        p90_vel >= (EMO_VEL_ANGRY_MIN + 2.0)
        and p90_osc >= (EMO_OSC_ANGRY_MIN + 0.4)
        and high_vel_ratio >= 0.35
        and high_osc_ratio >= 0.35
    ):
        return "ANGRY", (0, 80, 255)

    # Happy（使用者指定標準）：Hmed >= 150px 且 Tip speed 中位數 <= 10
    if (
        med_hgt >= EMO_HEIGHT_HAPPY_MIN
        and med_vel <= EMO_VEL_HAPPY_MAX
    ):
        return "HAPPY", (0, 220, 255)

    # Focused / Thinking：其餘落在中速/低頻或混合狀態
    return "FOCUSED", (180, 180, 180)


def classify_tail(angle):
    """
    回傳 (label_str, color)
    """
    if angle is None:
        return "UNKNOWN", COLOR_UNKNOWN
    if angle >= CURV_STRAIGHT_MIN:
        return "STRAIGHT", COLOR_STRAIGHT
    if CURV_SLIGHT_MIN <= angle < CURV_STRAIGHT_MIN:
        return "SLIGHT_BEND", COLOR_SLIGHT_BEND
    return "BENT", COLOR_BENT


# ======================================================
#  進階分析：方向 / 速度 / 角速度
# ======================================================

# 8 方位名稱（順時鐘，0°=右）
_DIR8 = ["RIGHT", "D-RIGHT", "DOWN", "D-LEFT",
         "LEFT",  "U-LEFT",  "UP",   "U-RIGHT"]

def calc_tail_direction(kpts, kpt_conf):
    """
    尾巴整體朝向：tail_base(14) → tail_tip(16) 向量角度。
    回傳 (angle_deg_0to360, direction_label) 或 (None, None)。
    angle_deg: 0=右, 90=下(圖像座標), 180=左, 270=上
    """
    if kpt_conf[IDX_TAIL_BASE] < KP_CONF_THRES or kpt_conf[IDX_TAIL_TIP] < KP_CONF_THRES:
        return None, None
    dx = float(kpts[IDX_TAIL_TIP][0] - kpts[IDX_TAIL_BASE][0])
    dy = float(kpts[IDX_TAIL_TIP][1] - kpts[IDX_TAIL_BASE][1])
    ang = float(np.degrees(np.arctan2(dy, dx)))   # -180 ~ 180
    ang360 = ang % 360                             # 0 ~ 360
    sector = int((ang360 + 22.5) / 45) % 8
    return ang360, _DIR8[sector]


def calc_body_axis_side(kpts, kpt_conf):
    """
    判斷尾巴相對身體軸的擺動側（LEFT / RIGHT / CENTER）。
    體軸：kp3(shoulder) → kp5(hip)；尾軸：kp14 → kp16。
    叉積 z 分量 > 0 → 尾巴在體軸右側（圖像坐標）。
    """
    need = [3, 5, IDX_TAIL_BASE, IDX_TAIL_TIP]
    if any(kpt_conf[i] < KP_CONF_THRES for i in need):
        return None
    body = kpts[5].astype(float) - kpts[3].astype(float)   # 體軸向量
    tail = kpts[IDX_TAIL_TIP].astype(float) - kpts[IDX_TAIL_BASE].astype(float)
    cross_z = body[0] * tail[1] - body[1] * tail[0]        # 2D 叉積 z
    if abs(cross_z) < 50:    # 幾乎與體軸平行
        return "CENTER"
    return "RIGHT" if cross_z > 0 else "LEFT"


def calc_tip_speed(tip_pos, prev_pos, body_len):
    """
    tail_tip 每幀位移，除以體長歸一化。
    回傳 normalized_speed 或 None。
    """
    if tip_pos is None or prev_pos is None or body_len < 1e-3:
        return None
    dist = float(np.linalg.norm(tip_pos - prev_pos))
    return dist / body_len * 100   # 單位：% 體長/幀


def get_tail_length_ref(kpts, kpt_conf):
    """以 tail_base(14) 到 tail_tip(16) 距離作為速度正規化尺度。"""
    if kpt_conf[IDX_TAIL_BASE] > KP_CONF_THRES and kpt_conf[IDX_TAIL_TIP] > KP_CONF_THRES:
        return max(float(np.linalg.norm(kpts[IDX_TAIL_TIP] - kpts[IDX_TAIL_BASE])), 1e-3)
    return 1e-3


def calc_tail_swing_sign(kpts, kpt_conf):
    """以 (base->mid) x (mid->tip) 的符號估計尾巴左右擺動方向。"""
    need = [IDX_TAIL_BASE, IDX_TAIL_MID, IDX_TAIL_TIP]
    if any(kpt_conf[i] < KP_CONF_THRES for i in need):
        return 0
    v1 = kpts[IDX_TAIL_MID].astype(float) - kpts[IDX_TAIL_BASE].astype(float)
    v2 = kpts[IDX_TAIL_TIP].astype(float) - kpts[IDX_TAIL_MID].astype(float)
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    if abs(cross_z) < 1e-3:
        return 0
    return 1 if cross_z > 0 else -1


def calc_tail_oscillation_hz(sign_hist, fps):
    """以符號翻轉率估計尾巴擺動頻率（Hz）。"""
    signs = [s for s in sign_hist if s != 0]
    if len(signs) < 3 or fps <= 0:
        return None
    flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    duration_s = len(signs) / fps
    if duration_s <= 1e-6:
        return None
    return float((flips / 2.0) / duration_s)


def get_body_length(kpts, kpt_conf):
    """kp0(頭) → kp5(臀) 距離作為體長參考，不足則回傳 1e-3"""
    if kpt_conf[0] > KP_CONF_THRES and kpt_conf[5] > KP_CONF_THRES:
        return float(np.linalg.norm(kpts[0] - kpts[5]))
    return 1e-3


# ======================================================
#  視覺化：BBox + 全骨架 + 尾巴特寫 + 角度弧線
# ======================================================

def draw_bbox(frame, result):
    """繪製所有偵測到的貓咪 BBox"""
    if result.boxes is None:
        return
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2)
        label = f"cat {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), COLOR_BBOX, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def draw_all_keypoints_and_skeleton(frame, kpts, kpt_conf):
    """繪製全部 17 個關鍵點（小圓點）＋全骨架（細暗線），尾巴三點除外（留給特寫）"""
    TAIL_SET = {IDX_TAIL_BASE, IDX_TAIL_MID, IDX_TAIL_TIP}

    # 骨架連線（暗灰）
    for a, b in SKEL_LINKS:
        if a in TAIL_SET or b in TAIL_SET:
            continue   # 尾巴線段留給 draw_tail_focus
        if kpt_conf[a] > KP_CONF_THRES and kpt_conf[b] > KP_CONF_THRES:
            pa = kpts[a].astype(int)
            pb = kpts[b].astype(int)
            cv2.line(frame, tuple(pa), tuple(pb), COLOR_SKEL, 2)

    # 一般關鍵點（小圓點）
    for i, (x, y) in enumerate(kpts):
        if i in TAIL_SET:
            continue
        if kpt_conf[i] < KP_CONF_THRES:
            continue
        px, py = int(x), int(y)
        cv2.circle(frame, (px, py), 4, COLOR_KPT, -1)
        cv2.circle(frame, (px, py), 4, (0, 0, 0), 1)
        draw_text_outlined(frame, str(i), px + 5, py - 4,
                           scale=0.38, thickness=1, fg=COLOR_KPT)


def draw_trajectory(frame, traj, tail_color):
    """
    畫尾巴 tip 軌跡拖尾。
    traj: deque of (x, y)，最新在末尾。
    """
    pts = list(traj)
    n = len(pts)
    if n < 2:
        return
    for i in range(1, n):
        alpha = i / n           # 越新越亮
        thick = max(1, int(alpha * 4))
        # 顏色由暗到亮
        c = tuple(int(ch * alpha) for ch in tail_color)
        cv2.line(frame, pts[i-1], pts[i], c, thick, cv2.LINE_AA)
    # 尾端畫小圓
    cv2.circle(frame, pts[-1], 5, tail_color, -1)
    cv2.circle(frame, pts[-1], 5, (255,255,255), 1)


def draw_direction_arrow(frame, kpts, kpt_conf, tail_color, dir_label, ang360):
    """
    從 tail_base 畫一條方向箭頭，標示尾巴整體朝向。
    """
    if ang360 is None or kpt_conf[IDX_TAIL_BASE] < KP_CONF_THRES:
        return
    px, py = kpts[IDX_TAIL_BASE].astype(int)
    arrow_len = 50
    rad = np.deg2rad(ang360)
    ex = int(px + arrow_len * np.cos(rad))
    ey = int(py + arrow_len * np.sin(rad))
    cv2.arrowedLine(frame, (px, py), (ex, ey),
                    (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.35)
    cv2.arrowedLine(frame, (px, py), (ex, ey),
                    tail_color, 2, cv2.LINE_AA, tipLength=0.35)
    draw_text_outlined(frame, dir_label,
                       ex + 4, ey + 4, scale=0.48, thickness=1,
                       fg=(255, 220, 100))


def draw_tail_focus(frame, kpts, kpt_conf, angle, tail_color):
    """尾巴三點特寫：發光外環 + 粗連線 + 角度弧 + 標籤"""
    p_base = kpts[IDX_TAIL_BASE].astype(int)
    p_mid  = kpts[IDX_TAIL_MID].astype(int)
    p_tip  = kpts[IDX_TAIL_TIP].astype(int)

    # ── 尾巴骨架連線（先畫背光加粗版，再畫彩色細版）──────
    pairs = []
    if kpt_conf[IDX_TAIL_BASE] > KP_CONF_THRES and kpt_conf[IDX_TAIL_MID] > KP_CONF_THRES:
        pairs.append((p_base, p_mid))
    if kpt_conf[IDX_TAIL_MID] > KP_CONF_THRES and kpt_conf[IDX_TAIL_TIP] > KP_CONF_THRES:
        pairs.append((p_mid, p_tip))
    # 也連 tail_base 到 kp5（脊椎末）
    if kpt_conf[5] > KP_CONF_THRES and kpt_conf[IDX_TAIL_BASE] > KP_CONF_THRES:
        pairs.append((kpts[5].astype(int), p_base))
    for pa, pb in pairs:
        cv2.line(frame, tuple(pa), tuple(pb), (0, 0, 0),   7)   # 外邊黑
        cv2.line(frame, tuple(pa), tuple(pb), tail_color,  4)   # 彩色線

    # ── 關鍵點：三層圓（外暈 → 白環 → 彩心）────────────────
    kpt_meta = [
        (IDX_TAIL_BASE, "14", "base"),
        (IDX_TAIL_MID,  "15", "mid"),
        (IDX_TAIL_TIP,  "16", "tip"),
    ]
    for idx, num_tag, name_tag in kpt_meta:
        if kpt_conf[idx] < KP_CONF_THRES:
            continue
        px, py = kpts[idx].astype(int)
        overlay = frame.copy()
        cv2.circle(overlay, (px, py), 20, tail_color, -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.circle(frame, (px, py), 12, (255, 255, 255), 2)
        cv2.circle(frame, (px, py), 9,  tail_color, -1)
        cv2.circle(frame, (px, py), 9,  (0, 0, 0),  1)
        draw_text_outlined(frame, num_tag, px - 5, py + 5, scale=0.45, thickness=1, fg=(0, 0, 0), bg=(255, 255, 255))
        draw_text_outlined(frame, f"{name_tag} {kpt_conf[idx]:.2f}", px + 14, py - 8, scale=0.48, thickness=1, fg=(255, 255, 200))

    # ── 角度弧線（以 tail_mid 為圓心）────────────────────
    if angle is not None and kpt_conf[IDX_TAIL_MID] > KP_CONF_THRES:
        arc_r = 32
        v1 = kpts[IDX_TAIL_BASE].astype(float) - kpts[IDX_TAIL_MID].astype(float)
        v2 = kpts[IDX_TAIL_TIP].astype(float)  - kpts[IDX_TAIL_MID].astype(float)
        ang1 = float(np.degrees(np.arctan2(v1[1], v1[0])))
        ang2 = float(np.degrees(np.arctan2(v2[1], v2[0])))
        a_start = min(ang1, ang2)
        a_end   = max(ang1, ang2)
        if (a_end - a_start) > 180:
            a_start, a_end = a_end, a_end + (360 - (a_end - a_start))
        cv2.ellipse(frame, tuple(p_mid),
                    (arc_r, arc_r), 0, a_start, a_end,
                    (255, 255, 255), 2, cv2.LINE_AA)
        draw_text_outlined(frame,
                           f"{angle:.1f}deg",
                           p_mid[0] + arc_r + 4, p_mid[1] + 6,
                           scale=0.6, thickness=2,
                           fg=COLOR_ANGLE)


def draw_tail_visualization(frame, kpts, kpt_conf, result, angle, color,
                             traj, dir_label, ang360):
    """整合：BBox → 全關鍵點骨架 → 軌跡 → 方向箭頭 → 尾巴特寫"""
    draw_bbox(frame, result)
    draw_all_keypoints_and_skeleton(frame, kpts, kpt_conf)
    if SHOW_TAIL_TRAJECTORY:
        draw_trajectory(frame, traj, color)
    draw_direction_arrow(frame, kpts, kpt_conf, color, dir_label or "", ang360)
    draw_tail_focus(frame, kpts, kpt_conf, angle, color)


def draw_status_banner(frame, label, color, angle, model_idx, video_idx,
                        fps, frame_id, total_frames, frame_step, play_mode,
                        dir_label=None, side=None, speed=None, ang_vel=None,
                        happy_hgt_med=None, happy_upa_med=None):
    """左上角狀態資訊 + 右上角中文大標語（PIL 渲染） + 進階資訊"""
    h, w = frame.shape[:2]

    # ── 頂部色條 ──────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 6), color, -1)

    # ── 右上角英文大標語（無底色，僅文字）────────────────
    banner_w, banner_h = 340, 80
    bx, by = w - banner_w - 10, 24
    main_label = {
        "STRAIGHT": "STRAIGHT",
        "SLIGHT_BEND": "SLIGHT BEND",
        "BENT": "BENT",
        "UNKNOWN": "UNKNOWN",
    }.get(label, "UNKNOWN")
    draw_text_outlined(frame, main_label, bx + 16, by + 16, scale=1.4, thickness=3, fg=color)

    # (移除右上角角度與門檻提示，讓版面更簡潔)

    # ── 左上角小字資訊 ──────────────────────────────────
    mode_str = "PLAY" if play_mode else "PAUSE"
    lines = [
        f"{mode_str}  Model:{model_idx+1}/{len(MODEL_LIST)}  Video:{video_idx+1}/{len(VIDEO_LIST)}",
        f"Frame:{frame_id}/{total_frames}  Step:{frame_step}",
        f"FPS:{fps:.1f}  kp_conf_thres:{KP_CONF_THRES}",
    ]
    y = 18
    for line in lines:
        draw_text_outlined(frame, line, 12, y, scale=0.48, thickness=1)
        y += 17

    # ── 進階資訊面板（左側垂直區塊）────────────────────
    panel_x = 12
    py_start = y + 6
    adv_lines = []
    if dir_label is not None:
        adv_lines.append((f"Direction : {dir_label}",  (255, 220, 80)))
    if side is not None:
        side_color = (100, 200, 255) if side == "LEFT" else \
                     (100, 255, 150) if side == "RIGHT" else \
                     (200, 200, 200)
        adv_lines.append((f"Body side : {side}", side_color))
    if speed is not None:
        spd_color = (0, 80, 255) if speed >= SPEED_FAST_THRES else (180, 255, 180)
        fast_tag = "  << FAST >>" if speed >= SPEED_FAST_THRES else ""
        adv_lines.append((f"Tip speed : {speed:.1f} %/fr{fast_tag}", spd_color))
    if ang_vel is not None:
        av_color = (0, 160, 255) if abs(ang_vel) > 5 else (180, 255, 180)
        adv_lines.append((f"Ang.vel   : {ang_vel:+.1f} deg/fr", av_color))
    if happy_hgt_med is not None:
        hgt_color = (120, 255, 200) if happy_hgt_med >= EMO_HEIGHT_HAPPY_MIN else (180, 180, 180)
        adv_lines.append((f"Happy Hmed : {happy_hgt_med:.1f} px", hgt_color))
    if happy_upa_med is not None:
        upa_color = (120, 255, 200) if happy_upa_med >= EMO_UPANGLE_HAPPY_MIN else (180, 180, 180)
        adv_lines.append((f"Happy Umed : {happy_upa_med:.1f} deg", upa_color))

    for txt, fc in adv_lines:
        draw_text_outlined(frame, txt, panel_x, py_start, scale=0.46,
                           thickness=1, fg=fc)
        py_start += 17

    # ── 底部操作提示 ────────────────────────────────────
    video_keys = "".join(chr(k) for k in VIDEO_KEY_MAP.keys())
    folder_switch = "  [/] :prev/next" if HAS_FOLDER_SOURCE else ""
    hint = f"SPACE:play/pause  A/D:frame step  Z/X:adjust step  T:goto  1-3:model{folder_switch}  {video_keys}:video  Q:quit"
    draw_text_outlined(frame, hint, 8, h - 8, scale=0.38, thickness=1,
                       fg=(160, 160, 160))


# ======================================================
#  初始化
# ======================================================

def load_model(path):
    m = YOLO(path)
    m.to("cuda")
    print(f"[INFO] Model loaded: {path}")
    return m


def load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {path}")
        return None
    print(f"[INFO] Video loaded: {path}")
    return cap


VIDEO_LIST = build_video_list(VIDEO_SOURCES)
if not VIDEO_LIST:
    raise RuntimeError("No readable videos found. Please set VIDEO_SOURCES to a valid file or folder path.")

HAS_FOLDER_SOURCE = any(Path(src).expanduser().is_dir() for src in VIDEO_SOURCES)

VIDEO_KEY_MAP = build_video_key_map(len(VIDEO_LIST))
if not VIDEO_KEY_MAP:
    raise RuntimeError("No usable video switch keys available.")

if len(VIDEO_LIST) > len(VIDEO_SWITCH_KEY_ORDER):
    print(f"[WARN] Found {len(VIDEO_LIST)} videos, but only first {len(VIDEO_SWITCH_KEY_ORDER)} can be switched by keyboard.")

if TARGET_VIDEO_KEY and ord(TARGET_VIDEO_KEY) in VIDEO_KEY_MAP:
    TARGET_VIDEO_INDEX = VIDEO_KEY_MAP[ord(TARGET_VIDEO_KEY)]
else:
    TARGET_VIDEO_INDEX = max(0, min(TARGET_VIDEO_INDEX, len(VIDEO_LIST) - 1))


model_index = 0
video_index = 0

model = load_model(MODEL_LIST[model_index])
cap   = load_video(VIDEO_LIST[video_index])

if AUTO_LOAD_TARGET_VIDEO and video_index != TARGET_VIDEO_INDEX:
    if cap is not None:
        cap.release()
    video_index = TARGET_VIDEO_INDEX
    cap = load_video(VIDEO_LIST[video_index])
    print(f"[INFO] Auto-switched to target video key '{TARGET_VIDEO_KEY}' (index={TARGET_VIDEO_INDEX})")

tail_ts_file = None
tail_ts_writer = None
if SAVE_TAIL_TS:
    TAIL_TS_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_ONLY_TARGET_VIDEO:
        ts_name = time.strftime(f"tail_kpt_timeseries_key{TARGET_VIDEO_KEY}_%Y%m%d_%H%M%S.csv")
    else:
        ts_name = time.strftime("tail_kpt_timeseries_%Y%m%d_%H%M%S.csv")
    tail_ts_path = TAIL_TS_DIR / ts_name
    tail_ts_file = open(tail_ts_path, "w", newline="", encoding="utf-8")
    tail_ts_writer = csv.writer(tail_ts_file)
    tail_ts_writer.writerow([
        "frame_id", "time_sec", "model_index", "video_index", "video_key", "video_path",
        "emotion", "tail_class", "norm_method",
        "base_x", "base_y", "mid_x", "mid_y", "tip_x", "tip_y",
        "base_x_norm", "base_y_norm", "mid_x_norm", "mid_y_norm", "tip_x_norm", "tip_y_norm",
    ])
    print(f"[INFO] Tail time-series CSV: {tail_ts_path}")

play_mode  = True
prev_time  = time.time()
frame      = None

# 角度平滑（簡單移動平均，視窗=5）
SMOOTH_N   = 5
angle_buf  = []

# ── 進階追蹤狀態 ──────────────────────────────────────
tail_traj      = deque(maxlen=TRAJ_LEN)   # tail_tip 軌跡
prev_tip_pos   = None                     # 上一幀 tip 座標
speed_buf      = []                       # 速度平滑緩衝
prev_angle     = None                     # 上一幀彎曲角度
angvel_buf     = []                       # 角速度平滑緩衝
swing_sign_buf = deque(maxlen=30)         # 擺動方向符號序列（估頻）
emotion_curv_hist = deque(maxlen=180)     # 情緒判斷曲率歷史
emotion_vel_hist  = deque(maxlen=180)     # 情緒判斷速度歷史
emotion_osc_hist  = deque(maxlen=180)     # 情緒判斷擺頻歷史
emotion_height_hist = deque(maxlen=180)   # 情緒判斷高度歷史
emotion_upang_hist = deque(maxlen=180)    # 情緒判斷上舉角歷史

video_switch_hint = "".join(chr(k) for k in VIDEO_KEY_MAP.keys())

print("\n============= TAIL BEND DETECTOR =============")
print("SPACE=play/pause  A/D=frame±  Z/X=adjust step")
if HAS_FOLDER_SOURCE:
    print(f"T=goto frame  R=restart  1/2/3=model  [ / ] prev-next  {video_switch_hint}=video")
else:
    print(f"T=goto frame  R=restart  1/2/3=model  {video_switch_hint}=video")
print("Curvature class: 160~180 STRAIGHT | 130~159 SLIGHT_BEND | <=129 BENT")
print(f"Emotion window: {EMO_WINDOW_SECONDS:.1f}s")
print(f"Min data before decision: {EMO_CONFIRM_SECONDS:.1f}s (show ANALYZING first)")
print(f"Happy rule: Hmed>{EMO_HEIGHT_HAPPY_MIN:.1f}px and TipSpeed<{EMO_VEL_HAPPY_MAX:.1f}%")
print(f"Emotion confirm: any emotion must hold for {EMO_CONFIRM_SECONDS:.1f}s")
print(f"Tail trajectory visual follow: {'ON' if SHOW_TAIL_TRAJECTORY else 'OFF'}")
print(f"Upright straight curvature threshold: >= {CURV_UPRIGHT_STRAIGHT_MIN:.1f} deg")
print(f"Video lock: {'ON (target key='+TARGET_VIDEO_KEY+')' if LOG_ONLY_TARGET_VIDEO else 'OFF'}")
print(f"Video source count: {len(VIDEO_LIST)}")
print(f"Folder source mode: {'ON ([ and ] to switch)' if HAS_FOLDER_SOURCE else 'OFF'}")
print("===============================================\n")


def switch_video_by_index(next_video_index):
    """切換到指定影片索引並重置所有與影片相關的狀態。"""
    global cap, video_index, video_fps, frame_interval, emo_window_len, emotion_confirm_frames
    global emotion_curv_hist, emotion_vel_hist, emotion_osc_hist, emotion_height_hist, emotion_upang_hist
    global emotion_candidate, emotion_candidate_frames, emotion_confirmed
    global prev_tip_pos, prev_angle, frame

    if next_video_index < 0 or next_video_index >= len(VIDEO_LIST):
        return

    if LOG_ONLY_TARGET_VIDEO and next_video_index != TARGET_VIDEO_INDEX:
        print(f"[INFO] Logging lock enabled: only key '{TARGET_VIDEO_KEY}' video is allowed. Switch ignored.")
        return

    video_index = next_video_index
    cap.release()
    cap = load_video(VIDEO_LIST[video_index])
    video_fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else 30
    frame_interval = int(round(video_fps / fps_limit)) if video_fps > fps_limit else 1
    effective_window_seconds = max(EMO_WINDOW_SECONDS, EMO_CONFIRM_SECONDS)
    emo_window_len = max(45, int(max(video_fps, 1.0) * effective_window_seconds))
    emotion_confirm_frames = max(1, int(max(video_fps, 1.0) * EMO_CONFIRM_SECONDS))
    emotion_curv_hist = deque(maxlen=emo_window_len)
    emotion_vel_hist = deque(maxlen=emo_window_len)
    emotion_osc_hist = deque(maxlen=emo_window_len)
    emotion_height_hist = deque(maxlen=emo_window_len)
    emotion_upang_hist = deque(maxlen=emo_window_len)
    emotion_candidate = "ANALYZING"
    emotion_candidate_frames = 0
    emotion_confirmed = "ANALYZING"
    angle_buf.clear(); tail_traj.clear()
    speed_buf.clear(); angvel_buf.clear()
    swing_sign_buf.clear()
    prev_tip_pos = None; prev_angle = None
    frame = None


# ======================================================
#  主迴圈
# ======================================================
fps_limit = 30
video_fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else 30
frame_interval = int(round(video_fps / fps_limit)) if video_fps > fps_limit else 1
frame_counter = 0
effective_window_seconds = max(EMO_WINDOW_SECONDS, EMO_CONFIRM_SECONDS)
emo_window_len = max(45, int(max(video_fps, 1.0) * effective_window_seconds))
emotion_confirm_frames = max(1, int(max(video_fps, 1.0) * EMO_CONFIRM_SECONDS))
emotion_curv_hist = deque(maxlen=emo_window_len)
emotion_vel_hist = deque(maxlen=emo_window_len)
emotion_osc_hist = deque(maxlen=emo_window_len)
emotion_height_hist = deque(maxlen=emo_window_len)
emotion_upang_hist = deque(maxlen=emo_window_len)
emotion_candidate = "ANALYZING"
emotion_candidate_frames = 0
emotion_confirmed = "ANALYZING"

while True:
    if cap is None:
        break

    # ── 讀取影格 ───────────────────────────────────────
    if play_mode:
        # 跳過多餘幀以達到fps限制
        for _ in range(frame_interval):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                angle_buf.clear()
                break
        if not ret:
            continue
        frame = prescale_frame(frame)
    else:
        if frame is None:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = prescale_frame(frame)

    display = frame.copy()

    # ── YOLO 推論 ────────────────────────────────────
    result = model.predict(
        display,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        half=True,
        verbose=False,
    )[0]

    # ── FPS ──────────────────────────────────────────
    now = time.time()
    fps = 1 / max(now - prev_time, 1e-6)
    prev_time = now

    # ── 取第一隻貓的尾巴關鍵點 ───────────────────────
    # 新增尾巴特徵
    tail_curvature = None
    tail_height = None
    tail_up_angle = None
    tail_velocity = None
    tail_osc_hz = None
    happy_hgt_med = None
    happy_upa_med = None
    label, color = "UNKNOWN", COLOR_UNKNOWN
    emotion, emotion_color = emotion_confirmed, get_emotion_color(emotion_confirmed)

    if result.keypoints is not None and len(result.keypoints.xy) > 0:
        kpts     = result.keypoints.xy[0].cpu().numpy()    # (17, 2)
        kpt_conf = result.keypoints.conf[0].cpu().numpy()  # (17,)

        # 尾巴曲率
        tail_curvature = calc_tail_curvature(kpts, kpt_conf)
        # 尾巴高度 / 上舉角
        tail_height = calc_tail_height(kpts, kpt_conf)
        tail_up_angle = calc_tail_up_angle(kpts, kpt_conf)

        # 平滑
        if tail_curvature is not None:
            angle_buf.append(tail_curvature)
            if len(angle_buf) > SMOOTH_N:
                angle_buf.pop(0)
            tail_curvature = float(np.mean(angle_buf))
        else:
            angle_buf.clear()

        label, color = classify_tail(tail_curvature)

        # 方向 & 擎動側
        ang360, dir_label = calc_tail_direction(kpts, kpt_conf)
        side              = calc_body_axis_side(kpts, kpt_conf)

        # 軌跡更新
        if kpt_conf[IDX_TAIL_TIP] > KP_CONF_THRES:
            tip_xy = tuple(kpts[IDX_TAIL_TIP].astype(int))
            tail_traj.append(tip_xy)
        else:
            tip_xy = None

        # 速度（以尾巴長度歸一化）
        body_len  = get_tail_length_ref(kpts, kpt_conf)
        tip_np    = kpts[IDX_TAIL_TIP] if tip_xy else None
        raw_speed = calc_tip_speed(tip_np, prev_tip_pos, body_len)
        if raw_speed is not None:
            speed_buf.append(raw_speed)
            if len(speed_buf) > SPEED_SMOOTH_N:
                speed_buf.pop(0)
            tail_velocity = float(np.mean(speed_buf))
            emotion_vel_hist.append(tail_velocity)
        else:
            tail_velocity = None
        prev_tip_pos = tip_np

        # 擺動頻率（Hz）
        swing_sign = calc_tail_swing_sign(kpts, kpt_conf)
        if swing_sign != 0:
            swing_sign_buf.append(swing_sign)
        tail_osc_hz = calc_tail_oscillation_hz(swing_sign_buf, video_fps if video_fps > 0 else 30.0)
        if tail_osc_hz is not None:
            emotion_osc_hist.append(tail_osc_hz)

        if tail_height is not None:
            emotion_height_hist.append(tail_height)
        if tail_up_angle is not None:
            emotion_upang_hist.append(tail_up_angle)

        # 角速度（度/幀）
        if tail_curvature is not None and prev_angle is not None:
            angvel_buf.append(tail_curvature - prev_angle)
            if len(angvel_buf) > ANGVEL_SMOOTH_N:
                angvel_buf.pop(0)
            smooth_angvel = float(np.mean(angvel_buf))
        else:
            smooth_angvel = None
        prev_angle = tail_curvature

        if tail_curvature is not None:
            emotion_curv_hist.append(tail_curvature)

        # 推論情緒
        if len(emotion_height_hist) > 0:
            happy_hgt_med = float(np.median(np.array(emotion_height_hist, dtype=float)))
        if len(emotion_upang_hist) > 0:
            happy_upa_med = float(np.median(np.array(emotion_upang_hist, dtype=float)))

        candidate_emotion, _ = infer_cat_emotion_window(
            emotion_curv_hist,
            emotion_vel_hist,
            emotion_osc_hist,
            emotion_height_hist,
            emotion_upang_hist,
            video_fps if video_fps > 0 else 30.0,
        )

        # 任一情緒都需要連續維持 emotion_confirm_frames 幀才正式切換
        if candidate_emotion != emotion_candidate:
            emotion_candidate = candidate_emotion
            emotion_candidate_frames = 1
        else:
            emotion_candidate_frames += 1

        if candidate_emotion == "ANALYZING":
            emotion_confirmed = "ANALYZING"
        elif emotion_candidate_frames >= emotion_confirm_frames:
            emotion_confirmed = candidate_emotion

        emotion = emotion_confirmed
        emotion_color = get_emotion_color(emotion_confirmed)

        # 儲存尾巴三點座標時間序列（原始 + 正規化）
        if tail_ts_writer is not None:
            should_log = (not LOG_ONLY_TARGET_VIDEO) or (video_index == TARGET_VIDEO_INDEX)
            if not should_log:
                triplet = None
            else:
                triplet = None
            h, w = display.shape[:2]
            if should_log:
                triplet = extract_tail_triplet(kpts, kpt_conf, w, h)
            if triplet is not None:
                frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                t_sec = frame_id / max(video_fps, 1.0)
                video_key = chr(next((k for k, v in VIDEO_KEY_MAP.items() if v == video_index), ord("?")))
                tail_ts_writer.writerow([
                    frame_id,
                    f"{t_sec:.4f}",
                    model_index,
                    video_index,
                    video_key,
                    VIDEO_LIST[video_index],
                    emotion,
                    label,
                    "xy/frame_wh",
                    f"{triplet['base_x']:.3f}", f"{triplet['base_y']:.3f}",
                    f"{triplet['mid_x']:.3f}", f"{triplet['mid_y']:.3f}",
                    f"{triplet['tip_x']:.3f}", f"{triplet['tip_y']:.3f}",
                    f"{triplet['base_x_norm']:.6f}", f"{triplet['base_y_norm']:.6f}",
                    f"{triplet['mid_x_norm']:.6f}", f"{triplet['mid_y_norm']:.6f}",
                    f"{triplet['tip_x_norm']:.6f}", f"{triplet['tip_y_norm']:.6f}",
                ])

        # 繪製
        draw_tail_visualization(display, kpts, kpt_conf, result, tail_curvature, color,
                                tail_traj, dir_label, ang360)

    # ── 狀態橫幅 ─────────────────────────────────────
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 顯示情緒與尾巴特徵
    draw_status_banner(
        display, label, color, tail_curvature,
        model_index, video_index,
        fps, current_frame, total_frames,
        FRAME_STEP, play_mode,
        dir_label=dir_label if 'dir_label' in locals() else None,
        side=side           if 'side'      in locals() else None,
        speed=tail_velocity if 'tail_velocity' in locals() else None,
        ang_vel=smooth_angvel if 'smooth_angvel' in locals() else None,
        happy_hgt_med=happy_hgt_med,
        happy_upa_med=happy_upa_med,
    )
    # 右下角角落顯示情緒狀態（英文，無底色，置右下）
    h, w = display.shape[:2]
    margin_x, margin_y = 18, 18
    text = f"Emotion: {emotion}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    text_x = w - tw - margin_x
    text_y = h - margin_y
    draw_text_outlined(display, text, text_x, text_y, scale=1.1, thickness=3, fg=emotion_color)

    cv2.imshow("Cat Tail Bend Detector", display)

    # ── 按鍵處理 ─────────────────────────────────────
    delay   = 1 if play_mode else 0
    raw_key = cv2.waitKeyEx(delay)
    key     = raw_key & 0xFF

    if key == ord("q"):
        break

    if HAS_FOLDER_SOURCE and key in (ord("["), ord("]")):
        delta = -1 if key == ord("[") else 1
        next_video_index = (video_index + delta) % len(VIDEO_LIST)
        switch_video_by_index(next_video_index)
        continue

    if key == 32:  # SPACE
        play_mode = not play_mode
        angle_buf.clear()
        print(f"[INFO] {'PLAY' if play_mode else 'PAUSE'}")

    if key == ord("z"):
        FRAME_STEP = max(1, FRAME_STEP - 1)
        print(f"[INFO] STEP={FRAME_STEP}")

    if key == ord("x"):
        FRAME_STEP = min(200, FRAME_STEP + 1)
        print(f"[INFO] STEP={FRAME_STEP}")

    if key == ord("r"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        angle_buf.clear()
        tail_traj.clear()
        speed_buf.clear()
        angvel_buf.clear()
        swing_sign_buf.clear()
        emotion_curv_hist.clear()
        emotion_vel_hist.clear()
        emotion_osc_hist.clear()
        emotion_height_hist.clear()
        emotion_upang_hist.clear()
        emotion_candidate = "ANALYZING"
        emotion_candidate_frames = 0
        emotion_confirmed = "ANALYZING"
        prev_tip_pos = None
        prev_angle   = None
        frame = None
        print("[INFO] Restarted")

    if key == ord("t"):
        play_mode = False
        try:
            target = int(input("\n[INPUT] Jump to frame: "))
            frame = jump_to_frame(cap, target)
            angle_buf.clear()
        except Exception:
            print("[ERROR] Invalid input")

    if not play_mode:
        if key == ord("a"):
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            frame = jump_to_frame(cap, pos - FRAME_STEP)
            angle_buf.clear(); tail_traj.clear()
            speed_buf.clear(); angvel_buf.clear()
            swing_sign_buf.clear()
            emotion_curv_hist.clear(); emotion_vel_hist.clear(); emotion_osc_hist.clear()
            emotion_height_hist.clear(); emotion_upang_hist.clear()
            emotion_candidate = "ANALYZING"; emotion_candidate_frames = 0; emotion_confirmed = "ANALYZING"
            prev_tip_pos = None; prev_angle = None
        elif key == ord("d"):
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            frame = jump_to_frame(cap, pos + FRAME_STEP)
            angle_buf.clear(); tail_traj.clear()
            speed_buf.clear(); angvel_buf.clear()
            swing_sign_buf.clear()
            emotion_curv_hist.clear(); emotion_vel_hist.clear(); emotion_osc_hist.clear()
            emotion_height_hist.clear(); emotion_upang_hist.clear()
            emotion_candidate = "ANALYZING"; emotion_candidate_frames = 0; emotion_confirmed = "ANALYZING"
            prev_tip_pos = None; prev_angle = None

    # 切換模型
    if key in [ord("1"), ord("2"), ord("3")]:
        idx = int(chr(key)) - 1
        if idx < len(MODEL_LIST):
            model_index = idx
            model = load_model(MODEL_LIST[model_index])
            angle_buf.clear()

    # 切換影片
    if key in VIDEO_KEY_MAP:
        switch_video_by_index(VIDEO_KEY_MAP[key])

# ── 結束 ──────────────────────────────────────────────
if cap is not None:
    cap.release()
if tail_ts_file is not None:
    tail_ts_file.close()
cv2.destroyAllWindows()
