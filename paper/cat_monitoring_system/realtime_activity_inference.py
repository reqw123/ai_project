"""
即時影片推論：Activity Score 視覺化

將 visualize_activity_score.py 的合成資料改為真實影片推論。
在 OpenCV 視窗即時顯示：位移量、activity_value、activity_score、行為標籤。
推論結束後自動儲存：逐幀 CSV、摘要 TXT、時序 PNG。

執行：
    python realtime_activity_inference.py
    python realtime_activity_inference.py --video path/to/video.mp4
"""

import sys
import os
import csv
import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# 把 cat_monitoring_system 目錄和 paper/ 根目錄都加入路徑，確保 config.py 可以 import
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO
from config import BehaviorTrackingConfig as _BTConfig
from config import AnomalyDetectionConfig as _ADConfig
from config import STGCNConfig as _STGCNConfig
from utils.constants import LOW_CONF_ID

# ST-GCN 依賴為可選，若模型不存在或路徑空則自動跳過
_GCN_AVAILABLE = False
try:
    from detectors.behavior_classifier import BehaviorClassifier
    from models.stgcn_model import (
        interpolate_missing,
        flip_normalize,
        orientation_normalize,
        normalize_skeleton_coords,
        build_feature_tensor,
    )
    _GCN_AVAILABLE = True
except Exception as _e:
    print(f"[WARN] ST-GCN 模組載入失敗（僅顯示 YOLO 結果）: {_e}")


# ══════════════════════════════════════════════════════════════════════
# 路徑設定
# ══════════════════════════════════════════════════════════════════════

DEFAULT_VIDEO      = r"C:\Users\homec\OneDrive\圖片\貓咪\自行拍攝\影片"
DEFAULT_YOLO       = r"C:\ai_project\cat_pose\v11s_91.pt"
DEFAULT_STGCN      = r"C:\Users\homec\Downloads\stgcn_best_022_xy_v_att_on.pth"
OUTPUT_BASE        = Path(__file__).parent / "activity_inference_logs"
SAVE_OUTPUT_VIDEO  = True   # 是否同步存下疊加 overlay 的影片

# ══════════════════════════════════════════════════════════════════════
# 演算法參數與門檻說明
# ══════════════════════════════════════════════════════════════════════

# ── YOLO ──────────────────────────────────────────────────────────────
YOLO_IMGSZ    = 640
# [門檻] YOLO 偵測信心門檻
# 降低 → 偵測率上升但誤框增加；升高 → 漏偵測增加
YOLO_CONF     = 0.50

# [門檻] 有效關鍵點信心門檻（參與位移計算）
# 降低 → 更多低信心關鍵點納入計算，position noise 可能拉高位移讀值
# 升高 → 只保留高信心點，計算更可靠但 active_kpts 數量下降
KP_CONF_THRES = float(_ADConfig.KP_CONF_THRES)

# ── 關鍵點座標 EMA（對應 STGCNConfig.KP_EMA_ALPHA）─────────────────────
# [門檻] 關鍵點座標的指數移動平均係數
# 1.0 = 完全不平滑（用原始偵測值）；0.5 = 強平滑（位移計算更穩，但骨架顯示有延遲）
# ⚠ 此值與訓練時 train_gcn.py 的 KP_EMA_ALPHA 必須一致，否則 ST-GCN 輸入分佈會偏移
EMA_ALPHA     = float(_STGCNConfig.KP_EMA_ALPHA)

# ── 位移前置平滑（原系統沒有，此腳本額外加入）────────────────────────────
# [門檻] 對計算出的逐幀位移量再做一層 EMA，然後才轉換成 activity_value
# 作用：消除骨架偵測噪音造成的瞬間大位移，讓 activity_value 曲線更平穩
# 1.0  = 不平滑（原始行為，activity_score 容易不穩定）
# 0.4  = 建議值：消除一半以上的逐幀抖動，同時保留真實動作的響應
# 0.15 = 強平滑：曲線非常平穩，但對突發動作的反應會延遲約 3-5 幀
DISP_EMA_ALPHA = 0.4

# ── 位移正規化方式 ─────────────────────────────────────────────────────
# [門檻] 是否以身體尺度（chest→hip 距離）正規化位移量
# True  = body-scale 正規化（推薦）：動作幅度不受貓咪距離鏡頭遠近影響
#         activity_value = clamp(disp / body_scale / MAX_MOTION_RATIO, 0, 1) × 100
# False = 固定像素正規化（原 anomaly_detector 設計）：
#         activity_value = clamp(disp / MAX_MOTION, 0, 1) × 100
#         ⚠ 近距離影片 disp 容易超過 20px → activity_value 長期卡在 100
NORMALIZE_BY_BODY_SCALE = True

# [門檻] body-scale 正規化時的最大位移比率（NORMALIZE_BY_BODY_SCALE=True 時有效）
# 含義：單幀平均位移 ≥ body_scale × MAX_MOTION_RATIO → activity_value = 100
# 建議範圍：
#   0.05 ~ 0.10 = 高靈敏度（靜態監控，微小動作也要顯示）
#   0.12 ~ 0.18 = 中等靈敏度（一般行走/搔抓場景，推薦預設）
#   0.20 ~ 0.35 = 低靈敏度（高動態場景，防止長期過高）
# 降低 → 更敏感（同樣動作得到更高的 activity_value，score 容易偏高）
# 升高 → 更遲鈍（不容易過高，但小動作可能顯示不出來）
MAX_MOTION_RATIO = 0.15

# [門檻] 固定像素正規化時的最大位移量（NORMALIZE_BY_BODY_SCALE=False 時有效）
# 原始 anomaly_detector 預設值：20px
# ⚠ 對近距離影片（貓咪佔畫面較大比例）容易讓 activity_value 長期為 100
MAX_MOTION    = float(_ADConfig.MAX_MOTION)

# [門檻] 異常位移門檻（像素，NORMALIZE_BY_BODY_SCALE=False 時有效）
# 原始設計：ema_motion > 0.2px → abnormal=True（閾值非常低，幾乎每幀都觸發）
# ⚠ 此旗標目前只寫入 CSV，未接入任何警報邏輯
ABNORMAL_THRES_PX = float(getattr(_ADConfig, 'ABNORMAL_THRESHOLD', 0.2))

# ── activity_score 滾動窗口 ─────────────────────────────────────────────
# [門檻] 滾動平均的時間窗口長度（秒）
# 越短 → 反應越快、起伏越大（1~2s 視窗下，瞬間動作就能讓 score 衝高）
# 越長 → 越平穩，但對真正的活動變化反應也越慢
# 原系統設定：3.0s（對活躍貓咪仍偏短，若不穩可調到 5.0~8.0）
ACTIVITY_WIN_SEC  = float(_BTConfig.ACTIVITY_SCORE_WINDOW_SECONDS)

# [門檻] 低信心幀在 activity_score 中的 weight
# 含義：GCN confidence < GCN_CONF_THRES 的幀，其 activity_value 以此 weight 計入加權平均
# 降低 → 低信心幀對 score 貢獻更小（分數更保守）
LOW_CONF_WEIGHT   = float(_BTConfig.LOW_CONFIDENCE_ACTIVITY_WEIGHT)

# ── ST-GCN ────────────────────────────────────────────────────────────
# [門檻] ST-GCN 輸出信心門檻
# 低於此值 → behavior_id = LOW_CONF_ID(-1)，不顯示行為標籤
# 升高 → 標籤更嚴格，LOW CONF 出現頻率上升
GCN_CONF_THRES    = float(_BTConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD)

SEQUENCE_LENGTH   = int(_STGCNConfig.SEQUENCE_LENGTH)
CLASSIFY_STRIDE   = 2   # 每 N 幀做一次 GCN 推論（降低 GPU 負載）

# ══════════════════════════════════════════════════════════════════════
# 顯示設定
# ══════════════════════════════════════════════════════════════════════

VIDEO_AREA_W   = 720    # 影片顯示區域寬度
INFO_PANEL_W   = 260    # 右側資訊面板寬度
DISPLAY_W      = VIDEO_AREA_W + INFO_PANEL_W   # = 980
DISPLAY_H      = 480    # 影片顯示高度
CHART_H        = 130    # 底部滾動折線圖高度
CANVAS_H       = DISPLAY_H + CHART_H           # 總高度
CHART_BUF      = 240    # 折線圖保留的歷史幀數（對應約 8 秒@30fps）
WINDOW_NAME    = "Cat Activity Monitor"

# ══════════════════════════════════════════════════════════════════════
# 行為標籤與顏色（英文標籤）
# ══════════════════════════════════════════════════════════════════════

# 所有文字輸出一律使用英文
BEHAVIOR_LABEL = {
     0: "WALK",
     1: "LICK",
     2: "SCRATCH",
     3: "SHAKE",
     4: "STOP",
    -1: "LOW CONF",
    -2: "NOT DETECTED",
}

# BGR 顏色（OpenCV 格式）
BEHAVIOR_COLOR = {
     0: (80,  200, 80),    # green
     1: (200, 160, 80),    # cyan-orange
     2: (80,  80,  220),   # red
     3: (80,  180, 220),   # yellow-orange
     4: (140, 140, 140),   # gray
    -1: (80,  80,  80),
    -2: (50,  50,  50),
}

# 貓咪骨架連線（17 關鍵點版本）
# kpt index：0=nose 1=L_ear 2=R_ear 3=chest 4=mid_back 5=hip
#             6=LF_elbow 7=LF_paw 8=RF_elbow 9=RF_paw
#             10=LH_knee 11=LH_paw 12=RH_knee 13=RH_paw
#             14=tail_root 15=tail_mid 16=tail_tip
SKELETON_LINKS = [
    (0, 1), (0, 2),              # nose → L_ear / R_ear
    (0, 3),                      # nose → chest（頭部與身體的連接）
    (3, 4), (4, 5),              # chest → mid_back → hip（脊椎）
    (3, 6), (6, 7),              # chest → LF_elbow → LF_paw
    (3, 8), (8, 9),              # chest → RF_elbow → RF_paw
    (5, 10), (10, 11),           # hip → LH_knee → LH_paw
    (5, 12), (12, 13),           # hip → RH_knee → RH_paw
    (5, 14), (14, 15), (15, 16), # hip → tail_root → tail_mid → tail_tip
]

# ══════════════════════════════════════════════════════════════════════
# 演算法複製（對應 anomaly_detector + behavior_tracker）
# ══════════════════════════════════════════════════════════════════════

def displacement_to_activity_value(disp_px: float, body_scale: float = 0.0) -> int:
    """
    將平滑後的位移量轉換為 activity_value（0-100）。

    NORMALIZE_BY_BODY_SCALE=True（推薦）：
        norm = disp_px / body_scale / MAX_MOTION_RATIO
        → 不受貓咪距離鏡頭遠近影響

    NORMALIZE_BY_BODY_SCALE=False（原始設計）：
        norm = disp_px / MAX_MOTION
        → 近距離影片容易長期卡在 100

    [門檻] MAX_MOTION_RATIO / MAX_MOTION：決定何種動作幅度才算「滿分」
    """
    if NORMALIZE_BY_BODY_SCALE and body_scale > 1.0:
        norm = disp_px / max(body_scale * MAX_MOTION_RATIO, 1e-6)
    else:
        norm = disp_px / max(MAX_MOTION, 1e-6)
    return int(min(norm, 1.0) * 100)


class ActivityScoreTracker:
    """
    對應 behavior_tracker.get_activity_score() 的滾動計算器。

    [可評估] activity_score 是 ACTIVITY_WIN_SEC 秒內的加權平均，
    可用來比較不同 window 長度對平滑度與靈敏度的取捨。
    """
    def __init__(self):
        # 每筆：{"t": float, "av": int, "w": float}
        self._window: list = []

    def update(self, t: float, av: int, weight: float = LOW_CONF_WEIGHT) -> int:
        self._window.append({"t": t, "av": av, "w": weight})
        # 清除超過時間窗的舊資料
        self._window = [e for e in self._window if (t - e["t"]) < ACTIVITY_WIN_SEC]
        tw = sum(e["w"] for e in self._window)
        if tw < 1e-9:
            return 50
        sc = round(sum(e["av"] * e["w"] for e in self._window) / tw)
        return max(0, min(100, sc))


# ══════════════════════════════════════════════════════════════════════
# OpenCV 繪圖工具
# ══════════════════════════════════════════════════════════════════════

_FONT = cv2.FONT_HERSHEY_SIMPLEX

def _text(img, txt, pos, scale=0.5, color=(220, 220, 220), thick=1):
    """cv2.putText 的簡短包裝"""
    cv2.putText(img, txt, pos, _FONT, scale, color, thick, cv2.LINE_AA)


def _text_bg(img, txt, pos, scale=0.45, color=(220, 220, 220), bg=(20, 30, 50)):
    """帶半透明背景的文字（避免與影像混淆）"""
    (w, h), bl = cv2.getTextSize(txt, _FONT, scale, 1)
    x, y = pos
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 2, y - h - 2), (x + w + 2, y + bl + 2), bg, -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.putText(img, txt, pos, _FONT, scale, color, 1, cv2.LINE_AA)


def draw_bar(img, x, y, w, h, value, max_val=100,
             bar_color=(80, 200, 80), bg_color=(30, 40, 55), show_val=True):
    """橫條圖：顯示 0-max_val 的比率"""
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    filled = max(0, int(value / max(max_val, 1) * w))
    cv2.rectangle(img, (x, y), (x + filled, y + h), bar_color, -1)
    # 邊框
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 70, 90), 1)
    if show_val:
        _text(img, str(int(value)), (x + w + 6, y + h - 1), scale=0.42, color=(220, 220, 220))


def draw_skeleton(img, kpts, kpt_conf, scale_xy=(1.0, 1.0)):
    """
    在影像上繪製貓咪骨架。

    [可評估] 骨架繪製質量可用來目測 YOLO 關鍵點是否穩定，
    搭配 jitter 指標使用（compare_pose_jitter_models.py）
    """
    ox, oy = 0, 0  # 偏移（當影片縮放到 VIDEO_AREA_W 時自動計算）
    sw, sh = scale_xy

    def kp(i):
        x = int(kpts[i][0] * sw)
        y = int(kpts[i][1] * sh)
        return (x, y), kpt_conf[i] > KP_CONF_THRES

    # 骨架連線
    for a, b in SKELETON_LINKS:
        pa, ca = kp(a)
        pb, cb = kp(b)
        if ca and cb:
            cv2.line(img, pa, pb, (60, 220, 60), 1, cv2.LINE_AA)

    # 關鍵點圓點
    for i in range(len(kpts)):
        pt, valid = kp(i)
        if valid:
            cv2.circle(img, pt, 3, (0, 240, 240), -1, cv2.LINE_AA)


def draw_rolling_chart(canvas, av_buf, score_buf, x, y, w, h):
    """
    在 canvas 上繪製最近 CHART_BUF 幀的 activity_value 和 activity_score 折線圖。

    [可評估]
    - av（藍線）：瞬時位移量，反映動作幅度
    - score（橙線）：滾動平均，反映整體活動趨勢
    - 兩線差距越大 → 位移變化越劇烈（score 追不上 av）
    """
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:] = (18, 28, 45)

    # 格線 25 / 50 / 75 / 100
    for pct, alpha in [(25, 30), (50, 60), (75, 30), (100, 30)]:
        gy = h - 1 - int(pct / 100 * (h - 2))
        cv2.line(bg, (0, gy), (w, gy), (50, 65, 90), 1)
        _text(bg, str(pct), (2, gy - 2), scale=0.30, color=(80, 95, 110))

    n = len(av_buf)
    if n < 2:
        canvas[y:y + h, x:x + w] = bg
        return

    def val_to_y(v):
        return h - 1 - max(0, min(h - 2, int(v / 100 * (h - 2))))

    def idx_to_x(i, total):
        return max(0, min(w - 1, int(i / max(total - 1, 1) * (w - 1))))

    # activity_value（天藍色）
    pts_av = [(idx_to_x(i, n), val_to_y(v)) for i, v in enumerate(av_buf)]
    for i in range(len(pts_av) - 1):
        cv2.line(bg, pts_av[i], pts_av[i + 1], (56, 189, 248), 1, cv2.LINE_AA)

    # activity_score（橙色，較粗）
    m = len(score_buf)
    pts_sc = [(idx_to_x(i, m), val_to_y(v)) for i, v in enumerate(score_buf)]
    for i in range(len(pts_sc) - 1):
        cv2.line(bg, pts_sc[i], pts_sc[i + 1], (249, 115, 22), 2, cv2.LINE_AA)

    # 標籤
    _text(bg, "ACTIVITY VALUE", (6, 14), scale=0.36, color=(56, 189, 248))
    _text(bg, "ACTIVITY SCORE", (6, 28), scale=0.36, color=(249, 115, 22))
    _text(bg, f"← {int(CHART_BUF / 30)}s", (w - 50, h - 5), scale=0.32, color=(80, 95, 110))

    canvas[y:y + h, x:x + w] = bg


def draw_info_panel(canvas, x, y, w, h,
                    behavior_id, confidence, av, score,
                    abnormal, body_scale, active_kpts,
                    frame_idx, detection_rate, mean_conf_yolo):
    """
    在畫布右側繪製資訊面板。

    [可評估] 面板顯示的每個指標都可作為系統評估維度：
    - BEHAVIOR + CONFIDENCE：GCN 有效率和分類準確度
    - ACTIVITY VALUE：即時反應速度
    - ACTIVITY SCORE：長期趨勢穩定性
    - ABNORMAL：位移異常幀比率（尚未接入警報系統）
    - BODY SCALE：YOLO 偵測的身體大小，正規化分母
    - ACTIVE KPTS：有效關鍵點比率，影響 activity_value 的可靠性
    - DET RATE：YOLO 偵測率，反映模型在此場景的覆蓋能力
    """
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (15, 22, 38)

    b_color = BEHAVIOR_COLOR.get(behavior_id, (120, 120, 120))
    b_label = BEHAVIOR_LABEL.get(behavior_id, "UNKNOWN")

    cy = 20

    # ── 標題列 ────────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (w, 36), (22, 35, 60), -1)
    _text(panel, "CAT ACTIVITY MONITOR", (8, 24), scale=0.48,
          color=(180, 200, 240), thick=1)

    cy = 54
    # ── BEHAVIOR ─────────────────────────────────────────────────────
    _text(panel, "BEHAVIOR", (8, cy), scale=0.38, color=(120, 140, 170))
    cy += 18
    cv2.rectangle(panel, (8, cy - 14), (w - 8, cy + 4), b_color, -1)
    _text(panel, b_label, (12, cy), scale=0.55, color=(10, 10, 10), thick=2)
    cy += 16
    _text(panel, f"CONFIDENCE  {int(confidence * 100)}%", (8, cy),
          scale=0.38, color=(160, 180, 200))

    cy += 20
    cv2.line(panel, (8, cy), (w - 8, cy), (40, 55, 75), 1)
    cy += 12

    # ── ACTIVITY VALUE ────────────────────────────────────────────────
    _text(panel, "ACTIVITY VALUE", (8, cy), scale=0.38, color=(56, 189, 248))
    cy += 14
    draw_bar(panel, 8, cy, w - 50, 12, av, bar_color=(56, 189, 248))
    cy += 22

    # ── ACTIVITY SCORE ────────────────────────────────────────────────
    _text(panel, f"ACTIVITY SCORE  ({ACTIVITY_WIN_SEC:.0f}s avg)", (8, cy),
          scale=0.38, color=(249, 115, 22))
    cy += 14
    draw_bar(panel, 8, cy, w - 50, 12, score, bar_color=(249, 115, 22))
    cy += 22

    # ── ABNORMAL ──────────────────────────────────────────────────────
    _text(panel, "ABNORMAL", (8, cy), scale=0.38, color=(120, 140, 170))
    ab_color = (60, 60, 220) if abnormal else (60, 180, 60)
    ab_text  = "YES" if abnormal else "NO"
    _text(panel, ab_text, (w - 55, cy), scale=0.45, color=ab_color, thick=2)
    cy += 20

    cv2.line(panel, (8, cy), (w - 8, cy), (40, 55, 75), 1)
    cy += 12

    # ── 次要指標 ──────────────────────────────────────────────────────
    def _stat(label, val_str):
        nonlocal cy
        _text(panel, label, (8, cy), scale=0.35, color=(100, 120, 150))
        _text(panel, val_str, (w - 10 - len(val_str) * 7, cy),
              scale=0.38, color=(200, 215, 230))
        cy += 16

    _stat("BODY SCALE",  f"{body_scale:.1f} px")  # [可評估] 偵測距離一致性
    _stat("ACTIVE KPTS", f"{active_kpts}/17")      # [可評估] 關鍵點可見率
    _stat("DET RATE",    f"{detection_rate:.1f}%") # [可評估] YOLO 偵測率
    _stat("YOLO CONF",   f"{mean_conf_yolo:.2f}")  # [可評估] YOLO 平均信心
    _stat("FRAME",       f"#{frame_idx}")

    canvas[y:y + h, x:x + w] = panel


# ══════════════════════════════════════════════════════════════════════
# 報告儲存
# ══════════════════════════════════════════════════════════════════════

def save_report(output_dir: Path, records: list, video_path: str,
                model_yolo: str, model_stgcn: str):
    """
    推論結束後儲存三份報告：
      frame_data.csv      逐幀原始資料
      summary.txt         整體統計摘要
      activity_timeseries.png  完整時序圖（matplotlib）
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 逐幀 CSV ───────────────────────────────────────────────────
    csv_path = output_dir / "frame_data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        # [可評估] 每欄都是可分析的評估維度
        w.writerow([
            "frame_idx", "timestamp_sec",
            "detected",          # YOLO 是否偵測到貓
            "body_scale_px",     # [可評估] 身體比例，正規化基準
            "active_kpts",       # [可評估] 有效關鍵點數
            "mean_kpt_conf",     # [可評估] YOLO 關鍵點平均信心
            "displacement_px",   # [可評估] 原始像素位移量
            "activity_value",    # [可評估] 正規化位移 0-100
            "activity_score",    # [可評估] 3s 滾動平均
            "abnormal",          # [可評估] 是否超過異常閾值
            "behavior_id",       # [可評估] ST-GCN 行為 ID
            "behavior_label",    # 英文行為名稱
            "gcn_confidence",    # [可評估] ST-GCN 信心值
        ])
        for r in records:
            w.writerow([
                r["frame_idx"], f"{r['ts']:.3f}",
                int(r["detected"]),
                f"{r['body_scale']:.2f}",
                r["active_kpts"],
                f"{r['mean_kpt_conf']:.4f}",
                f"{r['disp_px']:.4f}",
                r["av"],
                r["score"],
                int(r["abnormal"]),
                r["behavior_id"],
                BEHAVIOR_LABEL.get(r["behavior_id"], "UNKNOWN"),
                f"{r['gcn_conf']:.4f}",
            ])

    # ── 2. 文字摘要 ───────────────────────────────────────────────────
    det_frames   = [r for r in records if r["detected"]]
    n_total      = len(records)
    n_det        = len(det_frames)
    avs          = [r["av"]    for r in det_frames] or [0]
    scores       = [r["score"] for r in records]    or [0]
    disps        = [r["disp_px"] for r in det_frames] or [0]
    abnorm_cnt   = sum(1 for r in det_frames if r["abnormal"])

    # 行為分佈（只計 GCN 有效幀）
    from collections import Counter
    behavior_counts = Counter(
        r["behavior_id"] for r in records
        if r["behavior_id"] >= 0
    )

    lines = [
        "=" * 60,
        "  CAT ACTIVITY INFERENCE SUMMARY REPORT",
        "=" * 60,
        f"  Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Video        : {Path(video_path).name}",
        f"  YOLO Model   : {Path(model_yolo).name}",
        f"  ST-GCN Model : {Path(model_stgcn).name if model_stgcn else 'N/A'}",
        "",
        "  ── Frame Stats ──────────────────────────────────────",
        f"  Total frames     : {n_total}",
        f"  Detected frames  : {n_det}  ({n_det/max(n_total,1)*100:.1f}%)",
        f"  Missing frames   : {n_total - n_det}  ({(n_total-n_det)/max(n_total,1)*100:.1f}%)",
        f"  Abnormal frames  : {abnorm_cnt}  ({abnorm_cnt/max(n_det,1)*100:.1f}% of detected)",
        "",
        "  ── Activity Value  (instantaneous, 0-100) ───────────",
        f"  Mean             : {np.mean(avs):.2f}",
        f"  Std              : {np.std(avs):.2f}",
        f"  P95              : {np.percentile(avs, 95):.2f}",
        f"  Max              : {np.max(avs):.2f}",
        "",
        "  ── Activity Score  ({:.0f}s rolling avg, 0-100) ───────".format(ACTIVITY_WIN_SEC),
        f"  Mean             : {np.mean(scores):.2f}",
        f"  Std              : {np.std(scores):.2f}",
        f"  P95              : {np.percentile(scores, 95):.2f}",
        "",
        "  ── Displacement (px, before normalisation) ──────────",
        f"  Mean             : {np.mean(disps):.2f}",
        f"  Std              : {np.std(disps):.2f}",
        f"  Max              : {np.max(disps):.2f}",
        f"  ABNORMAL_THRES   : {ABNORMAL_THRES_PX:.2f} px  (used for abnormal flag)",
        f"  MAX_MOTION       : {MAX_MOTION:.1f} px  (activity_value = 100 at this disp)",
        "",
        "  ── Behavior Distribution (GCN valid frames) ─────────",
    ]
    gcn_total = sum(behavior_counts.values())
    for bid, cnt in sorted(behavior_counts.items()):
        pct = cnt / max(gcn_total, 1) * 100
        bar = "█" * int(pct / 5)
        lines.append(f"  {BEHAVIOR_LABEL.get(bid,'?'):<12} : {cnt:>5} frames  {pct:5.1f}%  {bar}")
    lines += ["", "=" * 60]

    txt_path = output_dir / "summary.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    # ── 3. 時序圖（matplotlib）────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        ts_all   = [r["ts"]    for r in records]
        av_all   = [r["av"]    for r in records]
        sc_all   = [r["score"] for r in records]
        bid_all  = [r["behavior_id"] for r in records]
        conf_all = [r["gcn_conf"]    for r in records]

        fig = plt.figure(figsize=(16, 9))
        fig.patch.set_facecolor("#0f172a")
        gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.35,
                               top=0.93, bottom=0.07, left=0.07, right=0.97,
                               height_ratios=[1.2, 1.2, 0.6])

        def _ax(idx):
            ax = fig.add_subplot(gs[idx])
            ax.set_facecolor("#1e293b")
            for sp in ax.spines.values():
                sp.set_color("#334155")
            ax.tick_params(colors="#94a3b8", labelsize=8)
            return ax

        # Panel 1: activity_value + activity_score
        ax1 = _ax(0)
        ax1.plot(ts_all, av_all, color="#38bdf8", lw=1.2, label="activity_value", alpha=0.8)
        ax1.plot(ts_all, sc_all, color="#f97316", lw=2.0, label="activity_score")
        ax1.axhline(50, color="#475569", ls=":", lw=0.8)
        ax1.set_ylabel("Value (0-100)", color="#94a3b8", fontsize=9)
        ax1.set_title("Activity Value & Score Timeline", color="white", fontsize=10, loc="left")
        ax1.legend(fontsize=8, facecolor="#0f172a", labelcolor="#94a3b8", edgecolor="#334155")
        ax1.set_ylim(-2, 105)
        ax1.tick_params(labelbottom=False)

        # Panel 2: displacement + abnormal flags
        ax2 = _ax(1)
        ax2.plot(ts_all, [r["disp_px"] for r in records],
                 color="#a78bfa", lw=1.2, label="displacement (px)", alpha=0.85)
        ax2.axhline(ABNORMAL_THRES_PX, color="#ef4444", ls="--", lw=0.9,
                    label=f"abnormal threshold ({ABNORMAL_THRES_PX}px)")
        ax2.axhline(MAX_MOTION, color="#fbbf24", ls="--", lw=0.8,
                    label=f"MAX_MOTION ({MAX_MOTION}px)")
        # 標記異常幀
        for r in records:
            if r["abnormal"]:
                ax2.axvline(r["ts"], color="#ef4444", lw=0.4, alpha=0.3)
        ax2.set_ylabel("Displacement (px)", color="#94a3b8", fontsize=9)
        ax2.set_title("Raw Displacement  +  Abnormal Frames", color="white", fontsize=10, loc="left")
        ax2.legend(fontsize=8, facecolor="#0f172a", labelcolor="#94a3b8", edgecolor="#334155")
        ax2.tick_params(labelbottom=False)

        # Panel 3: behavior timeline（色帶）
        ax3 = _ax(2)
        for r in records:
            color_rgb = BEHAVIOR_COLOR.get(r["behavior_id"], (80, 80, 80))
            # BGR → RGB
            color_mpl = (color_rgb[2]/255, color_rgb[1]/255, color_rgb[0]/255)
            ax3.axvspan(r["ts"], r["ts"] + 1/30, ymin=0, ymax=1,
                        color=color_mpl, alpha=0.85)
        ax3.set_ylabel("Behavior", color="#94a3b8", fontsize=9)
        ax3.set_xlabel("Time (s)", color="#94a3b8", fontsize=9)
        ax3.set_title("Behavior Timeline", color="white", fontsize=10, loc="left")
        ax3.set_yticks([])
        # 圖例
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=(c[2]/255, c[1]/255, c[0]/255), label=BEHAVIOR_LABEL[bid])
            for bid, c in BEHAVIOR_COLOR.items() if bid >= 0
        ]
        ax3.legend(handles=legend_handles, fontsize=7, facecolor="#0f172a",
                   labelcolor="#94a3b8", edgecolor="#334155",
                   loc="upper right", ncol=5)

        fig.suptitle(
            f"Activity Inference Report  ·  {Path(video_path).name}",
            color="white", fontsize=12, fontweight="bold",
        )
        png_path = output_dir / "activity_timeseries.png"
        plt.savefig(png_path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[REPORT] 時序圖: {png_path}")
    except Exception as e:
        print(f"[WARN] matplotlib 圖表儲存失敗: {e}")

    print(f"[REPORT] CSV   : {csv_path}")
    print(f"[REPORT] 摘要  : {txt_path}")
    return output_dir


# ══════════════════════════════════════════════════════════════════════
# 主推論迴圈
# ══════════════════════════════════════════════════════════════════════

def load_models(yolo_path: str, stgcn_path: str, device: str):
    """模型只在啟動時載入一次，切換影片時重複使用。"""
    print(f"[INIT] 載入 YOLO: {yolo_path}")
    yolo = YOLO(yolo_path)
    classifier = None
    if _GCN_AVAILABLE and stgcn_path and Path(stgcn_path).is_file():
        print(f"[INIT] 載入 ST-GCN: {stgcn_path}")
        try:
            classifier = BehaviorClassifier(stgcn_path, device=device)
        except Exception as e:
            print(f"[WARN] ST-GCN 初始化失敗: {e}")
    return yolo, classifier


def run(video_path: str, yolo, classifier, device: str,
        yolo_path: str = "", stgcn_path: str = "",
        video_idx: int = 0, video_total: int = 1):
    """
    推論單支影片。

    Returns
    -------
    "prev"  : 使用者按 1 → 切換到上一支影片
    "next"  : 使用者按 2 → 切換到下一支影片
    "quit"  : 使用者按 q/ESC → 結束整批
    """

    # ── 開啟影片 ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片: {video_path}")

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx        = VIDEO_AREA_W / src_w   # 水平縮放比例
    sy        = DISPLAY_H   / src_h   # 垂直縮放比例

    # ── 輸出目錄：時間戳 + 影片檔名，批次處理時各自獨立 ──────────────
    ts_tag     = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = Path(video_path).stem
    output_dir = OUTPUT_BASE / f"{ts_tag}_{video_stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    vout = None
    if SAVE_OUTPUT_VIDEO:
        vout_path = output_dir / "output.mp4"
        vout = cv2.VideoWriter(
            str(vout_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            src_fps, (DISPLAY_W, CANVAS_H),
        )

    # ── 狀態變數 ──────────────────────────────────────────────────────
    score_tracker  = ActivityScoreTracker()
    kpts_buffer    = deque(maxlen=SEQUENCE_LENGTH)  # ST-GCN 輸入 buffer
    av_history     = deque(maxlen=CHART_BUF)        # 折線圖歷史
    score_history  = deque(maxlen=CHART_BUF)

    prev_kpts      = None   # 上一幀關鍵點（用於位移計算）
    ema_kpts       = None   # EMA 平滑後的關鍵點（用於顯示）
    disp_ema       = 0.0    # 位移前置 EMA：消除骨架偵測噪音造成的瞬間大位移
    behavior_id    = LOW_CONF_ID
    gcn_conf       = 0.0
    infer_count    = 0      # 已處理幀數（含 detected + missed）
    detect_count   = 0      # 成功偵測幀數
    conf_sum       = 0.0
    conf_n         = 0

    records: list = []      # 完整逐幀記錄（供報告用）
    _nav    = "next"        # 預設：影片自然播完 → 前往下一支

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, CANVAS_H)

    use_half = device.lower().startswith("cuda")
    t_start  = time.time()
    paused   = False        # 空白鍵切換暫停/播放
    canvas   = np.zeros((CANVAS_H, DISPLAY_W, 3), dtype=np.uint8)  # 暫停時保持最後一幀

    while True:
        # ── 暫停時不讀新幀，只等待按鍵 ──────────────────────────────
        if paused:
            # 在導覽列上疊加 PAUSED 提示
            pause_overlay = canvas.copy()
            cv2.rectangle(pause_overlay, (0, 0), (VIDEO_AREA_W, 20), (10, 16, 30), -1)
            _text(pause_overlay,
                  f"VIDEO {video_idx + 1}/{video_total}: {Path(video_path).name}"
                  f"  [SPACE] RESUME  [1] PREV  [2] NEXT  [Q] QUIT  ⏸ PAUSED",
                  (6, 14), scale=0.38, color=(240, 200, 60))
            cv2.imshow(WINDOW_NAME, pause_overlay)
            key = cv2.waitKey(30)
            if key == ord(" "):
                paused = False
            elif key == ord("q") or key == 27:
                _nav = "quit"; break
            elif key == ord("1"):
                _nav = "prev"; break
            elif key == ord("2"):
                _nav = "next"; break
            continue

        ret, frame = cap.read()
        if not ret:
            # 影片播完，重新回到開頭繼續循環（按 q 結束）
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # 重置跨幀姿態狀態，避免結尾→開頭的跳接造成虛假大位移
            prev_kpts = None
            ema_kpts  = None
            disp_ema  = 0.0
            kpts_buffer.clear()
            continue

        t_now = infer_count / src_fps

        # ── YOLO 推論 ─────────────────────────────────────────────────
        result   = yolo.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,
                                half=use_half, verbose=False, device=device)[0]
        detected = result.keypoints is not None and len(result.keypoints.xy) > 0

        # 預設此幀的值
        disp_px    = 0.0
        av         = 0
        score      = 50
        abnormal   = False
        body_scale = 0.0
        active_kpts_n = 0
        mean_kpt_conf = 0.0
        kpts_arr   = None
        kpt_conf_arr = None

        if detected:
            detect_count += 1
            kpts_arr     = result.keypoints.xy[0].cpu().numpy()     # (17, 2)
            kpt_conf_arr = (
                result.keypoints.conf[0].cpu().numpy()
                if result.keypoints.conf is not None
                else np.ones(17, dtype=np.float32)
            )

            # EMA 平滑（用於顯示與位移計算）
            if ema_kpts is None:
                ema_kpts = kpts_arr.copy()
            else:
                ema_kpts = EMA_ALPHA * kpts_arr + (1.0 - EMA_ALPHA) * ema_kpts

            valid_mask    = kpt_conf_arr > KP_CONF_THRES
            active_kpts_n = int(np.sum(valid_mask))
            mean_kpt_conf = float(np.mean(kpt_conf_arr))
            conf_sum     += mean_kpt_conf
            conf_n       += 1

            # 計算位移量（對應 anomaly_detector.detect）
            if prev_kpts is not None and np.any(valid_mask):
                diffs      = np.linalg.norm(ema_kpts - prev_kpts, axis=1)
                disp_px    = float(np.mean(diffs[valid_mask]))
                body_scale = float(np.linalg.norm(kpts_arr[3] - kpts_arr[5]))
                # 位移前置 EMA：消除骨架偵測噪音（DISP_EMA_ALPHA 越小越平穩）
                disp_ema   = DISP_EMA_ALPHA * disp_px + (1.0 - DISP_EMA_ALPHA) * disp_ema
                # [可評估] abnormal：用原始 disp_px 判斷（不受平滑影響）
                abnormal   = disp_px > ABNORMAL_THRES_PX

            prev_kpts = ema_kpts.copy()

            # ST-GCN buffer
            kpts_buffer.append((kpts_arr, kpt_conf_arr))

        else:
            prev_kpts = None  # 偵測中斷時重置，避免跨場景虛假位移
            ema_kpts  = None

        # [可評估] activity_value：以平滑後位移 + body_scale 正規化
        av    = displacement_to_activity_value(disp_ema, body_scale) if detected else 0
        score = score_tracker.update(t_now, av)

        av_history.append(av)
        score_history.append(score)

        # ── ST-GCN 推論（每 CLASSIFY_STRIDE 幀） ────────────────────
        if (classifier is not None
                and len(kpts_buffer) >= SEQUENCE_LENGTH
                and infer_count % CLASSIFY_STRIDE == 0):
            try:
                kpts_seq = np.array([x[0] for x in kpts_buffer])
                conf_seq = np.array([x[1] for x in kpts_buffer])
                seq_in   = interpolate_missing(kpts_seq, conf_seq)
                model_obj = getattr(classifier, "model", None)
                if model_obj is not None and getattr(model_obj, "in_channels", 4) != 4:
                    seq_in = flip_normalize(seq_in)
                    seq_in = orientation_normalize(seq_in)
                    seq_in = normalize_skeleton_coords(seq_in)
                    feat   = build_feature_tensor(seq_in, conf_seq,
                                                  model_obj.feature_mode)
                    b, c, _ = classifier.classify(feat, precomputed=True)
                else:
                    b, c, _ = classifier.classify(seq_in)
                if b is not None and c >= GCN_CONF_THRES:
                    behavior_id = int(b)
                    gcn_conf    = float(c)
                else:
                    behavior_id = LOW_CONF_ID
                    gcn_conf    = float(c) if c is not None else 0.0
            except Exception:
                pass

        # ── 組合顯示畫面 ──────────────────────────────────────────────
        canvas = np.zeros((CANVAS_H, DISPLAY_W, 3), dtype=np.uint8)

        # 影片區域（左側）
        frame_disp = cv2.resize(frame, (VIDEO_AREA_W, DISPLAY_H))
        canvas[:DISPLAY_H, :VIDEO_AREA_W] = frame_disp

        # 骨架 overlay（在縮放後的影片上）
        if detected and kpts_arr is not None and kpt_conf_arr is not None:
            draw_skeleton(canvas, ema_kpts, kpt_conf_arr, scale_xy=(sx, sy))

        # 右側資訊面板
        det_rate = detect_count / max(infer_count + 1, 1) * 100
        mean_y_conf = conf_sum / max(conf_n, 1)
        draw_info_panel(
            canvas, VIDEO_AREA_W, 0, INFO_PANEL_W, DISPLAY_H,
            behavior_id=behavior_id,
            confidence=gcn_conf,
            av=av,
            score=score,
            abnormal=abnormal,
            body_scale=body_scale,
            active_kpts=active_kpts_n,
            frame_idx=infer_count,
            detection_rate=det_rate,
            mean_conf_yolo=mean_y_conf,
        )

        # 底部滾動折線圖
        draw_rolling_chart(canvas, list(av_history), list(score_history),
                           0, DISPLAY_H, DISPLAY_W, CHART_H)

        # 影片導覽列（影片區域頂端）
        nav_txt = (f"VIDEO {video_idx + 1}/{video_total}: "
                   f"{Path(video_path).name}  "
                   f"[SPACE] PAUSE  [1] PREV  [2] NEXT  [Q] QUIT")
        cv2.rectangle(canvas, (0, 0), (VIDEO_AREA_W, 20), (10, 16, 30), -1)
        _text(canvas, nav_txt, (6, 14), scale=0.38, color=(160, 200, 240))

        # ── 記錄 ──────────────────────────────────────────────────────
        records.append({
            "frame_idx":   infer_count,
            "ts":          t_now,
            "detected":    detected,
            "body_scale":  body_scale,
            "active_kpts": active_kpts_n,
            "mean_kpt_conf": mean_kpt_conf,
            "disp_px":     disp_px,
            "av":          av,
            "score":       score,
            "abnormal":    abnormal,
            "behavior_id": behavior_id,
            "gcn_conf":    gcn_conf,
        })

        if vout:
            vout.write(canvas)
        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1)
        if key == ord(" "):
            paused = True
        elif key == ord("q") or key == 27:
            _nav = "quit"; break
        elif key == ord("1"):
            _nav = "prev"; break
        elif key == ord("2"):
            _nav = "next"; break

        infer_count += 1
        if infer_count % 100 == 0:
            elapsed = time.time() - t_start
            fps_real = infer_count / max(elapsed, 1e-6)
            print(f"[RUN] frame={infer_count}  av={av}  score={score}"
                  f"  behavior={BEHAVIOR_LABEL.get(behavior_id,'?')}"
                  f"  fps={fps_real:.1f}")

    cap.release()
    if vout:
        vout.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE] 推論完成，共 {infer_count} 幀")

    # ── 儲存報告 ──────────────────────────────────────────────────────
    save_report(output_dir, records, video_path, yolo_path, stgcn_path or "")
    print(f"[DONE] 報告已儲存至: {output_dir.resolve()}\n")
    return _nav


# ══════════════════════════════════════════════════════════════════════
# 影片收集工具
# ══════════════════════════════════════════════════════════════════════

# 支援的影片副檔名（不分大小寫）
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}


def collect_videos(path: str) -> list:
    """
    接受單一檔案路徑或資料夾路徑，回傳排序後的影片路徑清單。

    - 單一檔案：直接回傳 [path]（不檢查副檔名，交由 cv2 決定）
    - 資料夾  ：遞迴搜尋 _VIDEO_EXTS 所有符合的影片，按名稱排序
    """
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        videos = sorted(
            f for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in _VIDEO_EXTS
        )
        if not videos:
            raise FileNotFoundError(f"資料夾內找不到影片檔案: {path}")
        return [str(v) for v in videos]
    raise FileNotFoundError(f"路徑不存在: {path}")


# ══════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Real-time cat activity inference with overlay visualization."
    )
    p.add_argument("--video",  default=DEFAULT_VIDEO,
                   help="影片路徑（單一 .mp4 等）或資料夾路徑（遞迴搜尋所有影片）")
    p.add_argument("--yolo",   default=DEFAULT_YOLO,   help="YOLO 模型路徑")
    p.add_argument("--stgcn",  default=DEFAULT_STGCN,  help="ST-GCN 模型路徑（可選）")
    p.add_argument("--device", default="cuda",          help="cuda / cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # 自動降級 device
    try:
        import torch
        if args.device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA 不可用，改用 CPU")
            args.device = "cpu"
    except ImportError:
        args.device = "cpu"

    # 收集影片清單（單一檔案或資料夾）
    video_list = collect_videos(args.video)
    n = len(video_list)
    print(f"[BATCH] 共找到 {n} 支影片")

    # 模型只載入一次，切換影片時重複使用
    yolo, classifier = load_models(args.yolo, args.stgcn, args.device)

    idx = 0
    while 0 <= idx < n:
        vp = video_list[idx]
        print(f"\n[BATCH] ── {idx + 1}/{n}: {Path(vp).name} ──")
        try:
            nav = run(
                video_path=vp,
                yolo=yolo,
                classifier=classifier,
                device=args.device,
                yolo_path=args.yolo,
                stgcn_path=args.stgcn,
                video_idx=idx,
                video_total=n,
            )
        except Exception as e:
            print(f"[ERROR] {Path(vp).name} 處理失敗，跳過: {e}")
            nav = "next"

        if nav == "quit":
            break
        elif nav == "prev":
            idx = (idx - 1) % n   # 第一支按 1 → 跳到最後一支
        else:                     # "next" 或自然播完
            idx = (idx + 1) % n   # 最後一支播完 → 回到第一支

    print(f"\n[BATCH] 結束，報告儲存於: {OUTPUT_BASE.resolve()}")
