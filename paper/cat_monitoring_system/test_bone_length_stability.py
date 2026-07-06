"""
Bone Length Stability Analysis — 骨段長度一致性診斷工具
多影片版（1~5 支）：依序播放每支影片，結果分開存，最後再疊圖比較。

════════════════════════════════════════════════════════════════════
目的
════════════════════════════════════════════════════════════════════
為「GCN 分類為主、幾何判斷為輔」的雙重判定機制找適當門檻值。

原理：真實貓咪骨架的骨段（如 胸→肘、髖→膝）長度在正規化座標下應該
維持穩定，即使動作很快（走路/甩頭）也一樣——因為那是剛體約束。
如果 YOLO 偵測到的關鍵點抖動/錯位，骨段長度會在短時間窗口內劇烈起伏，
這是跟「動作快慢」無關的雜訊訊號，正好可以用來偵測「這段骨架還能不能信」。

本腳本重用 models/stgcn_model.py 既有的前處理管線（interpolate_missing →
flip_normalize → orientation_normalize → normalize_skeleton_coords）與
compute_bone_feature() 的骨架拓撲（同一份 _parents_17，不重複定義），
確保這裡看到的數值跟未來真正接進 behavior_classifier.py 的判斷邏輯一致。

════════════════════════════════════════════════════════════════════
使用方式
════════════════════════════════════════════════════════════════════
  1. 設定 INPUT_MODE 選擇輸入方式（二選一）：
       "paths"  → 手動列出下方 VIDEO_PATHS 裡的 1~5 支個別影片路徑
       "folder" → 改指定 VIDEO_FOLDER 一個資料夾，自動抓裡面的影片
                  （依檔名排序，超過 5 支只取前 5 支）
     建議至少一支是追蹤穩定的「正常」影片、一支是已知會被誤判成
     walk/shake 的「抖動」影片，這樣最後的疊圖比較才看得出差異
  2. 執行腳本：依序播放每支影片，GUI 視窗右上角面板顯示目前 16 幀
     窗口內各骨段長度的變異係數（CV），數值越大代表越不穩定
  3. 播放時可按 [space] 暫停、[q] 跳到下一支影片（不是整個結束）
  4. 每支影片各自的 CSV + 雙圖表（時序線圖 + 分佈直方圖）分開存到
     OUTPUT_DIR / run_YYYYMMDD_HHMMSS / <影片檔名>/
  5. 全部影片跑完後，額外產生一張「多影片疊圖比較」：把每支影片的
     時序線與分佈直方圖疊在同一張圖上（不同顏色），方便直接比較
     「正常」跟「抖動」影片的 CV 落點差在哪裡，存到 run 根目錄的
     comparison_chart.png，並在最後互動顯示
"""
import sys
import csv
from pathlib import Path
from collections import deque
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from detectors.keypoint_detector import KeypointDetector
from models.stgcn_model import (
    interpolate_missing,
    flip_normalize,
    orientation_normalize,
    normalize_skeleton_coords,
    compute_bone_feature,
)

# ╔══════════════════════════════════════════════════════════════════╗
# ║          使 用 者 設 定 區（每次執行前只需修改此區）             ║
# ╚══════════════════════════════════════════════════════════════════╝

# 兩種輸入模式二選一，用 INPUT_MODE 切換：
#   "paths"  → 用下面 VIDEO_PATHS 手動列出的 1~5 支個別影片檔案路徑（原本的做法）
#   "folder" → 改成指定 VIDEO_FOLDER 一個資料夾，自動抓出裡面的影片檔（依檔名排序，
#              最多取前 5 支，超過會印警告並截斷）——不用手動一支一支列路徑
INPUT_MODE = "paths"  # "paths" 或 "folder"

VIDEO_PATHS = [
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\walk\walk_1.mp4",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\walk\walk_2.mp4",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\walk\walk_3.mp4",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\walk\walk_4.mp4",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\walk\walk_5.mp4"
]
# 1~5 支「單一影片檔案」路徑（不是資料夾），依序處理。建議至少放一支
# 追蹤穩定的正常影片、一支已知會誤判成 walk/shake 的抖動影片做對照。
# 只有 INPUT_MODE = "paths" 時才會用到這份清單。

VIDEO_FOLDER = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\泛化測試"
VIDEO_FOLDER_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
# 只有 INPUT_MODE = "folder" 時才會用到：資料夾底下（不含子資料夾）
# 副檔名符合上面清單的影片檔，依檔名排序後最多取前 5 支。

YOLO_MODEL_PATH = r"C:\AI_Project\cat_pose\v11s_113.pt"
INFERENCE_DEVICE = "cuda"
YOLO_IMGSZ = 640

OUTPUT_DIR = Path(r"C:\ai_project\paper\output\bone_length_stability")

SEQUENCE_LENGTH = 16          # 跟 ST-GCN 實際推論窗口一致（T=16）
BONE_CONF_THRESHOLD = 0.3     # 骨段兩端關鍵點信心低於此值，該幀不納入該骨段的 CV 計算
MIN_VALID_FRAMES_RATIO = 0.5  # 窗口內有效幀數低於此比例，該骨段本次 CV 標記為無效（NaN）

# 排除的關鍵點（不納入骨段穩定性分析）——尾巴天生會彎曲，長度變化是真實生理現象、
# 不是偵測雜訊，混進來會污染訊號，預設排除（跟 test_pose_jitter_analysis.py 的預設一致）
EXCLUDED_KEYPOINTS: set = {14, 15, 16}

# 面板/圖表用的候選門檻線（先猜一個，看圖表分佈後再調整，不代表最終建議值）
CANDIDATE_CV_THRESHOLD = 0.15
# midback offset jitter 專屬的候選門檻——先假設跟 CV 用同一個值當起點，
# jitter_threshold_chart 就是用來驗證/調整這個假設對不對，找到後改這裡即可，
# 不影響 CV 的門檻設定（兩者定義相同但代表的物理意義不同，不該共用同一條線）。
CANDIDATE_JITTER_THRESHOLD = 0.15
# 色階換算用的 CV 上限（超過這個值視覺上就全紅了，只影響顏色深淺，不影響數值）
MAX_CV_FOR_COLOR = 0.4

# 只在每 N 幀重算一次骨段 CV（其餘幀沿用上次結果顯示，不重新跑前處理），
# 對齊正式推論的 STGCNConfig.WINDOW_STRIDE 預設值——減少重複運算量，
# 同時讓這裡取樣的頻率更貼近正式部署時這個門檻機制實際會跑的頻率。
OVERLAY_STRIDE = 2

WINDOW_NAME = "Bone Length Stability"
DISPLAY_SIZE = (1080, 720)

# ===== 信心值門檻設定（bbox conf / keypoint conf，集中管理）=====
YOLO_CONF_THRESHOLD = 0.5      # YOLO bbox 偵測信心門檻
DRAW_KP_CONF_THRESHOLD = 0.25  # 畫骨架線段與關鍵點圓點用門檻（>此值才畫）

# ╔══════════════════════════════════════════════════════════════════╗
# ║                      以 下 無 需 修 改                           ║
# ╚══════════════════════════════════════════════════════════════════╝

KEYPOINT_NAMES = [
    "Nose", "L.Ear", "R.Ear", "Chest", "M.Back", "Hip",
    "LF.Elbow", "LF.Paw", "RF.Elbow", "RF.Paw",
    "LH.Knee", "LH.Paw", "RH.Knee", "RH.Paw",
    "T.Root", "T.Mid", "T.Tip",
]
# 對應 models/stgcn_model.py::compute_bone_feature 的 _parents_17，joint 0（Nose）
# 本身是根節點，bone 向量恆為 0，不納入分析。
BONE_SEGMENT_NAMES = {
    1: "Nose-LEar", 2: "Nose-REar", 3: "Nose-Chest", 4: "Chest-MBack",
    5: "MBack-Hip", 6: "Chest-LFElb", 7: "LFElb-LFPaw", 8: "Chest-RFElb",
    9: "RFElb-RFPaw", 10: "Hip-LHKnee", 11: "LHKnee-LHPaw", 12: "Hip-RHKnee",
    13: "RHKnee-RHPaw", 14: "Hip-TRoot", 15: "TRoot-TMid", 16: "TMid-TTip",
}
_ACTIVE_BONE_IDS = [i for i in range(1, 17) if i not in EXCLUDED_KEYPOINTS]

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 2), (0, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (3, 8), (8, 9),
    (5, 10), (10, 11), (5, 12), (12, 13),
    (5, 14), (14, 15), (15, 16),
]

WHITE = (255, 255, 255)
BLACK = (30, 30, 30)

# 從影片路徑（含資料夾名，因為你的檔案習慣常把行為標在上層資料夾，
# 例如 ...\walk\1 (43).mp4，檔名本身反而沒有）比對這五類關鍵字，
# 大小寫不敏感，比對不到回傳 "N/A"。
BEHAVIOR_KEYWORDS = ["walk", "lick", "scratch", "shake", "stop"]

# 每支影片依「索引」（VIDEO_PATHS 裡的順序，1-based）給一個獨立顏色，
# 不再依行為分組——同一類行為的影片也會各自用不同顏色，方便逐一辨認曲線。
# 行為標籤仍會顯示在圖例文字跟標題裡，只是不再拿來決定顏色。
VIDEO_INDEX_COLORS = [
    "#3498db",  # 1 藍
    "#f39c12",  # 2 金黃色（原本的橘色 #e67e22 跟 4 號紅色色相太接近，改用色相差更大的金黃）
    "#2ecc71",  # 3 綠
    "#c0392b",  # 4 磚紅色（原本的紅色 #e74c3c 偏橘調，改成更深、更偏紫的紅，跟 2 號拉開差距）
    "#9b59b6",  # 5 紫
]


def color_for_index(index: int) -> str:
    return VIDEO_INDEX_COLORS[(index - 1) % len(VIDEO_INDEX_COLORS)]


def ascii_safe_label(text: str, fallback: str) -> str:
    """matplotlib 預設字型（DejaVu Sans）沒有中文字符集，圖表裡混進中文
    （例如影片檔名剛好是中文，如「7月2日 (4).mp4」）會整段顯示成方塊。
    這裡把非 ASCII 字元換成空格（不是直接刪除）再收斂多餘空白，避免像
    「7月2日」這種中間插著中文字的字串被砍字砍成看起來像另一個數字的
    「72」——保留空格能讓「7」「2」維持是分開的兩截，不會誤黏在一起。
    濾完是空字串就用 fallback（例如 video_4）。"""
    cleaned = "".join(ch if ord(ch) < 128 else " " for ch in text)
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned else fallback


def infer_behavior_label(video_path) -> str:
    """從完整路徑（資料夾名或檔名任一處）比對 walk/lick/scratch/shake/stop，
    大小寫不敏感，找不到回傳 "N/A"。五個關鍵字彼此不互為子字串，不會誤配。"""
    lowered = str(video_path).lower()
    for kw in BEHAVIOR_KEYWORDS:
        if kw in lowered:
            return kw
    return "N/A"


def compute_bone_stability_overlay(seq_window, conf_window):
    """seq_window: (T, 17, 2) 原始座標；conf_window: (T, 17)。
    回傳每個骨段在這個窗口內的變異係數（CV = std/mean），未通過信心門檻的
    幀不計入該骨段，有效幀比例過低則該骨段本次標記為 NaN。

    前處理管線與 skeleton_visualizer.py::compute_velocity_overlay 一致，
    確保這裡看到的數值跟真正的推論路徑同源。

    同時回傳 avg_body_size_px：正規化「之前」的原始像素體型（胸(3)→髖(5)
    像素距離），用來驗證「鏡頭距離／貓咪佔畫面大小」是否是 CV 偏高的
    混淆變因——注意這**不是**正規化後的骨長（那個定義上恆等於 ~1.0，
    沒有鑑別力），是 normalize_skeleton_coords() 內部拿來當除數的那個值，
    這裡用同一條公式獨立算一次（平移/翻轉/旋轉都不改變距離，所以直接在
    原始座標上算跟在前處理後的座標上算數值相同，不需要重跑整條管線）。
    """
    seq = interpolate_missing(seq_window, conf_window, threshold=0.1)
    T = seq_window.shape[0]
    min_valid = max(2, int(round(T * MIN_VALID_FRAMES_RATIO)))

    chest_hip_valid = (conf_window[:, 3] >= BONE_CONF_THRESHOLD) & (conf_window[:, 5] >= BONE_CONF_THRESHOLD)
    if np.any(chest_hip_valid):
        avg_body_size_px = float(np.mean(np.linalg.norm(
            seq[chest_hip_valid, 3, :2] - seq[chest_hip_valid, 5, :2], axis=1
        )))
    else:
        avg_body_size_px = float("nan")

    # 虛擬點：胸(3)與髖(5)的中點——mid_back(4) 理論上應該落在這條線附近。
    # 這裡量的是「mid_back 偏離這個虛擬中點的距離 ÷ 體型」，是跟骨段長度
    # CV（量測時間上的變異）互補的另一種訊號：CV 抓的是「這段時間骨架
    # 有沒有在跳動」，這個比例抓的是「單獨這一幀 mid_back 的偵測位置，
    # 跟胸/髖兩點暗示的位置合不合理」。比值本身是距離除以距離，具尺度
    # 不變性，不受鏡頭遠近影響，不需要額外正規化。
    midback_valid = chest_hip_valid & (conf_window[:, 4] >= BONE_CONF_THRESHOLD)
    midback_offset_ratio = float("nan")
    midback_offset_jitter = float("nan")
    if np.any(midback_valid):
        virtual_pt = (seq[:, 3, :2] + seq[:, 5, :2]) / 2.0
        raw_offset = np.linalg.norm(seq[:, 4, :2] - virtual_pt, axis=1)
        body_size_per_frame = np.linalg.norm(seq[:, 3, :2] - seq[:, 5, :2], axis=1)
        frame_ok = midback_valid & (body_size_per_frame > 1e-6)
        if np.any(frame_ok):
            ratio_vals = raw_offset[frame_ok] / body_size_per_frame[frame_ok]
            midback_offset_ratio = float(np.mean(ratio_vals))
            # 抖動程度：跟骨段 CV 同一套定義（std/mean），量的是這個窗口內
            # mid_back 偏移比例本身有沒有在跳動——跟上面「單幀是否合理」的
            # 平均值互補，這裡抓的是「這段時間內是否忽大忽小」。有效幀數
            # 太少時（跟骨段 CV 用同一個 min_valid 門檻）不計，避免雜訊。
            if int(np.sum(frame_ok)) >= min_valid and midback_offset_ratio > 1e-9:
                midback_offset_jitter = float(np.std(ratio_vals) / midback_offset_ratio)

    seq = flip_normalize(seq)
    seq = orientation_normalize(seq)
    seq = normalize_skeleton_coords(seq)

    bone_xy = compute_bone_feature(seq)          # (T, 17, 2)
    bone_len = np.linalg.norm(bone_xy, axis=-1)   # (T, 17)

    _parents_17 = np.array([0, 0, 0, 0, 3, 4, 3, 6, 3, 8, 5, 10, 5, 12, 5, 14, 15])

    seg_cv = np.full(17, np.nan, dtype=np.float64)
    for j in _ACTIVE_BONE_IDS:
        parent = int(_parents_17[j])
        valid = (conf_window[:, j] >= BONE_CONF_THRESHOLD) & (conf_window[:, parent] >= BONE_CONF_THRESHOLD)
        if int(np.sum(valid)) < min_valid:
            continue
        vals = bone_len[valid, j]
        mean = float(np.mean(vals))
        if mean < 1e-9:
            continue
        seg_cv[j] = float(np.std(vals) / mean)

    valid_cv = seg_cv[~np.isnan(seg_cv)]
    agg_cv = float(np.mean(valid_cv)) if valid_cv.size > 0 else float("nan")
    max_cv = float(np.max(valid_cv)) if valid_cv.size > 0 else float("nan")
    max_idx = int(np.nanargmax(seg_cv)) if valid_cv.size > 0 else -1

    top_indices = [i for i in np.argsort(np.nan_to_num(seg_cv, nan=-1.0))[::-1] if not np.isnan(seg_cv[i])][:3]

    return {
        "seg_cv": seg_cv,
        "agg_cv": agg_cv,
        "max_cv": max_cv,
        "max_seg_idx": max_idx,
        "top_entries": [
            {"seg_idx": int(i), "name": BONE_SEGMENT_NAMES.get(int(i), str(i)), "cv": float(seg_cv[i])}
            for i in top_indices
        ],
        "avg_body_size_px": avg_body_size_px,
        "midback_offset_ratio": midback_offset_ratio,
        "midback_offset_jitter": midback_offset_jitter,
    }


def _heat_color(value, max_value):
    green = np.array([80, 220, 60], dtype=np.float32)
    yellow = np.array([0, 210, 255], dtype=np.float32)
    red = np.array([50, 50, 255], dtype=np.float32)
    if not np.isfinite(value) or max_value <= 1e-9:
        return tuple(int(v) for v in green)
    ratio = float(np.clip(value / max_value, 0.0, 1.0))
    if ratio < 0.5:
        c = green + (yellow - green) * (ratio / 0.5)
    else:
        c = yellow + (red - yellow) * ((ratio - 0.5) / 0.5)
    return tuple(int(np.clip(v, 0, 255)) for v in c)


FONT_SCALE_MULT = 1.5  # 面板整體字體放大倍率
NEUTRAL_TEXT_COLOR = (0, 200, 255)  # 中等亮度、高飽和的琥珀色——比灰白更顯眼，但不刺眼


def draw_bone_stability_panel(frame, ovl):
    if ovl is None:
        return frame
    h, w = frame.shape[:2]
    ui_scale = max(0.9, min(1.6, np.hypot(w, h) / 1500.0))
    panel_w = int(430 * ui_scale)
    panel_h = int(270 * ui_scale)
    # 貼齊畫面左上角邊界，不留額外偏移（原本 y0 多加了 44px 的空白間距，
    # 使用者要求整個面板要貼齊邊界，這裡拿掉那段偏移）。
    pad = int(6 * ui_scale)
    x0 = pad
    y0 = pad
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (120, 120, 120), 1, cv2.LINE_AA)

    tx = x0 + int(8 * ui_scale)
    ty = y0 + int(22 * ui_scale)
    f = FONT_SCALE_MULT
    cv2.putText(frame, "Bone Length CV (window)", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * f * ui_scale, NEUTRAL_TEXT_COLOR, 2, cv2.LINE_AA)
    ty += int(30 * ui_scale)
    agg = ovl["agg_cv"]
    agg_color = _heat_color(agg, MAX_CV_FOR_COLOR) if np.isfinite(agg) else (150, 150, 150)
    cv2.putText(frame, f"aggregate: {agg:.4f}" if np.isfinite(agg) else "aggregate: --",
                (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * f * ui_scale, agg_color, 2, cv2.LINE_AA)
    ty += int(30 * ui_scale)
    # 原始像素體型（非正規化，胸→髖），拿來目視核對是否跟上面的 CV 反向變化——
    # 若貓靠近鏡頭時這裡的數字變大、CV 同時下降，就直接證實鏡頭距離是混淆變因。
    # 這一項本身沒有「好壞」門檻（純參考數值），固定用中等亮度的綠色，
    # 不套 heat color，避免跟真正代表風險程度的三項指標混淆。
    body_px = ovl.get("avg_body_size_px", float("nan"))
    cv2.putText(frame, f"body size: {body_px:.1f}px" if np.isfinite(body_px) else "body size: --",
                (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * f * ui_scale, (100, 210, 100), 2, cv2.LINE_AA)
    ty += int(30 * ui_scale)
    # mid_back 偏離「胸髖虛擬中點」的比例（尺度不變，不受鏡頭遠近影響）——
    # 跟上面的骨段 CV 是互補訊號：CV 抓時間上的跳動，這個抓單幀 mid_back
    # 偵測位置跟胸/髖兩點是否吻合。
    mb_ratio = ovl.get("midback_offset_ratio", float("nan"))
    mb_color = _heat_color(mb_ratio, MAX_CV_FOR_COLOR) if np.isfinite(mb_ratio) else (150, 150, 150)
    cv2.putText(frame, f"midback offset: {mb_ratio:.4f}" if np.isfinite(mb_ratio) else "midback offset: --",
                (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * f * ui_scale, mb_color, 2, cv2.LINE_AA)
    ty += int(34 * ui_scale)

    # mid_back offset 比例「本身」在這個窗口內的抖動程度（std/mean，跟骨段
    # CV 同一套定義）——上面 midback offset 只看平均值是否合理，這裡額外
    # 看它在窗口內有沒有忽大忽小，兩者互補。這是目前判斷骨架可不可信最
    # 關鍵的單一訊號，所以字級再放大一輪、加粗，並用背景色塊獨立標出來，
    # 跟面板其他數值明確做出視覺區隔。
    mb_jitter = ovl.get("midback_offset_jitter", float("nan"))
    mb_jitter_color = _heat_color(mb_jitter, MAX_CV_FOR_COLOR) if np.isfinite(mb_jitter) else (150, 150, 150)
    jitter_scale = 0.62 * f * ui_scale
    jitter_text = f"OFFSET JITTER: {mb_jitter:.4f}" if np.isfinite(mb_jitter) else "OFFSET JITTER: --"
    (jw, jh), _ = cv2.getTextSize(jitter_text, cv2.FONT_HERSHEY_SIMPLEX, jitter_scale, 3)
    hl_pad = int(6 * ui_scale)
    hl_x0 = tx - hl_pad
    hl_y0 = ty - jh - hl_pad
    hl_x1 = tx + jw + hl_pad
    hl_y1 = ty + hl_pad
    cv2.rectangle(frame, (hl_x0, hl_y0), (hl_x1, hl_y1), (35, 35, 35), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (hl_x0, hl_y0), (hl_x1, hl_y1), mb_jitter_color, 2, cv2.LINE_AA)
    cv2.putText(frame, jitter_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, jitter_scale, mb_jitter_color, 3, cv2.LINE_AA)
    ty += int(38 * ui_scale)

    bar_x = tx
    bar_w = int(panel_w * 0.45)
    for i, entry in enumerate(ovl["top_entries"]):
        score = entry["cv"]
        color = _heat_color(score, MAX_CV_FOR_COLOR)
        row_y = ty + i * int(27 * ui_scale)
        cv2.putText(frame, entry["name"], (bar_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * f * ui_scale, NEUTRAL_TEXT_COLOR, 1, cv2.LINE_AA)
        bx0 = x0 + int(panel_w * 0.42)
        by0 = row_y - int(14 * ui_scale)
        bx1 = bx0 + bar_w
        by1 = by0 + int(14 * ui_scale)
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), (50, 50, 50), -1)
        ratio = float(np.clip(score / MAX_CV_FOR_COLOR, 0.0, 1.0)) if np.isfinite(score) else 0.0
        fx1 = bx0 + int(round(bar_w * ratio))
        if fx1 > bx0:
            cv2.rectangle(frame, (bx0, by0), (fx1, by1), color, -1)
        cv2.putText(frame, f"{score:.3f}", (bx1 + int(6 * ui_scale), row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * f * ui_scale, NEUTRAL_TEXT_COLOR, 1, cv2.LINE_AA)
    return frame


def draw_skeleton(frame, kpts, kpt_conf, sx, sy, conf_thresh=DRAW_KP_CONF_THRESHOLD):
    for (a, b) in SKELETON_EDGES:
        if kpt_conf[a] < conf_thresh or kpt_conf[b] < conf_thresh:
            continue
        pa = (int(kpts[a, 0] * sx), int(kpts[a, 1] * sy))
        pb = (int(kpts[b, 0] * sx), int(kpts[b, 1] * sy))
        cv2.line(frame, pa, pb, (0, 200, 0), 2, cv2.LINE_AA)
    for i in range(kpts.shape[0]):
        if kpt_conf[i] < conf_thresh:
            continue
        p = (int(kpts[i, 0] * sx), int(kpts[i, 1] * sy))
        cv2.circle(frame, p, 4, (0, 0, 220), -1, cv2.LINE_AA)
    return frame


def analyze_single_video(video_path: str, detector: KeypointDetector, out_dir: Path, video_index: int):
    """播放單一影片、收集逐窗口 CV 記錄，並輸出該影片自己的 CSV + 雙圖表。
    video_index 是這支影片在 VIDEO_PATHS 裡的順序（1-based）——用來決定
    這支影片專屬的顏色，也會寫進 CSV 的 video_index 欄位，方便日後把多支
    影片的 CSV 合併時，用同一個索引把資料列跟圖表上的曲線對起來。

    只在 frame_idx % OVERLAY_STRIDE == 0 時才重算 CV（其餘幀沿用上一次
    結果顯示，不重跑前處理），對齊正式推論的 WINDOW_STRIDE 節奏。

    回傳 (records, behavior)，behavior 是從路徑推斷出的行為標籤。
    按 [q] 會提前結束「這一支」影片（不是整個批次）。
    """
    video_path_obj = Path(video_path)
    if video_path_obj.is_dir():
        print(f"❌ 略過（指向資料夾，不是影片檔案）: {video_path}")
        return None
    if not video_path_obj.exists():
        print(f"❌ 略過（找不到檔案）: {video_path}")
        return None

    behavior = infer_behavior_label(video_path)
    color = color_for_index(video_index)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 略過（無法開啟）: {video_path}")
        return None

    kp_buffer = deque(maxlen=SEQUENCE_LENGTH)
    conf_buffer = deque(maxlen=SEQUENCE_LENGTH)

    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    records = []
    ovl = None  # 持續保留到下一次重算，避免節流幀顯示空白

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *DISPLAY_SIZE)

    print(f"▶ 開始分析 #{video_index}: {video_path}  [{behavior}]")
    print(f"  [space]=暫停/繼續  [q]=跳到下一支影片  （每 {OVERLAY_STRIDE} 幀重算一次 CV）")

    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            kpts, kpt_conf, bbox, _det_conf = detector.detect(frame)
            if kpts is None:
                kp_buffer.clear()
                conf_buffer.clear()
                ovl = None
            else:
                kp_buffer.append(kpts.copy())
                conf_buffer.append(kpt_conf.copy())
                if len(kp_buffer) == SEQUENCE_LENGTH:
                    if frame_idx % OVERLAY_STRIDE == 0:
                        seq_window = np.stack(kp_buffer, axis=0)
                        conf_window = np.stack(conf_buffer, axis=0)
                        ovl = compute_bone_stability_overlay(seq_window, conf_window)
                        records.append({
                            "video_index": video_index,
                            "video_name": video_path_obj.stem,
                            "behavior": behavior,
                            "frame": frame_idx,
                            "time_sec": round(frame_idx / fps, 3),
                            "agg_cv": ovl["agg_cv"],
                            "max_cv": ovl["max_cv"],
                            "max_seg": BONE_SEGMENT_NAMES.get(ovl["max_seg_idx"], ""),
                            "avg_body_size_px": ovl["avg_body_size_px"],
                            "midback_offset_ratio": ovl["midback_offset_ratio"],
                            "midback_offset_jitter": ovl["midback_offset_jitter"],
                        })
                    # else: 節流幀，沿用上一次算好的 ovl 顯示，不重算也不新增記錄
                else:
                    ovl = None

            h, w = frame.shape[:2]
            sx = DISPLAY_SIZE[0] / w
            sy = DISPLAY_SIZE[1] / h
            display = cv2.resize(frame, DISPLAY_SIZE)
            if kpts is not None:
                display = draw_skeleton(display, kpts, kpt_conf, sx, sy)
            if ovl is not None:
                display = draw_bone_stability_panel(display, ovl)
            cv2.putText(display, f"#{video_index} {video_path_obj.name} [{behavior}]  frame {frame_idx}", (10, DISPLAY_SIZE[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1 if not paused else 50) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            paused = not paused

    cap.release()

    if not records:
        print(f"⚠ {video_path_obj.name}: 沒有收集到任何有效窗口資料（影片太短或骨架偵測失敗），略過輸出")
        return None

    video_out_dir = out_dir / video_path_obj.stem
    video_out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = video_out_dir / "bone_length_cv_timeseries.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_index", "video_name", "behavior", "frame", "time_sec",
            "agg_cv", "max_cv", "max_seg", "avg_body_size_px", "midback_offset_ratio",
            "midback_offset_jitter",
        ])
        writer.writeheader()
        writer.writerows(records)

    frames = [r["frame"] for r in records]
    agg_cvs = [r["agg_cv"] for r in records]
    body_sizes = [r["avg_body_size_px"] for r in records]
    mb_ratios = [r["midback_offset_ratio"] for r in records]
    mb_jitters = [r["midback_offset_jitter"] for r in records]
    agg_arr = np.array(agg_cvs, dtype=np.float64)
    valid_arr = agg_arr[~np.isnan(agg_arr)]

    mb_arr_all = np.array(mb_ratios, dtype=np.float64)
    mb_valid_all = mb_arr_all[~np.isnan(mb_arr_all)]

    mb_jitter_arr_all = np.array(mb_jitters, dtype=np.float64)
    mb_jitter_valid_all = mb_jitter_arr_all[~np.isnan(mb_jitter_arr_all)]

    safe_name = ascii_safe_label(video_path_obj.name, f"video_{video_index}")

    # 拿掉 body size 時序圖（純參考用途、已驗證過沒有簡單反向關係，
    # 版面優先讓給 jitter）——jitter 是目前判斷骨架可信度最關鍵的
    # 單一訊號，用 height_ratios 讓它的時序圖＋分布圖拿到最大版面。
    fig, axes = plt.subplots(
        6, 1, figsize=(12, 20), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 0.8, 1.0, 0.8, 1.5, 1.2]},
    )
    axes[0].plot(frames, agg_cvs, linewidth=1, color=color, label=f"#{video_index} {behavior}")
    axes[0].axhline(CANDIDATE_CV_THRESHOLD, color="red", linestyle="--", linewidth=1,
                     label=f"candidate threshold = {CANDIDATE_CV_THRESHOLD}")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("aggregate bone-length CV")
    axes[0].set_title(f"Bone Length CV over time - #{video_index} {safe_name} [{behavior}]", fontsize=10)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if valid_arr.size > 0:
        axes[1].hist(valid_arr, bins=40, color=color, edgecolor="white", label=f"#{video_index} {behavior}")
        axes[1].axvline(CANDIDATE_CV_THRESHOLD, color="red", linestyle="--", linewidth=1,
                         label=f"candidate threshold = {CANDIDATE_CV_THRESHOLD}")
        axes[1].set_xlabel("aggregate bone-length CV")
        axes[1].set_ylabel("frame count")
        axes[1].set_title("CV distribution", fontsize=10)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    # mid_back 偏離胸髖虛擬中點的比例——跟骨段 CV 互補的單幀一致性訊號，
    # 尺度不變（距離除以距離），不受鏡頭遠近影響。
    axes[2].plot(frames, mb_ratios, linewidth=1, color=color)
    axes[2].set_xlabel("frame")
    axes[2].set_ylabel("midback offset ratio")
    axes[2].set_title("mid_back offset ratio over time (scale-invariant)", fontsize=10)
    axes[2].grid(alpha=0.3)

    if mb_valid_all.size > 0:
        axes[3].hist(mb_valid_all, bins=40, color=color, edgecolor="white", label=f"#{video_index} {behavior}")
        axes[3].set_xlabel("midback offset ratio")
        axes[3].set_ylabel("frame count")
        axes[3].set_title("midback offset ratio distribution", fontsize=10)
        axes[3].legend()
        axes[3].grid(alpha=0.3)

    # midback offset ratio 本身的窗口內抖動程度（std/mean，跟骨段 CV 同一套
    # 定義）——上面兩張圖看的是「offset 平均值多大／落在哪個範圍」，這裡看
    # 「offset 在窗口內穩不穩」，是更貼近「抖動」字面意義的訊號，也是目前
    # 判斷骨架可信度最重要的一項，這裡用最大的版面呈現。
    axes[4].plot(frames, mb_jitters, linewidth=1.3, color=color)
    axes[4].axhline(CANDIDATE_JITTER_THRESHOLD, color="red", linestyle="--", linewidth=1,
                     label=f"candidate threshold = {CANDIDATE_JITTER_THRESHOLD}")
    axes[4].set_xlabel("frame")
    axes[4].set_ylabel("midback offset jitter (std/mean)")
    axes[4].set_title("midback offset jitter over time (std/mean, same def as bone CV) [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[4].legend()
    axes[4].grid(alpha=0.3)

    if mb_jitter_valid_all.size > 0:
        axes[5].hist(mb_jitter_valid_all, bins=40, color=color, edgecolor="white", label=f"#{video_index} {behavior}")
        axes[5].axvline(CANDIDATE_JITTER_THRESHOLD, color="red", linestyle="--", linewidth=1,
                         label=f"candidate threshold = {CANDIDATE_JITTER_THRESHOLD}")
        axes[5].set_xlabel("midback offset jitter (std/mean)")
        axes[5].set_ylabel("frame count")
        axes[5].set_title("midback offset jitter distribution [KEY METRIC]", fontsize=11, fontweight="bold")
        axes[5].legend()
        axes[5].grid(alpha=0.3)

    png_path = video_out_dir / "bone_length_cv_chart.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"✅ {video_path_obj.name} 完成 — CSV: {csv_path}")
    print(f"   有效窗口數: {valid_arr.size} / {len(agg_cvs)}")
    if valid_arr.size > 0:
        print(f"   CV 分佈: min={valid_arr.min():.4f} median={np.median(valid_arr):.4f} "
              f"mean={valid_arr.mean():.4f} p95={np.percentile(valid_arr, 95):.4f} max={valid_arr.max():.4f}")
    body_arr = np.array(body_sizes, dtype=np.float64)
    body_valid = body_arr[~np.isnan(body_arr)]
    if body_valid.size > 0:
        print(f"   體型(px): min={body_valid.min():.1f} max={body_valid.max():.1f} "
              f"（變化幅度 {body_valid.max()-body_valid.min():.1f}px，越大代表貓在畫面裡的大小變化越劇烈）")
    mb_arr = np.array(mb_ratios, dtype=np.float64)
    mb_valid = mb_arr[~np.isnan(mb_arr)]
    if mb_valid.size > 0:
        print(f"   midback offset: min={mb_valid.min():.4f} median={np.median(mb_valid):.4f} "
              f"mean={mb_valid.mean():.4f} max={mb_valid.max():.4f}")
    mb_jitter_arr = np.array(mb_jitters, dtype=np.float64)
    mb_jitter_valid = mb_jitter_arr[~np.isnan(mb_jitter_arr)]
    if mb_jitter_valid.size > 0:
        print(f"   midback offset jitter: min={mb_jitter_valid.min():.4f} median={np.median(mb_jitter_valid):.4f} "
              f"mean={mb_jitter_valid.mean():.4f} max={mb_jitter_valid.max():.4f}")

    return records, behavior


def build_comparison_chart(video_results: dict, out_dir: Path):
    """video_results: {video_name: (records, behavior, video_index)}。把每支
    影片疊在同一張時序圖跟同一張分佈直方圖上，顏色依「索引」區分（每支影片
    獨立顏色，不再依行為分組），並在時序線的右端標上索引數字，不用回頭查
    圖例就能認出哪條線是哪支影片；行為標籤仍保留在圖例文字裡。
    density 正規化避免因影片長度不同而失真。拿掉了純參考用的原始像素體型
    子圖（已驗證過沒有簡單的鏡頭距離反向關係），版面優先給 midback offset
    jitter（窗口內 std/mean，跟骨段 CV 同一套定義，目前判斷骨架可信度
    最關鍵的單一訊號）——用 height_ratios 讓它的時序＋分布兩張圖拿到
    最大版面。存到 run 根目錄的 comparison_chart.png。
    """
    fig, axes = plt.subplots(
        6, 1, figsize=(13, 24), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 0.8, 1.0, 0.8, 1.5, 1.2]},
    )

    for name, (records, behavior, video_index) in video_results.items():
        color = color_for_index(video_index)
        safe_name = ascii_safe_label(name, f"video_{video_index}")
        legend_label = f"#{video_index} {behavior}: {safe_name}"
        times = [r["time_sec"] for r in records]
        agg_cvs = np.array([r["agg_cv"] for r in records], dtype=np.float64)
        axes[0].plot(times, agg_cvs, linewidth=1, color=color, alpha=0.85, label=legend_label)

        # 在曲線右端標上索引數字，方便不查圖例也能認出是哪支影片
        valid_mask = ~np.isnan(agg_cvs)
        if np.any(valid_mask):
            last_i = np.where(valid_mask)[0][-1]
            axes[0].annotate(
                str(video_index), xy=(times[last_i], agg_cvs[last_i]),
                xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=11, fontweight="bold", va="center",
            )

        valid = agg_cvs[valid_mask]
        if valid.size > 0:
            axes[1].hist(valid, bins=40, color=color, alpha=0.45, density=True, label=legend_label, edgecolor="none")

        mb_ratios = np.array([r["midback_offset_ratio"] for r in records], dtype=np.float64)
        axes[2].plot(times, mb_ratios, linewidth=1, color=color, alpha=0.85, label=legend_label)
        mb_valid_mask = ~np.isnan(mb_ratios)
        if np.any(mb_valid_mask):
            last_mi = np.where(mb_valid_mask)[0][-1]
            axes[2].annotate(
                str(video_index), xy=(times[last_mi], mb_ratios[last_mi]),
                xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=11, fontweight="bold", va="center",
            )

        mb_valid = mb_ratios[mb_valid_mask]
        if mb_valid.size > 0:
            axes[3].hist(mb_valid, bins=40, color=color, alpha=0.45, density=True, label=legend_label, edgecolor="none")

        mb_jitters = np.array([r["midback_offset_jitter"] for r in records], dtype=np.float64)
        axes[4].plot(times, mb_jitters, linewidth=1.3, color=color, alpha=0.9, label=legend_label)
        mb_jitter_mask = ~np.isnan(mb_jitters)
        if np.any(mb_jitter_mask):
            last_ji = np.where(mb_jitter_mask)[0][-1]
            axes[4].annotate(
                str(video_index), xy=(times[last_ji], mb_jitters[last_ji]),
                xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=11, fontweight="bold", va="center",
            )

        mb_jitter_valid = mb_jitters[mb_jitter_mask]
        if mb_jitter_valid.size > 0:
            axes[5].hist(mb_jitter_valid, bins=40, color=color, alpha=0.45, density=True, label=legend_label, edgecolor="none")

    axes[0].axhline(CANDIDATE_CV_THRESHOLD, color="black", linestyle="--", linewidth=1,
                     label=f"candidate threshold = {CANDIDATE_CV_THRESHOLD}")
    axes[0].set_xlabel("time (sec)")
    axes[0].set_ylabel("aggregate bone-length CV")
    axes[0].set_title("Bone Length CV over time (line-end number = video index)", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].axvline(CANDIDATE_CV_THRESHOLD, color="black", linestyle="--", linewidth=1,
                     label=f"candidate threshold = {CANDIDATE_CV_THRESHOLD}")
    axes[1].set_xlabel("aggregate bone-length CV")
    axes[1].set_ylabel("density")
    axes[1].set_title("CV distribution (density-normalized)", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].set_xlabel("time (sec)")
    axes[2].set_ylabel("midback offset ratio")
    axes[2].set_title("mid_back offset ratio over time (scale-invariant)", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    axes[3].set_xlabel("midback offset ratio")
    axes[3].set_ylabel("density")
    axes[3].set_title("midback offset ratio distribution (density-normalized)", fontsize=10)
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.3)

    axes[4].axhline(CANDIDATE_JITTER_THRESHOLD, color="black", linestyle="--", linewidth=1,
                     label=f"candidate threshold = {CANDIDATE_JITTER_THRESHOLD}")
    axes[4].set_xlabel("time (sec)")
    axes[4].set_ylabel("midback offset jitter (std/mean)")
    axes[4].set_title("midback offset jitter over time (std/mean, same def as bone CV) [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[4].legend(fontsize=8)
    axes[4].grid(alpha=0.3)

    axes[5].axvline(CANDIDATE_JITTER_THRESHOLD, color="black", linestyle="--", linewidth=1,
                     label=f"candidate threshold = {CANDIDATE_JITTER_THRESHOLD}")
    axes[5].set_xlabel("midback offset jitter (std/mean)")
    axes[5].set_ylabel("density")
    axes[5].set_title("midback offset jitter distribution (density-normalized) [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[5].legend(fontsize=8)
    axes[5].grid(alpha=0.3)

    png_path = out_dir / "comparison_chart.png"
    fig.savefig(png_path, dpi=150)
    print(f"\n✅ 多影片比較圖已存: {png_path}")
    plt.show()


def build_jitter_threshold_chart(video_results: dict, out_dir: Path):
    """專門用來找 midback offset jitter 異常門檻值的輔助圖表——先假設一個
    候選門檻（CANDIDATE_JITTER_THRESHOLD，預設沿用 CV 的 0.15 當起點），
    畫三張圖讓你事後檢驗這個假設合不合理，而不是憑感覺猜一個數字：

      1. 逐支影片的經驗累積分布（ECDF）：x 軸是 jitter 值，y 軸是「小於等於
         這個值的幀數佔全部幀數的比例」。理想情況下，穩定影片的曲線應該在
         候選門檻線之前就已經接近頂端（代表幾乎所有幀都在門檻內），問題
         影片的曲線則會在門檻線之後還有明顯一段沒爬完——兩者在門檻附近
         的垂直距離，就是這個門檻分不分得開的直接證據。
      2. 逐支影片「超過候選門檻的幀數比例」長條圖：把①的判讀量化成一個
         數字，直接排序比較，一眼看出哪支影片被判定為「常常抖」。
      3. 全部影片合併的 pooled 直方圖（y 軸用 log scale，因為真正的異常
         爆衝通常只佔一小撮幀數，線性座標會被大量正常幀壓成看不見），
         搭配候選門檻線，找整體分布有沒有自然斷層（gap）。

    調整門檻時只需要改上面的 CANDIDATE_JITTER_THRESHOLD 常數，重跑一次
    這張圖，看①②的分離效果跟③的斷層位置有沒有變好即可，不用改任何
    計算邏輯。存到 run 根目錄的 jitter_threshold_chart.png。
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), constrained_layout=True)

    pooled = []
    flag_rates = []  # [(video_index, behavior, safe_name, flag_rate_pct, color), ...]

    for name, (records, behavior, video_index) in video_results.items():
        color = color_for_index(video_index)
        safe_name = ascii_safe_label(name, f"video_{video_index}")
        legend_label = f"#{video_index} {behavior}: {safe_name}"

        jitters = np.array([r["midback_offset_jitter"] for r in records], dtype=np.float64)
        valid = jitters[~np.isnan(jitters)]
        if valid.size == 0:
            continue
        pooled.append(valid)

        sorted_vals = np.sort(valid)
        cum_frac = np.arange(1, sorted_vals.size + 1) / sorted_vals.size
        axes[0].plot(sorted_vals, cum_frac, linewidth=1.4, color=color, label=legend_label)

        flag_rate = float(np.mean(valid > CANDIDATE_JITTER_THRESHOLD)) * 100.0
        flag_rates.append((video_index, behavior, safe_name, flag_rate, color))

    axes[0].axvline(CANDIDATE_JITTER_THRESHOLD, color="black", linestyle="--", linewidth=1.5,
                     label=f"candidate threshold = {CANDIDATE_JITTER_THRESHOLD}")
    for ref in (0.90, 0.95, 0.99):
        axes[0].axhline(ref, color="gray", linestyle=":", linewidth=0.7)
    axes[0].set_xlabel("midback offset jitter (std/mean)")
    axes[0].set_ylabel("cumulative fraction of frames")
    axes[0].set_title("ECDF per video (dashed = candidate threshold, dotted = 90/95/99%)", fontsize=10)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    flag_rates.sort(key=lambda t: t[3], reverse=True)
    # 影片名稱後面加上排名（No.1 = flag rate 最高），排序邏輯本身沒變，
    # 只是把「第幾名」明確標在標籤上，不用自己數長條圖位置。
    labels = [f"No.{rank} #{idx} {beh}\n{nm}" for rank, (idx, beh, nm, _rate, _c) in enumerate(flag_rates, start=1)]
    rates = [rate for *_, rate, _c in flag_rates]
    bar_colors = [c for *_, c in flag_rates]
    bars = axes[1].bar(labels, rates, color=bar_colors)
    for bar, rate in zip(bars, rates):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{rate:.1f}%",
                      ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("% of frames with jitter > threshold")
    axes[1].set_title(f"Flag rate per video at threshold = {CANDIDATE_JITTER_THRESHOLD} (sorted, high to low)", fontsize=10)
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    if pooled:
        pooled_all = np.concatenate(pooled)
        counts, bin_edges, _patches = axes[2].hist(pooled_all, bins=60, color="#7f8c8d", edgecolor="white")
        axes[2].axvline(CANDIDATE_JITTER_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
                         label=f"candidate threshold = {CANDIDATE_JITTER_THRESHOLD}")
        # 門檻線下方標出確切數值（紅字，跟門檻線同色）——用 get_xaxis_transform()
        # 讓 x 還是照資料座標對齊門檻線，但 y 改用「軸的比例」（0=底部），
        # 這樣不管 log 座標的 y 範圍怎麼變，這個數字永遠穩穩貼在 x 軸正下方。
        axes[2].text(CANDIDATE_JITTER_THRESHOLD, -0.06, f"{CANDIDATE_JITTER_THRESHOLD:.3f}",
                      transform=axes[2].get_xaxis_transform(), color="red", fontsize=9,
                      fontweight="bold", ha="center", va="top", clip_on=False)
        axes[2].set_yscale("log")
        axes[2].set_xlabel("midback offset jitter (std/mean)")
        axes[2].set_ylabel("frame count (log scale)")
        axes[2].set_title("Pooled distribution across all videos (log y-axis, look for a natural gap)", fontsize=10)
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3)

        # 5 支影片加總的總幀數、門檻以下的幀數、佔比——三項數字直接寫在
        # 圖上（右上角文字框），不用再自己心算長條圖裡有幾根、加起來多少。
        total_frames = int(pooled_all.size)
        below_count = int(np.sum(pooled_all <= CANDIDATE_JITTER_THRESHOLD))
        below_pct = below_count / total_frames * 100.0 if total_frames > 0 else float("nan")
        summary_text = (
            f"Total frames (all videos): {total_frames}\n"
            f"Frames <= threshold ({CANDIDATE_JITTER_THRESHOLD:.3f}): {below_count}\n"
            f"% below threshold: {below_pct:.1f}%"
        )
        axes[2].text(0.98, 0.95, summary_text, transform=axes[2].transAxes,
                      ha="right", va="top", fontsize=9,
                      bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

        print(f"\n📊 Pooled 總覽（{len(pooled)} 支影片加總）：總幀數 {total_frames}，"
              f"門檻（{CANDIDATE_JITTER_THRESHOLD:.3f}）以下 {below_count} 幀，"
              f"佔比 {below_pct:.1f}%")

        # 每一根非空的長條上方標出確切幀數——log 座標只能目測大概高度，
        # 直接把數字寫在長條頂端最直觀。用直式文字（rotation=90）避免
        # 低數值區間相鄰長條的標籤互相重疊。
        bin_width = bin_edges[1] - bin_edges[0]
        for lo, cnt in zip(bin_edges[:-1], counts):
            if cnt > 0:
                axes[2].text(lo + bin_width / 2, cnt, str(int(cnt)),
                             rotation=90, ha="center", va="bottom", fontsize=6.5)

        # log 座標的長條圖只能目測大概高度，這裡額外把每一格 bin 的精確
        # 幀數印到終端機（只列有資料的 bin，大部分空 bin 沒必要洗版），
        # 跟長條上方的標籤互相對照，方便精確核對「斷層」從哪個數值開始。
        print(f"\n📊 Pooled 直方圖逐格明細（bin 寬度 = {bin_edges[1] - bin_edges[0]:.4f}，"
              f"共 {int(pooled_all.size)} 幀，{len(pooled)} 支影片加總）：")
        for lo, hi, cnt in zip(bin_edges[:-1], bin_edges[1:], counts):
            if cnt > 0:
                marker = "  <-- threshold" if lo <= CANDIDATE_JITTER_THRESHOLD < hi else ""
                print(f"   [{lo:.4f}, {hi:.4f}): {int(cnt)} 幀{marker}")

    png_path = out_dir / "jitter_threshold_chart.png"
    fig.savefig(png_path, dpi=150)
    print(f"\n✅ jitter 門檻分析圖已存: {png_path}")

    print(f"\n📊 Jitter 門檻分析摘要（候選門檻 = {CANDIDATE_JITTER_THRESHOLD}）：")
    for idx, beh, nm, rate, _c in flag_rates:
        print(f"   #{idx} {beh} ({nm}): {rate:.2f}% 的幀超過門檻")
    plt.show()


def resolve_video_paths() -> list:
    """依 INPUT_MODE 決定實際要處理的影片路徑清單："paths" 直接回傳
    VIDEO_PATHS；"folder" 則掃描 VIDEO_FOLDER（不含子資料夾）裡符合
    VIDEO_FOLDER_EXTENSIONS 的影片檔，依檔名排序，最多取前 5 支
    （超過 5 支會印警告並截斷，維持跟 "paths" 模式一樣 1~5 支的限制）。"""
    if INPUT_MODE == "paths":
        return list(VIDEO_PATHS)

    if INPUT_MODE == "folder":
        folder = Path(VIDEO_FOLDER)
        if not folder.is_dir():
            print(f"❌ VIDEO_FOLDER 不是有效的資料夾路徑: {folder}")
            return []
        found = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_FOLDER_EXTENSIONS
        )
        if not found:
            print(f"❌ 在資料夾裡找不到任何符合 {VIDEO_FOLDER_EXTENSIONS} 的影片: {folder}")
            return []
        if len(found) > 5:
            print(f"⚠ 資料夾裡有 {len(found)} 支影片，本工具最多同時分析 5 支，"
                  f"只取檔名排序後的前 5 支：{[p.name for p in found[:5]]}")
        return [str(p) for p in found[:5]]

    print(f"❌ INPUT_MODE 只能是 \"paths\" 或 \"folder\"，目前設定了 {INPUT_MODE!r}")
    return []


def main():
    video_paths = resolve_video_paths()
    n = len(video_paths)
    if not (1 <= n <= 5):
        print(f"❌ 需要 1~5 支影片，目前解析出 {n} 支（INPUT_MODE = {INPUT_MODE!r}）")
        return

    out_dir = OUTPUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"輸出目錄: {out_dir}")
    print(f"共 {n} 支影片待分析\n")

    detector = KeypointDetector(YOLO_MODEL_PATH, device=INFERENCE_DEVICE, imgsz=YOLO_IMGSZ, conf_thres=YOLO_CONF_THRESHOLD)

    video_results = {}
    for video_index, video_path in enumerate(video_paths, start=1):
        result = analyze_single_video(video_path, detector, out_dir, video_index)
        if result is not None:
            records, behavior = result
            video_results[Path(video_path).stem] = (records, behavior, video_index)
        print()

    cv2.destroyAllWindows()

    if not video_results:
        print("⚠ 所有影片都沒有產生有效資料，無法產生比較圖")
        return

    if len(video_results) >= 2:
        build_comparison_chart(video_results, out_dir)
    else:
        print("只有 1 支影片有有效資料，略過多影片比較圖（該影片自己的圖表已存在對應子資料夾）")
        # 單支影片時也把它的圖秀出來，維持互動檢視體驗
        only_name = next(iter(video_results))
        img = plt.imread(str(out_dir / only_name / "bone_length_cv_chart.png"))
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.show()

    # jitter 門檻分析圖不需要多支影片才有意義（單支影片也能看自己的
    # ECDF／flag rate），所以獨立於上面的比較圖分支之外，只要有資料就跑。
    build_jitter_threshold_chart(video_results, out_dir)


if __name__ == "__main__":
    main()
