from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import shutil
from datetime import datetime

from constants import (
    EAR_DISTANCE_SKELETON_EDGES as SKELETON_EDGES,
    EAR_DISTANCE_EDGE_COLORS as SKELETON_EDGE_COLORS,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =====================================================
# ⭐ 模型設定
# =====================================================
MODELS = {
    "640.1": {
        "path": str(_PROJECT_ROOT / "yolo_models" / "v11s_149.pt"),
        "imgsz": 640,
        # label will be set to the .pt filename below
    },
    "640.2": {
        "path": str(_PROJECT_ROOT / "yolo_models" / "v11s_149.pt"),
        "imgsz": 640,
        # label will be set to the .pt filename below
    }
}

# 若設定 YOLO_MODEL_PATH 環境變數，優先套用到 MODELS 的每一個模型（覆蓋上面寫死的
# path，對應 settings_window.py 的「🧠 模型路徑」欄位）。COMPARE_TWO_MODELS=False（單
# 模型模式）時只有 SINGLE_MODEL_KEY 那個模型會被實際用到，但這裡仍統一覆寫全部項目，
# 維持跟預設值「兩個 key 本來就指向同一支模型」一致的行為。
_env_yolo_model = os.getenv("YOLO_MODEL_PATH", "").strip()
if _env_yolo_model:
    for cfg in MODELS.values():
        cfg["path"] = _env_yolo_model

# Automatically set label to the .pt filename
for cfg in MODELS.values():
    cfg["label"] = Path(cfg["path"]).name

# =====================================================
# ⭐ 推論模式旗標
# =====================================================
# True  = 雙模型比較：載入 MODELS 中前兩個模型，產生並排偏移比較圖
# False = 單模型推論：只載入 SINGLE_MODEL_KEY 指定的模型，產生單張標註圖
COMPARE_TWO_MODELS = False

# COMPARE_TWO_MODELS=False 時使用；值必須是 MODELS 中已存在的 key。
SINGLE_MODEL_KEY = "640.1"

INPUT_DIR = r"C:\Users\homec\OneDrive\圖片\Screenshots\cat_test_image2"

# 若設定 TEST_VIDEO_PATH 環境變數且指向資料夾，優先只處理該資料夾（覆蓋上面寫死的
# INPUT_DIR，對應 settings_window.py 的「🎬 影片路徑」欄位）。本腳本處理的是圖片
# 資料夾而非單一影片檔，所以只在填的是資料夾路徑時才生效；填的是單一檔案則安靜
# 忽略，跟沒填一樣。
_env_test_video = os.getenv("TEST_VIDEO_PATH", "").strip()
if _env_test_video and os.path.isdir(_env_test_video):
    INPUT_DIR = _env_test_video

# 前處理：YOLO Pose 訓練時用 imgsz=640，這裡先把 INPUT_DIR 底下的圖片等比
# 壓縮到最長邊 640px，寫進獨立的工作副本快取資料夾（RESIZED_CACHE_DIR），
# 不動 INPUT_DIR 的原始檔案；後續推論、存檔（compare_output / offset_dataset）
# 全部沿用這份 640 版本的工作副本。
RESIZE_MAX_SIDE = 640

CONF_THRES = 0.9
KP_CONF_THRES = 0.8       # 關鍵點信心門檻：低於此值的點不列入偏移比較
DIFF_THRES_PERCENT = 2.0  # 偏移閾值：圖片對角線的百分比
TOTAL_KPTS = 17

# 是否在輸出圖上畫骨架連線（骨架定義與配色改從 constants.py 統一 import，
# 該檔案直接複製自 paper/cat_monitoring_system/utils/constants.py 目前使用中
# 的版本，17 個關鍵點索引順序與主專案一致）
DRAW_SKELETON_LINES = True

# True  = 推論前先將圖片等比縮放至最長邊 640px（模擬訓練解析度）
# False = 直接傳原圖路徑，YOLO 內部自動 letterbox 縮放（預設行為）
RESIZE_INPUT_TO_640 = True

# 輸出並排圖最大寬度（像素），超過則等比縮小存檔；設 None 不限制
MAX_OUTPUT_WIDTH = 3840

# 推論全部完成後，開啟人工審核 GUI。
# D 為紅色待刪標記；A 為綠色選取，可將對應原圖複製到 selected_images。
# 「標記刪除」只會加入待刪清單；必須再按「確定刪除已標記圖片」
# 並通過確認視窗，才會刪除 INPUT_DIR 中對應的原始圖片。
ENABLE_REVIEW_GUI = True

# =====================================================
# ⭐ cat_Compare 主資料夾
# =====================================================
BASE_DIR = Path(r"C:\cat_pose\cat_Compare")
COMPARE_DIR = BASE_DIR / "compare_output"
SINGLE_OUTPUT_DIR = BASE_DIR / "single_output"
OFFSET_DIR = BASE_DIR / "offset_dataset"
# GUI 按 A 綠色選取的原始圖片會複製到這裡；此資料夾不會在每次執行時清空。
SELECTED_IMAGES_DIR = BASE_DIR / "selected_images"
# 640px 工作副本快取（前處理縮圖寫在這裡，絕不動 INPUT_DIR 的原始檔案）
RESIZED_CACHE_DIR = BASE_DIR / "_resized_640_cache"
DELETE_LOG_PATH = BASE_DIR / "deleted_original_images.log"

COMPARE_DIR.mkdir(parents=True, exist_ok=True)
SINGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OFFSET_DIR.mkdir(parents=True, exist_ok=True)
SELECTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# 建立 offset_x/original + offset_x/yolo
for i in range(TOTAL_KPTS + 1):
    (OFFSET_DIR / f"offset_{i}" / "original").mkdir(parents=True, exist_ok=True)
    (OFFSET_DIR / f"offset_{i}" / "yolo").mkdir(parents=True, exist_ok=True)

# =====================================================
# 清空輸出資料夾
# =====================================================
print("🧹 清空輸出資料夾...")

if COMPARE_TWO_MODELS:
    # 雙模型模式只清理自己的輸出，保留上一次單模型結果。
    for file in COMPARE_DIR.glob("*.jpg"):
        file.unlink()

    for i in range(TOTAL_KPTS + 1):
        original_dir = OFFSET_DIR / f"offset_{i}" / "original"
        yolo_dir = OFFSET_DIR / f"offset_{i}" / "yolo"

        for file in original_dir.glob("*"):
            if file.is_file():
                file.unlink()

        for file in yolo_dir.glob("*"):
            if file.is_file():
                file.unlink()
else:
    # 單模型模式只清理自己的輸出，保留上一次雙模型比較結果。
    for file in SINGLE_OUTPUT_DIR.glob("*.jpg"):
        file.unlink()

print("✅ 資料夾已清空")

# =====================================================
# 載入模型
# =====================================================
if COMPARE_TWO_MODELS:
    active_model_names = list(MODELS.keys())[:2]
    if len(active_model_names) < 2:
        raise ValueError("雙模型比較模式至少需要在 MODELS 中設定兩個模型")
    mode_text = "雙模型比較"
else:
    if SINGLE_MODEL_KEY not in MODELS:
        raise ValueError(
            f"SINGLE_MODEL_KEY={SINGLE_MODEL_KEY!r} 不存在，"
            f"可用模型：{list(MODELS.keys())}"
        )
    active_model_names = [SINGLE_MODEL_KEY]
    mode_text = "單模型推論"

print(f"🔄 載入模型中（{mode_text}）：{active_model_names}")
models = {
    name: YOLO(MODELS[name]["path"])
    for name in active_model_names
}
print("✅ 模型載入成功")

# =====================================================
# 工具函式
# =====================================================
def calculate_diagonal(img_shape):
    """計算圖片對角線長度"""
    h, w = img_shape[:2]
    return np.sqrt(h**2 + w**2)


def resize_to_fit(img, max_side=640):
    """等比縮放，使最長邊不超過 max_side；已在範圍內則原樣返回"""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def get_scale(img_shape, base=640):
    """根據圖片短邊計算比例因子，基準為 640px"""
    h, w = img_shape[:2]
    return min(h, w) / base


def pad_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    return cv2.copyMakeBorder(
        img, 0, target_h - h, 0, 0,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def draw_label(img, text, bg_color=(0, 0, 0), scale=1.0):
    s = max(scale, 0.4)
    rect_w = int(320 * s)
    rect_h = int(56 * s)
    font_scale = max(0.5, 1.0 * s)
    thickness = max(1, int(2 * s))
    text_y = int(38 * s)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (rect_w, rect_h), bg_color, -1)
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    cv2.putText(
        img, text, (int(10 * s), text_y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )
    return img


def compute_keypoint_diff(k_ref, k_cmp, diagonal, conf_ref=None, conf_cmp=None):
    """計算關鍵點差異；任一模型信心不足的點差值強制設為 0，不列入偏移統計"""
    diffs_px = np.linalg.norm(k_ref - k_cmp, axis=1)
    diffs_percent = (diffs_px / diagonal) * 100

    # 過濾低信心點
    if conf_ref is not None and conf_cmp is not None:
        low_conf_mask = (conf_ref < KP_CONF_THRES) | (conf_cmp < KP_CONF_THRES)
        diffs_px[low_conf_mask] = 0.0
        diffs_percent[low_conf_mask] = 0.0

    max_idx = int(np.argmax(diffs_percent))
    return diffs_percent, diffs_px, max_idx


def draw_skeleton_lines(img, keypoints, kpt_conf=None, conf_thres=KP_CONF_THRES, scale=1.0):
    """畫骨架連線（骨架定義/配色參考主專案 utils/constants.py）。
    任一端點信心低於 conf_thres 就跳過該條線，避免畫到不可靠的關鍵點位置。"""
    thickness = max(1, int(2 * max(scale, 0.3)))
    img_out = img.copy()
    for edge_idx, (i, j) in enumerate(SKELETON_EDGES):
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        if kpt_conf is not None and (kpt_conf[i] < conf_thres or kpt_conf[j] < conf_thres):
            continue
        pt1 = tuple(keypoints[i].astype(int))
        pt2 = tuple(keypoints[j].astype(int))
        color = SKELETON_EDGE_COLORS[edge_idx] if edge_idx < len(SKELETON_EDGE_COLORS) else (180, 180, 180)
        cv2.line(img_out, pt1, pt2, color, thickness, cv2.LINE_AA)
    return img_out


def mark_offset_points_ultra_clear(img, keypoints, diffs_percent, threshold_percent, scale=1.0, kpt_conf=None):
    """
    超清晰標記 - 異常點與正常點大小一致，避免遮擋
    特點：
    1. 異常點保持多層同心圓顏色，但尺寸縮小至正常點大小
    2. 只顯示編號，不顯示百分比
    3. 編號文字超小
    4. 視覺層級清晰
    5. 所有尺寸依圖片解析度動態縮放
    6. 信心值低於 KP_CONF_THRES 的點直接跳過，不繪製
    """
    s = max(scale, 0.3)
    r1 = max(2, int(12 * s))
    r2 = max(2, int(10 * s))
    r3 = max(2, int(8 * s))
    r4 = max(1, int(6 * s))
    rc = max(1, int(2 * s))
    rg = max(1, int(6 * s))
    rgo = max(1, int(8 * s))
    font_scale = max(0.3, 0.45 * s)
    txt_thickness = max(1, int(2 * s))
    txt_offset_x = int(16 * s)
    txt_offset_y = int(8 * s)

    img_marked = img.copy()
    
    for i, (kpt, diff_pct) in enumerate(zip(keypoints, diffs_percent)):
        # 信心不足 → 跳過，不繪製
        if kpt_conf is not None and kpt_conf[i] < KP_CONF_THRES:
            continue

        x, y = kpt.astype(int)

        if diff_pct > threshold_percent:
            # ==========================================
            # 🔴 偏移點 - 保持顏色警告，但尺寸與正常點一致
            # ==========================================
            
            # 第1層：最外圈白色光暈（醒目）
            cv2.circle(img_marked, (x, y), r1, (255, 255, 255), 1, cv2.LINE_AA)
            
            # 第2層：黃色警告圈（中層）
            cv2.circle(img_marked, (x, y), r2, (0, 255, 255), 1, cv2.LINE_AA)
            
            # 第3層：橙色過渡圈
            cv2.circle(img_marked, (x, y), r3, (0, 165, 255), 1, cv2.LINE_AA)
            
            # 第4層：紅色主圈
            cv2.circle(img_marked, (x, y), r4, (0, 0, 255), -1, cv2.LINE_AA)
            
            # 白色中心點（對比）
            cv2.circle(img_marked, (x, y), rc, (255, 255, 255), -1, cv2.LINE_AA)
            
            # ==========================================
            # 📝 編號標記 - 超小，只顯示編號
            # ==========================================
            text_num = f"#{i}"
            text_pos = (x + txt_offset_x, y - txt_offset_y)
            
            # 精簡黑色陰影（2方向）
            for dx, dy in [(1, 1), (-1, -1)]:
                cv2.putText(
                    img_marked, text_num,
                    (text_pos[0] + dx, text_pos[1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), txt_thickness, cv2.LINE_AA
                )
            
            # 白色主體文字（超小）
            cv2.putText(
                img_marked, text_num, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, txt_thickness - 1), cv2.LINE_AA
            )
            
        else:
            # ==========================================
            # ✅ 正確點 - 低調但清晰
            # ==========================================
            # 雙層設計
            cv2.circle(img_marked, (x, y), rg, (0, 255, 0), -1, cv2.LINE_AA)   # 綠色實心
            cv2.circle(img_marked, (x, y), rgo, (255, 255, 255), 1, cv2.LINE_AA)  # 白色外圈
            cv2.circle(img_marked, (x, y), rc, (255, 255, 255), -1, cv2.LINE_AA)  # 白色中心點
    
    return img_marked


def create_side_by_side_comparison(img_ref, img_cmp, k_ref, k_cmp, diffs_percent, diffs_px,
                                   threshold_percent, model_names, conf_ref=None, conf_cmp=None):
    """
    創建並排對比圖 - 骨架連線 + 清晰標記，所有文字/圖形依解析度縮放
    """
    # 以參考圖短邊計算全局比例因子
    s = get_scale(img_ref.shape)
    KEYPOINT_NAMES = [
        "nose",               # 0
        "left_ear_tip",       # 1
        "right_ear_tip",      # 2
        "chest",              # 3
        "mid_back",           # 4
        "hip",                # 5
        "left_front_elbow",   # 6
        "left_front_paw",     # 7
        "right_front_elbow",  # 8
        "right_front_paw",    # 9
        "left_hind_knee",     # 10
        "left_hind_paw",      # 11
        "right_hind_knee",    # 12
        "right_hind_paw",     # 13
        "tail_base",          # 14
        "tail_mid",           # 15
        "tail_tip"            # 16
    ]
    
    # 先畫骨架連線，再疊加關鍵點標記（連線在下層，不會蓋住標記點）
    if DRAW_SKELETON_LINES:
        img_ref = draw_skeleton_lines(img_ref, k_ref, kpt_conf=conf_ref, scale=s)
        img_cmp = draw_skeleton_lines(img_cmp, k_cmp, kpt_conf=conf_cmp, scale=s)

    # 在兩張圖上標記
    img_ref_marked = mark_offset_points_ultra_clear(img_ref, k_ref, diffs_percent, threshold_percent, scale=s, kpt_conf=conf_ref)
    img_cmp_marked = mark_offset_points_ultra_clear(img_cmp, k_cmp, diffs_percent, threshold_percent, scale=s, kpt_conf=conf_cmp)
    
    # 添加模型標籤（不同顏色）
    img_ref_marked = draw_label(img_ref_marked, MODELS[model_names[0]]["label"], (60, 60, 180), scale=s)
    img_cmp_marked = draw_label(img_cmp_marked, MODELS[model_names[1]]["label"], (60, 180, 60), scale=s)
    
    # 統一高度並排
    max_h = max(img_ref_marked.shape[0], img_cmp_marked.shape[0])
    img_ref_marked = pad_to_height(img_ref_marked, max_h)
    img_cmp_marked = pad_to_height(img_cmp_marked, max_h)
    
    comparison = np.hstack([img_ref_marked, img_cmp_marked])
    
    # ==========================================
    # 底部信息面板（依比例縮放）
    # ==========================================
    panel_height = max(200, int(250 * s))
    panel = np.zeros((panel_height, comparison.shape[1], 3), dtype=np.uint8)

    # 縮放後的字體與間距
    fs_title  = max(0.6, 1.3 * s)
    fs_stats  = max(0.45, 0.75 * s)
    fs_detail = max(0.4, 0.8 * s)
    fs_list   = max(0.35, 0.55 * s)
    fs_note   = max(0.35, 0.6 * s)
    th_title  = max(1, int(3 * s))
    th_std    = max(1, int(2 * s))
    line_h_lg = max(20, int(45 * s))
    line_h_md = max(16, int(35 * s))
    line_h_sm = max(12, int(30 * s))
    line_h_xs = max(10, int(25 * s))
    pad       = max(8, int(15 * s))
    legend_cr = max(5, int(15 * s))
    legend_co = max(4, int(10 * s))

    offset_indices = np.where(diffs_percent > threshold_percent)[0]
    correct_count = len(diffs_percent) - len(offset_indices)
    
    y_pos = max(20, int(35 * s))
    
    # 主標題
    cv2.putText(panel, "OFFSET ANALYSIS REPORT",
               (pad, y_pos), cv2.FONT_HERSHEY_SIMPLEX, fs_title, (0, 255, 255), th_title)
    y_pos += line_h_lg
    
    # 統計信息
    stats_text = f"Threshold: {threshold_percent:.1f}%  |  Total Points: {len(diffs_percent)}  |  Offset: {len(offset_indices)}  |  Correct: {correct_count}"
    cv2.putText(panel, stats_text,
               (pad, y_pos), cv2.FONT_HERSHEY_SIMPLEX, fs_stats, (255, 255, 255), th_std)
    y_pos += line_h_md
    
    # 分隔線
    cv2.line(panel, (pad, y_pos), (comparison.shape[1] - pad, y_pos), (100, 100, 100), th_std)
    y_pos += line_h_sm
    
    # 圖例（右上角）
    legend_x = comparison.shape[1] - max(300, int(450 * s))
    legend_y = max(15, int(30 * s))
    legend_gap = max(20, int(40 * s))

    # 偏移點圖例
    cv2.circle(panel, (legend_x, legend_y + legend_cr // 2), legend_cr, (0, 0, 255), -1)
    cv2.circle(panel, (legend_x, legend_y + legend_cr // 2), legend_cr + max(1, int(3 * s)), (255, 255, 255), th_std)
    cv2.putText(panel, "= OFFSET POINT (Error)",
               (legend_x + legend_cr + max(8, int(15 * s)), legend_y + legend_cr),
               cv2.FONT_HERSHEY_SIMPLEX, fs_stats, (255, 255, 255), th_std)
    
    # 正確點圖例
    cv2.circle(panel, (legend_x, legend_y + legend_gap + legend_co), legend_co, (0, 255, 0), -1)
    cv2.circle(panel, (legend_x, legend_y + legend_gap + legend_co), legend_co + max(1, int(3 * s)), (255, 255, 255), th_std)
    cv2.putText(panel, "= CORRECT POINT",
               (legend_x + legend_co + max(8, int(15 * s)), legend_y + legend_gap + legend_co + max(3, int(5 * s))),
               cv2.FONT_HERSHEY_SIMPLEX, fs_stats, (255, 255, 255), th_std)
    
    # 偏移詳細列表
    cv2.putText(panel, "OFFSET DETAILS:",
               (pad, y_pos), cv2.FONT_HERSHEY_SIMPLEX, fs_detail, (255, 255, 0), th_std)
    y_pos += line_h_sm
    
    # 分三欄顯示
    col_width = comparison.shape[1] // 3
    y_cols = [y_pos, y_pos, y_pos]
    
    for idx_num, idx in enumerate(offset_indices):
        col = idx_num % 3
        
        if y_cols[col] < panel_height - pad:
            kpt_name = KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else f"kpt_{idx}"
            text = f"#{idx:2d} {kpt_name:12s}: {diffs_percent[idx]:4.1f}%"
            
            x_pos = pad + col * col_width
            cv2.putText(panel, text, (x_pos, y_cols[col]),
                       cv2.FONT_HERSHEY_SIMPLEX, fs_list, (0, 0, 255), max(1, th_std - 1))
            y_cols[col] += line_h_xs
    
    # 如果列表太長
    if any(y >= panel_height - pad for y in y_cols):
        cv2.putText(panel, "... (see console for complete list)",
                   (pad, panel_height - pad),
                   cv2.FONT_HERSHEY_SIMPLEX, fs_note, (150, 150, 150), 1)
    
    # 組合
    final_img = np.vstack([comparison, panel])
    return final_img


# =====================================================
# 推論完成後的人工刪圖審核 GUI
# =====================================================
def launch_review_gui(review_records):
    """
    顯示已推論的結果圖，讓使用者標記待刪圖片或綠色選取圖片。

    review_records 每筆資料至少包含：
        preview_path: GUI 顯示的推論結果圖
        source_path:  INPUT_DIR 中真正的原始圖片

    「標記刪除」不會立即動到檔案；只有在刪除確認視窗再次按下
    「永久刪除以上原始圖片」時，才會刪除 source_path。
    按 A 加入綠色選取後，可將對應原始圖片複製到 SELECTED_IMAGES_DIR。
    """
    if not review_records:
        print("⚠ 沒有可供 GUI 審核的圖片")
        return

    try:
        import tkinter as tk
        from tkinter import messagebox
        from PIL import Image, ImageOps, ImageTk
    except ImportError as exc:
        print(f"⚠ 無法開啟審核 GUI：{exc}")
        print("請先安裝 Pillow：pip install pillow")
        return

    input_root = Path(INPUT_DIR).resolve()
    marked_paths = set()
    selected_paths = set()
    current_index = 0
    current_photo = None
    resize_job = None

    root = tk.Tk()
    root.title("YOLO Pose 圖片篩選與原圖管理")
    root.geometry("1600x1000")
    root.minsize(900, 650)
    root.configure(bg="#111827")
    root.grid_columnconfigure(0, weight=1)
    # 只有圖片列可以伸縮，其餘控制列保留固定高度，不會被圖片擠出畫面。
    root.grid_rowconfigure(2, weight=1, minsize=300)

    # Windows 啟動時直接最大化，讓推論圖取得最多顯示空間。
    try:
        root.state("zoomed")
    except tk.TclError:
        pass

    title_label = tk.Label(
        root,
        text=f"{mode_text}｜推論結果人工審核",
        font=("Microsoft JhengHei UI", 13, "bold"),
        fg="#F9FAFB",
        bg="#111827",
    )
    title_label.grid(row=0, column=0, sticky="ew", pady=(5, 0))

    status_label = tk.Label(
        root,
        text="",
        font=("Microsoft JhengHei UI", 9),
        fg="#CBD5E1",
        bg="#111827",
    )
    status_label.grid(row=1, column=0, sticky="ew", pady=(0, 3))

    image_frame = tk.Frame(root, bg="#030712", highlightthickness=2, highlightbackground="#374151")
    image_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=2)
    image_frame.grid_propagate(False)

    image_label = tk.Label(
        image_frame,
        text="正在載入圖片...",
        font=("Microsoft JhengHei UI", 14),
        fg="#E5E7EB",
        bg="#030712",
    )
    image_label.pack(fill="both", expand=True, padx=2, pady=2)

    source_label = tk.Label(
        root,
        text="",
        font=("Microsoft JhengHei UI", 9),
        fg="#D1D5DB",
        bg="#1F2937",
        anchor="w",
        justify="left",
        padx=7,
        pady=3,
    )
    source_label.grid(row=3, column=0, sticky="ew", padx=6, pady=(3, 2))

    help_label = tk.Label(
        root,
        text="快捷鍵：← 上一張　→／Space 下一張　A 綠色選取　D 紅色待刪　Esc 關閉",
        font=("Microsoft JhengHei UI", 8),
        fg="#94A3B8",
        bg="#111827",
    )
    help_label.grid(row=4, column=0, sticky="ew", pady=(0, 2))

    button_frame = tk.Frame(root, bg="#111827")
    button_frame.grid(row=5, column=0, sticky="ew", padx=6, pady=(1, 6))

    # 分成兩排，避免低解析度或較窄視窗把右側功能擠出畫面。
    navigation_row = tk.Frame(button_frame, bg="#111827")
    navigation_row.pack(fill="x", pady=(0, 3))
    action_row = tk.Frame(button_frame, bg="#111827")
    action_row.pack(fill="x")

    for column in range(4):
        navigation_row.grid_columnconfigure(column, weight=1, uniform="navigation")
    for column in range(3):
        action_row.grid_columnconfigure(column, weight=1, uniform="actions")

    def make_button(parent, text, command, bg, width=15):
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            font=("Microsoft JhengHei UI", 9, "bold"),
            fg="white",
            bg=bg,
            activeforeground="white",
            activebackground=bg,
            relief="flat",
            bd=0,
            padx=5,
            pady=4,
            cursor="hand2",
        )

    def current_record():
        if not review_records:
            return None
        return review_records[current_index]

    def render_current():
        nonlocal current_photo

        record = current_record()
        if record is None:
            image_label.configure(image="", text="沒有剩餘的審核圖片")
            source_label.configure(text="")
            status_label.configure(text="已完成所有圖片審核")
            mark_button.configure(text="D 待刪", state="disabled", bg="#DC2626")
            select_button.configure(text="A 選取", state="disabled", bg="#15803D")
            image_frame.configure(highlightbackground="#374151")
            current_photo = None
            return

        source_key = str(Path(record["source_path"]).resolve())
        is_marked = source_key in marked_paths
        is_selected = source_key in selected_paths
        if record.get("mode") == "compare":
            inference_detail = f"偏移點 {record.get('offset_count', 0)}"
        else:
            inference_detail = f"姿態實例 {record.get('detected_instances', 0)}"

        status_label.configure(
            text=(
                f"第 {current_index + 1} / {len(review_records)} 張　｜　"
                f"{inference_detail}　｜　綠色 {len(selected_paths)} 張　｜　"
                f"待刪 {len(marked_paths)} 張"
            ),
            fg="#FCA5A5" if is_marked else ("#86EFAC" if is_selected else "#CBD5E1"),
        )
        source_label.configure(text=f"原始圖片：{record['source_path']}")
        mark_button.configure(
            text="D 取消待刪" if is_marked else "D 待刪",
            state="normal",
            bg="#F59E0B" if is_marked else "#DC2626",
            activebackground="#D97706" if is_marked else "#B91C1C",
        )
        select_button.configure(
            text="A 取消選取" if is_selected else "A 選取",
            state="normal",
            bg="#65A30D" if is_selected else "#15803D",
            activebackground="#4D7C0F" if is_selected else "#166534",
        )
        if is_marked:
            border_color = "#EF4444"
        elif is_selected:
            border_color = "#22C55E"
        else:
            border_color = "#374151"
        image_frame.configure(highlightbackground=border_color)

        try:
            with Image.open(record["preview_path"]) as opened:
                display_img = ImageOps.exif_transpose(opened).convert("RGB")

            max_w = max(500, image_frame.winfo_width() - 8)
            max_h = max(350, image_frame.winfo_height() - 8)
            resampling = getattr(Image, "Resampling", Image).LANCZOS

            # thumbnail() 只會縮小、不會放大；這裡自行計算比例，讓較小的
            # 推論圖也能放大到圖片區邊界，並維持原始長寬比、不裁切內容。
            scale = min(max_w / display_img.width, max_h / display_img.height)
            display_size = (
                max(1, int(display_img.width * scale)),
                max(1, int(display_img.height * scale)),
            )
            if display_size != display_img.size:
                display_img = display_img.resize(display_size, resampling)
            current_photo = ImageTk.PhotoImage(display_img)
            image_label.configure(image=current_photo, text="")
        except Exception as exc:
            current_photo = None
            image_label.configure(
                image="",
                text=f"無法讀取推論結果圖\n{record['preview_path']}\n\n{exc}",
            )

    def previous_image(event=None):
        nonlocal current_index
        if review_records:
            current_index = (current_index - 1) % len(review_records)
            render_current()

    def next_image(event=None):
        nonlocal current_index
        if review_records:
            current_index = (current_index + 1) % len(review_records)
            render_current()

    def toggle_mark(event=None):
        record = current_record()
        if record is None:
            return

        source_key = str(Path(record["source_path"]).resolve())
        if source_key in marked_paths:
            marked_paths.remove(source_key)
        else:
            # 紅色待刪與綠色選取互斥，避免同一張圖同時被保留與刪除。
            selected_paths.discard(source_key)
            marked_paths.add(source_key)
        render_current()

    def toggle_selected(event=None):
        record = current_record()
        if record is None:
            return

        source_key = str(Path(record["source_path"]).resolve())
        if source_key in selected_paths:
            selected_paths.remove(source_key)
        else:
            # 綠色選取與紅色待刪互斥。
            marked_paths.discard(source_key)
            selected_paths.add(source_key)
        render_current()

    def export_selected_originals():
        if not selected_paths:
            messagebox.showinfo("沒有綠色選取", "目前沒有任何綠色選取圖片。", parent=root)
            return

        copied = []
        failed = []
        SELECTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        for source_key in sorted(selected_paths):
            source_path = Path(source_key).resolve()
            try:
                # 只允許匯出本次 INPUT_DIR 建立映射的原始圖片。
                source_path.relative_to(input_root)
                if not source_path.is_file():
                    raise FileNotFoundError("原始圖片不存在或不是檔案")

                destination = SELECTED_IMAGES_DIR / source_path.name
                if source_path == destination.resolve():
                    raise ValueError("來源與目的路徑相同")
                shutil.copy2(source_path, destination)
                copied.append(destination)
            except Exception as exc:
                failed.append((source_path, str(exc)))

        render_current()
        if failed:
            details = "\n".join(f"{path}: {reason}" for path, reason in failed[:10])
            messagebox.showwarning(
                "部分圖片匯出失敗",
                f"成功複製：{len(copied)} 張\n失敗：{len(failed)} 張\n\n{details}",
                parent=root,
            )
        else:
            messagebox.showinfo(
                "綠色選取匯出完成",
                f"已複製 {len(copied)} 張原始圖片。\n\n輸出位置：{SELECTED_IMAGES_DIR}",
                parent=root,
            )

    def delete_marked_originals():
        nonlocal current_index

        if not marked_paths:
            messagebox.showinfo("沒有標記", "目前沒有任何待刪除圖片。", parent=root)
            return

        targets = []
        unsafe_targets = []
        for source_key in sorted(marked_paths):
            source_path = Path(source_key).resolve()
            try:
                source_path.relative_to(input_root)
                targets.append(source_path)
            except ValueError:
                unsafe_targets.append(source_path)

        if unsafe_targets:
            messagebox.showerror(
                "路徑安全檢查失敗",
                "下列檔案不位於 INPUT_DIR 中，已停止刪除：\n\n"
                + "\n".join(str(path) for path in unsafe_targets[:10]),
                parent=root,
            )
            return

        confirm_window = tk.Toplevel(root)
        confirm_window.title("確認刪除原始圖片")
        confirm_window.geometry("850x560")
        confirm_window.minsize(650, 420)
        confirm_window.configure(bg="#111827")
        confirm_window.transient(root)
        confirm_window.grab_set()

        tk.Label(
            confirm_window,
            text=f"即將永久刪除 {len(targets)} 張原始圖片",
            font=("Microsoft JhengHei UI", 16, "bold"),
            fg="#FCA5A5",
            bg="#111827",
        ).pack(pady=(16, 4))

        tk.Label(
            confirm_window,
            text="請核對完整路徑。刪除後無法由本程式復原。",
            font=("Microsoft JhengHei UI", 10),
            fg="#E5E7EB",
            bg="#111827",
        ).pack(pady=(0, 10))

        list_frame = tk.Frame(confirm_window, bg="#111827")
        list_frame.pack(fill="both", expand=True, padx=16, pady=4)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        path_list = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            font=("Consolas", 10),
            bg="#030712",
            fg="#F9FAFB",
            selectbackground="#374151",
            activestyle="none",
        )
        path_list.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=path_list.yview)
        for target in targets:
            path_list.insert("end", str(target))

        confirm_buttons = tk.Frame(confirm_window, bg="#111827")
        confirm_buttons.pack(fill="x", padx=16, pady=14)

        def perform_delete():
            nonlocal current_index
            deleted = []
            failed = []

            for target in targets:
                try:
                    if not target.is_file():
                        raise FileNotFoundError("原始圖片不存在或不是檔案")
                    target.unlink()
                    deleted.append(target)
                except Exception as exc:
                    failed.append((target, str(exc)))

            if deleted:
                try:
                    DELETE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with DELETE_LOG_PATH.open("a", encoding="utf-8") as log_file:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for target in deleted:
                            log_file.write(f"{timestamp}\tDELETED\t{target}\n")
                except Exception as exc:
                    print(f"⚠ 無法寫入刪除紀錄：{exc}")

                deleted_keys = {str(path.resolve()) for path in deleted}
                review_records[:] = [
                    item
                    for item in review_records
                    if str(Path(item["source_path"]).resolve()) not in deleted_keys
                ]
                marked_paths.difference_update(deleted_keys)
                selected_paths.difference_update(deleted_keys)

            confirm_window.destroy()
            if review_records:
                current_index = min(current_index, len(review_records) - 1)
            else:
                current_index = 0
            render_current()

            if failed:
                details = "\n".join(f"{path}: {reason}" for path, reason in failed[:10])
                messagebox.showwarning(
                    "部分檔案刪除失敗",
                    f"成功刪除：{len(deleted)} 張\n失敗：{len(failed)} 張\n\n{details}",
                    parent=root,
                )
            else:
                messagebox.showinfo(
                    "刪除完成",
                    f"已刪除 {len(deleted)} 張原始圖片。\n\n刪除紀錄：{DELETE_LOG_PATH}",
                    parent=root,
                )

        make_button(
            confirm_buttons,
            "取消",
            confirm_window.destroy,
            "#4B5563",
            width=14,
        ).pack(side="left")
        make_button(
            confirm_buttons,
            "永久刪除以上原始圖片",
            perform_delete,
            "#B91C1C",
            width=24,
        ).pack(side="right")

        confirm_window.protocol("WM_DELETE_WINDOW", confirm_window.destroy)
        confirm_window.focus_set()

    previous_button = make_button(
        navigation_row, "← 上一張", previous_image, "#2563EB", width=0
    )
    previous_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))

    next_button = make_button(
        navigation_row, "下一張 →", next_image, "#2563EB", width=0
    )
    next_button.grid(row=0, column=1, sticky="ew", padx=3)

    select_button = make_button(
        navigation_row,
        "A 選取",
        toggle_selected,
        "#15803D",
        width=0,
    )
    select_button.grid(row=0, column=2, sticky="ew", padx=3)

    mark_button = make_button(
        navigation_row, "D 待刪", toggle_mark, "#DC2626", width=0
    )
    mark_button.grid(row=0, column=3, sticky="ew", padx=(3, 0))

    export_button = make_button(
        action_row,
        "匯出綠色圖片",
        export_selected_originals,
        "#166534",
        width=0,
    )
    export_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))

    delete_button = make_button(
        action_row,
        "刪除紅色圖片",
        delete_marked_originals,
        "#7F1D1D",
        width=0,
    )
    delete_button.grid(row=0, column=1, sticky="ew", padx=3)

    close_button = make_button(
        action_row, "關閉", root.destroy, "#4B5563", width=0
    )
    close_button.grid(row=0, column=2, sticky="ew", padx=(3, 0))

    def rerender_after_resize():
        nonlocal resize_job
        resize_job = None
        render_current()

    def schedule_rerender(event):
        nonlocal resize_job
        if event.widget is not root:
            return
        if resize_job is not None:
            root.after_cancel(resize_job)
        resize_job = root.after(120, rerender_after_resize)

    root.bind("<Left>", previous_image)
    root.bind("<Right>", next_image)
    root.bind("<space>", next_image)
    root.bind("<Key-d>", toggle_mark)
    root.bind("<Key-D>", toggle_mark)
    root.bind("<Key-a>", toggle_selected)
    root.bind("<Key-A>", toggle_selected)
    root.bind("<Escape>", lambda event: root.destroy())
    root.bind("<Configure>", schedule_rerender)

    root.after(100, render_current)
    root.mainloop()


# =====================================================
# 前處理：壓縮圖片至 640 + 讀取圖片
# =====================================================
IMAGE_EXT = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

src_paths = []
for ext in IMAGE_EXT:
    src_paths.extend(Path(INPUT_DIR).glob(ext))
src_paths.sort(key=lambda path: path.name.lower())

if not src_paths:
    print("⚠ 找不到任何圖片")
    exit()

print(f"🔧 前處理：壓縮 {len(src_paths)} 張圖片至最長邊 {RESIZE_MAX_SIDE}px（寫入工作副本，原始檔案不動）...")

RESIZED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
image_items = []
for src_path in src_paths:
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"  ⚠ 無法讀取，略過: {src_path.name}")
        continue
    cache_path = RESIZED_CACHE_DIR / src_path.name
    cv2.imwrite(str(cache_path), resize_to_fit(img, max_side=RESIZE_MAX_SIDE))
    # 同時保留「真正原圖」與「640 工作副本」的映射，GUI 刪除時只使用 source_path。
    image_items.append({
        "source_path": src_path.resolve(),
        "cache_path": cache_path.resolve(),
    })

print(f"✅ 前處理完成，工作副本存於: {RESIZED_CACHE_DIR}（{INPUT_DIR} 內的原始檔案未被修改）")

# =====================================================
# 主流程
# =====================================================
model_names = active_model_names

if COMPARE_TWO_MODELS:
    ref_name, cmp_name = model_names
    if MODELS[ref_name]["path"] == MODELS[cmp_name]["path"]:
        print(
            f"⚠️ MODELS 設定中「{ref_name}」與「{cmp_name}」目前指向同一個模型檔案"
            f"（{MODELS[ref_name]['path']}），比較結果偏移量必然全為 0，等於沒有在做真正的比較。"
            f"請把其中一個模型路徑改成你真正要比較的另一顆模型。"
        )
else:
    single_name = model_names[0]

offset_hist = {i: 0 for i in range(TOTAL_KPTS + 1)}
single_detected_images = 0
single_no_detection_images = 0
single_total_instances = 0
total_images = 0
diff_threshold_px_list = []
review_records = []

for idx, image_item in enumerate(image_items, start=1):
    source_path = image_item["source_path"]
    img_path = image_item["cache_path"]
    review_id = f"{idx:06d}"
    print(f"[{idx}/{len(image_items)}] {source_path.name}")

    # ---------- 載入 640 工作副本 ----------
    original_img = cv2.imread(str(img_path))

    # ---------- 決定推論輸入 ----------
    if RESIZE_INPUT_TO_640:
        infer_img = resize_to_fit(original_img, max_side=640)
    else:
        infer_img = str(img_path)

    if COMPARE_TWO_MODELS:
        # =====================================================
        # 模式一：雙模型比較
        # =====================================================
        ref_shape = infer_img.shape if RESIZE_INPUT_TO_640 else original_img.shape
        diagonal = calculate_diagonal(ref_shape)
        diff_threshold_px = diagonal * (DIFF_THRES_PERCENT / 100)
        diff_threshold_px_list.append(diff_threshold_px)

        results = {}
        for name in model_names:
            cfg = MODELS[name]
            results[name] = models[name].predict(
                infer_img,
                imgsz=cfg["imgsz"],
                conf=CONF_THRES,
                verbose=False,
            )[0]

        ref_keypoints = results[ref_name].keypoints
        cmp_keypoints = results[cmp_name].keypoints
        has_ref = ref_keypoints is not None and len(ref_keypoints) > 0
        has_cmp = cmp_keypoints is not None and len(cmp_keypoints) > 0
        has_both = has_ref and has_cmp

        if not has_both:
            offset_count = TOTAL_KPTS
            print("  ⚠ 至少一個模型無法檢測到關鍵點")
        else:
            k_ref = ref_keypoints.xy[0].cpu().numpy()
            k_cmp = cmp_keypoints.xy[0].cpu().numpy()
            conf_ref = ref_keypoints.conf[0].cpu().numpy()
            conf_cmp = cmp_keypoints.conf[0].cpu().numpy()
            diffs_percent, diffs_px, max_idx = compute_keypoint_diff(
                k_ref, k_cmp, diagonal, conf_ref, conf_cmp
            )
            offset_count = int((diffs_percent > DIFF_THRES_PERCENT).sum())

            if offset_count > 0:
                offset_points = np.where(diffs_percent > DIFF_THRES_PERCENT)[0]
                print(
                    f"  🔴 偏移 {offset_count} 點: {list(offset_points)}, "
                    f"最大 #{max_idx} ({diffs_percent[max_idx]:.2f}%)"
                )
            else:
                print("  ✅ 完全匹配")

        offset_hist[offset_count] += 1

        # offset_dataset/original 延續保存 640 工作副本，不會改動真正原圖。
        shutil.copy(
            img_path,
            OFFSET_DIR / f"offset_{offset_count}" / "original" / img_path.name,
        )

        if has_both:
            yolo_img = results[ref_name].plot(kpt_line=False)
            draw_scale = get_scale(yolo_img.shape)
            if DRAW_SKELETON_LINES:
                yolo_img = draw_skeleton_lines(
                    yolo_img, k_ref, kpt_conf=conf_ref, scale=draw_scale
                )
            yolo_img = mark_offset_points_ultra_clear(
                yolo_img,
                k_ref,
                diffs_percent,
                DIFF_THRES_PERCENT,
                scale=draw_scale,
                kpt_conf=conf_ref,
            )
            yolo_img = draw_label(
                yolo_img, MODELS[ref_name]["label"], scale=draw_scale
            )
            yolo_out = (
                OFFSET_DIR
                / f"offset_{offset_count}"
                / "yolo"
                / f"{review_id}_{img_path.stem}_yolo.jpg"
            )
            cv2.imwrite(str(yolo_out), yolo_img)

            img_ref = results[ref_name].plot(kpt_line=False)
            img_cmp = results[cmp_name].plot(kpt_line=False)
            comparison = create_side_by_side_comparison(
                img_ref,
                img_cmp,
                k_ref,
                k_cmp,
                diffs_percent,
                diffs_px,
                DIFF_THRES_PERCENT,
                model_names,
                conf_ref=conf_ref,
                conf_cmp=conf_cmp,
            )

            if MAX_OUTPUT_WIDTH is not None and comparison.shape[1] > MAX_OUTPUT_WIDTH:
                scale_down = MAX_OUTPUT_WIDTH / comparison.shape[1]
                comparison = cv2.resize(
                    comparison,
                    (MAX_OUTPUT_WIDTH, int(comparison.shape[0] * scale_down)),
                    interpolation=cv2.INTER_AREA,
                )

            preview_path = COMPARE_DIR / f"{review_id}_{img_path.stem}_compare.jpg"
            cv2.imwrite(str(preview_path), comparison)
        else:
            fallback = original_img.copy()
            banner_h = min(max(48, fallback.shape[0] // 10), 100)
            cv2.rectangle(fallback, (0, 0), (fallback.shape[1], banner_h), (0, 0, 170), -1)
            cv2.putText(
                fallback,
                "KEYPOINTS NOT DETECTED - MANUAL REVIEW REQUIRED",
                (12, max(30, int(banner_h * 0.65))),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.45, min(fallback.shape[:2]) / 900),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            preview_path = COMPARE_DIR / f"{review_id}_{img_path.stem}_no_detection.jpg"
            cv2.imwrite(str(preview_path), fallback)

        record_data = {
            "mode": "compare",
            "offset_count": offset_count,
        }

    else:
        # =====================================================
        # 模式二：單模型推論
        # =====================================================
        cfg = MODELS[single_name]
        result = models[single_name].predict(
            infer_img,
            imgsz=cfg["imgsz"],
            conf=CONF_THRES,
            verbose=False,
        )[0]

        keypoints = result.keypoints
        has_detection = keypoints is not None and len(keypoints) > 0

        if has_detection:
            detected_instances = len(keypoints)
            single_detected_images += 1
            single_total_instances += detected_instances
            print(f"  ✅ 偵測到 {detected_instances} 個姿態實例")

            annotated = result.plot(kpt_line=False)
            draw_scale = get_scale(annotated.shape)

            # 單模型模式會替圖片內每一隻偵測到的貓畫上自訂骨架。
            if DRAW_SKELETON_LINES:
                all_xy = keypoints.xy.cpu().numpy()
                all_conf = (
                    keypoints.conf.cpu().numpy()
                    if keypoints.conf is not None
                    else [None] * len(all_xy)
                )
                for instance_xy, instance_conf in zip(all_xy, all_conf):
                    annotated = draw_skeleton_lines(
                        annotated,
                        instance_xy,
                        kpt_conf=instance_conf,
                        scale=draw_scale,
                    )

            annotated = draw_label(
                annotated, MODELS[single_name]["label"], scale=draw_scale
            )
            preview_path = (
                SINGLE_OUTPUT_DIR
                / f"{review_id}_{img_path.stem}_single.jpg"
            )
            cv2.imwrite(str(preview_path), annotated)
        else:
            detected_instances = 0
            single_no_detection_images += 1
            print("  ⚠ 單模型無法檢測到關鍵點")

            fallback = original_img.copy()
            banner_h = min(max(48, fallback.shape[0] // 10), 100)
            cv2.rectangle(fallback, (0, 0), (fallback.shape[1], banner_h), (0, 0, 170), -1)
            cv2.putText(
                fallback,
                "NO POSE DETECTED - MANUAL REVIEW REQUIRED",
                (12, max(30, int(banner_h * 0.65))),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.45, min(fallback.shape[:2]) / 900),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            preview_path = (
                SINGLE_OUTPUT_DIR
                / f"{review_id}_{img_path.stem}_no_detection.jpg"
            )
            cv2.imwrite(str(preview_path), fallback)

        record_data = {
            "mode": "single",
            "detected_instances": detected_instances,
        }

    total_images += 1
    review_records.append({
        "review_id": review_id,
        "source_path": source_path,
        "preview_path": preview_path.resolve(),
        **record_data,
    })

# =====================================================
# 統計輸出
# =====================================================
avg_threshold_px = np.mean(diff_threshold_px_list) if diff_threshold_px_list else 0

print("\n" + "=" * 85)
if COMPARE_TWO_MODELS:
    print("📊 雙模型 Keypoint 偏移統計報告")
    print("=" * 85)
    print(
        f"偏移閾值：{DIFF_THRES_PERCENT}% of diagonal "
        f"(平均 {avg_threshold_px:.1f} px)"
    )
    print(f"總圖片數：{total_images}")
    if total_images > 0:
        print(
            f"✅ 完美匹配：{offset_hist[0]:4d} "
            f"({offset_hist[0] / total_images * 100:5.1f}%)"
        )
        print(
            f"🔴 有偏移：  {total_images - offset_hist[0]:4d} "
            f"({(total_images - offset_hist[0]) / total_images * 100:5.1f}%)"
        )
        print("-" * 85)
        for i in range(TOTAL_KPTS + 1):
            if offset_hist[i] > 0:
                bar = "█" * int(offset_hist[i] / total_images * 50)
                print(
                    f"偏移 {i:2d} 點：{offset_hist[i]:4d} 張 "
                    f"({offset_hist[i] / total_images * 100:5.1f}%) {bar}"
                )
    print("=" * 85)
    print("\n📁 輸出目錄：")
    print(f"  compare_output: {COMPARE_DIR}")
    print(f"  offset_dataset: {OFFSET_DIR}")
    print("\n💡 雙模型模式：紅色為偏移點、綠色為符合點，GUI 顯示並排比較圖。")
else:
    print("📊 單模型姿態推論統計報告")
    print("=" * 85)
    print(f"使用模型：{SINGLE_MODEL_KEY} ({MODELS[single_name]['label']})")
    print(f"總圖片數：{total_images}")
    print(f"✅ 成功偵測圖片：{single_detected_images}")
    print(f"⚠ 未偵測圖片：  {single_no_detection_images}")
    print(f"偵測姿態總數：  {single_total_instances}")
    print("=" * 85)
    print("\n📁 輸出目錄：")
    print(f"  single_output: {SINGLE_OUTPUT_DIR}")
    print("\n💡 單模型模式：GUI 顯示指定模型的框、關鍵點與自訂骨架。")

if ENABLE_REVIEW_GUI:
    print(f"\n🖥 {mode_text}已全部完成，正在開啟圖片審核 GUI...")
    launch_review_gui(review_records)
else:
    input("\n按 Enter 結束...")
