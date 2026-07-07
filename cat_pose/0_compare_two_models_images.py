from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import shutil

# =====================================================
# ⭐ 模型設定
# =====================================================
MODELS = {
    "640.1": {
        "path": r"C:\ai_project\cat_pose\v11s_121.pt",
        "imgsz": 640,
        # label will be set to the .pt filename below
    },
    "640.2": {
        "path": r"C:\ai_project\cat_pose\v11s_121.pt",
        "imgsz": 640,
        # label will be set to the .pt filename below
    }
}

# Automatically set label to the .pt filename
for cfg in MODELS.values():
    cfg["label"] = Path(cfg["path"]).name

INPUT_DIR = r"C:\Users\homec\OneDrive\圖片\Screenshots"

# 前處理：YOLO Pose 訓練時用 imgsz=640，這裡先把 INPUT_DIR 底下的圖片等比
# 壓縮到最長邊 640px，直接覆寫回原檔案（不留備份，原始解析度會永久消失）；
# 後續推論、存檔（compare_output / offset_dataset）全部沿用覆寫後的 640 版本。
RESIZE_MAX_SIDE = 640

CONF_THRES = 0.9
KP_CONF_THRES = 0.8       # 關鍵點信心門檻：低於此值的點不列入偏移比較
DIFF_THRES_PERCENT = 2.0  # 偏移閾值：圖片對角線的百分比
TOTAL_KPTS = 17

# 是否在輸出圖上畫骨架連線（骨架定義與配色參考主專案
# paper/cat_monitoring_system/utils/constants.py 的 ALL_SKELETON /
# EAR_DISTANCE_EDGE_COLORS，17 個關鍵點索引順序與主專案一致）
DRAW_SKELETON_LINES = True

# 17-kpt 骨架連線（索引對應 KEYPOINT_NAMES：nose/ear_tip*2/chest/mid_back/hip/
# 前肢*2/後肢*2/tail_base/mid/tip），與主專案 ALL_SKELETON 相同
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 2),                    # 頭部
    (0, 3), (3, 4), (4, 5),                    # 身體
    (3, 6), (6, 7), (3, 8), (8, 9),             # 前肢
    (5, 10), (10, 11), (5, 12), (12, 13),       # 後肢
    (5, 14), (14, 15), (15, 16),                # 髖→尾根→尾中→尾尖
]
SKELETON_EDGE_COLORS = [
    (255, 120, 60), (255, 120, 60), (255, 120, 60),
    (220, 220, 60), (200, 220, 60), (160, 220, 60),
    (102, 85, 255), (102, 85, 255), (255, 68, 204), (255, 68, 204),
    (255, 170, 34), (255, 170, 34), (0, 153, 255), (0, 153, 255),
    (80, 200, 160), (60, 170, 130), (40, 140, 100),
]

# True  = 推論前先將圖片等比縮放至最長邊 640px（模擬訓練解析度）
# False = 直接傳原圖路徑，YOLO 內部自動 letterbox 縮放（預設行為）
RESIZE_INPUT_TO_640 = True

# 輸出並排圖最大寬度（像素），超過則等比縮小存檔；設 None 不限制
MAX_OUTPUT_WIDTH = 3840

# =====================================================
# ⭐ cat_Compare 主資料夾
# =====================================================
BASE_DIR = Path(r"C:\cat_pose\cat_Compare")
COMPARE_DIR = BASE_DIR / "compare_output"
OFFSET_DIR = BASE_DIR / "offset_dataset"

COMPARE_DIR.mkdir(parents=True, exist_ok=True)
OFFSET_DIR.mkdir(parents=True, exist_ok=True)

# 建立 offset_x/original + offset_x/yolo
for i in range(TOTAL_KPTS + 1):
    (OFFSET_DIR / f"offset_{i}" / "original").mkdir(parents=True, exist_ok=True)
    (OFFSET_DIR / f"offset_{i}" / "yolo").mkdir(parents=True, exist_ok=True)

# =====================================================
# 清空輸出資料夾
# =====================================================
print("🧹 清空輸出資料夾...")

# 清空 compare_output
for file in COMPARE_DIR.glob("*.jpg"):
    file.unlink()

# 清空 offset_dataset 中所有子資料夾的檔案
for i in range(TOTAL_KPTS + 1):
    original_dir = OFFSET_DIR / f"offset_{i}" / "original"
    yolo_dir = OFFSET_DIR / f"offset_{i}" / "yolo"
    
    for file in original_dir.glob("*"):
        if file.is_file():
            file.unlink()
    
    for file in yolo_dir.glob("*"):
        if file.is_file():
            file.unlink()

print("✅ 資料夾已清空")

# =====================================================
# 載入模型
# =====================================================
print("🔄 載入模型中...")
models = {name: YOLO(cfg["path"]) for name, cfg in MODELS.items()}
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
# 前處理：壓縮圖片至 640 + 讀取圖片
# =====================================================
IMAGE_EXT = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

src_paths = []
for ext in IMAGE_EXT:
    src_paths.extend(Path(INPUT_DIR).glob(ext))

if not src_paths:
    print("⚠ 找不到任何圖片")
    exit()

print(f"🔧 前處理：壓縮 {len(src_paths)} 張圖片至最長邊 {RESIZE_MAX_SIDE}px（直接覆寫原檔，不留備份）...")

for src_path in src_paths:
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"  ⚠ 無法讀取，略過: {src_path.name}")
        continue
    cv2.imwrite(str(src_path), resize_to_fit(img, max_side=RESIZE_MAX_SIDE))

print(f"✅ 前處理完成，已覆寫: {INPUT_DIR}")

image_paths = list(src_paths)

# =====================================================
# 主流程
# =====================================================
model_names = list(MODELS.keys())
ref_name, cmp_name = model_names

offset_hist = {i: 0 for i in range(TOTAL_KPTS + 1)}
total_images = 0
diff_threshold_px_list = []

for idx, img_path in enumerate(image_paths, start=1):
    print(f"[{idx}/{len(image_paths)}] {img_path.name}")

    # ---------- 載入原圖 ----------
    original_img = cv2.imread(str(img_path))

    # ---------- 決定推論輸入（依旗標決定是否預先 resize） ----------
    if RESIZE_INPUT_TO_640:
        infer_img = resize_to_fit(original_img, max_side=640)
    else:
        infer_img = str(img_path)  # 傳路徑，YOLO 內部自動 letterbox

    # 對角線以實際推論圖尺寸為準，確保偏移百分比一致
    ref_shape = infer_img.shape if RESIZE_INPUT_TO_640 else original_img.shape
    diagonal = calculate_diagonal(ref_shape)
    diff_threshold_px = diagonal * (DIFF_THRES_PERCENT / 100)
    diff_threshold_px_list.append(diff_threshold_px)

    # ---------- 推論 ----------
    results = {}
    for name, cfg in MODELS.items():
        results[name] = models[name].predict(
            infer_img,
            imgsz=cfg["imgsz"],
            conf=CONF_THRES,
            verbose=False
        )[0]

    # ---------- 無 keypoints ----------
    if (len(results[ref_name].keypoints) == 0 or
        len(results[cmp_name].keypoints) == 0):
        offset_count = TOTAL_KPTS
        print(f"  ⚠ 無法檢測到關鍵點")
    else:
        k_ref = results[ref_name].keypoints.xy[0].cpu().numpy()
        k_cmp = results[cmp_name].keypoints.xy[0].cpu().numpy()
        conf_ref = results[ref_name].keypoints.conf[0].cpu().numpy()
        conf_cmp = results[cmp_name].keypoints.conf[0].cpu().numpy()
        diffs_percent, diffs_px, max_idx = compute_keypoint_diff(k_ref, k_cmp, diagonal, conf_ref, conf_cmp)
        offset_count = int((diffs_percent > DIFF_THRES_PERCENT).sum())
        
        if offset_count > 0:
            offset_points = np.where(diffs_percent > DIFF_THRES_PERCENT)[0]
            print(f"  🔴 偏移 {offset_count} 點: {list(offset_points)}, 最大 #{max_idx} ({diffs_percent[max_idx]:.2f}%)")
        else:
            print(f"  ✅ 完全匹配")

    offset_hist[offset_count] += 1
    total_images += 1

    # 複製原圖
    shutil.copy(img_path, OFFSET_DIR / f"offset_{offset_count}" / "original" / img_path.name)

    # 存單張標記圖
    if len(results[ref_name].keypoints) > 0 and len(results[cmp_name].keypoints) > 0:
        yolo_img = results[ref_name].plot(kpt_line=False)  # 關掉 ultralytics 內建骨架線，改用自訂的貓骨架
        _s = get_scale(yolo_img.shape)
        if DRAW_SKELETON_LINES:
            yolo_img = draw_skeleton_lines(yolo_img, k_ref, kpt_conf=conf_ref, scale=_s)
        yolo_img_marked = mark_offset_points_ultra_clear(yolo_img, k_ref, diffs_percent, DIFF_THRES_PERCENT, scale=_s, kpt_conf=conf_ref)
        yolo_img_marked = draw_label(yolo_img_marked, MODELS[ref_name]["label"], scale=_s)

        yolo_out = OFFSET_DIR / f"offset_{offset_count}" / "yolo" / f"{img_path.stem}_yolo.jpg"
        cv2.imwrite(str(yolo_out), yolo_img_marked)

    # 創建並排對比圖
    if len(results[ref_name].keypoints) > 0 and len(results[cmp_name].keypoints) > 0:
        img_ref = results[ref_name].plot(kpt_line=False)  # 關掉 ultralytics 內建骨架線，改用自訂的貓骨架
        img_cmp = results[cmp_name].plot(kpt_line=False)
        
        comparison = create_side_by_side_comparison(
            img_ref, img_cmp, k_ref, k_cmp,
            diffs_percent, diffs_px,
            DIFF_THRES_PERCENT, model_names,
            conf_ref=conf_ref, conf_cmp=conf_cmp
        )
        
        # 超寬時等比縮小存檔（不影響繪製內容）
        if MAX_OUTPUT_WIDTH is not None and comparison.shape[1] > MAX_OUTPUT_WIDTH:
            scale_down = MAX_OUTPUT_WIDTH / comparison.shape[1]
            new_w = MAX_OUTPUT_WIDTH
            new_h = int(comparison.shape[0] * scale_down)
            comparison = cv2.resize(comparison, (new_w, new_h), interpolation=cv2.INTER_AREA)

        out_path = COMPARE_DIR / f"{img_path.stem}_compare.jpg"
        cv2.imwrite(str(out_path), comparison)

# =====================================================
# 統計輸出
# =====================================================
avg_threshold_px = np.mean(diff_threshold_px_list) if diff_threshold_px_list else 0

print("\n" + "=" * 85)
print(f"📊 Keypoint 偏移統計報告")
print("=" * 85)
print(f"偏移閾值：{DIFF_THRES_PERCENT}% of diagonal (平均 {avg_threshold_px:.1f} px)")
print(f"總圖片數：{total_images}")
print(f"✅ 完美匹配：{offset_hist[0]:4d} ({offset_hist[0]/total_images*100:5.1f}%)")
print(f"🔴 有偏移：  {total_images - offset_hist[0]:4d} ({(total_images - offset_hist[0])/total_images*100:5.1f}%)")
print("-" * 85)
for i in range(TOTAL_KPTS + 1):
    if offset_hist[i] > 0:
        bar = "█" * int(offset_hist[i]/total_images * 50)
        print(f"偏移 {i:2d} 點：{offset_hist[i]:4d} 張 ({offset_hist[i]/total_images*100:5.1f}%) {bar}")
print("=" * 85)

print(f"\n📁 輸出目錄：")
print(f"  compare_output: {COMPARE_DIR}")
print(f"  offset_dataset: {OFFSET_DIR}")

print(f"\n💡 視覺標記說明：")
print(f"  🔴 偏移點（超明顯）：")
print(f"     • 5層同心圓（白→黃→橙→紅→紅）")
print(f"     • 最大半徑 40px")
print(f"     • 編號文字帶 8 方向黑色陰影")
print(f"     • 偏移百分比顯示")
print(f"  🟢 正確點（簡潔）：")
print(f"     • 綠色實心圓 + 白色外圈")
print(f"  ✓ 骨架連線（參考主專案配色，DRAW_SKELETON_LINES={DRAW_SKELETON_LINES}）")
print(f"  ✓ 並排對比 - 直觀清晰")

input("\n按 Enter 結束...")