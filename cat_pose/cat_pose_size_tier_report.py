import os
import shutil
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==================== 顏色定義 ====================
GREEN = (0, 255, 0)
RED   = (0, 0, 255)
BLUE  = (255, 0, 0)

COLOR_HEAD = (255, 255, 0)
COLOR_BODY = (0, 255, 0)
COLOR_TAIL = (255, 0, 255)
COLOR_KPT  = (0, 0, 255)

COLOR_LEFT_FRONT  = (255, 0, 255)
COLOR_RIGHT_FRONT = (0, 255, 255)
COLOR_LEFT_HIND   = (255, 165, 0)
COLOR_RIGHT_HIND  = (0, 255, 0)

# ==================== 骨架連結 ====================
HEAD_LINKS  = [(0,1),(0,2),(1,2)]
BODY_LINKS  = [(0,3),(3,4),(4,5)]
TAIL_LINKS  = [(5,14),(14,15),(15,16)]

LEFT_FRONT_LINKS  = [(3,6),(6,7)]    # 左前肢
RIGHT_FRONT_LINKS = [(3,8),(8,9)]    # 右前肢
LEFT_HIND_LINKS   = [(5,10),(10,11)] # 左後肢
RIGHT_HIND_LINKS  = [(5,12),(12,13)] # 右後肢

KPT_CONF_THRESH = 0.0  # 0.0 = 不過濾，只跳過絕對無效點 (0,0)


def draw_skeleton(vis, kpts_xy, kpts_conf, box_thickness):
    """畫出骨架連線與關鍵點，只過濾絕對無效點"""
    def is_valid(idx):
        if idx >= len(kpts_xy):
            return False
        x, y = float(kpts_xy[idx][0]), float(kpts_xy[idx][1])
        if x == 0.0 and y == 0.0:  # 僅跳過完全無效的原點
            return False
        return True

    def draw_links(links, color):
        for i, j in links:
            if is_valid(i) and is_valid(j):
                pt1 = (int(kpts_xy[i][0]), int(kpts_xy[i][1]))
                pt2 = (int(kpts_xy[j][0]), int(kpts_xy[j][1]))
                cv2.line(vis, pt1, pt2, color, box_thickness)

    draw_links(HEAD_LINKS, COLOR_HEAD)
    draw_links(BODY_LINKS, COLOR_BODY)
    draw_links(TAIL_LINKS, COLOR_TAIL)
    draw_links(LEFT_FRONT_LINKS,  COLOR_LEFT_FRONT)
    draw_links(RIGHT_FRONT_LINKS, COLOR_RIGHT_FRONT)
    draw_links(LEFT_HIND_LINKS,   COLOR_LEFT_HIND)
    draw_links(RIGHT_HIND_LINKS,  COLOR_RIGHT_HIND)

    for idx in range(len(kpts_xy)):
        if is_valid(idx):
            x, y = int(kpts_xy[idx][0]), int(kpts_xy[idx][1])
            cv2.circle(vis, (x, y), max(3, box_thickness + 1), COLOR_KPT, -1)

# -------------------------
# 你要改的部分
# -------------------------
MODEL_PATH = r"C:\ai_project\cat_pose\v11s_114.pt"  # 換成你的模型
FOLDER = r"C:\Users\homec\OneDrive\圖片\Screenshots\screen_cat"            # 換成你的資料夾
OUTPUT_DIR = r"C:\Users\homec\OneDrive\圖片\Screenshots\screen_cat\class"  # 預設輸出資料夾，請自行修改
# 視覺化與判定設定
MIN_RATIO = 0.50  # 門檻：低於此值視為 small，高於此值分層 5 級 (50% ~ 100%)
DIR_NO = os.path.join(OUTPUT_DIR, "no")
DIR_SMALL = os.path.join(OUTPUT_DIR, "small")
# 5個等級文件夾，每級占 (1 - MIN_RATIO) / 5 = 8% 範圍
TIER_DIRS = [os.path.join(OUTPUT_DIR, f"tier_{i+1}") for i in range(5)]
CHART_PATH = os.path.join(OUTPUT_DIR, "summary.png")
# -------------------------

model = YOLO(MODEL_PATH)


def ensure_output_dir():
    # 清除舊的輸出資料夾
    if os.path.exists(DIR_NO):
        shutil.rmtree(DIR_NO)
    if os.path.exists(DIR_SMALL):
        shutil.rmtree(DIR_SMALL)
    for tier_dir in TIER_DIRS:
        if os.path.exists(tier_dir):
            shutil.rmtree(tier_dir)
    
    # 重新建立
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DIR_NO, exist_ok=True)
    os.makedirs(DIR_SMALL, exist_ok=True)
    for tier_dir in TIER_DIRS:
        os.makedirs(tier_dir, exist_ok=True)


def get_tier_by_ratio(ratio):
    """根據比例分配等級 (0-4) 或 small (-1)"""
    if ratio < MIN_RATIO:
        return -1  # small
    # 將 [MIN_RATIO, 1.0) 分成 5 等份
    normalized = (ratio - MIN_RATIO) / (1.0 - MIN_RATIO)
    tier = int(normalized * 5)
    return min(tier, 4)  # 上限為 tier 4


def annotate_and_save(frame, file, status_text, color, ratio, dest_dir):
    """在圖上標註並輸出視覺化結果"""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # 動態字體縮放（以640px為基準）
    base_width = 640
    font_scale = max(1.0, (w / base_width) * 1.2)
    thickness = max(2, int((w / base_width) * 3))
    box_height = int(50 * (w / base_width))

    text = f"{status_text} | ratio: {ratio*100:.1f}%"
    # 動態計算文字寬度，避免黑底太窄或太寬
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(vis, (5, 5), (text_w + 20, box_height), (0, 0, 0), -1)
    cv2.putText(vis, text, (10, int(box_height * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    out_path = os.path.join(dest_dir, file)
    cv2.imwrite(out_path, vis)


def analyze_folder(folder):
    ensure_output_dir()
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    summary = []  # (file, ratio, tier)
    tier_counts = [0] * 5  # tier_1 ~ tier_5 的計數
    small_count = 0
    miss_count = 0

    for file in files:
        path = os.path.join(folder, file)
        img = cv2.imread(path)

        if img is None:
            print(f"{file}: ❌ 無法讀取圖片")
            continue

        h, w = img.shape[:2]
        img_area = w * h

        result = model(img, verbose=False)[0]

        if result.boxes is None or len(result.boxes) == 0:
            print(f"{file}: ⚠ 未偵測到貓")
            miss_count += 1
            annotate_and_save(img, file, "NO CAT", (0, 0, 255), 0.0, DIR_NO)
            summary.append((file, 0.0, "no_cat"))
            continue

        # 取第一隻貓（若有多隻可改成迴圈）
        box = result.boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = box.astype(int)
        # 保護：避免 bbox 反轉導致負面積
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)

        ratio = max(0.0, min(1.0, bbox_area / img_area))
        tier = get_tier_by_ratio(ratio)
        tier = max(-1, min(tier, 4))  # 安全 clamp


        # 設定顏色：red for small, 漸層綠色 for tier
        color_map = [(0, 0, 255), (0, 100, 200), (0, 150, 150), (0, 200, 100), (100, 200, 0), (0, 200, 0)]
        color = color_map[tier + 1]  # small=-1 映射到 index 0

        if tier == -1:
            status = "SMALL"
            small_count += 1
            dest_dir = DIR_SMALL
        else:
            status = f"TIER_{tier+1}"
            tier_counts[tier] += 1
            dest_dir = TIER_DIRS[tier]

        vis = img.copy()

        # 動態字體縮放（以640px為基準）
        base_width = 640
        font_scale = max(1.0, (w / base_width) * 1.2)
        thickness = max(2, int((w / base_width) * 3))
        box_thickness = max(2, int((w / base_width) * 3))


        # 畫出貓的骨架（如果有 keypoints）
        if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints) > 0:
            kpts_xy   = result.keypoints.xy[0].cpu().numpy()    # shape: (num_kpts, 2)
            kpts_conf = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else None
            draw_skeleton(vis, kpts_xy, kpts_conf, box_thickness)

        # 畫出偵測框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, box_thickness)
        label = f"cat {ratio*100:.1f}%"
        cv2.putText(vis, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

        annotate_and_save(vis, file, status, color, ratio, dest_dir)

        print(f"{file}: {w}x{h}, 貓佔比 = {ratio:.3f} -> {status}")
        summary.append((file, ratio, status.lower()))

    # 整體摘要
    total = len(files)
    passed_count = sum(tier_counts)  # 通過門檻的圖片數量
    print("\n===== Summary =====")
    print(f"Total images: {total}")
    print(f"No cat:  {miss_count}")
    print(f"Small:   {small_count}")
    print(f"Passed (tier_1~5): {passed_count}")
    for i, c in enumerate(tier_counts):
        low  = (MIN_RATIO + i / 5 * (1 - MIN_RATIO)) * 100
        high = (MIN_RATIO + (i + 1) / 5 * (1 - MIN_RATIO)) * 100
        print(f"  Tier {i+1} ({low:.0f}% - {high:.0f}%): {c}")

    # 輸出所有結果資料夾的絕對路徑
    print("\n===== Output Folders =====")
    print(f"All results saved in: {os.path.abspath(OUTPUT_DIR)}")
    print(f"  No cat:   {os.path.abspath(DIR_NO)}")
    print(f"  Small:    {os.path.abspath(DIR_SMALL)}")
    for i, d in enumerate(TIER_DIRS):
        print(f"  Tier {i+1}:  {os.path.abspath(d)}")
    print(f"  Chart:    {os.path.abspath(CHART_PATH)}")

    # 畫長條圖
    all_labels = ["no"] + [f"tier_{i+1}" for i in range(5)] + ["small"]
    all_counts = [miss_count] + tier_counts + [small_count]
    colors_chart = ["red"] + ["#FF6B35", "#F7931E", "#FDB833", "#90EE90", "#228B22"] + ["orange"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(all_labels, all_counts, color=colors_chart)
    for bar, c in zip(bars, all_counts):
        if c > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(c), ha="center", va="bottom")
    plt.title("Cat size distribution")
    plt.ylabel("count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHART_PATH)
    plt.close()
    print(f"Chart saved: {CHART_PATH}")

    # 列出最小的前5張
    summary_sorted = sorted([s for s in summary if s[2] != "no_cat"], key=lambda x: x[1])
    top_n = min(5, len(summary_sorted))
    if top_n > 0:
        print("\nSmallest cats:")
        for i in range(top_n):
            f, r, st = summary_sorted[i]
            print(f"  {f}: {r*100:.2f}% ({st})")

# 執行分析
analyze_folder(FOLDER)
