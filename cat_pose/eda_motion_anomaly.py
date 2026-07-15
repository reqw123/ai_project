from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# ==================== 基本設定 ====================
MODEL_PATH = r"C:\ai_project\cat_pose\v11s_128.pt"
VIDEO_PATH = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\1_貓咪姿勢影片分類\模型專用\walk\walk_12.mp4"

IMGSZ = 640
CONF_THRES = 0.5
KP_CONF_THRES = 0.5
TOTAL_KPTS = 17
MAX_FRAMES = 'MAX'   # 可設數值(如600)或'MAX'代表全片
EMA_ALPHA = 0.7    # EMA 平滑係數 (0-1，越大越平滑)

# 追蹤中斷處理參數
MAX_FRAME_GAP = 5          # 最大容許的幀間隔（超過則重置追蹤）
MAX_NORM_DISP = 0.5        # 最大合理位移（相對於 body_scale）
MIN_TRACKING_FRAMES = 3    # 最少連續追蹤幀數（用於穩定 EMA）

# 貓完整性檢測參數
MIN_VISIBLE_KPT_RATIO = 0.75    # 最少可見關鍵點比例（75%）
BORDER_MARGIN = 50              # 邊界邊距（像素）- 關鍵點不應靠近邊緣
MIN_BODY_SCALE_RATIO = 0.15     # 最小身體尺度比例（相對影片寬度）
MAX_BODY_SCALE_RATIO = 0.8      # 最大身體尺度比例（避免過近特寫）

# ==================== 檢查檔案是否存在 ====================
import os
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found: {MODEL_PATH}")
    exit(1)
if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: Video file not found: {VIDEO_PATH}")
    exit(1)

print(f"Model: {MODEL_PATH}")
print(f"Video: {VIDEO_PATH}")

# ==================== 載入模型 ====================
print("Loading model...")
model = YOLO(MODEL_PATH)
model.to("cuda")
print("Model loaded successfully")

# ==================== 開啟影片 ====================
print("Opening video...")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Cannot open video file: {VIDEO_PATH}")
    exit(1)

# 取得影片資訊
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video opened successfully")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps:.2f}")
print(f"  Total frames: {total_frames}")
if isinstance(MAX_FRAMES, str) and MAX_FRAMES.upper() == 'MAX':
    print(f"  Will process up to {total_frames} frames")
else:
    print(f"  Will process up to {min(MAX_FRAMES, total_frames)} frames")
print()

records = []
prev_kpts = None
prev_prev_kpts = None  # 用於計算加速度
prev_frame_id = -1     # 上次成功檢測的幀號
frame_id = 0
detection_count = 0
no_detection_count = 0
low_body_scale_count = 0
tracking_reset_count = 0       # 追蹤重置次數
filtered_outlier_count = 0     # 過濾的異常值數量
continuous_tracking_frames = 0 # 連續追蹤幀數
incomplete_cat_count = 0       # 不完整的貓（部分入鏡）
too_close_count = 0            # 過近（特寫）
too_far_count = 0              # 過遠（太小）

# EMA 追蹤器（每個關鍵點一個）

ema_disp = np.zeros(TOTAL_KPTS, dtype=np.float32)

# ==================== 收集資料 ====================
print("Processing frames...")
# 動態決定 MAX_FRAMES（放在 while 迴圈前）
if isinstance(MAX_FRAMES, str) and MAX_FRAMES.upper() == 'MAX':
    MAX_FRAMES = total_frames
    print(f"MAX_FRAMES set to total_frames: {MAX_FRAMES}")

while cap.isOpened() and frame_id < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    result = model.predict(
        frame,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        half=True,
        verbose=False
    )[0]

    if result.keypoints is None or len(result.keypoints.xy) == 0:
        no_detection_count += 1
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"  Processed {frame_id}/{MAX_FRAMES} frames...")
        continue

    detection_count += 1
    kpts = result.keypoints.xy[0].cpu().numpy()
    confs = result.keypoints.conf[0].cpu().numpy()

    # ==================== 完整性檢測 ====================
    # 1. 檢查可見關鍵點比例
    visible_kpts = np.sum(confs > KP_CONF_THRES)
    visible_ratio = visible_kpts / TOTAL_KPTS
    
    if visible_ratio < MIN_VISIBLE_KPT_RATIO:
        incomplete_cat_count += 1
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"  Processed {frame_id}/{MAX_FRAMES} frames...")
        continue
    
    # 2. 檢查關鍵點是否靠近邊界
    near_border = False
    for i, (x, y) in enumerate(kpts):
        if confs[i] > KP_CONF_THRES:
            if (x < BORDER_MARGIN or x > width - BORDER_MARGIN or 
                y < BORDER_MARGIN or y > height - BORDER_MARGIN):
                near_border = True
                break
    
    if near_border:
        incomplete_cat_count += 1
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"  Processed {frame_id}/{MAX_FRAMES} frames...")
        continue
    
    # 3. 身體尺度檢查（相對於影片寬度）
    body_scale = np.linalg.norm(kpts[3] - kpts[5])
    body_scale_ratio = body_scale / width
    
    if body_scale_ratio < MIN_BODY_SCALE_RATIO:
        too_far_count += 1
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"  Processed {frame_id}/{MAX_FRAMES} frames...")
        continue
    
    if body_scale_ratio > MAX_BODY_SCALE_RATIO:
        too_close_count += 1
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"  Processed {frame_id}/{MAX_FRAMES} frames...")
        continue
    
    # 4. 基本身體尺度檢查
    if body_scale < 1e-3:
        low_body_scale_count += 1
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"  Processed {frame_id}/{MAX_FRAMES} frames...")
        continue

    if prev_kpts is not None:
        # 檢查幀間隔（否追蹤中斷）
        frame_gap = frame_id - prev_frame_id
        
        if frame_gap > MAX_FRAME_GAP:
            # 追蹤中斷，重置狀態
            print(f"  [Frame {frame_id}] Tracking lost for {frame_gap} frames, resetting...")
            prev_kpts = kpts.copy()
            prev_prev_kpts = None
            prev_frame_id = frame_id
            ema_disp = np.zeros(TOTAL_KPTS, dtype=np.float32)
            continuous_tracking_frames = 0
            tracking_reset_count += 1
            frame_id += 1
            if frame_id % 100 == 0:
                print(f"  Processed {frame_id}/{MAX_FRAMES} frames...")
            continue
        
        diffs = kpts - prev_kpts
        dists = np.linalg.norm(diffs, axis=1)
        norm_dists = dists / body_scale
        
        # 異常值過濾：檢查是否有不合理的大位移
        max_norm_disp = norm_dists.max()
        if max_norm_disp > MAX_NORM_DISP:
            print(f"  [Frame {frame_id}] Outlier detected: max_disp={max_norm_disp:.3f} > {MAX_NORM_DISP}, filtering...")
            # 過濾異常值：將超過門檻的位移裁剪到門檻值
            norm_dists = np.clip(norm_dists, 0, MAX_NORM_DISP)
            dists = norm_dists * body_scale
            filtered_outlier_count += 1
        
        # 更新 EMA（只有連續追蹤時才完全信任 EMA）
        if continuous_tracking_frames >= MIN_TRACKING_FRAMES:
            ema_disp = EMA_ALPHA * ema_disp + (1 - EMA_ALPHA) * norm_dists
        else:
            # 初始階段，直接使用當前值
            ema_disp = norm_dists
        
        # 計算加速度（如果有前前幀且追蹤連續）
        if prev_prev_kpts is not None and frame_gap == 1:
            prev_diffs = prev_kpts - prev_prev_kpts
            prev_dists = np.linalg.norm(prev_diffs, axis=1)
            accel = dists - prev_dists  # 位移的變化
        else:
            accel = np.zeros(TOTAL_KPTS)

        for i in range(TOTAL_KPTS):
            if confs[i] > KP_CONF_THRES:
                records.append({
                    "frame": frame_id,
                    "kpt_id": i,
                    "conf": confs[i],
                    "dx": diffs[i, 0],
                    "dy": diffs[i, 1],
                    "disp": dists[i],
                    "norm_disp": norm_dists[i],
                    "ema_disp": ema_disp[i],
                    "accel": accel[i],
                    "body_scale": body_scale,
                    "frame_gap": frame_gap,
                    "tracking_quality": min(continuous_tracking_frames / MIN_TRACKING_FRAMES, 1.0)
                })
        
        continuous_tracking_frames += 1
    else:
        # 第一次檢測到
        continuous_tracking_frames = 0

    prev_prev_kpts = prev_kpts.copy() if prev_kpts is not None else None
    prev_kpts = kpts.copy()
    prev_frame_id = frame_id
    frame_id += 1
    if frame_id % 100 == 0:
        print(f"  Processed {frame_id}/{MAX_FRAMES} frames...")

cap.release()

print(f"\nProcessing complete!")
print(f"  Total frames processed: {frame_id}")
print(f"  Frames with detections: {detection_count}")
print(f"  Frames without detections: {no_detection_count}")
print()
print(f"Quality Filtering:")
print(f"  Incomplete cats (partial view): {incomplete_cat_count}")
print(f"  Too close (close-up): {too_close_count}")
print(f"  Too far (too small): {too_far_count}")
print(f"  Low body scale: {low_body_scale_count}")
print()
print(f"Tracking Statistics:")
print(f"  Tracking resets: {tracking_reset_count}")
print(f"  Filtered outliers: {filtered_outlier_count}")
print(f"  Valid records collected: {len(records)}")
if detection_count > 0:
    valid_rate = len(records) / (detection_count * TOTAL_KPTS) * 100
    print(f"  Data validity rate: {valid_rate:.1f}%")
print()

# ==================== 轉 DataFrame ====================
df = pd.DataFrame(records)
df.to_csv("eda_keypoint_jitter.csv", index=False)

print("Saved: eda_keypoint_jitter.csv")

if df.empty:
    print("Warning: No keypoint data collected. DataFrame is empty.")
    print(f"Total frames processed: {frame_id}")
    print("Possible reasons:")
    print("1. No detections found in the video")
    print("2. All keypoint confidences below threshold")
    print("3. Body scale too small (< 1e-3)")
else:
    print(f"Collected {len(df)} keypoint records from {frame_id} frames")
    print(df.describe())
    print()
    
    # ==================== 關鍵點穩定性分析 ====================
    print("=" * 80)
    print("KEYPOINT STABILITY ANALYSIS")
    print("=" * 80)
    
    kpt_names = [
                "nose", "left_ear_tip", "right_ear_tip",
                "chest", "mid_back", "hip",
                "left_front_elbow", "left_front_paw",
                "right_front_elbow", "right_front_paw",
                "left_hind_knee", "left_hind_paw",
                "right_hind_knee", "right_hind_paw",
                "tail_base", "tail_mid", "tail_tip"
                ]
    
    stability_stats = []
    for kpt_id in range(TOTAL_KPTS):
        kpt_data = df[df["kpt_id"] == kpt_id]
        if len(kpt_data) > 0:
            stability_stats.append({
                "kpt_id": kpt_id,
                "kpt_name": kpt_names[kpt_id],
                "mean_disp": kpt_data["norm_disp"].mean(),
                "std_disp": kpt_data["norm_disp"].std(),
                "max_disp": kpt_data["norm_disp"].max(),
                "mean_conf": kpt_data["conf"].mean(),
                "count": len(kpt_data)
            })
    
    stability_df = pd.DataFrame(stability_stats).sort_values("mean_disp", ascending=False)
    
    print("\nMost Unstable Keypoints (Top 5):")
    for idx, row in stability_df.head(5).iterrows():
        print(f"  {row['kpt_name']:15s} - Mean: {row['mean_disp']:.4f}, "
              f"Std: {row['std_disp']:.4f}, Max: {row['max_disp']:.4f}")
    
    print("\nMost Stable Keypoints (Top 5):")
    for idx, row in stability_df.tail(5).iterrows():
        print(f"  {row['kpt_name']:15s} - Mean: {row['mean_disp']:.4f}, "
              f"Std: {row['std_disp']:.4f}, Max: {row['max_disp']:.4f}")
    
    print("\n" + "=" * 80)
    print()
    
    # ==================== EDA 增強視覺化 ====================
    
    # 1. 時間序列：body_scale 變化
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Body scale 變化
    frame_stats = df.groupby("frame").agg({
        "body_scale": "first",
        "norm_disp": "max"
    }).reset_index()
    
    axes[0].plot(frame_stats["frame"], frame_stats["body_scale"], linewidth=1.5)
    axes[0].set_xlabel("Frame", fontsize=11)
    axes[0].set_ylabel("Body Scale (pixels)", fontsize=11)
    axes[0].set_title("Cat Body Scale Over Time", fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 最大位移變化
    axes[1].plot(frame_stats["frame"], frame_stats["norm_disp"], 
                linewidth=1.5, color='orange')
    axes[1].set_xlabel("Frame", fontsize=11)
    axes[1].set_ylabel("Max Normalized Displacement", fontsize=11)
    axes[1].set_title("Maximum Keypoint Movement Over Time", fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 2. 關鍵點穩定性排名
    fig, ax = plt.subplots(figsize=(12, 6))
    stability_df_sorted = stability_df.sort_values("mean_disp")
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(stability_df_sorted)))
    
    bars = ax.barh(stability_df_sorted["kpt_name"], 
                   stability_df_sorted["mean_disp"], 
                   color=colors)
    ax.set_xlabel("Mean Normalized Displacement", fontsize=11)
    ax.set_title("Keypoint Stability Ranking (Lower = More Stable)", 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    # 3. Confidence vs Displacement（保留原有）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(df["conf"], df["disp"], s=5, alpha=0.3, c='blue')
    axes[0].set_xlabel("Keypoint Confidence", fontsize=11)
    axes[0].set_ylabel("Pixel Displacement", fontsize=11)
    axes[0].set_title("Confidence vs Pixel Jitter", fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(df["conf"], df["norm_disp"], s=5, alpha=0.3, c='green')
    axes[1].set_xlabel("Keypoint Confidence", fontsize=11)
    axes[1].set_ylabel("Normalized Displacement", fontsize=11)
    axes[1].set_title("Confidence vs Normalized Jitter", fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 4. EMA vs Raw Displacement 比較
    plt.figure(figsize=(14, 5))
    sample_kpt = 7  # left_paw
    sample_data = df[df["kpt_id"] == sample_kpt].head(100)
    
    plt.plot(sample_data["frame"], sample_data["norm_disp"], 
            label='Raw Displacement', alpha=0.6, linewidth=1)
    plt.plot(sample_data["frame"], sample_data["ema_disp"], 
            label='EMA Smoothed', linewidth=2)
    plt.xlabel("Frame", fontsize=11)
    plt.ylabel("Normalized Displacement", fontsize=11)
    plt.title(f"Raw vs EMA Smoothed Displacement ({kpt_names[sample_kpt]})", 
             fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 5. 加速度分析
    plt.figure(figsize=(14, 5))
    df.boxplot(column="accel", by="kpt_id", grid=True, figsize=(14, 5))
    plt.suptitle("")
    plt.title("Acceleration Distribution per Keypoint", fontsize=12, fontweight='bold')
    plt.xlabel("Keypoint ID", fontsize=11)
    plt.ylabel("Acceleration (pixel/frame²)", fontsize=11)
    plt.xticks(range(1, TOTAL_KPTS+1), 
              [kpt_names[i] for i in range(TOTAL_KPTS)], 
              rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
