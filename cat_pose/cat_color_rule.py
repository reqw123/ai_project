import os
import cv2
import numpy as np
from pathlib import Path
import csv
from sklearn.cluster import KMeans
from collections import Counter
import warnings

# 忽略 K-Means 收斂警告
warnings.filterwarnings('ignore', category=Warning)

# ========== 參數設定 ==========
IMG_DIR = r"C:\cat_pose\train\images"
OUTPUT_CSV = r"C:\cat_pose\cat_color_advanced_result.csv"
OUTPUT_DIR = r"C:\cat_pose\cat_color_advanced_folders"
DEBUG_DIR = r"C:\cat_pose\debug_visualizations"
IMG_EXTS = (".jpg", ".png")
MODEL_PATH = r"C:\cat_pose\no_aug.pt"
ENABLE_DEBUG = False

# ========== 顏色範圍定義 ==========
COLOR_RANGES_HSV = {
    'orange': [
        {'h_min': 0, 'h_max': 15, 's_min': 60, 's_max': 255, 'v_min': 90, 'v_max': 255},
        {'h_min': 165, 'h_max': 180, 's_min': 60, 's_max': 255, 'v_min': 90, 'v_max': 255},
        {'h_min': 10, 'h_max': 25, 's_min': 40, 's_max': 100, 'v_min': 120, 'v_max': 255},
    ],
    'gray': [
        {'h_min': 0, 'h_max': 180, 's_min': 0, 's_max': 35, 'v_min': 75, 'v_max': 185},
    ],
    'white': [
        {'h_min': 0, 'h_max': 180, 's_min': 0, 's_max': 30, 'v_min': 185, 'v_max': 255},
    ],
    'black': [
        {'h_min': 0, 'h_max': 180, 's_min': 0, 's_max': 255, 'v_min': 0, 'v_max': 75},
    ]
}

# ========== 工具函數 ==========
def normalize_lighting(roi):
    """使用 CLAHE 改善光線不均"""
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def get_cat_roi_yolo(img, model):
    """使用 YOLO 偵測貓並取得 ROI"""
    results = model.predict(img, conf=0.4, verbose=False)
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            bx1, by1, bx2, by2 = r.boxes.xyxy[0].cpu().numpy().astype(int)
            h, w = img.shape[:2]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)
            roi = img[by1:by2, bx1:bx2]
            
            if hasattr(r, 'keypoints') and r.keypoints is not None and r.keypoints.shape[1] > 0:
                kpts_xy = r.keypoints[0].xy[0].cpu().numpy().astype(int)
                polygon = []
                for idx in [3, 4, 5, 6, 8]:
                    if idx < len(kpts_xy):
                        x, y = kpts_xy[idx][0] - bx1, kpts_xy[idx][1] - by1
                        if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]:
                            polygon.append([x, y])
                
                if len(polygon) >= 3:
                    mask = np.zeros(roi.shape[:2], np.uint8)
                    cv2.fillPoly(mask, [np.array(polygon)], 255)
                    roi = cv2.bitwise_and(roi, roi, mask=mask)
                    return roi
            
            mask = np.zeros(roi.shape[:2], np.uint8)
            hh, ww = roi.shape[:2]
            cv2.ellipse(mask, (ww//2, hh//2), (int(ww*0.42), int(hh*0.52)), 0, 0, 360, 255, -1)
            roi = cv2.bitwise_and(roi, roi, mask=mask)
            return roi
    
    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    size = min(w, h) // 2
    x1, y1 = cx - size//2, cy - size//2
    x2, y2 = cx + size//2, cy + size//2
    roi = img[y1:y2, x1:x2]
    
    mask = np.zeros(roi.shape[:2], np.uint8)
    hh, ww = roi.shape[:2]
    cv2.ellipse(mask, (ww//2, hh//2), (int(ww*0.42), int(hh*0.52)), 0, 0, 360, 255, -1)
    return cv2.bitwise_and(roi, roi, mask=mask)

def check_color_in_range(hsv_pixel, color_name):
    """檢查 HSV 像素是否符合特定顏色範圍"""
    h, s, v = hsv_pixel
    ranges = COLOR_RANGES_HSV.get(color_name, [])
    
    for range_def in ranges:
        h_match = range_def['h_min'] <= h <= range_def['h_max']
        s_match = range_def['s_min'] <= s <= range_def['s_max']
        v_match = range_def['v_min'] <= v <= range_def['v_max']
        
        if h_match and s_match and v_match:
            return True
    return False

def sample_multi_scale_regions(roi):
    """多尺度採樣，加入區域權重"""
    h, w = roi.shape[:2]
    regions = []
    
    regions.append(('full', roi, 1.5))
    
    if h >= 30 and w >= 30:
        regions.append(('upper', roi[0:h//3, :], 1.0))
        regions.append(('middle', roi[h//3:2*h//3, :], 1.2))
        regions.append(('lower', roi[2*h//3:h, :], 1.0))
    
    if h >= 45 and w >= 45:
        regions.append(('center', roi[h//3:2*h//3, w//3:2*w//3], 2.0))
        regions.append(('top_center', roi[0:h//3, w//3:2*w//3], 0.8))
        regions.append(('bottom_center', roi[2*h//3:h, w//3:2*w//3], 0.8))
        regions.append(('left_center', roi[h//3:2*h//3, 0:w//3], 0.8))
        regions.append(('right_center', roi[h//3:2*h//3, 2*w//3:w], 0.8))
    
    return [(name, r, weight) for name, r, weight in regions if r.size > 100]

def get_dominant_colors_advanced(roi, k=6):
    """進階 K-Means 聚類（修正版）"""
    pixels = roi.reshape(-1, 3).astype(np.float32)
    
    # 過濾掉太暗的像素
    brightness = np.sum(pixels, axis=1)
    valid_mask = brightness > 20
    valid_pixels = pixels[valid_mask]
    
    if len(valid_pixels) < 30:
        return [], [], []
    
    # 動態調整 k 值
    unique_colors = len(np.unique(valid_pixels.view([('', valid_pixels.dtype)] * valid_pixels.shape[1])))
    k_actual = min(k, len(valid_pixels), max(2, unique_colors // 50))
    
    try:
        kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(valid_pixels)
        
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = counts / counts.sum()
        
        sorted_idx = np.argsort(-percentages)
        sorted_centers = centers[sorted_idx]
        sorted_percentages = percentages[sorted_idx]
        sorted_labels = unique_labels[sorted_idx]
        
        return sorted_centers, sorted_percentages, sorted_labels
    except:
        return [], [], []

def classify_bgr_color(bgr):
    """使用 HSV 和 Lab 雙重驗證判定顏色"""
    bgr_pixel = np.uint8([[bgr]])
    hsv = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
    lab = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)[0][0]
    
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    L, A, B = int(lab[0]), int(lab[1]), int(lab[2])
    
    if check_color_in_range(hsv, 'white'):
        return 'white'
    
    if check_color_in_range(hsv, 'black'):
        return 'black'
    
    if check_color_in_range(hsv, 'gray'):
        return 'gray'
    
    if check_color_in_range(hsv, 'orange'):
        if B > 128:
            return 'orange'
    
    if 5 <= h <= 30 and 35 <= s <= 120 and 30 <= v <= 100:
        return 'other'
    
    return 'other'

def check_color_consistency(centers, percentages):
    """檢測顏色分佈的一致性（修正版）"""
    if len(centers) == 0 or len(percentages) == 0:
        return 0.0
    
    # 主色佔比
    main_color_ratio = percentages[0]
    
    # 計算有效顏色數（佔比 > 5% 的顏色）
    significant_colors = np.sum(percentages > 0.05)
    
    # 計算顏色分散度（標準差）
    color_std = np.std(percentages)
    
    # 一致性分數：
    # - 主色佔比高 → 高一致性
    # - 有效顏色少 → 高一致性
    # - 分散度低 → 高一致性
    consistency = main_color_ratio * (1.0 - (significant_colors - 1) * 0.15) * (1.0 + color_std)
    
    # 限制在 0-1 範圍
    consistency = max(0.0, min(1.0, consistency))
    
    return consistency

def analyze_region_voting(region):
    """分析單一區域並產生顏色投票"""
    region_norm = normalize_lighting(region)
    
    centers, percentages, _ = get_dominant_colors_advanced(region_norm, k=6)
    
    votes = {'white': 0, 'black': 0, 'orange': 0, 'gray': 0, 'other': 0}
    
    if len(centers) == 0:
        return votes, 0.0
    
    for center, pct in zip(centers, percentages):
        if pct < 0.02:
            continue
        
        color_class = classify_bgr_color(center)
        votes[color_class] += pct
    
    consistency = check_color_consistency(centers, percentages)
    
    return votes, consistency

def detect_texture_pattern(roi):
    """檢測紋理變化"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    edge_density = np.mean(gradient_magnitude)
    texture_variance = np.var(gray)
    
    has_stripes = edge_density > 25 and texture_variance > 600
    
    return {
        'edge_density': edge_density,
        'texture_variance': texture_variance,
        'has_stripes': has_stripes
    }

def classify_cat_color_final(roi):
    """最終分類邏輯（修正版）"""
    regions = sample_multi_scale_regions(roi)
    
    total_votes = {'white': 0, 'black': 0, 'orange': 0, 'gray': 0, 'other': 0}
    total_consistency = 0
    region_count = 0
    
    for name, region, weight in regions:
        if region.size == 0:
            continue
        
        votes, consistency = analyze_region_voting(region)
        
        region_weight = (region.size / roi.size) * weight
        region_count += 1
        total_consistency += consistency
        
        for color, score in votes.items():
            total_votes[color] += score * region_weight
    
    total_score = sum(total_votes.values())
    if total_score > 0:
        for key in total_votes:
            total_votes[key] /= total_score
    
    avg_consistency = total_consistency / region_count if region_count > 0 else 0
    
    texture_info = detect_texture_pattern(roi)
    
    white = total_votes['white']
    black = total_votes['black']
    orange = total_votes['orange']
    gray = total_votes['gray']
    other = total_votes['other']
    
    # 黑白雙色判定
    bw_threshold = 0.15 if avg_consistency > 0.5 else 0.18
    if white > bw_threshold and black > bw_threshold:
        if white + black > 0.42:
            return 'black_white', total_votes, texture_info, avg_consistency
    
    # 單色判定
    max_color = max(total_votes, key=total_votes.get)
    max_score = total_votes[max_color]
    
    # 動態閾值
    if avg_consistency > 0.5:
        min_threshold = 0.28
    else:
        min_threshold = 0.32
    
    if max_score < min_threshold:
        return 'other', total_votes, texture_info, avg_consistency
    
    # 檢查次要顏色
    sorted_votes = sorted(total_votes.items(), key=lambda x: -x[1])
    if len(sorted_votes) >= 2 and sorted_votes[0][1] > 0:
        second_color_ratio = sorted_votes[1][1] / sorted_votes[0][1]
        
        if second_color_ratio > 0.50:
            if 'white' in [sorted_votes[0][0], sorted_votes[1][0]]:
                non_white = sorted_votes[0][0] if sorted_votes[0][0] != 'white' else sorted_votes[1][0]
                if non_white == 'black':
                    return 'black_white', total_votes, texture_info, avg_consistency
                return 'other', total_votes, texture_info, avg_consistency
    
    return max_color, total_votes, texture_info, avg_consistency

def save_debug_visualization(img_path, roi, color_label, votes, texture_info, consistency):
    """儲存分析過程的視覺化圖片"""
    if not ENABLE_DEBUG:
        return
    
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    vis_img = cv2.resize(roi, (400, 400))
    info_panel = np.zeros((400, 300, 3), dtype=np.uint8)
    
    y_offset = 25
    cv2.putText(info_panel, f"Class: {color_label}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += 30
    
    cv2.putText(info_panel, f"Consistency: {consistency:.3f}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 25
    
    cv2.putText(info_panel, "Color Scores:", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 20
    
    for color, score in sorted(votes.items(), key=lambda x: -x[1]):
        text = f"  {color}: {score:.3f}"
        color_rgb = (255, 255, 255)
        if color == color_label or (color_label == 'black_white' and color in ['black', 'white']):
            color_rgb = (0, 255, 255)
        
        cv2.putText(info_panel, text, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_rgb, 1)
        
        bar_width = int(score * 200)
        cv2.rectangle(info_panel, (150, y_offset-10), (150+bar_width, y_offset-2), 
                     color_rgb, -1)
        y_offset += 20
    
    y_offset += 10
    cv2.putText(info_panel, "Texture Analysis:", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 20
    
    cv2.putText(info_panel, f"  Edge: {texture_info['edge_density']:.1f}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    y_offset += 18
    
    cv2.putText(info_panel, f"  Variance: {texture_info['texture_variance']:.1f}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    y_offset += 18
    
    stripe_text = "Yes" if texture_info['has_stripes'] else "No"
    stripe_color = (0, 255, 0) if texture_info['has_stripes'] else (100, 100, 100)
    cv2.putText(info_panel, f"  Stripes: {stripe_text}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, stripe_color, 1)
    
    combined = np.hstack([vis_img, info_panel])
    
    filename = os.path.basename(img_path)
    save_path = os.path.join(DEBUG_DIR, f"debug_{filename}")
    cv2.imwrite(save_path, combined)

def main():
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
    
    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if ENABLE_DEBUG:
        if os.path.exists(DEBUG_DIR):
            shutil.rmtree(DEBUG_DIR)
        os.makedirs(DEBUG_DIR, exist_ok=True)
    
    six_classes = ["white", "black", "black_white", "orange", "gray", "other"]
    for cls in six_classes:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)
    
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    
    img_list = [p for ext in IMG_EXTS for p in Path(IMG_DIR).glob(f"*{ext}")]
    
    results = []
    color_count = {}
    
    print(f"🚀 開始處理 {len(img_list)} 張圖片...")
    print(f"除錯模式: {'開啟' if ENABLE_DEBUG else '關閉'}")
    print(f"改進功能: 修正一致性計算、動態 K 值、警告處理")
    print()
    
    for idx, img_path in enumerate(img_list):
        if (idx + 1) % 50 == 0:
            print(f"進度: {idx + 1}/{len(img_list)} ({(idx+1)/len(img_list)*100:.1f}%)")
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        roi = get_cat_roi_yolo(img, model)
        if roi is None or roi.size == 0:
            continue
        
        color_label, votes, texture_info, consistency = classify_cat_color_final(roi)
        
        if color_label not in six_classes:
            color_label = 'other'
        
        results.append({
            'filename': os.path.basename(img_path),
            'color_label': color_label,
            'white_score': round(votes['white'], 3),
            'black_score': round(votes['black'], 3),
            'orange_score': round(votes['orange'], 3),
            'gray_score': round(votes['gray'], 3),
            'other_score': round(votes['other'], 3),
            'consistency': round(consistency, 3),
            'edge_density': round(texture_info['edge_density'], 2),
            'texture_var': round(texture_info['texture_variance'], 2),
            'has_stripes': texture_info['has_stripes']
        })
        
        color_count[color_label] = color_count.get(color_label, 0) + 1
        
        color_folder = os.path.join(OUTPUT_DIR, color_label)
        shutil.copy(str(img_path), os.path.join(color_folder, os.path.basename(img_path)))
        
        save_debug_visualization(img_path, roi, color_label, votes, texture_info, consistency)
    
    if results:
        with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        
        print(f"\n✅ 分類完成！")
        print(f"📄 結果 CSV: {OUTPUT_CSV}")
        print(f"📁 分類資料夾: {OUTPUT_DIR}")
        if ENABLE_DEBUG:
            print(f"🔍 除錯圖片: {DEBUG_DIR}")
        
        print("\n📊 各類別統計：")
        print("-" * 50)
        for cls in six_classes:
            count = color_count.get(cls, 0)
            pct = count / len(results) * 100 if results else 0
            bar = "█" * int(pct / 2)
            print(f"{cls:15s}: {count:4d} ({pct:5.1f}%) {bar}")
        print("-" * 50)
        print(f"總計: {len(results)} 張圖片")
        
        avg_consistency = np.mean([r['consistency'] for r in results])
        print(f"\n平均顏色一致性: {avg_consistency:.3f}")
        
        # 顯示一致性分佈
        consistencies = [r['consistency'] for r in results]
        print(f"一致性範圍: {min(consistencies):.3f} - {max(consistencies):.3f}")
        print(f"一致性中位數: {np.median(consistencies):.3f}")
        
    else:
        print("❌ 沒有可分類的圖片。")

if __name__ == "__main__":
    main()