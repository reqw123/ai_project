"""
EDA for Cat Skeleton JSON Dataset
====================================
分析 skeletons 目錄下所有 YOLO-Pose 17 關鍵點 JSON 檔案：
- 缺失幀比例
- 各關鍵點 x/y 分布
- 信心分數分布
- 骨架長度分布
- 每檔案統計摘要

輸出：統計摘要、分布圖（可選）
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ==================== Config ====================

SKELETON_DIR = r"C:\cat_pose\gcn_pose\skeletons"
EDA_IMG_DIR = r"C:\cat_pose\gcn_pose\EDA_IMG"
NUM_KEYPOINTS = 17

# 建立輸出資料夾
os.makedirs(EDA_IMG_DIR, exist_ok=True)

# ==================== Helper ====================
def load_all_jsons(folder):
    files = list(Path(folder).glob("*.json"))
    all_data = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as jf:
            data = json.load(jf)
            all_data.append((f.name, data))
    return all_data

# ==================== EDA ====================
def eda_skeletons():
    all_jsons = load_all_jsons(SKELETON_DIR)
    print(f"共讀取 {len(all_jsons)} 個 JSON 檔案")

    # 統計容器
    total_frames = 0
    total_detected = 0
    kpt_x = [[] for _ in range(NUM_KEYPOINTS)]
    kpt_y = [[] for _ in range(NUM_KEYPOINTS)]
    kpt_conf = [[] for _ in range(NUM_KEYPOINTS)]
    skeleton_lengths = []
    file_stats = []

    for fname, data in all_jsons:
        frames = data['frames']
        n = len(frames)
        detected = sum(1 for f in frames if f['detected'])
        total_frames += n
        total_detected += detected
        # Per-file統計
        file_stats.append({
            'file': fname,
            'frames': n,
            'detected': detected,
            'detect_rate': detected / n if n else 0
        })
        # 關鍵點統計
        for f in frames:
            if f['detected'] and len(f['keypoints']) == NUM_KEYPOINTS:
                for i, kpt in enumerate(f['keypoints']):
                    kpt_x[i].append(kpt['x'])
                    kpt_y[i].append(kpt['y'])
                    kpt_conf[i].append(kpt['conf'])
                # 骨架長度（頸到下體）
                neck = f['keypoints'][3]
                lower = f['keypoints'][5]
                length = np.linalg.norm([neck['x']-lower['x'], neck['y']-lower['y']])
                skeleton_lengths.append(length)

    print(f"總幀數: {total_frames}")
    print(f"有偵測骨架幀數: {total_detected} ({total_detected/total_frames*100:.1f}%)")

    # --- 偵測率長條圖 ---
    plt.figure(figsize=(12,5))
    files = [stat['file'] for stat in file_stats]
    rates = [stat['detect_rate']*100 for stat in file_stats]
    plt.bar(range(len(files)), rates, color='teal', alpha=0.7)
    plt.xticks([])
    plt.ylabel('Detection Rate (%)')
    plt.title('Detection Rate per File')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_IMG_DIR, 'detection_rate_bar.png'))
    plt.close()

    # 儲存偵測率 csv
    import pandas as pd
    df = pd.DataFrame(file_stats)
    df.to_csv(os.path.join(EDA_IMG_DIR, 'detection_rate_per_file.csv'), index=False)

    # --- 偵測率分布箱型圖 ---
    plt.figure(figsize=(6,4))
    plt.boxplot(rates, vert=False, patch_artist=True, boxprops=dict(facecolor='orange'))
    plt.xlabel('Detection Rate (%)')
    plt.title('Detection Rate Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_IMG_DIR, 'detection_rate_box.png'))
    plt.close()

    # 關鍵點分布
    print("\n各關鍵點 x/y/conf 分布摘要:")
    # 儲存統計摘要 csv
    kpt_summary = []
    for i in range(NUM_KEYPOINTS):
        if kpt_x[i]:
            kpt_summary.append({
                'kpt': i,
                'x_min': np.min(kpt_x[i]),
                'x_max': np.max(kpt_x[i]),
                'y_min': np.min(kpt_y[i]),
                'y_max': np.max(kpt_y[i]),
                'conf_mean': np.mean(kpt_conf[i]),
                'conf_std': np.std(kpt_conf[i])
            })
            print(f"KPT {i:2d}: x=[{np.min(kpt_x[i]):.1f},{np.max(kpt_x[i]):.1f}] "
                  f"y=[{np.min(kpt_y[i]):.1f},{np.max(kpt_y[i]):.1f}] "
                  f"conf: mean={np.mean(kpt_conf[i]):.3f} std={np.std(kpt_conf[i]):.3f}")
        else:
            print(f"KPT {i:2d}: 無資料")
    if kpt_summary:
        pd.DataFrame(kpt_summary).to_csv(os.path.join(EDA_IMG_DIR, 'kpt_summary.csv'), index=False)

    # --- 各關鍵點 x/y 箱型圖 ---
    plt.figure(figsize=(14,6))
    plt.boxplot([kpt_x[i] for i in range(NUM_KEYPOINTS)], labels=[str(i) for i in range(NUM_KEYPOINTS)])
    plt.title('Keypoint X Distribution (Boxplot)')
    plt.xlabel('Keypoint Index')
    plt.ylabel('X (pixels)')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_IMG_DIR, 'kpt_x_box.png'))
    plt.close()

    plt.figure(figsize=(14,6))
    plt.boxplot([kpt_y[i] for i in range(NUM_KEYPOINTS)], labels=[str(i) for i in range(NUM_KEYPOINTS)])
    plt.title('Keypoint Y Distribution (Boxplot)')
    plt.xlabel('Keypoint Index')
    plt.ylabel('Y (pixels)')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_IMG_DIR, 'kpt_y_box.png'))
    plt.close()

    # 骨架長度分布
    if skeleton_lengths:
        print(f"\n骨架長度: mean={np.mean(skeleton_lengths):.1f} std={np.std(skeleton_lengths):.1f} min={np.min(skeleton_lengths):.1f} max={np.max(skeleton_lengths):.1f}")
        # 直方圖
        plt.hist(skeleton_lengths, bins=30, color='skyblue')
        plt.title('Skeleton Length Distribution (neck-lower body)')
        plt.xlabel('Length (pixels)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_IMG_DIR, 'skeleton_length_hist.png'))
        plt.close()
        # 箱型圖
        plt.figure(figsize=(6,4))
        plt.boxplot(skeleton_lengths, vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        plt.xlabel('Skeleton Length (pixels)')
        plt.title('Skeleton Length Boxplot')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_IMG_DIR, 'skeleton_length_box.png'))
        plt.close()

    # 可選: 各關鍵點信心分數分布圖
    for i in range(NUM_KEYPOINTS):
        if kpt_conf[i]:
            plt.hist(kpt_conf[i], bins=30, alpha=0.7, color='purple')
            plt.title(f'KPT {i} Confidence Distribution')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(EDA_IMG_DIR, f'kpt{i}_conf_hist.png'))
            plt.close()

    # --- 關鍵點分布熱力圖（所有點彙整）---
    all_x = np.concatenate([np.array(kpt_x[i]) for i in range(NUM_KEYPOINTS) if kpt_x[i]])
    all_y = np.concatenate([np.array(kpt_y[i]) for i in range(NUM_KEYPOINTS) if kpt_y[i]])
    if len(all_x) > 0 and len(all_y) > 0:
        plt.figure(figsize=(8,6))
        plt.hexbin(all_x, all_y, gridsize=80, cmap='hot', bins='log')
        plt.colorbar(label='log(count)')
        plt.title('All Keypoints Position Heatmap')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_IMG_DIR, 'all_kpt_heatmap.png'))
        plt.close()

if __name__ == "__main__":
    eda_skeletons()
