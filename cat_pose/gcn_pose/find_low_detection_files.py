"""
Find Low-Detection Skeleton JSON Files
=====================================
列出 skeletons 目錄下偵測率低於指定閾值的 JSON 檔案，方便人工檢查。
"""

import os
import json
from pathlib import Path



SKELETON_DIR = r"C:\cat_pose\gcn_pose\skeletons"
LABELS_FILE = r"C:\cat_pose\gcn_pose\labels.json"
THRESHOLD = 0.8  # 偵測率低於 80% 視為容易沒偵測到骨架
REMOVE_FILES = False # 設為 True 時自動刪除低於門檻的 JSON

# 讀取 labels
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, 'r', encoding='utf-8') as lf:
        labels_dict = json.load(lf)
else:
    labels_dict = {}

results = []

class_stats = {}  # {label: [detect_rate, ...]}
class_low_files = {}  # {label: [file, ...]}
class_total_frames = {}  # {label: 幀數總和}

for f in Path(SKELETON_DIR).glob("*.json"):
    video_id = f.stem
    label = labels_dict.get(video_id, 'unknown')
    with open(f, 'r', encoding='utf-8') as jf:
        data = json.load(jf)
        frames = data['frames']
        n = len(frames)
        detected = sum(1 for frame in frames if frame['detected'])
        detect_rate = detected / n if n else 0
        # 統計各類別偵測率
        class_stats.setdefault(label, []).append(detect_rate)
        class_total_frames[label] = class_total_frames.get(label, 0) + n
        if detect_rate < THRESHOLD:
            results.append({
                'file': f.name,
                'frames': n,
                'detected': detected,
                'detect_rate': detect_rate,
                'path': str(f),
                'label': label
            })
            class_low_files.setdefault(label, []).append(f.name)

if REMOVE_FILES and results:
    print(f"\n自動刪除偵測率低於 {THRESHOLD*100:.0f}% 的檔案...")
    for r in results:
        try:
            os.remove(r['path'])
            print(f"已刪除: {r['file']}")
        except Exception as e:
            print(f"刪除失敗: {r['file']} ({e})")


print(f"\n=== 各類別偵測率統計 ===")
for label, rates in class_stats.items():
    avg = sum(rates)/len(rates) if rates else 0
    below = sum(1 for r in rates if r < THRESHOLD)
    total_frames = class_total_frames.get(label, 0)
    print(f"類別: {str(label):10s}  檔案數: {len(rates):3d}  總幀數: {total_frames:6d}  平均偵測率: {avg*100:.1f}%  低於門檻: {below}")

print(f"\n=== 各類別低於 {THRESHOLD*100:.0f}% 偵測率的檔案 ===")
for label, files in class_low_files.items():
    print(f"[{label}] 共 {len(files)} 個:")
    for fname in files:
        print(f"  {fname}")
if not results:
    print("全部檔案偵測率都高於閾值。")
