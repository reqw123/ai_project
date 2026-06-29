"""
統計各行為類別可切出多少筆 ST-GCN 訓練序列，並計算有效資料總秒數。

有效秒數公式：
    effective_seconds = labeled_frame_count / SOURCE_FPS
    其中 labeled_frame_count = 該類別中 label != 'unannotated' 的幀總數
    SOURCE_FPS = 骨架提取時使用的 TARGET_FPS（預設 30fps）

注意：這裡計算的是「唯一標注幀」，不受滑動視窗重疊影響；
      sequence count 因視窗重疊會比秒數換算的理論值高。
"""

import json
from collections import Counter
from pathlib import Path

# ==================== 可調整變數 ====================

# 骨架 JSON 資料夾路徑
SKELETONS_ROOT = Path(r"C:\ai_project\paper\skeletons")

# ST-GCN 時間窗長度（幀數），需與訓練設定一致
SEQUENCE_LENGTH = 16

# 滑動視窗步長（幀數），需與 stgcn_config.yaml 的 WINDOW_STRIDE 一致
WINDOW_STRIDE = 8

# 關節點數，用來驗證幀格式（不符時以零填充）
NUM_JOINTS = 17

# True  = 嚴格模式：視窗內不得有 unannotated 幀
# False = 寬鬆模式：只要有 ≥1 幀有標籤即計入，以多數標籤作為視窗類別
STRICT_WINDOW_FILTER = True

# 視窗內允許 bbox 缺失（YOLO 未偵測到貓）的最大幀數；超過此數則過濾該視窗
# 設為 0 代表完全不允許缺失；設為 SEQUENCE_LENGTH 代表不過濾
MAX_NO_DETECT_FRAMES = 2

# 骨架提取時的 TARGET_FPS（用於換算秒數）
# 公式：effective_seconds = labeled_frame_count / SOURCE_FPS
SOURCE_FPS = 30

# ==================== 統計邏輯 ====================

counts: Counter = Counter()            # 各類別可切出的 sequence 數
frame_counts: Counter = Counter()      # 各類別有效標注幀總數（唯一，不含重疊）
raw_video_secs: Counter = Counter()    # 各類別原始影片總秒數（含未標注段）
video_count: Counter = Counter()       # 各類別影片數量

rejected_unann_count: Counter = Counter()      # 因含 unannotated 被過濾，按主類別計
rejected_no_detect_count: Counter = Counter() # 因 bbox 缺失過多被過濾，按主類別計
rejected_no_detect_detail: list[dict] = []    # 詳細清單（可反推影片）

for p in sorted(SKELETONS_ROOT.glob("*.json")):
    data = json.loads(p.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not frames or "label" not in frames[0]:
        continue

    # 讀取實際 FPS（JSON 有記錄時優先使用，確保秒數精確）
    actual_fps = data.get("video_metadata", {}).get("actual_fps", SOURCE_FPS)

    keypoint_frames, frame_labels, frame_detected = [], [], []
    for f in frames:
        kpts = f.get("keypoints", [])
        if len(kpts) == NUM_JOINTS:
            keypoint_frames.append([(k["x"], k["y"]) for k in kpts])
        else:
            keypoint_frames.append([(0.0, 0.0)] * NUM_JOINTS)
        label = f.get("label", "unannotated")
        frame_labels.append(label)
        # bbox 存在代表該幀 YOLO 有偵測到貓
        frame_detected.append(f.get("bbox") is not None)
        if label != "unannotated":
            frame_counts[label] += 1

    # 以該影片的主類別（多數標注 label）歸類，累加原始影片秒數
    non_unann = [l for l in frame_labels if l != "unannotated"]
    if non_unann:
        video_class = Counter(non_unann).most_common(1)[0][0]
        raw_video_secs[video_class] += len(frames) / actual_fps
        video_count[video_class] += 1

    T = len(keypoint_frames)
    if T < SEQUENCE_LENGTH:
        continue

    for start in range(0, T - SEQUENCE_LENGTH + 1, WINDOW_STRIDE):
        window_labels   = frame_labels[start : start + SEQUENCE_LENGTH]
        window_detected = frame_detected[start : start + SEQUENCE_LENGTH]

        if STRICT_WINDOW_FILTER:
            if "unannotated" in window_labels:
                ann_only = [l for l in window_labels if l != "unannotated"]
                best_unann = Counter(ann_only).most_common(1)[0][0] if ann_only else "unannotated"
                rejected_unann_count[best_unann] += 1
                continue
            lab_counts = Counter(window_labels)
            best, cnt = lab_counts.most_common(1)[0]
        else:
            annotated = [l for l in window_labels if l != "unannotated"]
            if not annotated:
                continue
            best = Counter(annotated).most_common(1)[0][0]

        # bbox 缺失過濾：與 STRICT_WINDOW_FILTER 無關，永遠套用
        no_detect = window_detected.count(False)
        if no_detect > MAX_NO_DETECT_FRAMES:
            rejected_no_detect_count[best] += 1
            rejected_no_detect_detail.append({
                "file": p.name,
                "start": start,
                "end": start + SEQUENCE_LENGTH - 1,
                "label": best,
                "no_detect": no_detect,
            })
            continue

        counts[best] += 1

# ==================== 輸出結果 ====================

print(f"SKELETONS_ROOT       : {SKELETONS_ROOT}")
print(f"SEQUENCE_LENGTH      : {SEQUENCE_LENGTH}")
print(f"WINDOW_STRIDE        : {WINDOW_STRIDE}")
print(f"STRICT_WINDOW_FILTER : {STRICT_WINDOW_FILTER}")
print(f"MAX_NO_DETECT_FRAMES : {MAX_NO_DETECT_FRAMES}")
print(f"SOURCE_FPS           : {SOURCE_FPS}")
print()

# 公式：effective_seconds = labeled_frame_count / SOURCE_FPS
all_classes = sorted(set(list(counts.keys()) + list(frame_counts.keys())))
print(f"{'Class':<12}  {'Sequences':>10}  {'Labeled Frames':>14}  {'Seconds':>10}  {'min:sec':>8}  {'Videos':>7}  {'Raw Video s':>12}  {'raw min:sec':>11}")
print("─" * 96)
total_seq = total_frames = 0
total_raw_secs = 0.0
total_videos   = 0
for k in all_classes:
    seq   = counts[k]
    frms  = frame_counts[k]
    secs  = frms / SOURCE_FPS          # effective_seconds = labeled_frame_count / SOURCE_FPS
    mins  = int(secs // 60)
    s_rem = secs % 60
    raw   = raw_video_secs[k]
    r_min = int(raw // 60)
    r_rem = raw % 60
    vids  = video_count[k]
    print(f"  {k:<12}  {seq:>10}  {frms:>14}  {secs:>9.1f}s  {mins:>3}:{s_rem:04.1f}  {vids:>7}  {raw:>11.1f}s  {r_min:>3}:{r_rem:04.1f}")
    total_seq      += seq
    total_frames   += frms
    total_raw_secs += raw
    total_videos   += vids

total_secs = total_frames / SOURCE_FPS
total_mins = int(total_secs // 60)
total_srem = total_secs % 60
tr_min = int(total_raw_secs // 60)
tr_rem = total_raw_secs % 60
print("─" * 96)
print(f"  {'TOTAL':<12}  {total_seq:>10}  {total_frames:>14}  {total_secs:>9.1f}s  {total_mins:>3}:{total_srem:04.1f}  {total_videos:>7}  {total_raw_secs:>11.1f}s  {tr_min:>3}:{tr_rem:04.1f}")

# ==================== 過濾統計 ====================

if STRICT_WINDOW_FILTER:
    print()
    print(f"{'─' * 60}")
    print(f"【STRICT 模式過濾統計】")

    total_unann_rej = sum(rejected_unann_count.values())
    print(f"\n  因含 unannotated 幀被過濾：{total_unann_rej} 個視窗")
    for cls in sorted(rejected_unann_count):
        print(f"    {cls:<12}  {rejected_unann_count[cls]:>6} 個")

total_no_detect_rej = sum(rejected_no_detect_count.values())
print(f"\n  因 bbox 缺失 > {MAX_NO_DETECT_FRAMES} 幀被過濾：{total_no_detect_rej} 個視窗")
for cls in sorted(rejected_no_detect_count):
    print(f"    {cls:<12}  {rejected_no_detect_count[cls]:>6} 個")

if rejected_no_detect_detail:
    # 按影片彙整：顯示哪些影片有幾個視窗被過濾、最多缺失幾幀
    from collections import defaultdict
    file_summary: dict = defaultdict(lambda: {"count": 0, "max_no_detect": 0, "labels": Counter()})
    for r in rejected_no_detect_detail:
        s = file_summary[r["file"]]
        s["count"] += 1
        s["max_no_detect"] = max(s["max_no_detect"], r["no_detect"])
        s["labels"][r["label"]] += 1

    print(f"\n  影片彙整（共 {len(file_summary)} 支影片有被過濾的視窗）：")
    print(f"  {'File':<45}  {'Rejected':>8}  {'MaxMiss':>7}  Labels")
    print(f"  {'─'*45}  {'─'*8}  {'─'*7}  {'─'*20}")
    for fname, s in sorted(file_summary.items(), key=lambda x: -x[1]["count"]):
        label_str = "  ".join(f"{k}:{v}" for k, v in s["labels"].most_common())
        print(f"  {fname:<45}  {s['count']:>8}  {s['max_no_detect']:>7}  {label_str}")

    print(f"\n  視窗詳細清單（共 {len(rejected_no_detect_detail)} 筆，按缺失幀數降冪）：")
    print(f"  {'File':<45}  {'Start':>6}  {'End':>5}  {'Label':<10}  {'Missing':>7}")
    print(f"  {'─'*45}  {'─'*6}  {'─'*5}  {'─'*10}  {'─'*7}")
    for r in sorted(rejected_no_detect_detail, key=lambda x: -x["no_detect"]):
        print(f"  {r['file']:<45}  {r['start']:>6}  {r['end']:>5}  {r['label']:<10}  {r['no_detect']:>7}")
