# ============================================================
#  骨架資料集收集與手動標注工具
# ============================================================
#  frame label 三種狀態：
#
#  狀態  說明                                    frame label        需處理
#  ────  ──────────────────────────────────────  ─────────────────  ──────
#  1     批次提取（process_all_videos），          全段 = 資料夾名稱   否
#        從未開啟標注模式                          (walk/lick/…)
#
#  2     開啟標注模式，有標記至少一個區間           區間內 = 行為標籤   否
#                                                其餘幀 = unannotated
#                                                （訓練時自動過濾）
#
#  3     開啟標注模式，未標任何區間直接儲存         保留原有 label      否
#        （保護機制：action_intervals 為空時       不覆寫              ）
#        不覆寫 frame label）
# ============================================================

import os
import re
import json
import time
import cv2
import numpy as np
import shutil
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

_BEHAVIOR_RE    = re.compile(r"(walk|lick|scratch|shake|stop)", re.IGNORECASE)
_BEHAVIOR_ORDER = ['walk', 'lick', 'scratch', 'shake', 'stop']

def _parse_behavior(name: str):
    """從檔名抽取行為關鍵字，找不到回傳 None。"""
    m = _BEHAVIOR_RE.search(name)
    return m.group(1).lower() if m else None

# ==================== Configuration ====================
# ==================== Configuration ====================
VIDEO_FOLDERS = [
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\walk",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\lick",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\scratch",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\shake",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\stop",
]
OUTPUT_FOLDER = r"C:\ai_project\paper\skeletons/"
MODEL_PATH = r"C:\ai_project\cat_pose\v11s_106.pt"  # You can use yolov8s-pose.pt, yolov8m-pose.pt for better accuracy
TARGET_FPS = 30
IMGSZ = 640
CONF_THRESHOLD = 0.5
KP_CONF_THRESHOLD = 0.5
# ==================== Main Processing Function ====================

# 恢復批次推論多資料夾影片功能
def process_all_videos():
    """
    批次提取骨架（增量模式）。
    每次執行前先比對 OUTPUT_FOLDER 內現有 JSON，已存在的影片直接跳過，
    不清空資料夾。顯示各類別現有資料量後等待確認才載入模型。
    """
    print("="*60)
    print("Skeleton Extraction Pipeline (Batch)")
    print("="*60)
    setup_directories()

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']

    # ── 步驟 1：收集所有影片，與 OUTPUT_FOLDER 現有 JSON 比對檔名 ─────────────
    existing_stems = {p.stem for p in Path(OUTPUT_FOLDER).glob("*.json")}

    video_files = []
    for folder in VIDEO_FOLDERS:
        video_folder = Path(folder)
        if not video_folder.exists():
            print(f"[Warning] Video folder not found: {folder}")
            continue
        found = [f for f in video_folder.iterdir() if f.suffix.lower() in video_extensions]
        video_files.extend(found)

    if not video_files:
        print(f"✗ No video files found in any of the specified folders.")
        print(f"  Supported formats: {', '.join(video_extensions)}")
        return

    skip_list = [v for v in video_files if v.stem in existing_stems]
    todo_list = [v for v in video_files if v.stem not in existing_stems]

    print(f"\n影片總數：{len(video_files)}   已存在（跳過）：{len(skip_list)}   待提取：{len(todo_list)}")
    if skip_list:
        print("  [Skip]", "  ".join(v.name for v in skip_list))
    if todo_list:
        print("  [Todo]")
        for v in todo_list:
            print(f"    {v.name}")

    # ── 步驟 2：統計現有 skeleton 資料夾各類別數量 ────────────────────────────
    from collections import Counter
    existing_jsons = list(Path(OUTPUT_FOLDER).glob("*.json"))
    class_counts: Counter = Counter()
    for jp in existing_jsons:
        behavior = _parse_behavior(jp.stem) or 'unknown'
        class_counts[behavior] += 1

    print(f"\n現有 skeleton 各類別（共 {len(existing_jsons)} 筆）：")
    sep = "─" * 40
    print(sep)
    for cls in _BEHAVIOR_ORDER + ['unknown']:
        if cls in class_counts:
            bar = '█' * class_counts[cls]
            print(f"  {cls:<10} {class_counts[cls]:>4} 筆  {bar}")
    print(sep)

    if not todo_list:
        print("\n✓ 所有影片皆已提取，無需重新處理。")
        return

    # ── 步驟 3：確認後才載入模型 ─────────────────────────────────────────────
    confirm = input('\n確認後輸入 "ok" 開始提取（其他任意鍵取消）：').strip().lower()
    if confirm != "ok":
        print("✗ 已取消。")
        return

    print()
    pose_extractor = PoseExtractor(
        model_path=MODEL_PATH,
        imgsz=IMGSZ,
        conf_threshold=CONF_THRESHOLD
    )

    results_summary = []
    for idx, video_path in enumerate(todo_list, 1):
        print(f"[{idx}/{len(todo_list)}] Processing: {video_path.name}")
        video_id    = video_path.stem
        output_path = Path(OUTPUT_FOLDER) / f"{video_id}.json"

        # 從資料夾名稱取得行為標籤（walk/lick/scratch/shake/stop）
        label = video_path.parent.name.lower()
        result = extract_skeleton_from_video(
            video_path,
            pose_extractor,
            target_fps=TARGET_FPS,
            label=label
        )
        if result is None:
            print(f"  ✗ Failed to process video")
            continue
        skeleton_data, actual_fps = result
        video_metadata = {
            "video_id": video_id,
            "video_filename": video_path.name,
            "video_path": str(video_path),
            "target_fps": TARGET_FPS,
            "actual_fps": actual_fps,   # 記錄實際來源 FPS，供訓練腳本做時基補償
            "model_used": MODEL_PATH,
            "imgsz": IMGSZ,
            "conf_threshold": CONF_THRESHOLD,
            "kp_conf_threshold": KP_CONF_THRESHOLD
        }
        save_skeleton_data(skeleton_data, output_path, video_metadata)
        detected_frames = sum(1 for f in skeleton_data if f['detected'])
        detection_rate = (detected_frames / len(skeleton_data) * 100) if skeleton_data else 0
        results_summary.append({
            "video_id": video_id,
            "total_frames": len(skeleton_data),
            "detected_frames": detected_frames,
            "detection_rate": detection_rate
        })
        print()
    print("="*60)
    print("Processing Summary")
    print("="*60)
    for result in results_summary:
        print(f"Video: {result['video_id']}")
        print(f"  Total frames: {result['total_frames']}")
        print(f"  Detected frames: {result['detected_frames']}")
        print(f"  Detection rate: {result['detection_rate']:.1f}%")
        print()
    print(f"✓ All done! Skeleton data saved to: {OUTPUT_FOLDER}")

# ==================== Setup ====================
def setup_directories():
    """Create necessary directories if they do not exist"""
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory created: {OUTPUT_FOLDER}")


def clear_output_folder():
    """Clear all existing files/folders under OUTPUT_FOLDER."""
    setup_directories()
    for f in Path(OUTPUT_FOLDER).glob("*"):
        try:
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
        except Exception as e:
            print(f"[Warning] Failed to delete {f}: {e}")
    print(f"✓ Cleared skeletons folder: {OUTPUT_FOLDER}")


# ==================== Video Processing ====================
def get_video_fps(cap):
    """Get the FPS of the video"""
    return cap.get(cv2.CAP_PROP_FPS)


def should_process_frame(frame_count, video_fps, target_fps):
    """
    Determine if a frame should be processed based on target FPS
    
    Args:
        frame_count: Current frame number
        video_fps: Original video FPS
        target_fps: Target FPS for extraction
    
    Returns:
        bool: True if frame should be processed
    """
    if video_fps <= target_fps:
        return True
    
    # Calculate frame interval (use max to prevent division by zero)
    interval = video_fps / target_fps
    interval_int = max(1, round(interval))  # Use round instead of int, ensure >= 1
    return frame_count % interval_int == 0


# ==================== YOLO-Pose Inference ====================
class PoseExtractor:
    """Wrapper class for YOLO-Pose inference"""
    
    def __init__(self, model_path, imgsz=640, conf_threshold=0.5):
        """
        Initialize the pose extractor
        
        Args:
            model_path: Path to YOLO-Pose model
            imgsz: Input image size
            conf_threshold: Confidence threshold for detection
        """
        print(f"Loading YOLO-Pose model from {model_path}...")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        
        # Try to use GPU if available
        self.use_half = False
        try:
            self.model.to("cuda")
            self.use_half = True
            print("✓ Model loaded on GPU")
        except:
            print("✓ Model loaded on CPU")

    def extract_keypoints(self, frame):
        """
        Extract keypoints from a single frame

        Args:
            frame: Input frame (numpy array)

        Returns:
            dict: Dictionary containing keypoints and metadata
                  Returns None if no person/cat detected
        """
        # Run inference
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            half=self.use_half,
            verbose=False
        )[0]
        
        # Check if keypoints are detected
        if results.keypoints is None or len(results.keypoints.xy) == 0:
            return None
        
        # Warn if multiple cats detected
        num_detected = len(results.keypoints.xy)
        if num_detected > 1:
            print(f"  ⚠ Warning: {num_detected} objects detected, using only the first one")
        
        # Get keypoints for the first detected object (assuming single cat in frame)
        keypoints_xy = results.keypoints.xy[0].cpu().numpy()  # Shape: (num_keypoints, 2)
        keypoints_conf = results.keypoints.conf[0].cpu().numpy()  # Shape: (num_keypoints,)
        
        # Get bounding box if available
        bbox = None
        if results.boxes is not None and len(results.boxes) > 0:
            box = results.boxes[0]
            bbox = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
        
        # Format keypoints as list of dictionaries
        keypoints_list = []
        for i, (xy, conf) in enumerate(zip(keypoints_xy, keypoints_conf)):
            keypoints_list.append({
                "joint_id": i,
                "x": float(xy[0]),
                "y": float(xy[1]),
                "conf": float(conf)
            })
        
        return {
            "keypoints": keypoints_list,
            "bbox": bbox,
            "num_keypoints": len(keypoints_list)
        }


# ==================== Video Processing Pipeline ====================
def extract_skeleton_from_video(video_path, pose_extractor, target_fps=30, label=None):
    """
    Extract skeleton sequence from a single video
    
    Args:
        video_path: Path to input video
        pose_extractor: PoseExtractor instance
        target_fps: Target FPS for skeleton extraction
    
    Returns:
        list: List of frame data dictionaries
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"✗ Failed to open video: {video_path}")
        return None
    
    # Get video properties
    video_fps = get_video_fps(cap)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Guard: 某些格式無法讀取 FPS（回傳 0），fallback 到 target_fps 避免 ZeroDivisionError
    if video_fps <= 0:
        print(f"  ⚠ Cannot read FPS from video, assuming {target_fps:.0f}fps")
        video_fps = float(target_fps)

    # 低 FPS 警告：ST-GCN velocity 特徵是逐幀差分，時基不同會讓幅度與訓練資料不一致
    if video_fps < 24:
        print(f"  ⚠ Source FPS={video_fps:.1f} < 24fps — 16 frames will cover "
              f"{16/video_fps:.2f}s instead of {16/target_fps:.2f}s; "
              f"velocity magnitude will differ from {target_fps:.0f}fps training data")

    print(f"  Source FPS: {video_fps:.2f} → target {target_fps}fps  |  Total frames: {total_frames}")
    print(f"  Extracting at {target_fps} FPS...")
    
    skeleton_data = []
    frame_count = 0
    processed_count = 0
    
    # Progress bar
    pbar = tqdm(total=total_frames, desc="  Processing", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if we should process this frame based on target FPS
        if should_process_frame(frame_count, video_fps, target_fps):
            # Extract keypoints
            pose_data = pose_extractor.extract_keypoints(frame)
            
            # Store frame data
            frame_data = {
                "frame_id": processed_count,
                "original_frame_id": frame_count,
                "timestamp": frame_count / video_fps,
                "detected": pose_data is not None
            }
            if label is not None:
                frame_data["label"] = label

            if pose_data is not None:
                frame_data.update(pose_data)
            else:
                # No detection - store empty keypoints
                frame_data["keypoints"] = []
                frame_data["bbox"] = None
                frame_data["num_keypoints"] = 0
            
            skeleton_data.append(frame_data)
            processed_count += 1
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"  ✓ Extracted {processed_count} frames with skeleton data")
    return skeleton_data, float(video_fps)


# ==================== Data Export ====================
def save_skeleton_data(skeleton_data, output_path, video_metadata):
    """
    Save skeleton data to JSON file
    
    Args:
        skeleton_data: List of frame dictionaries
        output_path: Path to output JSON file
        video_metadata: Metadata about the video
    """
    output_data = {
        "video_metadata": video_metadata,
        "frames": skeleton_data,
        "total_frames": len(skeleton_data)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved to: {output_path}")


# ==================== Main Processing Function ====================

def process_single_video():
    """
    只處理單一影片，推論骨架並存成 skeleton json，供標註用。
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(title="選擇影片檔案", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv")])
    if not video_path:
        print("✗ 未選擇影片")
        return

    setup_directories()
    pose_extractor = PoseExtractor(
        model_path=MODEL_PATH,
        imgsz=IMGSZ,
        conf_threshold=CONF_THRESHOLD
    )
    video_id = Path(video_path).stem
    # 從資料夾名稱自動推斷標籤；若資料夾名非行為類別可事後手動修改 JSON
    label = Path(video_path).parent.name.lower()
    result = extract_skeleton_from_video(
        video_path,
        pose_extractor,
        target_fps=TARGET_FPS,
        label=label
    )
    if result is None:
        print(f"  ✗ Failed to process video")
        return
    skeleton_data, actual_fps = result
    video_metadata = {
        "video_id": video_id,
        "video_filename": Path(video_path).name,
        "video_path": str(video_path),
        "target_fps": TARGET_FPS,
        "actual_fps": actual_fps,
        "model_used": MODEL_PATH,
        "imgsz": IMGSZ,
        "conf_threshold": CONF_THRESHOLD,
        "kp_conf_threshold": KP_CONF_THRESHOLD
    }
    output_path = Path(OUTPUT_FOLDER) / f"{video_id}.json"
    save_skeleton_data(skeleton_data, output_path, video_metadata)
    print(f"\n✓ Skeleton JSON 已儲存: {output_path}\n可直接用於手動標註模式。\n")


# ==================== CSV Export (Alternative Format) ====================
def save_skeleton_data_csv(skeleton_data, output_path):
    """
    Alternative function to save skeleton data in CSV format
    
    Args:
        skeleton_data: List of frame dictionaries
        output_path: Path to output CSV file
    """
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Determine number of keypoints (assuming consistent across frames)
        num_kpts = 0
        for frame in skeleton_data:
            if frame['detected'] and len(frame['keypoints']) > 0:
                num_kpts = len(frame['keypoints'])
                break
        
        # Create header
        header = ['frame_id', 'original_frame_id', 'timestamp', 'detected']
        for i in range(num_kpts):
            header.extend([f'joint{i}_x', f'joint{i}_y', f'joint{i}_conf'])
        
        writer = csv.writer(f)
        writer.writerow(header)
        
        # Write data
        for frame in skeleton_data:
            row = [
                frame['frame_id'],
                frame['original_frame_id'],
                frame['timestamp'],
                int(frame['detected'])
            ]
            
            # Add keypoint data
            if frame['detected']:
                for kpt in frame['keypoints']:
                    row.extend([kpt['x'], kpt['y'], kpt['conf']])
            else:
                # Fill with zeros if not detected
                row.extend([0.0] * (num_kpts * 3))
            
            writer.writerow(row)
    
    print(f"  ✓ CSV saved to: {output_path}")


# ==================== Main Entry Point ====================

# ==================== Manual Action Labeling ====================

# 骨架連線（與 cat_pose 腳本保持一致）
_ANNOT_EDGES = [
    (0, 1), (0, 2), (1, 2),
    (0, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (3, 8), (8, 9),
    (5, 10), (10, 11), (5, 12), (12, 13),
    (5, 14), (14, 15), (15, 16),
]
_ANNOT_EDGE_COLORS = [
    (255,120,60),(255,120,60),(255,120,60),
    (220,220,60),(200,220,60),(160,220,60),
    (102,85,255),(102,85,255),(255,68,204),(255,68,204),
    (255,170,34),(255,170,34),(0,153,255),(0,153,255),
    (80,200,160),(60,170,130),(40,140,100),
]


def _list_json_files_menu(folder: str, last_annotated: str = None):
    """
    在終端列出 folder 內所有 skeleton JSON，依行為關鍵字分組顯示標注進度。
    比對失敗的檔案歸入「未分類」群組（不捨棄）。
    回傳選中的路徑字串；q 或無可選時回傳 None。
    """
    p = Path(folder)
    if not p.exists():
        print(f"[Error] 資料夾不存在: {folder}")
        return None

    json_files = sorted(p.glob("*.json"),
                        key=lambda f: [int(t) if t.isdigit() else t
                                       for t in re.split(r'(\d+)', f.name.lower())])
    if not json_files:
        print(f"[Error] 找不到任何 JSON 檔案: {folder}")
        return None

    # 輕量掃描：只讀 action_intervals 與 total_frames
    file_infos = []
    for jf in json_files:
        info = {'path': jf, 'n_intervals': 0, 'total_frames': 0, 'label': '', 'video': ''}
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                d = json.load(f)
            info['n_intervals']  = len(d.get('action_intervals', []))
            info['total_frames'] = d.get('total_frames', len(d.get('frames', [])))
            meta = d.get('video_metadata', {})
            info['label']  = meta.get('label', '') or \
                             (d['frames'][0].get('label', '') if d.get('frames') else '')
            info['video']  = Path(meta.get('video_filename', '')).name
        except Exception:
            pass
        # 從檔名抽取行為關鍵字，比對失敗的放入 'unknown'（不捨棄）
        info['behavior'] = _parse_behavior(jf.stem) or 'unknown'
        file_infos.append(info)

    # 依行為分組（保留順序：walk/lick/scratch/shake/stop/unknown）
    groups = {b: [] for b in _BEHAVIOR_ORDER}
    groups['unknown'] = []
    for fi in file_infos:
        groups[fi['behavior']].append(fi)

    total  = len(file_infos)
    n_done = sum(1 for fi in file_infos if fi['n_intervals'] > 0)
    sep    = '─' * 72

    # 計算上次標注的全域索引
    last_idx = None
    last_name = Path(last_annotated).name if last_annotated else None

    print(f"\n{'='*72}")
    print(f"  Annotation file list   {folder}")
    print(f"  總進度: {n_done}/{total} 已完成  ({total - n_done} 待標注)")

    # 全域流水號（1-based），讓使用者直接輸入
    global_idx = 0
    ordered_flat = []   # [(global_1based, fi), ...]

    for behavior in list(_BEHAVIOR_ORDER) + ['unknown']:
        grp = groups[behavior]
        if not grp:
            continue

        g_done  = sum(1 for fi in grp if fi['n_intervals'] > 0)
        g_total = len(grp)

        print(sep)
        if behavior == 'unknown':
            print(f"  ⚠  未分類（檔名無法比對行為關鍵字）  {g_done}/{g_total} 已完成")
        else:
            bar = '█' * g_done + '░' * (g_total - g_done)
            print(f"  [{behavior.upper():<8}]  {g_done}/{g_total} 已完成  {bar}")

        for fi in grp:
            global_idx += 1
            ordered_flat.append(fi)
            status   = '✓' if fi['n_intervals'] > 0 else '·'
            detail   = (f"{fi['n_intervals']} 區間"
                        if fi['n_intervals'] > 0
                        else f"{fi['total_frames']} 幀  未標注")
            vid_hint = f"  ← {fi['video']}" if fi['video'] else ''
            # 標記「上次標注」的那一行
            last_mark = '  ← 上次' if (last_name and
                                        fi['path'].name == last_name) else ''
            if last_mark and last_idx is None:
                last_idx = global_idx
            print(f"    [{global_idx:3d}] {status}  {fi['path'].name:<38}  {detail}{vid_hint}{last_mark}")

    print(sep)
    if last_idx is not None and last_name:
        print(f"  上次標注: [{last_idx}]  {last_name}")
    print("  輸入編號選擇  |  直接 Enter = 自動跳下一個未標注  |  q = 離開")

    while True:
        choice = input("  > ").strip()
        if choice.lower() == 'q':
            return None
        if choice == '':
            # 優先從上次標注的位置往後找，找不到就從頭
            start = (last_idx or 0)          # last_idx 是 1-based
            candidates = (list(range(start, len(ordered_flat))) +
                          list(range(0, start)))
            for i in candidates:
                if ordered_flat[i]['n_intervals'] == 0:
                    chosen = ordered_flat[i]
                    print(f"  → 自動選擇: [{i+1}] {chosen['path'].name}  [{chosen['behavior'].upper()}]")
                    return str(chosen['path'])
            print("  所有檔案都已標注完成。")
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(ordered_flat):
                return str(ordered_flat[idx]['path'])
            print(f"  請輸入 1～{len(ordered_flat)} 之間的數字")
        except ValueError:
            print("  請輸入數字或 q")


def _annotate_single_skeleton(json_path, file_index=None, total_files=None):
    """
    對單一 skeleton JSON 檔案進行手動標註並儲存。
    改善版：終端選檔、骨架連線繪製、標記狀態視覺、底部時間軸、HUD 面板。
    """
    import tkinter as tk
    from tkinter import filedialog

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    frames       = data['frames']
    total_frames = len(frames)

    # 載入既有標注區間（支援重新標注同一檔案）
    intervals = []
    for iv in data.get('action_intervals', []):
        intervals.append((iv['start'], iv['end'], iv['action']))

    # 影片路徑
    video_path = None
    if 'video_metadata' in data and 'video_path' in data['video_metadata']:
        video_path = data['video_metadata']['video_path']
    elif 'video_path' in data:
        video_path = data['video_path']
    if not video_path or not os.path.exists(video_path):
        print("✗ 找不到影片路徑，請手動選擇影片檔案（可直接關閉視窗僅標骨架）")
        root2 = tk.Tk()
        root2.withdraw()
        root2.attributes('-topmost', True)
        video_path = filedialog.askopenfilename(
            title="選擇對應影片檔案（可略過）",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv")]
        )
        root2.destroy()
        if not video_path or not os.path.exists(video_path):
            video_path = None

    VALID_ACTIONS = ['walk', 'lick', 'scratch', 'shake', 'stop']
    ACTION_COLORS = {
        'walk':        (0,  200,  0),
        'scratch':     (0,  140, 255),
        'lick':        (0,  220, 220),
        'shake':       (60,  60, 220),
        'stop':        (0,  165, 255),
        'unannotated': (40,  40,  40),
    }
    # 從檔名關鍵字預設行為；比對失敗則 fallback 為 walk
    _default_action = _parse_behavior(Path(json_path).stem)
    current_action = _default_action if _default_action in VALID_ACTIONS else 'walk'
    if _default_action:
        print(f"  [檔名預設行為]  {current_action.upper()}  （可按 1-5 手動切換）")

    print("\n操作說明：")
    print("  1/2/3/4/5 切換行為  |  s 標記起點/終點  |  u 撤銷上一個區間")
    print("  a/d 前/後幀  |  z/x 調整步長  |  SPACE 播放/暫停  |  t 跳轉秒數")
    print("  q 儲存並離開    [未標記片段訓練時自動捨棄]\n")

    marking           = False
    start_idx         = None
    cur_idx           = 0
    cap               = None
    playing           = False
    skip_n            = 1
    needs_seek        = True
    cap_next_orig_fid = -1
    last_advance_t    = 0.0
    render_needed     = True
    cached_frame_img  = None
    cached_frame_idx  = -1
    flash_msg         = ''
    flash_until       = 0.0   # time.time() + duration

    MAX_DISP_W, MAX_DISP_H = 1280, 720
    frame_shape = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
        ret, sample = cap.read()
        if ret:
            h, w = sample.shape[:2]
            if w > MAX_DISP_W or h > MAX_DISP_H:
                r = min(MAX_DISP_W / w, MAX_DISP_H / h)
                sample = cv2.resize(sample, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
            frame_shape = sample.shape
    if frame_shape is None:
        frame_shape = (540, 960, 3)

    play_delay_ms = 33
    if cap is not None and cap.isOpened():
        _fps = cap.get(cv2.CAP_PROP_FPS)
        if _fps > 0:
            play_delay_ms = max(16, min(int(1000 / _fps), 66))

    frame_interval = 1
    if len(frames) >= 2:
        _gaps = [frames[i+1].get('original_frame_id', i+1) - frames[i].get('original_frame_id', i)
                 for i in range(min(10, len(frames) - 1))]
        if _gaps:
            frame_interval = max(1, int(round(sum(_gaps) / len(_gaps))))

    # ── 輔助繪圖函式 ──────────────────────────────────────────────────────────

    def draw_skeleton_with_edges(img, keypoints, kp_scale=1.0):
        """彩色骨架連線 + 關節點，取代純點繪製。"""
        kd  = {kpt['joint_id']: kpt for kpt in keypoints}
        w   = img.shape[1]
        lw  = max(1, int(w / 480))
        ro   = max(3, int(w / 320))
        ri   = max(2, int(w / 426))
        for ei, (a, b) in enumerate(_ANNOT_EDGES):
            ka, kb = kd.get(a), kd.get(b)
            if ka and kb and ka['conf'] > 0.2 and kb['conf'] > 0.2:
                pa = (int(ka['x'] * kp_scale), int(ka['y'] * kp_scale))
                pb = (int(kb['x'] * kp_scale), int(kb['y'] * kp_scale))
                col = _ANNOT_EDGE_COLORS[ei] if ei < len(_ANNOT_EDGE_COLORS) else (160, 160, 160)
                cv2.line(img, pa, pb, col, lw, cv2.LINE_AA)
        for kpt in keypoints:
            if kpt['conf'] > 0.2:
                x, y = int(kpt['x'] * kp_scale), int(kpt['y'] * kp_scale)
                cv2.circle(img, (x, y), ro, (0, 0, 0),   -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), ri, (0, 220, 60), -1, cv2.LINE_AA)

    def draw_timeline(img, total_f, cur_i, ivs, act_cols):
        """底部橫向時間軸：顯示各標注區間與當前位置。"""
        h, w  = img.shape[:2]
        bh    = max(10, int(h * 0.022))
        mx    = 6
        by    = h - bh - mx
        bx1, bx2 = mx, w - mx
        bw    = bx2 - bx1
        cv2.rectangle(img, (bx1, by), (bx2, by + bh), (25, 25, 25), -1)
        for s, e, act in ivs:
            x1 = bx1 + int(s / max(total_f, 1) * bw)
            x2 = bx1 + int((e + 1) / max(total_f, 1) * bw)
            col = act_cols.get(act, (80, 80, 80))
            cv2.rectangle(img, (x1, by + 1), (max(x1 + 1, x2), by + bh - 1), col, -1)
        cx = bx1 + int(cur_i / max(total_f - 1, 1) * bw)
        cv2.line(img, (cx, by - 3), (cx, by + bh + 3), (255, 255, 255), 2)
        cv2.circle(img, (cx, by + bh // 2), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.rectangle(img, (bx1, by), (bx2, by + bh), (70, 70, 70), 1)

    def draw_marking_indicator(img, act, act_col, dur_s, sc):
        """標記進行中：粗彩色外框 + REC 徽章。"""
        h, w = img.shape[:2]
        bw   = max(4, int(9 * sc))
        cv2.rectangle(img, (bw // 2, bw // 2), (w - bw // 2, h - bw // 2), act_col, bw)
        badge = f"  REC [{act.upper()}]  {dur_s:.1f}s  "
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.65 * sc, max(1, int(2 * sc)))
        bx = w - tw - int(12 * sc)
        by = int(12 * sc)
        ov = img.copy()
        cv2.rectangle(ov, (bx - 6, by - 6), (bx + tw + 6, by + th + 8), act_col, -1)
        cv2.addWeighted(ov, 0.82, img, 0.18, 0, img)
        cv2.putText(img, badge, (bx, by + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65 * sc, (10, 10, 10),
                    max(1, int(2 * sc)), cv2.LINE_AA)
        cv2.circle(img, (bx + int(7 * sc), by + th // 2 + int(2 * sc)),
                   max(3, int(5 * sc)), (0, 0, 200), -1, cv2.LINE_AA)

    def draw_hud(img, line1, line2, sc):
        """底部半透明 HUD（在時間軸上方）。"""
        h, w  = img.shape[:2]
        tl_h  = max(10, int(h * 0.022)) + 6
        hh    = max(46, int(58 * sc))
        hy    = h - tl_h - hh
        ov    = img.copy()
        cv2.rectangle(ov, (0, hy), (w, h - tl_h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
        lh = int(hh * 0.46)
        cv2.putText(img, line1, (int(8 * sc), hy + lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52 * sc, (210, 210, 210),
                    max(1, int(sc)), cv2.LINE_AA)
        cv2.putText(img, line2, (int(8 * sc), hy + lh * 2 - int(3 * sc)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47 * sc, (150, 150, 150),
                    max(1, int(sc)), cv2.LINE_AA)

    def draw_top_bar(img, text, sc):
        """頂部細資訊欄。"""
        bh = max(22, int(27 * sc))
        ov = img.copy()
        cv2.rectangle(ov, (0, 0), (img.shape[1], bh), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.62, img, 0.38, 0, img)
        cv2.putText(img, text, (int(8 * sc), bh - int(5 * sc)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47 * sc, (170, 210, 255),
                    max(1, int(sc)), cv2.LINE_AA)

    def draw_flash(img, msg, sc):
        """畫面中央短暫提示訊息。"""
        h, w = img.shape[:2]
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX,
                                       0.85 * sc, max(1, int(2 * sc)))
        x, y = (w - tw) // 2, h // 2
        ov = img.copy()
        cv2.rectangle(ov, (x - 14, y - th - 14), (x + tw + 14, y + 14), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.72, img, 0.28, 0, img)
        cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85 * sc, (80, 240, 80), max(1, int(2 * sc)), cv2.LINE_AA)

    # ── 視窗 ──────────────────────────────────────────────────────────────────
    file_name  = Path(json_path).name
    # 從 video_path 父資料夾推斷行為類別（比 frames[0]['label'] 穩定，不受標注狀態影響）
    _vp = data.get('video_metadata', {}).get('video_path', '')
    _folder = Path(_vp).parent.name.lower() if _vp else ''
    file_label = _folder if _folder in ('walk', 'lick', 'scratch', 'shake', 'stop') \
                 else (frames[0].get('label', '') if frames else '')
    ctx_prefix = f"[{file_index}/{total_files}] " if file_index and total_files else ""
    win_name   = "Annotation"   # 固定名稱，確保每次重用同一個視窗
    cv2.namedWindow(win_name, cv2.WINDOW_KEEPRATIO)
    if frame_shape:
        cv2.resizeWindow(win_name, frame_shape[1], frame_shape[0])

    # ── 主迴圈 ────────────────────────────────────────────────────────────────
    while True:
        frame_data = frames[cur_idx]

        if cur_idx != cached_frame_idx:
            show_base = None
            kp_scale  = 1.0
            if cap is not None and cap.isOpened():
                target_orig_fid = frame_data.get('original_frame_id', cur_idx)
                gap = target_orig_fid - cap_next_orig_fid
                if needs_seek or gap < 0 or gap > skip_n * frame_interval + 30:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_orig_fid)
                    cap_next_orig_fid = target_orig_fid
                    needs_seek = False
                elif gap > 0:
                    for _ in range(gap):
                        cap.grab()
                    cap_next_orig_fid = target_orig_fid
                ret, img = cap.read()
                cap_next_orig_fid += 1
                if ret:
                    h, w = img.shape[:2]
                    if w > MAX_DISP_W or h > MAX_DISP_H:
                        kp_scale = min(MAX_DISP_W / w, MAX_DISP_H / h)
                        img = cv2.resize(img, (int(w * kp_scale), int(h * kp_scale)),
                                         interpolation=cv2.INTER_LINEAR)
                    show_base = img
            if show_base is None:
                h, w = frame_shape[:2]
                show_base = np.ones((h, w, 3), dtype=np.uint8) * 30
            if frame_data['detected'] and frame_data.get('keypoints'):
                draw_skeleton_with_edges(show_base, frame_data['keypoints'], kp_scale)
            cached_frame_img = show_base
            cached_frame_idx = cur_idx
            render_needed    = True

        if render_needed or time.time() < flash_until:
            show_img  = cached_frame_img.copy()
            sc        = max(0.5, show_img.shape[1] / 960.0)
            timestamp = frame_data.get('timestamp')
            orig_fid  = frame_data.get('original_frame_id')
            n_kpts    = len(frame_data.get('keypoints', []))
            t_str     = f"{timestamp:.2f}s" if timestamp is not None else '--'
            play_str  = '[PLAY]' if playing else '[PAUSE]'

            # 行為在某區間內時高亮現有標注
            for s, e, act in intervals:
                if s <= cur_idx <= e:
                    col = ACTION_COLORS.get(act, (80, 80, 80))
                    cv2.rectangle(show_img, (2, 2),
                                  (show_img.shape[1] - 2, show_img.shape[0] - 2), col, 2)

            # 標記中：粗外框 + REC 徽章
            if marking:
                s_ts  = frames[start_idx].get('timestamp', 0) or 0
                c_ts  = frame_data.get('timestamp', 0) or 0
                dur_s = abs(c_ts - s_ts)
                act_col = ACTION_COLORS.get(current_action, (200, 200, 200))
                draw_marking_indicator(show_img, current_action, act_col, dur_s, sc)
                line2 = (f"  from frame {start_idx+1} ({s_ts:.1f}s)  ->  now ({c_ts:.1f}s)"
                         f"  dur={dur_s:.2f}s  |  s=END MARK  u=cancel  q=SAVE")
            else:
                act_col = ACTION_COLORS.get(current_action, (200, 200, 200))
                line2 = (f"  1=walk 2=lick 3=scratch 4=shake 5=stop  |  "
                         f"s=START MARK  u=UNDO  a/d=nav  z/x=skip  t=jump  SPACE  q=SAVE")

            line1 = (f"{play_str}  Frame {cur_idx+1}/{total_frames}  ({t_str})"
                     f"  skip:{skip_n}  |  Intervals:{len(intervals)}"
                     + (f"  VidFr:{orig_fid}" if orig_fid is not None else "")
                     + ("" if frame_data.get('detected') else "  [NO DETECT]"))

            draw_timeline(show_img, total_frames, cur_idx, intervals, ACTION_COLORS)
            draw_hud(show_img, line1, line2, sc)

            # 頂部欄
            lbl_tag = f" [{file_label.upper()}]" if file_label else ''
            top_text = (f"{ctx_prefix}{file_name}{lbl_tag}"
                        f"  |  {len(intervals)} interval(s)"
                        + (f"  kpts:{n_kpts}" if frame_data.get('detected') else "  [NO DETECT]"))
            draw_top_bar(show_img, top_text, sc)

            # 當前行為色塊（左下角小標籤）
            h_img = show_img.shape[0]
            tl_h  = max(10, int(h_img * 0.022)) + 6
            hh    = max(46, int(58 * sc))
            lbl_y = h_img - tl_h - hh - int(6 * sc)
            cv2.putText(show_img, f'[{current_action.upper()}]',
                        (int(8 * sc), lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72 * sc, act_col,
                        max(1, int(2 * sc)), cv2.LINE_AA)

            # 閃爍訊息
            if time.time() < flash_until:
                draw_flash(show_img, flash_msg, sc)
                render_needed = True   # 保持更新直到 flash 結束
            else:
                render_needed = False

            cv2.imshow(win_name, show_img)

        # 按鍵等待
        if playing or time.time() < flash_until:
            elapsed      = time.time() - last_advance_t
            remaining_ms = max(30, int((play_delay_ms / 1000.0 - elapsed) * 1000)) if playing else 30
            key = cv2.waitKey(remaining_ms) & 0xFF
        else:
            key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            playing = not playing
            if playing:
                last_advance_t = time.time()
            render_needed = True
        elif key == ord('a'):
            playing    = False
            cur_idx    = max(0, cur_idx - skip_n)
            needs_seek = True
            render_needed = True
        elif key == ord('d'):
            playing   = False
            cur_idx   = min(total_frames - 1, cur_idx + skip_n)
            render_needed = True
        elif key == ord('z'):
            skip_n = max(1, skip_n - 1)
            render_needed = True
        elif key == ord('x'):
            skip_n += 1
            render_needed = True
        elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            current_action = VALID_ACTIONS[key - ord('1')]
            render_needed  = True
        elif key == ord('s'):
            if not marking:
                start_idx  = cur_idx
                marking    = True
                s_ts = frames[start_idx].get('timestamp', 0) or 0
                flash_msg   = f"MARK START  [{current_action.upper()}]  @ {s_ts:.2f}s"
                flash_until = time.time() + 2.5
                print(f"  [{current_action}] 標記起點: {start_idx+1} ({s_ts:.2f}s)")
            else:
                end_idx = cur_idx
                if end_idx < start_idx:
                    start_idx, end_idx = end_idx, start_idx
                intervals.append((start_idx, end_idx, current_action))
                s_ts = frames[start_idx].get('timestamp', 0) or 0
                e_ts = frames[end_idx].get('timestamp', 0) or 0
                dur  = abs(e_ts - s_ts)
                flash_msg   = f"SAVED  [{current_action.upper()}]  {s_ts:.2f}s - {e_ts:.2f}s  (dur {dur:.2f}s)"
                flash_until = time.time() + 5.0
                print(f"  [{current_action}]  {s_ts:.2f}s - {e_ts:.2f}s  ({dur:.2f}s)  [total {len(intervals)} interval(s)]")
                marking = False
            render_needed = True
        elif key == ord('u'):
            if marking:
                marking   = False
                flash_msg = 'MARK CANCELLED'
            elif intervals:
                removed   = intervals.pop()
                flash_msg = f"UNDO  [{removed[2].upper()}]  frames {removed[0]+1}~{removed[1]+1}"
                print(f"  ↩ 撤銷: {removed}")
            else:
                flash_msg = '(nothing to undo)'
            flash_until   = time.time() + 1.2
            render_needed = True
        elif key == ord('t'):
            try:
                raw = input("跳轉（幀號整數 或 秒數如 3.5 / 3.5s）: ").strip()
                if '.' in raw or raw.endswith('s'):
                    sec = np.float64(raw.rstrip('s'))
                    cur_idx = min(range(total_frames),
                                  key=lambda i: abs(frames[i].get('timestamp', 0) - sec))
                else:
                    # 1-based 幀號輸入
                    cur_idx = max(0, min(total_frames - 1, int(raw) - 1))
                needs_seek = True
                playing    = False
                render_needed = True
                print(f"  跳轉到第 {cur_idx+1} 幀 ({frames[cur_idx].get('timestamp', 0):.2f}s)")
            except Exception as e:
                print(f"  ✗ 跳轉失敗: {e}")
        elif playing and key == 0xFF:
            if cur_idx < total_frames - 1:
                cur_idx        = min(cur_idx + 1, total_frames - 1)
                last_advance_t = time.time()
                render_needed  = True
            else:
                playing    = False
                needs_seek = True
                render_needed  = True

    cv2.destroyAllWindows()
    for _ in range(5):   # Windows 需要多次 pump 才能真正關閉視窗
        cv2.waitKey(1)
    if cap:
        cap.release()

    # ── frame label 三種狀態 ────────────────────────────────────────────────────
    # 狀態 1：批次提取（process_all_videos）、從未開啟標注模式
    #         → 每幀 label = 資料夾名稱（walk/lick/…），全段有效，無需處理
    # 狀態 2：開啟標注模式並標記了至少一個區間後儲存
    #         → 區間內 = 行為標籤，其餘幀 = 'unannotated'（訓練時自動過濾）
    # 狀態 3：開啟標注模式但未標記任何區間直接儲存（保護機制）
    #         → 偵測到 action_intervals 為空，保留原有 frame label 不覆寫
    # ────────────────────────────────────────────────────────────────────────────

    # 依行為類別分組並合併各自重疊區段
    from collections import defaultdict
    def merge_intervals_for_action(raw):
        raw = sorted([(min(s, e), max(s, e)) for s, e in raw])
        merged = []
        for s, e in raw:
            if not merged or merged[-1][1] < s - 1:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        return merged

    by_action = defaultdict(list)
    for s, e, act in intervals:
        by_action[act].append((s, e))

    action_intervals = []
    for act, segs in by_action.items():
        for s, e in merge_intervals_for_action(segs):
            action_intervals.append({"action": act, "start": int(s), "end": int(e)})
    action_intervals.sort(key=lambda x: x['start'])

    # 產生 frame-level label
    # 保護：若本次未標記任何區間，保留幀的原有標籤（批次提取的資料整段皆有效）
    # 只有在有區間標記時才覆寫，避免開啟後直接 q 存檔把批次標籤全清成 unannotated
    all_intervals = sorted(action_intervals, key=lambda x: x['start'])

    if not action_intervals:
        print("  [保護] 未標記任何區間，保留原有 frame label（不覆寫）")
    else:
        frame_labels = ['unannotated'] * total_frames
        for iv in action_intervals:
            for i in range(iv['start'], iv['end'] + 1):
                if 0 <= i < total_frames:
                    frame_labels[i] = iv['action']
        for i, frame in enumerate(frames):
            frame['label'] = frame_labels[i]

    out_json = data.copy()
    out_json['action_intervals'] = all_intervals
    out_json['frames'] = frames
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 已直接覆蓋原標註檔案: {json_path}\n✓ 已自動合併重疊區段，frames 內每一幀都含 label 欄位")


def manual_action_labeling():
    """
    連續標記多個 skeleton JSON 檔案。
    標記完成後自動回到列表，輸入 q 退出。
    """
    print("\n=== Annotation Mode ===")
    print(f"Folder: {OUTPUT_FOLDER}\n")

    last_annotated = None
    while True:
        json_path = _list_json_files_menu(OUTPUT_FOLDER, last_annotated=last_annotated)
        if not json_path:
            print("\n[Done] Annotation session ended.")
            break

        # 計算在整個 JSON 列表中的位置，供視窗標題顯示進度
        try:
            all_jsons   = sorted(Path(OUTPUT_FOLDER).glob("*.json"))
            file_index  = next((i + 1 for i, jf in enumerate(all_jsons)
                                if str(jf) == json_path), None)
            total_files = len(all_jsons)
        except Exception:
            file_index = total_files = None

        _annotate_single_skeleton(json_path,
                                   file_index=file_index,
                                   total_files=total_files)

        # 標記完成後在終端列印明確摘要
        sep = '=' * 62
        print(f"\n{sep}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            ivs      = d.get('action_intervals', [])
            n_frames = d.get('total_frames', 0)
            labeled  = sum(1 for fr in d.get('frames', [])
                           if fr.get('label', 'unannotated') != 'unannotated')
            pct      = labeled / n_frames * 100 if n_frames > 0 else 0.0

            # 統計各行為的區間數
            from collections import Counter
            act_counts = Counter(iv['action'] for iv in ivs)
            act_str    = '  '.join(f"{act}x{cnt}" for act, cnt in sorted(act_counts.items()))

            print(f"  ANNOTATED : {Path(json_path).name}")
            print(f"  Intervals : {len(ivs)}   ({act_str if act_str else 'none'})")
            print(f"  Labeled   : {labeled}/{n_frames} frames  ({pct:.1f}%)")
        except Exception as e:
            print(f"  ANNOTATED : {Path(json_path).name}")
            print(f"  (Could not read summary: {e})")
        print(f"{sep}\n")
        last_annotated = json_path   # 供下次列表顯示「上次標記」

    print("\n[Done] All annotation tasks completed.")


def reextract_preserve_labels():
    """
    以新 YOLO 模型重新推論骨架，但保留既有 JSON 的 action_intervals 與 frame label。
    前提：TARGET_FPS 不變，則抽幀順序相同，區間 index 可直接重用。
    """
    print("="*60)
    print("Skeleton Re-extraction (preserve annotations)")
    print("="*60)
    setup_directories()

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']

    video_files = []
    for folder in VIDEO_FOLDERS:
        vp = Path(folder)
        if not vp.exists():
            print(f"[Warning] 資料夾不存在: {folder}")
            continue
        video_files.extend(f for f in vp.iterdir() if f.suffix.lower() in video_extensions)

    if not video_files:
        print("✗ 找不到任何影片。")
        return

    # 讀取所有既有 JSON 的 action_intervals
    saved_intervals: dict[str, list] = {}
    for vf in video_files:
        jp = Path(OUTPUT_FOLDER) / f"{vf.stem}.json"
        if jp.exists():
            try:
                with open(jp, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                ivs = d.get('action_intervals', [])
                saved_intervals[vf.stem] = ivs
            except Exception as e:
                print(f"  [Warning] 無法讀取 {jp.name}: {e}")

    annotated   = [v for v in video_files if saved_intervals.get(v.stem)]
    unannotated = [v for v in video_files if not saved_intervals.get(v.stem)]

    print(f"\n影片總數：{len(video_files)}")
    print(f"  含標注（保留區間）：{len(annotated)}")
    print(f"  無標注（全段重推）：{len(unannotated)}")
    if annotated:
        print("\n  [含標注]")
        for v in annotated:
            n = len(saved_intervals[v.stem])
            print(f"    {v.name}  →  {n} 個區間將保留")

    confirm = input('\n確認後輸入 "ok" 開始重新推論（其他任意鍵取消）：').strip().lower()
    if confirm != "ok":
        print("✗ 已取消。")
        return

    pose_extractor = PoseExtractor(
        model_path=MODEL_PATH,
        imgsz=IMGSZ,
        conf_threshold=CONF_THRESHOLD
    )

    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] {video_path.name}")
        video_id    = video_path.stem
        output_path = Path(OUTPUT_FOLDER) / f"{video_id}.json"
        label       = video_path.parent.name.lower()

        result = extract_skeleton_from_video(
            video_path, pose_extractor, target_fps=TARGET_FPS, label=label
        )
        if result is None:
            print("  ✗ 推論失敗，跳過")
            continue
        skeleton_data, actual_fps = result

        # 還原既有標注
        ivs = saved_intervals.get(video_id, [])
        if ivs:
            total_f = len(skeleton_data)
            frame_labels = ['unannotated'] * total_f
            for iv in ivs:
                for i in range(iv['start'], iv['end'] + 1):
                    if 0 <= i < total_f:
                        frame_labels[i] = iv['action']
            for i, fd in enumerate(skeleton_data):
                fd['label'] = frame_labels[i]
            print(f"  ✓ 還原 {len(ivs)} 個 action_intervals，frame label 已重新套用")
        else:
            print("  → 無既有標注，frame label 保持資料夾名稱")

        video_metadata = {
            "video_id": video_id,
            "video_filename": video_path.name,
            "video_path": str(video_path),
            "target_fps": TARGET_FPS,
            "actual_fps": actual_fps,
            "model_used": MODEL_PATH,
            "imgsz": IMGSZ,
            "conf_threshold": CONF_THRESHOLD,
            "kp_conf_threshold": KP_CONF_THRESHOLD
        }
        out_data = {
            "video_metadata": video_metadata,
            "frames": skeleton_data,
            "total_frames": len(skeleton_data),
            "action_intervals": ivs,
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 已儲存: {output_path}")

    print("\n✓ 全部重新推論完成。")


if __name__ == "__main__":
    print("\n==== Cat Skeleton 批次推論/手動標註 ====")
    print("1. 批次推論五個資料夾影片 (YOLO-Pose)  [增量，跳過已有 JSON]")
    print("   → 影片依資料夾名稱 (walk/lick/scratch/shake/stop) 自動標記，可直接訓練")
    print("2. 連續手動標記多個 skeleton JSON")
    print("   → 適用影片含多種行為、需精確逐段標記的情況")
    print("3. 重新推論骨架（新模型），保留既有 action_intervals 與 frame label")
    print("   → YOLO 模型更換後使用；TARGET_FPS 不變則標注 index 可直接重用")
    mode = input("請選擇模式 (1/2/3): ").strip()
    if mode == '1':
        process_all_videos()
    elif mode == '2':
        manual_action_labeling()
    elif mode == '3':
        reextract_preserve_labels()
    else:
        print("✗ 未選擇正確模式，程式結束。")