

import os
import json
import time
import cv2
import numpy as np
import shutil
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ==================== Configuration ====================
# ==================== Configuration ====================
VIDEO_FOLDERS = [
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\walk",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\scratch",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\lick",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\模型專用\shake"
]
OUTPUT_FOLDER = r"C:\cat_pose\gcn_pose\skeletons/"
MODEL_PATH = r"C:\cat_pose\v11s_54.pt"  # You can use yolov8s-pose.pt, yolov8m-pose.pt for better accuracy
TARGET_FPS = 30
IMGSZ = 640
CONF_THRESHOLD = 0.5
KP_CONF_THRESHOLD = 0.3
# ==================== Main Processing Function ====================

# 恢復批次推論多資料夾影片功能
def process_all_videos():
    """
    Main function to process all videos in the input folders (VIDEO_FOLDERS)
    """
    print("="*60)
    print("Skeleton Extraction Pipeline (Batch)")
    print("="*60)
    clear_output_folder()

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    for folder in VIDEO_FOLDERS:
        video_folder = Path(folder)
        if not video_folder.exists():
            print(f"[Warning] Video folder not found: {folder}")
            continue
        found = [f for f in video_folder.iterdir() if f.suffix.lower() in video_extensions]
        print(f"{folder} 影片數量: {len(found)}")

    pose_extractor = PoseExtractor(
        model_path=MODEL_PATH,
        imgsz=IMGSZ,
        conf_threshold=CONF_THRESHOLD
    )

    video_files = []
    for folder in VIDEO_FOLDERS:
        video_folder = Path(folder)
        if not video_folder.exists():
            continue
        found = [f for f in video_folder.iterdir() if f.suffix.lower() in video_extensions]
        video_files.extend(found)

    if len(video_files) == 0:
        print(f"✗ No video files found in any of the specified folders.")
        print(f"  Supported formats: {', '.join(video_extensions)}")
        return

    print(f"\nFound {len(video_files)} video(s) to process from all folders\n")
    results_summary = []
    for idx, video_path in enumerate(video_files, 1):
        print(f"[{idx}/{len(video_files)}] Processing: {video_path.name}")
        video_id = video_path.stem
        # 從資料夾名稱取得行為標籤（walk/lick/scratch/shake）
        label = video_path.parent.name.lower()
        skeleton_data = extract_skeleton_from_video(
            video_path,
            pose_extractor,
            target_fps=TARGET_FPS,
            label=label
        )
        if skeleton_data is None:
            print(f"  ✗ Failed to process video")
            continue
        video_metadata = {
            "video_id": video_id,
            "video_filename": video_path.name,
            "video_path": str(video_path),
            "target_fps": TARGET_FPS,
            "model_used": MODEL_PATH,
            "imgsz": IMGSZ,
            "conf_threshold": CONF_THRESHOLD,
            "kp_conf_threshold": KP_CONF_THRESHOLD
        }
        output_path = Path(OUTPUT_FOLDER) / f"{video_id}.json"
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
        try:
            self.model.to("cuda")
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
    
    print(f"  Original FPS: {video_fps:.2f}, Total Frames: {total_frames}")
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
    return skeleton_data


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
    skeleton_data = extract_skeleton_from_video(
        video_path,
        pose_extractor,
        target_fps=TARGET_FPS,
        label=label
    )
    if skeleton_data is None:
        print(f"  ✗ Failed to process video")
        return
    video_metadata = {
        "video_id": video_id,
        "video_filename": Path(video_path).name,
        "video_path": str(video_path),
        "target_fps": TARGET_FPS,
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

def _annotate_single_skeleton(json_path):
    """
    對單一 skeleton JSON 檔案進行手動標註並儲存。
    """
    import tkinter as tk
    from tkinter import filedialog
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    frames = data['frames']
    total_frames = len(frames)

    # 讀取影片路徑（若 metadata 有 video_path）
    video_path = None
    # 支援舊格式與新格式
    if 'video_metadata' in data and 'video_path' in data['video_metadata']:
        video_path = data['video_metadata']['video_path']
    elif 'video_path' in data:
        video_path = data['video_path']
    # 檢查影片路徑
    if not video_path or not os.path.exists(video_path):
        print("✗ 找不到影片路徑，請手動選擇影片檔案（可直接關閉視窗僅標骨架）")
        root2 = tk.Tk()
        root2.withdraw()
        root2.attributes('-topmost', True)
        video_path = filedialog.askopenfilename(title="選擇對應影片檔案（可略過）", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv")])
        root2.destroy()
        if not video_path or not os.path.exists(video_path):
            video_path = None

    VALID_ACTIONS = ['walk', 'lick', 'scratch', 'shake']
    ACTION_COLORS = {
        'walk':        (0, 255, 0),
        'scratch':     (0, 165, 255),
        'lick':        (255, 255, 0),
        'shake':       (0, 0, 255),
        'unannotated': (40, 40, 40),
    }
    current_action = 'walk'

    print("\n操作說明：")
    print("  - 1/2/3/4: 切換當前行為 (1=walk, 2=lick, 3=scratch, 4=shake)")
    print("  - s: 標記區間起點/終點")
    print("  - a/d: 前/後 skip_n 幀（預設1）")
    print("  - z/x: 減少/增加跳幀數 (skip_n)")
    print("  - SPACE: 播放 / 暫停")
    print("  - t: 跳到指定秒數")
    print("  - q: 離開並儲存")
    print("  [未標記片段訓練時會自動捨棄]\n")

    intervals = []  # list of (start_idx, end_idx, action_name)
    marking = False
    start_idx = None
    cur_idx = 0
    cap = None  # 初始化 cap 避免作用域問題
    playing = False       # 空白鍵切換播放 / 暫停
    skip_n = 1            # a/d/playback 每次跳躍幀數，z/x 調整
    needs_seek = True     # 首幀或手動跳轉後需要 seek
    cap_next_orig_fid = -1  # cap 下一次 read() 會回傳的 original_frame_id
    last_advance_t = 0.0  # 上次推幀的時間戳記（時間驅動播放用）
    render_needed = True   # 只在幀/狀態改變時才重新解碼 + imshow，避免播放時重複 seek 造成抖動
    cached_frame_img = None    # 已解碼的基底幀（含骨架，無 overlay）；只在 cur_idx 改變時重新解碼
    cached_frame_idx = -1      # 對應 cached_frame_img 的 cur_idx
    cached_kp_scale  = 1.0     # cached_frame_img 使用的 kp_scale

    # 只讀一幀取得解析度，保持 cap 開啟供逐幀 seek；不預快取全部幀
    MAX_DISP_W, MAX_DISP_H = 960, 540  # 降低顯示尺寸以加快 seek
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
        frame_shape = (720, 1080, 3)

    # 依影片 FPS 計算播放每幀延遲（限 15~60 fps 之間）
    play_delay_ms = 33
    if cap is not None and cap.isOpened():
        _fps = cap.get(cv2.CAP_PROP_FPS)
        if _fps > 0:
            play_delay_ms = max(16, min(int(1000 / _fps), 66))

    # 計算 skeleton frames 間的 original_frame_id 間隔（供智慧循序讀取使用）
    frame_interval = 1
    if len(frames) >= 2:
        _gaps = [frames[i+1].get('original_frame_id', i+1) - frames[i].get('original_frame_id', i)
                 for i in range(min(10, len(frames) - 1))]
        if _gaps:
            frame_interval = max(1, int(round(sum(_gaps) / len(_gaps))))

    def draw_skeleton(frame_img, keypoints, kp_scale=1.0):
        r = max(2, int(frame_img.shape[1] / 426))  # ~3px at 1280w, scales with resolution
        for kpt in keypoints:
            x, y, conf = int(kpt['x'] * kp_scale), int(kpt['y'] * kp_scale), kpt['conf']
            if conf > 0.2:
                cv2.circle(frame_img, (x, y), r, (0,255,0), -1)
        return frame_img

    while True:
        frame_data = frames[cur_idx]

        # ── 只在 cur_idx 改變時才解碼影片幀（高碼率影片每次 decode 需 80~100ms）──
        # space/1234/s 等只改狀態的按鍵不觸發解碼，主執行緒保持可響應
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
                        cap.grab()  # 快速跳過（不解碼，比 read() 快數倍）
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
            if frame_data['detected'] and frame_data['keypoints']:
                draw_skeleton(show_base, frame_data['keypoints'], kp_scale)
            cached_frame_img = show_base
            cached_frame_idx = cur_idx
            cached_kp_scale  = kp_scale
            render_needed    = True   # 新幀解碼完成，必須重繪 overlay

        # ── 重繪 overlay（狀態文字 + 顏色框）：直接複製快取幀，不重新解碼 ──
        if render_needed:
            show_img = cached_frame_img.copy()

            timestamp = frame_data.get('timestamp', None)
            orig_fid  = frame_data.get('original_frame_id', None)
            n_kpts    = len(frame_data['keypoints']) if frame_data.get('keypoints') else 0
            time_str  = f" ({timestamp:.2f}s)" if timestamp is not None else ""
            orig_str  = f" | VideoFrame: {orig_fid}" if orig_fid is not None else ""
            kpt_str   = f" | Keypoints: {n_kpts}"
            play_str  = "[PLAY]" if playing else "[PAUSE]"
            status    = f"{play_str} Frame {cur_idx+1}/{total_frames}{time_str}{orig_str}{kpt_str} skip:{skip_n}"
            if marking:
                start_time = frames[start_idx].get('timestamp', None)
                cur_time   = frame_data.get('timestamp', None)
                if start_time is not None and cur_time is not None:
                    mark_elapsed = abs(cur_time - start_time)
                    status += f"  [{current_action} marking: from {start_idx+1}, {mark_elapsed:.2f}s]"
                else:
                    status += f"  [{current_action} marking: from {start_idx+1}]"
            else:
                status += f"  [Action: {current_action} | 1=walk 2=lick 3=scratch 4=shake | s=mark]"

            scale      = show_img.shape[1] / 1280.0
            max_width  = show_img.shape[1] - int(20 * scale)
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = round(0.9 * scale, 2)
            thickness  = max(1, int(round(2 * scale)))
            words = status.split(' ')
            lines, cur_line = [], ''
            for word in words:
                test_line = (cur_line + ' ' + word).strip()
                (text_w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                if text_w > max_width and cur_line:
                    lines.append(cur_line)
                    cur_line = word
                else:
                    cur_line = test_line
            if cur_line:
                lines.append(cur_line)
            y0     = max(20, int(30 * scale))
            line_h = max(20, int(35 * scale))
            for i, line in enumerate(lines):
                cv2.putText(show_img, line, (int(10 * scale), y0 + i * line_h),
                            font, font_scale, (0, 255, 255), thickness)

            for (s, e, act) in intervals:
                if s <= cur_idx <= e:
                    color = ACTION_COLORS.get(act, (0, 0, 255))
                    cv2.rectangle(show_img, (0, 0),
                                  (show_img.shape[1], show_img.shape[0]),
                                  color, max(2, int(8 * scale)))
            act_color = ACTION_COLORS.get(current_action, (255, 255, 255))
            cv2.putText(show_img, f"[{current_action.upper()}]",
                        (int(10 * scale), show_img.shape[0] - max(10, int(15 * scale))),
                        cv2.FONT_HERSHEY_SIMPLEX, round(1.2 * scale, 2),
                        act_color, max(2, int(3 * scale)))
            cv2.imshow("Manual Action Labeling", show_img)
            render_needed = False

        # ── 等待按鍵 ─────────────────────────────────────────────────────────
        if playing:
            elapsed      = time.time() - last_advance_t
            remaining_ms = max(15, int((play_delay_ms / 1000.0 - elapsed) * 1000))
            key = cv2.waitKey(remaining_ms) & 0xFF
        else:
            key = cv2.waitKey(0) & 0xFF

        # ── 按鍵處理 ─────────────────────────────────────────────────────────
        if key == ord('q'):
            break
        elif key == ord(' '):
            playing = not playing
            if playing:
                last_advance_t = time.time()
            # 不設 needs_seek：空白鍵只切換播放狀態，不觸發重新解碼
            render_needed = True
        elif key == ord('a'):
            playing    = False
            cur_idx    = max(0, cur_idx - skip_n)
            needs_seek = True   # 往後退必須 seek
            render_needed = True
        elif key == ord('d'):
            playing = False
            cur_idx = min(total_frames - 1, cur_idx + skip_n)
            # 不設 needs_seek：gap 邏輯會自動判斷循序 grab 或 seek
            render_needed = True
        elif key == ord('z'):
            skip_n = max(1, skip_n - 1)
            print(f"  skip_n = {skip_n}")
            render_needed = True
        elif key == ord('x'):
            skip_n += 1
            print(f"  skip_n = {skip_n}")
            render_needed = True
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            current_action = VALID_ACTIONS[key - ord('1')]
            print(f"  當前行為切換為: {current_action}")
            render_needed = True
        elif key == ord('s'):
            if not marking:
                start_idx = cur_idx
                marking   = True
                print(f"  [{current_action}] 標註起點: {start_idx+1}")
            else:
                end_idx = cur_idx
                if end_idx < start_idx:
                    start_idx, end_idx = end_idx, start_idx
                intervals.append((start_idx, end_idx, current_action))
                print(f"  [{current_action}] 區間完成: {start_idx+1} ~ {end_idx+1}")
                marking = False
            render_needed = True
        elif key == ord('t'):
            try:
                input_sec   = input("請輸入要跳轉的秒數（如 12.5）: ").strip()
                sec         = float(input_sec)
                closest_idx = min(range(total_frames),
                                  key=lambda i: abs(frames[i].get('timestamp', 0) - sec))
                cur_idx    = closest_idx
                needs_seek = True
                playing    = False
                render_needed = True
                print(f"  已跳轉到第 {cur_idx+1} 幀（{frames[cur_idx].get('timestamp', 0):.2f}s）")
            except Exception as e:
                print(f"  ✗ 跳轉失敗: {e}")
        elif playing and key == 0xFF:
            # 播放模式：waitKey 超時 → 推進到下一幀（固定+1；skip_n 只影響 a/d）
            if cur_idx < total_frames - 1:
                cur_idx        = min(cur_idx + 1, total_frames - 1)
                last_advance_t = time.time()
                render_needed  = True
            else:
                playing    = False
                needs_seek = True
                render_needed  = True

    cv2.destroyAllWindows()
    if cap:
        cap.release()

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

    # 產生 frame-level label（預設為 unannotated，只有明確標記的區段才有標籤）
    frame_labels = ['unannotated'] * total_frames
    for iv in action_intervals:
        for i in range(iv['start'], iv['end'] + 1):
            if 0 <= i < total_frames:
                frame_labels[i] = iv['action']

    # 空白區間維持 unannotated，訓練時用 label != 'unannotated' 過濾，不自動填 normal
    all_intervals = sorted(action_intervals, key=lambda x: x['start'])

    # 寫回每一幀
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
    連續標記多個 skeleton JSON 檔案；每次標記完後詢問是否繼續，無需重新執行腳本。
    """
    import tkinter as tk
    from tkinter import filedialog

    print("\n=== 連續標記模式 ===")
    print("每次標記完一個檔案後，可選擇繼續下一個。\n")

    while True:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        json_path = filedialog.askopenfilename(
            title="選擇 skeleton JSON 檔案",
            filetypes=[("JSON files", "*.json")]
        )
        root.destroy()
        if not json_path:
            print("✗ 未選擇檔案，結束標記")
            break

        _annotate_single_skeleton(json_path)

        cont = input("\n是否繼續標記下一個檔案？(y/n，預設 y): ").strip().lower()
        if cont == 'n':
            break

    print("\n✓ 標記作業全部完成")


if __name__ == "__main__":
    print("\n==== Cat Skeleton 批次推論/手動標註 ====")
    print("1. 批次推論四個資料夾影片 (YOLO-Pose)")
    print("   → 影片依資料夾名稱 (walk/lick/scratch/shake) 自動標記，可直接訓練")
    print("2. 連續手動標記多個 skeleton JSON")
    print("   → 適用影片含多種行為、需精確逐段標記的情況")
    mode = input("請選擇模式 (1/2): ").strip()
    if mode == '1':
        process_all_videos()
    elif mode == '2':
        manual_action_labeling()
    else:
        print("✗ 未選擇正確模式，程式結束。")