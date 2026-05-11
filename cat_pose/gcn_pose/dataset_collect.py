

import os
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ==================== Configuration ====================
# ==================== Configuration ====================
VIDEO_FOLDERS = [
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\walk",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\lying",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\lick",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\shake"
]
OUTPUT_FOLDER = r"C:\cat_pose\gcn_pose\skeletons/"
MODEL_PATH = r"C:\cat_pose\no_aug.pt"  # You can use yolov8s-pose.pt, yolov8m-pose.pt for better accuracy
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
    setup_directories()
    for f in Path(OUTPUT_FOLDER).glob("*"):
        try:
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                import shutil
                shutil.rmtree(f)
        except Exception as e:
            print(f"[Warning] Failed to delete {f}: {e}")
    print(f"✓ Cleared skeletons folder: {OUTPUT_FOLDER}")

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
        skeleton_data = extract_skeleton_from_video(
            video_path,
            pose_extractor,
            target_fps=TARGET_FPS
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
    
    # Calculate frame interval
    interval = video_fps / target_fps
    return frame_count % int(interval) == 0


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
def extract_skeleton_from_video(video_path, pose_extractor, target_fps=30):
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
    video_path = filedialog.askopenfilename(title="選擇影片檔案", filetypes=[("Video files", ".mp4 .avi .mov .mkv .flv")])
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
    skeleton_data = extract_skeleton_from_video(
        video_path,
        pose_extractor,
        target_fps=TARGET_FPS
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

def manual_action_labeling():
    """
    啟動即選擇 skeleton json，僅手動標註模式，明確顯示 frame 與標註狀態，可多次 s 鍵標註區間。
    """
    import tkinter as tk
    from tkinter import filedialog

    # 選擇 skeleton json 檔案（修正卡住問題）
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    json_path = filedialog.askopenfilename(title="選擇 skeleton JSON 檔案", filetypes=[("JSON files", "*.json")])
    root.destroy()
    if not json_path:
        print("✗ 未選擇檔案")
        return

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
        video_path = filedialog.askopenfilename(title="選擇對應影片檔案（可略過）", filetypes=[("Video files", ".mp4 .avi .mov .mkv .flv")])
        root2.destroy()
        if not video_path or not os.path.exists(video_path):
            video_path = None

    # 選擇標註行為名稱
    action_name = input("請輸入要標註的行為名稱（如 shake）：").strip()
    if not action_name:
        action_name = "target"

    print("\n操作說明：\n  - s: 標記區間起點/終點\n  - a/d: 前/後一幀\n  - t: 跳到指定秒數\n  - q: 離開並儲存\n")

    intervals = []
    marking = False
    start_idx = None
    cur_idx = 0

    # 預先快取所有 skeleton json 需要的 frame
    cached_imgs = []
    frame_shape = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
        video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 依照 skeleton json 的 original_frame_id 順序快取
        needed_fids = [frame_data.get('original_frame_id', idx) for idx, frame_data in enumerate(frames)]
        fid_set = set(needed_fids)
        fid_to_img = {}
        cur_fid = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break
            if cur_fid in fid_set:
                if frame_shape is None:
                    frame_shape = img.shape
                fid_to_img[cur_fid] = img.copy()
            cur_fid += 1
        cap.release()
        # 按 skeleton json 的順序快取
        for idx, frame_data in enumerate(frames):
            fid = frame_data.get('original_frame_id', idx)
            img = fid_to_img.get(fid, None)
            if img is not None:
                cached_imgs.append(img)
            else:
                h, w = frame_shape[:2] if frame_shape is not None else (720, 1080)
                cached_imgs.append(np.ones((h, w, 3), dtype=np.uint8) * 30)
    else:
        h, w = 720, 1080
        for _ in frames:
            cached_imgs.append(np.ones((h, w, 3), dtype=np.uint8) * 30)
        frame_shape = (h, w, 3)

    def draw_skeleton(frame_img, keypoints):
        for kpt in keypoints:
            x, y, conf = int(kpt['x']), int(kpt['y']), kpt['conf']
            if conf > 0.2:
                cv2.circle(frame_img, (x, y), 3, (0,255,0), -1)
        return frame_img

    while True:
        frame_data = frames[cur_idx]
        if cur_idx < len(cached_imgs):
            show_img = cached_imgs[cur_idx].copy()
        else:
            h, w = frame_shape[:2] if frame_shape is not None else (720, 1080)
            show_img = np.ones((h, w, 3), dtype=np.uint8) * 30
        if frame_data['detected'] and frame_data['keypoints']:
            show_img = draw_skeleton(show_img, frame_data['keypoints'])

        # 狀態文字
        timestamp = frame_data.get('timestamp', None)
        orig_fid = frame_data.get('original_frame_id', None)
        n_kpts = len(frame_data['keypoints']) if frame_data.get('keypoints') else 0
        if timestamp is not None:
            time_str = f" ({timestamp:.2f}s)"
        else:
            time_str = ""
        orig_str = f" | VideoFrame: {orig_fid}" if orig_fid is not None else ""
        kpt_str = f" | Keypoints: {n_kpts}"
        status = f"Frame {cur_idx+1}/{total_frames}{time_str}{orig_str}{kpt_str}"
        if marking:
            start_time = frames[start_idx].get('timestamp', None)
            cur_time = frame_data.get('timestamp', None)
            if start_time is not None and cur_time is not None:
                elapsed = abs(cur_time - start_time)
                status += f"  [標註中: 起點 {start_idx+1}, 已過 {elapsed:.2f}s]"
            else:
                status += f"  [標註中: 起點 {start_idx+1}]"
        else:
            status += "  [等待 s 開始]"

        # 狀態列自動換行，避免超出邊界
        # 估算單行最大長度（像素）
        max_width = show_img.shape[1] - 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        words = status.split(' ')
        lines = []
        cur_line = ''
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
        # 寫多行
        y0 = 30
        for i, line in enumerate(lines):
            y = y0 + i*35
            cv2.putText(show_img, line, (10, y), font, font_scale, (0,255,255), thickness)

        # 顯示已標註區間
        for (s, e) in intervals:
            if s <= cur_idx <= e:
                cv2.rectangle(show_img, (0,0), (show_img.shape[1], show_img.shape[0]), (0,0,255), 8)
        cv2.imshow("Manual Action Labeling", show_img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            cur_idx = max(0, cur_idx-1)
        elif key == ord('d'):
            cur_idx = min(total_frames-1, cur_idx+1)
        elif key == ord('s'):
            if not marking:
                start_idx = cur_idx
                marking = True
                print(f"  標註起點: {start_idx+1}")
            else:
                end_idx = cur_idx
                if end_idx < start_idx:
                    start_idx, end_idx = end_idx, start_idx
                intervals.append((start_idx, end_idx))
                print(f"  區間完成: {start_idx+1} ~ {end_idx+1}")
                marking = False
        elif key == ord('t'):
            # 跳到指定秒數
            try:
                input_sec = input("請輸入要跳轉的秒數（如 12.5）: ").strip()
                sec = float(input_sec)
                # 找到最接近該 timestamp 的 frame
                closest_idx = min(range(total_frames), key=lambda i: abs(frames[i].get('timestamp', 0) - sec))
                cur_idx = closest_idx
                print(f"  已跳轉到第 {cur_idx+1} 幀（{frames[cur_idx].get('timestamp', 0):.2f}s）")
            except Exception as e:
                print(f"  ✗ 跳轉失敗: {e}")

    cv2.destroyAllWindows()
    if cap:
        cap.release()

    # 合併重疊區段
    def merge_intervals(intervals):
        intervals = sorted([(min(s,e), max(s,e)) for s,e in intervals])
        merged = []
        for s, e in intervals:
            if not merged or merged[-1][1] < s-1:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        return merged

    merged_intervals = merge_intervals(intervals)

    # 匯出 action_intervals 格式（保留原始區段）
    action_intervals = [{"action": action_name, "start": int(s), "end": int(e)} for s, e in merged_intervals]
    # normal 區間
    normal_intervals = []
    last = 0
    for s, e in merged_intervals:
        if last < s:
            normal_intervals.append({"action": "normal", "start": int(last), "end": int(s-1)})
        last = e+1
    if last < total_frames:
        normal_intervals.append({"action": "normal", "start": int(last), "end": int(total_frames-1)})
    all_intervals = normal_intervals + action_intervals
    all_intervals = sorted(all_intervals, key=lambda x: x['start'])

    # 產生 frame-level label
    frame_labels = ['normal'] * total_frames
    for interval in action_intervals:
        for i in range(interval['start'], interval['end']+1):
            if 0 <= i < total_frames:
                frame_labels[i] = action_name
    # 寫回每一幀
    for i, frame in enumerate(frames):
        frame['label'] = frame_labels[i]

    out_json = data.copy()
    out_json['action_intervals'] = all_intervals
    out_json['frames'] = frames
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 已直接覆蓋原標註檔案: {json_path}\n✓ 已自動合併重疊區段，frames 內每一幀都含 label 欄位")






if __name__ == "__main__":
    print("\n==== Cat Skeleton 批次推論/手動標註 ====")
    print("1. 批次推論四個資料夾影片 (YOLO-Pose)")
    print("2. 單一影片手動標註 (action_intervals)")
    mode = input("請選擇模式 (1/2): ").strip()
    if mode == '1':
        process_all_videos()
    elif mode == '2':
        manual_action_labeling()
    else:
        print("✗ 未選擇正確模式，程式結束。")