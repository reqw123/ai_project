"""
Cat Behavior Data Collector
4 behaviors: Normal, Licking, Scratching, Head Shaking

Performance Requirements:
- Inference FPS should be 1.2-1.5× video FPS for best quality
- Green FPS indicator = GOOD (≥target)
- Yellow FPS indicator = OK (≥video FPS, may have minor drops)
- Red FPS indicator = LOW (<video FPS, will drop frames)
- If FPS is low: reduce IMGSZ (640→480→320)
"""
from ultralytics import YOLO
import cv2
import numpy as np
import json
from pathlib import Path
from collections import deque
import time

# ==================== Configuration ====================
MODEL_PATH = r"C:\cat_pose\2222.pt"
VIDEO_PATH = r"C:\tinycnn\0_Cat_Yellow_1920x1080.mp4"

IMGSZ = 640
CONF_THRES = 0.50
KP_CONF_THRES = 0.50
TOTAL_KPTS = 17
SEQUENCE_LENGTH = 30  # 30 frames (~1 second)

# 4 behavior classes
BEHAVIOR_CLASSES = {
    '0': 'normal',
    '1': 'licking',
    '2': 'scratching',
    '3': 'head_shaking',
}

# Behavior names (English)
BEHAVIOR_NAMES_ZH = {
    'normal': 'Normal',
    'licking': 'Licking',
    'scratching': 'Scratching',
    'head_shaking': 'Head Shaking'
}

# Output directories
OUTPUT_DIR = Path(r"C:\cat_pose\lstm_model\behavior_data")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_VIDEO_DIR = Path("cat_behavior_videos")  # Save verification videos
OUTPUT_VIDEO_DIR.mkdir(exist_ok=True)

# Colors
BEHAVIOR_COLORS = {
    'normal': (0, 255, 0),
    'licking': (255, 200, 100),
    'scratching': (100, 100, 255),
    'head_shaking': (255, 100, 255)
}

# 四隻脚的颜色（左前、右前、左後、右後）
COLOR_LEFT_FRONT  = (255, 0, 255)   # 洋紅色 (magenta)
COLOR_RIGHT_FRONT = (0, 255, 255)   # 青色 (cyan)
COLOR_LEFT_HIND   = (255, 165, 0)   # 橙色 (orange)
COLOR_RIGHT_HIND  = (0, 255, 0)     # 綠色 (green)

# ==================== Skeleton Links ====================
HEAD_LINKS = [(0,1), (0,2), (1,2)]
BODY_LINKS = [(0,3), (3,4), (4,5)]
FRONT_LIMBS = [(3,6), (6,7), (3,8), (8,9)]
HIND_LIMBS = [(5,10), (10,11), (5,12), (12,13)]
TAIL_LINKS = [(5,14), (14,15), (15,16)]

# ==================== Normalization ====================
def normalize_keypoints(kpts, kpt_conf, conf_thres=0.3):
    """
    Normalize keypoint coordinates (body-scale normalization)
    kpts: (17, 2) keypoint coordinates
    kpt_conf: (17,) keypoint confidence
    Returns: (17, 3) normalized coordinates [x, y, conf]
    """
    kpts = kpts.copy()
    
    # Check if chest(3) and hip(5) are visible
    if kpt_conf[3] < conf_thres or kpt_conf[5] < conf_thres:
        # Use image size normalization
        scale = max(kpts.max(), 1e-6)
        kpts = kpts / scale
    else:
        # Body scale normalization
        chest = kpts[3]
        hip = kpts[5]
        body_scale = np.linalg.norm(chest - hip)
        
        if body_scale < 1e-6:
            scale = max(kpts.max(), 1e-6)
            kpts = kpts / scale
        else:
            # Use chest as origin, body length as unit
            kpts = (kpts - chest) / body_scale
    
    # Merge coordinates and confidence
    normalized = np.column_stack([kpts, kpt_conf])
    return normalized.astype(np.float32)

# ==================== Data Collector ====================
class BehaviorDataCollector:
    def __init__(self):
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)  # Store original frames
        self.current_label = None
        self.recording = False
        
        self.collected_data = {
            'normal': [],
            'licking': [],
            'scratching': [],
            'head_shaking': []
        }
        
        # Auto-detect existing files to avoid overwriting
        self.sequence_counter = self._scan_existing_files()
        
        # Track session totals (not cleared on write)
        self.session_total = {
            'normal': 0,
            'licking': 0,
            'scratching': 0,
            'head_shaking': 0
        }
        
        # Track last saved for undo
        self.last_saved = {
            'label_name': None,
            'video_path': None,
            'data_index': None
        }
    
    def _scan_existing_files(self):
        """Scan existing JSON and video files to get next available number"""
        counter = {
            'normal': 0,
            'licking': 0,
            'scratching': 0,
            'head_shaking': 0
        }
        
        # Scan JSON files
        for label_name in counter.keys():
            json_pattern = OUTPUT_DIR / f"{label_name}_*.json"
            existing_files = list(OUTPUT_DIR.glob(f"{label_name}_*.json"))
            
            if existing_files:
                max_num = 0
                for file in existing_files:
                    try:
                        # Extract number from filename (e.g., normal_3.json -> 3)
                        num = int(file.stem.split('_')[-1])
                        max_num = max(max_num, num)
                    except:
                        pass
                counter[label_name] = max_num
        
        # Scan video files (in case more videos than JSON)
        for label_name in counter.keys():
            existing_videos = list(OUTPUT_VIDEO_DIR.glob(f"{label_name}_*.mp4"))
            
            if existing_videos:
                max_num = 0
                for file in existing_videos:
                    try:
                        # Extract number from filename (e.g., normal_0005.mp4 -> 5)
                        num = int(file.stem.split('_')[-1])
                        max_num = max(max_num, num)
                    except:
                        pass
                # Use the maximum between JSON and video counts
                counter[label_name] = max(counter[label_name], max_num)
        
        # Print detected counts
        if any(counter.values()):
            print("\n📂 Detected existing files:")
            for label_name, count in counter.items():
                if count > 0:
                    print(f"  {label_name}: {count} files (next will be {count+1})")
        
        return counter
    
    def set_label(self, label_key):
        """Set current label"""
        if label_key in BEHAVIOR_CLASSES:
            self.current_label = label_key
            print(f"✅ Label: {BEHAVIOR_CLASSES[label_key].upper()}")
    
    def start_recording(self):
        """Start recording"""
        if self.current_label:
            self.recording = True
            print("🔴 Recording...")
    
    def stop_recording(self):
        """Stop recording"""
        self.recording = False
        print("⏸️  Paused")
    
    def toggle_recording(self):
        """Toggle recording state"""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def add_to_buffer(self, features, frame=None):
        """Add features to buffer"""
        self.sequence_buffer.append(features)
        if frame is not None:
            self.frame_buffer.append(frame.copy())  # Store frame for video export
    
    def save_sequence(self):
        """Save current buffer (JSON + video)"""
        if not self.current_label:
            print("❌ No label selected")
            return False
        
        if len(self.sequence_buffer) < SEQUENCE_LENGTH:
            print(f"❌ Short: {len(self.sequence_buffer)}/{SEQUENCE_LENGTH}")
            return False
        
        label_name = BEHAVIOR_CLASSES[self.current_label]
        sequence = list(self.sequence_buffer)
        
        self.collected_data[label_name].append({
            'sequence': sequence,
            'label': int(self.current_label)
        })
        
        count = len(self.collected_data[label_name])
        self.session_total[label_name] += 1  # Track session total
        
        # Calculate global video number (existing + current session)
        global_video_num = self.sequence_counter[label_name] + count
        
        # Save video clip for verification
        video_saved = False
        video_path = None
        if len(self.frame_buffer) == SEQUENCE_LENGTH:
            try:
                video_path = OUTPUT_VIDEO_DIR / f"{label_name}_{global_video_num:04d}.mp4"
                frame_example = self.frame_buffer[0]
                h, w = frame_example.shape[:2]
                
                # Use H264 codec for better compatibility
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
                
                for frame in self.frame_buffer:
                    out.write(frame)
                
                out.release()
                video_saved = True
            except Exception as e:
                print(f"⚠️ Video save failed: {e}")
        
        # Store last saved info for undo
        self.last_saved = {
            'label_name': label_name,
            'video_path': video_path,
            'data_index': count - 1
        }
        
        status = "💾+🎬" if video_saved else "💾"
        print(f"{status} Saved: {label_name.upper()} (total: {count})")
        
        return True
    
    def undo_last_save(self):
        """Undo the last saved sequence (delete from memory and video file)"""
        label_name = self.last_saved['label_name']
        video_path = self.last_saved['video_path']
        data_index = self.last_saved['data_index']
        
        if label_name is None:
            print("⚠️ Nothing to undo (no recent save)")
            return False
        
        # Check if data still exists in memory
        if data_index is None or data_index >= len(self.collected_data[label_name]):
            print("❌ Cannot undo: Data already written to file (press W)")
            print("   To remove, manually delete the JSON/video files")
            return False
        
        # Remove from collected data
        self.collected_data[label_name].pop(data_index)
        self.session_total[label_name] -= 1
        
        # Delete video file if exists
        if video_path and video_path.exists():
            try:
                video_path.unlink()
                print(f"♻️ Undo: Deleted {label_name.upper()} sequence and video")
            except Exception as e:
                print(f"⚠️ Failed to delete video: {e}")
        else:
            print(f"♻️ Undo: Deleted {label_name.upper()} sequence")
        
        # Clear last saved info
        self.last_saved = {
            'label_name': None,
            'video_path': None,
            'data_index': None
        }
        
        return True
    
    def write_to_file(self):
        """Write all data to file"""
        saved_count = 0
        for label_name, data_list in self.collected_data.items():
            if data_list:
                counter = self.sequence_counter[label_name]
                file_path = OUTPUT_DIR / f"{label_name}_{counter+1}.json"
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, ensure_ascii=False, indent=2)
                
                self.sequence_counter[label_name] += 1
                saved_count += len(data_list)
                print(f"📝 Written: {file_path.name} ({len(data_list)} seqs)")
                
                self.collected_data[label_name] = []  # Clear after writing
        
        if saved_count > 0:
            print(f"✅ Saved {saved_count} sequences")
        else:
            print("⚠️ No data to write")
        
        return saved_count > 0

# ==================== Drawing Functions ====================
def draw_skeleton(frame, kpts, kpt_conf):
    """Draw skeleton (fast version)"""
    # Draw by body parts using different colors for legs
    # Head
    for a, b in HEAD_LINKS:
        if kpt_conf[a] > KP_CONF_THRES and kpt_conf[b] > KP_CONF_THRES:
            pt1 = (int(kpts[a][0]), int(kpts[a][1]))
            pt2 = (int(kpts[b][0]), int(kpts[b][1]))
            cv2.line(frame, pt1, pt2, (255, 255, 0), 2)  # Yellow
    
    # Body
    for a, b in BODY_LINKS:
        if kpt_conf[a] > KP_CONF_THRES and kpt_conf[b] > KP_CONF_THRES:
            pt1 = (int(kpts[a][0]), int(kpts[a][1]))
            pt2 = (int(kpts[b][0]), int(kpts[b][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Green
    
    # Four legs with different colors
    for a, b in [(3,6), (6,7)]:
        if kpt_conf[a] > KP_CONF_THRES and kpt_conf[b] > KP_CONF_THRES:
            pt1 = (int(kpts[a][0]), int(kpts[a][1]))
            pt2 = (int(kpts[b][0]), int(kpts[b][1]))
            cv2.line(frame, pt1, pt2, COLOR_LEFT_FRONT, 2)  # Left front - Magenta
    
    for a, b in [(3,8), (8,9)]:
        if kpt_conf[a] > KP_CONF_THRES and kpt_conf[b] > KP_CONF_THRES:
            pt1 = (int(kpts[a][0]), int(kpts[a][1]))
            pt2 = (int(kpts[b][0]), int(kpts[b][1]))
            cv2.line(frame, pt1, pt2, COLOR_RIGHT_FRONT, 2)  # Right front - Cyan
    
    for a, b in [(5,10), (10,11)]:
        if kpt_conf[a] > KP_CONF_THRES and kpt_conf[b] > KP_CONF_THRES:
            pt1 = (int(kpts[a][0]), int(kpts[a][1]))
            pt2 = (int(kpts[b][0]), int(kpts[b][1]))
            cv2.line(frame, pt1, pt2, COLOR_LEFT_HIND, 2)  # Left hind - Orange
    
    for a, b in [(5,12), (12,13)]:
        if kpt_conf[a] > KP_CONF_THRES and kpt_conf[b] > KP_CONF_THRES:
            pt1 = (int(kpts[a][0]), int(kpts[a][1]))
            pt2 = (int(kpts[b][0]), int(kpts[b][1]))
            cv2.line(frame, pt1, pt2, COLOR_RIGHT_HIND, 2)  # Right hind - Green
    
    # Tail
    for a, b in TAIL_LINKS:
        if kpt_conf[a] > KP_CONF_THRES and kpt_conf[b] > KP_CONF_THRES:
            pt1 = (int(kpts[a][0]), int(kpts[a][1]))
            pt2 = (int(kpts[b][0]), int(kpts[b][1]))
            cv2.line(frame, pt1, pt2, (255, 0, 255), 2)  # Magenta
    
    # Keypoints with leg-specific colors
    for i, (x, y) in enumerate(kpts):
        if kpt_conf[i] > KP_CONF_THRES:
            if i in [6, 7]:  # Left front leg
                color = COLOR_LEFT_FRONT
            elif i in [8, 9]:  # Right front leg
                color = COLOR_RIGHT_FRONT
            elif i in [10, 11]:  # Left hind leg
                color = COLOR_LEFT_HIND
            elif i in [12, 13]:  # Right hind leg
                color = COLOR_RIGHT_HIND
            else:
                color = (0, 0, 255)  # Red for other keypoints
            
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)

def draw_ui(frame, collector):
    """Draw UI (English only, fast)"""
    h, w = frame.shape[:2]
    
    # Top info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 220), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Bottom control panel
    y_start = h - 130
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (10, y_start - 10), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "Cat Behavior Data Collector", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Status
    status = "RECORDING" if collector.recording else "PAUSED"
    color = (0, 255, 0) if collector.recording else (0, 0, 255)
    cv2.putText(frame, f"Status: {status}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Buffer
    cv2.putText(frame, f"Buffer: {len(collector.sequence_buffer)}/{SEQUENCE_LENGTH}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Current label
    if collector.current_label:
        label_name = BEHAVIOR_CLASSES[collector.current_label]
        label_color = BEHAVIOR_COLORS[label_name]
        cv2.putText(frame, f"Label: {label_name}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
    
    # Data statistics
    y_offset = 165
    for i, (key, name) in enumerate(BEHAVIOR_CLASSES.items()):
        count = len(collector.collected_data[name])
        color = BEHAVIOR_COLORS[name]
        text = f"{key}: {name} = {count}"
        cv2.putText(frame, text, (20, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Instructions
    instructions = [
        "Controls:",
        "0:normal  1:licking  2:scratching  3:head_shaking",
        "SPACE:start/stop  S:save  U:undo  W:write  Q:quit"
    ]
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (20, y_start + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

# ==================== Main Program ====================
def main():
    print("="*70)
    print("Cat Behavior Data Collection Tool")
    print("="*70)
    print("\n📝 Behavior Categories:")
    for key, name in BEHAVIOR_CLASSES.items():
        print(f"  Press {key}: {name}")
    
    print("\n💡 Collection Tips:")
    print("  - Collect 100-150 sequences per behavior")
    print("  - Include different angles and lighting")
    print("  - Behaviors should be clear and typical")
    print("  - Keep class counts balanced")
    
    print("\n⌨️  Controls:")
    print("  1. Press 0-3 to select behavior category")
    print("  2. Press SPACE to start/stop recording")
    print("  3. Press S to save current sequence")
    print("  4. Press U to undo last save")
    print("  5. Press W to write to file")
    print("  6. Press Q to quit")
    
    print("\n" + "="*70)
    
    # Load model
    print(f"\n📦 Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Open video
    print(f"📹 Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("❌ Cannot open video")
        return
    
    # Create collector
    collector = BehaviorDataCollector()
    
    # Get video FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = video_fps * 1.3  # Target: 1.3x video FPS
    
    # FPS tracking
    fps = 0
    fps_time = time.time()
    frame_count = 0
    fps_history = []  # Track FPS stability
    
    print(f"\n🎬 Video FPS: {video_fps:.1f}")
    print(f"🎯 Target FPS: {target_fps:.1f}+ (for continuous capture)")
    print("\n✅ Ready!\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        
        # YOLO inference
        results = model(frame, imgsz=IMGSZ, conf=CONF_THRES, verbose=False)
        
        kpts = None
        kpt_conf = None
        features = None
        
        if len(results[0].keypoints) > 0:
            kpts_data = results[0].keypoints.data[0]
            kpts = kpts_data[:, :2].cpu().numpy()
            kpt_conf = kpts_data[:, 2].cpu().numpy()
            
            # Normalize
            features = normalize_keypoints(kpts, kpt_conf).flatten()
        
        # Record data
        if collector.recording and features is not None:
            collector.add_to_buffer(features.tolist(), frame)
        
        # Draw skeleton
        if kpts is not None:
            draw_skeleton(frame, kpts, kpt_conf)
        
        # Draw UI
        frame = draw_ui(frame, collector)
        
        # FPS calculation (every 10 frames for stability)
        if frame_count % 10 == 0:
            current_fps = 10 / (time.time() - fps_time)
            fps = current_fps if fps == 0 else 0.8 * fps + 0.2 * current_fps  # Smooth
            fps_time = time.time()
            
            # Track stability
            if len(fps_history) >= 10:
                fps_history.pop(0)
            fps_history.append(fps)
        
        # FPS display with color coding
        h, w = frame.shape[:2]
        if fps >= target_fps:
            fps_color = (0, 255, 0)  # Green: Good
            status = "GOOD"
        elif fps >= video_fps:
            fps_color = (0, 255, 255)  # Yellow: OK
            status = "OK"
        else:
            fps_color = (0, 0, 255)  # Red: Low
            status = "LOW"
        
        # FPS text with background
        fps_text = f"FPS:{fps:.1f} {status}"
        (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (w - tw - 25, 10), (w - 5, 40), (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (w - tw - 20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        
        # Show target FPS hint
        target_text = f"Target:{target_fps:.0f}+"
        cv2.putText(frame, target_text, (w - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        cv2.imshow("Behavior Data Collector", frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
            collector.set_label(chr(key))
        elif key == ord(' '):
            collector.toggle_recording()
        elif key == ord('s'):
            collector.save_sequence()
        elif key == ord('u'):
            collector.undo_last_save()
        elif key == ord('w'):
            collector.write_to_file()
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # FPS Statistics
    if fps_history:
        avg_fps = sum(fps_history) / len(fps_history)
        fps_std = (sum((x - avg_fps)**2 for x in fps_history) / len(fps_history))**0.5
        print("\n" + "="*70)
        print("📊 FPS Statistics:")
        print(f"  Average: {avg_fps:.1f} fps")
        print(f"  Stability: ±{fps_std:.1f} fps")
        print(f"  Video FPS: {video_fps:.1f} fps")
        print(f"  Target FPS: {target_fps:.1f} fps")
        if avg_fps >= target_fps:
            print("  ✅ Performance: EXCELLENT")
        elif avg_fps >= video_fps:
            print("  ⚠️ Performance: ACCEPTABLE (may drop frames)")
        else:
            print("  ❌ Performance: LOW (will drop frames)")
            print("  💡 Tip: Reduce IMGSZ or simplify visualization")
    
    # Statistics (show session totals)
    print("\n" + "="*70)
    print("📊 Collection Statistics (This Session):")
    total = sum(collector.session_total.values())
    for name in ['normal', 'licking', 'scratching', 'head_shaking']:
        count = collector.session_total[name]
        print(f"  {name}: {count} sequences")
    print(f"Total: {total} sequences")
    
    # Check for unsaved data
    unsaved = sum(len(data) for data in collector.collected_data.values())
    if unsaved > 0:
        print(f"\n⚠️  {unsaved} sequences not yet written to file!")
        response = input("Save unsaved data? (y/n): ")
        if response.lower() == 'y':
            collector.write_to_file()
    
    print("\n✅ Data collection complete!")
    print("="*70)

if __name__ == "__main__":
    main()
