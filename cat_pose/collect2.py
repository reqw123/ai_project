"""
Cat Behavior Data Collection Tool
4 Classes: Licking, Scratching, Head Shaking, Normal
"""
from ultralytics import YOLO
import cv2
import numpy as np
import json
from pathlib import Path
from collections import deque
import time
from PIL import Image, ImageDraw, ImageFont

# ==================== CONFIGURATION ====================
MODEL_PATH = r"C:\cat_pose\no_aug.pt"
VIDEO_PATH = r"C:\cat_pose\模型測試影片\cat5.mp4"

IMGSZ = 640
CONF_THRES = 0.50
KP_CONF_THRES = 0.50
TOTAL_KPTS = 17
SEQUENCE_LENGTH = 30  # 30-frame sequence (approx. 1 second)

# Behavior Class Definitions
BEHAVIOR_CLASSES = {
    '0': 'normal',          # Normal posture
    '1': 'licking',         # Licking
    '2': 'scratching',      # Scratching
    '3': 'head_shaking',    # Head shaking
}

BEHAVIOR_NAMES_EN = {
    'normal': 'Normal',
    'licking': 'Licking',
    'scratching': 'Scratching',
    'head_shaking': 'Head Shaking'
}

# Color Definitions
BEHAVIOR_COLORS = {
    'normal': (0, 255, 0),          # Green
    'licking': (255, 200, 100),     # Light Blue
    'scratching': (100, 100, 255),  # Red
    'head_shaking': (255, 100, 255) # Purple
}

OUTPUT_DIR = Path("cat_behavior_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Font path (Update if necessary for your OS)
FONT_PATH = r'C:\Windows\Fonts\arial.ttf' 

# ==================== SKELETON LINKS ====================
HEAD_LINKS = [(0,1),(0,2),(1,2)]
BODY_LINKS = [(3,5),(3,4),(4,5),(3,6),(6,7),(3,8)]
FRONT_LIMBS = [(8,9),(10,11),(12,13),(5,10),(5,12)]
HIND_LIMBS = [(5,14),(14,15),(15,16)]
TAIL_LINKS = []

# ==================== NORMALIZATION ====================
def normalize_keypoints(kpts, kpt_conf, conf_thres=0.3):
    """
    Normalize keypoint coordinates (Body-scale normalization)
    kpts: (17, 2) keypoint coords
    kpt_conf: (17,) keypoint confidence
    Returns: (17, 3) normalized coords [x, y, conf]
    """
    kpts = kpts.copy()
    
    # Check if chest (3) and hip (5) are visible
    if kpt_conf[3] < conf_thres or kpt_conf[5] < conf_thres:
        # Fallback: normalize by image size scale
        scale = max(kpts.max(), 1e-6)
        kpts = kpts / scale
    else:
        # Body-scale normalization
        chest = kpts[3]
        hip = kpts[5]
        body_scale = np.linalg.norm(chest - hip)
        
        if body_scale < 1e-6:
            scale = max(kpts.max(), 1e-6)
            kpts = kpts / scale
        else:
            # Origin at chest, units in body length
            kpts = (kpts - chest) / body_scale
    
    # Concatenate coordinates and confidence
    normalized = np.column_stack([kpts, kpt_conf])
    return normalized.astype(np.float32)

# ==================== DATA COLLECTOR ====================
class BehaviorDataCollector:
    def __init__(self):
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.current_label = None
        self.recording = False
        
        self.collected_data = {
            'normal': [],
            'licking': [],
            'scratching': [],
            'head_shaking': []
        }
        
        self.sequence_counter = {
            'normal': 0,
            'licking': 0,
            'scratching': 0,
            'head_shaking': 0
        }
    
    def set_label(self, label_key):
        """Set the current label"""
        if label_key in BEHAVIOR_CLASSES:
            self.current_label = label_key
            print(f"✅ Label switched to: {BEHAVIOR_NAMES_EN[BEHAVIOR_CLASSES[label_key]]}")
    
    def start_recording(self):
        """Start recording"""
        if self.current_label:
            self.recording = True
            print("🔴 Recording Started")
    
    def stop_recording(self):
        """Stop recording"""
        self.recording = False
        print("⏸️ Recording Stopped")
    
    def toggle_recording(self):
        """Toggle recording state"""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def add_to_buffer(self, features):
        """Add features to buffer"""
        self.sequence_buffer.append(features)
    
    def save_sequence(self):
        """Save current buffer sequence"""
        if not self.current_label:
            print("❌ No label selected")
            return False
        
        if len(self.sequence_buffer) < SEQUENCE_LENGTH:
            print(f"❌ Sequence too short ({len(self.sequence_buffer)}/{SEQUENCE_LENGTH})")
            return False
        
        label_name = BEHAVIOR_CLASSES[self.current_label]
        sequence = list(self.sequence_buffer)
        
        self.collected_data[label_name].append({
            'sequence': sequence,
            'label': int(self.current_label)
        })
        
        count = len(self.collected_data[label_name])
        en_name = BEHAVIOR_NAMES_EN[label_name]
        print(f"💾 Sequence Saved: {en_name} (Total: {count})")
        
        return True
    
    def write_to_file(self):
        """Write all collected data to file"""
        saved_count = 0
        for label_name, data_list in self.collected_data.items():
            if data_list:
                counter = self.sequence_counter[label_name]
                file_path = OUTPUT_DIR / f"{label_name}_{counter+1}.json"
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, ensure_ascii=False, indent=2)
                
                self.sequence_counter[label_name] += 1
                en_name = BEHAVIOR_NAMES_EN[label_name]
                print(f"📝 File Written: {file_path.name} ({len(data_list)} sequences, {en_name})")
                
                self.collected_data[label_name] = []
                saved_count += len(data_list)
        
        if saved_count > 0:
            print(f"✅ Successfully wrote {saved_count} sequences")
        else:
            print("⚠️ No data to write")
        
        return saved_count > 0

# ==================== DRAWING FUNCTIONS ====================
def put_text_batch(frame, text_list):
    """Batch draw text using PIL for better quality"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    for text, position, font_size, color in text_list:
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except:
            font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=color[::-1])  # BGR to RGB
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_skeleton(frame, kpts, kpt_conf):
    """Draw skeleton lines and points"""
    all_links = HEAD_LINKS + BODY_LINKS + FRONT_LIMBS + HIND_LIMBS + TAIL_LINKS
    
    for a, b in all_links:
        if kpt_conf[a] > KP_CONF_THRES and kpt_conf[b] > KP_CONF_THRES:
            pt1 = tuple(kpts[a].astype(int))
            pt2 = tuple(kpts[b].astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    for i, (x, y) in enumerate(kpts):
        if kpt_conf[i] > KP_CONF_THRES:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(i), (int(x)+5, int(y)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def draw_ui(frame, collector):
    """Draw UI overlay"""
    h, w = frame.shape[:2]
    
    # Top info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 220), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Bottom command panel
    y_start = h - 130
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (10, y_start - 10), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    
    text_list = []
    
    # Title
    text_list.append(("Cat Behavior Data Collector", (20, 20), 30, (0, 255, 255)))
    
    # Status
    status = "RECORDING" if collector.recording else "IDLE"
    color = (0, 255, 0) if collector.recording else (0, 0, 255)
    text_list.append((f"Status: {status}", (20, 60), 24, color))
    
    # Buffer
    text_list.append((f"Buffer: {len(collector.sequence_buffer)}/{SEQUENCE_LENGTH}",
                      (20, 90), 20, (255, 255, 255)))
    
    # Current Label
    if collector.current_label:
        label_name = BEHAVIOR_CLASSES[collector.current_label]
        en_name = BEHAVIOR_NAMES_EN[label_name]
        label_color = BEHAVIOR_COLORS[label_name]
        text_list.append((f"Label: {en_name} ({label_name})", (20, 120), 24, label_color))
    
    # Stats
    y_offset = 165
    for i, (key, name) in enumerate(BEHAVIOR_CLASSES.items()):
        count = len(collector.collected_data[name])
        en_name = BEHAVIOR_NAMES_EN[name]
        color = BEHAVIOR_COLORS[name]
        text = f"Key {key}: {en_name} = {count}"
        text_list.append((text, (20, y_offset + i * 25), 18, color))
    
    # Bottom Instructions
    instructions = [
        "Hotkeys:",
        "0:Normal  1:Licking  2:Scratching  3:Head Shaking",
        "SPACE: Start/Stop  S: Save Sequence  W: Write to File  Q: Quit"
    ]
    for i, text in enumerate(instructions):
        text_list.append((text, (20, y_start + i * 30), 18, (255, 255, 255)))
    
    return put_text_batch(frame, text_list)

# ==================== MAIN PROGRAM ====================
def main():
    print("="*70)
    print("Cat Behavior Data Collection Tool")
    print("="*70)
    print("\n📝 Behavior Classes:")
    for key, name in BEHAVIOR_CLASSES.items():
        en_name = BEHAVIOR_NAMES_EN[name]
        print(f"  Press {key}: {en_name} ({name})")
    
    print("\n💡 Collection Tips:")
    print("  - Collect 100-150 sequences per behavior")
    print("  - Include different angles and lighting conditions")
    print("  - Ensure behaviors are clear and typical")
    print("  - Maintain class balance")
    
    print("\n⌨️  Operations:")
    print("  1. Press 0-3 to select behavior class")
    print("  2. Press SPACE to Start/Stop recording")
    print("  3. Press S to save the current sequence")
    print("  4. Press W to write data to disk")
    print("  5. Press Q to Quit")
    
    print("\n" + "="*70)
    
    # Load Model
    print(f"\n📦 Loading Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Open Video
    print(f"📹 Opening Video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("❌ Could not open video source")
        return
    
    collector = BehaviorDataCollector()
    
    # FPS calculation
    fps = 0
    fps_time = time.time()
    frame_count = 0
    
    print("\n✅ Ready to collect data...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        
        # YOLO Inference
        results = model(frame, imgsz=IMGSZ, conf=CONF_THRES, verbose=False)
        
        kpts = None
        kpt_conf = None
        features = None
        
        if len(results[0].keypoints) > 0:
            kpts_data = results[0].keypoints.data[0]
            kpts = kpts_data[:, :2].cpu().numpy()
            kpt_conf = kpts_data[:, 2].cpu().numpy()
            
            # Normalization
            features = normalize_keypoints(kpts, kpt_conf).flatten()
        
        # Record Data
        if collector.recording and features is not None:
            collector.add_to_buffer(features.tolist())
        
        # Draw Skeleton
        if kpts is not None:
            draw_skeleton(frame, kpts, kpt_conf)
        
        # Draw UI
        frame = draw_ui(frame, collector)
        
        # FPS Calculation
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_time)
            fps_time = time.time()
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Behavior Data Collector", frame)
        
        # Input Handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
            collector.set_label(chr(key))
        elif key == ord(' '):
            collector.toggle_recording()
        elif key == ord('s'):
            collector.save_sequence()
        elif key == ord('w'):
            collector.write_to_file()
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final Stats
    print("\n" + "="*70)
    print("📊 Collection Statistics:")
    total = 0
    for name, data in collector.collected_data.items():
        en_name = BEHAVIOR_NAMES_EN[name]
        count = len(data)
        total += count
        print(f"  {en_name}: {count} sequences")
    print(f"Total: {total} sequences")
    
    if total > 0:
        response = input("\nSave pending data before exiting? (y/n): ")
        if response.lower() == 'y':
            collector.write_to_file()
    
    print("\n✅ Collection Completed!")
    print("="*70)

if __name__ == "__main__":
    main()