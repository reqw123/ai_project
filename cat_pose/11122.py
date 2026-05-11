"""
貓咪行為資料收集系統
四種行為：正常、舔拭、搔抓、甩頭
"""

from ultralytics import YOLO
import cv2
import numpy as np
import json
from pathlib import Path
from collections import deque
import time

# ===== 中文字型（PIL）=====
from PIL import Image, ImageDraw, ImageFont
font_path = r"C:\Windows\Fonts\msyh.ttc"
FONT_TITLE  = ImageFont.truetype(font_path, 28)
FONT_NORMAL = ImageFont.truetype(font_path, 22)
FONT_SMALL  = ImageFont.truetype(font_path, 18)

# ==================== 基本配置 ====================
MODEL_PATH = r"C:\tinycnn\adam.pt"
VIDEO_PATH = r"C:\tinycnn\0_Cat_Ginger_Cat_1280x720.mp4"
OUTPUT_DIR = Path("behavior_data")
OUTPUT_DIR.mkdir(exist_ok=True)

IMGSZ = 640
CONF_THRES = 0.50
KP_CONF_THRES = 0.50
TOTAL_KPTS = 17
SEQUENCE_LENGTH = 30   # 約 1 秒

# ==================== 行為定義（中文） ====================
BEHAVIOR_CLASSES = {
    '0': '正常',
    '1': '舔拭',
    '2': '搔抓',
    '3': '甩頭',
}

BEHAVIOR_COLORS = {
    '正常': (0, 255, 0),
    '舔拭': (255, 200, 100),
    '搔抓': (100, 100, 255),
    '甩頭': (255, 100, 255)
}

# ==================== Skeleton ====================
HEAD_LINKS = [(0,1), (0,2), (1,2)]
BODY_LINKS = [(0,3), (3,4), (4,5)]
FRONT_LIMBS = [(3,6), (6,7), (3,8), (8,9)]
HIND_LIMBS = [(5,10), (10,11), (5,12), (12,13)]
TAIL_LINKS = [(5,14), (14,15), (15,16)]
ALL_LINKS = HEAD_LINKS + BODY_LINKS + FRONT_LIMBS + HIND_LIMBS + TAIL_LINKS

# ==================== Normalize ====================
def normalize_keypoints(kpts, kpt_conf, conf_thres=0.3):
    kpts = kpts.copy()

    if kpt_conf[3] < conf_thres or kpt_conf[5] < conf_thres:
        scale = max(kpts.max(), 1e-6)
        kpts = kpts / scale
    else:
        chest = kpts[3]
        hip = kpts[5]
        body_scale = np.linalg.norm(chest - hip)
        if body_scale < 1e-6:
            scale = max(kpts.max(), 1e-6)
            kpts = kpts / scale
        else:
            kpts = (kpts - chest) / body_scale

    return np.column_stack([kpts, kpt_conf]).astype(np.float32)

# ==================== Data Collector ====================
class BehaviorDataCollector:
    def __init__(self):
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.current_label = None
        self.recording = False

        self.collected_data = {name: [] for name in BEHAVIOR_CLASSES.values()}
        self.sequence_counter = {name: 0 for name in BEHAVIOR_CLASSES.values()}

    def set_label(self, key):
        if key in BEHAVIOR_CLASSES:
            self.current_label = key
            print(f"✅ 標記姿勢：{BEHAVIOR_CLASSES[key]}")

    def toggle_recording(self):
        self.recording = not self.recording
        print("🔴 錄製中" if self.recording else "⏸️ 暫停")

    def add_to_buffer(self, features):
        self.sequence_buffer.append(features)

    def save_sequence(self):
        if not self.current_label:
            print("❌ 尚未選擇姿勢")
            return
        if len(self.sequence_buffer) < SEQUENCE_LENGTH:
            print("❌ 序列長度不足")
            return

        name = BEHAVIOR_CLASSES[self.current_label]
        self.collected_data[name].append({
            "label": int(self.current_label),
            "sequence": list(self.sequence_buffer)
        })
        print(f"💾 已儲存 {name}（共 {len(self.collected_data[name])} 筆）")

    def write_to_file(self):
        for name, data in self.collected_data.items():
            if not data:
                continue
            idx = self.sequence_counter[name] + 1
            path = OUTPUT_DIR / f"{name}_{idx}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.sequence_counter[name] += 1
            self.collected_data[name] = []
            print(f"📝 寫入 {path}")

# ==================== Drawing ====================
def draw_skeleton(frame, kpts, conf):
    for a, b in ALL_LINKS:
        if conf[a] > KP_CONF_THRES and conf[b] > KP_CONF_THRES:
            cv2.line(frame,
                     (int(kpts[a][0]), int(kpts[a][1])),
                     (int(kpts[b][0]), int(kpts[b][1])),
                     (0,255,0), 2)
    for i, (x, y) in enumerate(kpts):
        if conf[i] > KP_CONF_THRES:
            cv2.circle(frame, (int(x), int(y)), 3, (0,0,255), -1)

def draw_ui(frame, collector, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10,10), (w-10,210), (0,0,0), -1)
    cv2.rectangle(overlay, (10,h-120), (w-10,h-10), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    draw.text((20,20), "貓咪行為資料收集系統", font=FONT_TITLE, fill=(255,255,0))
    draw.text((20,60),
              f"狀態：{'錄製中' if collector.recording else '暫停'}",
              font=FONT_NORMAL,
              fill=(0,255,0) if collector.recording else (255,0,0))

    draw.text((20,90),
              f"序列緩衝：{len(collector.sequence_buffer)}/{SEQUENCE_LENGTH}",
              font=FONT_NORMAL, fill=(255,255,255))

    if collector.current_label:
        name = BEHAVIOR_CLASSES[collector.current_label]
        draw.text((20,120),
                  f"目前姿勢：{name}",
                  font=FONT_NORMAL,
                  fill=BEHAVIOR_COLORS[name])

    y = 150
    for k, name in BEHAVIOR_CLASSES.items():
        draw.text((20,y), f"{k}：{name}", font=FONT_SMALL,
                  fill=BEHAVIOR_COLORS[name])
        y += 22

    draw.text((20,h-110),
              "0~3：選擇姿勢   空白鍵：開始/暫停",
              font=FONT_SMALL, fill=(255,255,255))
    draw.text((20,h-80),
              "S：儲存序列   W：寫入檔案   Q：離開",
              font=FONT_SMALL, fill=(255,255,255))

    draw.text((w-140,20), f"FPS: {fps:.1f}", font=FONT_SMALL, fill=(255,255,255))

    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame

# ==================== Main ====================
def main():
    print("📦 載入模型...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ 無法開啟影片")
        return

    collector = BehaviorDataCollector()

    fps = 0
    t0 = time.time()
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        count += 1
        results = model(frame, imgsz=IMGSZ, conf=CONF_THRES, verbose=False)

        if len(results[0].keypoints) > 0:
            kpt = results[0].keypoints.data[0]
            kpts = kpt[:,:2].cpu().numpy()
            conf = kpt[:,2].cpu().numpy()

            feat = normalize_keypoints(kpts, conf).flatten()
            if collector.recording:
                collector.add_to_buffer(feat.tolist())

            draw_skeleton(frame, kpts, conf)

        if count % 30 == 0:
            fps = 30 / (time.time() - t0)
            t0 = time.time()

        frame = draw_ui(frame, collector, fps)
        cv2.imshow("Cat Behavior Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key in map(ord, ['0','1','2','3']):
            collector.set_label(chr(key))
        elif key == ord(' '):
            collector.toggle_recording()
        elif key == ord('s'):
            collector.save_sequence()
        elif key == ord('w'):
            collector.write_to_file()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
