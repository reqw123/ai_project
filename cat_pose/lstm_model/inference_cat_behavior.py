"""
猫咪行为实时识别 (YOLO Pose + LSTM)
整合原有的异常检测代码 + LSTM分类
"""
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import time
import threading
from queue import Queue
from collections import deque
from pathlib import Path
import math

from cat_behavior_lstm import CatBehaviorLSTM
from PIL import Image, ImageDraw, ImageFont

# ==================== 基本設定 ====================
YOLO_MODEL_PATH = r"C:\cat_pose\no_aug.pt"
LSTM_MODEL_PATH = r"C:\cat_pose\lstm_model\checkpoints\best_cat_behavior_model.pth"
VIDEO_PATH = r"C:\cat_pose\模型測試影片\cat5.mp4"
font_path = 'C:\\Windows\\Fonts\\msyh.ttc'

IMGSZ = 640
CONF_THRES = 0.50
KP_CONF_THRES = 0.50
TOTAL_KPTS = 17
SEQUENCE_LENGTH = 30

# 行为类别
BEHAVIOR_CLASSES = ['一般', '舔拭', '搔抓', '甩头']
BEHAVIOR_COLORS = {
    0: (0, 255, 0),         # 一般 - 绿色
    1: (255, 200, 100),     # 舔拭 - 浅蓝
    2: (100, 100, 255),     # 搔抓 - 红色
    3: (255, 100, 255)      # 甩头 - 紫色
}

# 異常設計參數（保留原有功能）
EMA_ALPHA = 0.7
ABNORMAL_THRES = 0.2
MIN_BODY_SCALE = 1e-3
STABILITY_K = 4.0

# 顏色定義
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
HIGHLIGHT_COLOR = (0, 255, 255)

COLOR_HEAD = (255, 255, 0)
COLOR_BODY = (0, 255, 0)
COLOR_LIMB = (255, 0, 0)
COLOR_TAIL = (255, 0, 255)
COLOR_KPT = (0, 0, 255)

# 四隻腳的顏色（左前、右前、左後、右後）
COLOR_LEFT_FRONT  = (255, 0, 255)   # 洋紅色 (magenta)
COLOR_RIGHT_FRONT = (0, 255, 255)   # 青色 (cyan)
COLOR_LEFT_HIND   = (255, 165, 0)   # 橙色 (orange)
COLOR_RIGHT_HIND  = (0, 255, 0)     # 綠色 (green)

# 骨架連結
HEAD_LINKS = [(0,1),(0,2),(1,2)]
BODY_LINKS = [(0,3),(3,4),(4,5)]
FRONT_LIMBS = [(3,6),(6,7),(3,8),(8,9)]
HIND_LIMBS = [(5,10),(10,11),(5,12),(12,13)]
TAIL_LINKS = [(5,14),(14,15),(15,16)]

# ==================== 高效影片讀取 ====================
class VideoStream:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.queue = Queue(maxsize=3)
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.queue.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        return self.queue.get() if not self.queue.empty() else None

# ==================== 工具函数 ====================
def normalize_keypoints(kpts, kpt_conf):
    """归一化关键点（与训练时一致）"""
    if kpt_conf[3] > KP_CONF_THRES and kpt_conf[5] > KP_CONF_THRES:
        body_scale = np.linalg.norm(kpts[3] - kpts[5])
        if body_scale < 1e-3:
            body_scale = 1.0
    else:
        body_scale = 1.0
    
    body_center = np.mean(kpts, axis=0)
    norm_kpts = (kpts - body_center) / body_scale
    features = np.concatenate([norm_kpts, kpt_conf.reshape(-1, 1)], axis=1)
    return features.flatten()

def compute_body_scale(kpts):
    return float(np.linalg.norm(kpts[3] - kpts[5]))

# ==================== 视觉化函式 ====================
def draw_vertex(frame, x, y, idx, conf, size=5):
    # 根據關鍵點索引決定顏色
    if idx in [6, 7]:  # 左前腳
        point_color = COLOR_LEFT_FRONT
    elif idx in [8, 9]:  # 右前腳
        point_color = COLOR_RIGHT_FRONT
    elif idx in [10, 11]:  # 左後腳
        point_color = COLOR_LEFT_HIND
    elif idx in [12, 13]:  # 右後腳
        point_color = COLOR_RIGHT_HIND
    else:
        point_color = COLOR_KPT
    
    # 十字（使用對應顏色）
    cv2.line(frame, (x-size, y), (x+size, y), point_color, 2)
    cv2.line(frame, (x, y-size), (x, y+size), point_color, 2)
    
    # index（使用對應顏色）
    cv2.putText(frame, f"{idx}", (x+6, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, point_color, 2)
    
    text = f"{conf:.2f}"
    cv2.putText(frame, text, (x+6, y+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 3)
    cv2.putText(frame, text, (x+6, y+12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,255), 1)

def draw_links(frame, kpts, conf, links, color):
    for a, b in links:
        if conf[a] > KP_CONF_THRES and conf[b] > KP_CONF_THRES:
            cv2.line(frame, tuple(kpts[a].astype(int)),
                    tuple(kpts[b].astype(int)), color, 2)

def draw_skeleton(frame, kpts, conf):
    draw_links(frame, kpts, conf, HEAD_LINKS, COLOR_HEAD)
    draw_links(frame, kpts, conf, BODY_LINKS, COLOR_BODY)
    
    # 四隻腳分別用不同顏色
    draw_links(frame, kpts, conf, [(3,6), (6,7)], COLOR_LEFT_FRONT)   # 左前腳
    draw_links(frame, kpts, conf, [(3,8), (8,9)], COLOR_RIGHT_FRONT)  # 右前腳
    draw_links(frame, kpts, conf, [(5,10), (10,11)], COLOR_LEFT_HIND)   # 左後腳
    draw_links(frame, kpts, conf, [(5,12), (12,13)], COLOR_RIGHT_HIND)  # 右後腳
    
    draw_links(frame, kpts, conf, TAIL_LINKS, COLOR_TAIL)
    
    for i, (x, y) in enumerate(kpts):
        if conf[i] > KP_CONF_THRES:
            draw_vertex(frame, int(x), int(y), i, conf[i])
    
    return frame

def draw_behavior_info(frame, behavior_id, confidence, top_k_probs):
    """绘制行为识别信息（支持中文）"""
    h, w = frame.shape[:2]
    
    # 标题（无背景）
    cv2.putText(frame, "Behavior Recognition", (w - 270, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 当前行为（使用PIL绘制中文）
    behavior_name = BEHAVIOR_CLASSES[behavior_id]
    behavior_color = BEHAVIOR_COLORS[behavior_id]
    
    # 转换为PIL格式以支持中文
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_large = ImageFont.truetype(font_path, 40)
    font_small = ImageFont.truetype(font_path, 16)
    
    # 绘制行为名称（中文）
    draw.text((w - 270, 45), behavior_name, font=font_large, fill=(behavior_color[2], behavior_color[1], behavior_color[0]))
    
    # 转换回OpenCV格式
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    cv2.putText(frame, f"{confidence:.1%}", (w - 270, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Top-K概率 (显示所有4个类别)
    y_offset = 130
    for class_id, prob in top_k_probs:
        bar_width = int(200 * prob)
        color = BEHAVIOR_COLORS[class_id]
        
        # 进度条
        cv2.rectangle(frame, (w - 265, y_offset - 12),
                     (w - 265 + bar_width, y_offset - 2), color, -1)
        
        # 文字（使用PIL绘制中文）
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        text = f"{BEHAVIOR_CLASSES[class_id]}: {prob:.0%}"
        draw.text((w - 265, y_offset - 12), text, font=font_small, fill=(255, 255, 255))
        frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        y_offset += 18

def draw_buffer_status(frame, buffer_size, required_size):
    """绘制序列缓冲区状态"""
    h, w = frame.shape[:2]
    
    bar_w = 200
    bar_h = 20
    x0 = w - 220
    y0 = h - 40
    
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
    
    fill_ratio = min(buffer_size / required_size, 1.0)
    fill_w = int(bar_w * fill_ratio)
    color = (0, 255, 0) if buffer_size >= required_size else (0, 255, 255)
    
    cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), color, -1)
    
    label = f"Buffer: {buffer_size}/{required_size}"
    cv2.putText(frame, label, (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_abnormality_bar(frame, frame_score):
    """异常分数条"""
    h, w = frame.shape[:2]
    bar_w = 220
    bar_h = 16
    x0 = 20
    y0 = h - 55
    
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
    
    if frame_score > 0.4:
        color = (0, 0, 255)
    elif frame_score > 0.2:
        color = (0, 255, 255)
    else:
        color = (0, 255, 0)
    
    fill_w = int(bar_w * min(frame_score, 1.0))
    cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), color, -1)
    
    label = f"Abnormality: {frame_score:.2f}"
    cv2.putText(frame, label, (x0, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

def draw_stability_bar(frame, stability):
    """稳定度条"""
    h, w = frame.shape[:2]
    bar_w = 220
    bar_h = 16
    x0 = 20
    y0 = h - 25
    
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
    
    if stability > 0.7:
        color = (0, 255, 0)
    elif stability > 0.4:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)
    
    fill_w = int(bar_w * stability)
    cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), color, -1)
    
    label = f"Stability: {stability:.2f}"
    cv2.putText(frame, label, (x0, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

# ==================== 主程式 ====================
def main():
    print("="*70)
    print("猫咪行为实时识别 (YOLO Pose + LSTM)")
    print("="*70)
    
    # 加载YOLO
    print("\n加载YOLO Pose模型...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_model.to("cuda")
    
    # 加载LSTM
    print("加载LSTM模型...")
    if not Path(LSTM_MODEL_PATH).exists():
        print(f"❌ 找不到LSTM模型: {LSTM_MODEL_PATH}")
        print("请先运行 train_cat_behavior.py 训练模型")
        return
    
    checkpoint = torch.load(LSTM_MODEL_PATH)
    config = checkpoint['config']
    
    lstm_model = CatBehaviorLSTM(
        input_size=51,
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        num_classes=4,
        dropout=0.0
    )
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_model = lstm_model.to(device)
    
    print(f"✅ 模型加载完成 (设备: {device})")
    
    # 打开视频
    vs = VideoStream(VIDEO_PATH).start()
    
    # 状态变量
    prev_kpts = None
    prev_time = time.time()
    ema_norm_disp = np.zeros(TOTAL_KPTS, dtype=np.float32)
    
    # LSTM序列缓冲
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    behavior_history = deque(maxlen=5)  # 行为平滑
    
    frame_idx = 0
    fps = 0
    
    print("\n开始推理... (按 'q' 退出)\n")
    
    while not vs.stopped:
        frame = vs.read()
        if frame is None:
            continue
        
        frame_idx += 1
        
        # YOLO推理
        result = yolo_model.predict(
            frame, imgsz=IMGSZ, conf=CONF_THRES,
            half=True, verbose=False
        )[0]
        
        # FPS
        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now
        
        # 初始化变量
        active_kpts = 0
        abnormal = False
        frame_score = 0.0
        stability = 1.0
        
        behavior_id = 0  # 默认一般
        confidence = 0.0
        top_k_probs = []
        
        # 绘制bbox
        if result.boxes is not None and len(result.boxes) > 0:
            box = result.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Cat {conf:.2f}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 处理关键点
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            kpts = result.keypoints.xy[0].cpu().numpy()
            kpt_conf = result.keypoints.conf[0].cpu().numpy()
            
            active_kpts = int(np.sum(kpt_conf > KP_CONF_THRES))
            
            # 异常检测（原有功能）
            body_scale = compute_body_scale(kpts)
            if body_scale > MIN_BODY_SCALE and prev_kpts is not None:
                diffs = np.linalg.norm(kpts - prev_kpts, axis=1)
                norm_disp = diffs / body_scale
                ema_norm_disp = EMA_ALPHA * ema_norm_disp + (1.0 - EMA_ALPHA) * norm_disp
                contrib = kpt_conf * ema_norm_disp
                
                valid = kpt_conf > KP_CONF_THRES
                if np.any(valid):
                    frame_score = float(np.max(contrib[valid]))
                    abnormal = frame_score > ABNORMAL_THRES
            
            stability = math.exp(-STABILITY_K * frame_score)
            prev_kpts = kpts.copy()
            
            # LSTM行为识别
            features = normalize_keypoints(kpts, kpt_conf)
            sequence_buffer.append(features)
            
            if len(sequence_buffer) >= SEQUENCE_LENGTH:
                seq_array = np.array(list(sequence_buffer)[-SEQUENCE_LENGTH:])
                seq_tensor = torch.FloatTensor(seq_array).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits, attn_weights = lstm_model(seq_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    
                    behavior_id = probs.argmax().item()
                    confidence = probs[behavior_id].item()
                    
                    # Fixed order: 0,1,2,3 (一般、舔拭、搔抓、甩头)
                    top_k_probs = [(i, probs[i].item()) for i in range(4)]
                
                behavior_history.append(behavior_id)
                
                # 平滑（众数）
                if len(behavior_history) >= 3:
                    from collections import Counter
                    behavior_id = Counter(behavior_history).most_common(1)[0][0]
            
            # 绘制骨架
            frame = draw_skeleton(frame, kpts, kpt_conf)
        
        # 绘制信息
        draw_behavior_info(frame, behavior_id, confidence, top_k_probs)
        draw_buffer_status(frame, len(sequence_buffer), SEQUENCE_LENGTH)
        draw_abnormality_bar(frame, frame_score)
        draw_stability_bar(frame, stability)
        
        # 左上资讯
        info = [
            f"frame: {frame_idx}",
            f"fps: {fps:.1f}",
            f"kpts: {active_kpts}/{TOTAL_KPTS}",
            f"abnormal: {'YES' if abnormal else 'NO'}"
        ]
        
        for i, txt in enumerate(info):
            cv2.putText(frame, txt, (20, 35 + i * 26),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLACK, 2)
        
        cv2.imshow("Cat Behavior Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vs.stopped = True
    cv2.destroyAllWindows()
    print("\n✅ 程序结束")

if __name__ == "__main__":
    main()
