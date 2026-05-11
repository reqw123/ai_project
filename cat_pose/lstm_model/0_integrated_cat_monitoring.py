"""
🐱 猫咪健康监测系统 - 完整整合版
整合功能：
- YOLO Pose 关键点检测
- LSTM 行为分类（一般、舔拭、搔抓、甩头）
- 异常运动检测（EMA算法）
- Flask Web 串流服务
- Node-RED 数据推送
- 行为追踪与健康警报
- CSV 日志记录
"""

from flask import Flask, Response, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import time
import csv
import math
import threading
import requests
import socket
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from cat_behavior_lstm import CatBehaviorLSTM

# ==================== 串流顯示設定 ====================
STREAM_DRAW_OVERLAY = True   # ❌ Node-RED 用（不卡）
DEBUG_DRAW_OVERLAY  = False    # ✅ 本地 / 開發用

# ==================== 配置参数 ====================
# 模型路径
YOLO_MODEL_PATH = r"C:\cat_pose\no_aug.pt"
LSTM_MODEL_PATH = r"C:\cat_pose\lstm_model\checkpoints\best_cat_behavior_model.pth"
VIDEO_PATH = r"C:\cat_pose\模型測試影片\cat5.mp4"

# YOLO 参数
IMGSZ = 640
CONF_THRES = 0.50
KP_CONF_THRES = 0.50
TOTAL_KPTS = 17
SEQUENCE_LENGTH = 30

# 异常检测参数
EMA_ALPHA = 0.7
ABNORMAL_THRES = 0.2
MIN_BODY_SCALE = 1e-3
STABILITY_K = 4.0

# CSV 日志
CSV_PATH = "cat_monitoring_log.csv"

# 字体路径（中文支持）
FONT_PATH = 'C:\\Windows\\Fonts\\msyh.ttc'

# Node-RED 配置
NODE_RED_HOST = "127.0.0.1"
NODE_RED_PORT = 1880
URL_NOTIFY = f"http://{NODE_RED_HOST}:{NODE_RED_PORT}/python_online"
URL_RESULT = f"http://{NODE_RED_HOST}:{NODE_RED_PORT}/yolo_result"

# ==================== 颜色定义 ====================
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
HIGHLIGHT_COLOR = (0, 255, 255)

COLOR_HEAD = (255, 255, 0)
COLOR_BODY = (0, 255, 0)
COLOR_LIMB = (255, 0, 0)
COLOR_TAIL = (255, 0, 255)
COLOR_KPT = (0, 0, 255)

# ==================== 骨架连结 ====================
HEAD_LINKS = [(0,1), (0,2), (1,2)]
BODY_LINKS = [(0,3), (3,4), (4,5)]
FRONT_LIMBS = [(3,6), (6,7), (3,8), (8,9)]
HIND_LIMBS = [(5,10), (10,11), (5,12), (12,13)]
TAIL_LINKS = [(5,14), (14,15), (15,16)]
ALL_SKELETON = HEAD_LINKS + BODY_LINKS + FRONT_LIMBS + HIND_LIMBS + TAIL_LINKS

# ==================== LSTM 行为类别 ====================
BEHAVIOR_CLASSES = ['一般', '舔拭', '搔抓', '甩头']
BEHAVIOR_COLORS = {
    0: (0, 255, 0),         # 一般 - 绿色
    1: (255, 200, 100),     # 舔拭 - 浅蓝
    2: (100, 100, 255),     # 搔抓 - 红色
    3: (255, 100, 255)      # 甩头 - 紫色
}

BEHAVIOR_TEXT_MAP = {
    0: "一般活动",
    1: "舔拭理毛",
    2: "搔抓动作",
    3: "甩头动作"
}

BEHAVIOR_EMOJI_MAP = {
    0: "🚶",
    1: "🧼",
    2: "🐾",
    3: "甩頭"
}

# ==================== 工具函数 ====================
def get_ip():
    """获取本机IP"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

LOCAL_IP = get_ip()

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
    """计算身体比例"""
    return float(np.linalg.norm(kpts[3] - kpts[5]))

# ==================== 行为追踪系统 ====================
class ImprovedBehaviorTracker:
    def __init__(self):
        # 行为时间记录（秒数）
        self.behavior_time = {
            "normal": 0.0,
            "groom": 0.0,
            "scratch": 0.0,
            "shake": 0.0
        }
        
        # 行为次数记录
        self.behavior_count = {
            "normal": 0,
            "groom": 0,
            "scratch": 0,
            "shake": 0
        }
        
        # 行为记录历史
        self.behavior_history = deque(maxlen=100)
        
        # 当前行为状态
        self.current_behavior = None
        self.behavior_start_time = time.time()
        
        # 每日重置
        self.last_reset = datetime.now().date()
        
        # 活动力窗口
        self.activity_window = deque(maxlen=60)
        
    def check_daily_reset(self):
        """每日重置统计"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.behavior_time = {k: 0.0 for k in self.behavior_time}
            self.behavior_count = {k: 0 for k in self.behavior_count}
            self.last_reset = today
    
    def update(self, behavior_id, activity_value):
        """更新行为记录"""
        self.check_daily_reset()
        
        # 映射LSTM类别到追踪器类别
        behavior_map = {
            0: "normal",
            1: "groom",
            2: "scratch",
            3: "shake"
        }
        behavior = behavior_map.get(behavior_id, "normal")
        
        now = time.time()
        duration = now - self.behavior_start_time
        
        # 行为转换检测
        if behavior != self.current_behavior:
            # 记录上一个行为
            if self.current_behavior is not None:
                # 累积时间
                if self.current_behavior in self.behavior_time:
                    self.behavior_time[self.current_behavior] += duration
                
                # 次数+1
                if self.current_behavior in self.behavior_count:
                    self.behavior_count[self.current_behavior] += 1
                
                # 记录到历史
                record = {
                    "behavior": self.current_behavior,
                    "behavior_id": behavior_id,
                    "timestamp": datetime.now(),
                    "duration": round(duration, 1),
                    "activity": activity_value
                }
                self.behavior_history.append(record)
            
            # 更新状态
            self.current_behavior = behavior
            self.behavior_start_time = now
        
        # 活动力记录
        self.activity_window.append({
            "time": now,
            "activity": activity_value,
            "weight": duration if duration > 0 else 0.5
        })
    
    def get_activity_score(self):
        """计算活动力分数（时间加权平均）"""
        if len(self.activity_window) == 0:
            return 50
        
        now = time.time()
        recent = [r for r in self.activity_window if (now - r["time"]) < 1]
        
        if len(recent) == 0:
            return 50
        
        total_weight = sum(r["weight"] for r in recent)
        weighted_sum = sum(r["activity"] * r["weight"] for r in recent)
        
        return round(weighted_sum / total_weight) if total_weight > 0 else 50
    
    def get_alerts(self):
        """生成健康警示"""
        alerts = []
        
        # 警示1: 搔抓时间过长
        scratch_time = self.behavior_time.get("scratch", 0)
        scratch_count = self.behavior_count.get("scratch", 0)
        
        if scratch_time > 60:
            alerts.append({
                "level": "high",
                "icon": "🚨",
                "title": "搔抓时间异常",
                "message": f"今日累积搔抓 {scratch_time:.1f} 秒（{scratch_count}次）",
                "suggestion": "请检查皮膚是否有红肿、掉毛、伤口",
                "action": "联络兽医"
            })
        elif scratch_count >= 5:
            alerts.append({
                "level": "medium",
                "icon": "⚠️",
                "title": "搔抓频率偏高",
                "message": f"今日已搔抓 {scratch_count} 次（累积{scratch_time:.1f}秒）",
                "suggestion": "建议观察是否有皮肤不适症状",
                "action": "持续观察"
            })
        
        # 警示2: 理毛时间过长
        groom_time = self.behavior_time.get("groom", 0)
        groom_count = self.behavior_count.get("groom", 0)
        
        if groom_time > 180:
            alerts.append({
                "level": "medium",
                "icon": "🧼",
                "title": "理毛时间较长",
                "message": f"今日理毛 {groom_time:.1f} 秒（{groom_count}次）",
                "suggestion": "可能有压力或皮肤问题",
                "action": "观察精神状态"
            })
        
        # 警示3: 甩头频繁
        shake_count = self.behavior_count.get("shake", 0)
        if shake_count >= 10:
            alerts.append({
                "level": "medium",
                "icon": "🔄",
                "title": "甩头动作频繁",
                "message": f"今日甩头 {shake_count} 次",
                "suggestion": "可能有耳部不适",
                "action": "检查耳朵"
            })
        
        return alerts
    
    def get_behavior_log(self, limit=10):
        """获取行为记录（格式化）"""
        logs = []
        for record in list(self.behavior_history)[-limit:]:
            behavior_id = record.get("behavior_id", 0)
            logs.append({
                "time": record["timestamp"].strftime("%H:%M"),
                "behavior": BEHAVIOR_CLASSES[behavior_id],
                "emoji": BEHAVIOR_EMOJI_MAP.get(behavior_id, "❓"),
                "duration": int(record["duration"]),
                "alert": behavior_id in [1, 2, 3]  # 舔拭、搔抓、甩头
            })
        return list(reversed(logs))
    
    def get_today_stats(self):
        return {
        "scratch": self.behavior_count.get("scratch", 0),
        "scratch_time": round(self.behavior_time.get("scratch", 0), 1),

        "groom": self.behavior_count.get("groom", 0),
        "groom_time": round(self.behavior_time.get("groom", 0), 1),

        "shake": self.behavior_count.get("shake", 0),
        "shake_time": round(self.behavior_time.get("shake", 0), 1),

        "normal": self.behavior_count.get("normal", 0),
        "normal_time": round(self.behavior_time.get("normal", 0), 1),

        "active_time": round(
            self.behavior_time.get("normal", 0) +
            self.behavior_time.get("scratch", 0) +
            self.behavior_time.get("groom", 0), 1
        ),

        "rest_time": round(
            self.behavior_time.get("groom", 0), 1
        )
    }


# 初始化追踪器
tracker = ImprovedBehaviorTracker()

# ==================== 可视化函数 ====================
def draw_links_fast(frame, kpts, conf, links, color):
    """快速绘制骨架连线"""
    for a, b in links:
        if conf[a] > KP_CONF_THRES and conf[b] > KP_CONF_THRES:
            cv2.line(frame,
                     (int(kpts[a][0]), int(kpts[a][1])),
                     (int(kpts[b][0]), int(kpts[b][1])),
                     color, 2)

def draw_skeleton_fast(frame, kpts, conf):
    """优化的骨架绘制"""
    draw_links_fast(frame, kpts, conf, HEAD_LINKS, COLOR_HEAD)
    draw_links_fast(frame, kpts, conf, BODY_LINKS, COLOR_BODY)
    draw_links_fast(frame, kpts, conf, FRONT_LIMBS, COLOR_LIMB)
    draw_links_fast(frame, kpts, conf, HIND_LIMBS, COLOR_LIMB)
    draw_links_fast(frame, kpts, conf, TAIL_LINKS, COLOR_TAIL)
    
    # 绘制关键点
    for i, (x, y) in enumerate(kpts):
        if conf[i] > KP_CONF_THRES:
            cv2.circle(frame, (int(x), int(y)), 3, COLOR_KPT, -1)

def draw_status_light(frame, abnormal):
    """状态指示灯（右上）"""
    h, w = frame.shape[:2]
    color = RED if abnormal else GREEN
    text = "ABNORMAL" if abnormal else "NORMAL"
    cx, cy = w - 80, 25
    cv2.circle(frame, (cx, cy), 8, color, -1)
    cv2.putText(frame, text, (cx - 70, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_trigger_marker(frame, kpts, idx, score):
    """高亮触发关键点"""
    if idx is None:
        return
    
    x, y = int(kpts[idx][0]), int(kpts[idx][1])
    cv2.circle(frame, (x, y), 12, YELLOW, 2)
    cv2.circle(frame, (x, y), 4, YELLOW, -1)
    
    label = f"KPT{idx} {score:.3f}"
    cv2.putText(frame, label, (x + 15, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2)

def draw_progress_bar(frame, value, y_pos, label, max_val=1.0):
    """通用进度条（左下）"""
    h, w = frame.shape[:2]
    bar_w = 200
    bar_h = 14
    x0 = 15
    y0 = h - y_pos
    
    # 背景
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (40, 40, 40), -1)
    
    # 颜色
    if value > 0.6:
        color = RED
    elif value > 0.3:
        color = YELLOW
    else:
        color = GREEN
    
    fill_w = int(bar_w * min(value / max_val, 1.0))
    cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), color, -1)
    
    # 标签
    cv2.putText(frame, f"{label}: {value:.2f}", (x0, y0 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

def draw_behavior_info(frame, behavior_id, confidence, top_k_probs):
    """绘制LSTM行为识别信息（支持中文）"""
    h, w = frame.shape[:2]
    
    # 标题
    cv2.putText(frame, "Behavior Recognition", (w - 270, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, HIGHLIGHT_COLOR, 2)
    
    # 当前行为（使用PIL绘制中文）
    behavior_name = BEHAVIOR_CLASSES[behavior_id]
    behavior_color = BEHAVIOR_COLORS[behavior_id]
    
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font_large = ImageFont.truetype(FONT_PATH, 40)
        font_small = ImageFont.truetype(FONT_PATH, 16)
        
        # 绘制行为名称
        draw.text((w - 270, 45), behavior_name, font=font_large, 
                 fill=(behavior_color[2], behavior_color[1], behavior_color[0]))
        
        # 转换回OpenCV格式
        frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        # 降级方案：使用英文
        cv2.putText(frame, behavior_name, (w - 270, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, behavior_color, 2)
    
    # 置信度
    cv2.putText(frame, f"{confidence:.1%}", (w - 270, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    
    # Top-K概率
    y_offset = 130
    for class_id, prob in top_k_probs:
        bar_width = int(200 * prob)
        color = BEHAVIOR_COLORS[class_id]
        
        # 进度条
        cv2.rectangle(frame, (w - 265, y_offset - 12),
                     (w - 265 + bar_width, y_offset - 2), color, -1)
        
        # 文字
        text = f"{BEHAVIOR_CLASSES[class_id]}: {prob:.0%}"
        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text((w - 265, y_offset - 12), text, font=font_small, fill=(255, 255, 255))
            frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except:
            cv2.putText(frame, text, (w - 265, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
        
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
    color = GREEN if buffer_size >= required_size else YELLOW
    
    cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), color, -1)
    
    label = f"Buffer: {buffer_size}/{required_size}"
    cv2.putText(frame, label, (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

# ==================== Node-RED 通讯 ====================
def notify_nodered_online():
    """通知 Node-RED Python 已上线"""
    try:
        data = {
            "status": "online",
            "ip": LOCAL_IP,
            "port": 5000,
            "stream_url": f"http://{LOCAL_IP}:5000/stream",
            "status_url": f"http://{LOCAL_IP}:5000/status",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "v3.0-integrated",
            "model": "YOLO-Pose + LSTM",
            "message": "😺 猫咪健康监测系统已启动（整合版）"
        }
        response = requests.post(URL_NOTIFY, json=data, timeout=2)
        if response.status_code == 200:
            print(f"✅ 已通知 Node-RED，Python 上线于 {LOCAL_IP}:5000")
        else:
            print(f"⚠️ Node-RED 回应状态码: {response.status_code}")
    except Exception as e:
        print(f"⚠️ 通知 Node-RED 失败: {e}")
        print("   (系统仍可正常运行，仅自动IP显示功能受影响)")

def send_owner_friendly_data(behavior_id, activity_value):
    """发送飼主友善数据给 Node-RED"""
    def _post():
        try:
            # 更新追踪器
            tracker.update(behavior_id, activity_value)
            
            # 获取今日统计
            today_stats = tracker.get_today_stats()
            
            # 组装数据
            data = {
                "current": {
                    "text": BEHAVIOR_CLASSES[behavior_id],
                    "emoji": BEHAVIOR_EMOJI_MAP.get(behavior_id, "❓"),
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                },
                "activity_score": tracker.get_activity_score(),
                "today_stats": today_stats,
                "behavior_log": tracker.get_behavior_log(10),
                "alerts": tracker.get_alerts(),
                "system": {
                    "ip": LOCAL_IP,
                    "model": "YOLO-Pose + LSTM",
                    "version": "v3.0-integrated"
                }
            }
            
            requests.post(URL_RESULT, json=data, timeout=0.3)
        except Exception as e:
            pass  # 静默失败
    
    threading.Thread(target=_post, daemon=True).start()

# ==================== 主处理逻辑 ====================
class CatMonitoringSystem:
    def __init__(self, width=None, height=None):
        print("="*70)
        print("🐱 猫咪健康监测系统 - 完整整合版")
        print("="*70)
        
        # 加载YOLO模型
        print("\n🔄 加载 YOLO Pose 模型...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.yolo_model.to("cuda")
        print("✅ YOLO 模型加载成功")
        
        # 加载LSTM模型
        print("🔄 加载 LSTM 行为分类模型...")
        if not Path(LSTM_MODEL_PATH).exists():
            print(f"❌ 找不到LSTM模型: {LSTM_MODEL_PATH}")
            print("   将使用基础检测模式（无行为分类）")
            self.lstm_model = None
        else:
            checkpoint = torch.load(LSTM_MODEL_PATH)
            config = checkpoint['config']
            
            self.lstm_model = CatBehaviorLSTM(
                input_size=51,
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                num_classes=4,
                dropout=0.0
            )
            self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
            self.lstm_model.eval()
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.lstm_model = self.lstm_model.to(self.device)
            print(f"✅ LSTM 模型加载成功 (设备: {self.device})")
        
        # 视频源
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        if not self.cap.isOpened():
            print("❌ 无法打开视频文件")
            exit()
        # 設定解析度（如有指定）
        if width is not None and height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # 状态变量
        self.prev_kpts = None
        self.prev_time = time.time()
        self.ema_norm_disp = np.zeros(TOTAL_KPTS, dtype=np.float32)
        self.sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.behavior_history = deque(maxlen=5)
        self.frame_idx = 0
        self.fps_display = 0
        self.last_send_time = 0
        
        # CSV日志
        self.csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame", "timestamp", "behavior", "confidence", 
            "abnormal", "trigger_kpt", "frame_score", "stability"
        ])
        
        # 通知Node-RED
        notify_nodered_online()
        
        print("\n✅ 系统初始化完成")
        print(f"📺 视频源: {VIDEO_PATH}")
        print(f"📝 日志文件: {CSV_PATH}")
        print("="*70)
    
    def process_frame(self, frame):
        """处理单帧，回傳(繪製後, 原始)"""
        self.frame_idx += 1
        frame_raw = frame.copy()  # 保留原始

        # YOLO推理
        result = self.yolo_model.predict(
            frame,
            imgsz=IMGSZ,
            conf=CONF_THRES,
            half=True,
            verbose=False,
            device=0
        )[0]

        # FPS计算
        now = time.time()
        fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now

        if self.frame_idx % 10 == 0:
            self.fps_display = fps

        # 初始化变量
        active_kpts = 0
        abnormal = False
        trigger_idx = None
        trigger_score = 0.0
        frame_score = 0.0
        stability = 1.0
        behavior_id = 0
        confidence = 0.0
        top_k_probs = []
        activity_value = 0

        # 绘制bbox
        if result.boxes is not None and len(result.boxes) > 0:
            box = result.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            if STREAM_DRAW_OVERLAY or DEBUG_DRAW_OVERLAY:
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                cv2.putText(frame, f'Cat {conf:.2f}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

        # 处理关键点
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            kpts = result.keypoints.xy[0].cpu().numpy()
            kpt_conf = result.keypoints.conf[0].cpu().numpy()

            active_kpts = int(np.sum(kpt_conf > KP_CONF_THRES))

            # ===== 异常检测 =====
            body_scale = compute_body_scale(kpts)
            if body_scale > MIN_BODY_SCALE and self.prev_kpts is not None:
                diffs = np.linalg.norm(kpts - self.prev_kpts, axis=1)
                norm_disp = diffs / body_scale

                self.ema_norm_disp = EMA_ALPHA * self.ema_norm_disp + (1.0 - EMA_ALPHA) * norm_disp
                contrib = kpt_conf * self.ema_norm_disp

                valid = kpt_conf > KP_CONF_THRES
                if np.any(valid):
                    valid_indices = np.where(valid)[0]
                    best_local = valid_indices[np.argmax(contrib[valid])]
                    trigger_idx = int(best_local)
                    trigger_score = float(contrib[trigger_idx])
                    frame_score = float(np.max(contrib[valid]))

                    abnormal = frame_score > ABNORMAL_THRES
                    stability = math.exp(-STABILITY_K * frame_score)

                # 计算活动力
                activity_value = float(np.sum(norm_disp[valid]))* 10 if np.any(valid) else 0

            self.prev_kpts = kpts.copy()

            # ===== LSTM 行为识别 =====
            if self.lstm_model is not None:
                features = normalize_keypoints(kpts, kpt_conf)
                self.sequence_buffer.append(features)

                if len(self.sequence_buffer) >= SEQUENCE_LENGTH:
                    seq_array = np.array(list(self.sequence_buffer)[-SEQUENCE_LENGTH:])
                    seq_tensor = torch.FloatTensor(seq_array).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        logits, _ = self.lstm_model(seq_tensor)
                        probs = torch.softmax(logits, dim=1)[0]

                        behavior_id = probs.argmax().item()
                        confidence = probs[behavior_id].item()

                        # Top-K概率
                        top_k_probs = [(i, probs[i].item()) for i in range(4)]

                    self.behavior_history.append(behavior_id)

                    # 平滑（众数）
                    if len(self.behavior_history) >= 3:
                        from collections import Counter
                        behavior_id = Counter(self.behavior_history).most_common(1)[0][0]

            # ===== 繪圖區塊（可開關） =====
            if STREAM_DRAW_OVERLAY or DEBUG_DRAW_OVERLAY:
                draw_skeleton_fast(frame, kpts, kpt_conf)
                draw_status_light(frame, abnormal)

                if trigger_idx is not None:
                    draw_trigger_marker(frame, kpts, trigger_idx, trigger_score)

                if self.lstm_model is not None:
                    draw_behavior_info(frame, behavior_id, confidence, top_k_probs)
                    draw_buffer_status(frame, len(self.sequence_buffer), SEQUENCE_LENGTH)

                draw_progress_bar(frame, frame_score, 50, "Abnormal")
                draw_progress_bar(frame, stability, 25, "Stability")

            # ===== CSV记录 =====
            if abnormal:
                self.csv_writer.writerow([
                    self.frame_idx,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    BEHAVIOR_CLASSES[behavior_id],
                    f"{confidence:.4f}",
                    "YES",
                    trigger_idx if trigger_idx is not None else "",
                    f"{frame_score:.6f}",
                    f"{stability:.6f}"
                ])

        # ===== 左上信息 =====
        info = [
            f"Frame: {self.frame_idx}",
            f"FPS: {self.fps_display:.1f}",
            f"Kpts: {active_kpts}/{TOTAL_KPTS}",
            f"Abnormal: {'YES' if abnormal else 'NO'}"
        ]

        if STREAM_DRAW_OVERLAY or DEBUG_DRAW_OVERLAY:
            for i, txt in enumerate(info):
                cv2.putText(frame, txt, (15, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 3)
                cv2.putText(frame, txt, (15, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        # ===== 发送数据给 Node-RED =====
        now = time.time()
        if now - self.last_send_time >= 0.5:
            send_owner_friendly_data(behavior_id, activity_value)
            self.last_send_time = now

        return frame, frame_raw
    
    def run_display(self):
        """运行显示模式（本地窗口）"""
        print("\n▶️ 开始处理... (按 'q' 退出)\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame = self.process_frame(frame)
            
            cv2.imshow("Cat Health Monitoring System", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def generate_frames(self):
        """生成Flask串流帧"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame = self.process_frame(frame)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                   buffer.tobytes() + b'\r\n')
    
    def cleanup(self):
        """清理资源"""
        self.cap.release()
        self.csv_file.close()
        cv2.destroyAllWindows()
        print("\n✅ 系统已关闭")
        print(f"📝 日志已保存: {CSV_PATH}")

# ==================== Flask Web服务器 ====================

app = Flask(__name__)
monitoring_system = None

# ====== 單一影像讀取執行緒與 frame buffer ======
import threading

class SharedFrameStreamer:
    def __init__(self, monitoring_system):
        self.monitoring_system = monitoring_system
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def _update_frame(self):
        while self.running:
            ret, frame = self.monitoring_system.cap.read()
            if not ret:
                self.monitoring_system.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            processed_frame, raw_frame = self.monitoring_system.process_frame(frame)
            with self.lock:
                if STREAM_DRAW_OVERLAY:
                    self.latest_frame = processed_frame.copy()
                else:
                    self.latest_frame = raw_frame.copy()

    def get_jpeg(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            _, buffer = cv2.imencode('.jpg', self.latest_frame, encode_param)
            return buffer.tobytes()

    def stop(self):
        self.running = False
        self.thread.join()

shared_streamer = None

@app.route('/stream')
def stream():
    """视频串流端点 (多 client 共用單一 frame buffer)"""
    global monitoring_system, shared_streamer
    if monitoring_system is None:
        monitoring_system = CatMonitoringSystem(width=width, height=height)
    if shared_streamer is None:
        shared_streamer = SharedFrameStreamer(monitoring_system)

    def mjpeg_stream():
        while True:
            jpeg = shared_streamer.get_jpeg()
            if jpeg is not None:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
                time.sleep(0.03)   # 約 30 FPS
            else:
                time.sleep(0.01)
    return Response(mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """状态API"""
    stats = tracker.get_today_stats()
    return jsonify({
        "status": "running",
        "port": 5000,
        "ip": LOCAL_IP,
        "activity_score": tracker.get_activity_score(),
        "today_stats": stats,
        "alerts": tracker.get_alerts(),
        "alerts_count": len(tracker.get_alerts()),
        "version": "v3.0-integrated"
    })

@app.route('/')
def index():
    """简单的首页"""
    return f"""
    <html>
    <head><title>猫咪健康监测系统</title></head>
    <body style="font-family: Arial; background: #1a1a1a; color: white; padding: 20px;">
        <h1>😺 猫咪健康监测系统</h1>
        <p>版本: v3.0-integrated (YOLO Pose + LSTM)</p>
        <hr>
        <h2>📺 实时串流</h2>
        <img src="/stream" width="960" style="border: 2px solid #4CAF50; border-radius: 8px;">
        <hr>
        <h2>📊 快速连结</h2>
        <ul>
            <li><a href="/stream" style="color: #4CAF50;">视频串流</a></li>
            <li><a href="/status" style="color: #4CAF50;">状态API (JSON)</a></li>
        </ul>
        <hr>
        <p style="color: #888;">系统IP: {LOCAL_IP}:5000</p>
    </body>
    </html>
    """

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    print("="*70)
    print("🐱 猫咪健康监测系统启动器 (僅Flask Web服務)")
    print("="*70)
    # 如需自訂解析度，請修改下方兩行
    # 例如: width, height = 1280, 720
    width, height = 640, 640  # 預設為None，使用原始影片解析度
    monitoring_system = CatMonitoringSystem(width=width, height=height)
    print(f"\n📺 Web服务器启动于 http://{LOCAL_IP}:5000")
    print(f"📊 串流网址: http://{LOCAL_IP}:5000/stream")
    print(f"📈 状态API: http://{LOCAL_IP}:5000/status")
    print("="*70)
    app.run(host="0.0.0.0", port=5000, threaded=True)
