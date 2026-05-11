from ultralytics import YOLO
import cv2
import numpy as np
import time
import threading
from queue import Queue
from collections import deque
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# ==================== 基本設定 ====================
MODEL_PATH = r"C:\tinycnn\640_best2.pt"
VIDEO_PATH = r"C:\tinycnn\0_Cat_Jumping_1280x720.mp4"
IMGSZ = 640
if IMGSZ != 640:
    raise ValueError("❌ 本程式已鎖定尺寸 640x640，請勿修改 IMGSZ！")

CONF_THRES = 0.50
KP_CONF_THRES = 0.30
DEVIATION_THRES = 0.60
TOTAL_KPTS = 17

# ==================== DTW 設定 ====================
NORMAL_PATTERN_PATH = r"C:\tinycnn\normal_pattern.npy"
SEQ_MAX_LEN = 30
DTW_THRES = 800

# ==================== 顏色 ====================
BLACK=(0,0,0); GREEN=(0,255,0); RED=(0,0,255); BLUE=(255,0,0)
COLOR_HEAD=(255,255,0); COLOR_BODY=(0,255,0)
COLOR_LIMB=(0,150,255); COLOR_TAIL=(255,0,255)
COLOR_KPT=(0,0,255); COLOR_ABN=(0,255,255)  # ⚠ 異常關節標記（亮黃）

# ==================== 骨架連結 ====================
HEAD_LINKS=[(0,1),(0,2),(1,2)]
BODY_LINKS=[(0,3),(3,4),(4,5)]
FRONT_LIMBS=[(3,6),(6,7),(3,8),(8,9)]
HIND_LIMBS=[(5,10),(10,11),(5,12),(12,13)]
TAIL_LINKS=[(5,14),(14,15),(15,16)]


# ==================== 影片讀取（固定為640） ====================
# ==================== 高效讀取（自動resize成640） ====================
class VideoStream:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.queue = Queue(maxsize=3)
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, daemon=True)
        t.start()
        return self  # ←❗修正 this → self

    def update(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                frame = cv2.resize(frame, (IMGSZ, IMGSZ))
                self.queue.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        return self.queue.get() if not self.queue.empty() else None



# ==================== BBox（正常狀態） ====================
def draw_bbox(frame, result, color=BLUE, thickness=2, shake=0):
    # shake 用於震動特效
    dx = np.random.randint(-shake, shake+1) if shake > 0 else 0
    dy = np.random.randint(-shake, shake+1) if shake > 0 else 0

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
            conf = float(box.conf[0].cpu().numpy())

            cv2.rectangle(frame,(x1+dx,y1+dy),(x2+dx,y2+dy),color,thickness)
            label = f"cat {conf:.2f}"
            cv2.putText(frame,label,(x1+dx,y1+dy-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)


# ==================== 骨架繪製 ====================
def draw_skeleton(frame, kpts, conf, abn_idx=None):
    for links,color in [(HEAD_LINKS,COLOR_HEAD),(BODY_LINKS,COLOR_BODY),
                        (FRONT_LIMBS,COLOR_LIMB),(HIND_LIMBS,COLOR_LIMB),(TAIL_LINKS,COLOR_TAIL)]:
        for a,b in links:
            if conf[a]>KP_CONF_THRES and conf[b]>KP_CONF_THRES:
                cv2.line(frame,tuple(kpts[a].astype(int)),tuple(kpts[b].astype(int)),color,2)

    # ⚠ 標記最異常關節（如果存在）
    for i,(x,y) in enumerate(kpts):
        if i == abn_idx:
            cv2.circle(frame,(int(x),int(y)),10,(0,255,255),-1)     # ⚠亮黃色點
            cv2.putText(frame,f"ABN-{i}",(int(x)+10,int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),3)
        elif conf[i] > KP_CONF_THRES:
            cv2.circle(frame,(int(x),int(y)),5,(0,0,255),-1)

    return frame


# ==================== 異常判斷（逐幀） ====================
def pose_deviation(prev_kpts,curr_kpts,kpt_conf):
    if prev_kpts is None: return False,None
    body_scale = np.linalg.norm(curr_kpts[3]-curr_kpts[5])
    if body_scale < 1e-3: return False,None
    diff = np.linalg.norm(curr_kpts-prev_kpts,axis=1)/body_scale
    valid = kpt_conf>KP_CONF_THRES
    if not np.any(valid): return False,None
    abn_idx = np.argmax(diff)
    return (np.max(diff[valid])>DEVIATION_THRES), abn_idx


# ==================== 大螢幕異常提示（背景紅光 + 文字） ====================
def abnormal_effect(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay,(0,0),(frame.shape[1],frame.shape[0]),(0,0,255),-1)
    frame = cv2.addWeighted(overlay,0.18,frame,0.82,0)

    cv2.putText(frame,"⚠ ABNORMAL POSE DETECTED ⚠",
                (40, frame.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX,
                1.3, (0,0,255), 4)
    return frame


# ==================== DTW 特徵 ====================
def extract_pose_feature(kpts,kpt_conf):
    visible = kpt_conf > KP_CONF_THRES
    if not np.any(visible): return None
    center = kpts[visible].mean(axis=0)
    scale = np.linalg.norm(kpts[3]-kpts[5]); scale = 1 if scale<1e-3 else scale
    return ((kpts-center)/scale).flatten()


# ==================== 載入 baseline ====================
try:
    normal_pattern = np.load(NORMAL_PATTERN_PATH)
    if normal_pattern.ndim==1:
        normal_pattern = normal_pattern.reshape(-1,normal_pattern.shape[0])
    print("[DTW] Loaded:", normal_pattern.shape)
except FileNotFoundError:
    print("[DTW] ❌ normal_pattern.npy not found → DTW停用")
    normal_pattern = None

sequence_buffer = deque(maxlen=SEQ_MAX_LEN)


# ==================== 主運行 ====================
model = YOLO(MODEL_PATH)
model.to("cuda")

cap = VideoStream(VIDEO_PATH).start()  # ←❗啟動執行緒


prev_kpts = None
last_dtw = None
prev_time = time.time()

while not cap.stopped:
    frame = cap.read()
    if frame is None: continue

    result = model.predict(frame, imgsz=IMGSZ, conf=CONF_THRES, verbose=False, half=True)[0]

    now=time.time(); fps=1/(now-prev_time); prev_time=now
    abnormal_dev=False; abnormal_dtw=False
    abn_joint=None  # <---記錄最異常關節編號

    # 💀 Pose 與 DTW
    if result.keypoints is not None and len(result.keypoints.xy)>0:
        kpts = result.keypoints.xy[0].cpu().numpy()
        conf = result.keypoints.conf[0].cpu().numpy()

        abnormal_dev,abn_joint = pose_deviation(prev_kpts,kpts,conf)
        prev_kpts = kpts.copy()

        feat = extract_pose_feature(kpts,conf)
        if feat is not None: sequence_buffer.append(feat)
        if normal_pattern is not None and len(sequence_buffer)>5:
            seq = np.array(sequence_buffer)
            last_dtw,_ = fastdtw(seq,normal_pattern,dist=euclidean)
            abnormal_dtw = last_dtw > DTW_THRES

        frame = draw_skeleton(frame,kpts,conf,abn_idx=abn_joint)

    # ❗最終異常判斷
    abnormal = abnormal_dev or abnormal_dtw

    # 🚨 異常 → 特效 + 震動 BBox + 關節提示
    if abnormal:
        frame = abnormal_effect(frame)
        draw_bbox(frame, result, color=(0,0,255), thickness=4, shake=6)  # 抖動值6
    else:
        draw_bbox(frame, result, color=BLUE, thickness=2, shake=0)

    # 📍資訊面板
    text = [
        "imgsz: 640 (locked)",
        f"FPS: {fps:.1f}",
        f"DTW: {last_dtw:.1f}" if last_dtw else "DTW: -",
        f"ABN JOINT: {abn_joint}" if abn_joint is not None else "",
    ]
    for i,t in enumerate(text):
        cv2.putText(frame,t,(20,35+30*i),cv2.FONT_HERSHEY_SIMPLEX,0.8,BLACK,2)

    cv2.imshow("Cat Pose Warning System (DTW + Vibration + Joint Alert)",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()
