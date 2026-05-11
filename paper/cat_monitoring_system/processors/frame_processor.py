
"""
幀處理管道（整合 Node-RED、CSV、異常檢測、overlay 控制）
"""
import numpy as np
import cv2
import time
from collections import deque
from detectors.keypoint_detector import KeypointDetector
from detectors.behavior_classifier import BehaviorClassifier
from trackers.behavior_tracker import ImprovedBehaviorTracker
from processors.anomaly_detector import AnomalyDetector
from processors.visualizer import Visualizer
from communication.nodered_client import NodeRedClient
from logutils.csv_logger import CSVLogger, BehaviorSegmentLogger
from utils.helpers import get_ip
from utils.constants import *
from models.stgcn_model import interpolate_missing

class FrameProcessor:
    def __init__(self, yolo_model_path, stgcn_model_path, video_path,
                 csv_path=None, segments_csv_path=None, nodered_url=None, nodered_notify_url=None, device='cuda', imgsz=640, conf_thres=0.5, sequence_length=32,
                 overlay=True, width=None, height=None, normalize=True, kp_ema_alpha=0.5):
        self.local_ip = get_ip()
        self.cap = cv2.VideoCapture(video_path)
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.keypoint_detector = KeypointDetector(yolo_model_path, device=device, imgsz=imgsz, conf_thres=conf_thres)
        self.behavior_classifier = BehaviorClassifier(stgcn_model_path, device=device, sequence_length=sequence_length, normalize=normalize)
        self.tracker = ImprovedBehaviorTracker()
        self.anomaly_detector = AnomalyDetector()
        self.visualizer = Visualizer()
        # 傳遞 yolo_model_path 給 visualizer 以供 overlay 顯示
        self.visualizer.yolo_model_path = yolo_model_path
        self.visualizer.frame_idx = 0
        self.keypoints_buffer = deque(maxlen=sequence_length)
        self.sequence_length = sequence_length
        self.overlay = overlay
        self.fps_display = 0.0
        self.prev_time = time.time()
        self.last_send_time = time.time()
        self.nodered = None
        if nodered_url:
            self.nodered = NodeRedClient(nodered_url, url_notify=nodered_notify_url, local_ip=self.local_ip)
        self.csv_logger = CSVLogger(csv_path) if csv_path else None
        self.segment_logger = BehaviorSegmentLogger(segments_csv_path) if segments_csv_path else None
        self.frame_idx = 0
        # 關鍵點 EMA 平滑狀態：None = 尚未初始化（貓咪初次出現或重新出現時）
        self.kp_ema_alpha = kp_ema_alpha
        self._ema_kpts = None

    def process(self, frame):
        self.frame_idx += 1
        # FPS 計算
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt > 0:
            self.fps_display = 0.9 * self.fps_display + 0.1 * (1.0 / dt)
        self.prev_time = current_time

        kpts, kpt_conf, bbox, conf = self.keypoint_detector.detect(frame)
        behavior_id, confidence, class_probs = 0, 0.0, [0.0]*4
        abnormal, activity_value = False, 0.0

        if kpts is not None:
            # === 關鍵點 EMA 平滑（與訓練時 KP_EMA_ALPHA 保持一致） ===
            if self._ema_kpts is None:
                self._ema_kpts = kpts.copy()
            else:
                self._ema_kpts = self.kp_ema_alpha * kpts + (1.0 - self.kp_ema_alpha) * self._ema_kpts
            kpts = self._ema_kpts.copy()
            # === 異常檢測 ===
            abnormal, activity_value = self.anomaly_detector.detect(kpts, kpt_conf)

            # === ST-GCN 行為推論 ===
            self.keypoints_buffer.append((kpts, kpt_conf))
            if len(self.keypoints_buffer) >= self.sequence_length:
                kpts_arr = np.array([item[0] for item in self.keypoints_buffer])   # (T, 17, 2)
                conf_arr = np.array([item[1] for item in self.keypoints_buffer])   # (T, 17)
                seq_array = interpolate_missing(kpts_arr, conf_arr)                # 與訓練一致
                behavior_id, confidence, class_probs = self.behavior_classifier.classify(seq_array)
                if behavior_id is None:
                    behavior_id = LOW_CONF_ID
                    confidence = 0.0
                elif confidence < CONFIDENCE_THRESHOLD:
                    behavior_id = LOW_CONF_ID

            # === 行為追蹤 ===
            len_before = len(self.tracker.behavior_history)
            self.tracker.update(behavior_id, activity_value)
            if self.segment_logger and len(self.tracker.behavior_history) > len_before:
                rec = list(self.tracker.behavior_history)[-1]
                if rec["gcn_behavior_id"] != LOW_CONF_ID:
                    self.segment_logger.log_segment(
                        rec["gcn_behavior_id"],
                        BEHAVIOR_TEXT_MAP.get(rec["gcn_behavior_id"], rec["behavior"]),
                        rec["duration"],
                        rec.get("activity", 0),
                    )

            # === Node-RED 資料推送 ===
            now = time.time()
            if self.nodered and (now - self.last_send_time >= 0.5):
                data = {
                    "current": {
                        "behavior_id": int(behavior_id),
                        "text": LOW_CONF_TEXT if behavior_id == LOW_CONF_ID else BEHAVIOR_TEXT_MAP.get(behavior_id, "未知"),
                        "behavior": LOW_CONF_TEXT if behavior_id == LOW_CONF_ID else BEHAVIOR_TEXT_MAP.get(behavior_id, "未知"),
                        "emoji": LOW_CONF_EMOJI if behavior_id == LOW_CONF_ID else BEHAVIOR_EMOJI_MAP.get(behavior_id, "❓"),
                        "timestamp": time.strftime("%H:%M:%S")
                    },
                    "activity_score": int(self.tracker.get_activity_score()),
                    "today_stats": self.tracker.get_today_stats(),
                    "behavior_log": [
                        {
                            "behavior": rec["behavior"],
                            "gcn_id": rec["gcn_behavior_id"],
                            "time": rec["timestamp"].strftime("%H:%M:%S"),
                            "duration": rec["duration"]
                        }
                        for rec in list(self.tracker.behavior_history)[-10:]
                    ],
                    "alerts": self.tracker.get_alerts(),
                    "system": {
                        "ip": self.local_ip,
                        "model": "YOLO-Pose + ST-GCN",
                        "version": "v4.0-stgcn",
                        "gcn_confidence": round(float(confidence), 3)
                    }
                }
                self.nodered.send_data(data)
                self.last_send_time = now

            # === CSV 日誌 ===
            if self.csv_logger and abnormal and behavior_id != LOW_CONF_ID:
                self.csv_logger.log(
                    self.frame_idx,
                    BEHAVIOR_CLASSES[behavior_id],
                    confidence,
                    abnormal,
                    self.anomaly_detector.ema_motion,
                    1.0
                )

            # === Overlay 畫圖 ===
            if self.overlay:
                # 傳遞 fps 與 frame_idx 給 visualizer 以供 overlay 顯示
                self.visualizer.fps = self.fps_display
                self.visualizer.frame_idx = self.frame_idx
                frame = self.visualizer.draw(frame, kpts, kpt_conf, bbox, conf, behavior_id, confidence, class_probs)

        else:
            # 貓咪消失時重置 EMA，避免重新出現時使用過時的平均值
            self._ema_kpts = None

        return frame, behavior_id, confidence, class_probs, abnormal, activity_value

    def cleanup(self):
        self.cap.release()
        if self.csv_logger:
            self.csv_logger.close()
        if self.segment_logger:
            self.segment_logger.close()
        cv2.destroyAllWindows()
