
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
from utils.helpers import get_ip, get_behavior_name
from utils.constants import *
from models.stgcn_model import (
    interpolate_missing,
    flip_normalize,
    orientation_normalize,
    normalize_skeleton_coords,
    build_feature_tensor,
    get_in_channels_for_mode,
)
from config import NodeRedConfig, BehaviorTrackingConfig, STGCNConfig, SystemInfo

class FrameProcessor:
    def __init__(self, yolo_model_path, stgcn_model_path, video_path,
                 nodered_url=None, device='cuda', imgsz=640, conf_thres=0.5, sequence_length=STGCNConfig.SEQUENCE_LENGTH,
                 overlay=True, width=None, height=None, normalize=True, kp_ema_alpha=STGCNConfig.KP_EMA_ALPHA,
                 feature_mode=None, window_stride=None):
        self.local_ip = get_ip()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_path}")
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            self.keypoint_detector = KeypointDetector(yolo_model_path, device=device, imgsz=imgsz, conf_thres=conf_thres)
            self.behavior_classifier = BehaviorClassifier(
                stgcn_model_path, device=device, sequence_length=sequence_length,
                normalize=normalize, feature_mode=feature_mode,
            )
        except Exception:
            self.cap.release()
            raise
        self.tracker = ImprovedBehaviorTracker()
        self.anomaly_detector = AnomalyDetector()
        self.visualizer = Visualizer()
        self.keypoints_buffer = deque(maxlen=sequence_length)
        self.sequence_length = sequence_length
        self.window_stride = window_stride if window_stride is not None else STGCNConfig.WINDOW_STRIDE
        self._infer_frame_count = 0  # 累積有效幀計數器，以 window_stride 取模決定推論時機（不重置）
        self.overlay = overlay
        self.fps_display = 0.0
        self.prev_time = time.time()
        self.last_send_time = time.time()
        self.nodered = None
        if nodered_url:
            self.nodered = NodeRedClient(nodered_url, local_ip=self.local_ip)
        self.csv_logger = CSVLogger()
        self.segment_logger = BehaviorSegmentLogger()
        self.frame_idx = 0
        # 推論時機快取：避免 process() 熱路徑上每幀重複 getattr
        _m = getattr(self.behavior_classifier, 'model', None)
        self._use_multichannel  = (_m is not None and getattr(_m, 'in_channels', 4) != 4)
        self._model_normalize   = getattr(_m, 'normalize',     True)    if _m else True
        self._model_feature_mode = getattr(_m, 'feature_mode', 'xy_v') if _m else 'xy_v'
        # 關鍵點 EMA：用於 overlay 顯示與異常偵測（不進入 ST-GCN buffer）
        self.kp_ema_alpha = kp_ema_alpha
        self._ema_kpts = None
        # 保存上次推論結果，非推論幀沿用，避免標籤閃爍
        self._last_behavior_id = LOW_CONF_ID
        self._last_confidence = 0.0
        self._last_class_probs = [0.0] * 5

    def process(self, frame):
        self.frame_idx += 1
        # FPS 計算
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt > 0:
            self.fps_display = 0.9 * self.fps_display + 0.1 * (1.0 / dt)
        self.prev_time = current_time

        kpts, kpt_conf, bbox, conf = self.keypoint_detector.detect(frame)
        # 沿用上次推論結果；僅在本幀推論成功時更新
        behavior_id = self._last_behavior_id
        confidence = self._last_confidence
        class_probs = self._last_class_probs
        is_still, activity_value = False, 0.0

        if kpts is not None:
            raw_kpts = kpts.copy()

            # === Frame-level EMA：僅用於 overlay 顯示與異常偵測，原始 raw_kpts 進 ST-GCN buffer ===
            # 注意：此 EMA 不影響 STGCN 推論路徑；ST-GCN 輸入的唯一平滑來源是下方 window-level EMA
            if self._ema_kpts is None:
                self._ema_kpts = raw_kpts.copy()
            else:
                self._ema_kpts = self.kp_ema_alpha * raw_kpts + (1.0 - self.kp_ema_alpha) * self._ema_kpts
            display_kpts = self._ema_kpts.copy()

            # === 靜止偵測（使用 EMA 平滑後的關鍵點） ===
            is_still, activity_value = self.anomaly_detector.detect(display_kpts, kpt_conf)

            # === ST-GCN 行為推論（buffer 儲存 raw kpts，與訓練前處理順序一致） ===
            self.keypoints_buffer.append((raw_kpts, kpt_conf))
            self._infer_frame_count += 1
            should_infer = (
                len(self.keypoints_buffer) >= self.sequence_length
                and (self._infer_frame_count % max(1, self.window_stride) == 0)
            )
            if should_infer:
                kpts_arr = np.array([item[0] for item in self.keypoints_buffer])   # (T, 17, 2)
                conf_arr = np.array([item[1] for item in self.keypoints_buffer])   # (T, 17)
                seq_array = interpolate_missing(kpts_arr, conf_arr)
                # Window-level EMA：STGCN 輸入的唯一平滑步驟，須與訓練時使用的 KP_EMA_ALPHA 一致
                # alpha=1.0（預設）表示不平滑；調低時須確認訓練也用相同數值，切勿在此之外另加平滑
                if self.kp_ema_alpha < 1.0:
                    for t in range(1, seq_array.shape[0]):
                        seq_array[t] = (self.kp_ema_alpha * seq_array[t]
                                        + (1.0 - self.kp_ema_alpha) * seq_array[t - 1])
                if self._use_multichannel:
                    if self._model_normalize:
                        seq_array = flip_normalize(seq_array)
                        seq_array = orientation_normalize(seq_array)
                        seq_array = normalize_skeleton_coords(seq_array)
                    seq_features = build_feature_tensor(seq_array, conf_arr, self._model_feature_mode)
                    new_bid, new_conf, new_probs = self.behavior_classifier.classify(seq_features, precomputed=True)
                else:
                    new_bid, new_conf, new_probs = self.behavior_classifier.classify(seq_array)
                if new_bid is None:
                    new_bid = LOW_CONF_ID
                    new_conf = 0.0
                elif new_conf < BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD:
                    new_bid = LOW_CONF_ID
                # 更新持久化結果，本幀也立即採用
                self._last_behavior_id = new_bid
                self._last_confidence = new_conf
                self._last_class_probs = new_probs if new_probs is not None else [0.0] * 5
                behavior_id = self._last_behavior_id
                confidence = self._last_confidence
                class_probs = self._last_class_probs
            # 以 display_kpts 替換後續用到 kpts 的位置
            kpts = display_kpts

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
            if self.nodered and (now - self.last_send_time >= NodeRedConfig.PUSH_INTERVAL): #設定的最小送出間隔
                # 根據顯示門檻決定呈現標籤（若 confidence 未達 BEHAVIOR_MIN_CONFIDENCE 則顯示為 LOW_CONF_TEXT）
                is_display_normal = (behavior_id == LOW_CONF_ID) or (float(confidence) < BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD)
                if is_display_normal:
                    display_behavior = LOW_CONF_TEXT
                    display_text = LOW_CONF_TEXT
                    display_emoji = LOW_CONF_EMOJI
                else:
                    display_behavior = BEHAVIOR_CLASSES[int(behavior_id)] if 0 <= int(behavior_id) < len(BEHAVIOR_CLASSES) else "unknown"
                    display_text = BEHAVIOR_TEXT_MAP.get(behavior_id, "未知")
                    display_emoji = BEHAVIOR_EMOJI_MAP.get(behavior_id, "❓")

                data = {
                    "current": {
                        "behavior_id": int(behavior_id),
                        "text": display_text,
                        "behavior": display_behavior,
                        "emoji": display_emoji,
                        "timestamp": time.strftime("%H:%M:%S")
                    },
                    "activity_score": int(self.tracker.get_activity_score()),
                    "today_stats": self.tracker.get_today_stats(),
                    "behavior_log": [
                        {
                            "behavior": rec["behavior"],
                            "gcn_id": rec["gcn_behavior_id"],
                            "time": (rec["timestamp"].strftime("%H:%M:%S")
                                     if hasattr(rec["timestamp"], "strftime")
                                     else str(rec["timestamp"])),
                            "duration": rec["duration"]
                        }
                        for rec in list(self.tracker.behavior_history)[-10:]
                    ],
                    "alerts": self.tracker.get_alerts(),
                    "system": {
                        "ip": self.local_ip,
                        "model": "YOLO-Pose + ST-GCN",
                        "version": SystemInfo.VERSION,
                        "gcn_confidence": round(float(confidence), 3)
                    }
                }
                self.nodered.send_data(data)
                self.last_send_time = now

            # === CSV 日誌 ===
            # CSV 日誌只在貓咪活動中（非靜止）且行為信心足夠時寫入
            if self.csv_logger and not is_still and float(confidence) >= BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD and behavior_id != LOW_CONF_ID:
                behavior_name = get_behavior_name(behavior_id, use_text=False, fallback="未知", confidence=confidence)
                self.csv_logger.log(
                    self.frame_idx,
                    behavior_name,
                    confidence,
                    is_still,
                    self.anomaly_detector.last_motion_score,
                )

            # === Overlay 畫圖 ===
            if self.overlay:
                frame = self.visualizer.draw(frame, kpts, kpt_conf, bbox, conf, behavior_id, confidence, class_probs)

        else:
            # 貓咪消失時重置 EMA、推論計數器、keypoints buffer 與上次推論結果
            # _infer_frame_count 重置確保貓重新出現後推論時機從 0 對齊，不受之前計數影響
            # keypoints_buffer 清除確保舊幀不污染下次推論窗口
            self._ema_kpts = None
            self._infer_frame_count = 0
            self.keypoints_buffer.clear()
            self._last_behavior_id = LOW_CONF_ID
            self._last_confidence = 0.0
            self._last_class_probs = [0.0] * 5
            self.tracker.update(NOT_VISIBLE_ID, 0.0)
            # Node-RED 推送：通知貓咪不在畫面
            now = time.time()
            if self.nodered and (now - self.last_send_time >= NodeRedConfig.PUSH_INTERVAL):
                data = {
                    "current": {
                        "behavior_id": NOT_VISIBLE_ID,
                        "text": NOT_VISIBLE_DISPLAY_TEXT,
                        "behavior": NOT_VISIBLE_TEXT,
                        "emoji": NOT_VISIBLE_EMOJI,
                        "timestamp": time.strftime("%H:%M:%S")
                    },
                    "activity_score": int(self.tracker.get_activity_score()),
                    "today_stats": self.tracker.get_today_stats(),
                    "behavior_log": [
                        {
                            "behavior": rec["behavior"],
                            "gcn_id": rec["gcn_behavior_id"],
                            "time": (rec["timestamp"].strftime("%H:%M:%S")
                                     if hasattr(rec["timestamp"], "strftime")
                                     else str(rec["timestamp"])),
                            "duration": rec["duration"]
                        }
                        for rec in list(self.tracker.behavior_history)[-10:]
                    ],
                    "alerts": self.tracker.get_alerts(),
                    "system": {
                        "ip": self.local_ip,
                        "model": "YOLO-Pose + ST-GCN",
                        "version": SystemInfo.VERSION,
                        "gcn_confidence": 0.0
                    }
                }
                self.nodered.send_data(data)
                self.last_send_time = now

        return frame, behavior_id, confidence, class_probs, is_still, activity_value

    def reset_ema(self):
        """重置 EMA 狀態（影片重播或貓咪重新出現時由外部呼叫）。"""
        self._ema_kpts = None

    def cleanup(self):
        self.cap.release()
        if self.csv_logger:
            self.csv_logger.close()
        if self.segment_logger:
            self.segment_logger.close()
        cv2.destroyAllWindows()
