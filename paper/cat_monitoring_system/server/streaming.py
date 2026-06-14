"""
MJPEG 串流管理
"""
import logging
import threading
import cv2
import time
from collections import deque
from config import FlaskConfig, STGCNConfig, VisualizationConfig

_TARGET_MODEL_FPS = STGCNConfig.TARGET_MODEL_FPS
_ENABLE_FPS_DOWNSAMPLE = STGCNConfig.ENABLE_FPS_DOWNSAMPLE
_STREAM_DISPLAY_SIZE = VisualizationConfig.STREAM_DISPLAY_SIZE
_CLIP_SECONDS = VisualizationConfig.CLIP_SECONDS


class SharedFrameStreamer:
    """單一寫入執行緒負責所有幀處理與 JPEG 編碼；消費者（路由、Node-RED）
    只讀取已編碼的 bytes，不重複編碼，確保每幀 CPU 開銷固定為一次。

    設計不變式：
    - latest_jpeg 為 Python bytes（不可變），消費者可直接回傳參考，無需額外拷貝。
    - JPEG 品質與編碼參數於寫入執行緒啟動時快取，避免熱路徑上重複建立 list。
    - clip_buffer 保存 BGR numpy 供 /video_clip，與 JPEG 快取使用各自獨立鎖。
    """

    def __init__(self, frame_processor):
        self.frame_processor = frame_processor
        self.latest_jpeg: bytes | None = None
        self.lock = threading.Lock()
        clip_maxlen = max(30, int(_TARGET_MODEL_FPS * _CLIP_SECONDS))
        self.clip_buffer = deque(maxlen=clip_maxlen)
        self.clip_lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def _update_frame(self):
        cap = self.frame_processor.cap
        if cap is None or not cap.isOpened():
            logging.error("SharedFrameStreamer: VideoCapture is not available")
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 1:
            source_fps = _TARGET_MODEL_FPS

        frame_step = 1
        if _ENABLE_FPS_DOWNSAMPLE and source_fps > _TARGET_MODEL_FPS + 1e-6:
            frame_step = max(1, int(round(source_fps / _TARGET_MODEL_FPS)))

        # 快取編碼參數，避免每幀重新建立 list
        _q = max(1, min(int(FlaskConfig.JPEG_QUALITY), 100))
        _encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), _q]

        raw_frame_count = 0

        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    raw_frame_count = 0
                    self.frame_processor.reset_ema()
                    continue

                raw_frame_count += 1
                if frame_step > 1 and ((raw_frame_count - 1) % frame_step != 0):
                    continue

                processed_frame, *_ = self.frame_processor.process(frame)

                if _STREAM_DISPLAY_SIZE is not None:
                    processed_frame = cv2.resize(processed_frame, _STREAM_DISPLAY_SIZE)

                # 每幀只編碼一次；bytes 不可變，所有消費者共享同一物件
                _, buf = cv2.imencode('.jpg', processed_frame, _encode_param)
                jpeg_bytes: bytes = buf.tobytes()

                with self.lock:
                    self.latest_jpeg = jpeg_bytes
                with self.clip_lock:
                    self.clip_buffer.append(processed_frame.copy())

            except Exception as e:
                logging.error("SharedFrameStreamer._update_frame error: %s", e)
                time.sleep(0.1)  # 防止 tight error loop 佔滿 CPU

    def get_jpeg(self) -> bytes | None:
        """回傳最新已編碼的 JPEG bytes。
        bytes 不可變，消費者直接持有參考即可，無需複製。
        """
        with self.lock:
            return self.latest_jpeg

    def get_clip_frames(self) -> list:
        with self.clip_lock:
            return list(self.clip_buffer)

    def stop(self):
        self.running = False
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            logging.warning("SharedFrameStreamer: _update_frame thread did not stop within 5 s")
