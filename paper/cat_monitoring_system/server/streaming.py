"""
MJPEG 串流管理
"""
import threading
import cv2
import time
from collections import deque
from config import STGCNConfig, VisualizationConfig

_TARGET_MODEL_FPS = STGCNConfig.TARGET_MODEL_FPS
_ENABLE_FPS_DOWNSAMPLE = STGCNConfig.ENABLE_FPS_DOWNSAMPLE
_STREAM_DISPLAY_SIZE = VisualizationConfig.STREAM_DISPLAY_SIZE
_FAST_STREAM_OVERLAY = VisualizationConfig.FAST_STREAM_OVERLAY
_CLIP_SECONDS = VisualizationConfig.CLIP_SECONDS


class SharedFrameStreamer:
    def __init__(self, frame_processor):
        self.frame_processor = frame_processor
        self.latest_frame = None
        self.lock = threading.Lock()
        clip_maxlen = max(30, int(_TARGET_MODEL_FPS * _CLIP_SECONDS))
        self.clip_buffer = deque(maxlen=clip_maxlen)
        self.clip_lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def _update_frame(self):
        import logging
        cap = self.frame_processor.cap
        if cap is None or not cap.isOpened():
            logging.error("SharedFrameStreamer: VideoCapture is not available")
            return

        # --- FPS 同步：計算降採樣步長 ---
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 1:
            source_fps = _TARGET_MODEL_FPS

        frame_step = 1
        if _ENABLE_FPS_DOWNSAMPLE and source_fps > _TARGET_MODEL_FPS + 1e-6:
            frame_step = max(1, int(round(source_fps / _TARGET_MODEL_FPS)))

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

                with self.lock:
                    self.latest_frame = processed_frame.copy()
                with self.clip_lock:
                    self.clip_buffer.append(processed_frame.copy())
            except Exception as e:
                logging.error("SharedFrameStreamer._update_frame error: %s", e)
                time.sleep(0.1)  # 防止 tight error loop 佔滿 CPU

    def get_jpeg(self):
        # 鎖內只 copy frame，鎖外再做 CPU 密集的 JPEG 編碼，避免阻塞寫入 thread
        with self.lock:
            if self.latest_frame is None:
                return None
            frame_copy = self.latest_frame.copy()

        try:
            from config import FlaskConfig
            q = int(getattr(FlaskConfig, 'JPEG_QUALITY', 60))
        except Exception:
            q = 60
        q = max(1, min(q, 100))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        _, buffer = cv2.imencode('.jpg', frame_copy, encode_param)
        return buffer.tobytes()

    def get_clip_frames(self):
        with self.clip_lock:
            return list(self.clip_buffer)

    def stop(self):
        self.running = False
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            import logging
            logging.warning("SharedFrameStreamer: _update_frame thread did not stop within 5 s")
