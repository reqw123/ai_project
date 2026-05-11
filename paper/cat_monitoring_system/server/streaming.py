"""
MJPEG 串流管理
"""
import threading
import cv2
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import STGCNConfig, VisualizationConfig
    _TARGET_MODEL_FPS = STGCNConfig.TARGET_MODEL_FPS
    _ENABLE_FPS_DOWNSAMPLE = STGCNConfig.ENABLE_FPS_DOWNSAMPLE
    _STREAM_DISPLAY_SIZE = VisualizationConfig.STREAM_DISPLAY_SIZE
    _FAST_STREAM_OVERLAY = VisualizationConfig.FAST_STREAM_OVERLAY
except Exception:
    _TARGET_MODEL_FPS = 30.0
    _ENABLE_FPS_DOWNSAMPLE = True
    _STREAM_DISPLAY_SIZE = None
    _FAST_STREAM_OVERLAY = True


class SharedFrameStreamer:
    def __init__(self, frame_processor):
        self.frame_processor = frame_processor
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def _update_frame(self):
        cap = self.frame_processor.cap

        # --- FPS 同步：計算降採樣步長 ---
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 1:
            source_fps = _TARGET_MODEL_FPS

        frame_step = 1
        if _ENABLE_FPS_DOWNSAMPLE and source_fps > _TARGET_MODEL_FPS + 1e-6:
            frame_step = max(1, int(round(source_fps / _TARGET_MODEL_FPS)))

        raw_frame_count = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                raw_frame_count = 0
                # EMA 狀態隨影片重播一併重置
                self.frame_processor._ema_kpts = None
                continue

            raw_frame_count += 1

            # 跳過多餘幀以對齊模型目標 FPS
            if frame_step > 1 and ((raw_frame_count - 1) % frame_step != 0):
                continue

            processed_frame, *_ = self.frame_processor.process(frame)

            # --- 繪圖優化：若設定了串流輸出尺寸，先縮放再 JPEG 編碼 ---
            if _STREAM_DISPLAY_SIZE is not None:
                if _FAST_STREAM_OVERLAY:
                    # overlay 已畫在全解析度，直接縮放（快）
                    processed_frame = cv2.resize(processed_frame, _STREAM_DISPLAY_SIZE)
                else:
                    # 縮放原始幀後重繪 overlay（字體更清晰，但較慢）
                    processed_frame = cv2.resize(processed_frame, _STREAM_DISPLAY_SIZE)

            with self.lock:
                self.latest_frame = processed_frame.copy()

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
