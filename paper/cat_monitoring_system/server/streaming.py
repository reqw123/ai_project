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
    - _client_count 追蹤目前活躍的串流客戶端數；無客戶端時跳過 JPEG 編碼以節省 CPU，
      但推論執行緒仍持續運行（行為追蹤不中斷）。
    """

    def __init__(self, frame_processor):
        self.frame_processor = frame_processor
        self.latest_jpeg: bytes | None = None
        self.lock = threading.Lock()
        clip_maxlen = max(30, int(_TARGET_MODEL_FPS * _CLIP_SECONDS))
        self.clip_buffer = deque(maxlen=clip_maxlen)
        self.clip_lock = threading.Lock()
        self._client_count = 0
        self._client_lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def acquire_client(self) -> None:
        with self._client_lock:
            self._client_count += 1

    def release_client(self) -> None:
        with self._client_lock:
            if self._client_count > 0:
                self._client_count -= 1

    def _update_frame(self):
        source_fps = self.frame_processor.source_fps

        frame_step = 1
        if _ENABLE_FPS_DOWNSAMPLE and source_fps > _TARGET_MODEL_FPS + 1e-6:
            frame_step = max(1, int(round(source_fps / _TARGET_MODEL_FPS)))

        # 快取編碼參數，避免每幀重新建立 list
        _q = max(1, min(int(FlaskConfig.JPEG_QUALITY), 100))
        _encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), _q]

        raw_frame_count = 0

        while self.running:
            try:
                ret, frame = self.frame_processor.read_raw_frame()
                if not ret:
                    raw_frame_count = 0
                    continue

                raw_frame_count += 1
                if frame_step > 1 and ((raw_frame_count - 1) % frame_step != 0):
                    continue

                processed_frame, *_ = self.frame_processor.process(frame)

                display_frame = processed_frame
                if _STREAM_DISPLAY_SIZE is not None:
                    h, w = processed_frame.shape[:2]
                    tw, th = _STREAM_DISPLAY_SIZE
                    if w > 0 and h > 0 and tw > 0 and th > 0:
                        display_frame = cv2.resize(processed_frame, _STREAM_DISPLAY_SIZE)
                    else:
                        logging.warning(
                            "SharedFrameStreamer: 跳過 resize，尺寸無效 frame=(%d,%d) target=(%d,%d)",
                            w, h, tw, th,
                        )

                with self.clip_lock:
                    self.clip_buffer.append(display_frame.copy())

                # 無客戶端時跳過 JPEG 編碼，節省 CPU；推論已在上方完成不受影響
                with self._client_lock:
                    has_client = self._client_count > 0
                if not has_client:
                    continue

                # 每幀只編碼一次；bytes 不可變，所有消費者共享同一物件
                _, buf = cv2.imencode('.jpg', display_frame, _encode_param)
                with self.lock:
                    self.latest_jpeg = buf.tobytes()

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
