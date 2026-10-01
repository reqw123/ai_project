"""
MJPEG 串流管理
"""

import logging
import os
import sys
import threading
import time
from collections import deque

import cv2
from config import FlaskConfig, STGCNConfig, VisualizationConfig

_TARGET_MODEL_FPS = STGCNConfig.TARGET_MODEL_FPS
_ENABLE_FPS_DOWNSAMPLE = STGCNConfig.ENABLE_FPS_DOWNSAMPLE
_STREAM_DISPLAY_SIZE = VisualizationConfig.STREAM_DISPLAY_SIZE
_CLIP_SECONDS = VisualizationConfig.CLIP_SECONDS

# 本機錄影（影片時鐘模式）播完後自動結束行程前，等 Node-RED 最後一筆推送送出的上限秒數
_MEDIA_EXIT_FLUSH_SECONDS = 3.0


def _exit_process() -> None:
    """結束整個行程（Flask 主執行緒卡在 accept()，只能用 os._exit）；獨立成函式方便測試替換。
    os._exit 不會清空輸出緩衝區，先 flush，否則輸出導到檔案時最後幾行訊息會不見。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(0)


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
        # 排程「區段執行」用：暫停時完全不讀取/不推論/不寫入任何統計，但保留
        # 模型與 VideoCapture 不釋放，恢復時可立即從暫停當下的位置繼續，不需重新載入。
        # 與下面的 finished 分開：paused 是可被排程恢復的暫時狀態，finished 是本機
        # 影片檔案播畢的終止狀態，不會因為進入排程允許時間就被自動復活。
        self.paused = False
        # 本機影片檔案播完（非串流來源）：True 代表已經放完，不再自動重播/重試讀取。
        self.finished = False
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def acquire_client(self) -> None:
        """登記一個活躍的串流客戶端連線（用於決定是否需要編碼 JPEG）。"""
        with self._client_lock:
            self._client_count += 1

    def release_client(self) -> None:
        """釋放一個串流客戶端連線。"""
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
        is_stream = self.frame_processor.is_stream_source()

        while self.running:
            try:
                if self.paused or self.finished:
                    raw_frame_count = 0
                    time.sleep(0.2)
                    continue

                ret, frame = self.frame_processor.read_raw_frame()
                if not ret:
                    raw_frame_count = 0
                    if not is_stream and not self.finished:
                        # 本機影片檔案已放完：不循環、不重試，直接停在這裡等待人工處理
                        # （例如換一支影片、重啟程式），避免統計被無限重播的相同片段疊加。
                        self.finished = True
                        logging.info(
                            "SharedFrameStreamer: 本機影片檔案已播畢，停止處理（不自動循環）"
                        )
                        # 讓舔毛外掛結算最後一段未結束的 bout（說明書第一階段
                        # 「Session 生命週期：finish_session 結算 active bout」）
                        _finish = getattr(
                            self.frame_processor, "finish_plugin_sessions", None
                        )
                        if _finish is not None:
                            try:
                                _finish()
                            except Exception:
                                pass
                        # 本機錄影（影片時鐘模式）播完就把那一天寫進多天歷史——影片錄到
                        # 24:00 結束，之後沒有下一幀能觸發跨日（見 docs/錄影推論改用影片時間-待辦.md）
                        _finish_day = getattr(
                            self.frame_processor, "finish_media_day", None
                        )
                        if _finish_day is not None:
                            try:
                                _finish_day()
                            except Exception as e:
                                logging.error("finish_media_day 失敗：%s", e)
                        # 影片時鐘模式＝為基線收集資料的錄影推論：播完代表那一天的資料已收齊，
                        # 直接結束行程，確保統計停在影片結尾，不會再有任何東西被計入那一天
                        if getattr(self.frame_processor, "uses_media_clock", False):
                            self._exit_after_media_finished()
                            return
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
                        display_frame = cv2.resize(
                            processed_frame, _STREAM_DISPLAY_SIZE
                        )
                    else:
                        logging.warning(
                            "SharedFrameStreamer: 跳過 resize，尺寸無效 frame=(%d,%d) target=(%d,%d)",
                            w,
                            h,
                            tw,
                            th,
                        )

                with self.clip_lock:
                    self.clip_buffer.append(display_frame.copy())

                # 無客戶端時跳過 JPEG 編碼，節省 CPU；推論已在上方完成不受影響
                with self._client_lock:
                    has_client = self._client_count > 0
                if not has_client:
                    continue

                # 每幀只編碼一次；bytes 不可變，所有消費者共享同一物件
                _, jpeg_buffer = cv2.imencode(".jpg", display_frame, _encode_param)
                with self.lock:
                    self.latest_jpeg = jpeg_buffer.tobytes()

            except Exception as e:
                logging.error("SharedFrameStreamer._update_frame error: %s", e)
                time.sleep(0.1)  # 防止 tight error loop 佔滿 CPU

    def _exit_after_media_finished(self) -> None:
        """錄影資料收集完畢：送完 Node-RED 最後一筆、關閉記錄器後結束行程。"""
        self.running = False
        print("🏁 影片已推論完畢，資料收集完成：系統自動結束（統計停在影片結尾）")
        try:
            self.frame_processor.cleanup(flush_timeout=_MEDIA_EXIT_FLUSH_SECONDS)
        except Exception as e:
            logging.error("影片播完自動結束前清理失敗（仍會結束行程）：%s", e)
        try:
            print(self.frame_processor.run_summary_line())  # 終端最後一行
        except Exception as e:
            logging.error("計算此次運行總監測時長失敗：%s", e)
        _exit_process()

    def get_jpeg(self) -> bytes | None:
        """回傳最新已編碼的 JPEG bytes。
        bytes 不可變，消費者直接持有參考即可，無需複製。
        """
        with self.lock:
            return self.latest_jpeg

    def get_clip_frames(self) -> list:
        """回傳目前 ring buffer 中保存的所有畫面（供 /video_clip 使用）。"""
        with self.clip_lock:
            return list(self.clip_buffer)
