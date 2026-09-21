"""
Unit Test：utils/gui_display.py 的 letterbox()（本地 GUI 視窗貼合，避免畫面被拉伸）。

只測純運算；fit_to_window() 需要真的 cv2 視窗（getWindowImageRect），屬 GUI IO，不在此列。
"""

import numpy as np
import pytest

from utils.gui_display import letterbox


def _solid(h, w, value=255, channels=3):
    shape = (h, w, channels) if channels else (h, w)
    return np.full(shape, value, dtype=np.uint8)


class TestLetterbox:
    def test_wider_window_adds_side_bars_and_keeps_aspect(self):
        # ESP32-CAM 640×480（4:3）放進 16:9 的 1280×720：內容 960×720、左右各 160 黑邊
        out = letterbox(_solid(480, 640), 1280, 720)
        assert out.shape == (720, 1280, 3)
        assert out[:, :160].max() == 0 and out[:, -160:].max() == 0
        assert out[:, 160:-160].min() == 255

    def test_taller_window_adds_top_bottom_bars(self):
        out = letterbox(_solid(480, 640), 640, 960)
        assert out.shape == (960, 640, 3)
        assert out[:240].max() == 0 and out[-240:].max() == 0
        assert out[240:-240].min() == 255

    def test_same_aspect_ratio_fills_window_without_bars(self):
        out = letterbox(_solid(480, 640), 1280, 960)
        assert out.shape == (960, 1280, 3)
        assert out.min() == 255

    def test_square_stays_square_in_wide_window(self):
        # 直接證明不拉伸：正方形內容放進 16:9 視窗，非黑區域的寬高必須相等
        frame = _solid(400, 400)
        out = letterbox(frame, 1280, 720)
        ys, xs = np.nonzero(out[:, :, 0])
        assert (xs.max() - xs.min() + 1) == (ys.max() - ys.min() + 1) == 720

    def test_upscales_small_frame_to_fill_window(self):
        out = letterbox(_solid(48, 64), 640, 480)
        assert out.shape == (480, 640, 3)
        assert out.min() == 255

    def test_downscales_large_frame(self):
        out = letterbox(_solid(1080, 1920), 640, 360)
        assert out.shape == (360, 640, 3)

    @pytest.mark.parametrize("w,h", [(0, 720), (1280, 0), (0, 0), (-1, 100)])
    def test_non_positive_window_returns_frame_unchanged(self, w, h):
        # 視窗被最小化時 getWindowImageRect 會回傳 0：不能崩、不能做無意義縮放
        frame = _solid(480, 640)
        assert letterbox(frame, w, h) is frame

    def test_grayscale_frame_supported(self):
        out = letterbox(_solid(480, 640, channels=0), 1280, 720)
        assert out.shape == (720, 1280)
        assert out[:, :160].max() == 0

    def test_dtype_preserved(self):
        frame = np.full((480, 640, 3), 0.5, dtype=np.float32)
        assert letterbox(frame, 1280, 720).dtype == np.float32
