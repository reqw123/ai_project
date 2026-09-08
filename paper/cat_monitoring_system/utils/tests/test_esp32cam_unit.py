"""
Unit Test：utils/esp32cam.py 的純函式（framesize 換算、控制端點推導）。

只測不需要對外發網路請求的部分：
- `framesize_value_for()` —— 目標尺寸 -> ESP32-CAM framesize 列舉值的純對照/挑選邏輯
- `_control_base_url()` —— 從串流網址推導 /control 端點的純字串處理

真正送 HTTP 控制請求的 `configure_stream()` 需要對外連線，屬 IO，不在此列。
"""

import pytest

from utils.esp32cam import _control_base_url, framesize_value_for


# ============================================================================
# framesize_value_for()
# ============================================================================
class TestFramesizeValueFor:
    @pytest.mark.parametrize(
        "width,height,expected_val",
        [
            (640, 480, 8),      # VGA 完全相符
            (800, 600, 9),      # SVGA 完全相符
            (320, 240, 5),      # QVGA 完全相符
            (1280, 720, 11),    # HD 完全相符
            (1600, 1200, 13),   # UXGA 完全相符
        ],
    )
    def test_exact_match(self, width, height, expected_val):
        _, _, val = framesize_value_for(width, height)
        assert val == expected_val

    def test_non_standard_size_rounds_down_not_up(self):
        # 700x500 沒有完全相符的標準尺寸，應退回「寬高都不超過」的最大者 = VGA
        w, h, val = framesize_value_for(700, 500)
        assert (w, h, val) == (640, 480, 8)

    def test_smaller_than_every_standard_size_returns_smallest(self):
        w, h, val = framesize_value_for(50, 50)
        assert (w, h, val) == (96, 96, 0)


# ============================================================================
# _control_base_url()
# ============================================================================
class TestControlBaseUrl:
    def test_swaps_stream_port_for_control_port(self):
        assert (
            _control_base_url("http://192.168.0.120:81/stream", 80)
            == "http://192.168.0.120:80/control"
        )

    def test_control_port_zero_keeps_stream_port(self):
        assert (
            _control_base_url("http://192.168.0.120:81/stream", 0)
            == "http://192.168.0.120:81/control"
        )

    def test_https_scheme_preserved(self):
        assert (
            _control_base_url("https://cam.local:81/stream", 80)
            == "https://cam.local:80/control"
        )

    @pytest.mark.parametrize(
        "url",
        [
            "rtsp://192.168.0.120:554/stream1",
            "not-a-url",
            "",
        ],
    )
    def test_non_http_source_returns_none(self, url):
        assert _control_base_url(url, 80) is None

    @pytest.mark.parametrize(
        "url",
        [
            "https://rr3---sn-abc.googlevideo.com:443/videoplayback?x=1",
            "http://example.com:81/stream",
            "http://8.8.8.8:81/stream",
        ],
    )
    def test_public_host_returns_none(self, url):
        # 只對區網裝置送控制請求；公開串流來源不該被推導出 /control 端點
        assert _control_base_url(url, 80) is None

    def test_mdns_local_host_allowed(self):
        assert (
            _control_base_url("http://esp32cam.local:81/stream", 80)
            == "http://esp32cam.local:80/control"
        )
