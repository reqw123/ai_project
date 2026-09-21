"""
Unit Test：utils/esp32cam.py 的純函式（framesize 換算、控制端點推導）。

只測不需要對外發網路請求的部分：
- `framesize_value_for()` —— 目標尺寸 -> ESP32-CAM framesize 列舉值的純對照/挑選邏輯
- `_control_base_url()` —— 從串流網址推導 /control 端點的純字串處理

真正送 HTTP 控制請求的 `configure_stream()` 需要對外連線，屬 IO，不在此列。
"""

import pytest

from utils import esp32cam
from utils.esp32cam import (
    _control_base_url,
    _parse_device_resolutions,
    configure_stream,
    framesize_value_for,
)

# Arduino-ESP32 3.x 客製韌體實機 /status 回報的 resolutions（2026-09-20 實測）
_DEVICE_STATUS = {"resolutions": {"qvga": 6, "vga": 10, "svga": 11}}


# ============================================================================
# framesize_value_for()
# ============================================================================
class TestFramesizeValueFor:
    @pytest.mark.parametrize(
        "width,height,expected_val",
        [
            (640, 480, 10),     # VGA 完全相符（新版 esp32-camera 編號）
            (800, 600, 11),     # SVGA 完全相符
            (320, 240, 6),      # QVGA 完全相符
            (1280, 720, 13),    # HD 完全相符
            (1600, 1200, 15),   # UXGA 完全相符
        ],
    )
    def test_exact_match(self, width, height, expected_val):
        _, _, val = framesize_value_for(width, height)
        assert val == expected_val

    def test_non_standard_size_rounds_down_not_up(self):
        # 700x500 沒有完全相符的標準尺寸，應退回「寬高都不超過」的最大者 = VGA
        w, h, val = framesize_value_for(700, 500)
        assert (w, h, val) == (640, 480, 10)

    def test_smaller_than_every_standard_size_returns_smallest(self):
        w, h, val = framesize_value_for(50, 50)
        assert (w, h, val) == (96, 96, 0)

    def test_explicit_table_overrides_builtin_numbering(self):
        # 舊版 esp32-camera 編號（VGA=8）：傳入自訂表時必須照表走，不能用內建的新版編號
        legacy = [(320, 240, 5), (640, 480, 8), (800, 600, 9)]
        assert framesize_value_for(640, 480, legacy) == (640, 480, 8)

    def test_device_table_only_offers_sizes_the_device_reports(self):
        table = _parse_device_resolutions(_DEVICE_STATUS)
        # 目標比裝置支援的最大尺寸還大 -> 退回最大的可用尺寸 SVGA，而不是編出一個裝置不認得的編號
        assert framesize_value_for(1280, 720, table) == (800, 600, 11)
        # 目標比最小的還小 -> 最小的可用尺寸 QVGA
        assert framesize_value_for(100, 100, table) == (320, 240, 6)


# ============================================================================
# _parse_device_resolutions()
# ============================================================================
class TestParseDeviceResolutions:
    def test_parses_reported_numbering(self):
        assert sorted(_parse_device_resolutions(_DEVICE_STATUS)) == [
            (320, 240, 6), (640, 480, 10), (800, 600, 11),
        ]

    @pytest.mark.parametrize(
        "status",
        [
            None,
            [],
            {},
            {"framesize": 10},                      # 原廠 CameraWebServer：沒有 resolutions
            {"resolutions": "vga"},
            {"resolutions": {}},
            {"resolutions": {"vga": "10"}},         # 非整數不採用
            {"resolutions": {"vga": True}},         # bool 不能被當成編號 1
        ],
    )
    def test_unusable_status_returns_none(self, status):
        assert _parse_device_resolutions(status) is None

    def test_partial_report_keeps_valid_entries(self):
        table = _parse_device_resolutions({"resolutions": {"vga": 10, "svga": "x"}})
        assert table == [(640, 480, 10)]


# ============================================================================
# configure_stream()：只驗證「實際送出去的 framesize 編號」，網路呼叫用 monkeypatch 取代
# ============================================================================
class TestConfigureStreamSendsCorrectNumber:
    URL = "http://192.168.0.120:81/stream"

    def _capture_requests(self, monkeypatch, device_table):
        sent = []
        monkeypatch.setattr(esp32cam, "_device_framesize_table", lambda base, timeout: device_table)
        monkeypatch.setattr(esp32cam, "_http_get", lambda url, timeout: sent.append(url) or 200)
        return sent

    def test_uses_device_reported_number(self, monkeypatch):
        sent = self._capture_requests(monkeypatch, _parse_device_resolutions(_DEVICE_STATUS))
        ok, detail = configure_stream(self.URL, 640, 480)
        assert ok
        assert sent == ["http://192.168.0.120:80/control?var=framesize&val=10"]
        assert "裝置回報" in detail

    def test_falls_back_to_builtin_table_when_device_gives_none(self, monkeypatch):
        sent = self._capture_requests(monkeypatch, None)
        ok, detail = configure_stream(self.URL, 640, 480)
        assert ok
        assert sent == ["http://192.168.0.120:80/control?var=framesize&val=10"]
        assert "內建對照表" in detail

    def test_legacy_device_numbering_is_respected(self, monkeypatch):
        # 舊版韌體自報 VGA=8：不能被內建新版編號（10）蓋掉
        legacy_status = {"resolutions": {"qvga": 5, "vga": 8, "svga": 9}}
        sent = self._capture_requests(monkeypatch, _parse_device_resolutions(legacy_status))
        configure_stream(self.URL, 640, 480)
        assert sent == ["http://192.168.0.120:80/control?var=framesize&val=8"]

    def test_quality_sent_only_when_non_negative(self, monkeypatch):
        sent = self._capture_requests(monkeypatch, None)
        configure_stream(self.URL, 640, 480, quality=-1)
        assert len(sent) == 1
        sent.clear()
        configure_stream(self.URL, 640, 480, quality=12)
        assert sent[-1].endswith("?var=quality&val=12")


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
