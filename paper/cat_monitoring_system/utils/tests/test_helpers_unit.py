"""
Unit Test：utils/helpers.py 的 Pure Function

第二階段（Unit Test）優先順序第 6 項。這個模組只依賴 `config.py`（stdlib）
與 `utils/constants.py`（純常數），不需要 GPU/YOLO/Flask/Node-RED，任何
環境都能跑。

只測真正的純函式：
- `is_stream_url()` —— 純 regex 比對
- `get_behavior_name()` —— 純邏輯，依賴的 `BehaviorTrackingConfig`/常數都是
  確定性讀取，不涉時鐘/IO

`resolve_video_source()` 只測「非 YouTube 網址原樣傳回」這條純函式路徑；
真正解析 YouTube 網址那條路徑需要呼叫 `yt_dlp` 對外發真實網路請求，違反
Unit Test「不得依賴外部網路」的規則，故不在這裡測試（那條路徑本質上是
IO，不是純函式）。

`get_ip()` 需要查詢真實網路介面/開 socket，不是純函式，同樣不在此列。
"""

import pytest

from utils.helpers import get_behavior_name, is_stream_url, resolve_video_source

# ============================================================================
# is_stream_url()
# ============================================================================


class TestIsStreamUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "http://example.com/stream",
            "https://example.com/stream",
            "rtsp://192.168.0.192:554/stream1",
            "rtsps://example.com/stream",
            "rtmp://example.com/live",
            "HTTP://example.com",  # 大小寫不敏感
            "RTSP://192.168.0.1/s",
        ],
    )
    def test_recognized_stream_schemes_return_true(self, url):
        assert is_stream_url(url) is True

    @pytest.mark.parametrize(
        "path",
        [
            r"C:\Users\homec\videos\cat.mp4",
            "./relative/path.mp4",
            "cat.mp4",
            "ftp://example.com/video.mp4",  # 不在支援清單內的協定
        ],
    )
    def test_local_paths_and_unsupported_schemes_return_false(self, path):
        assert is_stream_url(path) is False

    def test_non_string_input_returns_false(self):
        """攝影機來源用整數 index 表示，不是字串，應直接判定為 False。"""
        assert is_stream_url(0) is False
        assert is_stream_url(None) is False


# ============================================================================
# resolve_video_source()（只測非 YouTube 的純函式段落）
# ============================================================================


class TestResolveVideoSourceNonYoutubePassthrough:
    def test_camera_index_int_is_returned_unchanged(self):
        assert resolve_video_source(0) == 0

    def test_local_file_path_is_returned_unchanged(self):
        path = r"C:\Users\homec\videos\cat.mp4"
        assert resolve_video_source(path) == path

    def test_rtsp_url_is_returned_unchanged(self):
        url = "rtsp://192.168.0.192:554/stream1"
        assert resolve_video_source(url) == url

    def test_non_youtube_http_url_is_returned_unchanged(self):
        url = "http://example.com/some_other_video.mp4"
        assert resolve_video_source(url) == url

    @pytest.mark.parametrize(
        "youtube_looking_but_not_quite_url",
        [
            "https://notyoutube.com/watch?v=abc123",
            "https://www.youtube.com.evil.com/watch?v=abc",
        ],
    )
    def test_urls_that_merely_resemble_youtube_are_not_matched(
        self, youtube_looking_but_not_quite_url
    ):
        """regex 用 `^https?://(www\\.)?(youtube\\.com/...)`，網域必須精確匹配，
        不是「網址裡有出現 youtube 字樣」就算數，避免誤判相似網域。"""
        assert (
            resolve_video_source(youtube_looking_but_not_quite_url)
            == youtube_looking_but_not_quite_url
        )


# ============================================================================
# get_behavior_name()
# ============================================================================


class TestGetBehaviorName:
    @pytest.mark.parametrize(
        "behavior_id,expected_english",
        [(0, "walk"), (1, "lick"), (2, "scratch"), (3, "shake"), (4, "stop")],
    )
    def test_valid_id_without_use_text_returns_english_class_name(
        self, behavior_id, expected_english
    ):
        assert get_behavior_name(behavior_id, use_text=False) == expected_english

    @pytest.mark.parametrize(
        "behavior_id,expected_chinese",
        [(0, "走動"), (1, "舔舐"), (2, "搔抓"), (3, "甩頭"), (4, "靜止")],
    )
    def test_valid_id_with_use_text_returns_chinese_display_name(
        self, behavior_id, expected_chinese
    ):
        assert get_behavior_name(behavior_id, use_text=True) == expected_chinese

    def test_low_conf_id_returns_low_conf_text_regardless_of_use_text(self):
        assert get_behavior_name(-1, use_text=False) == "LOW_CONF"
        assert get_behavior_name(-1, use_text=True) == "LOW_CONF"

    @pytest.mark.parametrize("behavior_id", [5, 99, -2, -5])
    def test_out_of_range_id_returns_fallback(self, behavior_id):
        assert get_behavior_name(behavior_id, fallback="未知") == "未知"

    def test_custom_fallback_is_respected(self):
        assert get_behavior_name(999, fallback="自訂預設值") == "自訂預設值"

    def test_non_numeric_id_returns_fallback(self):
        assert get_behavior_name("not_a_number", fallback="未知") == "未知"
        assert get_behavior_name(None, fallback="未知") == "未知"

    def test_confidence_below_threshold_forces_low_conf_text(self):
        """即使 behavior_id 有效，confidence 低於門檻時仍應顯示 LOW_CONF。"""
        result = get_behavior_name(0, use_text=False, confidence=0.1)
        assert result == "LOW_CONF"

    def test_confidence_at_or_above_threshold_returns_normal_name(self):
        result = get_behavior_name(0, use_text=False, confidence=0.99)
        assert result == "walk"

    def test_confidence_none_skips_confidence_check(self):
        """未提供 confidence 時完全不檢查信心值，只看 behavior_id。"""
        assert get_behavior_name(0, use_text=False, confidence=None) == "walk"

    def test_malformed_confidence_is_silently_ignored(self):
        """confidence 給了無法轉成 float 的值時，該檢查被安全略過，不拋例外。"""
        result = get_behavior_name(0, use_text=False, confidence="not_a_float")
        assert result == "walk"
