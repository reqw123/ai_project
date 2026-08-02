"""
Unit Test：config.py 的環境變數解析函式 + RunModeConfig

第二階段（Unit Test）優先順序第 3 項。這個模組只依賴 stdlib（os/datetime/
pathlib），完全不需要 GPU/YOLO/Flask/Node-RED/外部網路/真實攝影機，任何
環境都能跑。

環境變數解析函式（`_env_str`/`_env_int`/`_env_float`/`_env_bool`/
`_env_video_input`/`_env_size`）用 `monkeypatch.setenv()` 設定真實環境變數
後直接呼叫函式驗證；`RunModeConfig.is_within_active_window()` 因為讀的是
class 屬性（模組載入當下就從環境變數算好，不會因為之後改環境變數而改變），
改用 `monkeypatch.setattr()` 直接覆寫 `SCHEDULED_START_HHMM`/
`SCHEDULED_END_HHMM` 這兩個已解析好的屬性來測試各種排程情境。
"""

from datetime import datetime

import pytest

from config import (
    RunModeConfig,
    _env_bool,
    _env_float,
    _env_int,
    _env_size,
    _env_str,
    _env_video_input,
    _get_stgcn_feature_spec,
    _is_valid_port,
    _normalize_feature_mode,
    _parse_hhmm,
)

_ENV_NAME = "CAT_MONITORING_TEST_UNIT_VAR"  # 測試專用的環境變數名稱，不影響任何正式設定


# ============================================================================
# _env_str()
# ============================================================================


class TestEnvStr:
    def test_unset_variable_returns_default(self, monkeypatch):
        monkeypatch.delenv(_ENV_NAME, raising=False)
        assert _env_str(_ENV_NAME, "fallback") == "fallback"

    def test_set_variable_returns_stripped_value(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "  hello  ")
        assert _env_str(_ENV_NAME, "fallback") == "hello"

    def test_whitespace_only_value_returns_default(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "   ")
        assert _env_str(_ENV_NAME, "fallback") == "fallback"


# ============================================================================
# _env_int()
# ============================================================================


class TestEnvInt:
    def test_unset_variable_returns_default(self, monkeypatch):
        monkeypatch.delenv(_ENV_NAME, raising=False)
        assert _env_int(_ENV_NAME, 42) == 42

    def test_valid_integer_string_is_parsed(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "123")
        assert _env_int(_ENV_NAME, 42) == 123

    def test_invalid_string_returns_default(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "not_a_number")
        assert _env_int(_ENV_NAME, 42) == 42

    def test_float_looking_string_returns_default(self, monkeypatch):
        """int() 對 "12.5" 這種字串會直接拋例外，應該安全回退到 default。"""
        monkeypatch.setenv(_ENV_NAME, "12.5")
        assert _env_int(_ENV_NAME, 42) == 42


# ============================================================================
# _env_float()
# ============================================================================


class TestEnvFloat:
    def test_unset_variable_returns_default(self, monkeypatch):
        monkeypatch.delenv(_ENV_NAME, raising=False)
        assert _env_float(_ENV_NAME, 1.5) == 1.5

    def test_valid_float_string_is_parsed(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "3.14")
        assert _env_float(_ENV_NAME, 1.5) == pytest.approx(3.14)

    def test_invalid_string_returns_default(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "not_a_float")
        assert _env_float(_ENV_NAME, 1.5) == 1.5


# ============================================================================
# _env_bool()
# ============================================================================


class TestEnvBool:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "y", "on", "On"])
    def test_truthy_strings_return_true(self, monkeypatch, value):
        monkeypatch.setenv(_ENV_NAME, value)
        assert _env_bool(_ENV_NAME, False) is True

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "n", "off"])
    def test_falsy_strings_return_false(self, monkeypatch, value):
        monkeypatch.setenv(_ENV_NAME, value)
        assert _env_bool(_ENV_NAME, True) is False

    def test_unrecognized_string_returns_default(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "maybe")
        assert _env_bool(_ENV_NAME, True) is True
        assert _env_bool(_ENV_NAME, False) is False

    def test_unset_variable_returns_default(self, monkeypatch):
        monkeypatch.delenv(_ENV_NAME, raising=False)
        assert _env_bool(_ENV_NAME, True) is True


# ============================================================================
# _env_video_input()
# ============================================================================


class TestEnvVideoInput:
    def test_unset_variable_returns_default(self, monkeypatch):
        monkeypatch.delenv(_ENV_NAME, raising=False)
        assert _env_video_input(_ENV_NAME, "default.mp4") == "default.mp4"

    def test_numeric_string_becomes_camera_index_int(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "0")
        result = _env_video_input(_ENV_NAME, "default.mp4")
        assert result == 0
        assert isinstance(result, int)

    def test_non_numeric_string_stays_as_string(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, r"C:\videos\cat.mp4")
        result = _env_video_input(_ENV_NAME, "default.mp4")
        assert result == r"C:\videos\cat.mp4"
        assert isinstance(result, str)

    def test_empty_string_returns_default(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "")
        assert _env_video_input(_ENV_NAME, "default.mp4") == "default.mp4"


# ============================================================================
# _env_size()
# ============================================================================


class TestEnvSize:
    def test_unset_variable_returns_default(self, monkeypatch):
        monkeypatch.delenv(_ENV_NAME, raising=False)
        assert _env_size(_ENV_NAME, None) is None

    @pytest.mark.parametrize(
        "value", ["640x480", "640,480", "640 480", "640X480"]
    )
    def test_supported_separator_formats_all_parse_to_same_tuple(self, monkeypatch, value):
        monkeypatch.setenv(_ENV_NAME, value)
        assert _env_size(_ENV_NAME, None) == (640, 480)

    @pytest.mark.parametrize("value", ["none", "null", "off", "false", "NONE"])
    def test_explicit_disable_keywords_return_default(self, monkeypatch, value):
        monkeypatch.setenv(_ENV_NAME, value)
        assert _env_size(_ENV_NAME, None) is None

    def test_wrong_number_of_parts_returns_default(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "640,480,720")
        assert _env_size(_ENV_NAME, None) is None

    def test_non_numeric_parts_return_default(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "abc,def")
        assert _env_size(_ENV_NAME, None) is None

    def test_zero_or_negative_dimension_returns_default(self, monkeypatch):
        monkeypatch.setenv(_ENV_NAME, "0,480")
        assert _env_size(_ENV_NAME, None) is None
        monkeypatch.setenv(_ENV_NAME, "-640,480")
        assert _env_size(_ENV_NAME, None) is None


# ============================================================================
# _is_valid_port()
# ============================================================================


class TestIsValidPort:
    @pytest.mark.parametrize("port", [1, 80, 1880, 5000, 65535])
    def test_valid_ports_return_true(self, port):
        assert _is_valid_port(port) is True

    @pytest.mark.parametrize("port", [0, -1, 65536, 100000])
    def test_out_of_range_ports_return_false(self, port):
        assert _is_valid_port(port) is False

    def test_non_integer_type_returns_false(self):
        assert _is_valid_port("5000") is False
        assert _is_valid_port(5000.0) is False
        assert _is_valid_port(None) is False


# ============================================================================
# _parse_hhmm()
# ============================================================================


class TestParseHHMM:
    def test_valid_time_string_is_parsed(self):
        assert _parse_hhmm("06:00") == (6, 0)
        assert _parse_hhmm("23:59") == (23, 59)
        assert _parse_hhmm("00:00") == (0, 0)

    def test_empty_string_returns_none(self):
        assert _parse_hhmm("") is None

    def test_missing_colon_returns_none(self):
        assert _parse_hhmm("0600") is None

    def test_non_numeric_parts_return_none(self):
        assert _parse_hhmm("aa:bb") is None

    @pytest.mark.parametrize("value", ["24:00", "-1:00", "12:60", "12:-1"])
    def test_out_of_range_hour_or_minute_returns_none(self, value):
        assert _parse_hhmm(value) is None


# ============================================================================
# _normalize_feature_mode() / _get_stgcn_feature_spec()
# ============================================================================


class TestNormalizeFeatureMode:
    def test_lowercases_and_strips_whitespace(self):
        assert _normalize_feature_mode("  XY_Conf  ") == "xy_conf"

    def test_empty_string_defaults_to_xy(self):
        assert _normalize_feature_mode("") == "xy"


class TestGetStgcnFeatureSpec:
    @pytest.mark.parametrize(
        "mode,expected_channels",
        [
            ("xy", 2),
            ("xy_conf", 3),
            ("xy_conf_v", 5),
            ("xy_conf_v_bone", 7),
            ("xy_conf_v_bone_bmotion", 9),
        ],
    )
    def test_known_modes_return_correct_channel_count(self, mode, expected_channels):
        assert _get_stgcn_feature_spec(mode)["in_channels"] == expected_channels

    def test_unknown_mode_raises_value_error(self):
        with pytest.raises(ValueError):
            _get_stgcn_feature_spec("not_a_real_mode")


# ============================================================================
# RunModeConfig.is_within_active_window()
# ============================================================================


class TestRunModeConfigIsWithinActiveWindow:
    def test_no_schedule_configured_is_always_active(self, monkeypatch):
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_START_HHMM", None)
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_END_HHMM", None)
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 3, 0)) is True
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 23, 0)) is True

    def test_only_start_configured_is_inactive_before_start(self, monkeypatch):
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_START_HHMM", (6, 0))
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_END_HHMM", None)
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 5, 59)) is False

    def test_only_start_configured_is_active_at_and_after_start(self, monkeypatch):
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_START_HHMM", (6, 0))
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_END_HHMM", None)
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 6, 0)) is True
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 12, 0)) is True

    def test_only_start_configured_pauses_at_midnight_until_next_start(self, monkeypatch):
        """只設開始時間：實際（且經確認符合預期）的行為是「每天固定 [開始時刻, 24:00)
        運行、[00:00, 開始時刻) 暫停」的每日重複排程，不是字面上「觸發一次後永遠
        運行、不受跨天影響」——docstring 目前的措辭容易被誤解成後者，跟這裡驗證的
        實際行為不一致（純文件措辭問題，不是邏輯錯誤，已跟使用者確認過）。"""
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_START_HHMM", (6, 0))
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_END_HHMM", None)
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 23, 59)) is True
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 2, 0, 0)) is False
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 2, 5, 59)) is False
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 2, 6, 0)) is True

    def test_both_start_and_end_empty_means_always_active_continuously(self, monkeypatch):
        """若想讓系統一直運行、完全不受排程限制，開始與結束時間都留空（None）即可。"""
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_START_HHMM", None)
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_END_HHMM", None)
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 0, 0)) is True
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 2, 23, 59)) is True

    def test_same_day_window_is_active_inside_range(self, monkeypatch):
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_START_HHMM", (8, 0))
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_END_HHMM", (18, 0))
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 12, 0)) is True

    def test_same_day_window_is_inactive_outside_range(self, monkeypatch):
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_START_HHMM", (8, 0))
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_END_HHMM", (18, 0))
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 7, 0)) is False
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 19, 0)) is False

    def test_same_day_window_end_time_is_exclusive(self, monkeypatch):
        """區間定義為 [開始, 結束)，結束時刻本身不算在區間內。"""
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_START_HHMM", (8, 0))
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_END_HHMM", (18, 0))
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 18, 0)) is False
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 8, 0)) is True  # 開始時刻本身算在內

    def test_overnight_window_is_active_after_start_before_midnight(self, monkeypatch):
        """跨午夜區間（例如 22:00~06:00）：晚上時段應視為在區間內。"""
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_START_HHMM", (22, 0))
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_END_HHMM", (6, 0))
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 23, 0)) is True

    def test_overnight_window_is_active_after_midnight_before_end(self, monkeypatch):
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_START_HHMM", (22, 0))
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_END_HHMM", (6, 0))
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 3, 0)) is True

    def test_overnight_window_is_inactive_during_daytime_gap(self, monkeypatch):
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_START_HHMM", (22, 0))
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_END_HHMM", (6, 0))
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 12, 0)) is False

    def test_only_end_configured_treats_start_as_midnight(self, monkeypatch):
        """只設結束時間、沒設開始：開始時間視為當天 00:00。"""
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_START_HHMM", None)
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_END_HHMM", (12, 0))
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 0, 30)) is True
        assert RunModeConfig.is_within_active_window(datetime(2026, 1, 1, 13, 0)) is False

    def test_now_defaults_to_real_current_time_when_not_passed(self, monkeypatch):
        """不傳 now 參數時，應該退回使用真實系統時間（不拋例外即可，不驗證確切結果）。"""
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_START_HHMM", None)
        monkeypatch.setattr(RunModeConfig, "SCHEDULED_END_HHMM", None)
        assert RunModeConfig.is_within_active_window() is True
