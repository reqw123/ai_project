"""
Unit Test：server/routes.py 的純函式 `_daily_record_from_dict()` 與
`_dataclass_to_jsonable()`

第二階段（Unit Test）優先順序第 4 項。這兩個函式本身邏輯單純（dict/
dataclass 轉換），完全不用 Flask/GPU/Node-RED，但因為 `server/routes.py`
這支檔案在頂層 `import cv2`、`from flask import ...`、
`from processors.frame_processor import FrameProcessor`（間接需要 torch），
只要 import 這個模組就需要完整的 yolo_new 環境（cv2/flask/torch 都要裝），
即使測試本身完全不會用到 Flask app、不會發任何 HTTP 請求、不會做任何
GPU 推論。用 `pytest.importorskip("cv2")` 讓沒裝的環境安全跳過收集。
"""

from datetime import date

import pytest

pytest.importorskip(
    "cv2",
    reason="server/routes.py 檔案層級 import cv2/flask/torch 才能載入，此環境未安裝",
)

from analytics.baseline import Baseline
from server.routes import _daily_record_from_dict, _dataclass_to_jsonable

# ============================================================================
# _daily_record_from_dict()
# ============================================================================


class TestDailyRecordFromDict:
    def test_minimal_dict_uses_dataclass_defaults_for_missing_fields(self):
        record = _daily_record_from_dict({"date": "2026-07-01"})
        assert record.day == date(2026, 7, 1)
        assert record.walk_time == 0.0
        assert record.walk_count == 0
        assert record.monitoring_seconds == 0.0

    def test_full_dict_maps_every_field_correctly(self):
        record = _daily_record_from_dict(
            {
                "date": "2026-07-01",
                "monitoring_seconds": 7200,
                "walk_time": 120.5,
                "walk_count": 10,
                "stop_time": 300.0,
                "stop_count": 3,
                "lick_time": 45.0,
                "lick_count": 2,
                "scratch_time": 15.0,
                "scratch_count": 1,
                "shake_count": 4,
                "active_time": 180.5,
                "rest_time": 300.0,
            }
        )
        assert record.day == date(2026, 7, 1)
        assert record.monitoring_seconds == 7200
        assert record.walk_time == 120.5
        assert record.walk_count == 10
        assert record.stop_time == 300.0
        assert record.stop_count == 3
        assert record.lick_time == 45.0
        assert record.lick_count == 2
        assert record.scratch_time == 15.0
        assert record.scratch_count == 1
        assert record.shake_count == 4
        assert record.active_time == 180.5
        assert record.rest_time == 300.0

    def test_partial_fields_only_sets_the_ones_present(self):
        record = _daily_record_from_dict({"date": "2026-07-01", "walk_time": 99.0})
        assert record.walk_time == 99.0
        assert record.lick_time == 0.0  # 未提供，沿用預設值

    def test_unknown_extra_keys_are_silently_ignored(self):
        """dict 裡混入不認識的欄位（例如前端多送的 metadata）不該讓函式出錯。"""
        record = _daily_record_from_dict(
            {"date": "2026-07-01", "some_unknown_field": "whatever"}
        )
        assert record.day == date(2026, 7, 1)

    def test_iso_datetime_string_is_truncated_to_date_part(self):
        """呼叫端若送完整 ISO datetime（含時間），函式只取前 10 字元的日期部分。"""
        record = _daily_record_from_dict({"date": "2026-07-01T10:30:00"})
        assert record.day == date(2026, 7, 1)

    def test_non_iso_date_format_raises_value_error(self):
        """例如 toLocaleDateString('zh-TW') 產生的 "2026/7/1"，應明確報錯，
        不嘗試猜測格式（docstring 明確說明這是刻意設計，避免靜默解析錯誤）。"""
        with pytest.raises(ValueError):
            _daily_record_from_dict({"date": "2026/7/1"})

    def test_missing_date_key_raises_value_error(self):
        with pytest.raises(ValueError):
            _daily_record_from_dict({"walk_time": 10.0})

    def test_error_message_includes_the_offending_raw_value(self):
        """錯誤訊息應該包含原始收到的值，方便除錯，而不是只講「格式錯誤」。"""
        with pytest.raises(ValueError, match="2026/7/1"):
            _daily_record_from_dict({"date": "2026/7/1"})


# ============================================================================
# _dataclass_to_jsonable()
# ============================================================================


class TestDataclassToJsonable:
    def test_simple_dataclass_converts_to_plain_dict(self):
        from server.routes import _daily_record_from_dict

        record = _daily_record_from_dict({"date": "2026-07-01", "walk_time": 5.0})
        result = _dataclass_to_jsonable(record)
        assert isinstance(result, dict)
        assert result["walk_time"] == 5.0
        assert result["day"] == date(2026, 7, 1)

    def test_nested_dataclass_is_recursively_converted(self):
        """`Baseline` 內部的 `metrics` 欄位是 dict[str, MetricStats]（MetricStats
        本身也是 dataclass），dataclasses.asdict() 應該遞迴展開成純 dict/list。"""
        baseline = Baseline(
            computed_at="2026-07-01T00:00:00",
            days_count=7,
            required_days=7,
            confidence="High",
        )
        result = _dataclass_to_jsonable(baseline)
        assert isinstance(result, dict)
        assert result["days_count"] == 7
        assert result["confidence"] == "High"
        assert isinstance(result["metrics"], dict)
