"""dashboard/refresher.py 的單元測試（compute_and_cache_once()）。

只測試純函式邏輯，不測試 start_background_refresh() 真正啟動的執行緒/
time.sleep 迴圈——那部分是無限迴圈，測不出什麼，用人工檢查
（見 refresher.py docstring）取代自動化測試就夠了。

2026-08-29：`compute_and_cache_once()` 改從 `analytics/live_adapter` 讀即時
資料，不再延遲匯入 `server.routes`，因此本檔案不再需要 cv2/torch。即時
tracker 透過 `live_adapter._active_tracker`（正式碼由 `_build_frame_processor()`
用 `set_active_tracker()` 註冊）注入。
"""

from datetime import date, timedelta

import pytest

from analytics import daily_store, live_adapter
from analytics.baseline import DailyRecord
from config import LoggingConfig

from dashboard import cache
from dashboard.refresher import compute_and_cache_once


class _FakeTracker:
    def __init__(self, stats):
        self._stats = stats

    def get_today_stats(self):
        return self._stats


_DEFAULT_STATS = {
    "walk": 5, "walk_time": 100.0,
    "stop": 3, "stop_time": 50.0,
    "lick": 2, "lick_time": 30.0,
    "scratch": 1, "scratch_time": 10.0,
    "shake": 0,
}


@pytest.fixture(autouse=True)
def _reset_cache():
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_daily_history.db")
    monkeypatch.setattr(LoggingConfig, "DAILY_HISTORY_DB_PATH", db_path)
    yield db_path
    # daily_store 依路徑快取連線（見 2026-08-11 效能修正），測試結束後顯式
    # 關閉，避免 Windows 上檔案控制代碼佔用導致 tmp_path 清不掉。
    daily_store.close_connection(db_path)


def _seed_history(db_path, n=7):
    for i in range(n):
        daily_store.save_day(
            DailyRecord(
                day=date(2026, 1, 1) + timedelta(days=i),
                monitoring_seconds=7200,
                walk_time=100.0,
                walk_count=5,
                lick_time=50.0,
                lick_count=2,
            ),
            db_path=db_path,
        )


def test_no_frame_processor_returns_false_and_does_not_touch_cache(
    isolated_db, monkeypatch
):
    monkeypatch.setattr(live_adapter, "_active_tracker", None)
    ok = compute_and_cache_once()
    assert ok is False
    assert cache.get_latest() == {"status": "not_yet_computed"}


def test_insufficient_history_returns_false_but_writes_insufficient_status(
    isolated_db, monkeypatch
):
    monkeypatch.setattr(
        live_adapter, "_active_tracker", _FakeTracker(_DEFAULT_STATS)
    )
    _seed_history(isolated_db, n=2)  # 少於預設 min_days=7
    ok = compute_and_cache_once()
    assert ok is False
    result = cache.get_latest()
    assert result["status"] == "insufficient_data"
    assert result["current_days"] == 2


def test_sufficient_history_writes_ok_status_with_full_schema(
    isolated_db, monkeypatch
):
    monkeypatch.setattr(
        live_adapter, "_active_tracker", _FakeTracker(_DEFAULT_STATS)
    )
    _seed_history(isolated_db, n=7)
    ok = compute_and_cache_once()
    assert ok is True
    result = cache.get_latest()
    assert result["status"] == "ok"
    for key in ("baseline", "deviation", "fusion"):
        assert key in result
    # Class C 分析尚未搬進 Python（見 refresher.py 開頭說明），這裡刻意
    # 不去讀任何 Node-RED 資料，class_c_score 應該固定是 0。
    assert result["fusion"]["class_c_score"] == 0.0


def test_excluded_dates_from_daily_store_are_respected(isolated_db, monkeypatch):
    """透過 analytics/manage_baseline_history.py（或直接呼叫 daily_store.
    set_excluded()）排除的日期，背景排程重算時應該真的少採用那幾天，
    不是「存了但沒人讀」。"""
    monkeypatch.setattr(
        live_adapter, "_active_tracker", _FakeTracker(_DEFAULT_STATS)
    )
    _seed_history(isolated_db, n=8)

    ok = compute_and_cache_once()
    assert ok is True
    assert cache.get_latest()["baseline"]["days_count"] == 8

    daily_store.set_excluded(date(2026, 1, 1), excluded=True, db_path=isolated_db)
    ok = compute_and_cache_once()
    assert ok is True
    assert cache.get_latest()["baseline"]["days_count"] == 7


def test_does_not_depend_on_any_nodered_global_state(isolated_db, monkeypatch):
    """核心承諾的迴歸測試：整個計算過程不應該呼叫任何 Node-RED 相關的東西
    （沒有 HTTP 呼叫、沒有讀 global.json）。用「完全沒有網路/檔案系統以外
    的依賴」間接驗證——這裡改用『刻意不設定任何 Node-RED 端點/IP』的方式，
    確認純靠 daily_store + 即時 tracker 資料就能算出結果。"""
    monkeypatch.setattr(
        live_adapter, "_active_tracker", _FakeTracker(_DEFAULT_STATS)
    )
    _seed_history(isolated_db, n=10)
    ok = compute_and_cache_once()
    assert ok is True
    assert cache.get_latest()["status"] == "ok"


# ── _cache_is_fresh()：避免背景執行緒跟 Node-RED 觸發的 /api/deviation ──────
# ── 重複計算（2026-08-11 code review 發現的資源浪費問題）──────────────────


def test_cache_is_fresh_returns_false_when_never_cached():
    from dashboard.refresher import _cache_is_fresh

    cache.clear()
    assert _cache_is_fresh(max_age_sec=999.0) is False


def test_cache_is_fresh_true_within_window_false_after():
    from dashboard.refresher import _cache_is_fresh

    cache.clear()
    cache.set_latest({"status": "ok"})
    assert _cache_is_fresh(max_age_sec=999.0) is True
    assert _cache_is_fresh(max_age_sec=0.0) is False
