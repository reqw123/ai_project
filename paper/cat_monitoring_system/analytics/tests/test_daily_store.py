"""analytics/daily_store.py 的單元測試。

全部用 tmp_path 帶入的獨立 db_path，不碰任何正式資料。
"""

from datetime import date

import pytest

from analytics import daily_store
from analytics.baseline import DailyRecord


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test_daily_history.db")
    yield path
    # daily_store 現在會依路徑快取連線（見 2026-08-11 的效能修正），測試
    # 結束後顯式關閉，避免 Windows 上檔案控制代碼佔用導致 tmp_path 清不掉。
    daily_store.close_connection(path)


def _mk(day, **overrides):
    kwargs = dict(day=day, monitoring_seconds=7200.0, walk_time=100.0, walk_count=5)
    kwargs.update(overrides)
    return DailyRecord(**kwargs)


# ── save_day / load_history / record_count ─────────────────────────────────


def test_save_and_load_round_trips_all_fields(db_path):
    record = DailyRecord(
        day=date(2026, 1, 1),
        monitoring_seconds=7200.0,
        walk_time=100.0,
        walk_count=5,
        stop_time=50.0,
        stop_count=3,
        lick_time=30.0,
        lick_count=2,
        scratch_time=10.0,
        scratch_count=1,
        shake_count=0,
        active_time=140.0,
        rest_time=50.0,
    )
    daily_store.save_day(record, db_path=db_path)

    history = daily_store.load_history(db_path=db_path)
    assert len(history) == 1
    got = history[0]
    assert got.day == record.day
    assert got.monitoring_seconds == record.monitoring_seconds
    assert got.walk_time == record.walk_time
    assert got.walk_count == record.walk_count
    assert got.lick_count == record.lick_count
    assert got.shake_count == record.shake_count


def test_save_day_upserts_same_day(db_path):
    daily_store.save_day(_mk(date(2026, 1, 1), walk_time=100.0), db_path=db_path)
    daily_store.save_day(_mk(date(2026, 1, 1), walk_time=999.0), db_path=db_path)

    history = daily_store.load_history(db_path=db_path)
    assert len(history) == 1
    assert history[0].walk_time == 999.0


def test_load_history_sorted_ascending(db_path):
    daily_store.save_day(_mk(date(2026, 1, 3)), db_path=db_path)
    daily_store.save_day(_mk(date(2026, 1, 1)), db_path=db_path)
    daily_store.save_day(_mk(date(2026, 1, 2)), db_path=db_path)

    history = daily_store.load_history(db_path=db_path)
    assert [r.day for r in history] == [
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 3),
    ]


def test_load_history_limit_days_keeps_most_recent(db_path):
    for i in range(1, 6):
        daily_store.save_day(_mk(date(2026, 1, i)), db_path=db_path)

    history = daily_store.load_history(db_path=db_path, limit_days=2)
    assert [r.day for r in history] == [date(2026, 1, 4), date(2026, 1, 5)]


def test_record_count(db_path):
    assert daily_store.record_count(db_path=db_path) == 0
    daily_store.save_day(_mk(date(2026, 1, 1)), db_path=db_path)
    daily_store.save_day(_mk(date(2026, 1, 2)), db_path=db_path)
    assert daily_store.record_count(db_path=db_path) == 2


# ── set_excluded / load_excluded_dates ──────────────────────────────────────


def test_no_exclusions_by_default(db_path):
    assert daily_store.load_excluded_dates(db_path=db_path) == []


def test_set_excluded_true_then_false(db_path):
    daily_store.set_excluded(date(2026, 1, 1), excluded=True, db_path=db_path)
    assert daily_store.load_excluded_dates(db_path=db_path) == ["2026-01-01"]

    daily_store.set_excluded(date(2026, 1, 1), excluded=False, db_path=db_path)
    assert daily_store.load_excluded_dates(db_path=db_path) == []


def test_set_excluded_accepts_iso_string_too(db_path):
    daily_store.set_excluded("2026-02-14", excluded=True, db_path=db_path)
    assert daily_store.load_excluded_dates(db_path=db_path) == ["2026-02-14"]


def test_set_excluded_true_twice_is_idempotent(db_path):
    daily_store.set_excluded(date(2026, 1, 1), excluded=True, db_path=db_path)
    daily_store.set_excluded(date(2026, 1, 1), excluded=True, db_path=db_path)
    assert daily_store.load_excluded_dates(db_path=db_path) == ["2026-01-01"]


def test_set_excluded_false_on_never_excluded_day_is_a_noop(db_path):
    daily_store.set_excluded(date(2026, 1, 1), excluded=False, db_path=db_path)
    assert daily_store.load_excluded_dates(db_path=db_path) == []


def test_exclusion_does_not_touch_daily_history_data(db_path):
    """排除只影響 excluded_dates 表，不會動到 daily_history 的原始統計數字
    ——跟 Node-RED 的 v2_excluded_dates 語意一致（排除，不是刪除）。"""
    daily_store.save_day(_mk(date(2026, 1, 1), walk_time=123.0), db_path=db_path)
    daily_store.set_excluded(date(2026, 1, 1), excluded=True, db_path=db_path)

    history = daily_store.load_history(db_path=db_path)
    assert len(history) == 1
    assert history[0].walk_time == 123.0


def test_excluded_dates_sorted_ascending(db_path):
    daily_store.set_excluded(date(2026, 1, 5), excluded=True, db_path=db_path)
    daily_store.set_excluded(date(2026, 1, 1), excluded=True, db_path=db_path)
    daily_store.set_excluded(date(2026, 1, 3), excluded=True, db_path=db_path)

    assert daily_store.load_excluded_dates(db_path=db_path) == [
        "2026-01-01",
        "2026-01-03",
        "2026-01-05",
    ]


def test_excluded_dates_feed_directly_into_compute_baseline(db_path):
    """端到端：exclude 一天之後，compute_baseline 真的少採用那天。"""
    from analytics.baseline import compute_baseline

    for i in range(1, 9):
        daily_store.save_day(
            _mk(date(2026, 1, i), monitoring_seconds=7200.0), db_path=db_path
        )
    history = daily_store.load_history(db_path=db_path)

    baseline_all = compute_baseline(history, min_days=7)
    assert baseline_all.days_count == 8

    daily_store.set_excluded(date(2026, 1, 1), excluded=True, db_path=db_path)
    excluded_dates = daily_store.load_excluded_dates(db_path=db_path)
    baseline_excl = compute_baseline(
        history, min_days=7, excluded_dates=excluded_dates
    )
    assert baseline_excl.days_count == 7
