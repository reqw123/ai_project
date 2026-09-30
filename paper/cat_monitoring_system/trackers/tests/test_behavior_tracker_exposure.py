"""
BehaviorTracker 曝露時間（分母）與跨日持久化（2026-09-29）

定義：
  系統運行時間 run_seconds        ＝ 逐幀累加的 dt（超過 MAX_FRAME_GAP_SECONDS 的空白不算）
  貓在畫面時間 monitoring_seconds ＝ 五類行為 + low_conf 幀的 dt
  貓不在畫面   not_detected_time  ＝ NOT_VISIBLE 幀的 dt
  恆等式：run_seconds == monitoring_seconds + not_detected_time
  low_conf 再拆 warmup / uncertain / sqa 三種來源。
"""

import json
from datetime import datetime

import pytest

from config import BehaviorTrackingConfig, LoggingConfig
from trackers import behavior_tracker as behavior_tracker_module
from trackers.behavior_tracker import ImprovedBehaviorTracker, summarize_periods

from _scenario_utils import FakeClock

WALK, LICK, LOW, GONE = 0, 1, -1, -2


@pytest.fixture
def clock():
    return FakeClock(datetime(2026, 1, 1, 20, 0, 0))


@pytest.fixture
def make_tracker(tmp_path, monkeypatch, clock):
    monkeypatch.setattr(LoggingConfig, "TRACKER_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(behavior_tracker_module.time, "time", clock.time)
    monkeypatch.setattr(behavior_tracker_module, "datetime", clock)
    return ImprovedBehaviorTracker


def _feed(tracker, clock, steps):
    for dt, bid, reason in steps:
        clock.advance(dt)
        tracker.update(bid, 10, low_conf_reason=reason)


def test_run_equals_visible_plus_not_detected(make_tracker, clock):
    t = make_tracker()
    _feed(t, clock, [
        (1, WALK, None), (1, WALK, None),          # 在畫面 2
        (1, GONE, None), (2, GONE, None),          # 不在畫面 3
        (1, LOW, "warmup"), (1, LOW, "uncertain"), (1, LOW, "sqa"), (1, LOW, None),  # 在畫面 4
        (1, LICK, None),                           # 在畫面 1
    ])
    assert t.run_seconds == pytest.approx(10.0)
    assert t.monitoring_seconds == pytest.approx(7.0)
    assert t.not_detected_time == pytest.approx(3.0)
    assert t.run_seconds == pytest.approx(t.monitoring_seconds + t.not_detected_time)
    # low_conf 拆三種來源；沒給來源的歸 uncertain
    assert t.low_conf_time == pytest.approx(4.0)
    assert t.low_conf_breakdown == pytest.approx({"warmup": 1.0, "uncertain": 2.0, "sqa": 1.0})

    stats = t.get_today_stats()
    assert stats["run_seconds"] == stats["total_uptime"] == pytest.approx(10.0)
    assert stats["monitoring_seconds"] == stats["cat_visible_time"] == pytest.approx(7.0)
    assert stats["date"] == "2026-01-01"
    assert stats["low_conf_breakdown"]["sqa"] == pytest.approx(1.0)

    bucket = stats["hourly_distribution"]["20"]
    assert bucket["monitoring_sec"] == pytest.approx(10.0)
    assert bucket["visible_sec"] == pytest.approx(7.0)
    assert bucket["not_detected_sec"] == pytest.approx(3.0)
    assert bucket["low_conf_sec"] == pytest.approx(4.0)


def test_gap_longer_than_limit_counts_nothing_and_ends_event_at_last_frame(
    make_tracker, clock, monkeypatch
):
    """排程暫停／串流卡住：空白不算運行時間，進行中的 walk 在上一幀結束，不吃進空白。"""
    monkeypatch.setattr(BehaviorTrackingConfig, "MAX_FRAME_GAP_SECONDS", 5.0)
    t = make_tracker()
    _feed(t, clock, [(1, WALK, None), (1, WALK, None), (1, WALK, None)])  # walk 3 秒
    _feed(t, clock, [(2 * 3600, WALK, None)])  # 暫停 2 小時後恢復（仍同一天），仍是 walk
    _feed(t, clock, [(1, WALK, None)])

    assert t.run_seconds == pytest.approx(4.0)  # 3 + 恢復後 1，空白 0
    assert t.monitoring_seconds == pytest.approx(4.0)
    assert t.behavior_count["walk"] == 1  # 中斷前那段已結算成 1 個事件
    assert t.behavior_max_duration["walk"] == pytest.approx(2.0)  # 事件從第 1 幀到第 3 幀
    assert t.behavior_time["walk"] <= 3.0 + 1e-6  # 沒有把 2 小時灌進去
    total_hourly = sum(b.get("monitoring_sec", 0) for b in t.hourly_distribution.values())
    assert total_hourly == pytest.approx(4.0)


def test_summarize_periods_sums_hours_into_four_periods():
    hourly = {
        "18": {"walk": 10, "lick": 5, "monitoring_sec": 100, "visible_sec": 60,
               "not_detected_sec": 40, "low_conf_sec": 3},
        "23": {"stop": 7, "monitoring_sec": 50, "visible_sec": 50},
        "02": {"walk": 1, "monitoring_sec": 2, "visible_sec": 2},
    }
    p = summarize_periods(hourly)
    assert set(p) == {"00-06", "06-12", "12-18", "18-24"}
    assert p["18-24"]["monitoring_sec"] == 150
    assert p["18-24"]["visible_sec"] == 110
    assert p["18-24"]["not_detected_sec"] == 40
    assert p["18-24"]["low_conf_sec"] == 3
    assert p["18-24"]["total"] == 22
    assert p["00-06"]["walk"] == 1
    assert p["06-12"]["monitoring_sec"] == 0


def test_stale_state_from_a_day_stopped_before_midnight_is_persisted(make_tracker, clock):
    """只錄 18~23 點、午夜前關掉程式：隔天啟動時要把前一天補寫進 daily_history.db。"""
    from analytics import daily_store

    t = make_tracker()
    _feed(t, clock, [(1, WALK, None)] * 5 + [(1, GONE, None)] * 2 + [(1, LOW, "sqa")] * 3)
    t.save_state()
    day_one = clock.now().date()

    clock.advance(20 * 3600)  # 隔天 16:00 才重開
    fresh = make_tracker()
    assert fresh.run_seconds == 0.0  # 不還原舊日期

    history = daily_store.load_history()
    assert [r.day for r in history] == [day_one]
    rec = history[0]
    assert rec.run_seconds == pytest.approx(10.0)
    assert rec.monitoring_seconds == pytest.approx(8.0)
    assert rec.not_detected_time == pytest.approx(2.0)
    assert rec.low_conf_sqa_time == pytest.approx(3.0)
    assert rec.periods["18-24"]["visible_sec"] == pytest.approx(8.0)
    assert rec.periods["18-24"]["not_detected_sec"] == pytest.approx(2.0)


def test_empty_stale_state_is_not_persisted(make_tracker, clock, tmp_path):
    from analytics import daily_store

    with open(LoggingConfig.TRACKER_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"date": "2025-12-31", "behavior_time": {}, "monitoring_seconds": 0}, f)
    make_tracker()
    assert daily_store.load_history() == []


def test_midnight_reset_persists_new_fields(make_tracker, clock):
    """程式開著跑過午夜：check_daily_reset() 寫入的紀錄也要帶新欄位與時段。"""
    from analytics import daily_store

    t = make_tracker()
    _feed(t, clock, [(1, LICK, None)] * 4 + [(1, GONE, None)])
    day_one = clock.now().date()
    clock.advance(24 * 3600)
    t.get_today_stats()

    rec = daily_store.load_history()[0]
    assert rec.day == day_one
    assert rec.run_seconds == pytest.approx(5.0)
    assert rec.monitoring_seconds == pytest.approx(4.0)
    assert rec.not_detected_time == pytest.approx(1.0)
    assert rec.periods["18-24"]["monitoring_sec"] == pytest.approx(5.0)
    assert t.run_seconds == 0.0


def test_day_with_no_time_at_all_is_not_persisted_on_midnight_reset(make_tracker, clock):
    """管線整天沒處理任何畫面就跨日（例如只開著 Flask）：不寫空白天，跟 Node-RED 同一條規則。"""
    from analytics import daily_store

    t = make_tracker()
    clock.advance(24 * 3600)
    t.get_today_stats()  # 觸發跨日重置
    assert daily_store.load_history() == []


def test_short_day_is_still_persisted(make_tracker, clock):
    """只跑了幾秒也照寫（不設時數門檻）；能不能進基線交給 compute_baseline 的門檻。"""
    from analytics import daily_store

    t = make_tracker()
    _feed(t, clock, [(1, WALK, None)] * 3)
    clock.advance(24 * 3600)
    t.get_today_stats()
    assert [r.run_seconds for r in daily_store.load_history()] == [pytest.approx(3.0)]


# ── 2026-09-30：本機錄影改用影片時間（docs/錄影推論改用影片時間-待辦.md）──────────


class _MediaClock:
    """模擬「錄影開始時間 + 影片位置」：跟電腦時鐘（FakeClock）完全無關。"""

    def __init__(self, start):
        self.ts = start.timestamp()

    def __call__(self):
        return self.ts


def _run_recording(make_tracker, clock, wall_sec_per_frame, *, hours=6, fps=2,
                   start=datetime(2026, 9, 30, 18, 0, 0)):
    """處理一段從 start 開始、hours 小時、fps 幀/秒的錄影；wall_sec_per_frame＝電腦處理一幀
    花的時間（模擬處理快慢）。行為：前半 walk、後半 lick，每 10 分鐘有 1 分鐘貓不在畫面。"""
    media = _MediaClock(start)
    t = make_tracker()
    t.enable_media_clock(media)
    n = int(hours * 3600 * fps)
    for i in range(n):
        media.ts = start.timestamp() + i / fps
        clock.advance(wall_sec_per_frame)
        sec = i / fps
        if (sec % 600) >= 540:
            bid = GONE
        else:
            bid = WALK if sec < hours * 1800 else LICK
        t.update(bid, 10)
    return t


def test_media_clock_results_do_not_depend_on_processing_speed(make_tracker, clock, tmp_path):
    """同一段 6 小時錄影，電腦處理得很快（每幀 0.001 秒）或很慢（每幀 0.9 秒、甚至超過
    影片本身）→ 時長、每小時分桶、日期全部相同。"""
    fast = _run_recording(make_tracker, clock, 0.001).get_today_stats()
    slow = _run_recording(make_tracker, clock, 0.9).get_today_stats()
    for key in ("run_seconds", "monitoring_seconds", "not_detected_time",
                "walk_time", "lick_time", "date"):
        assert fast[key] == slow[key], key
    assert fast["hourly_distribution"] == slow["hourly_distribution"]

    # 影片時間正確：6 小時 − 最後一幀，每 10 分鐘有 1 分鐘不在畫面
    assert fast["run_seconds"] == pytest.approx(6 * 3600 - 0.5, abs=0.1)
    assert fast["not_detected_time"] == pytest.approx(36 * 60, abs=1.0)
    assert fast["date"] == "2026-09-30"
    assert sorted(fast["hourly_distribution"]) == ["18", "19", "20", "21", "22", "23"]
    for h, b in fast["hourly_distribution"].items():
        assert b["monitoring_sec"] == pytest.approx(3600, abs=1.0), h


def test_media_clock_does_not_touch_state_file_and_finish_writes_recording_day(
    make_tracker, clock
):
    from analytics import daily_store

    t = _run_recording(make_tracker, clock, 0.01, hours=1)
    assert not __import__("os").path.exists(LoggingConfig.TRACKER_STATE_PATH)  # 不讀寫存檔

    day = t.finish_media_day()
    assert str(day) == "2026-09-30"  # 錄影日期，不是電腦的日期（FakeClock 是 2026-01-01）
    rec = daily_store.load_history()[0]
    assert str(rec.day) == "2026-09-30"
    assert rec.run_seconds == pytest.approx(3600 - 0.5, abs=0.1)
    assert rec.periods["18-24"]["monitoring_sec"] == pytest.approx(3600, abs=1.0)
    assert rec.lick_time > 0 and rec.walk_time > 0


def test_enable_media_clock_starts_clean_even_after_restoring_live_state(make_tracker, clock):
    """__init__ 可能已還原今天的即時監測資料；切到影片時間要從 0 開始，而且存檔不被覆蓋。"""
    live = make_tracker()
    _feed(live, clock, [(1, WALK, None)] * 5)
    live.save_state()
    saved = open(LoggingConfig.TRACKER_STATE_PATH, encoding="utf-8").read()

    t = make_tracker()
    assert t.run_seconds == pytest.approx(5.0)  # 同一天：還原了即時監測資料
    media = _MediaClock(datetime(2026, 9, 30, 18, 0, 0))
    t.enable_media_clock(media)
    assert t.run_seconds == 0.0 and t.behavior_time["walk"] == 0.0 and t.hourly_distribution == {}
    media.ts += 1
    t.update(WALK, 10)
    t.save_state()
    assert open(LoggingConfig.TRACKER_STATE_PATH, encoding="utf-8").read() == saved


def test_media_clock_recording_that_crosses_midnight_splits_days(make_tracker, clock):
    """錄影稍微超過 24:00：跨過午夜那一刻封存 9/30，之後算 10/1。"""
    from analytics import daily_store

    start = datetime(2026, 9, 30, 23, 59, 50)
    media = _MediaClock(start)
    t = make_tracker()
    t.enable_media_clock(media)
    for i in range(41):  # 23:59:50 → 00:00:10，每 0.5 秒一幀
        media.ts = start.timestamp() + i * 0.5
        t.update(WALK, 10)
    assert [str(r.day) for r in daily_store.load_history()] == ["2026-09-30"]
    assert t.get_today_stats()["date"] == "2026-10-01"
    assert t.run_seconds == pytest.approx(10.0, abs=0.6)


def test_recording_day_is_not_overwritten_by_stale_live_state_on_later_restarts(
    make_tracker, clock
):
    """9/30 白天用測試影片（電腦時鐘）累積了存檔 → 10/1 用影片時間跑 9/30 的錄影、播完寫 DB
    → 10/2 再開程式：舊的 9/30 存檔不能再被補寫一次、把錄影紀錄蓋掉。"""
    import os
    from analytics import daily_store

    clock._dt = datetime(2026, 9, 30, 9, 0, 0)
    live = make_tracker()
    _feed(live, clock, [(1, WALK, None)] * 7)  # 即時／測試資料：運行 7 秒
    live.save_state()

    clock._dt = datetime(2026, 10, 1, 10, 0, 0)
    t = make_tracker()  # 啟動：補寫 9/30 舊存檔，並把它改名封存
    assert [r.run_seconds for r in daily_store.load_history()] == [pytest.approx(7.0)]
    assert not os.path.exists(LoggingConfig.TRACKER_STATE_PATH)
    assert os.path.exists(LoggingConfig.TRACKER_STATE_PATH + ".persisted-2026-09-30")

    start = datetime(2026, 9, 30, 18, 0, 0)
    media = _MediaClock(start)
    t.enable_media_clock(media)
    for i in range(121):  # 錄影 60 秒
        media.ts = start.timestamp() + i * 0.5
        t.update(LICK, 10)
    t.finish_media_day()
    assert daily_store.load_history()[0].run_seconds == pytest.approx(60.0)

    clock._dt = datetime(2026, 10, 2, 10, 0, 0)
    make_tracker()  # 再開一次程式
    rows = daily_store.load_history()
    assert len(rows) == 1 and rows[0].run_seconds == pytest.approx(60.0)  # 錄影版沒被蓋掉
