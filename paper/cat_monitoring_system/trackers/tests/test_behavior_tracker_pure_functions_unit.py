"""
Unit Test：ImprovedBehaviorTracker 的純狀態函式

第二階段（Unit Test）優先順序第 5 項。這裡測的是三個「給定明確輸入/狀態，
輸出完全確定，不依賴系統時鐘或 IO」的函式，用單一函式的獨立呼叫驗證，
不像 `test_behavior_tracker_regression.py` 那樣要跑一整段腳本化情境：

- `map_gcn_to_tracker()`：純 dict 查表
- `get_alerts()`：只讀 `self.behavior_time`/`self.behavior_count` 與
  `BehaviorTrackingConfig` 門檻常數，不依賴時鐘/IO
- `_settle_current_behavior(now, ...)`：`now` 由呼叫端明確傳入（不是內部呼叫
  `time.time()`），只修改 `self.*`，可以直接餵固定的 epoch 數字測試，不需要
  假時鐘

警報門檻一律從 `BehaviorTrackingConfig` 讀取（不寫死數字），這樣之後如果
調整門檻值，測試仍然正確反映「相對門檻」的觸發/不觸發邊界。
"""

import pytest

from config import BehaviorTrackingConfig, LoggingConfig
from trackers.behavior_tracker import ImprovedBehaviorTracker


@pytest.fixture
def isolated_tracker(tmp_path, monkeypatch):
    """建立一個不會碰到正式 tracker_state.json 的 tracker（乾淨初始狀態）。"""
    monkeypatch.setattr(
        LoggingConfig, "TRACKER_STATE_PATH", str(tmp_path / "test_tracker_state.json")
    )
    return ImprovedBehaviorTracker()


# ============================================================================
# map_gcn_to_tracker()
# ============================================================================


class TestMapGcnToTracker:
    @pytest.mark.parametrize(
        "behavior_id,expected_name",
        [(0, "walk"), (1, "lick"), (2, "scratch"), (3, "shake"), (4, "stop")],
    )
    def test_known_ids_map_to_correct_names(self, isolated_tracker, behavior_id, expected_name):
        assert isolated_tracker.map_gcn_to_tracker(behavior_id) == expected_name

    def test_unknown_id_defaults_to_walk(self, isolated_tracker):
        assert isolated_tracker.map_gcn_to_tracker(999) == "walk"

    def test_low_conf_sentinel_id_defaults_to_walk(self, isolated_tracker):
        """-1（LOW_CONF）/-2（NOT_VISIBLE）不在 BEHAVIOR_CATEGORIES 裡，應該也 fallback 到 walk
        （呼叫端在 update() 裡會提前用 if/elif 攔截這兩個特殊值，不會真的走到這裡，
        但函式本身面對這種輸入時的行為仍應該是安全的 fallback，不拋例外）。"""
        assert isolated_tracker.map_gcn_to_tracker(-1) == "walk"
        assert isolated_tracker.map_gcn_to_tracker(-2) == "walk"


# ============================================================================
# get_alerts()
# ============================================================================


class TestGetAlerts:
    def _set_behavior_time(self, tracker, **kwargs):
        for key, value in kwargs.items():
            tracker.behavior_time[key] = value

    def _set_behavior_count(self, tracker, **kwargs):
        for key, value in kwargs.items():
            tracker.behavior_count[key] = value

    def test_all_values_below_thresholds_produces_no_alerts(self, isolated_tracker):
        assert isolated_tracker.get_alerts() == []

    def test_scratch_time_over_threshold_triggers_high_level_alert(self, isolated_tracker):
        self._set_behavior_time(
            isolated_tracker,
            walk=BehaviorTrackingConfig.LOW_ACTIVITY_TIME_THRESHOLD_SECONDS + 1,
            scratch=BehaviorTrackingConfig.SCRATCH_ALERT_TIME_SECONDS + 0.1,
        )
        alerts = isolated_tracker.get_alerts()
        titles = {a["title"]: a for a in alerts}
        assert "搔抓時間異常" in titles
        assert titles["搔抓時間異常"]["level"] == "high"

    def test_scratch_time_exactly_at_threshold_does_not_trigger(self, isolated_tracker):
        """程式碼用嚴格 `>`，剛好等於門檻不算異常。"""
        self._set_behavior_time(
            isolated_tracker,
            walk=BehaviorTrackingConfig.LOW_ACTIVITY_TIME_THRESHOLD_SECONDS + 1,
            scratch=BehaviorTrackingConfig.SCRATCH_ALERT_TIME_SECONDS,
        )
        titles = {a["title"] for a in isolated_tracker.get_alerts()}
        assert "搔抓時間異常" not in titles

    def test_scratch_count_at_threshold_triggers_medium_alert_when_time_is_low(
        self, isolated_tracker
    ):
        """搔抓次數（非時長）達到門檻時，走 elif 分支觸發「頻率偏高」（用 >=）。"""
        self._set_behavior_time(
            isolated_tracker, walk=BehaviorTrackingConfig.LOW_ACTIVITY_TIME_THRESHOLD_SECONDS + 1
        )
        self._set_behavior_count(
            isolated_tracker, scratch=BehaviorTrackingConfig.SCRATCH_ALERT_COUNT_THRESHOLD
        )
        titles = {a["title"] for a in isolated_tracker.get_alerts()}
        assert "搔抓頻率偏高" in titles

    def test_scratch_time_alert_takes_priority_over_count_alert(self, isolated_tracker):
        """時長跟次數門檻同時達標時，只會觸發時長那個（if/elif 互斥）。"""
        self._set_behavior_time(
            isolated_tracker,
            walk=BehaviorTrackingConfig.LOW_ACTIVITY_TIME_THRESHOLD_SECONDS + 1,
            scratch=BehaviorTrackingConfig.SCRATCH_ALERT_TIME_SECONDS + 0.1,
        )
        self._set_behavior_count(
            isolated_tracker, scratch=BehaviorTrackingConfig.SCRATCH_ALERT_COUNT_THRESHOLD
        )
        alerts = isolated_tracker.get_alerts()
        titles = [a["title"] for a in alerts if "搔抓" in a["title"]]
        assert titles == ["搔抓時間異常"]

    def test_lick_time_over_threshold_triggers_alert(self, isolated_tracker):
        self._set_behavior_time(
            isolated_tracker,
            walk=BehaviorTrackingConfig.LOW_ACTIVITY_TIME_THRESHOLD_SECONDS + 1,
            lick=BehaviorTrackingConfig.LICK_ALERT_TIME_SECONDS + 0.1,
        )
        titles = {a["title"] for a in isolated_tracker.get_alerts()}
        assert "舔舐時間較長" in titles

    def test_shake_count_at_threshold_triggers_alert(self, isolated_tracker):
        self._set_behavior_time(
            isolated_tracker, walk=BehaviorTrackingConfig.LOW_ACTIVITY_TIME_THRESHOLD_SECONDS + 1
        )
        self._set_behavior_count(
            isolated_tracker, shake=BehaviorTrackingConfig.SHAKE_ALERT_COUNT_THRESHOLD
        )
        titles = {a["title"] for a in isolated_tracker.get_alerts()}
        assert "甩頭動作頻繁" in titles

    def test_stop_time_over_threshold_triggers_alert(self, isolated_tracker):
        self._set_behavior_time(
            isolated_tracker,
            walk=BehaviorTrackingConfig.LOW_ACTIVITY_TIME_THRESHOLD_SECONDS + 1,
            stop=BehaviorTrackingConfig.STOP_ALERT_TIME_SECONDS + 0.1,
        )
        titles = {a["title"] for a in isolated_tracker.get_alerts()}
        assert "長時間靜止不動" in titles

    def test_low_walk_time_triggers_low_activity_alert_only_when_total_time_positive(
        self, isolated_tracker
    ):
        """walk_time 低於門檻且完全沒有任何活動紀錄（total_time=0）時，不該觸發
        「活動度過低」——這代表根本還沒開始監測，不是活動度真的過低。"""
        assert isolated_tracker.get_alerts() == []  # 全部為 0，total_time=0，不觸發

        self._set_behavior_time(isolated_tracker, lick=1.0)  # 隨便一點點活動，total_time>0
        titles = {a["title"] for a in isolated_tracker.get_alerts()}
        assert "活動度過低" in titles  # walk_time 仍是 0 < 門檻

    def test_high_walk_time_does_not_trigger_low_activity_alert(self, isolated_tracker):
        self._set_behavior_time(
            isolated_tracker, walk=BehaviorTrackingConfig.LOW_ACTIVITY_TIME_THRESHOLD_SECONDS + 1
        )
        titles = {a["title"] for a in isolated_tracker.get_alerts()}
        assert "活動度過低" not in titles


# ============================================================================
# _settle_current_behavior()
# ============================================================================


class TestSettleCurrentBehavior:
    def test_returns_false_and_does_nothing_when_no_current_behavior(self, isolated_tracker):
        assert isolated_tracker.current_behavior is None
        settled = isolated_tracker._settle_current_behavior(now=100.0)
        assert settled is False
        assert isolated_tracker.behavior_count["walk"] == 0

    def test_settling_adds_elapsed_duration_and_increments_count(self, isolated_tracker):
        isolated_tracker.current_behavior = "walk"
        isolated_tracker.current_gcn_id = 0
        isolated_tracker.behavior_start_time = 100.0
        isolated_tracker.current_event_start_time = 100.0

        settled = isolated_tracker._settle_current_behavior(now=105.0)

        assert settled is True
        assert isolated_tracker.behavior_time["walk"] == pytest.approx(5.0)
        assert isolated_tracker.behavior_count["walk"] == 1
        assert isolated_tracker.current_behavior is None
        assert isolated_tracker.current_gcn_id is None

    def test_settling_appends_a_behavior_history_entry_with_full_event_duration(
        self, isolated_tracker
    ):
        """duration 記錄的是「完整事件時長」（now - current_event_start_time），
        跟 behavior_time 累積用的 `now - behavior_start_time` 是不同基準
        （事件可能經過多次定期累積，behavior_start_time 會被重置多次）。"""
        isolated_tracker.current_behavior = "lick"
        isolated_tracker.current_gcn_id = 1
        isolated_tracker.behavior_start_time = 103.0  # 假設已經定期累積過一次
        isolated_tracker.current_event_start_time = 100.0  # 但事件真正開始的時間點

        isolated_tracker._settle_current_behavior(now=110.0, next_activity_value=42)

        assert len(isolated_tracker.behavior_history) == 1
        entry = isolated_tracker.behavior_history[-1]
        assert entry["behavior"] == "lick"
        assert entry["gcn_behavior_id"] == 1
        assert entry["duration"] == pytest.approx(10.0)  # 110-100，不是 110-103
        assert entry["activity"] == 42

    def test_min_and_max_duration_track_shortest_and_longest_events(self, isolated_tracker):
        # 第一次事件：5 秒
        isolated_tracker.current_behavior = "walk"
        isolated_tracker.current_gcn_id = 0
        isolated_tracker.behavior_start_time = 0.0
        isolated_tracker.current_event_start_time = 0.0
        isolated_tracker._settle_current_behavior(now=5.0)
        assert isolated_tracker.behavior_min_duration["walk"] == pytest.approx(5.0)
        assert isolated_tracker.behavior_max_duration["walk"] == pytest.approx(5.0)

        # 第二次事件：更短（2 秒）－應更新 min，不動 max
        isolated_tracker.current_behavior = "walk"
        isolated_tracker.current_gcn_id = 0
        isolated_tracker.behavior_start_time = 10.0
        isolated_tracker.current_event_start_time = 10.0
        isolated_tracker._settle_current_behavior(now=12.0)
        assert isolated_tracker.behavior_min_duration["walk"] == pytest.approx(2.0)
        assert isolated_tracker.behavior_max_duration["walk"] == pytest.approx(5.0)

        # 第三次事件：更長（20 秒）－應更新 max，不動 min
        isolated_tracker.current_behavior = "walk"
        isolated_tracker.current_gcn_id = 0
        isolated_tracker.behavior_start_time = 20.0
        isolated_tracker.current_event_start_time = 20.0
        isolated_tracker._settle_current_behavior(now=40.0)
        assert isolated_tracker.behavior_min_duration["walk"] == pytest.approx(2.0)
        assert isolated_tracker.behavior_max_duration["walk"] == pytest.approx(20.0)

    def test_negative_duration_is_clamped_to_zero(self, isolated_tracker):
        """理論上不該發生（now 應該永遠 >= behavior_start_time），但程式碼有
        `max(0.0, duration)` 防護，驗證這個保護確實生效，不會讓累積時間變負數。"""
        isolated_tracker.current_behavior = "walk"
        isolated_tracker.current_gcn_id = 0
        isolated_tracker.behavior_start_time = 100.0
        isolated_tracker.current_event_start_time = 100.0

        isolated_tracker._settle_current_behavior(now=50.0)  # now < behavior_start_time

        assert isolated_tracker.behavior_time["walk"] == 0.0
