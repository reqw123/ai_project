"""Shared scripted scenario + fake clock for BehaviorTracker regression tests.

見 `test_behavior_tracker_regression.py` 模組 docstring 的完整手算說明。
"""

from datetime import datetime, timedelta
from pathlib import Path

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "behavior_tracker_scripted_scenario.json"


class FakeClock:
    """可手動推進的假時鐘，同步提供 time.time() 與 datetime.now() 的假值。"""

    def __init__(self, start_datetime: datetime):
        self._dt = start_datetime

    def advance(self, seconds: float) -> None:
        self._dt += timedelta(seconds=seconds)

    def time(self) -> float:
        return self._dt.timestamp()

    def now(self, tz=None) -> datetime:
        return self._dt


# 腳本化情境：每一步是 (推進秒數, behavior_id, activity_value)。
SCRIPT = [
    (0.0, 0, 50),  # t=0  walk 開始
    (3.0, 0, 55),  # t=3  仍是 walk，定期累積 3.0s
    (2.0, 1, 60),  # t=5  切到 lick：結算 walk（duration=5.0, count=1），transition walk->lick
    (12.0, 1, 20),  # t=17 仍是 lick，定期累積 12.0s
    (3.0, 2, 10),  # t=20 切到 scratch：結算 lick（duration=15.0, count=1），transition lick->scratch
    (11.0, 2, 10),  # t=31 仍是 scratch，定期累積 11.0s
    (1.0, 4, 0),  # t=32 切到 stop：結算 scratch（duration=12.0, count=1），transition scratch->stop
    (1.0, -2, 0),  # t=33 貓消失：結算 stop（duration=1.0, count=1），not_detected_time+=1.0
    (2.0, -1, 0),  # t=35 低信心進入：low_conf_time+=2.0, low_conf_count=1
    (1.0, -1, 0),  # t=36 仍低信心：low_conf_time+=1.0（累積至 3.0），count 不變
    (1.0, 0, 45),  # t=37 回到 walk：transition stop->walk（跨過 NOT_VISIBLE/LOW_CONF 空窗）
]


def run_script(tracker, clock: FakeClock) -> None:
    for dt_seconds, behavior_id, activity_value in SCRIPT:
        clock.advance(dt_seconds)
        tracker.update(behavior_id, activity_value)


def full_state_snapshot(tracker) -> dict:
    return {
        "behavior_time": tracker.behavior_time,
        "behavior_count": tracker.behavior_count,
        "transition_matrix": tracker.transition_matrix,
        "not_detected_time": tracker.not_detected_time,
        "low_conf_time": tracker.low_conf_time,
        "low_conf_count": tracker.low_conf_count,
        "behavior_min_duration": tracker.behavior_min_duration,
        "behavior_max_duration": tracker.behavior_max_duration,
        "current_behavior": tracker.current_behavior,
        "alerts": [a["title"] for a in tracker.get_alerts()],
    }
