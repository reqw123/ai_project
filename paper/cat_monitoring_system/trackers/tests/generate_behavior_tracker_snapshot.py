"""
一次性腳本：跑過 `_scenario_utils.SCRIPT` 這段固定腳本，把 BehaviorTracker
的最終完整狀態凍結成 JSON 快照，供 test_behavior_tracker_regression.py 的
`test_scripted_scenario_matches_frozen_snapshot` 做回歸比對。

⚠ 這不是 pytest 測試，是「建立/更新基準快照」的手動工具。這段腳本的正確性
已經由 test_behavior_tracker_regression.py 裡其他明確斷言的測試驗證過
（duration/count/transition/alert 各自的手算期望值），所以這裡產生的快照
可以放心當作基準；但若之後修改了 BehaviorTracker 的邏輯、需要更新基準，
一樣要先確認新的輸出合理，才重新執行這支腳本覆蓋舊快照。

執行方式（不需要 GPU/真實模型，任何裝了本專案依賴的 Python 環境都能跑）：

    python trackers/tests/generate_behavior_tracker_snapshot.py

只會寫入自己的 snapshots/ 快照檔案，不會碰任何正式的 tracker_state.json
（見下方 LoggingConfig 攔截說明），也不修改任何正式程式碼。
"""

import json
import sys
from datetime import datetime
from pathlib import Path

_tests_dir = Path(__file__).resolve().parent
_cat_monitoring_system_dir = _tests_dir.parents[1]
_paper_dir = _cat_monitoring_system_dir.parent
for _p in (_cat_monitoring_system_dir, _paper_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import LoggingConfig  # noqa: E402
from trackers import behavior_tracker as behavior_tracker_module  # noqa: E402
from trackers.behavior_tracker import ImprovedBehaviorTracker  # noqa: E402

from _scenario_utils import (  # noqa: E402
    SNAPSHOT_PATH,
    FakeClock,
    full_state_snapshot,
    run_script,
)


def main() -> None:
    scratch_dir = _tests_dir / "_scratch"
    scratch_dir.mkdir(exist_ok=True)
    # 攔截正式的 tracker_state.json 路徑，避免讀到/寫到真實累積資料
    # （記憶體內暫時覆寫，腳本結束後沒有任何殘留效果，不影響 config.py 原始碼）。
    LoggingConfig.TRACKER_STATE_PATH = str(scratch_dir / "scratch_tracker_state.json")

    clock = FakeClock(datetime(2026, 1, 1, 8, 0, 0))
    # 直接覆寫模組層級的 time/datetime 名稱（跟 conftest 內 pytest monkeypatch
    # 的做法相同，只是這裡沒有 monkeypatch fixture 可用，手動做、腳本結束即失效）。
    behavior_tracker_module.time.time = clock.time
    behavior_tracker_module.datetime = clock

    tracker = ImprovedBehaviorTracker()
    run_script(tracker, clock)
    snapshot = full_state_snapshot(tracker)

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    print(f"已寫入快照：{SNAPSHOT_PATH}")
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
