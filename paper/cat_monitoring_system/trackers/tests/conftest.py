"""Ensure `cat_monitoring_system/` and `paper/` are importable as package roots.

Mirrors `analytics/tests/conftest.py` / `processors/tests/conftest.py`.
`behavior_tracker.py` does `from config import ...`, so we need `paper/`
(one level above `cat_monitoring_system/`) on sys.path as well.
"""

import sys
from pathlib import Path

import pytest

_cat_monitoring_system_dir = Path(__file__).resolve().parents[2]
_paper_dir = _cat_monitoring_system_dir.parent

for _p in (_cat_monitoring_system_dir, _paper_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 腳本情境（_scenario_utils.SCRIPT）每步最長推進 12 秒，要大於它才不會被當成中斷
SCENARIO_MAX_FRAME_GAP_SECONDS = 60.0


@pytest.fixture(autouse=True)
def _isolate_tracker_side_effects(tmp_path, monkeypatch):
    """每個測試都：
    - 把 daily_history.db 換成 tmp_path 底下的檔案——tracker 在 load_state() 讀到
      舊日期存檔、或跨日重置時會寫多天歷史，絕不能寫進正式資料庫；
    - 把幀間隔上限放寬到腳本步長以上（個別測試可再覆寫）。"""
    from config import BehaviorTrackingConfig, LoggingConfig

    db_path = str(tmp_path / "autouse_daily_history.db")
    monkeypatch.setattr(LoggingConfig, "DAILY_HISTORY_DB_PATH", db_path)
    monkeypatch.setattr(
        BehaviorTrackingConfig, "MAX_FRAME_GAP_SECONDS", SCENARIO_MAX_FRAME_GAP_SECONDS
    )
    yield
    from analytics import daily_store

    daily_store.close_connection(db_path)
