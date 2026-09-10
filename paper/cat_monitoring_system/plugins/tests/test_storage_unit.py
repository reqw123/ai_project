"""第二階段（M2）：LickStorage 事件表 / 視窗表持久化 + Session 結算。

不需要 cv2 —— storage.py 只用 sqlite3 / csv 標準庫。
"""

import os
import sqlite3

import pytest

from plugins.lick_stage.analysis_context import AnalysisContext
from plugins.lick_stage.storage import LickStorage


def test_disabled_storage_is_noop():
    st = LickStorage(db_path=None)
    assert st.enabled is False
    st.open_session(AnalysisContext(session_id="S0"))
    st.write_event({"duration_sec": 1.0, "zone_l1": "TORSO"})
    st.write_window({"window_start": 0.0})
    st.close_session()  # 不應拋例外
    st.close()


def test_events_and_windows_persist_and_session_closes_cleanly(tmp_path):
    db = str(tmp_path / "lick.db")
    st = LickStorage(db_path=db)
    assert st.enabled is True

    ctx = AnalysisContext(
        session_id="S20260910_001",
        video_id="cam01.mp4",
        cat_id="cat_A",
        period="PM",
        model_version="stgcn_v4",
        config_hash="sha256:abc",
        source_fps=30.0,
    )
    st.open_session(ctx)
    st.write_event(
        {
            "start_source_ts": 1.0,
            "end_source_ts": 2.0,
            "duration_sec": 1.0,
            "start_frame": 30,
            "end_frame": 60,
            "zone_l1": "TORSO",
            "zone_l2": None,
            "zone_switch_count": 0,
            "assigned_ratio": 1.0,
            "action_score_mean": 0.9,
            "action_score_min": 0.85,
            "raw_bout": True,
        }
    )
    st.write_event({"duration_sec": 0.5, "zone_l1": "FORELIMB", "raw_bout": True})
    st.write_window(
        {
            "window_start": 0.0,
            "window_end": 60.0,
            "window_sec": 60.0,
            "period": "PM",
            "observed_sec": 60.0,
            "valid_observed_sec": 55.0,
            "stgcn_lick_sec": 12.0,
            "assigned_zone_sec": 10.0,
            "unassigned_lick_sec": 2.0,
            "bout_count": 2,
            "torso_sec": 8.0,
            "forelimb_sec": 2.0,
            "zone_coverage_ratio": 0.833,
        }
    )
    st.close_session()
    st.close()

    conn = sqlite3.connect(db)
    events = conn.execute(
        "SELECT event_id, session_id, cat_id, video_id, zone_l1, duration_sec, raw_bout "
        "FROM lick_events ORDER BY id"
    ).fetchall()
    assert len(events) == 2
    assert events[0][0] == "S20260910_001_E00001"
    assert events[0][1] == "S20260910_001"
    assert events[0][2] == "cat_A"  # 由 context 補上
    assert events[0][4] == "TORSO"
    assert events[0][6] == 1  # raw_bout → 1

    windows = conn.execute(
        "SELECT session_id, stgcn_lick_sec, zone_coverage_ratio FROM lick_window_summary"
    ).fetchall()
    assert len(windows) == 1
    assert windows[0][1] == pytest.approx(12.0)

    sess = conn.execute(
        "SELECT event_count, window_count, closed_cleanly, finished_at "
        "FROM lick_sessions WHERE session_id=?",
        ("S20260910_001",),
    ).fetchone()
    assert sess[0] == 2 and sess[1] == 1 and sess[2] == 1
    assert sess[3] is not None
    conn.close()

    # CSV 匯出
    assert os.path.exists(str(tmp_path / "lick.lick_events.csv"))
    assert os.path.exists(str(tmp_path / "lick.lick_window_summary.csv"))


def test_bad_db_path_degrades_to_disabled(tmp_path):
    # 指到一個「父路徑是檔案」的無效位置
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    st = LickStorage(db_path=str(blocker / "nested" / "lick.db"))
    assert st.enabled is False
    st.write_event({"duration_sec": 1.0})  # 不拋例外
