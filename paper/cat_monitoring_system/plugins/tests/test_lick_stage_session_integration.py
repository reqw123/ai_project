"""第二階段（M2）整合：跑一整段 Session（LickStagePlugin）並確認
事件 / 視窗 / session row 落地，且 session close 無遺失。
"""

import sqlite3

import numpy as np
import pytest

pytest.importorskip("cv2", reason="plugins/lick_stage 需要 cv2")

from plugins.lick_stage.config import LickConfig as _C
from plugins.lick_stage.manager import LickStagePlugin

NUM_JOINTS = 17
_FPS = 30.0


def _pose():
    kpts = np.zeros((NUM_JOINTS, 2), dtype=np.float64)
    kpts[_C.KP_NOSE] = (0.0, 120.0)
    kpts[_C.KP_LEFT_EAR] = (-18.0, -6.0)
    kpts[_C.KP_RIGHT_EAR] = (18.0, -6.0)
    kpts[_C.KP_CHEST] = (0.0, 0.0)
    kpts[_C.KP_MID_BACK] = (0.0, 180.0)
    kpts[_C.KP_HIP] = (0.0, 360.0)
    return kpts, np.ones(NUM_JOINTS, dtype=np.float64)


def test_full_session_persists_events_and_windows(tmp_path):
    db = str(tmp_path / "sess.db")
    plugin = LickStagePlugin(nodered_url=None, storage_db_path=db)
    plugin.start_session(
        "S_ITEST", video_id="clip.mp4", cat_id="cat_A", period="PM", source_fps=_FPS
    )

    kpts, conf = _pose()
    f = 0
    # 3 段舔毛，中間插非舔毛，間隔長度刻意超過 config.py 的
    # GAP_TOLERANCE_SEC（M6 起 manager.py 真的接上 action_gate.py，見
    # _make_event_aggregator()）——確保 3 段真的形成 3 個獨立事件，不會被
    # M6 的短暫中斷合併機制吸收掉；驗證「短間隔會被合併」是另一個測試
    # （test_short_gap_across_session_merges_into_one_event）的工作，兩者
    # 刻意分開，各自驗證一種行為。
    _gap_frames = int(_C.GAP_TOLERANCE_SEC * _FPS) + 5  # 明確超過門檻
    for _bout in range(3):
        for _ in range(20):  # 每段 20 幀 ≈ 0.67 秒
            plugin.update(
                kpts, conf, source_timestamp=f / _FPS, frame_idx=f,
                cat_present=True, is_lick=True, lick_confidence=0.9,
                session_id="S_ITEST",
            )
            f += 1
        for _ in range(_gap_frames):  # 非舔毛間隔（超過 GAP_TOLERANCE_SEC）
            plugin.update(
                None, None, source_timestamp=f / _FPS, frame_idx=f,
                cat_present=True, is_lick=False, session_id="S_ITEST",
            )
            f += 1

    # 最後一段舔毛不接非舔毛就直接結束 —— finish_session 必須把它結算掉
    for _ in range(15):
        plugin.update(
            kpts, conf, source_timestamp=f / _FPS, frame_idx=f,
            cat_present=True, is_lick=True, lick_confidence=0.9, session_id="S_ITEST",
        )
        f += 1

    plugin.finish_session(end_source_timestamp=f / _FPS)
    plugin.close()

    conn = sqlite3.connect(db)
    events = conn.execute(
        "SELECT event_id, zone_l1, duration_sec, start_frame, end_frame FROM lick_events ORDER BY id"
    ).fetchall()
    # 3 個間隔事件 + 1 個結尾事件（靠 finish_session 結算）= 4
    assert len(events) == 4, events
    # 每個事件都有完整的起訖 frame 與正的 duration
    for _eid, _l1, dur, sf, ef in events:
        assert dur > 0.0
        assert ef >= sf

    windows = conn.execute(
        "SELECT stgcn_lick_sec, assigned_zone_sec, unassigned_lick_sec, "
        "no_cat_sec, observed_sec FROM lick_window_summary ORDER BY id"
    ).fetchall()
    assert len(windows) >= 1
    # 不變式：stgcn == assigned + unassigned（逐視窗）
    for stg, asg, una, _nc, _obs in windows:
        assert stg == pytest.approx(asg + una, abs=1e-6)

    sess = conn.execute(
        "SELECT event_count, closed_cleanly FROM lick_sessions WHERE session_id=?",
        ("S_ITEST",),
    ).fetchone()
    assert sess == (4, 1)
    conn.close()


def test_short_gap_across_session_merges_into_one_event(tmp_path):
    """M6：跟上面那個測試互補——這裡間隔刻意設在 GAP_TOLERANCE_SEC 之內，
    驗證 manager.py 真的把 config.py 的值接進 EventAggregator/ActionGate
    （見 _make_event_aggregator()），不是只有寫好但沒接上。"""
    db = str(tmp_path / "sess.db")
    plugin = LickStagePlugin(nodered_url=None, storage_db_path=db)
    plugin.start_session(
        "S_MERGE", video_id="clip.mp4", cat_id="cat_A", period="PM", source_fps=_FPS
    )

    kpts, conf = _pose()
    f = 0
    _gap_frames = max(1, int(_C.GAP_TOLERANCE_SEC * _FPS) - 5)  # 明確低於門檻
    for _bout in range(3):
        for _ in range(20):
            plugin.update(
                kpts, conf, source_timestamp=f / _FPS, frame_idx=f,
                cat_present=True, is_lick=True, lick_confidence=0.9,
                session_id="S_MERGE",
            )
            f += 1
        for _ in range(_gap_frames):
            plugin.update(
                None, None, source_timestamp=f / _FPS, frame_idx=f,
                cat_present=True, is_lick=False, session_id="S_MERGE",
            )
            f += 1

    plugin.finish_session(end_source_timestamp=f / _FPS)
    plugin.close()

    conn = sqlite3.connect(db)
    events = conn.execute(
        "SELECT start_frame, end_frame FROM lick_events WHERE session_id=? ORDER BY id",
        ("S_MERGE",),
    ).fetchall()
    conn.close()
    assert len(events) == 1  # 3 段短間隔的舔毛全部合併成一個事件
    assert events[0][0] == 0
    assert events[0][1] == f - _gap_frames - 1  # 最後一段 lick 的最後一幀


def test_session_disabled_storage_still_runs(tmp_path):
    """未給 storage_db_path 時，整段 Session 照跑不落地、不報錯。"""
    plugin = LickStagePlugin(nodered_url=None)  # storage 停用
    plugin.start_session("S_OFF", source_fps=_FPS)
    kpts, conf = _pose()
    for i in range(30):
        plugin.update(
            kpts, conf, source_timestamp=i / _FPS, cat_present=True, is_lick=True,
            lick_confidence=0.9,
        )
    plugin.finish_session()
    plugin.close()
    assert plugin._storage.enabled is False
