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
    # 3 段舔毛，中間插非舔毛 —— 應形成 3 個事件
    for _bout in range(3):
        for _ in range(20):  # 每段 20 幀 ≈ 0.67 秒
            plugin.update(
                kpts, conf, source_timestamp=f / _FPS, frame_idx=f,
                cat_present=True, is_lick=True, lick_confidence=0.9,
                session_id="S_ITEST",
            )
            f += 1
        for _ in range(10):  # 非舔毛間隔
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
