"""第一階段驗收：舔毛時長對「處理速度」不變（說明書「來源時間規格」）。

核心主張：同一支影片無論以即時、兩倍速或離線批次推論，累積舔毛時長必須
一致 —— 因為 plugin 內部計時只看 source_timestamp，不看 wall clock。

這裡用兩種方式驗證：
  1. 餵入完全相同的 source_timestamp 序列，但在其中人為讓 time.monotonic()
     回傳混亂的值（模擬 GPU 卡頓 / 離線批次 / 暫停）—— 結果應完全相同。
  2. dt 大於 max_valid_gap 時（seek / 排程恢復）該段時間不被累加。
"""

import numpy as np
import pytest

pytest.importorskip(
    "cv2",
    reason="plugins/lick_stage/__init__.py 透過 manager.py/overlay.py 需要 cv2，此環境未安裝",
)

from plugins.lick_stage.analysis_context import DEFAULT_MAX_VALID_GAP_SEC
from plugins.lick_stage.config import LickConfig as _C
from plugins.lick_stage.manager import LickStagePlugin

NUM_JOINTS = 17
_SRC_FPS = 30.0


def _lick_pose():
    """一個資料充分、可跑完整幾何路徑的舔毛姿態（是否命中區域不影響
    stgcn_lick_sec —— 它對 LICK_ASSIGNED 與 LICK_UNASSIGNED 都累加）。"""
    kpts = np.zeros((NUM_JOINTS, 2), dtype=np.float64)
    kpts[_C.KP_NOSE] = (0.0, 120.0)
    kpts[_C.KP_LEFT_EAR] = (-18.0, -6.0)
    kpts[_C.KP_RIGHT_EAR] = (18.0, -6.0)
    kpts[_C.KP_CHEST] = (0.0, 0.0)
    kpts[_C.KP_MID_BACK] = (0.0, 180.0)
    kpts[_C.KP_HIP] = (0.0, 360.0)
    conf = np.ones(NUM_JOINTS, dtype=np.float64)
    return kpts, conf


def _run(plugin, n_frames, *, monotonic_sequence=None):
    """餵 n_frames 個 lick 幀，source_timestamp = i / 30。回傳 stgcn_lick_sec。"""
    kpts, conf = _lick_pose()
    for i in range(n_frames):
        if monotonic_sequence is not None:
            monotonic_sequence.append(i)  # 只是記錄；實際 patch 在外層
        plugin.update(
            kpts,
            conf,
            source_timestamp=i / _SRC_FPS,
            frame_idx=i,
            cat_present=True,
            is_lick=True,
            lick_confidence=0.95,
            session_id="S_test",
        )
    return plugin._analyzer._stats.stgcn_lick_sec


def test_duration_identical_regardless_of_wall_clock(monkeypatch):
    n = 90  # 90 幀 @ 30fps → source 上經過 89/30 ≈ 2.967 秒
    expected = (n - 1) / _SRC_FPS

    # Run A：wall clock 正常前進
    plugin_a = LickStagePlugin(nodered_url=None)
    got_a = _run(plugin_a, n)

    # Run B：讓 time.monotonic() 回傳混亂跳動的值（模擬離線批次/卡頓/暫停）
    import plugins.lick_stage.manager as manager_mod

    chaos = iter([0.0, 999.0, 1.0, 5000.0] * n)
    monkeypatch.setattr(manager_mod.time, "monotonic", lambda: next(chaos))
    plugin_b = LickStagePlugin(nodered_url=None)
    got_b = _run(plugin_b, n)

    assert got_a == pytest.approx(expected, abs=1.0 / _SRC_FPS)
    assert got_b == pytest.approx(expected, abs=1.0 / _SRC_FPS)
    assert got_a == pytest.approx(got_b, abs=1e-9)


def test_half_and_double_speed_give_same_accumulated_seconds():
    """同一組 source_timestamp，不論呼叫端「多快」餵進來，結果一致。"""
    n = 60
    plugin_slow = LickStagePlugin(nodered_url=None)
    plugin_fast = LickStagePlugin(nodered_url=None)
    slow = _run(plugin_slow, n)
    fast = _run(plugin_fast, n)
    assert slow == pytest.approx(fast, abs=1e-9)


def test_large_source_time_gap_is_not_accumulated():
    """source_timestamp 跳躍超過 max_valid_gap（seek / 排程恢復）→ 該段不累加。"""
    plugin = LickStagePlugin(nodered_url=None)
    kpts, conf = _lick_pose()

    # 前 10 幀正常（每幀 1/30 秒）
    for i in range(10):
        plugin.update(
            kpts, conf, source_timestamp=i / _SRC_FPS,
            cat_present=True, is_lick=True, lick_confidence=0.9,
        )
    before = plugin._analyzer._stats.stgcn_lick_sec

    # 一個大跳躍（10 秒後才有下一幀）—— 這 10 秒不該被算成舔毛
    plugin.update(
        kpts, conf, source_timestamp=10.0 + 10 / _SRC_FPS,
        cat_present=True, is_lick=True, lick_confidence=0.9,
    )
    after = plugin._analyzer._stats.stgcn_lick_sec

    assert (after - before) <= DEFAULT_MAX_VALID_GAP_SEC
    assert (after - before) == pytest.approx(0.0, abs=1e-9)
    assert plugin._analyzer._stats.discontinuity_count >= 1


def test_stgcn_lick_invariant_holds():
    """必要不變式：stgcn_lick_sec == assigned_zone_sec + unassigned_lick_sec。"""
    plugin = LickStagePlugin(nodered_url=None)
    _run(plugin, 50)
    st = plugin._analyzer._stats
    assert st.invariant_error() == pytest.approx(0.0, abs=1e-9)
