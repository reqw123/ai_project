"""M6：ActionGate（bout 邊界狀態機）單元測試。

不需要 cv2 —— action_gate.py 只依賴 analysis_context.py（純函式/常數）。
"""

import pytest

from plugins.lick_stage.action_gate import ActionGate
from plugins.lick_stage.analysis_context import FrameState

_DT = 1.0 / 30.0  # 30fps


def _feed_n(gate, frame_state, n, start_frame=0, start_ts=0.0, dt=_DT):
    """餵 n 幀同一個 frame_state，回傳「第一次 closed 非 None」那一刻的
    (in_bout, closed)——bout 一旦關閉，之後同樣的 NOT_LICK 幀不會再回傳
    closed（gate 已經是 CLOSED，沒有東西可關），只回傳最後一幀的結果會
    誤判成「沒有關閉」。找不到就回傳最後一幀的結果（in_bout, None）。"""
    in_bout, closed = gate.is_open, None
    for i in range(n):
        ts = start_ts + i * dt
        in_bout, closed = gate.feed(
            ts=ts, frame_idx=start_frame + i, dt=dt, frame_state=frame_state
        )
        if closed is not None:
            return in_bout, closed
    return in_bout, closed


class TestBasicOpenClose:
    def test_lick_assigned_opens_bout(self):
        gate = ActionGate(gap_tolerance_sec=0.0, min_bout_sec=0.0)
        in_bout, closed = gate.feed(
            ts=0.0, frame_idx=0, dt=_DT, frame_state=FrameState.LICK_ASSIGNED
        )
        assert in_bout is True
        assert closed is None

    def test_not_lick_never_opens_bout(self):
        gate = ActionGate(gap_tolerance_sec=0.0, min_bout_sec=0.0)
        in_bout, closed = gate.feed(
            ts=0.0, frame_idx=0, dt=_DT, frame_state=FrameState.NOT_LICK
        )
        assert in_bout is False
        assert closed is None

    def test_zero_gap_tolerance_closes_immediately_on_non_lick(self):
        """gap_tolerance_sec=0.0（跟 EventAggregator 預設一致）：跟 M2 舊行為
        一樣，一幀非 lick 就立刻關閉，沒有任何寬限期。"""
        gate = ActionGate(gap_tolerance_sec=0.0, min_bout_sec=0.0)
        gate.feed(ts=0.0, frame_idx=0, dt=_DT, frame_state=FrameState.LICK_ASSIGNED)
        in_bout, closed = gate.feed(
            ts=_DT, frame_idx=1, dt=_DT, frame_state=FrameState.NOT_LICK
        )
        assert in_bout is False
        assert closed is not None
        assert closed["active_sec"] == pytest.approx(_DT)

    def test_low_lick_conf_does_not_open_a_bout(self):
        """LOW_LICK_CONF 只用於「維持已開啟的 bout」，不能單獨開啟一個。"""
        gate = ActionGate(gap_tolerance_sec=0.0, min_bout_sec=0.0)
        in_bout, closed = gate.feed(
            ts=0.0, frame_idx=0, dt=_DT, frame_state=FrameState.LOW_LICK_CONF
        )
        assert in_bout is False
        assert closed is None


class TestGapTolerance:
    def test_short_gap_within_tolerance_does_not_split(self):
        gate = ActionGate(gap_tolerance_sec=0.5, min_bout_sec=0.0)
        gate.feed(ts=0.0, frame_idx=0, dt=_DT, frame_state=FrameState.LICK_ASSIGNED)
        # 3 幀非 lick（合計 0.1s，遠低於 0.5s 寬限）
        in_bout, closed = _feed_n(
            gate, FrameState.NOT_LICK, 3, start_frame=1, start_ts=_DT
        )
        assert in_bout is True  # 還在寬限期內，bout 沒關
        assert closed is None
        # 恢復 lick：應該視為同一個 bout 沒中斷過
        in_bout, closed = gate.feed(
            ts=4 * _DT, frame_idx=4, dt=_DT, frame_state=FrameState.LICK_ASSIGNED
        )
        assert in_bout is True
        assert closed is None

    def test_gap_exceeding_tolerance_closes_bout(self):
        gate = ActionGate(gap_tolerance_sec=0.1, min_bout_sec=0.0)
        gate.feed(ts=0.0, frame_idx=0, dt=_DT, frame_state=FrameState.LICK_ASSIGNED)
        # 非 lick 累積時間超過 0.1s 寬限（0.1s ≈ 3 幀 @30fps，餵到第 5 幀確定超過）
        in_bout, closed = _feed_n(
            gate, FrameState.NOT_LICK, 5, start_frame=1, start_ts=_DT
        )
        assert in_bout is False
        assert closed is not None
        # active_sec 只算真正 lick 的那一幀（寬限期本身不計入舔毛時長）
        assert closed["active_sec"] == pytest.approx(_DT)

    def test_low_lick_conf_extends_bout_without_counting_as_gap(self):
        """LOW_LICK_CONF 滿足 CONTINUE，不會累積寬限期秒數，也會累積進
        active_sec（見 action_gate.py 開頭對 hysteresis 的說明）。"""
        gate = ActionGate(gap_tolerance_sec=0.05, min_bout_sec=0.0)
        gate.feed(ts=0.0, frame_idx=0, dt=_DT, frame_state=FrameState.LICK_ASSIGNED)
        # 10 幀 LOW_LICK_CONF（合計遠超過 0.05s 的寬限期，但因為滿足
        # CONTINUE、不是「寬限期」，bout 不會因此關閉）
        in_bout, closed = _feed_n(
            gate, FrameState.LOW_LICK_CONF, 10, start_frame=1, start_ts=_DT
        )
        assert in_bout is True
        assert closed is None


class TestMinBout:
    def test_bout_shorter_than_min_bout_is_discarded(self):
        gate = ActionGate(gap_tolerance_sec=0.0, min_bout_sec=1.0)
        gate.feed(ts=0.0, frame_idx=0, dt=_DT, frame_state=FrameState.LICK_ASSIGNED)
        in_bout, closed = gate.feed(
            ts=_DT, frame_idx=1, dt=_DT, frame_state=FrameState.NOT_LICK
        )
        assert in_bout is False  # gate 確實關閉了
        assert closed is None  # 但太短，不回傳給呼叫端

    def test_bout_reaching_min_bout_is_kept(self):
        gate = ActionGate(gap_tolerance_sec=0.0, min_bout_sec=0.05)
        # 餵到累積 active_sec 剛好達標（0.05s ≈ 2 幀 @30fps）
        _feed_n(gate, FrameState.LICK_ASSIGNED, 2, start_frame=0, start_ts=0.0)
        in_bout, closed = gate.feed(
            ts=2 * _DT, frame_idx=2, dt=_DT, frame_state=FrameState.NOT_LICK
        )
        assert closed is not None
        assert closed["active_sec"] == pytest.approx(2 * _DT)


class TestDiscontinuity:
    def test_discontinuity_force_closes_open_bout(self):
        gate = ActionGate(gap_tolerance_sec=10.0, min_bout_sec=0.0)  # 寬限期很長
        gate.feed(ts=0.0, frame_idx=0, dt=_DT, frame_state=FrameState.LICK_ASSIGNED)
        in_bout, closed = gate.feed(
            ts=999.0, frame_idx=1, dt=0.0,
            frame_state=FrameState.NOT_LICK, discontinuity=True,
        )
        # discontinuity 不受 gap_tolerance 保護，立刻關閉——即使寬限期還很長
        assert in_bout is False
        assert closed is not None


class TestFinalize:
    def test_finalize_closes_open_bout(self):
        gate = ActionGate(gap_tolerance_sec=0.0, min_bout_sec=0.0)
        gate.feed(ts=0.0, frame_idx=0, dt=_DT, frame_state=FrameState.LICK_ASSIGNED)
        assert gate.is_open is True
        closed = gate.finalize()
        assert closed is not None
        assert gate.is_open is False

    def test_finalize_with_nothing_open_returns_none(self):
        gate = ActionGate(gap_tolerance_sec=0.0, min_bout_sec=0.0)
        assert gate.finalize() is None
