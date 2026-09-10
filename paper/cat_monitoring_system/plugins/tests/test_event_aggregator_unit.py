"""第二階段（M2）：EventAggregator 把逐幀狀態流聚合成事件 / 視窗。

不需要 cv2 —— event_aggregator.py 只依賴 analysis_context.py（純函式）。
"""

import pytest

from plugins.lick_stage.analysis_context import FrameState, ZoneL1
from plugins.lick_stage.event_aggregator import EventAggregator

_FPS = 30.0
_DT = 1.0 / _FPS


def _feed_lick_run(agg, start_frame, n, zone="BODY_CENTER", assigned=True):
    fs = FrameState.LICK_ASSIGNED if assigned else FrameState.LICK_UNASSIGNED
    for i in range(n):
        f = start_frame + i
        agg.feed(
            source_ts=f * _DT,
            frame_idx=f,
            dt_sec=_DT,
            frame_state=fs,
            zone_label=zone,
            action_score=0.9,
            reason_code=None if assigned else "NO_REGION_HIT",
        )


def test_contiguous_lick_run_becomes_one_event_on_close():
    events = []
    agg = EventAggregator(on_event=events.append)
    _feed_lick_run(agg, 0, 10)  # 10 幀 @30fps ≈ 0.33 秒
    assert events == []  # 尚未結算
    # 一個非 lick 幀 → 結算
    agg.feed(source_ts=10 * _DT, frame_idx=10, dt_sec=_DT,
             frame_state=FrameState.NOT_LICK, zone_label="NO_TARGET")
    assert len(events) == 1
    ev = events[0]
    assert ev["zone_l1"] == ZoneL1.TORSO
    assert ev["start_frame"] == 0 and ev["end_frame"] == 9
    assert ev["duration_sec"] == pytest.approx(10 * _DT, abs=1e-3)
    assert ev["assigned_ratio"] == pytest.approx(1.0)
    assert ev["raw_bout"] is True


def test_zone_switch_within_bout_is_counted_not_split():
    events = []
    agg = EventAggregator(on_event=events.append)
    _feed_lick_run(agg, 0, 5, zone="FL")
    _feed_lick_run(agg, 5, 5, zone="HR")
    agg.finalize()
    assert len(events) == 1
    ev = events[0]
    assert ev["zone_switch_count"] == 1
    # primary zone = 時間較多者；此處兩段等長，取字典 max（實作細節）——至少要是其一
    assert ev["zone_l1"] in (ZoneL1.FORELIMB, ZoneL1.HINDLIMB)


def test_unassigned_lick_still_forms_event_with_reason_mode():
    events = []
    agg = EventAggregator(on_event=events.append)
    _feed_lick_run(agg, 0, 6, assigned=False)
    agg.finalize()
    assert len(events) == 1
    assert events[0]["assigned_ratio"] == pytest.approx(0.0)
    assert events[0]["unknown_reason_mode"] == "NO_REGION_HIT"
    assert events[0]["zone_l1"] == ZoneL1.UNKNOWN


def test_window_summary_emitted_on_boundary_cross():
    windows = []
    agg = EventAggregator(window_sec=1.0, on_window=windows.append)
    # 45 幀 @30fps = 1.5 秒 → 跨過 1 個視窗邊界
    for f in range(45):
        agg.feed(source_ts=f * _DT, frame_idx=f, dt_sec=_DT,
                 frame_state=FrameState.LICK_ASSIGNED, zone_label="BODY_CENTER",
                 action_score=0.9)
    assert len(windows) == 1
    w0 = windows[0]
    assert w0["window_sec"] == pytest.approx(1.0)
    assert w0["stgcn_lick_sec"] == pytest.approx(1.0, abs=_DT)
    assert w0["assigned_zone_sec"] == pytest.approx(w0["torso_sec"], abs=1e-6)

    agg.finalize()
    assert len(windows) == 2  # 最後未滿的視窗也會被結算


def test_discontinuity_closes_bout_without_bridging():
    events = []
    agg = EventAggregator(on_event=events.append)
    _feed_lick_run(agg, 0, 5)
    agg.feed(source_ts=999.0, frame_idx=5, dt_sec=0.0,
             frame_state=FrameState.LICK_ASSIGNED, zone_label="BODY_CENTER",
             discontinuity=True)
    _feed_lick_run(agg, 6, 5)
    agg.finalize()
    assert len(events) == 2  # 斷點兩側是兩個獨立事件
