"""M6（第二部分）：ZoneHysteresis（同一個 raw bout 內部的 zone 切分）單元
測試 + 透過 EventAggregator 的端到端整合測試。

不需要 cv2 —— bout_aggregator.py 只依賴 analysis_context.py（純函式/常數）。
"""

import pytest

from plugins.lick_stage.analysis_context import FrameState, ZoneL1
from plugins.lick_stage.bout_aggregator import ZoneHysteresis
from plugins.lick_stage.event_aggregator import EventAggregator

_DT = 1.0 / 30.0  # 30fps


def _feed_zone(seg, l1, n, start_frame=0, start_ts=0.0, dt=_DT, assigned=True, l2=None):
    """餵 n 幀同一個 zone，回傳第一次非 None 的切分結果（若中途沒有觸發
    切分，回傳最後一幀的 None）。**會提早返回**——只適合用在「這段本身
    預期不會觸發切分」或「只關心第一次切分結果」的情境；要完整餵完
    n 幀（含切分後續內容）請用 `_feed_zone_collect`。"""
    finished = None
    for i in range(n):
        ts = start_ts + i * dt
        finished = seg.feed(ts, start_frame + i, dt, l1, l2, None, None, None, assigned)
        if finished is not None:
            return finished
    return finished


def _feed_zone_collect(seg, l1, n, start_frame=0, start_ts=0.0, dt=_DT, assigned=True, l2=None):
    """餵滿 n 幀同一個 zone（不提早返回），回傳這 n 幀期間所有非 None 的
    切分結果組成的 list（可能是空的）。"""
    out = []
    for i in range(n):
        ts = start_ts + i * dt
        finished = seg.feed(ts, start_frame + i, dt, l1, l2, None, None, None, assigned)
        if finished is not None:
            out.append(finished)
    return out


class TestShadowDefault:
    """zone_switch_min_sec=None：完全不啟用切分，等同 M6 前半（action_gate.py
    已生效，但 bout 內部從不因為 zone 改變而拆事件）。"""

    def test_zone_change_never_splits_without_threshold(self):
        seg = ZoneHysteresis(zone_switch_min_sec=None)
        # 前半段 FORELIMB，後半段 HINDLIMB，各餵很久——沒有門檻，永遠不切分
        assert _feed_zone(seg, ZoneL1.FORELIMB, 30, start_frame=0, start_ts=0.0) is None
        assert (
            _feed_zone(seg, ZoneL1.HINDLIMB, 30, start_frame=30, start_ts=30 * _DT)
            is None
        )
        ev = seg.flush_final()
        assert ev is not None
        assert ev["raw_bout"] is True  # 從沒切分過，整段就是一筆事件
        assert ev["start_frame"] == 0
        assert ev["end_frame"] == 59


class TestConfirmedSwitch:
    def test_sustained_zone_change_triggers_split(self):
        """候選 zone 連續累積時間達到門檻 -> feed() 提前送出子事件
        （raw_bout=False），新內容接續累積到新 zone。"""
        seg = ZoneHysteresis(zone_switch_min_sec=0.1)  # 0.1s ≈ 3 幀 @30fps
        assert _feed_zone(seg, ZoneL1.FORELIMB, 10, start_frame=0, start_ts=0.0) is None
        finished = _feed_zone(
            seg, ZoneL1.HINDLIMB, 10, start_frame=10, start_ts=10 * _DT
        )
        assert finished is not None
        assert finished["raw_bout"] is False
        assert finished["zone_l1"] == ZoneL1.FORELIMB
        assert finished["start_frame"] == 0
        # 最後一段（切分後接續累積的 HINDLIMB 內容）用 flush_final 結算
        tail = seg.flush_final()
        assert tail is not None
        assert tail["raw_bout"] is False  # 這個 bout 觸發過切分，最後一段也不是 raw
        assert tail["zone_l1"] == ZoneL1.HINDLIMB

    def test_multiple_sustained_switches_produce_multiple_events(self):
        """A -> B -> C 都持續夠久：兩次中途切分 + 一次 flush_final，共 3 段。
        用 `_feed_zone_collect`（不提早返回）確保每段真的餵滿 10 幀，這樣
        每段的 primary zone 才會是明確多數，不受切分當下那 1~2 幀少數
        內容干擾。"""
        seg = ZoneHysteresis(zone_switch_min_sec=0.1)
        events = []
        events += _feed_zone_collect(seg, ZoneL1.TORSO, 10, start_frame=0, start_ts=0.0)
        events += _feed_zone_collect(
            seg, ZoneL1.FORELIMB, 10, start_frame=10, start_ts=10 * _DT
        )
        events += _feed_zone_collect(
            seg, ZoneL1.HINDLIMB, 10, start_frame=20, start_ts=20 * _DT
        )
        events.append(seg.flush_final())
        assert [ev["zone_l1"] for ev in events] == [
            ZoneL1.TORSO,
            ZoneL1.FORELIMB,
            ZoneL1.HINDLIMB,
        ]
        for ev in events:
            assert ev["raw_bout"] is False


class TestFlickerAbsorbed:
    def test_brief_flicker_below_threshold_does_not_split(self):
        """候選 zone 沒撐夠久就跳回原 zone：streak 歸零，不觸發切分，flicker
        幀的內容誠實留在同一段裡（影響 zone_sec 但不影響 primary）。"""
        seg = ZoneHysteresis(zone_switch_min_sec=0.5)  # 門檻遠高於下面的抖動長度
        _feed_zone(seg, ZoneL1.TORSO, 20, start_frame=0, start_ts=0.0)
        # 短暫抖動成 FORELIMB（2 幀 ≈ 0.067s，遠低於 0.5s 門檻）
        assert (
            _feed_zone(seg, ZoneL1.FORELIMB, 2, start_frame=20, start_ts=20 * _DT)
            is None
        )
        # 跳回 TORSO
        assert (
            _feed_zone(seg, ZoneL1.TORSO, 20, start_frame=22, start_ts=22 * _DT)
            is None
        )
        ev = seg.flush_final()
        assert ev["raw_bout"] is True  # 從沒真正切分過
        assert ev["zone_l1"] == ZoneL1.TORSO  # 抖動幀是少數，不影響 primary
        assert ev["zone_switch_count"] == 2  # 但原始切換次數誠實記錄（TORSO<->FORELIMB 各一次）

    def test_unassigned_frames_do_not_advance_candidate_streak(self):
        """assigned=False（LICK_UNASSIGNED，例如 NO_REGION_HIT）不會推進候選
        streak，也不會打斷——即使夾在候選 zone 中間、且幀數很多，只要真正
        assigned 的候選累積秒數沒到門檻，就不觸發切分。"""
        seg = ZoneHysteresis(zone_switch_min_sec=0.1)
        _feed_zone(seg, ZoneL1.TORSO, 10, start_frame=0, start_ts=0.0)
        # 候選 FORELIMB 只餵 1 幀（遠低於 0.1s 門檻），中間穿插大量
        # unassigned 幀——unassigned 幀不會讓候選 streak 誤判成已經達標
        assert (
            _feed_zone(
                seg, ZoneL1.FORELIMB, 1, start_frame=10, start_ts=10 * _DT
            )
            is None
        )
        finished = _feed_zone(
            seg,
            ZoneL1.UNKNOWN,
            50,
            start_frame=11,
            start_ts=11 * _DT,
            assigned=False,
        )
        assert finished is None  # 50 幀 unassigned 不會湊出切分門檻
        ev = seg.flush_final()
        assert ev["raw_bout"] is True


class TestEventAggregatorIntegration:
    """透過 EventAggregator.feed() 的端到端驗證：zone_label 字串 ->
    canonical_zone -> ZoneHysteresis 全流程。"""

    def _feed_run(self, agg, start_frame, n, zone):
        for i in range(n):
            f = start_frame + i
            agg.feed(
                source_ts=f * _DT,
                frame_idx=f,
                dt_sec=_DT,
                frame_state=FrameState.LICK_ASSIGNED,
                zone_label=zone,
                action_score=0.9,
            )

    def test_disabled_by_default_single_event_per_bout(self):
        """EventAggregator 不傳 zone_switch_min_sec：shadow，即使 zone
        中途換了兩次，還是只送出一筆事件（raw_bout=True）。"""
        events = []
        agg = EventAggregator(on_event=events.append)
        self._feed_run(agg, 0, 10, "FL")
        self._feed_run(agg, 10, 10, "HL")
        agg.finalize()
        assert len(events) == 1
        assert events[0]["raw_bout"] is True

    def test_enabled_splits_sustained_zone_change_into_two_events(self):
        events = []
        agg = EventAggregator(zone_switch_min_sec=0.1, on_event=events.append)
        self._feed_run(agg, 0, 10, "FL")  # FORELIMB，10 幀
        self._feed_run(agg, 10, 10, "HL")  # HINDLIMB，10 幀，遠超過 0.1s 門檻
        agg.finalize()
        assert len(events) == 2
        assert events[0]["zone_l1"] == ZoneL1.FORELIMB
        assert events[0]["raw_bout"] is False
        assert events[1]["zone_l1"] == ZoneL1.HINDLIMB
        assert events[1]["raw_bout"] is False
        # 兩段合計時長跟總幀數一致（沒有幀被漏掉或重複計；容許 1e-3 是因為
        # 兩段各自的 duration_sec 在 to_event() 已各自四捨五入到小數 4 位，
        # 加總會有微小的捨入誤差，不是計算錯誤）。
        total = events[0]["duration_sec"] + events[1]["duration_sec"]
        assert total == pytest.approx(20 * _DT, abs=1e-3)

    def test_enabled_but_brief_zone_flicker_still_one_event(self):
        """啟用切分，但抖動時間不到門檻：行為等同沒啟用（一筆事件）。"""
        events = []
        agg = EventAggregator(zone_switch_min_sec=0.5, on_event=events.append)
        self._feed_run(agg, 0, 20, "FL")
        self._feed_run(agg, 20, 2, "HL")  # 短暫抖動，遠低於 0.5s 門檻
        self._feed_run(agg, 22, 20, "FL")
        agg.finalize()
        assert len(events) == 1
        assert events[0]["raw_bout"] is True
        assert events[0]["zone_l1"] == ZoneL1.FORELIMB

    def test_split_events_both_counted_into_window_bout_secs(self):
        """window summary 的 bout_count 反映切分後的子事件數（誠實的副作用，
        見 event_aggregator.py 模組開頭 M6 更新說明）。"""
        windows = []
        agg = EventAggregator(zone_switch_min_sec=0.1, window_sec=10.0, on_window=windows.append)
        self._feed_run(agg, 0, 10, "FL")
        self._feed_run(agg, 10, 10, "HL")
        agg.finalize()
        assert len(windows) == 1
        assert windows[0]["bout_count"] == 2
