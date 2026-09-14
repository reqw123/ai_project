"""把逐幀狀態流聚合成 canonical 舔毛事件與固定視窗摘要。

對應說明書「單一事件聚合原則」與「事件與視窗資料設計」。

M2 範圍（刻意最小化，見下方 M6 更新）：
  * 事件 = **連續**的 stgcn-lick 幀（LICK_ASSIGNED 或 LICK_UNASSIGNED）構成
    一段 bout；一遇到非 lick 幀（或時間不連續）就結算。
  * 視窗 = 固定長度（預設 60 秒，來源時間）；跨越邊界時結算並輸出。

M6 更新（bout 邊界狀態機，見 action_gate.py）：事件邊界判斷改委派給
`ActionGate`——預設 `gap_tolerance_sec`/`min_bout_sec` 都是 0.0，行為
跟 M2 完全一樣（shadow 模式，呼叫端不傳新參數就沒有變化）；呼叫端傳非零值
才會啟用「容忍短暫中斷」「丟掉太短的雜訊 bout」。zone hysteresis（同一個
bout 內部依 zone 是否持續改變切分子事件）是 bout_aggregator.py 的工作，
這裡還沒做，事件的 zone_l1/zone_l2 仍然是整個 bout 累積時間最長的那個。

本模組純資料處理：不做 I/O、不匯入 numpy / cv2。由呼叫端（LickAnalyzer）
每幀 `feed(...)`，事件 / 視窗完成時透過 callback 交給 storage。
"""

from __future__ import annotations

import statistics as _pystats
from typing import Callable, Optional

from plugins.lick_stage.action_gate import ActionGate
from plugins.lick_stage.analysis_context import (
    FrameState,
    ZoneL1,
    canonical_zone,
)

DEFAULT_WINDOW_SEC = 60.0


class _ActiveBout:
    __slots__ = (
        "start_ts",
        "end_ts",
        "start_frame",
        "end_frame",
        "duration_sec",
        "zone_sec",  # {zone_l1: sec}
        "zone_l2_sec",  # {(l1,l2): sec}
        "zone_switch_count",
        "_last_zone_l1",
        "action_scores",
        "pose_qualities",
        "reason_codes",
        "assigned_sec",
    )

    def __init__(self, ts, frame):
        self.start_ts = ts
        self.end_ts = ts
        self.start_frame = frame
        self.end_frame = frame
        self.duration_sec = 0.0
        self.zone_sec = {}
        self.zone_l2_sec = {}
        self.zone_switch_count = 0
        self._last_zone_l1 = None
        self.action_scores = []
        self.pose_qualities = []
        self.reason_codes = []
        self.assigned_sec = 0.0

    def add(self, ts, frame, dt, l1, l2, action_score, pose_quality, reason_code, assigned):
        self.end_ts = ts
        self.end_frame = frame
        self.duration_sec += dt
        if assigned:
            self.assigned_sec += dt
            self.zone_sec[l1] = self.zone_sec.get(l1, 0.0) + dt
            key = (l1, l2)
            self.zone_l2_sec[key] = self.zone_l2_sec.get(key, 0.0) + dt
            if self._last_zone_l1 is not None and l1 != self._last_zone_l1:
                self.zone_switch_count += 1
            self._last_zone_l1 = l1
        if action_score is not None:
            self.action_scores.append(float(action_score))
        if pose_quality is not None:
            self.pose_qualities.append(float(pose_quality))
        if reason_code:
            self.reason_codes.append(reason_code)

    def to_event(self) -> dict:
        if self.zone_sec:
            primary_l1 = max(self.zone_sec, key=self.zone_sec.get)
        else:
            primary_l1 = ZoneL1.UNKNOWN
        # primary_l2：在 primary_l1 底下累積時間最長的 l2
        primary_l2 = None
        best = -1.0
        for (l1, l2), sec in self.zone_l2_sec.items():
            if l1 == primary_l1 and l2 is not None and sec > best:
                best, primary_l2 = sec, l2

        reason_mode = None
        if self.reason_codes:
            reason_mode = _pystats.mode(self.reason_codes)

        n = max(self.duration_sec, 1e-9)
        return {
            "start_source_ts": round(self.start_ts, 4),
            "end_source_ts": round(self.end_ts, 4),
            "duration_sec": round(self.duration_sec, 4),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "zone_l1": primary_l1,
            "zone_l2": primary_l2,
            "zone_score_mean": (
                round(_pystats.fmean(self.action_scores), 4)
                if self.action_scores
                else None
            ),
            "zone_switch_count": self.zone_switch_count,
            "assigned_ratio": round(self.assigned_sec / n, 4),
            "action_score_mean": (
                round(_pystats.fmean(self.action_scores), 4)
                if self.action_scores
                else None
            ),
            "action_score_min": (
                round(min(self.action_scores), 4) if self.action_scores else None
            ),
            "pose_quality_mean": (
                round(_pystats.fmean(self.pose_qualities), 4)
                if self.pose_qualities
                else None
            ),
            "unknown_reason_mode": reason_mode,
            # M6（2026-09-14）：min_bout / gap merge 已透過 action_gate.py 的
            # ActionGate 完成，但同一個 bout 內部的 zone hysteresis
            # （bout_aggregator.py）還沒做——這個事件的 zone_l1/zone_l2 仍然
            # 只是整個 bout 累積時間最長的那個，還不能反映「同一段裡換了
            # 部位」這種情況。等 bout_aggregator.py 也做完，這裡才會真的變
            # False；維持 True 是誠實反映「還沒完全過完 M6」，不是沒接上
            # ActionGate。
            "raw_bout": True,
        }


class _Window:
    __slots__ = (
        "start",
        "end",
        "observed_sec",
        "valid_observed_sec",
        "no_cat_sec",
        "stgcn_lick_sec",
        "assigned_zone_sec",
        "unassigned_lick_sec",
        "zone_l1_sec",
        "bout_secs",
    )

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.observed_sec = 0.0
        self.valid_observed_sec = 0.0
        self.no_cat_sec = 0.0
        self.stgcn_lick_sec = 0.0
        self.assigned_zone_sec = 0.0
        self.unassigned_lick_sec = 0.0
        self.zone_l1_sec = {
            ZoneL1.TORSO: 0.0,
            ZoneL1.FORELIMB: 0.0,
            ZoneL1.HINDLIMB: 0.0,
            ZoneL1.TAIL: 0.0,
        }
        self.bout_secs = []

    def to_summary(self, period: str = "") -> dict:
        cov = (
            self.assigned_zone_sec / self.stgcn_lick_sec
            if self.stgcn_lick_sec > 1e-9
            else 0.0
        )
        bs = sorted(self.bout_secs)
        return {
            "window_start": round(self.start, 3),
            "window_end": round(self.end, 3),
            "window_sec": round(self.end - self.start, 3),
            "period": period,
            "observed_sec": round(self.observed_sec, 3),
            "valid_observed_sec": round(self.valid_observed_sec, 3),
            "no_cat_sec": round(self.no_cat_sec, 3),
            "stale_sec": 0.0,  # M3 傳輸層才有 stale 概念
            "stgcn_lick_sec": round(self.stgcn_lick_sec, 3),
            "assigned_zone_sec": round(self.assigned_zone_sec, 3),
            "unassigned_lick_sec": round(self.unassigned_lick_sec, 3),
            "bout_count": len(bs),
            "median_bout_sec": round(_pystats.median(bs), 3) if bs else 0.0,
            "max_bout_sec": round(bs[-1], 3) if bs else 0.0,
            "torso_sec": round(self.zone_l1_sec[ZoneL1.TORSO], 3),
            "forelimb_sec": round(self.zone_l1_sec[ZoneL1.FORELIMB], 3),
            "hindlimb_sec": round(self.zone_l1_sec[ZoneL1.HINDLIMB], 3),
            "tail_sec": round(self.zone_l1_sec[ZoneL1.TAIL], 3),
            "zone_coverage_ratio": round(cov, 4),
        }


class EventAggregator:
    """逐幀 feed，事件 / 視窗完成時呼叫 sink callback。

    `gap_tolerance_sec`/`min_bout_sec` 預設都是 0.0——**完全重現 M2 的舊行為**
    （一遇到非 lick 幀就立刻結算、不篩掉任何長度的 bout），這是刻意的 shadow
    模式：呼叫端（manager.py／ext_body_zones/plugin.py）不主動傳新參數就
    跟以前一模一樣，既有測試/行為都不受影響。要啟用 M6 的邊界狀態機（容忍
    短暫中斷、丟掉太短的雜訊 bout），呼叫端要明確傳非零值——通常是
    `config.py` 的 `GAP_TOLERANCE_SEC`/`MIN_BOUT_SEC`。

    邊界判斷本身委派給 `action_gate.ActionGate`（見該檔案開頭的完整說明），
    這裡只負責：gate 說「在 bout 裡」時把內容（zone/動作分數等）累積進
    `_ActiveBout`，gate 說「bout 關閉」時結算並回呼 `on_event`。"""

    def __init__(
        self,
        *,
        window_sec: float = DEFAULT_WINDOW_SEC,
        period: str = "",
        gap_tolerance_sec: float = 0.0,
        min_bout_sec: float = 0.0,
        on_event: Optional[Callable[[dict], None]] = None,
        on_window: Optional[Callable[[dict], None]] = None,
    ):
        self.window_sec = float(window_sec)
        self.period = period
        self._on_event = on_event
        self._on_window = on_window
        self._bout: Optional[_ActiveBout] = None
        self._window: Optional[_Window] = None
        self._elapsed = 0.0  # 累積來源時間，用來決定視窗邊界
        self._gate = ActionGate(
            gap_tolerance_sec=gap_tolerance_sec, min_bout_sec=min_bout_sec
        )

    # ── 每幀 ─────────────────────────────────────────────────────────────
    def feed(
        self,
        *,
        source_ts,
        frame_idx: int,
        dt_sec: float,
        frame_state: str,
        zone_label: str,
        action_score=None,
        pose_quality=None,
        reason_code=None,
        discontinuity: bool = False,
    ) -> None:
        if discontinuity:
            # 時間不連續：結算開放中的 bout（不跨斷點），視窗計時也不前進
            was_open = self._gate.is_open
            _, gate_closed = self._gate.feed(
                ts=self._elapsed, frame_idx=frame_idx, dt=0.0,
                frame_state=frame_state, discontinuity=True,
            )
            self._on_gate_transition(was_open, gate_closed)
            return

        dt = max(0.0, float(dt_sec))
        if dt <= 0.0 and frame_state != FrameState.NO_CAT:
            # 第一幀 dt=0：仍要初始化視窗
            pass

        ts = self._elapsed if source_ts is None else float(source_ts)
        self._ensure_window(ts)
        self._accumulate_window(dt, frame_state, zone_label)

        was_open = self._gate.is_open
        _, gate_closed = self._gate.feed(
            ts=ts, frame_idx=frame_idx, dt=dt, frame_state=frame_state,
        )

        # 只有真的判成 lick 的幀（LICK_ASSIGNED/LICK_UNASSIGNED）才把內容
        # 累積進 _ActiveBout——gate 的 CONTINUE 條件另外容許 LOW_LICK_CONF
        # 撐住 bout 不中斷，但那種幀沒有可信的 zone/動作分數，不該進統計
        # （見 action_gate.py 開頭「為什麼 START/CONTINUE 不是兩個信心數值
        # 門檻」的說明）。這個分支跟下面 gate 關閉的分支互斥（gate 只會在
        # 「不滿足 CONTINUE」的幀上關閉，LICK 幀必然滿足 CONTINUE），
        # 兩者執行順序不影響正確性。
        if frame_state in FrameState.LICK:
            l1, l2 = canonical_zone(zone_label)
            assigned = frame_state == FrameState.LICK_ASSIGNED
            if self._bout is None:
                self._bout = _ActiveBout(ts, frame_idx)
            self._bout.add(
                ts,
                frame_idx,
                dt,
                l1,
                l2,
                action_score,
                pose_quality,
                reason_code,
                assigned,
            )

        self._on_gate_transition(was_open, gate_closed)

        self._elapsed += dt
        self._roll_window_if_needed(self._elapsed)

    def finalize(self, end_source_ts=None) -> None:
        """Session 結束：結算最後一段 bout 與最後一個未滿的視窗。"""
        was_open = self._gate.is_open
        gate_closed = self._gate.finalize()
        self._on_gate_transition(was_open, gate_closed)
        if self._window is not None:
            self._emit_window()
            self._window = None

    # ── 私有 ─────────────────────────────────────────────────────────────
    def _on_gate_transition(self, was_open: bool, gate_closed) -> None:
        """`gate_closed` 是 `ActionGate.feed()`/`finalize()` 的回傳值：非 None
        代表 gate 判定一個 bout 真正關閉且通過 min_bout 過濾，應該結算並送出
        事件；`was_open and gate_closed is None` 代表 gate 剛把一個開著的
        bout 關閉，但太短被丟棄（沒通過 min_bout），對應的 `_ActiveBout`
        內容要安靜丟掉，不送事件、也不計進視窗的 bout_secs。"""
        just_closed = was_open and not self._gate.is_open
        if not just_closed:
            return
        bout, self._bout = self._bout, None
        if gate_closed is None or bout is None:
            return  # 太短被 gate 丟棄，或 gate 判定的邊界內完全沒有 LICK 內容幀
        event = bout.to_event()
        if self._window is not None:
            self._window.bout_secs.append(event["duration_sec"])
        if self._on_event is not None and event["duration_sec"] > 0.0:
            self._on_event(event)

    def _ensure_window(self, ts) -> None:
        if self._window is None:
            start = (ts // self.window_sec) * self.window_sec
            self._window = _Window(start, start + self.window_sec)

    def _accumulate_window(self, dt, frame_state, zone_label) -> None:
        w = self._window
        w.observed_sec += dt
        if frame_state == FrameState.NO_CAT:
            w.no_cat_sec += dt
            return
        w.valid_observed_sec += dt
        if frame_state == FrameState.LICK_ASSIGNED:
            w.stgcn_lick_sec += dt
            w.assigned_zone_sec += dt
            l1, _ = canonical_zone(zone_label)
            if l1 in w.zone_l1_sec:
                w.zone_l1_sec[l1] += dt
        elif frame_state == FrameState.LICK_UNASSIGNED:
            w.stgcn_lick_sec += dt
            w.unassigned_lick_sec += dt

    def _roll_window_if_needed(self, elapsed) -> None:
        w = self._window
        if w is None:
            return
        # 用累積來源時間對齊視窗；跨過一個或多個邊界都補齊
        while elapsed >= w.end:
            self._emit_window()
            new_start = w.end
            self._window = _Window(new_start, new_start + self.window_sec)
            w = self._window

    def _emit_window(self) -> None:
        if self._on_window is not None:
            self._on_window(self._window.to_summary(self.period))
