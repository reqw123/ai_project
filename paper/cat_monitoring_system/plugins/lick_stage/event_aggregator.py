"""把逐幀狀態流聚合成 canonical 舔毛事件與固定視窗摘要。

對應說明書「單一事件聚合原則」與「事件與視窗資料設計」。

M2 範圍（刻意最小化）：
  * 事件 = **連續**的 stgcn-lick 幀（LICK_ASSIGNED 或 LICK_UNASSIGNED）構成
    一段 bout；一遇到非 lick 幀（或時間不連續）就結算。
  * 尚未實作 min_bout / gap merge / zone hysteresis —— 那是第二階段
    （M6）action_gate.py / bout_aggregator.py 的工作。這裡產出的事件明確
    標為 `raw_bout=True`，論文使用前必須先過 M6 的聚合。
  * 視窗 = 固定長度（預設 60 秒，來源時間）；跨越邊界時結算並輸出。

本模組純資料處理：不做 I/O、不匯入 numpy / cv2。由呼叫端（LickAnalyzer）
每幀 `feed(...)`，事件 / 視窗完成時透過 callback 交給 storage。
"""

from __future__ import annotations

import statistics as _pystats
from typing import Callable, Optional

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
            "raw_bout": True,  # 尚未過 M6 的 min_bout / gap merge / hysteresis
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
    """逐幀 feed，事件 / 視窗完成時呼叫 sink callback。"""

    def __init__(
        self,
        *,
        window_sec: float = DEFAULT_WINDOW_SEC,
        period: str = "",
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
            self._close_bout()
            return

        dt = max(0.0, float(dt_sec))
        if dt <= 0.0 and frame_state != FrameState.NO_CAT:
            # 第一幀 dt=0：仍要初始化視窗
            pass

        ts = self._elapsed if source_ts is None else float(source_ts)
        self._ensure_window(ts)
        self._accumulate_window(dt, frame_state, zone_label)

        is_lick = frame_state in FrameState.LICK
        if is_lick:
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
        else:
            self._close_bout()

        self._elapsed += dt
        self._roll_window_if_needed(self._elapsed)

    def finalize(self, end_source_ts=None) -> None:
        """Session 結束：結算最後一段 bout 與最後一個未滿的視窗。"""
        self._close_bout()
        if self._window is not None:
            self._emit_window()
            self._window = None

    # ── 私有 ─────────────────────────────────────────────────────────────
    def _close_bout(self) -> None:
        if self._bout is None:
            return
        event = self._bout.to_event()
        if self._window is not None:
            self._window.bout_secs.append(event["duration_sec"])
        self._bout = None
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
