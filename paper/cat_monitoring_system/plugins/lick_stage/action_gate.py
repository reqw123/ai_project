"""M6：舔毛「動作邊界」狀態機——決定一段連續（或近似連續）的舔毛訊號算不算
一個 bout，以及 bout 該從哪裡開始/結束。取代 event_aggregator.py 舊版「一
遇到非 lick 幀就立刻結算」的粗聚合（見該檔案開頭的 `raw_bout=True` 說明）。

**為什麼 START/CONTINUE 不是兩個信心數值門檻**：呼叫端（frame_processor.py）
傳進來的 `lick_confidence` 只有在 `is_lick=True`（`behavior_id` 剛好判成
lick）時才真的是「P(lick)」——`is_lick=False` 時，那個數字其實是「ST-GCN
判成的其他行為」的信心值，跟舔毛信心無關，拿它跟 lick 的門檻比較是誤用。
而且 `analysis_context.compute_frame_state()`（M1）已經把 `lick_confidence`
跟 `low_conf_threshold` 折算進 `FrameState` 了（過門檻→`LICK_ASSIGNED`/
`LICK_UNASSIGNED`，沒過→`LOW_LICK_CONF`）——再拿原始浮點數開一套新門檻，
會製造兩份彼此不同步、各自要校準的信心判準。所以這裡的 hysteresis 改成
兩組寬鬆度不同的 `FrameState` 集合：

- **START**（開啟一個新 bout）：`frame_state ∈ FrameState.LICK`
  （`LICK_ASSIGNED`／`LICK_UNASSIGNED`）——跟 M2 原本開始一個 bout 的條件
  一致，沒有放寬也沒有收緊。
- **CONTINUE**（維持已經開啟的 bout）：上面那組 **+ `LOW_LICK_CONF`**——
  已經在舔毛的情況下，信心值短暫掉到 M1 的門檻以下（貓稍微側過頭、模型
  信心浮動）不會馬上結束事件；還沒開始的時候用同一組寬鬆判準當「開始」
  條件則太容易誤觸發，兩種情境的合理寬鬆度本來就不一樣，這正是
  hysteresis 要處理的問題。

再疊兩層時間容忍（見 config.py 的 GAP_TOLERANCE_SEC／MIN_BOUT_SEC）：

- **gap 寬限**：CONTINUE 條件都不滿足（真的是 `NOT_LICK`/`NO_CAT`）時，
  不會立刻結束 bout，先進入 PENDING_CLOSE、給一段寬限期；寬限期內 CONTINUE
  條件恢復，視為同一個 bout 沒有中斷過（寬限期本身不計入回報的「舔毛時長」，
  只延伸事件的結束時間戳）。超過寬限期才真正關閉。
- **min_bout 過濾**：bout 真正關閉時，若累積的（CONTINUE 條件下的）舔毛
  時長低於門檻，整段直接丟棄，不產生事件——避免零星幾幀誤判被聚成一筆
  「事件」污染事件表。

本模組是純狀態機：不知道 zone/geometry，只回答「這一幀算不算某個 bout
的一部分、bout 有沒有在這一幀關閉」；zone 相關的聚合/hysteresis 是
bout_aggregator.py 的工作，兩者刻意分工，不合併成一個檔案。
"""

from __future__ import annotations

from typing import Optional

from plugins.lick_stage.analysis_context import FrameState
from plugins.lick_stage.config import LickConfig as _C

_CLOSED = "CLOSED"
_OPEN = "OPEN"
_PENDING_CLOSE = "PENDING_CLOSE"


class ActionGate:
    """逐幀 `feed()`；回傳 `(in_bout, closed_bout)`。

    `closed_bout`：非 None 時代表這一幀讓一個 bout 真正關閉並通過
    `min_bout_sec` 過濾，內容是 `{start_ts, end_ts, start_frame, end_frame,
    active_sec}`——呼叫端（event_aggregator.py）據此去結算對應的
    `_ActiveBout`（zone/動作分數等內容累積）並產出事件。太短被丟棄的 bout
    不會出現在這裡，呼叫端不需要另外判斷。
    """

    def __init__(
        self,
        *,
        gap_tolerance_sec: float = None,
        min_bout_sec: float = None,
    ):
        self.gap_tolerance_sec = (
            _C.GAP_TOLERANCE_SEC if gap_tolerance_sec is None else float(gap_tolerance_sec)
        )
        self.min_bout_sec = (
            _C.MIN_BOUT_SEC if min_bout_sec is None else float(min_bout_sec)
        )
        self._reset()

    def _reset(self) -> None:
        self._state = _CLOSED
        self._start_ts = None
        self._start_frame = None
        self._active_sec = 0.0
        self._gap_sec = 0.0
        # 最後一個滿足 CONTINUE 條件的幀——bout 真正關閉時，end_ts/end_frame
        # 用這個，不是「關閉當下那一幀」（那一幀可能已經是寬限期之後的
        # NOT_LICK/NO_CAT，不該算進事件的時間範圍）。
        self._last_active_ts = None
        self._last_active_frame = None

    @property
    def is_open(self) -> bool:
        return self._state != _CLOSED

    def feed(
        self,
        *,
        ts: float,
        frame_idx: int,
        dt: float,
        frame_state: str,
        discontinuity: bool = False,
    ) -> tuple:
        """回傳 `(in_bout, closed_bout)`。

        `discontinuity=True`（來源時間不連續，例如影片被 seek）時，視同
        寬限期立刻用盡：強制關閉目前的 bout（一樣過 min_bout 過濾），不會
        讓 bout 跨過時間跳躍——跟 event_aggregator.py 既有的 discontinuity
        處理原則一致。
        """
        if discontinuity:
            closed = self._close_if_open()
            return self.is_open, closed

        dt = max(0.0, float(dt))
        starts = frame_state in FrameState.LICK
        continues = starts or frame_state == FrameState.LOW_LICK_CONF

        closed_bout = None

        if self._state == _CLOSED:
            if starts:
                self._state = _OPEN
                self._start_ts = ts
                self._start_frame = frame_idx
                self._active_sec = dt
                self._last_active_ts = ts
                self._last_active_frame = frame_idx
        elif continues:
            if self._state == _PENDING_CLOSE:
                self._state = _OPEN
                self._gap_sec = 0.0
            self._active_sec += dt
            self._last_active_ts = ts
            self._last_active_frame = frame_idx
        else:
            if self._state == _OPEN:
                self._state = _PENDING_CLOSE
                self._gap_sec = 0.0
            self._gap_sec += dt
            if self._gap_sec >= self.gap_tolerance_sec:
                closed_bout = self._close_if_open()

        return self.is_open, closed_bout

    def finalize(self) -> Optional[dict]:
        """Session 結束：強制關閉仍開著的 bout（一樣過 min_bout 過濾）。"""
        return self._close_if_open()

    def _close_if_open(self) -> Optional[dict]:
        if self._state == _CLOSED:
            return None
        bout = None
        if self._active_sec >= self.min_bout_sec:
            bout = {
                "start_ts": self._start_ts,
                "end_ts": self._last_active_ts,
                "start_frame": self._start_frame,
                "end_frame": self._last_active_frame,
                "active_sec": self._active_sec,
            }
        self._reset()
        return bout
