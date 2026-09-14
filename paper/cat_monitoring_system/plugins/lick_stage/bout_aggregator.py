"""M6（第二部分）：同一個 raw bout 內部依 zone 是否『持續』改變切分子事件
（zone hysteresis），取代 event_aggregator.py 舊版「整個 bout 只回報累積
時間最長的 zone、raw_bout 恆為 True」的粗略作法。

跟 action_gate.py 的關係
────────────────────────
`action_gate.py` 決定 bout 的『時間邊界』（有沒有在舔、bout 何時開始/
結束）；這裡處理的是同一段時間邊界『內部』的『舔的是哪裡』——貓咪真的
換部位時（例如先舔前爪、再換舔後爪）該切成兩筆事件；但候選評分（見
contact_regions.py / regions.py 的 AMBIGUITY_MARGIN）在相鄰 zone 間偶爾
抖動、或短暫幾幀被 NO_REGION_HIT 判成 unassigned，都不該被當成「真的換
部位」而切出一堆碎片事件。兩種 hysteresis 維度不同（時間 vs 部位），
刻意分成兩個檔案，不合併成一個過度肥大的狀態機。

設計：Schmitt-trigger 式的「候選 streak」
──────────────────────────────────────
- 一個 bout 內部維護一個「已確認 zone」（confirmed zone）與目前正在累積
  中的 `_ActiveBout` 內容（等同舊版 event_aggregator.py 裡的那個類別，
  這裡改由本模組持有，因為內容累積本來就是「同一段 zone 的內容該歸
  哪一筆事件」這個問題的一部分）。
- 收到 assigned 幀且其 zone_l1 != confirmed zone 時，累加一個「候選
  streak」（同一個候選 zone 連續累積的秒數；中間穿插 unassigned/
  LOW_LICK_CONF 幀不會推進也不會打斷 streak——這些幀根本不會被餵進來，
  見下方 `feed()` 呼叫端的既有條件，等同 gap 寬限同一種「短暫沒命中不該
  讓已累積的證據歸零」的精神）。
- 候選 streak 累積到 `zone_switch_min_sec` 才『確認』真的換部位：把
  目前累積中的內容結算成一筆子事件（`raw_bout=False`），並開一個新的
  `_ActiveBout` 接續。**新事件的起點是「確認的這一幀」，不是回溯到
  streak 開始的那一幀**——候選期間的幀誠實留在舊事件裡。這是刻意的
  簡化：不做回溯式重新分配（需要額外的逐幀緩衝區/復原機制），讓狀態機
  保持跟 `action_gate.py` 一樣單純、單向、可逐幀驗證；代價是真正換部位
  的邊界時間戳最多會有 `zone_switch_min_sec` 的滯後，這裡明確記錄，
  不是 bug。
- 候選 zone 又變回 confirmed zone（或候選中途換成第三種 zone）：候選
  streak 歸零重算，不觸發切分——這就是防抖動的核心。

Shadow 模式
───────────
`zone_switch_min_sec=None`（`EventAggregator` 的預設）＝完全不啟用這個
模組的切分邏輯，`_ActiveBout` 貫穿整個 raw bout 沒有中途切分（跟 M2、
M6 前半——只有 action_gate.py 生效時——完全一樣，`raw_bout` 恆為
`True`）。這裡刻意不比照 `action_gate.py` 用 `0.0` 當「關閉」的哨兵值：
`0.0` 對時間類參數（寬限期/最短長度）本身就是合法邊界（沒有寬限、沒有
最短限制，仍是完整定義的行為）；但 `zone_switch_min_sec=0.0` 語意上是
「任何一幀不同 zone 就切」——是完全不同、更激進的合法行為，不能拿來
借代『關閉』。所以這裡改用 `None` 當唯一的『未啟用』哨兵，`event_
aggregator.py` 沒收到明確數值就完全不建立切分邏輯。

呼叫端注意事項（min_bout_sec 與 zone_switch_min_sec 的交互）
──────────────────────────────────────────────────────────
`action_gate.py` 的 `min_bout_sec` 過濾是在**整個 raw bout 真正關閉時**
才判斷（根據 bout 的總累積時長丟棄太短的 bout）；但 zone 切分可能在 bout
仍開著、遠早於 bout 結束前就已經送出子事件。如果 `min_bout_sec` 設得比
`zone_switch_min_sec` 還大很多，理論上可能出現「已經送出的子事件，事後
發現整個 raw bout 被 min_bout 丟棄」的邊界情形——這裡不做跨模組的交易式
回滾（複雜度不成比例）。目前的預設值 `MIN_BOUT_SEC=0.5s` ≤
`ZONE_SWITCH_MIN_SEC=1.0s`（見 config.py），不會觸發這個情形；自訂設定
時請維持 `min_bout_sec` 不明顯大於 `zone_switch_min_sec`。
"""

from __future__ import annotations

import statistics as _pystats
from typing import Optional

from plugins.lick_stage.analysis_context import ZoneL1


class _ActiveBout:
    """單一子事件（可能就是整個 raw bout，也可能是切分後的一段）的內容
    累加器。逐幀 `add()`，結束時 `to_event()` 產出事件 dict。"""

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

    def to_event(self, raw_bout: bool = True) -> dict:
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
            # M6（2026-09-14 第二部分）：raw_bout=True 代表這筆事件對應
            # 「整個」raw action-gate bout，沒有被 zone hysteresis 切分過
            # （zone_switch_min_sec 未啟用，或啟用了但這段 bout 裡 zone
            # 從沒持續切換過）；raw_bout=False 代表這是切分後的子事件之一。
            "raw_bout": raw_bout,
        }


class ZoneHysteresis:
    """單一 raw bout 內的 zone 切分狀態機，見模組開頭完整設計說明。

    呼叫端（`event_aggregator.py`）只在 `frame_state in FrameState.LICK`
    的幀呼叫 `feed()`（LOW_LICK_CONF / NOT_LICK / NO_CAT 幀完全不會餵進
    來，就跟舊版 `_ActiveBout.add()` 的既有呼叫條件一模一樣，是 shadow
    模式成立的前提之一）。`feed()` 回傳非 None 代表一段子事件因為 zone
    真正切換而提前結算，呼叫端要立刻當成一筆完整事件送出（不等 bout
    整個關閉）。bout 真正關閉時，呼叫端另外呼叫 `flush_final()` 結算
    最後（或唯一）一段內容。
    """

    def __init__(self, *, zone_switch_min_sec: Optional[float] = None):
        self.zone_switch_min_sec = (
            None if zone_switch_min_sec is None else float(zone_switch_min_sec)
        )
        self._active: Optional[_ActiveBout] = None
        self._confirmed_zone = None
        self._streak_zone = None
        self._streak_sec = 0.0
        self._split_count = 0

    def feed(
        self, ts, frame_idx, dt, l1, l2, action_score, pose_quality, reason_code, assigned
    ) -> Optional[dict]:
        if self._active is None:
            self._active = _ActiveBout(ts, frame_idx)

        finished = None
        if self.zone_switch_min_sec is not None and assigned and l1 != ZoneL1.UNKNOWN:
            if self._confirmed_zone is None:
                # 這個 bout 第一個有效命中的 zone——直接當作已確認 zone，
                # 不需要（也不能）經過候選 streak。
                self._confirmed_zone = l1
            elif l1 != self._confirmed_zone:
                if l1 == self._streak_zone:
                    self._streak_sec += dt
                else:
                    self._streak_zone = l1
                    self._streak_sec = dt
                if self._streak_sec >= self.zone_switch_min_sec:
                    # 候選 streak 撐夠久，確認真的換部位：結算目前累積的
                    # 內容為一筆子事件，開新的 _ActiveBout 接續（見模組
                    # 開頭「新事件起點」的說明）。
                    finished = self._active.to_event(raw_bout=False)
                    self._split_count += 1
                    self._active = _ActiveBout(ts, frame_idx)
                    self._confirmed_zone = l1
                    self._streak_zone = None
                    self._streak_sec = 0.0
            else:
                # 候選又跳回已確認 zone（或本來就沒有候選）：抖動吸收，
                # streak 歸零，不觸發切分。
                self._streak_zone = None
                self._streak_sec = 0.0

        self._active.add(ts, frame_idx, dt, l1, l2, action_score, pose_quality, reason_code, assigned)
        return finished

    def flush_final(self) -> Optional[dict]:
        """bout 真正關閉（或 session 結束）時呼叫：結算最後一段內容。
        `raw_bout` 只有在這個 bout 全程從未觸發過切分時才是 `True`——
        觸發過切分的話，就連最後這一段也只是眾多子事件之一，誠實標
        `False`。"""
        if self._active is None:
            return None
        ev = self._active.to_event(raw_bout=(self._split_count == 0))
        self._active = None
        return ev
