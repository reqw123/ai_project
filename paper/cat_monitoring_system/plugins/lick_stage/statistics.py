"""舔毛統計累計：動作偵測時間與部位定位時間**分開計算**。

說明書「統計分母與計算口徑」的落地點。現行 `total_lick_time` 僅由「成功
指派部位」加總，會讓高 NO_TARGET 的資料看起來舔毛時間較少，並錯誤提高
已命中部位的比例。這裡把時間拆成三個互不重疊的分母：

    stgcn_lick_sec      = 所有 LICK_ASSIGNED + LICK_UNASSIGNED 的 dt 總和
    assigned_zone_sec   = 只有 LICK_ASSIGNED 的 dt 總和
    unassigned_lick_sec = 只有 LICK_UNASSIGNED 的 dt 總和

必要不變式（允許一個影格誤差）：
    stgcn_lick_sec == assigned_zone_sec + unassigned_lick_sec
    sum(每個 zone 的 time_sec) == assigned_zone_sec

另外累計觀測分母，供 lick_sec_per_valid_hour 等跨時段比較使用：
    observed_sec        = 所有幀的 dt 總和（含 NO_CAT）
    no_cat_sec          = NO_CAT 的 dt 總和
    valid_observed_sec  = observed_sec - no_cat_sec（畫面裡有貓的時間）
"""

from plugins.lick_stage.analysis_context import FrameState

_ZONES = ("BODY", "FL", "FR", "HL", "HR", "AMBIGUOUS")

# Maps raw zone labels to statistics keys
_LABEL_TO_KEY = {
    "BODY_CENTER": "BODY",
    "FL": "FL",
    "FR": "FR",
    "HL": "HL",
    "HR": "HR",
    # M5：候選評分最高分/次高分差距太小時的結果（見 contact_regions.py
    # find_nearest_zone()）。歸為自己獨立的一個 zone bucket，語意是「確實
    # 碰觸到身體，只是幾何上無法可靠分辨是哪個相鄰區域」——跟 NO_TARGET
    # （完全沒碰到任何區域）是不同的失敗模式，不該混在一起。
    "AMBIGUOUS": "AMBIGUOUS",
}


def _stats_key(zone_label: str) -> str:
    return _LABEL_TO_KEY.get(zone_label, "")


class LickStatistics:
    """
    Accumulates per-zone lick time and hit counts with bout tracking.
    Not thread-safe; must be updated from a single thread.
    """

    def __init__(self):
        self._time: dict = {z: 0.0 for z in _ZONES}
        self._hits: dict = {z: 0 for z in _ZONES}
        # bout tracking
        self._bout_count: dict = {z: 0 for z in _ZONES}
        self._bout_sec: dict = {z: 0.0 for z in _ZONES}
        self._active_bout_zone: str = ""
        self._active_bout_sec: float = 0.0
        self._prev_key: str = ""

        # ── 第一階段新增：三種舔毛秒數 + 觀測分母（說明書「統計分母」）──────
        self.observed_sec: float = 0.0
        self.no_cat_sec: float = 0.0
        self.valid_observed_sec: float = 0.0
        self.stgcn_lick_sec: float = 0.0
        self.assigned_zone_sec: float = 0.0
        self.unassigned_lick_sec: float = 0.0
        # 未過門檻的 lick 候選秒數：另存，不納入主統計（LOW_LICK_CONF）
        self.low_conf_lick_sec: float = 0.0
        # 時間不連續 / 倒退事件次數（seek、排程恢復、大幅掉幀），僅供稽核
        self.discontinuity_count: int = 0

    # ──────────────────────────────────────────────────────────────────────
    def update(
        self,
        zone_label: str,
        dt_sec: float,
        frame_state: str = FrameState.NO_CAT,
        *,
        discontinuity: bool = False,
    ) -> None:
        """Call once per frame.

        zone_label  — raw label from find_nearest_zone (BODY_CENTER / FL / … / NO_TARGET)
        dt_sec      — 來源時間差（秒），已由 SourceClock 夾鉗
        frame_state — 見 analysis_context.FrameState（五種互斥狀態之一）
        discontinuity — True 時這一幀的 dt 完全不累加（僅記入 discarded_dt_sec）
        """
        dt = max(0.0, float(dt_sec))

        if discontinuity:
            self.discontinuity_count += 1
        if discontinuity or dt <= 0.0:
            # 時間不連續：結算開放中的 bout，避免跨越斷點的 bout 被縫在一起
            self._close_active_bout()
            self._prev_key = ""
            return

        self.observed_sec += dt

        if frame_state == FrameState.NO_CAT:
            self.no_cat_sec += dt
            self._close_active_bout()
            self._prev_key = ""
            return

        # 以下 frame_state ∈ {NOT_LICK, LOW_LICK_CONF, LICK_ASSIGNED, LICK_UNASSIGNED}
        self.valid_observed_sec += dt

        if frame_state == FrameState.LOW_LICK_CONF:
            self.low_conf_lick_sec += dt
            self._close_active_bout()
            self._prev_key = ""
            return

        if frame_state == FrameState.LICK_UNASSIGNED:
            self.stgcn_lick_sec += dt
            self.unassigned_lick_sec += dt
            self._close_active_bout()
            self._prev_key = ""
            return

        if frame_state == FrameState.LICK_ASSIGNED:
            self.stgcn_lick_sec += dt
            self.assigned_zone_sec += dt
            self._accumulate_zone(zone_label, dt)
            return

        # NOT_LICK（或未知狀態）：不累加任何舔毛時間，只結算 bout
        self._close_active_bout()
        self._prev_key = ""

    # ── 私有：per-zone 時間 / hits / bout ─────────────────────────────────
    def _accumulate_zone(self, zone_label: str, dt: float) -> None:
        key = _stats_key(zone_label)
        if not key:
            # LICK_ASSIGNED 理論上一定帶合法 zone；萬一沒有，當成未指派處理，
            # 維持 assigned_zone_sec == sum(zone time) 的不變式不被破壞。
            self.assigned_zone_sec -= dt
            self.unassigned_lick_sec += dt
            self._close_active_bout()
            self._prev_key = ""
            return

        # Hit = transition *into* a lick zone
        if key != self._prev_key:
            self._hits[key] += 1
        self._time[key] += dt

        if key == self._active_bout_zone:
            self._active_bout_sec += dt
        else:
            self._close_active_bout()
            self._active_bout_zone = key
            self._active_bout_sec = dt

        self._prev_key = key

    def _close_active_bout(self) -> None:
        if self._active_bout_zone and self._active_bout_sec > 0:
            self._bout_count[self._active_bout_zone] += 1
            self._bout_sec[self._active_bout_zone] += self._active_bout_sec
        self._active_bout_zone = ""
        self._active_bout_sec = 0.0

    def finalize(self) -> None:
        """Session 結束時呼叫：結算最後一段尚未關閉的 bout（說明書
        「Session 生命週期：finish_session 結算 active bout」）。"""
        self._close_active_bout()
        self._prev_key = ""

    # ── 讀取 ─────────────────────────────────────────────────────────────
    def best_zone(self) -> str:
        """Return the zone with the highest accumulated lick time."""
        total = sum(self._time.values())
        if total <= 0:
            return "NO_TARGET"
        return max(self._time, key=self._time.get)

    def zone_stats(self, key: str):
        """Return (hits, time_sec) for the given statistics key."""
        return self._hits.get(key, 0), self._time.get(key, 0.0)

    def zone_bout(self, key: str):
        """Return (bout_count, bout_sec) for the given statistics key."""
        return self._bout_count.get(key, 0), self._bout_sec.get(key, 0.0)

    def zone_coverage_ratio(self) -> float:
        """assigned_zone_sec ÷ stgcn_lick_sec —— 判斷部位分布是否可信。"""
        if self.stgcn_lick_sec <= 1e-9:
            return 0.0
        return self.assigned_zone_sec / self.stgcn_lick_sec

    def unknown_rate(self) -> float:
        """unassigned_lick_sec ÷ stgcn_lick_sec —— 回報系統可觀測限制。"""
        if self.stgcn_lick_sec <= 1e-9:
            return 0.0
        return self.unassigned_lick_sec / self.stgcn_lick_sec

    def invariant_error(self) -> float:
        """|stgcn_lick_sec - (assigned + unassigned)| —— 應接近 0。"""
        return abs(
            self.stgcn_lick_sec - (self.assigned_zone_sec + self.unassigned_lick_sec)
        )
