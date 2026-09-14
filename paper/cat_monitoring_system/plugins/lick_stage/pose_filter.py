"""M4：共用 PoseFilter——confidence-aware EMA + 缺點 hold decay + 骨長品質，
輸出 `pose_frame.PoseFrame`。見 `pose_frame.py` 開頭說明目前的交付階段（模組
本身完整、尚未接進 `frame_processor.py`/`analyzer.py`/`ext_body_zones`）。

**跟 `analyzer.py` 現有 `EMA_ALPHA` 的關係與差異**：`analyzer.py` 現在的
`self._ema_kpts` 邏輯是「全部關鍵點用同一個固定 `alpha` 值」的樸素 EMA，
不看每個點各自的信心值。這裡的 `PoseFilter` 是它的**父集合**、不是取代它的
另一套邏輯——`alpha` 這個建構參數維持一模一樣的語意（含 `>=1.0` 即完全
不平滑這個既有慣例，見 `config.py` 的 `EMA_ALPHA` 註解「1.0 = 不平滑，直接
使用原始值」），只是在此基礎上疊加「按每個關鍵點各自的信心值調整實際套用
的平滑係數」，`min_conf`/`hold_decay_frames`/`bone_pairs` 全部留空、不給時，
數值上會跟 `analyzer.py` 現有的樸素 EMA 完全等價（見
`test_pose_filter_unit.py::TestParityWithLegacyEma`）。

**三個功能，刻意保持互相獨立、按需啟用**：
1. **confidence-aware EMA**（一律生效，`alpha<1.0` 時）：每個關鍵點各自的
   實際平滑係數 `effective_alpha_i = alpha * clamp(conf_i, 0, 1)`——信心
   越高，`effective_alpha_i` 越接近 `alpha`（跟樸素 EMA 一樣正常平滑）；
   信心越低，`effective_alpha_i` 越接近 0（幾乎完全不信任這幀的新讀數，
   平滑後的座標幾乎不動）。信心值是 `NaN`（型別上代表「完全沒有這個量測」，
   跟「量到但信心很低」不同）時視同 0 處理，數學上自然落在同一個公式裡，
   不需要另外開分支。
2. **缺點 hold decay**（只有給 `min_conf` 才啟用）：`min_conf` 是一個顯式
   的硬門檻——低於它的讀數不再套用上面的連續信心縮放，改成「凍結」在
   目前平滑值不動（`effective_alpha_i = 0`，不管實際信心值多少），並用
   per-point 計數器累積「連續幾幀都低於門檻」。連續超過 `hold_decay_frames`
   幀仍低於門檻，代表這不是短暫誤判、可能是真的長期遮蔽或追蹤異常，
   凍結太久反而會讓平滑值卡死在舊位置——超過門檻後改用一個很小的固定
   `decay_alpha`（預設 0.05）緩慢地把平滑值拉回真實（哪怕不可靠的）讀數，
   避免永久卡住。`min_conf=None`（預設）＝不啟用這層，所有點永遠走第 1 點
   的連續縮放公式（信心值本身已經隱含「越不可信越少採信」，不是每個呼叫端
   都需要額外的硬門檻）。
3. **骨長品質**（只有給 `bone_pairs` 才啟用）：對每一對定義好的骨骼端點
   （例如 `(KP_CHEST, KP_HIP)`），追蹤一份獨立的「預期骨長」EMA（`bone_
   len_ema_alpha`，預設 0.05——刻意跟位置平滑分開、更慢，避免一次量測異常
   就把「預期值」本身帶偏），每幀比較「這幀量到的骨長」跟「預期骨長」的
   相對偏差，取所有骨骼對的平均偏差換算成 `[0,1]` 的品質分數（1.0＝完全
   一致）。第一幀（每個骨骼對都還沒有「預期值」可比較）回傳 `quality=None`
   （見 `pose_frame.PoseFrame` 的欄位說明，`None` 是「尚不可得」不是
   「品質為 0」）。`bone_pairs=None`（預設）＝完全不追蹤，`quality` 永遠是
   `None`。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from plugins.lick_stage.pose_frame import PoseFrame

_BYPASS_EPS = 1.0 - 1e-9  # 跟 analyzer.py 既有的 `_C.EMA_ALPHA < 1.0 - 1e-9` 同一個門檻


class PoseFilter:
    """逐幀 `update(kpts, kpt_conf, ...)` -> `PoseFrame`。有狀態（維護跨幀的
    平滑位置／缺點計數／骨長 EMA），呼叫端要在新 track/session 開始時呼叫
    `reset()`（本模組不自動偵測「這是不是新的一段」，交由呼叫端決定政策，
    見下方 `update(None, None)` 的說明）。"""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        min_conf: Optional[float] = None,
        hold_decay_frames: int = 0,
        decay_alpha: float = 0.05,
        bone_pairs: Optional[Sequence[Tuple[int, int]]] = None,
        bone_len_ema_alpha: float = 0.05,
    ):
        self.alpha = float(alpha)
        self.min_conf = None if min_conf is None else float(min_conf)
        self.hold_decay_frames = int(hold_decay_frames)
        self.decay_alpha = float(decay_alpha)
        self.bone_pairs = list(bone_pairs) if bone_pairs else []
        self.bone_len_ema_alpha = float(bone_len_ema_alpha)
        self._reset_state()

    def _reset_state(self) -> None:
        self._smoothed: Optional[np.ndarray] = None
        self._miss_streak: Optional[np.ndarray] = None
        self._bone_len_ema = [None] * len(self.bone_pairs)

    def reset(self) -> None:
        """清空跨幀平滑狀態（新 track / 新 session / 明確來源切換時呼叫），
        跟 `analyzer.py` 的 `_reset_transient_state()` 是同一種考量：不該讓
        陳舊姿態繼續拉動下一段真正的資料。"""
        self._reset_state()

    def update(
        self,
        kpts,
        kpt_conf,
        *,
        frame_idx: int = 0,
        source_timestamp: Optional[float] = None,
    ) -> PoseFrame:
        """整幀缺失（`kpts`/`kpt_conf` 任一為 `None`）時，回傳全 `None` 的
        `PoseFrame`，且**不改動任何內部平滑狀態**——這一幀「沒有發生過」，
        下一個真正有姿態的幀會延續中斷前的平滑狀態，不會被當成缺點計入
        `hold_decay_frames`。這跟 `analyzer.py` 現在每次非 lick 幀都主動
        呼叫 `_reset_transient_state()` 清空 `_ema_kpts` 的政策不同——那是
        `analyzer.py` 自己的選擇（避免陳舊姿態拉動下一段真正的舔毛判定），
        不是這個共用模組該內建的假設；要沿用那個政策，呼叫端在對應的時機
        自行呼叫 `reset()` 即可。"""
        if kpts is None or kpt_conf is None:
            return PoseFrame(
                raw_kpts=None,
                raw_conf=None,
                smoothed_kpts=None,
                quality=None,
                frame_idx=frame_idx,
                source_timestamp=source_timestamp,
            )

        raw_kpts = np.asarray(kpts, dtype=np.float64)
        raw_conf = np.asarray(kpt_conf, dtype=np.float64)

        if self.alpha >= _BYPASS_EPS:
            # alpha>=1.0：完全不平滑，直接回傳原始值的拷貝（不維護任何內部
            # 平滑狀態——跟 analyzer.py 的既有 bypass 分支語意一致）。
            smoothed = raw_kpts.copy()
        else:
            smoothed = self._smooth(raw_kpts, raw_conf)

        quality = self._compute_quality(smoothed) if self.bone_pairs else None

        return PoseFrame(
            raw_kpts=raw_kpts,
            raw_conf=raw_conf,
            smoothed_kpts=smoothed,
            quality=quality,
            frame_idx=frame_idx,
            source_timestamp=source_timestamp,
        )

    # ── 私有 ─────────────────────────────────────────────────────────────
    def _smooth(self, raw_kpts: np.ndarray, raw_conf: np.ndarray) -> np.ndarray:
        n = raw_kpts.shape[0]
        if self._smoothed is None or self._smoothed.shape != raw_kpts.shape:
            self._smoothed = raw_kpts.copy()
            self._miss_streak = np.zeros(n, dtype=np.int64)
            return self._smoothed.copy()

        eff_alpha = np.empty(n, dtype=np.float64)
        for i in range(n):
            c = raw_conf[i]
            missing_by_threshold = self.min_conf is not None and (
                not np.isfinite(c) or c < self.min_conf
            )
            if missing_by_threshold:
                self._miss_streak[i] += 1
                if self._miss_streak[i] > self.hold_decay_frames:
                    eff_alpha[i] = self.decay_alpha
                else:
                    eff_alpha[i] = 0.0  # hold：凍結在目前平滑值
            else:
                self._miss_streak[i] = 0
                c_clamped = c if np.isfinite(c) else 0.0
                c_clamped = min(max(c_clamped, 0.0), 1.0)
                eff_alpha[i] = self.alpha * c_clamped

        eff_alpha_col = eff_alpha[:, None]  # broadcast 到 (N,2) 的 x/y
        self._smoothed = eff_alpha_col * raw_kpts + (1.0 - eff_alpha_col) * self._smoothed
        return self._smoothed.copy()

    def _compute_quality(self, kpts: np.ndarray) -> Optional[float]:
        deviations = []
        for idx, (i, j) in enumerate(self.bone_pairs):
            cur_len = float(np.hypot(kpts[i, 0] - kpts[j, 0], kpts[i, 1] - kpts[j, 1]))
            expected = self._bone_len_ema[idx]
            if expected is None:
                self._bone_len_ema[idx] = cur_len
                continue  # 第一次看到這對骨骼，還沒有預期值可比較
            if expected > 1e-6:
                deviations.append(abs(cur_len - expected) / expected)
            self._bone_len_ema[idx] = (
                self.bone_len_ema_alpha * cur_len
                + (1.0 - self.bone_len_ema_alpha) * expected
            )
        if not deviations:
            return None
        mean_dev = sum(deviations) / len(deviations)
        return max(0.0, 1.0 - mean_dev)
