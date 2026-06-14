"""
靜止偵測（滾動均值閾值）

改進說明（v2）：
1. 正規化：motion_score 除以 body_size（胸部(3)→髖部(5) 距離 ÷ 100），與訓練管線
   normalize_skeleton_coords 一致，消除拍攝距離對閾值的影響。
   STILL_MOTION_THRESHOLD 單位為 body_fraction × 100。

2. 排除尾巴關鍵點（14/15/16）：貓休息時尾巴甩動不代表活動狀態，
   改用核心骨架（鼻尖、耳朵、胸、中背、髖、四肢）計算位移。
   若核心關鍵點全部信心不足，fallback 至所有有效關鍵點。

3. 偵測遺失處理：短暫遺失（≤ MAX_MISS_FRAMES 幀）保留 prev_kpts，
   避免 YOLO 在活動中漏幀、重新偵測後立刻誤算大位移；
   長時間遺失（貓離開畫面）才清空以防止跨段假位移。
   遺失期間 is_still 根據現有視窗狀態判斷（不強制設為 False）。
"""

from collections import deque
import numpy as np
from config import AnomalyDetectionConfig

_CHEST_IDX    = 3
_HIP_IDX      = 5
_TAIL_JOINTS  = (14, 15, 16)
_MAX_MISS_FRAMES = 5   # 超過此連續遺失幀數才清空 prev_kpts


class AnomalyDetector:
    """基於關鍵點位移滾動均值的靜止偵測器（體型正規化版）。

    motion_score 單位：body_fraction × 100（每幀位移 / 胸→髖距離 × 100）
      - 靜止（呼吸抖動）：< 2
      - 小動作（尾巴/眨眼）：2 ~ 5
      - 舔毛 / 抓撓：        5 ~ 15
      - 走路：                > 10

    still_threshold / rolling_window 可覆寫 config 預設值，方便測試腳本獨立調整。
    """

    def __init__(self, still_threshold=None, rolling_window=None,
                 kp_conf_thres=None, stride=None):
        self.prev_kpts         = None
        self._miss_count       = 0      # 連續偵測遺失幀計數
        self.last_motion_score = 0.0

        window_size = (int(rolling_window) if rolling_window is not None
                       else AnomalyDetectionConfig.ROLLING_WINDOW_SIZE)
        self._motion_window: deque = deque(maxlen=window_size)
        self._still_threshold = (
            float(still_threshold) if still_threshold is not None
            else float(AnomalyDetectionConfig.STILL_MOTION_THRESHOLD)
        )
        self._kp_conf_thres = (
            float(kp_conf_thres) if kp_conf_thres is not None
            else float(AnomalyDetectionConfig.KP_CONF_THRES)
        )
        self._stride       = max(1, int(stride)) if stride is not None else 1
        self._stride_count = 0

    def reset(self):
        """清空所有累積狀態（motion_window、prev_kpts 等），保留門檻與視窗大小設定。
        切換影片或回到影片開頭時呼叫，確保前段資料不污染新段判斷。"""
        self.prev_kpts         = None
        self._miss_count       = 0
        self.last_motion_score = 0.0
        self._motion_window.clear()
        self._stride_count     = 0

    # ── 內部工具 ──────────────────────────────────────────────────────────────

    def _is_still_from_window(self) -> bool:
        """依現有視窗內容判斷靜止狀態；視窗未暖機時預設非靜止。"""
        if len(self._motion_window) < 2:
            return False
        return (sum(self._motion_window) / len(self._motion_window)) < self._still_threshold

    def _body_size(self, kpts_xy: np.ndarray, valid: np.ndarray) -> float:
        """胸(3)→髖(5) 距離 ÷ 100 作為正規化分母，與訓練管線一致。
        除以 100 使 motion_score 放大 100 倍，數值更易閱讀與設定門檻。
        信心不足時回傳 0.01（保持相同縮放比例）。"""
        if (_CHEST_IDX < len(valid) and _HIP_IDX < len(valid)
                and valid[_CHEST_IDX] and valid[_HIP_IDX]):
            d = float(np.linalg.norm(kpts_xy[_CHEST_IDX] - kpts_xy[_HIP_IDX]))
            if d > 1.0:
                return d / 100.0
        return 0.01

    # ── 主要偵測介面 ──────────────────────────────────────────────────────────

    def detect(self, kpts, kpt_conf):
        """輸入：當前幀關鍵點 kpts (V,2) 與信心 kpt_conf (V,)

        返回：(is_still: bool, activity_value: int[0-100])
        """
        if kpts is None or kpt_conf is None:
            self._miss_count += 1
            if self._miss_count > _MAX_MISS_FRAMES:
                self.prev_kpts = None   # 長時間遺失→清空，防止跨段假大位移
            # 遺失期間依現有視窗狀態判斷，不強制回 False
            return self._is_still_from_window(), 0

        kpts_xy = np.asarray(kpts, dtype=np.float32)
        if kpts_xy.ndim == 2 and kpts_xy.shape[1] > 2:
            kpts_xy = kpts_xy[:, :2]

        valid   = np.asarray(kpt_conf) > self._kp_conf_thres
        body_sz = self._body_size(kpts_xy, valid)

        if self.prev_kpts is not None:
            disp = np.linalg.norm(kpts_xy - self.prev_kpts, axis=1)

            # 排除尾巴：貓靜止時尾巴甩動不代表活動
            core_valid = valid.copy()
            for t in _TAIL_JOINTS:
                if t < len(core_valid):
                    core_valid[t] = False

            if np.any(core_valid):
                motion_score = float(np.mean(disp[core_valid])) / body_sz
            elif np.any(valid):
                motion_score = float(np.mean(disp[valid])) / body_sz   # fallback
            else:
                motion_score = 0.0

            self.last_motion_score = motion_score
            self._stride_count += 1
            if self._stride_count % self._stride == 0:
                self._motion_window.append(motion_score)

        self.prev_kpts   = kpts_xy.copy()
        self._miss_count = 0

        is_still       = self._is_still_from_window()
        norm_motion    = min(self.last_motion_score / max(AnomalyDetectionConfig.MAX_MOTION, 1e-6), 1.0)
        activity_value = int(norm_motion * 100)

        return is_still, activity_value
