"""
異常偵測（EMA）

說明：
- `motion_score`：對於每個有效關鍵點（以信心閾值判定），計算當前幀與前一幀座標之位移向量長度，取平均作為當幀的 motion_score。此值代表瞬時運動量。
- `ema_motion`：對 motion_score 做指數移動平均（EMA）以產生穩定的活動度估計；若 EMA 超過門檻則視為異常（abnormal）。

用途：motion_score（及其 EMA）會用於 UI 的活動分數、CSV 的 Motion_Score 欄位，以及作為是否記錄/觸發警報的判斷依據。

"""

import numpy as np
from config import AnomalyDetectionConfig


class AnomalyDetector:
    """基於關鍵點位移的異常檢測器。

    參數:
      - alpha: EMA 平滑係數（0-1），若為 None 則使用 `AnomalyDetectionConfig.EMA_ALPHA`。
      - abnormal_thres: EMA 超過此值視為異常，若為 None 則使用 `AnomalyDetectionConfig.ABNORMAL_THRESHOLD`。
    """

    def __init__(self, alpha=None, abnormal_thres=None):
        self.ema_motion = 0.0
        self.alpha = float(alpha) if alpha is not None else float(AnomalyDetectionConfig.EMA_ALPHA)
        self.abnormal_thres = float(abnormal_thres) if abnormal_thres is not None else float(AnomalyDetectionConfig.ABNORMAL_THRESHOLD)
        self.prev_kpts = None

    def detect(self, kpts, kpt_conf):
        """輸入：當前幀關鍵點 `kpts` (V,2) 與信心 `kpt_conf` (V,)

        返回：(abnormal: bool, activity_value: int[0-100])
        """
        if kpts is None or kpt_conf is None:
            return False, 0
        # 確保只使用 xy 座標（防止傳入含 conf 的 (V,3) 格式）
        kpts_xy = np.asarray(kpts)[:, :2]

        abnormal = False
        activity_value = 0.0
        valid = kpt_conf > AnomalyDetectionConfig.KP_CONF_THRES

        if self.prev_kpts is not None and np.any(valid):
            disp = np.linalg.norm(kpts_xy - self.prev_kpts, axis=1)
            motion_score = float(np.mean(disp[valid]))
            self.ema_motion = self.alpha * motion_score + (1.0 - self.alpha) * self.ema_motion
            abnormal = self.ema_motion > self.abnormal_thres
            norm_motion = min(motion_score / max(AnomalyDetectionConfig.MAX_MOTION, 1e-6), 1.0)
            activity_value = int(norm_motion * 100)
            # 更新 prev_kpts 僅在有有效關鍵點時，避免低信心幀造成虛假大位移
            self.prev_kpts = kpts_xy.copy()
        elif self.prev_kpts is None:
            # 初始化
            self.prev_kpts = kpts_xy.copy()
        # 若 np.any(valid) 為 False，則 prev_kpts 保持不變（跳過這一幀）

        return abnormal, activity_value
