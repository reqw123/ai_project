"""逐關鍵點 2D 等速度（constant-velocity）Kalman filter。

**動機**：`stgcn_model.interpolate_missing()` 用線性插值補低信心/缺測的關鍵點
——本質上是「事後在兩個已知點之間走直線」，需要未來的有效幀才能插值，且
不利用「這個關節現在往哪個方向、多快在移動」這個資訊。Kalman filter 是
`interpolate_missing()` 同一個插入點（餵進 ST-GCN 前的關鍵點序列前處理）的
替代方案：遮蔽期間用當下估計的速度外推位置（predict-only），而不是單純
補直線。

**這是「深入研究 T-LEAP 式時序遮蔽處理」評估工作的第一步（路線 A：Kalman
filter 後處理，不需要重新訓練 ST-GCN 就能先驗證關鍵點層級的效果）**，
詳見 `docs/YOLO-Pose應用文獻與專案優化建議.md`「深入」章節。

**重要限制（誠實揭露）**：目前部署的 ST-GCN checkpoint（見
`config.py` 的 `STGCNTrainingConfig`／`stgcn_config.yaml` 的
`KP_EMA_ALPHA: 1.0`）是用「完全不平滑」的關鍵點序列訓練出來的。用這個
模組把關鍵點平滑過再餵給那個 checkpoint，等於讓模型吃到跟訓練時分布不同
的輸入——`tools/eval_accuracy_smoothing_compare.py` 就是為了實測這個分布偏移在
實務上是幫助還是傷害而寫的，結果不能想當然爾地預設會變好，需要真的跑過
評估腳本比較準確度才算數。

只用 numpy，不依賴 scipy/filterpy，跟 `analytics/deviation.py` 的
stdlib-only 慣例一致。
"""

from __future__ import annotations

import numpy as np


class KeypointKalmanFilter:
    """單一 2D 關鍵點的等速度 Kalman filter。

    狀態向量 ``[x, y, vx, vy]``；量測只有 ``[x, y]``。信心值低於門檻時視為
    「這一幀沒有量測」，只做 predict（用當下的速度估計外推位置），不做
    update——這是相對於線性插值的核心差異：遮蔽期間仍然有「速度」這個
    先驗可用，不是插值那種「假設兩個已知點之間走直線」的事後補值，也不
    需要往未來看才能補值（線上即時也適用，見 docs 的「深入」章節路線 A）。

    時間單位用「幀」而非真實時間戳（`dt=1.0`），因為輸入序列本來就是等
    間隔取樣（`STGCNConfig.WINDOW_STRIDE` 固定步長），不需要額外處理變動
    時間間隔。
    """

    def __init__(self, process_noise: float = 90.0, measurement_noise: float = 70.0):
        # process_noise 越大 → 越信任「最近的量測」、平滑程度越弱；
        # measurement_noise 越大 → 越信任「等速度運動模型的預測」、平滑程度越強。
        # 兩者的相對大小（Q/R 比值）決定平滑強度，比絕對值本身更重要。
        #
        # 2026-08-11 用真實 YOLO 偵測結果實測校準（見
        # docs/YOLO-Pose應用文獻與專案優化建議.md「後續實作進度」）：
        # - measurement_noise：拿 stop 類別影片（幾乎靜止）算高信心幀間的
        #   座標抖動變異數，實測 ≈70（單位：像素²，因為 Kalman filter 在
        #   frame_processor.py/0_train_gcn.py 的插入點都還在 YOLO 原始像素
        #   座標，flip_normalize/normalize_skeleton_coords 之前套用）。
        # - process_noise：拿 shake 類別影片（快速動作）算「真實軌跡 vs.
        #   等速度模型預測」的殘差變異數，實測 ≈94（同樣是像素²）。
        # 這兩個數字取代原本沒有實測依據的 1.0/5.0——原本的絕對尺度小了
        # 快兩個數量級，相對於實際關鍵點座標（數百像素量級）的變化幅度，
        # 會讓 Kalman 的協方差很快收斂到近乎零、對新量測反應過度遲鈍。
        #
        # **已知限制，不是最終答案**：這只是單支 stop 影片 + 單支 shake
        # 影片的一次性測量，沒有做到每個關節分開估、沒有跨多支影片取統計
        # 穩健值，而且 stop／shake 兩種行為的真實 Q 差異很大（shake 需要
        # 的 Q 遠高於 stop，用同一組全域參數本質上是妥協）——真的要選定
        # 正式使用的數值，應該用 tools/eval_accuracy_smoothing_compare.py
        # 的 SMOOTHING_CONFIGS／KALMAN_PARAM_SWEEP 掃過幾組實際跑準確度
        # 比較，而不是只看這裡的統計量。
        self.F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None

    def _init_state(self, z: np.ndarray) -> None:
        self.x = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0  # 初始不確定性大，讓第一筆量測快速拉近

    def _predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _update(self, z: np.ndarray) -> None:
        innovation = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ innovation
        self.P = (np.eye(4) - k @ self.H) @ self.P

    def step(self, z: np.ndarray, has_measurement: bool) -> np.ndarray:
        """處理一幀。``has_measurement=False`` 時只 predict、不 update
        （模擬遮蔽/低信心幀）。回傳這一幀的估計位置 ``[x, y]``。"""
        if self.x is None:
            if has_measurement:
                self._init_state(z)
                return self.x[:2].copy()
            return z  # 還沒收過任何有效量測，沒有狀態可外推，原樣回傳
        self._predict()
        if has_measurement:
            self._update(z)
        return self.x[:2].copy()


def kalman_smooth_sequence(
    sequence: np.ndarray,
    conf: np.ndarray,
    threshold: float = 0.1,
    process_noise: float = 90.0,
    measurement_noise: float = 70.0,
) -> np.ndarray:
    """跟 ``stgcn_model.interpolate_missing()`` 同一個插入點的替代方案。

    Parameters
    ----------
    sequence : (T, V, 2) 原始關鍵點座標
    conf : (T, V) 每幀每個關鍵點的信心值
    threshold : 信心值門檻，跟 ``interpolate_missing`` 預設值一致（0.1），
        低於此值視為「這一幀這個關節缺測」
    process_noise, measurement_noise : 見 ``KeypointKalmanFilter`` 說明

    Returns
    -------
    (T, V, 2) 平滑後的關鍵點座標
    """
    seq = sequence.copy()
    _, num_joints, _ = seq.shape
    for v in range(num_joints):
        kf = KeypointKalmanFilter(process_noise, measurement_noise)
        for t in range(seq.shape[0]):
            has_meas = bool(conf[t, v] > threshold)
            seq[t, v] = kf.step(seq[t, v], has_meas)
    return seq
