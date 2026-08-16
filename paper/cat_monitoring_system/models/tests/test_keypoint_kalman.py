"""models/keypoint_kalman.py 的單元測試。

第 0 階段驗證（見對話紀錄／docs 的「深入」章節）：純合成軌跡＋已知真值，
不碰任何真實影片或模型，快速確認 Kalman filter 的行為在數學上是合理的，
再往下做真的要花算力的 tools/eval_accuracy_*.py 影片比較。
"""

import numpy as np
import pytest

from models.keypoint_kalman import KeypointKalmanFilter, kalman_smooth_sequence
from models.stgcn_model import interpolate_missing


def _linear_trajectory(n_frames, start=(0.0, 0.0), velocity=(2.0, 1.0)):
    """已知真值：等速度直線運動，方便手算誤差。"""
    t = np.arange(n_frames, dtype=np.float64)
    x = start[0] + velocity[0] * t
    y = start[1] + velocity[1] * t
    return np.stack([x, y], axis=1)  # (T, 2)


def _rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


# ── KeypointKalmanFilter 基本行為 ────────────────────────────────────────────


def test_first_measurement_initializes_state_exactly():
    kf = KeypointKalmanFilter()
    z = np.array([10.0, 20.0])
    est = kf.step(z, has_measurement=True)
    assert np.allclose(est, z)


def test_no_measurement_before_any_init_returns_raw_value_unchanged():
    """對齊 interpolate_missing() 「完全沒有有效幀時設為 0」的精神——
    還沒收過任何量測時沒有狀態可外推，只能原樣回傳（呼叫端傳進來的通常
    就是 0,0 這種佔位值）。"""
    kf = KeypointKalmanFilter()
    z = np.array([0.0, 0.0])
    est = kf.step(z, has_measurement=False)
    assert np.allclose(est, z)
    assert kf.x is None  # 狀態仍未初始化


def test_tracks_noiseless_linear_motion_closely():
    """完全沒有雜訊、每幀都有量測時，Kalman 應該幾乎完美貼合真值
    （不能因為做了平滑反而讓乾淨訊號變得不準）。"""
    traj = _linear_trajectory(20)
    kf = KeypointKalmanFilter(process_noise=1.0, measurement_noise=1.0)
    estimates = np.array([kf.step(z, True) for z in traj])
    # 前幾幀狀態還在收斂，只檢查穩定後（第 5 幀之後）的精度
    assert _rmse(estimates[5:], traj[5:]) < 0.5


# ── kalman_smooth_sequence：跟 interpolate_missing() 的核心比較情境 ─────────


def test_gap_at_end_of_window_kalman_beats_flatline_interpolation():
    """關鍵情境（對應對話紀錄的「深入」討論）：遮蔽發生在視窗**尾端**，
    後面沒有任何有效幀可以當內插的另一端——interpolate_missing() 這時
    只能沿用最後一個有效值（flatline，np.interp 對範圍外的點外推成常數），
    Kalman 則會用目前估計的速度繼續外推，對持續移動中的關鍵點應該明顯
    更接近真值。
    """
    n_frames = 20
    gap_start = 14  # 最後 6 幀（含）都遮蔽，視窗內沒有更晚的有效幀
    traj = _linear_trajectory(n_frames, velocity=(3.0, -2.0))  # 持續移動，非靜止

    conf = np.ones((n_frames, 1))
    conf[gap_start:, 0] = 0.0  # 低於門檻，視為缺測
    seq = traj.reshape(n_frames, 1, 2).copy()
    # 缺測幀的「原始觀測值」本身不可信，用真值加大雜訊模擬感測器亂跳，
    # 藉此確認兩種方法在缺測期間真的是各自的補值邏輯在起作用，不是剛好
    # 原始值就已經很準。
    rng = np.random.default_rng(0)
    seq[gap_start:, 0] += rng.normal(scale=5.0, size=(n_frames - gap_start, 2))

    kalman_out = kalman_smooth_sequence(
        seq, conf, threshold=0.5, process_noise=0.5, measurement_noise=2.0
    )
    interp_out = interpolate_missing(seq, conf, threshold=0.5)

    gt_gap = traj[gap_start:]
    kalman_err = _rmse(kalman_out[gap_start:, 0], gt_gap)
    interp_err = _rmse(interp_out[gap_start:, 0], gt_gap)

    assert kalman_err < interp_err, (
        f"預期 Kalman 在視窗尾端缺測情境下誤差更小："
        f"kalman={kalman_err:.3f}, interpolate_missing={interp_err:.3f}"
    )


def test_gap_in_middle_both_methods_reasonable_but_kalman_not_worse_by_much():
    """遮蔽在視窗**中間**、前後都有有效幀時，interpolate_missing() 的線性
    插值對等速度直線運動來說幾乎是最佳解（真值本來就是直線）——這裡不
    要求 Kalman 贏，只要求它不會明顯輸給插值太多，避免「只在極端情境
    測贏、正常情境卻更差」這種選擇性驗證的陷阱。
    """
    n_frames = 20
    traj = _linear_trajectory(n_frames, velocity=(2.0, 1.5))
    conf = np.ones((n_frames, 1))
    conf[8:12, 0] = 0.0  # 中間 4 幀缺測，前後都還有效
    seq = traj.reshape(n_frames, 1, 2).copy()

    kalman_out = kalman_smooth_sequence(
        seq, conf, threshold=0.5, process_noise=0.5, measurement_noise=2.0
    )
    interp_out = interpolate_missing(seq, conf, threshold=0.5)

    gt_gap = traj[8:12]
    kalman_err = _rmse(kalman_out[8:12, 0], gt_gap)
    interp_err = _rmse(interp_out[8:12, 0], gt_gap)

    # interp_err 在這個合成情境下會趨近 0（真值本來就是直線，線性插值幾乎
    # 精確重建），所以用「相對倍率」比較沒有意義（任何非零數乘上 0 還是
    # 0）——改成一個跟軌跡尺度相稱的絕對誤差上限：這條軌跡每幀位移約
    # 2.5 單位，容許 Kalman 誤差在同一個數量級內都算「沒有輸太多」。
    assert kalman_err < 1.0, (
        f"Kalman 在中段缺測（插值理論最佳情境）誤差應該還是很小："
        f"kalman={kalman_err:.3f}, interpolate_missing={interp_err:.3f}"
    )


def test_all_frames_missing_returns_input_unchanged():
    """全程都缺測（例如貓整段時間都不在畫面內）：沒有任何狀態可以外推，
    整段原樣回傳，行為上對齊 interpolate_missing() 的「完全沒有有效幀
    時設為 0」——呼叫端傳進來的缺測佔位值通常就是 0。"""
    seq = np.zeros((10, 1, 2))
    conf = np.zeros((10, 1))
    out = kalman_smooth_sequence(seq, conf, threshold=0.5)
    assert np.allclose(out, seq)


def test_shape_preserved_for_multi_joint_input():
    """確認函式對 (T, V, 2) 的多關節輸入逐關節獨立處理，shape 不變。"""
    n_frames, n_joints = 16, 17
    rng = np.random.default_rng(1)
    seq = rng.normal(size=(n_frames, n_joints, 2))
    conf = np.ones((n_frames, n_joints))
    out = kalman_smooth_sequence(seq, conf)
    assert out.shape == seq.shape
