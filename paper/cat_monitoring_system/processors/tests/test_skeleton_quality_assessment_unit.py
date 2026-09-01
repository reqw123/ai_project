"""
Unit Test：processors/skeleton_quality_assessment.py

第二階段（Unit Test）優先順序第 1 項。這個模組全部是純函式（見模組自己的
docstring：「只依賴 numpy 跟 models.stgcn_model.interpolate_missing」），
每個測試都用手算/獨立幾何公式驗證過的固定座標，不依賴 GPU/YOLO/Flask/
Node-RED/外部網路/真實攝影機。

唯一需要說明的例外：`from models.stgcn_model import interpolate_missing`
這一行，雖然 `interpolate_missing()` 本身是純 numpy 函式、不需要 GPU，但
`models/stgcn_model.py` 這個檔案本身在檔案層級 `import torch`（因為同一個
檔案還定義了 nn.Module 模型類別），所以只要 import 這個模組就需要環境裝有
`torch` 套件——不需要真的有 GPU/CUDA、也不需要任何訓練好的模型檔，純粹是
「torch 這個 Python 套件要能 import」。目前只有 `yolo_new` conda 環境有裝，
所以這份測試在沒裝 torch 的環境下會被安全跳過（`pytest.importorskip`），
不會報錯中斷。

固定座標的選擇：所有測試都只設定 Chest(3)/MidBack(4)/Hip(5) 這三個關節，
其餘 14 個關節保持 (0,0)（本模組所有函式都不會讀取其他關節）。用到的角度/
比例數值都先用獨立的向量幾何公式（不呼叫模組本身的函式）算過一次，確認跟
模組輸出一致，而不是「跑一次相信現在的輸出」。
"""

import numpy as np
import pytest

pytest.importorskip(
    "torch",
    reason=(
        "processors/skeleton_quality_assessment.py 透過 models.stgcn_model "
        "間接需要 torch 套件才能 import（該檔案另外定義了 nn.Module 類別），"
        "此環境未安裝"
    ),
)

from processors import skeleton_quality_assessment as sqa

NUM_JOINTS = 17


def _make_frame(chest=(0, 2), midback=(0, 0), hip=(2, 0)):
    """建立一幀 17 個關鍵點座標，只設定 Chest(3)/MidBack(4)/Hip(5)。"""
    kpts = np.zeros((NUM_JOINTS, 2), dtype=np.float64)
    kpts[3] = chest
    kpts[4] = midback
    kpts[5] = hip
    return kpts


def _make_conf(chest=1.0, midback=1.0, hip=1.0, default=1.0):
    conf = np.full(NUM_JOINTS, default, dtype=np.float64)
    conf[3] = chest
    conf[4] = midback
    conf[5] = hip
    return conf


def _make_window(chest, midback, hip, t=20):
    """建立 T 幀、每幀座標都相同的窗口（信心值全滿），供 evaluate_window() 系列測試使用。"""
    seq = np.zeros((t, NUM_JOINTS, 2), dtype=np.float64)
    conf = np.ones((t, NUM_JOINTS), dtype=np.float64)
    for i in range(t):
        seq[i, 3] = chest
        seq[i, 4] = midback
        seq[i, 5] = hip
    return seq, conf


# ============================================================================
# compute_midback_angle()
# ============================================================================


class TestComputeMidbackAngle:
    def test_right_angle_returns_90_degrees(self):
        kpts = _make_frame(chest=(0, 1), midback=(0, 0), hip=(1, 0))
        assert sqa.compute_midback_angle(kpts, _make_conf()) == pytest.approx(90.0)

    def test_60_degree_angle(self):
        # hip 取單位圓上 60 度處：(cos60, sin60) = (0.5, sqrt(3)/2)
        kpts = _make_frame(chest=(1, 0), midback=(0, 0), hip=(0.5, 0.8660254))
        assert sqa.compute_midback_angle(kpts, _make_conf()) == pytest.approx(60.0, abs=1e-4)

    def test_collinear_points_return_180_degrees(self):
        kpts = _make_frame(chest=(-1, 0), midback=(0, 0), hip=(1, 0))
        assert sqa.compute_midback_angle(kpts, _make_conf()) == pytest.approx(180.0)

    def test_low_chest_confidence_returns_none(self):
        kpts = _make_frame()
        conf = _make_conf(chest=0.1)
        assert sqa.compute_midback_angle(kpts, conf) is None

    def test_low_midback_confidence_returns_none(self):
        kpts = _make_frame()
        conf = _make_conf(midback=0.1)
        assert sqa.compute_midback_angle(kpts, conf) is None

    def test_low_hip_confidence_returns_none(self):
        kpts = _make_frame()
        conf = _make_conf(hip=0.1)
        assert sqa.compute_midback_angle(kpts, conf) is None

    def test_confidence_exactly_at_threshold_is_accepted(self):
        """信心值剛好等於門檻（非低於）應視為足夠：程式碼用 `< kpt_conf_thresh` 判斷。"""
        kpts = _make_frame(chest=(0, 1), midback=(0, 0), hip=(1, 0))
        conf = _make_conf(chest=sqa.BONE_KPT_CONF_THRESHOLD)
        assert sqa.compute_midback_angle(kpts, conf) is not None

    def test_midback_coincident_with_chest_returns_none(self):
        """MidBack 與 Chest 重合（向量長度為 0）時應回傳 None，不應該除以零。"""
        kpts = _make_frame(chest=(0, 0), midback=(0, 0), hip=(1, 0))
        assert sqa.compute_midback_angle(kpts, _make_conf()) is None


# ============================================================================
# compute_bone_stability_overlay()
# ============================================================================


class TestComputeBoneStabilityOverlay:
    def test_offset_ratio_and_angle_match_hand_calculation(self):
        """chest=(0,2), midback=(0,0), hip=(2,0)：
        virtual_pt=(1,1), offset=|midback-virtual_pt|=sqrt(2), body_size=|chest-hip|=sqrt(8)
        → offset_ratio = sqrt(2)/sqrt(8) = 0.5（獨立驗證過的手算值）。
        """
        seq, conf = _make_window((0, 2), (0, 0), (2, 0), t=2)
        result = sqa.compute_bone_stability_overlay(seq, conf)
        assert result["midback_offset_ratio"] == pytest.approx(0.5)
        assert result["midback_angle"] == pytest.approx(90.0)

    def test_low_midback_confidence_gives_nan_offset_ratio(self):
        """所有幀的 MidBack 信心都不足時，沒有任何有效幀可採信，應回傳 NaN。"""
        seq, conf = _make_window((0, 2), (0, 0), (2, 0), t=2)
        conf[:, 4] = 0.1
        result = sqa.compute_bone_stability_overlay(seq, conf)
        assert np.isnan(result["midback_offset_ratio"])

    def test_midback_angle_uses_last_frame_only(self):
        """midback_angle 應該取窗口「最後一幀」，前面幀的角度改變不應影響結果。"""
        seq, conf = _make_window((0, 2), (0, 0), (2, 0), t=3)
        seq[0, 3] = (-1, 0)  # 把第一幀改成完全不同的共線幾何
        seq[0, 4] = (0, 0)
        seq[0, 5] = (1, 0)
        result = sqa.compute_bone_stability_overlay(seq, conf)
        assert result["midback_angle"] == pytest.approx(90.0)  # 仍等於最後一幀（未被修改）的值


# ============================================================================
# compute_body_axis_geometry()
# ============================================================================


class TestComputeBodyAxisGeometry:
    def test_matches_independently_derived_ratios_and_score(self):
        """chest=(0,2), midback=(0,0), hip=(2,0)，用獨立公式算過：
        r2=r3=sqrt(2)/2≈0.7071068，geometry_error≈0.043478，body_axis_score≈93.9778。
        """
        kpts = _make_frame(chest=(0, 2), midback=(0, 0), hip=(2, 0))
        result = sqa.compute_body_axis_geometry(kpts)
        assert result["chest_midback_ratio"] == pytest.approx(0.7071068, abs=1e-6)
        assert result["midback_hip_ratio"] == pytest.approx(0.7071068, abs=1e-6)
        assert result["geometry_error"] == pytest.approx(0.043478, abs=1e-5)
        assert result["body_axis_score"] == pytest.approx(93.9778, abs=1e-3)

    def test_reference_ratios_exactly_matched_gives_score_100(self):
        """R2/R3 恰好等於 BODY_AXIS_REFERENCE_RATIOS 時，geometry_error=0，score=100。"""
        chest_hip_dist = 10.0
        r2 = sqa.BODY_AXIS_REFERENCE_RATIOS["chest_midback"]
        r3 = sqa.BODY_AXIS_REFERENCE_RATIOS["midback_hip"]
        chest = np.array([0.0, 0.0])
        hip = np.array([chest_hip_dist, 0.0])
        # 解兩個距離方程式求 midback 座標（chest_midback=r2*10, midback_hip=r3*10）
        cm = r2 * chest_hip_dist
        mh = r3 * chest_hip_dist
        x = (cm**2 - mh**2 + chest_hip_dist**2) / (2 * chest_hip_dist)
        y = np.sqrt(max(cm**2 - x**2, 0.0))
        kpts = _make_frame(chest=tuple(chest), midback=(x, y), hip=tuple(hip))
        result = sqa.compute_body_axis_geometry(kpts)
        assert result["geometry_error"] == pytest.approx(0.0, abs=1e-9)
        assert result["body_axis_score"] == pytest.approx(100.0, abs=1e-6)

    def test_degenerate_zero_chest_hip_distance_returns_nan(self):
        """Chest 與 Hip 重合（距離為 0）時，所有比例都無法計算，應全部回傳 NaN。"""
        kpts = _make_frame(chest=(1, 1), midback=(0, 0), hip=(1, 1))
        result = sqa.compute_body_axis_geometry(kpts)
        assert np.isnan(result["body_axis_score"])
        assert np.isnan(result["geometry_error"])
        assert np.isnan(result["chest_midback_ratio"])
        assert np.isnan(result["midback_hip_ratio"])


# ============================================================================
# compute_body_axis_score_jitter()
# ============================================================================


class TestComputeBodyAxisScoreJitter:
    def test_amplitude_equals_max_minus_min_of_per_frame_scores(self):
        """驗證振幅定義：逐幀呼叫 compute_body_axis_geometry() 算出的
        body_axis_score，取 max-min——直接用該函式自己算出的分數驗證，
        確保兩者的計算方式互相一致（不是各自獨立猜測公式）。"""
        frame1 = _make_frame(chest=(0, 2), midback=(0, 0), hip=(2, 0))
        frame2 = _make_frame(chest=(0, 2), midback=(5, 5), hip=(2, 0))
        window = np.stack([frame1, frame2, frame1.copy()])

        amplitude, valid_count = sqa.compute_body_axis_score_jitter(window)

        score1 = sqa.compute_body_axis_geometry(frame1)["body_axis_score"]
        score2 = sqa.compute_body_axis_geometry(frame2)["body_axis_score"]
        expected_amplitude = max(score1, score2) - min(score1, score2)

        assert valid_count == 3
        assert amplitude == pytest.approx(expected_amplitude)

    def test_constant_window_has_zero_amplitude(self):
        """每一幀幾何都相同時，振幅應該剛好是 0（不是接近 0）。"""
        frame = _make_frame(chest=(0, 2), midback=(0, 0), hip=(2, 0))
        window = np.stack([frame, frame.copy(), frame.copy()])
        amplitude, valid_count = sqa.compute_body_axis_score_jitter(window)
        assert valid_count == 3
        assert amplitude == pytest.approx(0.0, abs=1e-9)

    def test_too_few_valid_samples_returns_nan(self):
        """有效樣本數低於 BODY_AXIS_MIN_VALID_SAMPLES 時，振幅應為 NaN。"""
        degenerate = _make_frame(chest=(1, 1), midback=(0, 0), hip=(1, 1))  # chest_hip 距離為 0 → NaN
        valid = _make_frame(chest=(0, 2), midback=(0, 0), hip=(2, 0))
        window = np.stack([degenerate, degenerate.copy(), valid])  # 只有 1 個有效樣本
        amplitude, valid_count = sqa.compute_body_axis_score_jitter(window)
        assert valid_count == 1
        assert valid_count < sqa.BODY_AXIS_MIN_VALID_SAMPLES
        assert np.isnan(amplitude)


# ============================================================================
# _is_bad()
# ============================================================================


class TestIsBad:
    def test_above_direction_flags_value_greater_than_threshold(self):
        assert sqa._is_bad(5.0, 3.0, "above") is True

    def test_above_direction_does_not_flag_value_at_threshold(self):
        assert sqa._is_bad(3.0, 3.0, "above") is False

    def test_above_direction_does_not_flag_value_below_threshold(self):
        assert sqa._is_bad(2.0, 3.0, "above") is False

    def test_below_direction_flags_value_less_than_threshold(self):
        assert sqa._is_bad(1.0, 3.0, "below") is True

    def test_below_direction_does_not_flag_value_at_threshold(self):
        assert sqa._is_bad(3.0, 3.0, "below") is False

    def test_outside_range_flags_value_below_low(self):
        assert sqa._is_bad(10.0, (20.0, 160.0), "outside_range") is True

    def test_outside_range_flags_value_above_high(self):
        assert sqa._is_bad(170.0, (20.0, 160.0), "outside_range") is True

    def test_outside_range_does_not_flag_value_inside_range(self):
        assert sqa._is_bad(90.0, (20.0, 160.0), "outside_range") is False

    def test_outside_range_boundary_values_are_not_flagged(self):
        """邊界值本身（等於 low 或 high）不算異常，只有「超出」才算。"""
        assert sqa._is_bad(20.0, (20.0, 160.0), "outside_range") is False
        assert sqa._is_bad(160.0, (20.0, 160.0), "outside_range") is False


# ============================================================================
# _enabled_thresholds()
# ============================================================================


class TestEnabledThresholds:
    def test_all_three_checks_enabled_by_default(self):
        thresholds = sqa._enabled_thresholds()
        assert set(thresholds.keys()) == {
            "midback_offset_ratio",
            "midback_angle",
            "body_axis_score_jitter",
        }

    def test_disabling_one_check_removes_only_that_entry(self, monkeypatch):
        monkeypatch.setattr(sqa, "ENABLE_MIDBACK_OFFSET_CHECK", False)
        thresholds = sqa._enabled_thresholds()
        assert "midback_offset_ratio" not in thresholds
        assert "midback_angle" in thresholds
        assert "body_axis_score_jitter" in thresholds

    def test_disabling_all_checks_returns_empty_dict(self, monkeypatch):
        monkeypatch.setattr(sqa, "ENABLE_MIDBACK_OFFSET_CHECK", False)
        monkeypatch.setattr(sqa, "ENABLE_MIDBACK_ANGLE_CHECK", False)
        monkeypatch.setattr(sqa, "ENABLE_SCORE_JITTER_CHECK", False)
        assert sqa._enabled_thresholds() == {}


# ============================================================================
# evaluate_window()（唯一對外進入點，整合上面所有函式）
# ============================================================================


class TestEvaluateWindow:
    def test_normal_window_is_reliable_with_no_failed_checks(self):
        seq, conf = _make_window((0, 2), (0, 0), (2, 0))
        reliable, details = sqa.evaluate_window(seq, conf)
        assert reliable is True
        assert details["failed_checks"] == []

    def test_excessive_midback_offset_marks_window_unreliable(self):
        seq, conf = _make_window((0, 2), (5, 5), (2, 0))  # MidBack 嚴重偏移
        reliable, details = sqa.evaluate_window(seq, conf)
        assert reliable is False
        assert "midback_offset_ratio" in details["failed_checks"]

    def test_collinear_midback_angle_marks_window_unreliable(self):
        seq, conf = _make_window((-1, 0), (0, 0), (1, 0))  # 三點共線，180 度
        reliable, details = sqa.evaluate_window(seq, conf)
        assert reliable is False
        assert "midback_angle" in details["failed_checks"]

    def test_high_score_jitter_marks_window_unreliable(self):
        """窗口前半段跟後半段的 MidBack 位置差異很大，body_axis_score 振幅超標。"""
        t = 20
        seq = np.zeros((t, NUM_JOINTS, 2), dtype=np.float64)
        conf = np.ones((t, NUM_JOINTS), dtype=np.float64)
        for i in range(t):
            seq[i, 3] = (0, 2)
            seq[i, 5] = (2, 0)
            seq[i, 4] = (0, 0) if i < t // 2 else (5, 5)
        reliable, details = sqa.evaluate_window(seq, conf)
        assert reliable is False
        assert "body_axis_score_jitter" in details["failed_checks"]

    def test_disabling_all_checks_makes_every_window_reliable(self, monkeypatch):
        """三個指標開關全關時，即使幾何明顯異常也不該被判定為不可信。"""
        monkeypatch.setattr(sqa, "ENABLE_MIDBACK_OFFSET_CHECK", False)
        monkeypatch.setattr(sqa, "ENABLE_MIDBACK_ANGLE_CHECK", False)
        monkeypatch.setattr(sqa, "ENABLE_SCORE_JITTER_CHECK", False)
        seq, conf = _make_window((-1, 0), (0, 0), (1, 0))  # 明顯異常的共線幾何
        reliable, details = sqa.evaluate_window(seq, conf)
        assert reliable is True
        assert details["failed_checks"] == []

    def test_malformed_input_fails_safe_and_returns_reliable_true(self):
        """Fail-safe 承諾：任何內部例外都要被攔截，回傳 (True, {})，不得往外拋出例外。"""
        reliable, details = sqa.evaluate_window(None, None)
        assert reliable is True
        assert details == {}
