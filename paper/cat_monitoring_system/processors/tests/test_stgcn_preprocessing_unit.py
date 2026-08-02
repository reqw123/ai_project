"""
Unit Test：models/stgcn_model.py 前處理管線

第二階段（Unit Test）優先順序第 2 項。涵蓋訓練與推論共用的前處理鏈：
`interpolate_missing → flip_normalize → orientation_normalize →
normalize_skeleton_coords → add_velocity_feature → compute_bone_feature →
build_feature_tensor`——文件明確要求這條鏈「順序不可更動」，是全系統
正確性的地基。

全部是純 numpy 函式，測試座標都先用獨立計算（手算幾何、或用旋轉矩陣/
線性插值公式反推）驗證過，不是「跑一次相信現在的輸出」。

需要環境裝有 `torch`（`models/stgcn_model.py` 檔案層級 `import torch`，
因為同一支檔案也定義了神經網路類別；這裡用到的函式本身純 numpy、不需要
GPU/CUDA），沒裝的環境會用 `pytest.importorskip` 安全跳過。
"""

import numpy as np
import pytest

pytest.importorskip(
    "torch",
    reason="models/stgcn_model.py 檔案層級 import torch 才能載入，此環境未安裝",
)

from models.stgcn_model import (
    add_velocity_feature,
    build_feature_tensor,
    compute_bone_feature,
    compute_bone_motion_feature,
    flip_normalize,
    interpolate_missing,
    normalize_skeleton_coords,
    orientation_normalize,
)

NUM_JOINTS = 6  # 測試只需要 chest(3)/midback(4)/hip(5) 這幾個索引位置存在即可


# ============================================================================
# interpolate_missing()
# ============================================================================


class TestInterpolateMissing:
    def test_high_confidence_values_pass_through_unchanged(self):
        seq = np.zeros((5, 1, 2))
        seq[:, 0, 0] = [0, 1, 2, 3, 4]
        conf = np.ones((5, 1))
        out = interpolate_missing(seq, conf, threshold=0.1)
        assert out[:, 0, 0].tolist() == pytest.approx([0, 1, 2, 3, 4])

    def test_single_low_confidence_frame_gets_linearly_interpolated(self):
        """第 2 幀信心不足，應該被鄰近幀線性插值補上（座標本身在同一條直線上，
        插值結果應等於原值 2.0）。"""
        seq = np.zeros((5, 1, 2))
        seq[:, 0, 0] = [0, 1, 2, 3, 4]
        conf = np.array([[1.0], [1.0], [0.05], [1.0], [1.0]])
        out = interpolate_missing(seq, conf, threshold=0.1)
        assert out[2, 0, 0] == pytest.approx(2.0)

    def test_all_frames_low_confidence_zeros_out_that_joint(self):
        seq = np.ones((3, 1, 2)) * 5.0
        conf = np.full((3, 1), 0.05)
        out = interpolate_missing(seq, conf, threshold=0.1)
        assert np.all(out[:, 0, :] == 0.0)

    def test_confidence_exactly_at_threshold_is_treated_as_invalid(self):
        """程式碼用 `conf > threshold`（嚴格大於），等於門檻值本身應視為無效。"""
        seq = np.zeros((3, 1, 2))
        seq[:, 0, 0] = [10, 20, 30]
        conf = np.array([[1.0], [0.1], [1.0]])  # 中間那幀信心「剛好」等於門檻
        out = interpolate_missing(seq, conf, threshold=0.1)
        assert out[1, 0, 0] == pytest.approx(20.0)  # 仍被視為缺失、需插值（剛好插出原值）

    def test_does_not_mutate_the_input_array(self):
        seq = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        seq_copy = seq.copy()
        conf = np.zeros((2, 1))  # 全部視為缺失，觸發賦值路徑
        interpolate_missing(seq, conf, threshold=0.1)
        assert np.array_equal(seq, seq_copy), "interpolate_missing 不應該修改傳入的原始陣列"


# ============================================================================
# flip_normalize()
# ============================================================================


class TestFlipNormalize:
    def test_midback_left_of_hip_gets_flipped_to_the_right(self):
        seq = np.zeros((2, NUM_JOINTS, 2))
        seq[:, 4, 0] = [3, 3]
        seq[:, 5, 0] = [13, 13]
        seq[:, 4, 1] = [1, 1]
        seq[:, 5, 1] = [2, 2]
        out = flip_normalize(seq)
        assert out[:, 4, 0].tolist() == pytest.approx([13, 13])
        assert out[:, 5, 0].tolist() == pytest.approx([3, 3])

    def test_y_coordinates_are_never_changed(self):
        seq = np.zeros((2, NUM_JOINTS, 2))
        seq[:, 4, 0] = [3, 3]
        seq[:, 5, 0] = [13, 13]
        seq[:, 4, 1] = [1, 1]
        seq[:, 5, 1] = [2, 2]
        out = flip_normalize(seq)
        assert out[:, 4, 1].tolist() == pytest.approx([1, 1])
        assert out[:, 5, 1].tolist() == pytest.approx([2, 2])

    def test_midback_already_right_of_hip_is_not_flipped(self):
        seq = np.zeros((2, NUM_JOINTS, 2))
        seq[:, 4, 0] = [13, 13]
        seq[:, 5, 0] = [3, 3]
        out = flip_normalize(seq)
        assert out[:, 4, 0].tolist() == pytest.approx([13, 13])
        assert out[:, 5, 0].tolist() == pytest.approx([3, 3])

    def test_zero_x_coordinate_is_treated_as_missing_not_flipped(self):
        """x=0 是「未偵測到」的 sentinel 值，不是真實座標——midback_x=0 時
        該幀不該被多數決採納，全部幀都是 0 時應該完全不觸發翻轉。"""
        seq = np.zeros((2, NUM_JOINTS, 2))
        seq[:, 4, 0] = [0, 0]
        seq[:, 5, 0] = [10, 10]
        out = flip_normalize(seq)
        assert out[:, 4, 0].tolist() == pytest.approx([0, 0])
        assert out[:, 5, 0].tolist() == pytest.approx([10, 10])


# ============================================================================
# orientation_normalize()
# ============================================================================


class TestOrientationNormalize:
    def test_axis_already_pointing_up_is_unchanged(self):
        """mid_back→hip 已經指向 +y，旋轉角應為 0，座標不變。"""
        seq = np.zeros((1, NUM_JOINTS, 2))
        seq[0, 4] = [2, 3]
        seq[0, 5] = [2, 8]
        out = orientation_normalize(seq)
        assert out[0, 4] == pytest.approx([2, 3])
        assert out[0, 5] == pytest.approx([2, 8])

    def test_axis_pointing_right_gets_rotated_90_degrees_to_point_up(self):
        """mid_back→hip 原本指向 +x，旋轉後應指向 +y，長度不變（5.0）。"""
        seq = np.zeros((1, NUM_JOINTS, 2))
        seq[0, 4] = [0, 0]  # midback（旋轉中心）
        seq[0, 5] = [5, 0]  # hip
        out = orientation_normalize(seq)
        assert out[0, 4] == pytest.approx([0, 0], abs=1e-9)  # 中心點不動
        assert out[0, 5] == pytest.approx([0, 5], abs=1e-9)

    def test_other_joints_rotate_around_midback_consistently(self):
        """跟 hip 用同一個旋轉矩陣：(10,0) 相對 midback 旋轉 90 度後應變成 (0,10)。"""
        seq = np.zeros((1, NUM_JOINTS, 2))
        seq[0, 4] = [0, 0]
        seq[0, 5] = [5, 0]
        seq[0, 3] = [10, 0]
        out = orientation_normalize(seq)
        assert out[0, 3] == pytest.approx([0, 10], abs=1e-9)


# ============================================================================
# normalize_skeleton_coords()
# ============================================================================


class TestNormalizeSkeletonCoords:
    def test_center_joint_becomes_origin(self):
        seq = np.zeros((2, NUM_JOINTS, 2))
        seq[:, 3] = [1, 4]
        seq[:, 4] = [1, 1]
        seq[:, 5] = [1, -2]
        out = normalize_skeleton_coords(seq)
        assert out[:, 4] == pytest.approx(np.zeros((2, 2)))

    def test_scales_by_average_chest_hip_distance(self):
        """chest/hip 中心化後距離為 6.0，除以 body_size(6.0) 後應變成單位距離 1.0。"""
        seq = np.zeros((2, NUM_JOINTS, 2))
        seq[:, 3] = [1, 4]
        seq[:, 4] = [1, 1]
        seq[:, 5] = [1, -2]
        out = normalize_skeleton_coords(seq)
        assert out[:, 3] == pytest.approx(np.array([[0, 0.5], [0, 0.5]]))
        assert out[:, 5] == pytest.approx(np.array([[0, -0.5], [0, -0.5]]))

    def test_near_zero_body_size_skips_scaling(self):
        """chest 與 hip 幾乎重合（body_size≈0）時應該只做中心化、不除以趨近 0 的值。"""
        seq = np.zeros((1, NUM_JOINTS, 2))
        seq[0, 3] = [1, 1]
        seq[0, 4] = [5, 5]
        seq[0, 5] = [1, 1]
        out = normalize_skeleton_coords(seq)
        assert np.all(np.isfinite(out)), "body_size 趨近 0 時不應該產生 inf/nan"
        assert out[0, 4] == pytest.approx([0, 0])


# ============================================================================
# add_velocity_feature()
# ============================================================================


class TestAddVelocityFeature:
    def test_output_shape_adds_two_velocity_channels(self):
        seq = np.zeros((3, 2, 2))
        out = add_velocity_feature(seq)
        assert out.shape == (3, 2, 4)

    def test_first_frame_velocity_is_always_zero(self):
        seq = np.zeros((3, 1, 2))
        seq[:, 0, 0] = [5, 10, 20]
        out = add_velocity_feature(seq)
        assert out[0, 0, 2:].tolist() == pytest.approx([0, 0])

    def test_velocity_equals_frame_to_frame_difference(self):
        seq = np.zeros((3, 1, 2))
        seq[:, 0, 0] = [0, 2, 4]
        out = add_velocity_feature(seq)
        assert out[:, 0, 2].tolist() == pytest.approx([0, 2, 2])

    def test_position_channels_are_preserved_unchanged(self):
        seq = np.zeros((3, 1, 2))
        seq[:, 0, 0] = [0, 2, 4]
        out = add_velocity_feature(seq)
        assert out[:, 0, :2] == pytest.approx(seq[:, 0, :])


# ============================================================================
# compute_bone_feature()
# ============================================================================


class TestComputeBoneFeature:
    def test_bone_vector_equals_child_minus_parent(self):
        """joint 4（mid_back）的父節點是 joint 3（chest）（見 _parents_17）。"""
        seq = np.zeros((1, 17, 2))
        seq[0, 3] = [1, 0]
        seq[0, 4] = [3, 0]
        out = compute_bone_feature(seq)
        assert out[0, 4] == pytest.approx([2, 0])

    def test_root_joint_bone_is_always_zero(self):
        """joint 0 是根節點，程式碼明確把 bone[:,0,:] 強制設為 0。"""
        seq = np.zeros((1, 17, 2))
        seq[0, 0] = [99, 99]  # 即使根節點座標非零
        out = compute_bone_feature(seq)
        assert out[0, 0] == pytest.approx([0, 0])

    def test_supports_truncated_14_joint_skeleton(self):
        """NUM_JOINTS=14（排除尾巴三點）時，父節點索引表應正確截斷，不報錯。"""
        seq = np.zeros((1, 14, 2))
        seq[0, 3] = [1, 0]
        seq[0, 4] = [3, 0]
        out = compute_bone_feature(seq)
        assert out.shape == (1, 14, 2)
        assert out[0, 4] == pytest.approx([2, 0])


# ============================================================================
# build_feature_tensor()
# ============================================================================


class TestBuildFeatureTensor:
    def _make_sequence(self):
        seq = np.zeros((3, 17, 2))
        seq[:, 3, 0] = [1, 2, 3]  # chest x 隨時間變化，讓速度非零
        seq[:, 4, 0] = [5, 5, 5]
        conf = np.random.default_rng(0).uniform(0, 1, size=(3, 17))
        return seq, conf

    def test_xy_mode_returns_2_channels_equal_to_input(self):
        seq, conf = self._make_sequence()
        out = build_feature_tensor(seq, conf, "xy")
        assert out.shape == (3, 17, 2)
        assert out == pytest.approx(seq)

    def test_xy_conf_mode_appends_confidence_channel(self):
        seq, conf = self._make_sequence()
        out = build_feature_tensor(seq, conf, "xy_conf")
        assert out.shape == (3, 17, 3)
        assert out[:, :, 2] == pytest.approx(conf)

    def test_xy_conf_v_mode_appends_velocity_channels(self):
        seq, conf = self._make_sequence()
        out = build_feature_tensor(seq, conf, "xy_conf_v")
        assert out.shape == (3, 17, 5)
        expected_velocity = add_velocity_feature(seq)[:, :, 2:]
        assert out[:, :, 3:5] == pytest.approx(expected_velocity)

    def test_xy_conf_v_bone_mode_appends_bone_channels(self):
        seq, conf = self._make_sequence()
        out = build_feature_tensor(seq, conf, "xy_conf_v_bone")
        assert out.shape == (3, 17, 7)
        expected_bone = compute_bone_feature(seq)
        assert out[:, :, 5:7] == pytest.approx(expected_bone)

    def test_xy_conf_v_bone_bmotion_mode_appends_bone_motion_channels(self):
        seq, conf = self._make_sequence()
        out = build_feature_tensor(seq, conf, "xy_conf_v_bone_bmotion")
        assert out.shape == (3, 17, 9)
        expected_bone_motion = compute_bone_motion_feature(seq)
        assert out[:, :, 7:9] == pytest.approx(expected_bone_motion)

    def test_mode_string_is_case_and_whitespace_insensitive(self):
        seq, conf = self._make_sequence()
        out_normal = build_feature_tensor(seq, conf, "xy_conf")
        out_messy = build_feature_tensor(seq, conf, "  XY_CONF  ")
        assert out_normal == pytest.approx(out_messy)

    def test_unknown_mode_raises_value_error(self):
        seq, conf = self._make_sequence()
        with pytest.raises(ValueError):
            build_feature_tensor(seq, conf, "not_a_real_mode")
