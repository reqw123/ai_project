"""
Unit Test：plugins/lick_stage/contact_regions.py 的幾何函式

第二階段（Unit Test）優先順序第 7 項（plugins 幾何函式，第 1 個檔案）。
模組自己的 docstring 明確聲明「All functions are pure (no side effects,
no I/O)」。所有測試座標都先用獨立公式（歐氏距離、橢圓方程式、線性插值）
驗證過。

需要環境裝有 `cv2`：`plugins/lick_stage/__init__.py` 會 import
`plugins.lick_stage.manager`，該檔案又 import `overlay.py`（需要 cv2 畫圖），
所以只要 import `plugins.lick_stage.*` 底下任何模組，就會觸發這條 import
鏈——即使這裡測試的函式本身完全不用 cv2。沒裝的環境會安全跳過。
"""

import numpy as np
import pytest

pytest.importorskip(
    "cv2",
    reason="plugins/lick_stage/__init__.py 透過 manager.py/overlay.py 需要 cv2，此環境未安裝",
)

from plugins.lick_stage.contact_regions import (
    _aabb_overlap,
    _compute_midback_angle_deg,
    _curvature_size_boost,
    _distance_point_to_segment,
    _point_in_oriented_ellipse,
    _point_in_polygon,
    _polygon_aabb,
    _polygon_contacts_circle,
    compute_geometry,
    trap_dir_from_perp,
)
from plugins.lick_stage.config import LickConfig as _C

NUM_JOINTS = 17  # compute_geometry() 內部經 _build_limb_joint_targets() 會索引到
# LIMB_SEGMENTS 的腳掌關鍵點（最大到 index 13），故陣列大小需比對完整 17 點骨架，
# 即使本檔案的測試案例本身不關心四肢/尾巴部位。


def _make_frame(nose=(0, -5), left_ear=(-2, -8), right_ear=(2, -8), chest=(0, 0), hip=(0, 10)):
    kpts = np.zeros((NUM_JOINTS, 2), dtype=np.float64)
    kpts[_C.KP_NOSE] = nose
    kpts[_C.KP_LEFT_EAR] = left_ear
    kpts[_C.KP_RIGHT_EAR] = right_ear
    kpts[_C.KP_CHEST] = chest
    kpts[_C.KP_HIP] = hip
    return kpts


def _full_conf():
    """所有 17 個關鍵點皆給滿信心值；未在 _make_frame() 明確設定座標的四肢/尾巴
    關鍵點座標預設為 (0,0)，但這裡的測試案例都不驗證四肢/尾巴幾何，不受影響。"""
    return np.ones(NUM_JOINTS, dtype=np.float64)


# ============================================================================
# trap_dir_from_perp()
# ============================================================================


class TestTrapDirFromPerp:
    @pytest.mark.parametrize(
        "trap_perp,expected",
        [
            ([1, 0], [0, 1]),
            ([0, 1], [-1, 0]),
            ([0, -1], [1, 0]),
            ([-1, 0], [0, 1]),
        ],
    )
    def test_always_points_toward_positive_y(self, trap_perp, expected):
        result = trap_dir_from_perp(trap_perp)
        assert result == pytest.approx(expected)
        assert result[1] >= 0.0  # 永遠指向影像座標系「下方」


# ============================================================================
# _compute_midback_angle_deg()
# ============================================================================


class TestComputeMidbackAngleDeg:
    def test_right_angle(self):
        # chest=(1,0), midback=(0,0), hip=(0,1) → v1=(1,0), v2=(0,1) → 90 度
        assert _compute_midback_angle_deg((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0)

    def test_collinear_gives_180(self):
        # chest/hip 在 midback 兩側同一直線上 → 180 度（接近共線）
        assert _compute_midback_angle_deg((-1, 0), (0, 0), (1, 0)) == pytest.approx(180.0)

    def test_same_side_gives_0(self):
        # chest/hip 在 midback 同一側同一直線上 → 0 度（夾角過尖的極端）
        assert _compute_midback_angle_deg((1, 0), (0, 0), (2, 0)) == pytest.approx(0.0)

    def test_degenerate_vector_returns_nan(self):
        # chest 與 midback 重合，v1 長度為 0，無法定義夾角
        assert np.isnan(_compute_midback_angle_deg((0, 0), (0, 0), (0, 1)))


# ============================================================================
# _curvature_size_boost()
# ============================================================================


class TestCurvatureSizeBoost:
    def test_max_angle_gives_min_boost(self):
        """夾角最大（脊椎最直）→ 縮放倍率最小。"""
        assert _curvature_size_boost(_C.CURVATURE_ANGLE_MAX_DEG) == pytest.approx(
            _C.CURVATURE_BOOST_MIN
        )

    def test_min_angle_gives_max_boost(self):
        """夾角最小（脊椎最彎/蜷曲）→ 縮放倍率最大。"""
        assert _curvature_size_boost(_C.CURVATURE_ANGLE_MIN_DEG) == pytest.approx(
            _C.CURVATURE_BOOST_MAX
        )

    def test_midpoint_angle_gives_midpoint_boost(self):
        mid_angle = (_C.CURVATURE_ANGLE_MIN_DEG + _C.CURVATURE_ANGLE_MAX_DEG) / 2
        expected = (_C.CURVATURE_BOOST_MIN + _C.CURVATURE_BOOST_MAX) / 2
        assert _curvature_size_boost(mid_angle) == pytest.approx(expected)

    def test_nan_input_returns_neutral_scale(self):
        assert _curvature_size_boost(float("nan")) == 1.0

    def test_none_input_returns_neutral_scale(self):
        assert _curvature_size_boost(None) == 1.0

    def test_above_max_angle_is_clamped(self):
        """比 CURVATURE_ANGLE_MAX_DEG 更直（角度更大）仍然只夾在
        CURVATURE_BOOST_MIN，不會更小。"""
        assert _curvature_size_boost(_C.CURVATURE_ANGLE_MAX_DEG + 20) == pytest.approx(
            _C.CURVATURE_BOOST_MIN
        )

    def test_below_min_angle_is_clamped(self):
        """比 CURVATURE_ANGLE_MIN_DEG 更彎（角度更小）仍然只夾在
        CURVATURE_BOOST_MAX，不會更大。"""
        assert _curvature_size_boost(_C.CURVATURE_ANGLE_MIN_DEG - 10) == pytest.approx(
            _C.CURVATURE_BOOST_MAX
        )


# ============================================================================
# _distance_point_to_segment()
# ============================================================================


class TestDistancePointToSegment:
    def test_point_beyond_segment_start_clamps_to_endpoint(self):
        assert _distance_point_to_segment((0, 0), (0, 1), (0, 3)) == pytest.approx(1.0)

    def test_point_on_segment_has_zero_distance(self):
        assert _distance_point_to_segment((0, 2), (0, 1), (0, 3)) == pytest.approx(0.0)

    def test_point_perpendicular_to_segment_uses_perpendicular_distance(self):
        assert _distance_point_to_segment((3, 2), (0, 1), (0, 3)) == pytest.approx(3.0)

    def test_degenerate_zero_length_segment_uses_direct_distance(self):
        assert _distance_point_to_segment((3, 4), (0, 0), (0, 0)) == pytest.approx(5.0)


# ============================================================================
# _point_in_polygon()
# ============================================================================


class TestPointInPolygon:
    _square = [(0, 0), (10, 0), (10, 10), (0, 10)]

    def test_point_inside_returns_true(self):
        assert _point_in_polygon((5, 5), self._square) is True

    def test_point_outside_returns_false(self):
        assert _point_in_polygon((15, 15), self._square) is False

    def test_point_on_boundary_returns_true(self):
        assert _point_in_polygon((5, 0), self._square) is True

    def test_degenerate_polygon_with_too_few_points_returns_false(self):
        assert _point_in_polygon((0, 0), [(0, 0), (1, 1)]) is False


# ============================================================================
# _polygon_aabb() / _aabb_overlap()
# ============================================================================


class TestPolygonAabbAndOverlap:
    def test_aabb_matches_min_max_of_points(self):
        square = np.array([(0, 0), (10, 0), (10, 10), (0, 10)])
        assert _polygon_aabb(square) == pytest.approx((0.0, 0.0, 10.0, 10.0))

    def test_overlapping_boxes_return_true(self):
        assert _aabb_overlap((0, 0, 10, 10), (5, 5, 15, 15)) is True

    def test_non_overlapping_boxes_return_false(self):
        assert _aabb_overlap((0, 0, 10, 10), (20, 20, 30, 30)) is False

    def test_touching_edges_count_as_overlap(self):
        assert _aabb_overlap((0, 0, 10, 10), (10, 0, 20, 10)) is True


# ============================================================================
# _point_in_oriented_ellipse()
# ============================================================================


class TestPointInOrientedEllipse:
    _center = np.array([0.0, 0.0])
    _axis_u = np.array([1.0, 0.0])
    _axis_v = np.array([0.0, 1.0])

    def test_point_within_major_axis_radius_is_inside(self):
        pt = np.array([3.0, 0.0])
        assert _point_in_oriented_ellipse(pt, self._center, self._axis_u, self._axis_v, 5, 3) is True

    def test_point_beyond_major_axis_radius_is_outside(self):
        pt = np.array([6.0, 0.0])
        assert _point_in_oriented_ellipse(pt, self._center, self._axis_u, self._axis_v, 5, 3) is False

    def test_point_exactly_on_minor_axis_boundary_is_inside(self):
        pt = np.array([0.0, 3.0])
        assert _point_in_oriented_ellipse(pt, self._center, self._axis_u, self._axis_v, 5, 3) is True


# ============================================================================
# _polygon_contacts_circle()
# ============================================================================


class TestPolygonContactsCircle:
    _square = [(0, 0), (10, 0), (10, 10), (0, 10)]

    def test_circle_center_inside_polygon_counts_as_contact(self):
        assert _polygon_contacts_circle(self._square, (5, 5), 1) is True

    def test_far_away_circle_has_no_contact(self):
        assert _polygon_contacts_circle(self._square, (20, 20), 1) is False

    def test_circle_touching_polygon_edge_counts_as_contact(self):
        assert _polygon_contacts_circle(self._square, (11, 5), 2) is True

    def test_circle_just_short_of_polygon_edge_has_no_contact(self):
        assert _polygon_contacts_circle(self._square, (15, 5), 2) is False

    def test_zero_radius_never_contacts(self):
        assert _polygon_contacts_circle(self._square, (5, 5), 0) is False


# ============================================================================
# compute_geometry()（主要進入點，高層次驗證）
# ============================================================================


class TestComputeGeometry:
    def test_returns_none_when_input_is_none(self):
        assert compute_geometry(None, None) is None

    def test_returns_none_when_nose_confidence_too_low(self):
        kpts = _make_frame()
        conf = _full_conf()
        conf[_C.KP_NOSE] = 0.01
        assert compute_geometry(kpts, conf) is None

    def test_returns_none_when_chest_and_hip_coincide(self):
        """body_len 趨近 0（chest==hip）時無法定義身體軸線，應回傳 None。"""
        kpts = _make_frame(chest=(0, 0), hip=(0, 0))
        assert compute_geometry(kpts, _full_conf()) is None

    def test_valid_frame_returns_dict_with_expected_keys(self):
        result = compute_geometry(_make_frame(), _full_conf())
        assert result is not None
        for key in (
            "nose",
            "body_center",
            "body_len",
            "nose_contact_trapezoid",
            "trap_perp",
            "trap_dir",
        ):
            assert key in result

    def test_body_len_matches_chest_hip_distance(self):
        result = compute_geometry(_make_frame(chest=(0, 0), hip=(0, 10)), _full_conf())
        assert result["body_len"] == pytest.approx(10.0)

    def test_nose_trapezoid_has_four_corners(self):
        result = compute_geometry(_make_frame(), _full_conf())
        assert result["nose_contact_trapezoid"].shape == (4, 2)

    def test_missing_ears_falls_back_to_body_normal_for_trap_perp(self):
        """兩耳信心都不足時，trap_perp 改用 body_normal 方向，仍應回傳有效結果
        （不因為缺耳朵就整個判定失敗——只有 nose/chest/hip 是必要關鍵點）。"""
        kpts = _make_frame()
        conf = _full_conf()
        conf[_C.KP_LEFT_EAR] = 0.01
        conf[_C.KP_RIGHT_EAR] = 0.01
        result = compute_geometry(kpts, conf)
        assert result is not None
        assert result["ear_center"] == pytest.approx(result["nose"])  # fallback 用 nose 當 ear_center
