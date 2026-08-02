"""
Unit Test：plugins/lick_stage/head_direction.py 的頭部朝向推論與平滑函式

第二階段（Unit Test）優先順序第 7 項（plugins 幾何函式，第 2 個檔案）。
所有函式都是純函式（不觸碰 cv2/GPU/IO），但 import 路徑仍需要 cv2
（見 test_contact_regions_unit.py 開頭說明），故同樣加上 importorskip 保護。

所有期望值都直接依照 head_direction.py 目前的原始碼手動推導：
- compute_head_ear_angle()：向量夾角公式（acos(dot/normA*normB)）
- infer_face_state_cat_centric()：forward_norm/lateral_norm 與
  CAT_FRONT_FORWARD_MIN/CAT_BACK_FORWARD_MIN/CAT_LR_MARGIN 門檻比較
- infer_face_state_user_rules()：三個依序 if（非 elif）的規則區塊
- stabilize_direction_vector()：翻轉感知 EMA
- smooth_state()：Counter 多數決
- check_front_view_guard()：body_ear_ratio 門檻比較
"""

import math
from collections import deque

import numpy as np
import pytest

pytest.importorskip(
    "cv2",
    reason="plugins/lick_stage/__init__.py 透過 manager.py/overlay.py 需要 cv2，此環境未安裝",
)

from plugins.lick_stage.config import LickConfig as _C
from plugins.lick_stage.head_direction import (
    check_front_view_guard,
    compute_head_ear_angle,
    infer_face_state_cat_centric,
    infer_face_state_user_rules,
    smooth_state,
    stabilize_direction_vector,
)

NUM_JOINTS = 3  # nose(0)/left_ear(1)/right_ear(2) 已足夠


def _kpts(nose=(0, 0), left_ear=(-1, 1), right_ear=(1, 1)):
    kpts = np.zeros((NUM_JOINTS, 2), dtype=np.float64)
    kpts[_C.KP_NOSE] = nose
    kpts[_C.KP_LEFT_EAR] = left_ear
    kpts[_C.KP_RIGHT_EAR] = right_ear
    return kpts


def _conf(nose=1.0, left_ear=1.0, right_ear=1.0):
    conf = np.zeros(NUM_JOINTS, dtype=np.float64)
    conf[_C.KP_NOSE] = nose
    conf[_C.KP_LEFT_EAR] = left_ear
    conf[_C.KP_RIGHT_EAR] = right_ear
    return conf


# ============================================================================
# compute_head_ear_angle()
# ============================================================================


class TestComputeHeadEarAngle:
    def test_symmetric_ears_give_90_degrees(self):
        """nose=(0,0), left=(-1,1), right=(1,1) → 兩向量互為鏡像，夾角剛好 90 度。"""
        result = compute_head_ear_angle(_kpts(), _conf())
        assert result == pytest.approx(90.0)

    def test_ears_aligned_with_nose_give_0_degrees(self):
        left_ear = (1, 1)
        right_ear = (2, 2)  # 與 left_ear 同方向（同一射線上）
        result = compute_head_ear_angle(_kpts(left_ear=left_ear, right_ear=right_ear), _conf())
        # acos() 對接近 1.0 的輸入極度敏感於浮點誤差，允許極小的絕對誤差容忍值。
        assert result == pytest.approx(0.0, abs=1e-3)

    def test_low_nose_confidence_returns_nan(self):
        result = compute_head_ear_angle(_kpts(), _conf(nose=0.1))
        assert math.isnan(result)

    def test_low_ear_confidence_returns_nan(self):
        result = compute_head_ear_angle(_kpts(), _conf(left_ear=0.1))
        assert math.isnan(result)

    def test_ear_coincident_with_nose_returns_nan(self):
        """向量長度趨近 0（耳朵與鼻子重合）時分母趨近 0，回傳 NaN 而非除以零錯誤。"""
        result = compute_head_ear_angle(_kpts(left_ear=(0, 0)), _conf())
        assert math.isnan(result)


# ============================================================================
# infer_face_state_cat_centric()
# ============================================================================


def _geom(nose, body_center=(0, 0), body_axis_unit=(0, 1), body_normal=(1, 0), body_len=10.0):
    return {
        "nose": np.array(nose, dtype=np.float64),
        "body_center": np.array(body_center, dtype=np.float64),
        "body_axis_unit": np.array(body_axis_unit, dtype=np.float64),
        "body_normal": np.array(body_normal, dtype=np.float64),
        "body_len": body_len,
    }


class TestInferFaceStateCatCentric:
    def test_none_geom_returns_unknown_with_nans(self):
        state, fwd, lat, gaze = infer_face_state_cat_centric(None, nose_ok=True)
        assert state == _C.STATE_UNKNOWN
        assert math.isnan(fwd) and math.isnan(lat) and math.isnan(gaze)

    def test_nose_not_ok_returns_unknown_with_nans(self):
        state, fwd, lat, gaze = infer_face_state_cat_centric(_geom((0, -5)), nose_ok=False)
        assert state == _C.STATE_UNKNOWN
        assert math.isnan(fwd) and math.isnan(lat) and math.isnan(gaze)

    def test_degenerate_body_len_returns_unknown_with_nans(self):
        state, fwd, lat, gaze = infer_face_state_cat_centric(
            _geom((0, -5), body_len=1e-9), nose_ok=True
        )
        assert state == _C.STATE_UNKNOWN
        assert math.isnan(fwd) and math.isnan(lat) and math.isnan(gaze)

    def test_nose_toward_chest_and_centered_is_facing_camera(self):
        state, fwd, lat, gaze = infer_face_state_cat_centric(_geom((0, -5)), nose_ok=True)
        assert state == _C.STATE_FRONT
        assert fwd == pytest.approx(0.5)
        assert lat == pytest.approx(0.0)
        assert gaze == pytest.approx(0.0)

    def test_nose_toward_hip_and_centered_is_back_view(self):
        state, fwd, lat, gaze = infer_face_state_cat_centric(_geom((0, 5)), nose_ok=True)
        assert state == _C.STATE_BACK
        assert fwd == pytest.approx(-0.5)
        assert gaze == pytest.approx(180.0)

    def test_front_facing_and_offset_left_is_front_left(self):
        state, fwd, lat, _ = infer_face_state_cat_centric(_geom((-2, -5)), nose_ok=True)
        assert state == _C.STATE_FRONT_LEFT
        assert lat == pytest.approx(-0.2)

    def test_front_facing_and_offset_right_is_front_right(self):
        state, fwd, lat, _ = infer_face_state_cat_centric(_geom((2, -5)), nose_ok=True)
        assert state == _C.STATE_FRONT_RIGHT
        assert lat == pytest.approx(0.2)

    def test_weak_forward_component_is_unknown(self):
        """forward_norm 落在 CAT_FRONT_FORWARD_MIN 與 -CAT_BACK_FORWARD_MIN 之間的模糊地帶。"""
        state, fwd, _, _ = infer_face_state_cat_centric(_geom((0, -0.5)), nose_ok=True)
        assert state == _C.STATE_UNKNOWN
        assert fwd == pytest.approx(0.05)


# ============================================================================
# infer_face_state_user_rules()
# ============================================================================


class TestInferFaceStateUserRules:
    def test_low_nose_conf_with_large_ear_distance_is_back_view(self):
        state, applied = infer_face_state_user_rules(
            head_ear_angle_deg=float("nan"), dist_norm=float("nan"), dist_px=10.0, nose_conf=0.3
        )
        assert (state, applied) == (_C.STATE_BACK, True)

    def test_low_nose_conf_with_non_finite_dist_px_is_back_view(self):
        state, applied = infer_face_state_user_rules(
            head_ear_angle_deg=float("nan"), dist_norm=float("nan"), dist_px=float("nan"),
            nose_conf=0.3,
        )
        assert (state, applied) == (_C.STATE_BACK, True)

    def test_low_nose_conf_but_small_ear_distance_falls_through_to_angle_rule(self):
        state, applied = infer_face_state_user_rules(
            head_ear_angle_deg=50.0, dist_norm=0.5, dist_px=1.0, nose_conf=0.3
        )
        assert (state, applied) == (_C.STATE_FRONT, True)

    def test_high_nose_conf_with_wide_angle_and_distance_is_facing_camera(self):
        state, applied = infer_face_state_user_rules(
            head_ear_angle_deg=50.0, dist_norm=0.5, dist_px=100.0, nose_conf=0.9
        )
        assert (state, applied) == (_C.STATE_FRONT, True)

    def test_narrow_angle_does_not_trigger_front_rule(self):
        state, applied = infer_face_state_user_rules(
            head_ear_angle_deg=10.0, dist_norm=0.5, dist_px=100.0, nose_conf=0.9
        )
        assert (state, applied) == (_C.STATE_UNKNOWN, False)

    def test_all_signals_absent_returns_unknown(self):
        state, applied = infer_face_state_user_rules(
            head_ear_angle_deg=float("nan"), dist_norm=float("nan"), dist_px=float("nan"),
            nose_conf=0.9,
        )
        assert (state, applied) == (_C.STATE_UNKNOWN, False)


# ============================================================================
# stabilize_direction_vector()
# ============================================================================


class TestStabilizeDirectionVector:
    def test_no_previous_vector_normalizes_and_returns_new(self):
        result = stabilize_direction_vector((3, 4), None, alpha=0.5, flip_margin=0.15)
        assert result == pytest.approx([0.6, 0.8])

    def test_zero_new_vector_with_no_previous_returns_raw_zero(self):
        result = stabilize_direction_vector((0, 0), None, alpha=0.5, flip_margin=0.15)
        assert result == pytest.approx([0.0, 0.0])

    def test_zero_new_vector_with_previous_returns_previous_unchanged(self):
        result = stabilize_direction_vector((0, 0), (1, 0), alpha=0.5, flip_margin=0.15)
        assert result == pytest.approx([1.0, 0.0])

    def test_same_direction_stays_stable_regardless_of_alpha(self):
        result = stabilize_direction_vector((2, 0), (1, 0), alpha=0.3, flip_margin=0.15)
        assert result == pytest.approx([1.0, 0.0])

    def test_opposite_direction_flips_then_blends_back_to_same_side(self):
        result = stabilize_direction_vector((-1, 0), (1, 0), alpha=0.5, flip_margin=0.15)
        assert result == pytest.approx([1.0, 0.0])

    def test_perpendicular_direction_blends_without_flipping(self):
        result = stabilize_direction_vector((0, 1), (1, 0), alpha=0.5, flip_margin=0.15)
        expected = 1.0 / math.sqrt(2)
        assert result == pytest.approx([expected, expected])


# ============================================================================
# smooth_state()
# ============================================================================


class TestSmoothState:
    def test_empty_history_returns_unknown_with_zero_stability(self):
        state, stability = smooth_state(deque())
        assert (state, stability) == (_C.STATE_UNKNOWN, 0.0)

    def test_majority_state_wins_with_correct_stability_fraction(self):
        state, stability = smooth_state(deque(["A", "A", "B"]))
        assert state == "A"
        assert stability == pytest.approx(2.0 / 3.0)

    def test_unanimous_history_gives_full_stability(self):
        state, stability = smooth_state(deque(["A", "A", "A"]))
        assert (state, stability) == ("A", 1.0)

    def test_tie_breaks_toward_first_encountered_value(self):
        state, stability = smooth_state(deque(["A", "B"]))
        assert state == "A"
        assert stability == pytest.approx(0.5)


# ============================================================================
# check_front_view_guard()
# ============================================================================


class TestCheckFrontViewGuard:
    def test_low_ratio_triggers_guard(self):
        conf = _conf()
        assert check_front_view_guard(conf, dist_px=100.0, body_ear_ratio=0.5) is True

    def test_high_ratio_does_not_trigger_guard(self):
        conf = _conf()
        assert check_front_view_guard(conf, dist_px=100.0, body_ear_ratio=0.9) is False

    def test_low_ear_confidence_disables_guard(self):
        conf = _conf(left_ear=0.1)
        assert check_front_view_guard(conf, dist_px=100.0, body_ear_ratio=0.5) is False

    def test_non_finite_ratio_disables_guard(self):
        conf = _conf()
        assert check_front_view_guard(conf, dist_px=100.0, body_ear_ratio=float("nan")) is False

    def test_guard_disabled_by_config_always_returns_false(self, monkeypatch):
        monkeypatch.setattr(_C, "FRONT_VIEW_GUARD_ENABLED", False)
        conf = _conf()
        assert check_front_view_guard(conf, dist_px=100.0, body_ear_ratio=0.1) is False
