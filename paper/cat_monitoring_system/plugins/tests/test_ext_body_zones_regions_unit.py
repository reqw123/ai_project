"""
Unit Test：plugins/lick_stage/ext_body_zones/regions.py 的 7 區身體分區幾何函式

第二階段（Unit Test）優先順序第 7 項（plugins 幾何函式，第 4 個檔案，也是最後
一個）。模組自己的 docstring 明確聲明「No side effects, no I/O」。
import 路徑仍需要 cv2（`ext_body_zones` 是 `plugins.lick_stage` 的子套件，
import 它一樣會觸發 `plugins/lick_stage/__init__.py` → manager.py → overlay.py
→ cv2 這條鏈，見 test_contact_regions_unit.py 開頭說明）。

所有期望值直接依照原始碼手動推導，關鍵設計：
- body_len = |Hip - Chest|，會被 BODY_LEN_MIN_PX(300)/BODY_LEN_MAX_PX(650) 夾鉗
  成 eff_len，所有其餘比例（半徑/半寬）都是 eff_len 的倍數
- classify_zone() 判定優先序：四肢腳掌圓 > 四肢長條 > 尾巴長條 >
  （頭/胸口判定停用）> 軀幹橢圓（腹側 vs 側背）> NO_TARGET
"""

import math

import numpy as np
import pytest

pytest.importorskip(
    "cv2",
    reason="plugins/lick_stage/__init__.py 透過 manager.py/overlay.py 需要 cv2，此環境未安裝",
)

from plugins.lick_stage.ext_body_zones.config import ExtZoneConfig as _C
from plugins.lick_stage.ext_body_zones.regions import (
    _conf_ok,
    _norm,
    _perp,
    _point_in_circle,
    _point_on_strip,
    build_zone_targets,
    classify_zone,
    targets_to_geometry_payload,
)

NUM_JOINTS = 17  # 完整 17 點 YOLO-Pose 骨架
EFF_LEN = _C.BODY_LEN_MIN_PX  # body_len=100 < 300 一律被夾鉗到下限 300


def _kpts(overrides=None):
    kpts = np.zeros((NUM_JOINTS, 2), dtype=np.float64)
    kpts[_C.KP_CHEST] = (0, 0)
    kpts[_C.KP_HIP] = (0, 100)  # body_len = 100（會被夾鉗到 EFF_LEN=300）
    for idx, pt in (overrides or {}).items():
        kpts[idx] = pt
    return kpts


def _full_conf(overrides=None):
    """關鍵點索引是 int，不能用 **{idx: val} 展開成 kwargs（keyword 必須是字串），
    故一律改用一般 dict 參數傳入覆寫值。"""
    conf = np.zeros(NUM_JOINTS, dtype=np.float64)
    conf[_C.KP_CHEST] = 1.0
    conf[_C.KP_HIP] = 1.0
    for idx, val in (overrides or {}).items():
        conf[idx] = val
    return conf


# ============================================================================
# 底層幾何工具
# ============================================================================


class TestPerpAndNorm:
    def test_perp_rotates_90_degrees_counterclockwise(self):
        assert _perp((1, 0)) == pytest.approx([0.0, 1.0])
        assert _perp((0, 1)) == pytest.approx([-1.0, 0.0])

    def test_norm_matches_euclidean_distance(self):
        assert _norm((3, 4)) == pytest.approx(5.0)


class TestConfOk:
    def test_confidence_above_threshold_is_ok(self):
        conf = [0.0, 0.51]
        assert _conf_ok(conf, 1) is True

    def test_confidence_exactly_at_threshold_is_not_ok(self):
        """比較用嚴格大於（>），恰好等於門檻視為不合格。"""
        conf = [0.0, _C.CONF_THRESHOLD]
        assert _conf_ok(conf, 1) is False

    def test_confidence_below_threshold_is_not_ok(self):
        conf = [0.0, 0.1]
        assert _conf_ok(conf, 1) is False


class TestPointInCircle:
    def test_point_within_radius_hits(self):
        hit, dist = _point_in_circle((3, 4), (0, 0), 5)
        assert hit is True
        assert dist == pytest.approx(5.0)

    def test_point_beyond_radius_misses(self):
        hit, dist = _point_in_circle((3, 4), (0, 0), 4)
        assert hit is False
        assert dist == pytest.approx(5.0)


class TestPointOnStrip:
    _p0, _p1 = (0, 0), (10, 0)

    def test_point_within_strip_hits_with_correct_perp_distance(self):
        hit, perp = _point_on_strip((5, 1), self._p0, self._p1, half_width=2)
        assert hit is True
        assert perp == pytest.approx(1.0)

    def test_point_beyond_half_width_misses(self):
        hit, perp = _point_on_strip((5, 3), self._p0, self._p1, half_width=2)
        assert hit is False
        assert perp == pytest.approx(3.0)

    def test_point_before_segment_start_misses(self):
        hit, perp = _point_on_strip((-1, 0), self._p0, self._p1, half_width=2)
        assert hit is False
        assert perp == pytest.approx(0.0)

    def test_degenerate_zero_length_segment_never_hits(self):
        hit, perp = _point_on_strip((0, 0), (5, 5), (5, 5), half_width=2)
        assert hit is False
        assert math.isinf(perp)


# ============================================================================
# build_zone_targets()
# ============================================================================


class TestBuildZoneTargets:
    def test_none_inputs_return_none(self):
        assert build_zone_targets(None, None) is None

    def test_missing_chest_confidence_returns_none(self):
        result = build_zone_targets(_kpts(), _full_conf({_C.KP_CHEST: 0.1}))
        assert result is None

    def test_missing_hip_confidence_returns_none(self):
        conf = _full_conf()
        conf[_C.KP_HIP] = 0.1
        assert build_zone_targets(_kpts(), conf) is None

    def test_coincident_chest_and_hip_returns_none(self):
        kpts = _kpts()
        kpts[_C.KP_HIP] = kpts[_C.KP_CHEST]
        assert build_zone_targets(kpts, _full_conf()) is None

    def test_body_axis_unit_points_from_chest_to_hip(self):
        result = build_zone_targets(_kpts(), _full_conf())
        assert result["body_axis_unit"] == pytest.approx([0.0, 1.0])

    def test_short_body_length_is_clamped_to_minimum(self):
        result = build_zone_targets(_kpts(), _full_conf())
        assert result["head_radius"] == pytest.approx(EFF_LEN * _C.HEAD_RADIUS_RATIO)
        assert result["neck_radius"] == pytest.approx(EFF_LEN * _C.NECK_RADIUS_RATIO)

    def test_confident_mid_back_is_used_as_torso_center(self):
        kpts = _kpts({_C.KP_MID_BACK: (5, 50)})
        conf = _full_conf({_C.KP_MID_BACK: 1.0})
        result = build_zone_targets(kpts, conf)
        assert result["torso_center"] == pytest.approx([5.0, 50.0])

    def test_missing_mid_back_falls_back_to_chest_hip_midpoint(self):
        result = build_zone_targets(_kpts(), _full_conf())
        assert result["torso_center"] == pytest.approx([0.0, 50.0])

    def test_both_ears_confident_uses_ear_midpoint_as_head_center(self):
        kpts = _kpts({_C.KP_LEFT_EAR: (-10, -10), _C.KP_RIGHT_EAR: (10, -10)})
        conf = _full_conf({_C.KP_LEFT_EAR: 1.0, _C.KP_RIGHT_EAR: 1.0})
        result = build_zone_targets(kpts, conf)
        assert result["head_center"] == pytest.approx([0.0, -10.0])

    def test_only_nose_confident_uses_nose_as_head_center(self):
        kpts = _kpts({_C.KP_NOSE: (3, -20)})
        conf = _full_conf({_C.KP_NOSE: 1.0})
        result = build_zone_targets(kpts, conf)
        assert result["head_center"] == pytest.approx([3.0, -20.0])

    def test_no_head_keypoints_falls_back_to_axis_projection(self):
        result = build_zone_targets(_kpts(), _full_conf())
        expected = np.array([0.0, 0.0]) - np.array([0.0, 1.0]) * EFF_LEN * 0.5
        assert result["head_center"] == pytest.approx(expected)

    def test_no_confident_knees_defaults_ventral_sign_positive(self):
        result = build_zone_targets(_kpts(), _full_conf())
        assert result["ventral_sign"] == pytest.approx(1.0)

    def test_knee_on_negative_normal_side_gives_positive_ventral_sign(self):
        """body_normal=[-1,0]；膝蓋 x 座標小於 torso_center.x 時，
        dot(knee-torso_center, body_normal) > 0 → ventral_sign = +1。"""
        kpts = _kpts({_C.KP_FL_KNEE: (-10, 50)})
        conf = _full_conf({_C.KP_FL_KNEE: 1.0})
        result = build_zone_targets(kpts, conf)
        assert result["ventral_sign"] == pytest.approx(1.0)

    def test_knee_on_positive_normal_side_gives_negative_ventral_sign(self):
        kpts = _kpts({_C.KP_FL_KNEE: (10, 50)})
        conf = _full_conf({_C.KP_FL_KNEE: 1.0})
        result = build_zone_targets(kpts, conf)
        assert result["ventral_sign"] == pytest.approx(-1.0)

    def test_confident_limb_pair_produces_one_segment_and_paw(self):
        kpts = _kpts({_C.KP_FL_KNEE: (100, 0), _C.KP_FL_PAW: (100, 100)})
        conf = _full_conf({_C.KP_FL_KNEE: 1.0, _C.KP_FL_PAW: 1.0})
        result = build_zone_targets(kpts, conf)
        assert len(result["limbs"]["FORELIMB"]["segments"]) == 1
        assert len(result["limbs"]["FORELIMB"]["paws"]) == 1
        assert len(result["limbs"]["HINDLIMB"]["segments"]) == 0

    def test_incomplete_limb_pair_is_excluded(self):
        """膝蓋信心足夠但腳掌不足時，該肢體不應被納入。"""
        kpts = _kpts({_C.KP_FL_KNEE: (100, 0), _C.KP_FL_PAW: (100, 100)})
        conf = _full_conf({_C.KP_FL_KNEE: 1.0, _C.KP_FL_PAW: 0.01})
        result = build_zone_targets(kpts, conf)
        assert len(result["limbs"]["FORELIMB"]["segments"]) == 0

    def test_all_tail_points_confident_produces_two_segments(self):
        kpts = _kpts(
            {
                _C.KP_TAIL_ROOT: (0, 150),
                _C.KP_TAIL_MID: (0, 200),
                _C.KP_TAIL_TIP: (0, 250),
            }
        )
        conf = _full_conf(
            {_C.KP_TAIL_ROOT: 1.0, _C.KP_TAIL_MID: 1.0, _C.KP_TAIL_TIP: 1.0}
        )
        result = build_zone_targets(kpts, conf)
        assert len(result["tail_segs"]) == 2

    def test_incomplete_tail_produces_no_segments(self):
        kpts = _kpts({_C.KP_TAIL_ROOT: (0, 150), _C.KP_TAIL_MID: (0, 200)})
        conf = _full_conf({_C.KP_TAIL_ROOT: 1.0, _C.KP_TAIL_MID: 1.0})
        result = build_zone_targets(kpts, conf)
        assert result["tail_segs"] == []


# ============================================================================
# classify_zone()
# ============================================================================


def _base_targets():
    return build_zone_targets(_kpts(), _full_conf())


class TestClassifyZone:
    def test_none_targets_returns_no_target(self):
        zone_id, name, conf = classify_zone((0, 0), None)
        assert (zone_id, name, conf) == (_C.ZONE_NO_TARGET, "NO_TARGET", 0.0)

    def test_none_nose_point_returns_no_target(self):
        zone_id, name, conf = classify_zone(None, _base_targets())
        assert (zone_id, name, conf) == (_C.ZONE_NO_TARGET, "NO_TARGET", 0.0)

    def test_nose_exactly_on_paw_gives_forelimb_full_confidence(self):
        kpts = _kpts({_C.KP_FL_KNEE: (100, 0), _C.KP_FL_PAW: (100, 100)})
        conf = _full_conf({_C.KP_FL_KNEE: 1.0, _C.KP_FL_PAW: 1.0})
        targets = build_zone_targets(kpts, conf)
        zone_id, name, zconf = classify_zone((100, 100), targets)
        assert (zone_id, name) == (_C.ZONE_FORELIMB, "FORELIMB")
        assert zconf == pytest.approx(1.0)

    def test_nose_near_limb_strip_but_outside_paw_circle_gives_forelimb(self):
        kpts = _kpts({_C.KP_FL_KNEE: (100, 0), _C.KP_FL_PAW: (100, 100)})
        conf = _full_conf({_C.KP_FL_KNEE: 1.0, _C.KP_FL_PAW: 1.0})
        targets = build_zone_targets(kpts, conf)
        paw_radius = EFF_LEN * _C.LIMB_PAW_RADIUS_RATIO
        assert paw_radius == pytest.approx(15.0)
        zone_id, name, zconf = classify_zone((105, 50), targets)
        assert (zone_id, name) == (_C.ZONE_FORELIMB, "FORELIMB")
        limb_strip_hw = EFF_LEN * _C.LIMB_STRIP_HW_RATIO
        assert limb_strip_hw == pytest.approx(18.0)
        assert zconf == pytest.approx(1.0 - 5.0 / 18.0)

    def test_nose_on_tail_strip_gives_tail_when_no_limb_hit(self):
        kpts = _kpts(
            {
                _C.KP_TAIL_ROOT: (0, 150),
                _C.KP_TAIL_MID: (0, 200),
                _C.KP_TAIL_TIP: (0, 250),
            }
        )
        conf = _full_conf(
            {_C.KP_TAIL_ROOT: 1.0, _C.KP_TAIL_MID: 1.0, _C.KP_TAIL_TIP: 1.0}
        )
        targets = build_zone_targets(kpts, conf)
        zone_id, name, zconf = classify_zone((5, 175), targets)
        assert (zone_id, name) == (_C.ZONE_TAIL, "TAIL")
        tail_hw = EFF_LEN * _C.TAIL_STRIP_HW_RATIO
        assert tail_hw == pytest.approx(13.5)
        assert zconf == pytest.approx(1.0 - 5.0 / 13.5)

    def test_nose_on_ventral_side_of_torso_gives_abdomen(self):
        """torso_center=(0,50)（無 mid_back fallback），body_normal=[-1,0]，
        ventral_sign 預設 +1：鼻子 x<=0 側（v=-rel.x>=0）判定為腹側。"""
        targets = _base_targets()
        zone_id, name, zconf = classify_zone((-10, 50), targets)
        assert (zone_id, name) == (_C.ZONE_ABDOMEN, "ABDOMEN")
        assert zconf == pytest.approx(1.0 - math.sqrt((10.0 / 90.0) ** 2))

    def test_nose_on_dorsal_side_of_torso_gives_side_back(self):
        targets = _base_targets()
        zone_id, name, zconf = classify_zone((10, 50), targets)
        assert (zone_id, name) == (_C.ZONE_SIDE_BACK, "SIDE_BACK")
        assert zconf == pytest.approx(1.0 - math.sqrt((10.0 / 90.0) ** 2))

    def test_nose_far_from_everything_gives_no_target(self):
        targets = _base_targets()
        zone_id, name, zconf = classify_zone((5000, 5000), targets)
        assert (zone_id, name, zconf) == (_C.ZONE_NO_TARGET, "NO_TARGET", 0.0)


# ============================================================================
# targets_to_geometry_payload()
# ============================================================================


class TestTargetsToGeometryPayload:
    def test_none_targets_returns_empty_dict(self):
        assert targets_to_geometry_payload(None) == {}

    def test_valid_targets_produce_expected_top_level_keys(self):
        payload = targets_to_geometry_payload(_base_targets())
        for key in (
            "head",
            "neck",
            "torso",
            "forelimb_segs",
            "forelimb_paws",
            "hindlimb_segs",
            "hindlimb_paws",
            "tail_segs",
            "limb_hw",
            "tail_hw",
        ):
            assert key in payload

    def test_torso_payload_reflects_axis_and_normal(self):
        payload = targets_to_geometry_payload(_base_targets())
        torso = payload["torso"]
        assert torso["ux"] == pytest.approx(0.0)
        assert torso["uy"] == pytest.approx(1.0)
        assert torso["vx"] == pytest.approx(-1.0)
        assert torso["vy"] == pytest.approx(0.0)
        assert torso["ventral_sign"] == pytest.approx(1.0)

    def test_empty_limbs_and_tail_produce_empty_lists(self):
        payload = targets_to_geometry_payload(_base_targets())
        assert payload["forelimb_segs"] == []
        assert payload["forelimb_paws"] == []
        assert payload["tail_segs"] == []

    def test_populated_forelimb_produces_one_segment_and_paw_entry(self):
        kpts = _kpts({_C.KP_FL_KNEE: (100, 0), _C.KP_FL_PAW: (100, 100)})
        conf = _full_conf({_C.KP_FL_KNEE: 1.0, _C.KP_FL_PAW: 1.0})
        targets = build_zone_targets(kpts, conf)
        payload = targets_to_geometry_payload(targets)
        assert payload["forelimb_segs"] == [{"p0": [100.0, 0.0], "p1": [100.0, 100.0]}]
        assert len(payload["forelimb_paws"]) == 1
        assert payload["forelimb_paws"][0]["cx"] == pytest.approx(100.0)
        assert payload["forelimb_paws"][0]["cy"] == pytest.approx(100.0)
