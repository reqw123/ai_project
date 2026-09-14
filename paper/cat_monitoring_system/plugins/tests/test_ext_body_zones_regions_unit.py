"""
Unit Test：plugins/lick_stage/ext_body_zones/regions.py 的 7 區身體分區幾何函式

第二階段（Unit Test）優先順序第 7 項（plugins 幾何函式，第 4 個檔案，也是最後
一個）。模組自己的 docstring 明確聲明「No side effects, no I/O」。
import 路徑仍需要 cv2（`ext_body_zones` 是 `plugins.lick_stage` 的子套件，
import 它一樣會觸發 `plugins/lick_stage/__init__.py` → manager.py → overlay.py
→ cv2 這條鏈，見 test_contact_regions_unit.py 開頭說明）。

所有期望值直接依照原始碼手動推導，關鍵設計：
- body_len = |Hip - Chest|；M5 移除絕對像素夾鉗，改用 _compute_body_scale() 混合尺度
  （mid_back 信心足夠時用 chest-midback-hip 折線長，否則退回 chest-hip 直線距離，
  再退回 bbox fallback）算出 eff_len，所有其餘比例（半徑/半寬）都是 eff_len 的倍數。
  預設測試 fixture 不設 mid_back，故落在「chest_hip」分支，eff_len = body_len = 100
  （不再夾鉗到 300）。
- classify_zone()：M5 起改成候選評分制，不再是固定優先序——四肢（腳掌圓/長條
  取距離較短者）、尾巴長條、軀幹橢圓（腹側 vs 側背，或 ventral_sign_known=False
  時的 TORSO_UNSPECIFIED——見 build_zone_targets() 的 ventral_sign 說明）各自算
  正規化分數，取最高分；最高分與次高分差距小於 AMBIGUITY_MARGIN 時回傳
  AMBIGUOUS。都沒有候選命中才是 NO_TARGET。頭/胸口判定仍然停用（理由見
  classify_zone() docstring）。
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
    _compute_body_scale,
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
EFF_LEN = 100.0  # 預設 fixture 沒有 mid_back，混合尺度退回 chest_hip 直線距離（不夾鉗）


def _kpts(overrides=None):
    kpts = np.zeros((NUM_JOINTS, 2), dtype=np.float64)
    kpts[_C.KP_CHEST] = (0, 0)
    kpts[_C.KP_HIP] = (0, 100)  # body_len = 100 = EFF_LEN（無 mid_back 時不再夾鉗）
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
# _compute_body_scale()（M5：混合尺度，取代舊的絕對像素夾鉗）
# ============================================================================


class TestComputeBodyScale:
    def test_prefers_spine_path_when_mid_back_confident(self):
        kpts = _kpts({_C.KP_MID_BACK: (0.0, 50.0)})
        eff_len, source = _compute_body_scale(
            kpts, _full_conf(), np.array([0.0, 0.0]), np.array([0.0, 100.0]), True, 100.0
        )
        assert source == "spine_path"
        assert eff_len == pytest.approx(100.0)

    def test_spine_path_resists_curl_compression(self):
        kpts = _kpts({_C.KP_MID_BACK: (40.0, 30.0)})
        eff_len, source = _compute_body_scale(
            kpts, _full_conf(), np.array([0.0, 0.0]), np.array([0.0, 60.0]), True, 60.0
        )
        assert source == "spine_path"
        assert eff_len > 60.0

    def test_falls_back_to_chest_hip_when_mid_back_not_confident(self):
        eff_len, source = _compute_body_scale(
            _kpts(), _full_conf(), np.array([0.0, 0.0]), np.array([0.0, 50.0]), False, 50.0
        )
        assert source == "chest_hip"
        assert eff_len == pytest.approx(50.0)  # 不再夾鉗到絕對像素下限

    def test_falls_back_to_bbox_when_spine_and_chest_hip_both_degenerate(self):
        kpts = _kpts(
            {_C.KP_MID_BACK: (0.0, 0.0), _C.KP_NOSE: (0.0, -30.0), _C.KP_HIP: (0.0, 5.0)}
        )
        conf = np.zeros(NUM_JOINTS, dtype=np.float64)
        conf[_C.KP_NOSE] = 1.0
        conf[_C.KP_HIP] = 1.0
        eff_len, source = _compute_body_scale(
            kpts, conf, np.array([0.0, 0.0]), np.array([0.0, 5.0]), True, 5.0
        )
        assert source == "bbox_fallback"
        assert eff_len == pytest.approx(35.0 * _C.BBOX_TO_BODY_LEN_RATIO)

    def test_degenerate_when_bbox_also_unavailable(self):
        conf = np.zeros(NUM_JOINTS, dtype=np.float64)
        eff_len, source = _compute_body_scale(
            _kpts(), conf, np.array([0.0, 0.0]), np.array([0.0, 0.0]), False, 0.0
        )
        assert source == "degenerate"
        assert eff_len == pytest.approx(1e-6)


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

    def test_short_body_length_uses_chest_hip_distance_unclamped(self):
        """M5：mid_back 不可信時退回 chest-hip 直線距離，不再夾鉗到絕對像素下限。"""
        result = build_zone_targets(_kpts(), _full_conf())
        assert result["eff_len"] == pytest.approx(EFF_LEN)
        assert result["scale_source"] == "chest_hip"
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
        """+1.0 只是沒有證據時的任意預設值——M5 起還要確認
        `ventral_sign_known=False`，classify_zone() 才知道不能拿這個值去猜
        ABDOMEN/SIDE_BACK（見 TestClassifyZone 的 TORSO_UNSPECIFIED 測試）。"""
        result = build_zone_targets(_kpts(), _full_conf())
        assert result["ventral_sign"] == pytest.approx(1.0)
        assert result["ventral_sign_known"] is False

    def test_knee_on_negative_normal_side_gives_positive_ventral_sign(self):
        """body_normal=[-1,0]；膝蓋 x 座標小於 torso_center.x 時，
        dot(knee-torso_center, body_normal) > 0 → ventral_sign = +1。"""
        kpts = _kpts({_C.KP_FL_KNEE: (-10, 50)})
        conf = _full_conf({_C.KP_FL_KNEE: 1.0})
        result = build_zone_targets(kpts, conf)
        assert result["ventral_sign"] == pytest.approx(1.0)
        assert result["ventral_sign_known"] is True

    def test_knee_on_positive_normal_side_gives_negative_ventral_sign(self):
        kpts = _kpts({_C.KP_FL_KNEE: (10, 50)})
        conf = _full_conf({_C.KP_FL_KNEE: 1.0})
        result = build_zone_targets(kpts, conf)
        assert result["ventral_sign"] == pytest.approx(-1.0)
        assert result["ventral_sign_known"] is True

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
        """M5 起，同一個 zone（這裡是 FORELIMB）不管命中的是 paw 圓還是
        strip 長條，一律用同一個特徵尺度（paw_radius）正規化分數——跟
        find_nearest_zone() 的 limb_scale 同一套邏輯，取代舊版「paw 用
        paw_radius、strip 用 limb_strip_hw」各自分母的算法。這裡命中 strip
        （perp=5.0），分數 = 1 - 5.0/paw_radius(5.0) = 0.0，不是舊版的
        1 - 5.0/limb_strip_hw(6.0)。"""
        kpts = _kpts({_C.KP_FL_KNEE: (100, 0), _C.KP_FL_PAW: (100, 100)})
        conf = _full_conf({_C.KP_FL_KNEE: 1.0, _C.KP_FL_PAW: 1.0})
        targets = build_zone_targets(kpts, conf)
        paw_radius = EFF_LEN * _C.LIMB_PAW_RADIUS_RATIO
        assert paw_radius == pytest.approx(5.0)
        zone_id, name, zconf = classify_zone((105, 50), targets)
        assert (zone_id, name) == (_C.ZONE_FORELIMB, "FORELIMB")
        limb_strip_hw = EFF_LEN * _C.LIMB_STRIP_HW_RATIO
        assert limb_strip_hw == pytest.approx(6.0)
        assert zconf == pytest.approx(1.0 - 5.0 / 5.0)

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
        zone_id, name, zconf = classify_zone((2, 175), targets)
        assert (zone_id, name) == (_C.ZONE_TAIL, "TAIL")
        tail_hw = EFF_LEN * _C.TAIL_STRIP_HW_RATIO
        assert tail_hw == pytest.approx(4.5)
        assert zconf == pytest.approx(1.0 - 2.0 / 4.5)

    def _targets_with_ventral_evidence(self):
        """torso 判定要吃到 ABDOMEN/SIDE_BACK（而不是 M5 新增的
        TORSO_UNSPECIFIED），前提是 ventral_sign_known=True——用跟
        test_knee_on_negative_normal_side_gives_positive_ventral_sign 同一組
        膝蓋覆寫（ventral_sign=+1.0，維持跟舊版預設值相同的判定結果）。"""
        kpts = _kpts({_C.KP_FL_KNEE: (-10, 50)})
        conf = _full_conf({_C.KP_FL_KNEE: 1.0})
        return build_zone_targets(kpts, conf)

    def test_nose_on_ventral_side_of_torso_gives_abdomen(self):
        """torso_center=(0,50)（無 mid_back fallback），body_normal=[-1,0]，
        ventral_sign=+1（有膝蓋證據）：鼻子 x<=0 側（v=-rel.x>=0）判定為腹側。"""
        targets = self._targets_with_ventral_evidence()
        zone_id, name, zconf = classify_zone((-10, 50), targets)
        assert (zone_id, name) == (_C.ZONE_ABDOMEN, "ABDOMEN")
        assert zconf == pytest.approx(1.0 - math.sqrt((10.0 / 30.0) ** 2))

    def test_nose_on_dorsal_side_of_torso_gives_side_back(self):
        targets = self._targets_with_ventral_evidence()
        zone_id, name, zconf = classify_zone((10, 50), targets)
        assert (zone_id, name) == (_C.ZONE_SIDE_BACK, "SIDE_BACK")
        assert zconf == pytest.approx(1.0 - math.sqrt((10.0 / 30.0) ** 2))

    def test_torso_hit_without_ventral_evidence_gives_torso_unspecified(self):
        """M5：鼻子確實命中軀幹橢圓，但 _base_targets()（無膝蓋關鍵點）沒有
        ventral_sign 的真實證據——不該硬猜 ABDOMEN/SIDE_BACK，改回傳
        TORSO_UNSPECIFIED。同一個鼻子座標，_targets_with_ventral_evidence()
        版本會判成 ABDOMEN（見上一個測試），差別只在有沒有膝蓋證據。"""
        targets = _base_targets()
        assert targets["ventral_sign_known"] is False
        zone_id, name, zconf = classify_zone((-10, 50), targets)
        assert (zone_id, name) == (_C.ZONE_TORSO_UNSPECIFIED, "TORSO_UNSPECIFIED")
        assert zconf == pytest.approx(1.0 - math.sqrt((10.0 / 30.0) ** 2))

    def test_tied_forelimb_and_hindlimb_scores_give_ambiguous(self):
        """M5：鼻子跟前肢/後肢的 paw 距離完全相等（都是 3px，paw_radius=5px，
        兩者分數都是 1-3/5=0.4，差距 0 < AMBIGUITY_MARGIN）——候選評分機制
        不該武斷選其中一個，改回傳 AMBIGUOUS。舊版優先序判定（四肢腳掌圓
        > 四肢長條）會直接選 FORELIMB（迴圈先跑到），不會偵測到這種對稱情況
        下的真正不確定性。"""
        kpts = _kpts(
            {
                _C.KP_FL_KNEE: (3, 100),
                _C.KP_FL_PAW: (3, 0),
                _C.KP_HL_KNEE: (-3, 100),
                _C.KP_HL_PAW: (-3, 0),
            }
        )
        conf = _full_conf(
            {
                _C.KP_FL_KNEE: 1.0,
                _C.KP_FL_PAW: 1.0,
                _C.KP_HL_KNEE: 1.0,
                _C.KP_HL_PAW: 1.0,
            }
        )
        targets = build_zone_targets(kpts, conf)
        zone_id, name, zconf = classify_zone((0, 0), targets)
        assert (zone_id, name) == (_C.ZONE_AMBIGUOUS, "AMBIGUOUS")
        assert zconf == pytest.approx(0.4)

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
