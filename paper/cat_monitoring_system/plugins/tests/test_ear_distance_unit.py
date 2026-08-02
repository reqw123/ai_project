"""
Unit Test：plugins/lick_stage/ear_distance.py 的 compute_ear_distance()

第二階段（Unit Test）優先順序第 7 項（plugins 幾何函式，第 3 個檔案）。
純函式，但 import 路徑仍需要 cv2（見 test_contact_regions_unit.py 開頭說明）。

所有期望值直接依照原始碼手動推導：
dist_px = |left_ear - right_ear|
body_scale = |chest - hip|
dist_norm = dist_px / body_scale（僅在兩者皆為有限值且 dist_px > 1e-6 時計算）
body_ear_ratio = body_scale / dist_px（同上條件）
valid = 兩耳信心值皆達門檻
"""

import math

import numpy as np
import pytest

pytest.importorskip(
    "cv2",
    reason="plugins/lick_stage/__init__.py 透過 manager.py/overlay.py 需要 cv2，此環境未安裝",
)

from plugins.lick_stage.config import LickConfig as _C
from plugins.lick_stage.ear_distance import compute_ear_distance

NUM_JOINTS = 6


def _kpts(left_ear=(0, 0), right_ear=(10, 0), chest=(0, 0), hip=(0, 10)):
    kpts = np.zeros((NUM_JOINTS, 2), dtype=np.float64)
    kpts[_C.KP_LEFT_EAR] = left_ear
    kpts[_C.KP_RIGHT_EAR] = right_ear
    kpts[_C.KP_CHEST] = chest
    kpts[_C.KP_HIP] = hip
    return kpts


def _full_conf():
    return np.ones(NUM_JOINTS, dtype=np.float64)


class TestComputeEarDistance:
    def test_all_keypoints_confident_returns_correct_values(self):
        dist_px, dist_norm, valid, body_scale, body_ear_ratio = compute_ear_distance(
            _kpts(), _full_conf()
        )
        assert dist_px == pytest.approx(10.0)
        assert dist_norm == pytest.approx(1.0)
        assert valid is True
        assert body_scale == pytest.approx(10.0)
        assert body_ear_ratio == pytest.approx(1.0)

    def test_low_ear_confidence_makes_dist_px_nan_and_invalid(self):
        conf = _full_conf()
        conf[_C.KP_LEFT_EAR] = 0.1
        dist_px, dist_norm, valid, body_scale, body_ear_ratio = compute_ear_distance(
            _kpts(), conf
        )
        assert math.isnan(dist_px)
        assert valid is False
        assert body_scale == pytest.approx(10.0)  # chest/hip 仍有效，不受耳朵影響
        assert math.isnan(dist_norm)
        assert math.isnan(body_ear_ratio)

    def test_low_chest_confidence_makes_body_scale_nan(self):
        conf = _full_conf()
        conf[_C.KP_CHEST] = 0.1
        dist_px, dist_norm, valid, body_scale, body_ear_ratio = compute_ear_distance(
            _kpts(), conf
        )
        assert dist_px == pytest.approx(10.0)
        assert valid is True
        assert math.isnan(body_scale)
        assert math.isnan(dist_norm)
        assert math.isnan(body_ear_ratio)

    def test_low_hip_confidence_makes_body_scale_nan(self):
        conf = _full_conf()
        conf[_C.KP_HIP] = 0.1
        _, dist_norm, _, body_scale, body_ear_ratio = compute_ear_distance(_kpts(), conf)
        assert math.isnan(body_scale)
        assert math.isnan(dist_norm)
        assert math.isnan(body_ear_ratio)

    def test_coincident_ears_give_zero_dist_px_and_nan_ratios(self):
        """dist_px 恰好為 0（未達 >1e-6 門檻）時，dist_norm/body_ear_ratio 應為 NaN
        （避免除以 0），但 valid 仍為 True（信心值本身沒問題）。"""
        dist_px, dist_norm, valid, body_scale, body_ear_ratio = compute_ear_distance(
            _kpts(left_ear=(5, 5), right_ear=(5, 5)), _full_conf()
        )
        assert dist_px == pytest.approx(0.0)
        assert valid is True
        assert math.isnan(dist_norm)
        assert math.isnan(body_ear_ratio)

    def test_ear_confidence_exactly_at_threshold_is_invalid(self):
        """門檻比較用 >=（EAR_CONF_THRESHOLD=0.3），恰好等於門檻應視為有效。"""
        conf = _full_conf()
        conf[_C.KP_LEFT_EAR] = _C.EAR_CONF_THRESHOLD
        conf[_C.KP_RIGHT_EAR] = _C.EAR_CONF_THRESHOLD
        _, _, valid, _, _ = compute_ear_distance(_kpts(), conf)
        assert valid is True

    def test_body_ear_ratio_is_reciprocal_of_dist_norm(self):
        dist_px, dist_norm, _, body_scale, body_ear_ratio = compute_ear_distance(
            _kpts(left_ear=(0, 0), right_ear=(20, 0), chest=(0, 0), hip=(0, 5)), _full_conf()
        )
        assert dist_px == pytest.approx(20.0)
        assert body_scale == pytest.approx(5.0)
        assert dist_norm == pytest.approx(4.0)
        assert body_ear_ratio == pytest.approx(0.25)
        assert body_ear_ratio == pytest.approx(1.0 / dist_norm)
