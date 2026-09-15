"""M6（2026-09-15，ext_body_zones 統一 ontology 融合）：
`ExtBodyZonePlugin.last_zone_name`/`last_confidence` 單元測試——這兩個屬性
是新公開狀態，供 `frame_processor.py::_notify_plugins()` 讀取後轉餵給
`LickStagePlugin`（見該方法內的說明），本檔案只驗證這兩個屬性本身在各種
`update()` 呼叫情境下是否正確更新，不涉及 frame_processor 的融合邏輯
（那部分見 test_frame_processor_pose_filter_wiring.py）。

跟 test_ext_body_zones_regions_unit.py 用同一套 fixture 慣例（見該檔案開頭
說明，import 需要 cv2）。
"""

import numpy as np
import pytest

pytest.importorskip(
    "cv2",
    reason="plugins/lick_stage/__init__.py 透過 manager.py/overlay.py 需要 cv2，此環境未安裝",
)

from plugins.lick_stage.ext_body_zones.config import ExtZoneConfig as _C
from plugins.lick_stage.ext_body_zones.plugin import ExtBodyZonePlugin

NUM_JOINTS = 17


def _kpts_hit_forelimb():
    """nose 剛好落在前肢腳掌圓中心 → classify_zone() 應回傳 FORELIMB，
    信心值 1.0（照抄 test_ext_body_zones_regions_unit.py 的同款 fixture）。"""
    kpts = np.zeros((NUM_JOINTS, 2), dtype=np.float64)
    kpts[_C.KP_CHEST] = (0, 0)
    kpts[_C.KP_HIP] = (0, 100)
    kpts[_C.KP_FL_KNEE] = (100, 0)
    kpts[_C.KP_FL_PAW] = (100, 100)
    kpts[_C.KP_NOSE] = (100, 100)
    return kpts


def _full_conf():
    conf = np.zeros(NUM_JOINTS, dtype=np.float64)
    conf[_C.KP_CHEST] = 1.0
    conf[_C.KP_HIP] = 1.0
    conf[_C.KP_FL_KNEE] = 1.0
    conf[_C.KP_FL_PAW] = 1.0
    conf[_C.KP_NOSE] = 1.0
    return conf


class TestLastResultDefaultsAndReset:
    def test_default_before_any_update_is_no_target(self):
        p = ExtBodyZonePlugin(csv_path=None, mqtt_enabled=False, nodered_enabled=False)
        assert p.last_zone_name == "NO_TARGET"
        assert p.last_confidence is None

    def test_no_cat_frame_resets_to_no_target(self):
        p = ExtBodyZonePlugin(csv_path=None, mqtt_enabled=False, nodered_enabled=False)
        p.update(_kpts_hit_forelimb(), _full_conf(), cat_present=True, is_lick=True)
        assert p.last_zone_name == "FORELIMB"

        p.update(None, None, cat_present=False, is_lick=False)
        assert p.last_zone_name == "NO_TARGET"
        assert p.last_confidence is None

    def test_not_lick_frame_resets_to_no_target(self):
        p = ExtBodyZonePlugin(csv_path=None, mqtt_enabled=False, nodered_enabled=False)
        p.update(_kpts_hit_forelimb(), _full_conf(), cat_present=True, is_lick=True)
        assert p.last_zone_name == "FORELIMB"

        p.update(None, None, cat_present=True, is_lick=False)
        assert p.last_zone_name == "NO_TARGET"
        assert p.last_confidence is None

    def test_lick_without_pose_resets_to_no_target(self):
        p = ExtBodyZonePlugin(csv_path=None, mqtt_enabled=False, nodered_enabled=False)
        p.update(_kpts_hit_forelimb(), _full_conf(), cat_present=True, is_lick=True)
        assert p.last_zone_name == "FORELIMB"

        p.update(None, None, cat_present=True, is_lick=True)
        assert p.last_zone_name == "NO_TARGET"
        assert p.last_confidence is None


class TestLastResultReflectsClassification:
    def test_forelimb_hit_updates_both_fields(self):
        p = ExtBodyZonePlugin(csv_path=None, mqtt_enabled=False, nodered_enabled=False)
        p.update(
            _kpts_hit_forelimb(), _full_conf(),
            cat_present=True, is_lick=True, frame_idx=0,
        )
        assert p.last_zone_name == "FORELIMB"
        assert p.last_confidence == pytest.approx(1.0)

    def test_no_region_hit_gives_no_target_with_zero_confidence_sentinel(self):
        """鼻子完全不在任何候選區域內：classify_zone() 回傳 (ZONE_NO_TARGET,
        "NO_TARGET", 0.0) —— 0.0 是哨兵值不是真信心值，這裡只驗證屬性值
        本身如實反映 classify_zone() 的回傳，「0.0 不該被當真信心值使用」
        的判斷留給呼叫端（frame_processor.py，見該檔案 _notify_plugins()
        的說明），本模組不做過濾。"""
        p = ExtBodyZonePlugin(csv_path=None, mqtt_enabled=False, nodered_enabled=False)
        kpts = np.zeros((NUM_JOINTS, 2), dtype=np.float64)
        kpts[_C.KP_CHEST] = (0, 0)
        kpts[_C.KP_HIP] = (0, 100)
        conf = np.zeros(NUM_JOINTS, dtype=np.float64)
        conf[_C.KP_CHEST] = 1.0
        conf[_C.KP_HIP] = 1.0
        # 鼻子遠離所有候選區域（軀幹橢圓/四肢/尾巴）
        kpts[_C.KP_NOSE] = (100000.0, 100000.0)
        conf[_C.KP_NOSE] = 1.0
        p.update(kpts, conf, cat_present=True, is_lick=True, frame_idx=0)
        assert p.last_zone_name == "NO_TARGET"
        assert p.last_confidence == pytest.approx(0.0)
