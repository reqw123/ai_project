"""M6（2026-09-15，ext_body_zones 統一 ontology 融合）：`canonical_ext_zone()`
單元測試。純函式/常數，不需要 cv2。
"""

from plugins.lick_stage.analysis_context import ZoneL1, canonical_ext_zone


class TestCanonicalExtZone:
    def test_limb_and_tail_map_directly(self):
        assert canonical_ext_zone("FORELIMB") == ZoneL1.FORELIMB
        assert canonical_ext_zone("HINDLIMB") == ZoneL1.HINDLIMB
        assert canonical_ext_zone("TAIL") == ZoneL1.TAIL

    def test_torso_subregions_collapse_to_torso(self):
        """NECK_CHEST/SIDE_BACK/ABDOMEN/TORSO_UNSPECIFIED 都是軀幹次分區，
        ZoneL1 這一層粗粒度統一成 TORSO（細節留在 ext_zone_mode 原始標籤，
        見 bout_aggregator.py）。"""
        for label in ("NECK_CHEST", "SIDE_BACK", "ABDOMEN", "TORSO_UNSPECIFIED"):
            assert canonical_ext_zone(label) == ZoneL1.TORSO

    def test_head_maps_to_its_own_l1(self):
        """HEAD 是 canonical_zone()（lick_stage 自己）永遠不會產生、只有
        ext_body_zones 才可能產生的類別（見 ZoneL1.HEAD 的說明）。"""
        assert canonical_ext_zone("HEAD") == ZoneL1.HEAD

    def test_no_target_and_ambiguous_are_unknown(self):
        assert canonical_ext_zone("NO_TARGET") == ZoneL1.UNKNOWN
        assert canonical_ext_zone("AMBIGUOUS") == ZoneL1.UNKNOWN

    def test_none_and_unrecognized_string_are_unknown(self):
        assert canonical_ext_zone(None) == ZoneL1.UNKNOWN
        assert canonical_ext_zone("SOME_FUTURE_ZONE_NOT_YET_MAPPED") == ZoneL1.UNKNOWN
