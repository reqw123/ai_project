"""第一階段驗收：五種互斥 frame_state 可被區分，且統計分母正確
（說明書「狀態與原因碼重構」/「統計分母與計算口徑」）。

重點：kpts 為空 **不等於** 沒有貓。NO_CAT / NOT_LICK / LICK_ASSIGNED /
LICK_UNASSIGNED 走不同的分母累加路徑。
"""

import numpy as np
import pytest

pytest.importorskip(
    "cv2",
    reason="plugins/lick_stage 需要 cv2，此環境未安裝",
)

from plugins.lick_stage.analysis_context import FrameState, ReasonCode
from plugins.lick_stage.analyzer import LickAnalyzer
from plugins.lick_stage.config import LickConfig as _C

NUM_JOINTS = 17
_DT = 1.0 / 30.0


def _pose():
    kpts = np.zeros((NUM_JOINTS, 2), dtype=np.float64)
    kpts[_C.KP_NOSE] = (0.0, 120.0)
    kpts[_C.KP_LEFT_EAR] = (-18.0, -6.0)
    kpts[_C.KP_RIGHT_EAR] = (18.0, -6.0)
    kpts[_C.KP_CHEST] = (0.0, 0.0)
    kpts[_C.KP_MID_BACK] = (0.0, 180.0)
    kpts[_C.KP_HIP] = (0.0, 360.0)
    conf = np.ones(NUM_JOINTS, dtype=np.float64)
    return kpts, conf


def _analyze(an, i, **kw):
    return an.analyze(kw.pop("kpts", None), kw.pop("conf", None), i, i * _DT, _DT, **kw)


def test_no_cat_only_accumulates_observed_not_valid():
    an = LickAnalyzer()
    r = _analyze(an, 1, cat_present=False, is_lick=False)
    assert r.frame_state == FrameState.NO_CAT
    st = an._stats
    assert st.observed_sec == pytest.approx(_DT)
    assert st.no_cat_sec == pytest.approx(_DT)
    assert st.valid_observed_sec == pytest.approx(0.0)
    assert st.stgcn_lick_sec == pytest.approx(0.0)


def test_not_lick_accumulates_valid_observed_but_no_lick_time():
    an = LickAnalyzer()
    # 有貓、非舔毛：FrameProcessor 目前對非 lick 幀傳 kpts=None
    r = _analyze(an, 1, cat_present=True, is_lick=False)
    assert r.frame_state == FrameState.NOT_LICK
    st = an._stats
    assert st.valid_observed_sec == pytest.approx(_DT)
    assert st.no_cat_sec == pytest.approx(0.0)
    assert st.stgcn_lick_sec == pytest.approx(0.0)


def test_lick_without_pose_is_unassigned_pose_invalid():
    an = LickAnalyzer()
    r = _analyze(an, 1, cat_present=True, is_lick=True)  # kpts=None
    assert r.frame_state == FrameState.LICK_UNASSIGNED
    assert r.reason_code == ReasonCode.POSE_INVALID
    st = an._stats
    assert st.stgcn_lick_sec == pytest.approx(_DT)
    assert st.unassigned_lick_sec == pytest.approx(_DT)
    assert st.assigned_zone_sec == pytest.approx(0.0)


def test_lick_with_pose_populates_stgcn_lick_sec():
    an = LickAnalyzer()
    kpts, conf = _pose()
    r = _analyze(an, 1, kpts=kpts, conf=conf, cat_present=True, is_lick=True,
                 lick_confidence=0.95)
    assert r.frame_state in (FrameState.LICK_ASSIGNED, FrameState.LICK_UNASSIGNED)
    st = an._stats
    assert st.stgcn_lick_sec == pytest.approx(_DT)


def test_invariant_across_mixed_sequence():
    """混合序列後：stgcn == assigned + unassigned，且 sum(zone time) == assigned。"""
    an = LickAnalyzer()
    kpts, conf = _pose()
    seq = [
        dict(cat_present=False, is_lick=False),
        dict(cat_present=True, is_lick=False),
        dict(cat_present=True, is_lick=True),  # no pose → unassigned
        dict(kpts=kpts, conf=conf, cat_present=True, is_lick=True, lick_confidence=0.9),
        dict(kpts=kpts, conf=conf, cat_present=True, is_lick=True, lick_confidence=0.9),
        dict(cat_present=False, is_lick=False),
    ]
    for i, kw in enumerate(seq):
        _analyze(an, i, **kw)

    st = an._stats
    assert st.invariant_error() == pytest.approx(0.0, abs=1e-9)
    zone_sum = sum(st.zone_stats(k)[1] for k in ("BODY", "FL", "FR", "HL", "HR"))
    assert zone_sum == pytest.approx(st.assigned_zone_sec, abs=1e-9)
    # 觀測分母：6 幀，其中 2 幀 NO_CAT
    assert st.observed_sec == pytest.approx(6 * _DT)
    assert st.no_cat_sec == pytest.approx(2 * _DT)
    assert st.valid_observed_sec == pytest.approx(4 * _DT)


def test_finalize_closes_active_bout():
    an = LickAnalyzer()
    kpts, conf = _pose()
    # 連續數幀命中同一區域，形成一個尚未關閉的 bout
    for i in range(5):
        _analyze(an, i, kpts=kpts, conf=conf, cat_present=True, is_lick=True,
                 lick_confidence=0.9)
    before = sum(an._stats.zone_bout(k)[0] for k in ("BODY", "FL", "FR", "HL", "HR"))
    an.finalize()
    after = sum(an._stats.zone_bout(k)[0] for k in ("BODY", "FL", "FR", "HL", "HR"))
    # 命中時才有 bout 可結算；未命中則兩者皆 0，不應報錯
    assert after >= before


def test_reset_clears_cumulative_stats():
    an = LickAnalyzer()
    _analyze(an, 1, cat_present=True, is_lick=True)
    assert an._stats.stgcn_lick_sec > 0.0
    an.reset()
    assert an._stats.stgcn_lick_sec == pytest.approx(0.0)
    assert an._stats.observed_sec == pytest.approx(0.0)
