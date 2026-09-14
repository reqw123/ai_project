"""M4（第一階段：模組本身，尚未接進 frame_processor.py/analyzer.py）：
`pose_filter.PoseFilter` 單元測試。

不需要 cv2——pose_filter.py/pose_frame.py 只依賴 numpy。
"""

import numpy as np
import pytest

from plugins.lick_stage.pose_filter import PoseFilter

# 3 個關鍵點：0=nose, 1=chest, 2=hip（骨長品質測試用 (0,1)/(1,2) 兩段骨骼）
_N = 3


def _kpts(nose_x=0.0, nose_y=0.0, chest=(0.0, 100.0), hip=(0.0, 200.0)):
    return np.array([[nose_x, nose_y], list(chest), list(hip)], dtype=np.float64)


def _conf(vals=(1.0, 1.0, 1.0)):
    return np.array(vals, dtype=np.float64)


class TestBypassAtAlphaOne:
    """alpha>=1.0：明確要求「完全不平滑」，不管信心值多低、跑幾幀，
    smoothed_kpts 永遠逐元素等於 raw_kpts。"""

    def test_alpha_exactly_one_is_pure_passthrough(self):
        pf = PoseFilter(alpha=1.0)
        for i in range(5):
            frame = pf.update(_kpts(nose_x=float(i)), _conf((0.01, 0.01, 0.01)), frame_idx=i)
            np.testing.assert_array_equal(frame.smoothed_kpts, frame.raw_kpts)

    def test_alpha_above_one_also_bypasses(self):
        """>1.0（不合理但不該崩潰）也視同不平滑，跟 `_BYPASS_EPS` 的判斷式
        (`alpha >= 1.0 - 1e-9`) 一致。"""
        pf = PoseFilter(alpha=1.5)
        frame = pf.update(_kpts(), _conf())
        np.testing.assert_array_equal(frame.smoothed_kpts, frame.raw_kpts)

    def test_bypass_returns_copy_not_same_array(self):
        """raw_kpts 跟 smoothed_kpts 內容相等，但不是同一個物件——避免呼叫端
        修改其中一份意外污染另一份。"""
        pf = PoseFilter(alpha=1.0)
        kpts = _kpts()
        frame = pf.update(kpts, _conf())
        assert frame.smoothed_kpts is not frame.raw_kpts
        frame.smoothed_kpts[0, 0] = 999.0
        assert frame.raw_kpts[0, 0] != 999.0


class TestParityWithLegacyEma:
    """min_conf/hold_decay_frames/bone_pairs 全部不給、信心值全部 1.0 時，
    數值上要跟 analyzer.py 既有的樸素 EMA
    （`alpha*new + (1-alpha)*old`）逐幀完全一致。"""

    def test_matches_naive_ema_formula_across_frames(self):
        alpha = 0.4
        pf = PoseFilter(alpha=alpha)
        expected = None
        rng = np.random.default_rng(0)
        for i in range(10):
            raw = _kpts(nose_x=float(rng.uniform(-50, 50)))
            frame = pf.update(raw, _conf(), frame_idx=i)
            if expected is None:
                expected = raw.copy()
            else:
                expected = alpha * raw + (1.0 - alpha) * expected
            np.testing.assert_allclose(frame.smoothed_kpts, expected)


class TestConfidenceAwareScaling:
    def test_low_confidence_point_moves_less_than_high_confidence_point(self):
        """同一幀內，同樣的座標跳動量，高信心的點應該比低信心的點更快跟上
        新讀數（effective_alpha 正比於信心值）。"""
        pf = PoseFilter(alpha=0.5)
        pf.update(_kpts(nose_x=0.0, chest=(0.0, 100.0)), _conf((1.0, 1.0, 1.0)), frame_idx=0)
        # 第二幀：nose 高信心跳到 100；chest 低信心也跳到同樣幅度
        frame = pf.update(
            _kpts(nose_x=100.0, chest=(100.0, 100.0)),
            _conf((1.0, 0.1, 1.0)),
            frame_idx=1,
        )
        nose_moved = abs(frame.smoothed_kpts[0, 0] - 0.0)
        chest_moved = abs(frame.smoothed_kpts[1, 0] - 0.0)
        assert nose_moved > chest_moved

    def test_nan_confidence_treated_as_zero_without_min_conf(self):
        """沒設 min_conf 時，NaN 信心值透過同一條公式自然變成
        effective_alpha=0（凍結），不需要另外的分支。"""
        pf = PoseFilter(alpha=0.5)
        pf.update(_kpts(nose_x=0.0), _conf(), frame_idx=0)
        frame = pf.update(
            _kpts(nose_x=999.0), np.array([float("nan"), 1.0, 1.0]), frame_idx=1
        )
        assert frame.smoothed_kpts[0, 0] == pytest.approx(0.0)  # 完全沒被新讀數拉動


class TestHoldDecay:
    def test_holds_position_while_within_hold_decay_frames(self):
        pf = PoseFilter(alpha=0.5, min_conf=0.3, hold_decay_frames=3)
        pf.update(_kpts(nose_x=0.0), _conf(), frame_idx=0)
        for i in range(1, 4):  # 3 幀低信心（未超過 hold_decay_frames=3）
            frame = pf.update(_kpts(nose_x=100.0), _conf((0.05, 1.0, 1.0)), frame_idx=i)
            assert frame.smoothed_kpts[0, 0] == pytest.approx(0.0)

    def test_decays_toward_raw_after_hold_decay_frames_exceeded(self):
        pf = PoseFilter(alpha=0.5, min_conf=0.3, hold_decay_frames=2, decay_alpha=0.2)
        pf.update(_kpts(nose_x=0.0), _conf(), frame_idx=0)
        for i in range(1, 3):  # 2 幀凍結期
            pf.update(_kpts(nose_x=100.0), _conf((0.05, 1.0, 1.0)), frame_idx=i)
        # 第 3 幀低信心：超過 hold_decay_frames，開始用 decay_alpha 緩慢拉回
        frame = pf.update(_kpts(nose_x=100.0), _conf((0.05, 1.0, 1.0)), frame_idx=3)
        assert frame.smoothed_kpts[0, 0] == pytest.approx(0.2 * 100.0)

    def test_recovering_confidence_resets_streak_and_resumes_normal_scaling(self):
        pf = PoseFilter(alpha=0.5, min_conf=0.3, hold_decay_frames=1)
        pf.update(_kpts(nose_x=0.0), _conf(), frame_idx=0)
        pf.update(_kpts(nose_x=100.0), _conf((0.05, 1.0, 1.0)), frame_idx=1)  # hold
        # 信心恢復：立刻回到正常的 confidence-aware EMA（不是還卡在 decay_alpha）
        frame = pf.update(_kpts(nose_x=100.0), _conf((1.0, 1.0, 1.0)), frame_idx=2)
        assert frame.smoothed_kpts[0, 0] == pytest.approx(0.5 * 100.0)

    def test_without_min_conf_hold_decay_tier_never_triggers(self):
        """min_conf=None（預設）：完全不啟用這層，低信心走第 1 點的連續縮放
        公式，不是凍結在 0。"""
        pf = PoseFilter(alpha=0.5)  # min_conf 預設 None
        pf.update(_kpts(nose_x=0.0), _conf(), frame_idx=0)
        frame = pf.update(_kpts(nose_x=100.0), _conf((0.1, 1.0, 1.0)), frame_idx=1)
        assert frame.smoothed_kpts[0, 0] == pytest.approx(0.5 * 0.1 * 100.0)


class TestMissingWholeFrame:
    def test_none_kpts_returns_all_none_pose_frame(self):
        pf = PoseFilter(alpha=0.5)
        frame = pf.update(None, None, frame_idx=0)
        assert frame.raw_kpts is None
        assert frame.raw_conf is None
        assert frame.smoothed_kpts is None
        assert frame.quality is None
        assert frame.has_pose is False

    def test_missing_frame_does_not_disturb_smoothing_state(self):
        """整幀缺失不算進 hold_decay 的缺點計數，也不會重置平滑狀態——
        中斷後恢復時，平滑狀態延續中斷前的樣子。"""
        pf = PoseFilter(alpha=0.4)
        pf.update(_kpts(nose_x=0.0), _conf(), frame_idx=0)
        before = pf.update(_kpts(nose_x=100.0), _conf(), frame_idx=1).smoothed_kpts.copy()
        pf.update(None, None, frame_idx=2)  # 整幀缺失，不應該影響下面的結果
        raw_again = _kpts(nose_x=100.0)
        after = pf.update(raw_again, _conf(), frame_idx=3).smoothed_kpts
        # 直接再餵一次同樣的讀數（等同 frame_idx=1 之後緊接著再餵一次
        # nose_x=100.0），驗證平滑狀態確實是從 `before` 接續，不是被清空
        # 重新從 raw_again 起跳
        expected = 0.4 * raw_again + 0.6 * before
        np.testing.assert_allclose(after, expected)


class TestBoneLengthQuality:
    _BONES = [(0, 1), (1, 2)]  # nose-chest, chest-hip

    def test_quality_is_none_without_bone_pairs(self):
        pf = PoseFilter(alpha=1.0)
        frame = pf.update(_kpts(), _conf())
        assert frame.quality is None

    def test_first_frame_quality_is_none_not_enough_history(self):
        pf = PoseFilter(alpha=1.0, bone_pairs=self._BONES)
        frame = pf.update(_kpts(), _conf())
        assert frame.quality is None

    def test_consistent_bone_length_gives_high_quality(self):
        pf = PoseFilter(alpha=1.0, bone_pairs=self._BONES, bone_len_ema_alpha=0.5)
        pf.update(_kpts(chest=(0.0, 100.0), hip=(0.0, 200.0)), _conf())
        frame = pf.update(_kpts(chest=(0.0, 100.0), hip=(0.0, 200.0)), _conf())
        assert frame.quality == pytest.approx(1.0, abs=1e-6)

    def test_deviated_bone_length_lowers_quality(self):
        pf = PoseFilter(alpha=1.0, bone_pairs=self._BONES, bone_len_ema_alpha=0.5)
        pf.update(_kpts(chest=(0.0, 100.0), hip=(0.0, 200.0)), _conf())  # 骨長各 100
        # 第二幀 hip 突然跳很遠：chest-hip 骨長劇變
        frame = pf.update(_kpts(chest=(0.0, 100.0), hip=(0.0, 500.0)), _conf())
        assert frame.quality < 0.5

    def test_quality_computed_even_when_alpha_bypasses_position_smoothing(self):
        """骨長品質是獨立測量，alpha=1.0（不平滑位置）時仍要正常運作。"""
        pf = PoseFilter(alpha=1.0, bone_pairs=self._BONES, bone_len_ema_alpha=0.5)
        pf.update(_kpts(chest=(0.0, 100.0), hip=(0.0, 200.0)), _conf())
        frame = pf.update(_kpts(chest=(0.0, 100.0), hip=(0.0, 200.0)), _conf())
        assert frame.quality is not None


class TestReset:
    def test_reset_clears_smoothing_state(self):
        pf = PoseFilter(alpha=0.4, bone_pairs=[(0, 1)])
        pf.update(_kpts(nose_x=0.0), _conf())
        pf.update(_kpts(nose_x=100.0), _conf())
        pf.reset()
        # reset 後第一幀應該像全新開始：smoothed == raw（沒有歷史可 blend）
        frame = pf.update(_kpts(nose_x=999.0), _conf())
        np.testing.assert_array_equal(frame.smoothed_kpts, frame.raw_kpts)
        assert frame.quality is None  # 骨長歷史也一併清空
