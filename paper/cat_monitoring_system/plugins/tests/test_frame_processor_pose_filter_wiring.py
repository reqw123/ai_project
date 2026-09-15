"""M4 第二階段（接線）：`processors/frame_processor.py` 的 `_notify_plugins()`
如何建構／使用共用 `PoseFilter` 的單元測試。

不透過真正的 `FrameProcessor.__init__`（需要 YOLO/ST-GCN 模型檔案），改用
`types.SimpleNamespace` 組一個最小的 fake `self`，直接呼叫未綁定方法
`FrameProcessor._notify_plugins(fake_self, ...)`——只驗證這一段接線邏輯本身
（PoseFilter 建構參數、smoothed_kpts/pose_quality 有沒有正確傳給 plugin、
reset() 有沒有在正確時機呼叫），不是端到端整合測試。
"""

import types

import numpy as np
import pytest

import processors.frame_processor as fp
from plugins.lick_stage.config import LickConfig as _C


class _RecordingPlugin:
    def __init__(self):
        self.calls = []

    def update(self, kpts, kpt_conf, **ctx):
        self.calls.append(
            {
                "kpts": None if kpts is None else np.asarray(kpts).copy(),
                "pose_quality": ctx.get("pose_quality"),
            }
        )


def _make_fake_processor(pose_filter):
    fake = types.SimpleNamespace()
    fake._plugin_source_fps = 30.0
    fake._pose_filter = pose_filter
    fake._plugins = [_RecordingPlugin()]
    fake.frame_idx = 0
    fake._plugin_session_id = "S_test"
    fake.cap = None

    def _current_source_timestamp(self):
        return self.frame_idx / self._plugin_source_fps

    def _ensure_plugin_sessions(self):
        pass

    fake._current_source_timestamp = types.MethodType(_current_source_timestamp, fake)
    fake._ensure_plugin_sessions = types.MethodType(_ensure_plugin_sessions, fake)
    # _call_plugin_update 是 staticmethod，直接把底層函式掛到 fake 物件上
    # 即可（SimpleNamespace 不是真正的 FrameProcessor 實例，不會自動繼承
    # 類別方法；staticmethod 呼叫沒有 self 綁定問題，直接指派函式參照即可）。
    fake._call_plugin_update = fp.FrameProcessor._call_plugin_update
    return fake


def _default_pose_filter():
    hold_decay_frames = round(_C.POSE_FILTER_HOLD_DECAY_SEC * 30.0)
    return fp._PoseFilter(
        alpha=_C.POSE_FILTER_ALPHA,
        min_conf=_C.POSE_FILTER_MIN_CONF,
        hold_decay_frames=hold_decay_frames,
        bone_pairs=fp.EAR_DISTANCE_SKELETON_EDGES,
    )


class TestPoseFilterAvailability:
    def test_module_resolves_pose_filter_and_config(self):
        """plugins/lick_stage 存在時，frame_processor.py 的 optional import
        guard 應該成功解析成真正的類別，不是停用中的 None（見檔案開頭的
        try/except _PoseFilter/_LickConfig 區塊）。"""
        assert fp._PoseFilter is not None
        assert fp._LickConfig is not None

    def test_default_construction_uses_full_17_edge_skeleton(self):
        pf = _default_pose_filter()
        assert pf.bone_pairs == list(fp.EAR_DISTANCE_SKELETON_EDGES)
        assert len(pf.bone_pairs) == 17


class TestNotifyPluginsWiring:
    def test_smoothed_kpts_passed_to_plugin_not_raw(self):
        """兩幀連續 lick、第二幀關鍵點跳動：plugin 收到的應該是平滑後的值
        （落後 raw），且數值符合 alpha=POSE_FILTER_ALPHA（沿用既有
        EMA_ALPHA）的信心值感知 EMA 公式（conf=1.0 時退化成樸素 EMA）。"""
        fake = _make_fake_processor(_default_pose_filter())
        kpts1 = np.zeros((17, 2), dtype=np.float64)
        conf = np.ones((17,), dtype=np.float64)
        kpts2 = np.full((17, 2), 10.0, dtype=np.float64)

        fp.FrameProcessor._notify_plugins(
            fake, kpts1, conf, cat_present=True, is_lick=True, lick_confidence=0.9
        )
        fake.frame_idx = 1
        fp.FrameProcessor._notify_plugins(
            fake, kpts2, conf, cat_present=True, is_lick=True, lick_confidence=0.9
        )

        plugin = fake._plugins[0]
        assert len(plugin.calls) == 2
        np.testing.assert_array_equal(plugin.calls[0]["kpts"], kpts1)
        expected_alpha = _C.POSE_FILTER_ALPHA
        expected_frame2 = expected_alpha * kpts2 + (1.0 - expected_alpha) * kpts1
        np.testing.assert_allclose(plugin.calls[1]["kpts"], expected_frame2)
        # 確實有平滑效果（不是逐位元等於原始跳動值）
        assert not np.allclose(plugin.calls[1]["kpts"], kpts2)

    def test_pose_quality_none_until_bone_length_history_available(self):
        """骨長品質第一幀必然是 None（見 pose_filter.py 的
        `_compute_quality()` 說明：第一次看到每對骨骼還沒有預期值可比較）；
        餵入穩定的第二幀後應該產生實際的浮點分數。"""
        fake = _make_fake_processor(_default_pose_filter())
        kpts = np.zeros((17, 2), dtype=np.float64)
        for i, (x, y) in enumerate([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]):
            kpts = np.arange(34, dtype=np.float64).reshape(17, 2) + np.array([x, y])
            conf = np.ones((17,), dtype=np.float64)
            fake.frame_idx = i
            fp.FrameProcessor._notify_plugins(
                fake, kpts, conf, cat_present=True, is_lick=True, lick_confidence=0.9
            )

        plugin = fake._plugins[0]
        assert plugin.calls[0]["pose_quality"] is None
        assert plugin.calls[-1]["pose_quality"] is not None
        assert 0.0 <= plugin.calls[-1]["pose_quality"] <= 1.0

    def test_non_lick_frame_resets_filter_and_feeds_none(self):
        """is_lick=False（含有貓但非舔毛、或整幀無貓）時，plugin 仍應收到
        kpts=None（維持既有「不污染統計」行為），且共用 PoseFilter 的跨幀
        平滑狀態要被 reset()，跟 lick_stage/analyzer.py 既有的
        `_reset_transient_state()` 政策保持一致（見 frame_processor.py
        `_notify_plugins()` 的說明）。"""
        pose_filter = _default_pose_filter()
        fake = _make_fake_processor(pose_filter)
        kpts = np.ones((17, 2), dtype=np.float64) * 5.0
        conf = np.ones((17,), dtype=np.float64)
        fp.FrameProcessor._notify_plugins(
            fake, kpts, conf, cat_present=True, is_lick=True, lick_confidence=0.9
        )
        assert pose_filter._smoothed is not None

        fake.frame_idx = 1
        fp.FrameProcessor._notify_plugins(
            fake, kpts, conf, cat_present=True, is_lick=False, lick_confidence=0.1
        )

        plugin = fake._plugins[0]
        assert plugin.calls[-1]["kpts"] is None
        assert plugin.calls[-1]["pose_quality"] is None
        assert pose_filter._smoothed is None  # reset() 已清空跨幀狀態

    def test_disabled_pose_filter_falls_back_to_raw_passthrough(self):
        """plugins/lick_stage 整個被移除（import guard 觸發，`_pose_filter`
        為 None）時，_notify_plugins() 必須退回舊行為：kpts 原樣傳給
        plugin、pose_quality 恆為 None，不能拋例外。"""
        fake = _make_fake_processor(None)
        kpts = np.ones((17, 2), dtype=np.float64) * 7.0
        conf = np.ones((17,), dtype=np.float64)
        fp.FrameProcessor._notify_plugins(
            fake, kpts, conf, cat_present=True, is_lick=True, lick_confidence=0.9
        )
        plugin = fake._plugins[0]
        np.testing.assert_array_equal(plugin.calls[0]["kpts"], kpts)
        assert plugin.calls[0]["pose_quality"] is None
