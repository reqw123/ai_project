"""M6（2026-09-15，ext_body_zones 統一 ontology 融合）：
`frame_processor.py::_notify_plugins()` 如何把 `ExtBodyZonePlugin` 的逐幀
分類結果轉餵給 `LickStagePlugin` 的單元測試。

跟 test_frame_processor_pose_filter_wiring.py 同一套手法：不透過真正的
`FrameProcessor.__init__`，用 `types.SimpleNamespace` 組最小 fake `self`，
直接呼叫未綁定方法 `FrameProcessor._notify_plugins(fake_self, ...)`。

這裡用**真正的** `ExtBodyZonePlugin`/`LickStagePlugin` 子類別（覆寫
`__init__`/`update()` 跳過重量級初始化，只留下讓 `isinstance()` 判斷成立
所需要的繼承關係）而不是完全獨立的假物件——因為 `_notify_plugins()` 的融合
邏輯是用 `isinstance(_plugin, _ExtBodyZonePlugin)`/`isinstance(_plugin,
_LickStagePlugin)` 判斷要不要套用融合邏輯，純 duck-typing 的假物件不會被
辨識出來，測不到這裡真正要驗證的行為。
"""

import types

import numpy as np
import pytest

pytest.importorskip(
    "cv2",
    reason="plugins/lick_stage/__init__.py 透過 manager.py/overlay.py 需要 cv2，此環境未安裝",
)

import processors.frame_processor as fp
from plugins.lick_stage.ext_body_zones.plugin import ExtBodyZonePlugin as _RealExtPlugin
from plugins.lick_stage.manager import LickStagePlugin as _RealLickPlugin


class _FakeExtPlugin(_RealExtPlugin):
    """繼承真正的 ExtBodyZonePlugin（讓 isinstance 判斷成立），但跳過重量級
    __init__（CSV/MQTT/Node-RED 輸出端）與真正的分類演算法，改成純記錄呼叫
    參數＋讓測試自己指定 last_zone_name/last_confidence。"""

    def __init__(self, zone_name="NO_TARGET", confidence=None):
        self.calls = []
        self.last_zone_name = zone_name
        self.last_confidence = confidence

    def update(self, kpts, kpt_conf, **ctx):
        self.calls.append({"kpts": kpts, "ctx": ctx})


class _FakeLickPlugin(_RealLickPlugin):
    """繼承真正的 LickStagePlugin（讓 isinstance 判斷成立），跳過重量級
    __init__（analyzer/storage/publisher），只記錄收到的呼叫參數。"""

    def __init__(self):
        self.calls = []

    def update(self, kpts, kpt_conf, **ctx):
        self.calls.append({"kpts": kpts, "ctx": ctx})


class _FakeOtherPlugin:
    """既非 ext_body_zones 也非 lick_stage 的第三方外掛（純 duck-typing，
    驗證融合邏輯不會誤把 ext_zone_name/ext_zone_confidence 塞給不相關的
    外掛）。"""

    def __init__(self):
        self.calls = []

    def update(self, kpts, kpt_conf, **ctx):
        self.calls.append({"kpts": kpts, "ctx": ctx})


def _make_fake_processor(plugins):
    fake = types.SimpleNamespace()
    fake._plugin_source_fps = 30.0
    fake._pose_filter = None  # 這組測試不關心 M4 平滑，維持停用最單純
    fake._plugins = plugins
    fake.frame_idx = 0
    fake._plugin_session_id = "S_test"
    fake.cap = None

    def _current_source_timestamp(self):
        return self.frame_idx / self._plugin_source_fps

    def _ensure_plugin_sessions(self):
        pass

    fake._current_source_timestamp = types.MethodType(_current_source_timestamp, fake)
    fake._ensure_plugin_sessions = types.MethodType(_ensure_plugin_sessions, fake)
    fake._call_plugin_update = fp.FrameProcessor._call_plugin_update
    return fake


_KPTS = np.zeros((17, 2), dtype=np.float64)
_CONF = np.ones((17,), dtype=np.float64)


class TestExtZoneFusionWiring:
    def test_ext_result_forwarded_to_lick_stage_kwargs(self):
        ext = _FakeExtPlugin(zone_name="ABDOMEN", confidence=0.77)
        lick = _FakeLickPlugin()
        fake = _make_fake_processor([lick, ext])  # 刻意反過來註冊順序
        fp.FrameProcessor._notify_plugins(
            fake, _KPTS, _CONF, cat_present=True, is_lick=True, lick_confidence=0.9
        )
        assert lick.calls[0]["ctx"]["ext_zone_name"] == "ABDOMEN"
        assert lick.calls[0]["ctx"]["ext_zone_confidence"] == pytest.approx(0.77)

    def test_ext_plugin_itself_does_not_receive_ext_zone_kwargs(self):
        """ext_body_zones 不需要、也不該收到自己剛算出來的結果——這兩個
        kwargs 只塞給 LickStagePlugin。"""
        ext = _FakeExtPlugin(zone_name="ABDOMEN", confidence=0.77)
        lick = _FakeLickPlugin()
        fake = _make_fake_processor([ext, lick])
        fp.FrameProcessor._notify_plugins(
            fake, _KPTS, _CONF, cat_present=True, is_lick=True, lick_confidence=0.9
        )
        assert "ext_zone_name" not in ext.calls[0]["ctx"]
        assert "ext_zone_confidence" not in ext.calls[0]["ctx"]

    def test_third_party_plugin_does_not_receive_ext_zone_kwargs(self):
        ext = _FakeExtPlugin(zone_name="ABDOMEN", confidence=0.77)
        other = _FakeOtherPlugin()
        fake = _make_fake_processor([ext, other])
        fp.FrameProcessor._notify_plugins(
            fake, _KPTS, _CONF, cat_present=True, is_lick=True, lick_confidence=0.9
        )
        assert "ext_zone_name" not in other.calls[0]["ctx"]
        assert "ext_zone_confidence" not in other.calls[0]["ctx"]

    def test_no_target_is_filtered_to_none(self):
        """ext_body_zones 這幀分類是 NO_TARGET（沒有具體 zone）：轉餵給
        lick_stage 的兩個欄位都應該是 None，不是字面的 "NO_TARGET" 字串或
        0.0 哨兵信心值（見 _notify_plugins() 的說明）。"""
        ext = _FakeExtPlugin(zone_name="NO_TARGET", confidence=0.0)
        lick = _FakeLickPlugin()
        fake = _make_fake_processor([ext, lick])
        fp.FrameProcessor._notify_plugins(
            fake, _KPTS, _CONF, cat_present=True, is_lick=True, lick_confidence=0.9
        )
        assert lick.calls[0]["ctx"]["ext_zone_name"] is None
        assert lick.calls[0]["ctx"]["ext_zone_confidence"] is None

    def test_no_ext_plugin_registered_gives_none_without_error(self):
        """完全沒有註冊 ext_body_zones：lick_stage 仍應正常收到呼叫，兩個
        欄位是 None，不拋例外（見「可以整個被刪除」的既有 fail-safe 原則）。"""
        lick = _FakeLickPlugin()
        fake = _make_fake_processor([lick])
        fp.FrameProcessor._notify_plugins(
            fake, _KPTS, _CONF, cat_present=True, is_lick=True, lick_confidence=0.9
        )
        assert lick.calls[0]["ctx"]["ext_zone_name"] is None
        assert lick.calls[0]["ctx"]["ext_zone_confidence"] is None

    def test_ext_updated_before_lick_receives_its_result_even_when_registered_after(self):
        """驗證真正的呼叫順序無關性：即使 ext_body_zones 在註冊列表裡排在
        lick_stage 之後（例如 tools/verify_lick_stage_m2.py 的註冊順序），
        融合邏輯仍要先跑 ext 才能把結果餵給 lick。這裡用一個會在 update()
        當下才動態設定 last_zone_name 的 ext 假外掛，模擬「真正算出結果」
        這件事發生在 update() 呼叫時，不是建構時。"""

        class _LazyExtPlugin(_FakeExtPlugin):
            def update(self, kpts, kpt_conf, **ctx):
                super().update(kpts, kpt_conf, **ctx)
                self.last_zone_name = "HINDLIMB"
                self.last_confidence = 0.42

        ext = _LazyExtPlugin()
        lick = _FakeLickPlugin()
        fake = _make_fake_processor([lick, ext])  # lick_stage 先註冊
        fp.FrameProcessor._notify_plugins(
            fake, _KPTS, _CONF, cat_present=True, is_lick=True, lick_confidence=0.9
        )
        assert lick.calls[0]["ctx"]["ext_zone_name"] == "HINDLIMB"
        assert lick.calls[0]["ctx"]["ext_zone_confidence"] == pytest.approx(0.42)
