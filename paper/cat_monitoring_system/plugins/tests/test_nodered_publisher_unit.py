"""涵蓋 2026-08-10 的 Node-RED 端點集中化修正：

1. LickConfig/ExtZoneConfig 的 NODERED_URL 預設值應該跟主專案
   config.NodeRedConfig.HOST/PORT 一致（不是各自寫死 127.0.0.1:1880）。
2. NodeRedPublisher/ZoneHttpPublisher 推送失敗時應該記警告（節流過，
   同一實例短時間內重複失敗不會洗版），不再是完全靜默的 except: pass。
"""

import logging

import pytest

from config import NodeRedConfig
from plugins.lick_stage.config import LickConfig
from plugins.lick_stage.ext_body_zones.config import ExtZoneConfig
from plugins.lick_stage.publisher import NodeRedPublisher
from plugins.lick_stage.ext_body_zones.output import ZoneHttpPublisher


def test_lick_config_url_derives_from_shared_nodered_config():
    assert LickConfig.NODERED_URL == (
        f"http://{NodeRedConfig.HOST}:{NodeRedConfig.PORT}/lick_zone_result"
    )


def test_ext_zone_config_url_derives_from_shared_nodered_config():
    assert ExtZoneConfig.NODERED_URL == (
        f"http://{NodeRedConfig.HOST}:{NodeRedConfig.PORT}/ext_zone_result"
    )


class _FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def time(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def fake_requests_that_always_fail(monkeypatch):
    import plugins.lick_stage.publisher as publisher_module

    class _Boom:
        @staticmethod
        def post(*a, **kw):
            raise ConnectionError("simulated Node-RED unreachable")

    monkeypatch.setattr(publisher_module, "_requests", _Boom)
    monkeypatch.setattr(publisher_module, "_HAS_REQUESTS", True)
    return publisher_module


def test_lick_publisher_logs_warning_on_failure_and_throttles(
    fake_requests_that_always_fail, monkeypatch, caplog
):
    publisher_module = fake_requests_that_always_fail
    clock = _FakeClock()
    monkeypatch.setattr(publisher_module.time, "time", clock.time)

    pub = NodeRedPublisher(url="http://example.invalid/x", timeout=0.1)
    caplog.set_level(logging.WARNING, logger=publisher_module.__name__)
    pub._post({"a": 1})  # 第一次失敗：應該警告
    pub._post({"a": 2})  # 馬上又失敗：在節流窗口內，不應該再警告
    assert len(caplog.records) == 1

    caplog.clear()
    clock.advance(publisher_module._WARN_INTERVAL_SEC + 1)
    pub._post({"a": 3})  # 節流窗口過了，應該再警告一次
    assert len(caplog.records) == 1


@pytest.fixture
def fake_ext_zone_requests_that_always_fail(monkeypatch):
    import plugins.lick_stage.ext_body_zones.output as output_module

    class _Boom:
        @staticmethod
        def post(*a, **kw):
            raise ConnectionError("simulated Node-RED unreachable")

    monkeypatch.setattr(output_module, "_requests", _Boom)
    monkeypatch.setattr(output_module, "_HAS_REQUESTS", True)
    return output_module


def test_ext_zone_http_publisher_logs_warning_on_failure_and_throttles(
    fake_ext_zone_requests_that_always_fail, monkeypatch, caplog
):
    output_module = fake_ext_zone_requests_that_always_fail
    clock = _FakeClock()
    monkeypatch.setattr(output_module.time, "time", clock.time)

    pub = ZoneHttpPublisher(url="http://example.invalid/x", timeout=0.1)
    caplog.set_level(logging.WARNING, logger=output_module.__name__)
    pub._post({"a": 1})
    pub._post({"a": 2})  # 節流窗口內，不應該再警告

    assert len(caplog.records) == 1
