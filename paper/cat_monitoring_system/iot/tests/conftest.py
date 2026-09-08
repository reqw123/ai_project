"""讓 ``iot`` 能以套件根匯入（``from iot.xxx import ...``）。

比照 ``analytics/tests/conftest.py``：把 ``cat_monitoring_system/`` 插進
sys.path，pytest 不論從哪個 cwd 執行都能解析。**不需要** paho-mqtt / cv2 / torch。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_cat_monitoring_system_dir = Path(__file__).resolve().parents[2]
if str(_cat_monitoring_system_dir) not in sys.path:
    sys.path.insert(0, str(_cat_monitoring_system_dir))

from iot.storage import store  # noqa: E402


@pytest.fixture
def db_path(tmp_path) -> str:
    """每個測試各自一份 SQLite，測完關閉快取連線（Windows 上避免檔案被占用）。"""
    path = str(tmp_path / "iot_hub_test.db")
    yield path
    store.close_connection(path)


class RecordingPublisher:
    """假 MqttPublisher：記下所有送出的 (topic-kind, payload)。"""

    def __init__(self):
        self.derived: list[tuple[str, str, dict]] = []
        self.events: list[tuple[str, dict]] = []
        self.alerts: list[dict] = []

    def publish_derived(self, kind, source_id, payload, now=None):
        self.derived.append((kind, source_id, payload))
        return True

    def publish_event(self, kind, payload):
        self.events.append((kind, payload))

    def publish_alert(self, alert):
        self.alerts.append(alert.to_payload())


@pytest.fixture
def publisher() -> "RecordingPublisher":
    return RecordingPublisher()
