"""把 MQTT 訊息接到 ReportService，並把結果發回去。

``Router`` 是純邏輯（topic＋payload bytes → 要發的訊息），``test_runner.py`` 直接灌假訊息；
``run()`` 才建立真正的 MQTT 連線、裝中斷訊號、block 到 Ctrl+C。
"""

from __future__ import annotations

import json
import logging
import signal
import threading
from typing import Any, Callable, Optional

from iot.voice.config import VoiceReportConfig as _C
from iot.voice.service import ReportService

_log = logging.getLogger(__name__)

PublishFn = Callable[[str, dict, bool], None]   # (topic, payload, retain)


class Router:
    def __init__(self, service: ReportService, publish: PublishFn):
        self._service = service
        self._publish = publish

    def subscribe_topics(self) -> list[str]:
        return [_C.topic("utterance"), _C.topic("reports/delete"), _C.topic("reports/get")]

    def publish_recent(self) -> None:
        self._publish(_C.topic("reports/recent"), self._service.recent_snapshot(), True)

    def handle(self, topic: str, payload: bytes | str) -> None:
        data = _json_object(payload)
        if topic == _C.topic("utterance"):
            if data is None:
                _log.warning("utterance 不是 JSON 物件，略過")
                return
            result = self._service.handle_utterance(data)
            self._publish(_C.topic("result"), result, False)
            if result.get("reports"):
                self.publish_recent()
        elif topic == _C.topic("reports/delete"):
            if data is not None and self._service.handle_delete(data):
                self.publish_recent()
        elif topic == _C.topic("reports/get"):
            self.publish_recent()


def _json_object(payload: bytes | str) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(payload)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def run() -> None:
    logging.basicConfig(level=getattr(logging, _C.LOG_LEVEL.upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _log.info("飼主語音紀錄服務啟動中……（broker %s:%s，topic %s/*，DB %s）",
              _C.MQTT_HOST, _C.MQTT_PORT, _C.TOPIC_PREFIX, _C.DB_PATH)
    try:
        from iot.voice.transport import MqttTransport
    except ImportError:
        _log.error("缺少 paho-mqtt，無法啟動。請 `pip install paho-mqtt`。主系統不受影響。")
        return

    box: dict[str, Any] = {}
    router = Router(ReportService(), lambda t, p, r: box["t"].publish(t, p, retain=r))
    transport = MqttTransport(router.handle, router.subscribe_topics(), on_connected=router.publish_recent)
    box["t"] = transport
    transport.start()

    stop = threading.Event()

    def _stop(signum, frame):
        _log.info("收到中斷訊號（%s），關閉中……", signum)
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _stop)
    try:
        while not stop.wait(1.0):
            pass
    finally:
        transport.stop()
        from iot.voice import store
        store.close_connection()
        _log.info("飼主語音紀錄服務已停止。")
