"""paho-mqtt 薄殼（本服務自己的一份，不共用 ``iot.ingest``——那個綁著感測器 hub 的設定）。

* 相容 paho-mqtt 1.x／2.x；``connect_async``＋``loop_start``：broker 沒開也不會 crash，背景重連。
* 連上時送 ``status`` = online（retained），並設 MQTT 遺囑 offline（斷線／當掉時 broker 代發），
  Node-RED 儀表板靠它顯示「紀錄服務：在線／離線」。
* 收到訊息只呼叫注入的 ``on_message(topic, payload_bytes)``，例外吞掉不中斷網路迴圈。
* ``paho`` 延遲匯入：測試不需要安裝。
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from iot.voice.config import VoiceReportConfig as _C

_log = logging.getLogger(__name__)

OnMessage = Callable[[str, bytes], None]


class MqttTransport:
    def __init__(self, on_message: OnMessage, subscribe_topics: list[str], on_connected: Callable[[], None] | None = None):
        import paho.mqtt.client as mqtt

        try:
            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=_C.MQTT_CLIENT_ID, clean_session=True)
        except (AttributeError, TypeError):
            self._client = mqtt.Client(client_id=_C.MQTT_CLIENT_ID, clean_session=True)
        self._on_message = on_message
        self._topics = subscribe_topics
        self._on_connected = on_connected
        if _C.MQTT_USERNAME:
            self._client.username_pw_set(_C.MQTT_USERNAME, _C.MQTT_PASSWORD or None)
        self._client.will_set(_C.topic("status"), json.dumps({"online": False}), qos=1, retain=True)
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client.on_connect = self._handle_connect
        self._client.on_disconnect = lambda *a: _log.warning("與 MQTT broker 中斷連線，paho 會自動重連")
        self._client.on_message = self._handle_message

    def _handle_connect(self, client, userdata, *args):
        _log.info("已連上 MQTT broker %s:%s，訂閱 %s", _C.MQTT_HOST, _C.MQTT_PORT, self._topics)
        for topic in self._topics:
            client.subscribe(topic, qos=1)
        self.publish(_C.topic("status"), {"online": True}, retain=True)
        if self._on_connected:
            try:
                self._on_connected()
            except Exception:  # noqa: BLE001
                _log.exception("連線後的初始化失敗")

    def _handle_message(self, client, userdata, msg):
        try:
            self._on_message(msg.topic, msg.payload)
        except Exception:  # noqa: BLE001 — 絕不讓它中斷 paho 網路迴圈
            _log.exception("處理 MQTT 訊息時發生未預期例外（topic=%s）", msg.topic)

    def start(self) -> None:
        self._client.connect_async(_C.MQTT_HOST, _C.MQTT_PORT, keepalive=_C.MQTT_KEEPALIVE)
        self._client.loop_start()

    def stop(self) -> None:
        try:
            self.publish(_C.topic("status"), {"online": False}, retain=True)
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:  # noqa: BLE001
            pass

    def publish(self, topic: str, payload: dict, retain: bool = False) -> None:
        self._client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=1, retain=retain)
