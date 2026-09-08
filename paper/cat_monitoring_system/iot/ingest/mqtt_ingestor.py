"""paho-mqtt 訂閱者薄殼。

* 相容 paho-mqtt 1.x 與 2.x（callback API 版本自動偵測）。
* ``connect_async`` + ``loop_start`` + 內建自動重連——broker 沒開就啟動也不會
  crash，會持續在背景重試（比照 ext_body_zones/output.py 的 ZoneMqttPublisher）。
* 收到訊息只做一件事：呼叫注入的 ``on_message(topic, payload_bytes)``；任何
  例外都吞在這裡，不讓它中斷 paho 的網路迴圈。
* 同一個 client 也提供 ``publish()`` 給 output.publisher 共用，不用開兩條連線。

``paho`` 是延遲匯入的——沒安裝時 ``MqttIngestor`` 無法建立，但 import 這個模組
本身不會失敗（測試不需要 paho）。
"""

from __future__ import annotations

import logging
from typing import Callable

from iot.config import IotHubConfig as _C

_log = logging.getLogger(__name__)

OnMessage = Callable[[str, bytes], None]


def _make_client(client_id: str):
    """建立 paho Client，相容 1.x / 2.x。回傳 (client, is_v2)。"""
    import paho.mqtt.client as mqtt

    try:
        # paho-mqtt >= 2.0
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=True
        )
        return client, True
    except (AttributeError, TypeError):
        # paho-mqtt 1.x
        client = mqtt.Client(client_id=client_id, clean_session=True)
        return client, False


class MqttIngestor:
    def __init__(self, on_message: OnMessage, subscribe_topics: list[str] | None = None):
        self._on_message = on_message
        self._topics = subscribe_topics or _C.subscribe_topics()
        self._client, self._is_v2 = _make_client(_C.MQTT_CLIENT_ID)
        self._started = False

        if _C.MQTT_USERNAME:
            self._client.username_pw_set(
                _C.MQTT_USERNAME, _C.MQTT_PASSWORD or None
            )
        self._client.reconnect_delay_set(
            min_delay=_C.MQTT_RECONNECT_MIN_SEC,
            max_delay=_C.MQTT_RECONNECT_MAX_SEC,
        )
        self._client.on_connect = self._handle_connect
        self._client.on_disconnect = self._handle_disconnect
        self._client.on_message = self._handle_message

    # ── paho callbacks（用 *args 吸收 1.x / 2.x 的簽章差異）──────────────
    def _handle_connect(self, client, userdata, *args):
        # 1.x: (flags, rc) ; 2.x: (flags, reason_code, properties)
        rc = args[1] if len(args) >= 2 else None
        _log.info("已連上 MQTT broker %s:%s（rc=%s），訂閱 %s",
                  _C.MQTT_HOST, _C.MQTT_PORT, rc, self._topics)
        for topic in self._topics:
            client.subscribe(topic, qos=0)

    def _handle_disconnect(self, client, userdata, *args):
        _log.warning("與 MQTT broker 中斷連線（%s），paho 會自動重連", args)

    def _handle_message(self, client, userdata, msg):
        try:
            self._on_message(msg.topic, msg.payload)
        except Exception:  # noqa: BLE001 — 絕不讓它中斷 paho 網路迴圈
            _log.exception("處理 MQTT 訊息時發生未預期例外（topic=%s）", msg.topic)

    # ── 生命週期 ─────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        # connect_async：broker 沒開也不會拋，loop_start 的背景執行緒會持續重試。
        self._client.connect_async(
            _C.MQTT_HOST, _C.MQTT_PORT, keepalive=_C.MQTT_KEEPALIVE
        )
        self._client.loop_start()
        _log.info("MqttIngestor 啟動（broker=%s:%s）", _C.MQTT_HOST, _C.MQTT_PORT)

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:  # noqa: BLE001
            pass

    def publish(self, topic: str, payload: str) -> None:
        """給 output.publisher 共用同一條連線。未連線時 paho 會排隊或丟棄（qos=0）。"""
        self._client.publish(topic, payload, qos=0)
