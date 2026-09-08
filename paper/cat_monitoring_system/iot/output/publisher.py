"""把正規化讀數（derived）與告警發回 MQTT。

設計成只依賴一個 ``publish_fn(topic: str, payload: dict) -> None`` callable，
不直接綁 paho——runner 注入真正的 MQTT client.publish 包裝，測試注入假的並
檢查呼叫。任何 publish 例外都在這裡吞掉（fail-safe），最多記一次節流警告。

derived 訊息對每個 ``(kind, source_id)`` 做最小間隔節流
（``DERIVED_PUBLISH_MIN_INTERVAL_SEC``）；告警不節流（AlertEngine 已自帶冷卻）。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from iot.config import IotHubConfig as _C
from iot.output.alert_engine import Alert

_log = logging.getLogger(__name__)

PublishFn = Callable[[str, str], Any]

_WARN_INTERVAL_SEC = 30.0


class MqttPublisher:
    def __init__(
        self,
        publish_fn: PublishFn,
        derived_min_interval_sec: float | None = None,
    ):
        self._publish_fn = publish_fn
        self._min_interval = (
            _C.DERIVED_PUBLISH_MIN_INTERVAL_SEC
            if derived_min_interval_sec is None
            else derived_min_interval_sec
        )
        self._last_derived: dict[tuple[str, str], float] = {}
        self._last_warn = 0.0

    def _send(self, topic: str, payload: dict[str, Any]) -> None:
        try:
            self._publish_fn(topic, json.dumps(payload, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001 — fail-safe
            now = time.time()
            if now - self._last_warn >= _WARN_INTERVAL_SEC:
                self._last_warn = now
                _log.warning(
                    "MQTT publish 失敗（topic=%s）：%s"
                    "（此類錯誤 %d 秒內只警告一次）",
                    topic, exc, int(_WARN_INTERVAL_SEC),
                )

    def publish_derived(
        self, kind: str, source_id: str, payload: dict[str, Any], now: float | None = None
    ) -> bool:
        """回傳是否真的送出（False = 被節流）。"""
        now = time.time() if now is None else now
        key = (kind, source_id)
        last = self._last_derived.get(key)
        if last is not None and now - last < self._min_interval:
            return False
        self._last_derived[key] = now
        self._send(_C.derived_topic(kind), payload)
        return True

    def publish_event(self, kind: str, payload: dict[str, Any]) -> None:
        """發一筆 derived topic 訊息但**不**節流——給「進食事件」這種低頻但重要的訊息用。"""
        self._send(_C.derived_topic(kind), payload)

    def publish_alert(self, alert: Alert) -> None:
        self._send(_C.alert_topic(), alert.to_payload())
