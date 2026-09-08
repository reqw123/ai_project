"""把「MQTT topic + 原始 payload bytes」分派到對應的 parser。

topic 格式：``<TOPIC_PREFIX>/<kind>/<source_id>``，例如 ``cat/iot/env/living_room``。
未知 kind / JSON 壞掉 / parser 失敗都只記 log 並回傳 ``None``，絕不往外拋
（fail-safe：一顆壞掉的感測器不該讓整個 hub 停擺）。
"""

from __future__ import annotations

import json
import logging

from iot.config import IotHubConfig as _C
from iot.sensors.base import ParseError, Reading
from iot.sensors.bodytemp import BodyTempParser
from iot.sensors.environment import EnvironmentParser
from iot.sensors.motion import MotionParser
from iot.sensors.weight import WeightParser

_log = logging.getLogger(__name__)

_PARSERS = {
    "environment": EnvironmentParser(),
    "motion": MotionParser(),
    "weight": WeightParser(),
    "bodytemp": BodyTempParser(),
}


class SensorRouter:
    def __init__(self, topic_prefix: str | None = None, topic_map: dict | None = None):
        self._prefix = (topic_prefix or _C.TOPIC_PREFIX).strip("/")
        self._topic_map = dict(topic_map or _C.TOPIC_MAP)

    def _split_topic(self, topic: str) -> tuple[str, str] | None:
        topic = topic.strip("/")
        if not topic.startswith(self._prefix + "/"):
            return None
        rest = topic[len(self._prefix) + 1 :]
        parts = rest.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        return parts[0], parts[1]  # (kind, source_id)

    def route(self, topic: str, payload: bytes | str) -> Reading | None:
        """回傳解析後的讀數；任何問題都記 log 後回 ``None``。"""
        split = self._split_topic(topic)
        if split is None:
            _log.debug("略過非本 hub 或格式不符的 topic: %s", topic)
            return None
        kind, source_id = split

        parser_name = self._topic_map.get(kind)
        parser = _PARSERS.get(parser_name) if parser_name else None
        if parser is None:
            _log.debug("未知感測器類型 %r（topic=%s），略過", kind, topic)
            return None

        try:
            data = json.loads(payload)
        except (ValueError, TypeError) as exc:
            _log.warning("topic=%s payload 不是合法 JSON（略過）: %s", topic, exc)
            return None
        if not isinstance(data, dict):
            _log.warning("topic=%s payload 不是 JSON 物件（略過）: %r", topic, data)
            return None

        try:
            return parser.parse(source_id, data)
        except ParseError as exc:
            _log.warning("topic=%s 解析失敗（略過該筆）: %s", topic, exc)
            return None
        except Exception:  # noqa: BLE001 — fail-safe，parser 不該炸掉 hub
            _log.exception("topic=%s parser 發生未預期例外（略過該筆）", topic)
            return None
