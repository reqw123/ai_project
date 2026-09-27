"""把「MQTT topic + 原始 payload bytes」分派到對應的 parser。

topic 格式：``<TOPIC_PREFIX>/<kind>/<source_id>``，例如 ``cat/iot/env/living_room``。
未知 kind / JSON 壞掉 / parser 失敗都只記 log 並回傳 ``None``，絕不往外拋
（fail-safe：一顆壞掉的感測器不該讓整個 hub 停擺）。
"""

from __future__ import annotations

import json
import logging
import re
import time

from iot.config import IotHubConfig as _C
from iot.repeat_limiter import RepeatLimiter
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


# JSON 值位置上的 nan／inf（Arduino String(NAN) 的輸出），換成 null
_NAN_TOKEN = re.compile(r"([:\[,]\s*)-?(?:nan|inf(?:inity)?)(?=\s*[,}\]])", re.IGNORECASE)


class SensorRouter:
    def __init__(self, topic_prefix: str | None = None, topic_map: dict | None = None):
        self._prefix = (topic_prefix or _C.TOPIC_PREFIX).strip("/")
        self._topic_map = dict(topic_map or _C.TOPIC_MAP)
        # nan 警告：同一 topic 每 WARN_REPEAT_SEC 秒印一次（外出包每秒一筆，不能每筆都印）；0＝每個 topic 只印一次
        self._nan_log = RepeatLimiter(lambda: _C.WARN_REPEAT_SEC)
        # 最近一次 route() 的 payload 有 nan 時＝(kind, source_id)。讀數因此不能用（必填欄位是 nan）時，
        # runner 靠它告訴健康監測「這台在線、只是讀值失敗」，不會被當成離線，欄位消失照樣會報
        self.last_nan_node: tuple[str, str] | None = None

    def _warn_nan(self, topic: str, text: str) -> None:
        again = self._nan_log.hit(topic, time.monotonic())
        if again is None:
            return
        _log.warning(again + "topic=%s 有欄位是 nan（感測器讀值失敗，多半是沒接好），已當作沒有值、其他欄位照收：%s",
                     topic, text[:120])

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
        self.last_nan_node = None
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

        text = payload.decode("utf-8", "replace") if isinstance(payload, (bytes, bytearray)) else str(payload)
        constants: list[str] = []
        try:   # NaN／Infinity（Python 認得的寫法）→ 沒有值
            data = json.loads(payload, parse_constant=lambda c: constants.append(c))
        except (ValueError, TypeError) as exc:
            # Arduino 的 String(NAN) 是小寫「nan」（不是合法 JSON）：感測器讀值失敗（例如外出包 MLX90614 沒接）。
            # 以前整筆丟掉，連同一筆裡正常的欄位（DHT11 濕度）也沒了；改成把 nan 當作沒有值、其他欄位照收。
            fixed, n = _NAN_TOKEN.subn(r"\1null", text)
            try:
                data = json.loads(fixed) if n else None
            except ValueError:
                data = None
            if data is None:
                _log.warning("topic=%s payload 不是合法 JSON（略過）: %s", topic, exc)
                return None
            constants.append("nan")
        if constants:
            self.last_nan_node = (kind, source_id)
            self._warn_nan(topic, text)
        if not isinstance(data, dict):
            _log.warning("topic=%s payload 不是 JSON 物件（略過）: %r", topic, data)
            return None

        try:
            return parser.parse(source_id, data)
        except ParseError as exc:
            if self.last_nan_node:   # 必填欄位是 nan：上面的 nan 警告已經有節流地報了，這裡不再每筆都印
                _log.debug("topic=%s 含 nan、解析失敗（略過該筆）: %s", topic, exc)
            else:
                _log.warning("topic=%s 解析失敗（略過該筆）: %s", topic, exc)
            return None
        except Exception:  # noqa: BLE001 — fail-safe，parser 不該炸掉 hub
            _log.exception("topic=%s parser 發生未預期例外（略過該筆）", topic)
            return None
