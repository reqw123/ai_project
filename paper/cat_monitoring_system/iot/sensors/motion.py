"""PIR（HC-SR501）移動偵測 payload 解析。

接受兩種常見韌體寫法：
  * ``{"active": true}`` / ``{"motion": 1}`` / ``{"state": "on"}``
  * ``{"event": "enter"}`` / ``{"event": "leave"}``
"""

from __future__ import annotations

from typing import Any

from iot.sensors.base import MotionEvent, ParseError, coerce_ts

_TRUE_TOKENS = {"1", "true", "on", "yes", "enter", "entered", "detected", "motion"}
_FALSE_TOKENS = {"0", "false", "off", "no", "leave", "left", "clear", "idle"}


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
    return None


class MotionParser:
    kind = "motion"

    def parse(self, source_id: str, payload: dict[str, Any]) -> MotionEvent:
        for key in ("active", "motion", "state", "event", "value", "detected"):
            if key in payload:
                result = _as_bool(payload[key])
                if result is not None:
                    return MotionEvent(
                        source_id=source_id, active=result, ts=coerce_ts(payload)
                    )
        raise ParseError(f"motion payload 無法判定移動狀態: {payload!r}")
