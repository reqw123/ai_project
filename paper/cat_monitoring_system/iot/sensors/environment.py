"""環境感測（溫濕度 / 氣體 / 光照）payload 解析 + 範圍檢查 + EWMA 平滑。

原始值原封不動交給 store 落地；EWMA 平滑後的值只用於發回 MQTT 的 derived
topic（給 Node-RED Dashboard 畫比較不抖的曲線）。
"""

from __future__ import annotations

from typing import Any

from iot.config import IotHubConfig as _C
from iot.sensors.base import (
    EnvironmentReading,
    ParseError,
    check_range,
    coerce_float,
    coerce_ts,
)


class EnvironmentParser:
    kind = "env"

    def parse(self, source_id: str, payload: dict[str, Any]) -> EnvironmentReading:
        temp = check_range(
            coerce_float(payload, "temp_c", "temperature", "temp", "t"),
            *_C.ENV_TEMP_VALID_RANGE,
            label="temp_c",
        )
        humidity = check_range(
            coerce_float(payload, "humidity_pct", "humidity", "hum", "rh", "h"),
            *_C.ENV_HUMIDITY_VALID_RANGE,
            label="humidity_pct",
        )
        gas = check_range(
            coerce_float(payload, "gas_ppm", "gas", "co2", "voc", "ppm"),
            *_C.ENV_GAS_VALID_RANGE,
            label="gas_ppm",
        )
        lux = check_range(
            coerce_float(payload, "lux", "light", "illuminance"),
            *_C.ENV_LUX_VALID_RANGE,
            label="lux",
        )
        if temp is None and humidity is None and gas is None and lux is None:
            raise ParseError("environment payload 沒有任何可辨識的量測欄位")
        return EnvironmentReading(
            source_id=source_id,
            temp_c=temp,
            humidity_pct=humidity,
            gas_ppm=gas,
            lux=lux,
            ts=coerce_ts(payload),
        )


class EnvironmentSmoother:
    """對每個 source_id 各維護一組 EWMA。純函式狀態機，方便單元測試。"""

    _FIELDS = ("temp_c", "humidity_pct", "gas_ppm", "lux")

    def __init__(self, alpha: float | None = None):
        self._alpha = _C.ENV_EWMA_ALPHA if alpha is None else alpha
        self._state: dict[str, dict[str, float]] = {}

    def smooth(self, reading: EnvironmentReading) -> dict[str, float | None]:
        prev = self._state.setdefault(reading.source_id, {})
        out: dict[str, float | None] = {}
        for f in self._FIELDS:
            raw = getattr(reading, f)
            if raw is None:
                out[f] = prev.get(f)
                continue
            if f in prev:
                prev[f] = self._alpha * raw + (1.0 - self._alpha) * prev[f]
            else:
                prev[f] = raw
            out[f] = round(prev[f], 3)
        return out
