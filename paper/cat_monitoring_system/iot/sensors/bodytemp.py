"""體表溫度（MLX90614 紅外線非接觸測溫）payload 解析。

來源是 ``esp32_petbox`` 節點（衍生自舊專案 `寵物包/mqtt_all`）：MLX90614 的
「object temperature」對著貓量 → 體表溫度；「ambient temperature」→ 環境溫度。

⚠️ 體表 IR 溫度 ≠ 核心體溫，門檻與告警語意見 ``config.py`` 的說明。
"""

from __future__ import annotations

from typing import Any

from iot.config import IotHubConfig as _C
from iot.sensors.base import (
    BodyTempReading,
    ParseError,
    check_range,
    coerce_float,
    coerce_ts,
)


class BodyTempParser:
    kind = "bodytemp"

    def parse(self, source_id: str, payload: dict[str, Any]) -> BodyTempReading:
        surface = coerce_float(
            payload,
            "surface_temp_c",
            "object_temp_c",
            "object",
            "obj",
            "body_temp_c",
        )
        if surface is None:
            raise ParseError(f"bodytemp payload 沒有體表溫度欄位: {payload!r}")
        surface = check_range(
            surface, *_C.BODYTEMP_SURFACE_VALID_RANGE, label="surface_temp_c"
        )

        ambient = check_range(
            coerce_float(payload, "ambient_temp_c", "ambient", "amb"),
            *_C.ENV_TEMP_VALID_RANGE,
            label="ambient_temp_c",
        )
        return BodyTempReading(
            source_id=source_id,
            surface_temp_c=float(surface),
            ambient_temp_c=ambient,
            ts=coerce_ts(payload),
        )
