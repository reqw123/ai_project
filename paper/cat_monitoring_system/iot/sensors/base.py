"""共用型別：讀數 dataclass、parser 介面、例外。

所有讀數都帶 ``kind`` / ``source_id`` / ``ts``（UTC epoch 秒），方便 store 與
publisher 一致處理，不必為每種感測器各寫一套分支。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class ParseError(ValueError):
    """payload 格式錯誤或數值超出合理範圍。呼叫端應記 log 後丟棄該筆，不中斷。"""


def _now() -> float:
    return time.time()


@dataclass(frozen=True)
class EnvironmentReading:
    source_id: str
    temp_c: float | None = None
    humidity_pct: float | None = None
    gas_ppm: float | None = None
    lux: float | None = None
    ts: float = field(default_factory=_now)
    kind: str = "env"

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_id": self.source_id,
            "temp_c": self.temp_c,
            "humidity_pct": self.humidity_pct,
            "gas_ppm": self.gas_ppm,
            "lux": self.lux,
            "ts": self.ts,
        }


@dataclass(frozen=True)
class MotionEvent:
    source_id: str
    active: bool
    ts: float = field(default_factory=_now)
    kind: str = "motion"

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_id": self.source_id,
            "active": self.active,
            "ts": self.ts,
        }


@dataclass(frozen=True)
class WeightReading:
    source_id: str  # == scale_id
    grams: float
    ts: float = field(default_factory=_now)
    kind: str = "weight"

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_id": self.source_id,
            "grams": self.grams,
            "ts": self.ts,
        }


@dataclass(frozen=True)
class BodyTempReading:
    """MLX90614 紅外線非接觸測溫。``surface_temp_c`` = 對著貓量到的體表溫度
    （非核心體溫，見 config.py 的說明）；``ambient_temp_c`` = 感測器環境溫度。"""

    source_id: str
    surface_temp_c: float
    ambient_temp_c: float | None = None
    ts: float = field(default_factory=_now)
    kind: str = "bodytemp"

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_id": self.source_id,
            "surface_temp_c": self.surface_temp_c,
            "ambient_temp_c": self.ambient_temp_c,
            "ts": self.ts,
        }


Reading = EnvironmentReading | MotionEvent | WeightReading | BodyTempReading


class SensorParser(Protocol):
    """每個 parser 把「一則 MQTT payload（已解析成 dict）+ source_id」轉成一筆讀數。

    失敗時 raise :class:`ParseError`；不得回傳 ``None`` 或吞掉錯誤（由 router
    統一 try/except 記 log）。
    """

    kind: str

    def parse(self, source_id: str, payload: dict[str, Any]) -> Reading: ...


def coerce_ts(payload: dict[str, Any]) -> float:
    """payload 若帶 ``ts`` / ``timestamp``（epoch 秒）就用它，否則用收到的當下時間。

    ESP32 韌體多半不帶時間戳，但允許帶入方便測試，也讓有 RTC 的節點可自報時間。
    """
    for key in ("ts", "timestamp", "time"):
        if key in payload and payload[key] is not None:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                pass
    return _now()


def coerce_float(payload: dict[str, Any], *keys: str) -> float | None:
    """從 payload 依序找第一個存在且可轉 float 的鍵；都沒有回傳 ``None``。

    ESP32 韌體常見的鍵名不統一（``temp`` / ``temperature`` / ``t``），這個
    helper 讓 parser 可以一次列出所有別名。
    """
    for key in keys:
        if key in payload and payload[key] is not None:
            try:
                return float(payload[key])
            except (TypeError, ValueError) as exc:
                raise ParseError(f"欄位 {key!r} 無法轉成數字: {payload[key]!r}") from exc
    return None


def check_range(value: float | None, lo: float, hi: float, label: str) -> float | None:
    """``value`` 若非 ``None`` 但落在 [lo, hi] 之外就 raise ParseError。"""
    if value is None:
        return None
    if not (lo <= value <= hi):
        raise ParseError(f"{label}={value} 超出合理範圍 [{lo}, {hi}]（疑似感測器故障）")
    return value
