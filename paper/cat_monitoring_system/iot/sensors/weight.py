"""HX711 重量感測 payload 解析 + 進食 / 加料 / 飲水事件偵測。

事件偵測是一個 per-scale 的小狀態機，用「基準重量（去皮）」加上「變化需先
穩定下來才承認」的邏輯，濾掉貓走過秤台造成的瞬間晃動與感測器漂移。

所有帶時間的方法都可注入 ``now``（epoch 秒），方便單元測試不靠 sleep。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from iot.config import IotHubConfig as _C
from iot.sensors.base import (
    ParseError,
    WeightReading,
    check_range,
    coerce_float,
    coerce_ts,
)


class WeightParser:
    kind = "weight"

    def parse(self, source_id: str, payload: dict[str, Any]) -> WeightReading:
        grams = coerce_float(payload, "grams", "weight", "g", "value", "mass")
        if grams is None:
            raise ParseError(f"weight payload 沒有重量欄位: {payload!r}")
        grams = check_range(grams, *_C.WEIGHT_VALID_RANGE, label="grams")
        return WeightReading(
            source_id=source_id, grams=float(grams), ts=coerce_ts(payload)
        )


@dataclass(frozen=True)
class WeightChangeEvent:
    scale_id: str
    from_g: float
    to_g: float
    delta_g: float  # to_g - from_g（負 = 重量減少 = 進食 / 喝水）
    direction: str  # "decrease" | "increase"
    ts: float
    kind: str = "weight_change"

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_id": self.scale_id,
            "from_g": round(self.from_g, 2),
            "to_g": round(self.to_g, 2),
            "delta_g": round(self.delta_g, 2),
            "direction": self.direction,
            "ts": self.ts,
        }


@dataclass
class _ScaleState:
    baseline_g: float
    pending_g: float  # 目前正在觀察是否穩定下來的候選值
    pending_since: float  # pending_g 維持穩定的起始時間
    last_g: float
    last_ts: float


class WeightEventDetector:
    """對每個 scale_id 各維護一組去皮基準，偵測「穩定後」的重量變化事件。"""

    def __init__(
        self,
        event_delta_g: float | None = None,
        settle_sec: float | None = None,
    ):
        self._delta = (
            _C.WEIGHT_EVENT_DELTA_G if event_delta_g is None else event_delta_g
        )
        self._settle = _C.WEIGHT_SETTLE_SEC if settle_sec is None else settle_sec
        self._states: dict[str, _ScaleState] = {}

    def update(self, reading: WeightReading) -> WeightChangeEvent | None:
        """吃一筆讀數，回傳偵測到的變化事件或 ``None``。"""
        scale_id = reading.source_id
        g = reading.grams
        ts = reading.ts
        st = self._states.get(scale_id)
        if st is None:
            self._states[scale_id] = _ScaleState(
                baseline_g=g, pending_g=g, pending_since=ts, last_g=g, last_ts=ts
            )
            return None

        # 目前這筆跟「候選值」差多少——差太多代表還在變動，重設候選與計時。
        if abs(g - st.pending_g) > self._delta:
            st.pending_g = g
            st.pending_since = ts
            st.last_g, st.last_ts = g, ts
            return None

        st.last_g, st.last_ts = g, ts

        # 候選值已穩定夠久，跟基準比較。
        if ts - st.pending_since < self._settle:
            return None

        diff = st.pending_g - st.baseline_g
        if abs(diff) < self._delta:
            # 只是漂移，靜靜地把基準拉到新的穩定值，不算事件。
            st.baseline_g = st.pending_g
            return None

        event = WeightChangeEvent(
            scale_id=scale_id,
            from_g=st.baseline_g,
            to_g=st.pending_g,
            delta_g=diff,
            direction="decrease" if diff < 0 else "increase",
            ts=ts,
        )
        st.baseline_g = st.pending_g
        return event
