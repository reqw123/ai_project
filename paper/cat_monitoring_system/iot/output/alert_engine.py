"""門檻告警：環境數值超標、疑似長時間不進食。

每個 alert key 有獨立冷卻時間（``ALERT_COOLDOWN_SEC``），避免數值在門檻附近
抖動時洗版。冷卻狀態純記憶體，行程重啟後重置（可接受——重啟後最多多發一次）。

時間一律由呼叫端傳 ``now``（epoch 秒），方便測試。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from iot.config import IotHubConfig as _C
from iot.sensors.base import BodyTempReading, EnvironmentReading


@dataclass(frozen=True)
class Alert:
    key: str
    severity: str  # "warning" | "critical"
    message: str
    ts: float
    value: float | None = None
    threshold: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "severity": self.severity,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "ts": self.ts,
        }


class AlertEngine:
    def __init__(self, cooldown_sec: float | None = None):
        self._cooldown = (
            _C.ALERT_COOLDOWN_SEC if cooldown_sec is None else cooldown_sec
        )
        self._last_emit: dict[str, float] = {}

    # ── 冷卻控制 ──────────────────────────────────────────────────────────
    def _passes_cooldown(self, key: str, now: float) -> bool:
        last = self._last_emit.get(key)
        if last is not None and now - last < self._cooldown:
            return False
        self._last_emit[key] = now
        return True

    def _emit(
        self,
        key: str,
        severity: str,
        message: str,
        now: float,
        value: float | None = None,
        threshold: float | None = None,
    ) -> Alert | None:
        if not self._passes_cooldown(key, now):
            return None
        return Alert(
            key=key,
            severity=severity,
            message=message,
            ts=now,
            value=value,
            threshold=threshold,
        )

    # ── 環境門檻 ──────────────────────────────────────────────────────────
    def check_environment(
        self, reading: EnvironmentReading, now: float | None = None
    ) -> list[Alert]:
        now = time.time() if now is None else now
        alerts: list[Alert] = []
        sid = reading.source_id

        t = reading.temp_c
        if t is not None:
            if t > _C.TEMP_MAX_C:
                a = self._emit(
                    f"env.temp_high.{sid}", "warning",
                    f"環境溫度過高：{t:.1f}°C（上限 {_C.TEMP_MAX_C:.1f}°C）",
                    now, t, _C.TEMP_MAX_C,
                )
                if a:
                    alerts.append(a)
            elif t < _C.TEMP_MIN_C:
                a = self._emit(
                    f"env.temp_low.{sid}", "warning",
                    f"環境溫度過低：{t:.1f}°C（下限 {_C.TEMP_MIN_C:.1f}°C）",
                    now, t, _C.TEMP_MIN_C,
                )
                if a:
                    alerts.append(a)

        h = reading.humidity_pct
        if h is not None:
            if h > _C.HUMIDITY_MAX_PCT:
                a = self._emit(
                    f"env.humidity_high.{sid}", "warning",
                    f"環境濕度過高：{h:.0f}%（上限 {_C.HUMIDITY_MAX_PCT:.0f}%）",
                    now, h, _C.HUMIDITY_MAX_PCT,
                )
                if a:
                    alerts.append(a)
            elif h < _C.HUMIDITY_MIN_PCT:
                a = self._emit(
                    f"env.humidity_low.{sid}", "warning",
                    f"環境濕度過低：{h:.0f}%（下限 {_C.HUMIDITY_MIN_PCT:.0f}%）",
                    now, h, _C.HUMIDITY_MIN_PCT,
                )
                if a:
                    alerts.append(a)

        g = reading.gas_ppm
        if g is not None and g > _C.GAS_PPM_MAX:
            a = self._emit(
                f"env.gas_high.{sid}", "critical",
                f"空氣品質異常：{g:.0f} ppm（上限 {_C.GAS_PPM_MAX:.0f} ppm）",
                now, g, _C.GAS_PPM_MAX,
            )
            if a:
                alerts.append(a)

        return alerts

    # ── 體表溫度（初篩，非診斷）───────────────────────────────────────────
    def check_bodytemp(
        self, reading: BodyTempReading, now: float | None = None
    ) -> list[Alert]:
        now = time.time() if now is None else now
        sid = reading.source_id
        s = reading.surface_temp_c
        alerts: list[Alert] = []
        _note = "（體表 IR 溫度僅供初篩，非核心體溫，請獸醫量肛溫確認）"
        if s > _C.BODYTEMP_SURFACE_MAX_C:
            a = self._emit(
                f"bodytemp.high.{sid}", "warning",
                f"貓體表溫度偏高：{s:.1f}°C（門檻 {_C.BODYTEMP_SURFACE_MAX_C:.1f}°C）{_note}",
                now, s, _C.BODYTEMP_SURFACE_MAX_C,
            )
            if a:
                alerts.append(a)
        elif s < _C.BODYTEMP_SURFACE_MIN_C:
            a = self._emit(
                f"bodytemp.low.{sid}", "warning",
                f"貓體表溫度偏低：{s:.1f}°C（門檻 {_C.BODYTEMP_SURFACE_MIN_C:.1f}°C）{_note}",
                now, s, _C.BODYTEMP_SURFACE_MIN_C,
            )
            if a:
                alerts.append(a)
        return alerts

    # ── 時間型：疑似不進食 ────────────────────────────────────────────────
    def check_feeding_silence(
        self,
        last_feeding_ts_by_scale: dict[str, float | None],
        now: float | None = None,
    ) -> list[Alert]:
        """``last_feeding_ts_by_scale``：{scale_id: 最後一次「重量下降」事件的 ts 或 None}。

        由 runner 從 ``store.last_weight_event_ts(scale_id, "decrease")`` 組好傳進來
        （alert_engine 不直接碰 DB，維持純邏輯、好測）。
        """
        now = time.time() if now is None else now
        silence_sec = _C.FEEDING_SILENCE_HOURS * 3600.0
        alerts: list[Alert] = []
        for scale_id, last_ts in last_feeding_ts_by_scale.items():
            elapsed = silence_sec + 1 if last_ts is None else now - last_ts
            if elapsed >= silence_sec:
                hours = elapsed / 3600.0
                detail = (
                    "從未偵測到進食" if last_ts is None
                    else f"已 {hours:.1f} 小時未偵測到進食"
                )
                a = self._emit(
                    f"feeding.silence.{scale_id}", "warning",
                    f"疑似食慾不振（{scale_id}）：{detail}"
                    f"（門檻 {_C.FEEDING_SILENCE_HOURS:.0f} 小時）",
                    now, round(elapsed / 3600.0, 1), _C.FEEDING_SILENCE_HOURS,
                )
                if a:
                    alerts.append(a)
        return alerts
