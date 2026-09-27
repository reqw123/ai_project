"""門檻告警：環境數值超標、疑似長時間不進食。

每個 alert key 有獨立冷卻時間（``ALERT_COOLDOWN_SEC``），避免數值在門檻附近
抖動時洗版。冷卻狀態純記憶體，行程重啟後重置（可接受——重啟後最多多發一次）。

冷卻只管「發出去」（寫 DB、MQTT → Node-RED → Discord）。09-27 起另外記下「超標但在冷卻中」與「恢復正常」，
由 runner 用 ``drain()`` 取走、只寫日誌（超標期間持續提醒），Discord 不會因此洗版。

「恢復正常」的判斷（09-27 審查修正）：每次檢查記下「這次**有評估到**的條件」（欄位有值才算），
之前超標、這次有評估到而且沒超標的才算恢復。所以：
* 欄位沒值（例：MLX90614 送 nan → 溫度是 None）不會被當成恢復——溫度是未知，不是正常；
* 多個食盆一起檢查時，每個食盆各自判斷（以前第二個之後的食盆會被誤報恢復）。
paho 執行緒（環境、體表溫度）和週期檢查執行緒（食盆）會同時呼叫，所有狀態都在 lock 裡。

時間一律由呼叫端傳 ``now``（epoch 秒），方便測試。
"""

from __future__ import annotations

import threading
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
    label: str = ""       # 條件的中文名稱（「環境溫度過高」），「恢復正常」那行用
    source_id: str = ""

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
        self._lock = threading.RLock()
        self._last_emit: dict[str, float] = {}
        self._active: dict[str, str] = {}               # 目前超標中的 key → 「恢復正常」要印的文字
        self._suppressed: list[tuple[str, str]] = []    # (key, 訊息)：超標但在冷卻中
        self._recovered: list[tuple[str, str]] = []     # (key, 文字)：從超標回到正常

    # ── 一次檢查：評估了哪些條件、哪些超標 ─────────────────────────────────
    class _Check:
        def __init__(self):
            self.evaluated: set[str] = set()
            self.fired: dict[str, str] = {}

    def _settle(self, chk: "AlertEngine._Check") -> None:
        """之前超標、這次有評估到但沒超標 → 恢復；這次超標的記成超標中。呼叫端持有 lock。"""
        for key in sorted(chk.evaluated):
            if key in self._active and key not in chk.fired:
                self._recovered.append((key, self._active.pop(key)))
        self._active.update(chk.fired)

    def drain(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """取走（超標但冷卻中的 (key, 訊息)、恢復正常的 (key, 文字)），取完清空。"""
        with self._lock:
            out = (self._suppressed, self._recovered)
            self._suppressed, self._recovered = [], []
            return out

    # ── 冷卻控制 ──────────────────────────────────────────────────────────
    def _passes_cooldown(self, key: str, now: float) -> bool:
        last = self._last_emit.get(key)
        if last is not None and now - last < self._cooldown:
            return False
        self._last_emit[key] = now
        return True

    def _emit(
        self,
        chk: "AlertEngine._Check",
        key: str,
        severity: str,
        label: str,
        source_id: str,
        message: str,
        now: float,
        value: float | None = None,
        threshold: float | None = None,
    ) -> Alert | None:
        chk.fired[key] = f"{label}（{source_id}）"
        if not self._passes_cooldown(key, now):
            self._suppressed.append((key, message))
            return None
        return Alert(key=key, severity=severity, message=message, ts=now, value=value, threshold=threshold,
                     label=label, source_id=source_id)

    def _range(self, chk, alerts, sid, now, value, name, unit, fmt, lo, hi, kind, lo_word, hi_word):
        """value 有值才評估（沒值＝未知，不算恢復）；超過 hi／低於 lo 各一個 key。"""
        if value is None:
            return
        high, low = f"{kind}_high.{sid}", f"{kind}_low.{sid}"
        chk.evaluated |= {high, low}
        if hi is not None and value > hi:
            a = self._emit(chk, high, "warning", f"{name}{hi_word}", sid,
                           f"{name}{hi_word}：{value:{fmt}}{unit}（上限 {hi:{fmt}}{unit}）", now, value, hi)
        elif lo is not None and value < lo:
            a = self._emit(chk, low, "warning", f"{name}{lo_word}", sid,
                           f"{name}{lo_word}：{value:{fmt}}{unit}（下限 {lo:{fmt}}{unit}）", now, value, lo)
        else:
            a = None
        if a:
            alerts.append(a)

    # ── 環境門檻 ──────────────────────────────────────────────────────────
    def check_environment(
        self, reading: EnvironmentReading, now: float | None = None
    ) -> list[Alert]:
        now = time.time() if now is None else now
        alerts: list[Alert] = []
        sid = reading.source_id
        with self._lock:
            chk = self._Check()
            self._range(chk, alerts, sid, now, reading.temp_c, "環境溫度", "°C", ".1f",
                        _C.TEMP_MIN_C, _C.TEMP_MAX_C, "env.temp", "過低", "過高")
            self._range(chk, alerts, sid, now, reading.humidity_pct, "環境濕度", "%", ".0f",
                        _C.HUMIDITY_MIN_PCT, _C.HUMIDITY_MAX_PCT, "env.humidity", "過低", "過高")
            g = reading.gas_ppm
            if g is not None:
                key = f"env.gas_high.{sid}"
                chk.evaluated.add(key)
                if g > _C.GAS_PPM_MAX:
                    a = self._emit(chk, key, "critical", "空氣品質異常", sid,
                                   f"空氣品質異常：{g:.0f} ppm（上限 {_C.GAS_PPM_MAX:.0f} ppm）",
                                   now, g, _C.GAS_PPM_MAX)
                    if a:
                        alerts.append(a)
            self._settle(chk)
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
        with self._lock:
            chk = self._Check()
            if s is not None:
                high, low = f"bodytemp.high.{sid}", f"bodytemp.low.{sid}"
                chk.evaluated |= {high, low}
                a = None
                if s > _C.BODYTEMP_SURFACE_MAX_C:
                    a = self._emit(chk, high, "warning", "貓體表溫度偏高", sid,
                                   f"貓體表溫度偏高：{s:.1f}°C（門檻 {_C.BODYTEMP_SURFACE_MAX_C:.1f}°C）{_note}",
                                   now, s, _C.BODYTEMP_SURFACE_MAX_C)
                elif s < _C.BODYTEMP_SURFACE_MIN_C:
                    a = self._emit(chk, low, "warning", "貓體表溫度偏低", sid,
                                   f"貓體表溫度偏低：{s:.1f}°C（門檻 {_C.BODYTEMP_SURFACE_MIN_C:.1f}°C）{_note}",
                                   now, s, _C.BODYTEMP_SURFACE_MIN_C)
                if a:
                    alerts.append(a)
            self._settle(chk)
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
        with self._lock:
            chk = self._Check()
            for scale_id, last_ts in last_feeding_ts_by_scale.items():
                key = f"feeding.silence.{scale_id}"
                chk.evaluated.add(key)
                elapsed = silence_sec + 1 if last_ts is None else now - last_ts
                if elapsed >= silence_sec:
                    hours = elapsed / 3600.0
                    detail = (
                        "從未偵測到進食" if last_ts is None
                        else f"已 {hours:.1f} 小時未偵測到進食"
                    )
                    a = self._emit(
                        chk, key, "warning", "疑似食慾不振", scale_id,
                        f"疑似食慾不振（{scale_id}）：{detail}"
                        f"（門檻 {_C.FEEDING_SILENCE_HOURS:.0f} 小時）",
                        now, round(elapsed / 3600.0, 1), _C.FEEDING_SILENCE_HOURS,
                    )
                    if a:
                        alerts.append(a)
            self._settle(chk)   # 全部食盆評估完才結算一次（以前每個食盆結算一次，第二個之後被誤報恢復）
        return alerts
