"""感測器健康：ESP32 在線、資料照送，但某一顆感測器沒接好／壞了；或同種類裡某一台離線。

「整種感測器都沒收到資料」由 ``ingest/no_data_watchdog.py`` 負責；這裡抓的是更細的情況：

* **欄位消失**：同一個節點還在送，但某個欄位超過 ``missing_sec`` 秒都沒有值
  （例：外出包 DHT11 沒接 → 韌體直接省略 humidity_pct，hub 原本完全不會發現）。
  要檢查哪些欄位＝``expected``（config 的 SENSOR_EXPECTED_FIELDS，一開始就沒接也抓得到）∪ 看過的欄位。
* **卡在極端值**：類比感測器連續 ``rail_count`` 筆都是量測範圍的極端值（腳位接地／懸空／滿格），
  例：MQ-135 讀到 0 ppm（加熱中一定有電壓）、光敏電阻滿格。
* **同種類某一台離線**（09-27 審查修正）：env 有外出包（carrier）和 esp32_room（living_room）兩台，
  一台斷電時另一台還在送，watchdog 看「env 有資料」就不會報——這裡針對那一台報。
  整種都沒在送（唯一的一台離線）時交給 watchdog，不重複報。

異常持續時每 ``repeat_sec`` 秒再印一次（標「（持續中）」，0＝只警告一次），恢復時回報一行。
純邏輯、時間由呼叫端傳入；paho 執行緒（observe／touch）和 runner 主迴圈（tick）都會呼叫，所以用 lock。
"""

from __future__ import annotations

import threading
from typing import Callable, Optional, Union

from iot.repeat_limiter import RepeatLimiter

# 欄位 → 可能沒接好的感測器（警告訊息用）
FIELD_SENSORS = {
    "temp_c": "溫度感測器（外出包是 MLX90614 的環境溫度）",
    "humidity_pct": "DHT11 溫濕度感測器",
    "gas_ppm": "MQ-135 空氣品質感測器",
    "lux": "GL5528 光敏電阻",
    "surface_temp_c": "MLX90614 紅外線測溫",
    "ambient_temp_c": "MLX90614 紅外線測溫",
}
# 各種讀數要追蹤的欄位（motion／weight 只有一個必填欄位，沒有「欄位消失」的問題，但一樣追蹤節點在不在線）
TRACKED_FIELDS = {
    "env": ("temp_c", "humidity_pct", "gas_ppm", "lux"),
    "bodytemp": ("surface_temp_c", "ambient_temp_c"),
}
# 每個節點都一定要有的欄位（必填）：從來沒收到過也要檢查。例：MLX90614 一開機就沒接，體表溫度從頭到尾都是 nan，
# 沒列在 SENSOR_EXPECTED_FIELDS、也沒「看過」——以前因此永遠不會報
REQUIRED_FIELDS = {
    "bodytemp": ("surface_temp_c",),
}


class SensorHealthMonitor:
    def __init__(self, missing_sec: float = 60.0, rail_count: int = 6,
                 rails: Optional[dict] = None, expected: Optional[dict] = None,
                 repeat_sec: Union[float, Callable[[], float]] = 10.0):
        self._missing_sec = missing_sec
        self._rail_count = max(1, int(rail_count))
        self._rails = dict(rails or {})
        self._expected = {k: tuple(v) for k, v in (expected or {}).items()}
        self._lock = threading.Lock()
        self._first_msg: dict[str, float] = {}           # 節點 → 第一次收到的時間
        self._last_msg: dict[str, float] = {}            # 節點 → 最近一次收到的時間（含讀值失敗的那筆）
        self._field_seen: dict[tuple, float] = {}        # (節點, 欄位) → 最近一次有值的時間
        self._rail_run: dict[tuple, int] = {}            # (節點, 欄位) → 連續幾筆在極端值
        # 三種異常各一個節流器（key：(節點, 欄位) 或 節點）
        self._missing = RepeatLimiter(repeat_sec)
        self._rail = RepeatLimiter(repeat_sec)
        self._offline = RepeatLimiter(repeat_sec)

    def _fields_of(self, node: str) -> set:
        kind = node.split("/", 1)[0]
        tracked = set(TRACKED_FIELDS.get(kind, ()))
        seen = {f for (n, f) in self._field_seen if n == node}
        return (set(self._expected.get(node, ())) | set(REQUIRED_FIELDS.get(kind, ())) | seen) & tracked

    def _at_rail(self, field: str, value: float) -> bool:
        lo, hi = self._rails.get(field, (None, None))
        return (lo is not None and value <= lo) or (hi is not None and value >= hi)

    def _alive(self, node: str, now: float) -> list[tuple[str, str]]:
        """節點送來東西（不管能不能用）：記時間；先前報過離線 → 恢復。呼叫端持有 lock。"""
        self._first_msg.setdefault(node, now)
        self._last_msg[node] = now
        if self._offline.clear(node):
            return [("info", f"{node} 恢復送資料了")]
        return []

    def touch(self, kind: str, source_id: str, now: float) -> list[tuple[str, str]]:
        """節點有送、但這筆讀值失敗沒辦法用（例：必填的體表溫度是 nan）：只算「在線」，欄位照樣算沒值，
        超過 missing_sec 就會報「欄位消失」（MLX90614 沒接好），不會被當成離線。"""
        with self._lock:
            return self._alive(f"{kind}/{source_id}", now)

    def observe(self, reading, now: float) -> list[tuple[str, str]]:
        """收到一筆讀數。回傳 [(等級, 訊息)]：恢復（info）、卡在極端值（warning）。"""
        node = f"{reading.kind}/{reading.source_id}"
        out: list[tuple[str, str]] = []
        with self._lock:
            out += self._alive(node, now)
            for f in TRACKED_FIELDS.get(reading.kind, ()):
                v = getattr(reading, f, None)
                if v is None:
                    continue
                key = (node, f)
                self._field_seen[key] = now
                if self._missing.clear(key):
                    out.append(("info", f"{node} 的 {f} 恢復有值了（{FIELD_SENSORS.get(f, f)}）"))
                if f not in self._rails:
                    continue
                if self._at_rail(f, v):
                    self._rail_run[key] = self._rail_run.get(key, 0) + 1
                    if self._rail_run[key] >= self._rail_count:
                        again = self._rail.hit(key, now)
                        if again is not None:
                            out.append(("warning",
                                        f"{again}{node} 的 {f} 連續 {self._rail_run[key]} 筆都是 {v:g}（量測範圍的極端值）："
                                        f"{FIELD_SENSORS.get(f, f)} 疑似沒接好或斷線"))
                else:
                    self._rail_run[key] = 0
                    if self._rail.clear(key):
                        out.append(("info", f"{node} 的 {f} 離開極端值了（{v:g}），{FIELD_SENSORS.get(f, f)} 恢復正常"))
        return out

    def tick(self, now: float) -> list[tuple[str, str]]:
        """定期呼叫：某欄位太久沒值、同種類某一台離線 → warning（異常持續時每 repeat 秒再印一次）。"""
        out: list[tuple[str, str]] = []
        with self._lock:
            kinds_alive = {n.split("/", 1)[0] for n, last in self._last_msg.items() if now - last <= self._missing_sec}
            for node, last in self._last_msg.items():
                silent = now - last
                if silent > self._missing_sec:
                    # 整台沒在送：同種類還有別台在送才由這裡報（否則 watchdog 會報「沒收到資料」，不重複）
                    if node.split("/", 1)[0] in kinds_alive:
                        again = self._offline.hit(node, now)
                        if again is not None:
                            out.append(("warning",
                                        f"{again}{node} 已經 {silent:.0f} 秒沒送資料（同種類其他節點還在送）："
                                        "這台 ESP32 可能離線或斷電"))
                    continue
                for f in sorted(self._fields_of(node)):
                    key = (node, f)
                    since = self._field_seen.get(key, self._first_msg[node])
                    if now - since >= self._missing_sec:
                        again = self._missing.hit(key, now)
                        if again is not None:
                            out.append(("warning",
                                        f"{again}{node} 還在送資料，但 {f} 已經 {now - since:.0f} 秒沒有值："
                                        f"{FIELD_SENSORS.get(f, f)} 可能沒接好或故障"))
        return out
