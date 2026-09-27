"""「已連上 broker，但一直沒收到感測資料」警告。

MQTT 的 hub→broker、ESP32→broker 是兩條獨立的連線：日誌寫「已連上 MQTT broker」只代表前者通。
ESP32 連錯 WiFi、韌體裡的 broker IP 寫錯時，hub 只會安靜地等、不會報錯，看起來像正常在跑
（2026-09-27 實際踩過：broker IP 是電腦 Wi-Fi 網卡的位址，Wi-Fi 沒連上時 ESP32 的資料根本進不來）。

規則（09-27 起跟其他異常一致：「多久算異常」和「多久印一次」分開）：
* 連上 broker 後 ``first_check_sec``（預設 5 秒）先檢查一次：這段期間沒收到的種類 → 警告；
* 之後某種感測器超過 ``timeout_sec``（預設 60 秒）沒資料 → 警告；
* 異常沒解除時每 ``repeat_sec``（預設 10 秒）重印一次（標「持續中」，0＝只印一次）；收到資料 → 一行「恢復」。
斷線期間不檢查（斷線本身已經有警告），重連後從頭算。

純邏輯、時間由呼叫端傳入（方便測試）；paho 的執行緒和 runner 主迴圈都會呼叫，所以用 lock。
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Iterable, Optional, Union

from iot.repeat_limiter import RepeatLimiter

_HINT = ("ESP32 沒連上 broker（檢查 WiFi／韌體裡的 broker IP），或 ESP32 在線但感測器讀不到"
         "（設定視窗 {kind} 卡片有顯示 IP 的話，請檢查感測器接線）")


class NoDataWatchdog:
    def __init__(self, kinds: Iterable[str], topic_prefix: str, broker: str,
                 first_check_sec: float = 5.0, timeout_sec: float = 60.0,
                 repeat_sec: Union[float, Callable[[], float]] = 10.0):
        self._kinds = list(kinds)
        self._prefix = topic_prefix.rstrip("/") + "/"
        self._broker = broker
        self._first = first_check_sec
        self._timeout = timeout_sec
        self._lock = threading.Lock()
        self._connected_at: Optional[float] = None    # None＝沒連上 broker，不檢查
        self._first_done = False
        self._last_seen: dict[str, float] = {}        # 種類 → 最近一次收到資料的時間
        self._warned = RepeatLimiter(repeat_sec)      # 警告中的種類（共用的「持續中重印」節流）

    def kind_of(self, topic: str) -> Optional[str]:
        if not topic.startswith(self._prefix):
            return None
        kind = topic[len(self._prefix):].split("/", 1)[0]
        return kind if kind in self._kinds else None

    def on_connected(self, now: float) -> None:
        with self._lock:
            self._connected_at = now
            self._first_done = False

    def on_disconnected(self) -> None:
        with self._lock:
            self._connected_at = None

    def on_message(self, topic: str, now: Optional[float] = None) -> Optional[str]:
        """記下收到資料；如果這個種類正在警告中，回傳「恢復」訊息。"""
        kind = self.kind_of(topic)
        if kind is None:
            return None
        with self._lock:
            self._last_seen[kind] = time.monotonic() if now is None else now
            if self._warned.clear(kind):
                return f"收到 {kind} 資料了（先前警告過沒資料），ESP32 已經連上 broker"
        return None

    def tick(self, now: float) -> list[str]:
        """回傳這一刻該印的警告（每種一行）；沒連上 broker 或都正常時回傳空清單。"""
        out: list[str] = []
        with self._lock:
            if self._connected_at is None:
                return out
            first_now = not self._first_done and now >= self._connected_at + self._first
            if first_now:
                self._first_done = True
            for k in self._kinds:
                since = max(self._last_seen.get(k, self._connected_at), self._connected_at)
                silent = now - since
                got_since_connect = self._last_seen.get(k, -1e18) >= self._connected_at
                abnormal = (first_now and not got_since_connect) \
                    or k in self._warned or (self._first_done and silent >= self._timeout)
                if not abnormal:
                    continue
                again = self._warned.hit(k, now)
                if again is None:
                    continue
                out.append(f"{again}已連上 broker {self._broker}，但 {k} 已經 {silent:.0f} 秒沒有資料："
                           + _HINT.format(kind=k))
        return out
