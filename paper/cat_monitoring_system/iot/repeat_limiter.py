"""「異常持續中每隔幾秒重印一次」的共用節流（沒收到資料、欄位消失、極端值、nan、門檻告警都用這一份）。

規則：同一個 key 第一次一定印；之後 ``repeat_sec`` > 0 時每 ``repeat_sec`` 秒再印一次，0＝只印一次。
異常解除時呼叫 ``clear(key)``，回傳它是不是警告中（用來決定要不要印「恢復」），下次異常又從「第一次」算。

``repeat_sec`` 可以是數字或「回傳數字的函式」（runner／router 傳 ``lambda: _C.WARN_REPEAT_SEC``，測試改 config 立即生效）。
會被 paho 執行緒和 runner 主迴圈同時呼叫，內部自己加鎖。
"""

from __future__ import annotations

import threading
from typing import Callable, Hashable, Optional, Union

ONGOING = "（持續中）"


class RepeatLimiter:
    def __init__(self, repeat_sec: Union[float, Callable[[], float]] = 10.0):
        self._repeat = repeat_sec
        self._last: dict[Hashable, float] = {}
        self._lock = threading.Lock()

    def _repeat_sec(self) -> float:
        return float(self._repeat() if callable(self._repeat) else self._repeat)

    def hit(self, key: Hashable, now: float) -> Optional[str]:
        """該印了嗎？不該印回傳 None；第一次回傳 ""；重印回傳「（持續中）」（直接接在訊息前面）。"""
        with self._lock:
            last = self._last.get(key)
            if last is not None:
                repeat = self._repeat_sec()
                if repeat <= 0 or now - last < repeat:
                    return None
            self._last[key] = now
            return "" if last is None else ONGOING

    def clear(self, key: Hashable) -> bool:
        """異常解除；回傳它原本是不是警告中（是的話呼叫端印一行「恢復」）。"""
        with self._lock:
            return self._last.pop(key, None) is not None

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._last
