"""
個體化基線儀表板的記憶體快取層。

只做一件事：記住「最後一次 /api/deviation 算出來的結果」，供
GET /api/deviation/latest 立即回傳，不用等下一次 Node-RED 觸發。
刻意不認識 analytics/ 的 dataclass 型別，只搬移 routes.py 已經序列化好
的 dict——這樣這支檔案不需要知道 baseline/deviation/fusion 內部長什麼
樣子，未來 analytics/ 的回傳格式改了也不用動這裡。
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional

_lock = threading.Lock()
_latest: Optional[dict] = None
_cached_at: Optional[str] = None


def set_latest(result: dict) -> None:
    """儲存一次 /api/deviation 的計算結果（routes.py 序列化後的 dict）。"""
    global _latest, _cached_at
    with _lock:
        _latest = result
        _cached_at = datetime.now(timezone.utc).isoformat()


def get_latest() -> dict:
    """回傳最新快取；尚未有任何資料時回傳明確的 not_yet_computed 狀態，
    HTTP 層一律回 200，不用另外處理錯誤路徑（見 dashboard/views.py）。"""
    with _lock:
        if _latest is None:
            return {"status": "not_yet_computed"}
        payload = dict(_latest)
        payload["cached_at"] = _cached_at
        return payload


def clear() -> None:
    """僅供測試使用，重置快取到初始狀態。"""
    global _latest, _cached_at
    with _lock:
        _latest = None
        _cached_at = None
