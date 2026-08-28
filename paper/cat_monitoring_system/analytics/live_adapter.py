"""即時 BehaviorTracker 資料的中性轉接層。

**動機**：`dashboard/refresher.py` 過去 `from server.routes import _today_from_live_tracker,
_dataclass_to_jsonable`——dashboard（唯讀展示層）反向依賴 server 層的私有函式，
而 `server/routes.py` 又延遲 import `dashboard.cache`，形成雙向耦合。

本模組把「今日即時統計」的來源收成一個中性註冊點：

  - `server/routes.py` 建立 `FrameProcessor` 時呼叫 `set_active_tracker()` 註冊當前 tracker。
  - `routes.py` 的 `POST /api/deviation` 與 `dashboard/refresher.py` 都透過這裡讀取，
    兩邊都不需要 import 對方。

只依賴 `analytics.baseline.DailyRecord` 與 stdlib，不碰 cv2/torch/config/Flask。
"""

from __future__ import annotations

import datetime
import threading
from typing import Optional

from analytics.baseline import DailyRecord

_lock = threading.Lock()
_active_tracker = None  # 由 server/routes.py 的 _build_frame_processor() 註冊


def set_active_tracker(tracker) -> None:
    """註冊目前處理管線的 BehaviorTracker（`FrameProcessor.tracker`）。
    傳 None 可解除註冊（例如管線關閉）。"""
    global _active_tracker
    with _lock:
        _active_tracker = tracker


def get_active_tracker():
    """回傳目前註冊的 tracker，尚未啟動處理管線時為 None。"""
    with _lock:
        return _active_tracker


def today_from_tracker(tracker=None) -> Optional[dict]:
    """把 `tracker.get_today_stats()` 轉成 `analytics.deviation.compute_deviation()`
    需要的扁平鍵名（walk_time / walk_count / ...）。

    `tracker` 省略時使用目前註冊的 active tracker；沒有可用 tracker 時回傳
    None（呼叫端據此判斷「攝影機管線尚未啟動、沒有即時資料」）。
    """
    if tracker is None:
        tracker = get_active_tracker()
    if tracker is None:
        return None
    stats = tracker.get_today_stats()
    return {
        "walk_time": stats.get("walk_time", 0),
        "walk_count": stats.get("walk", 0),
        "stop_time": stats.get("stop_time", 0),
        "stop_count": stats.get("stop", 0),
        "lick_time": stats.get("lick_time", 0),
        "lick_count": stats.get("lick", 0),
        "scratch_time": stats.get("scratch_time", 0),
        "scratch_count": stats.get("scratch", 0),
        "shake_count": stats.get("shake", 0),
    }


def daily_record_from_dict(d: dict) -> DailyRecord:
    """解析 `POST /api/deviation` 請求 body 裡的一筆每日紀錄。

    ``date`` 欄位僅接受 ISO 格式（YYYY-MM-DD）。呼叫端若使用其他日期格式
    （例如 ``toLocaleDateString('zh-TW')`` 產生的 2026/7/2），需自行正規化後
    再送進來——刻意不猜測/相容多種日期格式，寧可讓格式錯誤在這裡就明確
    報錯，也不要靜默解析錯誤造成基線算錯。
    """
    raw_date = d.get("date")
    try:
        day = datetime.date.fromisoformat(str(raw_date)[:10])
    except (TypeError, ValueError):
        raise ValueError(f"date 必須是 ISO 格式 (YYYY-MM-DD)，收到: {raw_date!r}")

    kwargs = {"day": day}
    for field_name in (
        "monitoring_seconds",
        "walk_time",
        "walk_count",
        "stop_time",
        "stop_count",
        "lick_time",
        "lick_count",
        "scratch_time",
        "scratch_count",
        "shake_count",
        "active_time",
        "rest_time",
    ):
        if field_name in d:
            kwargs[field_name] = d[field_name]
    return DailyRecord(**kwargs)
