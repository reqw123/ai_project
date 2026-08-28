"""背景排程：在同一個 process 內直接算一次基線/偏差/融合結果，寫入
``dashboard/cache.py``，完全不透過 HTTP、不依賴 Node-RED 有沒有開。

**為什麼需要這個模組**：``server/routes.py`` 的 ``POST /api/deviation``
雖然已經改成預設吃 Python 自己獨立收集的資料（``analytics/daily_store.py``），
但「誰來觸發這次計算」在那之前完全沒變——目前唯一會呼叫這個端點的是 Node-RED
的 ``analytics_deviation_bridge.json``（給新舊引擎比對用），Python 端沒有任何
自己的排程會主動算。結果是：即使資料源已經獨立，只要 Node-RED 沒開、那個
bridge flow 沒觸發，``/dashboard/baseline`` 這個「只看新引擎」的展示頁就永遠
停在「尚未取得資料」——資料源獨立了，觸發權還握在 Node-RED 手上。

這個模組把觸發權也收回 Python 端：``start_background_refresh()`` 啟動一個
daemon thread，定期直接呼叫 ``analytics/`` 的計算函式（不繞 HTTP、不用自己
打自己的 ``/api/deviation``），算完直接寫進 ``dashboard/cache.py``。這樣
``/dashboard/baseline`` 這個頁面就跟 Node-RED 完全脫鉤——Node-RED 開不開、
``analytics_deviation_bridge.json`` 有沒有部署，都不影響這個頁面能不能看到
最新資料。

跟 ``POST /api/deviation`` 是兩條完全獨立、互不依賴的觸發路徑：
- Node-RED 觸發 → 算給 Node-RED 用（新舊引擎比對），順便也更新這裡的快取。
- 這裡的背景執行緒 → 定期自己算，只給這個展示頁用，不需要 Node-RED 參與。
任一邊沒運作都不影響另一邊。

``class_c_score``（節律/轉移模式分數）目前固定用 ``compute_fusion`` 的預設值
0——Class C 分析還沒搬進 Python（見 ``analytics/README.md``「還沒做的事」），
這裡刻意不去讀 Node-RED 的 ``global.v2_class_c_score`` 來補這個值，否則又會
製造一條隱性的 Node-RED 依賴，違背這個模組存在的目的。
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from analytics import daily_store
from analytics.baseline import (
    HISTORY_LOAD_LIMIT_DAYS,
    InsufficientDataError,
    compute_baseline,
)
from analytics.deviation import compute_deviation
from analytics.fusion import compute_fusion
from config import BaselineDashboardConfig

_started = False
_started_lock = threading.Lock()


def compute_and_cache_once() -> bool:
    """算一次目前的基線/偏差/融合結果並寫入 ``dashboard/cache.py``。

    回傳是否成功寫入「ok」狀態的完整結果。攝影機管線還沒啟動（沒有即時
    ``today`` 資料）或歷史天數不足時，回傳 ``False``——這是正常的「還沒
    準備好」狀態，不是錯誤；歷史不足的情況仍會把 ``insufficient_data``
    狀態寫進快取，讓頁面能顯示明確的等待訊息，而不是維持在完全沒收過任何
    回應的 ``not_yet_computed``。
    """
    # 2026-08-29：改從 analytics/live_adapter 讀即時資料，不再 import
    # server.routes 的私有函式（解除 dashboard → server 的反向耦合）。
    # live_adapter 只依賴 analytics.baseline + stdlib，沒有 cv2/torch 重依賴，
    # 故可在檔案層級 import；set_latest 仍延遲匯入維持既有 lazy 慣例。
    import dataclasses

    from analytics.live_adapter import today_from_tracker

    from dashboard.cache import set_latest

    _dataclass_to_jsonable = dataclasses.asdict
    today = today_from_tracker()
    if today is None:
        return False  # 攝影機管線還沒啟動，沒有即時資料可算

    # limit_days：見 analytics/config.py 的 HISTORY_LOAD_LIMIT_DAYS 說明——
    # 這個背景執行緒每 RECOMPUTE_INTERVAL_SEC（預設 2 秒）就會呼叫一次，
    # daily_history 只增不減，不限制讀取筆數的話系統跑越久這裡的成本越高。
    daily_records = daily_store.load_history(limit_days=HISTORY_LOAD_LIMIT_DAYS)
    excluded_dates = daily_store.load_excluded_dates()
    try:
        baseline = compute_baseline(daily_records, excluded_dates=excluded_dates)
    except InsufficientDataError as e:
        set_latest(
            {
                "status": "insufficient_data",
                "current_days": e.current_days,
                "required_days": e.required_days,
            }
        )
        return False

    deviation = compute_deviation(today=today, baseline=baseline)
    fusion = compute_fusion(deviation)

    set_latest(
        {
            "status": "ok",
            "baseline": _dataclass_to_jsonable(baseline),
            "deviation": _dataclass_to_jsonable(deviation),
            "fusion": _dataclass_to_jsonable(fusion),
        }
    )
    return True


def _cache_is_fresh(max_age_sec: float) -> bool:
    """快取是否已經在 ``max_age_sec`` 秒內被更新過——不分是這裡的背景執行緒
    自己算的，還是 Node-RED 觸發 ``POST /api/deviation`` 順便更新的（見
    routes.py ``api_deviation()`` 結尾寫入 ``dashboard/cache`` 那段）。

    2026-08-11 加入：Node-RED 正常運作時，它觸發 `/api/deviation` 的頻率
    跟這個背景執行緒的 `RECOMPUTE_INTERVAL_SEC` 差不多（都約 2 秒），而
    兩邊算的是同一份 daily_store 資料、寫進同一個快取——沒有這個判斷的話，
    兩條路徑會在 Node-RED 開著的正常情況下持續重複做幾乎一樣的統計計算+
    SQLite 讀取，卻只換來次秒級的時間差。這裡只在快取「不夠新鮮」時才真的
    重算，Node-RED 沒開／沒觸發時快取自然不會被另一條路徑刷新，這個背景
    執行緒照常自己算，不影響模組開頭說的「兩條路徑互不依賴」承諾。
    """
    from dashboard.cache import get_latest

    cached_at = get_latest().get("cached_at")
    if not cached_at:
        return False
    try:
        ts = datetime.fromisoformat(cached_at)
    except ValueError:
        return False
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    return age < max_age_sec


def _loop():
    interval = max(1.0, BaselineDashboardConfig.RECOMPUTE_INTERVAL_SEC)
    while True:
        try:
            if not _cache_is_fresh(interval):
                compute_and_cache_once()
        except Exception as e:
            # 背景執行緒裡任何未預期例外都只記警告、不能讓整個 daemon thread
            # 死掉——跟 behavior_tracker.py 的 save_state()/_persist_daily_record()
            # 容錯原則一致：這是輔助性的展示資料更新，不該影響主流程。
            logging.warning("Baseline dashboard 背景刷新失敗: %s", e)
        time.sleep(interval)


def start_background_refresh() -> None:
    """啟動背景執行緒，定期刷新 ``/dashboard/baseline`` 的資料。

    呼叫端（``main.py``）應該只在 ``BaselineDashboardConfig.ENABLED`` 為
    True 時呼叫——旗標關閉時整個 ``dashboard/`` 套件都不該被匯入，見
    ``config.BaselineDashboardConfig``。

    刻意不放進 ``server/flask_app.py`` 的 ``create_app()``：``create_app()``
    在測試裡會被重複呼叫來驗證旗標開關行為（見
    ``dashboard/tests/test_flag_integration.py``），如果背景執行緒的啟動
    也綁在那裡，每次測試呼叫 ``create_app()`` 都會多開一個 daemon thread、
    且會用預設（正式環境）路徑讀寫 SQLite，造成測試之間互相汙染、甚至
    意外碰到正式資料。放在 ``main.py`` 的 ``run_server_mode()``，跟
    `_scheduler_loop` 那個背景執行緒是同一層級的「真正啟動伺服器時才會
    發生」的副作用，`create_app()` 本身維持純粹、可重複呼叫。

    用一個模組層級旗標擋重複啟動——理論上呼叫端不會呼叫兩次，但用旗標
    擋住比讓兩個背景執行緒同時跑（雙倍不必要的 SQLite 讀取）更保險。
    """
    global _started
    with _started_lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_loop, daemon=True, name="baseline-dashboard-refresh").start()
