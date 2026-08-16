"""Python 端獨立的多天歷史持久化（SQLite）。

**設計動機**：Node-RED 的 ``cat_health_v3_flow.json`` 有自己一份多天歷史
（``global.v2_daily_history`` + ``daily_history.csv``），由它自己的「每日統計
彙整」JS function 在每天午夜彙整。過去 Python 端做基線比對（``/api/deviation``）
時，是靠 ``analytics_deviation_bridge.json`` 把 Node-RED 那份歷史整包轉發過來
當輸入——這代表 Python 算出來的結果跟 Node-RED 自己算的「新引擎」共用同一個
資料源，只要那份資料被寫壞（真實發生過：2026-08-09 假資料測試蓋掉
``v2_daily_history`` 7 天份紀錄且無法復原），兩邊會一起壞掉、比對也看不出異狀。

這個模組讓 Python 端有一份**完全獨立收集、獨立持久化**的多天歷史，不讀、不寫
Node-RED 的 ``global.json`` 或任何 Node-RED context——資料來源是
``trackers/behavior_tracker.py`` 自己每天累積的 ``get_today_stats()``（見
``ImprovedBehaviorTracker.check_daily_reset()`` 掛的持久化 hook）。

好處不只是少維護一份重複邏輯：兩邊資料來源獨立之後，若其中一邊被寫壞，比對
會出現真實分歧，等於多一層資料完整性的照妖鏡，而不是像現在這樣「同源，一起
壞掉、比對不出異狀」。

**已知取捨**：兩邊各自收集，跨日觸發時機不完全相同（Node-RED 靠 inject 準時
00:00 觸發、Node-RED 沒在跑當天會整天漏記；Python 是下一次有 frame 進來時
``check_daily_reset()`` 才觸發，不受 Node-RED 停機影響但時間點可能差幾分鐘），
比對時應留容許誤差，不要把這種良性分歧當異常。

用 SQLite（stdlib ``sqlite3``）而非 CSV：``behavior_segments_log.csv`` 已經
因為 ISO/本地日期格式混用炸過一次，SQLite 有型別欄位跟交易安全，不會重蹈覆轍；
也不需要像 Node-RED 的 ``gfile``（見 ``docs/資料層架構現況與統一管理評估.md``
第八節）那樣自己刻 atomic write——SQLite 本身的 transaction 就足夠。
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional

from analytics.baseline import DailyRecord

_FIELDS = (
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
)

_lock = threading.RLock()

# path -> 已開啟且已跑過 DDL 的 sqlite3.Connection，跨呼叫重用（見 _connect()
# 說明）。所有存取都經過 _lock 序列化，所以連線本身可以安全跨執行緒共用。
_connections: dict[str, sqlite3.Connection] = {}


def _default_db_path() -> str:
    # 延遲匯入 config，避免 analytics 套件在不需要 Flask/config 情境下
    # （例如純跑 pytest analytics/tests/）也被迫要求 config.py 可匯入成功。
    from config import LoggingConfig

    return LoggingConfig.DAILY_HISTORY_DB_PATH


def _open_connection(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # check_same_thread=False：這個連線會被快取重用、跨呼叫（因此也跨執行緒，
    # 例如 dashboard/refresher.py 的背景執行緒、Flask request 執行緒、即時
    # 追蹤迴圈）存活，sqlite3 預設的同執行緒限制在這裡不適用；安全性由上面
    # 的模組級 _lock 保證同一時間只有一個呼叫端在用這個連線。
    conn = sqlite3.connect(path, timeout=10.0, check_same_thread=False)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS daily_history (
            day TEXT PRIMARY KEY,
            {", ".join(f"{f} REAL" if f not in ("walk_count", "stop_count", "lick_count", "scratch_count", "shake_count") else f"{f} INTEGER" for f in _FIELDS)}
        )
        """
    )
    # 獨立小表，只記錄「這天要不要排除」——刻意不在 daily_history 上加
    # 一個 excluded 欄位：exclude/include 是使用者對既有天數的編輯決策，
    # 跟「這天原始統計數字是多少」是兩件不同性質的事，分開存也方便日後
    # 单獨清空排除清單而不動到任何統計資料。語意對齊 Node-RED 自己的
    # v2_excluded_dates（排除，不是刪除——資料還在，只是基線計算不採用）。
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS excluded_dates (
            day TEXT PRIMARY KEY
        )
        """
    )
    conn.commit()
    return conn


@contextmanager
def _connect(db_path: Optional[str] = None):
    """回傳（必要時建立並快取）指向 ``path`` 的連線。

    2026-08-11 修正：原本每次呼叫都 `sqlite3.connect()` 開一個全新連線、
    重跑兩次 `CREATE TABLE IF NOT EXISTS` DDL，用完立刻 `close()`——
    `dashboard/refresher.py` 的背景執行緒預設每 2 秒左右就會呼叫一次
    `load_history()`/`load_excluded_dates()`，等於程序存活期間持續每 2 秒
    重開連線+重解析 DDL，而這份資料實際上只有每日重置那一刻才會真的改變。
    改成依 ``path`` 快取連線、只在第一次開啟時跑 DDL，之後重用同一個連線。
    """
    path = db_path or _default_db_path()
    with _lock:
        conn = _connections.get(path)
        if conn is None:
            conn = _open_connection(path)
            _connections[path] = conn
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def close_connection(db_path: Optional[str] = None) -> None:
    """關閉並清掉 ``db_path``（或預設路徑）對應的快取連線。

    正式執行時不需要呼叫——SQLite 檔案在程序結束時由作業系統回收即可（見
    模組開頭「用 SQLite」段落的設計取捨）。主要給測試用：pytest 的
    ``tmp_path`` 每個測試各自獨立，若不顯式關閉，快取連線會在整個測試
    session 期間持續佔著該路徑的檔案控制代碼，Windows 上檔案被佔用中的
    暫存目錄可能造成 pytest 清理暫存目錄時出狀況。
    """
    path = db_path or _default_db_path()
    with _lock:
        conn = _connections.pop(path, None)
    if conn is not None:
        conn.close()


def save_day(record: DailyRecord, db_path: Optional[str] = None) -> None:
    """把一天的彙整資料寫入（upsert，同一天重複寫入會覆蓋，方便補寫/修正）。"""
    with _lock, _connect(db_path) as conn:
        cols = ("day",) + _FIELDS
        placeholders = ", ".join("?" for _ in cols)
        updates = ", ".join(f"{f}=excluded.{f}" for f in _FIELDS)
        values = (record.day.isoformat(),) + tuple(
            getattr(record, f) for f in _FIELDS
        )
        conn.execute(
            f"""
            INSERT INTO daily_history ({", ".join(cols)}) VALUES ({placeholders})
            ON CONFLICT(day) DO UPDATE SET {updates}
            """,
            values,
        )


def load_history(
    db_path: Optional[str] = None, limit_days: Optional[int] = None
) -> list:
    """讀出全部（或最近 limit_days 天）的歷史紀錄，依日期由舊到新排序。

    回傳 ``DailyRecord`` 列表，可直接餵給 ``analytics.baseline.compute_baseline``。
    """
    import datetime as _dt

    with _lock, _connect(db_path) as conn:
        cur = conn.execute(
            f"SELECT day, {', '.join(_FIELDS)} FROM daily_history ORDER BY day ASC"
        )
        rows = cur.fetchall()
    if limit_days is not None and limit_days > 0:
        rows = rows[-limit_days:]
    records = []
    for row in rows:
        day = _dt.date.fromisoformat(row[0])
        kwargs = {"day": day}
        for name, value in zip(_FIELDS, row[1:]):
            kwargs[name] = value
        records.append(DailyRecord(**kwargs))
    return records


def record_count(db_path: Optional[str] = None) -> int:
    """回傳目前已持久化的天數，供健康檢查/除錯使用。"""
    with _lock, _connect(db_path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM daily_history")
        return int(cur.fetchone()[0])


def set_excluded(day, excluded: bool, db_path: Optional[str] = None) -> None:
    """把某一天標記為排除／取消排除（不會動到 daily_history 裡的原始統計
    數字，只影響 ``analytics.baseline.compute_baseline`` 算基線時要不要
    採用這天）。``day`` 可以是 ``datetime.date`` 或 ISO 字串。"""
    iso = day.isoformat() if hasattr(day, "isoformat") else str(day)
    with _lock, _connect(db_path) as conn:
        if excluded:
            conn.execute(
                "INSERT INTO excluded_dates (day) VALUES (?) "
                "ON CONFLICT(day) DO NOTHING",
                (iso,),
            )
        else:
            conn.execute("DELETE FROM excluded_dates WHERE day = ?", (iso,))


def load_excluded_dates(db_path: Optional[str] = None) -> list:
    """回傳目前被排除的日期（ISO 字串列表），可直接傳給
    ``analytics.baseline.compute_baseline(..., excluded_dates=...)``。"""
    with _lock, _connect(db_path) as conn:
        cur = conn.execute("SELECT day FROM excluded_dates ORDER BY day ASC")
        return [row[0] for row in cur.fetchall()]
