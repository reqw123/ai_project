"""IoT Sensor Hub 的 SQLite 持久化。

結構與並行處理沿用 ``analytics/daily_store.py`` 的既有範式：
  * 依 ``db_path`` 快取連線、只在第一次開啟時跑 DDL；
  * 所有存取經模組級 ``_lock`` 序列化，連線 ``check_same_thread=False`` 可跨執行緒重用；
  * ``db_path`` 參數可注入，測試各自用 ``tmp_path``；
  * ``close_connection()`` 主要給測試收尾（正式執行靠行程結束由 OS 回收）。

寫入失敗一律往上拋給呼叫端（runner）決定要不要吞——runner 會 try/except 記
警告後繼續，符合「壞掉也沒關係」的定位；這裡不自己靜默吞掉，以免除錯時看不到。
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional

from iot.sensors.base import (
    BodyTempReading,
    EnvironmentReading,
    MotionEvent,
    WeightReading,
)
from iot.sensors.weight import WeightChangeEvent

_TABLES = (
    "env_readings",
    "motion_events",
    "weight_readings",
    "weight_events",
    "bodytemp_readings",
    "iot_alerts",
)

_lock = threading.RLock()
_connections: dict[str, sqlite3.Connection] = {}

_DDL = """
CREATE TABLE IF NOT EXISTS env_readings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    source_id   TEXT NOT NULL,
    temp_c      REAL,
    humidity_pct REAL,
    gas_ppm     REAL,
    lux         REAL
);
CREATE INDEX IF NOT EXISTS ix_env_ts ON env_readings (ts);
CREATE INDEX IF NOT EXISTS ix_env_source_ts ON env_readings (source_id, ts);

CREATE TABLE IF NOT EXISTS motion_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    source_id   TEXT NOT NULL,
    active      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_motion_ts ON motion_events (ts);

CREATE TABLE IF NOT EXISTS weight_readings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    source_id   TEXT NOT NULL,
    grams       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_weight_ts ON weight_readings (ts);
CREATE INDEX IF NOT EXISTS ix_weight_source_ts ON weight_readings (source_id, ts);

CREATE TABLE IF NOT EXISTS weight_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    scale_id    TEXT NOT NULL,
    from_g      REAL NOT NULL,
    to_g        REAL NOT NULL,
    delta_g     REAL NOT NULL,
    direction   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_weight_events_scale_ts ON weight_events (scale_id, ts);

CREATE TABLE IF NOT EXISTS bodytemp_readings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    source_id     TEXT NOT NULL,
    surface_temp_c REAL NOT NULL,
    ambient_temp_c REAL
);
CREATE INDEX IF NOT EXISTS ix_bodytemp_source_ts ON bodytemp_readings (source_id, ts);

CREATE TABLE IF NOT EXISTS iot_alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    key         TEXT NOT NULL,
    severity    TEXT NOT NULL,
    message     TEXT NOT NULL,
    value       REAL,
    threshold   REAL
);
CREATE INDEX IF NOT EXISTS ix_alerts_key_ts ON iot_alerts (key, ts);
"""


def _default_db_path() -> str:
    from iot.config import IotHubConfig

    return IotHubConfig.DB_PATH


def _open_connection(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_DDL)
    conn.commit()
    return conn


@contextmanager
def _connect(db_path: Optional[str] = None):
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
    path = db_path or _default_db_path()
    with _lock:
        conn = _connections.pop(path, None)
    if conn is not None:
        conn.close()


# ── 寫入 ──────────────────────────────────────────────────────────────────


def insert_env(reading: EnvironmentReading, db_path: Optional[str] = None) -> None:
    with _lock, _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO env_readings (ts, source_id, temp_c, humidity_pct, gas_ppm, lux)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                reading.ts,
                reading.source_id,
                reading.temp_c,
                reading.humidity_pct,
                reading.gas_ppm,
                reading.lux,
            ),
        )


def insert_motion(event: MotionEvent, db_path: Optional[str] = None) -> None:
    with _lock, _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO motion_events (ts, source_id, active) VALUES (?, ?, ?)",
            (event.ts, event.source_id, int(event.active)),
        )


def insert_weight(reading: WeightReading, db_path: Optional[str] = None) -> None:
    with _lock, _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO weight_readings (ts, source_id, grams) VALUES (?, ?, ?)",
            (reading.ts, reading.source_id, reading.grams),
        )


def insert_weight_event(
    event: WeightChangeEvent, db_path: Optional[str] = None
) -> None:
    with _lock, _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO weight_events (ts, scale_id, from_g, to_g, delta_g, direction)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.ts,
                event.scale_id,
                event.from_g,
                event.to_g,
                event.delta_g,
                event.direction,
            ),
        )


def insert_bodytemp(reading: BodyTempReading, db_path: Optional[str] = None) -> None:
    with _lock, _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO bodytemp_readings (ts, source_id, surface_temp_c, ambient_temp_c)"
            " VALUES (?, ?, ?, ?)",
            (
                reading.ts,
                reading.source_id,
                reading.surface_temp_c,
                reading.ambient_temp_c,
            ),
        )


def insert_alert(
    key: str,
    severity: str,
    message: str,
    ts: float,
    value: float | None = None,
    threshold: float | None = None,
    db_path: Optional[str] = None,
) -> None:
    with _lock, _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO iot_alerts (ts, key, severity, message, value, threshold)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (ts, key, severity, message, value, threshold),
        )


# ── 查詢（給 alert_engine / 除錯用）───────────────────────────────────────


def last_weight_event_ts(
    scale_id: str, direction: Optional[str] = None, db_path: Optional[str] = None
) -> Optional[float]:
    sql = "SELECT MAX(ts) FROM weight_events WHERE scale_id = ?"
    params: tuple = (scale_id,)
    if direction is not None:
        sql += " AND direction = ?"
        params += (direction,)
    with _lock, _connect(db_path) as conn:
        row = conn.execute(sql, params).fetchone()
    return row[0] if row and row[0] is not None else None


def row_count(table: str, db_path: Optional[str] = None) -> int:
    if table not in _TABLES:
        raise ValueError(f"未知資料表: {table!r}")
    with _lock, _connect(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def recent(table: str, limit: int = 50, db_path: Optional[str] = None) -> list[dict]:
    if table not in _TABLES:
        raise ValueError(f"未知資料表: {table!r}")
    with _lock, _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                f"SELECT * FROM {table} ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        finally:
            conn.row_factory = None
    return [dict(r) for r in rows]
