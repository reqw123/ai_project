"""飼主語音紀錄的 SQLite 持久化。

結構沿用 ``iot/storage/store.py``（也就是 ``analytics/daily_store.py``）的範式：依 ``db_path``
快取連線、第一次開啟時跑 DDL、模組級鎖序列化、``db_path`` 可注入給測試用。

兩張表
──────
* ``owner_reports``：分類出來的紀錄（一句話可以有好幾筆）。刪除是**軟刪除**（``deleted=1``＋時間），
  資料留著——飼主在儀表板刪掉的誤判本身就是論文 E1「實際使用時的誤判率」的資料。
* ``utterances``：每一句送進來的話（有沒有分類成紀錄都存），``reason`` 記沒分類的原因；
  之後人工標註「其實是回報卻漏掉」就能算召回率。可用 ``LOG_UNMATCHED=0`` 關掉未分類句的紀錄。
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Optional

from iot.voice.classifier import Classification

_lock = threading.RLock()
_connections: dict[str, sqlite3.Connection] = {}

_DDL = """
CREATE TABLE IF NOT EXISTS utterances (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT NOT NULL,
    ts          REAL NOT NULL,          -- 說話時間（epoch 秒）
    text        TEXT NOT NULL,          -- 語音辨識的原句
    normalized  TEXT NOT NULL,
    matched     INTEGER NOT NULL,       -- 分類出幾筆紀錄
    reason      TEXT NOT NULL DEFAULT '',
    device_id   TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL DEFAULT '',
    mode        TEXT NOT NULL DEFAULT ''    -- explicit＝飼主先說了「我要記錄貓咪」（v7.2）
);
CREATE INDEX IF NOT EXISTS ix_utt_ts ON utterances (ts);
CREATE UNIQUE INDEX IF NOT EXISTS ux_utt_request ON utterances (request_id);

CREATE TABLE IF NOT EXISTS owner_reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT NOT NULL,
    ts          REAL NOT NULL,          -- 說話時間
    day         TEXT NOT NULL,          -- 這件事是哪一天（「昨天帶去看醫生」→ 前一天）
    day_offset  INTEGER NOT NULL DEFAULT 0,
    kind        TEXT NOT NULL,          -- event / observation
    category    TEXT NOT NULL,
    label       TEXT NOT NULL,
    text        TEXT NOT NULL,
    matched     TEXT NOT NULL DEFAULT '',
    related     TEXT NOT NULL DEFAULT '',   -- 對應的系統行為類別（逗號分隔）
    device_id   TEXT NOT NULL DEFAULT '',
    source      TEXT NOT NULL DEFAULT '',
    deleted     INTEGER NOT NULL DEFAULT 0,
    deleted_ts  REAL
);
CREATE INDEX IF NOT EXISTS ix_rep_day ON owner_reports (day);
CREATE INDEX IF NOT EXISTS ix_rep_ts ON owner_reports (ts);
"""


def _default_db_path() -> str:
    from iot.voice.config import VoiceReportConfig

    return VoiceReportConfig.DB_PATH


def _open_connection(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_DDL)
    # v7.2：舊資料庫（v7.1 建的）補 mode 欄位
    cols = {r[1] for r in conn.execute("PRAGMA table_info(utterances)")}
    if "mode" not in cols:
        conn.execute("ALTER TABLE utterances ADD COLUMN mode TEXT NOT NULL DEFAULT ''")
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


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.row_factory = None


# ── 寫入 ──────────────────────────────────────────────────────────────────


def save(
    request_id: str,
    ts: float,
    result: Classification,
    device_id: str = "",
    source: str = "",
    log_unmatched: bool = True,
    db_path: Optional[str] = None,
    mode: str = "",
) -> list[int]:
    """存一句話與它的紀錄（同一個交易）；回傳新紀錄的 id。同一個 request_id 已存過 → 不重複寫、回傳舊的 id。"""
    with _lock, _connect(db_path) as conn:
        old = conn.execute("SELECT 1 FROM utterances WHERE request_id = ?", (request_id,)).fetchone()
        if old:
            return [r[0] for r in conn.execute(
                "SELECT id FROM owner_reports WHERE request_id = ? ORDER BY id", (request_id,)).fetchall()]
        if result.matched or log_unmatched:
            conn.execute(
                "INSERT INTO utterances (request_id, ts, text, normalized, matched, reason, device_id, source, mode)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (request_id, ts, result.text, result.normalized, len(result.reports), result.reason, device_id, source, mode),
            )
        ids = []
        for r in result.reports:
            cur = conn.execute(
                "INSERT INTO owner_reports (request_id, ts, day, day_offset, kind, category, label, text, matched,"
                " related, device_id, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (request_id, ts, r.day, r.day_offset, r.kind, r.category, r.label, result.text, r.matched,
                 ",".join(r.related_behaviors), device_id, source),
            )
            ids.append(int(cur.lastrowid))
        return ids


def set_deleted(report_id: int, deleted: bool = True, db_path: Optional[str] = None) -> bool:
    """軟刪除（或復原）一筆；回傳有沒有這筆。"""
    with _lock, _connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE owner_reports SET deleted = ?, deleted_ts = ? WHERE id = ?",
            (1 if deleted else 0, time.time() if deleted else None, int(report_id)),
        )
        return cur.rowcount > 0


# ── 查詢 ──────────────────────────────────────────────────────────────────


def recent_reports(limit: int = 30, include_deleted: bool = False, db_path: Optional[str] = None) -> list[dict]:
    where = "" if include_deleted else "WHERE deleted = 0"
    with _lock, _connect(db_path) as conn:
        return _rows(conn, f"SELECT * FROM owner_reports {where} ORDER BY ts DESC, id DESC LIMIT ?", (int(limit),))


def reports_between(
    day_from: Optional[str] = None,
    day_to: Optional[str] = None,
    include_deleted: bool = False,
    db_path: Optional[str] = None,
) -> list[dict]:
    """依「事件那一天」篩選（含頭尾，ISO 日期；省略＝不限）。"""
    cond, params = [], []
    if day_from:
        cond.append("day >= ?"); params.append(day_from)
    if day_to:
        cond.append("day <= ?"); params.append(day_to)
    if not include_deleted:
        cond.append("deleted = 0")
    where = ("WHERE " + " AND ".join(cond)) if cond else ""
    with _lock, _connect(db_path) as conn:
        return _rows(conn, f"SELECT * FROM owner_reports {where} ORDER BY day, ts, id", tuple(params))


def utterances_between(ts_from: Optional[float] = None, ts_to: Optional[float] = None,
                       db_path: Optional[str] = None) -> list[dict]:
    cond, params = [], []
    if ts_from is not None:
        cond.append("ts >= ?"); params.append(ts_from)
    if ts_to is not None:
        cond.append("ts <= ?"); params.append(ts_to)
    where = ("WHERE " + " AND ".join(cond)) if cond else ""
    with _lock, _connect(db_path) as conn:
        return _rows(conn, f"SELECT * FROM utterances {where} ORDER BY ts, id", tuple(params))


def day_counts(day: str, db_path: Optional[str] = None) -> dict[str, int]:
    """某一天（未刪除）的各類別筆數。"""
    with _lock, _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT category, COUNT(*) FROM owner_reports WHERE day = ? AND deleted = 0 GROUP BY category", (day,)
        ).fetchall()
    return {k: int(n) for k, n in rows}
