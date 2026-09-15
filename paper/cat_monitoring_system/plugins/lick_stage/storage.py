"""事件表 / 視窗摘要表持久化（說明書「事件與視窗資料設計」）。

設計原則：
  * 事件表保留「可重算的最小資訊」；視窗表供 Node-RED 與健康異常分析使用。
  * 主儲存為 SQLite（單檔、原子交易、可查詢）；CSV 僅作交換格式，於
    `close_session()` 時匯出。
  * fail-safe：初始化 / 寫入失敗只記 DEBUG，不拋例外、不影響主系統。
  * 歷史 `ext_body_zones/results.csv` 不轉進來 —— 那是累積快照，只標
    legacy_snapshot 作探索性分析（說明書「向後相容策略」）。

預設**停用**：`LickStorage(db_path=None)` 時所有方法都是 no-op。呼叫端
（manager / FrameProcessor）要明確給路徑才會落地，維持第一階段 shadow 精神。
"""

from __future__ import annotations

import csv
import logging
import os
import sqlite3
import threading
from typing import Optional

from plugins.lick_stage.analysis_context import SCHEMA_VERSION

_log = logging.getLogger(__name__)

# schema 版本化（說明書：任何欄位語意變更都提高版本號）
_DB_USER_VERSION = 1

_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS lick_events (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_version     TEXT,
    session_id         TEXT,
    event_id           TEXT,
    cat_id             TEXT,
    track_id           TEXT,
    video_id           TEXT,
    period             TEXT,
    start_source_ts    REAL,
    end_source_ts      REAL,
    duration_sec       REAL,
    start_frame        INTEGER,
    end_frame          INTEGER,
    action             TEXT,
    action_score_mean  REAL,
    action_score_min   REAL,
    zone_l1            TEXT,
    zone_l2            TEXT,
    zone_score_mean    REAL,
    zone_switch_count  INTEGER,
    assigned_ratio     REAL,
    pose_quality_mean  REAL,
    unknown_reason_mode TEXT,
    ext_zone_mode       TEXT,
    ext_zone_l1_mode    TEXT,
    ext_zone_confidence_mean REAL,
    raw_bout           INTEGER,
    model_version      TEXT,
    config_hash        TEXT,
    topology_version   TEXT,
    code_commit        TEXT
);
"""

_WINDOW_DDL = """
CREATE TABLE IF NOT EXISTS lick_window_summary (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_version      TEXT,
    session_id          TEXT,
    cat_id              TEXT,
    video_id            TEXT,
    window_start        REAL,
    window_end          REAL,
    window_sec          REAL,
    period              TEXT,
    observed_sec        REAL,
    valid_observed_sec  REAL,
    no_cat_sec          REAL,
    stale_sec           REAL,
    stgcn_lick_sec      REAL,
    assigned_zone_sec   REAL,
    unassigned_lick_sec REAL,
    bout_count          INTEGER,
    median_bout_sec     REAL,
    max_bout_sec        REAL,
    torso_sec           REAL,
    forelimb_sec        REAL,
    hindlimb_sec        REAL,
    tail_sec            REAL,
    zone_coverage_ratio REAL,
    model_version       TEXT,
    config_hash         TEXT
);
"""

_SESSION_DDL = """
CREATE TABLE IF NOT EXISTS lick_sessions (
    session_id      TEXT PRIMARY KEY,
    video_id        TEXT,
    cat_id          TEXT,
    period          TEXT,
    model_version   TEXT,
    config_hash     TEXT,
    source_fps      REAL,
    started_at      TEXT,
    finished_at     TEXT,
    event_count     INTEGER DEFAULT 0,
    window_count    INTEGER DEFAULT 0,
    closed_cleanly  INTEGER DEFAULT 0
);
"""

_EVENT_COLS = [
    "schema_version", "session_id", "event_id", "cat_id", "track_id", "video_id",
    "period", "start_source_ts", "end_source_ts", "duration_sec", "start_frame",
    "end_frame", "action", "action_score_mean", "action_score_min", "zone_l1",
    "zone_l2", "zone_score_mean", "zone_switch_count", "assigned_ratio",
    "pose_quality_mean", "unknown_reason_mode",
    # M6（2026-09-15，ext_body_zones 融合）：見 bout_aggregator.py::to_event()
    "ext_zone_mode", "ext_zone_l1_mode", "ext_zone_confidence_mean",
    "raw_bout", "model_version",
    "config_hash", "topology_version", "code_commit",
]

_WINDOW_COLS = [
    "schema_version", "session_id", "cat_id", "video_id", "window_start",
    "window_end", "window_sec", "period", "observed_sec", "valid_observed_sec",
    "no_cat_sec", "stale_sec", "stgcn_lick_sec", "assigned_zone_sec",
    "unassigned_lick_sec", "bout_count", "median_bout_sec", "max_bout_sec",
    "torso_sec", "forelimb_sec", "hindlimb_sec", "tail_sec",
    "zone_coverage_ratio", "model_version", "config_hash",
]


class LickStorage:
    """事件 / 視窗持久化。thread-safe（單一 lock 保護連線）。"""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._ctx: dict = {}
        self._session_id = ""
        self._event_seq = 0
        self._event_count = 0
        self._window_count = 0
        if db_path:
            self._init_db(db_path)

    @property
    def enabled(self) -> bool:
        return self._conn is not None

    def _init_db(self, path: str) -> None:
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(_SESSION_DDL)
            conn.execute(_EVENTS_DDL)
            conn.execute(_WINDOW_DDL)
            (cur_ver,) = conn.execute("PRAGMA user_version;").fetchone()
            if cur_ver == 0:
                conn.execute(f"PRAGMA user_version = {_DB_USER_VERSION};")
            conn.commit()
            self._conn = conn
        except Exception as exc:  # fail-safe
            _log.debug("LickStorage init failed (%s): %s", path, exc)
            self._conn = None

    # ── Session ─────────────────────────────────────────────────────────
    def open_session(self, context) -> None:
        """context 為 analysis_context.AnalysisContext。"""
        self._session_id = getattr(context, "session_id", "") or ""
        self._ctx = {
            "session_id": self._session_id,
            "cat_id": getattr(context, "cat_id", None),
            "video_id": getattr(context, "video_id", ""),
            "period": getattr(context, "period", ""),
            "model_version": getattr(context, "model_version", ""),
            "config_hash": getattr(context, "config_hash", ""),
            "topology_version": getattr(context, "topology_version", ""),
            "code_commit": getattr(context, "code_commit", ""),
        }
        self._event_seq = 0
        self._event_count = 0
        self._window_count = 0
        if not self.enabled:
            return
        try:
            import datetime as _dt

            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO lick_sessions "
                    "(session_id, video_id, cat_id, period, model_version, "
                    " config_hash, source_fps, started_at, finished_at, "
                    " event_count, window_count, closed_cleanly) "
                    "VALUES (?,?,?,?,?,?,?,?,NULL,0,0,0)",
                    (
                        self._session_id,
                        self._ctx["video_id"],
                        self._ctx["cat_id"],
                        self._ctx["period"],
                        self._ctx["model_version"],
                        self._ctx["config_hash"],
                        float(getattr(context, "source_fps", 0.0) or 0.0),
                        _dt.datetime.now().isoformat(timespec="seconds"),
                    ),
                )
                self._conn.commit()
        except Exception as exc:
            _log.debug("LickStorage.open_session failed: %s", exc)

    def write_event(self, event: dict) -> None:
        if not self.enabled:
            return
        self._event_seq += 1
        row = dict(event)
        row.setdefault("schema_version", SCHEMA_VERSION)
        row["session_id"] = self._session_id
        row["event_id"] = f"{self._session_id}_E{self._event_seq:05d}"
        row.setdefault("action", "lick")
        for k in ("cat_id", "track_id", "video_id", "period", "model_version",
                  "config_hash", "topology_version", "code_commit"):
            row.setdefault(k, self._ctx.get(k))
        row["raw_bout"] = 1 if row.get("raw_bout") else 0
        try:
            with self._lock:
                self._conn.execute(
                    f"INSERT INTO lick_events ({','.join(_EVENT_COLS)}) "
                    f"VALUES ({','.join('?' for _ in _EVENT_COLS)})",
                    [row.get(c) for c in _EVENT_COLS],
                )
                self._conn.commit()
            self._event_count += 1
        except Exception as exc:
            _log.debug("LickStorage.write_event failed: %s", exc)

    def write_window(self, window: dict) -> None:
        if not self.enabled:
            return
        row = dict(window)
        row.setdefault("schema_version", SCHEMA_VERSION)
        row["session_id"] = self._session_id
        for k in ("cat_id", "video_id", "model_version", "config_hash"):
            row.setdefault(k, self._ctx.get(k))
        try:
            with self._lock:
                self._conn.execute(
                    f"INSERT INTO lick_window_summary ({','.join(_WINDOW_COLS)}) "
                    f"VALUES ({','.join('?' for _ in _WINDOW_COLS)})",
                    [row.get(c) for c in _WINDOW_COLS],
                )
                self._conn.commit()
            self._window_count += 1
        except Exception as exc:
            _log.debug("LickStorage.write_window failed: %s", exc)

    def close_session(self, *, csv_export: bool = True) -> None:
        if not self.enabled:
            return
        try:
            import datetime as _dt

            with self._lock:
                self._conn.execute(
                    "UPDATE lick_sessions SET finished_at=?, event_count=?, "
                    "window_count=?, closed_cleanly=1 WHERE session_id=?",
                    (
                        _dt.datetime.now().isoformat(timespec="seconds"),
                        self._event_count,
                        self._window_count,
                        self._session_id,
                    ),
                )
                self._conn.commit()
        except Exception as exc:
            _log.debug("LickStorage.close_session update failed: %s", exc)
        if csv_export:
            self._export_csv()

    def _export_csv(self) -> None:
        base = os.path.splitext(self._db_path)[0]
        for table, cols in (
            ("lick_events", ["event_id"] + _EVENT_COLS),
            ("lick_window_summary", _WINDOW_COLS),
        ):
            path = f"{base}.{table}.csv"
            try:
                with self._lock:
                    rows = self._conn.execute(
                        f"SELECT {','.join(cols)} FROM {table} "
                        f"WHERE session_id=?",
                        (self._session_id,),
                    ).fetchall()
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(cols)
                    w.writerows(rows)
            except Exception as exc:
                _log.debug("LickStorage CSV export (%s) failed: %s", table, exc)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
