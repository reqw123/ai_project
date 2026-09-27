"""v7.2 明確記錄模式（飼主先說「我要記錄貓咪」，下一句才送來）＋舊資料庫補欄位。"""

import sqlite3
from datetime import datetime

import pytest

from iot.voice import store
from iot.voice.classifier import classify
from iot.voice.service import ReportService

NOW = datetime(2026, 9, 27, 21, 30).timestamp()


def cats(text):
    return sorted(r.category for r in classify(text, now=NOW, explicit=True).reports)


@pytest.mark.parametrize("text, expected", [
    ("剛剛吐了", ["vomit"]),                    # 不用再說「貓咪」
    ("牠今天一直甩頭", ["head_shake"]),
    ("今天去看醫生", ["vet"]),                  # 一般模式會被主詞規則擋掉
    ("有沒有吐我不確定", ["other"]),             # 不做問句排除；「沒有吐」不算嘔吐，但照樣記成其他（不丟掉）
    ("今天幫牠洗澡", ["other"]),                # 對不上類別 → 其他，不丟掉
    ("甩", ["other"]),
    ("昨天吃很少", ["appetite"]),
])
def test_explicit_records(text, expected):
    assert cats(text) == expected


@pytest.mark.parametrize("text", ["取消", "算了", "不用了", "好算了", "不要記了", "沒事了", "記錯了", "算了，不用記了。", "取消取消"])
def test_explicit_cancel(text):
    c = classify(text, now=NOW, explicit=True)
    assert c.cancelled and not c.matched and c.reason == "cancel"


def test_explicit_empty():
    c = classify("", now=NOW, explicit=True)
    assert not c.matched and not c.cancelled and c.reason == "empty"


def test_non_explicit_is_unchanged():
    assert not classify("剛剛吐了", now=NOW).matched            # 沒提到貓 → 一般模式不記
    assert classify("取消", now=NOW).reason == "cancel" and not classify("取消", now=NOW).cancelled


def test_service_explicit_and_cancel(db_path):
    svc = ReportService(db_path=db_path, log_unmatched=False)
    out = svc.handle_utterance({"id": "e1", "text": "剛剛吐了", "mode": "explicit", "ts": NOW * 1000})
    assert out["matched"] and out["mode"] == "explicit" and out["confirm_key"] == "vomit"
    out = svc.handle_utterance({"id": "e2", "text": "算了", "mode": "explicit", "ts": NOW * 1000})
    assert out["cancelled"] and out["confirm_key"] == "cancel" and out["reply"] == "好的，已取消記錄。" and not out["matched"]
    rows = store.utterances_between(db_path=db_path)
    assert [(r["request_id"], r["mode"], r["reason"]) for r in rows] == [("e1", "explicit", ""), ("e2", "explicit", "cancel")]   # 明確模式一定存


def test_old_database_gets_mode_column(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE utterances (id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL, ts REAL NOT NULL,
            text TEXT NOT NULL, normalized TEXT NOT NULL, matched INTEGER NOT NULL, reason TEXT NOT NULL DEFAULT '',
            device_id TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT '');
        INSERT INTO utterances (request_id, ts, text, normalized, matched) VALUES ('old', 1, '貓咪吐了', '貓咪吐了', 1);
    """)
    conn.commit(); conn.close()
    try:
        ReportService(db_path=path).handle_utterance({"id": "new", "text": "剛剛吐了", "mode": "explicit"})
        rows = store.utterances_between(db_path=path)
        assert [(r["request_id"], r["mode"]) for r in rows] == [("old", ""), ("new", "explicit")]
    finally:
        store.close_connection(path)


def test_explicit_long_sentence_with_cancel_word_is_recorded():
    # 長句裡有「不要」不是取消（「不要讓牠再吐了，牠今天吐了兩次」→ 嘔吐）
    assert cats("不要讓牠再吃草了今天吐了兩次") == ["vomit"]
