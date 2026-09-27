"""service＋store：存檔、去重、軟刪除、最近紀錄、匯出。"""

import csv
from datetime import datetime

from iot.voice import store
from iot.voice.export import export
from iot.voice.service import ReportService

NOW_MS = datetime(2026, 9, 27, 21, 30).timestamp() * 1000


def test_matched_utterance_is_saved(db_path):
    svc = ReportService(db_path=db_path)
    out = svc.handle_utterance({"id": "a1", "text": "貓咪剛剛吐了", "ts": NOW_MS, "deviceId": "ESP32S3_VOICE_F08804", "source": "voice"})
    assert out["ok"] and out["matched"]
    assert out["confirm_key"] == "vomit" and out["reply"] == "已記錄：嘔吐。"
    (r,) = out["reports"]
    assert r["kind"] == "observation" and r["day"] == "2026-09-27" and r["report_id"] > 0
    rows = store.recent_reports(db_path=db_path)
    assert rows[0]["device_id"] == "ESP32S3_VOICE_F08804" and rows[0]["text"] == "貓咪剛剛吐了"


def test_multiple_reports_in_one_sentence(db_path):
    out = ReportService(db_path=db_path).handle_utterance({"id": "m1", "text": "今天家裡有客人，貓咪一直躲起來", "ts": NOW_MS})
    assert out["confirm_key"] == "multi"
    assert out["reply"] == "已記錄：有客人來訪、情緒緊張。"
    assert len(store.recent_reports(db_path=db_path)) == 2


def test_unmatched_is_logged_but_not_reported(db_path):
    out = ReportService(db_path=db_path).handle_utterance({"id": "u1", "text": "貓咪好可愛", "ts": NOW_MS})
    assert out["ok"] and not out["matched"] and out["reason"] == "no_category" and out["reply"] == ""
    assert store.recent_reports(db_path=db_path) == []
    (u,) = store.utterances_between(db_path=db_path)
    assert u["matched"] == 0 and u["reason"] == "no_category"


def test_unmatched_logging_can_be_disabled(db_path):
    ReportService(db_path=db_path, log_unmatched=False).handle_utterance({"id": "u2", "text": "貓咪好可愛"})
    assert store.utterances_between(db_path=db_path) == []


def test_same_id_twice_is_not_duplicated(db_path):
    svc = ReportService(db_path=db_path)
    a = svc.handle_utterance({"id": "d1", "text": "貓咪吐了", "ts": NOW_MS})
    b = svc.handle_utterance({"id": "d1", "text": "貓咪吐了", "ts": NOW_MS})
    assert a == b
    # 服務重啟（記憶體去重清空）→ 資料庫的 request_id 也擋得住
    c = ReportService(db_path=db_path).handle_utterance({"id": "d1", "text": "貓咪吐了", "ts": NOW_MS})
    assert [r["report_id"] for r in c["reports"]] == [r["report_id"] for r in a["reports"]]
    assert len(store.recent_reports(db_path=db_path)) == 1


def test_missing_id_is_rejected(db_path):
    out = ReportService(db_path=db_path).handle_utterance({"text": "貓咪吐了"})
    assert not out["ok"]


def test_soft_delete_and_restore(db_path):
    svc = ReportService(db_path=db_path)
    rid = svc.handle_utterance({"id": "s1", "text": "貓咪吐了", "ts": NOW_MS})["reports"][0]["report_id"]
    assert svc.handle_delete({"report_id": rid})
    assert store.recent_reports(db_path=db_path) == []
    assert len(store.recent_reports(include_deleted=True, db_path=db_path)) == 1   # 資料還在
    assert svc.handle_delete({"report_id": rid, "restore": True})
    assert len(store.recent_reports(db_path=db_path)) == 1
    assert not svc.handle_delete({"report_id": 99999})
    assert not svc.handle_delete({"report_id": "abc"})


def test_recent_snapshot(db_path):
    svc = ReportService(db_path=db_path)
    svc.handle_utterance({"id": "r1", "text": "貓咪吐了", "ts": NOW_MS})
    svc.handle_utterance({"id": "r2", "text": "昨天帶貓咪去看醫生", "ts": NOW_MS})
    snap = svc.recent_snapshot(now=NOW_MS / 1000)
    assert snap["today"] == "2026-09-27"
    assert snap["today_counts"] == {"vomit": 1}          # 看醫生是昨天的事
    assert [r["category"] for r in snap["reports"]] == ["vet", "vomit"]   # 同一時間：新的 id 在前
    assert {c["key"] for c in snap["categories"]} >= {"vet", "vomit", "other"}


def test_export_csv(db_path, tmp_path):
    svc = ReportService(db_path=db_path)
    svc.handle_utterance({"id": "e1", "text": "貓咪吐了", "ts": NOW_MS})
    svc.handle_utterance({"id": "e2", "text": "貓咪好可愛", "ts": NOW_MS})
    rid = svc.handle_utterance({"id": "e3", "text": "貓咪拉肚子", "ts": NOW_MS})["reports"][0]["report_id"]
    svc.handle_delete({"report_id": rid})
    out = tmp_path / "reports.csv"
    assert export("reports", str(out), "2026-09-27", "2026-09-27", db_path=db_path) == 1
    assert export("reports", str(out), include_deleted=True, db_path=db_path) == 2
    rows = list(csv.DictReader(open(out, encoding="utf-8-sig")))
    assert {r["category"] for r in rows} == {"vomit", "litter"} and rows[0]["time"].startswith("2026-09-27 21:30")
    out2 = tmp_path / "utt.csv"
    assert export("utterances", str(out2), "2026-09-27", "2026-09-27", db_path=db_path) == 3
