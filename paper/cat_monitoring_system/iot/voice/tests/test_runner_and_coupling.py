"""MQTT 路由（不接真 broker、不裝 paho）＋低耦合規則的靜態檢查。"""

import ast
import json
from pathlib import Path

from iot.voice.config import VoiceReportConfig as C
from iot.voice.runner import Router
from iot.voice.service import ReportService


class Recorder:
    def __init__(self):
        self.sent = []

    def __call__(self, topic, payload, retain):
        json.dumps(payload, ensure_ascii=False)   # 一定要能序列化
        self.sent.append((topic, payload, retain))

    def topics(self):
        return [t for t, _, _ in self.sent]


def test_utterance_publishes_result_and_recent(db_path):
    rec = Recorder()
    router = Router(ReportService(db_path=db_path), rec)
    router.handle(C.topic("utterance"), json.dumps({"id": "x1", "text": "貓咪吐了"}).encode())
    assert rec.topics() == [C.topic("result"), C.topic("reports/recent")]
    result = rec.sent[0][1]
    assert result["id"] == "x1" and result["matched"] and rec.sent[0][2] is False
    assert rec.sent[1][2] is True   # 最近紀錄是 retained


def test_unmatched_utterance_only_publishes_result(db_path):
    rec = Recorder()
    Router(ReportService(db_path=db_path), rec).handle(C.topic("utterance"), json.dumps({"id": "x2", "text": "你好"}))
    assert rec.topics() == [C.topic("result")] and rec.sent[0][1]["matched"] is False


def test_delete_and_get(db_path):
    rec = Recorder()
    router = Router(ReportService(db_path=db_path), rec)
    router.handle(C.topic("utterance"), json.dumps({"id": "x3", "text": "貓咪吐了"}))
    rid = rec.sent[0][1]["reports"][0]["report_id"]
    rec.sent.clear()
    router.handle(C.topic("reports/delete"), json.dumps({"report_id": rid}))
    assert rec.topics() == [C.topic("reports/recent")] and rec.sent[0][1]["reports"] == []
    rec.sent.clear()
    router.handle(C.topic("reports/get"), b"")
    assert rec.topics() == [C.topic("reports/recent")]


def test_garbage_is_ignored(db_path):
    rec = Recorder()
    router = Router(ReportService(db_path=db_path), rec)
    router.handle(C.topic("utterance"), b"not json")
    router.handle(C.topic("utterance"), b"[1,2]")
    router.handle(C.topic("reports/delete"), b"{}")
    router.handle("other/topic", b"{}")
    assert rec.sent == []


# ── 低耦合：iot/voice 只能 import 標準函式庫、paho、iot.voice.* ────────────────
_ALLOWED_ROOTS = {"paho", "pytest"}


def test_voice_package_imports_only_itself_stdlib_and_paho():
    import sys

    pkg = Path(__file__).resolve().parents[1]
    stdlib = set(sys.stdlib_module_names)
    bad = []
    for py in pkg.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if name == "iot.voice" or name.startswith("iot.voice.") or root in stdlib or root in _ALLOWED_ROOTS:
                    continue
                bad.append(f"{py.relative_to(pkg)}: import {name}")
    assert not bad, "iot/voice 不可以 import 主系統或感測器 hub：\n" + "\n".join(bad)


def test_nobody_else_imports_voice_package():
    root = Path(__file__).resolve().parents[3]   # cat_monitoring_system/
    voice_dir = root / "iot" / "voice"
    bad = []
    for py in root.rglob("*.py"):
        if voice_dir in py.parents or "__pycache__" in py.parts:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "iot.voice" in text or "from iot import voice" in text:
            bad.append(str(py.relative_to(root)))
    assert not bad, "主系統／感測器 hub 不可以 import iot.voice：" + ", ".join(bad)
