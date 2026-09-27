"""CAT_MONITORING_IOT_KINDS：只處理部分感測器（設定視窗「IoT 子系統」分頁把各感測器分開啟動用）。"""

from iot.config import IotHubConfig as C
from iot.runner import Hub


def test_default_is_all_kinds(monkeypatch):
    monkeypatch.setattr(C, "KINDS", ())
    assert C.active_kinds() == list(C.TOPIC_MAP)
    assert C.subscribe_topics() == [f"{C.TOPIC_PREFIX}/{k}/#" for k in C.TOPIC_MAP]


def test_only_selected_kinds_are_subscribed(monkeypatch):
    monkeypatch.setattr(C, "KINDS", ("env", "bodytemp", "unknown"))
    assert C.active_kinds() == ["env", "bodytemp"]          # 不認得的種類忽略
    assert C.subscribe_topics() == [f"{C.TOPIC_PREFIX}/env/#", f"{C.TOPIC_PREFIX}/bodytemp/#"]


def test_feeding_silence_only_checked_when_weight_is_handled(monkeypatch, db_path, publisher):
    hub = Hub(publisher, db_path=db_path)
    monkeypatch.setattr(C, "KINDS", ("env",))
    hub.run_periodic_checks(now=10 ** 9)
    assert publisher.alerts == []                            # 只開環境感測：不能誤報「沒進食」
    monkeypatch.setattr(C, "KINDS", ("weight",))
    hub.run_periodic_checks(now=10 ** 9)
    assert any(a["key"].startswith("feeding.silence") for a in publisher.alerts)
