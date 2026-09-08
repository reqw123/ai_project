"""端到端（不接真 broker / 不裝 paho）：假 payload → Hub → store + publisher。"""

import json

from iot.config import IotHubConfig as C
from iot.output.alert_engine import AlertEngine
from iot.runner import Hub
from iot.sensors.weight import WeightEventDetector
from iot.storage import store


def _msg(hub, topic, obj):
    hub.handle_message(topic, json.dumps(obj).encode())


def test_env_flow_persists_and_publishes_and_alerts(db_path, publisher):
    hub = Hub(publisher, db_path=db_path)
    _msg(hub, "cat/iot/env/room", {"temp_c": C.TEMP_MAX_C + 4, "humidity_pct": 55, "ts": 1000.0})

    assert store.row_count("env_readings", db_path) == 1
    assert store.row_count("iot_alerts", db_path) == 1
    kinds = [d[0] for d in publisher.derived]
    assert "env" in kinds
    assert any(a["key"].startswith("env.temp_high.") for a in publisher.alerts)


def test_unparseable_message_is_swallowed(db_path, publisher):
    hub = Hub(publisher, db_path=db_path)
    hub.handle_message("cat/iot/env/room", b"garbage")
    hub.handle_message("cat/iot/env/room", json.dumps({"temp_c": 9999}).encode())
    assert store.row_count("env_readings", db_path) == 0
    assert publisher.alerts == []


def test_weight_flow_emits_feeding_event(db_path, publisher):
    hub = Hub(
        publisher,
        db_path=db_path,
        weight_detector=WeightEventDetector(event_delta_g=3.0, settle_sec=5.0),
    )
    for ts, g in [(0, 100), (1, 100), (20, 80), (21, 80), (30, 80)]:
        _msg(hub, "cat/iot/weight/food_bowl", {"grams": g, "ts": ts})

    assert store.row_count("weight_readings", db_path) == 5
    assert store.row_count("weight_events", db_path) == 1
    assert publisher.events, "應發出 weight_change derived 事件"
    kind, payload = publisher.events[-1]
    assert kind == "weight_change"
    assert payload["direction"] == "decrease"


def test_periodic_check_flags_no_feeding(db_path, publisher):
    hub = Hub(
        publisher,
        db_path=db_path,
        alert_engine=AlertEngine(cooldown_sec=1.0),
    )
    hub.run_periodic_checks(now=2_000_000.0)
    # 預設 FOOD_SCALE_IDS = ("food_bowl",)，從未進食 → 應有 feeding.silence 告警
    assert any(a["key"].startswith("feeding.silence.") for a in publisher.alerts)
    assert store.row_count("iot_alerts", db_path) >= 1


def test_bodytemp_flow(db_path, publisher):
    hub = Hub(publisher, db_path=db_path)
    _msg(hub, "cat/iot/bodytemp/carrier",
         {"surface_temp_c": C.BODYTEMP_SURFACE_MAX_C + 3, "ambient_temp_c": 27, "ts": 1.0})
    assert store.row_count("bodytemp_readings", db_path) == 1
    assert "bodytemp" in [d[0] for d in publisher.derived]
    assert any(a["key"].startswith("bodytemp.high.") for a in publisher.alerts)
    assert store.row_count("iot_alerts", db_path) == 1


def test_motion_flow(db_path, publisher):
    hub = Hub(publisher, db_path=db_path)
    _msg(hub, "cat/iot/motion/hall", {"active": True, "ts": 5.0})
    assert store.row_count("motion_events", db_path) == 1
    assert ("motion", "hall", {"kind": "motion", "source_id": "hall", "active": True, "ts": 5.0}) in [
        (k, s, p) for (k, s, p) in publisher.derived
    ]
