"""門檻告警一直超標：Discord／DB 照冷卻只發一次，日誌每 WARN_REPEAT_SEC 秒提醒（持續中），恢復記一行。"""

import logging

from iot.config import IotHubConfig as C
from iot.output.alert_engine import AlertEngine
from iot.runner import Hub
from iot.sensors.base import BodyTempReading


def test_engine_reports_suppressed_and_recovered():
    eng = AlertEngine(cooldown_sec=900)
    hot = lambda v: BodyTempReading("carrier", surface_temp_c=v)   # noqa: E731
    assert len(eng.check_bodytemp(hot(46.8), now=0)) == 1              # 第一次：發出去
    assert eng.drain() == ([], [])
    assert eng.check_bodytemp(hot(45.0), now=10) == []                 # 冷卻中：不發
    suppressed, recovered = eng.drain()
    assert [k for k, _ in suppressed] == ["bodytemp.high.carrier"] and "45.0" in suppressed[0][1]
    assert eng.check_bodytemp(hot(35.0), now=20) == []                 # 回到正常
    assert eng.drain() == ([], [("bodytemp.high.carrier", "貓體表溫度偏高（carrier）")])
    assert eng.drain() == ([], [])


def test_hub_logs_ongoing_alert_repeatedly_but_publishes_once(monkeypatch, caplog, db_path, publisher):
    monkeypatch.setattr(C, "WARN_REPEAT_SEC", 10.0)
    clock = [1000.0]
    monkeypatch.setattr("iot.runner.time.monotonic", lambda: clock[0])
    hub = Hub(publisher, db_path=db_path, alert_engine=AlertEngine(cooldown_sec=900))
    caplog.set_level(logging.INFO, logger="iot.runner")
    for i, v in enumerate([46.8, 41.0, 45.0, 50.0, 47.0]):              # 每 5 秒一筆，一直超標
        clock[0] = 1000.0 + i * 5
        hub.handle_message("cat/iot/bodytemp/carrier", f'{{"surface_temp_c": {v}}}'.encode())
    clock[0] = 1030.0
    hub.handle_message("cat/iot/bodytemp/carrier", b'{"surface_temp_c": 36.0}')
    msgs = [(r.levelname, r.getMessage()) for r in caplog.records if r.name == "iot.runner"]
    assert len(publisher.alerts) == 1                                   # Discord／MQTT 只發一次
    assert msgs[0][0] == "WARNING" and msgs[0][1].startswith("告警：貓體表溫度偏高：46.8")
    ongoing = [m for lv, m in msgs if "持續中" in m]
    assert len(ongoing) == 2 and "45.0" in ongoing[0] and "47.0" in ongoing[1]   # t=10、t=20
    assert msgs[-1] == ("INFO", "恢復正常：貓體表溫度偏高（carrier）")


def test_reading_log_interval(monkeypatch, caplog, db_path, publisher):
    """除錯：每個節點最新讀數每 READING_LOG_INTERVAL_SEC 秒印一行；0＝不印。"""
    clock = [0.0]
    monkeypatch.setattr("iot.runner.time.monotonic", lambda: clock[0])
    caplog.set_level(logging.INFO, logger="iot.runner")
    hub = Hub(publisher, db_path=db_path)
    readings = lambda: [r.getMessage() for r in caplog.records if r.getMessage().startswith("讀數")]   # noqa: E731
    monkeypatch.setattr(C, "READING_LOG_INTERVAL_SEC", 0.0)
    hub.handle_message("cat/iot/env/carrier", b'{"temp_c": 25.0}')
    assert readings() == []
    monkeypatch.setattr(C, "READING_LOG_INTERVAL_SEC", 5.0)
    for t in range(0, 11):                                              # 每秒一筆，只在 0、5、10 秒印
        clock[0] = float(t)
        hub.handle_message("cat/iot/env/carrier", f'{{"temp_c": {20 + t}.0, "humidity_pct": 50}}'.encode())
    assert len(readings()) == 3
    assert readings()[0] == "讀數 env/carrier：temp_c=20.0，humidity_pct=50.0，gas_ppm=None，lux=None"


def test_nan_warn_interval(monkeypatch, caplog):
    from iot.sensors.router import SensorRouter
    clock = [0.0]
    monkeypatch.setattr("iot.sensors.router.time.monotonic", lambda: clock[0])
    caplog.set_level(logging.WARNING, logger="iot.sensors.router")
    nan_logs = lambda: [r for r in caplog.records if "nan" in r.getMessage()]   # noqa: E731
    r = SensorRouter()
    monkeypatch.setattr(C, "WARN_REPEAT_SEC", 3.0)
    for t in range(0, 7):
        clock[0] = float(t)
        r.route("cat/iot/env/carrier", b'{"temp_c":nan,"humidity_pct":55}')
    assert len(nan_logs()) == 3                                         # 0、3、6 秒
    caplog.clear()
    monkeypatch.setattr(C, "WARN_REPEAT_SEC", 0.0)
    r2 = SensorRouter()
    for t in range(0, 7):
        clock[0] = float(t)
        r2.route("cat/iot/env/carrier", b'{"temp_c":nan}')
    assert len(nan_logs()) == 1                                         # 0＝只印一次
