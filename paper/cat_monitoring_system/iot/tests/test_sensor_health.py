"""ESP32 在線、但感測器沒接好：欄位消失、卡在極端值、韌體送 nan。"""

from iot.sensors.base import BodyTempReading, EnvironmentReading
from iot.sensors.health import SensorHealthMonitor
from iot.sensors.router import SensorRouter

RAILS = {"gas_ppm": (0.0, 1000.0), "lux": (None, 2000.0)}
EXPECTED = {"env/carrier": ("temp_c", "humidity_pct")}


def _mon(repeat_sec=0):
    return SensorHealthMonitor(missing_sec=60, rail_count=3, rails=RAILS, expected=EXPECTED, repeat_sec=repeat_sec)


def test_missing_field_repeats_while_ongoing():
    """09-27：異常沒解除就一直印（每 repeat 秒），恢復後停止。"""
    m = _mon(repeat_sec=10)
    for t in range(0, 100):
        m.observe(EnvironmentReading("carrier", temp_c=25.0), t)
        msgs = m.tick(t)
        if t in (60, 70, 80, 90):
            assert len(msgs) == 1 and "humidity_pct" in msgs[0][1], t
            assert ("（持續中）" in msgs[0][1]) == (t != 60)
        else:
            assert msgs == [], t
    assert m.observe(EnvironmentReading("carrier", temp_c=25.0, humidity_pct=50.0), 100)[0][0] == "info"
    assert m.tick(115) == []


def test_rail_repeats_while_ongoing():
    m = _mon(repeat_sec=10)
    out = [m.observe(EnvironmentReading("living_room", gas_ppm=0.0), t) for t in range(0, 50, 5)]
    warned_at = [t for t, msgs in zip(range(0, 50, 5), out) if msgs]
    assert warned_at == [10, 20, 30, 40]                      # 第 3 筆開始，之後每 10 秒


def test_dht_unplugged_from_the_start_is_caught_via_expected_fields():
    m = _mon()
    for t in range(0, 61):                                   # 外出包每秒送，只有溫度、沒有濕度
        m.observe(EnvironmentReading("carrier", temp_c=25.0), t)
    assert m.tick(59) == []
    msgs = m.tick(60)
    assert len(msgs) == 1 and msgs[0][0] == "warning"
    assert "humidity_pct" in msgs[0][1] and "DHT11" in msgs[0][1]
    assert m.tick(61) == []                                   # 只警告一次
    rec = m.observe(EnvironmentReading("carrier", temp_c=25.0, humidity_pct=50.0), 62)
    assert rec and rec[0][0] == "info" and "恢復" in rec[0][1]


def test_field_disappears_after_being_seen_on_unlisted_node():
    m = _mon()
    m.observe(EnvironmentReading("kitchen", gas_ppm=300.0, lux=100.0), 0)
    for t in range(1, 70):
        m.observe(EnvironmentReading("kitchen", gas_ppm=300.0), t)   # 光敏電阻那欄不見了
    msgs = m.tick(69)
    assert [lv for lv, _ in msgs] == ["warning"] and "lux" in msgs[0][1] and "光敏電阻" in msgs[0][1]


def test_unlisted_node_never_seen_field_is_not_guessed():
    m = _mon()
    for t in range(0, 100):
        m.observe(EnvironmentReading("kitchen", gas_ppm=300.0), t)
    assert m.tick(100) == []                                  # 沒列預期欄位、也沒看過 lux → 不亂猜


def test_whole_node_silent_is_left_to_no_data_watchdog():
    m = _mon()
    m.observe(EnvironmentReading("carrier", temp_c=25.0), 0)
    assert m.tick(500) == []                                  # 整台都沒在送：不是「欄位消失」


def test_rail_values():
    m = _mon()
    assert m.observe(EnvironmentReading("living_room", gas_ppm=0.0, lux=0.0), 0) == []
    assert m.observe(EnvironmentReading("living_room", gas_ppm=0.0, lux=0.0), 10) == []
    msgs = m.observe(EnvironmentReading("living_room", gas_ppm=0.0, lux=0.0), 20)
    assert len(msgs) == 1 and "gas_ppm" in msgs[0][1] and "MQ-135" in msgs[0][1]   # lux=0（全暗）不算
    assert m.observe(EnvironmentReading("living_room", gas_ppm=0.0), 30) == []       # 只警告一次
    rec = m.observe(EnvironmentReading("living_room", gas_ppm=250.0), 40)
    assert rec and rec[0][0] == "info"
    for t in (50, 60):
        m.observe(EnvironmentReading("living_room", lux=2000.0), t)
    assert "lux" in m.observe(EnvironmentReading("living_room", lux=2000.0), 70)[0][1]


def test_bodytemp_ambient_missing():
    m = SensorHealthMonitor(missing_sec=60, rail_count=3)
    m.observe(BodyTempReading("carrier", surface_temp_c=34.0, ambient_temp_c=26.0), 0)
    for t in range(1, 61):
        m.observe(BodyTempReading("carrier", surface_temp_c=34.0), t)
    assert "ambient_temp_c" in m.tick(60)[0][1]


def test_router_nan_is_treated_as_missing_not_dropped():
    """Arduino String(NAN) 會送出小寫 nan：以前整筆丟掉（連濕度一起），現在當作沒有值、其他欄位照收。"""
    r = SensorRouter()
    reading = r.route("cat/iot/env/carrier", b'{"temp_c":nan,"humidity_pct":55.0}')
    assert reading is not None and reading.temp_c is None and reading.humidity_pct == 55.0
    reading = r.route("cat/iot/env/carrier", b'{"temp_c":NaN,"humidity_pct":40}')
    assert reading.temp_c is None and reading.humidity_pct == 40
    reading = r.route("cat/iot/env/carrier", b'{"temp_c":25.5, "humidity_pct": -inf}')
    assert reading.temp_c == 25.5 and reading.humidity_pct is None
    assert r.route("cat/iot/env/carrier", b'{"temp_c": 25,,}') is None   # 真的壞掉的 JSON 還是丟掉
