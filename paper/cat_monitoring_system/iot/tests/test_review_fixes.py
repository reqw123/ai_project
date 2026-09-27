"""09-27 程式碼審查找到的問題，各一個回歸測試。"""

import logging
import threading

from iot.ingest.mqtt_ingestor import MqttIngestor
from iot.output.alert_engine import AlertEngine
from iot.repeat_limiter import ONGOING, RepeatLimiter
from iot.runner import Hub
from iot.sensors.base import EnvironmentReading
from iot.sensors.health import SensorHealthMonitor
from iot.sensors.router import SensorRouter


# ── #1 多個食盆：第二個之後不能被誤報恢復 ─────────────────────────────────
def test_feeding_silence_multiple_scales_not_falsely_recovered():
    eng = AlertEngine(cooldown_sec=900)
    both_silent = {"a": None, "b": None}
    assert {a.key for a in eng.check_feeding_silence(both_silent, now=0)} == {"feeding.silence.a", "feeding.silence.b"}
    assert eng.drain() == ([], [])                                   # 以前：b 馬上被當成恢復
    eng.check_feeding_silence(both_silent, now=300)
    suppressed, recovered = eng.drain()
    assert recovered == [] and {k for k, _ in suppressed} == {"feeding.silence.a", "feeding.silence.b"}
    eng.check_feeding_silence({"a": None, "b": 299.0}, now=600)      # b 吃了
    assert eng.drain()[1] == [("feeding.silence.b", "疑似食慾不振（b）")]


# ── #2 兩個執行緒同時檢查：不能誤報恢復、不能拋例外 ──────────────────────────
def test_alert_engine_threads_do_not_cross_talk():
    eng = AlertEngine(cooldown_sec=0)
    errors, recovered = [], []

    def env_loop():
        try:
            for i in range(2000):
                eng.check_environment(EnvironmentReading("x", temp_c=40.0), now=i)
                recovered.extend(eng.drain()[1])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def feed_loop():
        try:
            for i in range(2000):
                eng.check_feeding_silence({"a": None}, now=i)
                recovered.extend(eng.drain()[1])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=env_loop), threading.Thread(target=feed_loop)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and recovered == []                          # 溫度一直超標、食盆一直沒吃：不會有恢復


# ── #3 欄位沒值（nan → None）不算恢復 ────────────────────────────────────
def test_missing_field_is_not_recovery():
    eng = AlertEngine(cooldown_sec=900)
    eng.check_environment(EnvironmentReading("carrier", temp_c=35.0, humidity_pct=50), now=0)
    eng.drain()
    eng.check_environment(EnvironmentReading("carrier", humidity_pct=50), now=10)   # 溫度 nan → None
    assert eng.drain() == ([], [])
    eng.check_environment(EnvironmentReading("carrier", temp_c=25.0, humidity_pct=50), now=20)
    assert eng.drain()[1] == [("env.temp_high.carrier", "環境溫度過高（carrier）")]


# ── #4 必填欄位是 nan：解析失敗不洗版，健康監測知道這台在線 ────────────────────
def test_required_nan_not_spammed_and_node_kept_alive(monkeypatch, caplog, db_path, publisher):
    from iot.config import IotHubConfig as C
    monkeypatch.setattr(C, "WARN_REPEAT_SEC", 10.0)
    clock = [0.0]
    monkeypatch.setattr("iot.runner.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("iot.sensors.router.time.monotonic", lambda: clock[0])
    health = SensorHealthMonitor(missing_sec=60, repeat_sec=10)
    hub = Hub(publisher, db_path=db_path, health=health)
    caplog.set_level(logging.WARNING)
    for t in range(0, 70):                                           # MLX90614 沒接：體表溫度每秒 nan
        clock[0] = float(t)
        hub.handle_message("cat/iot/bodytemp/carrier", b'{"surface_temp_c":nan,"ambient_temp_c":nan}')
        hub.check_sensor_health()
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("解析失敗" in m for m in msgs)                     # 以前：每秒一行
    assert sum("有欄位是 nan" in m for m in msgs) == 7                # 0,10,…,60 秒
    assert not any("沒送資料" in m for m in msgs)                     # 有在送，不是離線
    missing = [m for m in msgs if "還在送資料" in m and "surface_temp_c" in m]
    assert missing and "MLX90614" in missing[0]                       # 必填欄位從沒收到過：一樣報「欄位消失」


# ── #5 同種類兩台，其中一台離線 ──────────────────────────────────────────
def test_one_of_two_env_nodes_offline_is_reported():
    m = SensorHealthMonitor(missing_sec=60, repeat_sec=10)
    m.observe(EnvironmentReading("carrier", temp_c=25.0), 0)          # 外出包送一筆後斷電
    out = []
    for t in range(0, 90):
        m.observe(EnvironmentReading("living_room", gas_ppm=300.0, lux=100.0), t)   # esp32_env 照送
        out += [(t, msg) for lv, msg in m.tick(t) if "沒送資料" in msg]
    assert [t for t, _ in out] == [61, 71, 81] and "env/carrier" in out[0][1]
    assert m.observe(EnvironmentReading("carrier", temp_c=25.0), 90)[0] == ("info", "env/carrier 恢復送資料了")


def test_only_node_of_kind_offline_left_to_watchdog():
    m = SensorHealthMonitor(missing_sec=60, repeat_sec=10)
    m.observe(EnvironmentReading("carrier", temp_c=25.0), 0)
    assert all("沒送資料" not in msg for t in range(0, 200) for _, msg in m.tick(t))


# ── #6 broker 拒絕連線：不能當成已連上 ───────────────────────────────────
class _FakeRC:
    def __init__(self, failure):
        self.is_failure = failure


class _FakeClient:
    def __init__(self):
        self.subscribed = []

    def subscribe(self, topic, qos=0):
        self.subscribed.append(topic)


def test_refused_connect_does_not_notify_watchdog(caplog):
    ing = MqttIngestor.__new__(MqttIngestor)          # 不建真的 paho client
    ing._topics = ["cat/iot/env/#"]
    notified = []
    ing._on_connected = lambda: notified.append(1)
    client = _FakeClient()
    ing._handle_connect(client, None, {}, _FakeRC(True), None)          # 2.x：帳密錯
    ing._handle_connect(client, None, {}, 5)                             # 1.x：rc=5
    assert notified == [] and client.subscribed == []
    assert any("拒絕連線" in r.getMessage() for r in caplog.records)
    ing._handle_connect(client, None, {}, _FakeRC(False), None)
    ing._handle_connect(client, None, {}, 0)
    assert notified == [1, 1] and client.subscribed == ["cat/iot/env/#"] * 2


# ── #9 共用節流 ──────────────────────────────────────────────────────
def test_repeat_limiter():
    lim = RepeatLimiter(10)
    assert lim.hit("k", 0) == "" and lim.hit("k", 5) is None and lim.hit("k", 10) == ONGOING
    assert lim.clear("k") and not lim.clear("k") and lim.hit("k", 11) == ""
    once = RepeatLimiter(0)
    assert once.hit("k", 0) == "" and once.hit("k", 1000) is None
    dyn = [10.0]
    lim2 = RepeatLimiter(lambda: dyn[0])
    lim2.hit("k", 0)
    dyn[0] = 2.0
    assert lim2.hit("k", 2) == ONGOING                               # 間隔改了立即生效


def test_router_nan_python_style_also_warned(caplog):
    caplog.set_level(logging.WARNING)
    r = SensorRouter()
    reading = r.route("cat/iot/env/carrier", b'{"temp_c": NaN, "humidity_pct": 50}')
    assert reading.temp_c is None and r.last_nan_node == ("env", "carrier")
    assert any("有欄位是 nan" in rec.getMessage() for rec in caplog.records)
    r.route("cat/iot/env/carrier", b'{"temp_c": 25}')
    assert r.last_nan_node is None
