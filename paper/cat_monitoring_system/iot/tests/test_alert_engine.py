import pytest

from iot.config import IotHubConfig as C
from iot.output.alert_engine import AlertEngine
from iot.sensors.base import BodyTempReading, EnvironmentReading


def _env(**kw):
    kw.setdefault("source_id", "s1")
    kw.setdefault("ts", 0.0)
    return EnvironmentReading(**kw)


def test_temp_high_alert():
    eng = AlertEngine(cooldown_sec=100.0)
    alerts = eng.check_environment(_env(temp_c=C.TEMP_MAX_C + 5), now=1000.0)
    assert len(alerts) == 1
    assert alerts[0].severity == "warning"
    assert alerts[0].key.startswith("env.temp_high.")
    assert alerts[0].value == C.TEMP_MAX_C + 5


def test_gas_is_critical():
    eng = AlertEngine(cooldown_sec=100.0)
    alerts = eng.check_environment(_env(gas_ppm=C.GAS_PPM_MAX + 1), now=1.0)
    assert alerts[0].severity == "critical"


def test_within_range_no_alert():
    eng = AlertEngine(cooldown_sec=100.0)
    mid_t = (C.TEMP_MIN_C + C.TEMP_MAX_C) / 2
    mid_h = (C.HUMIDITY_MIN_PCT + C.HUMIDITY_MAX_PCT) / 2
    assert eng.check_environment(_env(temp_c=mid_t, humidity_pct=mid_h, gas_ppm=0), now=1.0) == []


def test_cooldown_suppresses_repeat():
    eng = AlertEngine(cooldown_sec=100.0)
    r = _env(temp_c=C.TEMP_MAX_C + 5)
    assert len(eng.check_environment(r, now=0.0)) == 1
    assert eng.check_environment(r, now=50.0) == []       # 冷卻中
    assert len(eng.check_environment(r, now=101.0)) == 1  # 冷卻過了


def test_bodytemp_high_and_low():
    eng = AlertEngine(cooldown_sec=1.0)
    hi = eng.check_bodytemp(
        BodyTempReading(source_id="c", surface_temp_c=C.BODYTEMP_SURFACE_MAX_C + 3, ts=0),
        now=0.0,
    )
    assert len(hi) == 1 and hi[0].key == "bodytemp.high.c"
    lo = eng.check_bodytemp(
        BodyTempReading(source_id="c", surface_temp_c=C.BODYTEMP_SURFACE_MIN_C - 3, ts=0),
        now=10.0,
    )
    assert len(lo) == 1 and lo[0].key == "bodytemp.low.c"


def test_bodytemp_normal_no_alert():
    eng = AlertEngine(cooldown_sec=1.0)
    mid = (C.BODYTEMP_SURFACE_MIN_C + C.BODYTEMP_SURFACE_MAX_C) / 2
    assert eng.check_bodytemp(
        BodyTempReading(source_id="c", surface_temp_c=mid, ts=0), now=0.0
    ) == []


def test_feeding_silence_alert():
    eng = AlertEngine(cooldown_sec=1.0)
    now = 1_000_000.0
    silence = C.FEEDING_SILENCE_HOURS * 3600.0
    # 從未進食
    a1 = eng.check_feeding_silence({"food_bowl": None}, now=now)
    assert len(a1) == 1 and a1[0].key == "feeding.silence.food_bowl"
    # 剛吃過 → 無告警
    a2 = eng.check_feeding_silence({"food_bowl": now - 60}, now=now + 3600)
    assert a2 == []
    # 超過門檻
    a3 = eng.check_feeding_silence({"food_bowl": now - silence - 10}, now=now + silence + 10)
    assert len(a3) == 1
