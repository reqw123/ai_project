import pytest

from iot.sensors.base import ParseError
from iot.sensors.environment import EnvironmentParser, EnvironmentSmoother


def test_parse_full_payload():
    r = EnvironmentParser().parse(
        "living_room",
        {"temp_c": 26.4, "humidity_pct": 55, "gas_ppm": 120, "lux": 300},
    )
    assert r.source_id == "living_room"
    assert r.temp_c == 26.4
    assert r.humidity_pct == 55.0
    assert r.gas_ppm == 120.0
    assert r.lux == 300.0
    assert r.kind == "env"


def test_parse_accepts_alias_keys():
    r = EnvironmentParser().parse("s1", {"temperature": 20, "rh": 40})
    assert r.temp_c == 20.0
    assert r.humidity_pct == 40.0
    assert r.gas_ppm is None


def test_parse_rejects_out_of_range():
    with pytest.raises(ParseError):
        EnvironmentParser().parse("s1", {"temp_c": 999})


def test_parse_rejects_non_numeric():
    with pytest.raises(ParseError):
        EnvironmentParser().parse("s1", {"temp_c": "warm"})


def test_parse_rejects_empty_payload():
    with pytest.raises(ParseError):
        EnvironmentParser().parse("s1", {"note": "hello"})


def test_smoother_ewma_converges_and_is_per_source():
    sm = EnvironmentSmoother(alpha=0.5)
    p = EnvironmentParser()
    first = sm.smooth(p.parse("a", {"temp_c": 10}))
    assert first["temp_c"] == 10.0
    second = sm.smooth(p.parse("a", {"temp_c": 20}))
    assert second["temp_c"] == pytest.approx(15.0)
    # 另一個 source 的狀態獨立
    other = sm.smooth(p.parse("b", {"temp_c": 50}))
    assert other["temp_c"] == 50.0


def test_smoother_missing_field_holds_previous():
    sm = EnvironmentSmoother(alpha=0.5)
    p = EnvironmentParser()
    sm.smooth(p.parse("a", {"temp_c": 10, "humidity_pct": 50}))
    out = sm.smooth(p.parse("a", {"temp_c": 20}))
    assert out["humidity_pct"] == 50.0
