from iot.sensors.base import (
    BodyTempReading,
    EnvironmentReading,
    MotionEvent,
    WeightReading,
)
from iot.sensors.router import SensorRouter


def test_routes_env():
    r = SensorRouter().route(
        "cat/iot/env/living_room", b'{"temp_c": 25, "humidity_pct": 50}'
    )
    assert isinstance(r, EnvironmentReading)
    assert r.source_id == "living_room"
    assert r.temp_c == 25.0


def test_routes_motion_and_weight():
    m = SensorRouter().route("cat/iot/motion/hall", b'{"active": true}')
    assert isinstance(m, MotionEvent) and m.active is True
    w = SensorRouter().route("cat/iot/weight/food_bowl", b'{"grams": 88.5}')
    assert isinstance(w, WeightReading) and w.grams == 88.5


def test_source_id_may_contain_slashes():
    r = SensorRouter().route("cat/iot/env/floor2/bedroom", b'{"temp_c": 22}')
    assert isinstance(r, EnvironmentReading)
    assert r.source_id == "floor2/bedroom"


def test_routes_bodytemp():
    r = SensorRouter().route(
        "cat/iot/bodytemp/carrier", b'{"surface_temp_c": 34.0, "ambient_temp_c": 26.0}'
    )
    assert isinstance(r, BodyTempReading)
    assert r.surface_temp_c == 34.0


def test_unknown_kind_returns_none():
    assert SensorRouter().route("cat/iot/radiation/x", b'{"v": 1}') is None


def test_foreign_prefix_returns_none():
    assert SensorRouter().route("home/other/env/x", b'{"temp_c": 20}') is None


def test_bad_json_returns_none():
    assert SensorRouter().route("cat/iot/env/x", b"not json") is None


def test_json_array_returns_none():
    assert SensorRouter().route("cat/iot/env/x", b"[1,2,3]") is None


def test_parse_error_returns_none_not_raise():
    # temp 超出合理範圍 → parser 拋 ParseError → router 吞掉回 None
    assert SensorRouter().route("cat/iot/env/x", b'{"temp_c": 5000}') is None


def test_custom_prefix():
    r = SensorRouter(topic_prefix="lab/sensors").route(
        "lab/sensors/env/room1", b'{"temp_c": 21}'
    )
    assert isinstance(r, EnvironmentReading)
