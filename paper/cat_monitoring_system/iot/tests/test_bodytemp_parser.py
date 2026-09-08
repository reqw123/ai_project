import pytest

from iot.sensors.base import ParseError
from iot.sensors.bodytemp import BodyTempParser


def test_parse_basic():
    r = BodyTempParser().parse(
        "carrier", {"surface_temp_c": 34.2, "ambient_temp_c": 26.0}
    )
    assert r.source_id == "carrier"
    assert r.surface_temp_c == 34.2
    assert r.ambient_temp_c == 26.0
    assert r.kind == "bodytemp"


def test_parse_object_alias():
    # 舊專案 寵物包/mqtt_all 送的是 {"object":..,"ambient":..}
    r = BodyTempParser().parse("carrier", {"object": 33.0, "ambient": 25.0})
    assert r.surface_temp_c == 33.0
    assert r.ambient_temp_c == 25.0


def test_ambient_optional():
    r = BodyTempParser().parse("carrier", {"surface_temp_c": 30.0})
    assert r.ambient_temp_c is None


def test_missing_surface_rejected():
    with pytest.raises(ParseError):
        BodyTempParser().parse("carrier", {"ambient_temp_c": 25.0})


def test_out_of_range_rejected():
    with pytest.raises(ParseError):
        BodyTempParser().parse("carrier", {"surface_temp_c": 120})
