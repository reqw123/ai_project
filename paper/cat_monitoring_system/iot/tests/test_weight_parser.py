import pytest

from iot.sensors.base import ParseError, WeightReading
from iot.sensors.weight import WeightEventDetector, WeightParser


def test_parse_basic():
    r = WeightParser().parse("food_bowl", {"grams": 123.4})
    assert r.source_id == "food_bowl"
    assert r.grams == 123.4
    assert r.kind == "weight"


def test_parse_alias_and_reject():
    assert WeightParser().parse("s", {"weight": 5}).grams == 5.0
    with pytest.raises(ParseError):
        WeightParser().parse("s", {"foo": 1})
    with pytest.raises(ParseError):
        WeightParser().parse("s", {"grams": 999999})


def _feed(detector, scale_id, series):
    """series: list of (ts, grams) → 回傳每筆對應的 event（或 None）。"""
    out = []
    for ts, g in series:
        out.append(detector.update(WeightReading(source_id=scale_id, grams=g, ts=ts)))
    return out


def test_detects_feeding_after_settle():
    d = WeightEventDetector(event_delta_g=3.0, settle_sec=10.0)
    events = _feed(
        d,
        "food_bowl",
        [(0, 100), (1, 100), (20, 85), (21, 84), (22, 85), (33, 85)],
    )
    assert all(e is None for e in events[:-1])
    ev = events[-1]
    assert ev is not None
    assert ev.direction == "decrease"
    assert ev.delta_g == pytest.approx(-15.0)
    assert ev.from_g == pytest.approx(100.0)
    assert ev.to_g == pytest.approx(85.0)


def test_small_wobble_no_event():
    d = WeightEventDetector(event_delta_g=3.0, settle_sec=5.0)
    events = _feed(d, "food_bowl", [(0, 200), (1, 201), (7, 199), (13, 200.5)])
    assert all(e is None for e in events)


def test_sub_threshold_drift_never_fires():
    d = WeightEventDetector(event_delta_g=3.0, settle_sec=5.0)
    # 緩慢漂移，每筆與基準差都 < 門檻 → 永遠不觸發事件
    events = _feed(
        d, "food_bowl",
        [(0, 50.0), (1, 51.0), (10, 51.5), (20, 52.0), (30, 52.5), (40, 52.0)],
    )
    assert all(e is None for e in events)


def test_refill_is_increase():
    d = WeightEventDetector(event_delta_g=3.0, settle_sec=5.0)
    ev = _feed(d, "food_bowl", [(0, 10), (1, 10), (20, 90), (21, 90), (30, 90)])[-1]
    assert ev is not None
    assert ev.direction == "increase"
    assert ev.delta_g == pytest.approx(80.0)


def test_state_is_per_scale():
    d = WeightEventDetector(event_delta_g=3.0, settle_sec=5.0)
    _feed(d, "food_bowl", [(0, 100)])
    _feed(d, "water_bowl", [(0, 500)])
    ev = _feed(d, "water_bowl", [(20, 480), (21, 480), (30, 480)])[-1]
    assert ev is not None
    assert ev.scale_id == "water_bowl"
    assert ev.from_g == pytest.approx(500.0)
