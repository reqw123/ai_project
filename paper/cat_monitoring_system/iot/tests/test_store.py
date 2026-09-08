import pytest

from iot.sensors.base import (
    BodyTempReading,
    EnvironmentReading,
    MotionEvent,
    WeightReading,
)
from iot.sensors.weight import WeightChangeEvent
from iot.storage import store


def test_insert_and_count_all_tables(db_path):
    store.insert_env(
        EnvironmentReading(source_id="s1", temp_c=25.0, humidity_pct=50.0, ts=1.0),
        db_path=db_path,
    )
    store.insert_motion(MotionEvent(source_id="hall", active=True, ts=2.0), db_path=db_path)
    store.insert_weight(
        WeightReading(source_id="food_bowl", grams=100.0, ts=3.0), db_path=db_path
    )
    store.insert_weight_event(
        WeightChangeEvent(
            scale_id="food_bowl", from_g=100, to_g=90, delta_g=-10,
            direction="decrease", ts=4.0,
        ),
        db_path=db_path,
    )
    store.insert_bodytemp(
        BodyTempReading(source_id="carrier", surface_temp_c=34.0, ambient_temp_c=26.0, ts=6.0),
        db_path=db_path,
    )
    store.insert_alert("k1", "warning", "msg", ts=5.0, value=1.0, threshold=2.0, db_path=db_path)

    assert store.row_count("env_readings", db_path) == 1
    assert store.row_count("motion_events", db_path) == 1
    assert store.row_count("weight_readings", db_path) == 1
    assert store.row_count("weight_events", db_path) == 1
    assert store.row_count("bodytemp_readings", db_path) == 1
    assert store.row_count("iot_alerts", db_path) == 1


def test_last_weight_event_ts_direction_filter(db_path):
    for ts, direction, delta in [(10, "decrease", -5), (20, "increase", 40), (30, "decrease", -8)]:
        store.insert_weight_event(
            WeightChangeEvent(
                scale_id="food_bowl", from_g=0, to_g=0, delta_g=delta,
                direction=direction, ts=ts,
            ),
            db_path=db_path,
        )
    assert store.last_weight_event_ts("food_bowl", db_path=db_path) == 30
    assert store.last_weight_event_ts("food_bowl", "decrease", db_path=db_path) == 30
    assert store.last_weight_event_ts("food_bowl", "increase", db_path=db_path) == 20
    assert store.last_weight_event_ts("water_bowl", db_path=db_path) is None


def test_recent_orders_by_ts_desc(db_path):
    for ts in (1.0, 5.0, 3.0):
        store.insert_env(EnvironmentReading(source_id="s", temp_c=ts, ts=ts), db_path=db_path)
    rows = store.recent("env_readings", limit=2, db_path=db_path)
    assert [r["ts"] for r in rows] == [5.0, 3.0]


def test_row_count_rejects_unknown_table(db_path):
    with pytest.raises(ValueError):
        store.row_count("secret_table", db_path)


def test_db_file_is_isolated_per_path(tmp_path):
    a = str(tmp_path / "a.db")
    b = str(tmp_path / "b.db")
    store.insert_env(EnvironmentReading(source_id="s", temp_c=1.0, ts=1.0), db_path=a)
    assert store.row_count("env_readings", a) == 1
    assert store.row_count("env_readings", b) == 0
    store.close_connection(a)
    store.close_connection(b)
