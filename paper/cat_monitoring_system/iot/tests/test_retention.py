"""資料保留：超過 DATA_RETENTION_DAYS 天的原始讀數定時刪掉；進食事件、告警不刪；只清這個行程處理的感測器。"""

import pytest

from iot.config import IotHubConfig as C
from iot.runner import Hub
from iot.sensors.base import BodyTempReading, EnvironmentReading
from iot.sensors.weight import WeightChangeEvent
from iot.storage import store

DAY = 86400.0
NOW = 1_800_000_000.0


def _fill(db_path):
    for age_days in (40, 31, 29, 1):
        ts = NOW - age_days * DAY
        store.insert_env(EnvironmentReading("carrier", temp_c=25.0, ts=ts), db_path=db_path)
        store.insert_bodytemp(BodyTempReading("carrier", surface_temp_c=34.0, ts=ts), db_path=db_path)
    store.insert_weight_event(WeightChangeEvent(scale_id="food_bowl", from_g=100.0, to_g=80.0, delta_g=-20.0,
                                                direction="decrease", ts=NOW - 90 * DAY), db_path=db_path)


def test_purge_keeps_recent_and_events(monkeypatch, db_path, publisher):
    monkeypatch.setattr(C, "DATA_RETENTION_DAYS", 30.0)
    monkeypatch.setattr(C, "KINDS", ())
    _fill(db_path)
    deleted = Hub(publisher, db_path=db_path).purge_old_data(now=NOW)
    assert deleted["env_readings"] == 2 and deleted["bodytemp_readings"] == 2
    assert store.row_count("env_readings", db_path) == 2 and store.row_count("bodytemp_readings", db_path) == 2
    assert store.row_count("weight_events", db_path) == 1            # 90 天前的進食事件：不清


def test_purge_only_own_kinds(monkeypatch, db_path, publisher):
    monkeypatch.setattr(C, "DATA_RETENTION_DAYS", 30.0)
    monkeypatch.setattr(C, "KINDS", ("env",))                        # 只處理 env 的行程
    _fill(db_path)
    Hub(publisher, db_path=db_path).purge_old_data(now=NOW)
    assert store.row_count("env_readings", db_path) == 2
    assert store.row_count("bodytemp_readings", db_path) == 4        # 不是它的：不動


def test_retention_zero_disables(monkeypatch, db_path, publisher):
    monkeypatch.setattr(C, "DATA_RETENTION_DAYS", 0.0)
    _fill(db_path)
    assert Hub(publisher, db_path=db_path).purge_old_data(now=NOW) == {}
    assert store.row_count("env_readings", db_path) == 4


def test_purge_in_batches_and_rejects_other_tables(monkeypatch, db_path):
    monkeypatch.setattr(store, "_PURGE_BATCH", 3)
    for i in range(10):
        store.insert_env(EnvironmentReading("x", temp_c=20.0, ts=float(i)), db_path=db_path)
    assert store.purge_older_than(["env_readings"], 8.0, db_path=db_path) == {"env_readings": 8}
    with pytest.raises(ValueError):
        store.purge_older_than(["iot_alerts"], 8.0, db_path=db_path)
