"""
Unit Test：settings_gui/iot_config_overrides.py（設定視窗 IoT 分頁「⚙ 參數設定」：覆寫 iot/config.py）

預設值是用 ast 讀真的 iot/config.py（不 import iot 套件）；覆寫檔換到暫存資料夾。
啟動測試用假服務（只印出環境變數就結束），不會碰到使用者真的 python -m iot。
"""

import time

import pytest

from settings_gui import iot_config_overrides as ov
from settings_gui import iot_services as svc


@pytest.fixture
def tmp_overrides(tmp_path, monkeypatch):
    path = tmp_path / "overrides.json"
    monkeypatch.setattr(ov, "OVERRIDES_FILE", path)
    return path


def test_defaults_read_from_real_config():
    d = ov.read_defaults()
    assert d["CAT_MONITORING_IOT_TEMP_MAX_C"].type == "float"
    assert d["CAT_MONITORING_IOT_TEMP_MAX_C"].value == 28.0
    assert d["CAT_MONITORING_IOT_MQTT_PORT"].type == "int"
    assert d["CAT_MONITORING_IOT_DB_PATH"].value.endswith("iot_hub.db")   # 運算式：照 config.py 的算法算出實際路徑


def test_fields_cover_every_env_var_in_config():
    """config.py 新增可覆寫的環境變數卻沒列進 FIELDS → 設定視窗看不到它，這裡抓出來。"""
    in_config = set(ov.read_defaults()) - ov.EXCLUDED
    listed = {f.env for f in ov.FIELDS}
    assert in_config == listed, f"缺：{in_config - listed}；多：{listed - in_config}"


def test_validate_types_and_blank():
    clean, errors = ov.validate({"CAT_MONITORING_IOT_TEMP_MAX_C": " 30 ", "CAT_MONITORING_IOT_GAS_PPM_MAX": "",
                                 "CAT_MONITORING_IOT_MQTT_PORT": "1884"})
    assert errors == []
    assert clean == {"CAT_MONITORING_IOT_TEMP_MAX_C": "30", "CAT_MONITORING_IOT_MQTT_PORT": "1884"}   # 空白不存
    _, errors = ov.validate({"CAT_MONITORING_IOT_TEMP_MAX_C": "abc", "CAT_MONITORING_IOT_MQTT_PORT": "1.5"})
    assert len(errors) == 2


def test_validate_min_below_max_uses_defaults():
    _, errors = ov.validate({"CAT_MONITORING_IOT_TEMP_MIN_C": "29"})     # 預設上限 28
    assert any("環境溫度下限" in e for e in errors)
    _, errors = ov.validate({"CAT_MONITORING_IOT_TEMP_MIN_C": "29", "CAT_MONITORING_IOT_TEMP_MAX_C": "32"})
    assert errors == []
    _, errors = ov.validate({"CAT_MONITORING_IOT_ENV_EWMA_ALPHA": "1.5"})
    assert errors


def test_save_load_roundtrip_ignores_excluded(tmp_overrides):
    ov.save({"CAT_MONITORING_IOT_TEMP_MAX_C": "30"})
    assert ov.load() == {"CAT_MONITORING_IOT_TEMP_MAX_C": "30"}
    tmp_overrides.write_text('{"CAT_MONITORING_IOT_KINDS": "env", "OTHER": "x", "CAT_MONITORING_IOT_GAS_PPM_MAX": "800"}',
                             encoding="utf-8")
    assert ov.load() == {"CAT_MONITORING_IOT_GAS_PPM_MAX": "800"}
    tmp_overrides.write_text("壞掉的 json", encoding="utf-8")
    assert ov.load() == {}


def test_start_passes_overrides_to_sensor_but_not_voice(tmp_path, tmp_overrides, monkeypatch):
    ov.save({"CAT_MONITORING_IOT_TEMP_MAX_C": "31", "CAT_MONITORING_IOT_MQTT_PASSWORD": "secret"})
    mods = tmp_path / "mods"
    mods.mkdir()
    (mods / "fake_env_dump.py").write_text(
        "import os, time\nprint('TEMP_MAX=' + os.getenv('CAT_MONITORING_IOT_TEMP_MAX_C', '-'), flush=True)\n"
        "time.sleep(30)\n", encoding="utf-8")
    env = {"PYTHONPATH": str(mods)}
    services = (
        svc.Service(key="ka", title="假感測器", desc="", module="fake_env_dump", kind="ka", env={**env, svc._KINDS_ENV: "ka"}),
        svc.Service(key="va", title="假語音", desc="", module="fake_env_dump", env=env),
    )
    monkeypatch.setattr(svc, "SERVICES", services)
    monkeypatch.setattr(svc, "_BY_KEY", {s.key: s for s in services})
    monkeypatch.setattr(svc, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(svc, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.delenv("CAT_MONITORING_IOT_TEMP_MAX_C", raising=False)
    try:
        # 兩個假服務同一個模組：scan 會把它們當同一種；分開啟動、分開檢查
        assert svc.start("ka")[0]
        end = time.time() + 10
        while "TEMP_MAX=" not in svc.tail("ka") and time.time() < end:
            time.sleep(0.1)
        log = svc.tail("ka")
        assert "TEMP_MAX=31" in log
        assert "參數覆寫：TEMP_MAX_C = 31" in log and "secret" not in log   # 密碼不寫進紀錄
    finally:
        for s in services:
            st = svc.scan()[s.key]
            if st.running:
                svc.stop(s.key, pid=st.pid)
    # 語音紀錄：不帶 iot 覆寫（直接驗環境組法，不再開行程）
    assert svc.get("va").kind == ""


def test_broker_reachable_uses_override(tmp_overrides, monkeypatch):
    ov.save({"CAT_MONITORING_IOT_MQTT_HOST": "127.0.0.1", "CAT_MONITORING_IOT_MQTT_PORT": "1"})
    ok, where = svc.broker_reachable(timeout=0.2)
    assert where == "127.0.0.1:1" and not ok


def test_legacy_keys_migrate_to_merged_vars(tmp_path, monkeypatch):
    """09-27 合併變數：覆寫檔裡的舊名稱自動換成新的（重印間隔取最小＝印得最勤的）；新名稱已有值時以新的為準。"""
    import json
    from settings_gui import iot_config_overrides as ov
    monkeypatch.setattr(ov, "OVERRIDES_FILE", tmp_path / "ov.json")
    ov.OVERRIDES_FILE.write_text(json.dumps({
        "CAT_MONITORING_IOT_NO_DATA_CHECK_INTERVAL_SEC": "30",
        "CAT_MONITORING_IOT_SENSOR_NAN_WARN_SEC": "1",
        "CAT_MONITORING_IOT_SENSOR_FIELD_MISSING_SEC": "45",
        "CAT_MONITORING_IOT_TEMP_MAX_C": "30",
    }), encoding="utf-8")
    assert ov.load() == {
        "CAT_MONITORING_IOT_WARN_REPEAT_SEC": "1",
        "CAT_MONITORING_IOT_DATA_TIMEOUT_SEC": "45",
        "CAT_MONITORING_IOT_TEMP_MAX_C": "30",
    }
    ov.OVERRIDES_FILE.write_text(json.dumps({
        "CAT_MONITORING_IOT_WARN_REPEAT_SEC": "20", "CAT_MONITORING_IOT_ALERT_LOG_REPEAT_SEC": "1"}), encoding="utf-8")
    assert ov.load() == {"CAT_MONITORING_IOT_WARN_REPEAT_SEC": "20"}
    assert all(old not in {f.env for f in ov.FIELDS} for old in ov.LEGACY)   # 畫面上不再列舊變數


def test_db_path_default_is_shown_as_real_path():
    """DB_PATH 的預設在 config.py 是運算式（ast 讀不到）：參數設定要顯示實際路徑，不是「依程式計算」。"""
    from settings_gui import iot_config_overrides as ov
    d = ov.read_defaults()["CAT_MONITORING_IOT_DB_PATH"]
    assert d.text.endswith("iot_hub.db") and (ov.CONFIG_PATH.parent / "data" / "iot_hub.db") == __import__("pathlib").Path(d.text)
