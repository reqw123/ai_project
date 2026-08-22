"""settings_manager.py 的單元測試。

策略同 test_config_unit.py：不動專案根目錄真正的 runtime_settings.current.json，
用 monkeypatch 把 settings_manager 的模組層路徑常數（RUNTIME_SETTINGS_PATH/
BACKUP_SETTINGS_PATH）指到 tmp_path，並在每個測試前重設模組層快取
（_cache/_cache_load_error），確保測試之間互不干擾、也不干擾其他測試檔案
共用的同一個 settings_manager 模組實例。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import settings_manager as sm


@pytest.fixture(autouse=True)
def _reset_cache_and_redirect(tmp_path, monkeypatch):
    """每個測試都指到獨立的 tmp_path 檔案，並重設模組層快取。"""
    monkeypatch.setattr(sm, "RUNTIME_SETTINGS_PATH", tmp_path / "runtime_settings.current.json")
    monkeypatch.setattr(sm, "BACKUP_SETTINGS_PATH", tmp_path / "runtime_settings.previous.json")
    monkeypatch.setattr(sm, "_cache", None)
    monkeypatch.setattr(sm, "_cache_load_error", None)
    yield


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================================
# load_runtime_settings()
# ============================================================================


class TestLoadRuntimeSettings:
    def test_missing_file_returns_empty_dict(self):
        assert sm.load_runtime_settings() == {}
        assert sm.get_load_error() is None

    def test_valid_file_is_loaded(self):
        _write_json(sm.RUNTIME_SETTINGS_PATH, {"yolo": {"image_size": 800}})
        assert sm.load_runtime_settings() == {"yolo": {"image_size": 800}}
        assert sm.get_load_error() is None

    def test_corrupt_json_returns_empty_dict_with_error_recorded(self):
        sm.RUNTIME_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        sm.RUNTIME_SETTINGS_PATH.write_text("{not valid json", encoding="utf-8")
        assert sm.load_runtime_settings() == {}
        assert sm.get_load_error() is not None

    def test_non_object_json_returns_empty_dict_with_error(self):
        sm.RUNTIME_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        sm.RUNTIME_SETTINGS_PATH.write_text("[1, 2, 3]", encoding="utf-8")
        assert sm.load_runtime_settings() == {}
        assert sm.get_load_error() is not None

    def test_cache_reused_until_force_reload(self):
        _write_json(sm.RUNTIME_SETTINGS_PATH, {"yolo": {"image_size": 111}})
        assert sm.load_runtime_settings()["yolo"]["image_size"] == 111
        _write_json(sm.RUNTIME_SETTINGS_PATH, {"yolo": {"image_size": 222}})
        assert sm.load_runtime_settings()["yolo"]["image_size"] == 111  # 快取，尚未變
        assert sm.reload_runtime_settings()["yolo"]["image_size"] == 222  # 強制重讀


# ============================================================================
# get_runtime_value() / get_runtime_size()
# ============================================================================


class TestGetRuntimeValue:
    def test_missing_key_returns_fallback(self):
        _write_json(sm.RUNTIME_SETTINGS_PATH, {})
        assert sm.get_runtime_value("yolo.image_size", 640, value_type=int) == 640

    def test_present_key_returns_json_value(self):
        _write_json(sm.RUNTIME_SETTINGS_PATH, {"yolo": {"image_size": 800}})
        assert sm.get_runtime_value("yolo.image_size", 640, value_type=int) == 800

    def test_no_file_returns_fallback(self):
        assert sm.get_runtime_value("yolo.confidence_threshold", 0.5, value_type=float) == 0.5

    def test_type_mismatch_falls_back(self):
        _write_json(sm.RUNTIME_SETTINGS_PATH, {"yolo": {"image_size": "not_a_number"}})
        assert sm.get_runtime_value("yolo.image_size", 640, value_type=int) == 640

    def test_int_accepts_integral_float(self):
        _write_json(sm.RUNTIME_SETTINGS_PATH, {"yolo": {"image_size": 800.0}})
        assert sm.get_runtime_value("yolo.image_size", 640, value_type=int) == 800

    def test_no_value_type_returns_raw_value(self):
        _write_json(sm.RUNTIME_SETTINGS_PATH, {"model_paths": {"video_input": 0}})
        assert sm.get_runtime_value("model_paths.video_input", "x") == 0


class TestGetRuntimeSize:
    def test_missing_key_returns_fallback(self):
        _write_json(sm.RUNTIME_SETTINGS_PATH, {})
        assert sm.get_runtime_size("visualization.stream_display_size", None) is None

    def test_explicit_null_returns_none_even_with_non_none_fallback(self):
        _write_json(sm.RUNTIME_SETTINGS_PATH, {"visualization": {"stream_display_size": None}})
        assert sm.get_runtime_size("visualization.stream_display_size", (999, 999)) is None

    def test_valid_size_dict_returns_tuple(self):
        _write_json(
            sm.RUNTIME_SETTINGS_PATH,
            {"visualization": {"stream_display_size": {"width": 480, "height": 360}}},
        )
        assert sm.get_runtime_size("visualization.stream_display_size", None) == (480, 360)

    def test_malformed_size_falls_back(self):
        _write_json(
            sm.RUNTIME_SETTINGS_PATH, {"visualization": {"stream_display_size": {"width": 0}}}
        )
        assert sm.get_runtime_size("visualization.stream_display_size", None) is None


class TestGetSettingsSource:
    def test_env_var_set_wins(self, monkeypatch):
        monkeypatch.setenv("CAT_MONITORING_YOLO_IMAGE_SIZE", "800")
        _write_json(sm.RUNTIME_SETTINGS_PATH, {"yolo": {"image_size": 700}})
        assert sm.get_settings_source("yolo.image_size", "CAT_MONITORING_YOLO_IMAGE_SIZE") == "env"

    def test_json_present_without_env(self, monkeypatch):
        monkeypatch.delenv("CAT_MONITORING_YOLO_IMAGE_SIZE", raising=False)
        _write_json(sm.RUNTIME_SETTINGS_PATH, {"yolo": {"image_size": 700}})
        assert sm.get_settings_source("yolo.image_size", "CAT_MONITORING_YOLO_IMAGE_SIZE") == "json"

    def test_neither_set_is_default(self, monkeypatch):
        monkeypatch.delenv("CAT_MONITORING_YOLO_IMAGE_SIZE", raising=False)
        assert sm.get_settings_source("yolo.image_size", "CAT_MONITORING_YOLO_IMAGE_SIZE") == "default"


# ============================================================================
# validate_settings()
# ============================================================================


class TestValidateSettings:
    def test_empty_dict_is_valid(self):
        ok, errors, warnings = sm.validate_settings({})
        assert ok is True
        assert errors == []

    @pytest.mark.parametrize("port", [1, 80, 65535])
    def test_valid_ports_pass(self, port):
        ok, errors, _ = sm.validate_settings({"flask": {"port": port}})
        assert ok is True

    @pytest.mark.parametrize("port", [0, -1, 65536, 100000])
    def test_invalid_ports_fail(self, port):
        ok, errors, _ = sm.validate_settings({"flask": {"port": port}})
        assert ok is False
        assert errors

    @pytest.mark.parametrize("quality", [1, 50, 100])
    def test_valid_jpeg_quality_passes(self, quality):
        ok, _, _ = sm.validate_settings({"flask": {"jpeg_quality": quality}})
        assert ok is True

    @pytest.mark.parametrize("quality", [0, 101, -5])
    def test_invalid_jpeg_quality_fails(self, quality):
        ok, _, _ = sm.validate_settings({"flask": {"jpeg_quality": quality}})
        assert ok is False

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_unit_interval_boundaries_pass(self, value):
        ok, _, _ = sm.validate_settings({"yolo": {"confidence_threshold": value}})
        assert ok is True

    @pytest.mark.parametrize("value", [-0.01, 1.01, 2.0])
    def test_unit_interval_out_of_range_fails(self, value):
        ok, _, _ = sm.validate_settings({"yolo": {"confidence_threshold": value}})
        assert ok is False

    @pytest.mark.parametrize("value", ["06:00", "23:59", "00:00", ""])
    def test_valid_hhmm_passes(self, value):
        ok, _, _ = sm.validate_settings({"run_mode": {"scheduled_start_time": value}})
        assert ok is True

    @pytest.mark.parametrize("value", ["24:00", "6:00", "06:60", "abc", "0600"])
    def test_invalid_hhmm_fails(self, value):
        ok, _, _ = sm.validate_settings({"run_mode": {"scheduled_start_time": value}})
        assert ok is False

    def test_positive_int_zero_fails(self):
        ok, errors, _ = sm.validate_settings({"yolo": {"image_size": 0}})
        assert ok is False

    def test_positive_int_bool_is_rejected(self):
        """isinstance(True, int) 在 Python 是 True，這裡刻意排除，避免布林值被誤判成合法整數。"""
        ok, errors, _ = sm.validate_settings({"yolo": {"image_size": True}})
        assert ok is False

    def test_nonneg_float_negative_fails(self):
        ok, _, _ = sm.validate_settings(
            {"anomaly_detection": {"still_motion_threshold": -1.0}}
        )
        assert ok is False

    def test_stream_size_null_passes(self):
        ok, _, _ = sm.validate_settings({"visualization": {"stream_display_size": None}})
        assert ok is True

    def test_stream_size_valid_dict_passes(self):
        ok, _, _ = sm.validate_settings(
            {"visualization": {"stream_display_size": {"width": 480, "height": 360}}}
        )
        assert ok is True

    def test_stream_size_zero_dimension_fails(self):
        ok, _, _ = sm.validate_settings(
            {"visualization": {"stream_display_size": {"width": 0, "height": 360}}}
        )
        assert ok is False

    def test_required_file_missing_blocks(self, tmp_path):
        missing = str(tmp_path / "does_not_exist.pt")
        ok, errors, _ = sm.validate_settings({"model_paths": {"yolo_model": missing}})
        assert ok is False
        assert any("YOLO" in e for e in errors)

    def test_required_file_existing_passes(self, tmp_path):
        real_file = tmp_path / "model.pt"
        real_file.write_text("dummy")
        ok, _, _ = sm.validate_settings({"model_paths": {"yolo_model": str(real_file)}})
        assert ok is True

    def test_optional_file_missing_warns_but_does_not_block(self, tmp_path):
        missing = str(tmp_path / "no_such_profile.json")
        ok, errors, warnings = sm.validate_settings(
            {"cat_identity": {"target_cat_profile_path": missing}}
        )
        assert ok is True
        assert errors == []
        assert warnings

    def test_optional_file_empty_string_is_fine(self):
        ok, errors, warnings = sm.validate_settings(
            {"cat_identity": {"other_cat_profile_path": ""}}
        )
        assert ok is True
        assert warnings == []

    def test_output_path_existing_parent_is_fine(self, tmp_path):
        target = str(tmp_path / "out.csv")
        ok, errors, warnings = sm.validate_settings({"logging": {"csv_path": target}})
        assert ok is True
        assert errors == []
        assert warnings == []

    def test_output_path_creatable_parent_warns(self, tmp_path):
        target = str(tmp_path / "not_yet_created" / "out.csv")
        ok, errors, warnings = sm.validate_settings({"logging": {"csv_path": target}})
        assert ok is True
        assert warnings

    def test_enum_invalid_choice_fails(self):
        ok, errors, _ = sm.validate_settings({"run_mode": {"mode": "not_a_mode"}})
        assert ok is False

    def test_enum_valid_choice_passes(self):
        ok, _, _ = sm.validate_settings({"run_mode": {"mode": "gui"}})
        assert ok is True

    def test_sensitive_value_is_redacted_in_error_message(self):
        ok, errors, _ = sm.validate_settings(
            {"run_mode": {"mode": "rtsp://user:secretpass@host/stream"}}
        )
        assert ok is False
        joined = " ".join(errors)
        assert "secretpass" not in joined
        assert "***" in joined


# ============================================================================
# save_runtime_settings() / export_settings() / import_settings()
# ============================================================================


class TestSaveRuntimeSettings:
    def test_invalid_data_is_not_written(self):
        ok, errors, _ = sm.save_runtime_settings({"flask": {"port": 999999}})
        assert ok is False
        assert errors
        assert not sm.RUNTIME_SETTINGS_PATH.exists()

    def test_valid_data_is_written_atomically(self):
        ok, errors, _ = sm.save_runtime_settings({"yolo": {"image_size": 800}})
        assert ok is True
        assert errors == []
        assert sm.RUNTIME_SETTINGS_PATH.exists()
        with open(sm.RUNTIME_SETTINGS_PATH, "r", encoding="utf-8") as f:
            assert json.load(f) == {"yolo": {"image_size": 800}}

    def test_second_save_creates_backup_of_previous_content(self):
        sm.save_runtime_settings({"yolo": {"image_size": 800}})
        sm.save_runtime_settings({"yolo": {"image_size": 900}})
        assert sm.BACKUP_SETTINGS_PATH.exists()
        with open(sm.BACKUP_SETTINGS_PATH, "r", encoding="utf-8") as f:
            assert json.load(f) == {"yolo": {"image_size": 800}}
        with open(sm.RUNTIME_SETTINGS_PATH, "r", encoding="utf-8") as f:
            assert json.load(f) == {"yolo": {"image_size": 900}}

    def test_no_temp_file_left_behind(self):
        sm.save_runtime_settings({"yolo": {"image_size": 800}})
        leftovers = list(sm.RUNTIME_SETTINGS_PATH.parent.glob(".runtime_settings_*.tmp"))
        assert leftovers == []

    def test_save_updates_module_cache(self):
        sm.save_runtime_settings({"yolo": {"image_size": 800}})
        assert sm.load_runtime_settings() == {"yolo": {"image_size": 800}}


class TestExportImportSettings:
    def test_export_writes_validated_json(self, tmp_path):
        target = tmp_path / "exported" / "out.json"
        ok, errors, _ = sm.export_settings({"yolo": {"image_size": 800}}, target)
        assert ok is True
        assert errors == []
        assert json.loads(target.read_text(encoding="utf-8")) == {"yolo": {"image_size": 800}}

    def test_export_invalid_data_does_not_write(self, tmp_path):
        target = tmp_path / "out.json"
        ok, errors, _ = sm.export_settings({"flask": {"port": -1}}, target)
        assert ok is False
        assert not target.exists()

    def test_import_valid_file(self, tmp_path):
        src = tmp_path / "import_me.json"
        _write_json(src, {"yolo": {"image_size": 800}})
        ok, data, errors, warnings = sm.import_settings(src)
        assert ok is True
        assert data == {"yolo": {"image_size": 800}}
        assert errors == []

    def test_import_invalid_data_reports_errors_without_raising(self, tmp_path):
        src = tmp_path / "bad.json"
        _write_json(src, {"flask": {"port": -1}})
        ok, data, errors, _ = sm.import_settings(src)
        assert ok is False
        assert errors

    def test_import_missing_file_reports_error(self, tmp_path):
        ok, data, errors, _ = sm.import_settings(tmp_path / "nope.json")
        assert ok is False
        assert errors


class TestDiffSettings:
    def test_no_changes_returns_empty_list(self):
        old = {"yolo": {"image_size": 640}}
        assert sm.diff_settings(old, old) == []

    def test_changed_field_is_reported(self):
        old = {"yolo": {"image_size": 640}}
        new = {"yolo": {"image_size": 800}}
        changes = sm.diff_settings(old, new)
        keys = [c[0] for c in changes]
        assert "yolo.image_size" in keys

    def test_missing_vs_present_counts_as_change(self):
        changes = sm.diff_settings({}, {"yolo": {"image_size": 800}})
        assert any(c[0] == "yolo.image_size" for c in changes)


# ============================================================================
# 一致性測試：default_runtime_settings.json 不應跟 config.py 的硬編碼預設值漂移
# ============================================================================


class TestDefaultRuntimeSettingsConsistency:
    """在乾淨環境（無 CAT_MONITORING_* 環境變數、runtime_settings.current.json 暫時搬開）
    的子行程重新 import config，逐一比對 FIELD_SCHEMA 每筆 attr 的值是否等於
    default_runtime_settings.json 對應欄位——防止兩邊日後各自被改動而漂移。

    用子行程而非 importlib.reload(config)：config.py 的 class 只會在模組第一次
    載入時執行一次，同一行程內 reload 會產生新的 class 物件，其他早就
    `from config import XxxConfig` 的模組會拿到舊物件，drift 檢查用子行程隔離
    最乾淨，也不會影響同一個 pytest session 裡其他測試檔案。
    """

    def test_config_defaults_match_default_runtime_settings_json(self, tmp_path):
        paper_dir = Path(__file__).resolve().parents[1]
        real_runtime_settings = paper_dir / "runtime_settings.current.json"
        real_backup = paper_dir / "runtime_settings.previous.json"
        moved_aside = tmp_path / "runtime_settings.current.json.moved_for_test"

        had_real_file = real_runtime_settings.exists()
        if had_real_file:
            real_runtime_settings.rename(moved_aside)
        try:
            clean_env = {
                k: v for k, v in os.environ.items() if not k.startswith("CAT_MONITORING_")
            }
            script = (
                "import json, sys\n"
                "sys.path.insert(0, r'" + str(paper_dir) + "')\n"
                "import config\n"
                "from settings_manager import FIELD_SCHEMA\n"
                "out = {}\n"
                "for f in FIELD_SCHEMA:\n"
                "    cls_name, attr_name = f['attr']\n"
                "    value = getattr(getattr(config, cls_name), attr_name)\n"
                "    out[f['json_key']] = value\n"
                "print(json.dumps(out))\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(paper_dir),
                env=clean_env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, result.stderr
            resolved = json.loads(result.stdout.strip().splitlines()[-1])
        finally:
            if had_real_file:
                if real_runtime_settings.exists():
                    real_runtime_settings.unlink()
                moved_aside.rename(real_runtime_settings)
            if real_backup.exists():
                real_backup.unlink()

        defaults = json.loads(
            (paper_dir / "default_runtime_settings.json").read_text(encoding="utf-8")
        )
        mismatches = []
        for field in sm.FIELD_SCHEMA:
            key = field["json_key"]
            expected = sm._get_nested(defaults, key)
            actual = resolved.get(key)
            if field["value_type"] == "size":
                if expected is None:
                    expected_norm = None
                else:
                    expected_norm = (expected["width"], expected["height"])
                actual_norm = tuple(actual) if isinstance(actual, list) else actual
                if expected_norm != actual_norm:
                    mismatches.append((key, expected_norm, actual_norm))
            elif expected != actual:
                mismatches.append((key, expected, actual))
        assert mismatches == [], f"config.py 硬編碼預設值與 default_runtime_settings.json 不一致：{mismatches}"
