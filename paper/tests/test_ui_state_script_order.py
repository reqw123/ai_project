"""`settings_gui/ui_state.py` 的「獨立腳本工具顯示順序」偏好邏輯。"""

import json

import pytest

from settings_gui import ui_state


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_state, "_STATE_PATH", tmp_path / "gui_state.json")
    monkeypatch.setattr(ui_state, "_LEGACY_STATE_PATH", tmp_path / "_legacy_nope.json")
    return tmp_path


def test_empty_when_no_preference():
    assert ui_state.get_script_order() == []
    assert ui_state.apply_script_order(["a.py", "b.py"]) == ["a.py", "b.py"]


def test_set_and_get_roundtrip():
    ui_state.set_script_order([
        {"key": "b.py", "locked": True},
        {"key": "a.py", "locked": False},
    ])
    assert ui_state.get_script_order() == [
        {"key": "b.py", "locked": True},
        {"key": "a.py", "locked": False},
    ]


def test_apply_puts_preference_first_then_new_scripts_in_scan_order():
    ui_state.set_script_order([{"key": "c.py", "locked": False},
                              {"key": "a.py", "locked": False}])
    # 掃到 a c d e；偏好只涵蓋 c、a → c, a 在前，d e 依掃描順序接後
    assert ui_state.apply_script_order(["a.py", "c.py", "d.py", "e.py"]) == \
        ["c.py", "a.py", "d.py", "e.py"]


def test_new_script_appended_after_locked_and_ordered_scripts():
    # 使用者排過序、鎖了 b.py；之後新增 z.py 到 tools/
    ui_state.set_script_order([
        {"key": "b.py", "locked": True},
        {"key": "a.py", "locked": False},
        {"key": "c.py", "locked": False},
    ])
    ordered = ui_state.apply_script_order(["a.py", "b.py", "c.py", "z.py"])
    assert ordered == ["b.py", "a.py", "c.py", "z.py"]  # z.py 接在最後
    # 鎖定旗標仍讀得到（給 _discover_tool_scripts 的 locked set 用）
    assert {e["key"] for e in ui_state.get_script_order() if e["locked"]} == {"b.py"}


def test_apply_ignores_vanished_scripts_in_preference():
    ui_state.set_script_order([{"key": "gone.py", "locked": True},
                              {"key": "a.py", "locked": False}])
    assert ui_state.apply_script_order(["a.py", "b.py"]) == ["a.py", "b.py"]


def test_set_dedups_and_coerces():
    ui_state.set_script_order([
        {"key": "a.py", "locked": "yes"},
        {"key": "a.py", "locked": False},   # 重複 → 只留第一次
        {"key": "  ", "locked": False},     # 空 key → 丟掉
        "not-a-dict",
    ])
    assert ui_state.get_script_order() == [{"key": "a.py", "locked": True}]


def test_legacy_list_of_str(tmp_path):
    (tmp_path / "gui_state.json").write_text(
        json.dumps({"tool_script_order": ["a.py", "b.py"]}), encoding="utf-8"
    )
    assert ui_state.get_script_order() == [
        {"key": "a.py", "locked": False},
        {"key": "b.py", "locked": False},
    ]


def test_script_order_does_not_disturb_recent_video_paths(tmp_path):
    v = tmp_path / "vid"
    v.mkdir()
    ui_state.add_recent_video_path(str(v))
    ui_state.set_script_order([{"key": "a.py", "locked": False}])
    assert ui_state.get_recent_video_paths() == [str(v)]
    data = json.loads((tmp_path / "gui_state.json").read_text(encoding="utf-8"))
    assert "recent_video_paths" in data and "tool_script_order" in data
