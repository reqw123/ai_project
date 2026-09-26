"""
Unit Test：settings_gui/recent_paths.py 的「最近影片路徑」清單

重點在：新路徑插在未鎖定項目最前面、重複的會被提前（不分大小寫、\\ 與 / 視為相同）、
未鎖定超過上限會丟掉最舊的、鎖定項目位置完全固定（新增／移動／刪除都不會動到它、
也不會被擠掉）、存檔後讀得回來、第一次使用時拿舊的 last_tool_video_path 當種子。
全用 tmp_path，不碰真實的 ui_state.json。
"""

import json

import pytest

from settings_gui import recent_paths as rp
from settings_gui import ui_state


@pytest.fixture
def state_path(tmp_path, monkeypatch):
    p = tmp_path / "ui_state.json"
    monkeypatch.setattr(ui_state, "_STATE_PATH", p)
    return p


# ── normalize ─────────────────────────────────────────────────────────


def test_normalize_strips_quotes_and_unifies_slashes():
    assert rp.normalize('  "D:\\videos\\a b.mp4"  ') == "D:/videos/a b.mp4"
    assert rp.normalize("D:/videos/sub/../a.mp4") == "D:/videos/a.mp4"
    assert rp.normalize("   ") == ""
    assert rp.normalize(None) == ""


# ── add ───────────────────────────────────────────────────────────────


def test_add_puts_new_path_first():
    assert rp.add(["D:/a", "D:/b"], [], "D:/c") == ["D:/c", "D:/a", "D:/b"]


def test_add_existing_path_moves_to_front_case_and_slash_insensitive():
    assert rp.add(["D:/a", "D:/B/x.mp4"], [], "d:\\b\\X.MP4") == ["d:/b/X.MP4", "D:/a"]


def test_add_trims_oldest_unlocked_beyond_limit():
    paths = [f"D:/{i}" for i in range(3)]
    assert rp.add(paths, [], "D:/new", max_recent=3) == ["D:/new", "D:/0", "D:/1"]


def test_add_keeps_locked_position_and_never_evicts_locked():
    paths = ["D:/a", "D:/L", "D:/b", "D:/c"]
    out = rp.add(paths, ["D:/L"], "D:/new", max_recent=3)
    assert out == ["D:/new", "D:/L", "D:/a", "D:/b"]  # 鎖定的仍在第 2 位，最舊的 D:/c 被丟掉


def test_add_locked_path_is_noop():
    paths = ["D:/a", "D:/L"]
    assert rp.add(paths, ["D:/L"], "d:\\l") == paths


def test_add_empty_is_noop():
    assert rp.add(["D:/a"], [], "  ") == ["D:/a"]


# ── move / remove ─────────────────────────────────────────────────────


def test_move_jumps_over_locked_items():
    paths = ["D:/a", "D:/L", "D:/b"]
    out, j = rp.move(paths, ["D:/L"], 0, 1)
    assert out == ["D:/b", "D:/L", "D:/a"] and j == 2


def test_move_locked_or_at_edge_is_refused():
    paths = ["D:/a", "D:/L", "D:/b"]
    assert rp.move(paths, ["D:/L"], 1, 1) == (paths, None)
    assert rp.move(paths, ["D:/L"], 0, -1) == (paths, None)
    assert rp.move(["D:/L", "D:/a"], ["D:/L"], 1, -1) == (["D:/L", "D:/a"], None)


def test_remove_keeps_locked_in_place_and_refuses_locked():
    paths = ["D:/a", "D:/b", "D:/L", "D:/c"]
    assert rp.remove(paths, ["D:/L"], 0) == ["D:/b", "D:/c", "D:/L"]  # 鎖定的仍在第 3 位
    assert rp.remove(paths, ["D:/L"], 2) == paths


def test_remove_last_unlocked_before_trailing_locked():
    # 清單變短、鎖定項目原位置超出長度 → 依序接在最後
    assert rp.remove(["D:/a", "D:/L"], ["D:/L"], 0) == ["D:/L"]


# ── 存檔 ──────────────────────────────────────────────────────────────


def test_load_seeds_from_last_tool_video_path(state_path):
    state_path.write_text(json.dumps({"last_tool_video_path": "D:\\v\\a.mp4"}), encoding="utf-8")
    assert rp.load() == (["D:/v/a.mp4"], [])


def test_record_and_save_roundtrip(state_path):
    rp.record("D:/a.mp4")
    rp.record("D:/b")
    paths, locked = rp.load()
    assert paths == ["D:/b", "D:/a.mp4"] and locked == []
    rp.save(paths, ["D:/a.mp4", "D:/not-in-list"])
    assert rp.load() == (["D:/b", "D:/a.mp4"], ["D:/a.mp4"])
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["recent_video_locked"] == ["D:/a.mp4"]  # 不在清單裡的鎖定項目不會被存


def test_save_preserves_other_ui_state_keys(state_path):
    state_path.write_text(json.dumps({"last_tool_script": "x.py"}), encoding="utf-8")
    rp.record("D:/a")
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["last_tool_script"] == "x.py"


def test_load_survives_corrupt_file(state_path):
    state_path.write_text("{not json", encoding="utf-8")
    assert rp.load() == ([], [])
