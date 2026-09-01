"""`settings_gui/ui_state.py` 的「最近影片路徑」清單邏輯——鎖定 / 排序 / 上限 /
舊格式相容 / 讀取端存在性過濾。純檔案 I/O，不碰 Tk。"""

import json

import pytest

from settings_gui import ui_state


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_state, "_STATE_PATH", tmp_path / "gui_state.json")
    monkeypatch.setattr(ui_state, "_LEGACY_STATE_PATH", tmp_path / "_legacy_nope.json")
    return tmp_path


def _mkdir(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return str(d)


def test_add_orders_newest_first_and_dedups(tmp_path):
    a, b, c = (_mkdir(tmp_path, n) for n in ("a", "b", "c"))
    for p in (a, b, c, a):  # a 再用一次
        ui_state.add_recent_video_path(p)
    assert ui_state.get_recent_video_paths() == [a, c, b]


def test_nonexistent_path_not_added(tmp_path):
    ui_state.add_recent_video_path(str(tmp_path / "ghost"))
    assert ui_state.get_recent_video_paths() == []


def test_cap_evicts_oldest_unlocked(tmp_path):
    paths = [_mkdir(tmp_path, f"p{i}") for i in range(13)]
    for p in paths:
        ui_state.add_recent_video_path(p)
    got = ui_state.get_recent_video_paths()
    assert len(got) == ui_state._MAX_RECENT
    assert got == list(reversed(paths))[: ui_state._MAX_RECENT]


def test_locked_entry_survives_flood_and_keeps_position(tmp_path):
    keep = _mkdir(tmp_path, "keep")
    ui_state.add_recent_video_path(keep)

    entries = ui_state.get_recent_video_entries()
    entries[0]["locked"] = True
    ui_state.set_recent_video_entries(entries)

    for i in range(20):
        ui_state.add_recent_video_path(_mkdir(tmp_path, f"flood{i}"))

    final = ui_state.get_recent_video_entries()
    assert len(final) == ui_state._MAX_RECENT
    locked = [e for e in final if e["locked"]]
    assert [e["path"] for e in locked] == [keep]


def test_locked_used_again_does_not_jump_to_front(tmp_path):
    a, b = _mkdir(tmp_path, "a"), _mkdir(tmp_path, "b")
    ui_state.add_recent_video_path(a)
    ui_state.add_recent_video_path(b)  # [b, a]

    entries = ui_state.get_recent_video_entries()
    for e in entries:
        if e["path"] == a:
            e["locked"] = True
    ui_state.set_recent_video_entries(entries)  # [b, a(locked)]

    ui_state.add_recent_video_path(a)  # 鎖定項再用到 → 位置不動
    assert ui_state.get_recent_video_paths() == [b, a]


def test_reorder_via_set_entries_persists(tmp_path):
    a, b, c = (_mkdir(tmp_path, n) for n in ("a", "b", "c"))
    for p in (a, b, c):
        ui_state.add_recent_video_path(p)  # [c, b, a]

    entries = ui_state.get_recent_video_entries()
    entries.reverse()
    ui_state.set_recent_video_entries(entries)
    assert ui_state.get_recent_video_paths() == [a, b, c]


def test_locked_missing_shown_unlocked_missing_hidden(tmp_path):
    live = _mkdir(tmp_path, "live")
    gone_locked = _mkdir(tmp_path, "gone_locked")
    gone_plain = _mkdir(tmp_path, "gone_plain")
    for p in (live, gone_locked, gone_plain):
        ui_state.add_recent_video_path(p)

    entries = ui_state.get_recent_video_entries()
    for e in entries:
        if e["path"] == gone_locked:
            e["locked"] = True
    ui_state.set_recent_video_entries(entries)

    (tmp_path / "gone_locked").rmdir()
    (tmp_path / "gone_plain").rmdir()

    shown = ui_state.get_recent_video_paths()
    assert gone_locked in shown          # 鎖定 → 仍顯示
    assert gone_plain not in shown       # 未鎖定且找不到 → 不顯示
    assert live in shown
    # 磁碟上三筆都還在（讀取端過濾不刪檔）
    assert len(ui_state.get_recent_video_entries()) == 3


def test_legacy_list_of_str_format_is_read(tmp_path):
    live = _mkdir(tmp_path, "live")
    (tmp_path / "gui_state.json").write_text(
        json.dumps({"recent_video_paths": [live, str(tmp_path / "ghost")]}),
        encoding="utf-8",
    )
    assert ui_state.get_recent_video_paths() == [live]
    entries = ui_state.get_recent_video_entries()
    assert all(e["locked"] is False for e in entries)
    assert len(entries) == 2  # ghost 也在（get_recent_video_entries 不過濾）


def test_legacy_folders_key_is_read(tmp_path):
    live = _mkdir(tmp_path, "live")
    (tmp_path / "gui_state.json").write_text(
        json.dumps({"recent_video_folders": [live]}), encoding="utf-8"
    )
    assert ui_state.get_recent_video_paths() == [live]
    # 一次寫入後轉成新 key、舊 key 清掉
    ui_state.add_recent_video_path(live)
    data = json.loads((tmp_path / "gui_state.json").read_text(encoding="utf-8"))
    assert "recent_video_folders" not in data
    assert data["recent_video_paths"] == [{"path": live, "locked": False}]
