"""
Unit Test：tools/_rename_gui.py 的批次序號命名邏輯（不含 tkinter 視窗）

這組邏輯的存在理由是「不要覆蓋其他來源已放進行為資料夾的影片」，所以測試重點在：
起始序號怎麼避開既有檔名、被未選取檔案佔用的目標名稱要被標成衝突、選取範圍內互相
換位不能撞名、改名途中失敗要還原。只依賴 stdlib，全用 tmp_path，不碰真實資料夾。
"""

import sys
from pathlib import Path

import pytest

_tools_dir = Path(__file__).resolve().parents[1] / "tools"
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

from _rename_gui import (  # noqa: E402
    execute_renames,
    list_class_files,
    next_free_index,
    plan_renames,
)

EXTS = {".mp4", ".avi"}


def _make(folder: Path, name: str, content: str = "") -> Path:
    p = folder / name
    p.write_text(content or name, encoding="utf-8")
    return p


def test_list_class_files_natural_sort_and_ext_filter(tmp_path):
    for n in ["walk_10.mp4", "walk_2.mp4", "walk_1.avi", "note.txt", "clip.MP4"]:
        _make(tmp_path, n)
    (tmp_path / "subdir").mkdir()

    names = [f.name for f in list_class_files(tmp_path, EXTS)]

    assert names == ["clip.MP4", "walk_1.avi", "walk_2.mp4", "walk_10.mp4"]


def test_list_class_files_missing_folder_returns_empty(tmp_path):
    assert list_class_files(tmp_path / "nope", EXTS) == []


def test_next_free_index_empty_folder_is_one(tmp_path):
    assert next_free_index(tmp_path, "walk", EXTS) == 1


def test_next_free_index_follows_max_existing(tmp_path):
    for n in ["walk_3.mp4", "walk_10.avi", "walk_x.mp4", "lick_99.mp4", "other.mp4", "walk_5.txt"]:
        _make(tmp_path, n)

    assert next_free_index(tmp_path, "walk", EXTS) == 11


def test_plan_renames_numbers_in_given_order_keeps_extension(tmp_path):
    a = _make(tmp_path, "a.mp4")
    b = _make(tmp_path, "b.avi")

    plan, conflicts = plan_renames(tmp_path, [a, b], "walk", 5)

    assert [(s.name, d.name) for s, d in plan] == [("a.mp4", "walk_5.mp4"), ("b.avi", "walk_6.avi")]
    assert conflicts == []


def test_plan_renames_flags_target_taken_by_unselected_file(tmp_path):
    """其他來源的 walk_1.mp4 已在資料夾，且沒被選取：不能被覆蓋，必須列為衝突。"""
    other_source = _make(tmp_path, "walk_1.mp4", "other source")
    a = _make(tmp_path, "a.mp4")

    plan, conflicts = plan_renames(tmp_path, [a], "walk", 1)

    assert conflicts == ["walk_1.mp4"]
    assert other_source.read_text(encoding="utf-8") == "other source"


def test_plan_renames_conflict_check_is_case_insensitive(tmp_path):
    _make(tmp_path, "WALK_1.mp4")
    a = _make(tmp_path, "a.mp4")

    _plan, conflicts = plan_renames(tmp_path, [a], "walk", 1)

    assert conflicts == ["walk_1.mp4"]


def test_plan_renames_selected_file_holding_target_is_not_a_conflict(tmp_path):
    w1 = _make(tmp_path, "walk_1.mp4")
    a = _make(tmp_path, "a.mp4")

    _plan, conflicts = plan_renames(tmp_path, [a, w1], "walk", 1)

    assert conflicts == []


def test_plan_renames_target_taken_by_non_video_file_is_conflict(tmp_path):
    (tmp_path / "walk_1.mp4").mkdir()  # 資料夾同名也不能撞
    a = _make(tmp_path, "a.mp4")

    _plan, conflicts = plan_renames(tmp_path, [a], "walk", 1)

    assert conflicts == ["walk_1.mp4"]


def test_execute_renames_renames_and_preserves_content(tmp_path):
    a = _make(tmp_path, "a.mp4", "AAA")
    b = _make(tmp_path, "b.mp4", "BBB")
    plan, _ = plan_renames(tmp_path, [a, b], "stop", 3)

    assert execute_renames(plan) == 2

    assert (tmp_path / "stop_3.mp4").read_text(encoding="utf-8") == "AAA"
    assert (tmp_path / "stop_4.mp4").read_text(encoding="utf-8") == "BBB"
    assert not a.exists() and not b.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["stop_3.mp4", "stop_4.mp4"]  # 沒有殘留暫名


def test_execute_renames_swap_within_selection_does_not_collide(tmp_path):
    """walk_2 → walk_1 同時 walk_1 → walk_2：一階段直接改名會撞，兩階段必須成功且內容互換。"""
    w1 = _make(tmp_path, "walk_1.mp4", "ONE")
    w2 = _make(tmp_path, "walk_2.mp4", "TWO")
    plan, conflicts = plan_renames(tmp_path, [w2, w1], "walk", 1)
    assert conflicts == []

    execute_renames(plan)

    assert (tmp_path / "walk_1.mp4").read_text(encoding="utf-8") == "TWO"
    assert (tmp_path / "walk_2.mp4").read_text(encoding="utf-8") == "ONE"


def test_execute_renames_skips_noop(tmp_path):
    w1 = _make(tmp_path, "walk_1.mp4", "ONE")
    plan, _ = plan_renames(tmp_path, [w1], "walk", 1)

    assert execute_renames(plan) == 0
    assert w1.read_text(encoding="utf-8") == "ONE"


def test_execute_renames_failure_rolls_back_and_never_overwrites(tmp_path):
    """中途發現目標被佔用（例如規劃後才有別的程式放進同名檔）：丟例外、全部還原、不覆蓋。"""
    a = _make(tmp_path, "a.mp4", "AAA")
    b = _make(tmp_path, "b.mp4", "BBB")
    intruder = _make(tmp_path, "walk_2.mp4", "INTRUDER")
    plan = [(a, tmp_path / "walk_1.mp4"), (b, tmp_path / "walk_2.mp4")]

    with pytest.raises(OSError):
        execute_renames(plan)

    assert a.read_text(encoding="utf-8") == "AAA"
    assert b.read_text(encoding="utf-8") == "BBB"
    assert intruder.read_text(encoding="utf-8") == "INTRUDER"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.mp4", "b.mp4", "walk_2.mp4"]


# ---- 獨立子行程入口（stdin JSON）：參數錯誤要乾淨地以 exit code 2 結束，不能開視窗或卡住 ----
import os  # noqa: E402
import subprocess  # noqa: E402

_RENAME_GUI_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "_rename_gui.py"


@pytest.mark.parametrize(
    "stdin_text",
    [
        "",                                   # 什麼都沒傳
        "not json at all",                    # 不是 JSON
        '{"exts": [".mp4"]}',                 # 缺 dest_folders
        '{"dest_folders": {"walk": "x"}}',    # 缺 exts
        '{"dest_folders": 5, "exts": []}',    # 型別錯誤
        "[1, 2, 3]",                          # JSON 但不是物件
    ],
)
def test_cli_entry_rejects_bad_payload_with_exit_code_2(stdin_text):
    result = subprocess.run(
        [sys.executable, str(_RENAME_GUI_SCRIPT)],
        input=stdin_text, capture_output=True, text=True, encoding="utf-8", timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},  # 跟設定視窗啟動腳本時的條件一致
    )

    assert result.returncode == 2
    assert "參數格式錯誤" in result.stderr
