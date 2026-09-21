"""
Unit Test：tools/1_collect_class_videos.py 的同類影片聚集邏輯（不含 tkinter 視窗）

這支工具的核心承諾有三個，測試就圍繞它們：
1. 不混類別——只查找、只收納使用者選定的那一個行為類別。
2. 同名檔案絕不覆蓋——不同資料夾的同名檔案全部收進輸出資料夾，撞名自動改名。
3. 可反覆執行——內容已收納過的影片不會再被重複收進來。
只依賴 stdlib，全用 tmp_path，不碰真實資料夾。
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_paper_dir = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("collect_mod", _paper_dir / "tools" / "1_collect_class_videos.py")
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)


def _video(folder: Path, name: str, content: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text(content, encoding="utf-8")
    return p


def _names(folder: Path):
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def _run(items, mode="copy"):
    for it in items:
        if it.action == "copy":
            collect.execute_item(it, mode)


def test_behavior_classes_match_project_constant():
    """這裡刻意不 import 專案模組、自己存一份清單，所以要靠這個測試防止兩邊悄悄不一致。"""
    cms = _paper_dir / "cat_monitoring_system"
    if str(cms) not in sys.path:
        sys.path.insert(0, str(cms))
    from utils.constants import BEHAVIOR_CLASSES

    assert list(collect.BEHAVIOR_CLASSES) == list(BEHAVIOR_CLASSES)


# ---------------- 查找：不混類別 ----------------
def test_find_class_dirs_finds_only_the_selected_class(tmp_path):
    (tmp_path / "istock" / "class" / "walk").mkdir(parents=True)
    (tmp_path / "istock" / "class" / "lick").mkdir(parents=True)
    (tmp_path / "istock" / "class" / "stop").mkdir(parents=True)

    found = collect.find_class_dirs(tmp_path / "istock", "walk")

    assert found == [tmp_path / "istock" / "class" / "walk"]


def test_find_class_dirs_requires_exact_lowercase_name_by_default(tmp_path):
    (tmp_path / "a" / "walk").mkdir(parents=True)
    (tmp_path / "b" / "Walk").mkdir(parents=True)
    (tmp_path / "c" / "WALK").mkdir(parents=True)
    (tmp_path / "d" / "walk_old").mkdir(parents=True)
    (tmp_path / "e" / "walking").mkdir(parents=True)

    found = collect.find_class_dirs(tmp_path, "walk")

    assert [p.relative_to(tmp_path).as_posix() for p in found] == ["a/walk"]  # Walk／WALK／walk_old／walking 都不算


def test_find_class_dirs_can_ignore_case_when_asked(tmp_path):
    (tmp_path / "a" / "WALK").mkdir(parents=True)
    (tmp_path / "b" / "Walk").mkdir(parents=True)

    found = collect.find_class_dirs(tmp_path, "walk", case_sensitive=False)

    assert sorted(p.name for p in found) == ["WALK", "Walk"]


def test_find_class_dirs_default_depth_is_effectively_unlimited(tmp_path):
    """walk 藏得很深也要找得到，否則會被誤判成『缺少』而拒絕整合。"""
    deep = tmp_path / "r"
    for i in range(15):
        deep = deep / f"lvl{i}"
    (deep / "walk").mkdir(parents=True)

    assert len(collect.find_class_dirs(tmp_path / "r", "walk")) == 1


def test_find_class_dirs_includes_the_selected_folder_itself(tmp_path):
    walk = tmp_path / "somewhere" / "walk"
    walk.mkdir(parents=True)

    assert collect.find_class_dirs(walk, "walk") == [walk]
    assert collect.find_class_dirs(walk, "lick") == []  # 選到 walk 資料夾卻查 lick：不混類別，什麼都不找


def test_find_class_dirs_finds_multiple_and_respects_depth_limit(tmp_path):
    (tmp_path / "r" / "s1" / "class" / "walk").mkdir(parents=True)   # 深度 3
    (tmp_path / "r" / "s2" / "walk").mkdir(parents=True)              # 深度 2
    (tmp_path / "r" / "x" / "x" / "x" / "x" / "x" / "x" / "walk").mkdir(parents=True)  # 深度 7

    found = collect.find_class_dirs(tmp_path / "r", "walk", max_depth=3)

    assert sorted(p.relative_to(tmp_path / "r").as_posix() for p in found) == ["s1/class/walk", "s2/walk"]


def test_find_class_dirs_skips_hidden_and_missing_folders(tmp_path):
    (tmp_path / "r" / ".git" / "walk").mkdir(parents=True)
    (tmp_path / "r" / "$RECYCLE.BIN" / "walk").mkdir(parents=True)

    assert collect.find_class_dirs(tmp_path / "r", "walk") == []
    assert collect.find_class_dirs(tmp_path / "does_not_exist", "walk") == []


def test_unique_dirs_dedupes_same_folder_and_keeps_order(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()

    assert collect.unique_dirs([a, b, a, tmp_path / "a" / ".." / "a"]) == [a, b]


def test_selected_class_never_pulls_videos_of_other_classes(tmp_path):
    """端對端的不混類別：來源同時有 walk 與 lick，選 walk 只會收 walk 的影片。"""
    _video(tmp_path / "s" / "class" / "walk", "w.mp4", "W")
    _video(tmp_path / "s" / "class" / "lick", "l.mp4", "L")
    dest = tmp_path / "out"

    dirs = collect.find_class_dirs(tmp_path / "s", "walk")
    items = collect.plan_collect(dirs, dest, "walk")
    _run(items)

    assert _names(dest) == ["w.mp4"]


# ---------------- 規劃：同名不覆蓋、全部收納 ----------------
def test_plan_collects_from_multiple_folders(tmp_path):
    d1 = tmp_path / "s1" / "walk"; _video(d1, "a.mp4", "A")
    d2 = tmp_path / "s2" / "walk"; _video(d2, "b.mp4", "B")

    items = collect.plan_collect([d1, d2], tmp_path / "out", "walk")

    assert [it.dst.name for it in items] == ["a.mp4", "b.mp4"]


def test_same_name_in_different_folders_are_all_collected_without_overwrite(tmp_path):
    d1 = tmp_path / "s1" / "walk"; _video(d1, "clip.mp4", "FIRST")
    d2 = tmp_path / "s2" / "walk"; _video(d2, "clip.mp4", "SECOND!")
    d3 = tmp_path / "s3" / "walk"; _video(d3, "clip.mp4", "THIRD??!")
    dest = tmp_path / "out"

    items = collect.plan_collect([d1, d2, d3], dest, "walk")
    _run(items)

    assert _names(dest) == ["clip.mp4", "clip_2.mp4", "clip_3.mp4"]
    assert (dest / "clip.mp4").read_text(encoding="utf-8") == "FIRST"
    assert (dest / "clip_2.mp4").read_text(encoding="utf-8") == "SECOND!"
    assert (dest / "clip_3.mp4").read_text(encoding="utf-8") == "THIRD??!"
    assert items[1].note == "改名為 clip_2.mp4"


def test_existing_file_in_output_with_same_name_is_never_overwritten(tmp_path):
    existing = _video(tmp_path / "out", "clip.mp4", "ALREADY HERE")
    d1 = tmp_path / "s1" / "walk"; _video(d1, "clip.mp4", "NEW ONE")

    items = collect.plan_collect([d1], tmp_path / "out", "walk")
    _run(items)

    assert existing.read_text(encoding="utf-8") == "ALREADY HERE"
    assert (tmp_path / "out" / "clip_2.mp4").read_text(encoding="utf-8") == "NEW ONE"


def test_same_name_same_content_is_skipped_when_dedupe_on(tmp_path):
    d1 = tmp_path / "s1" / "walk"; _video(d1, "a.mp4", "SAMEBYTES")
    d2 = tmp_path / "s2" / "walk"; _video(d2, "a.mp4", "SAMEBYTES")

    items = collect.plan_collect([d1, d2], tmp_path / "out", "walk", dedupe=True)

    assert [it.action for it in items] == ["copy", "skip_duplicate"]


def test_same_content_is_still_collected_when_dedupe_off(tmp_path):
    d1 = tmp_path / "s1" / "walk"; _video(d1, "a.mp4", "SAMEBYTES")
    d2 = tmp_path / "s2" / "walk"; _video(d2, "a.mp4", "SAMEBYTES")
    dest = tmp_path / "out"

    _run(collect.plan_collect([d1, d2], dest, "walk", dedupe=False))

    assert _names(dest) == ["a.mp4", "a_2.mp4"]  # 「全部一起收」：取消勾選後連內容相同的也收


def test_renamed_duplicate_content_is_still_recognized(tmp_path):
    d1 = tmp_path / "s1" / "walk"; _video(d1, "a.mp4", "SAMEBYTES")
    d2 = tmp_path / "s2" / "walk"; _video(d2, "totally_other_name.mp4", "SAMEBYTES")

    items = collect.plan_collect([d1, d2], tmp_path / "out", "walk")

    assert [it.action for it in items] == ["copy", "skip_duplicate"]


def test_sequential_naming_continues_after_existing_max_serial(tmp_path):
    _video(tmp_path / "out", "walk_3.mp4", "x3")
    _video(tmp_path / "out", "walk_10.mp4", "x10")
    d1 = tmp_path / "s1" / "walk"; _video(d1, "a.mp4", "A"); _video(d1, "b.mp4", "B")
    d2 = tmp_path / "s2" / "walk"; _video(d2, "c.MP4", "C")

    items = collect.plan_collect([d1, d2], tmp_path / "out", "walk", naming="sequential")

    assert [it.dst.name for it in items] == ["walk_11.mp4", "walk_12.mp4", "walk_13.mp4"]  # 副檔名一律小寫


@pytest.mark.parametrize("naming", ["keep", "sequential"])
def test_rerun_is_idempotent(tmp_path, naming):
    d1 = tmp_path / "s1" / "walk"; _video(d1, "a.mp4", "A"); _video(d1, "dup.mp4", "X1")
    d2 = tmp_path / "s2" / "walk"; _video(d2, "a.mp4", "A-different"); _video(d2, "b.mp4", "B")
    dest = tmp_path / "out"

    _run(collect.plan_collect([d1, d2], dest, "walk", naming=naming))
    before = _names(dest)
    second = collect.plan_collect([d1, d2], dest, "walk", naming=naming)
    _run(second)

    assert second and all(it.action == "skip_duplicate" for it in second)
    assert _names(dest) == before


def test_output_folder_inside_a_class_dir_is_never_a_source(tmp_path):
    """輸出資料夾恰好就是某個來源的 walk 資料夾：不能把檔案聚集到自己身上。"""
    walk = tmp_path / "s" / "walk"; _video(walk, "a.mp4", "A")

    assert collect.plan_collect([walk], walk, "walk") == []


def test_ignores_non_video_files_and_part_files(tmp_path):
    d = tmp_path / "s" / "walk"
    _video(d, "a.mp4", "A"); _video(d, "notes.txt", "t"); _video(d, "b.mp4.part", "p")

    items = collect.plan_collect([d], tmp_path / "out", "walk")

    assert [it.src.name for it in items] == ["a.mp4"]


# ---------------- 執行 ----------------
def test_execute_copy_keeps_source_and_leaves_no_part_file(tmp_path):
    d = tmp_path / "s" / "shake"; src = _video(d, "a.mp4", "AAA")
    dest = tmp_path / "out"
    (item,) = collect.plan_collect([d], dest, "shake")

    collect.execute_item(item, "copy")

    assert (dest / "a.mp4").read_text(encoding="utf-8") == "AAA"
    assert src.exists()
    assert _names(dest) == ["a.mp4"]  # 沒有殘留 .part


def test_execute_move_removes_source(tmp_path):
    d = tmp_path / "s" / "lick"; src = _video(d, "a.mp4", "AAA")
    dest = tmp_path / "out"
    (item,) = collect.plan_collect([d], dest, "lick")

    collect.execute_item(item, "move")

    assert (dest / "a.mp4").read_text(encoding="utf-8") == "AAA"
    assert not src.exists()


def test_execute_never_overwrites_file_that_appeared_after_planning(tmp_path):
    d = tmp_path / "s" / "walk"; src = _video(d, "a.mp4", "MINE")
    dest = tmp_path / "out"
    (item,) = collect.plan_collect([d], dest, "walk")
    intruder = _video(dest, "a.mp4", "INTRUDER")  # 規劃之後才出現的同名檔

    with pytest.raises(FileExistsError):
        collect.execute_item(item, "copy")

    assert intruder.read_text(encoding="utf-8") == "INTRUDER"
    assert src.read_text(encoding="utf-8") == "MINE"


def test_write_manifest_never_overwrites_previous_one(tmp_path):
    p1 = collect.write_manifest(tmp_path, [["walk", "copy", "s", "d", ""]])
    p2 = collect.write_manifest(tmp_path, [["walk", "copy", "s2", "d2", ""]])

    assert p1 != p2 and p1.exists() and p2.exists()


# ---------------- 整合前的全數驗證 ----------------
def test_coverage_passes_when_every_selected_folder_has_the_class_folder(tmp_path):
    for name in ("s1", "s2", "s3"):
        (tmp_path / name / "class" / "walk").mkdir(parents=True)

    report = collect.check_coverage([tmp_path / "s1", tmp_path / "s2", tmp_path / "s3"], "walk")

    assert report.ok and report.missing == []
    assert [len(v) for v in report.hits.values()] == [1, 1, 1]


def test_coverage_refuses_when_even_one_selected_folder_lacks_the_class_folder(tmp_path):
    """選 10 個資料夾，只要有 1 個底下沒有 walk 就必須拒絕。"""
    folders = []
    for i in range(10):
        f = tmp_path / f"s{i}"
        (f / "class" / ("walk" if i != 6 else "lick")).mkdir(parents=True)   # 第 7 個只有 lick，沒有 walk
        folders.append(f)

    report = collect.check_coverage(folders, "walk")

    assert not report.ok
    assert report.missing == [tmp_path / "s6"]


def test_coverage_reports_every_missing_folder(tmp_path):
    (tmp_path / "ok" / "walk").mkdir(parents=True)
    (tmp_path / "bad1").mkdir(); (tmp_path / "bad2" / "stop").mkdir(parents=True)

    report = collect.check_coverage([tmp_path / "bad1", tmp_path / "ok", tmp_path / "bad2"], "walk")

    assert report.missing == [tmp_path / "bad1", tmp_path / "bad2"]


def test_coverage_treats_nonexistent_folder_as_missing(tmp_path):
    report = collect.check_coverage([tmp_path / "does_not_exist"], "walk")

    assert not report.ok
    assert "資料夾不存在" in collect.format_missing_message(report)


def test_coverage_selected_folder_named_after_the_class_counts_itself(tmp_path):
    walk = tmp_path / "anywhere" / "walk"; walk.mkdir(parents=True)

    assert collect.check_coverage([walk], "walk").ok


def test_one_wrapper_folder_containing_walk_folders_at_different_depths_works(tmp_path):
    """用一個資料夾包住 10 個不同層的 walk 資料夾：只選這一個外層資料夾也要能通過並全部收進來。"""
    wrapper = tmp_path / "wrapper"
    for i in range(10):
        extra_levels = [f"lvl{j}" for j in range(i % 5)]                  # 0~4 層額外巢狀，讓 walk 落在不同深度
        _video(wrapper.joinpath(f"g{i}", *extra_levels, "walk"), f"v{i}.mp4", f"content-{i}")
    dest = tmp_path / "out"

    report = collect.check_coverage([wrapper], "walk")
    items = collect.plan_collect(report.class_dirs(), dest, "walk")
    _run(items)

    assert report.ok
    assert len(report.hits[wrapper]) == 10
    assert len(_names(dest)) == 10


def test_coverage_mentions_case_mismatch_in_warning(tmp_path):
    (tmp_path / "s1" / "Walk").mkdir(parents=True)

    report = collect.check_coverage([tmp_path / "s1"], "walk")
    message = collect.format_missing_message(report)

    assert not report.ok
    assert report.near_misses == {tmp_path / "s1": ["Walk"]}
    assert "大小寫不符" in message and "Walk" in message


def test_coverage_message_lists_missing_folders_and_says_nothing_was_touched(tmp_path):
    report = collect.check_coverage([tmp_path / "a", tmp_path / "b"], "walk")

    message = collect.format_missing_message(report)

    assert str(tmp_path / "a") in message and str(tmp_path / "b") in message
    assert "已拒絕整合" in message and "沒有動任何檔案" in message


def test_coverage_class_dirs_are_deduped_when_folders_overlap(tmp_path):
    (tmp_path / "parent" / "child" / "walk").mkdir(parents=True)

    report = collect.check_coverage([tmp_path / "parent", tmp_path / "parent" / "child"], "walk")

    assert report.ok and len(report.class_dirs()) == 1  # 選了外層又選內層，同一個 walk 只算一次


# ---------------- 多選資料夾視窗用的純函式 ----------------
def test_list_subdirs_is_naturally_sorted_and_skips_hidden_and_files(tmp_path):
    for name in ("d10", "d2", "d1", ".git", "$RECYCLE.BIN"):
        (tmp_path / name).mkdir()
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")

    assert collect._list_subdirs(tmp_path) == ["d1", "d2", "d10"]


def test_list_subdirs_returns_empty_when_unreadable_or_missing(tmp_path):
    assert collect._list_subdirs(tmp_path / "does_not_exist") == []


def test_list_drives_is_never_empty():
    assert collect._list_drives()
