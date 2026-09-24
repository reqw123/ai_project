"""
Unit Test：tools/1_run_video_inference.py 的 Shift+A~E 影片分類（搬到行為資料夾／「已檢視」_2 資料夾）

該腳本一 import 就會載入模型與開視窗，沒辦法直接 import；這裡從原始碼取出
REVIEWED_SUFFIX 與分類相關的函式（_is_behavior_folder_name／find_class_root／
move_video_to_class_folder／classify_video_to_folder）單獨執行。全部在 tmp_path 的假資料夾上操作。
"""

import re
import shutil
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "1_run_video_inference.py"
BEHAVIORS = ["walk", "lick", "scratch", "shake", "stop"]


@pytest.fixture(scope="module")
def fn():
    src = SCRIPT.read_text(encoding="utf-8")
    suffix = re.search(r'^REVIEWED_SUFFIX = "([^"]+)"', src, re.M).group(1)
    a = src.index("def _is_behavior_folder_name")
    # 取到 classify_video_to_folder 結束：往後第一個「頂層（第 0 欄）且不是空行」的行（可能是下一個 def 或裝飾器）
    start = src.index("def classify_video_to_folder")
    body = src.index("\n", start) + 1  # 從 def 的下一行開始找
    b = body + re.search(r"^\S", src[body:], re.M).start()
    ns = {"Path": Path, "shutil": shutil, "BEHAVIOR_CLASSES": BEHAVIORS, "REVIEWED_SUFFIX": suffix}
    exec(src[a:b], ns)
    return ns


def _video(folder, name="a.mp4"):
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / name
    f.write_bytes(b"x")
    return f


def test_in_behavior_folder_and_press_same_class_goes_to_reviewed_folder(fn, tmp_path):
    """已經在 walk 裡按 Shift+A（檢視後確認是 walk）→ 搬到同層的 walk_2，沒有才建立。"""
    v = _video(tmp_path / "已改名" / "walk")
    dest = fn["move_video_to_class_folder"](v, "walk")
    assert dest == tmp_path / "已改名" / "walk_2" / "a.mp4"
    assert dest.exists() and not v.exists()


def test_reviewed_folder_is_reused_not_multiplied(fn, tmp_path):
    root = tmp_path / "已改名"
    fn["move_video_to_class_folder"](_video(root / "walk", "a.mp4"), "walk")
    fn["move_video_to_class_folder"](_video(root / "walk", "b.mp4"), "walk")
    assert sorted(p.name for p in root.iterdir()) == ["walk", "walk_2"]
    assert sorted(p.name for p in (root / "walk_2").iterdir()) == ["a.mp4", "b.mp4"]


def test_video_already_in_reviewed_folder_is_not_moved_back(fn, tmp_path):
    v = _video(tmp_path / "已改名" / "walk_2")
    assert fn["move_video_to_class_folder"](v, "walk") == v
    assert v.exists()
    assert not (tmp_path / "已改名" / "walk").exists()  # 不會因此建出 walk


def test_other_class_still_goes_to_sibling_class_folder(fn, tmp_path):
    root = tmp_path / "已改名"
    v = _video(root / "walk")
    dest = fn["move_video_to_class_folder"](v, "lick")
    assert dest == root / "lick" / "a.mp4"
    assert not (root / "lick_2").exists()  # 只有「同類別再按一次」才會建 _2


def test_video_in_reviewed_folder_can_be_reclassified_to_another_class(fn, tmp_path):
    """walk_2 裡的影片改判成別類：根目錄仍是 已改名（不會把 walk_2 當成根目錄而建出 walk_2\\lick）。"""
    root = tmp_path / "已改名"
    v = _video(root / "walk_2")
    assert fn["find_class_root"](v) == root
    assert fn["move_video_to_class_folder"](v, "lick") == root / "lick" / "a.mp4"


@pytest.mark.parametrize("behavior", BEHAVIORS)
def test_all_five_classes_use_the_same_logic(fn, tmp_path, behavior):
    root = tmp_path / "已改名"
    v = _video(root / behavior)
    dest = fn["move_video_to_class_folder"](v, behavior)
    assert dest == root / f"{behavior}_2" / "a.mp4"
    again = fn["move_video_to_class_folder"](dest, behavior)
    assert again == dest  # 已經在 _2 裡不動


def test_video_outside_any_class_folder_goes_into_plain_class_folder(fn, tmp_path):
    """沒分類的資料夾（不是 walk/…/walk_2）裡的影片：跟以前一樣建行為資料夾，不會直接跳到 _2。"""
    v = _video(tmp_path / "未分類")
    dest = fn["move_video_to_class_folder"](v, "walk")
    assert dest == tmp_path / "未分類" / "walk" / "a.mp4"


def test_name_matching_is_case_insensitive_and_strict(fn):
    f = fn["_is_behavior_folder_name"]
    assert f("walk") and f("WALK") and f("walk_2") and f("Lick_2")
    assert not f("walk_3") and not f("walk2") and not f("dance_2") and not f("已改名")


def test_duplicate_name_in_reviewed_folder_gets_a_suffix(fn, tmp_path):
    root = tmp_path / "已改名"
    _video(root / "walk_2", "a.mp4")
    v = _video(root / "walk", "a.mp4")
    dest = fn["move_video_to_class_folder"](v, "walk")
    assert dest == root / "walk_2" / "a_1.mp4"


def test_classify_messages_and_return_values(fn, tmp_path, capsys):
    root = tmp_path / "已改名"
    v = _video(root / "walk")
    assert fn["classify_video_to_folder"](v, "walk") is True         # 搬走了 → 要從播放清單移除
    assert "已檢視確認為 WALK" in capsys.readouterr().out
    moved = root / "walk_2" / "a.mp4"
    assert fn["classify_video_to_folder"](moved, "walk") is False    # 已在 walk_2 → 沒搬
    assert "已檢視資料夾" in capsys.readouterr().out
    v2 = _video(root / "walk", "b.mp4")
    assert fn["classify_video_to_folder"](v2, "lick") is True
    assert "已歸類為 LICK" in capsys.readouterr().out


# ── 2026-09-24：分類子資料夾建在「輸入資料夾」底下，不在同層 ─────────────────────

@pytest.fixture
def with_roots(fn):
    roots = fn["_CLASSIFY_ROOTS"]
    roots.clear()
    yield roots
    roots.clear()


def test_input_folder_named_like_a_class_gets_subfolder_not_sibling(fn, with_roots, tmp_path):
    """輸入資料夾本身叫 lick（例如 未被選擇的模型影片/lick）：按 Shift+A 搬到 lick/walk/，不是同層的 walk/。"""
    src = tmp_path / "未被選擇的模型影片" / "lick"
    with_roots.append(src.resolve())
    v = _video(src)
    assert fn["move_video_to_class_folder"](v, "walk") == src / "walk" / "a.mp4"
    assert not (tmp_path / "未被選擇的模型影片" / "walk").exists()


def test_same_class_as_input_folder_name_also_goes_into_subfolder(fn, with_roots, tmp_path):
    src = tmp_path / "未被選擇的模型影片" / "lick"
    with_roots.append(src.resolve())
    v = _video(src)
    assert fn["move_video_to_class_folder"](v, "lick") == src / "lick" / "a.mp4"


def test_video_in_nested_folder_of_input_uses_input_as_root(fn, with_roots, tmp_path):
    src = tmp_path / "輸入"
    with_roots.append(src.resolve())
    v = _video(src / "某批" / "更深")
    assert fn["move_video_to_class_folder"](v, "shake") == src / "shake" / "a.mp4"


def test_closest_input_folder_wins(fn, with_roots, tmp_path):
    outer = tmp_path / "外"
    inner = outer / "內"
    with_roots.extend([outer.resolve(), inner.resolve()])
    v = _video(inner)
    assert fn["find_class_root"](v) == inner.resolve()


@pytest.fixture(scope="module")
def resolve(fn):
    src = SCRIPT.read_text(encoding="utf-8")
    a = src.index("def resolve_video_paths")
    body = src.index("\n", a) + 1
    b = body + re.search(r"^\S", src[body:], re.M).start()
    exts = re.search(r"^SUPPORTED_VIDEO_EXTS = (\{.*?\})", src, re.M | re.S).group(1)
    ns = dict(fn)
    ns.update({"Iterable": list, "_is_stream_url": lambda s: False,
               "SUPPORTED_VIDEO_EXTS": eval(exts)})
    exec(src[a:b], ns)
    return ns["resolve_video_paths"]


def test_rescan_skips_classified_subfolders(resolve, with_roots, tmp_path):
    src = tmp_path / "輸入"
    _video(src, "left.mp4")
    _video(src / "walk", "done1.mp4")
    _video(src / "lick_2", "done2.mp4")
    _video(src / "其他批次", "keep.mp4")   # 不是行為名稱的子資料夾照樣掃
    names = sorted(Path(p).name for p in resolve([str(src)]))
    assert names == ["keep.mp4", "left.mp4"]
    assert src.resolve() in with_roots


def test_input_folder_itself_named_like_class_is_still_scanned(resolve, with_roots, tmp_path):
    """輸入資料夾本身叫 walk（例如 模型專用/train/walk）：裡面的影片照常播，只略過它底下的分類子資料夾。"""
    src = tmp_path / "train" / "walk"
    _video(src, "a.mp4")
    _video(src / "lick", "moved.mp4")
    assert [Path(p).name for p in resolve([str(src)])] == ["a.mp4"]
