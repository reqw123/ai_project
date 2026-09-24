"""
Unit Test：訓練前的切分檢查（gcn_dataset_manager.check_split_consistency 與
0_train_gcn.require_split_consistency 使用的四種錯誤）。重現「在檔案總管手動拖檔」會造成的問題。

全部在 tmp_path 底下造假的 模型專用/ 與 skeletons/，並把 VIDEO_ROOT／設定檔都指到 tmp_path，
不會碰到真正的影片或骨架。
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER / "cat_monitoring_system"))
from utils import skeleton_splits as ss  # noqa: E402

_spec = importlib.util.spec_from_file_location("gcn_dataset_manager", _PAPER / "tools" / "gcn_dataset_manager.py")
gdm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gdm)

PREFIXES = {"walk": 0, "lick": 1, "scratch": 2, "shake": 3, "stop": 4}


@pytest.fixture
def env(tmp_path, monkeypatch):
    video_root = tmp_path / "模型專用"
    skel_root = tmp_path / "skeletons"
    skel_root.mkdir()
    monkeypatch.setattr(ss, "VIDEO_ROOT", video_root)
    monkeypatch.setattr(gdm, "VIDEO_ROOT", video_root)
    monkeypatch.setattr(gdm, "_load_config",
                        lambda: {"SKELETON_DATA_FOLDER": str(skel_root), "BEHAVIOR_PREFIXES": PREFIXES})
    # 固定名單：lick_8 固定在 train
    (skel_root / "_split_rules.json").write_text(
        json.dumps({"fixed": {"train": ["lick_8"], "val": [], "test": []}}), encoding="utf-8")
    return video_root, skel_root


def _make(video_root, skel_root, split, stem, seed):
    """造一支影片 模型專用/<split>/<類別>/ 與骨架 skeletons/<split>/<類別>/，關鍵點依 seed 產生。"""
    cls = ss.class_folder_of(stem)
    video = video_root / split / cls / f"{stem}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"v")
    rng = np.random.default_rng(seed)
    frames = [{"keypoints": [{"x": float(x), "y": float(y), "conf": 1.0}
                             for x, y in rng.uniform(0, 1000, (17, 2))]} for _ in range(10)]
    js = skel_root / split / cls / f"{stem}.json"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps({"video_metadata": {"video_path": str(video)}, "frames": frames}), encoding="utf-8")
    return video, js


@pytest.fixture
def clean(env):
    video_root, skel_root = env
    files = {
        "walk5": _make(video_root, skel_root, "train", "walk5", 1),
        "lick_8": _make(video_root, skel_root, "train", "lick_8", 2),
        "shake_32": _make(video_root, skel_root, "val", "shake_32", 3),
        "stop_1": _make(video_root, skel_root, "test", "stop_1", 4),
    }
    return video_root, skel_root, files


def _blocking(issues):
    return {k: v for k, v in issues.items() if k != "missing" and v}


def test_clean_dataset_passes(clean):
    assert _blocking(gdm.check_split_consistency()) == {}


def test_copy_instead_of_move_is_clash(clean):
    """錯誤示範 1：在檔案總管把 walk5.json「複製」到 val/walk/（而不是搬移）。"""
    _, skel_root, files = clean
    (skel_root / "val" / "walk").mkdir(parents=True, exist_ok=True)
    shutil.copy(files["walk5"][1], skel_root / "val" / "walk" / "walk5.json")
    issues = gdm.check_split_consistency()
    assert [v for v, _ in issues["clash"]] == ["walk5"]
    assert not issues["video"] and not issues["fixed"]   # clash 的檔名不參與其他檢查


def test_move_skeleton_only_is_video_mismatch(clean):
    """錯誤示範 2：只把 walk5.json 拖到 val/walk/，影片留在 train。"""
    _, skel_root, files = clean
    (skel_root / "val" / "walk").mkdir(parents=True, exist_ok=True)
    shutil.move(str(files["walk5"][1]), str(skel_root / "val" / "walk" / "walk5.json"))
    assert gdm.check_split_consistency()["video"] == [("walk5", "val", "train")]


def test_move_video_only_is_video_mismatch(clean):
    """錯誤示範 3：只把 walk5.mp4 拖到 模型專用/test/walk/（JSON 記的路徑也跟著失效）。"""
    video_root, _, files = clean
    dst = video_root / "test" / "walk" / "walk5.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(files["walk5"][0]), str(dst))
    assert gdm.check_split_consistency()["video"] == [("walk5", "train", "test")]


def test_move_both_consistently_is_allowed(clean):
    """骨架和影片一起拖到 val：資料夾就是切分，這是合法的手動調整，不擋。"""
    video_root, skel_root, files = clean
    for src, dst in [(files["walk5"][1], skel_root / "val" / "walk" / "walk5.json"),
                     (files["walk5"][0], video_root / "val" / "walk" / "walk5.mp4")]:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    assert _blocking(gdm.check_split_consistency()) == {}


def test_moving_fixed_item_is_fixed_violation(clean):
    """錯誤示範 4：把固定在 train 的 lick_8（骨架＋影片）一起拖到 val。"""
    video_root, skel_root, files = clean
    for src, dst in [(files["lick_8"][1], skel_root / "val" / "lick" / "lick_8.json"),
                     (files["lick_8"][0], video_root / "val" / "lick" / "lick_8.mp4")]:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    issues = gdm.check_split_consistency()
    assert issues["fixed"] == [("lick_8", "val", "train")]
    assert not issues["video"]


def test_splitting_duplicate_group_is_group_violation(clean):
    """錯誤示範 5：放進一支跟 shake_32（val）內容相同的 shake_99，但放在 train。"""
    video_root, skel_root, _ = clean
    _make(video_root, skel_root, "train", "shake_99", 3)   # 同 seed ＝ 關鍵點完全相同
    issues = gdm.check_split_consistency()
    assert issues["same_content"] == [[("shake_32", "val"), ("shake_99", "train")]]
    assert not issues["hq_pair"]


def test_hq_pair_split_is_group_violation(clean):
    video_root, skel_root, _ = clean
    _make(video_root, skel_root, "test", "walk5_hq", 99)
    issues = gdm.check_split_consistency()
    assert issues["hq_pair"] == [[("walk5", "train"), ("walk5_hq", "test")]]
    assert not issues["same_content"]


def test_missing_video_does_not_block(clean):
    video_root, _, files = clean
    files["stop_1"][0].unlink()
    issues = gdm.check_split_consistency()
    assert issues["missing"] == ["stop_1"]
    assert _blocking(issues) == {}


def test_skeleton_in_wrong_class_folder_blocks(clean):
    """錯誤示範 6：lick_8.json 拖到同 split 的 walk 資料夾：檔名開頭就是類別，放錯就擋。"""
    _, skel_root, files = clean
    shutil.move(str(files["lick_8"][1]), str(skel_root / "train" / "walk" / "lick_8.json"))
    issues = gdm.check_split_consistency()
    assert issues["misplaced"] == [("lick_8", "骨架", "train/walk")]
    assert set(_blocking(issues)) == {"misplaced"}


def test_video_in_wrong_class_folder_blocks(clean):
    """lick_8.mp4 拖到 模型專用/train/walk/：抽骨架會被當成 walk，一樣擋。"""
    video_root, _, files = clean
    shutil.move(str(files["lick_8"][0]), str(video_root / "train" / "walk" / "lick_8.mp4"))
    issues = gdm.check_split_consistency()
    assert issues["misplaced"] == [("lick_8", "影片", "train/walk")]
    assert set(_blocking(issues)) == {"misplaced"}


def test_report_numbers_every_blocking_kind(clean):
    """split_issue_report：每一種擋下的問題一個編號區塊，missing 只放在提醒。"""
    video_root, skel_root, files = clean
    shutil.move(str(files["lick_8"][1]), str(skel_root / "train" / "walk" / "lick_8.json"))
    files["stop_1"][0].unlink()
    sections, notes = gdm.split_issue_report(gdm.check_split_consistency(), skel_root)
    assert [h for h, _ in sections] == ["[1] 放在別的類別資料夾（檔名開頭就是類別）：1 筆"]
    assert notes == [("找不到原始影片", ["stop_1"])]


def test_find_skeleton_finds_file_in_wrong_class_folder(tmp_path):
    """放錯類別資料夾的骨架也要找得到，否則重抽骨架會另外建一份新檔（同名兩份、標記遺失）。"""
    js = tmp_path / "train" / "walk" / "lick_10.json"
    js.parent.mkdir(parents=True)
    js.write_text("{}", encoding="utf-8")
    assert ss.find_skeleton(tmp_path, "lick_10") == js
    assert ss.skeleton_path_for(tmp_path, "lick_10", split="train") == js


def test_copied_json_renamed_is_foreign_and_duplicate(clean):
    """錯誤示範 7：把 stop_1.json（test）複製改名成 stop_777.json 放到 train：
    記錄的影片不是自己的，而且內容跟 stop_1 重複（資料洩漏）。"""
    _, skel_root, files = clean
    (skel_root / "train" / "stop").mkdir(parents=True, exist_ok=True)
    shutil.copy(files["stop_1"][1], skel_root / "train" / "stop" / "stop_777.json")
    issues = gdm.check_split_consistency()
    assert issues["foreign"] == [("stop_777", "stop_1.mp4")]
    assert issues["same_content"] == [[("stop_1", "test"), ("stop_777", "train")]]
    assert not issues["video"]   # 別人的影片位置不拿來比


def test_every_clash_copy_is_checked(clean):
    """同名多份：每一份都檢查類別資料夾，也都參加內容重複比對（使用者實測的 walk777 情境）。"""
    _, skel_root, files = clean
    for rel in ["train/walk", "val/walk", "test/stop"]:
        (skel_root / rel).mkdir(parents=True, exist_ok=True)
        shutil.copy(files["walk5"][1], skel_root / rel / "walk777.json")   # 同類別內容（walk33→walk777 的情境）
    issues = gdm.check_split_consistency()
    assert [v for v, _ in issues["clash"]] == ["walk777"]
    assert ("walk777", "骨架", "test/stop") in issues["misplaced"]
    grp = issues["same_content"][0]
    assert ("walk5", "train") in grp and ("walk777", "val/walk") in grp


def test_sequential_names_with_different_content_are_not_grouped(clean):
    """shake_1、shake_2 這種流水號，只要影片內容不同就不算重複（判斷看座標，不看檔名）。"""
    video_root, skel_root, _ = clean
    _make(video_root, skel_root, "train", "shake_1", 11)
    _make(video_root, skel_root, "val", "shake_2", 12)
    issues = gdm.check_split_consistency()
    assert not issues["same_content"] and not issues["hq_pair"]
