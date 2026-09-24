"""
Unit Test：新骨架的 split 跟著影片所在的資料夾（skeleton_splits.split_of_video /
skeleton_path_for 的 split 參數）。全部在 tmp_path 底下，不碰真正的影片或骨架。
"""

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER / "cat_monitoring_system"))
from utils import skeleton_splits as ss  # noqa: E402


def test_split_of_video_reads_split_folder(tmp_path):
    root = tmp_path / "模型專用"
    assert ss.split_of_video(root / "val" / "lick" / "lick_1.mp4", root) == "val"
    assert ss.split_of_video(root / "test" / "walk" / "walk_2.mp4", root) == "test"


def test_split_of_video_none_for_legacy_or_outside(tmp_path):
    root = tmp_path / "模型專用"
    assert ss.split_of_video(root / "lick" / "lick_1.mp4", root) is None       # 舊排法
    assert ss.split_of_video(tmp_path / "別處" / "lick_1.mp4", root) is None   # 不在 root 底下
    assert ss.split_of_video(root / "val", root) is None                        # 資料夾本身


def test_new_skeleton_follows_video_split(tmp_path):
    skel = tmp_path / "skeletons"
    assert ss.skeleton_path_for(skel, "lick_77", split="val") == skel / "val" / "lick" / "lick_77.json"
    assert ss.skeleton_path_for(skel, "walk_77") == skel / "train" / "walk" / "walk_77.json"
    assert ss.skeleton_path_for(skel, "walk_78", split=None) == skel / "train" / "walk" / "walk_78.json"


def test_existing_skeleton_keeps_its_split(tmp_path):
    skel = tmp_path / "skeletons"
    existing = skel / "test" / "lick" / "lick_5.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}", encoding="utf-8")
    # 影片在 val，但骨架已經在 test：重抽時寫回原位，不偷偷換 split（交給模式 2 的警告）
    assert ss.skeleton_path_for(skel, "lick_5", split="val") == existing


def test_find_video_by_name_any_split(tmp_path):
    root = tmp_path / "模型專用"
    v = root / "val" / "walk" / "walk12.mp4"
    v.parent.mkdir(parents=True)
    v.write_bytes(b"x")
    assert ss.find_video("walk12.mp4", root) == v
    assert ss.find_video("walk_12.mp4", root) is None


def test_find_video_legacy_and_ambiguous(tmp_path):
    root = tmp_path / "模型專用"
    legacy = root / "lick" / "lick_1.mp4"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"x")
    assert ss.find_video("lick_1.mp4", root) == legacy
    dup = root / "train" / "lick" / "lick_1.mp4"
    dup.parent.mkdir(parents=True)
    dup.write_bytes(b"x")
    assert ss.find_video("lick_1.mp4", root) is None   # 同名兩支：不猜
