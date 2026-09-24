"""
Unit Test：原始影片跟骨架同一套 train/val/test 切分（skeleton_splits 的影片函式、
gcn_dataset_manager 的模式 2 連帶搬影片、模式 3 同步、模式 4 還原）。

全部在 tmp_path 底下造假的 模型專用/ 與 skeletons/，並把 VIDEO_ROOT／
VIDEO_MOVES_LOG／骨架資料夾設定都指到 tmp_path，不會碰到真正的影片或骨架。
"""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PAPER / "cat_monitoring_system"))
from utils import skeleton_splits as ss  # noqa: E402

_spec = importlib.util.spec_from_file_location("gcn_dataset_manager", _PAPER / "tools" / "gcn_dataset_manager.py")
gdm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gdm)


@pytest.fixture
def env(tmp_path, monkeypatch):
    video_root = tmp_path / "模型專用"
    skel_root = tmp_path / "skeletons"
    monkeypatch.setattr(ss, "VIDEO_ROOT", video_root)
    monkeypatch.setattr(gdm, "VIDEO_ROOT", video_root)
    monkeypatch.setattr(gdm, "VIDEO_MOVES_LOG", video_root / "_video_moves_log.csv")
    monkeypatch.setattr(gdm, "_load_config", lambda: {"SKELETON_DATA_FOLDER": str(skel_root)})
    return video_root, skel_root


def _make(video_root, skel_root, split, cls, stem, video_dir=None):
    """造一支影片（預設放舊排法 模型專用/<類別>/）和它的骨架 skeletons/<split>/<類別>/<stem>.json。"""
    vdir = video_dir or (video_root / cls)
    vdir.mkdir(parents=True, exist_ok=True)
    video = vdir / f"{stem}.mp4"
    video.write_bytes(b"fake video " + stem.encode())
    js = skel_root / split / cls / f"{stem}.json"
    js.parent.mkdir(parents=True, exist_ok=True)
    # 故意用跟 0_dataset_collect.py 一樣的格式（縮排 2、非 ASCII 不跳脫），另外放一個
    # 浮點數，確認改寫 video_path 時其他內容一個位元組都沒動
    data = {"video_metadata": {"video_id": stem, "video_path": str(video), "actual_fps": 24.0},
            "frames": [{"frame_id": 0, "keypoints": [{"x": 867.0, "y": 1.25}]}]}
    js.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return video, js


def _recorded(js):
    return json.loads(js.read_text(encoding="utf-8"))["video_metadata"]["video_path"]


def test_move_video_with_skeleton_rewrites_only_video_path(env):
    video_root, skel_root = env
    video, js = _make(video_root, skel_root, "val", "lick", "lick_21")
    before = js.read_text(encoding="utf-8")

    old, new = ss.move_video_with_skeleton(js, "val")

    assert Path(old) == video and not video.exists()
    assert Path(new) == video_root / "val" / "lick" / "lick_21.mp4" and Path(new).exists()
    assert _recorded(js) == new
    after = js.read_text(encoding="utf-8")
    assert after == before.replace(json.dumps(str(video), ensure_ascii=False),
                                   json.dumps(new, ensure_ascii=False))
    assert ss.move_video_with_skeleton(js, "val") is None   # 已在正確位置 → 不動


def test_move_video_refuses_to_overwrite(env):
    video_root, skel_root = env
    video, js = _make(video_root, skel_root, "test", "stop", "stop_53")
    clash = video_root / "test" / "stop" / "stop_53.mp4"
    clash.parent.mkdir(parents=True)
    clash.write_bytes(b"other")
    with pytest.raises(FileExistsError):
        ss.move_video_with_skeleton(js, "test")
    assert video.exists() and _recorded(js) == str(video)


def test_video_outside_root_is_left_alone(env, tmp_path):
    video_root, skel_root = env
    _, js = _make(video_root, skel_root, "train", "walk", "walk_9", video_dir=tmp_path / "別的資料夾")
    assert ss.move_video_with_skeleton(js, "train") is None


def test_video_class_folders_covers_new_and_legacy_layouts(env):
    video_root, _ = env
    for d in ["train/walk", "val/walk", "test/lick", "walk", "lick"]:
        (video_root / d).mkdir(parents=True)
    (video_root / "walk" / "new.mp4").write_bytes(b"v")        # 舊排法裡還有影片 → 列入
    (video_root / "lick" / "筆記.txt").write_text("x", encoding="utf-8")   # 只剩非影片檔 → 不列
    got = [Path(p).relative_to(video_root).as_posix() for p in ss.video_class_folders()]
    assert got == ["train/walk", "val/walk", "walk", "test/lick"]
    assert [Path(p).parent.name for p in ss.video_class_folders(classes="lick")] == ["test"]


def test_sync_migrates_legacy_layout_and_undo_restores(env):
    video_root, skel_root = env
    made = {
        "walk_1": _make(video_root, skel_root, "train", "walk", "walk_1"),
        "lick_21": _make(video_root, skel_root, "val", "lick", "lick_21"),
        "stop_53": _make(video_root, skel_root, "test", "stop", "stop_53"),
    }
    orphan = video_root / "shake" / "shake_new.mp4"   # 新影片、還沒抽骨架
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"new")
    snapshot = {k: (js.read_text(encoding="utf-8"), v) for k, (v, js) in made.items()}

    gdm.run_video_sync(confirm=False)

    for stem, (video, js) in made.items():
        split = js.parent.parent.name
        dst = video_root / split / js.parent.name / video.name
        assert dst.exists() and not video.exists(), stem
        assert _recorded(js) == str(dst), stem
    assert (video_root / "train" / "shake" / "shake_new.mp4").exists()
    for sp in ss.SPLITS:                        # 每個 split 都建好五個類別資料夾
        for c in ss.VIDEO_CLASSES:
            assert (video_root / sp / c).is_dir()
    for c in ss.VIDEO_CLASSES:                  # 舊排法資料夾清空後移除
        assert not (video_root / c).exists()
    assert gdm.plan_video_sync(skel_root)["moves"] == []   # 再跑一次不用搬

    gdm.undo_last_video_sync(confirm=False)

    for stem, (text, video) in snapshot.items():
        js = made[stem][1]
        assert video.exists(), stem
        assert js.read_text(encoding="utf-8") == text, stem   # JSON 完整還原
    assert orphan.exists()


def test_sync_recovers_manually_moved_video_by_filename(env):
    video_root, skel_root = env
    video, js = _make(video_root, skel_root, "val", "walk", "walk_5")
    manual = video_root / "train" / "walk" / "walk_5.mp4"   # 使用者在檔案總管拖到錯的地方
    manual.parent.mkdir(parents=True)
    video.rename(manual)

    plan = gdm.plan_video_sync(skel_root)
    assert [(v, a) for v, _, a, _ in plan["moves"]] == [("walk_5", manual)]
    gdm.run_video_sync(confirm=False)

    dst = video_root / "val" / "walk" / "walk_5.mp4"
    assert dst.exists() and _recorded(js) == str(dst)


def test_sync_relinks_when_video_already_in_place(env):
    video_root, skel_root = env
    video, js = _make(video_root, skel_root, "train", "lick", "lick_3")
    right = video_root / "train" / "lick" / "lick_3.mp4"
    right.parent.mkdir(parents=True)
    video.rename(right)                         # 位置正確，但 JSON 還記著舊路徑

    plan = gdm.plan_video_sync(skel_root)
    assert plan["moves"] == [] and [r[0] for r in plan["relinks"]] == ["lick_3"]
    gdm.run_video_sync(confirm=False)
    assert _recorded(js) == str(right)


def test_sync_reports_missing_and_ambiguous_videos(env):
    video_root, skel_root = env
    video, _ = _make(video_root, skel_root, "train", "walk", "walk_7")
    video.unlink()
    v2, _ = _make(video_root, skel_root, "val", "stop", "stop_2")
    dup = video_root / "train" / "stop" / "stop_2.mp4"
    dup.parent.mkdir(parents=True)
    dup.write_bytes(b"x")
    v2.unlink()
    (video_root / "val" / "stop").mkdir(parents=True)
    (video_root / "val" / "stop" / "stop_2.mp4").write_bytes(b"y")

    problems = gdm.plan_video_sync(skel_root)["problems"]
    assert any("walk_7" in p and "找不到影片" in p for p in problems)
    assert any("stop_2" in p and "同名影片" in p for p in problems)


def test_split_move_carries_video_and_logs_it(env):
    video_root, skel_root = env
    _make(video_root, skel_root, "train", "walk", "walk_1")
    gdm.run_video_sync(confirm=False)
    js = skel_root / "train" / "walk" / "walk_1.json"
    state = {"root": skel_root, "paths": {"walk_1": js}, "current": {"walk_1": "train"}}

    gdm.apply_moves(state, [("walk_1", "train", "test")], "gui")

    new_js = skel_root / "test" / "walk" / "walk_1.json"
    dst = video_root / "test" / "walk" / "walk_1.mp4"
    assert new_js.exists() and dst.exists()
    assert _recorded(new_js) == str(dst)
    assert state["video_warnings"] == []
    with open(video_root / "_video_moves_log.csv", newline="", encoding="utf-8-sig") as f:
        last = list(csv.DictReader(f))[-1]
    assert (last["mode"], last["video_id"], last["to"]) == ("gui", "walk_1", str(dst))


def test_split_move_keeps_skeleton_move_when_video_is_blocked(env):
    video_root, skel_root = env
    _make(video_root, skel_root, "train", "walk", "walk_1")
    gdm.run_video_sync(confirm=False)
    blocker = video_root / "val" / "walk" / "walk_1.mp4"   # 目的地被佔住 → 影片搬不過去
    blocker.write_bytes(b"other")
    js = skel_root / "train" / "walk" / "walk_1.json"
    state = {"root": skel_root, "paths": {"walk_1": js}, "current": {"walk_1": "train"}}

    gdm.apply_moves(state, [("walk_1", "train", "val")], "gui")

    new_js = skel_root / "val" / "walk" / "walk_1.json"
    assert new_js.exists()                                  # 骨架照搬
    assert _recorded(new_js) == str(video_root / "train" / "walk" / "walk_1.mp4")   # 仍指向影片實際位置
    assert len(state["video_warnings"]) == 1 and "walk_1" in state["video_warnings"][0]
