"""
Unit Test：tools/1_keypoint_presence_lights.py 的缺失率統計（summarize／presence_mask）

模型是 run_analysis()／run_gui() 裡才載入，import 腳本本身不會跑推論，這裡直接餵假的信心值陣列。
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "1_keypoint_presence_lights.py"


@pytest.fixture(scope="module")
def mod():
    sys.path.insert(0, str(SCRIPT.parent))  # 腳本 import 同資料夾的 _window_zoom
    spec = importlib.util.spec_from_file_location("keypoint_presence_lights", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _confs(n, low_kps=(), low_every=1):
    """n 幀、全部 0.9；low_kps 裡的點每 low_every 幀有一幀掉到 0.1。"""
    c = np.full((n, 17), 0.9, dtype=np.float32)
    for k in low_kps:
        c[::low_every, k] = 0.1
    return c


def test_presence_mask_threshold_and_no_cat(mod):
    conf = np.full(17, 0.9)
    conf[10] = 0.69
    conf[11] = 0.7
    m = mod.presence_mask(conf, 0.7)
    assert not m[10] and m[11] and m.sum() == 16
    assert not mod.presence_mask(None, 0.7).any()


def test_video_average_not_dominated_by_long_video(mod):
    # lick 兩支片：長片 1000 幀完全不缺，短片 20 幀 LH_Knee(10) 全缺
    recs = [
        {"cls": "lick", "confs": _confs(1000), "n_no_cat": 0},
        {"cls": "lick", "confs": _confs(20, low_kps=[10]), "n_no_cat": 0},
    ]
    s = mod.summarize(recs, (0.5,), min_frames=10)[("lick", 0.5)]
    assert s["miss_video"][10] == pytest.approx(0.5)
    assert s["miss_frame"][10] == pytest.approx(20 / 1020)
    assert s["n_videos"] == 2


def test_no_cat_frames_excluded_from_keypoint_rate(mod):
    recs = [{"cls": "walk", "confs": _confs(30), "n_no_cat": 70}]
    s = mod.summarize(recs, (0.5,), min_frames=10)[("walk", 0.5)]
    assert np.all(s["miss_video"] == 0)
    assert s["no_cat_rate"] == pytest.approx(0.7)


def test_delta_marks_class_specific_missing(mod):
    # scratch 的 RF_Paw(9) 一半的幀缺，其他類別都不缺
    recs = [{"cls": c, "confs": _confs(40), "n_no_cat": 0} for c in ("walk", "lick", "shake", "stop")]
    recs.append({"cls": "scratch", "confs": _confs(40, low_kps=[9], low_every=2), "n_no_cat": 0})
    out = mod.summarize(recs, (0.5,), min_frames=10)
    assert out[("scratch", 0.5)]["delta"][9] == pytest.approx(0.5)
    assert out[("walk", 0.5)]["delta"][9] == pytest.approx(-0.5 / 4)


def test_short_videos_skipped_and_empty_class_is_nan(mod):
    recs = [{"cls": "shake", "confs": _confs(5), "n_no_cat": 0}]
    out = mod.summarize(recs, (0.5,), min_frames=10)
    assert out[("shake", 0.5)]["n_videos"] == 0
    assert np.all(np.isnan(out[("shake", 0.5)]["miss_video"]))
    assert np.all(np.isnan(out[("walk", 0.5)]["miss_frame"]))


def _record(cls, confs, pred):
    return {"video": f"{cls}.mp4", "cls": cls, "confs": confs, "pred": np.asarray(pred, np.int16)}


def test_group_by_folder_keeps_whole_video_and_counts_no_cat(mod):
    confs = _confs(4)
    confs[1] = np.nan  # 第 2 幀沒偵測到貓
    groups = mod.group_frames([_record("lick", confs, [0, 1, 2, -1])], "folder")
    assert len(groups) == 1
    g = groups[0]
    assert g["cls"] == "lick" and len(g["confs"]) == 3 and g["n_no_cat"] == 1


def test_group_by_stgcn_splits_frames_uncertain_kept_warmup_dropped(mod):
    # 資料夾是 lick；ST-GCN：第 1 幀 warmup、兩幀 walk、兩幀 lick、最後一幀信心不足
    confs = _confs(6, low_kps=[10])
    groups = mod.group_frames([_record("lick", confs, [-2, 0, 0, 1, 1, -1])], "stgcn")
    by_cls = {g["cls"]: g for g in groups}
    assert set(by_cls) == {"walk", "lick", "uncertain"}
    assert len(by_cls["walk"]["confs"]) == 2 and len(by_cls["lick"]["confs"]) == 2
    assert len(by_cls["uncertain"]["confs"]) == 1
    out = mod.summarize(groups, (0.5,), min_frames=1, classes=mod.STGCN_GROUPS)
    assert out[("walk", 0.5)]["miss_video"][10] == pytest.approx(1.0)


def test_folder_grouping_skips_videos_without_folder_label(mod):
    # ANALYSIS_SOURCE 指定跟行為無關的資料夾時 cls=None：只能依 ST-GCN 分組
    rec = _record(None, _confs(4), [0, 0, 3, 3])
    assert mod.group_frames([rec], "folder") == []
    assert {g["cls"] for g in mod.group_frames([rec], "stgcn")} == {"walk", "shake"}


def test_delta_baseline_excludes_uncertain(mod):
    # uncertain 幀 Hip(5) 全缺，但五個行為都不缺：各行為的 delta 不該被 uncertain 拉低
    recs = [{"cls": c, "confs": _confs(20), "n_no_cat": 0} for c in mod.BEHAVIOR_CLASSES]
    recs.append({"cls": "uncertain", "confs": _confs(20, low_kps=[5]), "n_no_cat": 0})
    out = mod.summarize(recs, (0.5,), min_frames=10, classes=mod.STGCN_GROUPS)
    assert out[("walk", 0.5)]["delta"][5] == pytest.approx(0.0)
    assert out[("uncertain", 0.5)]["delta"][5] == pytest.approx(1.0)


def test_count_dropouts_counts_on_to_off_transitions(mod):
    c = np.full((6, 17), 0.9, dtype=np.float32)
    c[[1, 2, 4], 7] = 0.1  # LF_Paw：有 無 無 有 無 有 → 兩次從有到無
    c[0, 9] = 0.1          # RF_Paw：一開始就無 → 0 次
    d = mod.count_dropouts(c, 0.5)
    assert d[7] == 2 and d[9] == 0 and d.sum() == 2


def test_count_dropouts_skips_no_cat_frames(mod):
    c = np.full((5, 17), 0.9, dtype=np.float32)
    c[1] = np.nan          # 貓整隻不見：不算 17 個點各缺一次
    c[3] = np.nan
    c[4, 10] = 0.1         # 有 → [沒貓] → 無：算一次
    d = mod.count_dropouts(c, 0.5)
    assert d[10] == 1 and d.sum() == 1


def test_gui_stats_match_count_dropouts(mod):
    rng = np.random.default_rng(0)
    c = rng.random((200, 17)).astype(np.float32)
    c[rng.random(200) < 0.1] = np.nan
    st = mod.PresenceStats()
    for row in c:
        st.update(None if np.isnan(row).all() else row, 0.5)
    assert np.array_equal(st.drop_count, mod.count_dropouts(c, 0.5))
    assert st.n_cat == int((~np.isnan(c).all(axis=1)).sum())
