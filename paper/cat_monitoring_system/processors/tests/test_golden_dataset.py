"""
Golden Dataset Regression Test —— 覆蓋 walk/lick/scratch/shake/stop 五種行為，
驗證 prediction / statistics / Node-RED payload / CSV / behavior segments
五類輸出是否與已凍結的基準快照一致。

跟 `test_frame_processor_characterization.py`（單一 walk 影片、只驗證
process() 回傳值、刻意排除 BehaviorTracker 累積統計）不同：這裡用
`GoldenClock` 把 `time.time()`/`time.strftime()`/`datetime.now()` 全部控制住
（見 `_golden_dataset_utils.py`），讓原本因為「真實處理速度」不可重現的
BehaviorTracker 累積時長、CSV 時間戳、Node-RED payload 時間戳，也都變成
確定性可重現，因此這裡**不排除**任何欄位，五類輸出全部納入比對。

同樣刻意不 mock YOLO/ST-GCN（Regression Test 要保護的是真實系統行為），
所以浮點數（confidence/class_probs/activity_value/CSV 內的數值欄位）仍用
容許誤差比較；behavior_id/is_still/次數/文字內容等離散值要求完全相等。

基準快照由 `generate_golden_dataset_snapshot.py` 產生，若不存在則測試直接
失敗並提示先執行該腳本，不會自動產生。
"""

import json
from pathlib import Path

import pytest

from config import LoggingConfig, ModelPaths
from logutils import csv_logger as csv_logger_module
from trackers import behavior_tracker as behavior_tracker_module

from _golden_dataset_utils import (
    GOLDEN_VIDEOS,
    SNAPSHOT_DIR,
    GoldenClock,
    process_video_golden,
    read_csv_rows,
)

_FLOAT_TOLERANCE = 1e-4  # GPU 浮點數合理誤差
_FIXED_START = __import__("datetime").datetime(2026, 1, 1, 8, 0, 0)

_MODELS_AVAILABLE = Path(ModelPaths.YOLO_MODEL).exists() and Path(
    ModelPaths.STGCN_MODEL
).exists()
_VIDEOS_AVAILABLE = all(Path(p).exists() for p, _ in GOLDEN_VIDEOS.values())

pytestmark = pytest.mark.skipif(
    not (_MODELS_AVAILABLE and _VIDEOS_AVAILABLE),
    reason="需要真實 YOLO/ST-GCN 模型檔與 5 支固定測試影片才能執行 Golden Dataset regression test",
)

# `frame_processor.py` 模組層級直接 `import cv2`；沒裝 cv2 的環境下光是收集
#這個測試檔案就會整個中斷（見 test_frame_processor_characterization.py 同樣
# 的處理方式），故一樣用 importorskip 讓 collection 安全跳過。
pytest.importorskip("cv2", reason="processors/frame_processor.py 需要 cv2，此環境未安裝")
from processors.frame_processor import FrameProcessor  # noqa: E402


def _load_snapshot(behavior_name: str) -> dict:
    path = SNAPSHOT_DIR / f"{behavior_name}.json"
    if not path.exists():
        pytest.fail(
            f"找不到 Golden Dataset 基準快照 {path}。\n"
            "請先手動執行一次：\n"
            "    python processors/tests/generate_golden_dataset_snapshot.py\n"
            "確認輸出合理後，把產生的快照檔一併提交，測試才有比較基準。"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def golden_result(request, tmp_path, monkeypatch):
    """對指定行為的固定影片，用完全受控的假時鐘跑過真實 pipeline，回傳跟
    generate_golden_dataset_snapshot.py 相同結構的結果 dict。"""
    behavior_name = request.param
    video_path, frame_count = GOLDEN_VIDEOS[behavior_name]

    clock = GoldenClock(_FIXED_START)
    monkeypatch.setattr("time.time", clock.time)
    monkeypatch.setattr("time.strftime", clock.strftime)
    monkeypatch.setattr(behavior_tracker_module, "datetime", clock)
    monkeypatch.setattr(csv_logger_module, "datetime", clock)

    csv_path = tmp_path / "cat_monitoring_log.csv"
    segments_path = tmp_path / "behavior_segments_log.csv"
    monkeypatch.setattr(LoggingConfig, "CSV_PATH", str(csv_path))
    monkeypatch.setattr(LoggingConfig, "SEGMENTS_CSV_PATH", str(segments_path))
    monkeypatch.setattr(LoggingConfig, "TRACKER_STATE_PATH", str(tmp_path / "tracker_state.json"))

    processor = FrameProcessor(
        yolo_model_path=ModelPaths.YOLO_MODEL,
        stgcn_model_path=ModelPaths.STGCN_MODEL,
        video_path=video_path,
        nodered_url=None,
        device="cuda",
        overlay=True,
    )
    try:
        predictions = process_video_golden(processor, clock, frame_count)
        nodered_payload = processor._build_nodered_payload(
            processor._display_behavior_id, processor._display_confidence
        )
        statistics = {
            "today_stats": processor.tracker.get_today_stats(),
            "activity_score": processor.tracker.get_activity_score(),
            "alerts": [a["title"] for a in processor.tracker.get_alerts()],
        }
    finally:
        processor.cleanup()

    return {
        "behavior": behavior_name,
        "prediction": predictions,
        "statistics": statistics,
        "nodered_payload": nodered_payload,
        "csv_rows": read_csv_rows(csv_path),
        "behavior_segment_rows": read_csv_rows(segments_path),
    }


def _assert_predictions_match(current: list, baseline: list, behavior_name: str):
    assert len(current) == len(baseline), (
        f"[{behavior_name}] 實際處理幀數（{len(current)}）與快照記錄的幀數"
        f"（{len(baseline)}）不一致"
    )
    for i, (cur, base) in enumerate(zip(current, baseline)):
        assert cur["behavior_id"] == base["behavior_id"], (
            f"[{behavior_name}] 第 {i} 幀 behavior_id 改變 "
            f"{base['behavior_id']} -> {cur['behavior_id']}"
        )
        assert cur["is_still"] == base["is_still"], f"[{behavior_name}] 第 {i} 幀 is_still 改變"
        assert cur["activity_value"] == base["activity_value"], (
            f"[{behavior_name}] 第 {i} 幀 activity_value 改變"
        )
        assert cur["confidence"] == pytest.approx(base["confidence"], abs=_FLOAT_TOLERANCE)
        assert cur["class_probs"] == pytest.approx(base["class_probs"], abs=_FLOAT_TOLERANCE)


def _assert_csv_rows_match(current: list, baseline: list, behavior_name: str, label: str):
    assert len(current) == len(baseline), (
        f"[{behavior_name}] {label} 筆數改變：{len(baseline)} -> {len(current)}"
    )
    _FLOAT_FIELDS = {"GCN_Confidence", "Motion_Score", "duration_sec", "activity"}
    for i, (cur, base) in enumerate(zip(current, baseline)):
        assert cur.keys() == base.keys(), f"[{behavior_name}] {label} 第 {i} 列欄位改變"
        for key in base:
            if key in _FLOAT_FIELDS:
                assert float(cur[key]) == pytest.approx(float(base[key]), abs=_FLOAT_TOLERANCE), (
                    f"[{behavior_name}] {label} 第 {i} 列 {key} 超出容許誤差"
                )
            else:
                assert cur[key] == base[key], (
                    f"[{behavior_name}] {label} 第 {i} 列 {key} 改變："
                    f"{base[key]!r} -> {cur[key]!r}"
                )


@pytest.mark.parametrize("golden_result", list(GOLDEN_VIDEOS.keys()), indirect=True)
def test_golden_dataset_prediction(golden_result):
    """prediction：逐幀 behavior_id/is_still/activity_value 需完全相等，confidence/class_probs 容許誤差。"""
    baseline = _load_snapshot(golden_result["behavior"])
    _assert_predictions_match(
        golden_result["prediction"], baseline["prediction"], golden_result["behavior"]
    )


@pytest.mark.parametrize("golden_result", list(GOLDEN_VIDEOS.keys()), indirect=True)
def test_golden_dataset_statistics(golden_result):
    """statistics：BehaviorTracker 累積統計（today_stats/activity_score/alerts）。"""
    baseline = _load_snapshot(golden_result["behavior"])
    behavior_name = golden_result["behavior"]

    cur_stats = golden_result["statistics"]["today_stats"]
    base_stats = baseline["statistics"]["today_stats"]
    assert cur_stats.keys() == base_stats.keys(), f"[{behavior_name}] today_stats 欄位改變"
    for key in base_stats:
        cur_v, base_v = cur_stats[key], base_stats[key]
        if isinstance(base_v, (int, float)) and not isinstance(base_v, bool):
            assert cur_v == pytest.approx(base_v, abs=_FLOAT_TOLERANCE), (
                f"[{behavior_name}] today_stats.{key} 改變：{base_v} -> {cur_v}"
            )
        else:
            assert cur_v == base_v, f"[{behavior_name}] today_stats.{key} 改變：{base_v} -> {cur_v}"

    assert golden_result["statistics"]["activity_score"] == baseline["statistics"]["activity_score"]
    assert golden_result["statistics"]["alerts"] == baseline["statistics"]["alerts"]


@pytest.mark.parametrize("golden_result", list(GOLDEN_VIDEOS.keys()), indirect=True)
def test_golden_dataset_nodered_payload(golden_result):
    """Node-RED payload：只驗證 payload 結構與內容一致，不實際發送任何網路請求。"""
    baseline = _load_snapshot(golden_result["behavior"])
    behavior_name = golden_result["behavior"]

    cur = golden_result["nodered_payload"]
    base = baseline["nodered_payload"]
    assert cur.keys() == base.keys(), f"[{behavior_name}] nodered_payload 頂層欄位改變"
    assert cur["current"] == base["current"], f"[{behavior_name}] nodered_payload['current'] 改變"
    assert cur["system"]["model"] == base["system"]["model"]
    assert cur["system"]["version"] == base["system"]["version"]
    assert cur["activity_score"] == base["activity_score"]


@pytest.mark.parametrize("golden_result", list(GOLDEN_VIDEOS.keys()), indirect=True)
def test_golden_dataset_csv(golden_result):
    """CSV：CSVLogger 實際寫出的逐幀紀錄。"""
    baseline = _load_snapshot(golden_result["behavior"])
    _assert_csv_rows_match(
        golden_result["csv_rows"], baseline["csv_rows"], golden_result["behavior"], "CSV"
    )


@pytest.mark.parametrize("golden_result", list(GOLDEN_VIDEOS.keys()), indirect=True)
def test_golden_dataset_behavior_segments(golden_result):
    """behavior segments：BehaviorSegmentLogger 實際寫出的行為區段紀錄。"""
    baseline = _load_snapshot(golden_result["behavior"])
    _assert_csv_rows_match(
        golden_result["behavior_segment_rows"],
        baseline["behavior_segment_rows"],
        golden_result["behavior"],
        "behavior segments",
    )
