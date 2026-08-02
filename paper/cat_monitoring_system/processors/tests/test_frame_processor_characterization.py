"""
FrameProcessor.process() Characterization / Regression Test
（Michael Feathers 風格：先把「目前系統對固定輸入的實際輸出」凍結成快照，
再用快照當回歸基準——不主張這個輸出「正確」，只主張「不應該無故改變」。）

前提與範圍
----------
`process()` 本身依賴真實 YOLO/ST-GCN 模型與真實影片幀，因此本測試**刻意
不 mock 這些依賴**（跟 Unit Test 階段「不得依賴 GPU/YOLO」的規則不同：
這裡是 Regression Test，目的是保護「真實系統」的行為，mock 掉核心依賴
會讓保護網失去意義）。因此本測試：

- 需要真實模型檔（`ModelPaths.YOLO_MODEL` / `ModelPaths.STGCN_MODEL`）與
  固定測試影片（`_snapshot_utils.FIXED_VIDEO`）存在，否則自動跳過。
- 浮點數（confidence / class_probs）使用容許誤差比較（見下方
  `_FLOAT_TOLERANCE`），不做完全相等比較——GPU 推論結果在不同硬體/驅動/
  cuDNN 版本下可能有極小的浮點差異，這是預期中的正常變動，不是回歸。
- `behavior_id` / `is_still` / `activity_value` 為離散值，仍要求完全相等：
  這些不該因為浮點誤差而改變。

未覆蓋範圍（刻意排除，原因見 `_snapshot_utils.py` 模組說明）：
- 疊圖後的畫面像素內容（只驗證 shape）
- BehaviorTracker 的累積時長類欄位（wall-clock 相依，交給 BehaviorTracker
  專屬的 regression test 處理）
- Node-RED payload 實際發送、CSV 實際寫入內容（交給後續的 Golden Dataset）

若快照檔不存在，測試會明確失敗並提示先執行
`generate_frame_processor_snapshot.py`，而不是自動幫你產生一份——自動產生
會讓這個測試永遠通過，喪失偵測回歸的能力。
"""

import json
from pathlib import Path

import pytest

from config import LoggingConfig, ModelPaths

from _snapshot_utils import FIXED_VIDEO, FRAME_COUNT, SNAPSHOT_PATH, capture_process_outputs

_FLOAT_TOLERANCE = 1e-4  # GPU 浮點數合理誤差；不同硬體/cuDNN 版本可能有極小差異

_MODELS_AVAILABLE = Path(ModelPaths.YOLO_MODEL).exists() and Path(
    ModelPaths.STGCN_MODEL
).exists()
_VIDEO_AVAILABLE = Path(FIXED_VIDEO).exists()

pytestmark = pytest.mark.skipif(
    not (_MODELS_AVAILABLE and _VIDEO_AVAILABLE),
    reason=(
        "需要真實 YOLO/ST-GCN 模型檔與固定測試影片才能執行 "
        "FrameProcessor characterization test（此為 Regression Test，"
        "刻意不 mock 核心依賴）"
    ),
)

# `frame_processor.py` 在模組層級直接 `import cv2`，若在沒裝 cv2 的環境（例如
# base conda env）collect 這個檔案，光是 import 就會丟 ModuleNotFoundError，
# 讓 `pytestmark` 的 skipif 完全來不及生效，甚至會讓同一次 pytest 呼叫的
# 其他測試檔案也一併收集失敗（pytest 遇到 collection error 預設會整個中斷）。
# 用 `pytest.importorskip` 把這個高風險 import 往後挪：cv2 不存在時直接跳過
# 這個檔案剩下的收集過程，不會拋出未被攔截的例外。
pytest.importorskip("cv2", reason="processors/frame_processor.py 需要 cv2，此環境未安裝")
from processors.frame_processor import FrameProcessor  # noqa: E402


@pytest.fixture
def isolated_frame_processor(tmp_path, monkeypatch):
    """建立一個不會碰到正式 CSV / tracker_state.json 的 FrameProcessor。

    只 monkeypatch 測試期間的 LoggingConfig 屬性值（pytest 測試結束後自動
    還原），不修改 config.py 原始碼、不影響正式環境的任何檔案。
    """
    monkeypatch.setattr(LoggingConfig, "CSV_PATH", str(tmp_path / "test_cat_monitoring_log.csv"))
    monkeypatch.setattr(
        LoggingConfig, "SEGMENTS_CSV_PATH", str(tmp_path / "test_behavior_segments_log.csv")
    )
    monkeypatch.setattr(
        LoggingConfig, "TRACKER_STATE_PATH", str(tmp_path / "test_tracker_state.json")
    )

    processor = FrameProcessor(
        yolo_model_path=ModelPaths.YOLO_MODEL,
        stgcn_model_path=ModelPaths.STGCN_MODEL,
        video_path=FIXED_VIDEO,
        nodered_url=None,  # 明確關閉 Node-RED 推送，測試不會發出任何網路請求
        device="cuda",
        overlay=True,
    )
    yield processor
    processor.cleanup()


def _load_snapshot() -> dict:
    if not SNAPSHOT_PATH.exists():
        pytest.fail(
            f"找不到基準快照 {SNAPSHOT_PATH}。\n"
            "這是 characterization test 的基準線，須先手動執行一次：\n"
            "    python processors/tests/generate_frame_processor_snapshot.py\n"
            "確認輸出合理後，把產生的快照檔一併提交，測試才有比較基準。"
        )
    with open(SNAPSHOT_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_process_output_matches_characterization_snapshot(isolated_frame_processor):
    """對固定影片的固定幀數逐一呼叫 process()，輸出需與已凍結的基準快照一致。"""
    snapshot = _load_snapshot()
    baseline_records = snapshot["records"]

    current_records = capture_process_outputs(isolated_frame_processor, FRAME_COUNT)

    assert len(current_records) == len(baseline_records), (
        f"實際處理幀數（{len(current_records)}）與快照記錄的幀數"
        f"（{len(baseline_records)}）不一致，代表影片讀取或 process() 提早/"
        "延後結束的行為被改變了。"
    )

    for i, (current, baseline) in enumerate(zip(current_records, baseline_records)):
        assert current["output_frame_shape"] == baseline["output_frame_shape"], (
            f"第 {i} 幀：輸出畫面 shape 改變 "
            f"{baseline['output_frame_shape']} -> {current['output_frame_shape']}"
        )
        assert current["behavior_id"] == baseline["behavior_id"], (
            f"第 {i} 幀：behavior_id 改變 {baseline['behavior_id']} -> {current['behavior_id']}"
        )
        assert current["is_still"] == baseline["is_still"], (
            f"第 {i} 幀：is_still 改變 {baseline['is_still']} -> {current['is_still']}"
        )
        assert current["activity_value"] == baseline["activity_value"], (
            f"第 {i} 幀：activity_value 改變 "
            f"{baseline['activity_value']} -> {current['activity_value']}"
        )
        assert current["confidence"] == pytest.approx(
            baseline["confidence"], abs=_FLOAT_TOLERANCE
        ), f"第 {i} 幀：confidence 超出容許誤差"
        assert current["class_probs"] == pytest.approx(
            baseline["class_probs"], abs=_FLOAT_TOLERANCE
        ), f"第 {i} 幀：class_probs 超出容許誤差"


def test_snapshot_records_at_least_one_behavior_transition():
    """健全性檢查：確保基準快照本身不是全部靜止/全部 LOW_CONF 的退化案例。

    這不是 process() 的行為驗證，是驗證「快照本身有沒有意義」——避免快照
    意外記錄到一段完全沒有偵測到貓、或推論從未觸發的空白影片，讓上面的
    回歸測試看起來通過、實際上什麼都沒保護到。
    """
    snapshot = _load_snapshot()
    behavior_ids = {r["behavior_id"] for r in snapshot["records"]}
    assert len(snapshot["records"]) > 0, "快照是空的"
    assert behavior_ids != {-2}, "快照全程都是「貓不在畫面」（NOT_VISIBLE），無法驗證有效推論路徑"
