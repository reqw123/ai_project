"""
一次性腳本：對 walk/lick/scratch/shake/stop 五種行為各一支固定影片，跑過真正
的 FrameProcessor pipeline（YOLO + ST-GCN + BehaviorTracker + CSVLogger +
BehaviorSegmentLogger），把下列五類輸出凍結成快照：

    1. prediction        —— 逐幀 process() 的數值輸出
    2. statistics        —— BehaviorTracker 的累積統計（today_stats、activity_score、alerts）
    3. Node-RED payload  —— _build_nodered_payload() 的完整內容（不實際發送）
    4. CSV               —— CSVLogger 實際寫出的逐幀紀錄
    5. behavior segments —— BehaviorSegmentLogger 實際寫出的行為區段紀錄

⚠ 這不是 pytest 測試，是「建立/更新 Golden Dataset 基準」的手動工具，只有
確認目前行為就是要保護的基準時才執行：

    python processors/tests/generate_golden_dataset_snapshot.py

用 GoldenClock 把 time.time()/time.strftime()/datetime.now() 全部控制住，
讓原本會受「真實處理速度」影響的 BehaviorTracker 累積時長、CSV 時間戳、
Node-RED payload 時間戳，都變成確定性可重現（不需要事後排除這些欄位）。
浮點數（confidence/class_probs/activity_value 等 ST-GCN 輸出）比對時仍需要
容許誤差，這是 GPU 推論的正常變動，不是回歸。

只會寫入自己的 _scratch/ 暫存目錄與 snapshots/golden_dataset/ 快照檔案，
不會修改任何正式程式碼、不會寫入正式 CSV，也不會碰正式的 tracker_state.json
（見下方 LoggingConfig 攔截說明）。
"""

import json
import sys
import time
from pathlib import Path

_processors_tests_dir = Path(__file__).resolve().parent
_cat_monitoring_system_dir = _processors_tests_dir.parents[1]
_paper_dir = _cat_monitoring_system_dir.parent
for _p in (_cat_monitoring_system_dir, _paper_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import LoggingConfig, ModelPaths  # noqa: E402
from logutils import csv_logger as csv_logger_module  # noqa: E402
from processors.frame_processor import FrameProcessor  # noqa: E402
from trackers import behavior_tracker as behavior_tracker_module  # noqa: E402

from _golden_dataset_utils import (  # noqa: E402
    GOLDEN_VIDEOS,
    SNAPSHOT_DIR,
    GoldenClock,
    process_video_golden,
    read_csv_rows,
)

_FIXED_START = __import__("datetime").datetime(2026, 1, 1, 8, 0, 0)


def _run_one_behavior(behavior_name: str, video_path: str, frame_count: int, scratch_dir: Path) -> dict:
    if not Path(video_path).exists():
        raise FileNotFoundError(f"[{behavior_name}] 固定測試影片不存在: {video_path}")

    clock = GoldenClock(_FIXED_START)
    # 統一攔截 time.time / time.strftime（全域 time 模組，frame_processor.py
    # 與 csv_logger.py 皆用得到）以及 behavior_tracker.py / csv_logger.py 各自
    # `from datetime import datetime` 進來的 datetime 名稱。純測試期間的
    # 記憶體覆寫，腳本結束後沒有任何殘留效果。
    time.time = clock.time
    time.strftime = clock.strftime
    behavior_tracker_module.datetime = clock
    csv_logger_module.datetime = clock

    behavior_dir = scratch_dir / behavior_name
    behavior_dir.mkdir(parents=True, exist_ok=True)
    LoggingConfig.CSV_PATH = str(behavior_dir / "cat_monitoring_log.csv")
    LoggingConfig.SEGMENTS_CSV_PATH = str(behavior_dir / "behavior_segments_log.csv")
    LoggingConfig.TRACKER_STATE_PATH = str(behavior_dir / "tracker_state.json")

    processor = FrameProcessor(
        yolo_model_path=ModelPaths.YOLO_MODEL,
        stgcn_model_path=ModelPaths.STGCN_MODEL,
        video_path=video_path,
        nodered_url=None,  # 明確關閉 Node-RED 推送，不會發出任何網路請求
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

    csv_rows = read_csv_rows(Path(LoggingConfig.CSV_PATH))
    segment_rows = read_csv_rows(Path(LoggingConfig.SEGMENTS_CSV_PATH))

    return {
        "behavior": behavior_name,
        "video": video_path,
        "frame_count": len(predictions),
        "prediction": predictions,
        "statistics": statistics,
        "nodered_payload": nodered_payload,
        "csv_rows": csv_rows,
        "behavior_segment_rows": segment_rows,
    }


def main() -> None:
    if not Path(ModelPaths.YOLO_MODEL).exists():
        raise FileNotFoundError(f"YOLO 模型不存在: {ModelPaths.YOLO_MODEL}")
    if not Path(ModelPaths.STGCN_MODEL).exists():
        raise FileNotFoundError(f"ST-GCN 模型不存在: {ModelPaths.STGCN_MODEL}")

    scratch_dir = _processors_tests_dir / "_scratch_golden"
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    for behavior_name, (video_path, frame_count) in GOLDEN_VIDEOS.items():
        print(f"處理 [{behavior_name}] {video_path} ({frame_count} 幀)...")
        result = _run_one_behavior(behavior_name, video_path, frame_count, scratch_dir)
        out_path = SNAPSHOT_DIR / f"{behavior_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  已寫入 {out_path}（{result['frame_count']} 幀，"
              f"{len(result['csv_rows'])} 筆 CSV，{len(result['behavior_segment_rows'])} 筆行為區段）")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
