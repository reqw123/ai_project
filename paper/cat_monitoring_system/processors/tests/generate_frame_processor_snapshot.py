"""
一次性腳本：對固定影片輸入跑過真正的 FrameProcessor.process()，把輸出凍結
成 JSON 快照，供 test_frame_processor_characterization.py 做回歸比對。

⚠ 這不是 pytest 測試，是「建立/更新基準快照」的手動工具。只有在你確認
「目前這個行為就是你要保護的基準」時才執行它。平常跑測試只會讀取既有的
快照做比較，不會自動重新產生——否則測試會失去偵測「行為被意外改變」的
能力。

執行方式（需要真實 YOLO/ST-GCN 模型檔 + yolo_new conda 環境，且需要
FIXED_VIDEO 指到的影片檔存在）：

    python processors/tests/generate_frame_processor_snapshot.py

本腳本只讀取模型/影片、寫入自己的 _scratch/ 暫存目錄與 snapshots/ 快照
檔案，不會修改任何正式程式碼、不會寫入正式 CSV，也不會讀取/寫入正式的
tracker_state.json（見下方 LoggingConfig 攔截說明）。
"""

import json
import sys
from pathlib import Path

_processors_tests_dir = Path(__file__).resolve().parent
_cat_monitoring_system_dir = _processors_tests_dir.parents[1]
_paper_dir = _cat_monitoring_system_dir.parent
for _p in (_cat_monitoring_system_dir, _paper_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import LoggingConfig, ModelPaths  # noqa: E402
from processors.frame_processor import FrameProcessor  # noqa: E402
from _snapshot_utils import (  # noqa: E402
    FIXED_VIDEO,
    FRAME_COUNT,
    SNAPSHOT_PATH,
    capture_process_outputs,
)


def main() -> None:
    scratch_dir = _processors_tests_dir / "_scratch"
    scratch_dir.mkdir(exist_ok=True)

    # FrameProcessor.__init__() 內部無條件建立 CSVLogger()/BehaviorSegmentLogger()/
    # ImprovedBehaviorTracker()，三者都沒有暴露建構參數可覆寫路徑，一律讀
    # config.py::LoggingConfig 的路徑。這裡把這三個路徑「暫時」重新導向到
    # 這支腳本自己的暫存目錄——這只是這次腳本執行期間、記憶體內的屬性覆寫，
    # 不會寫回 config.py 原始碼，腳本結束（程序退出）後就沒有任何殘留效果。
    # 若不攔截，會：
    #   1. 把假資料寫進正式的 cat_monitoring_log.csv / behavior_segments_log.csv
    #   2. 讀到你機器上真實累積的 tracker_state.json，讓「固定輸入」的起始
    #      狀態變得不固定（每次執行結果都不一樣）
    LoggingConfig.CSV_PATH = str(scratch_dir / "scratch_cat_monitoring_log.csv")
    LoggingConfig.SEGMENTS_CSV_PATH = str(scratch_dir / "scratch_behavior_segments_log.csv")
    LoggingConfig.TRACKER_STATE_PATH = str(scratch_dir / "scratch_tracker_state.json")

    if not Path(FIXED_VIDEO).exists():
        raise FileNotFoundError(f"固定測試影片不存在: {FIXED_VIDEO}")
    if not Path(ModelPaths.YOLO_MODEL).exists():
        raise FileNotFoundError(f"YOLO 模型不存在: {ModelPaths.YOLO_MODEL}")
    if not Path(ModelPaths.STGCN_MODEL).exists():
        raise FileNotFoundError(f"ST-GCN 模型不存在: {ModelPaths.STGCN_MODEL}")

    processor = FrameProcessor(
        yolo_model_path=ModelPaths.YOLO_MODEL,
        stgcn_model_path=ModelPaths.STGCN_MODEL,
        video_path=FIXED_VIDEO,
        nodered_url=None,  # 明確關閉 Node-RED 推送，腳本執行期間不會發出任何網路請求
        device="cuda",
        overlay=True,
    )
    try:
        records = capture_process_outputs(processor, FRAME_COUNT)
    finally:
        processor.cleanup()

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fixed_video": FIXED_VIDEO,
                "frame_count": len(records),
                "records": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"已寫入快照：{SNAPSHOT_PATH}（共 {len(records)} 幀）")


if __name__ == "__main__":
    main()
