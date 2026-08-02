"""Shared helpers for the FrameProcessor.process() characterization/regression
test (Michael Feathers 風格：先把「目前行為」凍結成快照，再拿快照當回歸基準）。

固定輸入：`FIXED_VIDEO` 的前 `FRAME_COUNT` 幀，逐幀依序餵進 process()。

刻意排除以下欄位，因為它們本質上不是「固定輸入 → 固定輸出」可重現的東西：

- 疊圖後的 frame 像素內容本身：cv2 繪製結果對 GPU/OpenCV 版本、字型渲染
  細節敏感，不是本測試要保護的邏輯核心（只保留 shape 供 sanity check）。
- BehaviorTracker 的累積時長類欄位（walk_time/lick_time 等）：這些是用
  `now - self.behavior_start_time`（真實 wall-clock 差值）算出來的，數值
  取決於這台機器這次執行實際跑多快，跟「輸入內容」無關，本質上就不可能
  重現。BehaviorTracker 自己的 regression test（見另一個模組）會改用
  外部注入固定 `now` 的方式測試累積邏輯，不受這個問題影響。
"""

from pathlib import Path

FIXED_VIDEO = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\walk\7月7日 (8).mp4"
FRAME_COUNT = 90  # 該影片總共 94 幀（30fps，約 3 秒）；90 幀涵蓋序列 buffer 填滿 + 多次推論
SNAPSHOT_PATH = (
    Path(__file__).parent / "snapshots" / "frame_processor_process_characterization.json"
)


def capture_process_outputs(processor, frame_count: int = FRAME_COUNT) -> list:
    """依序讀取固定數量的幀，逐幀呼叫 process()，收集可重現的數值輸出。"""
    records = []
    for _ in range(frame_count):
        ret, frame = processor.cap.read()
        if not ret:
            break
        out_frame, behavior_id, confidence, class_probs, is_still, activity_value = (
            processor.process(frame)
        )
        records.append(
            {
                "output_frame_shape": list(out_frame.shape),
                "behavior_id": int(behavior_id),
                "confidence": float(confidence),
                "class_probs": [float(p) for p in class_probs],
                "is_still": bool(is_still),
                "activity_value": float(activity_value),
            }
        )
    return records
