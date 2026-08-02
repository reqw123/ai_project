"""Shared helpers for the Golden Dataset regression test（覆蓋 walk/lick/
scratch/shake/stop 五種行為，保存 prediction / statistics / Node-RED payload /
CSV / behavior segments）。

跟 `_snapshot_utils.py`（單一 walk 影片、只驗證 process() 回傳值）不同，這裡
額外用假時鐘把 `time.time()`/`time.strftime()`/`datetime.now()` 全部控制住
（見 `GoldenClock`），讓原本因為「真實處理速度」而不可重現的 BehaviorTracker
累積統計、CSV 時間戳、Node-RED payload 內容，也都變成 100% 確定性可重現。

固定輸入：`GOLDEN_VIDEOS` 列出的 5 支影片（每種行為各一支，取自跟
`tools/eval_gcn_compare.py` 等既有工具共用的「主要測試」資料集），每支各處理
`frame_count` 幀。
"""

import time as _real_time_module
from datetime import datetime, timedelta
from pathlib import Path

_ORIGINAL_TIME_STRFTIME = _real_time_module.strftime  # 未被 monkeypatch 前的真正 time.strftime

_BASE = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試"

# behavior_name -> (影片路徑, 要處理的幀數)
GOLDEN_VIDEOS = {
    "walk": (_BASE + r"\walk\7月7日 (8).mp4", 90),
    "lick": (_BASE + r"\lick\7月3日(1).mp4", 90),
    "scratch": (_BASE + r"\scratch\7月3日 (2)(1).mp4", 90),
    "shake": (_BASE + r"\shake\7月7日 (5)(1).mp4", 90),
    "stop": (_BASE + r"\stop\7月7日 (8)(2).mp4", 50),  # 該影片僅有 50 幀
}

FRAME_DT = 1.0 / 30.0  # 固定模擬 30fps 的幀間隔，不受真實推論速度影響

SNAPSHOT_DIR = Path(__file__).parent / "snapshots" / "golden_dataset"


class GoldenClock:
    """統一控制 time.time() / time.strftime() / datetime.now()，讓整條
    pipeline（FrameProcessor → BehaviorTracker → CSVLogger）在測試期間
    完全不受真實系統時鐘影響。"""

    def __init__(self, start_datetime: datetime):
        self._dt = start_datetime

    def advance(self, seconds: float) -> None:
        self._dt += timedelta(seconds=seconds)

    def strftime(self, fmt: str, t=None) -> str:
        """相容 `time.strftime(format[, t])` 的完整簽名。

        CPython 的 `datetime.strftime()` 內部其實是委派給 `time.strftime()`
        實作（傳入該 datetime 實例自己的 `timetuple()`），所以這裡被
        monkeypatch 成全域 `time.strftime` 後，任何 `some_datetime.strftime(...)`
        呼叫都會間接經過這裡、且會帶著 `t` 參數進來——若忽略 `t`、只認第一個
        參數，會讓每個 datetime 實例都被錯誤地格式化成同一個「現在時間」。
        因此有帶 `t` 時原樣委派給真正的 `time.strftime`；沒有帶（例如
        `frame_processor.py` 直接呼叫 `time.strftime("%H:%M:%S")` 取得
        「現在」時）才使用假時鐘的當前時間。
        """
        if t is not None:
            return _ORIGINAL_TIME_STRFTIME(fmt, t)
        return _ORIGINAL_TIME_STRFTIME(fmt, self._dt.timetuple())

    def time(self) -> float:
        return self._dt.timestamp()

    def now(self, tz=None) -> datetime:
        return self._dt


def process_video_golden(processor, clock: GoldenClock, frame_count: int) -> list:
    """依序讀取固定數量的幀，每幀先推進假時鐘 FRAME_DT 秒再呼叫 process()。"""
    predictions = []
    for _ in range(frame_count):
        ret, frame = processor.cap.read()
        if not ret:
            break
        clock.advance(FRAME_DT)
        out_frame, behavior_id, confidence, class_probs, is_still, activity_value = (
            processor.process(frame)
        )
        predictions.append(
            {
                "behavior_id": int(behavior_id),
                "confidence": float(confidence),
                "class_probs": [float(p) for p in class_probs],
                "is_still": bool(is_still),
                "activity_value": float(activity_value),
            }
        )
    return predictions


def read_csv_rows(path: Path) -> list:
    import csv

    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))
