"""M4：共用 PoseFrame——`lick_stage`／`ext_body_zones` 兩個外掛共用的「這一幀的
姿態資料」封裝，純資料結構，不含任何平滑演算法（平滑邏輯在 `pose_filter.py`）。

**現況（2026-09-14）**：這裡只有資料結構＋`pose_filter.py` 的濾波器本體，
**尚未接進 `frame_processor.py`／`analyzer.py`／`ext_body_zones`**——是刻意
分兩階段交付的第一階段（模組本身先完整、有測試、可獨立審查），第二階段
（真的把兩個外掛的平滑邏輯換成呼叫這裡）是後續、需要另外確認才動的工作，
因為那一步要改到 `frame_processor.py`（全系統共用的核心檔案，不是像
`plugins/lick_stage/` 這樣的隔離沙盒）。`event_aggregator.py`／`analyzer.py`
裡 `pose_quality=None  # M4 由共用 PoseFrame 提供` 這個既有註解, 就是等這個
模組真正接上之後才會補上的那個值。

為什麼两個外掛值得共用一份：目前 `lick_stage/analyzer.py` 自己有一個很陽春
的關鍵點 EMA（`self._ema_kpts`，`config.py` 的 `EMA_ALPHA`），沒有信心值
感知、沒有缺點 hold decay、沒有骨長品質；`ext_body_zones` 完全獨立、沒有
共用這份邏輯的機制。兩個外掛各自對同一批原始關鍵點做（可能不一致的）平滑，
不只是重複工作，過濾出來的姿態品質也可能因此不同步。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PoseFrame:
    """一幀的姿態資料，raw + smoothed + conf + quality + timestamp。

    `raw_kpts`/`raw_conf` 是 YOLO-Pose 的原始輸出（未經任何濾波，`update()`
    的輸入原樣保留一份，方便需要「絕對真實值」的呼叫端繞過平滑）。
    `smoothed_kpts` 是 `PoseFilter.update()` 濾波後的結果——`alpha>=1.0` 時
    (完全不平滑) 這個欄位會跟 `raw_kpts`逐元素相等（同一份資料的拷貝，
    不是同一個物件參照，避免呼叫端不小心互相污染）。
    `quality` 是骨長一致性品質分數（見 `pose_filter.PoseFilter` 的
    `bone_pairs`/`_compute_quality()`），範圍 `[0.0, 1.0]`（1.0＝目前幀骨長
    跟歷史平均完全一致），`None` 代表尚未啟用骨長品質追蹤，或歷史資料還
    不足以計算（例如第一幀）——**呼叫端應該把 `None` 當「暫時不可得」處理，
    不要當成 0.0**（`event_aggregator.py` 的 `_ActiveBout.add()` 已經是
    `if pose_quality is not None:` 這種寫法，這裡的 `None` 語意與其一致）。

    整幀缺失（`update(None, None, ...)`，例如 NO_CAT/NOT_LICK）時，所有
    姿態欄位都是 `None`——呼叫端可以用 `raw_kpts is None` 判斷這幀完全沒有
    姿態可用，不用另外查 `quality`/`smoothed_kpts` 是否為 `None`（保證
    一致：要嘛全部有值，要嘛全部是 `None`）。
    """

    raw_kpts: Optional[np.ndarray]
    raw_conf: Optional[np.ndarray]
    smoothed_kpts: Optional[np.ndarray]
    quality: Optional[float]
    frame_idx: int = 0
    source_timestamp: Optional[float] = None

    @property
    def has_pose(self) -> bool:
        """這一幀是否真的有姿態資料可用（等同 `raw_kpts is not None`，
        提供具名的判斷式方便呼叫端閱讀）。"""
        return self.raw_kpts is not None
