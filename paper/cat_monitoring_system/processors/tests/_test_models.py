"""回歸測試（特徵快照、Golden Dataset）固定使用的測試基準模型。

測試模型跟 config.py 的正式預設模型（ModelPaths）分開：正式系統換模型不會讓回歸
測試壞掉，測試基準只在你自己決定要換時才換。

目前的測試基準模型（2026-09-30 從正式模型複製）：
- yolo_models/test_baseline.pt                  ← v11s_152.pt
- stgcn_models/test_baseline/test_baseline.pth  ← run_153_xy_conf_v_bone_att_on/153_best_model.pth
ST-GCN 的特徵模式／attention／關節數都從 checkpoint 自動判定，不看檔名或資料夾名。

換成更好的模型：
1. 用新模型覆蓋上面兩個檔案（檔名不變），順手更新這段說明的來源
2. 重跑 generate_frame_processor_snapshot.py（與 generate_golden_dataset_snapshot.py）
3. 快照會記下模型指紋（SHA-256）；只換模型沒重新產生快照時，測試會直接提示要重跑，
   不會變成一堆「第 N 幀數值改變」讓人以為程式壞了
"""

import hashlib
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
TEST_YOLO_MODEL = str(_PROJECT_ROOT / "yolo_models" / "test_baseline.pt")
TEST_STGCN_MODEL = str(_PROJECT_ROOT / "stgcn_models" / "test_baseline" / "test_baseline.pth")


def models_available() -> bool:
    return Path(TEST_YOLO_MODEL).exists() and Path(TEST_STGCN_MODEL).exists()


@lru_cache(maxsize=None)
def _sha256_prefix(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def model_fingerprint() -> dict:
    """寫進快照的模型指紋：檔案內容雜湊，換了模型檔就會不同。"""
    return {
        "yolo_sha256": _sha256_prefix(TEST_YOLO_MODEL),
        "stgcn_sha256": _sha256_prefix(TEST_STGCN_MODEL),
    }


def fingerprint_mismatch_message(snapshot: dict, generator_script: str):
    """快照的模型指紋跟目前測試模型不同時回傳說明文字；相同則回傳 None。"""
    recorded = snapshot.get("test_models")
    current = model_fingerprint()
    if recorded == current:
        return None
    return (
        "快照不是用目前的測試基準模型產生的"
        f"（快照記錄：{recorded or '沒有記錄模型指紋（舊版快照）'}；目前：{current}）。\n"
        "如果是你刻意換了測試模型，請重新產生快照：\n"
        f"    python processors/tests/{generator_script}\n"
        "確認輸出合理後，把新快照一併提交。"
    )
