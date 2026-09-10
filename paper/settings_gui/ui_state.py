"""設定視窗的「UI 狀態」持久化（非 config 欄位、純介面便利記憶）。

跟 `runtime_settings.current.json` 的差別：那份管的是 config.py 欄位的執行期
覆寫，是「設定」；這裡存的是「上次選了哪支腳本、上次填的影片路徑」這類
純介面狀態，跟程式行為無關，跟 `settings_gui/tool_order.json`（下拉選單自訂
排序）同一個層級 —— 存在 `settings_gui/ui_state.json`。

讀寫都是 best-effort：檔案不存在 / 壞掉 / 寫不進去都安靜降級，不影響 GUI。
"""

import json
from pathlib import Path

_STATE_PATH = Path(__file__).resolve().parent / "ui_state.json"


def load() -> dict:
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def get(key: str, default=None):
    value = load().get(key, default)
    return value if isinstance(value, str) else default


def update(**kwargs) -> None:
    """把 kwargs 合併進 ui_state.json（None 值代表刪除該鍵）。寫入失敗安靜略過。"""
    data = load()
    for k, v in kwargs.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    try:
        _STATE_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        pass
