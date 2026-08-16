"""
共用的環境變數解析 helper。

**背景（2026-08-11 code review）**：`analytics/config.py`、
`plugins/lick_stage/config.py`、`plugins/lick_stage/ext_body_zones/config.py`
三個檔案原本各自獨立實作了幾乎一模一樣的 `_env_str`/`_env_int`/`_env_float`
（頂層的 `paper/config.py` 也有自己一份，功能更多，包含 `_env_bool` 等這裡
沒有的變體，維持獨立不動）。同一段解析邏輯散落在三個檔案裡，之後要修
（例如目前所有版本都靜默吞掉解析失敗的值，不記任何警告）得手動改三份，
容易改了一份漏掉其他份。這裡把三個檔案共同的部分抽出來，讓它們改成匯入
這裡，而不是複製貼上。

`paper/config.py` 刻意不跟這裡共用——它是這幾個子模組共同依賴的頂層設定
檔，方向上不應該反過來依賴 `cat_monitoring_system` 套件內部的模組（避免
循環依賴／不自然的依賴方向）。
"""

import os


def env_str(name: str, default: str) -> str:
    """讀取字串型環境變數；未設定或空字串時回傳 default。"""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def env_int(name: str, default: int) -> int:
    """讀取整數型環境變數；未設定或轉型失敗時回傳 default。"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    """讀取浮點數型環境變數；未設定或轉型失敗時回傳 default。"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
