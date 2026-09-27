"""飼主語音紀錄服務的全部可調參數，集中一處。

比照 ``iot/config.py`` 的慣例：獨立 class、每個值都可用環境變數覆寫。**刻意不 import**
主專案 ``config.py``，也不 import ``iot.config``（感測器 hub 的設定）——兩個子系統
共用同一台 broker 只是部署上的巧合，設定各管各的，任一邊改動不會牽動另一邊。

環境變數統一用 ``CAT_MONITORING_VOICE_*`` 前綴。
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = (os.getenv(name) or "").strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


_PACKAGE_DIR = Path(__file__).resolve().parent


class VoiceReportConfig:
    """MQTT、topic、資料庫位置、分類規則的參數。"""

    # ── MQTT broker（跟 Node-RED 同一台 mosquitto）───────────────────────────
    MQTT_HOST = _env_str("CAT_MONITORING_VOICE_MQTT_HOST", "192.168.0.171")
    MQTT_PORT = _env_int("CAT_MONITORING_VOICE_MQTT_PORT", 1883)
    MQTT_KEEPALIVE = _env_int("CAT_MONITORING_VOICE_MQTT_KEEPALIVE", 60)
    MQTT_CLIENT_ID = _env_str("CAT_MONITORING_VOICE_MQTT_CLIENT_ID", "cat-voice-reports")
    MQTT_USERNAME = _env_str("CAT_MONITORING_VOICE_MQTT_USERNAME", "")
    MQTT_PASSWORD = _env_str("CAT_MONITORING_VOICE_MQTT_PASSWORD", "")

    # ── Topic 約定（見 README「MQTT 介面」）──────────────────────────────────
    #   收：<prefix>/utterance        一句話 {"id","text","ts"?,"deviceId"?,"source"?}
    #       <prefix>/reports/delete   刪除一筆（儀表板按鈕）{"report_id"}
    #       <prefix>/reports/get      要求重送最近紀錄（任何內容）
    #   發：<prefix>/result           每句話的分類結果（跟 utterance 的 id 對應）
    #       <prefix>/reports/recent   最近紀錄＋今日統計（retained，儀表板一開就有）
    #       <prefix>/status           服務上線／離線（retained；離線用 MQTT 遺囑）
    TOPIC_PREFIX = _env_str("CAT_MONITORING_VOICE_TOPIC_PREFIX", "cat/voice")

    @classmethod
    def topic(cls, name: str) -> str:
        return f"{cls.TOPIC_PREFIX.strip('/')}/{name}"

    # ── 持久化（自己的 SQLite，跟 iot_hub.db、baseline_data 完全分開）──────────
    DB_PATH = _env_str("CAT_MONITORING_VOICE_DB_PATH", str(_PACKAGE_DIR / "data" / "owner_reports.db"))

    # ── 分類規則 ──────────────────────────────────────────────────────────
    # 句子要提到這些字之一（或說「記錄」）才會被當成貓咪的紀錄；貓咪有名字的話加進來（逗號分隔）
    CAT_ALIASES = tuple(
        s.strip() for s in _env_str("CAT_MONITORING_VOICE_CAT_ALIASES", "貓,猫,咪").split(",") if s.strip()
    )
    # 沒被分類成紀錄的句子也存下來（text＋原因），論文 E1 可以回頭標註「漏掉的回報」算召回率
    LOG_UNMATCHED = _env_bool("CAT_MONITORING_VOICE_LOG_UNMATCHED", True)

    # ── 儀表板 ───────────────────────────────────────────────────────────
    RECENT_LIMIT = _env_int("CAT_MONITORING_VOICE_RECENT_LIMIT", 30)
    # 同一個 utterance id 重送（MQTT 重傳、Node-RED 重試）時回同一個結果、不重複寫入；記住最近幾個 id
    DEDUP_CACHE_SIZE = _env_int("CAT_MONITORING_VOICE_DEDUP_CACHE_SIZE", 256)

    LOG_LEVEL = _env_str("CAT_MONITORING_VOICE_LOG_LEVEL", "INFO")
