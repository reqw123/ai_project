"""IoT Sensor Hub 的全部可調參數，集中一處。

比照 ``plugins/lick_stage/config.py`` / ``analytics/config.py`` 的既有慣例：
獨立 class、每個值都可用環境變數覆寫、docstring 說明依據。**刻意不 import
主專案 ``config.py``**——連 broker host/port 也自己管，維持本套件「雙向零
依賴、可整包移除」的設計原則（見 ``iot/__init__.py``）。

環境變數統一用 ``CAT_MONITORING_IOT_*`` 前綴。未來若要接進 ``settings_window``
管理，沿用專案既有的「GUI 寫 env / JSON、config 讀 env」模式即可，本檔案不必改。
"""

from __future__ import annotations

import os
from pathlib import Path

# ── 精簡版 env helper（自帶一份，不 import 主專案 config.py / utils）──────────


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


_PACKAGE_DIR = Path(__file__).resolve().parent


class IotHubConfig:
    """MQTT broker、topic 對應、持久化路徑、告警門檻與各種節流參數。"""

    # ── MQTT broker（Node-RED 的 mosquitto）─────────────────────────────────
    # 預設指向這個環境裡跑 mosquitto / Node-RED 的那台（192.168.0.171）；
    # hub 跟 broker 同機時設 CAT_MONITORING_IOT_MQTT_HOST=127.0.0.1 即可。
    MQTT_HOST = _env_str("CAT_MONITORING_IOT_MQTT_HOST", "192.168.0.171")
    MQTT_PORT = _env_int("CAT_MONITORING_IOT_MQTT_PORT", 1883)
    MQTT_KEEPALIVE = _env_int("CAT_MONITORING_IOT_MQTT_KEEPALIVE", 60)
    # 只處理這幾種感測器（逗號分隔，例如 "env,bodytemp"；空＝TOPIC_MAP 全部）。設定視窗「IoT 子系統」分頁
    # 用它把各感測器分開啟動（每種一個行程，互不影響）；手動 `python -m iot` 不設＝跟原本一樣全部處理。
    KINDS = tuple(k.strip() for k in _env_str("CAT_MONITORING_IOT_KINDS", "").split(",") if k.strip())
    # 同一台 broker 上 client id 不能重複（重複會互踢）：只處理部分感測器時自動加上種類
    MQTT_CLIENT_ID = _env_str(
        "CAT_MONITORING_IOT_MQTT_CLIENT_ID",
        "cat-iot-hub" + ("-" + "-".join(KINDS) if KINDS else ""),
    )
    # 帳密：留空代表匿名連線（mosquitto 預設）。
    MQTT_USERNAME = _env_str("CAT_MONITORING_IOT_MQTT_USERNAME", "")
    MQTT_PASSWORD = _env_str("CAT_MONITORING_IOT_MQTT_PASSWORD", "")
    # 斷線後自動重連的退避秒數（min / max，paho 內建指數退避）。
    MQTT_RECONNECT_MIN_SEC = _env_int("CAT_MONITORING_IOT_MQTT_RECONNECT_MIN_SEC", 1)
    MQTT_RECONNECT_MAX_SEC = _env_int("CAT_MONITORING_IOT_MQTT_RECONNECT_MAX_SEC", 60)

    # ── Topic 約定 ─────────────────────────────────────────────────────────
    # 收：<PREFIX>/<kind>/<source_id>；kind ∈ {env, motion, weight}
    # 發：<PREFIX>/derived/<kind>、<PREFIX>/alert
    TOPIC_PREFIX = _env_str("CAT_MONITORING_IOT_TOPIC_PREFIX", "cat/iot")

    # kind -> parser 名稱（router 用）。改這裡就能加/換感測器類型。
    TOPIC_MAP = {
        "env": "environment",
        "motion": "motion",
        "weight": "weight",
        "bodytemp": "bodytemp",
    }

    # 訂閱清單：每個已知 kind 一條 ``<prefix>/<kind>/#``（用 ``#`` 讓 source_id
    # 可含斜線）。刻意不訂 ``<prefix>/#``——那會收到 hub 自己發出去的
    # ``derived/*`` 與 ``alert``，形成回授。
    @classmethod
    def subscribe_topics(cls) -> list[str]:
        return [f"{cls.TOPIC_PREFIX}/{kind}/#" for kind in cls.active_kinds()]

    @classmethod
    def active_kinds(cls) -> list[str]:
        """這個行程要處理的感測器種類（KINDS 裡不認得的種類忽略）。"""
        return [k for k in cls.TOPIC_MAP if not cls.KINDS or k in cls.KINDS]

    @classmethod
    def derived_topic(cls, kind: str) -> str:
        return f"{cls.TOPIC_PREFIX}/derived/{kind}"

    @classmethod
    def alert_topic(cls) -> str:
        return f"{cls.TOPIC_PREFIX}/alert"

    # ── 持久化（自己的 SQLite，跟 paper/baseline_data/ 完全分開）───────────
    DB_PATH = _env_str(
        "CAT_MONITORING_IOT_DB_PATH",
        str(_PACKAGE_DIR / "data" / "iot_hub.db"),
    )
    # 原始讀數保留幾天，舊的定時刪掉（0＝不清理）。外出包每秒一筆，24 小時連續跑一天約 17 萬筆、一個月約 400 MB。
    # 只清原始讀數（env／bodytemp／motion／weight）；進食事件、告警紀錄量小又有用，不清。
    # hub 啟動時清一次、之後每 DATA_PURGE_EVERY_HOURS 小時一次；每個 hub 行程只清自己處理的感測器（09-27）。
    DATA_RETENTION_DAYS = _env_float("CAT_MONITORING_IOT_DATA_RETENTION_DAYS", 30.0)
    DATA_PURGE_EVERY_HOURS = 6.0

    # ── 環境感測合理範圍（超出即視為感測器故障，丟棄該筆並記警告）────────
    ENV_TEMP_VALID_RANGE = (-20.0, 60.0)  # °C
    ENV_HUMIDITY_VALID_RANGE = (0.0, 100.0)  # %RH
    ENV_GAS_VALID_RANGE = (0.0, 10000.0)  # ppm（MQ 系列粗略上限）
    ENV_LUX_VALID_RANGE = (0.0, 200000.0)  # lux

    # EWMA 平滑係數（0<α≤1；越小越平滑）。只影響發回 MQTT 的 derived 值，
    # 原始值仍原封不動寫進 SQLite。
    ENV_EWMA_ALPHA = _env_float("CAT_MONITORING_IOT_ENV_EWMA_ALPHA", 0.3)

    # ── 環境告警門檻（舒適導向）──────────────────────────────────────────
    #
    # 健康成貓室內舒適範圍約 18–28°C / 40–70% RH（貓的熱中性區其實偏高、更
    # 耐熱不耐濕冷，但室內飼養仍以這個範圍當「該調空調了」的提醒）。
    # 特殊個體要自行收窄：幼貓 / 高齡 / 生病 / 過瘦 / 無毛品種對冷更敏感；
    # 短吻品種（波斯、異短）、肥胖貓對熱更敏感，尤其高濕時。
    # 想放寬回「只抓明顯異常」用環境變數覆寫即可（見下方變數名）。
    TEMP_MIN_C = _env_float("CAT_MONITORING_IOT_TEMP_MIN_C", 18.0)
    TEMP_MAX_C = _env_float("CAT_MONITORING_IOT_TEMP_MAX_C", 28.0)
    HUMIDITY_MIN_PCT = _env_float("CAT_MONITORING_IOT_HUMIDITY_MIN_PCT", 40.0)
    HUMIDITY_MAX_PCT = _env_float("CAT_MONITORING_IOT_HUMIDITY_MAX_PCT", 70.0)
    GAS_PPM_MAX = _env_float("CAT_MONITORING_IOT_GAS_PPM_MAX", 1000.0)

    # ── 體表溫度（MLX90614 紅外線非接觸測溫，來自 esp32_petbox 節點）─────────
    #
    # ⚠️ 這裡量的是「體表 / 毛髮表面」溫度，受量測距離、毛色、毛長、環境溫度
    # 影響很大，**不等於肛溫 / 核心體溫**。貓正常肛溫約 38.1–39.2°C，而 MLX90614
    # 對著貓身上讀到的體表值通常低 3–8°C 且變異大。因此下面的門檻只做「明顯
    # 異常、值得注意」的初篩，**不是發燒診斷**——真的要確認發燒/失溫還是要
    # 獸醫量肛溫。門檻預設放很寬，寧可漏報也不要一直誤報。
    BODYTEMP_SURFACE_VALID_RANGE = (0.0, 50.0)  # °C，超出視為感測器故障
    BODYTEMP_SURFACE_MIN_C = _env_float(
        "CAT_MONITORING_IOT_BODYTEMP_SURFACE_MIN_C", 20.0
    )
    BODYTEMP_SURFACE_MAX_C = _env_float(
        "CAT_MONITORING_IOT_BODYTEMP_SURFACE_MAX_C", 40.0
    )

    # ── 重量感測 / 進食偵測 ───────────────────────────────────────────────
    WEIGHT_VALID_RANGE = (-500.0, 20000.0)  # g（負值容許少量，代表去皮漂移）
    # 單筆變化量超過這個絕對值（g）才算一次「進食 / 加料 / 飲水」事件，
    # 用來濾掉貓走過造成的秤台輕微晃動。
    WEIGHT_EVENT_DELTA_G = _env_float("CAT_MONITORING_IOT_WEIGHT_EVENT_DELTA_G", 3.0)
    # 連續讀數穩定（變化量 < 此值）多久才更新「基準重量」（去皮），秒。
    WEIGHT_SETTLE_SEC = _env_float("CAT_MONITORING_IOT_WEIGHT_SETTLE_SEC", 10.0)
    # 食盆 scale_id（重量下降＝進食）。可用逗號分隔多個。
    FOOD_SCALE_IDS = tuple(
        s.strip()
        for s in _env_str("CAT_MONITORING_IOT_FOOD_SCALE_IDS", "food_bowl").split(",")
        if s.strip()
    )
    # 幾小時內沒偵測到食盆重量下降事件就告警（疑似不進食）。
    FEEDING_SILENCE_HOURS = _env_float("CAT_MONITORING_IOT_FEEDING_SILENCE_HOURS", 24.0)

    # ── 告警節流 ──────────────────────────────────────────────────────────
    # 同一個 alert key 兩次發送之間至少間隔幾秒（避免溫度在門檻附近抖動洗版）。
    ALERT_COOLDOWN_SEC = _env_float("CAT_MONITORING_IOT_ALERT_COOLDOWN_SEC", 900.0)

    # ── derived 發布節流 ──────────────────────────────────────────────────
    # 每個 (kind, source_id) 最多每幾秒發一次 derived（原始資料仍全部進 DB）。
    DERIVED_PUBLISH_MIN_INTERVAL_SEC = _env_float(
        "CAT_MONITORING_IOT_DERIVED_MIN_INTERVAL_SEC", 5.0
    )

    # ── 週期性檢查（時間型告警，例如「N 小時沒進食」）────────────────────
    PERIODIC_CHECK_INTERVAL_SEC = _env_float(
        "CAT_MONITORING_IOT_PERIODIC_CHECK_INTERVAL_SEC", 300.0
    )

    # ── 偵測條件：多久算異常（ingest/no_data_watchdog.py、sensors/health.py）──────────
    # 「已連上 broker」只代表 hub→broker 通，ESP32 連不上 broker 時 hub 只會安靜地等，所以要自己計時。
    # 連上 broker 後第一次檢查（5 秒：使用者要求盡快知道；移動偵測 heartbeat 30 秒、環境 10 秒一筆，
    # 開機時可能先警告、資料到了記一行「恢復」）。
    NO_DATA_FIRST_CHECK_SEC = _env_float("CAT_MONITORING_IOT_NO_DATA_FIRST_CHECK_SEC", 5.0)
    # 多久沒資料算異常：整台 ESP32 某種感測器都沒送、或 ESP32 在送但某欄位（例 DHT11 的 humidity_pct）沒值，
    # 兩者共用（09-27 合併原本的 NO_DATA_CHECK_INTERVAL_SEC、SENSOR_FIELD_MISSING_SEC）。
    # 韌體最慢的是移動偵測 heartbeat 30 秒，低於 30 秒會誤報。
    DATA_TIMEOUT_SEC = _env_float("CAT_MONITORING_IOT_DATA_TIMEOUT_SEC", 60.0)

    # ── 終端列印（除錯用；只影響 hub 日誌，跟 Discord 告警通知無關）──────────────────
    # 警報通知歸警報通知（Discord 照 ALERT_COOLDOWN_SEC 冷卻）。所有「異常持續中」的訊息——沒收到資料、
    # 欄位消失、卡在極端值、韌體送 nan、門檻告警超標——沒解除時每幾秒重印一次，0＝只印一次
    # （09-27 合併原本的 SENSOR_WARN_REPEAT_SEC、SENSOR_NAN_WARN_SEC、ALERT_LOG_REPEAT_SEC）。
    WARN_REPEAT_SEC = _env_float("CAT_MONITORING_IOT_WARN_REPEAT_SEC", 10.0)
    # 每個節點最新讀數每幾秒印一行（看數值有沒有進來、對不對）；0＝不印。
    READING_LOG_INTERVAL_SEC = _env_float("CAT_MONITORING_IOT_READING_LOG_INTERVAL_SEC", 0.0)

    # ── 感測器健康（sensors/health.py）：ESP32 在線、但某顆感測器沒接好 ──────────
    # 類比感測器連續這麼多筆都是量測範圍的極端值（腳位接地／懸空／滿格）→ 警告（esp32_room 10 秒一筆，6 筆≈1 分鐘）。
    SENSOR_RAIL_CONSECUTIVE = _env_int("CAT_MONITORING_IOT_SENSOR_RAIL_CONSECUTIVE", 6)
    # 各欄位的「極端值」(下限, 上限)，None＝不檢查那一端。對應 esp32_room（備用的 esp32_env 相同）的換算：ADC 0→0、4095→滿格。
    # 光照不檢查 0（全暗是正常的）；MQ-135 加熱中一定有電壓，讀到 0 幾乎就是沒接。
    SENSOR_RAILS = {"gas_ppm": (0.0, 1000.0), "lux": (None, 2000.0)}
    # 每個節點「應該要有」的欄位：hub 剛啟動、感測器一開始就沒接時也抓得到（不用先看過一次）。
    # key＝"<kind>/<source_id>"；沒列的節點就用「看過的欄位」判斷。
    SENSOR_EXPECTED_FIELDS = {
        "env/carrier": ("temp_c", "humidity_pct"),      # 外出包：MLX90614 環境溫度＋DHT11 濕度
        "env/living_room": ("gas_ppm", "lux"),          # esp32_room：MQ-135＋光敏電阻（改它的 SOURCE_ID 這裡也要改）
    }

    # ── log ───────────────────────────────────────────────────────────────
    LOG_LEVEL = _env_str("CAT_MONITORING_IOT_LOG_LEVEL", "INFO")
