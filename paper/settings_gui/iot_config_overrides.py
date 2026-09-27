"""IoT 感測器 hub（``iot/config.py``）的參數覆寫——設定視窗「📡 IoT 子系統」分頁「⚙ 參數設定」的邏輯層（不含 Tk）。

``iot/config.py`` 的每個可調值本來就能用 ``CAT_MONITORING_IOT_*`` 環境變數覆寫；這裡只是把使用者在設定視窗填的值
存成 ``iot_config_overrides.json``，由 ``iot_services.start()`` 在啟動 ``python -m iot`` 時放進子行程的環境變數。
**空白＝用 config.py 的預設值**（不寫進 JSON）。改完要重新啟動感測器服務才會生效（config 是啟動時讀一次）。

維持「主系統不 import iot 套件」的規則：預設值不是 import 來的，而是用 ast 讀 ``iot/config.py`` 原始碼裡
``_env_float("CAT_MONITORING_IOT_…", 預設值)`` 這種呼叫（改 config.py 的預設值，這裡自動跟著變）。
``FIELDS`` 是畫面上要列的欄位（中文名稱、單位、分組）；config.py 新增環境變數而這裡沒列時，單元測試會抓出來。
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_PAPER_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = _PAPER_DIR / "cat_monitoring_system" / "iot" / "config.py"
OVERRIDES_FILE = Path(__file__).resolve().parent / "iot_config_overrides.json"

_HELPERS = {"_env_str": "str", "_env_int": "int", "_env_float": "float", "_env_bool": "bool"}

# 不給使用者改的：KINDS 由各感測器卡片自己設（每種一個行程）；CLIENT_ID 預設會自動加上種類，覆寫成同一個值會互踢
EXCLUDED = frozenset({"CAT_MONITORING_IOT_KINDS", "CAT_MONITORING_IOT_MQTT_CLIENT_ID"})


@dataclass(frozen=True)
class Field:
    env: str
    label: str
    unit: str = ""
    group: str = ""
    hint: str = ""


# 09-27：告警通知（Discord）、感測器偵測條件、終端列印間隔（除錯）分成三組——警報通知歸警報通知，
# 終端訊息每一種都能自訂間隔
_G_ENV, _G_BODY, _G_WEIGHT, _G_ALERT, _G_SENSE, _G_LOG, _G_MQTT, _G_MISC = (
    "🌡 環境告警門檻", "🐾 體表溫度告警門檻", "⚖ 秤重／進食", "🔔 告警通知（Discord）與發布節流",
    "🩺 感測器偵測條件", "🖨 終端列印間隔（除錯）", "📶 MQTT broker", "🗂 其他")

FIELDS: tuple[Field, ...] = (
    Field("CAT_MONITORING_IOT_TEMP_MIN_C", "環境溫度下限", "°C", _G_ENV, "低於它 → 溫度過低告警"),
    Field("CAT_MONITORING_IOT_TEMP_MAX_C", "環境溫度上限", "°C", _G_ENV, "高於它 → 溫度過高告警"),
    Field("CAT_MONITORING_IOT_HUMIDITY_MIN_PCT", "濕度下限", "%RH", _G_ENV),
    Field("CAT_MONITORING_IOT_HUMIDITY_MAX_PCT", "濕度上限", "%RH", _G_ENV),
    Field("CAT_MONITORING_IOT_GAS_PPM_MAX", "空氣品質上限", "ppm", _G_ENV, "高於它 → critical 告警"),
    Field("CAT_MONITORING_IOT_ENV_EWMA_ALPHA", "EWMA 平滑係數", "", _G_ENV, "0<α≤1，越小越平滑（只影響 derived）"),
    Field("CAT_MONITORING_IOT_BODYTEMP_SURFACE_MIN_C", "體表溫度下限", "°C", _G_BODY, "IR 體表值，非肛溫"),
    Field("CAT_MONITORING_IOT_BODYTEMP_SURFACE_MAX_C", "體表溫度上限", "°C", _G_BODY, "IR 體表值，非肛溫"),
    Field("CAT_MONITORING_IOT_FEEDING_SILENCE_HOURS", "多久沒進食就告警", "小時", _G_WEIGHT),
    Field("CAT_MONITORING_IOT_WEIGHT_EVENT_DELTA_G", "進食事件最小變化量", "g", _G_WEIGHT, "濾掉秤台輕微晃動"),
    Field("CAT_MONITORING_IOT_WEIGHT_SETTLE_SEC", "穩定多久才更新基準重量", "秒", _G_WEIGHT),
    Field("CAT_MONITORING_IOT_FOOD_SCALE_IDS", "食盆 scale_id", "", _G_WEIGHT, "逗號分隔多個"),
    Field("CAT_MONITORING_IOT_ALERT_COOLDOWN_SEC", "同一告警冷卻時間", "秒", _G_ALERT,
          "Discord／資料庫：同一告警多久才再發一次（不影響終端列印）"),
    Field("CAT_MONITORING_IOT_DERIVED_MIN_INTERVAL_SEC", "derived 最短發布間隔", "秒", _G_ALERT),
    Field("CAT_MONITORING_IOT_PERIODIC_CHECK_INTERVAL_SEC", "週期檢查間隔", "秒", _G_ALERT, "「多久沒進食」多久檢查一次"),
    Field("CAT_MONITORING_IOT_NO_DATA_FIRST_CHECK_SEC", "連上 broker 後首次檢查", "秒", _G_SENSE,
          "連上後多久先檢查一次 ESP32 有沒有送資料"),
    Field("CAT_MONITORING_IOT_DATA_TIMEOUT_SEC", "多久沒資料算異常", "秒", _G_SENSE,
          "整台 ESP32 沒送、或某欄位（如 DHT11 濕度）沒值；低於 30 移動偵測會誤報"),
    Field("CAT_MONITORING_IOT_SENSOR_RAIL_CONSECUTIVE", "極端值連續幾筆算異常", "筆", _G_SENSE,
          "MQ-135 讀到 0、光敏電阻滿格＝疑似沒接"),
    Field("CAT_MONITORING_IOT_WARN_REPEAT_SEC", "異常持續中重印間隔", "秒", _G_LOG,
          "沒收到資料、欄位消失、極端值、nan、門檻告警沒解除時每隔幾秒再印；0＝只印一次"),
    Field("CAT_MONITORING_IOT_READING_LOG_INTERVAL_SEC", "讀數列印間隔", "秒", _G_LOG,
          "每個節點最新讀數每隔幾秒印一行；0＝不印"),
    Field("CAT_MONITORING_IOT_MQTT_HOST", "broker 位址", "", _G_MQTT),
    Field("CAT_MONITORING_IOT_MQTT_PORT", "broker port", "", _G_MQTT),
    Field("CAT_MONITORING_IOT_MQTT_KEEPALIVE", "keepalive", "秒", _G_MQTT),
    Field("CAT_MONITORING_IOT_MQTT_USERNAME", "帳號", "", _G_MQTT, "空白＝匿名"),
    Field("CAT_MONITORING_IOT_MQTT_PASSWORD", "密碼", "", _G_MQTT),
    Field("CAT_MONITORING_IOT_MQTT_RECONNECT_MIN_SEC", "重連退避下限", "秒", _G_MQTT),
    Field("CAT_MONITORING_IOT_MQTT_RECONNECT_MAX_SEC", "重連退避上限", "秒", _G_MQTT),
    Field("CAT_MONITORING_IOT_TOPIC_PREFIX", "topic 前綴", "", _G_MISC, "改了 ESP32 韌體也要跟著改"),
    Field("CAT_MONITORING_IOT_DB_PATH", "SQLite 路徑", "", _G_MISC),
    Field("CAT_MONITORING_IOT_DATA_RETENTION_DAYS", "原始讀數保留天數", "天", _G_MISC,
          "舊的定時刪掉（進食事件、告警不刪）；0＝不清理"),
    Field("CAT_MONITORING_IOT_LOG_LEVEL", "log 等級", "", _G_MISC, "DEBUG／INFO／WARNING"),
)
_BY_ENV = {f.env: f for f in FIELDS}

# (下限, 上限)：兩個都有值（覆寫或預設）時，下限必須小於上限
RANGE_PAIRS = (
    ("CAT_MONITORING_IOT_TEMP_MIN_C", "CAT_MONITORING_IOT_TEMP_MAX_C"),
    ("CAT_MONITORING_IOT_HUMIDITY_MIN_PCT", "CAT_MONITORING_IOT_HUMIDITY_MAX_PCT"),
    ("CAT_MONITORING_IOT_BODYTEMP_SURFACE_MIN_C", "CAT_MONITORING_IOT_BODYTEMP_SURFACE_MAX_C"),
    ("CAT_MONITORING_IOT_MQTT_RECONNECT_MIN_SEC", "CAT_MONITORING_IOT_MQTT_RECONNECT_MAX_SEC"),
)


# ── 預設值：讀 iot/config.py 原始碼 ─────────────────────────────────────────


@dataclass(frozen=True)
class Default:
    type: str                 # "str" | "int" | "float" | "bool"
    value: object = None      # 預設值；不是常數（例如用路徑組出來的）就是 None
    text: str = ""            # 顯示用


def read_defaults(path: Path = CONFIG_PATH) -> dict[str, Default]:
    """{環境變數: Default}。讀不到檔案／語法錯誤回空 dict（畫面照樣能開，只是看不到預設值）。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return {}
    out: dict[str, Default] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _HELPERS):
            continue
        if len(node.args) < 2 or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
            continue
        typ = _HELPERS[node.func.id]
        env = node.args[0].value
        try:
            value = ast.literal_eval(node.args[1])
            text = _fmt(value)
        except ValueError:
            value = _COMPUTED_DEFAULTS.get(env, lambda _p: None)(path)
            text = _fmt(value) if value is not None else "（依程式計算）"
        out[env] = Default(typ, value, text)
    return out


# config.py 裡預設值是運算式（ast 讀不到）的欄位：照 config.py 的算法在這裡算出來，畫面才看得到實際值
_COMPUTED_DEFAULTS = {
    # config.py：str(_PACKAGE_DIR / "data" / "iot_hub.db")，_PACKAGE_DIR＝config.py 所在的 iot/
    "CAT_MONITORING_IOT_DB_PATH": lambda config_path: str(Path(config_path).resolve().parent / "data" / "iot_hub.db"),
}


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return f"{value:.1f}"
    return str(value)


# ── 驗證 ──────────────────────────────────────────────────────────────────


def normalize(env: str, text: str, typ: str) -> tuple[bool, str]:
    """把使用者填的字串整理成要放進環境變數的值；回傳 (ok, 值或錯誤訊息)。空字串＝不覆寫，回 (True, "")。"""
    text = (text or "").strip()
    if not text:
        return True, ""
    label = _BY_ENV[env].label if env in _BY_ENV else env
    if typ == "int":
        try:
            return True, str(int(text))
        except ValueError:
            return False, f"「{label}」要填整數：{text}"
    if typ == "float":
        try:
            v = float(text)
        except ValueError:
            return False, f"「{label}」要填數字：{text}"
        if v != v or v in (float("inf"), float("-inf")):
            return False, f"「{label}」不是有效數字：{text}"
        return True, text
    if typ == "bool":
        low = text.lower()
        if low in {"1", "true", "yes", "y", "on"}:
            return True, "true"
        if low in {"0", "false", "no", "n", "off"}:
            return True, "false"
        return False, f"「{label}」要填 true／false：{text}"
    return True, text


def validate(values: dict[str, str], defaults: Optional[dict[str, Default]] = None) -> tuple[dict[str, str], list[str]]:
    """整批驗證畫面上的值（{環境變數: 字串}）。回傳 (要存的覆寫, 錯誤訊息清單)；有錯誤時不該存。"""
    defaults = read_defaults() if defaults is None else defaults
    clean: dict[str, str] = {}
    errors: list[str] = []
    for env, text in values.items():
        typ = defaults.get(env, Default("str")).type
        ok, v = normalize(env, text, typ)
        if not ok:
            errors.append(v)
        elif v:
            clean[env] = v
    for lo, hi in RANGE_PAIRS:
        a, b = _effective_num(lo, clean, defaults), _effective_num(hi, clean, defaults)
        if a is not None and b is not None and a >= b:
            errors.append(f"「{_BY_ENV[lo].label}」（{_fmt(a)}）要小於「{_BY_ENV[hi].label}」（{_fmt(b)}）")
    alpha = _effective_num("CAT_MONITORING_IOT_ENV_EWMA_ALPHA", clean, defaults)
    if alpha is not None and not 0 < alpha <= 1:
        errors.append(f"「EWMA 平滑係數」要在 0（不含）到 1 之間：{_fmt(alpha)}")
    for env in ("CAT_MONITORING_IOT_MQTT_PORT",):
        port = _effective_num(env, clean, defaults)
        if port is not None and not 1 <= port <= 65535:
            errors.append(f"「broker port」要在 1–65535：{_fmt(port)}")
    return clean, errors


def _effective_num(env, clean, defaults) -> Optional[float]:
    raw = clean.get(env)
    if raw is None:
        d = defaults.get(env)
        raw = d.value if d is not None else None
    try:
        return float(raw) if raw is not None and raw != "" else None
    except (TypeError, ValueError):
        return None


# ── 存取 ──────────────────────────────────────────────────────────────────


# 09-27 合併掉的舊變數 → 新變數。覆寫檔裡還有舊名稱時 load() 自動換成新的（同一個新變數有好幾個舊值時取最小，
# 也就是印得最勤的那個），存檔後舊名稱就消失。
LEGACY = {
    "CAT_MONITORING_IOT_NO_DATA_CHECK_INTERVAL_SEC": "CAT_MONITORING_IOT_WARN_REPEAT_SEC",
    "CAT_MONITORING_IOT_SENSOR_WARN_REPEAT_SEC": "CAT_MONITORING_IOT_WARN_REPEAT_SEC",
    "CAT_MONITORING_IOT_SENSOR_NAN_WARN_SEC": "CAT_MONITORING_IOT_WARN_REPEAT_SEC",
    "CAT_MONITORING_IOT_ALERT_LOG_REPEAT_SEC": "CAT_MONITORING_IOT_WARN_REPEAT_SEC",
    "CAT_MONITORING_IOT_SENSOR_FIELD_MISSING_SEC": "CAT_MONITORING_IOT_DATA_TIMEOUT_SEC",
}


def _migrate(data: dict[str, str]) -> dict[str, str]:
    out = {k: v for k, v in data.items() if k not in LEGACY}
    olds: dict[str, list[str]] = {}
    for old, new in LEGACY.items():
        if old in data:
            olds.setdefault(new, []).append(data[old])
    for new, values in olds.items():
        if new in out:
            continue   # 新名稱已經有值：以新的為準
        try:
            out[new] = min(values, key=float)
        except ValueError:
            out[new] = values[0]
    return out


def load() -> dict[str, str]:
    """已存的覆寫（只留 CAT_MONITORING_IOT_* 的非空值；KINDS／CLIENT_ID 就算被手動寫進去也忽略；舊變數名稱自動換新）。"""
    try:
        data = json.loads(OVERRIDES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return _migrate({str(k): str(v) for k, v in data.items() if str(k).startswith("CAT_MONITORING_IOT_")
                     and k not in EXCLUDED and v is not None and str(v).strip()})


def save(overrides: dict[str, str]) -> None:
    OVERRIDES_FILE.write_text(json.dumps(dict(sorted(overrides.items())), ensure_ascii=False, indent=2),
                              encoding="utf-8")


def env_for_start() -> dict[str, str]:
    """啟動 python -m iot 時要放進子行程的環境變數。"""
    return load()


def effective(env: str, default: str = "") -> str:
    """畫面／broker 檢查用：覆寫值 > 目前行程的環境變數 > config.py 預設值 > default。"""
    v = load().get(env) or os.getenv(env)
    if v:
        return v
    d = read_defaults().get(env)
    return d.text if d is not None and d.value is not None else default
