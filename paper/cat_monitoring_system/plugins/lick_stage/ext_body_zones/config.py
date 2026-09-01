"""All tunable parameters for the extended 7-zone body classifier.

Independent of plugins/lick_stage/config.py — this module must be removable
without touching the existing lick_stage plugin. The one exception is the
Node-RED HTTP endpoint host/port (see NODERED_URL below), which is shared
with the main project's config.NodeRedConfig for the same reason
plugins/lick_stage/config.py now shares it: a hardcoded 127.0.0.1:1880
here would silently stop matching reality if NodeRedConfig.HOST/PORT is
ever changed (e.g. Node-RED moved to another host) — see
docs/資料層架構現況與統一管理評估.md 第十一節 (Traditional Chinese).
"""

import os as _os

from config import NodeRedConfig as _NodeRedConfig

# 2026-08-11：_env_str/_env_float 原本在這裡各自重複實作一份，改成共用
# utils/env_parsing.py，見該檔案開頭說明；同套件內的純函式 helper，不影響
# 本檔案開頭講的「independent of plugins/lick_stage/config.py」這件事
# （那句話講的是不依賴同一個 plugin 的另一份設定，不是禁止共用套件內的
# 通用工具函式）。
from utils.env_parsing import env_float as _env_float, env_str as _env_str

_MODULE_DIR = _os.path.dirname(_os.path.abspath(__file__))


class ExtZoneConfig:
    """ExtBodyZonePlugin（7 區身體分區偵測）的關鍵點索引與所有可調參數。"""

    # ── Keypoint indices (17-pt YOLO-Pose layout, see utils/constants.py) ──
    KP_NOSE = 0
    KP_LEFT_EAR = 1
    KP_RIGHT_EAR = 2
    KP_CHEST = 3
    KP_MID_BACK = 4
    KP_HIP = 5
    KP_FL_KNEE = 6
    KP_FL_PAW = 7
    KP_FR_KNEE = 8
    KP_FR_PAW = 9
    KP_HL_KNEE = 10
    KP_HL_PAW = 11
    KP_HR_KNEE = 12
    KP_HR_PAW = 13
    KP_TAIL_ROOT = 14
    KP_TAIL_MID = 15
    KP_TAIL_TIP = 16

    KPT_CONF_THRESHOLD = 0.5  # nose / ear / chest / hip / mid-back
    LIMB_KPT_CONF_THRESHOLD = 0.5  # knees / paws / tail points

    # ── Zone ids — must match the 7-zone body diagram (1=Head .. 7=Tail) ──
    ZONE_NO_TARGET = 0
    ZONE_HEAD = 1
    ZONE_NECK_CHEST = 2
    ZONE_SIDE_BACK = 3
    ZONE_ABDOMEN = 4
    ZONE_FORELIMB = 5
    ZONE_HINDLIMB = 6
    ZONE_TAIL = 7

    ZONE_NAMES = {
        ZONE_NO_TARGET: "NO_TARGET",
        ZONE_HEAD: "HEAD",
        ZONE_NECK_CHEST: "NECK_CHEST",
        ZONE_SIDE_BACK: "SIDE_BACK",
        ZONE_ABDOMEN: "ABDOMEN",
        ZONE_FORELIMB: "FORELIMB",
        ZONE_HINDLIMB: "HINDLIMB",
        ZONE_TAIL: "TAIL",
    }

    # ── Geometry ratios, all relative to body_len = |Hip - Chest| ──────────
    HEAD_RADIUS_RATIO = 0.30
    NECK_RADIUS_RATIO = 0.22
    TORSO_HALF_LEN_RATIO = 0.55  # torso ellipse long-axis half length
    TORSO_HALF_WIDTH_RATIO = 0.30  # torso ellipse short-axis half length
    LIMB_STRIP_HW_RATIO = 0.06  # forelimb/hindlimb strip half width
    LIMB_PAW_RADIUS_RATIO = 0.05
    TAIL_STRIP_HW_RATIO = 0.045  # tail strip half width (single shared region)

    # Body length clamp (guards against exploding geometry on extreme poses)
    BODY_LEN_MIN_PX = 300.0
    BODY_LEN_MAX_PX = 650.0

    # ── Output (file / MQTT only — never fed back to the main program) ────
    OUTPUT_ENABLED = True
    OUTPUT_CSV_PATH = _os.path.join(_MODULE_DIR, "results.csv")
    LOG_INTERVAL_SEC = 2.0  # minimum seconds between persisted snapshot rows

    MQTT_ENABLED = False  # off by default; paho-mqtt is optional
    MQTT_HOST = "127.0.0.1"
    MQTT_PORT = 1883
    MQTT_TOPIC = "cat/ext_body_zone"

    # ── Node-RED HTTP output (raw geometry, for client-side visualization
    # only — Node-RED does the drawing; this module never renders anything) ──
    NODERED_ENABLED = True
    # Host/port default to the main project's NodeRedConfig (see module
    # docstring); CAT_MONITORING_EXT_ZONE_NODERED_URL overrides the whole
    # URL independently of NodeRedConfig.HOST/PORT if ever needed.
    NODERED_URL = _env_str(
        "CAT_MONITORING_EXT_ZONE_NODERED_URL",
        f"http://{_NodeRedConfig.HOST}:{_NodeRedConfig.PORT}/ext_zone_result",
    )
    NODERED_TIMEOUT = _env_float("CAT_MONITORING_EXT_ZONE_NODERED_TIMEOUT", 0.3)
    GEO_PUBLISH_INTERVAL_SEC = 0.3  # throttle: raw pixel coords, not every frame
