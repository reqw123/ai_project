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

    CONF_THRESHOLD = 0.5  # nose / ear / chest / hip / mid-back
    LIMB_CONF_THRESHOLD = 0.10  # knees / paws / tail points

    # ── Zone ids — must match the 7-zone body diagram (1=Head .. 7=Tail),
    # plus 8=TORSO_UNSPECIFIED (M5：ventral_sign 證據不足時的誠實回退，見下方)。
    ZONE_NO_TARGET = 0
    ZONE_HEAD = 1
    ZONE_NECK_CHEST = 2
    ZONE_SIDE_BACK = 3
    ZONE_ABDOMEN = 4
    ZONE_FORELIMB = 5
    ZONE_HINDLIMB = 6
    ZONE_TAIL = 7
    # M5：鼻子確實命中軀幹橢圓，但 build_zone_targets() 沒有任何信心足夠的膝蓋
    # 關鍵點可用來判斷 ventral_sign（哪一側是腹側）——這種情況下 classify_zone()
    # 不再沿用 ventral_sign 的任意預設值（+1.0）硬猜 ABDOMEN/SIDE_BACK，改回傳
    # 這個獨立的「軀幹，但無法分辨腹/背」標籤，如實反映幾何證據不足，而不是
    # 用一個看起來自信、實際上是猜的結果污染統計。
    ZONE_TORSO_UNSPECIFIED = 8
    # M5：候選評分最高分與次高分差距小於 AMBIGUITY_MARGIN，無法可靠分辨是哪個
    # 相鄰區域時的回傳值——跟 lick_stage/contact_regions.py find_nearest_zone()
    # 的 ZONE_AMBIGUOUS 同一個概念，這裡補齊讓兩個姊妹外掛的候選評分機制一致。
    ZONE_AMBIGUOUS = 9

    ZONE_NAMES = {
        ZONE_NO_TARGET: "NO_TARGET",
        ZONE_HEAD: "HEAD",
        ZONE_NECK_CHEST: "NECK_CHEST",
        ZONE_SIDE_BACK: "SIDE_BACK",
        ZONE_ABDOMEN: "ABDOMEN",
        ZONE_FORELIMB: "FORELIMB",
        ZONE_HINDLIMB: "HINDLIMB",
        ZONE_TAIL: "TAIL",
        ZONE_TORSO_UNSPECIFIED: "TORSO_UNSPECIFIED",
        ZONE_AMBIGUOUS: "AMBIGUOUS",
    }

    # ── 候選評分與 AMBIGUOUS（M5，取代 classify_zone() 舊版優先序判定）──────
    # 鼻子接觸範圍若同時跟兩個相鄰區域都有明顯重疊（例如剛好在軀幹與前肢
    # 交界處），舊版直接照固定優先序（四肢腳掌圓 > 四肢長條 > 尾巴 > 軀幹）
    # 選第一個命中的，武斷且沒有反映真正的不確定性。現在改成每個候選算
    # 正規化分數（1 - 距離/該候選區域特徵尺度），最高分跟次高分差距小於此值
    # 時回傳 ZONE_AMBIGUOUS，而不是硬選一個。跟 lick_stage 的 AMBIGUITY_MARGIN
    # 用同一個值，粗略預設值，之後可依實測重新校準。
    AMBIGUITY_MARGIN = 0.15

    # ── Geometry ratios, all relative to body_len = |Hip - Chest| ──────────
    HEAD_RADIUS_RATIO = 0.30
    NECK_RADIUS_RATIO = 0.22
    TORSO_HALF_LEN_RATIO = 0.55  # torso ellipse long-axis half length
    TORSO_HALF_WIDTH_RATIO = 0.30  # torso ellipse short-axis half length
    LIMB_STRIP_HW_RATIO = 0.06  # forelimb/hindlimb strip half width
    LIMB_PAW_RADIUS_RATIO = 0.05
    TAIL_STRIP_HW_RATIO = 0.045  # tail strip half width (single shared region)

    # ── Hybrid body scale (M5, replaces the old absolute BODY_LEN_MIN/MAX_PX
    # clamp) ── Fixed 300-650px clamp implicitly assumed roughly constant
    # camera distance; a curled-up grooming cat compresses the straight-line
    # chest-hip distance regardless of true scale, so clamping to a fixed
    # floor systematically distorted geometry sizing. See
    # _compute_body_scale() in regions.py: chest-midback-hip path length
    # (resists curl-compression) -> straight chest-hip distance -> bbox
    # diagonal * ratio below. Independent copy of the same idea implemented
    # in plugins/lick_stage/config.py — kept separate on purpose (this
    # module must stay zero-dependency on the sibling plugin).
    SCALE_DEGENERATE_LEN_PX = 20.0  # below this, treat chest-hip/path length as detection noise, not a curled pose
    BBOX_TO_BODY_LEN_RATIO = 0.65  # bbox_fallback: body_len ≈ bbox diagonal * this ratio; coarse default, recalibrate if verify tooling shows frequent bbox_fallback

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
