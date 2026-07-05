"""All tunable parameters for the extended 7-zone body classifier.

Independent of plugins/lick_stage/config.py — this module must be removable
without touching the existing lick_stage plugin.
"""
import os as _os

_MODULE_DIR = _os.path.dirname(_os.path.abspath(__file__))


class ExtZoneConfig:
    # ── Keypoint indices (17-pt YOLO-Pose layout, see utils/constants.py) ──
    KP_NOSE      = 0
    KP_LEFT_EAR  = 1
    KP_RIGHT_EAR = 2
    KP_CHEST     = 3
    KP_MID_BACK  = 4
    KP_HIP       = 5
    KP_FL_KNEE   = 6
    KP_FL_PAW    = 7
    KP_FR_KNEE   = 8
    KP_FR_PAW    = 9
    KP_HL_KNEE   = 10
    KP_HL_PAW    = 11
    KP_HR_KNEE   = 12
    KP_HR_PAW    = 13
    KP_TAIL_ROOT = 14
    KP_TAIL_MID  = 15
    KP_TAIL_TIP  = 16

    CONF_THRESHOLD       = 0.5    # nose / ear / chest / hip / mid-back
    LIMB_CONF_THRESHOLD  = 0.10   # knees / paws / tail points

    # ── Zone ids — must match the 7-zone body diagram (1=Head .. 7=Tail) ──
    ZONE_NO_TARGET  = 0
    ZONE_HEAD       = 1
    ZONE_NECK_CHEST = 2
    ZONE_SIDE_BACK  = 3
    ZONE_ABDOMEN    = 4
    ZONE_FORELIMB   = 5
    ZONE_HINDLIMB   = 6
    ZONE_TAIL       = 7

    ZONE_NAMES = {
        ZONE_NO_TARGET:  "NO_TARGET",
        ZONE_HEAD:       "HEAD",
        ZONE_NECK_CHEST: "NECK_CHEST",
        ZONE_SIDE_BACK:  "SIDE_BACK",
        ZONE_ABDOMEN:    "ABDOMEN",
        ZONE_FORELIMB:   "FORELIMB",
        ZONE_HINDLIMB:   "HINDLIMB",
        ZONE_TAIL:       "TAIL",
    }

    # ── Geometry ratios, all relative to body_len = |Hip - Chest| ──────────
    HEAD_RADIUS_RATIO      = 0.30
    NECK_RADIUS_RATIO      = 0.22
    TORSO_HALF_LEN_RATIO   = 0.55   # torso ellipse long-axis half length
    TORSO_HALF_WIDTH_RATIO = 0.30   # torso ellipse short-axis half length
    LIMB_STRIP_HW_RATIO    = 0.06   # forelimb/hindlimb strip half width
    LIMB_PAW_RADIUS_RATIO  = 0.05
    TAIL_STRIP_HW_RATIO    = 0.045  # tail strip half width (single shared region)

    # Body length clamp (guards against exploding geometry on extreme poses)
    BODY_LEN_MIN_PX = 300.0
    BODY_LEN_MAX_PX = 650.0

    # ── Output (file / MQTT only — never fed back to the main program) ────
    OUTPUT_ENABLED   = True
    OUTPUT_CSV_PATH  = _os.path.join(_MODULE_DIR, "results.csv")
    LOG_INTERVAL_SEC = 2.0   # minimum seconds between persisted snapshot rows

    MQTT_ENABLED = False               # off by default; paho-mqtt is optional
    MQTT_HOST    = "127.0.0.1"
    MQTT_PORT    = 1883
    MQTT_TOPIC   = "cat/ext_body_zone"

    # ── Node-RED HTTP output (raw geometry, for client-side visualization
    # only — Node-RED does the drawing; this module never renders anything) ──
    NODERED_ENABLED       = True
    NODERED_URL           = "http://127.0.0.1:1880/ext_zone_result"
    NODERED_TIMEOUT       = 0.3
    GEO_PUBLISH_INTERVAL_SEC = 0.3   # throttle: raw pixel coords, not every frame
