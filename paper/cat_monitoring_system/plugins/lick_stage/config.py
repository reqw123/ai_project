class LickConfig:
    # ── Keypoint indices ──────────────────────────────────────────────
    KP_NOSE          = 0
    KP_LEFT_EAR      = 1
    KP_RIGHT_EAR     = 2
    KP_CHEST         = 3
    KP_HIP           = 5

    # LIMB_SEGMENTS: (group_label, knee_idx, paw_idx)
    LIMB_SEGMENTS = [
        ("FL", 6, 7),   # front-left  knee=6, paw=7
        ("FR", 8, 9),   # front-right knee=8, paw=9
        ("HL", 10, 11), # hind-left   knee=10, paw=11
        ("HR", 12, 13), # hind-right  knee=12, paw=13
    ]

    # ── Confidence thresholds ─────────────────────────────────────────
    EAR_CONF_THRESHOLD  = 0.5
    NOSE_CONF_THRESHOLD = 0.3
    LIMB_CONF_THRESHOLD = 0.10
    BODY_KP_CONF        = 0.5  # chest / hip threshold for body scale

    # ── State smoothing ───────────────────────────────────────────────
    STATE_SMOOTH_WINDOW = 30

    # ── Face direction state labels ───────────────────────────────────
    STATE_UNKNOWN      = "UNKNOWN"
    STATE_FRONT        = "FACING_CAMERA"
    STATE_FRONT_LEFT   = "FRONT_LEFT"
    STATE_FRONT_RIGHT  = "FRONT_RIGHT"
    STATE_BACK         = "BACK_VIEW"
    STATE_FRONT_VIEW   = "FRONT_VIEW"
    STATE_NO_CAT       = "NO_CAT"

    # ── Cat-centric gaze thresholds ───────────────────────────────────
    CAT_FRONT_FORWARD_MIN = 0.08
    CAT_BACK_FORWARD_MIN  = 0.10
    CAT_LR_MARGIN         = 0.06
    CAT_LR_SIGN           = 1.0

    # ── User-rule thresholds ──────────────────────────────────────────
    FRONT_CAMERA_ANGLE_MIN_DEG = 45.0
    FRONT_CAMERA_NORM_MIN      = 0.30
    BACK_CAMERA_NOSE_CONF_MAX  = 0.5
    BACK_CAMERA_DIST_MIN_PX    = 3.0
    BACK_VIEW_REQUIRE_LOW_NOSE = True

    # ── Front-view guard ─────────────────────────────────────────────
    # body_scale_norm guard disabled — requires frame_diag not available in plugin
    FRONT_VIEW_GUARD_ENABLED      = True
    FRONT_VIEW_BODY_EAR_RATIO_MAX = 0.75

    # ── Body region geometry ──────────────────────────────────────────
    BODY_ELLIPSE_W_RATIO  = 0.65
    BODY_ELLIPSE_H_RATIO  = 0.27
    HEAD_RAY_LENGTH_RATIO = 1.60
    HEAD_RAY_MIN_PX       = 60.0

    # ── Nose contact trapezoid ────────────────────────────────────────
    NOSE_TRAP_THICKNESS_CM    = 2.2
    NOSE_TRAP_THICKNESS_SCALE = 1.0
    NOSE_TRAP_TOP_W_RATIO     = 0.10
    NOSE_TRAP_BOT_W_RATIO     = 0.20
    NOSE_TRAP_W_SCALE         = 1.15
    CAT_BODY_LENGTH_CM        = 40.0

    # ── Contact geometry clamp ────────────────────────────────────────
    CONTACT_BODY_LEN_MIN_PX = 300.0
    CONTACT_BODY_LEN_MAX_PX = 650.0
    LIMB_CONTACT_SCALE      = 1.0
    LIMB_PAW_CIRCLE_R_RATIO = 0.04
    LIMB_STRIP_HW_RATIO     = 0.055
    LIMB_STRIP_EDGE_GAP     = 0.0

    # ── EMA (1.0 = bypass) ────────────────────────────────────────────
    EMA_ALPHA = 1.0

    # ── Zone labels ──────────────────────────────────────────────────
    ZONE_NO_TARGET = "NO_TARGET"
    ZONE_BODY      = "BODY_CENTER"

    # ── Ellipse boundary sampling ─────────────────────────────────────
    ELLIPSE_SAMPLES = 40

    # ── Node-RED ──────────────────────────────────────────────────────
    NODERED_URL     = "http://127.0.0.1:1880/lick_zone_result"
    NODERED_TIMEOUT = 0.3
