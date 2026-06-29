class LickConfig:
    # ── 關鍵點索引（對應 YOLO-Pose 輸出的骨架編號） ───────────────────────
    KP_NOSE      = 0   # 鼻子
    KP_LEFT_EAR  = 1   # 左耳
    KP_RIGHT_EAR = 2   # 右耳
    KP_CHEST     = 3   # 胸部
    KP_HIP       = 5   # 髖部

    # LIMB_SEGMENTS: (群組標籤, 膝關節索引, 腳掌索引)
    LIMB_SEGMENTS = [
        ("FL", 6, 7),   # 前左肢  膝=6, 腳掌=7
        ("FR", 8, 9),   # 前右肢  膝=8, 腳掌=9
        ("HL", 10, 11), # 後左肢  膝=10, 腳掌=11
        ("HR", 12, 13), # 後右肢  膝=12, 腳掌=13
    ]

    # ── 關鍵點信心值門檻 ──────────────────────────────────────────────────
    EAR_CONF_THRESHOLD  = 0.5   # 耳朵關鍵點最低信心值，低於此值視為耳朵不可見
    NOSE_CONF_THRESHOLD = 0.5   # 鼻子關鍵點最低信心值，低於此值不執行接觸判定
    LIMB_CONF_THRESHOLD = 0.10  # 四肢關鍵點最低信心值，低於此值跳過該肢體
    BODY_KP_CONF        = 0.5   # 胸部/髖部信心值門檻，用於計算身體尺度

    # ── 狀態平滑窗口 ──────────────────────────────────────────────────────
    STATE_SMOOTH_WINDOW = 30    # 頭部朝向狀態平滑的滑動窗口大小（幀數）

    # ── 頭部朝向狀態標籤 ─────────────────────────────────────────────────
    STATE_UNKNOWN    = "UNKNOWN"       # 無法判斷朝向
    STATE_FRONT      = "FACING_CAMERA" # 正面朝向鏡頭
    STATE_FRONT_LEFT = "FRONT_LEFT"    # 偏左的正面
    STATE_FRONT_RIGHT= "FRONT_RIGHT"   # 偏右的正面
    STATE_BACK       = "BACK_VIEW"     # 背對鏡頭
    STATE_FRONT_VIEW = "FRONT_VIEW"    # 前視圖保護觸發（抑制舔舐判定）
    STATE_NO_CAT     = "NO_CAT"        # 畫面中無貓

    # ── 貓咪視角凝視方向門檻（以貓咪身體坐標系為基準） ──────────────────
    CAT_FRONT_FORWARD_MIN = 0.08  # 判定「朝前」所需的前向分量最小值
    CAT_BACK_FORWARD_MIN  = 0.10  # 判定「背後」所需的前向分量最小值
    CAT_LR_MARGIN         = 0.06  # 左右偏移容許誤差（小於此值視為正中）
    CAT_LR_SIGN           = 1.0   # 左右方向符號（+1 = 無翻轉；-1 = 鏡像）

    # ── 使用者規則門檻（角度/距離輔助判斷頭部朝向） ──────────────────────
    FRONT_CAMERA_ANGLE_MIN_DEG = 45.0  # 耳鼻夾角超過此值（度）判定為面朝鏡頭
    FRONT_CAMERA_NORM_MIN      = 0.30  # 耳間距標準化值超過此值輔助判定為正面
    BACK_CAMERA_NOSE_CONF_MAX  = 0.5   # 背對時鼻子信心值上限（高於此值不認為是背對）
    BACK_CAMERA_DIST_MIN_PX    = 3.0   # 判定背對所需的耳間距最小像素值
    BACK_VIEW_REQUIRE_LOW_NOSE = True  # True = 背對判定要求鼻子信心值必須低於門檻

    # ── 正面視圖保護（front-view guard）────────────────────────────────
    # 啟用後，當耳間距與身體尺度比值過大（貓面朝鏡頭）時，
    # 暫停舔舐接觸判定，避免誤判。
    FRONT_VIEW_GUARD_ENABLED      = True   # 是否啟用正面保護機制
    FRONT_VIEW_BODY_EAR_RATIO_MAX = 0.75   # 耳間距/身體尺度比值超過此值即觸發保護

    # ── 身體區域幾何參數 ──────────────────────────────────────────────────
    BODY_ELLIPSE_W_RATIO  = 0.65   # 身體橢圓寬度 = 身體長度 × 此比例
    BODY_ELLIPSE_H_RATIO  = 0.27   # 身體橢圓高度 = 身體長度 × 此比例
    HEAD_RAY_LENGTH_RATIO = 1.60   # 頭部方向射線長度 = 耳間距 × 此比例
    HEAD_RAY_MIN_PX       = 60.0   # 頭部射線最短像素長度下限

    # ── 鼻子接觸梯形幾何參數 ──────────────────────────────────────────────
    NOSE_TRAP_THICKNESS_CM    = 2.2   # 梯形厚度（以貓身體長度換算的公分數）
    NOSE_TRAP_THICKNESS_SCALE = 1.0   # 梯形厚度縮放係數（1.0 = 不縮放）
    NOSE_TRAP_TOP_W_RATIO     = 0.10  # 梯形頂寬 = 身體長度 × 此比例
    NOSE_TRAP_BOT_W_RATIO     = 0.20  # 梯形底寬 = 身體長度 × 此比例
    NOSE_TRAP_W_SCALE         = 1.15  # 梯形整體寬度額外放大係數
    CAT_BODY_LENGTH_CM        = 40.0  # 貓咪標準身體長度（公分），用於像素/公分換算

    # ── 接觸幾何尺寸夾鉗（防止極端姿勢下幾何爆炸） ──────────────────────
    CONTACT_BODY_LEN_MIN_PX = 300.0   # 身體長度像素下限，低於此值夾鉗到此
    CONTACT_BODY_LEN_MAX_PX = 650.0   # 身體長度像素上限，高於此值夾鉗到此
    LIMB_CONTACT_SCALE      = 1.0     # 四肢接觸區域整體縮放係數
    LIMB_PAW_CIRCLE_R_RATIO = 0.04    # 腳尖圓圈半徑 = 身體長度 × 此比例
    LIMB_STRIP_HW_RATIO     = 0.055   # 四肢長條寬度 = 身體長度 × 此比例
    LIMB_STRIP_EDGE_GAP     = 0.0     # 四肢長條與腳尖圓之間的間距（像素，0 = 緊鄰）

    # ── 關鍵點 EMA 平滑（1.0 = 不平滑，直接使用原始值） ─────────────────
    EMA_ALPHA = 1.0  # 指數移動平均係數；調低可平滑抖動，但會增加延遲

    # ── 區域標籤 ─────────────────────────────────────────────────────────
    ZONE_NO_TARGET = "NO_TARGET"    # 鼻子未命中任何區域
    ZONE_BODY      = "BODY_CENTER"  # 命中身體中心區域

    # ── 橢圓邊界取樣點數（用於判斷鼻子是否在橢圓內） ────────────────────
    ELLIPSE_SAMPLES = 40  # 取樣點越多精度越高，但計算量略增

    # ── Node-RED 推送設定 ─────────────────────────────────────────────────
    NODERED_URL     = "http://127.0.0.1:1880/lick_zone_result"  # 結果推送端點
    NODERED_TIMEOUT = 0.3  # HTTP 請求逾時秒數（避免卡住主迴圈）
