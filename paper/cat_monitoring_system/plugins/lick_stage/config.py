class LickConfig:
    # ── 關鍵點索引（對應 YOLO-Pose 輸出的骨架編號） ───────────────────────
    KP_NOSE      = 0   # 鼻子
    KP_LEFT_EAR  = 1   # 左耳
    KP_RIGHT_EAR = 2   # 右耳
    KP_CHEST     = 3   # 胸部
    KP_MID_BACK  = 4   # 背部中心（標記於背部最高點，非 chest-hip 連線中點）
    KP_HIP       = 5   # 髖部

    # LIMB_SEGMENTS: (群組標籤, 膝關節索引, 腳掌索引)
    LIMB_SEGMENTS = [
        ("FL", 6, 7),   # 前左肢  膝=6, 腳掌=7
        ("FR", 8, 9),   # 前右肢  膝=8, 腳掌=9
        ("HL", 10, 11), # 後左肢  膝=10, 腳掌=11
        ("HR", 12, 13), # 後右肢  膝=12, 腳掌=13
    ]

    # ── 關鍵點信心值門檻 ──────────────────────────────────────────────────
    EAR_CONF_THRESHOLD  = 0.3   # 耳朵關鍵點最低信心值，低於此值視為耳朵不可見
    NOSE_CONF_THRESHOLD = 0.3   # 鼻子關鍵點最低信心值，低於此值不執行接觸判定
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

    # ── 梯形彎曲自適應（mid_back 偏離 chest-hip 中點的比例，驅動梯形縮放）──
    # mid_back 標記在貓咪背部最高點（非 chest-hip 連線中點）。貓咪拱背/蜷曲
    # 理毛時，該點會明顯偏離 chest-hip 直線，此時 body_len（chest-hip 直線
    # 距離）會因姿態彎曲而低估「真實」身體尺度，導致鼻子接觸梯形在最需要
    # 精準判定的姿態下反而系統性偏小。用 mid_back_dist_pct（mid_back 到
    # chest-hip 中點的正規化距離百分比）線性內插出一個縮放倍率，只套用在
    # 梯形尺寸上（不影響身體橢圓/四肢區域，那些量測的是「當下真實」幾何，
    # 不該被歷史或姿態推測放大縮小）。
    CURVATURE_PCT_MIN   = 0.0   # mid_back_dist_pct 下限：脊椎接近打直
    CURVATURE_PCT_MAX   = 30.0  # mid_back_dist_pct 上限：明顯拱背/蜷曲
    CURVATURE_BOOST_MIN = 0.85  # 對應 CURVATURE_PCT_MIN 時的梯形縮放倍率（打直時縮小）
    CURVATURE_BOOST_MAX = 1.20  # 對應 CURVATURE_PCT_MAX 時的梯形縮放倍率（拱背時放大）

    # ── 接觸幾何尺寸夾鉗（防止極端姿勢下幾何爆炸） ──────────────────────
    CONTACT_BODY_LEN_MIN_PX = 300.0   # 身體長度像素下限，低於此值夾鉗到此
    CONTACT_BODY_LEN_MAX_PX = 650.0   # 身體長度像素上限，高於此值夾鉗到此
    LIMB_CONTACT_SCALE      = 1.0     # 四肢接觸區域整體縮放係數
    LIMB_PAW_CIRCLE_R_RATIO = 0.04    # 腳尖圓圈半徑 = 身體長度 × 此比例
    LIMB_STRIP_HW_RATIO     = 0.055   # 四肢長條寬度 = 身體長度 × 此比例
    LIMB_STRIP_EDGE_GAP     = 0.0     # 四肢長條與腳尖圓之間的間距（像素，0 = 緊鄰）

    # ── 關鍵點 EMA 平滑（1.0 = 不平滑，直接使用原始值） ─────────────────
    # 只影響本 plugin 內部的接觸判定/overlay 穩定度，與 ST-GCN 推論用的
    # kp_ema_alpha（frame_processor 傳入的是 raw_kpts）完全independent。
    EMA_ALPHA = 0.4  # 指數移動平均係數；調低可平滑抖動，但會增加延遲

    # ── 鼻子接觸梯形方向向量穩定化（解決貓側躺/頭部縮短時梯形抖動亂轉）──
    # trap_perp（耳線方向）對雜訊極敏感（耳間距短時尤其明顯），用「翻轉感知
    # EMA」+「連續反向才確認」跨幀穩定：新向量若與前一幀方向明顯相反，先翻轉
    # 再平均，避免兩個反向向量直接相加抵消；同時提供緩衝區間與確認幀數，噪音
    # 在邊界附近來回時不會讓 trap_perp 整個 180° 亂轉，只有連續多幀都反向
    # （例如貓真的轉身）才會被接受為真正的方向改變。
    TRAP_PERP_EMA_ALPHA    = 0.35  # trap_perp 的 EMA 係數（越小越穩定，但轉向反應越慢）
    TRAP_PERP_FLIP_MARGIN  = 0.15  # 內積需低於 -此值才視為「真的反向」，避免邊界抖動誤觸發翻轉
    # 連續幾幀都判定為「反向」才接受為真正的方向改變（例如貓轉身）；
    # 少於此幀數視為單幀雜訊，暫時忽略、沿用前一個穩定方向，避免被雜訊拉走
    TRAP_PERP_FLIP_CONFIRM_FRAMES = 6

    # trap_dir（梯形延伸方向）改由 trap_dir_from_perp() 從穩定後的 trap_perp
    # 決定性推導，強制指向影像下方（鼻子/短邊保證在上面，貓咪不會倒著理毛）。
    # 但這個推導出來的候選值仍會因 trap_perp 殘留的微小雜訊而在自己的判斷
    # 邊界（trap_perp 幾乎與畫面垂直，即耳朵連線幾乎垂直、貓側躺頭部縮短時）
    # 跳動，而且這個雜訊已經是 EMA 平滑過的、具有時間相關性，用「連續反向
    # 才確認」的計數方式反而容易被平滑雜訊的自然延續性誤判成真的轉向。
    # 因此 trap_dir 只用純翻轉感知 EMA（不做確認幀數計數），並採用更強的
    # 平滑（更小的 alpha）壓下殘留雜訊——若 trap_perp 真的持續反向，
    # trap_perp 自己的確認機制會先接受新方向，trap_dir 隨後自然會平滑跟上，
    # 不需要獨立的「瞬間接受」機制。
    #
    # 注意：FLIP_MARGIN 必須維持「窄」（接近 0），不能為了「減少誤觸發翻轉」
    # 而調寬——寬邊界會讓「明顯但未達邊界」的反向讀數被直接拿去平均而非先
    # 翻轉，兩個部分抵銷的向量疊加會讓 EMA 狀態的長度被拉向 0，之後重新
    # 正規化（除以趨近 0 的長度）會把方向放大成任意雜訊，反而更不穩定。
    TRAP_DIR_EMA_ALPHA    = 0.12
    TRAP_DIR_FLIP_MARGIN  = 0.15

    # trap_dir 的「長期方向卡死」安全網：上面的純 EMA 追蹤完全信任
    # _prev_trap_dir 這個歷史錨點，只要初始化那一刻（或貓消失重新出現後
    # 重新初始化那一次）trap_dir_from_perp() 定出的方向剛好不符合直覺
    # （鼻子/短邊在上、身體/長邊在下），後續就會一路「穩定地」錯下去，
    # 因為 EMA 只追求跟前一幀連續、不會主動驗證 y>=0 這個物理假設。
    # 這裡不做「每幀強制」（那樣會在梯形接近水平時造成硬性翻轉抖動，
    # 見上方 trap_dir 段落說明），而是隔離成一道獨立的安全網：只有連續
    # 好幾幀都偏離 y>=0（不是臨界抖動，是真的卡在錯誤方向）才強制拉回，
    # 平常的水平抖動只會讓計數器歸零，不會觸發翻轉。
    TRAP_DIR_WRONG_SIDE_CONFIRM_FRAMES = 10

    # ── 區域標籤 ─────────────────────────────────────────────────────────
    ZONE_NO_TARGET = "NO_TARGET"    # 鼻子未命中任何區域
    ZONE_BODY      = "BODY_CENTER"  # 命中身體中心區域

    # ── 橢圓邊界取樣點數（用於判斷鼻子是否在橢圓內） ────────────────────
    ELLIPSE_SAMPLES = 40  # 取樣點越多精度越高，但計算量略增

    # ── Node-RED 推送設定 ─────────────────────────────────────────────────
    NODERED_URL     = "http://127.0.0.1:1880/lick_zone_result"  # 結果推送端點
    NODERED_TIMEOUT = 0.3  # HTTP 請求逾時秒數（避免卡住主迴圈）
