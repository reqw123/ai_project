"""
貓咪監測系統配置檔案
方便管理所有設置，避免直接修改主程序
"""

import datetime as _datetime
import os
import builtins as _builtins
from pathlib import Path


def _env_str(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return _builtins.float(value)
    except (TypeError, ValueError):
        return default


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_video_input(name, default):
    """讀取影像來源：純數字 -> 攝影機 index，其餘保持字串。"""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _env_size(name, default):
    """讀取尺寸設定：支援 640x480、640,480、640 480；無效時回傳預設值。"""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if not value or value in {"none", "null", "off", "false"}:
        return default

    parts = [part for part in value.replace("x", ",").replace(" ", ",").split(",") if part]
    if len(parts) != 2:
        return default

    try:
        width = int(parts[0])
        height = int(parts[1])
    except (TypeError, ValueError):
        return default

    if width <= 0 or height <= 0:
        return default

    return (width, height)


def _is_valid_port(port):
    return isinstance(port, int) and 1 <= port <= 65535


def _parse_hhmm(value):
    """解析 'HH:MM'（24 小時制）字串，回傳 (hour, minute)；空字串/格式錯誤回傳 None。"""
    if not value:
        return None
    parts = value.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        return None
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return (hour, minute)
    return None


def _normalize_feature_mode(feature_mode):
    mode = str(feature_mode).strip().lower()
    if not mode:
        return "xy"
    return mode


def _get_stgcn_feature_spec(feature_mode):
    """回傳 ST-GCN 特徵模式的通道數與特徵名稱說明。"""
    mode = _normalize_feature_mode(feature_mode)
    specs = {
        "xy": {
            "in_channels": 2,
            "features": ["x", "y"],
            "description": "位置資訊",
        },
        "xy_conf": {
            "in_channels": 3,
            "features": ["x", "y", "conf"],
            "description": "位置 + 信心值",
        },
        "xy_conf_v": {
            "in_channels": 5,
            "features": ["x", "y", "conf", "vx", "vy"],
            "description": "位置 + 信心值 + 速度",
        },
        "xy_conf_v_bone": {
            "in_channels": 7,
            "features": ["x", "y", "conf", "vx", "vy", "bone_x", "bone_y"],
            "description": "位置 + 信心值 + 速度 + 骨架向量",
        },
        "xy_conf_v_bone_bmotion": {
            "in_channels": 9,
            "features": ["x", "y", "conf", "vx", "vy", "bone_x", "bone_y", "bone_mx", "bone_my"],
            "description": "位置 + 信心值 + 速度 + 骨架向量 + 骨架位移",
        },
    }
    if mode not in specs:
        raise ValueError(f"Unknown ST-GCN feature mode: {feature_mode!r}. 支援模式: {list(specs)}")
    return specs[mode]

# ==================== 模型和資料路徑 ====================
class ModelPaths:
    """模型和資料檔案路徑"""
    
    # YOLO 模型
    YOLO_MODEL = _env_str("CAT_MONITORING_YOLO_MODEL", r"C:\ai_project\cat_pose\v11s_128.pt")
    
    # ST-GCN 模型
    STGCN_MODEL = _env_str("CAT_MONITORING_STGCN_MODEL", r"C:\Users\homec\Downloads\stgcn_results\run_122_xy_conf_v_bone_att_on\122_best_model.pth")
    
    # 測試視頻
    VIDEO_INPUT = _env_video_input("CAT_MONITORING_VIDEO_INPUT", r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\泛化測試\5月5日(1).mp4")
                                                                  # rtsp://12345678:456456123@192.168.0.192:554/stream1
    # 日誌和輸出目錄                                             # "C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\泛化測試\5月5日(1).mp4"
    # 日誌和輸出目錄
    LOG_DIR = _env_str("CAT_MONITORING_LOG_DIR", "./logs")
    OUTPUT_DIR = _env_str("CAT_MONITORING_OUTPUT_DIR", "./output")
    
    @classmethod
    def ensure_dirs(cls):
        """確保所有目錄存在"""
        Path(cls.LOG_DIR).mkdir(exist_ok=True)
        Path(cls.OUTPUT_DIR).mkdir(exist_ok=True)
    
    @classmethod
    def validate(cls):
        """驗證模型檔案存在"""
        required_files = {
            "YOLO": cls.YOLO_MODEL,
            "ST-GCN": cls.STGCN_MODEL,
        }
        
        missing = []
        for name, path in required_files.items():
            if not Path(path).exists():
                missing.append(f"{name}: {path}")

        video_src = cls.VIDEO_INPUT
        if isinstance(video_src, int):
            pass
        elif isinstance(video_src, str):
            lower_src = video_src.lower()
            if lower_src.startswith(("rtsp://", "http://", "https://")):
                pass
            elif not Path(video_src).exists():
                missing.append(f"Video: {video_src}")
        else:
            missing.append(f"Video: 不支援的來源型別 {type(video_src).__name__}")
        
        if missing:
            print("⚠ 缺少的檔案:")
            for item in missing:
                print(f"  - {item}")
            return False
        
        return True

# ==================== YOLO 參數 ====================
class YOLOConfig:
    """YOLO 檢測參數

    裝置選擇不在此設定：實際執行時由 routes.py 的 _resolve_runtime_device() 自動偵測 CUDA 可用性。
    """

    # 推論參數
    IMAGE_SIZE = _env_int("CAT_MONITORING_YOLO_IMAGE_SIZE", 640)
    CONFIDENCE_THRESHOLD = _env_float("CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD", 0.50)

# ==================== ST-GCN 參數 ====================
class STGCNConfig:
    """ST-GCN 模型參數"""
    
    # 模型超參數
    SEQUENCE_LENGTH = _env_int("CAT_MONITORING_STGCN_SEQUENCE_LENGTH", 16)          # 時間窗長度（幀數）
    NUM_CLASSES = 5               # 行為類別數

    # 特徵模式（與 train_gcn.py / 推論腳本共用概念）
    # 預設值須與 ModelPaths.STGCN_MODEL 預設 checkpoint（076_xy_conf_v_bone_att_on.pth）一致；
    # 實際推論時 in_channels/attention 仍會依 checkpoint 內容自動偵測並覆寫此設定（見 stgcn_model.py）。
    FEATURE_MODE = _normalize_feature_mode(_env_str("CAT_MONITORING_STGCN_FEATURE_MODE", "xy_conf_v_bone"))
    _get_stgcn_feature_spec(FEATURE_MODE)  # 僅用於在載入時驗證 FEATURE_MODE 合法，實際通道數由 checkpoint 自動偵測

    # 推論用滑動步長（每幾幀執行一次 ST-GCN，對應 CLASSIFY_STRIDE）
    # 訓練用的步長由 stgcn_config.yaml 的 WINDOW_STRIDE 管理，與此無關
    WINDOW_STRIDE = _env_int("CAT_MONITORING_STGCN_WINDOW_STRIDE", 2)

    # 裝置選擇不在此設定：實際執行時由 routes.py 的 _resolve_runtime_device() 自動偵測 CUDA 可用性。

    # FPS 同步：對來源影片做降採樣，使模型輸入時基符合訓練設定。
    # TARGET_MODEL_FPS: 推論與串流使用的目標 FPS，須與訓練時一致。
    #   調低（例如 15）可降低 YOLO + ST-GCN 推論頻率，減少 CPU/GPU 負擔，但反應會變慢。
    TARGET_MODEL_FPS = _env_float("CAT_MONITORING_TARGET_MODEL_FPS", 30.0)
    # ENABLE_FPS_DOWNSAMPLE: True 代表來源 FPS 高於 TARGET_MODEL_FPS 時自動跳幀；False 代表每幀都處理。
    ENABLE_FPS_DOWNSAMPLE = _env_bool("CAT_MONITORING_ENABLE_FPS_DOWNSAMPLE", True)

    # 關鍵點 EMA 平滑（須與 train_gcn.py 的 KP_EMA_ALPHA 保持一致）
    # alpha 越大 → 越貼近原始偵測值；alpha 越小 → 越平滑但延遲增加
    KP_EMA_ALPHA = _env_float("CAT_MONITORING_KP_EMA_ALPHA", 1.0)
    # 行為類別名稱與顏色見 BehaviorTrackingConfig.BEHAVIOR_CATEGORIES / utils.constants.BEHAVIOR_COLORS

# ==================== ST-GCN 訓練設定（唯一權威來源：stgcn_config.yaml） ====================
class STGCNTrainingConfig:
    """0_train_gcn.py 訓練這個模型時實際使用的參數，全部從 stgcn_config.yaml
    讀取——SEQUENCE_LENGTH/NUM_JOINTS/WINDOW_STRIDE/SPATIAL_KERNEL_SIZE/
    FEATURE_MODE/BEHAVIOR_PREFIXES 等重要參數的權威來源都是那份 YAML，
    這裡不重新定義任何預設值，只負責讀出來供 get_config_summary()／
    validate_all_config() 顯示與比對，避免每次要確認訓練參數都要另外開檔案。

    刻意跟 STGCNConfig 分開類別：STGCNConfig 是「推論/串流」執行期設定
    （env 變數可覆寫，部分欄位如 in_channels/num_joints/attention 還會在
    載入 checkpoint 時被自動偵測結果覆寫，見 models/stgcn_model.py），
    跟「當初訓練這顆 checkpoint 時用的設定」是兩個不同時間點、不同目的
    的東西，不應該混在同一個類別、更不該互相覆寫。

    讀檔路徑跟 0_train_gcn.py 用同一個環境變數 STGCN_CONFIG_PATH（未設定
    時預設指向 cat_monitoring_system/stgcn_config.yaml），確保兩邊讀的是
    同一份檔案。找不到檔案、YAML 格式錯誤、或 PyYAML 未安裝時，get() 一律
    回傳 default（通常是 None）而不拋例外——config.py 被很多地方 import，
    這裡讀取失敗不該讓整個系統掛掉，只是配置摘要少顯示這幾行。
    """
    _DEFAULT_PATH = str(Path(__file__).parent / "cat_monitoring_system" / "stgcn_config.yaml")
    _PATH = _env_str("STGCN_CONFIG_PATH", _DEFAULT_PATH)
    _cache = None
    _load_error = None

    @classmethod
    def _load(cls):
        if cls._cache is not None:
            return cls._cache
        try:
            import yaml
            with open(cls._PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError(f"{cls._PATH} 內容不是一個 mapping/object")
            cls._cache = data
        except Exception as e:
            cls._load_error = str(e)
            cls._cache = {}
        return cls._cache

    @classmethod
    def get(cls, key, default=None):
        return cls._load().get(key, default)

    @classmethod
    def is_available(cls) -> bool:
        """True 代表成功讀到檔案內容；False 代表檔案不存在/格式錯誤/PyYAML
        未安裝——呼叫端可用這個判斷要不要顯示「讀取失敗」提示。"""
        cls._load()
        return cls._load_error is None

# ==================== 異常檢測參數 ====================
class AnomalyDetectionConfig:
    """靜止偵測與運動分析

    v2 起 motion_score 單位為 body_fraction × 100（每幀位移 / 胸→髖距離 × 100），
    與訓練管線 normalize_skeleton_coords 一致，消除拍攝距離影響。
    閾值參考值：靜止呼吸 < 2；舔毛/抓撓 5-15；走路 > 10
    """

    MAX_MOTION = 20.0            # motion_score 正規化分母（body_fraction×100）；走路約 10-20
    KP_CONF_THRES = 0.5          # 只使用高於此信心的關鍵點計算 motion_score
    ROLLING_WINDOW_SIZE = 30     # 滾動均值視窗大小（幀數，30fps ≈ 1 秒）
    STILL_MOTION_THRESHOLD = 3.0    # 滾動均值低於此值（body_fraction×100）判為靜止；呼吸抖動約 < 2

# ==================== Skeleton Quality Assessment（骨架品質雙重判定）====================
class SQAConfig:
    """「GCN 分類為主、幾何判斷為輔」雙重判定總開關。

    ENABLE_SQA_DUAL_JUDGMENT=True 時，FrameProcessor 會在每次 ST-GCN
    推論同一個窗口後，額外呼叫 skeleton_quality_assessment.evaluate_
    window() 做幾何合理性檢查；只要被判定為不可信，就把該幀的分類結果
    覆蓋成 LOW_CONF（信心值歸零）——跟診斷腳本 test_bone_length_
    stability.py（模式2）驗證過的覆蓋規則一致。

    這是唯一的總開關；3 項個別指標（midback_offset_ratio/midback_angle/
    body_axis_score_jitter）各自要不要參與判定，由
    cat_monitoring_system/processors/skeleton_quality_assessment.py 模組內的
    ENABLE_MIDBACK_OFFSET_CHECK/ENABLE_MIDBACK_ANGLE_CHECK/
    ENABLE_SCORE_JITTER_CHECK 三個變數各自控制，不在這裡設定——刻意保持
    低耦合：config.py 只決定「要不要啟用這整套機制」，機制內部的細節
    門檻/開關留在模組自己的檔案裡管理，兩者不互相知道對方的存在。

    evaluate_window() 本身是 fail-safe（任何內部錯誤都回傳
    reliable=True，等同不覆蓋），FrameProcessor 呼叫端也會再包一層
    try/except——即使這個模組完全壞掉或被整個刪除，都不會影響主系統
    其餘功能運行，最多只是這個雙重判定不生效。

    預設 False：這是還在校準門檻階段的新機制（門檻值目前只用少量影片
    校準過），正式套用前建議先在 GUI 模式肉眼比對過覆蓋規則是否合理，
    確認沒問題後再開啟。
    """
    ENABLE_SQA_DUAL_JUDGMENT = _env_bool("CAT_MONITORING_ENABLE_SQA_DUAL_JUDGMENT", True)


# ==================== 執行模式參數 ====================
class RunModeConfig:
    """控制 main.py 啟動時走哪一種模式，整體處理架構（FrameProcessor 等）不受影響。

    "server"（預設）：現行行為，啟動 Flask HTTP 伺服器 + Node-RED 上線通知
    "gui"           ：不啟動 Flask/Node-RED，直接用同一套 FrameProcessor 開本地視窗顯示
    """
    MODE = _env_str("CAT_MONITORING_RUN_MODE", "gui")

    # server 模式下，處理管線（開影片、載入 YOLO/ST-GCN、tracker 統計、CSV、Node-RED 推送）
    # 原本要等第一個打到 /stream 等路由的 HTTP 請求才會啟動（見 routes.py 的
    # _ensure_processor_started()），也就是實務上要有人打開 Dashboard 點播放才會真正開始跑。
    # 預錄影片、排程無人值守執行時沒人會去點播放，關掉這個延遲啟動，改成 Python 進程一啟動
    # 就立刻開始處理，不等任何請求。設為 False 可還原成原本「有人連線才啟動」的行為。
    AUTO_START_PROCESSING = _env_bool("CAT_MONITORING_AUTO_START_PROCESSING", True)

    # 排程啟動時間（24 小時制 "HH:MM"，例如 "06:00"）。設定後，處理管線在 Python 進程
    # 啟動的當下不會立刻執行，而是持續等待直到真實世界時間到達此時刻才開始跑；
    # 若進程啟動時已經過了這個時刻（例如排程 06:00 但 14:00 才啟動），視為時間已到，
    # 立即開始，不會傻等到隔天。留空（預設）代表不啟用排程，沿用 AUTO_START_PROCESSING
    # 的行為（一啟動就跑或永遠不自動跑）。用於預錄影片、無人值守的排程執行情境
    # （例如固定每天啟動一次，只想在 06:00 才開始處理當天份的影片）。
    SCHEDULED_START_TIME = _env_str("CAT_MONITORING_SCHEDULED_START_TIME", "")  #"06:00"
    SCHEDULED_START_HHMM = _parse_hhmm(SCHEDULED_START_TIME)

    # 排程結束時間（24 小時制 "HH:MM"，例如 "12:00"）。留空（預設）＝不設結束時間，
    # 只要 SCHEDULED_START_TIME 有設，一旦時間到就開始處理、之後永遠不會自動停止
    # （對應「排程時間到、之後一直運行」的用法）。
    # 若同時設定了開始與結束時間，就變成「區段執行」：只有現在時刻落在
    # [開始, 結束) 之間才處理，區間外自動暫停，且每一天都會依同一組 HH:MM 重新套用
    # （不需要重啟 Python，也不會重新載入模型——暫停只是不讀取/不推論，不釋放資源）。
    # 若只設結束、沒設開始，開始時間視為當天 00:00。
    SCHEDULED_END_TIME = _env_str("CAT_MONITORING_SCHEDULED_END_TIME", "")  #"12:00"
    SCHEDULED_END_HHMM = _parse_hhmm(SCHEDULED_END_TIME)

    @classmethod
    def is_within_active_window(cls, now=None):
        """判斷「現在」是否落在排程允許處理的時間內。

        - 開始/結束都沒設定：一律允許（沒有時間限制）。
        - 只設開始，沒設結束：現在 >= 今天的開始時刻就允許，且此後永遠允許
          （不會因為跨天而重新暫停——這是「排程時間到、之後一直運行」的模式）。
        - 有設結束（不論有沒有設開始）：視為每日重複的區段，只有現在落在
          [開始（預設 00:00）, 結束) 之間才允許；結束時刻比開始時刻早（例如
          22:00~06:00 跨午夜）時，視為跨天區間處理。
        """
        if now is None:
            now = _datetime.datetime.now()
        start_hhmm = cls.SCHEDULED_START_HHMM
        end_hhmm = cls.SCHEDULED_END_HHMM

        if start_hhmm is None and end_hhmm is None:
            return True

        if end_hhmm is None:
            start_dt = now.replace(hour=start_hhmm[0], minute=start_hhmm[1], second=0, microsecond=0)
            return now >= start_dt

        start_h, start_m = start_hhmm if start_hhmm is not None else (0, 0)
        start_dt = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        end_dt = now.replace(hour=end_hhmm[0], minute=end_hhmm[1], second=0, microsecond=0)
        if end_dt <= start_dt:
            return now >= start_dt or now < end_dt
        return start_dt <= now < end_dt

# ==================== Flask 服務參數 ====================
class FlaskConfig:
    """Flask Web 服務參數"""
    
    HOST = _env_str("CAT_MONITORING_FLASK_HOST", "0.0.0.0")
    PORT = _env_int("CAT_MONITORING_FLASK_PORT", 5000)
    DEBUG = _env_bool("CAT_MONITORING_FLASK_DEBUG", False)
    THREADED = _env_bool("CAT_MONITORING_FLASK_THREADED", True)
    
    # JPEG 壓縮品質 (1-100)
    JPEG_QUALITY = _env_int("CAT_MONITORING_JPEG_QUALITY", 30)

# ==================== Node-RED 參數 ====================
class NodeRedConfig:
    """Node-RED 通訊參數"""
    
    HOST = _env_str("CAT_MONITORING_NODERED_HOST", "127.0.0.1")
    PORT = _env_int("CAT_MONITORING_NODERED_PORT", 1880)
    
    # 推送間隔（秒）
    PUSH_INTERVAL = _env_float("CAT_MONITORING_NODERED_PUSH_INTERVAL", 2)

    ENDPOINT_NOTIFY = _env_str("CAT_MONITORING_NODERED_ENDPOINT_NOTIFY", f"http://{HOST}:{PORT}/python_online")
    ENDPOINT_RESULT = _env_str("CAT_MONITORING_NODERED_ENDPOINT_RESULT", f"http://{HOST}:{PORT}/yolo_result")
    ENDPOINT_RESULT_V2 = _env_str("CAT_MONITORING_NODERED_ENDPOINT_RESULT_V2", f"http://{HOST}:{PORT}/yolo_result_v2")
    
    # 超時時間（秒）
    TIMEOUT = _env_float("CAT_MONITORING_NODERED_TIMEOUT", 2)

# ==================== 行為追蹤參數 ====================
class BehaviorTrackingConfig:
    """行為統計和追蹤"""
    
    # 歷史記錄大小
    MAX_HISTORY_SIZE = _env_int("CAT_MONITORING_MAX_HISTORY_SIZE", 100)  # 行為歷史清單最多保留筆數

    # 活動力窗口
    ACTIVITY_WINDOW_SIZE = _env_int("CAT_MONITORING_ACTIVITY_WINDOW_SIZE", 54)   # 30fps × 1.8s = 54；須 ≥ TARGET_MODEL_FPS × ACTIVITY_SCORE_WINDOW_SECONDS

    # 行為轉換與活動分數參數
    MIN_RECORD_DURATION_SECONDS = _env_float("CAT_MONITORING_MIN_RECORD_DURATION_SECONDS", 2.0)  # 單一行為最短記錄秒數
    ACTIVITY_SCORE_WINDOW_SECONDS = _env_float("CAT_MONITORING_ACTIVITY_SCORE_WINDOW_SECONDS", 1.2)  # 活動分數取樣時間窗（秒）；越短反應越快
    LOW_CONFIDENCE_ACTIVITY_WEIGHT = _env_float("CAT_MONITORING_LOW_CONFIDENCE_ACTIVITY_WEIGHT", 0.5)  # 低信心幀的活動權重
    STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD = _env_float("CAT_MONITORING_STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD", 0.80,)  # ST-GCN 行為標籤輸出門檻；低於此值視為 normal

    # 顯示層 hysteresis：同一個新類別要連續達到這麼多次分類視窗（非幀數，見 STGCNConfig.WINDOW_STRIDE
    # 對應的 CLASSIFY_STRIDE）才會真的切換畫面上顯示的行為標籤，用來過濾單一視窗瞬間誤判造成的畫面閃爍
    # （例如動作轉換瞬間、windup 動作被誤判成別的類別）。只影響顯示，不影響底層逐視窗統計/紀錄。
    # 五個行為各自獨立設定（key 對應 BEHAVIOR_CATEGORIES 的 id：0 walk/1 lick/2 scratch/3 shake/4 stop）；
    # <=1 等同該行為關閉此機制，維持原本逐視窗即時顯示的行為。
    # shake 動作本身只持續 0.5~1 秒，換算成分類視窗數不多，門檻設太高會導致整個 shake 事件
    # 結束前都累積不到門檻次數、畫面永遠來不及切換顯示，因此預設給比其他行為低的門檻。
    DISPLAY_HYSTERESIS_WINDOWS_WALK = _env_int("CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_WALK", 3)
    DISPLAY_HYSTERESIS_WINDOWS_LICK = _env_int("CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_LICK", 3)
    DISPLAY_HYSTERESIS_WINDOWS_SCRATCH = _env_int("CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_SCRATCH", 3)
    DISPLAY_HYSTERESIS_WINDOWS_SHAKE = _env_int("CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_SHAKE", 3)
    DISPLAY_HYSTERESIS_WINDOWS_STOP = _env_int("CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_STOP", 3)
    DISPLAY_HYSTERESIS_WINDOWS = {
        0: DISPLAY_HYSTERESIS_WINDOWS_WALK,
        1: DISPLAY_HYSTERESIS_WINDOWS_LICK,
        2: DISPLAY_HYSTERESIS_WINDOWS_SCRATCH,
        3: DISPLAY_HYSTERESIS_WINDOWS_SHAKE,
        4: DISPLAY_HYSTERESIS_WINDOWS_STOP,
    }

    # 貓咪偵測消失容忍：YOLO 連續幾幀沒偵測到貓，才真的視為「貓消失」並重置 EMA/緩衝區。
    # 容忍期間內沿用最後一次偵測到的關鍵點，避免單幀漏偵測就整個中斷分類/顯示。
    # <=0 等同關閉此機制，維持原本單幀漏偵測就立即重置的行為。
    CAT_MISSING_TOLERANCE_FRAMES = _env_int("CAT_MONITORING_CAT_MISSING_TOLERANCE_FRAMES", 5)

    # 警報門檻
    SCRATCH_ALERT_TIME_SECONDS = _env_float("CAT_MONITORING_SCRATCH_ALERT_TIME_SECONDS", 10.0)  # 單日搔抓累積秒數警戒值
    SCRATCH_ALERT_COUNT_THRESHOLD = _env_int("CAT_MONITORING_SCRATCH_ALERT_COUNT_THRESHOLD", 5)  # 單日搔抓次數警戒值
    LICK_ALERT_TIME_SECONDS = _env_float("CAT_MONITORING_LICK_ALERT_TIME_SECONDS", 10.0)  # 單日舔舐累積秒數警戒值
    SHAKE_ALERT_COUNT_THRESHOLD = _env_int("CAT_MONITORING_SHAKE_ALERT_COUNT_THRESHOLD", 10)  # 單日甩頭次數警戒值
    STOP_ALERT_TIME_SECONDS = _env_float("CAT_MONITORING_STOP_ALERT_TIME_SECONDS", 300.0)  # 單日靜止累積秒數警戒值
    LOW_ACTIVITY_TIME_THRESHOLD_SECONDS = _env_float("CAT_MONITORING_LOW_ACTIVITY_TIME_THRESHOLD_SECONDS", 20.0)  # 活動度過低的 walk 時長門檻
    
    # 行為統計：四種行為完全獨立
    BEHAVIOR_CATEGORIES = {
        0: "walk",
        1: "lick",
        2: "scratch",
        3: "shake",
        4: "stop",
    }
    

# ==================== 貓咪身分（單一貓咪，固定 ID） ====================
class CatIdentityConfig:
    """
    本系統目前僅支援單一貓咪偵測，不做多貓身分辨識/re-ID（見 0_進度彙整.md）。
    個體化基線的前提是「同一份紀錄都來自同一隻貓」——這裡用一個固定的 CAT_ID
    把這個假設明確標記在每一筆 log／基線資料上，取代先前未強制的隱含假設；
    未來若要支援多貓，也有現成欄位可以擴充成真正依偵測結果變化的 ID。
    """
    CAT_ID = _env_str("CAT_MONITORING_CAT_ID", "cat_001")


# ==================== CSV 日誌參數 ====================
class LoggingConfig:
    """日誌記錄設置"""
    
    # Tracker 狀態持久化路徑（重啟後恢復當日累積資料）
    TRACKER_STATE_PATH = _env_str("CAT_MONITORING_TRACKER_STATE_PATH", r"C:\a\tracker_state.json")

    # CSV 絕對路徑（可由環境變數覆寫）
    CSV_PATH = _env_str("CAT_MONITORING_CSV_PATH", r"C:\ai_project\paper\cat_monitoring_log.csv")
    # 行為區段 CSV（BehaviorSegmentLogger）路徑 — 獨立檔案，避免與 CSV_PATH 混寫
    SEGMENTS_CSV_PATH = _env_str("CAT_MONITORING_SEGMENTS_CSV_PATH", r"C:\ai_project\paper\behavior_segments_log.csv")

# ==================== 顯示和視覺化參數 ====================
# 骨架/文字顏色與字型等實際繪圖參數已改由 utils/constants.py 與
# processors/visualizer.py 內部管理（此處舊有的同名設定從未被讀取），
# 故僅保留仍會影響串流行為的參數。
class VisualizationConfig:
    """串流輸出參數"""

    # STREAM_DISPLAY_SIZE: None 代表維持原始解析度；(寬, 高) 例如 (480, 480) 代表先縮小再編碼，降低頻寬但犧牲畫質。
    STREAM_DISPLAY_SIZE = _env_size("CAT_MONITORING_STREAM_DISPLAY_SIZE", None)

    # CLIP_SECONDS: /video_clip 保留的 ring buffer 秒數；記憶體佔用會隨這個值線性增加，但不會因長時間運行持續暴增。
    CLIP_SECONDS = _env_int("CAT_MONITORING_CLIP_SECONDS", 5)

    # SHOW_NOSE_TRAPEZOID: True = 在串流畫面上繪製鼻子接觸梯形 overlay（lick_stage plugin 專用）
    # 設為 False 可在不移除 plugin 的情況下完全隱藏此視覺效果。
    SHOW_NOSE_TRAPEZOID = _env_bool("CAT_MONITORING_SHOW_NOSE_TRAPEZOID", True)

# ==================== 系統識別 ====================
class SystemInfo:
    """系統識別和版本信息"""
    
    SYSTEM_NAME = "Cat Health Monitoring System"
    VERSION = "v4.0-stgcn"
    MODEL_TYPE = "YOLO-Pose + ST-GCN"
    
    # 幀尺寸（None = 使用原始尺寸）
    OUTPUT_WIDTH = 640
    OUTPUT_HEIGHT = 640

# ==================== 便利函數 ====================
def get_config_summary():
    """取得配置摘要"""
    # STGCNTrainingConfig 讀取 stgcn_config.yaml；找不到檔案/格式錯誤時
    # is_available() 為 False，下面顯示用的值全部會是 None，不會拋例外。
    _train_available = STGCNTrainingConfig.is_available()
    _train_seq_len = STGCNTrainingConfig.get("SEQUENCE_LENGTH")
    _train_num_joints = STGCNTrainingConfig.get("NUM_JOINTS")
    _train_window_stride = STGCNTrainingConfig.get("WINDOW_STRIDE")
    _train_spatial_kernel = STGCNTrainingConfig.get("SPATIAL_KERNEL_SIZE")
    _train_temporal_kernel = STGCNTrainingConfig.get("TEMPORAL_KERNEL_SIZE")
    _train_num_layers = STGCNTrainingConfig.get("NUM_STGCN_LAYERS")
    _train_feature_mode = STGCNTrainingConfig.get("FEATURE_MODE")
    _train_num_classes = STGCNTrainingConfig.get("NUM_CLASSES")
    _train_use_attention = STGCNTrainingConfig.get("USE_ATTENTION")
    _train_behavior_prefixes = STGCNTrainingConfig.get("BEHAVIOR_PREFIXES")

    # SEQUENCE_LENGTH/FEATURE_MODE 這兩項推論時「不會」依 checkpoint 自動偵測
    # 覆寫（跟 in_channels/num_joints/attention 不同，見 models/stgcn_model.py），
    # 訓練/推論兩邊不一致會是安靜的 bug，這裡直接標記出來提醒使用者。
    _seq_len_match = (
        "—" if _train_seq_len is None
        else ("✓" if _train_seq_len == STGCNConfig.SEQUENCE_LENGTH else "⚠ 不一致！")
    )
    _feature_mode_match = (
        "—" if _train_feature_mode is None
        else ("✓" if _normalize_feature_mode(_train_feature_mode) == STGCNConfig.FEATURE_MODE else "⚠ 不一致！")
    )

    summary = f"""
    ╔════════════════════════════════════════════════════════╗
    ║          貓咪監測系統配置摘要                         ║
    ╚════════════════════════════════════════════════════════╝

    📋 系統資訊  (硬編碼於 SystemInfo 類別: SYSTEM_NAME, VERSION, MODEL_TYPE, OUTPUT_WIDTH/HEIGHT；無 env 覆寫)
      - 名稱    : {SystemInfo.SYSTEM_NAME}
      - 版本    : {SystemInfo.VERSION}
      - 模型    : {SystemInfo.MODEL_TYPE}
      - 輸出尺寸: {SystemInfo.OUTPUT_WIDTH} × {SystemInfo.OUTPUT_HEIGHT}

    📷 YOLO 參數
      - 圖像尺寸          : {YOLOConfig.IMAGE_SIZE}
      - 偵測信心閾值      : {YOLOConfig.CONFIDENCE_THRESHOLD}

    🧠 ST-GCN 參數  (硬編碼於 STGCNConfig.NUM_CLASSES；無 env 覆寫)
      - 時間窗長度 (T)    : {STGCNConfig.SEQUENCE_LENGTH} 幀
      - 行為類別數        : {STGCNConfig.NUM_CLASSES}
      - 特徵模式          : {STGCNConfig.FEATURE_MODE}  (實際輸入通道數依 checkpoint 自動偵測，見 stgcn_model.py)
      - 推論滑動步長      : {STGCNConfig.WINDOW_STRIDE} 幀/次  (CLASSIFY_STRIDE，訓練步長由 stgcn_config.yaml 管理)
      - 目標模型 FPS      : {STGCNConfig.TARGET_MODEL_FPS}
      - FPS 降採樣        : {STGCNConfig.ENABLE_FPS_DOWNSAMPLE}
      - 關鍵點 EMA α      : {STGCNConfig.KP_EMA_ALPHA}  (1.0=不平滑)

    🎓 ST-GCN 訓練設定  (唯一權威來源: {STGCNTrainingConfig._PATH}；讀取{"成功" if _train_available else "失敗，以下皆為 None"})
      - 序列長度 (訓練)   : {_train_seq_len} 幀   [跟推論端 SEQUENCE_LENGTH 是否一致: {_seq_len_match}]
      - 特徵模式 (訓練)   : {_train_feature_mode}   [跟推論端 FEATURE_MODE 是否一致: {_feature_mode_match}]
      - 關節數 (NUM_JOINTS): {_train_num_joints}  (17=完整骨架, 14=排除尾巴三點；實際推論時由 checkpoint 自動偵測，不受這裡影響)
      - 訓練滑動步長      : {_train_window_stride} 幀  (跟上面推論滑動步長是不同概念，無需一致)
      - 空間 kernel 大小  : {_train_spatial_kernel}
      - 時間 kernel 大小  : {_train_temporal_kernel}
      - ST-GCN block 層數 : {_train_num_layers}
      - 行為類別數 (訓練) : {_train_num_classes}
      - 是否啟用 Attention: {_train_use_attention}
      - 行為前綴對應      : {_train_behavior_prefixes}

    🛑 靜止偵測（滾動均值閾值，純 CSV 記錄；單位 body_fraction×100）
      - 最大動作值        : {AnomalyDetectionConfig.MAX_MOTION}  （body_fraction×100；走路約 10-20）
      - 關鍵點信心門檻    : {AnomalyDetectionConfig.KP_CONF_THRES}
      - 滾動視窗大小      : {AnomalyDetectionConfig.ROLLING_WINDOW_SIZE} 幀
      - 靜止動作門檻      : {AnomalyDetectionConfig.STILL_MOTION_THRESHOLD}  （body_fraction×100；呼吸抖動約 < 2）

    🔬 Skeleton Quality Assessment（骨架品質雙重判定）
      - 總開關 ENABLE_SQA_DUAL_JUDGMENT: {SQAConfig.ENABLE_SQA_DUAL_JUDGMENT}
        （個別指標開關在 cat_monitoring_system/processors/skeleton_quality_assessment.py 模組內設定，此處不重複顯示）

    🕐 執行模式與排程
      - 執行模式          : {RunModeConfig.MODE}  ("server" 或 "gui")
      - 啟動即自動處理    : {RunModeConfig.AUTO_START_PROCESSING}  （False=等第一個 /stream 等請求才啟動處理管線）
      - 排程開始時間      : {RunModeConfig.SCHEDULED_START_TIME or "(未設定)"}
      - 排程結束時間      : {RunModeConfig.SCHEDULED_END_TIME or "(未設定，開始後永遠運行)"}
      - 目前是否在排程區間內: {RunModeConfig.is_within_active_window()}  （即時判斷，印出當下這一刻的狀態）

    🏷️ 行為追蹤門檻
      - ST-GCN 行為標籤門檻  : {BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD}
      - 最短記錄時長         : {BehaviorTrackingConfig.MIN_RECORD_DURATION_SECONDS} s
      - 活動分數時間窗        : {BehaviorTrackingConfig.ACTIVITY_SCORE_WINDOW_SECONDS} s
      - 活動力窗口大小        : {BehaviorTrackingConfig.ACTIVITY_WINDOW_SIZE} 幀
      - 行為歷史保留筆數      : {BehaviorTrackingConfig.MAX_HISTORY_SIZE} 筆
      - 低信心幀活動權重      : {BehaviorTrackingConfig.LOW_CONFIDENCE_ACTIVITY_WEIGHT}
      - 貓消失容忍幀數        : {BehaviorTrackingConfig.CAT_MISSING_TOLERANCE_FRAMES} 幀  （<=0 等同關閉此機制）
      - 顯示延遲窗口數(walk/lick/scratch/shake/stop):
          {BehaviorTrackingConfig.DISPLAY_HYSTERESIS_WINDOWS_WALK}/{BehaviorTrackingConfig.DISPLAY_HYSTERESIS_WINDOWS_LICK}/{BehaviorTrackingConfig.DISPLAY_HYSTERESIS_WINDOWS_SCRATCH}/{BehaviorTrackingConfig.DISPLAY_HYSTERESIS_WINDOWS_SHAKE}/{BehaviorTrackingConfig.DISPLAY_HYSTERESIS_WINDOWS_STOP}
      - 行為類別對照          : {BehaviorTrackingConfig.BEHAVIOR_CATEGORIES}
      - 搔抓警報秒數          : {BehaviorTrackingConfig.SCRATCH_ALERT_TIME_SECONDS} s
      - 搔抓警報次數          : {BehaviorTrackingConfig.SCRATCH_ALERT_COUNT_THRESHOLD} 次
      - 舔舐警報秒數          : {BehaviorTrackingConfig.LICK_ALERT_TIME_SECONDS} s
      - 甩頭警報次數          : {BehaviorTrackingConfig.SHAKE_ALERT_COUNT_THRESHOLD} 次
      - 靜止警報秒數          : {BehaviorTrackingConfig.STOP_ALERT_TIME_SECONDS} s
      - 低活動 walk 門檻      : {BehaviorTrackingConfig.LOW_ACTIVITY_TIME_THRESHOLD_SECONDS} s

    🐱 貓咪身分  (單一貓咪固定 ID，本系統不做多貓 re-ID)
      - CAT_ID            : {CatIdentityConfig.CAT_ID}

    🌐 Flask 服務
      - 主機        : {FlaskConfig.HOST}:{FlaskConfig.PORT}
      - JPEG 品質   : {FlaskConfig.JPEG_QUALITY}
      - Debug 模式  : {FlaskConfig.DEBUG}
      - Threaded    : {FlaskConfig.THREADED}

    🎞️ 串流視覺化
      - 串流縮放尺寸      : {VisualizationConfig.STREAM_DISPLAY_SIZE}
      - Ring Buffer 秒數  : {VisualizationConfig.CLIP_SECONDS} s
      - 鼻子梯形 overlay  : {VisualizationConfig.SHOW_NOSE_TRAPEZOID}

    🔗 Node-RED 連線
      - 主機        : {NodeRedConfig.HOST}:{NodeRedConfig.PORT}
      - 推送間隔    : {NodeRedConfig.PUSH_INTERVAL} s
      - 超時        : {NodeRedConfig.TIMEOUT} s
      - Notify 端點   : {NodeRedConfig.ENDPOINT_NOTIFY}
      - Result v1 端點: {NodeRedConfig.ENDPOINT_RESULT}
      - Result v2 端點: {NodeRedConfig.ENDPOINT_RESULT_V2}

    📄 日誌設定
      - 主要 CSV          : {LoggingConfig.CSV_PATH}
      - 行為區段 CSV      : {LoggingConfig.SEGMENTS_CSV_PATH}
      - Tracker 狀態檔    : {LoggingConfig.TRACKER_STATE_PATH}  （重啟後恢復當日累積資料）

    📁 路徑配置
      - YOLO 模型   : {ModelPaths.YOLO_MODEL}
      - ST-GCN 模型 : {ModelPaths.STGCN_MODEL}
      - 輸入視訊    : {ModelPaths.VIDEO_INPUT}
      - 日誌目錄    : {ModelPaths.LOG_DIR}
      - 輸出目錄    : {ModelPaths.OUTPUT_DIR}

    ╔════════════════════════════════════════════════════════╗
    """
    return summary

def validate_all_config():
    """驗證所有配置"""
    print("🔍 驗證配置...")
    
    def _validate_runtime_values():
        errors = []
        if not _is_valid_port(FlaskConfig.PORT):
            errors.append(f"Flask PORT 無效: {FlaskConfig.PORT}")
        if not _is_valid_port(NodeRedConfig.PORT):
            errors.append(f"Node-RED PORT 無效: {NodeRedConfig.PORT}")
        if not (0.0 <= YOLOConfig.CONFIDENCE_THRESHOLD <= 1.0):
            errors.append(f"YOLO CONFIDENCE_THRESHOLD 應在 [0,1]: {YOLOConfig.CONFIDENCE_THRESHOLD}")
        if STGCNConfig.SEQUENCE_LENGTH <= 0:
            errors.append(f"ST-GCN SEQUENCE_LENGTH 必須 > 0: {STGCNConfig.SEQUENCE_LENGTH}")
        if STGCNConfig.TARGET_MODEL_FPS <= 0:
            errors.append(f"ST-GCN TARGET_MODEL_FPS 必須 > 0: {STGCNConfig.TARGET_MODEL_FPS}")
        if not (0.0 < STGCNConfig.KP_EMA_ALPHA <= 1.0):
            errors.append(f"KP_EMA_ALPHA 應在 (0,1]: {STGCNConfig.KP_EMA_ALPHA}")
        if VisualizationConfig.STREAM_DISPLAY_SIZE is not None:
            stream_size = VisualizationConfig.STREAM_DISPLAY_SIZE
            valid_stream_size = (
                isinstance(stream_size, tuple)
                and len(stream_size) == 2
                and all(isinstance(value, int) and value > 0 for value in stream_size)
            )
            if not valid_stream_size:
                errors.append(f"STREAM_DISPLAY_SIZE 必須是 (寬, 高) 且都 > 0: {VisualizationConfig.STREAM_DISPLAY_SIZE}")
        if FlaskConfig.JPEG_QUALITY < 1 or FlaskConfig.JPEG_QUALITY > 100:
            errors.append(f"JPEG_QUALITY 應在 [1,100]: {FlaskConfig.JPEG_QUALITY}")
        if NodeRedConfig.TIMEOUT <= 0:
            errors.append(f"Node-RED TIMEOUT 必須 > 0: {NodeRedConfig.TIMEOUT}")

        if errors:
            print("  ✗ 參數範圍檢查")
            for err in errors:
                print(f"    - {err}")
            return False
        return True

    def _validate_train_inference_consistency():
        """SEQUENCE_LENGTH/FEATURE_MODE 這兩項推論時不會依 checkpoint 自動
        偵測覆寫（跟 in_channels/num_joints/attention 不同，見
        models/stgcn_model.py），訓練/推論兩邊不一致會是安靜的 bug——例如
        訓練時用 SEQUENCE_LENGTH=32，推論卻用 16，模型會吃到形狀對得上但
        語意錯誤的輸入，不會報錯，只會默默地推論結果不準。

        stgcn_config.yaml 讀不到（檔案不存在/PyYAML 未安裝）時視為無法比對，
        不當成錯誤——這項檢查是「有資料可比對時才擋」，不是強制要求一定要有
        這份訓練設定檔。
        """
        if not STGCNTrainingConfig.is_available():
            print("  ⚠ 找不到 stgcn_config.yaml 或讀取失敗，略過訓練/推論一致性比對")
            return True

        mismatches = []
        train_seq_len = STGCNTrainingConfig.get("SEQUENCE_LENGTH")
        if train_seq_len is not None and train_seq_len != STGCNConfig.SEQUENCE_LENGTH:
            mismatches.append(
                f"SEQUENCE_LENGTH 不一致：訓練={train_seq_len}，推論={STGCNConfig.SEQUENCE_LENGTH}"
            )

        train_feature_mode = STGCNTrainingConfig.get("FEATURE_MODE")
        if train_feature_mode is not None and _normalize_feature_mode(train_feature_mode) != STGCNConfig.FEATURE_MODE:
            mismatches.append(
                f"FEATURE_MODE 不一致：訓練={train_feature_mode}，推論={STGCNConfig.FEATURE_MODE}"
            )

        if mismatches:
            print("  ✗ 訓練/推論一致性比對")
            for m in mismatches:
                print(f"    - {m}")
            return False
        return True

    checks = [
        ("模型檔案", ModelPaths.validate),
        ("目錄結構", lambda: (ModelPaths.ensure_dirs(), True)[1]),
        ("參數範圍", _validate_runtime_values),
        ("訓練/推論一致性", _validate_train_inference_consistency),
    ]
    
    all_valid = True
    for check_name, check_func in checks:
        try:
            result = check_func()
            status = "✓" if result else "✗"
            print(f"  {status} {check_name}")
            if not result:
                all_valid = False
        except Exception as e:
            print(f"  ✗ {check_name}: {str(e)}")
            all_valid = False
    
    return all_valid


# ==================== 主測試 ====================
if __name__ == "__main__":
    print(get_config_summary())
    
    if validate_all_config():
        print("\n✅ 所有配置驗證通過！")
    else:
        print("\n⚠ 部分配置驗證失敗，請檢查。")
