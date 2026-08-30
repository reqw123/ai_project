"""
貓咪監測系統配置檔案
方便管理所有設置，避免直接修改主程序
"""

import builtins as _builtins
import datetime as _datetime
import os
import re
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    """讀取字串型環境變數；未設定或空字串時回傳 default。"""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    """讀取整數型環境變數；未設定或轉型失敗時回傳 default。"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """讀取浮點數型環境變數；未設定或轉型失敗時回傳 default。"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return _builtins.float(value)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    """讀取布林型環境變數（1/true/yes/y/on 為 True，0/false/no/n/off 為 False）；
    未設定或無法辨識時回傳 default。"""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_video_input(name: str, default: str | int) -> str | int:
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


def _env_size(name: str, default: tuple[int, int] | None) -> tuple[int, int] | None:
    """讀取尺寸設定：支援 640x480、640,480、640 480；無效時回傳預設值。"""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if not value or value in {"none", "null", "off", "false"}:
        return default

    parts = [
        part for part in value.replace("x", ",").replace(" ", ",").split(",") if part
    ]
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


def _is_valid_port(port: int) -> bool:
    """判斷是否為合法的 TCP port 號碼（1-65535）。"""
    return isinstance(port, int) and 1 <= port <= 65535


_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def _parse_hhmm(value: str) -> tuple[int, int] | None:
    """解析 'HH:MM'（24 小時制，兩位數，例如 '06:00'）字串，回傳 (hour, minute)；
    空字串/格式錯誤（含少一位數的 '6:00'、無效範圍如 '24:00'）回傳 None。

    2026-08-15：改用正則、要求兩位數小時，跟 settings_manager.py 的
    _HHMM_RE 用同一個 pattern——原本這裡只檢查數值範圍（0<=hour<=23），
    「6:00」這種少一位數的字串也算合法，但設定視窗那邊的驗證規則要求
    兩位數，兩邊標準不一致，會出現「config.py 自己執行不會擋、存進設定
    視窗卻被擋」的落差。統一成一樣嚴格，任何一邊都用同一套判斷標準。
    """
    if not value:
        return None
    match = _HHMM_RE.match(value.strip())
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)))


def _runtime_default(json_key: str, fallback, value_type=None):
    """JSON 執行期覆寫層的「預設值」來源：settings_manager.get_runtime_value() 讀
    runtime_settings.current.json，欄位缺漏/檔案不存在時直接回傳 fallback（原本寫死的
    字面值）。這個函式回傳的值只是拿去當 `_env_*()` 的 `default` 參數——最終
    優先順序仍是「環境變數（_env_* 本身的邏輯）> runtime_settings.current.json（這裡）
    > 寫死的 fallback」，`_env_*` 系列函式本身完全不變。

    刻意用 try/except 包住整個查詢：settings_manager.py 缺失、import 失敗、
    JSON 格式錯誤等任何情況都不該讓 config.py 整個 import 失敗——config.py
    被 39 個檔案 import，安全啟動比「JSON 一定要生效」更優先。
    """
    try:
        from settings_manager import get_runtime_value

        return get_runtime_value(json_key, fallback, value_type=value_type)
    except Exception as e:
        print(f"⚠ 讀取 runtime_settings.current.json 失敗（{json_key}），使用內建預設值: {e}")
        return fallback


def _runtime_default_size(json_key: str, fallback):
    """STREAM_DISPLAY_SIZE 專用版本：JSON 存 null 或 {"width":.., "height":..}。"""
    try:
        from settings_manager import get_runtime_size

        return get_runtime_size(json_key, fallback)
    except Exception as e:
        print(f"⚠ 讀取 runtime_settings.current.json 失敗（{json_key}），使用內建預設值: {e}")
        return fallback


def _normalize_feature_mode(feature_mode: str) -> str:
    """把特徵模式字串正規化為小寫、去除前後空白；空字串預設為 'xy'。"""
    mode = str(feature_mode).strip().lower()
    if not mode:
        return "xy"
    return mode


def _get_stgcn_feature_spec(feature_mode: str) -> dict:
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
            "features": [
                "x",
                "y",
                "conf",
                "vx",
                "vy",
                "bone_x",
                "bone_y",
                "bone_mx",
                "bone_my",
            ],
            "description": "位置 + 信心值 + 速度 + 骨架向量 + 骨架位移",
        },
    }
    if mode not in specs:
        raise ValueError(
            f"Unknown ST-GCN feature mode: {feature_mode!r}. 支援模式: {list(specs)}"
        )
    return specs[mode]


# ==================== 模型和資料路徑 ====================
# 專案根目錄（paper/ 的上一層）：用來組出 yolo_models/、stgcn_models/ 的預設路徑，
# 讓模型路徑不再寫死成特定電腦上的絕對路徑（例如 C:\ai_project\...、
# C:\Users\homec\Downloads\...），改成「相對於這份原始碼所在的專案」，
# 換一台電腦、換一個 clone 路徑都能直接跑，不需要每次都改 config 或設環境變數。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_project_path(path_str: str) -> str:
    """把相對路徑接到 _PROJECT_ROOT 上，絕對路徑原樣回傳。

    讓 runtime_settings.current.json／環境變數可以直接寫 "yolo_models\\v11s_143.pt"
    這種相對路徑，不受實際執行時的工作目錄（CWD）影響——不像 LOG_DIR/OUTPUT_DIR
    那樣依賴 CWD 剛好等於專案根目錄，這裡固定以原始碼所在專案為錨點解析。
    """
    path = Path(path_str)
    return str(path if path.is_absolute() else _PROJECT_ROOT / path)


class ModelPaths:
    """模型和資料檔案路徑"""

    # YOLO 模型：權重檔統一放在專案根目錄的 yolo_models/（原本直接散落在 cat_pose/ 下，
    # 沒有資料夾管理）。
    YOLO_MODEL = _resolve_project_path(
        _env_str(
            "CAT_MONITORING_YOLO_MODEL",
            _runtime_default(
                "model_paths.yolo_model",
                str(Path("yolo_models") / "v11s_147.pt"),
                value_type=str,
            ),
        )
    )

    # ST-GCN 模型：訓練結果統一放在專案根目錄的 stgcn_models/（原本在另一顆磁碟機
    # D:\stgcn_results，跨機器/跨磁碟機會直接找不到檔案）。
    STGCN_MODEL = _resolve_project_path(
        _env_str(
            "CAT_MONITORING_STGCN_MODEL",
            _runtime_default(
                "model_paths.stgcn_model",
                str(
                    Path("stgcn_models")
                    / "run_122_xy_conf_v_bone_att_on"
                    / "122_best_model.pth"
                ),
                value_type=str,
            ),
        )
    )

    # 測試視頻（可設為本機路徑、rtsp:// 串流網址、或攝影機 index）
    # 例：rtsp://12345678:456456123@192.168.0.192:554/stream1
    # "C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\泛化測試\7月18日.mp4"
    VIDEO_INPUT = _env_video_input(
        "CAT_MONITORING_VIDEO_INPUT",
        _runtime_default(
            "model_paths.video_input",
            r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\泛化測試\7月18日.mp4",
        ),
    )

    # 日誌和輸出目錄
    LOG_DIR = _env_str(
        "CAT_MONITORING_LOG_DIR", _runtime_default("model_paths.log_dir", "./logs", value_type=str)
    )
    OUTPUT_DIR = _env_str(
        "CAT_MONITORING_OUTPUT_DIR",
        _runtime_default("model_paths.output_dir", "./output", value_type=str),
    )

    @classmethod
    def ensure_dirs(cls) -> None:
        """確保日誌與輸出目錄存在，不存在則建立。"""
        Path(cls.LOG_DIR).mkdir(exist_ok=True)
        Path(cls.OUTPUT_DIR).mkdir(exist_ok=True)

    @classmethod
    def validate(cls) -> bool:
        """驗證模型檔案與視訊來源是否存在，缺少時印出清單並回傳 False。"""
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
            for missing_entry in missing:
                print(f"  - {missing_entry}")
            return False

        return True


# ==================== YOLO 參數 ====================
class YOLOConfig:
    """YOLO 檢測參數

    裝置選擇不在此設定：實際執行時由 routes.py 的 _resolve_runtime_device() 自動偵測 CUDA 可用性。
    """

    # 推論參數
    IMAGE_SIZE = _env_int(
        "CAT_MONITORING_YOLO_IMAGE_SIZE",
        _runtime_default("yolo.image_size", 640, value_type=int),
    )
    CONFIDENCE_THRESHOLD = _env_float(
        "CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD",
        _runtime_default("yolo.confidence_threshold", 0.50, value_type=float),
    )


# ==================== ST-GCN 參數 ====================
class STGCNConfig:
    """ST-GCN 模型參數"""

    # 模型超參數
    SEQUENCE_LENGTH = _env_int(
        "CAT_MONITORING_STGCN_SEQUENCE_LENGTH", 16
    )  # 時間窗長度（幀數）
    NUM_CLASSES = 5  # 行為類別數

    # 特徵模式（與 train_gcn.py / 推論腳本共用概念）
    # 預設值須與 ModelPaths.STGCN_MODEL 預設 checkpoint（run_122_xy_conf_v_bone_att_on/122_best_model.pth）一致；
    # 實際推論時 in_channels/attention 仍會依 checkpoint 內容自動偵測並覆寫此設定（見 stgcn_model.py）。
    FEATURE_MODE = _normalize_feature_mode(
        _env_str("CAT_MONITORING_STGCN_FEATURE_MODE", "xy_conf_v_bone")
    )
    _get_stgcn_feature_spec(
        FEATURE_MODE
    )  # 僅用於在載入時驗證 FEATURE_MODE 合法，實際通道數由 checkpoint 自動偵測

    # 推論用滑動步長（每幾幀執行一次 ST-GCN，對應 CLASSIFY_STRIDE）
    # 訓練用的步長由 stgcn_config.yaml 的 WINDOW_STRIDE 管理，與此無關
    WINDOW_STRIDE = _env_int(
        "CAT_MONITORING_STGCN_WINDOW_STRIDE",
        _runtime_default("stgcn_runtime.window_stride", 2, value_type=int),
    )

    # 裝置選擇不在此設定：實際執行時由 routes.py 的 _resolve_runtime_device() 自動偵測 CUDA 可用性。

    # FPS 同步：對來源影片做降採樣，使模型輸入時基符合訓練設定。
    # TARGET_MODEL_FPS: 推論與串流使用的目標 FPS，須與訓練時一致。
    #   調低（例如 15）可降低 YOLO + ST-GCN 推論頻率，減少 CPU/GPU 負擔，但反應會變慢。
    TARGET_MODEL_FPS = _env_float(
        "CAT_MONITORING_TARGET_MODEL_FPS",
        _runtime_default("stgcn_runtime.target_model_fps", 30.0, value_type=float),
    )
    # ENABLE_FPS_DOWNSAMPLE: True 代表來源 FPS 高於 TARGET_MODEL_FPS 時自動跳幀；False 代表每幀都處理。
    ENABLE_FPS_DOWNSAMPLE = _env_bool(
        "CAT_MONITORING_ENABLE_FPS_DOWNSAMPLE",
        _runtime_default("stgcn_runtime.enable_fps_downsample", True, value_type=bool),
    )

    # 關鍵點 EMA 平滑（須與 train_gcn.py 的 KP_EMA_ALPHA 保持一致）
    # alpha 越大 → 越貼近原始偵測值；alpha 越小 → 越平滑但延遲增加
    KP_EMA_ALPHA = _env_float(
        "CAT_MONITORING_KP_EMA_ALPHA",
        _runtime_default("stgcn_runtime.kp_ema_alpha", 1.0, value_type=float),
    )
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

    _DEFAULT_PATH = str(
        Path(__file__).parent / "cat_monitoring_system" / "stgcn_config.yaml"
    )
    _PATH = _env_str("STGCN_CONFIG_PATH", _DEFAULT_PATH)
    _cache = None
    _load_error = None

    @classmethod
    def _load(cls) -> dict:
        """讀取並快取 stgcn_config.yaml 內容；讀取失敗時回傳空 dict。"""
        if cls._cache is not None:
            return cls._cache
        try:
            import yaml

            with open(cls._PATH, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
            if not isinstance(yaml_data, dict):
                raise ValueError(f"{cls._PATH} 內容不是一個 mapping/object")
            cls._cache = yaml_data
        except Exception as e:
            cls._load_error = str(e)
            cls._cache = {}
        return cls._cache

    @classmethod
    def get(cls, key: str, default=None):
        """讀取訓練設定中的單一欄位；檔案不存在或無此鍵時回傳 default。"""
        return cls._load().get(key, default)

    @classmethod
    def is_available(cls) -> bool:
        """True 代表成功讀到檔案內容；False 代表檔案不存在/格式錯誤/PyYAML
        未安裝——呼叫端可用這個判斷要不要顯示「讀取失敗」提示。"""
        cls._load()
        return cls._load_error is None



# ==================== 執行模式參數 ====================
class RunModeConfig:
    """控制 main.py 啟動時走哪一種模式，整體處理架構（FrameProcessor 等）不受影響。

    "server"（預設）：現行行為，啟動 Flask HTTP 伺服器 + Node-RED 上線通知
    "gui"           ：不啟動 Flask/Node-RED，直接用同一套 FrameProcessor 開本地視窗顯示
    """

    MODE = _env_str(
        "CAT_MONITORING_RUN_MODE", _runtime_default("run_mode.mode", "server", value_type=str)
    )

    # server 模式下，處理管線（開影片、載入 YOLO/ST-GCN、tracker 統計、CSV、Node-RED 推送）
    # 原本要等第一個打到 /stream 等路由的 HTTP 請求才會啟動（見 routes.py 的
    # _ensure_processor_started()），也就是實務上要有人打開 Dashboard 點播放才會真正開始跑。
    # 預錄影片、排程無人值守執行時沒人會去點播放，關掉這個延遲啟動，改成 Python 進程一啟動
    # 就立刻開始處理，不等任何請求。設為 False 可還原成原本「有人連線才啟動」的行為。
    AUTO_START_PROCESSING = _env_bool(
        "CAT_MONITORING_AUTO_START_PROCESSING",
        _runtime_default("run_mode.auto_start_processing", True, value_type=bool),
    )

    # 排程啟動時間（24 小時制 "HH:MM"，例如 "06:00"）。設定後，處理管線在 Python 進程
    # 啟動的當下不會立刻執行，而是持續等待直到真實世界時間到達此時刻才開始跑；
    # 若進程啟動時已經過了這個時刻（例如排程 06:00 但 14:00 才啟動），視為時間已到，
    # 立即開始，不會傻等到隔天。留空（預設）代表不啟用排程，沿用 AUTO_START_PROCESSING
    # 的行為（一啟動就跑或永遠不自動跑）。用於預錄影片、無人值守的排程執行情境
    # （例如固定每天啟動一次，只想在 06:00 才開始處理當天份的影片）。
    # 2026-08-29：預設改為空字串（不啟用排程）。先前預設 "06:00"~"12:00" 會讓
    # 開箱即用的系統在 12:00–06:00 之間對 /stream 等端點一律回 503，使用者曾
    # 因忘記自己沒設過而誤判系統故障（見 server/routes.py::_schedule_unavailable_reason）。
    SCHEDULED_START_TIME = _env_str(
        "CAT_MONITORING_SCHEDULED_START_TIME",
        _runtime_default("run_mode.scheduled_start_time", "", value_type=str),
    )  # 預設 "" = 不啟用排程
    SCHEDULED_START_HHMM = _parse_hhmm(SCHEDULED_START_TIME)

    # 排程結束時間（24 小時制 "HH:MM"，例如 "12:00"）。留空（預設）＝不設結束時間，
    # 只要 SCHEDULED_START_TIME 有設，一旦時間到就開始處理、之後永遠不會自動停止
    # （對應「排程時間到、之後一直運行」的用法）。
    # 若同時設定了開始與結束時間，就變成「區段執行」：只有現在時刻落在
    # [開始, 結束) 之間才處理，區間外自動暫停，且每一天都會依同一組 HH:MM 重新套用
    # （不需要重啟 Python，也不會重新載入模型——暫停只是不讀取/不推論，不釋放資源）。
    # 若只設結束、沒設開始，開始時間視為當天 00:00。
    SCHEDULED_END_TIME = _env_str(
        "CAT_MONITORING_SCHEDULED_END_TIME",
        _runtime_default("run_mode.scheduled_end_time", "", value_type=str),
    )  # 預設 "" = 不設結束時間
    SCHEDULED_END_HHMM = _parse_hhmm(SCHEDULED_END_TIME)

    @classmethod
    def is_within_active_window(cls, now: _datetime.datetime | None = None) -> bool:
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
            start_dt = now.replace(
                hour=start_hhmm[0], minute=start_hhmm[1], second=0, microsecond=0
            )
            return now >= start_dt

        start_h, start_m = start_hhmm if start_hhmm is not None else (0, 0)
        start_dt = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        end_dt = now.replace(
            hour=end_hhmm[0], minute=end_hhmm[1], second=0, microsecond=0
        )
        if end_dt <= start_dt:
            return now >= start_dt or now < end_dt
        return start_dt <= now < end_dt


# ==================== Flask 服務參數 ====================
class FlaskConfig:
    """Flask Web 服務參數"""

    HOST = _env_str(
        "CAT_MONITORING_FLASK_HOST", _runtime_default("flask.host", "0.0.0.0", value_type=str)
    )
    PORT = _env_int(
        "CAT_MONITORING_FLASK_PORT", _runtime_default("flask.port", 5000, value_type=int)
    )
    DEBUG = _env_bool(
        "CAT_MONITORING_FLASK_DEBUG", _runtime_default("flask.debug", False, value_type=bool)
    )
    THREADED = _env_bool(
        "CAT_MONITORING_FLASK_THREADED",
        _runtime_default("flask.threaded", True, value_type=bool),
    )

    # JPEG 壓縮品質 (1-100)
    JPEG_QUALITY = _env_int(
        "CAT_MONITORING_JPEG_QUALITY",
        _runtime_default("flask.jpeg_quality", 30, value_type=int),
    )


# ==================== 個體化基線儀表板（Python 端唯讀展示頁）====================
class BaselineDashboardConfig:
    """新版個體化基線引擎（analytics/）的唯讀展示頁面設定。

    ENABLED 關閉時，server/routes.py、server/flask_app.py 都不會匯入
    dashboard/ 模組本身——不是路由回 404，是整個模組不被載入，不佔用任何
    記憶體或處理資源。這個頁面只讀 analytics/ 算好的結果，不讀 Node-RED
    的 global.json，也不做新舊引擎比對（那個已經有 analytics_deviation_bridge.json
    的「🔬 基線引擎比對」可以看）。
    """

    ENABLED = _env_bool(
        "CAT_MONITORING_BASELINE_DASHBOARD_ENABLED",
        _runtime_default("flask.baseline_dashboard_enabled", True, value_type=bool),
    )

    # 前端輪詢 /api/deviation/latest 的間隔（秒）。比照 NodeRedConfig.PUSH_INTERVAL
    # （Python 端每筆推論結果送出的間隔，預設 2 秒）訂一個略慢的節奏，避免每次
    # 輪詢都撲空、也不會累積沒必要的請求量。
    POLL_INTERVAL_SEC = _env_float(
        "CAT_MONITORING_BASELINE_DASHBOARD_POLL_INTERVAL",
        _runtime_default("flask.baseline_dashboard_poll_interval_sec", 3, value_type=float),
    )

    # dashboard/refresher.py 背景執行緒重新計算一次基線/偏差/融合結果的間隔（秒）。
    # 這是這個頁面「跟 Node-RED 毫無相干」的關鍵——POST /api/deviation 只在 Node-RED
    # 主動呼叫時才會算一次（那是給新舊引擎比對用的，見 analytics_deviation_bridge.json），
    # 這裡是同一個 process 內直接呼叫 analytics/ 算的獨立排程，不透過 HTTP、不依賴
    # Node-RED 有沒有開。見 docs/資料層架構現況與統一管理評估.md 第十節。
    #
    # 預設值刻意「跟隨」NodeRedConfig.PUSH_INTERVAL（Python 推送推論結果給
    # Node-RED 的間隔，預設 2 秒）——讀同一個環境變數
    # CAT_MONITORING_NODERED_PUSH_INTERVAL 當 fallback 預設值，不是寫死複製
    # 一份數字：改那個環境變數，這裡的預設節奏會一起變，兩邊「感覺一樣即時」。
    # NodeRedConfig 定義在這個 class 後面（第 510 行），沒辦法在這裡直接寫
    # `NodeRedConfig.PUSH_INTERVAL`（class body 執行時那個名字還不存在），
    # 所以用共用環境變數名稱做間接參照，而不是搬動 class 定義順序。
    # 想讓儀表板用跟推論推送不同的節奏，設 CAT_MONITORING_BASELINE_DASHBOARD_RECOMPUTE_INTERVAL
    # 即可獨立覆寫，不受 PUSH_INTERVAL 影響。
    #
    # 注意：compute_baseline/compute_deviation/compute_fusion 都是純 Python
    # 統計運算（不是 ML 推論），對預設 <=30 天的資料量而言，每 2 秒算一次的
    # CPU 成本可忽略；daily_store.load_history() 也只是一次輕量 SQLite 讀取。
    RECOMPUTE_INTERVAL_SEC = _env_float(
        "CAT_MONITORING_BASELINE_DASHBOARD_RECOMPUTE_INTERVAL",
        _runtime_default(
            "flask.baseline_dashboard_recompute_interval_sec",
            _env_float(
                "CAT_MONITORING_NODERED_PUSH_INTERVAL",
                _runtime_default("nodered.push_interval", 2, value_type=float),
            ),
            value_type=float,
        ),
    )


# ==================== Node-RED 參數 ====================
class NodeRedConfig:
    """Node-RED 通訊參數"""

    HOST = _env_str(
        "CAT_MONITORING_NODERED_HOST",
        _runtime_default("nodered.host", "127.0.0.1", value_type=str),
    )
    PORT = _env_int(
        "CAT_MONITORING_NODERED_PORT", _runtime_default("nodered.port", 1880, value_type=int)
    )

    # 推送間隔（秒）
    PUSH_INTERVAL = _env_float(
        "CAT_MONITORING_NODERED_PUSH_INTERVAL",
        _runtime_default("nodered.push_interval", 2, value_type=float),
    )

    # 端點預設由上面的 HOST/PORT（已含 env/JSON 覆寫結果）推導；仍可個別用環境變數
    # 或 runtime_settings.current.json 的 advanced.* 覆寫成完全不同的網址。
    ENDPOINT_NOTIFY = _env_str(
        "CAT_MONITORING_NODERED_ENDPOINT_NOTIFY",
        _runtime_default(
            "advanced.nodered_endpoint_notify", f"http://{HOST}:{PORT}/python_online", value_type=str
        ),
    )
    ENDPOINT_RESULT = _env_str(
        "CAT_MONITORING_NODERED_ENDPOINT_RESULT",
        _runtime_default(
            "advanced.nodered_endpoint_result", f"http://{HOST}:{PORT}/yolo_result", value_type=str
        ),
    )
    ENDPOINT_RESULT_V2 = _env_str(
        "CAT_MONITORING_NODERED_ENDPOINT_RESULT_V2",
        _runtime_default(
            "advanced.nodered_endpoint_result_v2",
            f"http://{HOST}:{PORT}/yolo_result_v2",
            value_type=str,
        ),
    )

    # 超時時間（秒）
    TIMEOUT = _env_float(
        "CAT_MONITORING_NODERED_TIMEOUT",
        _runtime_default("nodered.timeout", 2, value_type=float),
    )

    # 個體化基線共用儲存（原本是 Node-RED 內建 file-scoped context store；
    # 2026-08-10 起改為 settings.js functionGlobalContext.gfile 手動用 fs
    # 讀寫的固定路徑，Python 端與 Node-RED function node 都指向這裡）。
    # 存放 v2_daily_history / v2_today / v2_baseline 等個體化基線相關資料。
    # 只記錄路徑，尚未讀取／解析——之後 Python 端需要直接讀取這份資料時，
    # 由此常數取得路徑，避免路徑散落各處造成不一致（對照 tracker_state.json
    # 過去在 config.py 跟 Node-RED function node 裡各自寫死一份路徑的教訓）。
    # 2026-08-26：一度刻意「沒有」跟著 TRACKER_STATE_PATH/DAILY_HISTORY_DB_PATH 一起
    # 搬進 baseline_data/——真正的即時寫入者是 Node-RED（settings.js 的
    # functionGlobalContext.gfile 寫死路徑，不受這個 git 專案版本控制），當時只改
    # Python 端會讓兩邊分岔（Node-RED 繼續寫舊路徑，Python 讀到的變成搬遷當下凍結
    # 的快照）。
    # 2026-08-27：改用環境變數 CAT_MONITORING_NODERED_GLOBAL_CONTEXT_PATH 同時驅動
    # 兩邊——settings.js 的 gfile 也改成讀同一個環境變數（process.env，不是 Node-RED
    # 的「全域環境變數」面板，那個存在 flows.json 的 global-config 節點，settings.js
    # 啟動時還沒載入，用不到），設定這一個環境變數即可讓 Python／Node-RED 同時指向
    # paper/baseline_data/global.json。這裡的 fallback 刻意維持 C:\a\global.json
    # （沒有跟著改成新路徑）——settings.js 那邊的 fallback 也是同一個舊路徑，兩邊
    # fallback 保持一致，這樣即使環境變數還沒生效（例如剛設定、尚未登出重登），
    # 兩邊還是會一致地退回舊路徑，不會其中一邊已經切新路徑、另一邊還在舊路徑造成
    # 分岔。
    # 見 docs/資料層架構現況與統一管理評估.md 第十五節。
    GLOBAL_CONTEXT_PATH = _env_str(
        "CAT_MONITORING_NODERED_GLOBAL_CONTEXT_PATH",
        _runtime_default("nodered.global_context_path", r"C:\a\global.json", value_type=str),
    )


# ==================== 顯示和視覺化參數 ====================
# 骨架/文字顏色與字型等實際繪圖參數已改由 utils/constants.py 與
# processors/visualizer.py 內部管理（此處舊有的同名設定從未被讀取），
# 故僅保留仍會影響串流行為的參數。
class VisualizationConfig:
    """串流輸出參數"""

    # STREAM_DISPLAY_SIZE: None 代表維持原始解析度；(寬, 高) 例如 (480, 480) 代表先縮小再編碼，降低頻寬但犧牲畫質。
    STREAM_DISPLAY_SIZE = _env_size(
        "CAT_MONITORING_STREAM_DISPLAY_SIZE",
        _runtime_default_size("visualization.stream_display_size", None),
    )

    # CLIP_SECONDS: /video_clip 保留的 ring buffer 秒數；記憶體佔用會隨這個值線性增加，但不會因長時間運行持續暴增。
    CLIP_SECONDS = _env_int(
        "CAT_MONITORING_CLIP_SECONDS",
        _runtime_default("visualization.clip_seconds", 5, value_type=int),
    )

    # SHOW_NOSE_TRAPEZOID: True = 在串流畫面上繪製鼻子接觸梯形 overlay（lick_stage plugin 專用）
    # 設為 False 可在不移除 plugin 的情況下完全隱藏此視覺效果。
    SHOW_NOSE_TRAPEZOID = _env_bool(
        "CAT_MONITORING_SHOW_NOSE_TRAPEZOID",
        _runtime_default("visualization.show_nose_trapezoid", True, value_type=bool),
    )

    # 以下三個是 FrameProcessor.show_skeleton/show_label/show_bbox 的「啟動預設值」，
    # 決定 Python 進程一啟動時畫面上預設要不要畫這些疊圖；啟動後仍可透過 GUI 模式
    # 鍵盤 s/l/b 或 server 模式 /api/overlay 在執行期即時切換，不受這裡影響
    # （這裡只決定重啟後、切回來時的初始狀態）。
    SHOW_BBOX = _env_bool(
        "CAT_MONITORING_SHOW_BBOX",
        _runtime_default("visualization.show_bbox", True, value_type=bool),
    )  # 是否繪製 YOLO 偵測框(bbox)與信心值文字
    SHOW_SKELETON = _env_bool(
        "CAT_MONITORING_SHOW_SKELETON",
        _runtime_default("visualization.show_skeleton", True, value_type=bool),
    )  # 是否繪製骨架關鍵點與連線
    SHOW_GCN_RESULT = _env_bool(
        "CAT_MONITORING_SHOW_GCN_RESULT",
        _runtime_default("visualization.show_gcn_result", True, value_type=bool),
    )  # 是否顯示 ST-GCN 行為分類結果與機率條


# ==================== 異常檢測參數 ====================
class AnomalyDetectionConfig:
    """靜止偵測與運動分析

    v2 起 motion_score 單位為 body_fraction × 100（每幀位移 / 胸→髖距離 × 100），
    與訓練管線 normalize_skeleton_coords 一致，消除拍攝距離影響。
    閾值參考值：靜止呼吸 < 2；舔毛/抓撓 5-15；走路 > 10
    """

    # 2026-08-15：這四個欄位原本是純寫死常數，沒有 env 覆寫。納入 GUI 設定系統時
    # 補上對應環境變數（CAT_MONITORING_MAX_MOTION 等），屬於新增而非移除既有機制，
    # 讓它們也能走「環境變數 > runtime_settings.current.json > 內建預設值」這條鏈。
    MAX_MOTION = _env_float(
        "CAT_MONITORING_MAX_MOTION",
        _runtime_default("anomaly_detection.max_motion", 20.0, value_type=float),
    )  # motion_score 正規化分母（body_fraction×100）；走路約 10-20
    KP_CONF_THRES = _env_float(
        "CAT_MONITORING_KP_CONF_THRES",
        _runtime_default("anomaly_detection.kp_conf_thres", 0.5, value_type=float),
    )  # 只使用高於此信心的關鍵點計算 motion_score
    ROLLING_WINDOW_SIZE = _env_int(
        "CAT_MONITORING_ROLLING_WINDOW_SIZE",
        _runtime_default("anomaly_detection.rolling_window_size", 30, value_type=int),
    )  # 滾動均值視窗大小（幀數，30fps ≈ 1 秒）
    STILL_MOTION_THRESHOLD = _env_float(
        "CAT_MONITORING_STILL_MOTION_THRESHOLD",
        _runtime_default("anomaly_detection.still_motion_threshold", 3.0, value_type=float),
    )  # 滾動均值低於此值（body_fraction×100）判為靜止；呼吸抖動約 < 2


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

    正式套用前建議先在 GUI 模式肉眼比對過覆蓋規則是否合理， 確認沒問題後再開啟。
    """

    ENABLE_SQA_DUAL_JUDGMENT = _env_bool(
        "CAT_MONITORING_ENABLE_SQA_DUAL_JUDGMENT",
        _runtime_default("anomaly_detection.enable_sqa_dual_judgment", True, value_type=bool),
    )

# ==================== 行為追蹤參數 ====================
class BehaviorTrackingConfig:
    """行為統計和追蹤"""

    # 歷史記錄大小
    MAX_HISTORY_SIZE = _env_int(
        "CAT_MONITORING_MAX_HISTORY_SIZE",
        _runtime_default("behavior_tracking.max_history_size", 100, value_type=int),
    )  # 行為歷史清單最多保留筆數

    # 活動力窗口
    ACTIVITY_WINDOW_SIZE = _env_int(
        "CAT_MONITORING_ACTIVITY_WINDOW_SIZE",
        _runtime_default("behavior_tracking.activity_window_size", 54, value_type=int),
    )  # 30fps × 1.8s = 54；須 ≥ TARGET_MODEL_FPS × ACTIVITY_SCORE_WINDOW_SECONDS

    # 行為轉換與活動分數參數
    MIN_RECORD_DURATION_SECONDS = _env_float(
        "CAT_MONITORING_MIN_RECORD_DURATION_SECONDS",
        _runtime_default("behavior_tracking.min_record_duration_seconds", 2.0, value_type=float),
    )  # 單一行為最短記錄秒數
    ACTIVITY_SCORE_WINDOW_SECONDS = _env_float(
        "CAT_MONITORING_ACTIVITY_SCORE_WINDOW_SECONDS",
        _runtime_default(
            "behavior_tracking.activity_score_window_seconds", 1.2, value_type=float
        ),
    )  # 活動分數取樣時間窗（秒）；越短反應越快
    LOW_CONFIDENCE_ACTIVITY_WEIGHT = _env_float(
        "CAT_MONITORING_LOW_CONFIDENCE_ACTIVITY_WEIGHT",
        _runtime_default(
            "behavior_tracking.low_confidence_activity_weight", 0.5, value_type=float
        ),
    )  # 低信心幀的活動權重
    STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD = _env_float(
        "CAT_MONITORING_STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD",
        _runtime_default(
            "behavior_tracking.stgcn_behavior_label_confidence_threshold", 0.80, value_type=float
        ),
    )  # ST-GCN 行為標籤輸出門檻；低於此值視為 normal

    # 顯示層 hysteresis：同一個新類別要連續達到這麼多次分類視窗（非幀數，見 STGCNConfig.WINDOW_STRIDE
    # 對應的 CLASSIFY_STRIDE）才會真的切換畫面上顯示的行為標籤，用來過濾單一視窗瞬間誤判造成的畫面閃爍
    # （例如動作轉換瞬間、windup 動作被誤判成別的類別）。只影響顯示，不影響底層逐視窗統計/紀錄。
    # 五個行為各自獨立設定（key 對應 BEHAVIOR_CATEGORIES 的 id：0 walk/1 lick/2 scratch/3 shake/4 stop）；
    # <=1 等同該行為關閉此機制，維持原本逐視窗即時顯示的行為。
    # shake 動作本身只持續 0.5~1 秒，換算成分類視窗數不多，門檻設太高會導致整個 shake 事件
    # 結束前都累積不到門檻次數、畫面永遠來不及切換顯示，因此預設給比其他行為低的門檻。
    DISPLAY_HYSTERESIS_WINDOWS_WALK = _env_int(
        "CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_WALK",
        _runtime_default("behavior_tracking.display_hysteresis_windows.walk", 3, value_type=int),
    )
    DISPLAY_HYSTERESIS_WINDOWS_LICK = _env_int(
        "CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_LICK",
        _runtime_default("behavior_tracking.display_hysteresis_windows.lick", 3, value_type=int),
    )
    DISPLAY_HYSTERESIS_WINDOWS_SCRATCH = _env_int(
        "CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_SCRATCH",
        _runtime_default(
            "behavior_tracking.display_hysteresis_windows.scratch", 3, value_type=int
        ),
    )
    DISPLAY_HYSTERESIS_WINDOWS_SHAKE = _env_int(
        "CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_SHAKE",
        _runtime_default(
            "behavior_tracking.display_hysteresis_windows.shake", 3, value_type=int
        ),
    )
    DISPLAY_HYSTERESIS_WINDOWS_STOP = _env_int(
        "CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_STOP",
        _runtime_default("behavior_tracking.display_hysteresis_windows.stop", 3, value_type=int),
    )
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
    CAT_MISSING_TOLERANCE_FRAMES = _env_int(
        "CAT_MONITORING_CAT_MISSING_TOLERANCE_FRAMES",
        _runtime_default("behavior_tracking.cat_missing_tolerance_frames", 5, value_type=int),
    )

    # 警報門檻
    SCRATCH_ALERT_TIME_SECONDS = _env_float(
        "CAT_MONITORING_SCRATCH_ALERT_TIME_SECONDS",
        _runtime_default("behavior_tracking.scratch_alert_time_seconds", 10.0, value_type=float),
    )  # 單日搔抓累積秒數警戒值
    SCRATCH_ALERT_COUNT_THRESHOLD = _env_int(
        "CAT_MONITORING_SCRATCH_ALERT_COUNT_THRESHOLD",
        _runtime_default(
            "behavior_tracking.scratch_alert_count_threshold", 5, value_type=int
        ),
    )  # 單日搔抓次數警戒值
    LICK_ALERT_TIME_SECONDS = _env_float(
        "CAT_MONITORING_LICK_ALERT_TIME_SECONDS",
        _runtime_default("behavior_tracking.lick_alert_time_seconds", 10.0, value_type=float),
    )  # 單日舔舐累積秒數警戒值
    SHAKE_ALERT_COUNT_THRESHOLD = _env_int(
        "CAT_MONITORING_SHAKE_ALERT_COUNT_THRESHOLD",
        _runtime_default("behavior_tracking.shake_alert_count_threshold", 10, value_type=int),
    )  # 單日甩頭次數警戒值
    STOP_ALERT_TIME_SECONDS = _env_float(
        "CAT_MONITORING_STOP_ALERT_TIME_SECONDS",
        _runtime_default("behavior_tracking.stop_alert_time_seconds", 300.0, value_type=float),
    )  # 單日靜止累積秒數警戒值
    LOW_ACTIVITY_TIME_THRESHOLD_SECONDS = _env_float(
        "CAT_MONITORING_LOW_ACTIVITY_TIME_THRESHOLD_SECONDS",
        _runtime_default(
            "behavior_tracking.low_activity_time_threshold_seconds", 20.0, value_type=float
        ),
    )  # 活動度過低的 walk 時長門檻

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
    個體化基線的前提是「同一份紀錄都來自同一隻貓」。

    ENABLE_IDENTITY_VERIFICATION 是身分驗證（多貓辨識）的總開關，補上這個
    假設原本沒有強制檢查的部分——見 detectors/identity_verifier.py。跟
    SQAConfig 同一套「低耦合、fail-safe」慣例：這裡只決定要不要啟用、指定
    模型檔與目標貓類別名稱，機制內部細節（信心門檻、平滑視窗、裁切比例等）
    留在該模組自己管理；模型檔遺失/載入失敗、torch/torchvision 缺失，或
    啟用時 detectors/identity_verifier.py 整個被刪除，FrameProcessor 都會
    自動停用這一層、回退成「偵測到的貓一律視為目標貓」的原本行為，不影響
    其餘功能運行。目前預設關閉（False）——即使 IDENTITY_MODEL_PATH 指向的
    模型檔已存在，預設部署下這層過濾也不會生效。需要啟用身分過濾時，設定
    環境變數 CAT_MONITORING_ENABLE_IDENTITY_VERIFICATION=true（或透過
    settings_window.py 存進 runtime_settings.current.json）。

    方法：ImageNet 預訓練的 MobileNetV3-Small 微調成 N 類貓咪身分分類器
    （由 tools/cat_identity/2_train.py 訓練，權重存在 C:\\ai_project\\
    identity_models\\；latest.pt 永遠指向最近一次訓練的 best）。

    啟用後：FrameProcessor 只有在判定「這是目標貓（TARGET_CAT_CLASS）」時
    才會把偵測結果送進行為分類/追蹤/CSV/Node-RED；判定為其他貓或無法判定
    時，直接視同「這一幀貓不在畫面」處理（沿用既有的貓咪消失容忍/NOT_VISIBLE
    路徑），不會產生任何行為紀錄，也不會計入 Node-RED 的 today_stats。
    """

    # 貓咪 ID：只被 logutils/csv_logger.py 原樣寫進兩個 CSV 每一列的最後一欄，用來標記
    # 「這些紀錄都來自同一隻貓」。不影響身分驗證判斷、統計、基線，多天歷史 DB 也不含此欄。
    # 設定視窗「貓咪身份驗證」分頁、唯讀：identity_trainer_window.py 按「設為監控系統
    # 使用的模型」時，連同 identity_model_path 一起把所選模型的自訂顯示名稱
    # （run_meta.json 的 target_display_name）寫進 runtime_settings。
    CAT_ID = _env_str(
        "CAT_MONITORING_CAT_ID", _runtime_default("cat_identity.cat_id", "cat_001", value_type=str)
    )

    ENABLE_IDENTITY_VERIFICATION = _env_bool(
        "CAT_MONITORING_ENABLE_IDENTITY_VERIFICATION",
        _runtime_default("cat_identity.enable_identity_verification", False, value_type=bool),
    )
    # 身分辨識 CNN 權重檔（.pt）。由 tools/cat_identity/2_train.py 訓練產生，
    # 檔案自帶 class_names / image_size / normalize 參數。預設指向
    # C:\ai_project\identity_models\latest.pt（永遠是最近一次訓練的 best）。
    IDENTITY_MODEL_PATH = _resolve_project_path(
        _env_str(
            "CAT_MONITORING_IDENTITY_MODEL_PATH",
            _runtime_default(
                "cat_identity.identity_model_path",
                str(Path("identity_models") / "latest.pt"),
                value_type=str,
            ),
        )
    )
    # 「目標貓」在模型 class_names 裡的類別名稱（唯一會被納入統計的貓）。
    # 其餘所有類別、以及信心不足判為「未知」的偵測，都視為「不是目標貓」。
    TARGET_CAT_CLASS = _env_str(
        "CAT_MONITORING_TARGET_CAT_CLASS",
        _runtime_default("cat_identity.target_cat_class", "目標貓", value_type=str),
    )


# ==================== CSV 日誌參數 ====================
class LoggingConfig:
    """日誌記錄設置"""

    # Tracker 狀態持久化路徑（重啟後恢復當日累積資料）。
    # 2026-08-26：原本寫死 C:\a\tracker_state.json（跟專案原始碼分開存放，換一台
    # 機器/重新 clone 就找不到），這個檔案完全由 Python 端自己讀寫（behavior_tracker.py
    # 的 json.dump/json.load，Node-RED 不碰），搬進專案目錄下的 baseline_data/、
    # 改用 _resolve_project_path() 以相對路徑解析，換機器/換 clone 路徑不用再改設定。
    # 見 docs/資料層架構現況與統一管理評估.md 第十四節。
    # 注意：_resolve_project_path() 的錨點是 _PROJECT_ROOT（paper/ 的上一層，見該常數
    # 註解），不是 paper/ 本身，所以這裡的相對路徑要帶上 "paper/" 前綴才會落在
    # paper/baseline_data/（跟 CSV_PATH/SEGMENTS_CSV_PATH 等其他 LoggingConfig 輸出
    # 一致，都放在 paper/ 底下，而不是跟 yolo_models/、stgcn_models/ 一樣放在專案根目錄）。
    TRACKER_STATE_PATH = _resolve_project_path(
        _env_str(
            "CAT_MONITORING_TRACKER_STATE_PATH",
            _runtime_default(
                "logging.tracker_state_path",
                str(Path("paper") / "baseline_data" / "tracker_state.json"),
                value_type=str,
            ),
        )
    )

    # Python 端獨立的多天歷史（analytics/daily_store.py，SQLite）。刻意跟
    # NodeRedConfig.GLOBAL_CONTEXT_PATH（C:\a\global.json，Node-RED 自己的
    # v2_daily_history）分開存放、互不相干——見
    # docs/資料層架構現況與統一管理評估.md 第九節「Python 端獨立多天歷史」。
    # 2026-08-26：同樣是純 Python 端讀寫（daily_store.py 的 sqlite3），搬進專案目錄下
    # 的 baseline_data/，跟上面 TRACKER_STATE_PATH 同一套 _resolve_project_path() 相對
    # 路徑解析邏輯。跟 GLOBAL_CONTEXT_PATH 不同：這個路徑不涉及 Node-RED，搬遷不會
    # 造成任何資料來源分歧（見第十四節）。
    DAILY_HISTORY_DB_PATH = _resolve_project_path(
        _env_str(
            "CAT_MONITORING_DAILY_HISTORY_DB_PATH",
            _runtime_default(
                "logging.daily_history_db_path",
                str(Path("paper") / "baseline_data" / "daily_history.db"),
                value_type=str,
            ),
        )
    )

    # CSV 路徑：預設相對專案根解析（與 TRACKER_STATE_PATH 一致），env/JSON
    # 可用絕對路徑覆寫。2026-08-29：原本寫死 C:\ai_project\paper\... 絕對路徑，
    # 換機器/換 clone 位置就寫錯地方。
    CSV_PATH = _resolve_project_path(
        _env_str(
            "CAT_MONITORING_CSV_PATH",
            _runtime_default(
                "logging.csv_path",
                str(Path("paper") / "cat_monitoring_log.csv"),
                value_type=str,
            ),
        )
    )
    # 行為區段 CSV（BehaviorSegmentLogger）路徑 — 獨立檔案，避免與 CSV_PATH 混寫
    SEGMENTS_CSV_PATH = _resolve_project_path(
        _env_str(
            "CAT_MONITORING_SEGMENTS_CSV_PATH",
            _runtime_default(
                "logging.segments_csv_path",
                str(Path("paper") / "behavior_segments_log.csv"),
                value_type=str,
            ),
        )
    )


# ==================== 系統識別 ====================
class SystemInfo:
    """系統識別和版本信息"""

    SYSTEM_NAME = "Cat Health Monitoring System"
    VERSION = "v4.0-stgcn"
    MODEL_TYPE = "YOLO-Pose + ST-GCN"

    # 幀尺寸——不只是文件用的識別資訊，server/routes.py 的 _build_frame_processor()
    # 會把這兩個值當成 width/height 傳進 FrameProcessor，透過
    # cv2.VideoCapture.set(CAP_PROP_FRAME_WIDTH/HEIGHT) 要求攝影機來源改用這個
    # 解析度擷取（對影片檔/串流來源這個 .set() 是無效操作，只有真的接攝影機時
    # 才會實際生效）。
    #
    # 2026-08-15 修正：這裡原本是 640×640（正方形 1:1），跟 YOLOConfig.IMAGE_SIZE
    # 剛好同一個數字，懷疑是把「YOLO 推論輸入尺寸」誤用成「攝影機擷取解析度」——
    # 兩者是完全不同的概念。1:1 幾乎沒有消費級攝影機原生支援，驅動收到這種不支援
    # 的請求時行為完全看驅動實作，可能悄悄退回它自己的開機預設解析度（畫面被拉伸
    # 鋪滿顯示區域、像素感明顯），也可能退回原生最大解析度（畫面明顯過大，因為
    # VisualizationConfig.STREAM_DISPLAY_SIZE 預設不會再把它縮小）——這是實際
    # 使用網路攝影機時回報過的問題。改成 1280×720（16:9），絕大多數 USB/網路攝影機
    # 原生支援的標準比例，才比較可能被驅動真的採納。
    OUTPUT_WIDTH = 1280
    OUTPUT_HEIGHT = 720


# ==================== 便利函數 ====================
def get_config_summary() -> str:
    """組出可直接印出的完整配置摘要文字（含系統資訊、模型參數、行為追蹤門檻等）。"""
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
    _train_use_joint_prior_weights = STGCNTrainingConfig.get("USE_JOINT_PRIOR_WEIGHTS")
    _train_joint_prior_weights = STGCNTrainingConfig.get("JOINT_PRIOR_WEIGHTS")

    # SEQUENCE_LENGTH/FEATURE_MODE 這兩項推論時「不會」依 checkpoint 自動偵測
    # 覆寫（跟 in_channels/num_joints/attention 不同，見 models/stgcn_model.py），
    # 訓練/推論兩邊不一致會是安靜的 bug，這裡直接標記出來提醒使用者。
    _seq_len_match = (
        "—"
        if _train_seq_len is None
        else ("✓" if _train_seq_len == STGCNConfig.SEQUENCE_LENGTH else "⚠ 不一致！")
    )
    _feature_mode_match = (
        "—"
        if _train_feature_mode is None
        else (
            "✓"
            if _normalize_feature_mode(_train_feature_mode) == STGCNConfig.FEATURE_MODE
            else "⚠ 不一致！"
        )
    )

    summary = f"""
    ╔════════════════════════════════════════════════════════╗
    ║          貓咪監測系統配置摘要                         ║
    ╚════════════════════════════════════════════════════════╝

    📁 路徑配置
      - YOLO 模型   : {ModelPaths.YOLO_MODEL}
      - ST-GCN 模型 : {ModelPaths.STGCN_MODEL}
      - 輸入視訊    : {ModelPaths.VIDEO_INPUT}
      - 日誌目錄    : {ModelPaths.LOG_DIR}
      - 輸出目錄    : {ModelPaths.OUTPUT_DIR}

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
      - 關節先驗權重開關  : {_train_use_joint_prior_weights}  (僅 Attention 啟用時生效，強制放大特定關節訊號)
      - 關節先驗權重內容  : {_train_joint_prior_weights if _train_use_joint_prior_weights else "(未啟用)"}
      - 行為前綴對應      : {_train_behavior_prefixes}

    🕐 執行模式與排程
      - 執行模式          : {RunModeConfig.MODE}  ("server" 或 "gui")
      - 啟動即自動處理    : {RunModeConfig.AUTO_START_PROCESSING}  （False=等第一個 /stream 等請求才啟動處理管線）
      - 排程開始時間      : {RunModeConfig.SCHEDULED_START_TIME or "(未設定)"}
      - 排程結束時間      : {RunModeConfig.SCHEDULED_END_TIME or "(未設定，開始後永遠運行)"}
      - 目前是否在排程區間內: {RunModeConfig.is_within_active_window()}  （即時判斷，印出當下這一刻的狀態）

    🌐 Flask 服務
      - 主機        : {FlaskConfig.HOST}:{FlaskConfig.PORT}
      - JPEG 品質   : {FlaskConfig.JPEG_QUALITY}
      - Debug 模式  : {FlaskConfig.DEBUG}
      - Threaded    : {FlaskConfig.THREADED}

    🧮 個體化基線儀表板（Python 端唯讀展示頁）
      - 啟用          : {BaselineDashboardConfig.ENABLED}  （False=不匯入 dashboard/ 模組，不佔用任何資源）
      - 前端輪詢間隔  : {BaselineDashboardConfig.POLL_INTERVAL_SEC} s
      - 背景重算間隔  : {BaselineDashboardConfig.RECOMPUTE_INTERVAL_SEC} s

    🔗 Node-RED 連線
      - 主機        : {NodeRedConfig.HOST}:{NodeRedConfig.PORT}
      - 推送間隔    : {NodeRedConfig.PUSH_INTERVAL} s
      - 超時        : {NodeRedConfig.TIMEOUT} s
      - Notify 端點   : {NodeRedConfig.ENDPOINT_NOTIFY}
      - Result v1 端點: {NodeRedConfig.ENDPOINT_RESULT}
      - Result v2 端點: {NodeRedConfig.ENDPOINT_RESULT_V2}
      - 個體化基線共用儲存: {NodeRedConfig.GLOBAL_CONTEXT_PATH}

    🎞️ 串流視覺化
      - 串流縮放尺寸      : {VisualizationConfig.STREAM_DISPLAY_SIZE}
      - Ring Buffer 秒數  : {VisualizationConfig.CLIP_SECONDS} s
      - 鼻子梯形 overlay  : {VisualizationConfig.SHOW_NOSE_TRAPEZOID}
      - 偵測框 bbox (啟動預設): {VisualizationConfig.SHOW_BBOX}  （執行期間可用鍵盤 b 切換(僅限 gui 模式)）
      - 骨架關鍵點 (啟動預設): {VisualizationConfig.SHOW_SKELETON}  （執行期間可用鍵盤 s 切換(僅限 gui 模式)）
      - GCN 分類結果+機率條 (啟動預設): {VisualizationConfig.SHOW_GCN_RESULT}  （執行期間可用鍵盤 l 切換(僅限 gui 模式)）

    🛑 靜止偵測（滾動均值閾值，純 CSV 記錄；單位 body_fraction×100）
      - 最大動作值        : {AnomalyDetectionConfig.MAX_MOTION}  （body_fraction×100；走路約 10-20）
      - 關鍵點信心門檻    : {AnomalyDetectionConfig.KP_CONF_THRES}
      - 滾動視窗大小      : {AnomalyDetectionConfig.ROLLING_WINDOW_SIZE} 幀
      - 靜止動作門檻      : {AnomalyDetectionConfig.STILL_MOTION_THRESHOLD}  （body_fraction×100；呼吸抖動約 < 2）

    🔬 Skeleton Quality Assessment（骨架品質雙重判定）
      - 總開關 ENABLE_SQA_DUAL_JUDGMENT: {SQAConfig.ENABLE_SQA_DUAL_JUDGMENT}
        （個別指標開關在 cat_monitoring_system/processors/skeleton_quality_assessment.py 模組內設定，此處不重複顯示）

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

    🐱 貓咪身分與身分驗證
      - 貓咪 ID                   : {CatIdentityConfig.CAT_ID}  （唯讀，跟隨採用的身分模型自訂名稱；只寫進 CSV 每列最後一欄當篩選標記）
      - 身分驗證總開關（多貓過濾）: {CatIdentityConfig.ENABLE_IDENTITY_VERIFICATION}  （False=偵測到的貓一律視為目標貓，不做身分過濾）
      - 身分辨識 CNN 模型檔       : {CatIdentityConfig.IDENTITY_MODEL_PATH}
      - 目標貓類別名稱           : {CatIdentityConfig.TARGET_CAT_CLASS}  （對應模型 class_names；其餘類別/未知都視為非目標貓）

    📄 日誌設定
      - 主要 CSV          : {LoggingConfig.CSV_PATH}
      - 行為區段 CSV      : {LoggingConfig.SEGMENTS_CSV_PATH}
      - Tracker 狀態檔    : {LoggingConfig.TRACKER_STATE_PATH}  （重啟後恢復當日累積資料）
      - 多天歷史 SQLite   : {LoggingConfig.DAILY_HISTORY_DB_PATH}

    📋 系統資訊  (硬編碼於 SystemInfo 類別: SYSTEM_NAME, VERSION, MODEL_TYPE, OUTPUT_WIDTH/HEIGHT；無 env 覆寫)
      - 名稱    : {SystemInfo.SYSTEM_NAME}
      - 版本    : {SystemInfo.VERSION}
      - 模型    : {SystemInfo.MODEL_TYPE}
      - 輸出尺寸: {SystemInfo.OUTPUT_WIDTH} × {SystemInfo.OUTPUT_HEIGHT}  （會透過 cv2.VideoCapture.set() 要求攝影機來源改用此解析度擷取；對影片檔/串流無效）

    ╔════════════════════════════════════════════════════════╗
    """
    return summary


def validate_all_config() -> bool:
    """依序驗證模型檔案、目錄結構、參數範圍、訓練/推論一致性，全部通過才回傳 True。"""
    print("🔍 驗證配置...")

    def _validate_runtime_values() -> bool:
        errors = []
        if not _is_valid_port(FlaskConfig.PORT):
            errors.append(f"Flask PORT 無效: {FlaskConfig.PORT}")
        if not _is_valid_port(NodeRedConfig.PORT):
            errors.append(f"Node-RED PORT 無效: {NodeRedConfig.PORT}")
        if not (0.0 <= YOLOConfig.CONFIDENCE_THRESHOLD <= 1.0):
            errors.append(
                f"YOLO CONFIDENCE_THRESHOLD 應在 [0,1]: {YOLOConfig.CONFIDENCE_THRESHOLD}"
            )
        if STGCNConfig.SEQUENCE_LENGTH <= 0:
            errors.append(
                f"ST-GCN SEQUENCE_LENGTH 必須 > 0: {STGCNConfig.SEQUENCE_LENGTH}"
            )
        if STGCNConfig.TARGET_MODEL_FPS <= 0:
            errors.append(
                f"ST-GCN TARGET_MODEL_FPS 必須 > 0: {STGCNConfig.TARGET_MODEL_FPS}"
            )
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
                errors.append(
                    f"STREAM_DISPLAY_SIZE 必須是 (寬, 高) 且都 > 0: {VisualizationConfig.STREAM_DISPLAY_SIZE}"
                )
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

    def _validate_train_inference_consistency() -> bool:
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
        if (
            train_feature_mode is not None
            and _normalize_feature_mode(train_feature_mode) != STGCNConfig.FEATURE_MODE
        ):
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

    # 2026-08-15：直接修改這個檔案裡任何 _runtime_default() 的字面值（例如把
    # "server" 改成 "gui"）之後，default_runtime_settings.json（設定視窗
    # 「還原 GUI 預設值」用的範本）不會自動跟著變——那份 JSON 是獨立檔案，兩邊
    # 各自維護容易漂移。這裡讓 `python config.py` 順手把範本同步好，不用再手動
    # 對照兩邊改。只動範本檔，不動 runtime_settings.current.json（使用者實際套用中的
    # 設定，不該被這個動作悄悄覆蓋）。
    try:
        from settings_manager import regenerate_default_runtime_settings

        changed, changes = regenerate_default_runtime_settings()
        if changed:
            print(
                f"\n🔄 default_runtime_settings.json 已同步（{len(changes)} 個欄位"
                "跟目前硬編碼預設值不同，已更新為新值）："
            )
            for key, label, old, new in changes:
                print(f"  - {label}（{key}）：{old!r} → {new!r}")
        else:
            print("\n✓ default_runtime_settings.json 已跟目前硬編碼預設值一致，未變更。")
    except Exception as e:
        print(f"\n⚠ 同步 default_runtime_settings.json 失敗（不影響主系統）：{e}")
