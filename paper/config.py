"""
貓咪監測系統配置檔案
方便管理所有設置，避免直接修改主程序
"""

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
    YOLO_MODEL = _env_str("CAT_MONITORING_YOLO_MODEL", r"C:\ai_project\cat_pose\v11s_118.pt")
    
    # ST-GCN 模型
    STGCN_MODEL = _env_str("CAT_MONITORING_STGCN_MODEL", r"C:\Users\homec\Downloads\stgcn_results\run_085_xy_conf_v_bone_att_on\085_best_model.pth")
    
    # 測試視頻
    VIDEO_INPUT = _env_video_input("CAT_MONITORING_VIDEO_INPUT", r"C:\Users\homec\OneDrive\圖片\貓咪\自行拍攝\影片\1783259367519_0.mp4")
                                                                  # rtsp://12345678:456456123@192.168.0.46:554/stream1
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

# ==================== 執行模式參數 ====================
class RunModeConfig:
    """控制 main.py 啟動時走哪一種模式，整體處理架構（FrameProcessor 等）不受影響。

    "server"（預設）：現行行為，啟動 Flask HTTP 伺服器 + Node-RED 上線通知
    "gui"           ：不啟動 Flask/Node-RED，直接用同一套 FrameProcessor 開本地視窗顯示
    """
    MODE = _env_str("CAT_MONITORING_RUN_MODE", "server")

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
    DISPLAY_HYSTERESIS_WINDOWS_WALK = _env_int("CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_WALK", 2)
    DISPLAY_HYSTERESIS_WINDOWS_LICK = _env_int("CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_LICK", 2)
    DISPLAY_HYSTERESIS_WINDOWS_SCRATCH = _env_int("CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_SCRATCH", 2)
    DISPLAY_HYSTERESIS_WINDOWS_SHAKE = _env_int("CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_SHAKE", 2)
    DISPLAY_HYSTERESIS_WINDOWS_STOP = _env_int("CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_STOP", 2)
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

    🛑 靜止偵測（滾動均值閾值，純 CSV 記錄；單位 body_fraction×100）
      - 最大動作值        : {AnomalyDetectionConfig.MAX_MOTION}  （body_fraction×100；走路約 10-20）
      - 關鍵點信心門檻    : {AnomalyDetectionConfig.KP_CONF_THRES}
      - 滾動視窗大小      : {AnomalyDetectionConfig.ROLLING_WINDOW_SIZE} 幀
      - 靜止動作門檻      : {AnomalyDetectionConfig.STILL_MOTION_THRESHOLD}  （body_fraction×100；呼吸抖動約 < 2）

    🏷️ 行為追蹤門檻
      - ST-GCN 行為標籤門檻  : {BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD}
      - 最短記錄時長         : {BehaviorTrackingConfig.MIN_RECORD_DURATION_SECONDS} s
      - 活動分數時間窗        : {BehaviorTrackingConfig.ACTIVITY_SCORE_WINDOW_SECONDS} s
      - 活動力窗口大小        : {BehaviorTrackingConfig.ACTIVITY_WINDOW_SIZE} 幀
      - 搔抓警報秒數          : {BehaviorTrackingConfig.SCRATCH_ALERT_TIME_SECONDS} s
      - 搔抓警報次數          : {BehaviorTrackingConfig.SCRATCH_ALERT_COUNT_THRESHOLD} 次
      - 舔舐警報秒數          : {BehaviorTrackingConfig.LICK_ALERT_TIME_SECONDS} s
      - 甩頭警報次數          : {BehaviorTrackingConfig.SHAKE_ALERT_COUNT_THRESHOLD} 次
      - 靜止警報秒數          : {BehaviorTrackingConfig.STOP_ALERT_TIME_SECONDS} s
      - 低活動 walk 門檻      : {BehaviorTrackingConfig.LOW_ACTIVITY_TIME_THRESHOLD_SECONDS} s

    🌐 Flask 服務
      - 主機        : {FlaskConfig.HOST}:{FlaskConfig.PORT}
      - JPEG 品質   : {FlaskConfig.JPEG_QUALITY}
      - Debug 模式  : {FlaskConfig.DEBUG}

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
      - 主要 CSV        : {LoggingConfig.CSV_PATH}
      - 行為區段 CSV    : {LoggingConfig.SEGMENTS_CSV_PATH}

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

    checks = [
        ("模型檔案", ModelPaths.validate),
        ("目錄結構", lambda: (ModelPaths.ensure_dirs(), True)[1]),
        ("參數範圍", _validate_runtime_values),
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
