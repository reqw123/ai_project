"""
貓咪監測系統配置檔案
方便管理所有設置，避免直接修改主程序
"""

import os
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
        return float(value)
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


def _is_valid_port(port):
    return isinstance(port, int) and 1 <= port <= 65535

# ==================== 模型和資料路徑 ====================
class ModelPaths:
    """模型和資料檔案路徑"""
    
    # YOLO 模型
    YOLO_MODEL = _env_str("CAT_MONITORING_YOLO_MODEL", r"C:\AI_Project\cat_pose\v11s_68.pt")
    
    # ST-GCN 模型
    STGCN_MODEL = _env_str("CAT_MONITORING_STGCN_MODEL", r"C:\AI_Project\cat_pose\gcn_pose\models\stgcn_best_xyv.pth")
    
    # 測試視頻
    VIDEO_INPUT = _env_video_input("CAT_MONITORING_VIDEO_INPUT", 0)
    
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
    """YOLO 檢測參數"""
    
    # 推論參數
    IMAGE_SIZE = _env_int("CAT_MONITORING_YOLO_IMAGE_SIZE", 640)
    CONFIDENCE_THRESHOLD = _env_float("CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD", 0.50)
    KEYPOINT_CONFIDENCE_THRESHOLD = _env_float("CAT_MONITORING_YOLO_KEYPOINT_CONFIDENCE_THRESHOLD", 0.50)
    TOTAL_KEYPOINTS = 17
    
    # 硬體
    DEVICE = _env_str("CAT_MONITORING_YOLO_DEVICE", "cuda")  # 改為 "cpu" 如果無 GPU

# ==================== ST-GCN 參數 ====================
class STGCNConfig:
    """ST-GCN 模型參數"""
    
    # 模型超參數
    SEQUENCE_LENGTH = _env_int("CAT_MONITORING_STGCN_SEQUENCE_LENGTH", 32)          # 時間窗長度（幀數）
    NUM_CLASSES = 4               # 行為類別數
    IN_CHANNELS = 2               # 輸入通道數 (x, y)
    NUM_JOINTS = 17               # 關鍵點數
    NUM_LAYERS = 3                # ST-GCN 層數
    
    # 時間步參數
    WINDOW_STRIDE = _env_int("CAT_MONITORING_STGCN_WINDOW_STRIDE", 16)            # 滑動步長
    
    # 硬體
    DEVICE = _env_str("CAT_MONITORING_STGCN_DEVICE", "cuda")  # 改為 "cpu" 如果無 GPU

    # FPS 同步：對來源影片做降採樣，使模型輸入時基符合訓練設定
    TARGET_MODEL_FPS = _env_float("CAT_MONITORING_TARGET_MODEL_FPS", 30.0)       # 與訓練時序一致；來源 FPS 超過此值時才做降採樣
    ENABLE_FPS_DOWNSAMPLE = _env_bool("CAT_MONITORING_ENABLE_FPS_DOWNSAMPLE", True)  # False → 停用降採樣（直接餵所有幀）

    # 關鍵點 EMA 平滑（須與 train_gcn.py 的 KP_EMA_ALPHA 保持一致）
    # alpha 越大 → 越貼近原始偵測值；alpha 越小 → 越平滑但延遲增加
    KP_EMA_ALPHA = _env_float("CAT_MONITORING_KP_EMA_ALPHA", 0.5)

    # 行為類別
    CLASS_NAMES = ["walk", "lick", "scratch", "shake"]
    
    # 視覺化用的顏色 (BGR)
    CLASS_COLORS = [
        (0, 255, 0),      # walk - 綠色
        (0, 255, 255),    # lick - 黃色
        (255, 0, 0),      # scratch - 藍色
        (0, 0, 255)       # shake - 紅色
    ]

# ==================== 異常檢測參數 ====================
class AnomalyDetectionConfig:
    """異常檢測和運動分析"""
    
    # EMA 參數
    EMA_ALPHA = 1.0
    ABNORMAL_THRESHOLD = 0.2
    MIN_BODY_SCALE = 1e-3
    STABILITY_K = 4.0
    
    # 動作檢測敏感度 (0-1，越小越敏感)
    MOTION_SENSITIVITY = 0.5

# ==================== Flask 服務參數 ====================
class FlaskConfig:
    """Flask Web 服務參數"""
    
    HOST = _env_str("CAT_MONITORING_FLASK_HOST", "0.0.0.0")
    PORT = _env_int("CAT_MONITORING_FLASK_PORT", 5000)
    DEBUG = _env_bool("CAT_MONITORING_FLASK_DEBUG", False)
    THREADED = _env_bool("CAT_MONITORING_FLASK_THREADED", True)
    
    # JPEG 壓縮品質 (1-100)
    JPEG_QUALITY = _env_int("CAT_MONITORING_JPEG_QUALITY", 60)
    
    # 串流 FPS
    STREAM_FPS = _env_int("CAT_MONITORING_STREAM_FPS", 30)

# ==================== Node-RED 參數 ====================
class NodeRedConfig:
    """Node-RED 通訊參數"""
    
    HOST = _env_str("CAT_MONITORING_NODERED_HOST", "127.0.0.1")
    PORT = _env_int("CAT_MONITORING_NODERED_PORT", 1880)
    
    # 推送間隔（秒）
    PUSH_INTERVAL = _env_float("CAT_MONITORING_NODERED_PUSH_INTERVAL", 0.5)
    
    # 端點
    ENDPOINT_NOTIFY = _env_str("CAT_MONITORING_NODERED_ENDPOINT_NOTIFY", f"http://{HOST}:{PORT}/python_online")
    ENDPOINT_RESULT = _env_str("CAT_MONITORING_NODERED_ENDPOINT_RESULT", f"http://{HOST}:{PORT}/yolo_result")
    
    # 超時時間（秒）
    TIMEOUT = _env_float("CAT_MONITORING_NODERED_TIMEOUT", 2)

# ==================== 行為追蹤參數 ====================
class BehaviorTrackingConfig:
    """行為統計和追蹤"""
    
    # 歷史記錄大小
    MAX_HISTORY_SIZE = 100
    MAX_ALERTS_SIZE = 50
    
    # 活動力窗口
    ACTIVITY_WINDOW_SIZE = 60
    
    # 行為統計：四種行為完全獨立
    # normal 對應 walk，scratch 只對應 scratch
    BEHAVIOR_CATEGORIES = {
        0: "walk",
        1: "lick",
        2: "scratch",
        3: "shake"
    }
    
    DISPLAY_TEXT = {
        "walk": "一般活動",
        "scratch": "搔抓動作",
        "lick": "舔拭理毛",
        "shake": "甩頭動作"
    }
    
    DISPLAY_EMOJI = {
        "walk": "🚶",
        "scratch": "🐾",
        "lick": "🧼",
        "shake": "甩頭"
    }

# ==================== CSV 日誌參數 ====================
class LoggingConfig:
    """日誌記錄設置"""
    
    # CSV 檔案名
    CSV_FILENAME = "cat_monitoring_log.csv"
    
    # CSV 欄位
    CSV_COLUMNS = [
        "Frame", "Timestamp", "Behavior", "GCN_Confidence",
        "Abnormal", "Motion_Score", "Stability"
    ]
    
    # 日誌等級
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# ==================== 顯示和視覺化參數 ====================
class VisualizationConfig:
    """顯示和繪圖參數"""
    
    # 覆蓋層顯示設置
    DRAW_OVERLAY_STREAM = True    # Node-RED 串流用
    DRAW_OVERLAY_DEBUG = False     # 本地除錯用

    # 串流輸出優化
    # STREAM_DISPLAY_SIZE: None → 維持原始解析度；(寬, 高) → 先縮放再 JPEG 編碼
    STREAM_DISPLAY_SIZE = None
    # FAST_STREAM_OVERLAY: True → 在原始解析度畫完 overlay 後再縮放（縮圖速度快）
    #                      False → 先縮放再重繪 overlay（適合需要精確文字大小時）
    FAST_STREAM_OVERLAY = True
    
    # 骨架顏色 (BGR)
    COLOR_HEAD = (255, 255, 0)
    COLOR_BODY = (0, 255, 0)
    COLOR_LIMB = (255, 0, 0)
    COLOR_TAIL = (255, 0, 255)
    COLOR_KEYPOINT = (0, 0, 255)
    
    # 狀態顯示
    COLOR_NORMAL = (0, 255, 0)     # 綠色
    COLOR_ABNORMAL = (0, 0, 255)   # 紅色
    
    # 字體
    FONT_PATH = 'C:\\Windows\\Fonts\\msyh.ttc'  # Windows 中文字體
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    
    # 線條粗細
    LINE_WIDTH_SKELETON = 2
    LINE_WIDTH_BOX = 3
    POINT_RADIUS = 3

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
    ║     貓咪監測系統配置摘要                              ║
    ╚════════════════════════════════════════════════════════╝
    
    📋 系統信息:
      - 名稱: {SystemInfo.SYSTEM_NAME}
      - 版本: {SystemInfo.VERSION}
      - 模型: {SystemInfo.MODEL_TYPE}
    
    📷 YOLO 參數:
      - 圖像尺寸: {YOLOConfig.IMAGE_SIZE}
      - 信心閾值: {YOLOConfig.CONFIDENCE_THRESHOLD}
      - 關鍵點閾值: {YOLOConfig.KEYPOINT_CONFIDENCE_THRESHOLD}
      - 設備: {YOLOConfig.DEVICE}
    
    🧠 ST-GCN 參數:
      - 時間窗長度: {STGCNConfig.SEQUENCE_LENGTH} 幀
      - 行為類別: {STGCNConfig.CLASS_NAMES}
      - 層數: {STGCNConfig.NUM_LAYERS}
      - 設備: {STGCNConfig.DEVICE}
    
    🌐 Flask 服務:
      - 主機: {FlaskConfig.HOST}
      - 埠號: {FlaskConfig.PORT}
      - 串流 FPS: {FlaskConfig.STREAM_FPS}
    
    🔗 Node-RED 連線:
      - 主機: {NodeRedConfig.HOST}:{NodeRedConfig.PORT}
      - 推送間隔: {NodeRedConfig.PUSH_INTERVAL}s
      - 超時: {NodeRedConfig.TIMEOUT}s
    
    ⚠️ 異常偵測:
      - EMA 係數: {AnomalyDetectionConfig.EMA_ALPHA}
      - 異常閾值: {AnomalyDetectionConfig.ABNORMAL_THRESHOLD}
    
    📁 路徑配置:
      - YOLO 模型: {ModelPaths.YOLO_MODEL}
      - ST-GCN 模型: {ModelPaths.STGCN_MODEL}
      - 輸入視頻: {ModelPaths.VIDEO_INPUT}
      - 日誌目錄: {ModelPaths.LOG_DIR}
      - 輸出目錄: {ModelPaths.OUTPUT_DIR}
    
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
        if not (0.0 <= YOLOConfig.KEYPOINT_CONFIDENCE_THRESHOLD <= 1.0):
            errors.append(f"YOLO KEYPOINT_CONFIDENCE_THRESHOLD 應在 [0,1]: {YOLOConfig.KEYPOINT_CONFIDENCE_THRESHOLD}")
        if STGCNConfig.SEQUENCE_LENGTH <= 0:
            errors.append(f"ST-GCN SEQUENCE_LENGTH 必須 > 0: {STGCNConfig.SEQUENCE_LENGTH}")
        if not (0.0 < STGCNConfig.KP_EMA_ALPHA <= 1.0):
            errors.append(f"KP_EMA_ALPHA 應在 (0,1]: {STGCNConfig.KP_EMA_ALPHA}")
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


def get_runtime_config_snapshot():
    """回傳目前實際生效的主要設定（已包含環境變數覆寫結果）。"""
    return {
        "model_paths": {
            "yolo_model": ModelPaths.YOLO_MODEL,
            "stgcn_model": ModelPaths.STGCN_MODEL,
            "video_input": ModelPaths.VIDEO_INPUT,
            "log_dir": ModelPaths.LOG_DIR,
            "output_dir": ModelPaths.OUTPUT_DIR,
        },
        "yolo": {
            "image_size": YOLOConfig.IMAGE_SIZE,
            "confidence_threshold": YOLOConfig.CONFIDENCE_THRESHOLD,
            "keypoint_confidence_threshold": YOLOConfig.KEYPOINT_CONFIDENCE_THRESHOLD,
            "device": YOLOConfig.DEVICE,
        },
        "stgcn": {
            "sequence_length": STGCNConfig.SEQUENCE_LENGTH,
            "window_stride": STGCNConfig.WINDOW_STRIDE,
            "device": STGCNConfig.DEVICE,
            "target_model_fps": STGCNConfig.TARGET_MODEL_FPS,
            "enable_fps_downsample": STGCNConfig.ENABLE_FPS_DOWNSAMPLE,
            "kp_ema_alpha": STGCNConfig.KP_EMA_ALPHA,
        },
        "flask": {
            "host": FlaskConfig.HOST,
            "port": FlaskConfig.PORT,
            "debug": FlaskConfig.DEBUG,
            "threaded": FlaskConfig.THREADED,
            "jpeg_quality": FlaskConfig.JPEG_QUALITY,
            "stream_fps": FlaskConfig.STREAM_FPS,
        },
        "nodered": {
            "host": NodeRedConfig.HOST,
            "port": NodeRedConfig.PORT,
            "push_interval": NodeRedConfig.PUSH_INTERVAL,
            "endpoint_notify": NodeRedConfig.ENDPOINT_NOTIFY,
            "endpoint_result": NodeRedConfig.ENDPOINT_RESULT,
            "timeout": NodeRedConfig.TIMEOUT,
        },
        "system": {
            "name": SystemInfo.SYSTEM_NAME,
            "version": SystemInfo.VERSION,
            "model_type": SystemInfo.MODEL_TYPE,
        },
    }

# ==================== 骨架連結定義 ====================
class SkeletonLinks:
    """骨架連接定義（COCO 17 點格式）"""
    
    HEAD_LINKS = [(0,1), (0,2), (1,2)]
    BODY_LINKS = [(0,3), (3,4), (4,5)]
    FRONT_LIMBS = [(3,6), (6,7), (3,8), (8,9)]
    HIND_LIMBS = [(5,10), (10,11), (5,12), (12,13)]
    TAIL_LINKS = [(5,14), (14,15), (15,16)]
    
    ALL_LINKS = HEAD_LINKS + BODY_LINKS + FRONT_LIMBS + HIND_LIMBS + TAIL_LINKS

# ==================== 主測試 ====================
if __name__ == "__main__":
    print(get_config_summary())
    
    if validate_all_config():
        print("\n✅ 所有配置驗證通過！")
    else:
        print("\n⚠ 部分配置驗證失敗，請檢查。")
