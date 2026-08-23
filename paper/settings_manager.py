"""執行期設定（runtime_settings.current.json）的載入／驗證／合併／原子儲存。

**跟 config.py 的關係**：config.py 的每個可納管欄位在計算「硬編碼預設值」時，
會多包一層 `_runtime_default()`（定義在 config.py，內部呼叫這裡的
`get_runtime_value()`），讓優先順序變成「環境變數 > runtime_settings.current.json >
config.py 內建預設值」。這裡完全不 import config.py（避免循環依賴），也不重複
抄錄任何硬編碼預設值——GUI 要顯示「目前生效值」時，直接 `getattr` config.py
的 class attribute 現讀，不由這裡代管一份可能過期的複本。

**FIELD_SCHEMA 是唯一的欄位對照表**：dotted JSON key ↔ 環境變數名稱 ↔
`(class 名稱, attribute 名稱)` ↔ 型別 ↔ 驗證規則，settings_window.py 的欄位
渲染引擎跟這裡的 validate_settings() 都吃同一份表——新增一個可調整欄位只需要
在這裡加一筆，並在 config.py 對應屬性外包一層 `_runtime_default()`。

**刻意排除**：`STGCNTrainingConfig`（唯一權威來源 stgcn_config.yaml）與
`STGCNConfig.SEQUENCE_LENGTH`/`FEATURE_MODE`/`NUM_CLASSES`（訓練 checkpoint
架構相容性參數）完全不出現在 FIELD_SCHEMA，GUI 因此也看不到、存不了、蓋不掉
這些欄位——排除是「這份表裡沒有」，不是額外寫一份黑名單去擋。
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_PAPER_DIR = Path(__file__).resolve().parent
RUNTIME_SETTINGS_PATH = _PAPER_DIR / "runtime_settings.current.json"
DEFAULT_RUNTIME_SETTINGS_PATH = _PAPER_DIR / "default_runtime_settings.json"
BACKUP_SETTINGS_PATH = _PAPER_DIR / "runtime_settings.previous.json"


# ============================================================================
# FIELD_SCHEMA：唯一的欄位對照表        
# ============================================================================
# 每筆 dict 欄位：  
#   json_key   dotted key，對應 runtime_settings.current.json 巢狀路徑
#   env_var    config.py 用來覆寫這個欄位的環境變數名稱
#   attr       (class 名稱, attribute 名稱)，GUI 用 getattr(config.<class>, <attr>) 現讀生效值
#   tab        GUI 分頁名稱
#   label      GUI 顯示用中文標籤
#   value_type "bool"/"int"/"float"/"str"/"file"/"folder"/"video_input"/"size"/"hhmm"/"enum"
#   validate   驗證規則 tag（見 _VALIDATORS）
#   choices    僅 value_type == "enum" 使用
#   browse_filter 僅 file 型別使用，(說明文字, 副檔名 pattern)
FIELD_SCHEMA = [
    # ── 模型與輸入來源 ────────────────────────────────────────────────
    {
        "json_key": "model_paths.yolo_model", "env_var": "CAT_MONITORING_YOLO_MODEL",
        "attr": ("ModelPaths", "YOLO_MODEL"), "tab": "模型與輸入來源",
        "label": "YOLO 模型檔案", "value_type": "file", "validate": "required_file",
        "browse_filter": ("YOLO 權重檔", "*.pt"),
    },
    {
        "json_key": "model_paths.stgcn_model", "env_var": "CAT_MONITORING_STGCN_MODEL",
        "attr": ("ModelPaths", "STGCN_MODEL"), "tab": "模型與輸入來源",
        "label": "ST-GCN 模型檔案", "value_type": "file", "validate": "required_file",
        "browse_filter": ("ST-GCN 權重檔", "*.pth"),
    },
    {
        "json_key": "model_paths.video_input", "env_var": "CAT_MONITORING_VIDEO_INPUT",
        "attr": ("ModelPaths", "VIDEO_INPUT"), "tab": "模型與輸入來源",
        "label": "影像來源（本機路徑／RTSP-HTTP URL／攝影機索引）",
        "value_type": "video_input", "validate": "optional_file_warn",
    },
    {
        "json_key": "model_paths.log_dir", "env_var": "CAT_MONITORING_LOG_DIR",
        "attr": ("ModelPaths", "LOG_DIR"), "tab": "模型與輸入來源",
        "label": "日誌目錄", "value_type": "folder", "validate": "output_path",
    },
    {
        "json_key": "model_paths.output_dir", "env_var": "CAT_MONITORING_OUTPUT_DIR",
        "attr": ("ModelPaths", "OUTPUT_DIR"), "tab": "模型與輸入來源",
        "label": "輸出目錄", "value_type": "folder", "validate": "output_path",
    },
    # ── YOLO 推論 ────────────────────────────────────────────────────
    {
        "json_key": "yolo.image_size", "env_var": "CAT_MONITORING_YOLO_IMAGE_SIZE",
        "attr": ("YOLOConfig", "IMAGE_SIZE"), "tab": "YOLO 推論",
        "label": "推論圖像尺寸", "value_type": "int", "validate": "positive_int",
    },
    {
        "json_key": "yolo.confidence_threshold", "env_var": "CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD",
        "attr": ("YOLOConfig", "CONFIDENCE_THRESHOLD"), "tab": "YOLO 推論",
        "label": "偵測信心閾值", "value_type": "float", "validate": "unit_interval",
    },
    # ── ST-GCN 推論（不含訓練架構相容性欄位） ───────────────────────────
    {
        "json_key": "stgcn_runtime.window_stride", "env_var": "CAT_MONITORING_STGCN_WINDOW_STRIDE",
        "attr": ("STGCNConfig", "WINDOW_STRIDE"), "tab": "ST-GCN 推論",
        "label": "推論滑動步長（幀）", "value_type": "int", "validate": "positive_int",
    },
    {
        "json_key": "stgcn_runtime.target_model_fps", "env_var": "CAT_MONITORING_TARGET_MODEL_FPS",
        "attr": ("STGCNConfig", "TARGET_MODEL_FPS"), "tab": "ST-GCN 推論",
        "label": "目標模型 FPS", "value_type": "float", "validate": "positive_float",
    },
    {
        "json_key": "stgcn_runtime.enable_fps_downsample", "env_var": "CAT_MONITORING_ENABLE_FPS_DOWNSAMPLE",
        "attr": ("STGCNConfig", "ENABLE_FPS_DOWNSAMPLE"), "tab": "ST-GCN 推論",
        "label": "啟用 FPS 降採樣", "value_type": "bool", "validate": "bool",
    },
    {
        "json_key": "stgcn_runtime.kp_ema_alpha", "env_var": "CAT_MONITORING_KP_EMA_ALPHA",
        "attr": ("STGCNConfig", "KP_EMA_ALPHA"), "tab": "ST-GCN 推論",
        "label": "關鍵點 EMA α（1.0=不平滑）", "value_type": "float", "validate": "ema_alpha",
    },
    # ── 異常偵測與骨架品質 ────────────────────────────────────────────
    {
        "json_key": "anomaly_detection.max_motion", "env_var": "CAT_MONITORING_MAX_MOTION",
        "attr": ("AnomalyDetectionConfig", "MAX_MOTION"), "tab": "異常偵測與骨架品質",
        "label": "最大動作值正規化分母", "value_type": "float", "validate": "positive_float",
    },
    {
        "json_key": "anomaly_detection.kp_conf_thres", "env_var": "CAT_MONITORING_KP_CONF_THRES",
        "attr": ("AnomalyDetectionConfig", "KP_CONF_THRES"), "tab": "異常偵測與骨架品質",
        "label": "關鍵點信心門檻", "value_type": "float", "validate": "unit_interval",
    },
    {
        "json_key": "anomaly_detection.rolling_window_size", "env_var": "CAT_MONITORING_ROLLING_WINDOW_SIZE",
        "attr": ("AnomalyDetectionConfig", "ROLLING_WINDOW_SIZE"), "tab": "異常偵測與骨架品質",
        "label": "滾動視窗大小（幀）", "value_type": "int", "validate": "positive_int",
    },
    {
        "json_key": "anomaly_detection.still_motion_threshold", "env_var": "CAT_MONITORING_STILL_MOTION_THRESHOLD",
        "attr": ("AnomalyDetectionConfig", "STILL_MOTION_THRESHOLD"), "tab": "異常偵測與骨架品質",
        "label": "靜止動作門檻", "value_type": "float", "validate": "nonneg_float",
    },
    {
        "json_key": "anomaly_detection.enable_sqa_dual_judgment", "env_var": "CAT_MONITORING_ENABLE_SQA_DUAL_JUDGMENT",
        "attr": ("SQAConfig", "ENABLE_SQA_DUAL_JUDGMENT"), "tab": "異常偵測與骨架品質",
        "label": "啟用 SQA 雙重判定", "value_type": "bool", "validate": "bool",
    },
    # ── 執行模式與排程 ────────────────────────────────────────────────
    {
        "json_key": "run_mode.mode", "env_var": "CAT_MONITORING_RUN_MODE",
        "attr": ("RunModeConfig", "MODE"), "tab": "執行模式與排程",
        "label": "執行模式", "value_type": "enum", "validate": "enum",
        "choices": ["server", "gui"],
    },
    {
        "json_key": "run_mode.auto_start_processing", "env_var": "CAT_MONITORING_AUTO_START_PROCESSING",
        "attr": ("RunModeConfig", "AUTO_START_PROCESSING"), "tab": "執行模式與排程",
        "label": "啟動即自動處理", "value_type": "bool", "validate": "bool",
    },
    {
        "json_key": "run_mode.scheduled_start_time", "env_var": "CAT_MONITORING_SCHEDULED_START_TIME",
        "attr": ("RunModeConfig", "SCHEDULED_START_TIME"), "tab": "執行模式與排程",
        "label": "排程開始時間（HH:MM，留空=不啟用）", "value_type": "hhmm", "validate": "hhmm",
    },
    {
        "json_key": "run_mode.scheduled_end_time", "env_var": "CAT_MONITORING_SCHEDULED_END_TIME",
        "attr": ("RunModeConfig", "SCHEDULED_END_TIME"), "tab": "執行模式與排程",
        "label": "排程結束時間（HH:MM，留空=不設限）", "value_type": "hhmm", "validate": "hhmm",
    },
    # ── Flask 與 Node-RED ────────────────────────────────────────────
    {
        "json_key": "flask.host", "env_var": "CAT_MONITORING_FLASK_HOST",
        "attr": ("FlaskConfig", "HOST"), "tab": "Flask 與 Node-RED",
        "label": "Flask 主機位址", "value_type": "str", "validate": "str",
    },
    {
        "json_key": "flask.port", "env_var": "CAT_MONITORING_FLASK_PORT",
        "attr": ("FlaskConfig", "PORT"), "tab": "Flask 與 Node-RED",
        "label": "Flask Port", "value_type": "int", "validate": "port",
    },
    {
        "json_key": "flask.debug", "env_var": "CAT_MONITORING_FLASK_DEBUG",
        "attr": ("FlaskConfig", "DEBUG"), "tab": "Flask 與 Node-RED",
        "label": "Debug 模式（⚠ 生產環境請勿開啟）", "value_type": "bool", "validate": "bool",
    },
    {
        "json_key": "flask.threaded", "env_var": "CAT_MONITORING_FLASK_THREADED",
        "attr": ("FlaskConfig", "THREADED"), "tab": "Flask 與 Node-RED",
        "label": "Threaded", "value_type": "bool", "validate": "bool",
    },
    {
        "json_key": "flask.jpeg_quality", "env_var": "CAT_MONITORING_JPEG_QUALITY",
        "attr": ("FlaskConfig", "JPEG_QUALITY"), "tab": "Flask 與 Node-RED",
        "label": "JPEG 串流品質（1-100）", "value_type": "int", "validate": "jpeg_quality",
    },
    {
        "json_key": "flask.baseline_dashboard_enabled", "env_var": "CAT_MONITORING_BASELINE_DASHBOARD_ENABLED",
        "attr": ("BaselineDashboardConfig", "ENABLED"), "tab": "Flask 與 Node-RED",
        "label": "個體化基線儀表板啟用", "value_type": "bool", "validate": "bool",
    },
    {
        "json_key": "flask.baseline_dashboard_poll_interval_sec", "env_var": "CAT_MONITORING_BASELINE_DASHBOARD_POLL_INTERVAL",
        "attr": ("BaselineDashboardConfig", "POLL_INTERVAL_SEC"), "tab": "Flask 與 Node-RED",
        "label": "儀表板前端輪詢間隔（秒）", "value_type": "float", "validate": "positive_float",
    },
    {
        "json_key": "flask.baseline_dashboard_recompute_interval_sec", "env_var": "CAT_MONITORING_BASELINE_DASHBOARD_RECOMPUTE_INTERVAL",
        "attr": ("BaselineDashboardConfig", "RECOMPUTE_INTERVAL_SEC"), "tab": "Flask 與 Node-RED",
        "label": "儀表板背景重算間隔（秒）", "value_type": "float", "validate": "positive_float",
    },
    {
        "json_key": "nodered.host", "env_var": "CAT_MONITORING_NODERED_HOST",
        "attr": ("NodeRedConfig", "HOST"), "tab": "Flask 與 Node-RED",
        "label": "Node-RED 主機位址", "value_type": "str", "validate": "str",
    },
    {
        "json_key": "nodered.port", "env_var": "CAT_MONITORING_NODERED_PORT",
        "attr": ("NodeRedConfig", "PORT"), "tab": "Flask 與 Node-RED",
        "label": "Node-RED Port", "value_type": "int", "validate": "port",
    },
    {
        "json_key": "nodered.push_interval", "env_var": "CAT_MONITORING_NODERED_PUSH_INTERVAL",
        "attr": ("NodeRedConfig", "PUSH_INTERVAL"), "tab": "Flask 與 Node-RED",
        "label": "推送間隔（秒）", "value_type": "float", "validate": "positive_float",
    },
    {
        "json_key": "nodered.timeout", "env_var": "CAT_MONITORING_NODERED_TIMEOUT",
        "attr": ("NodeRedConfig", "TIMEOUT"), "tab": "Flask 與 Node-RED",
        "label": "逾時時間（秒）", "value_type": "float", "validate": "positive_float",
    },
    {
        "json_key": "nodered.global_context_path", "env_var": "CAT_MONITORING_NODERED_GLOBAL_CONTEXT_PATH",
        "attr": ("NodeRedConfig", "GLOBAL_CONTEXT_PATH"), "tab": "Flask 與 Node-RED",
        "label": "個體化基線共用儲存路徑（global.json）", "value_type": "file", "validate": "output_path",
    },
    # ── 行為追蹤與警報門檻 ────────────────────────────────────────────
    {
        "json_key": "behavior_tracking.max_history_size", "env_var": "CAT_MONITORING_MAX_HISTORY_SIZE",
        "attr": ("BehaviorTrackingConfig", "MAX_HISTORY_SIZE"), "tab": "行為追蹤與警報門檻",
        "label": "行為歷史保留筆數", "value_type": "int", "validate": "positive_int",
    },
    {
        "json_key": "behavior_tracking.activity_window_size", "env_var": "CAT_MONITORING_ACTIVITY_WINDOW_SIZE",
        "attr": ("BehaviorTrackingConfig", "ACTIVITY_WINDOW_SIZE"), "tab": "行為追蹤與警報門檻",
        "label": "活動力窗口大小（幀）", "value_type": "int", "validate": "positive_int",
    },
    {
        "json_key": "behavior_tracking.min_record_duration_seconds", "env_var": "CAT_MONITORING_MIN_RECORD_DURATION_SECONDS",
        "attr": ("BehaviorTrackingConfig", "MIN_RECORD_DURATION_SECONDS"), "tab": "行為追蹤與警報門檻",
        "label": "單一行為最短記錄秒數", "value_type": "float", "validate": "nonneg_float",
    },
    {
        "json_key": "behavior_tracking.activity_score_window_seconds", "env_var": "CAT_MONITORING_ACTIVITY_SCORE_WINDOW_SECONDS",
        "attr": ("BehaviorTrackingConfig", "ACTIVITY_SCORE_WINDOW_SECONDS"), "tab": "行為追蹤與警報門檻",
        "label": "活動分數取樣時間窗（秒）", "value_type": "float", "validate": "positive_float",
    },
    {
        "json_key": "behavior_tracking.low_confidence_activity_weight", "env_var": "CAT_MONITORING_LOW_CONFIDENCE_ACTIVITY_WEIGHT",
        "attr": ("BehaviorTrackingConfig", "LOW_CONFIDENCE_ACTIVITY_WEIGHT"), "tab": "行為追蹤與警報門檻",
        "label": "低信心幀活動權重", "value_type": "float", "validate": "unit_interval",
    },
    {
        "json_key": "behavior_tracking.stgcn_behavior_label_confidence_threshold", "env_var": "CAT_MONITORING_STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD",
        "attr": ("BehaviorTrackingConfig", "STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD"), "tab": "行為追蹤與警報門檻",
        "label": "ST-GCN 行為標籤輸出門檻", "value_type": "float", "validate": "unit_interval",
    },
    {
        "json_key": "behavior_tracking.display_hysteresis_windows.walk", "env_var": "CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_WALK",
        "attr": ("BehaviorTrackingConfig", "DISPLAY_HYSTERESIS_WINDOWS_WALK"), "tab": "行為追蹤與警報門檻",
        "label": "顯示延遲窗口數 - walk（<=1 關閉）", "value_type": "int", "validate": "nonneg_int",
    },
    {
        "json_key": "behavior_tracking.display_hysteresis_windows.lick", "env_var": "CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_LICK",
        "attr": ("BehaviorTrackingConfig", "DISPLAY_HYSTERESIS_WINDOWS_LICK"), "tab": "行為追蹤與警報門檻",
        "label": "顯示延遲窗口數 - lick（<=1 關閉）", "value_type": "int", "validate": "nonneg_int",
    },
    {
        "json_key": "behavior_tracking.display_hysteresis_windows.scratch", "env_var": "CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_SCRATCH",
        "attr": ("BehaviorTrackingConfig", "DISPLAY_HYSTERESIS_WINDOWS_SCRATCH"), "tab": "行為追蹤與警報門檻",
        "label": "顯示延遲窗口數 - scratch（<=1 關閉）", "value_type": "int", "validate": "nonneg_int",
    },
    {
        "json_key": "behavior_tracking.display_hysteresis_windows.shake", "env_var": "CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_SHAKE",
        "attr": ("BehaviorTrackingConfig", "DISPLAY_HYSTERESIS_WINDOWS_SHAKE"), "tab": "行為追蹤與警報門檻",
        "label": "顯示延遲窗口數 - shake（<=1 關閉）", "value_type": "int", "validate": "nonneg_int",
    },
    {
        "json_key": "behavior_tracking.display_hysteresis_windows.stop", "env_var": "CAT_MONITORING_DISPLAY_HYSTERESIS_WINDOWS_STOP",
        "attr": ("BehaviorTrackingConfig", "DISPLAY_HYSTERESIS_WINDOWS_STOP"), "tab": "行為追蹤與警報門檻",
        "label": "顯示延遲窗口數 - stop（<=1 關閉）", "value_type": "int", "validate": "nonneg_int",
    },
    {
        "json_key": "behavior_tracking.cat_missing_tolerance_frames", "env_var": "CAT_MONITORING_CAT_MISSING_TOLERANCE_FRAMES",
        "attr": ("BehaviorTrackingConfig", "CAT_MISSING_TOLERANCE_FRAMES"), "tab": "行為追蹤與警報門檻",
        "label": "貓消失容忍幀數（<=0 關閉此機制）", "value_type": "int", "validate": "int_any",
    },
    {
        "json_key": "behavior_tracking.scratch_alert_time_seconds", "env_var": "CAT_MONITORING_SCRATCH_ALERT_TIME_SECONDS",
        "attr": ("BehaviorTrackingConfig", "SCRATCH_ALERT_TIME_SECONDS"), "tab": "行為追蹤與警報門檻",
        "label": "搔抓警報累積秒數", "value_type": "float", "validate": "nonneg_float",
    },
    {
        "json_key": "behavior_tracking.scratch_alert_count_threshold", "env_var": "CAT_MONITORING_SCRATCH_ALERT_COUNT_THRESHOLD",
        "attr": ("BehaviorTrackingConfig", "SCRATCH_ALERT_COUNT_THRESHOLD"), "tab": "行為追蹤與警報門檻",
        "label": "搔抓警報次數", "value_type": "int", "validate": "nonneg_int",
    },
    {
        "json_key": "behavior_tracking.lick_alert_time_seconds", "env_var": "CAT_MONITORING_LICK_ALERT_TIME_SECONDS",
        "attr": ("BehaviorTrackingConfig", "LICK_ALERT_TIME_SECONDS"), "tab": "行為追蹤與警報門檻",
        "label": "舔舐警報累積秒數", "value_type": "float", "validate": "nonneg_float",
    },
    {
        "json_key": "behavior_tracking.shake_alert_count_threshold", "env_var": "CAT_MONITORING_SHAKE_ALERT_COUNT_THRESHOLD",
        "attr": ("BehaviorTrackingConfig", "SHAKE_ALERT_COUNT_THRESHOLD"), "tab": "行為追蹤與警報門檻",
        "label": "甩頭警報次數", "value_type": "int", "validate": "nonneg_int",
    },
    {
        "json_key": "behavior_tracking.stop_alert_time_seconds", "env_var": "CAT_MONITORING_STOP_ALERT_TIME_SECONDS",
        "attr": ("BehaviorTrackingConfig", "STOP_ALERT_TIME_SECONDS"), "tab": "行為追蹤與警報門檻",
        "label": "靜止警報累積秒數", "value_type": "float", "validate": "nonneg_float",
    },
    {
        "json_key": "behavior_tracking.low_activity_time_threshold_seconds", "env_var": "CAT_MONITORING_LOW_ACTIVITY_TIME_THRESHOLD_SECONDS",
        "attr": ("BehaviorTrackingConfig", "LOW_ACTIVITY_TIME_THRESHOLD_SECONDS"), "tab": "行為追蹤與警報門檻",
        "label": "低活動 walk 時長門檻（秒）", "value_type": "float", "validate": "nonneg_float",
    },
    # ── 貓咪身份驗證 ─────────────────────────────────────────────────
    {
        "json_key": "cat_identity.cat_id", "env_var": "CAT_MONITORING_CAT_ID",
        "attr": ("CatIdentityConfig", "CAT_ID"), "tab": "貓咪身份驗證",
        "label": "貓咪 ID", "value_type": "str", "validate": "str",
    },
    {
        "json_key": "cat_identity.enable_identity_verification", "env_var": "CAT_MONITORING_ENABLE_IDENTITY_VERIFICATION",
        "attr": ("CatIdentityConfig", "ENABLE_IDENTITY_VERIFICATION"), "tab": "貓咪身份驗證",
        "label": "啟用身份驗證", "value_type": "bool", "validate": "bool",
    },
    {
        "json_key": "cat_identity.target_cat_profile_path", "env_var": "CAT_MONITORING_TARGET_CAT_PROFILE_PATH",
        "attr": ("CatIdentityConfig", "TARGET_CAT_PROFILE_PATH"), "tab": "貓咪身份驗證",
        "label": "目標貓特徵基準檔", "value_type": "file", "validate": "optional_file_warn",
        "browse_filter": ("特徵基準檔", "*.json"),
    },
    {
        "json_key": "cat_identity.other_cat_profile_path", "env_var": "CAT_MONITORING_OTHER_CAT_PROFILE_PATH",
        "attr": ("CatIdentityConfig", "OTHER_CAT_PROFILE_PATH"), "tab": "貓咪身份驗證",
        "label": "其他已知貓特徵基準檔（可留空）", "value_type": "file", "validate": "optional_file_warn",
        "browse_filter": ("特徵基準檔", "*.json"),
    },
    # ── 日誌、CSV、資料庫與輸出路徑 ────────────────────────────────────
    {
        "json_key": "logging.tracker_state_path", "env_var": "CAT_MONITORING_TRACKER_STATE_PATH",
        "attr": ("LoggingConfig", "TRACKER_STATE_PATH"), "tab": "日誌、CSV、資料庫與輸出路徑",
        "label": "Tracker 狀態檔路徑", "value_type": "file", "validate": "output_path",
    },
    {
        "json_key": "logging.daily_history_db_path", "env_var": "CAT_MONITORING_DAILY_HISTORY_DB_PATH",
        "attr": ("LoggingConfig", "DAILY_HISTORY_DB_PATH"), "tab": "日誌、CSV、資料庫與輸出路徑",
        "label": "多天歷史 SQLite 路徑", "value_type": "file", "validate": "output_path",
    },
    {
        "json_key": "logging.csv_path", "env_var": "CAT_MONITORING_CSV_PATH",
        "attr": ("LoggingConfig", "CSV_PATH"), "tab": "日誌、CSV、資料庫與輸出路徑",
        "label": "主要 CSV 路徑", "value_type": "file", "validate": "output_path",
    },
    {
        "json_key": "logging.segments_csv_path", "env_var": "CAT_MONITORING_SEGMENTS_CSV_PATH",
        "attr": ("LoggingConfig", "SEGMENTS_CSV_PATH"), "tab": "日誌、CSV、資料庫與輸出路徑",
        "label": "行為區段 CSV 路徑", "value_type": "file", "validate": "output_path",
    },
    # ── 視覺化與串流顯示 ─────────────────────────────────────────────
    {
        "json_key": "visualization.stream_display_size", "env_var": "CAT_MONITORING_STREAM_DISPLAY_SIZE",
        "attr": ("VisualizationConfig", "STREAM_DISPLAY_SIZE"), "tab": "視覺化與串流顯示",
        "label": "串流縮放尺寸（留空=維持原始解析度）", "value_type": "size", "validate": "stream_size",
    },
    {
        "json_key": "visualization.clip_seconds", "env_var": "CAT_MONITORING_CLIP_SECONDS",
        "attr": ("VisualizationConfig", "CLIP_SECONDS"), "tab": "視覺化與串流顯示",
        "label": "Ring Buffer 秒數", "value_type": "int", "validate": "positive_int",
    },
    {
        "json_key": "visualization.show_nose_trapezoid", "env_var": "CAT_MONITORING_SHOW_NOSE_TRAPEZOID",
        "attr": ("VisualizationConfig", "SHOW_NOSE_TRAPEZOID"), "tab": "視覺化與串流顯示",
        "label": "顯示鼻子梯形 overlay", "value_type": "bool", "validate": "bool",
    },
    {
        "json_key": "visualization.show_bbox", "env_var": "CAT_MONITORING_SHOW_BBOX",
        "attr": ("VisualizationConfig", "SHOW_BBOX"), "tab": "視覺化與串流顯示",
        "label": "顯示偵測框 bbox（啟動預設）", "value_type": "bool", "validate": "bool",
    },
    {
        "json_key": "visualization.show_skeleton", "env_var": "CAT_MONITORING_SHOW_SKELETON",
        "attr": ("VisualizationConfig", "SHOW_SKELETON"), "tab": "視覺化與串流顯示",
        "label": "顯示骨架關鍵點（啟動預設）", "value_type": "bool", "validate": "bool",
    },
    {
        "json_key": "visualization.show_gcn_result", "env_var": "CAT_MONITORING_SHOW_GCN_RESULT",
        "attr": ("VisualizationConfig", "SHOW_GCN_RESULT"), "tab": "視覺化與串流顯示",
        "label": "顯示 GCN 分類結果（啟動預設）", "value_type": "bool", "validate": "bool",
    },
    # ── 進階設定：Node-RED 端點覆寫（預設由 HOST/PORT 推導） ──────────────
    {
        "json_key": "advanced.nodered_endpoint_notify", "env_var": "CAT_MONITORING_NODERED_ENDPOINT_NOTIFY",
        "attr": ("NodeRedConfig", "ENDPOINT_NOTIFY"), "tab": "進階設定",
        "label": "上線通知端點", "value_type": "str", "validate": "str",
    },
    {
        "json_key": "advanced.nodered_endpoint_result", "env_var": "CAT_MONITORING_NODERED_ENDPOINT_RESULT",
        "attr": ("NodeRedConfig", "ENDPOINT_RESULT"), "tab": "進階設定",
        "label": "推論結果端點（v1）", "value_type": "str", "validate": "str",
    },
    {
        "json_key": "advanced.nodered_endpoint_result_v2", "env_var": "CAT_MONITORING_NODERED_ENDPOINT_RESULT_V2",
        "attr": ("NodeRedConfig", "ENDPOINT_RESULT_V2"), "tab": "進階設定",
        "label": "推論結果端點（v2）", "value_type": "str", "validate": "str",
    },
]

TAB_ORDER = [
    "模型與輸入來源", "YOLO 推論", "ST-GCN 推論",
    "執行模式與排程", "Flask 與 Node-RED", "視覺化與串流顯示", "異常偵測與骨架品質",
    "行為追蹤與警報門檻", "貓咪身份驗證", "日誌、CSV、資料庫與輸出路徑", "進階設定",
]

# 敏感字串遮蔽：scheme://user:pass@host 型態的憑證，任何要印出來的訊息都先過這裡。
_CREDENTIAL_RE = re.compile(r"(://[^/@\s:]+:)[^/@\s]+(@)")


def _redact(text) -> str:
    """遮蔽字串中形如 rtsp://user:pass@host 的憑證，避免 GUI 日誌／錯誤視窗印出密碼。"""
    text = str(text)
    return _CREDENTIAL_RE.sub(r"\1***\2", text)


def _get_nested(data: dict, dotted_key: str):
    """依 dotted key 逐層查找；任何一層缺漏回傳 _MISSING 哨兵。"""
    node = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _set_nested(data: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


class _Missing:
    def __repr__(self):
        return "<MISSING>"


_MISSING = _Missing()


# ============================================================================
# 載入（含快取）
# ============================================================================

_cache = None  # type: dict | None
_cache_load_error = None  # type: str | None


def _load_json_file(path: Path) -> tuple[dict, str | None]:
    """讀取單一 JSON 檔案；不存在回傳 ({}, None)；讀取/格式錯誤回傳 ({}, 錯誤原因)。"""
    if not path.exists():
        return {}, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}, f"{path} 內容不是一個 JSON object"
        return data, None
    except Exception as e:
        return {}, f"讀取 {path} 失敗：{e}"


def load_runtime_settings(force_reload: bool = False) -> dict:
    """載入 runtime_settings.current.json，模組層快取。損壞/不存在都安全回傳 {}，絕不 raise。"""
    global _cache, _cache_load_error
    if _cache is not None and not force_reload:
        return _cache
    data, error = _load_json_file(RUNTIME_SETTINGS_PATH)
    _cache = data
    _cache_load_error = error
    if error:
        print(f"⚠ runtime_settings.current.json 讀取失敗，回退到環境變數／內建預設值：{error}")
    return _cache


def get_load_error() -> str | None:
    """回傳最近一次 load_runtime_settings() 的讀取錯誤原因；正常或檔案不存在時為 None。"""
    return _cache_load_error


def reload_runtime_settings() -> dict:
    """強制重新讀取（GUI「載入目前設定」用；config.py 正常流程不需要，第一版不熱重載）。"""
    return load_runtime_settings(force_reload=True)


def runtime_settings_exists() -> bool:
    return RUNTIME_SETTINGS_PATH.exists()


def get_runtime_settings_path() -> str:
    return str(RUNTIME_SETTINGS_PATH)


def get_last_modified():
    """回傳 runtime_settings.current.json 最後修改時間（datetime），檔案不存在回傳 None。"""
    if not RUNTIME_SETTINGS_PATH.exists():
        return None
    import datetime

    return datetime.datetime.fromtimestamp(RUNTIME_SETTINGS_PATH.stat().st_mtime)


# ============================================================================
# 供 config.py 使用：get_runtime_value() / get_runtime_size()
# ============================================================================


def get_runtime_value(json_key: str, fallback, value_type=None):
    """回傳 runtime_settings.current.json 中 dotted key 對應的值；沒有該欄位或型別不符時回退 fallback。

    型別檢查刻意寬鬆處理 int/float 混淆（JSON 數字寫成 5.0 這種情況），其餘型別
    不符就直接印警告、安全回退——這裡讀到的壞值最終會變成 config.py 的
    class attribute，寧可用內建預設值也不要讓奇怪型別流進主系統。
    """
    data = load_runtime_settings()
    value = _get_nested(data, json_key)
    if value is _MISSING:
        return fallback
    if value_type is None:
        return value
    if value_type is bool:
        if isinstance(value, bool):
            return value
    elif value_type is int:
        if isinstance(value, bool):
            pass
        elif isinstance(value, int):
            return value
        elif isinstance(value, float) and value.is_integer():
            return int(value)
    elif value_type is float:
        if isinstance(value, bool):
            pass
        elif isinstance(value, (int, float)):
            return float(value)
    elif value_type is str:
        if isinstance(value, str):
            return value
    else:
        return value
    print(
        f"⚠ runtime_settings.current.json 欄位 {json_key!r} 型別不符（預期 {value_type.__name__}，"
        f"實際 {type(value).__name__}），回退內建預設值"
    )
    return fallback


def get_runtime_size(json_key: str, fallback):
    """STREAM_DISPLAY_SIZE 專用：JSON 存 null 或 {"width":.., "height":..}，回傳 None 或 (w, h)。"""
    data = load_runtime_settings()
    value = _get_nested(data, json_key)
    if value is _MISSING or value is None:
        return fallback if value is _MISSING else None
    if (
        isinstance(value, dict)
        and isinstance(value.get("width"), int)
        and isinstance(value.get("height"), int)
        and value["width"] > 0
        and value["height"] > 0
    ):
        return (value["width"], value["height"])
    print(f"⚠ runtime_settings.current.json 欄位 {json_key!r} 格式不符，回退內建預設值")
    return fallback


def get_settings_source(json_key: str, env_var: str) -> str:
    """回傳 "env" / "json" / "default"，GUI 顯示「目前生效值的來源」用。"""
    if env_var and os.getenv(env_var) is not None:
        return "env"
    data = load_runtime_settings()
    if _get_nested(data, json_key) is not _MISSING:
        return "json"
    return "default"


# ============================================================================
# 驗證
# ============================================================================


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_positive_int(v, label):
    if isinstance(v, bool) or not isinstance(v, int):
        return f"{label}：必須是整數（目前為 {type(v).__name__}）"
    if v <= 0:
        return f"{label}：必須大於 0（目前為 {v}）"
    return None


def _validate_nonneg_int(v, label):
    if isinstance(v, bool) or not isinstance(v, int):
        return f"{label}：必須是整數（目前為 {type(v).__name__}）"
    if v < 0:
        return f"{label}：不可為負數（目前為 {v}）"
    return None


def _validate_int_any(v, label):
    if isinstance(v, bool) or not isinstance(v, int):
        return f"{label}：必須是整數（目前為 {type(v).__name__}）"
    return None


def _validate_positive_float(v, label):
    if not _is_number(v):
        return f"{label}：必須是數字（目前為 {type(v).__name__}）"
    if float(v) <= 0:
        return f"{label}：必須大於 0（目前為 {v}）"
    return None


def _validate_nonneg_float(v, label):
    if not _is_number(v):
        return f"{label}：必須是數字（目前為 {type(v).__name__}）"
    if float(v) < 0:
        return f"{label}：不可為負數（目前為 {v}）"
    return None


def _validate_unit_interval(v, label):
    if not _is_number(v):
        return f"{label}：必須是數字（目前為 {type(v).__name__}）"
    if not (0.0 <= float(v) <= 1.0):
        return f"{label}：必須介於 0 到 1 之間（目前為 {v}）"
    return None


def _validate_ema_alpha(v, label):
    if not _is_number(v):
        return f"{label}：必須是數字（目前為 {type(v).__name__}）"
    if not (0.0 < float(v) <= 1.0):
        return f"{label}：必須介於 0（不含）到 1 之間（目前為 {v}）"
    return None


def _validate_port(v, label):
    if isinstance(v, bool) or not isinstance(v, int):
        return f"{label}：必須是整數（目前為 {type(v).__name__}）"
    if not (1 <= v <= 65535):
        return f"{label}：必須介於 1 到 65535 之間（目前為 {v}）"
    return None


def _validate_jpeg_quality(v, label):
    if isinstance(v, bool) or not isinstance(v, int):
        return f"{label}：必須是整數（目前為 {type(v).__name__}）"
    if not (1 <= v <= 100):
        return f"{label}：必須介於 1 到 100 之間（目前為 {v}）"
    return None


_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def _validate_hhmm(v, label):
    if not isinstance(v, str):
        return f"{label}：必須是字串（目前為 {type(v).__name__}）"
    if v == "":
        return None
    if not _HHMM_RE.match(v):
        return f"{label}：格式須為 24 小時制 HH:MM（目前為 {_redact(v)!r}）"
    return None


def _validate_str(v, label):
    if not isinstance(v, str):
        return f"{label}：必須是字串（目前為 {type(v).__name__}）"
    return None


def _validate_bool(v, label):
    if not isinstance(v, bool):
        return f"{label}：必須是布林值（目前為 {type(v).__name__}）"
    return None


def _validate_enum(v, label, choices):
    if v not in choices:
        return f"{label}：必須是 {choices} 其中之一（目前為 {_redact(v)!r}）"
    return None


def _looks_like_url_or_camera_index(v) -> bool:
    if isinstance(v, int):
        return True
    if isinstance(v, str) and v.lower().startswith(("rtsp://", "http://", "https://")):
        return True
    return False


def _validate_required_file(v, label):
    if not isinstance(v, str) or not v.strip():
        return f"{label}：必須指定檔案路徑"
    if not Path(v).is_file():
        return f"{label}：找不到檔案 {_redact(v)}"
    return None


def _validate_optional_file_warn(v, label):
    """回傳 (error, warning)：檔案缺漏只警告，不擋存檔——本函式回傳 warning 訊息或 None。"""
    if v is None:
        return None
    if isinstance(v, int):
        return None  # video_input 的攝影機索引
    if not isinstance(v, str) or v.strip() == "":
        return None
    if _looks_like_url_or_camera_index(v):
        return None
    if not Path(v).exists():
        return f"{label}：路徑不存在（{_redact(v)}），啟用此功能前請確認檔案已備妥"
    return None


def _validate_output_path(v, label):
    """回傳 (error, warning)：output_path 規則——父目錄存在→ok；不存在但可建立→警告；不可建立→錯誤。"""
    if not isinstance(v, str) or not v.strip():
        return (f"{label}：必須指定路徑", None)
    parent = Path(v).parent
    if parent == Path("") or str(parent) in (".", ""):
        return (None, None)
    if parent.exists():
        if parent.is_dir():
            return (None, None)
        return (f"{label}：父路徑 {_redact(str(parent))} 已存在但不是資料夾", None)
    ancestor = parent
    while not ancestor.exists() and ancestor.parent != ancestor:
        ancestor = ancestor.parent
    if not ancestor.exists():
        return (f"{label}：路徑所在磁碟機／根目錄不存在（{_redact(v)}）", None)
    if os.access(ancestor, os.W_OK):
        return (None, f"{label}：父目錄尚不存在，執行時需自動建立（{_redact(str(parent))}）")
    return (f"{label}：父目錄不存在且無法建立（無寫入權限：{_redact(str(ancestor))}）", None)


def _validate_stream_size(v, label):
    if v is None:
        return None
    if not isinstance(v, dict):
        return f"{label}：必須是 null 或 {{width, height}}"
    w, h = v.get("width"), v.get("height")
    if isinstance(w, bool) or isinstance(h, bool) or not isinstance(w, int) or not isinstance(h, int):
        return f"{label}：寬高必須是整數"
    if w <= 0 or h <= 0:
        return f"{label}：寬高必須都大於 0（目前為 {w} x {h}）"
    return None


_SIMPLE_VALIDATORS = {
    "positive_int": _validate_positive_int,
    "nonneg_int": _validate_nonneg_int,
    "int_any": _validate_int_any,
    "positive_float": _validate_positive_float,
    "nonneg_float": _validate_nonneg_float,
    "unit_interval": _validate_unit_interval,
    "ema_alpha": _validate_ema_alpha,
    "port": _validate_port,
    "jpeg_quality": _validate_jpeg_quality,
    "hhmm": _validate_hhmm,
    "str": _validate_str,
    "bool": _validate_bool,
    "required_file": _validate_required_file,
}


def validate_settings(nested_data: dict) -> tuple[bool, list, list]:
    """驗證整份巢狀設定 dict。回傳 (是否全部通過, 錯誤清單, 警告清單)。

    只驗證 FIELD_SCHEMA 裡有出現、且這份 nested_data 實際有帶到的欄位——GUI
    存檔時一定帶完整表單，但匯入的 JSON 可能只有部分欄位，缺漏的欄位視為
    「沿用現況」，不因為沒帶到就報錯。
    """
    errors = []
    warnings = []
    for field in FIELD_SCHEMA:
        value = _get_nested(nested_data, field["json_key"])
        if value is _MISSING:
            continue
        rule = field["validate"]
        label = field["label"]
        if rule == "video_input":
            continue  # video_input 沒有獨立規則，型別即代表意義（int=攝影機索引/str=路徑或URL）
        if rule == "enum":
            err = _validate_enum(value, label, field.get("choices", []))
            if err:
                errors.append(err)
        elif rule == "optional_file_warn":
            warn = _validate_optional_file_warn(value, label)
            if warn:
                warnings.append(warn)
        elif rule == "output_path":
            err, warn = _validate_output_path(value, label)
            if err:
                errors.append(err)
            if warn:
                warnings.append(warn)
        elif rule == "stream_size":
            err = _validate_stream_size(value, label)
            if err:
                errors.append(err)
        else:
            validator = _SIMPLE_VALIDATORS.get(rule)
            if validator is None:
                continue
            err = validator(value, label)
            if err:
                errors.append(err)
    return (len(errors) == 0, errors, warnings)


# ============================================================================
# 儲存（原子寫入 + 備份）
# ============================================================================


def save_runtime_settings(nested_data: dict) -> tuple[bool, list, list]:
    """驗證後原子寫入 runtime_settings.current.json；驗證失敗完全不動檔案。

    儲存前若已有舊檔，先複製一份到 runtime_settings.previous.json；寫入走
    「同目錄暫存檔 + os.replace()」，避免程式中斷留下損壞的 JSON。
    """
    ok, errors, warnings = validate_settings(nested_data)
    if not ok:
        return (False, errors, warnings)

    RUNTIME_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RUNTIME_SETTINGS_PATH.exists():
        try:
            shutil.copyfile(RUNTIME_SETTINGS_PATH, BACKUP_SETTINGS_PATH)
        except OSError as e:
            warnings.append(f"備份舊設定檔失敗（仍會繼續儲存）：{e}")

    fd, tmp_path = tempfile.mkstemp(
        dir=str(RUNTIME_SETTINGS_PATH.parent), prefix=".runtime_settings_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(nested_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, RUNTIME_SETTINGS_PATH)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    global _cache
    _cache = nested_data
    return (True, [], warnings)


def export_settings(nested_data: dict, target_path) -> tuple[bool, list, list]:
    """驗證後另存一份 JSON 到任意路徑（匯出，不影響 runtime_settings.current.json）。"""
    ok, errors, warnings = validate_settings(nested_data)
    if not ok:
        return (False, errors, warnings)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(nested_data, f, ensure_ascii=False, indent=2)
    return (True, [], warnings)


def import_settings(source_path) -> tuple[bool, dict, list, list]:
    """讀取並驗證一份外部 JSON（匯入用）。回傳 (是否通過驗證, 內容 dict, 錯誤, 警告)。

    跟 load_runtime_settings() 不同：那裡「檔案不存在」是正常情況（代表尚未
    存過任何執行期設定，安全回退預設值）；這裡是使用者主動選了一個檔案要
    匯入，檔案不存在應視為明確的錯誤，不能悄悄回傳空結果。
    """
    path = Path(source_path)
    if not path.exists():
        return (False, {}, [f"找不到檔案：{_redact(str(path))}"], [])
    data, error = _load_json_file(path)
    if error:
        return (False, {}, [error], [])
    ok, errors, warnings = validate_settings(data)
    return (ok, data, errors, warnings)


def restore_defaults() -> dict:
    """讀取 default_runtime_settings.json，回傳 dict 供 GUI 填表用；不寫入 runtime_settings.current.json。"""
    data, error = _load_json_file(DEFAULT_RUNTIME_SETTINGS_PATH)
    if error:
        print(f"⚠ default_runtime_settings.json 讀取失敗：{error}")
    return data


def diff_settings(old: dict, new: dict) -> list:
    """回傳 [(json_key, label, old_value, new_value), ...]，僅列出實際不同的欄位（匯入預覽用）。"""
    changes = []
    for field in FIELD_SCHEMA:
        key = field["json_key"]
        old_v = _get_nested(old, key)
        new_v = _get_nested(new, key)
        old_v = None if old_v is _MISSING else old_v
        new_v = None if new_v is _MISSING else new_v
        if old_v != new_v:
            changes.append((key, field["label"], old_v, new_v))
    return changes


# ============================================================================
# default_runtime_settings.json 自動同步（跟 config.py 硬編碼字面值防漂移）
# ============================================================================


def _compute_pure_hardcoded_defaults() -> dict:
    """取得完全不受環境變數／runtime_settings.current.json 影響的「純硬編碼預設值」。

    做法：把 runtime_settings.current.json 暫時搬開、環境變數清成不含任何
    CAT_MONITORING_* 的乾淨副本，在子行程重新 import config，讀出
    FIELD_SCHEMA 每個 attr 當下的值——這時候 _runtime_default() 內部查
    runtime_settings.current.json 一定撲空、環境變數也一定沒設，回傳的就是 config.py
    寫死的那個字面值本身。跟 test_settings_manager.py 的一致性測試用同一招，
    刻意不用 importlib.reload(config)：同一行程內 reload 會產生新的 class
    物件，其他早就 import 過 config 的模組（這裡是 settings_manager 自己）
    會對不上，子行程隔離最乾淨。
    """
    moved_aside = None
    if RUNTIME_SETTINGS_PATH.exists():
        moved_aside = RUNTIME_SETTINGS_PATH.with_name(
            RUNTIME_SETTINGS_PATH.name + ".tmp_regen_hide"
        )
        RUNTIME_SETTINGS_PATH.rename(moved_aside)
    try:
        clean_env = {
            k: v for k, v in os.environ.items() if not k.startswith("CAT_MONITORING_")
        }
        script = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(_PAPER_DIR)!r})\n"
            "import config\n"
            "from settings_manager import FIELD_SCHEMA\n"
            "out = {}\n"
            "for f in FIELD_SCHEMA:\n"
            "    cls_name, attr_name = f['attr']\n"
            "    out[f['json_key']] = getattr(getattr(config, cls_name), attr_name)\n"
            "print(json.dumps(out))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(_PAPER_DIR),
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"子行程讀取 config.py 純預設值失敗：{result.stderr.strip()}")
        flat = json.loads(result.stdout.strip().splitlines()[-1])
    finally:
        if moved_aside is not None:
            if RUNTIME_SETTINGS_PATH.exists():
                RUNTIME_SETTINGS_PATH.unlink()
            moved_aside.rename(RUNTIME_SETTINGS_PATH)

    nested = {}
    for field in FIELD_SCHEMA:
        key = field["json_key"]
        value = flat.get(key)
        if field["value_type"] == "size" and isinstance(value, list):
            value = None if len(value) != 2 else {"width": value[0], "height": value[1]}
        _set_nested(nested, key, value)
    return nested


def regenerate_default_runtime_settings() -> tuple[bool, list]:
    """把 default_runtime_settings.json 重新對齊 config.py 目前的硬編碼預設值。

    回傳 (是否有變更, 變更清單)；變更清單格式同 diff_settings()。這支函式是
    「config.py 改了字面值 → default_runtime_settings.json 卻沒跟著動」這種
    漂移的正式解法，取代手動同步——config.py 的 `if __name__ == "__main__":`
    區塊會呼叫這裡，`python config.py` 就會順便把這份範本檔同步好。

    注意：**只動 default_runtime_settings.json（範本／還原用），不動
    runtime_settings.current.json（使用者目前實際套用中的執行期設定）**——後者可能
    存著使用者刻意透過 GUI 存下來的值，即使剛好跟舊的硬編碼預設值長得一樣，
    也不該被這支函式悄悄覆蓋掉。
    """
    new_defaults = _compute_pure_hardcoded_defaults()
    old_defaults, _ = _load_json_file(DEFAULT_RUNTIME_SETTINGS_PATH)
    changes = diff_settings(old_defaults, new_defaults)
    if not changes:
        return False, []

    DEFAULT_RUNTIME_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(DEFAULT_RUNTIME_SETTINGS_PATH.parent),
        prefix=".default_runtime_settings_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(new_defaults, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, DEFAULT_RUNTIME_SETTINGS_PATH)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return True, changes
