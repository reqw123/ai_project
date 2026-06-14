# 與 `main.py` / `config.py` 相關檔案：綁定（runtime）vs 獨立腳本

更新日期: 2026-05-05

說明：
- 「綁定（runtime）」表示該檔案為系統運行時必要或由 `main.py` / Flask app 直接或間接匯入、執行的模組。
- 「獨立腳本」表示主要用於訓練、測試、匯出或資料處理，可獨立呼叫，不是啟動服務時必須被載入的核心模組。

**綁定（Runtime / 被 main.py / Flask app 直接或間接使用）**
- `cat_monitoring_system/server/flask_app.py` : Flask app 工廠，啟動與註冊路由（必載）。
- `cat_monitoring_system/server/routes.py` : 定義 `/stream`、`/` 等路由（被 app 載入）。
- `cat_monitoring_system/server/streaming.py` : SharedFrameStreamer 與背景串流管理（被 routes / app 啟動使用）。
- `cat_monitoring_system/processors/frame_processor.py` : 幀級處理核心（YOLO→GCN→overlay→輸出），runtime pipeline 核心。
- `cat_monitoring_system/processors/anomaly_detector.py` : 異常檢測模組（frame_processor 使用）。
- `cat_monitoring_system/processors/visualizer.py` : 視覺化覆蓋層（frame_processor / streaming 使用）。
- `cat_monitoring_system/detectors/keypoint_detector.py` : YOLO-Pose 偵測封裝（frame_processor 與其他模組共用）。
- `cat_monitoring_system/detectors/behavior_classifier.py` : ST-GCN 分類器封裝（frame_processor 使用）。
- `cat_monitoring_system/models/stgcn_model.py` : ST-GCN 模型實作（行為分類 runtime 需要）。
- `cat_monitoring_system/trackers/behavior_tracker.py` : 行為追蹤與統計（Node-RED 輸出來源）。
- `cat_monitoring_system/logutils/csv_logger.py` : CSV 日誌寫入（runtime logging）。
- `cat_monitoring_system/communication/nodered_client.py` : Node-RED 通訊封裝（main.py 與 pipeline 可呼叫）。
- `cat_monitoring_system/utils/constants.py` : 共用顏色、骨架連線、行為名稱與低信心 sentinel 常量（visualizer / frame_processor / helpers 會用到）。
- `cat_monitoring_system/utils/helpers.py` : 常用工具（含 `get_ip()`，main.py 使用）。
- `config.py` : 全域設定與檢查（全系統共用設定中心）。
- `cat_monitoring_system/main.py` : 啟動點（當然屬於 runtime）。

**獨立腳本（可獨立執行 / 非 main 啟動必要）**
- `cat_monitoring_system/train_gcn.py` : 訓練 ST-GCN 的腳本（訓練用，非服務運行所需）。
- `cat_monitoring_system/test_video_inference_ema.py` : 推論測試（EMA 版本），離線驗證用。
- `cat_monitoring_system/test_video_inference_7ch.py` : 7 通道推論測試腳本，離線驗證用。
- `cat_monitoring_system/test2.py` : 測試/實驗腳本（獨立執行）。
- `cat_monitoring_system/export_keypoint_timeseries.py` : 匯出關鍵點時間序列（資料處理/匯出工具）。
- `cat_monitoring_system/dataset_collect.py` : 資料標註/收集工具（互動式或批次執行）。
- `cat_monitoring_system/measure_ear_distance_single_video.py` : 二階段部位判讀分析腳本（用於分析單支影片，獨立執行）。
- `cat_monitoring_system/measure_ear_distance_single_video copy.py` : 上項的複本（同為獨立腳本）。
- `cat_monitoring_system/count_video_frames.py` : 影片幀計數工具（獨立工具）。

**備註（灰色地帶）**
- 有些「獨立腳本」仍會匯入核心模組（如 `detectors.keypoint_detector`、`detectors.behavior_classifier`、`config.py` 等）來重用邏輯；但匯入不等同於被 `main.py` 所依賴。判斷依據是：啟動服務時（`main.py` → `create_app()` → routes/streaming）是否會載入該檔案。
- 若你想把某些獨立腳本納入服務（例如把 `measure_ear_distance_single_video.py` 改為可由 API 呼叫），可把它包成模組並由 `frame_processor` 或一個 background task 呼叫。

---

如果你要，我可以：
- 把這個分類結果合併回 `RELATED_TO_MAIN_CONFIG.md`，或
- 列出每個「綁定」檔案的主要類別/函式與它們被哪個模組呼叫（依賴圖）。

（已將此檔案儲存為 `MAIN_CONFIG_SCRIPT_CLASSIFICATION.md`）

## _append_test_

### 綁定檔案：主要類別 / 函式 與 被呼叫情況（1/2）

- `cat_monitoring_system/server/flask_app.py`
	- 主要符號：`create_app()`（建立 Flask 應用並註冊路由）
	- 被呼叫者：由 `cat_monitoring_system/main.py` 在啟動時呼叫。

- `cat_monitoring_system/server/routes.py`
	- 主要符號：`register_routes(app)`、路由處理函式如 `/stream`, `/api/behavior_history`, `/`, `/python_online`。
	- 呼叫關係：由 `create_app()` 透過 `register_routes()` 註冊。路由內會建立或使用 `SharedFrameStreamer`、`FrameProcessor` 與 `ImprovedBehaviorTracker`。
	- 其他依賴：`config.py`（讀取 ModelPaths / FlaskConfig / NodeRedConfig）、`server/streaming.py`、`processors/frame_processor.py`、`communication/nodered_client.py`、`models.stgcn_model.interpolate_missing()`、`utils.helpers.get_ip()`。

- `cat_monitoring_system/server/streaming.py`
	- 主要類別：`SharedFrameStreamer`（方法：`__init__`, `_update_frame`, `get_jpeg`, `stop`）
	- 被呼叫者：由 `routes.stream` 與 `routes._ensure_processor_started()` 在需要串流時建立。
	- 依賴：`FrameProcessor.process()`（每幀處理）、`config.py`（STGCNConfig、VisualizationConfig 參數）。

- `cat_monitoring_system/processors/frame_processor.py`
	- 主要類別：`FrameProcessor`（方法：`__init__`, `process`, `cleanup`）
	- 被呼叫者：`SharedFrameStreamer` 會呼叫 `process(frame)`；`routes._ensure_processor_started()` 建立此物件。
	- 內部使用／注入：`detectors.keypoint_detector.KeypointDetector`、`detectors.behavior_classifier.BehaviorClassifier`、`processors.anomaly_detector.AnomalyDetector`、`processors.visualizer.Visualizer`、`trackers.behavior_tracker.ImprovedBehaviorTracker`、`communication.nodered_client.NodeRedClient`、`logutils.csv_logger.CSVLogger`、`logutils.csv_logger.BehaviorSegmentLogger`。

- `cat_monitoring_system/processors/anomaly_detector.py`
	- 主要類別：`AnomalyDetector`（方法：`detect(kpts, kpt_conf)`）
	- 被呼叫者：由 `FrameProcessor.process()` 在每個有效關鍵點時調用以計算 `abnormal` 與 `activity_value`。

- `cat_monitoring_system/processors/visualizer.py`
	- 主要類別：`Visualizer`（方法：`draw`, `draw_prediction_on_frame`, `draw_probability_bars`）
	- 被呼叫者：`FrameProcessor` 在需要 overlay 時呼叫 `visualizer.draw(...)` 產生輸出影像。

- `cat_monitoring_system/detectors/keypoint_detector.py`
	- 主要類別：`KeypointDetector`（方法：`detect(frame)`）
	- 被呼叫者：`FrameProcessor.process()`、測試腳本（例如 `test_video_inference_*.py`）等直接呼叫以取得 17 點 keypoints 與 confidence。
	- 依賴：第三方模型（Ultralytics YOLO）；`config.py` 中的模型路徑可作為建構引數。

- `cat_monitoring_system/detectors/behavior_classifier.py`
	- 主要類別：`BehaviorClassifier`（方法：`classify(keypoints_sequence, precomputed=False)`），內部封裝 `models.stgcn_model.CatBehaviorSTGCN`。
	- 被呼叫者：`FrameProcessor` 在序列長度滿時呼叫進行行為推論；亦被離線推論腳本呼叫。

- `cat_monitoring_system/models/stgcn_model.py`
	- 主要符號：`interpolate_missing(sequence, conf)`、`build_feature_tensor(...)`、`CatBehaviorSTGCN` 類（`predict`, `normalize_keypoints`, `__call__`）及底層 `STGCN` 網路類別。
	- 被呼叫者：`FrameProcessor`（透過 `BehaviorClassifier`）與 `routes._get_latest_behavior()`（會呼叫 `interpolate_missing` 與 classifier）。

- `cat_monitoring_system/trackers/behavior_tracker.py`
	- 主要類別：`ImprovedBehaviorTracker`（方法：`update`, `get_activity_score`, `get_today_stats`, `get_alerts`, `add_alert`）
	- 被呼叫者：`FrameProcessor` 在每次推論後呼叫 `tracker.update(...)`；`routes.status` 與 `/api/behavior_history` 讀取 tracker 提供的統計資料。

- `cat_monitoring_system/logutils/csv_logger.py`
	- 主要類別：`CSVLogger.log(...)`, `BehaviorSegmentLogger.log_segment(...)`
	- 被呼叫者：`FrameProcessor` 在條件滿足時寫入逐幀或區段 CSV。

- `cat_monitoring_system/communication/nodered_client.py`
	- 主要類別：`NodeRedClient`（方法：`send_data(data)`），會先嘗試發送 `python_online` notify（如設定），再發送 `yolo_result`。
	- 被呼叫者：`FrameProcessor` 週期性呼叫以推送結果；`main.py` 亦在背景執行緒啟動時透過 HTTP 向 Node-RED 通知 Python 上線端點。

- `cat_monitoring_system/utils/helpers.py`
	- 主要函式：`get_ip()`（回傳本機 IP，`main.py` 與 `routes` 用於顯示/通知）。

- `cat_monitoring_system/utils/constants.py`
	- 主要內容：`WHITE` / `BLACK` / `GREEN` / `RED` / `BLUE` 等繪圖顏色，`HEAD_LINKS` / `BODY_LINKS` / `FRONT_LIMBS` / `HIND_LIMBS` / `TAIL_LINKS` 等骨架連線，`BEHAVIOR_CLASSES` / `BEHAVIOR_COLORS` / `BEHAVIOR_TEXT_MAP` / `BEHAVIOR_EMOJI_MAP` 等行為對照表，以及 `LOW_CONF_ID` / `LOW_CONF_TEXT` / `LOW_CONF_EMOJI` 這類低信心 sentinel 常量。
	- 被呼叫者：`frame_processor`、`visualizer`、`routes`、`helpers` 與測試腳本會直接匯入使用，用來統一畫面疊圖、骨架繪製與行為顯示。

- `config.py`
	- 主要內容：設定類別（`ModelPaths`, `YOLOConfig`, `STGCNConfig`, `FlaskConfig`, `NodeRedConfig`, `VisualizationConfig` 等）、`validate_all_config()`, `get_config_summary()`, `get_runtime_config_snapshot()`。
	- 被呼叫者：幾乎所有模組（routes、frame_processor、streaming、detectors、models 等）會讀取其屬性或環境變數覆寫值。

---

這段附加內容已直接寫入本檔案下方，若要我再將每個綁定檔案產生更詳細的呼叫圖（DOT / Mermaid），或把函式/類別的參數與回傳型別列成表格，我可以接著產生並附加在後面。

## Mermaid 依賴圖（模組間呼叫 / 匯入關係）

下面為專案中主要綁定模組之間的依賴關係圖（Mermaid 格式）。我已將此圖附加在本檔案，方便在支援 Mermaid 的檢視器中直接顯示。

```mermaid
graph LR
	main["main.py"]
	flask["server/flask_app.py\ncreate_app()"]
	routes["server/routes.py"]
	streaming["server/streaming.py\nSharedFrameStreamer"]
	frame["processors/frame_processor.py\nFrameProcessor"]
	keydet["detectors/keypoint_detector.py\nKeypointDetector"]
	behav["detectors/behavior_classifier.py\nBehaviorClassifier"]
	model["models/stgcn_model.py\nCatBehaviorSTGCN"]
	anomaly["processors/anomaly_detector.py\nAnomalyDetector"]
	viz["processors/visualizer.py\nVisualizer"]
	tracker["trackers/behavior_tracker.py\nImprovedBehaviorTracker"]
	csv["logutils/csv_logger.py\nCSVLogger / SegmentLogger"]
	nodered["communication/nodered_client.py\nNodeRedClient"]
	constants["utils/constants.py\nShared constants"]
	utils["utils/helpers.py\nget_ip()"]
	config["config.py\n(Flask/Model/Node-RED settings)"]

	main --> flask
	main --> nodered

	flask --> routes

	routes --> config
	routes --> streaming
	routes --> frame
	routes --> nodered
	routes --> tracker
	routes --> model
	routes --> utils

	streaming --> frame
	streaming --> config

	frame --> keydet
	frame --> behav
	frame --> anomaly
	frame --> viz
	frame --> tracker
	frame --> nodered
	frame --> csv
	frame --> model
	frame --> config
	frame --> constants

	behav --> model
	keydet --> "Ultralytics YOLO (external)"
	model --> config

	tracker --> config
	nodered --> config
	csv --> config
	viz --> config
	viz --> constants
	helpers --> constants

	style main fill:#f9f,stroke:#333,stroke-width:1px
	style config fill:#ffeb99,stroke:#333,stroke-width:1px
	style constants fill:#d9edf7,stroke:#333,stroke-width:1px
```

如果你需要，我可以：
- 把這個 Mermaid 圖也嵌到 `RELATED_TO_MAIN_CONFIG.md`，或
- 產生針對每個函式/類別的更詳細互動式圖表（Mermaid subgraphs 或 DOT + SVG）。