# 與 `main.py` 和 `config.py` 相關的系統檔案清單

更新日期: 2026-05-05

以下列出本專案中與 `main.py`（啟動點）和 `config.py`（設定中心）直接相關的主要程式檔案與簡短說明，按邏輯分群（Server / Pipeline / 模型 / 支援）。

**Server（Flask + 串流 / API）**

- [cat_monitoring_system/main.py](cat_monitoring_system/main.py#L1) : 程式入口，建立 Flask app、取得本機 IP、啟動背景通知 Node-RED 的執行緒，並啟動 Flask。
- [cat_monitoring_system/server/flask_app.py](cat_monitoring_system/server/flask_app.py#L1) : Flask 應用工廠，註冊 Blueprint、初始化背景服務與單例物件。
- [cat_monitoring_system/server/routes.py](cat_monitoring_system/server/routes.py#L1) : 定義 `/`, `/stream`, `/status` 等 HTTP API 路由。
- [cat_monitoring_system/server/streaming.py](cat_monitoring_system/server/streaming.py#L1) : 背景串流與 SharedFrameStreamer，負責讀幀、維護 MJPEG 串流緩存並觸發 `FrameProcessor`。

**資料處理管線（Frame → Detect → Classify → Output）**

- [cat_monitoring_system/processors/frame_processor.py](cat_monitoring_system/processors/frame_processor.py#L1) : 幀級處理主入口，整合 Keypoint 檢測、序列緩衝、ST-GCN 推論、異常偵測、Overlay 繪製與輸出（stream / CSV / Node-RED）。
- [cat_monitoring_system/processors/anomaly_detector.py](cat_monitoring_system/processors/anomaly_detector.py#L1) : 異常與運動分析（EMA 平滑、motion score、abnormal flag）。
- [cat_monitoring_system/processors/visualizer.py](cat_monitoring_system/processors/visualizer.py#L1) : 視覺化覆蓋層：骨架、框、文字、顏色等。

**偵測與分類模組（模型封裝）**

- [cat_monitoring_system/detectors/keypoint_detector.py](cat_monitoring_system/detectors/keypoint_detector.py#L1) : YOLO-Pose（關鍵點檢測）封裝，產生 17 點 keypoints 與 confidence、bbox 等。
- [cat_monitoring_system/detectors/behavior_classifier.py](cat_monitoring_system/detectors/behavior_classifier.py#L1) : 行為分類器封裝（ST-GCN wrapper），提供 `classify()` / `predict()` 等介面。
- [cat_monitoring_system/models/stgcn_model.py](cat_monitoring_system/models/stgcn_model.py#L1) : ST-GCN 模型實作與載入、forward 與權重處理。

**追蹤、統計與日誌**

- [cat_monitoring_system/trackers/behavior_tracker.py](cat_monitoring_system/trackers/behavior_tracker.py#L1) : 行為統計、事件合併、歷史緩衝與活動力計算（對 `/status` 與 Node-RED 輸出提供資料）。
- [cat_monitoring_system/logutils/csv_logger.py](cat_monitoring_system/logutils/csv_logger.py#L1) : CSV 日誌寫入（`LoggingConfig.CSV_COLUMNS` 對應欄位）。

**通訊 / 外部整合**

- [cat_monitoring_system/communication/nodered_client.py](cat_monitoring_system/communication/nodered_client.py#L1) : 與 Node-RED 的 HTTP/JSON 通訊封裝（notify / result endpoints），`main.py` 會通知 Node-RED Python 服務上線。

**工具與輔助**

- [cat_monitoring_system/export_keypoint_timeseries.py](cat_monitoring_system/export_keypoint_timeseries.py#L1) : 匯出關鍵點時間序列以供離線分析或訓練資料生成（可被 pipeline 或獨立呼叫）。
- [cat_monitoring_system/utils/helpers.py](cat_monitoring_system/utils/helpers.py#L1) : 常用工具函式（包含 `get_ip()`，`main.py` 直接使用）。
- [config.py](config.py#L1) : 全域設定中心（環境變數覆寫、模型/路徑參數、Flask 與 Node-RED 設定、驗證函式）。

**測試 / 推論 / 訓練（與運行時有關）**

- [cat_monitoring_system/test_video_inference_ema.py](cat_monitoring_system/test_video_inference_ema.py#L1) : 推論測試腳本（EMA 平滑流程），用於驗證 ST-GCN 與 pipeline 行為分類效果。
- [cat_monitoring_system/test_video_inference_7ch.py](cat_monitoring_system/test_video_inference_7ch.py#L1) : 7 通道特徵的推論測試腳本（另一路驗證）。
- [cat_monitoring_system/train_gcn.py](cat_monitoring_system/train_gcn.py#L1) : ST-GCN 訓練腳本（與模型權重產生有關，影響行為分類品質）。

---

備註與讀取建議：

- 若要理解服務啟動流程，先讀 `main.py` → `server/flask_app.py` → `server/routes.py`。
- 若要理解資料流與推論，讀 `processors/frame_processor.py` → `detectors/keypoint_detector.py` → `detectors/behavior_classifier.py` → `models/stgcn_model.py`。
- 若要檢查外部整合（Node-RED），查看 `communication/nodered_client.py` 與 `config.py` 中 `NodeRedConfig` 設定。

本檔案已儲存於專案根目錄：`RELATED_TO_MAIN_CONFIG.md`。