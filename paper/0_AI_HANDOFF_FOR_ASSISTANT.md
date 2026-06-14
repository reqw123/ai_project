```mermaid
flowchart LR

    A["影片來源<br/>攝影機 RTSP 測試影片"] --> B["資料流層"]
    B --> C["模型流層"]
    C --> D["服務流層"]

    D --> E["Flask 網頁 API"]
    D --> F["Node-RED"]
    D --> G["CSV 日誌統計"]
    D --> H["即時影像串流"]

    subgraph L2["模型流層"]

        C1["ST-GCN 特徵輸入"]
        C2["ST-GCN 行為分類"]
        C3["分類機率 / 信心值"]
        C4["信心門檻判定"]
        C5["行為標籤（含低信心回退）"]

        C1 --> C2
        C2 --> C3
        C3 --> C4
        C4 --> C5

    end

    subgraph L1["資料流層"]

        B1["擷取 Frame"]
        B2["YOLO-Pose 關鍵點偵測"]
        B3["關鍵點座標 / 信心值"]
        B4["EMA 平滑（逐幀關鍵點）"]
        B5["時間序列 Buffer（累積多幀）"]
        B6["時序補點 / 對齊 / 正規化（ST-GCN 輸入）"]

        B1 --> B2
        B2 --> B3
        B3 --> B4
        B4 --> B5
        B5 --> B6

    end

    subgraph L3["服務流層"]

        D1["FrameProcessor 編排"]
        D2["BehaviorTracker 統計"]
        D3["CSVLogger 記錄"]
        D4["NodeRedClient 推送"]
        D5["Visualizer 疊圖"]
        D6["SharedFrameStreamer 串流"]

        D1 --> D2
        D2 --> D3
        D2 --> D4
        D1 --> D5
        D1 --> D6

        D6 --> Eresponse系統首先建立同一隻貓的長期正常行為軌跡，包括：

日常舔舐頻率
搔抓比例
活動量
休息比例
行為分布趨勢

作為其個體化正常行為模型。

後續系統再持續監測：

當前行為是否偏離歷史基線
是否出現持續性異常變化
是否出現行為比例失衡
是否出現活動模式改變

若偏離程度持續增加，則視為：

潛在異常行為
健康狀態可能變化
需進一步觀察之早期警示訊號

因此，本研究之核心目標並非直接診斷疾病，而是透過 YOLO-Pose 與 ST-GCN 建立長期、客觀且非侵入式的行為監測系統，協助飼主及後續專業人員更早發現可能異常之行為趨勢。
        D4 --> F
        D3 --> G
        D5 --> H

    end

    subgraph CFG["配置層"]

        K["config.py<br/>模型路徑 閾值 EMA ST-GCN Flask Node-RED"]

    end

    K -.-> B
    K -.-> C
    K -.-> D
```
# 貓咪行為辨識系統 - AI 技術交接文件

更新日期：2026-06-01

系統名稱:貓咪行為辨識系統

這份文件的目的，是讓下一個 AI 助手能快速接手目前專案，不需要重新從零摸索。內容盡量以「實作現況」為準，而不是只寫概念。

---

## 1. 專案目的

本專案是一套貓咪行為辨識系統，核心流程是：

1. 讀取影片來源或攝影機串流。
2. 使用 YOLO-Pose 擷取貓咪骨架關鍵點。
3. 對每幀關鍵點做 EMA 平滑。
4. 將連續幀放入時間序列 buffer。
5. 在送進 ST-GCN 前做序列補點、對齊與正規化。
6. 以 ST-GCN 輸出行為分類與信心值。
7. 再交給 BehaviorTracker、CSVLogger、Node-RED、Flask 串流與 overlay 顯示。

系統目前已把大部分重要參數統一集中在 [config.py](config.py)。

---

## 2. 建議優先閱讀的文件

如果要快速理解專案，請按下面順序看：

1. [config.py](config.py)
2. [cat_monitoring_system/main.py](cat_monitoring_system/main.py)
3. [cat_monitoring_system/server/routes.py](cat_monitoring_system/server/routes.py)
4. [cat_monitoring_system/processors/frame_processor.py](cat_monitoring_system/processors/frame_processor.py)
5. [cat_monitoring_system/models/stgcn_model.py](cat_monitoring_system/models/stgcn_model.py)
6. [cat_monitoring_system/processors/anomaly_detector.py](cat_monitoring_system/processors/anomaly_detector.py)
7. [cat_monitoring_system/detectors/keypoint_detector.py](cat_monitoring_system/detectors/keypoint_detector.py)
8. [cat_monitoring_system/trackers/behavior_tracker.py](cat_monitoring_system/trackers/behavior_tracker.py)
9. [cat_monitoring_system/communication/nodered_client.py](cat_monitoring_system/communication/nodered_client.py)
10. [cat_monitoring_system/utils/constants.py](cat_monitoring_system/utils/constants.py)
11. [cat_monitoring_system/mermaid.md](cat_monitoring_system/mermaid.md)
12. [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md)
13. [THREE_LAYER_FLOW.md](THREE_LAYER_FLOW.md)
14. [TRAINING_INFERENCE_GUIDE.md](TRAINING_INFERENCE_GUIDE.md)閾
15. [NODERED_UPDATE_GUIDE.md](NODERED_UPDATE_GUIDE.md)
16. [MAIN_CONFIG_SCRIPT_CLASSIFICATION.md](MAIN_CONFIG_SCRIPT_CLASSIFICATION.md)
17. [SCRIPT_SYNC_SUMMARY.md](SCRIPT_SYNC_SUMMARY.md)

---

## 3. 目前系統架構

### 3.1 三層架構

- 資料流層：Frame 擷取、YOLO-Pose、EMA 平滑、時間序列 buffer、補點與正規化前處理。
- 模型流層：ST-GCN 行為分類、異常分析、信心門檻判定。
- 服務流層：Flask /stream、Node-RED 推送、CSV 記錄、overlay 顯示。

### 3.2 實際資料順序

目前實作順序為：

1. 擷取 Frame
2. YOLO-Pose 關鍵點偵測
3. 關鍵點座標 / 信心值
4. EMA 平滑（逐幀關鍵點）
5. 時間序列 Buffer（累積多幀）
6. 時序補點 / 對齊 / 正規化（ST-GCN 輸入）
7. ST-GCN 分類
8. 行為標籤門檻判定
9. BehaviorTracker / CSV / Node-RED / overlay

這個順序和 [cat_monitoring_system/mermaid.md](cat_monitoring_system/mermaid.md) 的流程圖一致。

---

## 4. 目前已確認的關鍵參數

以下為目前實作中最重要、且已確認會影響行為辨識的設定：

| 項目 | 值 | 來源 |
|---|---:|---|
| YOLO 關鍵點數量 | 17 | [config.py](config.py) 的 `YOLOConfig.TOTAL_KEYPOINTS` |
| ST-GCN 時間窗長度 | 16 | [config.py](config.py) 的 `STGCNConfig.SEQUENCE_LENGTH` |
| ST-GCN 類別數 | 5 | [config.py](config.py) 的 `STGCNConfig.NUM_CLASSES` |
| ST-GCN 關節數 | 17 | [config.py](config.py) 的 `STGCNConfig.NUM_JOINTS` |
| ST-GCN 層數 | 3 | [config.py](config.py) 的 `STGCNConfig.NUM_LAYERS` |
| 預設特徵模式 | xy_v | [config.py](config.py) 的 `STGCNConfig.FEATURE_MODE` |
| xy_v 通道數 | 4 | `x, y, vx, vy` |
| xy_conf_v 通道數 | 5 | `x, y, conf, vx, vy` |
| xy_conf_v_bone 通道數 | 7 | `x, y, conf, vx, vy, bone_x, bone_y` |
| xy_conf_v_bone_bmotion 通道數 | 9 | `x, y, conf, vx, vy, bone_x, bone_y, bone_mx, bone_my` |
| 關鍵點 EMA 係數 | 1.0 | `STGCNConfig.KP_EMA_ALPHA` |
| 異常 motion EMA 係數 | 1.0 | `AnomalyDetectionConfig.EMA_ALPHA` |
| 關鍵點信心閾值 | 0.5 | `AnomalyDetectionConfig.KP_CONF_THRES` |
| 活動分數上限 | 20.0 | `AnomalyDetectionConfig.MAX_MOTION` |
| 行為標籤門檻 | 0.80 | `BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD` |

---

## 5. 模組責任對照

### 5.1 核心 runtime 模組

- [cat_monitoring_system/main.py](cat_monitoring_system/main.py)
  - 主入口。
  - 建立 Flask app。
  - 啟動背景執行緒通知 Node-RED Python IP。

- [cat_monitoring_system/server/flask_app.py](cat_monitoring_system/server/flask_app.py)
  - Flask app factory。

- [cat_monitoring_system/server/routes.py](cat_monitoring_system/server/routes.py)
  - Flask routes：`/`, `/stream`, `/python_online`。
  - 會建立 `FrameProcessor` 與 `SharedFrameStreamer`。

- [cat_monitoring_system/server/streaming.py](cat_monitoring_system/server/streaming.py)
  - 管理 MJPEG 串流與 frame 更新。

- [cat_monitoring_system/processors/frame_processor.py](cat_monitoring_system/processors/frame_processor.py)
  - 幀處理核心。
  - 負責 YOLO、EMA、buffer、AnomalyDetector、ST-GCN、BehaviorTracker、CSV、Node-RED、overlay。

- [cat_monitoring_system/detectors/keypoint_detector.py](cat_monitoring_system/detectors/keypoint_detector.py)
  - YOLO-Pose 封裝。

- [cat_monitoring_system/detectors/behavior_classifier.py](cat_monitoring_system/detectors/behavior_classifier.py)
  - ST-GCN 封裝。

- [cat_monitoring_system/models/stgcn_model.py](cat_monitoring_system/models/stgcn_model.py)
  - ST-GCN 模型本體、正規化、補點、特徵建構。

- [cat_monitoring_system/processors/anomaly_detector.py](cat_monitoring_system/processors/anomaly_detector.py)
  - 異常與活動度分析。

- [cat_monitoring_system/processors/visualizer.py](cat_monitoring_system/processors/visualizer.py)
  - 骨架與結果 overlay。

- [cat_monitoring_system/trackers/behavior_tracker.py](cat_monitoring_system/trackers/behavior_tracker.py)
  - 行為統計、歷史、警示。

- [cat_monitoring_system/logutils/csv_logger.py](cat_monitoring_system/logutils/csv_logger.py)
  - CSV logging。

- [cat_monitoring_system/communication/nodered_client.py](cat_monitoring_system/communication/nodered_client.py)
  - 與 Node-RED 通訊。

- [cat_monitoring_system/utils/constants.py](cat_monitoring_system/utils/constants.py)
  - 共用常量定義。
  - 提供畫框與骨架繪製用的顏色、骨架連線、行為類別名稱、行為顏色、行為文字對照、emoji 對照，以及低信心 sentinel 值。

- [cat_monitoring_system/utils/helpers.py](cat_monitoring_system/utils/helpers.py)
  - `get_ip()` 等工具。

### 5.2 資料流上的實際責任切分

- YOLO-Pose 之後就會拿到 `kpts` 與 `kpt_conf`。
- EMA 是先於 ST-GCN 前的逐幀平滑。
- `interpolate_missing()` 與 `flip_normalize / orientation_normalize / normalize_skeleton_coords` 是送進 ST-GCN 前的序列前處理。
- ST-GCN 的輸入形狀是 `(N, C, T, V)`。

### 5.3 ST-GCN 前處理中的「正規化」其實不只一項

這份專案裡的「正規化」不是單一的尺度縮放，而是由多個步驟組成，且訓練與推論的順序一致：

1. `interpolate_missing()`：先補時間序列中低信心或缺失的關鍵點。
2. `flip_normalize()`：先做左右翻轉對齊，讓骨架朝向一致。
3. `orientation_normalize()`：再把身體主軸旋轉到固定方向，減少姿態朝向差異。
4. `normalize_skeleton_coords()`：最後做中心化與尺度正規化，讓不同體型與畫面位置更可比。
5. `add_velocity_feature()`：若目前 feature mode 需要速度通道，則在正規化後再加入時間差分特徵。

因此，若要描述這段流程，建議不要只寫「尺度正規化」，而是寫成「補點 → 翻轉對齊 → 方向正規化 → 中心化 / 尺度正規化 → 速度特徵建構」。這樣才和 [cat_monitoring_system/models/stgcn_model.py](cat_monitoring_system/models/stgcn_model.py) 以及 [cat_monitoring_system/0_train_gcn.py](cat_monitoring_system/0_train_gcn.py) 的實作一致。

---

## 6. Node-RED 現況與注意事項

目前 Node-RED flow 已經整理成以「Python 上線通知 + 即時結果顯示 + Discord 告警」為主的 Dashboard 架構。核心設計如下：

### 6.1 Python 與 Node-RED 的資料交換

- Python 啟動後會向 Node-RED 的 `POST /python_online` 送出上線資訊，內容主要是 Python 端 IP。
- Node-RED 收到後會將 IP 存入 `global context`，並用 `http://<ip>:5000/stream` 組出影像串流位址。
- 後續 UI 只需要讀取全域儲存的 IP，不必再依賴 Python 重複推送完整 payload。
- Node-RED 另外透過 `POST /yolo_result` 接收 Python 的行為辨識資料，再分發到狀態卡、時間軸、統計、健康警示與儀表板。

### 6.2 Flow 內的主要節點

- `python_online` (`http in`)：接收 Python 上線通知。
- `parse_json` (`json`)：解析上線通知 payload。
- `build_response` (`function`)：整理 IP、寫入 `global python_ip`，並回傳串流 URL。
- `從持久化context恢復串流` (`inject` + `function`)：Node-RED 啟動時從持久化 context 恢復影像串流。
- `接收Python數據` (`http in`, 對應 `/yolo_result`)：接收 Python 推論結果。
- `解析JSON` (`json`)：解析推論結果資料。
- `數據分發器` (`function`)：補齊 `today_stats`、`current`、`behavior_log` 等欄位，並分流到各個 Dashboard 元件。
- `健康引擎分析` (`function`)：根據 today stats 計算 health score 與 alerts。
- `發送提醒` / `組成 Discord Webhook 格式`：將健康警示送到 Discord webhook。

### 6.3 Dashboard 組件對應

- `影像串流`：以 `msg.payload.ip` 或 `msg.stream_url` 顯示 Flask `:5000/stream` 影像。
- `即時狀態卡片`：顯示當前行為、時間戳、活動力分數與背景圖。
- `行為時間紀錄`：顯示近期行為時間軸。
- `詳細統計`：顯示 walk / lick / scratch / shake / stop 的次數與持續時間。
- `健康警示`：顯示 health score、警示內容與建議。
- `活動力儀表`：以 gauge 顯示 activity score。

### 6.4 資料契約與欄位

Node-RED flow 預期 Python 端送出的資料至少包含：

- `current`：當前狀態物件，例如 `behavior`、`text`、`emoji`、`timestamp`。
- `today_stats`：包含 `walk`、`walk_time`、`lick`、`lick_time`、`scratch`、`scratch_time`、`shake`、`shake_time`、`stop`、`stop_time`、`active_time`、`rest_time`。
- `behavior_log`：行為時間軸資料，至少包含 `behavior`、`time`、`duration`。
- `activity_score` / `health_score` / `alerts`：供儀表與健康警示面板使用。

目前 flow 內的 function 已經有防呆補值：如果缺少欄位，會自動補成預設值，避免 Dashboard 因 null 而中斷。

### 6.5 持久化 context 的前提

- 如果要在 Node-RED 重啟後仍保留 Python IP，`settings.js` 必須啟用 file-based context storage，也就是 `localfilesystem`。
- 若沒有啟用，`global.get("python_ip", "file")` 仍可讀取，但重啟後不會真正持久保存。

Node-RED 相關檔案：

- [cat_monitoring_system/flows (7).json](cat_monitoring_system/flows%20(7).json)
- [cat_monitoring_system/ip取得.json](cat_monitoring_system/ip取得.json)
- [NODERED_UPDATE_GUIDE.md](NODERED_UPDATE_GUIDE.md)

### 6.6 `/camera` 即時截圖功能（2026-06-03 新增）

Messenger 的 `/camera` 指令已從傳送靜態圖片改為擷取 Flask 當前串流幀。

**Flask 端**：`server/routes.py` 新增 `/snapshot` endpoint，呼叫 `SharedFrameStreamer.get_jpeg()` 回傳目前最新幀的 JPEG binary。若尚無可用幀則回傳 503。

**Node-RED 端**（`gpt_api.json`）：`/camera` 觸發後的完整流程：
```
攝影機畫面 → 取得截圖(GET /snapshot, binary)
    → 處理截圖
          out1 → 儲存截圖(file write, C:\a\snapshot_<時間戳>.jpg)
          out2 → 即時截圖卡片(ui_template, Dashboard 顯示 base64 圖片)
          out3 → FB API(Messenger 文字通知)
```

**Messenger echo 過濾**：`解析Messenger訊息` 已加入三道防護（`!event.message`、`is_echo`、`!text`），避免 bot 自身發出的訊息觸發重複執行。

### 6.7 目前需要留意的點

- `flows (7).json` 內的 Dashboard 元件大多已對齊現有 Python runtime，但仍建議在 Python 端資料格式變動時，同步檢查 `current`、`today_stats`、`behavior_log` 的欄位。
- 如果未來調整 ST-GCN 行為類別或顏色，Node-RED 的時間軸、統計卡與警示文字也要一起同步。
- `/snapshot` 回傳的是 `SharedFrameStreamer` 的最新快取幀，不會額外觸發影片讀取，對推論管線無效能影響。

---

## 7. 訓練 / 推論 / 測試相關文件

建議接手的人至少看過以下幾份：

- [TRAINING_INFERENCE_GUIDE.md](TRAINING_INFERENCE_GUIDE.md)
  - 訓練與推論環境變數、特徵模式、測試腳本用法。

- [SCRIPT_SYNC_SUMMARY.md](SCRIPT_SYNC_SUMMARY.md)
  - 訓練與推論腳本之間的同步結果。

- [MAIN_CONFIG_SCRIPT_CLASSIFICATION.md](MAIN_CONFIG_SCRIPT_CLASSIFICATION.md)
  - 哪些檔案是 runtime 綁定、哪些是獨立腳本。

---

## 8. 目前已完成的文件整理

以下文件已經可以直接拿來作為架構與交接參考：

- [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md)
- [THREE_LAYER_FLOW.md](THREE_LAYER_FLOW.md)
- [cat_monitoring_system/mermaid.md](cat_monitoring_system/mermaid.md)
- [TRAINING_INFERENCE_GUIDE.md](TRAINING_INFERENCE_GUIDE.md)
- [NODERED_UPDATE_GUIDE.md](NODERED_UPDATE_GUIDE.md)
- [MAIN_CONFIG_SCRIPT_CLASSIFICATION.md](MAIN_CONFIG_SCRIPT_CLASSIFICATION.md)
- [SCRIPT_SYNC_SUMMARY.md](SCRIPT_SYNC_SUMMARY.md)
- [cat_monitoring_system/INTEGRATION_SUMMARY.md](cat_monitoring_system/INTEGRATION_SUMMARY.md)

另外，Node-RED 的實際設計現在以 [cat_monitoring_system/flows (7).json](cat_monitoring_system/flows%20(7).json) 為準，後續若要改 Dashboard 或警示流程，建議直接以這份 flow 當主來源。

---

## 9. 已知待確認事項


1. 若要讓 Node-RED 的 IP 在重啟後也保留，需確認 `settings.js` 是否已啟用 `localfilesystem`。
2. 若未來修改 ST-GCN 特徵模式，請同步檢查：
   - `config.py`
   - `models/stgcn_model.py`
  - `0_train_gcn.py`
   - `test_video_inference_ema.py`
   - `test_video_inference_7ch.py`

---

## 10. 給下一個 AI 的工作建議

若要接手，建議優先順序如下：

1. 先讀 [config.py](config.py) 與 [cat_monitoring_system/processors/frame_processor.py](cat_monitoring_system/processors/frame_processor.py)。
2. 若要處理啟動問題，先查 [cat_monitoring_system/main.py](cat_monitoring_system/main.py) 的 exit code 1 原因。
3. 若要修改流程圖，先保持 [cat_monitoring_system/mermaid.md](cat_monitoring_system/mermaid.md) 的流程，再只調整附錄細節表。

---

## 11. 補充：這份交接文件的定位

這份文件不是原始論文內容，也不是使用手冊，而是「給 AI 接手專案用的實作導覽」。
因此內容刻意包含：

- 現況架構
- 核心參數
- 模組責任
- Node-RED 狀態
- 已知待確認事項

這樣下一個 AI 可以先建立全局理解，再去處理具體 bug、文件整理或功能修改。


### 12.3 雙模型對比評估腳本

`cat_monitoring_system/eval_model_four_videos.py` 已重新設計為**雙模型對比評估**：

- 同時評估兩個模型對同五支影片（walk / lick / scratch / shake / stop）
- 輸出指標：
  1. 離散準確率（argmax == true label）— 硬指標
  2. 真實類別平均機率（avg true-class probability）— 軟指標 / 信心度
  3. Overall Accuracy + Macro F1
  4. 混淆矩陣（row-wise recall 正規化顯示）
- 自動生成：`accuracy_comparison.png`（含贏家標示 ★ 與 Δ 差距）、`confusion_matrices.png`（並排）、`comparison_summary.csv`（含 winner 欄）
- 執行結束後在終端列印結構化分析（`print_final_summary`）：贏家判定、最大差距 class 提示、使用建議
- 輸出目錄採流水號：`eval_results/comparison_001_modelA_vs_modelB/`


