# AI 專案導覽地圖

> 本文件目的：讓任何 AI 助手（或新接手的人）在最短時間內建立「這個專案長什麼樣子、程式碼怎麼分工、Node-RED 那幾個 json 各自是做什麼的」的正確心智模型，並知道要去哪個檔案找答案。
>
> 內容以**實際原始碼現況**為準（校對日期：2026-07-02），刻意不重複程式碼細節，只給「地圖」——路徑、職責、彼此的呼叫關係。
>
> ⚠️ **本文件之外的舊文件已有過時內容**，見文末「九、舊文件可信度備註」，請先看那一節再決定要不要參考它們。

---

## 一、專案一句話說明

這是一套**貓咪行為辨識與健康監測系統**：YOLO-Pose 擷取貓咪骨架 → ST-GCN 分類行為（walk/lick/scratch/shake/stop）→ Python（Flask）產出即時串流與統計 → 多個 Node-RED flow 負責 Dashboard 呈現、健康評分、個體化基線比對、Discord/Messenger 通知。

---

## 二、五分鐘搞懂：建議依序閱讀的檔案

以下路徑皆相對於 `paper/`：

| 順序 | 檔案 | 為什麼看這個 |
|---|---|---|
| 1 | `config.py` | 所有可調參數的單一來源（模型路徑、閾值、Flask/Node-RED 設定） |
| 2 | `cat_monitoring_system/main.py` | 程式進入點，看懂啟動流程 |
| 3 | `cat_monitoring_system/server/routes.py` | Flask 路由總表，看懂 Python 對外暴露哪些 HTTP 端點、插件如何註冊 |
| 4 | `cat_monitoring_system/processors/frame_processor.py` | 整條資料流的實際編排者（YOLO→EMA→buffer→ST-GCN→統計→推送），是理解全系統的核心檔案 |
| 5 | `cat_monitoring_system/models/stgcn_model.py` | ST-GCN 模型本體與前處理函式（訓練/推論共用） |
| 6 | `0_進度彙整.md` | 論文進度與系統參數的敘事版本（已校對至現況） |
| 7 | 本文件的「五、監控層 Node-RED 對應」一節 | 看懂 4 個 json flow 各自負責什麼 |

---

## 三、Python 資料層模組化架構

程式碼位於 `cat_monitoring_system/`，依套件（資料夾）劃分職責：

### 3.1 入口與服務骨架

| 路徑 | 職責 |
|---|---|
| `cat_monitoring_system/main.py` | 程式進入點：建立 Flask app、啟動背景執行緒向 Node-RED 回報 Python IP（`POST /python_online`）、啟動 `app.run()` |
| `cat_monitoring_system/server/flask_app.py` | `create_app()`：Flask app factory，呼叫 `register_routes()` |
| `cat_monitoring_system/server/routes.py` | 定義所有 Flask 路由（見「六、Flask 端點總表」），並在首次請求時（`_ensure_processor_started`）建立 `FrameProcessor`、註冊 `LickStagePlugin`／`ExtBodyZonePlugin` |
| `cat_monitoring_system/server/streaming.py` | `SharedFrameStreamer`：管理 MJPEG 串流用的最新幀快取、client 計數、ring buffer（供 `/video_clip` 用） |

### 3.2 資料流層（Data Layer）— YOLO 骨架擷取

| 路徑 | 職責 |
|---|---|
| `cat_monitoring_system/detectors/keypoint_detector.py` | `KeypointDetector`：封裝 YOLO-Pose 推論，輸出 `kpts (17,2)` / `kpt_conf (17,)` / bbox / 偵測信心 |

### 3.3 模型流層（Model Layer）— ST-GCN 行為分類

| 路徑 | 職責 |
|---|---|
| `cat_monitoring_system/detectors/behavior_classifier.py` | `BehaviorClassifier`：ST-GCN 封裝，統一接收 `(T,V,2)` 原始座標 + `(T,V)` 信心值，內部依模型 `in_channels` 自動決定前處理路徑（14 關節截斷、補點、翻轉、正規化、特徵組裝皆在此觸發） |
| `cat_monitoring_system/models/stgcn_model.py` | ST-GCN 模型本體（`CatBehaviorSTGCN`、`JointAttention`、`SpatialGraphConv`、`MultiScaleTemporalConv`）與所有前處理純函式（`interpolate_missing`、`flip_normalize`、`orientation_normalize`、`normalize_skeleton_coords`、`build_feature_tensor`、`add_velocity_feature`、`compute_bone_feature` 等）；訓練腳本 `0_train_gcn.py` 與推論路徑共用同一份函式 |

### 3.4 服務流層（Service Layer）— 統計、記錄、推送、視覺化

| 路徑 | 職責 |
|---|---|
| `cat_monitoring_system/processors/frame_processor.py` | **整條 pipeline 的編排核心**：持有 `KeypointDetector`、`BehaviorClassifier`、`ImprovedBehaviorTracker`、`AnomalyDetector`、`Visualizer`、`NodeRedClient`、`CSVLogger`/`BehaviorSegmentLogger`，並管理已註冊插件（`self._plugins`）的 `update()`/`draw_overlay()`/`close()` 呼叫時機 |
| `cat_monitoring_system/trackers/behavior_tracker.py` | `ImprovedBehaviorTracker`：行為轉換偵測、時間累積、次數統計、`today_stats`、警報門檻判定 |
| `cat_monitoring_system/processors/anomaly_detector.py` | `AnomalyDetector`：以 body_fraction×100 正規化的關鍵點位移計算活動力分數，滾動均值判斷靜止；排除尾巴關節（14/15/16） |
| `cat_monitoring_system/processors/visualizer.py` | `Visualizer`：骨架連線、關鍵點、bbox、行為標籤、機率條的 overlay 繪製 |
| `cat_monitoring_system/communication/nodered_client.py` | `NodeRedClient`：非阻塞雙端點（v1 `/yolo_result` + v2 `/yolo_result_v2`）背景推送，各自獨立 daemon thread，佇列容量=1（drop-on-full，只送最新資料） |
| `cat_monitoring_system/logutils/csv_logger.py` | `CSVLogger`（逐幀/事件記錄）、`BehaviorSegmentLogger`（行為區段記錄，寫入 `behavior_segments_log.csv`） |
| `cat_monitoring_system/utils/constants.py` | 共用常量：骨架連線定義（`ALL_SKELETON`）、行為類別名稱/顏色/文字對照、低信心 sentinel 值 |
| `cat_monitoring_system/utils/helpers.py` | `get_ip()`、`get_behavior_name()` 等工具函式 |

### 3.5 外掛系統（Plugins）— 舔舐部位精細化分析

`FrameProcessor` 在每幀 YOLO 偵測完成後（ST-GCN 的 14 點截斷發生**之前**），會把完整未截斷的 17 點 `kpts`/`kpt_conf` 傳給所有已註冊插件的 `update()`；插件彼此獨立、互不依賴，可個別移除。

| 路徑 | 職責 | 詳細文件 |
|---|---|---|
| `cat_monitoring_system/plugins/lick_stage/` | `LickStagePlugin`：以鼻尖位置判斷貼近身體的區域（BODY/FL/FR/HL/HR），推論頭部朝向，結果 POST 至 Node-RED `/lick_zone_result` | `plugins/lick_stage/舔舐行為二階段分析模組說明.md`（已校對） |
| `cat_monitoring_system/plugins/lick_stage/ext_body_zones/` | `ExtBodyZonePlugin`：獨立姊妹插件，7 區身體分區偵測（HEAD/NECK_CHEST/SIDE_BACK/ABDOMEN/FORELIMB/HINDLIMB/TAIL），結果 POST 至 Node-RED `/ext_zone_result` | 無獨立說明文件，直接看 `plugin.py`/`regions.py` |

### 3.6 訓練 / 評估 / 離線工具腳本（非 runtime 必要）

這些腳本**不會**被 `main.py` 啟動的 Flask 服務呼叫，是研究方法（訓練、消融實驗、資料收集）用的獨立工具：

| 路徑 | 用途 |
|---|---|
| `cat_monitoring_system/0_train_gcn.py` | ST-GCN 訓練腳本主體 |
| `cat_monitoring_system/train_data/0_dataset_collect.py` | 骨架資料集收集與手動標注工具 |
| `cat_monitoring_system/1_eval_ema_ablation.py` | 不同 KP EMA alpha 消融實驗評估 |
| `cat_monitoring_system/1_eval_gcn_model.py` / `1_eval_pose_models.py` / `1_eval_model_worst_videos.py` | 模型評估腳本（GCN 模型評估、姿態模型評估、最差表現影片挑選） |
| `cat_monitoring_system/1_run_video_inference.py` | 單支影片離線推論 |
| `cat_monitoring_system/1_skeleton_visualizer.py` | 骨架視覺化腳本 |
| `cat_monitoring_system/1_export_keypoint_timeseries.py` / `1_measure_ear_distance_single_video.py` / `test_pose_jitter_analysis.py` / `test_anomaly_detection.py` / `1_visualize_activity_score.py` | 各類量測/除錯用的獨立分析腳本 |
| `cat_monitoring_system/1_classify_and_sort_videos.py` / `1_heic_av_png.py` / `1_多重命名.py` / `plugins/lick_stage/1_自動抓取.py` | 資料整理/格式轉換/爬蟲類雜項工具，與核心 pipeline 無程式碼依賴 |

---

## 四、資料流總覽（一句話版）

```
攝影機/影片 → YOLO-Pose(17點) → EMA平滑 → 時間序列buffer(T=16)
    → [ST-GCN 專用路徑] 14點截斷+補點+翻轉+正規化 → ST-GCN → 行為標籤
    → BehaviorTracker統計 / CSVLogger記錄 / NodeRedClient推送 / Visualizer疊圖
    → Flask /stream 串流輸出

（並行）YOLO原始17點 → LickStagePlugin / ExtBodyZonePlugin → 各自 POST 到 Node-RED
```

---

## 五、監控層 Node-RED 對應

Node-RED flow 檔案全部位於 `paper/`（**不在** `cat_monitoring_system/` 下）。目前共 4 個檔案：

### 5.1 `貓咪主控.json` —— 主控中心 / 核心健康監測 Dashboard

- 唯一 tab：「😺 貓咪健康監測系統」
- **接收**：`POST /python_online`（Python 上線通知）、`POST /yolo_result`（v1 行為推論資料）
- **對外呼叫**：Discord webhook（上線通知 + 健康風險告警，告警用的 webhook URL 讀自 `global.v2_user_settings.discord_webhook`）；組出 `http://<python_ip>:5000/stream` 供影像卡片使用
- **功能**：健康/風險評分引擎、CSV 寫入、Discord 告警、即時狀態卡片、行為時間軸、詳細統計、活動力儀表、影像串流卡片
- 這是**目前運行中的主要 Dashboard**，對應 `config.py` 的 `NodeRedConfig.ENDPOINT_NOTIFY`（`/python_online`）與 `ENDPOINT_RESULT`（`/yolo_result`）

### 5.2 `cat_health_v3_flow.json` —— 個體化基線分析引擎（v3 版）

四個 tab，構成一條分層 pipeline：

1. **第1層 核心資料流**：接收 `POST /yolo_result_v2`，累積行為統計，餵給 P1/P2 面板
2. **第2層 行為分析引擎**：行為分布 → 節律分析 → 偏差分析 → 健康預警引擎 → Discord 告警，餵給 P3/P4 面板
3. **第3層 基線引擎**：個體化正常行為基線計算（mean/std/median），寫入 `baseline.csv`/`daily_history.csv`/`deviation_log.csv`；也管理 `v2_user_settings`（含 Discord webhook 設定）
4. **定時任務**：每日午夜彙整、每小時偏差快照、系統啟動跨日清除、手動觸發基線重算

- **接收**：`POST /yolo_result_v2`（對應 `config.py` 的 `NodeRedConfig.ENDPOINT_RESULT_V2`）
- **對外呼叫**：Discord webhook（自動偏差告警 + 手動測試通知）
- 這就是使用者所稱的「**個體化基線**」模組——「先建立同一隻貓的長期正常行為基線（舔舐頻率、搔抓比例、活動量、休息比例、行為分布趨勢），再持續監測當前行為是否偏離基線」的核心邏輯即在這支 flow 的第2、3層

> `貓咪主控.json`（v1，`/yolo_result`）與 `cat_health_v3_flow.json`（v3，`/yolo_result_v2`）是**兩條並行的 flow**，Python 端 `NodeRedClient` 確實會同時推送 v1 與 v2 兩個端點（見 `communication/nodered_client.py`），兩者各自獨立運作、互不依賴。

### 5.3 `GPT 健康報告.json` —— Messenger 機器人 + GPT 健康報告（⚠️ 目前 disabled）

- 唯一 tab：「CSV AI分析系統」，**`"disabled": true`**——目前未在運行中的 Node-RED 實例內生效
- **接收**：`GET/POST /messengerwebhook`（Facebook Messenger webhook 驗證與事件接收）、`POST /ui-trigger-health`、`POST /ui-trigger-record`
- **對外呼叫**：`https://api.openai.com/v1/chat/completions`（GPT 生成健康報告）、`https://graph.facebook.com/v23.0/me/messages`（Messenger 回覆）、`http://<python_ip>:5000/video_clip`（`/camera` 指令觸發，**不是** `/snapshot`）、`http://<python_ip>:5000/status`（`/status` 指令觸發）
- **文字指令**：哈基米（觸發 GPT 分析）、`/camera`（錄 5 秒短片）、`/status`（查詢即時狀態）、`/help`
- ⚠️ **已知問題**：`/status` 指令呼叫的 Flask `/status` 路由**目前不存在**於 `server/routes.py`（只有 `/`, `/stream`, `/snapshot`, `/video_clip`, `/api/behavior_history`, `/api/overlay`），該指令會持續失敗

### 5.4 `lick_stage2_nodered.json` —— 舔舐部位分析 Dashboard

兩個 tab：

1. **第5層 舔舐部位分析**：接收 `POST /lick_zone_result`（`LickStagePlugin` 的輸出）與 `POST /lick_python_online`（獨立於主流程的 Python 上線通知，使用獨立的 `lick_python_ip` 全域變數），渲染關鍵指標 / 各區域時長 / 耳距與頭部朝向面板
2. **🦴 擴充身體區域 (7區)**：接收 `POST /ext_zone_result`（`ExtBodyZonePlugin` 的輸出），定期與舊版梯形區域資料合併顯示

- **接收**：`/lick_zone_result`、`/lick_python_online`、`/ext_zone_result`
- **對外呼叫**：無（純接收端，不主動呼叫外部服務）
- 這支 flow 對應 `plugins/lick_stage/config.py` 的 `NODERED_URL`（預設 `http://127.0.0.1:1880/lick_zone_result`）

---

## 六、Python ↔ Node-RED 端點總表

| Flask 端點（Python 提供） | 呼叫方 | 對應 Node-RED flow |
|---|---|---|
| `POST /python_online`（Node-RED 提供，Python 呼叫） | `main.py` 背景執行緒 | `貓咪主控.json` |
| `POST /yolo_result`（Node-RED 提供） | `NodeRedClient`（v1） | `貓咪主控.json` |
| `POST /yolo_result_v2`（Node-RED 提供） | `NodeRedClient`（v2） | `cat_health_v3_flow.json` |
| `POST /lick_zone_result`（Node-RED 提供） | `LickStagePlugin` → `NodeRedPublisher` | `lick_stage2_nodered.json` |
| `POST /ext_zone_result`（Node-RED 提供） | `ExtBodyZonePlugin` | `lick_stage2_nodered.json` |
| `GET /stream` | 各 flow 的影像卡片 | 全部 |
| `GET /snapshot` | （目前無 flow 使用；`GPT 健康報告.json` 改用 `/video_clip`） | — |
| `GET /video_clip` | `GPT 健康報告.json` 的 `/camera` 指令 | `GPT 健康報告.json` |
| `GET /status`（**不存在，會 404**） | `GPT 健康報告.json` 的 `/status` 指令 | `GPT 健康報告.json`（已知壞掉） |
| `GET/POST /api/overlay` | 目前無 flow 使用 | — |
| `GET /api/behavior_history` | 目前無 flow 使用 | — |

> 前三列的「提供方」寫反了方向：`/python_online`、`/yolo_result`、`/yolo_result_v2`、`/lick_zone_result`、`/ext_zone_result` 都是 **Node-RED 提供、Python 呼叫**的端點；其餘（`/stream`、`/snapshot`、`/video_clip`、`/api/*`）才是 **Python(Flask) 提供、Node-RED 呼叫**的端點。

---

## 七、config.py 與各層的對應關係

- `ModelPaths` / `YOLOConfig` / `STGCNConfig` → 影響「三、3.2/3.3」的資料流層與模型流層
- `AnomalyDetectionConfig` / `BehaviorTrackingConfig` → 影響 `AnomalyDetector`、`ImprovedBehaviorTracker`（3.4）
- `FlaskConfig` → 影響 `server/flask_app.py`、`server/routes.py`
- `NodeRedConfig` → 影響 `NodeRedClient` 推送的目標端點（對應「五、六」的 `/python_online`、`/yolo_result`、`/yolo_result_v2`）
- `LoggingConfig` → 影響 `CSVLogger`/`BehaviorSegmentLogger` 的輸出路徑
- `VisualizationConfig` → 影響 `Visualizer` 與 `LickStagePlugin.draw_overlay()` 的疊圖行為

lick_stage / ext_body_zones 插件**不吃** `config.py`，而是各自有獨立的 `plugins/lick_stage/config.py`／`plugins/lick_stage/ext_body_zones/config.py`。

---

## 八、目前已知問題（尚未修正）

1. `GPT 健康報告.json` 的 `/status` 指令呼叫 Flask 不存在的 `/status` 路由，會持續失敗（見 0_進度彙整.md 的已知問題註記）。
2. `GPT 健康報告.json` 整個 tab 目前是 `disabled: true`，Messenger/GPT 健康報告功能目前未啟用。
3. `貓咪主控.json`（v1/`yolo_result`）與 `cat_health_v3_flow.json`（v2/`yolo_result_v2`）各自維護獨立的 Discord webhook 設定與告警邏輯，未來若要合併兩條 flow 需注意告警可能重複觸發。

---

## 九、舊文件可信度備註

以下既有文件內容**已與現況不符**，閱讀時請以本文件與各模組內的最新說明（如 `plugins/lick_stage/舔舐行為二階段分析模組說明.md`、`0_進度彙整.md`）為準：

- `0_AI_HANDOFF_FOR_ASSISTANT.md`：多處引用不存在的檔案（`cat_monitoring_system/mermaid.md`、`THREE_LAYER_FLOW.md`、`NODERED_UPDATE_GUIDE.md`、`MAIN_CONFIG_SCRIPT_CLASSIFICATION.md`、`SCRIPT_SYNC_SUMMARY.md`、`flows (7).json`、`ip取得.json`），且參數表寫 `NUM_JOINTS=17`／`WINDOW_STRIDE(推論)=16`，與現行 `config.py`（`NUM_JOINTS=14`、`WINDOW_STRIDE` 預設 2）不符。
- `0_ARCHITECTURE_DESIGN.md`：架構圖與參數表同樣寫 `V=17`／`WINDOW_STRIDE(推論)=16`，與現況不符；其餘前處理管線順序描述仍正確可用。
- `NODE_RED_FUNCTIONS.md`（位於 `cat_monitoring_system/` 下）：引用 `cat_monitoring_system/node-red.json` 與 `gpt_api.json`，這兩個檔案**目前不存在於該路徑**（現行 Node-RED flow 全部搬到 `paper/` 根目錄下，且檔名已改為本文件「五」所列的 4 個檔案）。
- `貓咪個體化基線.md`：文件頭標註「對應檔案：`cat_health_v2_flow.json`」，但該檔案已不存在——現行對應檔案是 `cat_health_v3_flow.json`（v3）。內文的基線計算邏輯描述仍大致可參考，但檔名與部分流程細節建議以本文件「5.2」與實際 json 為準。

若要修正上述舊文件，建議另外開任務處理，避免與本次導覽文件的建立混在一起。
