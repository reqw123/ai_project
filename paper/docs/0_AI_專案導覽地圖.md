# 📘 AI 專案導覽地圖

> [!IMPORTANT]
> **本文件目的**
>
> 讓任何 AI 助手（或新接手的人）在最短時間內建立「這個專案長什麼樣子、程式碼怎麼分工、Node-RED 那幾個 json 各自是做什麼的」的正確心智模型，並知道要去哪個檔案找答案。
>
> 內容以**實際原始碼現況**為準（校對日期：2026-07-15；2026-08-11 已核對並補上期間新增的模組/工具，見文中個別標註的日期，非全文重新校對），刻意不重複程式碼細節，只給「地圖」——路徑、職責、彼此的呼叫關係。
>
> 🚨 **本文件之外的舊文件已有過時內容**，見文末「九、舊文件可信度備註」，請先看那一節再決定要不要參考它們。

---

## 🚀 30 秒快速看懂（TL;DR）

> [!TIP]
> ### 如果只看一頁，請看這裡
>
> - 🎯 **一句話**：YOLO-Pose 擷取骨架 → ST-GCN 分類行為 → Flask 產出串流/統計 → 多個 Node-RED flow 負責 Dashboard/健康評分/通知。
> - 📂 **想快速上手**：照「二、五分鐘搞懂」表格的順序看 7 個檔案就夠。
> - 🏗️ **Python 端三層**：感知層（YOLO）→ 分析層（ST-GCN）→ 監測層（統計/記錄/推送/視覺化），見「三」。
> - 📡 **Node-RED 端 4 個 flow**：`貓咪主控.json`（主 Dashboard，運行中）、`cat_health_v3_flow.json`（個體化基線分析，運行中）、`GPT 健康報告.json`（**目前 disabled**）、`lick_stage2_nodered.json`（舔舐部位分析，運行中），見「五」。
> - 🚨 **看其他舊文件前，先看「九、舊文件可信度備註」**——好幾份舊文件的參數表/檔名已經跟現況不符。

---

## 📖 如何使用本文件（標示說明）

| 標示 | 意義 |
|:---:|---|
| 🚨 | 必看、容易踩坑、或已知過時處 |
| ⚠️ | 注意事項 |
| 💡 | 重點整理 |
| ✅ | 已修正/已完成 |
| 🔄 | 進行中/研究中功能 |
| 🧠 | 原理解析 |
| 📌 | 本節結論 |
| 📎 | 延伸閱讀／相關檔案 |

---

## 🧭 快速導航（目錄）

1. [🎯 一、專案一句話說明](#一專案一句話說明)
2. [📂 二、五分鐘搞懂：建議依序閱讀的檔案](#二五分鐘搞懂建議依序閱讀的檔案)
3. [🏗️ 三、Python 資料層模組化架構](#三python-資料層模組化架構)
   - [3.1 入口與服務骨架](#31-入口與服務骨架)
   - [3.2 感知層（Perception Layer）](#32-感知層perception-layer--yolo-骨架擷取)
   - [3.3 分析層（Analysis Layer）](#33-分析層analysis-layer--st-gcn-行為分類)
   - [3.3.1 🔄 Joint Prior Weights（研究中功能）](#331-joint-prior-weights關節先驗權重進行中的研究功能)
   - [3.4 監測層（Monitoring Layer）](#34-監測層monitoring-layer--統計記錄推送視覺化)
   - [3.4.1 🚨 Skeleton Quality Assessment](#341-skeleton-quality-assessment骨架品質雙重判定2026-07-新增2026-08-11-使用者確認改為預設開啟)
   - [3.5 外掛系統（Plugins）](#35-外掛系統plugins--舔舐部位精細化分析)
   - [3.6 訓練/評估/離線工具腳本](#36-訓練--評估--離線工具腳本非-runtime-必要)
4. [🔀 四、資料流總覽（一句話版）](#四資料流總覽一句話版)
5. [📡 五、監控層 Node-RED 對應](#五監控層-node-red-對應)
6. [🔌 六、Python ↔ Node-RED 端點總表](#六python--node-red-端點總表)
7. [⚙️ 七、config.py 與各層的對應關係](#七configpy-與各層的對應關係)
8. [🚨 八、目前已知問題（尚未修正）](#八目前已知問題尚未修正)
9. [🚨 九、舊文件可信度備註](#九舊文件可信度備註)

---

## 一、專案一句話說明

這是一套**貓咪行為辨識與健康監測系統**：YOLO-Pose 擷取貓咪骨架 → ST-GCN 分類行為（walk/lick/scratch/shake/stop）→ Python（Flask）產出即時串流與統計 → 多個 Node-RED flow 負責 Dashboard 呈現、健康評分、個體化基線比對、Discord/Messenger 通知。

---

## 二、五分鐘搞懂：建議依序閱讀的檔案

> 📖 **一句話摘要**：照順序看完這 7 項，就能理解「怎麼啟動 → 資料怎麼流 → 模型怎麼判斷 → Node-RED 那邊在幹嘛」。

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

> 📖 **一句話摘要**：程式碼在 `cat_monitoring_system/`，依套件分職責——入口、感知層、分析層、監測層、外掛、離線工具，共 6 大塊。

程式碼位於 `cat_monitoring_system/`，依套件（資料夾）劃分職責：

### 3.1 入口與服務骨架

| 路徑 | 職責 |
|---|---|
| `cat_monitoring_system/main.py` | 程式進入點：建立 Flask app、啟動背景執行緒向 Node-RED 回報 Python IP（`POST /python_online`）、啟動 `app.run()` |
| `cat_monitoring_system/server/flask_app.py` | `create_app()`：Flask app factory，呼叫 `register_routes()` |
| `cat_monitoring_system/server/routes.py` | 定義所有 Flask 路由（見「六、Flask 端點總表」），並在首次請求時（`_ensure_processor_started`）建立 `FrameProcessor`、註冊 `LickStagePlugin`／`ExtBodyZonePlugin` |
| `cat_monitoring_system/server/streaming.py` | `SharedFrameStreamer`：管理 MJPEG 串流用的最新幀快取、client 計數、ring buffer（供 `/video_clip` 用） |

### 3.2 感知層（Perception Layer）— YOLO 骨架擷取

| 路徑 | 職責 |
|---|---|
| `cat_monitoring_system/detectors/keypoint_detector.py` | `KeypointDetector`：封裝 YOLO-Pose 推論，輸出 `kpts (17,2)` / `kpt_conf (17,)` / bbox / 偵測信心 |

### 3.3 分析層（Analysis Layer）— ST-GCN 行為分類

| 路徑 | 職責 |
|---|---|
| `cat_monitoring_system/detectors/behavior_classifier.py` | `BehaviorClassifier`：ST-GCN 封裝，統一接收 `(T,V,2)` 原始座標 + `(T,V)` 信心值，內部依模型 `in_channels` 自動決定前處理路徑（14 關節截斷、補點、翻轉、正規化、特徵組裝皆在此觸發） |
| `cat_monitoring_system/models/stgcn_model.py` | ST-GCN 模型本體（`CatBehaviorSTGCN`、`JointAttention`、`SpatialGraphConv`、`MultiScaleTemporalConv`）與所有前處理純函式（`interpolate_missing`、`flip_normalize`、`orientation_normalize`、`normalize_skeleton_coords`、`build_feature_tensor`、`add_velocity_feature`、`compute_bone_feature` 等）；訓練腳本 `0_train_gcn.py` 與推論路徑共用同一份函式 |
| `cat_monitoring_system/models/keypoint_kalman.py`（2026-08 新增） | 逐關鍵點等速度 Kalman filter（`KeypointKalmanFilter`/`kalman_smooth_sequence`），跟 `interpolate_missing()` 同一個插入點的替代方案；透過 `stgcn_config.yaml` 的 `SMOOTHING_KIND: "kalman"` 選用。實測結論：推論期套用、單參數/三參數匹配重訓皆未找到顯著贏過無平滑的證據，目前 `stgcn_config.yaml` 已改回 `SMOOTHING_KIND: "ema"`，這個模組保留但未啟用於正式部署（詳見 `docs/YOLO-Pose應用文獻與專案優化建議.md`） |

### 3.3.1 Joint Prior Weights（關節先驗權重，進行中的研究功能）

> 🔄 **研究中功能**：架構層面強制放大特定關節訊號，測試對辨識率的影響，目前結果**尚未達統計顯著**。

- **設定**：`stgcn_config.yaml` 的 `USE_JOINT_PRIOR_WEIGHTS`（開關）/`JOINT_PRIOR_WEIGHTS`（`{關節名稱: 權重}`）；**目前設定為啟用**（`true`），權重為 `Nose:2.0, Left_Ear:1.5, Right_Ear:1.5, LF_Paw:1.5, RF_Paw:1.5`，其餘關節=1.0。
- **實作**：`models/stgcn_model.py` 的 `JointAttention`（`prior_weights` 以 `register_buffer` 存放，非學習參數，但隨 checkpoint 一起存/讀，推論端`eval_gcn_compare.py`/`1_run_video_inference.py` 完全不需額外設定即自動生效）；`0_train_gcn.py` 的 `_build_joint_prior_weights()` 負責讀取 config 並在訓練時建構這組權重。
- **原因**：`0_train_gcn.py` 的 `diagnose_keypoint_motion()`（逐關節動作幅度診斷）分析驗證集發現，lick/stop 正確分類樣本的判別訊號高度集中在 Nose（動作幅度差異是第二名 Right_Ear 的 2.5 倍以上，其餘關節幾乎無差異），但既有 `JointAttention` 只是逐關節獨立的 sigmoid gate（類似 SE-Net channel gate），關節間無互動、不是 Transformer/GAT 式的 self-attention，模型不一定會自己學會多看鼻子。
- **目的**：測試「架構層面強制放大特定關節訊號」是否能改善辨識率。
- **目前結果（三輪 McNemar 檢定，皆基於 `eval_gcn_compare.py` 獨立測試集）**：
  - `Nose:3.0`：stop 顯著改善（p=0.0000），但 scratch 顯著犧牲（p=0.0156）——真實 trade-off，非全面提升。
  - `Nose:5.0`：沒有換到額外好處（stop/scratch 皆無進一步改善），反而讓 lick 顯著崩潰在單一影片上（p=0.0002）——過頭了。
  - `Nose:2.0 + Left_Ear/Right_Ear:1.5 + LF_Paw/RF_Paw:1.5`（目前設定）：walk/scratch/shake 方向一致小幅改善、stop 小幅回檔，但 `n_discordant` 僅 2~5，**未達統計顯著**，需要多組 `RANDOM_SEED` 重複訓練＋彙總比較才能確認效果是否穩定。
  - 附註：scratch 類別的獨立測試集準確率偏低已確認主因是「搔抓片段佔比短、被大量非搔抓畫面稀釋」的評估方式問題（`max_true_prob`/`event_detected` 顯示模型仍抓到高信心事件），並非單純判別力不足，因此 scratch 相關的權重調整效果不能只看這個準確率數字判斷。

### 3.4 監測層（Monitoring Layer）— 統計、記錄、推送、視覺化

| 路徑 | 職責 |
|---|---|
| `cat_monitoring_system/processors/frame_processor.py` | **整條 pipeline 的編排核心**：持有 `KeypointDetector`、`BehaviorClassifier`、`ImprovedBehaviorTracker`、`AnomalyDetector`、`Visualizer`、`NodeRedClient`、`CSVLogger`/`BehaviorSegmentLogger`，並管理已註冊插件（`self._plugins`）的 `update()`/`draw_overlay()`/`close()` 呼叫時機 |
| `cat_monitoring_system/trackers/behavior_tracker.py` | `ImprovedBehaviorTracker`：行為轉換偵測、時間累積、次數統計、`today_stats`、警報門檻判定 |
| `cat_monitoring_system/processors/anomaly_detector.py` | `AnomalyDetector`：以 body_fraction×100 正規化的關鍵點位移計算活動力分數，滾動均值判斷靜止；排除尾巴關節（14/15/16） |
| `cat_monitoring_system/processors/skeleton_quality_assessment.py` | 「GCN 分類為主、幾何判斷為輔」雙重判定（見下方 3.4.1），純函式模組，不依賴 `FrameProcessor`/`config.py` |
| `cat_monitoring_system/processors/visualizer.py` | `Visualizer`：骨架連線、關鍵點、bbox、行為標籤、機率條的 overlay 繪製 |
| `cat_monitoring_system/communication/nodered_client.py` | `NodeRedClient`：非阻塞雙端點（v1 `/yolo_result` + v2 `/yolo_result_v2`）背景推送，各自獨立 daemon thread，佇列容量=1（drop-on-full，只送最新資料） |
| `cat_monitoring_system/logutils/csv_logger.py` | `CSVLogger`（逐幀/事件記錄）、`BehaviorSegmentLogger`（行為區段記錄，寫入 `behavior_segments_log.csv`） |
| `cat_monitoring_system/utils/constants.py` | 共用常量：骨架連線定義（`ALL_SKELETON`）、行為類別名稱/顏色/文字對照、低信心 sentinel 值 |
| `cat_monitoring_system/utils/helpers.py` | `get_ip()`、`get_behavior_name()` 等工具函式 |

### 3.4.1 Skeleton Quality Assessment（骨架品質雙重判定，2026-07 新增；2026-08-11 使用者確認改為預設開啟）

> 📖 **一句話摘要**：獨立於 ST-GCN 的幾何合理性檢查層，3 項指標判斷骨架窗口可不可信，不可信就覆蓋成 LOW_CONF；fail-safe，壞掉不影響主系統。

- **設計動機**：ST-GCN 分類器有時會對明顯不合理的骨架（YOLO 誤檢背景棉被/枕頭、關鍵點嚴重飄移）給出高信心的錯誤分類。這個模組不取代 ST-GCN，而是額外算一組「這個窗口的骨架幾何合不合理」的獨立訊號，當幾何判斷認為不可信時，把該幀的分類結果覆蓋成 `LOW_CONF`——即「GCN 分類為主、幾何判斷為輔」。
- **3 項指標**（皆從同一個 ST-GCN 推論窗口的原始關鍵點算出，不需要額外緩衝區）：
  - `midback_offset_ratio`：MidBack 偏離 Chest-Hip 虛擬中點的距離比例，超過解剖合理性上限視為可疑。
  - `midback_angle`：Chest-MidBack-Hip 夾角（取窗口最後一幀），太接近 180 度（幾乎共線）或太小（過尖）都視為可疑。
  - `body_axis_score_jitter`：身體主軸比例分數在窗口內的振幅，振幅過大代表骨架偵測反覆跳動、不穩定。
- **移植來源**：邏輯移植自 `cat_monitoring_system/tools/test_bone_length_stability.py`（模式2/GUI 視覺偵測）——那支腳本仍保留、持續用來肉眼校準門檻，跟這裡的正式整合版本共用同一套公式。
- **兩層開關**（刻意分開管理，保持低耦合）：
  - 總開關：`config.py` 的 `SQAConfig.ENABLE_SQA_DUAL_JUDGMENT`（**目前預設 `True`**——門檻值目前僅用少量影片校準過，套用前建議先用 `tools/test_bone_length_stability.py` 或 GUI 模式肉眼比對覆蓋規則是否合理）。
  - 個別指標開關：模組內的 `ENABLE_MIDBACK_OFFSET_CHECK`/`ENABLE_MIDBACK_ANGLE_CHECK`/`ENABLE_SCORE_JITTER_CHECK`，不在 `config.py` 設定。
- **Fail-safe 保證**：唯一對外函式 `evaluate_window()` 保證不拋出例外，任何內部錯誤都回傳「可信、不覆蓋」；`FrameProcessor` 呼叫端（`processors/frame_processor.py`）的 import 與呼叫也都各自包一層防護，即使這個模組完全壞掉或被刪除，也不會影響主系統其餘功能運行。

### 3.5 外掛系統（Plugins）— 舔舐部位精細化分析

> 📖 **一句話摘要**：兩個選用插件在 14 點截斷之前拿到完整 17 點資料，各自獨立、互不依賴，可個別移除。

`FrameProcessor` 在每幀 YOLO 偵測完成後（ST-GCN 的 14 點截斷發生**之前**），會把完整未截斷的 17 點 `kpts`/`kpt_conf` 傳給所有已註冊插件的 `update()`；插件彼此獨立、互不依賴，可個別移除。

| 路徑 | 職責 | 詳細文件 |
|---|---|---|
| `cat_monitoring_system/plugins/lick_stage/` | `LickStagePlugin`：以鼻尖位置判斷貼近身體的區域（BODY/FL/FR/HL/HR），推論頭部朝向，結果 POST 至 Node-RED `/lick_zone_result` | `plugins/lick_stage/舔舐行為二階段分析模組說明.md`（已校對） |
| `cat_monitoring_system/plugins/lick_stage/ext_body_zones/` | `ExtBodyZonePlugin`：獨立姊妹插件，7 區身體分區偵測（HEAD/NECK_CHEST/SIDE_BACK/ABDOMEN/FORELIMB/HINDLIMB/TAIL），結果 POST 至 Node-RED `/ext_zone_result` | 無獨立說明文件，直接看 `plugin.py`/`regions.py` |

### 3.6 訓練 / 評估 / 離線工具腳本（非 runtime 必要）

> 📖 **一句話摘要**：`tools/` 底下一大堆訓練/消融/校準/資料整理腳本，都不會被 Flask 服務呼叫，是研究方法用的獨立工具。

這些腳本**不會**被 `main.py` 啟動的 Flask 服務呼叫，是研究方法（訓練、消融實驗、資料收集）用的獨立工具：

2026-07 已把這批腳本全部搬進 `cat_monitoring_system/tools/`（核心模組資料夾底下不再混雜非模組腳本），下表路徑已同步更新：

| 路徑 | 用途 |
|---|---|
| `cat_monitoring_system/tools/0_train_gcn.py` | ST-GCN 訓練腳本主體 |
| `cat_monitoring_system/tools/train_data/0_dataset_collect.py` | 骨架資料集收集與手動標注工具，共 7 種模式（1~5 為訓練資料，6~7 為 2026-08-11 新增的獨立測試集標註/檢視工具，輸出到獨立的 `TEST_OUTPUT_FOLDER`，不會混進訓練用 `skeletons/`） |
| `cat_monitoring_system/tools/eval_ema_ablation.py` | 不同 KP EMA alpha 消融實驗評估 |
| `cat_monitoring_system/tools/eval_gcn_compare.py` / `eval_pose_compare.py` / `eval_model_worst_videos.py` | 模型評估腳本（GCN 模型評估、姿態模型評估、最差表現影片挑選）—— 2026-07 由 `eval_gcn_model.py`/`eval_pose_models.py` 改名而來 |
| `cat_monitoring_system/tools/eval_class_source_distribution.py` | 檢查各行為類別的訓練樣本是否過度集中於少數影片/場景（過擬合假象排查） |
| `cat_monitoring_system/tools/eval_lighting_distribution.py`（2026-08-11 新增） | 抽樣訓練影片幾幀算 HSV V channel 均值當亮度代理指標，統計各行為類別的光照分布、記錄快照供之後比對；實測發現訓練資料幾乎不含低光照場景（501支僅1.2%落在「很暗」區間） |
| `cat_monitoring_system/tools/2_run_dual_model_compare.py` | 即時視覺化並排比較兩個模型的推論結果（疊圖顯示），跟 `eval_gcn_compare.py`（數字/統計量比較）用途不同、互補 |
| `cat_monitoring_system/tools/3_cat_identity_verification_test.py` | 貓咪個體身分辨識測試腳本 |
| `cat_monitoring_system/tools/test_bone_length_stability.py` | 骨架穩定度診斷/校準工具（骨段長度一致性、脊椎中點偏移/角度、Body Axis score jitter 等指標的離線分析與正常基線建立，`processors/skeleton_quality_assessment.py` 的邏輯即由此腳本的模式 2 移植而來） |
| `cat_monitoring_system/tools/1_run_video_inference.py` | 單支影片離線推論 |
| `cat_monitoring_system/tools/1_skeleton_visualizer.py` | 骨架視覺化腳本 |
| `cat_monitoring_system/tools/1_visualize_three_normalizations.py` | 互動式 Demo：對照 flip_normalize / orientation_normalize / normalize_skeleton_coords 三種正規化步驟的視覺效果 |
| `cat_monitoring_system/tools/1_visualize_interpolation.py` | 視覺化 `interpolate_missing()` 補點前後的差異 |
| `cat_monitoring_system/tools/run_keypoint_trend_from_videos.py` | 直接對原始影片重新推論，檢視單一行為類別各關鍵點動作幅度與跨影片趨勢（不依賴既有訓練資料集） |
| `cat_monitoring_system/tools/1_export_keypoint_timeseries.py` / `1_measure_ear_distance_single_video.py` / `test_pose_jitter_analysis.py` / `test_anomaly_detection.py` / `1_visualize_activity_score.py` | 各類量測/除錯用的獨立分析腳本 |
| `cat_monitoring_system/tools/1_classify_and_sort_videos.py` / `1_heic_av_png.py` / `1_多重命名.py` / `1_自動抓取.py` / `影片拼接.py` | 資料整理/格式轉換/爬蟲/影片合併類雜項工具，與核心 pipeline 無程式碼依賴（`1_自動抓取.py` 2026-07 已由 `plugins/lick_stage/` 移入此資料夾，不再是 plugin 專屬工具） |

> [!WARNING]
> 🚨 `run_keypoint_verification.bat` 呼叫的 `1_check_keypoint_importance.py` 目前已不存在於專案中（僅剩 `__pycache__` 殘留的 `.pyc`），此 `.bat` 目前已失效、待清理或補回對應腳本。

---

## 四、資料流總覽（一句話版）

> 📖 **一句話摘要**：主線是「攝影機→YOLO→ST-GCN→行為標籤→統計/記錄/推送/串流」，插件走另一條並行的原始 17 點路徑。

```
攝影機/影片 → YOLO-Pose(17點) → EMA平滑 → 時間序列buffer(T=16)
    → [ST-GCN 專用路徑] 14點截斷+補點+翻轉+正規化 → ST-GCN → 行為標籤
    → BehaviorTracker統計 / CSVLogger記錄 / NodeRedClient推送 / Visualizer疊圖
    → Flask /stream 串流輸出

（並行）YOLO原始17點 → LickStagePlugin / ExtBodyZonePlugin → 各自 POST 到 Node-RED
```

---

## 五、監控層 Node-RED 對應

> 📖 **一句話摘要**：4 個 flow 檔案都放在 `paper/`（不在 `cat_monitoring_system/`），其中 `GPT 健康報告.json` 目前 disabled，其餘 3 個都在運行中。

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

> [!NOTE]
> `貓咪主控.json`（v1，`/yolo_result`）與 `cat_health_v3_flow.json`（v3，`/yolo_result_v2`）是**兩條並行的 flow**，Python 端 `NodeRedClient` 確實會同時推送 v1 與 v2 兩個端點（見 `communication/nodered_client.py`），兩者各自獨立運作、互不依賴。

### 5.3 `GPT 健康報告.json` —— Messenger 機器人 + GPT 健康報告（🚨 目前 disabled）

- 唯一 tab：「CSV AI分析系統」，**`"disabled": true`**——目前未在運行中的 Node-RED 實例內生效
- **接收**：`GET/POST /messengerwebhook`（Facebook Messenger webhook 驗證與事件接收）、`POST /ui-trigger-health`、`POST /ui-trigger-record`
- **對外呼叫**：`https://api.openai.com/v1/chat/completions`（GPT 生成健康報告）、`https://graph.facebook.com/v23.0/me/messages`（Messenger 回覆）、`http://<python_ip>:5000/video_clip`（`/camera` 指令觸發，**不是** `/snapshot`）
- **文字指令**：哈基米（觸發 GPT 分析）、`/camera`（錄 5 秒短片）、`/help`
- ✅ 2026-07 更新：原本的 `/status` 指令呼叫 Flask 端不存在的 `/status` 路由、會持續失敗，已將該指令從「指令分流」switch、對應的請求/回覆節點、以及 `/help` 說明文字中整個移除

### 5.4 `lick_stage2_nodered.json` —— 舔舐部位分析 Dashboard

兩個 tab：

1. **第5層 舔舐部位分析**：接收 `POST /lick_zone_result`（`LickStagePlugin` 的輸出）與 `POST /lick_python_online`（獨立於主流程的 Python 上線通知，使用獨立的 `lick_python_ip` 全域變數），渲染關鍵指標 / 各區域時長 / 耳距與頭部朝向面板
2. **🦴 擴充身體區域 (7區)**：接收 `POST /ext_zone_result`（`ExtBodyZonePlugin` 的輸出），定期與舊版梯形區域資料合併顯示

- **接收**：`/lick_zone_result`、`/lick_python_online`、`/ext_zone_result`
- **對外呼叫**：無（純接收端，不主動呼叫外部服務）
- 這支 flow 對應 `plugins/lick_stage/config.py` 的 `NODERED_URL`（預設 `http://127.0.0.1:1880/lick_zone_result`）

> 📌 **本節結論**：4 個 flow 各司其職，`貓咪主控.json`／`cat_health_v3_flow.json` 並行運作互不依賴，`lick_stage2_nodered.json` 純接收插件資料，`GPT 健康報告.json` 目前停用。

---

## 六、Python ↔ Node-RED 端點總表

> 📖 **一句話摘要**：Node-RED 提供的端點是 Python 主動呼叫，Flask 提供的端點是 Node-RED 主動呼叫——方向不要搞反。

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
| `GET/POST /api/overlay` | 目前無 flow 使用 | — |
| `GET /api/behavior_history` | 目前無 flow 使用 | — |

> [!WARNING]
> ⚠️ 前三列的「提供方」寫反了方向：`/python_online`、`/yolo_result`、`/yolo_result_v2`、`/lick_zone_result`、`/ext_zone_result` 都是 **Node-RED 提供、Python 呼叫**的端點；其餘（`/stream`、`/snapshot`、`/video_clip`、`/api/*`）才是 **Python(Flask) 提供、Node-RED 呼叫**的端點。

---

## 七、config.py 與各層的對應關係

> 📖 **一句話摘要**：`config.py` 底下每個 Config class 對應一到多個模組；lick_stage/ext_body_zones 插件不吃這份，各自有獨立設定檔。

- `ModelPaths` / `YOLOConfig` / `STGCNConfig` → 影響「三、3.2/3.3」的感知層與分析層
- `AnomalyDetectionConfig` / `BehaviorTrackingConfig` → 影響 `AnomalyDetector`、`ImprovedBehaviorTracker`（3.4）
- `SQAConfig`（僅 `ENABLE_SQA_DUAL_JUDGMENT` 一個總開關）→ 影響 `processors/skeleton_quality_assessment.py` 是否被 `FrameProcessor` 呼叫（3.4.1）；3 項個別指標開關不在 `config.py`，在模組自己的檔案裡
- `FlaskConfig` → 影響 `server/flask_app.py`、`server/routes.py`
- `NodeRedConfig` → 影響 `NodeRedClient` 推送的目標端點（對應「五、六」的 `/python_online`、`/yolo_result`、`/yolo_result_v2`）
- `LoggingConfig` → 影響 `CSVLogger`/`BehaviorSegmentLogger` 的輸出路徑
- `VisualizationConfig` → 影響 `Visualizer` 與 `LickStagePlugin.draw_overlay()` 的疊圖行為

⚠️ lick_stage / ext_body_zones 插件**不吃** `config.py`，而是各自有獨立的 `plugins/lick_stage/config.py`／`plugins/lick_stage/ext_body_zones/config.py`。

---

## 八、目前已知問題（尚未修正）

> 📖 **一句話摘要**：兩個已知未修正問題——GPT 報告停用中、雙引擎各自獨立告警未來合併時要小心重複觸發。

1. `GPT 健康報告.json` 整個 tab 目前是 `disabled: true`，Messenger/GPT 健康報告功能目前未啟用。
2. `貓咪主控.json`（v1/`yolo_result`）與 `cat_health_v3_flow.json`（v2/`yolo_result_v2`）各自維護獨立的 Discord webhook 設定與告警邏輯，未來若要合併兩條 flow 需注意告警可能重複觸發。

> [!NOTE]
> ✅ 已修正：`GPT 健康報告.json` 原本的 `/status` 指令呼叫 Flask 不存在的 `/status` 路由會持續失敗，2026-07 已將該指令（switch 規則、對應請求/回覆節點、`/help` 說明文字）整個移除。

---

## 九、舊文件可信度備註

> [!CAUTION]
> 🚨 以下既有文件內容**已與現況不符**，閱讀時請以本文件與各模組內的最新說明（如 `plugins/lick_stage/舔舐行為二階段分析模組說明.md`、`0_進度彙整.md`）為準。

- ❌ `0_AI_HANDOFF_FOR_ASSISTANT.md`：多處引用不存在的檔案（`cat_monitoring_system/mermaid.md`、`THREE_LAYER_FLOW.md`、`NODERED_UPDATE_GUIDE.md`、`MAIN_CONFIG_SCRIPT_CLASSIFICATION.md`、`SCRIPT_SYNC_SUMMARY.md`、`flows (7).json`、`ip取得.json`），且參數表寫 `NUM_JOINTS=17`／`WINDOW_STRIDE(推論)=16`，與現行 `config.py`（`NUM_JOINTS=14`、`WINDOW_STRIDE` 預設 2）不符。
- ❌ `0_ARCHITECTURE_DESIGN.md`：架構圖與參數表同樣寫 `V=17`／`WINDOW_STRIDE(推論)=16`，與現況不符；其餘前處理管線順序描述仍正確可用。
- ❌ `貓咪個體化基線.md`：文件頭標註「對應檔案：`cat_health_v2_flow.json`」，但該檔案已不存在——現行對應檔案是 `cat_health_v3_flow.json`（v3）。內文的基線計算邏輯描述仍大致可參考，但檔名與部分流程細節建議以本文件「5.2」與實際 json 為準。

✅ `NODE_RED_FUNCTIONS.md` 已於 2026-07 依現行 `貓咪主控.json`／`GPT 健康報告.json` 逐節點重新校對（檔名、`/camera` 已改用 `/video_clip`、新增的健康風險評分引擎／CSV 寫入／行為時間軸引擎等皆已補上），可信任其內容；但它仍只涵蓋這 2 個 flow，另外 2 個 flow（`cat_health_v3_flow.json`／`lick_stage2_nodered.json`）請看本文件「五」。

💡 若要修正上述舊文件，建議另外開任務處理，避免與本次導覽文件的建立混在一起。
