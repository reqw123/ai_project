# ARCHITECTURE.md

> 貓咪行為辨識 + 個體化基線健康監測系統 — 架構與資料流參考
> 更新：2026-08-29（依實際程式碼核對，取代 `paper/docs/0_ARCHITECTURE_DESIGN.md` 中已過時的部分）
> 詳細逐模組職責另見 `paper/docs/模組責任畫分.md`。

---

## 1. 系統概觀

```
                        ┌─────────────────────────────────────────────┐
                        │              進入點 main.py                   │
                        │   RunModeConfig.MODE = "server" | "gui"       │
                        └───────────────┬──────────────┬───────────────┘
                                        │ server       │ gui
                        ┌───────────────▼──┐        ┌──▼───────────────┐
                        │ Flask + Node-RED │        │ OpenCV 本地視窗   │
                        │ + 排程迴圈        │        │ （無 HTTP/NodeRed）│
                        └───────────────┬──┘        └──┬───────────────┘
                                        │  共用          │
                                  ┌─────▼──────────────▼─────┐
                                  │   FrameProcessor.process  │  ← 核心編排（單幀）
                                  └─────┬────────────┬────────┘
              ┌────────────────────────┘            └────────────────────────┐
     ┌────────▼─────────┐                                        ┌───────────▼──────────┐
     │  感知層 Perception │                                        │  監測層 Monitoring     │
     │  YOLO-Pose         │                                        │  AnomalyDetector      │
     │  骨架前處理         │                                        │  BehaviorTracker      │
     └────────┬─────────┘                                        │  Visualizer / CSV     │
     ┌────────▼─────────┐                                        │  NodeRedClient        │
     │  分析層 Analysis   │                                        │  Plugins (lick/zones) │
     │  ST-GCN + SQA     │────────────────────────────────────────▶└───────────┬──────────┘
     └──────────────────┘                                                     │
                                                          ┌───────────────────▼─────────────────┐
                                                          │  個體化基線 analytics/（獨立於攝影機）  │
                                                          │  daily_store(SQLite) → baseline →     │
                                                          │  deviation → fusion → /dashboard      │
                                                          └──────────────────────────────────────┘
```

### 分層職責

| 層 | 輸入 → 輸出 | 模組 |
|---|---|---|
| **感知 Perception** | frame → `kpts(17,2)`, `kpt_conf(17,)`, `bbox`, `bbox_conf` → `(T=16, V, C=7)` 特徵張量 | `detectors/keypoint_detector.py`、`models/stgcn_model.py`（`interpolate_missing` → `flip_normalize` → `orientation_normalize` → `normalize_skeleton_coords` → `build_feature_tensor`） |
| **分析 Analysis** | `(1, C, 16, V)` → `behavior_id ∈ {0..4, -1 LOW_CONF, -2 NO_CAT}`, `confidence`, `probs[5]` | `detectors/behavior_classifier.py`、`models/stgcn_model.py`（STGCN）、`processors/skeleton_quality_assessment.py`（幾何否決 → LOW_CONF） |
| **監測 Monitoring** | 上述 + `activity_value` → 每日統計、CSV、Node-RED payload、疊圖畫面、警報 | `processors/anomaly_detector.py`、`trackers/behavior_tracker.py`、`processors/visualizer.py`、`logutils/csv_logger.py`、`communication/nodered_client.py`、`plugins/*` |
| **個體化分析 Analytics** | 多天 `DailyRecord` + 今日統計 → `Baseline`、`DeviationResult`、`FusionResult`（風險等級） | `analytics/{daily_store,baseline,deviation,fusion,config}.py`、`dashboard/*` |

---

## 2. 單幀資料流（`FrameProcessor.process()`）

```
frame
 │
 ├─▶ KeypointDetector.detect()                     # YOLO-Pose，多貓時以 IoU 延續鎖定同一隻
 │     └─ kpts / kpt_conf / bbox / bbox_conf  (或全 None)
 │
 ├─▶ IdentityVerifier.verify()  [預設關閉]          # 非目標貓 → kpts = None
 │
 ├─▶ 貓消失容忍  (CAT_MISSING_TOLERANCE_FRAMES = 5)  # 連續漏偵測未超過門檻 → 沿用 _last_known_kpts
 │
 ├─▶ Plugin.update(raw_kpts, kpt_conf)              # ⚠ 僅在已確認 lick 時餵真值，否則 (None, None)
 │
 ├─▶ Frame-level EMA (KP_EMA_ALPHA = 1.0 → no-op)   # 僅供 overlay / 靜止偵測；不進 ST-GCN
 │     └─▶ AnomalyDetector.detect(display_kpts)     # 位移滾動均值（體型正規化，排除尾巴）→ is_still, activity
 │
 ├─▶ keypoints_buffer.append((raw_kpts, kpt_conf))  # deque(maxlen = SEQUENCE_LENGTH = 16)
 │     每 WINDOW_STRIDE(=2) 幀 且 buffer 滿 16：
 │       interpolate_missing(kpts_arr, conf_arr)
 │        └─▶ BehaviorClassifier.classify(seq, conf)
 │              flip_normalize → orientation_normalize → normalize_skeleton_coords
 │              → build_feature_tensor(seq, conf, "xy_conf_v_bone")   # 7ch: x,y,conf,vx,vy,bone_x,bone_y
 │              → STGCN.forward → softmax → (behavior_id, confidence, probs)
 │        confidence < STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD(0.80) → LOW_CONF_ID
 │        SQA.evaluate_window(原始未插值 kpts_arr) 不可信 → LOW_CONF_ID (confidence = 0)
 │        └─▶ _update_display_hysteresis()          # 顯示層：同類連續 DISPLAY_HYSTERESIS_WINDOWS 次才切換標籤
 │
 ├─▶ BehaviorTracker.update(behavior_id, activity)  # ← 用「即時」結果（非 hysteresis）
 │     ├─ behavior_time / behavior_count / transition_matrix / hourly_distribution
 │     ├─ 事件結算 _settle_current_behavior() → behavior_history + BehaviorSegmentLogger
 │     ├─ save_state()（事件完成 或 每 30s；原子寫入 tracker_state.json）
 │     └─ check_daily_reset() → 跨日 → daily_store.save_day()（SQLite，鎖外執行）
 │
 ├─▶ NodeRedClient.send_data(payload)               # 每 PUSH_INTERVAL(2s)；← 用「hysteresis 後」的顯示結果
 │     └─ v1 /yolo_result + v2 /yolo_result_v2，各自 daemon 緒、drop-on-full
 │
 ├─▶ CSVLogger.log()                                # 非靜止 且 confidence ≥ 0.80 且 非 LOW_CONF
 │
 └─▶ Visualizer.draw() + plugin.draw_overlay()      # 骨架 / bbox / 標籤 / 機率條 / 梯形 overlay
       └─▶ SharedFrameStreamer：clip_buffer.append(copy)；有 client 時 JPEG 編碼 → /stream MJPEG
```

### 三層狀態哨兵（`utils/constants.py`）

| id | 名稱 | 意義 | 顯示 |
|---|---|---|---|
| `-2` | `NOT_VISIBLE_ID` | YOLO 未偵測到貓，ST-GCN 不執行 | 「不在畫面」🔍 |
| `-1` | `LOW_CONF_ID` | 有骨架但 ST-GCN 信心 < 0.80，或 SQA 幾何否決 | 「正常/低信心」❓ |
| `0..4` | walk / lick / scratch / shake / stop | 有效行為 | 走動🐾 / 舔舐👅 / 搔抓🦶 / 甩頭🌀 / 靜止💤 |

---

## 3. 個體化基線分析（與攝影機管線解耦）

```
BehaviorTracker（每日累積） ──跨日──▶ daily_store.save_day() ──▶ SQLite: daily_history
                                                                       │
   觸發者 A：dashboard/refresher.py 背景緒（每 ~2s，_cache_is_fresh 去重）
   觸發者 B：Node-RED analytics_deviation_bridge.json → POST /api/deviation（新舊引擎比對）
                                                                       │
                          compute_baseline(history, min_days=7, max_days=30, excluded_dates)
                            ├─ 連續指標 (walk/stop/lick/scratch _time)：mean/std/median/MAD/EWMA
                            └─ 稀疏計數 (walk/stop/lick/scratch/shake _count)：保留每日原始序列
                                                                       │
                          compute_deviation(today, baseline)
                            ├─ 連續 → robust z-score = (today - median) / (MAD × 1.4826)
                            └─ 計數 → Poisson（或過度離散時 Negative-Binomial）尾機率 → sigma_equivalent
                                                                       │
                          compute_fusion(deviation, class_c_score=0)
                            Class A（lick/scratch 時長+次數，權重和 0.85）45%   ← 可單獨 override 等級
                            Class B（shake/walk/stop）                    25%
                            Class C（節律/轉移，暫由 Node-RED 傳入）        30%
                            → score 0-100 → Normal(<20) / Mild(<45) / Moderate(<70) / Severe(≥70)
                                                                       │
                          dashboard/cache.set_latest() → GET /api/deviation/latest → /dashboard/baseline
```

> ⚠ 融合權重與 sigma 門檻（2.5/3.0/4.0）為專家經驗值，**無資料驅動校準**（`analytics/config.py` 逐項標註證據等級；論文中誠實列為限制）。

---

## 4. 主要 HTTP 端點（`server/routes.py` + `dashboard/views.py`）

| 端點 | 用途 |
|---|---|
| `GET /stream` | MJPEG 即時疊圖串流 |
| `GET /snapshot` | 單張 JPEG |
| `GET /video_clip` | 最近 `CLIP_SECONDS` 秒 ring buffer 編成短片 + base64 縮圖 |
| `GET /api/behavior_history?limit=` | 行為區段歷史 |
| `GET/POST /api/overlay` | 讀/切換 skeleton / label / bbox / master 疊圖旗標 |
| `POST /api/deviation` | 基線+偏差+融合橋接（新舊引擎比對；欄位全可省略，預設走 Python 自己的 daily_store） |
| `GET /api/deviation/latest` | 回傳快取的最新基線結果 |
| `GET /dashboard/baseline` | Python 端唯讀基線儀表板頁 |

排程時段外：`/stream` 等回 **503** 並附說明（`RunModeConfig.SCHEDULED_START_TIME`~`END_TIME`）。

---

## 5. 設定系統

```
┌──────────────┐   優先序高
│ 環境變數      │   CAT_MONITORING_*
│  ↓           │
│ runtime_settings.current.json   ← settings_window.py 編輯（原子寫入 + .previous.json 備份）
│  ↓           │   欄位表：settings_manager.FIELD_SCHEMA（json_key ↔ env_var ↔ (class, attr) ↔ 型別 ↔ 驗證）
│ config.py 硬編碼 fallback │
└──────────────┘   優先序低

排除在外（不進 GUI）：
  STGCNTrainingConfig（權威 = stgcn_config.yaml）
  STGCNConfig.SEQUENCE_LENGTH / FEATURE_MODE / NUM_CLASSES（checkpoint 架構相容性）
```

### config.py 內的設定 class

`ModelPaths` · `YOLOConfig` · `STGCNConfig` · `STGCNTrainingConfig`(讀 yaml) · `RunModeConfig` · `FlaskConfig` · `BaselineDashboardConfig` · `NodeRedConfig` · `VisualizationConfig` · `AnomalyDetectionConfig` · `SQAConfig` · `BehaviorTrackingConfig` · `CatIdentityConfig` · `LoggingConfig` · `SystemInfo`

### 關鍵參數（現值）

| 參數 | 值 | 位置 | 備註 |
|---|---|---|---|
| `SEQUENCE_LENGTH` (T) | 16 | config + yaml | 16 幀 @ 30fps ≈ 0.53s |
| `WINDOW_STRIDE`（推論） | 2 | config | 每 2 幀推論一次 |
| `WINDOW_STRIDE`（訓練） | 8 | yaml | 50% 重疊 |
| `TARGET_MODEL_FPS` | 30 | config | 來源 FPS 更高則降採樣 |
| `FEATURE_MODE` | `xy_conf_v_bone` (7ch) | config + yaml | 推論時由 checkpoint 自動偵測驗證 |
| `NUM_JOINTS` | yaml 寫 **17**，部署 checkpoint 為 **14**（排除尾巴） | ⚠ 不一致，見 PROJECT_ANALYSIS B8 |
| `USE_ATTENTION` | true | yaml | JointAttention + JOINT_PRIOR_WEIGHTS（Nose×2.0） |
| `STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD` | 0.80 | config | 低於 → LOW_CONF |
| `KP_EMA_ALPHA` | 1.0 | config + yaml | 1.0 = 不平滑 |
| `SQA` 雙重判定 | **啟用** | config | fail-safe，可覆蓋為 LOW_CONF |
| `CAT_MISSING_TOLERANCE_FRAMES` | 5 | config | |
| `PUSH_INTERVAL` | 2s | config | Node-RED 推送節奏 |
| `SCHEDULED_START/END` | **06:00 / 12:00**（預設！） | config | ⚠ 開箱即用只在此區間運行 |
| `YOLO 模型` | `yolo_models/v11s_147.pt` | config | 相對專案根解析 |
| `ST-GCN 模型` | `stgcn_models/run_122_xy_conf_v_bone_att_on/122_best_model.pth` | config | 相對專案根解析 |

---

## 6. ST-GCN 模型結構（`models/stgcn_model.py`）

```
輸入 (N=1, C=7, T=16, V=14)
  BatchNorm2d(7)
  JointAttention: Conv2d(7→1) + Sigmoid × prior_weights(Nose 2.0…) → gate (N,1,T,V)；x = bn_x × gate
  Block 1  SpatialGraphConv(7→64, K=3 分區: 自身/向心/離心) + MultiScaleTCN(k=3,5,9, stride=1)  → (N,64,16,14)
  Block 2  SGC(64→128) + MST-TCN(stride=2)                                                      → (N,128,8,14)
  Block 3  SGC(128→128) + MST-TCN(stride=1)                                                     → (N,128,8,14)
  AdaptiveAvgPool2d(1,1) → (N,128) → Dropout(0.5) → Linear(128→5)
  Softmax → [walk, lick, scratch, shake, stop]
```

- **分區鄰接**：以 `mid_back(4)` 為中心 BFS，PYSKL 式向心/離心分區；`D^-0.5 A D^-0.5` 對稱正規化。
- **可學習參數**：`partition_importance`（每分區）、`branch_logits`（多尺度 TCN softmax 權重）、JointAttention conv。
- **checkpoint 自動偵測**：`in_channels`（由 `bn_input.weight`）、`num_classes`（`fc.weight`）、`num_joints`（adjacency buffer）、`use_attention`（key 前綴）、TCN 分支數。**不自動偵測** `SEQUENCE_LENGTH` / `FEATURE_MODE`（訓練/推論不一致會是安靜的 bug）。
- **訓練/推論共用**：`0_train_gcn.py` 的 `build_feature_tensor` 只是 `models.stgcn_model.build_feature_tensor` 的 thin wrapper（前處理確實同源）。

---

## 7. Runtime 狀態模型（目前為隱式，建議顯式化 — 見 PROJECT_ANALYSIS §10）

### FrameProcessor（每幀）
```
                    ┌──────────┐  YOLO 有偵測 + (身分驗證通過)
          ┌────────▶│ BEHAVIOR │◀────────┐
          │         │  (0..4)  │         │ 連續同類 ≥ hysteresis
   信心≥0.80        └────┬─────┘         │
   且 SQA 可信           │ 信心<0.80 或 SQA 否決
          │              ▼
          │         ┌──────────┐
          └─────────│ LOW_CONF │
                    └────┬─────┘
                         │ YOLO 連續漏偵測
                         ▼
              ┌──────────────────────┐  streak < TOLERANCE(5)：沿用 _last_known_kpts
              │ CAT_MISSING_TOLERATED │
              └──────────┬───────────┘  streak ≥ TOLERANCE
                         ▼
                    ┌──────────┐
                    │  NO_CAT  │  → 重置 EMA / buffer / _infer_frame_count / _last_behavior
                    └──────────┘
```

### SharedFrameStreamer（管線）
```
        start_background_refresh
   ────────────────────────────▶ RUNNING ◀────────────────┐
                                  │  │                     │ 排程進入區間
                    排程離開區間   │  │ 本機影片 EOF         │
                                  ▼  ▼                     │
                              PAUSED  FINISHED ────────────┘（PAUSED 可恢復；FINISHED 不會）
```
> **B2 修正點**：`PAUSED → RUNNING` 轉移時應重置 `BehaviorTracker.last_update_time` 與 plugin `_last_wall_t`，否則恢復時的巨大 `dt` 污染統計。

---

## 8. 部署形態與環境

- Python 3.11（conda 環境 `yolo_new`），`torch 2.5.1+cu121`、`ultralytics 8.4.87`、`opencv-python`、`Flask`。
- 選用：`pywin32`（子行程視窗置頂）、`paho-mqtt`（ext_body_zones MQTT）、`yt-dlp`（YouTube 來源）。
- Node-RED（獨立行程，port 1880）：Dashboard、Discord 告警、部分歷史彙整。共用檔案 `paper/baseline_data/global.json`（Node-RED 端 `settings.js` 的 `gfile` 寫入，Python 端目前僅記錄路徑）。
- 資料檔：`paper/baseline_data/`（`tracker_state.json`、`daily_history.db`）、`paper/cat_monitoring_log.csv`、`paper/behavior_segments_log.csv`。

---

## 9. 已知架構債（詳見 PROJECT_ANALYSIS.md）

| 代碼 | 摘要 |
|---|---|
| C1 | `config.py` 在被設定的套件外，靠 `sys.path` hack import |
| C2 | `dashboard/refresher.py` import `server.routes` 私有函式（雙向耦合） |
| C3 | 所有 plugin 被 `is_lick_behavior` 閘控（泛用協定實為 lick-scoped） |
| C6/B8 | ST-GCN 架構參數 3 處（config / yaml / checkpoint）可能不一致，只交叉檢查 2/3 |
| C7 | processor 生命週期分散於 main.py / routes.py / streaming.py |
