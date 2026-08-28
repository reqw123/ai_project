# PROJECT_ANALYSIS.md

> 第一階段分析報告。
> 掃描日期：2026-08-29
> **註**：Phase 2 執行時經實測/grep 驗證，部分項目已更正（B3/P3 FP16 為誤判、
> U5/U6 非死碼、B1/B8 經使用者澄清為預期行為）——內文已就地標記 ✅/更正/~~刪除線~~，
> 完整處理記錄見 `TODO.md` 的「Phase 2 執行記錄」。
> 範圍：`paper/`（核心系統 + tools/ + docs/）、`cat_pose/`、`stgcn_models/`、`yolo_models/`、Node-RED flow JSON、根目錄設定檔。
> 追蹤檔案數：1332（含 415 個產物檔，見 §8）。實際 Python 原始碼 ~150 檔。

---

## 1. 專案用途與目前架構

### 用途
**貓咪行為辨識 + 個體化基線健康監測系統**（碩士論文專案）。

- 以 **YOLO-Pose** 偵測貓咪 17 點骨架，**ST-GCN** 分類 5 類行為（walk / lick / scratch / shake / stop）。
- 為單一貓咪建立**個體化行為基線**（近 30 天每日指標的 robust 統計），偵測偏離日常模式的健康異常訊號。
- 對外輸出：Flask MJPEG 即時串流、Node-RED Dashboard、Python 端唯讀基線儀表板。

### 架構分層（三層流水線）

| 層 | 模組 | 職責 |
|---|---|---|
| 感知層 Perception | `detectors/keypoint_detector.py`、`models/stgcn_model.py` 前處理函式 | YOLO-Pose 推論 → 關鍵點 → 插值/翻轉/方向/中心化/特徵建構 |
| 分析層 Analysis | `detectors/behavior_classifier.py`、`models/stgcn_model.py`（STGCN）、`processors/skeleton_quality_assessment.py` | ST-GCN 行為分類 + 幾何合理性雙重判定（SQA） |
| 監測層 Monitoring | `processors/anomaly_detector.py`、`trackers/behavior_tracker.py`、`processors/visualizer.py`、`communication/nodered_client.py`、`logutils/csv_logger.py`、`analytics/*` | 靜止偵測、每日統計/警報、疊圖、Node-RED 推送、CSV、個體化基線 |

核心編排者：**`processors/frame_processor.py::FrameProcessor.process()`** — 單幀從偵測到通報的完整流程。

### 執行形態
- **server 模式**（預設）：`main.py` → Flask（`server/flask_app.py` + `routes.py`）+ Node-RED 上線通知 + 背景排程迴圈。
- **gui 模式**：`main.py` 直接開 OpenCV 視窗，重用同一套 `FrameProcessor`，不啟動 Flask/Node-RED。
- **settings_window.py**：獨立 tkinter GUI，編輯 `runtime_settings.current.json`（不熱重載，需重啟主程式）。
- **tools/**：~40 支研究/評估/資料處理腳本（訓練、消融、雙模型比較、耳距量測…），非 runtime 必要。
- **Node-RED**（`paper/*.json`）：系統的「另一半」，負責 Dashboard UI、Discord 告警、部分歷史彙整。**目前有新舊兩套健康評分引擎並存**（見 §6、ADR 0001）。

### 設定優先序（三層覆寫）
```
環境變數 (CAT_MONITORING_*)  >  runtime_settings.current.json  >  config.py 硬編碼 fallback
```
由 `config.py::_runtime_default()` + `settings_manager.py::get_runtime_value()` 實作。
ST-GCN **訓練**參數的唯一權威來源是 `cat_monitoring_system/stgcn_config.yaml`（不進 GUI）。

---

## 2. 所有主要模組與資料流

### 2.1 模組清單（runtime 核心）

| 檔案 | 職責 | 主要依賴 |
|---|---|---|
| `paper/config.py` (1363 行) | 13 個設定 class + env 解析 + 摘要/驗證 + 預設值同步 | `settings_manager` |
| `paper/settings_manager.py` (1010 行) | `FIELD_SCHEMA` 單一欄位表、驗證、原子儲存、預設值防漂移 | — |
| `cat_monitoring_system/main.py` | 進入點、Ctrl+C 硬關閉、排程迴圈、gui 模式迴圈 | `config`, `server.*` |
| `server/flask_app.py` / `routes.py` | Flask app factory + 全部 HTTP 路由 + processor 生命週期 | `config`, `processors`, `analytics`, `dashboard` |
| `server/streaming.py` | `SharedFrameStreamer`：單一寫入緒 + JPEG 編碼 + clip ring buffer | `config` |
| `processors/frame_processor.py` (775 行) | **核心編排** | 幾乎所有子模組 |
| `detectors/keypoint_detector.py` | YOLO-Pose 封裝 + 多貓 IoU 追蹤鎖定 | `ultralytics` |
| `detectors/behavior_classifier.py` | ST-GCN 前處理路徑分派（legacy 4ch / multichannel） | `models.stgcn_model` |
| `detectors/identity_verifier.py` | 多貓身分過濾（預設關閉，fail-safe） | — |
| `models/stgcn_model.py` (760 行) | STGCN 網路 + **訓練/推論共用**前處理與特徵建構函式 | `torch` |
| `models/keypoint_kalman.py` | Kalman 平滑（研究用，runtime 未接線，見 §8） | `numpy` |
| `processors/anomaly_detector.py` | 位移滾動均值靜止偵測（體型正規化） | `config` |
| `processors/skeleton_quality_assessment.py` | 3 指標幾何合理性判定，覆蓋為 LOW_CONF（fail-safe） | `models.stgcn_model` |
| `processors/visualizer.py` (576 行) | 骨架/bbox/標籤/機率條/GIF 疊圖 | `PIL`, `cv2` |
| `trackers/behavior_tracker.py` (568 行) | 每日時長/次數/轉移矩陣/活動分數/警報 + 狀態持久化 + 跨日重置 | `config`, `analytics.daily_store` |
| `communication/nodered_client.py` | 非阻塞雙端點（v1/v2）背景推送，drop-on-full | `requests` |
| `logutils/csv_logger.py` | 逐幀 CSV + 行為區段 CSV（append 模式） | `config` |
| `utils/{helpers,constants,env_parsing}.py` | IP 偵測、YouTube 解析、行為名稱、骨架常數、env 解析 | — |
| `analytics/{baseline,deviation,fusion,daily_store,config}.py` | 個體化基線計算 + 今日偏差評分（robust-z / Poisson-NB 尾機率）+ Class A/B/C 融合 + SQLite 多天歷史 | `numpy`（daily_store 只用 stdlib） |
| `dashboard/{views,cache,refresher}.py` | Python 端唯讀基線儀表板 + 記憶體快取 + 背景重算緒 | `analytics`, `server.routes`（⚠ 見 §6） |
| `plugins/lick_stage/*` | 舔舐二階段：鼻子接觸梯形、部位命中、頭部朝向平滑（可插拔、fail-safe） | `numpy`, `config`（僅 Node-RED 端點） |
| `plugins/lick_stage/ext_body_zones/*` | 7 區身體分區（可插拔、fail-safe，CSV/MQTT/HTTP 輸出） | 同上 |
| `settings_gui/*` | tkinter 元件：console panel、process manager、欄位搜尋、樣式 | `pywin32`（選用） |

### 2.2 主資料流（server 模式，單幀）

```
VideoCapture ─(串流則經 _LatestFrameGrabber 抽乾緩衝)→ read_raw_frame()
  → SharedFrameStreamer._update_frame（FPS 降採樣 frame_step = src_fps / TARGET_MODEL_FPS）
    → FrameProcessor.process(frame):
        1. KeypointDetector.detect() → kpts(17,2), kpt_conf(17,), bbox, det_conf
        2. IdentityVerifier.verify()（預設關閉）→ 非目標貓則 kpts=None
        3. 貓消失容忍（CAT_MISSING_TOLERANCE_FRAMES=5）：沿用 _last_known_kpts
        4. Plugin.update(raw_kpts, kpt_conf)  ← 僅在已判定 lick 時餵真值，否則 (None,None)
        5. Frame-level EMA（α=1.0 預設=no-op）→ display_kpts → AnomalyDetector.detect() → is_still, activity
        6. keypoints_buffer.append((raw_kpts, kpt_conf))；每 WINDOW_STRIDE(2) 幀且 buffer≥16：
             interpolate_missing → BehaviorClassifier.classify()
               (flip_normalize → orientation_normalize → normalize_skeleton_coords → build_feature_tensor)
               → STGCN → (behavior_id, confidence, probs)
             confidence < 0.80 → LOW_CONF_ID
             SQA.evaluate_window(原始 kpts_arr) 不可信 → LOW_CONF_ID
        7. _update_display_hysteresis()（顯示層連續 N 視窗才切換標籤）
        8. BehaviorTracker.update(behavior_id, activity)  ← 用「未經 hysteresis」的即時結果
             → 每日統計 / 轉移矩陣 / 活動視窗 / 事件結算 → save_state()（事件完成或每 30s）
             → 跨日 → daily_store.save_day()（SQLite）
        9. NodeRedClient.send_data()（每 PUSH_INTERVAL=2s）  ← 用「經 hysteresis」的顯示結果
       10. CSVLogger.log()（非靜止 且 confidence≥0.80 且 非 LOW_CONF）
       11. Visualizer.draw() + plugin.draw_overlay()
    → JPEG 編碼（僅有串流 client 時）→ /stream MJPEG
```

### 2.3 基線分析資料流（與攝影機管線獨立）

```
BehaviorTracker 跨日 → daily_store.save_day() → SQLite daily_history
                                                      │
      ┌───────────────────────────────────────────────┴──────────────┐
      │                                                              │
dashboard/refresher.py（每 RECOMPUTE_INTERVAL≈2s，_cache_is_fresh 去重）   Node-RED analytics_deviation_bridge.json
      │                                                              │  → POST /api/deviation
      └──────────────► compute_baseline() ◄──────────────────────────┘
                       compute_deviation()  (robust-z / Poisson-NB tail → sigma_equivalent)
                       compute_fusion()      (Class A 45% / B 25% / C 30%，Class A 可 override 等級)
                       → dashboard/cache.set_latest() → GET /api/deviation/latest → /dashboard/baseline
```

---

## 3. 潛在 Bug

> 標記：🔴 影響正確性/資料 ・ 🟠 效能/可靠性 ・ 🟡 小瑕疵

| # | 位置 | 問題 | 影響 |
|---|---|---|---|
| B1 🔴 | `frame_processor.py` `process()` + `trackers/behavior_tracker.py` `add_monitoring_seconds()` | `monitoring_seconds` **只在 Node-RED 推送路徑累加**（`if self.nodered and ...`）。gui 模式 `nodered=None` → 永遠不累加。跨日持久化的 `DailyRecord.monitoring_seconds = 0`。 | `compute_baseline()` 以 `monitoring_seconds >= 3600` 篩「有效天」→ **gui 模式收集的每一天都會被基線排除**。`total_uptime` / Dashboard「監測時長」在 gui 模式恆為 0。 |
| B2 🔴 | `main.py` 排程 `_pause_processing` / `streaming.py` `paused` / gui 暫停迴圈 | 暫停期間不呼叫 `process()`，`BehaviorTracker.last_update_time`、plugin `_last_wall_t` 不更新。恢復後第一幀的 `dt = now - last_update_time` 可能是「數分鐘/數小時」。 | 恢復瞬間把巨大 `dt` 灌進 `hourly_distribution["monitoring_sec"]` 與 `not_detected_time`（或當下行為時長）→ 每次排程區段邊界都產生時間尖峰，污染統計與基線。 |
| ~~B3~~ ✅ | `detectors/keypoint_detector.py` `detect()` | ~~`quantize=16` 非 ultralytics 參數~~ **已於 2026-08-29 驗證：誤判**。ultralytics 8.4.x `default.yaml` 明載 `quantize`（`predict` 用 16/fp16）**取代已棄用的 `half` 參數**。現行程式碼正確。 | 無需處理。 |
| B4 🟡 | `processors/visualizer.py` L28-33 | `HIP_IMAGE_PATH = Path(__file__).resolve().parent.parent.parent / r"C:\ai_project\...\h6bxw-tkcsv.gif"` — `Path / 絕對路徑字串` 在 Windows 上「碰巧」得到該絕對路徑，前綴 `parent.parent.parent` 是死碼；非 Windows 直接壞。且該 gif 不在 repo。 | 目前靠 `_load_overlay_frames` 回傳 `[], []` 優雅降級。屬硬編碼 + 混淆碼。 |
| B5 🟡 | `processors/visualizer.py` `HIP_IMAGE_SCALE = 0` | `0` 被當「停用」魔術值，但 `if HIP_IMAGE_SCALE is not None:` 為真 → `w = max(1, int(round(src_w * 0))) = 1` → 每幀仍嘗試貼一張 1px 圖。註解卻說 `None=跟隨 bbox`。 | 每幀無謂運算；語意矛盾。 |
| B6 🟠 | `config.py` `RunModeConfig` 預設值 | `SCHEDULED_START_TIME="06:00"` / `SCHEDULED_END_TIME="12:00"` 為**出廠預設**。開箱即用時 `/stream` 等端點在 12:00–06:00 一律回 503。 | 已知踩過（`routes.py::_schedule_unavailable_reason` 註解記錄使用者忘記自己設過而誤判系統壞掉）。限制性排程當預設是 footgun。 |
| B7 🟡 | `models/stgcn_model.py` `interpolate_missing()` | 某關節整個窗口都 `conf ≤ 0.1` → `seq[:, v, :] = 0`（全零）。若 `mid_back(4)` 全零，後續 `normalize_skeleton_coords` 中心化 + `orientation_normalize` 對全零旋轉 → 退化輸入餵進 STGCN。 | 可能產生虛假分類。SQA 部分能攔（門檻經驗值、少量影片校準）。 |
| B8 🟡 | `stgcn_config.yaml` `NUM_JOINTS: 17` vs 部署 checkpoint | `stgcn_models/run_122/params_snapshot.json` → `num_joints: 14`（訓練時排除尾巴 3 點）。推論端 `CatBehaviorSTGCN` 由 checkpoint 自動偵測 → 正確跑 14；但 `config.py::_validate_train_inference_consistency` **只比對 SEQUENCE_LENGTH / FEATURE_MODE，不比對 NUM_JOINTS**。 | 設定檔誤述部署模型。照 yaml 現值重訓會得到行為不同的 17 點模型且無警告。ARCHITECTURE_DESIGN.md 也仍寫 17。 |
| B9 🟠 | `server/routes.py` 模組全域 `frame_streamer` / `frame_processor` | 多個 route handler 未持鎖直接讀取；`_ensure_processor_started` 為 double-checked locking，但讀取端沒有一致保護。 | 排程開始時刻邊界曾撞出 `AttributeError`（註解已記錄，逐一補 `if X is None` 防護，非根治）。 |
| B10 🟡 | `logutils/csv_logger.py` | `cat_monitoring_log.csv` 逐幀 append、無輪替/上限。長時間運行無限增長。 | 磁碟；讀取工具變慢。`behavior_segments_log.csv` 亦已知 ISO/本地日期混用 bug（`routes.py` 註解提及）。 |
| B11 🟡 | `frame_processor.py` 身分驗證 + 貓消失容忍交互 | 非目標貓入鏡使 `kpts=None` 後，容忍機制會**沿用目標貓的 `_last_known_kpts`** 最多 5 幀。 | 非目標貓出現後短暫仍以目標貓陳舊姿態處理/統計。 |
| B12 🟡 | `analytics/daily_store.py` | 連線快取於模組 dict、`check_same_thread=False`、全部存取序列化於單一模組 `_lock`。慢的 analytics 計算持鎖時會阻塞即時管線的 `check_daily_reset`（雖已把 persist 移出 tracker 鎖，daily_store 自己的 `_lock` 仍序列化）。 | 跨日那一刻若碰上 refresher/`manage_baseline_history` 併發，即時管線可能短暫卡住（`timeout=10.0`）。 |

---

## 4. 重複程式碼

| # | 內容 | 位置 | 備註 |
|---|---|---|---|
| D1 | env 解析 (`_env_str/int/float`) | `paper/config.py`（完整版含 `_env_bool/_env_video_input/_env_size`）、`utils/env_parsing.py`（str/int/float）；後者被 `analytics/config.py`、`plugins/lick_stage/config.py`、`plugins/lick_stage/ext_body_zones/config.py` 共用 | 已部分收斂（2026-08 抽 `env_parsing.py`）。`config.py` 刻意不共用（避免反向依賴 `cat_monitoring_system` 套件）。`_env_bool` 仍只有 config.py 一份。 |
| D2 | `ZoneHttpPublisher` ≈ `NodeRedPublisher` | `ext_body_zones/output.py` vs `lick_stage/publisher.py` | ~40 行幾乎逐字複製（含 `_warn_lock` 節流邏輯）。docstring 自承 "mirroring"。可抽 `plugins/_common/http_publisher.py`。 |
| D3 | `_HHMM_RE` 正則 | `config.py` 與 `settings_manager.py` | 刻意保持一致（註解說明），非疏漏。 |
| D4 | Chest-MidBack-Hip 夾角 / 曲率公式 | `processors/skeleton_quality_assessment.py` 與 `plugins/lick_stage/contact_regions.py` | 刻意各實作一份以維持 plugin 低耦合（註解說明「未來重新校準時兩處要一起改」）。 |
| D5 | 骨架關節索引常數 (0=nose,3=chest,4=mid_back,5=hip,14-16=tail…) | `stgcn_model._BONE_PARENTS_17`、`get_adjacency_matrix` edges、`anomaly_detector._CHEST_IDX/_HIP_IDX/_TAIL_JOINTS`、`skeleton_quality_assessment`、`lick_stage/config.py`、`utils/constants.EAR_DISTANCE_SKELETON_EDGES` … | **無單一 `KEYPOINT_INDEX` 定義**。重新命名/排序關節需改 ~10 檔。見 §10。 |
| D6 | tools/ 推論腳本 | `1_run_video_inference.py` docstring 自承「其餘功能與 test_video_inference.py 完全相同」；`eval_ema_ablation.py` / `eval_gcn_compare.py` / `eval_model_worst_videos.py` 共用大量流程（部分已抽 `_smoothing_eval_common.py`） | tools/ 是最大重複面。 |
| D7 | 骨架/配色常數跨專案複製 | `cat_pose/constants.py` docstring：「直接複製自 paper/cat_monitoring_system/utils/constants.py」 | 兩個子專案各一份。 |
| D8 | 基線數學雙語言實作 | `analytics/baseline.py`（Python）↔ `node_red_tests/baseline_calculator.js` / `cat_health_v3_flow.json`（JS） | 遷移進行中（`baseline.py` docstring 說明數值刻意對齊 JS 原版）。 |

---

## 5. 效能問題

| # | 位置 | 問題 | 估計成本 |
|---|---|---|---|
| P1 🔴 | `server/streaming.py` `_update_frame` | `self.clip_buffer.append(display_frame.copy())` **每幀無條件執行**（不管 `/video_clip` 是否曾被呼叫）。deque maxlen = `TARGET_FPS(30) × CLIP_SECONDS(5)` = 150。 | 1280×720×3 ≈ 2.7 MB/frame × 150 ≈ **~415 MB 常駐** + 每幀一次全幀 memcpy。應改成「有需求才錄」或大幅降低。 |
| P2 🟠 | `processors/visualizer.py` `_draw_text_with_pil` | 行為標籤含中文 → 每幀 `cvtColor(BGR2RGB)` + `Image.fromarray` + `np.array(pil)` + `cvtColor(RGB2BGR)`，**4 次全幀轉換**（機率條走 cv2 較便宜）。 | 1280×720 每幀數 ms。可把標籤 raster 依 `(text, color)` 快取後 alpha-blit。 |
| ~~P3~~ ✅ | `detectors/keypoint_detector.py` | 依 B3 更正：FP16（`quantize=16`）在 ultralytics 8.4.x 正常運作，無此問題。 | — |
| P4 🟡 | `dashboard/refresher.py` + Node-RED | 兩條路徑各每 ~2s 重算完整 `compute_baseline`（重建 up to 120 天 `DailyRecord` + 全量統計）。`_cache_is_fresh` 已去重其一。 | 120 天資料量下每次 µs–低 ms，可接受；屬「定時全量重算」模式。 |
| P5 🟡 | `trackers/behavior_tracker.py` `save_state()` | 每個完成的行為事件（`_event_completed`）都 `json.dump` + `os.fsync`；另每 30s 一次。 | 慢磁碟上 fsync 可能在熱路徑短暫 stall。 |
| P6 🟡 | `config.py` import | 被 39 個模組 import；所有 class body 於首次 import 執行 `_env_*` + `_runtime_default`（讀 JSON 檔一次、可能 print 警告）。 | 一次性，可接受；但 import 有副作用（讀檔/印字）。 |
| P7 🟡 | `_LatestFrameGrabber._grab_loop`（串流） | 專用解碼緒 + sleep 節流至來源 FPS，避免搶 GIL。設計合理，但串流來源多一條常駐解碼緒 + 每幀一次 lock。 | 可接受。 |

---

## 6. 不合理的耦合

| # | 內容 |
|---|---|
| C1 | **`config.py` 在被設定的套件外面**。`paper/config.py` 被 `cat_monitoring_system/**` 依賴，靠 `main.py`、`routes.py`、`settings_window.py` 各自 `sys.path.insert()` 才 import 得到。無正式 packaging，import 路徑脆弱。 |
| C2 | **`dashboard/refresher.py` import `server.routes` 的私有函式**（`_dataclass_to_jsonable`、`_today_from_live_tracker`）；同時 `server.routes` 又延遲 import `dashboard.cache` → 雙向耦合。「從 live tracker 取今日資料」的邏輯應放中性模組（如 `analytics/live_adapter.py`），而非 `routes`。 |
| C3 | **`frame_processor` 把所有 plugin 都以 `is_lick_behavior` 閘控**。`register_plugin`/`update` 是泛用協定，卻隱性只服務 lick；`ExtBodyZonePlugin` docstring 卻自稱「fully independent」。未來非 lick plugin 會被餓死。 |
| C4 | **`SystemInfo.OUTPUT_WIDTH/HEIGHT` 雙用途**（文件識別資訊 + 攝影機擷取解析度請求），經 `routes._build_frame_processor` 傳進 `FrameProcessor`。攝影機設定放在 `SystemInfo` 很意外。 |
| C5 | **`from utils.constants import *` 萬用 import**（`frame_processor`、`visualizer`、`routes`、`helpers`）→ 命名空間污染，難追符號來源。 |
| C6 | **ST-GCN 架構參數 3 處可能不一致**：`config.STGCNConfig`（env 可覆寫）、`stgcn_config.yaml`（權威）、checkpoint 自動偵測。只有 2/3 有交叉檢查（B8）。 |
| C7 | **processor 生命週期邏輯分散**於 `main.py`（排程迴圈、gui 迴圈、Ctrl+C 清理）+ `routes.py`（`_build_frame_processor`、`_ensure_processor_started`、`_pause_processing`、plugin 註冊、`frame_processor` 全域）+ `streaming.py`（`paused`/`finished`）。`main.py` 直接戳 `routes` 內部與全域變數。 |
| C8 | 正面例子：**`analytics/` 高度隔離**（自己的 config、延遲 import `config`、`daily_store` 只用 stdlib），`plugins/lick_stage` 亦然（僅共用 `interpolate_missing` 與 Node-RED 端點）。 |

---

## 7. 可維護性問題

| # | 問題 |
|---|---|
| M1 | **無根目錄 `.gitignore`**；415 個產物檔進版控：`__pycache__/`、`*.pyc`、`paper/build/`（PyInstaller 中間產物 http_1/http_2）、`.pytest_cache/`。 |
| M2 | **183 MB 模型權重進 git**（`yolo_models/` 152 MB、`stgcn_models/` 31 MB）：9 個 stgcn run + ~10 個 yolo 權重，預設只用 `run_122` + `v11s_147.pt`。歷史已膨脹（需 rewrite 才能瘦身）。 |
| M3 | 無 `pyproject.toml` / `setup.cfg` / packaging；3+ 進入點 `sys.path` 硬插。無 linter/formatter 設定檔、無 CI。 |
| M4 | **`pytest.ini` 含非 ASCII 註解** → 在 iniconfig 使用 locale codec 的環境直接壞（本次重現：`UnicodeDecodeError: 'cp950'`，`pytest` 無法啟動）。應移到 `pyproject.toml [tool.pytest.ini_options]`（永遠 UTF-8）或移除中文註解。 |
| M5 | **超大檔案**：`config.py` 1363、`settings_window.py` 1993、`tools/1_measure_ear_distance_single_video.py` 3029、`tools/0_train_gcn.py` 2561、`tools/train_data/0_dataset_collect.py` 2278、`tools/1_run_video_inference.py` 2184。 |
| M6 | **註解訊噪比低**：函式內夾大量帶日期的變更史敘事（如 `behavior_tracker.check_daily_reset` ~25 行沿革）。脈絡有價值但應進 git/ADR，目前傷害掃讀性且易與程式漂移。 |
| M7 | **混語言檔名**：`影片拼接.py`、`影片降1080p.py`、`1_多重命名.py`、`1_自動抓取.py`、`建立攝影伺服器.py`、`之川論文.docx`。**流水號檔名**：`paper/1110.py`、`cat_pose/111.py`。 |
| M8 | **tools/ 混雜三類**：(a) 真正可重用工具、(b) 帶硬編碼 `MODE=`/`SOURCE_FOLDER=r"C:\Users\homec\..."` 模組常數的一次性資料處理腳本、(c) 命名為 `test_*.py` 但其實是互動式 GUI 除錯器（非 pytest 測試）——`pytest.ini testpaths` 是繞過這個陷阱的 workaround。 |
| M9 | **多份重疊架構文件**：`docs/模組責任畫分.md`、`docs/0_AI_專案導覽地圖.md`、`docs/0_ARCHITECTURE_DESIGN.md`（後者 2026-06-01、標示「FC 輸出 (N=1, 4)」與 17 點，皆與現況不符，且引用一個叫 `先不管.md` 的檔案）。 |
| M10 | **懸空引用**：程式碼/文件多次提到 `tools/eval_accuracy_smoothing_compare.py`、`eval_model_four_videos.py`、`test_video_inference.py`、`test_video_inference_ema copy.py`——皆不在版控檔案清單中。 |
| M11 | **stale docstring**：`settings_manager._validate_optional_file_warn` docstring 說「回傳 (error, warning)」但實際回傳單值；`deviation.py` docstring 引用不存在的腳本。 |
| M12 | 測試覆蓋不錯（analytics/processors/trackers/server/utils/plugins/dashboard/config/settings_manager 皆有），但 `detectors/*`（需 YOLO/torch）、`visualizer.py`、`communication/` 無測試。`processors/tests` 有 `frame_processor` characterization 測試（好）。 |

---

## 8. 已經沒有使用的程式碼或檔案

| # | 項目 | 狀態 |
|---|---|---|
| U1 | `paper/build/http_1/`、`paper/build/http_2/` | PyInstaller 中間產物，應進 `.gitignore`。build target 名稱與現行進入點對不上。 |
| U2 | `cat_pose/tello_drone_archive/` | 明確「archive」（Tello 無人機控制，與現行系統無關）。 |
| U3 | `models/keypoint_kalman.py` + `stgcn_config.yaml` `KALMAN_*` | 研究產物。`SMOOTHING_KIND: "ema"` → runtime 與訓練皆不呼叫。yaml 註解自承實驗結論「與不平滑相比 McNemar p=1.0，不顯著，已改回 ema」。刻意保留為紀錄。 |
| ~~U4~~ | Legacy 4-channel ST-GCN 路徑 | **使用者決定保留**（可能載入舊 4ch checkpoint）。runtime 未走此路但保留能力。 |
| ~~U5~~ | `models/stgcn_model.py::compute_bone_motion_feature()` | **更正**：非死碼——`test_stgcn_preprocessing_unit.py` 有測試覆蓋，屬 `stgcn_model` 共用前處理 public API。 |
| ~~U6~~ | `detectors/keypoint_detector.py::reset_track()` | **更正**：非死碼——`tools/` 底下約 10 支腳本呼叫（新影片切換時重置追蹤鎖定）。原分析僅搜尋核心模組故誤判。 |
| ~~U7~~ | `models/stgcn_model.py::_ZeroResidual` | `STGCN.__init__` 目前一律 `residual=True`，故未實例化；但屬 `STGCNBlock(residual=...)` public 參數的防禦性完整實作，移除價值低。**保留**。 |
| U8 | GIF 疊圖功能 | `HIP_IMAGE_SCALE=0` 實質關閉、gif 不在 repo，但每幀仍執行 `_get_overlay_frame` + 1px blit 嘗試（見 B5）。 |
| U9 | `paper/貓咪主控.json`（Node-RED 舊引擎）+ 對應 Discord 告警 | ADR 0001 已決定停用、統一到 Class A/B/C，「實際 Node-RED 節點停用尚未執行」。 |
| U10 | 一次性腳本 | `paper/1110.py`、`cat_pose/111.py`、`cat_pose/pose_single_image_test.py`、`paper/_tools/1_read_context.py`、`paper/_tools/2_backfill_daily_store.py`、`tools/1_多重命名.py`。 |
| U11 | 非程式碼產物進版控 | `之川論文.docx`（根目錄 80 KB）、`paper/GPT 健康報告.json`、`paper/skeletons_test_labeled/*(N)(M).json`（看似重複匯出）、`runtime_settings.previous.json`（自動備份）。 |

---

## 9. 設定值是否過度寫死

| # | 位置 | 硬編碼內容 | 建議 |
|---|---|---|---|
| H1 🔴 | `config.py` | `VIDEO_INPUT = r"C:\Users\homec\OneDrive\圖片\..."`、`CSV_PATH` / `SEGMENTS_CSV_PATH = r"C:\ai_project\paper\..."`（絕對，未用 `_resolve_project_path`）、`TARGET_CAT_PROFILE_PATH` / `OTHER_CAT_PROFILE_PATH = r"C:\ai_project\paper\..."`、`GLOBAL_CONTEXT_PATH` fallback `r"C:\a\global.json"` | 不一致：YOLO/STGCN 路徑已改用 `_resolve_project_path()`，CSV/profile 卻沒。統一為相對專案根。 |
| H2 🟠 | `stgcn_config.yaml` | `SKELETON_DATA_FOLDER: "C:/ai_project/paper/skeletons"`、`RESULTS_FOLDER: "C:/ai_project/stgcn_models"` 絕對路徑；`NUM_JOINTS: 17` 與部署 checkpoint（14）不符（B8） | 相對路徑 + 修正 NUM_JOINTS 或加註。 |
| H3 🟠 | `visualizer.py` | `HIP_IMAGE_PATH` 絕對硬編碼；`_FONT_CANDIDATES` 全為 `C:\Windows\Fonts\...`（Windows-only，有 `load_default()` 兜底） | 相對 assets/ + 跨平台字型搜尋。 |
| H4 🟡 | 關節索引 | `_CHEST_IDX=3`、`_HIP_IDX=5`、`_TAIL_JOINTS=(14,15,16)`、`_BONE_PARENTS_17`、`KP_NOSE=0`… 散落 ~10 檔 | 集中成 `KeypointIndex` enum（見 §10）。 |
| H5 🟡 | `SystemInfo.OUTPUT_WIDTH/HEIGHT = 1280/720` | 無 env 覆寫 | 併入 GUI 設定或加 env。 |
| H6 ✅ | `analytics/config.py`、`fusion.py`、`deviation.py` | 權重/門檻皆有 env 覆寫、且逐項標註證據等級（【文獻支持】/【與 Node-RED 一致】/【專家設定】/【數值穩定】） | **良好範例**，無需處理。 |
| H7 🟡 | `skeleton_quality_assessment.py` | 3 個 `ENABLE_*_CHECK` + 8 個門檻常數為模組層級（自承「少量影片校準、暫定」）。刻意不進 config.py（低耦合） | 可接受；調參需改碼。 |
| H8 🟡 | `plugins/lick_stage/config.py` | ~60 個幾何比例常數為 class attribute，僅 `NODERED_URL/TIMEOUT` 可 env 覆寫 | plugin 內合理，量大。 |
| H9 🟡 | `main.py` | `_SCHEDULER_POLL_SECONDS=20`、`_SHUTDOWN_GRACE_SECONDS=3.0`、`GUI_MAX_WIDTH/HEIGHT` | 低優先。 |

---

## 10. 適合模組化或 State Machine 化的地方

### State Machine 候選

| # | 現況 | 建議 |
|---|---|---|
| S1 🔴 | **`FrameProcessor` 單幀狀態**：隱式 FSM 涵蓋 `{NO_CAT, CAT_MISSING_TOLERATED, LOW_CONF, BEHAVIOR(0..4)}` + 顯示層 hysteresis 子狀態，由 ~10 個散落 instance 變數表達（`_cat_missing_streak`、`_last_known_*`×4、`_last_behavior_id/_confidence/_class_probs`、`_display_*`×3、`_hysteresis_candidate_id/_streak`）。 | 抽出顯式狀態物件（手刻或 `transitions`）：NO_CAT 的 EMA/buffer/counter 重置、容忍窗、hysteresis 合為一個可獨立測試的單元。 |
| S2 🟠 | **`SharedFrameStreamer` 生命週期**：`running` / `paused` / `finished` 三個 bool + 隱式優先序。 | 小 enum FSM，明確轉移（start→RUNNING、排程外→PAUSED、EOF→FINISHED、排程內→RUNNING）。**B2 的時間尖峰在 `PAUSED→RUNNING` transition hook 重置計時器即可根治**。 |
| S3 🟠 | **`BehaviorTracker` 事件生命週期**：`current_behavior` open/settle/switch、`_in_low_conf` enter/exit、跨日重置。`update()` 內 not_detected/low_conf/valid 的巢狀 if/elif 就是一台手寫狀態機。 | 明確狀態 + transition 表。 |
| S4 🟡 | **`LickAnalyzer` 方向穩定化**：`_trap_perp_flip_streak`、`_trap_dir_wrong_streak`、`_state_history` 多個 streak counter。 | 抽 `DirectionStabilizer` 狀態物件。 |
| S5 🟠 | **排程狀態**：`RunModeConfig.is_within_active_window` + `main._scheduler_loop`（`was_active` 邊緣偵測）+ `routes._ensure_processor_started`/`_pause_processing` 分散於 3 檔。 | `Scheduler` 物件持有狀態、發出 start/pause/resume 事件。 |

### 模組化候選

| # | 建議 |
|---|---|
| G1 | **拆 `config.py`**：`config/env.py`（解析器）、`config/paths.py`、`config/model.py`、`config/runtime.py`、`config/summary.py`（`get_config_summary` + `validate_all_config` + `__main__` regen）。至少先把 120 行 f-string 摘要與驗證搬走。 |
| G2 | **抽 `pipeline/supervisor.py`**（`ProcessorSupervisor`）：`_build_frame_processor` + plugin 註冊 + `_ensure_processor_started` + pause/resume + 排程。`main.py`（gui/server）與 `routes.py` 都依賴它，不再互戳全域（解 C2/C7）。 |
| G3 | **`skeleton.py` 單一骨架定義**：`KeypointIndex` enum + edges + `BONE_PARENTS` + partition 建構（收斂 D5/H4）。 |
| G4 | **`plugins/_common/http_publisher.py`**：dedupe `NodeRedPublisher` / `ZoneHttpPublisher`（D2）。 |
| G5 | **`analytics/live_adapter.py`**：把 `_today_from_live_tracker` / `_daily_record_from_dict` 從 `routes.py` 移出（解 C2）。 |
| G6 | **`toolkit/`**：把 tools/ 中真正可重用者（video IO、labeled-video 評估 harness — 已部分見 `_smoothing_eval_common.py`）收成小套件；一次性資料處理腳本標註/歸位到 `tools/oneoff/`。 |

---

## 附錄：整體評語

- **成熟度高**：核心 runtime 設計嚴謹（fail-safe plugin、原子寫入、非阻塞推送、訓練/推論共用前處理、三層設定覆寫、豐富的決策文件與 ADR）。多數「怪異」處都有註解說明是刻意取捨。
- **主要風險集中在邊界**：gui 模式資料完整性（B1）、排程 pause/resume（B2）、config/checkpoint 漂移（B8）、版控衛生（M1/M2）。
- **技術債主要在 `tools/` 與根目錄衛生**，核心 `cat_monitoring_system/` 相對乾淨。
- 已存在的 `docs/模組責任畫分.md` 品質良好，可作為 ARCHITECTURE.md 的基礎（本次另附精簡英中對照版）。
