# TODO.md

> 依 PROJECT_ANALYSIS.md 的發現，按優先順序排列的改善清單。
> 每項標註：影響、位置、**呼叫關係**（改動前必查）、風險。
> 狀態：`[ ]` 未開始　`[~]` 進行中　`[x]` 完成

---

## Phase 2 執行記錄（2026-08-29，branch `refactor/analysis-improvements`）

測試基準：`yolo` conda env `pytest` → 每步驟後執行，維持 **522 passed / 27 skipped**。

**已完成（本次）：**

| 項目 | 內容 | 驗證 |
|---|---|---|
| C-3 | 新增根目錄 `.gitignore`；`git rm --cached` 415 個產物檔（`__pycache__`/`*.pyc`/`paper/build/`）；untrack `runtime_settings.previous.json` | 工作樹完整、測試通過 |
| M-1 | `pytest.ini` → `pyproject.toml [tool.pytest.ini_options]`（修 UTF-8/cp950 崩潰）；順帶把漏掉的 `models/tests` 納入 testpaths | **base anaconda3 env（無 cv2）現在也能跑 pytest**：237 passed |
| H-5 / B6 | 排程預設 `06:00`/`12:00` → `""`（不啟用排程），config.py + `default_runtime_settings.json` 同步 | `python config.py` 驗證通過、defaults 一致 |
| H-4 | `config.py` 的 `CSV_PATH` / `SEGMENTS_CSV_PATH` / cat profile 路徑 → `_resolve_project_path()`（與 `TRACKER_STATE_PATH` 一致的相對專案根解析） | 解析結果在本機不變、測試通過 |
| C-2 | pause/resume 時間錨點重設：新增 `BehaviorTracker.reset_time_anchor()` + `FrameProcessor.mark_resumed()`，在 `routes._ensure_processor_started` 解除暫停處與 gui 空白鍵恢復處呼叫。**未改 `update()` dt 語意**（避免動到手算的 regression snapshot） | 測試通過；註：完整根治需 M-9 supervisor |
| M-5 | `visualizer.py::HIP_IMAGE_PATH` 絕對路徑 → 相對套件解析；57 MB GIF 疊圖幀改**延遲載入**（`_get_overlay_data` 首用才解碼，`HIP_IMAGE_SCALE==0` 時完全不碰檔案）。功能保留 | 測試 6.6s（原 11s）、visualizer 冒煙 import OK |
| H-6 | 新增 `analytics/live_adapter.py`（中性轉接層）；`dashboard/refresher.py` 不再 `from server.routes import ...`；`routes._build_frame_processor` 用 `set_active_tracker()` 註冊。**副效果：refresher 測試不再需要 cv2** | server/analytics/dashboard 測試通過；base env 也能跑 refresher 測試 |
| M-10 | `settings_manager._validate_optional_file_warn` docstring 更正；`docs/0_ARCHITECTURE_DESIGN.md` 加過時警告指向 `ARCHITECTURE.md` | — |
| ~~H-1~~ | **驗證後撤銷**：`quantize=16` 是 ultralytics 8.4.x 正確的 predict FP16 用法（取代已棄用的 `half`）。原分析誤判。 | 查 `ultralytics/cfg/default.yaml` |
| ~~M-6~~ | **驗證後撤銷**：`reset_track` 被 tools/ 約 10 支腳本使用；`compute_bone_motion_feature` 有測試覆蓋。原「死碼」清單未涵蓋 tools/ 與測試。 | grep 全樹 |

**本次刻意未做（風險/成本與收益不成比例，或需使用者決策）：**

| 項目 | 原因 |
|---|---|
| C-1 / B1（gui 模式 monitoring_seconds） | 使用者澄清：gui 模式是本地辨識、不納入統計分析，現行「該日被基線排除」的行為**符合預期**，非 bug。（僅留 Low 註記：gui 模式仍會寫 `monitoring_seconds=0` 的列進 `daily_history.db`，可考慮 gui 模式直接跳過 `_persist_daily_record`。） |
| C-4 / B8（NUM_JOINTS 17 vs 14） | 使用者澄清：14 是暫時測試用，骨架完整定義是 17。已在 `ARCHITECTURE.md` / `0_ARCHITECTURE_DESIGN.md` 標註不一致，不改 yaml。 |
| H-2（clip_buffer ~415MB） | 移除 `.copy()` 在串流 + 慢消費者情境有畫面損毀風險，此環境無法測串流路徑。建議做法（JPEG bytes 取代 raw BGR，~50× 記憶體）已寫在下方 H-2 條目，留待可測試時實作。 |
| H-3（visualizer 標籤快取） | 有視覺回歸風險，此環境無法目視驗證。 |
| M-2（http_publisher 抽共用） | ~40 行 dedup，但 `test_nodered_publisher_unit.py` 深度綁定模組層符號（`_requests`/`time`/`_WARN_INTERVAL_SEC`/logger 名稱），churn 大於收益。 |
| M-3（skeleton.py 收斂關節索引） | 高風險（全系統 load-bearing），需逐檔遷移 + 每步測試，單次 commit 不宜。 |
| M-7（`import *` → 具名） | 純機械式、低風險但 diff 大、零功能收益，留待專門的 lint pass。 |
| M-8（tools/ 整理） | 檔案搬移 churn 大，且 `git mv` 會讓本次 commit diff 難讀。 |
| M-9（ProcessorSupervisor）/ L-* | 大型結構重構，需獨立分支 + 大量手動驗證。 |

**已知測試污染（新發現，記於此）**：`paper/tests/test_settings_manager.py::TestDefaultRuntimeSettingsConsistency` 在 cleanup 時 `real_backup.unlink()`，會刪掉真實的 `paper/runtime_settings.previous.json`。已藉由 gitignore 該檔迴避版控影響；根治需讓該測試完全在 `tmp_path` 內操作。

---

## 🔴 Critical（資料正確性 / 開箱即用 / 版控衛生）

- [x] **C-1｜gui 模式 `monitoring_seconds`** — 使用者澄清為**預期行為**（gui 不納入統計），非 bug。詳見 Phase 2 記錄。下方原始分析保留供參考。
  - 位置：`trackers/behavior_tracker.py::add_monitoring_seconds`（呼叫端）、`frame_processor.py::process()` 的 Node-RED 推送區塊、`gui` 模式 `main.py::run_gui_mode`。
  - 呼叫關係：`add_monitoring_seconds` 目前**唯一呼叫點**在 `frame_processor.process()` 的 `if self.nodered and (now - last_send_time >= PUSH_INTERVAL)` 內。`monitoring_seconds` 被 `get_today_stats()`（→ Node-RED payload、`_today_from_live_tracker` → deviation）與 `_persist_daily_record()`（→ `DailyRecord.monitoring_seconds` → `compute_baseline` 的有效天篩選 `>= MIN_DAILY_MONITORING_SEC(3600)`）讀取。
  - 修法：把「累加已監測秒數」從 Node-RED 推送路徑移出，改成每幀（或每次 `process()`）依 `dt` 累加，與 `nodered` 是否啟用無關。注意不要與 B2 的 `dt` 尖峰疊加。
  - 風險：低-中。需確認 server 模式下數值不會因改動而重複計算（目前是「兩次推送間隔」，改成逐幀後語意等價但要驗）。加/改 `behavior_tracker` 測試。

- [x] **C-2｜排程 pause/resume 時間尖峰**（PROJECT_ANALYSIS B2）— 已加 resume hook（`mark_resumed`/`reset_time_anchor`）。完整根治（含 dt 上限）待 M-9。
  - 位置：`server/streaming.py`（`paused`/`finished` 迴圈）、`server/routes.py::_pause_processing`/`_ensure_processor_started`、`main.py::_scheduler_loop`、`main.py::run_gui_mode`（暫停分支）、`trackers/behavior_tracker.py::last_update_time`、`plugins/lick_stage/manager.py::_last_wall_t`、`plugins/.../ext_body_zones/plugin.py::_last_wall_t`。
  - 呼叫關係：暫停期間 `FrameProcessor.process()` 完全不被呼叫 → 所有以 `now - last_*_time` 計 `dt` 的地方在恢復時吃到巨大 `dt`。受影響：`BehaviorTracker.update`（`hourly_distribution["monitoring_sec"]`、`not_detected_time`、`behavior_time`）、兩個 plugin 的 `_elapsed_sec`。
  - 修法：在 `RUNNING` 恢復點（`_ensure_processor_started` 解除 paused、gui 解除 paused）呼叫新增的 `FrameProcessor.notify_resumed()` → 重設 `tracker.last_update_time = now`、plugin `_last_wall_t = now`。或在 `BehaviorTracker.update` 對單幀 `dt` 加上限（如 `min(dt, 5.0)`）作為防禦。建議兩者都做。
  - 風險：中。需釐清所有恢復路徑（server 排程、gui 空白鍵、gui 逐幀 a/d）。與 S2（SharedFrameStreamer FSM）一起做最乾淨。

- [x] **C-3｜建立根目錄 `.gitignore` 並停止追蹤產物檔**（PROJECT_ANALYSIS M1、U1）
  - 位置：新增 `C:\ai_project\.gitignore`。
  - 內容至少：`__pycache__/`、`*.pyc`、`.pytest_cache/`、`paper/build/`、`*.spec`、`dist/`、`runtime_settings.previous.json`（討論後決定）、`.vscode/`（討論）。
  - 動作：`git rm -r --cached` 上述已追蹤路徑（415 檔），保留工作目錄檔案。
  - 呼叫關係：無程式依賴。確認 `paper/build/` 不被任何腳本讀取（已查：無）。
  - 風險：低。純版控操作，需使用者確認 commit。

- [~] **C-4｜(標註不一致，不改 yaml — 見 Phase 2 記錄) `stgcn_config.yaml` `NUM_JOINTS` 與部署 checkpoint 不一致**（PROJECT_ANALYSIS B8、H2）
  - 位置：`cat_monitoring_system/stgcn_config.yaml`（`NUM_JOINTS: 17`）、部署 checkpoint `stgcn_models/run_122_.../params_snapshot.json`（`num_joints: 14`）、`config.py::_validate_train_inference_consistency`、`docs/0_ARCHITECTURE_DESIGN.md`。
  - 呼叫關係：推論端 `models/stgcn_model.py::CatBehaviorSTGCN` 由 checkpoint adjacency buffer 自動偵測 num_joints（正確）；`config.py` 一致性檢查**不含** num_joints；`0_train_gcn.py` 讀 yaml 的 `NUM_JOINTS`。
  - 修法（擇一，需與使用者確認意圖）：
    (a) 若部署模型確定用 14 點 → 把 yaml 改回 `NUM_JOINTS: 14` 並在 `params_snapshot` 加對照註解；
    (b) 若要回到 17 點 → 需重訓並更新 checkpoint 路徑。
    另：在 `_validate_train_inference_consistency` 加一條「yaml NUM_JOINTS vs checkpoint 自動偵測值」的比對警告（非阻擋）。
  - 風險：低（只改設定/加檢查，不動推論路徑）。

---

## 🟠 High（效能 / 可靠性 / 明確 bug）

- [x] **H-1｜YOLO FP16**（PROJECT_ANALYSIS B3）— **驗證後為誤判，無需修改**。
  ultralytics 8.4.x `default.yaml`：`quantize` 取代已棄用的 `half`；`quantize=16` 為 predict FP16 的正確用法。現行程式碼正確。

- [ ] **H-2｜`SharedFrameStreamer` clip_buffer 每幀無條件全幀 copy（~415 MB 常駐）**（PROJECT_ANALYSIS P1）— 功能保留（使用者要）
  - 位置：`server/streaming.py::_update_frame` 的 `self.clip_buffer.append(display_frame.copy())`、`get_clip_frames()`、`server/routes.py` `/video_clip` route（`writer.write(f)`、`frames[0].shape`、`frames[-1]` 縮圖）。
  - 呼叫關係：`clip_buffer` 只被 `GET /video_clip` 消費。`clip_maxlen = max(30, TARGET_MODEL_FPS × CLIP_SECONDS)` = 150。
  - **建議修法（JPEG bytes）**：`clip_buffer` 改存 `cv2.imencode('.jpg', display_frame, [IMWRITE_JPEG_QUALITY, q])` 的 bytes（q30 下 1280×720 ≈ 30–60 KB，150 幀 ≈ 8 MB，比 raw 少 ~50×）。`_update_frame` 已在有 client 時做 JPEG 編碼，這裡對「無 client 但要維護 clip」的情況多一次編碼（~1–3 ms/幀，有界）。`/video_clip` route 改用 `cv2.imdecode` 還原後 `writer.write()`；縮圖直接用 buffer 最後一筆 bytes（免重編）。
  - 為何不直接移除 `.copy()`：串流來源（`_LatestFrameGrabber`）+ 推論慢於來源時，同一 frame 物件可能被 `process()` 就地 draw 兩次，去掉 copy 會讓 buffer 內畫面損毀。此環境無法測串流路徑，故不採此法。
  - 風險：中。需 `test_routes_api_regression.py` 有 `/video_clip` 覆蓋（待確認）+ 手動測一次點播。

- [ ] **H-3｜`Visualizer` 中文標籤每幀 4 次全幀色彩轉換**（PROJECT_ANALYSIS P2）
  - 位置：`processors/visualizer.py::_draw_text_with_pil`、`draw_prediction_on_frame`（`use_pil` 分支）。
  - 呼叫關係：`Visualizer.draw()` → `draw_prediction_on_frame()` 每幀一次（行為標籤含中文時走 PIL）。機率條走 cv2（不受影響）。
  - 修法：把「標籤文字 → 帶 alpha 的小 RGBA raster」依 `(text, color, font_size)` 快取，之後每幀只做局部 alpha-blit（類似 `_overlay_image_centered`）。
  - 風險：低-中。需保留 outline 效果與背景框；加視覺回歸（characterization）比對。

- [x] **H-4｜`config.py` 機器相依絕對路徑預設值**（PROJECT_ANALYSIS H1）
  - 位置：`config.py` 的 `ModelPaths.VIDEO_INPUT`、`LoggingConfig.CSV_PATH` / `SEGMENTS_CSV_PATH`、`NodeRedConfig.GLOBAL_CONTEXT_PATH` fallback。（原列的 `CatIdentityConfig.TARGET_CAT_PROFILE_PATH` / `OTHER_CAT_PROFILE_PATH` 已於 2026-08 身分驗證改 CNN 時移除，換成走 `_resolve_project_path()` 的 `IDENTITY_MODEL_PATH`。）
  - 呼叫關係：這些是 `_runtime_default()` 的 fallback（env / JSON 未設時才用）。`settings_manager.FIELD_SCHEMA` 對應欄位、`default_runtime_settings.json`（由 `python config.py` 同步）。
  - 修法：`CSV_PATH` / `SEGMENTS_CSV_PATH` / cat profile 改用 `_resolve_project_path("paper/...")`（與 `TRACKER_STATE_PATH` 一致）。`VIDEO_INPUT` 預設改成 `0`（攝影機）或空字串 + 明確錯誤訊息。`GLOBAL_CONTEXT_PATH` fallback 與 Node-RED `settings.js` 兩邊必須一起改（見 config.py 該處長註解），暫緩或與使用者一起處理。
  - 風險：中。改預設值會連帶觸發 `regenerate_default_runtime_settings()`；需跑 `paper/tests/test_settings_manager.py` 的一致性測試。使用者現有 `runtime_settings.current.json` 若已設這些欄位則不受影響。

- [ ] **H-5｜排程預設 06:00–12:00 是 footgun**（PROJECT_ANALYSIS B6）
  - 位置：`config.py::RunModeConfig.SCHEDULED_START_TIME` / `SCHEDULED_END_TIME` 預設、`default_runtime_settings.json`、`settings_window.py` 對應欄位提示。
  - 呼叫關係：`is_within_active_window()` ← `main._scheduler_loop` / `routes._ensure_processor_started` / `_schedule_unavailable_reason`。
  - 修法：預設改為兩者皆空字串（＝不啟用排程，開箱即用一啟動就跑）。在 settings GUI 該欄位加更醒目的說明。
  - 風險：低。行為變更需在 release note / commit message 說明。跑 `test_config_unit.py`。

- [x] **H-6｜移除 / 隔離 `server.routes` 私有函式被 `dashboard` import**（PROJECT_ANALYSIS C2）
  - 位置：`dashboard/refresher.py` import `from server.routes import _dataclass_to_jsonable, _today_from_live_tracker`；`server/routes.py` 延遲 import `dashboard.cache`。
  - 呼叫關係：`_today_from_live_tracker` 讀 `frame_processor.tracker`；`_daily_record_from_dict` 只 routes 自己用。
  - 修法：新增 `analytics/live_adapter.py` 放 `today_from_tracker(tracker)` 與 `daily_record_from_dict(d)`；`routes.py`、`refresher.py` 都 import 它。`_dataclass_to_jsonable` 換成 `dataclasses.asdict` 直接用。
  - 風險：低。純搬移 + 改 import。跑 dashboard/server 測試。

---

## 🟡 Medium（結構 / 重複 / 小 bug / 衛生）

- [x] **M-1｜`pytest.ini` 非 ASCII → 部分環境無法啟動 pytest**（PROJECT_ANALYSIS M4）
  - 修法：把 `[pytest] testpaths` 搬到 `pyproject.toml [tool.pytest.ini_options]`（UTF-8 安全）或移除 `pytest.ini` 中文註解改英文。
  - 呼叫關係：無程式依賴。CI（若之後建）會讀。
  - 風險：低。

- [ ] **M-2｜抽 `plugins/_common/http_publisher.py`**（PROJECT_ANALYSIS D2）
  - 位置：`plugins/lick_stage/publisher.py::NodeRedPublisher`、`plugins/lick_stage/ext_body_zones/output.py::ZoneHttpPublisher`（~40 行逐字複製）。
  - 呼叫關係：`NodeRedPublisher` ← `lick_stage/manager.py`；`ZoneHttpPublisher` ← `ext_body_zones/plugin.py`。
  - 修法：共用 base class（ThreadPoolExecutor + `_warn_lock` 節流）；子類只給預設 URL/timeout。
  - 風險：低。plugin fail-safe，測試 `plugins/tests/test_nodered_publisher_unit.py`。

- [ ] **M-3｜建立 `skeleton.py` 單一骨架定義**（PROJECT_ANALYSIS D5、H4）
  - 位置：新增 `cat_monitoring_system/skeleton.py`（或 `utils/skeleton.py`）：`class KP(IntEnum)` + `EDGES` + `BONE_PARENTS` + partition 建構。
  - 呼叫關係（散落點，逐一遷移，各自獨立提交）：`models/stgcn_model.py`（`_BONE_PARENTS_17`、`get_adjacency_matrix`）、`processors/anomaly_detector.py`（`_CHEST_IDX`/`_HIP_IDX`/`_TAIL_JOINTS`）、`processors/skeleton_quality_assessment.py`、`plugins/lick_stage/config.py`（`KP_*`）、`utils/constants.py`（edges）。
  - 風險：中。骨架索引是全系統 load-bearing，逐檔遷移 + 每步跑測試，不要一次全改。

- [ ] **M-4｜`cat_monitoring_log.csv` 無上限增長**（PROJECT_ANALYSIS B10）
  - 位置：`logutils/csv_logger.py::CSVLogger`。
  - 修法：加日期輪替（每日一檔）或大小上限輪替。同時修 `behavior_segments_log.csv` 的 ISO/本地日期混用（`routes.py` 註解提及的已知 bug）。
  - 風險：低-中。下游讀取工具（`tools/`、Node-RED）可能假設單一檔名，需盤點。

- [x] **M-5｜`visualizer.py` `HIP_IMAGE_PATH` / `HIP_IMAGE_SCALE=0` 死碼 + 硬編碼**（PROJECT_ANALYSIS B4、B5、U8）
  - 修法：若確定不用貓臉 GIF → 整段移除（`_load_overlay_frames`、`HIP_IMAGE_*`、`_get_overlay_frame`、`_overlay_image_centered` 的呼叫）。若要保留 → 路徑改 `Path(__file__).parent.parent / "assets" / "..."`，`HIP_IMAGE_SCALE` 的「停用」用 `None` 表達並修正判斷。
  - 呼叫關係：`Visualizer.__init__` / `draw()`；無外部依賴。
  - 風險：低。與使用者確認此視覺功能去留。

- [x] **M-6｜移除確定的死碼** — **經 grep 驗證後取消**：
  - `compute_bone_motion_feature` 有測試覆蓋（public API）；`reset_track` 被 tools/ 約 10 支腳本使用；`_ZeroResidual` 是 `STGCNBlock` public 參數的防禦實作；4ch 路徑使用者決定保留。
  - 教訓：原「死碼」清單僅搜尋核心模組，未涵蓋 `tools/` 與測試，故 U5/U6 誤判。

- [ ] **M-7｜`from utils.constants import *` 改具名 import**（PROJECT_ANALYSIS C5）
  - 位置：`frame_processor.py`、`visualizer.py`、`server/routes.py`、`utils/helpers.py`。
  - 風險：低。純機械式，IDE 可輔助。

- [ ] **M-8｜整理 `tools/` 與根目錄雜項**（PROJECT_ANALYSIS M7、M8、U10、U11）
  - `tools/` 建 `oneoff/` 子目錄放帶硬編碼路徑的一次性腳本；非 pytest 的 `test_*.py` 互動除錯器改前綴（如 `debug_*.py` / `inspect_*.py`）並更新 `pytest.ini`/`pyproject` 說明。
  - 根目錄 `之川論文.docx`、`paper/1110.py`、`cat_pose/111.py` 等 → 移到 `docs/` / `scratch/` 或刪除（問使用者）。
  - 風險：低。改檔名/移動需 `git mv`，確認無 import 依賴（tools 多為獨立腳本）。

- [ ] **M-9｜`_scheduler_loop` / `is_within_active_window` / processor 生命週期收斂**（PROJECT_ANALYSIS C7、S2、S5）
  - 抽 `pipeline/supervisor.py`（`ProcessorSupervisor`）：建立/暫停/恢復/停止 + 排程狀態。`main.py`（server/gui）與 `routes.py` 都依賴它。
  - 風險：高。動到啟動流程與全域狀態，需大量手動驗證（server 冷啟動、排程邊界、gui 模式、Ctrl+C 清理）。**建議放在 Critical/High 修完、有回歸測試後再做**。

- [x] **M-10｜文件對齊**（PROJECT_ANALYSIS M9、M10、M11）
  - `docs/0_ARCHITECTURE_DESIGN.md`：修正「FC 輸出 (N=1, 4)」→ 5 類、17 點 → 部署 14 點；或標記為 deprecated 指向本 `ARCHITECTURE.md`。
  - 移除對不存在腳本的引用（`eval_accuracy_smoothing_compare.py` 等）或補回腳本。
  - 修 `settings_manager._validate_optional_file_warn` docstring。

---

## 🟢 Low（長期 / 大重構 / 選用）

- [ ] **L-1｜拆分 `config.py`（1363 行）**（PROJECT_ANALYSIS G1）—— 先搬 `get_config_summary` / `validate_all_config` / `__main__` regen 到 `config_report.py`。風險中（39 個 import 點）。
- [ ] **L-2｜拆分 `settings_window.py`（1993 行）** —— 已有 `settings_gui/` 元件化基礎，繼續把分頁渲染引擎、匯入預覽拆出。
- [ ] **L-3｜`FrameProcessor` 顯式狀態機**（PROJECT_ANALYSIS S1）—— 依賴 M-9 完成 + characterization 測試齊備。風險高。
- [ ] **L-4｜`BehaviorTracker` 事件生命週期狀態機**（S3）。
- [ ] **L-5｜正式 packaging**：`pyproject.toml` + 把 `config.py` 移進套件或改成 `cat_monitoring.config`，消除 `sys.path` hack（C1）。風險高。
- [ ] **L-6｜模型權重移出 git**（M2）：改用 Git LFS 或外部儲存 + 下載腳本。需 history rewrite，破壞性，最後做。
- [ ] **L-7｜`save_state()` fsync 頻率**（P5）—— 事件完成時改成非同步/批次寫入。
- [ ] **L-8｜`cat_pose/` 與 `paper/` 共用骨架常數**（D7）—— 建 `shared/` 或讓 `cat_pose` import `paper` 的 constants。
- [ ] **L-9｜Node-RED 舊引擎正式停用**（U9、ADR 0001）—— 停用 `貓咪主控.json` 的重複健康引擎與 Discord 告警節點。非 Python 工作。
- [ ] **L-10｜`plugins` 閘控協定明確化**（C3）—— `register_plugin(plugin, scope="lick"|"always")`，`FrameProcessor` 依 scope 決定餵 `(None,None)` 或真值。

---

## 建議執行順序

1. **C-3、M-1、H-5**（低風險、立即見效：版控衛生 + 開箱即用）
2. **C-4、H-1、H-4、M-10**（設定/文件對齊 + FP16，皆低風險有測試）
3. **C-1、C-2**（資料正確性核心 bug；需補 `behavior_tracker` 測試後再改）
4. **H-2、H-3**（效能；需視覺回歸）
5. **H-6、M-2、M-5、M-6、M-7、M-8**（結構清理，逐項獨立提交）
6. **M-3、M-9**（骨架收斂、supervisor；風險較高，回歸測試齊備後）
7. **L-*（長期）**

> 每次修改前：`grep` 確認呼叫點 → 跑相關測試（`pytest paper/cat_monitoring_system/<module>/tests`）→ 小步提交。
> 涉及行為/預設值變更者，在 commit message 明確標註。
