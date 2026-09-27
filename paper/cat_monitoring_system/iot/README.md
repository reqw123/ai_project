# IoT Sensor Hub（物聯網感測子系統）

> **定位**：整個貓咪監測系統的「錦上添花」部分。透過 MQTT 收 ESP32 環境感測、
> PIR 移動偵測、HX711 重量感測的資料，做正規化 / 持久化 / 門檻告警，再把
> 結果發回 MQTT 給 Node-RED Dashboard 使用。
>
> **壞掉也沒關係**：這個子系統是獨立行程，跟 `main.py`（YOLO/ST-GCN 推論主流程）
> 完全解耦、互不啟動對方。它整個掛掉，主系統照常運作。

---

## 設計原則（低耦合 / 高內聚）

| 原則 | 具體做法 |
|---|---|
| **雙向零 import** | `iot/` 不 import `config` / `analytics` / `processors` / `server` / `utils` … 任何主系統模組；主系統也不 import `iot/`。連 env 解析 helper、broker host/port 都自己管（`iot/config.py`）。 |
| **獨立行程** | `cd cat_monitoring_system && python -m iot` 啟動，是跟 `main.py` 平行的另一個行程。 |
| **唯一對外接觸點 = MQTT broker** | 沿用 Node-RED 的 mosquitto。不持有任何 Discord / 外部服務憑證——告警發到 `cat/iot/alert`，由 Node-RED 既有的「發送提醒」節點轉 Discord。 |
| **fail-safe** | MQTT 斷線自動重連；一筆壞 payload 記 log 丟棄、不中斷；SQLite 寫入失敗記警告後繼續。 |
| **高內聚** | 每種感測器一個模組（`sensors/environment.py` / `motion.py` / `weight.py`），解析 + 驗證 + 正規化 + 衍生事件偵測全在同一檔。 |
| **可整包移除** | 刪掉整個 `iot/` 資料夾 + 移除 `pytest.ini` 裡對應那一行 testpath 即可，主系統零影響。 |

---

## 資料流

```
ESP32 / 感測器節點
      │  MQTT publish  cat/iot/<kind>/<source_id>
      ▼
┌─────────────────┐        ┌──────────────┐
│  mosquitto      │───────▶│  Node-RED    │  （各自訂閱，互不知道對方存在）
│  (broker)       │        │  Dashboard   │
└────────┬────────┘        └──────────────┘
         │ 訂閱 cat/iot/env/#, cat/iot/motion/#, cat/iot/weight/#
         ▼
┌──────────────────────────────────────────────────────────────┐
│  python -m iot   （本子系統，獨立行程）                        │
│                                                              │
│  ingest/mqtt_ingestor  →  sensors/router  →  分派 parser       │
│         │                                                    │
│         ├─▶ storage/store        （SQLite：原始讀數全部落地）  │
│         ├─▶ sensors/*: 正規化 / EWMA 平滑 / 進食事件偵測       │
│         ├─▶ output/alert_engine  （門檻告警 + 每 key 冷卻）    │
│         └─▶ output/publisher     （發回 MQTT）                 │
│                    │                                         │
└────────────────────┼─────────────────────────────────────────┘
                     │ publish  cat/iot/derived/<kind> , cat/iot/alert
                     ▼
              mosquitto ──▶ Node-RED Dashboard / Discord
```

---

## MQTT topic 約定

`iot/config.py` 的 `TOPIC_PREFIX` 預設 `cat/iot`，`TOPIC_MAP` 定義有哪些 kind。

### 收（感測器 → hub）

| topic | payload（JSON 物件）| 說明 |
|---|---|---|
| `cat/iot/env/<source_id>` | `{"temp_c": 26.4, "humidity_pct": 55, "gas_ppm": 120, "lux": 300}` | 欄位皆可省略；接受別名（`temperature`/`rh`/`co2`/`light`…）。可選 `"ts"`（epoch 秒）。 |
| `cat/iot/motion/<source_id>` | `{"active": true}` 或 `{"event": "enter"}` | 接受 `active`/`motion`/`state`/`event`/`detected`。 |
| `cat/iot/weight/<scale_id>` | `{"grams": 123.4}` | 接受 `grams`/`weight`/`g`/`mass`。 |
| `cat/iot/bodytemp/<source_id>` | `{"surface_temp_c": 34.2, "ambient_temp_c": 26}` | MLX90614 紅外線非接觸測溫（`esp32_petbox` 節點）；接受 `surface_temp_c`/`object_temp_c`/`object`。**體表溫度非核心體溫，只做初篩**。 |

`<source_id>` 可含斜線（例如 `cat/iot/env/floor2/bedroom`）。

### 發（hub → Node-RED）

| topic | payload | 節流 |
|---|---|---|
| `cat/iot/derived/env` | EWMA 平滑後的環境讀數 | 每 `(kind, source_id)` 最小間隔 `DERIVED_PUBLISH_MIN_INTERVAL_SEC`（預設 5s） |
| `cat/iot/derived/motion` | 移動事件 | 同上 |
| `cat/iot/derived/weight` | 每筆重量讀數 | 同上 |
| `cat/iot/derived/weight_change` | `{"from_g","to_g","delta_g","direction",...}` 進食 / 加料 / 飲水事件 | 不節流（低頻但重要） |
| `cat/iot/derived/bodytemp` | `{"surface_temp_c","ambient_temp_c",...}` | 每 `(kind, source_id)` 最小間隔 `DERIVED_PUBLISH_MIN_INTERVAL_SEC` |
| `cat/iot/alert` | `{"key","severity","message","value","threshold","ts"}` | AlertEngine 每個 key 有 `ALERT_COOLDOWN_SEC` 冷卻（預設 15 分鐘） |

---

## 告警規則（`iot/config.py` 門檻，全部 env 可覆寫）

| key 前綴 | 觸發條件 | severity |
|---|---|---|
| `env.temp_high` / `env.temp_low` | 溫度超出 `[TEMP_MIN_C, TEMP_MAX_C]`（預設 18–28°C，舒適導向；幼貓/高齡/病貓要自行收窄） | warning |
| `env.humidity_high` / `env.humidity_low` | 濕度超出 `[HUMIDITY_MIN_PCT, HUMIDITY_MAX_PCT]`（預設 40–70%） | warning |
| `env.gas_high` | `gas_ppm > GAS_PPM_MAX`（預設 1000 ppm） | critical |
| `bodytemp.high` / `bodytemp.low` | 體表溫度超出 `[BODYTEMP_SURFACE_MIN_C, BODYTEMP_SURFACE_MAX_C]`（預設 20–40°C，門檻放很寬）。**初篩用，非發燒診斷**——體表 IR 溫度不等於肛溫 | warning |
| `feeding.silence` | `FEEDING_SILENCE_HOURS`（預設 24h）內偵測不到食盆重量下降事件 | warning |

食盆的 `scale_id` 由 `FOOD_SCALE_IDS`（預設 `food_bowl`，逗號分隔多個）指定。

### 連線／感測器健康警告（只記在 hub 日誌，設定視窗 IoT 紀錄框顯示成紅字；2026-09-27 起）

| 情況 | 怎麼判斷 | 模組 |
|---|---|---|
| 整台 ESP32 沒送資料 | 連上 broker 後 `NO_DATA_FIRST_CHECK_SEC`（5 秒）先檢查一次；之後某種感測器超過 `DATA_TIMEOUT_SEC`（60 秒）沒資料 | `ingest/no_data_watchdog.py` |
| ESP32 在線、某顆感測器沒接好：**欄位消失** | 節點還在送，但某欄位超過 `DATA_TIMEOUT_SEC` 沒值（例：外出包 DHT11 → `humidity_pct`）。`SENSOR_EXPECTED_FIELDS` 列的欄位一開始就沒接也抓得到 | `sensors/health.py` |
| ESP32 在線、類比感測器沒接好：**卡在極端值** | 連續 `SENSOR_RAIL_CONSECUTIVE`（6）筆是 `SENSOR_RAILS` 的極端值（MQ-135 讀到 0 或滿格、光敏電阻滿格；光照 0 不算，全暗是正常的） | `sensors/health.py` |
| 同種類有好幾台、**其中一台離線**（例：env 的外出包斷電，esp32_env 還在送） | 那一台超過 `DATA_TIMEOUT_SEC` 沒送、同種類還有別台在送（只剩一台的話由上面「沒送資料」報，不重複） | `sensors/health.py` |
| 韌體送出 `nan`（Arduino `String(NAN)`，例：MLX90614 沒接） | 當作沒有值、同一筆其他欄位照收（以前整筆丟掉）；必填欄位是 nan 時這筆不能用，但這台仍算在線，由「欄位消失」報 | `sensors/router.py` |
| hub 自己登入 broker 被拒（帳密錯） | 記 ERROR「broker 拒絕連線」，不會錯怪 ESP32 | `ingest/mqtt_ingestor.py` |

做不到的：PIR 空接跟「沒有動靜」在電氣上分不出來；荷重元斷線但 HX711 還在時送的是亂數。

**終端列印（除錯用，設定視窗「⚙ 參數設定 → 🖨 終端列印間隔」）**：警報通知歸警報通知——門檻告警寫 DB、
發 MQTT → Discord 照 `ALERT_COOLDOWN_SEC`（15 分鐘）冷卻；終端（hub 日誌）另外設：

| 變數 | 預設 | 印什麼 |
|---|---|---|
| `WARN_REPEAT_SEC` | 10 | 所有「異常持續中」——沒收到資料、欄位消失、極端值、nan、門檻告警超標——沒解除時每隔幾秒重印（標「持續中」）；0＝只印一次 |
| `READING_LOG_INTERVAL_SEC` | 0 | 每個節點最新讀數印一行（看數值有沒有進來）；0＝不印 |

2026-09-27 合併：`NO_DATA_CHECK_INTERVAL_SEC`＋`SENSOR_FIELD_MISSING_SEC` → `DATA_TIMEOUT_SEC`；
`SENSOR_WARN_REPEAT_SEC`＋`SENSOR_NAN_WARN_SEC`＋`ALERT_LOG_REPEAT_SEC` → `WARN_REPEAT_SEC`。
設定視窗的覆寫檔裡還有舊名稱時會自動換成新的（`iot_config_overrides.LEGACY`）。

恢復正常都會記一行；日誌的告警一律 WARNING（紅字）。

---

## 執行

### 前置

1. mosquitto broker 已在跑（Node-RED 環境通常已有）。
2. `pip install paho-mqtt`（已列在主專案 `requirements.txt`；或用 `iot/requirements.txt` 開獨立 venv）。

### 啟動

```bash
cd paper/cat_monitoring_system
python -m iot
```

Ctrl+C（Windows 也含 Ctrl+Break）會優雅關閉：停止 MQTT 迴圈、關閉 SQLite 連線。

### 從設定視窗啟動（2026-09-27 起）

`settings_window.py` 最後一個分頁「📡 IoT 子系統」是各子系統的啟動口：

- 5 個服務，各自一個背景行程：
  - 🌡 環境感測（`env`）
  - 👣 移動偵測（`motion`，PIR）
  - ⚖ 食盆秤重（`weight`）
  - 🐾 體表溫度（`bodytemp`）
  - 🎙 飼主語音紀錄（`python -m iot.voice`）
- 每個服務都有 ▶ 啟動、■ 停止、📄 紀錄三個按鈕，也有「全部啟動／全部停止」，並顯示 MQTT broker 連不連得到。
- 可以跟 main.py 同時跑：不佔 main.py 那一套「同一時間一支」的機制。
- 在 cmd 手動執行的也認得（掃行程指令列）。
- **關掉設定視窗時，執行中的 IoT 服務會一起強制關閉**（先 terminate、5 秒沒結束就 kill；cmd 手動執行的也算），紀錄檔最後會寫一行「設定視窗關閉，一併停止」。
- 紀錄檔在 `paper/logs/iot/<服務>.log`。
- 版面：整個分頁固定填滿「分頁頂端 → 全域終端機上緣」，不跟著外層捲動：
  - 上半是卡片區（放不下時自己捲）；
  - 下半是紀錄框，一打開就顯示最新的幾行；往上捲看舊紀錄時，新內容進來也不會跳走，捲回最底就恢復跟隨。
  - 全域終端機變高時，卡片區和紀錄框一起等比例縮，紀錄框不會被蓋住。
- 紀錄框的高度與字級：
  - 高度：拖紀錄框的**標題列**（上緣有一條綠色細條，滑鼠變成上下箭頭，整條都能抓），往上拖變高；雙擊回預設（可用高度的 45%）。
    - **最高跟全域終端機一樣**（拉到視窗標題列下方）：紀錄框跟全域終端機一樣浮在視窗上，拉得比分頁高時會蓋住卡片、分頁列；
      切到別的分頁時自動藏起來。要切分頁時先把它拉低或雙擊標題列。
  - 字級：A＋／A－ 按鈕、Ctrl＋滾輪、Ctrl＋＋／－；Ctrl＋0 或點「字級 N」回預設 12。
  - 兩者都記在 `ui_state.json`（`iot_log_ratio`＝紀錄框佔的比例、`iot_log_font`），下次開啟沿用。
  - 舊的 `iot_log_height` 已不再使用：像素高度在全域終端機變高時會把卡片擠掉，所以改記比例。
- 程式：`paper/settings_gui/iot_services.py`（邏輯，不 import 本套件，只用 subprocess 啟動）、`paper/settings_gui/iot_tab.py`（畫面）。

各感測器分開啟動靠 `CAT_MONITORING_IOT_KINDS`（見下表）：
- 每種一個 `python -m iot` 行程，client id 自動加上種類（例如 `cat-iot-hub-env`），不會互踢。
- 只有處理 `weight` 的行程才檢查「多久沒進食」，只開環境感測的行程不會誤報。
- 手動 `python -m iot`（不設）跟原本一樣，全部感測器在一個行程。

### 設定（環境變數，統一 `CAT_MONITORING_IOT_*` 前綴）

常用：

| 環境變數 | 預設 | 說明 |
|---|---|---|
| `CAT_MONITORING_IOT_KINDS` | 空（全部） | 只處理這幾種感測器，逗號分隔（`env`／`motion`／`weight`／`bodytemp`）；設定視窗分開啟動各感測器用 |
| `CAT_MONITORING_IOT_MQTT_HOST` / `_PORT` | `192.168.0.171` / `1883` | broker 位置（跟 broker 同機時設 `127.0.0.1`） |
| `CAT_MONITORING_IOT_MQTT_USERNAME` / `_PASSWORD` | 空（匿名） | broker 帳密 |
| `CAT_MONITORING_IOT_TOPIC_PREFIX` | `cat/iot` | topic 前綴 |
| `CAT_MONITORING_IOT_DB_PATH` | `iot/data/iot_hub.db` | SQLite 路徑 |
| `CAT_MONITORING_IOT_TEMP_MIN_C` / `_TEMP_MAX_C` | `18` / `28` | 溫度告警門檻（°C） |
| `CAT_MONITORING_IOT_HUMIDITY_MIN_PCT` / `_MAX_PCT` | `40` / `70` | 濕度告警門檻（%） |
| `CAT_MONITORING_IOT_BODYTEMP_SURFACE_MIN_C` / `_MAX_C` | `20` / `40` | 體表溫初篩門檻（很寬，非診斷） |
| `CAT_MONITORING_IOT_GAS_PPM_MAX` | `1000` | 氣體告警門檻 |
| `CAT_MONITORING_IOT_FOOD_SCALE_IDS` | `food_bowl` | 食盆 scale_id（逗號分隔） |
| `CAT_MONITORING_IOT_FEEDING_SILENCE_HOURS` | `24` | 不進食告警門檻 |

完整清單見 `iot/config.py`。

---

## 持久化

自己的 SQLite（`iot/data/iot_hub.db`，WAL 模式），跟 `paper/baseline_data/` 完全分開。
資料表：`env_readings` / `bodytemp_readings` / `motion_events` / `weight_readings` / `weight_events` / `iot_alerts`。
`data/` 已在 `iot/.gitignore`，不進版控。

**資料保留（2026-09-27 起）**：外出包每秒一筆，24 小時連續跑一天約 17 萬筆、一個月約 400 MB，所以原始讀數
（`env_readings`／`bodytemp_readings`／`motion_events`／`weight_readings`）超過 `DATA_RETENTION_DAYS`（預設 30 天，
0＝不清理）的定時刪掉：hub 啟動時一次、之後每 6 小時；每個 hub 行程只清自己處理的感測器，分批刪不長時間鎖庫。
`weight_events`（進食事件，「疑似食慾不振」要用）和 `iot_alerts` 量小又有用，不清。刪掉的空間 SQLite 會重複利用，
檔案大小會停在穩定值（不會自己變小）。設定視窗 IoT 分頁工具列下方顯示資料庫路徑、大小、保留天數。

---

## Node-RED 端要新增什麼

本子系統**不動任何 Node-RED flow 檔**。要在 Dashboard 看到感測資料，請自行在
Node-RED 編輯器加：

1. **`mqtt in` 節點** × 1，topic `cat/iot/derived/#`，接 broker → `json` 節點 → Dashboard 卡片（溫濕度曲線、進食事件時間軸）。
2. **`mqtt in` 節點** × 1，topic `cat/iot/alert` → `json` → 既有的「發送提醒」/ Discord webhook 節點（payload 已含 `message` 欄位，可直接送）。

（若 broker 設定跟現有 flow 共用同一個 `mqtt-broker` config 節點即可。）

---

## 測試

```bash
pytest paper/cat_monitoring_system/iot/tests
```

不需要 `cv2` / `torch` / GPU / 真的 broker（MQTT client 在測試裡用假的）。

---

## 韌體

`iot/firmware/` 有五支 ESP32 Arduino 參考 sketch（環境＝空氣品質＋光照 / 移動 / 房間＝前兩者合併成一片 / 重量 / 外出包＝溫濕度＋體表溫度），
只是範本，CI 不會編譯。見 `firmware/README.md`。

`esp32_petbox/` 衍生自舊 Arduino 專案「寵物包/mqtt_all」，用 MLX90614 做非接觸體表
測溫，保留原專案的非阻塞 WiFi/MQTT 重連與 Arduino OTA。

---

## 後續（尚未做）

- 整併進 `settings_window` 管理：沿用專案既有「GUI 寫 env / `runtime_settings.json`、`config.py` 讀 env」模式。`iot/config.py` 的 env 變數命名已預留。
- 可選的 debug 用 HTTP 讀取 API（`/iot/latest`、`/iot/history`、`/healthz`）：獨立小 Flask、另起 port。
- 穿戴式 IMU 項圈：`sensors/` 加一個新模組 + `TOPIC_MAP` 加一個 kind 即可。
- **RFID / NFC 個體辨識**：見 [`docs/RFID個體辨識_未來方向.md`](docs/RFID個體辨識_未來方向.md)。多貓家庭在食盆/貓砂盆旁放讀取器 + 項圈掛 tag，標記「這次進食/如廁是哪隻貓」。目前只留設計方向，未實作。
