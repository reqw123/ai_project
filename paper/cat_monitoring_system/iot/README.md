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

### 設定（環境變數，統一 `CAT_MONITORING_IOT_*` 前綴）

常用：

| 環境變數 | 預設 | 說明 |
|---|---|---|
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
資料表：`env_readings` / `motion_events` / `weight_readings` / `weight_events` / `iot_alerts`。
`data/` 已在 `iot/.gitignore`，不進版控。

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

`iot/firmware/` 有四支 ESP32 Arduino 參考 sketch（環境 / 移動 / 重量 / 外出包體溫），
只是範本，CI 不會編譯。見 `firmware/README.md`。

`esp32_petbox/` 衍生自舊 Arduino 專案「寵物包/mqtt_all」，用 MLX90614 做非接觸體表
測溫，保留原專案的非阻塞 WiFi/MQTT 重連與 Arduino OTA。

---

## 後續（尚未做）

- 整併進 `settings_window` 管理：沿用專案既有「GUI 寫 env / `runtime_settings.json`、`config.py` 讀 env」模式。`iot/config.py` 的 env 變數命名已預留。
- 可選的 debug 用 HTTP 讀取 API（`/iot/latest`、`/iot/history`、`/healthz`）：獨立小 Flask、另起 port。
- 穿戴式 IMU 項圈：`sensors/` 加一個新模組 + `TOPIC_MAP` 加一個 kind 即可。
- **RFID / NFC 個體辨識**：見 [`docs/RFID個體辨識_未來方向.md`](docs/RFID個體辨識_未來方向.md)。多貓家庭在食盆/貓砂盆旁放讀取器 + 項圈掛 tag，標記「這次進食/如廁是哪隻貓」。目前只留設計方向，未實作。
