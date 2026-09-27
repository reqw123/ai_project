# ESP32 韌體參考 sketch

**範本** sketch，示範怎麼把感測資料用 MQTT 送到 `cat/iot/<kind>/<source_id>`，
格式對齊 `iot/README.md` 的 topic 約定。**CI 不會編譯這些檔案**，它們只是給你
照著改的起點。

| 資料夾 | 感測器（完整型號） | 發布 topic |
|---|---|---|
| `esp32_env/` | （**備用**：感測器要分開放時才用，平常燒 `esp32_room`）Winsen **MQ-135** air quality gas sensor + **GL5528** CdS photoresistor（LDR）　※只負責空氣品質＋光照；溫濕度由 `esp32_petbox` 負責 | `cat/iot/env/<SOURCE_ID>` |
| `esp32_room/` | **主要使用**（2026-09-27 起）：esp32_env＋esp32_motion 合併成一片，MQ-135 ＋ GL5528 ＋ HC-SR501。感測器放同一個房間時用，省一片開發板；topic 跟分開的兩支完全一樣，hub 不用改。⚠ MQ-135 AOUT 要經 10k/20k 分壓再進 GPIO 34 | `cat/iot/env/<SOURCE_ID>` + `cat/iot/motion/<SOURCE_ID>` |
| `esp32_motion/` | （**備用**：感測器要分開放時才用，平常燒 `esp32_room`）**HC-SR501** passive infrared (PIR) motion sensor module | `cat/iot/motion/<SOURCE_ID>` |
| `esp32_weight/` | Avia Semiconductor **HX711** 24-bit load-cell ADC + straight-bar strain-gauge load cell（如 SparkFun **TAL220** 5 kg） | `cat/iot/weight/<SCALE_ID>` |
| `esp32_petbox/` | Melexis **MLX90614ESF-BAA** IR 非接觸測溫 + Aosong **DHT11** 溫濕度 + Sitronix **ST7789** 240x240 IPS TFT（舊專案「寵物包/mqtt_all」還原：卡片式 UI、`yuanpei` 開機圖、非阻塞重連、OTA；BluetoothSerial 已註解關閉，`// [BT]` 行取消註解可恢復） | `cat/iot/env/<SOURCE_ID>` + `cat/iot/bodytemp/<SOURCE_ID>`（另保留原專案的 `esp32/sensors`、`esp32/alert`） |

各 sketch 開頭的「硬體 / 感測器型號」註解區有完整型號、可替換方案與接腳對照。

## 共同需求

- **開發板**：ESP32（Arduino core for ESP32 ≥ 2.0）
- **函式庫**（Arduino Library Manager）：
  - `PubSubClient`（MQTT，Nick O'Leary）
  - `ArduinoJson`（≥ 6.x）
  - `DHT sensor library`（Adafruit，僅 petbox）
  - `HX711`（Bogdan Necula / Rob Tillaart 版皆可，僅 weight）
  - `Adafruit MLX90614 Library` + `TFT_eSPI`（僅 petbox；`yuanpei.h` 開機圖已附在 `esp32_petbox/` 資料夾內，跟 `.ino` 一起開）

## 設定

每支 sketch 最上面都有一段要改：

```cpp
const char* WIFI_SSID     = "CBN-B4640-2.4G";   // 已預填
const char* WIFI_PASSWORD = "110106291208";     // 已預填
const char* MQTT_HOST     = "192.168.0.171";    // 已預填（mosquitto / Node-RED 那台）
const uint16_t MQTT_PORT  = 1883;
const char* SOURCE_ID     = "living_room";      // 會變成 topic 最後一段
```

`SOURCE_ID` / `SCALE_ID` 要跟 `iot/config.py` 的 `FOOD_SCALE_IDS` 之類設定對得起來
（例如食盆秤台就設 `food_bowl`）。

## 網路主機名稱（2026-09-27 起）

每支 sketch 在 `WiFi.mode()` 前呼叫 `WiFi.setHostname("cat-<種類>-<節點>")`（底線換成連字號，主機名稱不能有底線）：

| sketch | 主機名稱 | 設定視窗顯示在哪張卡片 |
|---|---|---|
| esp32_env（備用） | `cat-env-living-room` | 環境感測（空氣品質＋光照） |
| esp32_room | `cat-room-living-room` | 環境感測（空氣品質＋光照）＋移動偵測 |
| esp32_motion（備用） | `cat-motion-food-area` | 移動偵測 |
| esp32_weight | `cat-weight-food-bowl` | 食盆秤重 |
| esp32_petbox（＝`C:\arduino\語音辨識-v7\環境監測\mqtt_all`） | `cat-petbox-carrier` | 環境感測（溫濕度）＋體表溫度 |
| 語音終端（`C:\arduino\語音辨識-v7`，不在這裡） | `cat-voice…` | 飼主語音紀錄 |

MQTT 訊息不帶發送者 IP；設定視窗「📡 IoT 子系統」把連進 broker 的 IP 拿去路由器 DNS 反查，
看主機名稱是 `cat-<種類>` 開頭就把 IP 顯示在對應卡片（環境感測卡片會同時列 esp32_env 和外出包兩個 IP）（`settings_gui/iot_services.py` 的 `_HOST_KINDS`）。
還沒燒新韌體的會是預設名稱（例如 `esp32-DA98B8`），顯示成「未辨識裝置」。
路由器可能要等 ESP32 重新向 DHCP 要 IP（重開機）才會更新名稱。

## payload 格式

- env：`{"gas_ppm":120.0,"lux":300.0}`（`esp32_env`）／`{"temp_c":26.4,"humidity_pct":55.0}`（`esp32_petbox`）——同一個 `cat/iot/env/<id>` topic（不同 id），欄位皆可省略，hub 端會各自併起來
- motion：`{"active":true}`（狀態改變時才送）
- weight：`{"grams":123.4}`
- bodytemp：`{"surface_temp_c":34.2,"ambient_temp_c":26.0}`（petbox；`surface_temp_c` = MLX90614 對著貓量到的體表溫度）

hub 端對缺欄位、別名欄位、超出合理範圍的值都有容錯（見 `sensors/`），但照上面
格式送最單純。

> ⚠️ `esp32_petbox` 的 `surface_temp_c` 是**體表 / 毛髮表面溫度，不是核心體溫**，
> 受量測距離、毛色影響大，hub 端只拿它做「明顯異常」初篩，不是發燒診斷。
