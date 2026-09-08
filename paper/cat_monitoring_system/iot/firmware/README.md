# ESP32 韌體參考 sketch

**範本** sketch，示範怎麼把感測資料用 MQTT 送到 `cat/iot/<kind>/<source_id>`，
格式對齊 `iot/README.md` 的 topic 約定。**CI 不會編譯這些檔案**，它們只是給你
照著改的起點。

| 資料夾 | 感測器（完整型號） | 發布 topic |
|---|---|---|
| `esp32_env/` | Winsen **MQ-135** air quality gas sensor + **GL5528** CdS photoresistor（LDR）　※溫濕度改由 `esp32_petbox` 負責 | `cat/iot/env/<SOURCE_ID>` |
| `esp32_motion/` | **HC-SR501** passive infrared (PIR) motion sensor module | `cat/iot/motion/<SOURCE_ID>` |
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

## payload 格式

- env：`{"gas_ppm":120.0,"lux":300.0}`（`esp32_env`）／`{"temp_c":26.4,"humidity_pct":55.0}`（`esp32_petbox`）——同一個 `cat/iot/env/<id>` topic，欄位皆可省略，hub 端會各自併起來
- motion：`{"active":true}`（狀態改變時才送）
- weight：`{"grams":123.4}`
- bodytemp：`{"surface_temp_c":34.2,"ambient_temp_c":26.0}`（petbox；`surface_temp_c` = MLX90614 對著貓量到的體表溫度）

hub 端對缺欄位、別名欄位、超出合理範圍的值都有容錯（見 `sensors/`），但照上面
格式送最單純。

> ⚠️ `esp32_petbox` 的 `surface_temp_c` 是**體表 / 毛髮表面溫度，不是核心體溫**，
> 受量測距離、毛色影響大，hub 端只拿它做「明顯異常」初篩，不是發燒診斷。
