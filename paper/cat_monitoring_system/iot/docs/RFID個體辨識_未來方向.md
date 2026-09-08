# RFID / NFC 個體辨識 —— 未來發展方向（設計筆記）

> 狀態：**尚未實作**，僅記錄設計方向。2026-09-07 建立。
> 這份不是實作計畫，是留給日後的起點。

---

## 動機

目前系統的個體辨識（`detectors/identity_verifier.py`）走的是視覺特徵比對，在多貓、
毛色相近、遮蔽、低光時可靠度有限——這也是口試三大硬核問題之一。

RFID / NFC 提供一條**物理層、高可靠**的補強路徑：貓的項圈掛一個被動式 tag，
在食盆 / 水盆 / 貓砂盆旁放讀取器，就能明確知道「這一次進食 / 飲水 / 如廁是哪一隻貓」。

用途定位（跟整個 `iot/` 一致）：**平行資料流、錦上添花**。
- 不取代視覺辨識，而是當它的**對照 / GT 來源**（讀取器旁同時有攝影機時，可以自動累積
  「視覺判為 A、RFID 也是 A」的一致率，量化視覺辨識的真實準確度）。
- 也可直接拿來把 `iot/` 收到的重量事件 / 移動事件標上貓 ID，讓每隻貓的進食量 /
  飲水量 / 如廁次數能分開統計。

---

## 現成素材（使用者舊 Arduino 專案）

`OneDrive/圖片/桌面/Arduino程式碼整理/` 底下：

| 專案 | 晶片 | 備註 |
|---|---|---|
| `RFID完全體/sketch_sep27a` | **MFRC522**（13.56 MHz，SPI） | 讀 UID + OLED + 蜂鳴器，程式較雜（混了跑馬燈 / 超音波），要抽乾淨 |
| `pn532版本集合體/sketch_sep20b` | **PN532**（13.56 MHz，I2C/SPI/HSU） | 較簡潔，`readPassiveTargetID` 讀 UID + `cardPresent` 進出偵測，已有「UID → 說明文字」對照的雛形 |

**建議用 PN532**：讀取距離比 MFRC522 遠（約 5–7 cm vs 2–3 cm），對「貓經過時項圈 tag
被讀到」的容錯高很多；`pn532版本集合體` 那份的結構也比較好改。

---

## 如果要做，大概長這樣

### 韌體：`iot/firmware/esp32_rfid/`

- ESP32 + PN532（I2C，SDA 21 / SCL 22）
- loop 裡 `readPassiveTargetID`，偵測到新卡（`cardPresent` 邊緣）就 publish：
  ```
  topic:   cat/iot/rfid/<reader_id>          （reader_id 例：food_bowl / litter_box）
  payload: {"uid": "04A1B2C3", "event": "enter"}   // 或 "leave"
  ```
- 沿用 `esp32_petbox` 的非阻塞 WiFi/MQTT 重連寫法

### Python 端（跟現有 `iot/` 完全一致的加法）

1. `config.py`：`TOPIC_MAP` 加 `"rfid": "rfid"`；加一張 `UID → 貓名` 對照
   （env 或小 JSON，例如 `CAT_MONITORING_IOT_RFID_TAG_MAP=04A1B2C3:mochi,05D6E7F8:kuro`）
2. `sensors/rfid.py`：`RfidParser` → `RfidEvent{reader_id, uid, cat_id, present, ts}`
   （`cat_id` 查對照表，查不到就留 `uid` 本身）
3. `storage/store.py`：加 `rfid_events` 表（`ts, reader_id, uid, cat_id, event`）
4. `runner.py`：`_handle_rfid` → 寫 store + `publish_derived("rfid", ...)`
5. 告警（可選）：某隻貓 `RFID_ABSENCE_HOURS` 小時內完全沒在任何讀取器出現 → 提醒

### 跟主系統的關聯（仍然零耦合）

- `iot/` 這邊只負責「哪隻貓、什麼時候、在哪個讀取器」的事實記錄與發布。
- 要做「視覺辨識 vs RFID 一致率」分析時，是在 **Node-RED 或另一支獨立分析腳本**
  裡把 `cat/iot/derived/rfid` 跟主系統推論結果對時間戳，`iot/` 本身不 import
  `detectors/`，維持雙向零依賴。

---

## 沒做的原因 / 前提

- 需要買 PN532 模組 + 貓項圈 tag，還要確認貓願意戴項圈。
- 讀取器要供電且放在食盆 / 貓砂盆旁，佈線與防護（貓砂粉塵、打翻水）要處理。
- 多貓時每隻要能穩定被讀到，實際擺位要現場調。

先把方向記著，等 `iot/` 主體（env / motion / weight / bodytemp）穩定跑一段時間、
確認 MQTT + SQLite 這套 pipeline 沒問題後再評估要不要加。
