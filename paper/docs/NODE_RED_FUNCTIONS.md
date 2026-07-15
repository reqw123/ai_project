# Node-RED 功能說明

更新日期：2026-06-03

這份文件整理 `paper/cat_monitoring_system/node-red.json` 與 `paper/cat_monitoring_system/gpt_api.json` 內的 Node-RED 功能，重點說明 Node-RED 在整個貓咪監測系統中的角色、資料流向、主要節點責任，以及和 GPT 分析 API 的串接方式。

---

## 1. Node-RED 在系統中的角色

Node-RED 負責把 Python 端的即時監測資料、影像串流、CSV 統計結果與 GPT 分析串接起來，並提供 Dashboard 顯示與外部通知能力。

它在系統中主要扮演四個工作：

1. 接收 Python 推論結果。
2. 整理與分發資料到 Dashboard 各區塊。
3. 提供使用者觸發健康分析的入口。
4. 串接 GPT API、Messenger、Discord 等外部服務。

---

## 2. 整體流程

### 2.1 即時監測流程

Python 偵測到貓咪行為後，會把結果送到 Node-RED 的 `/yolo_result`。
Node-RED 解析 JSON 後，將資料分發到：

- 即時狀態卡片
- 行為時間紀錄
- 詳細統計
- 健康警示
- 活動力儀表
- 影像串流卡片

### 2.2 GPT 健康分析流程

當使用者在 Dashboard 點擊健康分析按鈕，Node-RED 會：

1. 讀取 CSV 統計檔。
2. 將 CSV 原始資料包成 OpenAI Chat Completions 請求。
3. 呼叫 GPT 模型產生健康分析報告。
4. 將結果顯示在 Dashboard。
5. 若有 Messenger 需求，也會把摘要推播出去。

---

## 3. `node-red.json` 的主要功能

### 3.1 Python 上線通知與串流恢復

#### `python_online` (`http in`)
接收 Python 啟動後送來的上線資訊，主要內容是 Python 主機 IP。

#### `parse_json` (`json`)
把收到的 JSON 字串轉成物件。

#### `build_response` (`function`)
整理 IP、寫入 `global python_ip`，並組出影像串流 URL：

- `http://<ip>:5000/stream`

這樣 Dashboard 就能直接顯示 Flask MJPEG 串流。

#### `從持久化context恢復串流` (`function`)
Node-RED 重啟後，會從持久化 context 讀回 `python_ip`，避免每次都要等 Python 再送一次上線通知。

### 3.2 即時監測資料接收

#### `接收Python數據` (`http in`, `/yolo_result`)
Python 端把行為結果送進來的入口。

#### `解析JSON` (`json`)
將推論結果轉為可處理的物件。

#### `數據分發器` (`function`)
這是整個 Node-RED flow 最核心的整理節點，負責：

- 補齊 `current` 欄位。
- 補齊 `today_stats` 欄位。
- 整理 `behavior_log`。
- 計算 `activity_score` 與 `health_score` 的輔助資料。
- 把資料分流到不同 Dashboard 元件。

它同時也做防呆處理，例如：

- `walk_time`、`lick_time`、`scratch_time`、`shake_time`、`stop_time`
- `active_time`、`rest_time`、`not_detected_time`

如果欄位缺值，會補成預設值，避免 UI 出錯。

### 3.3 Dashboard 顯示元件

#### `即時狀態卡片`
顯示：

- 目前行為
- 時間戳
- 活動力分數
- 背景圖與狀態樣式

#### `行為時間紀錄`
顯示最近的行為事件時間軸，例如：

- 行走
- 舔舐
- 搔抓
- 甩頭
- 靜止
- 不在畫面

#### `詳細統計`
顯示今日統計數據，包括：

- walk / walk_time
- lick / lick_time
- scratch / scratch_time
- shake / shake_time
- stop / stop_time
- active_time
- rest_time

#### `健康警示`
顯示健康分數與風險提示，並根據統計資料推估是否需要關注異常舔舐、活動力下降或其他行為異常。

#### `活動力儀表`
用 gauge 顯示活動力分數，讓使用者快速判斷目前活躍程度。

### 3.4 啟動與通知

#### `主動推播`
可將異常警報主動送到 Messenger 指定使用者。

#### `一般回復`
當收到非特定指令時，回覆系統已收到訊息。

#### `攝影機畫面`
收到 `/camera` 指令時，觸發即時截圖流程（見 3.5 節）。

#### `健康報告`
當收到「哈基米」指令時，會記錄請求者 ID，並啟動 CSV 分析流程。

### 3.5 `/camera` 即時截圖流程

更新日期：2026-06-03

`/camera` 指令不再回傳靜態圖片，而是擷取 Flask 當前的串流幀，完整流程如下：

```
攝影機畫面（function）
    → 取得截圖（http request，GET binary）
        → 處理截圖（function，3 個輸出）
              out1 → 儲存截圖（file write）
              out2 → 即時截圖卡片（ui_template，Dashboard 顯示）
              out3 → FB API（Messenger 文字通知）
```

#### `攝影機畫面` (`function`)
從 `global.python_ip` 組出 `http://<ip>:5000/snapshot`，設定 `msg.method = 'GET'`，並將 `msg.sender` 暫存至 `msg._sender` 供後續 Messenger 回覆使用。

#### `取得截圖` (`http request`)
以 binary 模式（`ret: "bin"`）向 Flask `/snapshot` 發出 GET 請求，取得 JPEG 二進位資料。

#### `處理截圖` (`function`)
對收到的 buffer 做三件事：

1. 產生帶時間戳的本地存檔路徑（`C:\a\snapshot_<時間戳>.jpg`）
2. 轉換為 base64 data URL（`data:image/jpeg;base64,...`）
3. 分三路輸出：
   - `msg1`（buffer + filename）→ 寫入本地磁碟
   - `msg2`（b64 + ts + filename）→ Node-RED Dashboard 截圖卡片
   - `msg3`（Messenger payload）→ 通知 Messenger 使用者「截圖已擷取，請查看 Dashboard」

#### `儲存截圖` (`file`)
以 `msg.filename` 為路徑寫入 JPEG 檔，`createDir: true` 會自動建立目錄，`overwriteFile: true` 避免重複檔名錯誤。

#### `即時截圖卡片` (`ui_template`)
顯示於 Dashboard「AI健康分析」Tab 的「即時截圖」Group，以 `ng-src` 綁定 base64 圖片，並顯示擷取時間與本地儲存路徑。初始狀態顯示「尚未擷取截圖」提示。

---

### 3.6 Messenger webhook 防重複觸發

Facebook Messenger 在 bot 發出訊息後，會把該訊息以 `is_echo: true` 的事件再次送回 webhook，若不過濾會導致指令被重複執行。

`解析Messenger訊息` 目前加入三道過濾：

| 過濾條件 | 原因 |
|---|---|
| `!event.message` | delivery / read receipt，沒有訊息內容 |
| `event.message.is_echo` | bot 自己送出的訊息被 FB 回傳，不應處理 |
| `!event.message.text` | 貼圖、圖片、附件等非文字訊息 |

任一條件成立時，函數回傳 `null`，訊息不再進入後續分流。

---

## 4. `gpt_api.json` 的主要功能

這份 flow 是專門處理 CSV 交給 GPT 分析的流程，核心用途是產生「貓咪健康報告」。

### 4.1 CSV 讀取

#### `開始分析CSV` (`inject`)
手動或由其他流程觸發分析。

#### `讀取CSV` (`file in`)
讀取 CSV 檔案內容，目前設定的來源是 `C:\a\noda_data.csv`。

### 4.2 建立 GPT 請求

#### `建立GPT分析請求` (`function`)
這個節點會：

- 讀取環境變數 `OPENAI_API_KEY`
- 組成 OpenAI Chat Completions 請求
- 把 CSV 原始內容放入 user message
- 將系統提示詞寫入 system message
- 指定模型為 `gpt-4.1-mini`

系統提示詞明確定義：

- 可分析的欄位
- 舔舐、甩頭、活動力、靜止等判斷原則
- 不可捏造不存在數據
- 必須使用繁體中文輸出

### 4.3 呼叫 OpenAI API

#### `OpenAI GPT分析` (`http request`)
把請求送到：

- `https://api.openai.com/v1/chat/completions`

### 4.4 結果解析與顯示

#### `解析GPT結果` (`function`)
這個節點會：

- 解析 GPT 回傳的 `choices[0].message.content`
- 統計 token 使用量
- 估算成本
- 將結果轉回 `msg.payload.result`
- 若有 Messenger 請求者，會把摘要推播出去

#### `GPT分析結果` (`debug`)
用來檢查 GPT 回傳內容。

#### `AI健康卡片` (`ui_template`)
將 GPT 報告顯示成 Dashboard 卡片介面，並提供：

- 按鈕觸發
- 分析中狀態
- 結果分段顯示
- 最後分析時間

---

## 5. Node-RED 與 GPT API 的資料銜接方式

### 5.1 啟動健康分析

Dashboard 的按鈕會呼叫 `/ui-trigger-health`。

#### `UI觸發健康分析` (`http in`)
接收前端按鈕觸發。

#### `UI觸發-回應+啟動CSV` (`function`)
同時做兩件事：

- 立刻回應前端 `ok`
- 觸發 CSV 讀取與 GPT 分析流程

### 5.2 分析報告輸出

CSV 內容會進入 `建立GPT分析請求`，再送到 OpenAI。
GPT 回傳的內容最後會顯示在：

- Dashboard AI 健康卡片
- Debug 面板
- 若設定請求者，則推播到 Messenger

---

## 6. Node-RED 內的資料契約

Node-RED 目前預期 Python 端與 GPT 端輸入輸出符合下列欄位：

### 6.1 Python 推論資料

至少包含：

- `current`
- `today_stats`
- `behavior_log`
- `activity_score`
- `health_score`
- `alerts`

### 6.2 CSV 健康分析資料

CSV 欄位需包含：

- `timestamp`
- `walk`
- `walk_time`
- `lick`
- `lick_time`
- `scratch`
- `scratch_time`
- `shake`
- `shake_time`
- `stop`
- `stop_time`
- `active_time`
- `rest_time`
- `not_detected_time`
- `activity_score`
- `health_score`

GPT prompt 也已明確要求只能使用這些實際存在的欄位。

---

## 7. 這份 Node-RED 設計的重點

這份 flow 的核心不是單純顯示資料，而是把整個系統分成三個層次：

1. Python 端提供即時偵測與統計。
2. Node-RED 端負責資料整併、展示與外部通知。
3. GPT API 端負責把 CSV 統計轉成可讀的健康分析報告。

換句話說，Node-RED 是整個系統的「中控台」：

- 接資料
- 整資料
- 顯示資料
- 對外推播
- 觸發 GPT 分析

---

## 8. 建議後續可補充的文件

如果要繼續整理，建議下一步可以再拆成兩份：

1. `Node-RED Dashboard 元件說明`
2. `GPT 健康分析 Prompt 說明`

這樣會更容易維護，也方便之後調整 flow 或 prompt 時同步更新文件。
