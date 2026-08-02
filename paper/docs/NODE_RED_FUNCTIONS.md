# Node-RED 功能說明

更新日期：2026-07-20

這份文件整理 `paper/貓咪主控.json` 與 `paper/GPT 健康報告.json` 內的 Node-RED 功能，重點說明 Node-RED 在整個貓咪監測系統中的角色、資料流向、主要節點責任，以及和 GPT 分析 API 的串接方式。

> ⚠️ **範圍說明**：本文件只涵蓋這 2 個 flow。專案目前實際共有 4 個 Node-RED flow 檔案（皆位於 `paper/` 根目錄，**不在** `cat_monitoring_system/` 下），另外 2 個 `cat_health_v3_flow.json`（個體化基線分析引擎）與 `lick_stage2_nodered.json`（舔舐部位分析 Dashboard）請見 `0_AI_專案導覽地圖.md` 的「五、監控層 Node-RED 對應」一節。

---

## 1. Node-RED 在系統中的角色

Node-RED 負責把 Python 端的即時監測資料、影像串流、CSV 統計結果與 GPT 分析串接起來，並提供 Dashboard 顯示與外部通知能力。

它在系統中主要扮演四個工作：

1. 接收 Python 推論結果，計算健康風險分數並整理成 Dashboard 各區塊資料。
2. 定時把統計資料寫入 CSV，供 GPT 分析與人工查閱使用。
3. 提供使用者透過 Messenger 文字指令或 Dashboard 按鈕觸發健康分析、錄影的入口。
4. 串接 GPT API、Messenger、Discord 等外部服務。

---

## 2. 整體流程

### 2.1 即時監測流程

Python 偵測到貓咪行為後，會把結果送到 Node-RED 的 `/yolo_result`（`貓咪主控.json`）。
Node-RED 解析 JSON 後，`數據分發器` 節點會把資料同時分發到 7 條路徑：

- 即時狀態卡片
- 行為時間紀錄（近期 `behavior_log` 原始列表）
- 詳細統計
- 健康風險評分引擎（`健康引擎分析` → 異常時觸發 `發送提醒` 到 Discord）
- 活動力儀表
- CSV 統計寫入（每 5 分鐘節流一次，寫入 `C:\a\noda_data.csv`）
- 行為時間軸引擎（合併相鄰同行為片段，餵給「行為事件分析」卡片）

### 2.2 GPT 健康分析流程

GPT 健康分析可由兩個入口觸發：Messenger 文字指令「哈基米」，或 Dashboard 按鈕（呼叫 `/ui-trigger-health`）。觸發後 Node-RED 會：

1. 讀取 CSV 統計檔（`C:\a\noda_data.csv`，就是 `貓咪主控.json` 定時寫入的那份）。
2. 將 CSV 原始資料包成 OpenAI Chat Completions 請求。
3. 呼叫 GPT 模型產生健康分析報告。
4. 將結果顯示在 Dashboard「AI健康卡片」。
5. 若是由 Messenger 觸發（有記錄請求者 ID），額外把摘要推播回 Messenger。

---

## 3. `貓咪主控.json` 的主要功能

唯一 tab：「😺 貓咪健康監測系統」，**目前運行中的主要 Dashboard**（對應 `config.py` 的 `NodeRedConfig.ENDPOINT_NOTIFY`／`ENDPOINT_RESULT`）。

### 3.1 Python 上線通知與串流恢復

#### `python_online` (`http in`, `/python_online`)
接收 Python 啟動後送來的上線資訊，主要內容是 Python 主機 IP。

#### `parse_json` (`json`)
把收到的 JSON 字串轉成物件。

#### `build_response` (`function`)
整理 IP、寫入持久化 `global.python_ip`（`file` context，Node-RED 重啟後不遺失），並組出串流 URL `http://<ip>:5000/stream`，同時觸發 Discord 上線通知與影像串流卡片更新。

#### `從持久化context恢復串流` (`function`)
啟動後 2 秒觸發一次（`啟動時發送串流` inject），從持久化 context 讀回 `python_ip`，避免 Node-RED 重啟後要等 Python 再送一次上線通知，串流卡片才有畫面。

#### `組成 Discord Webhook 格式` (`function`) → `送到 Discord` (`http request`)
把 Python 上線事件包成 Discord embed（含 IP、時間戳），推播到寫死在節點內的 webhook URL。

#### 串流顯示元件
`影像串流`（`ui_template`，「即時影像串流」group）與 `懸浮影像（全域）`（`ui_template`，不屬於任何 group，跨 tab 常駐懸浮視窗）兩個元件都會接收 `build_response`／`從持久化context恢復串流` 的 `stream_url` 更新畫面。

### 3.2 即時監測資料接收與分發

#### `接收Python數據` (`http in`, `/yolo_result`)
Python 端把行為結果送進來的入口，立即以 `回應200` 回 200 OK。

#### `解析JSON` (`json`)
將推論結果轉為可處理的物件。

#### `數據分發器` (`function`，7 個輸出)
整個 flow 最核心的整理節點，負責：

- 補齊 `current`／`today_stats` 各欄位（`walk`/`walk_time`/`lick`/`lick_time`/`scratch`/`scratch_time`/`shake`/`shake_time`/`stop`/`stop_time`/`active_time`/`low_conf`/`low_conf_time`/`not_detected_time`），缺值一律補 0，避免 UI 出錯。
- 整理 `behavior_log`，補上對應 emoji。
- 計算 `avg_duration`（今日各行為平均時長，供時間軸風險評估用）與行為開始時間戳（`start_epoch_ms`，供時間軸「進行中」計時器使用）。
- 依「距上次寫檔是否已過 5 分鐘」決定要不要組出這次的 CSV 列（`flow.last_csv_write` 節流）。
- 把同一份／加工後的訊息分流到 7 條輸出：即時狀態卡片、行為時間紀錄、詳細統計、健康引擎分析、活動力儀表（`activity_score` 數值）、CSV 寫入（節流後才有值，否則為 `null`）、行為時間軸引擎。

### 3.3 健康風險評分引擎（舊文件未記載）

#### `健康引擎分析` (`function`)
比對舊版文件單純的「健康警示」，現行設計是一套四維加權風險評分：

- **dScore（行為佔比偏離，權重 30%）**：舔舐/搔抓/甩頭/靜止佔比超過可設定閾值（讀 `global.v2_user_settings`，未設定時走預設值：舔舐 20%／搔抓 15%／甩頭 10%／靜止 55%）。
- **fScore（頻率偏離，權重 35%）**：與 `global.v2_baseline`（`cat_health_v3_flow.json` 算出的個體化基線）比對百分比偏差；沒有基線時固定為 0。
- **rScore（節律偏離，權重 20%）**：凌晨 00-06 時段是否以舔舐/搔抓/甩頭為主、白天走動佔比是否異常偏低。
- **tScore（行為轉移偏離，權重 15%）**：`lick↔scratch` 循環次數、`shake→shake`／`stop→stop` 連續轉移次數異常。
- 綜合分數對應 4 個等級：Normal（<20）／Attention（<45）／Warning（<70）／High Risk（≥70），連同 `alerts` 陣列一起寫入 `data.risk`，同時把當日統計同步寫回 `global.v2_today`/`v2_hourly`/`v2_transition_matrix`（`file` context），讓 `cat_health_v3_flow.json` 的基線引擎能繼續運作。

#### `發送提醒` (`function`) → `送到 Discord` (`http request`)
`risk.level` 非 Normal 時才觸發；用「等級 + 異常類型」組成 dedup key（`flow.last_discord_alert`），同一組合不會重複發送。Webhook URL 讀自 `global.v2_user_settings.discord_webhook`（未設定時只印警告、不發送）——跟 3.1 節上線通知用的固定 webhook URL 是**不同**的設定來源。

### 3.4 CSV 統計寫入

#### `建立CSV` (`csv`) → `儲存CSV` (`file`)
把 `數據分發器` 節流後（每 5 分鐘一次）組出的統計列，以 `appendNewline` 附加寫入 `C:\a\noda_data.csv`，欄位順序為：

```
timestamp,walk,walk_time,lick,lick_time,scratch,scratch_time,shake,shake_time,
stop,stop_time,active_time,low_conf,low_conf_time,not_detected_time,activity_score
```

這份 CSV 就是 `GPT 健康報告.json` 讀取分析的資料來源（見 4.3 節）。

> 2026-07 更新：`GPT 健康報告.json` 的 `建立GPT分析請求` 節點（見 4.3）原本的 system prompt 描述 CSV 有 `rest_time`／`health_score` 欄位，跟這裡實際寫出的表頭對不上；已修正 prompt 改用 `low_conf`／`low_conf_time`，並移除 `health_score`（CSV 本來就沒有這欄）——「四、整體健康風險評估」也改成請 GPT 依 `activity_score`/`lick_time`/`scratch_time`/`shake`/`stop_ratio`/`low_conf_time` 自行綜合判斷風險等級，不再依賴不存在的 `health_score` 門檻值。

### 3.5 行為時間軸引擎（舊文件未記載）

#### `事件時間軸引擎` (`function`)
比舊文件的「行為時間紀錄」更進一步：讀取 `C:\a\tracker_state.json`（Python 端權威每日累積資料）校對日期，把 Python 送來的 `behavior_log` 依 `時間+行為+時長` 去重後累積進 `flow.evt_session`（上限 500 筆，跨日自動清空），並把時間間隔 ≤2 秒的相鄰同行為片段合併成一段完整事件（Python 端约每 2 秒送一段，合併後才是使用者看到的一次完整行為）。

#### `行為事件分析` (`ui_template`，獨立 group)
顯示合併後的行為事件時間軸，跟 3.2 節的「行為時間紀錄」卡片（顯示未合併的原始 `behavior_log`）是兩個不同用途的顯示元件。

### 3.6 Dashboard 顯示元件總覽

Tab「😺 貓咪健康監測系統」底下共 7 個 group：

| Group（顯示順序） | 元件 | 資料來源 |
|---|---|---|
| 即時健康狀態 (1) | `即時狀態卡片` | `數據分發器` 輸出 1 |
| 行為時間紀錄 (2) | `行為時間紀錄` | `數據分發器` 輸出 2（原始 `behavior_log`） |
| 詳細統計分析 (3) | `詳細統計` | `數據分發器` 輸出 3 |
| 健康警示系統 (4) | `健康警示` | `健康引擎分析` 的 `data.risk`/`data.alerts` |
| 活動力指標 (5) | `活動力儀表` (`ui_gauge`) | `數據分發器` 輸出 5（`activity_score`） |
| 即時影像串流 (6) | `影像串流` | `build_response`／`從持久化context恢復串流` |
| 行為事件分析 (7) | `行為事件分析` | `事件時間軸引擎`（合併後事件） |

另外兩個不屬於任何 group、跨 tab 常駐的元件：`🎨 Dashboard Global Theme`（CSS 主題注入）與 `懸浮影像（全域）`（浮動視窗版影像串流）。

---

## 4. `GPT 健康報告.json` 的主要功能

唯一 tab：「CSV AI分析系統」，**`"disabled": true`**——目前未在運行中的 Node-RED 實例內生效。

這份 flow 專門處理「Messenger 機器人指令」與「CSV 交給 GPT 分析」兩件事，核心用途是產生「貓咪健康報告」。

### 4.1 Messenger webhook 驗證與防重複觸發

#### `Webhook驗證` (`http in`, `GET /messengerwebhook`) → `解析訊息` (`function`)
Facebook 訂閱驗證流程：比對 `hub.verify_token` 是否等於寫死的 `cat_ai_system`，成功回傳 `hub.challenge`，失敗回 403。

#### `Messenger接收入口` (`http in`, `POST /messengerwebhook`) → `解析Messenger訊息` (`function`)
先立刻回 200（Facebook 要求快速 ACK），再解析事件。加入三道過濾避免重複觸發：

| 過濾條件 | 原因 |
|---|---|
| `!event.message` | delivery / read receipt，沒有訊息內容 |
| `event.message.is_echo` | bot 自己送出的訊息被 FB 回傳，不應處理 |
| `!event.message.text` | 貼圖、圖片、附件等非文字訊息 |

任一條件成立時，函數回傳 `null`，訊息不再進入後續分流。

### 4.2 指令分流

#### `指令分流` (`switch`)
依 `msg.text` 完全比對，目前 5 個輸出：`哈基米` → `健康報告`、`/camera` → `攝影機畫面`、`/help` → `/help`、`/日報`（規則存在但沒有接任何節點，目前無效果）、其餘 → `一般回復`。

> `/status` 指令原本會呼叫 Flask 端不存在的 `/status` 路由、持續失敗；2026-07 已把該指令從這個 switch、對應的請求/回覆節點、以及 `/help` 說明文字中整個移除。

#### `健康報告` (`function`)
收到「哈基米」時：記錄請求者 ID 到 `flow.health_requester`、回覆「正在分析，請稍候」，並觸發 CSV 讀取（見 4.3）。

#### `一般回復` (`function`)
收到不認得的文字時，回覆「🤖 AI 系統已收到：<原文>」。

### 4.3 CSV → GPT 分析流程

```
開始分析CSV / 健康報告 / UI觸發-回應+啟動CSV（三個入口皆可觸發）
    → 讀取CSV（file in，C:\a\noda_data.csv）
        → 建立GPT分析請求（function）
            → OpenAI GPT分析（http request → api.openai.com/v1/chat/completions）
                → 解析GPT結果（function）
                      → GPT分析結果（debug）+ AI健康卡片（ui_template，Dashboard 顯示）
                      → （若有 flow.health_requester）推播摘要到 Messenger
```

#### `讀取CSV` (`file in`)
讀取 `C:\a\noda_data.csv`（由 `貓咪主控.json` 的 `建立CSV`/`儲存CSV` 定時寫入，見 3.4 節）。

#### `建立GPT分析請求` (`function`)
讀取環境變數 `OPENAI_API_KEY`，組成 OpenAI Chat Completions 請求（`model: gpt-4.1-mini`），system prompt 明確定義行為類別語意、CSV 欄位說明（已對齊 3.4 節實際 CSV 表頭：`low_conf`/`low_conf_time`，不含 `health_score`）、分析原則（禁止捏造不存在欄位/數據）與輸出格式要求。

#### `解析GPT結果` (`function`)
解析 `choices[0].message.content`，統計並印出 token 用量與估算成本（GPT-4.1-mini 費率：輸入 $0.4/1M、輸出 $1.6/1M），回傳 `msg.payload.result`；若 `flow.health_requester` 有值，額外組一則 Messenger 訊息（超過 1900 字會截斷並附上「完整報告請查看 Dashboard」提示）推播給該使用者，並清空 `health_requester`。

#### `AI健康卡片` (`ui_template`)
Dashboard「GPT分析結果」group 內的卡片，前端 JS 會把 GPT 回傳文字依「數字開頭段落」規則切成多張子卡片顯示，並提供「開始健康分析」按鈕直接呼叫 `/ui-trigger-health`（不需要透過 Messenger）。

### 4.4 `/camera` 錄影流程

> ⚠️ **與舊版文件不同**：`/camera` 指令目前呼叫的是 Flask **`/video_clip`**（回傳最近幾秒的短片，見 `server/routes.py`），**不是**舊版文件描述的 `/snapshot`（單張截圖）。Flask 端已經把短片存檔並回傳 JSON（`path`/`duration`/`frames`/`ts`/`thumbnail`），Node-RED 這邊不需要也沒有再自己寫檔。

```
攝影機畫面 / UI錄影-組請求（function，組出 GET /video_clip 請求）
    → 取得截圖（http request，obj 模式）
        → 處理截圖（function，3 個輸出）
              out1 → null（Flask 已存檔，Node-RED 不需再寫檔）
              out2 → 即時截圖卡片（ui_template，Dashboard 顯示縮圖）
              out3 → FB API（Messenger 文字通知，僅在有 sender 時才送）
```

#### `攝影機畫面` (`function`)
Messenger `/camera` 指令觸發：從 `global.python_ip` 組出 `http://<ip>:5000/video_clip`，`msg.method = 'GET'`，並把 `msg.sender` 暫存至 `msg._sender` 供後續 Messenger 回覆使用。

#### `UI錄影-組請求` (`function`)
Dashboard「立即錄影」按鈕觸發的等價節點，同樣組 `GET /video_clip`，但 `msg._sender = null`（不透過 Messenger 通知）。

#### `取得截圖` (`http request`)
以物件模式（`ret: "obj"`）向 Flask `/video_clip` 發出 GET 請求，直接取得 JSON（Flask 已經把短片存檔並回傳中繼資料，不是二進位影像資料）。

#### `處理截圖` (`function`)
1. 若 Flask 回傳失敗（無 buffer/`data.error`/沒有 `data.path`），只在有 `sender`（即 Messenger 觸發）時回覆錯誤訊息，說明可能原因（系統尚未啟動、ring buffer 尚未累積滿 5 秒、Flask 未執行）。
2. 成功時三路輸出：`msg1=null`（不需要再存檔）、`msg2`（`ts`/`path`/`duration`/`frames`/`thumbnail`，餵給 Dashboard 卡片）、`msg3`（僅在有 `sender` 時組 Messenger 通知，含時長/幀數/存檔路徑）。

#### `即時截圖卡片` (`ui_template`)
顯示於 Dashboard「AI健康分析」Tab 的「即時截圖」Group，用 `ng-src` 綁定 base64 縮圖，並顯示擷取時間、時長、幀數、本地儲存路徑；也有自己的「立即錄影」按鈕直接呼叫 `/ui-trigger-record`。初始狀態顯示「尚未錄製」提示。

### 4.5 Dashboard 觸發端點

#### `UI觸發健康分析` (`http in`, `POST /ui-trigger-health`) → `UI觸發-回應+啟動CSV` (`function`)
立刻回應前端 `200 ok`，同時觸發 4.3 節的 CSV 讀取／GPT 分析流程（等同 Messenger 的「哈基米」指令，但不會有 `health_requester`，所以分析完只會顯示在 Dashboard，不會推播 Messenger）。

#### `UI觸發錄影` (`http in`, `POST /ui-trigger-record`) → `UI錄影-組請求` (`function`)
立刻回應前端 `200`，同時觸發 4.4 節的 `/video_clip` 錄影流程（`sender=null`，不推播 Messenger）。

### 4.6 其他

#### `儲存Token` (`function`)
由 `設定Token` inject 節點在 Node-RED 啟動 0.5 秒後觸發一次，把 Facebook Page Access Token **直接寫死在節點程式碼內**（並非讀取環境變數）存進 `global.FB_TOKEN`，供所有呼叫 Messenger API 的節點使用。

#### `異常警報` (`inject`) → `主動推播` (`function`)
手動測試用節點：注入一段固定文字，推播到程式碼裡寫死的單一 Messenger 使用者 ID，跟 3.3 節「Discord 異常告警」是兩條互相獨立的告警管道（一個推 Discord、一個推特定人的 Messenger），非同一套機制觸發。

---

## 5. Node-RED 與 GPT API 的資料銜接方式

### 5.1 啟動健康分析的兩種入口

- **Messenger**：使用者傳「哈基米」→ `指令分流` → `健康報告` function（記錄 `health_requester`）→ CSV 讀取流程。
- **Dashboard**：按鈕呼叫 `/ui-trigger-health` → `UI觸發-回應+啟動CSV` → CSV 讀取流程（不記錄 `health_requester`）。

兩者最終都會匯入同一條 `讀取CSV → 建立GPT分析請求 → OpenAI GPT分析 → 解析GPT結果` 管線（見 4.3 節），差別只在於有沒有 Messenger 使用者要推播結果。

### 5.2 分析報告輸出

GPT 回傳的內容最後會顯示在：

- Dashboard「AI健康卡片」
- Debug 面板
- 若由 Messenger 觸發（有 `health_requester`），額外推播到 Messenger

---

## 6. Node-RED 內的資料契約

### 6.1 Python 推論資料（`POST /yolo_result`）

`數據分發器` 實際會補齊／使用的欄位：

- `current`（含 `behavior`/`emoji`/`timestamp`/`start_epoch_ms`）
- `today_stats`（`walk`/`walk_time`/`lick`/`lick_time`/`scratch`/`scratch_time`/`shake`/`shake_time`/`stop`/`stop_time`/`active_time`/`low_conf`/`low_conf_time`/`not_detected_time`，缺值補 0）
- `behavior_log`（陣列，每筆含 `behavior`/`time`/`duration`）
- `activity_score`
- `hourly_distribution`（供健康引擎節律分析）
- `transition_matrix`（供健康引擎轉移偏離分析）

### 6.2 CSV 健康分析資料（`C:\a\noda_data.csv`）

**實際寫出的欄位**（`貓咪主控.json` 的 `建立CSV` 節點）：

```
timestamp, walk, walk_time, lick, lick_time, scratch, scratch_time,
shake, shake_time, stop, stop_time, active_time,
low_conf, low_conf_time, not_detected_time, activity_score
```

**GPT prompt 描述的欄位**（`GPT 健康報告.json` 的 `建立GPT分析請求` 節點）：2026-07 已同步改為上面這組實際欄位（`low_conf`/`low_conf_time`，不含 `health_score`），兩邊定義一致。

---

## 7. 這份 Node-RED 設計的重點

這兩條 flow 的核心不是單純顯示資料，而是把系統分成三個層次：

1. Python 端提供即時偵測與統計。
2. Node-RED 端負責資料整併、健康風險評分、Dashboard 展示與外部通知（Discord／Messenger）。
3. GPT API 端負責把 CSV 統計轉成可讀的健康分析報告。

換句話說，Node-RED 是整個系統的「中控台」：接資料、算風險分數、整資料、顯示資料、對外推播、觸發 GPT 分析。

專案實際上還有另外 2 條 flow（個體化基線引擎、舔舐部位分析）不在這份文件範圍內，整體 4 條 flow 的分工總覽請見 `0_AI_專案導覽地圖.md` 的「五、監控層 Node-RED 對應」。
