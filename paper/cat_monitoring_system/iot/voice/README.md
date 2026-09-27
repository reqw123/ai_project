# 飼主語音紀錄服務（Owner Voice Reports）

> **定位**：語音辨識系統與貓咪行為辨識系統之間的**飼主情境回饋通道**。
> 飼主對 ESP32 語音終端先說「**我要記錄貓咪**」，聽到「好的，請說貓咪發生了什麼事。」再說
> 「今天帶牠去看醫生」「剛剛吐了」；這裡把句子分類成**情境事件**或**飼主觀察**、存下來，
> 並寫進「健康監測 v2」P4 面板的事件標記（顯示「嘔吐 × 2」）。
>
> **壞掉也沒關係**：獨立行程，掛掉時語音辨識照常控制燈與電腦；說了「我要記錄貓咪」的話會聽到「沒有記錄成功」。
> 主系統（`main.py`）、感測器 hub（`python -m iot`）都不受影響。

---

## 為什麼要有這個（論文角度）

| 論文缺口 | 這裡做什麼 |
|---|---|
| **混淆因子**：看醫生、換飼料、訪客、緊張、服藥也會讓行為改變（Stella et al. 2011；Amat et al. 2016）；事件標記原本要自己開 P4 按 | **A. 情境事件**（整天）：用說的就記，寫進 P4 事件標記（同一套 6 類：vet/food/stranger/stress/medicine/other，標籤用 P4 自己的名稱） |
| **沒有 ground truth**：沒有真實異常事件可驗證風險分數 | **B. 飼主觀察**（時間點）：當作**弱標註**，跟系統的行為偏離預警做時間對照；每類標出對應的系統行為類別（抓癢或舔毛 → lick/scratch、甩頭 → shake…） |
| **攝影機看不到**、五類行為以外（嘔吐、食慾、如廁） | 飼主回報補上畫面外、類別外的事件 |
| 第四章要能評估 | 觸發後的每一句都存（`mode=explicit`，含取消）；`export` 匯出 CSV |

完整研究規劃見 `C:\arduino\語音辨識-v7\語音整合研究規劃.md`。

---

## 觸發模式（v7.2，使用者要求：不容易誤觸）

| 說法 | 結果 |
|---|---|
| 「我要記錄貓咪」（也接受「記錄貓咪」「幫我記錄貓咪的事」…） | ESP32 唸「好的，請說貓咪發生了什麼事。」；聆聽視窗模式自動開視窗（不用按按鈕）。**20 秒內這台講的下一句**才記錄 |
| 「我要記錄貓咪，牠剛剛吐了」（觸發句後面直接接內容） | 直接記錄後面那段 |
| 觸發後說「算了／取消／不用了」 | 「好的，已取消記錄。」 |
| 沒說觸發句的任何句子 | **不會送到這裡**（語音分頁照舊只比對燈、電腦等指令） |

觸發後的那一句是**明確記錄模式**（`classifier.classify(..., explicit=True)`）：不用再提到貓、不做問句排除；
對不上類別就記成「其他」（原句寫進 P4 備註），不會丟掉飼主想記的事。

---

## 設計原則（高內聚／低耦合）

每一層只知道下一層的「名字」，不知道它的內容：

```
語音分頁（語音辨識 v7，修補 26）── 動態 link call「owner-report」──▶ 橋接分頁（owner_reports_flow.json）
   只判斷觸發句、送 NOTE:ARM                                           │ MQTT cat/voice/*
                                                                       ▼
                                                        python -m iot.voice（這個資料夾）：分類、存檔
                                                                       │ 結果
                         橋接分頁 ── 動態 link call（依名稱）──▶「← 飼主語音紀錄」（健康監測流程裡、P4 旁邊）→「↩ 回到飼主語音紀錄」
                                                   寫進 v2_event_tags（P4 的格式只有這裡知道）→ 立即更新 P4
```

| 原則 | 具體做法 |
|---|---|
| **高內聚** | 「飼主回報」的全部知識都在這個資料夾：類別本體、分類規則、儲存、MQTT、匯出、兩個 Node-RED 檔的產生腳本、測試 |
| **零 import** | 只 import 標準函式庫、`paho`、`iot.voice.*`；主系統、感測器 hub 也都不 import 這裡。`tests/test_runner_and_coupling.py` 掃原始碼檢查 |
| **唯一對外接觸點＝MQTT** | `cat/voice/*`。Python 不知道 Node-RED 流程、ESP32、P4 的存在 |
| **fail-safe** | 服務離線時，橋接分頁看 retained 的 `status`（離線由 MQTT 遺囑代發）馬上回「沒有記錄」，不用等逾時 |
| **可整包移除** | 刪掉 `iot/voice/`、Node-RED 的「🎙 飼主語音紀錄」分頁、P4 旁邊的兩個節點、`pytest.ini` 的 `iot/voice/tests` 那行 |

---

## 檔案

| 檔案 | 內容 |
|---|---|
| `ontology.py` | 類別本體：情境事件 6 類、飼主觀察 7 類；中文標籤、觸發樣式、對應行為類別、確認句。**新增類別只改這裡** |
| `classifier.py` | 分類規則（純函式）：簡轉繁、明確記錄模式（取消）、一般模式（問句／主詞規則）、否定詞、今天／昨天／前天 |
| `store.py` | SQLite：`utterances`（每一句話，含 `mode`）、`owner_reports`（紀錄，軟刪除）；v7.1 的舊資料庫自動補欄位 |
| `service.py` | 協調層：分類 → 存檔 → 組回覆；同一個 id 重送不重複寫 |
| `transport.py`、`runner.py` | MQTT 介面（paho 1.x／2.x）；`python -m iot.voice` |
| `export.py` | 匯出 CSV：`python -m iot.voice.export reports --out r.csv --from 2026-09-01` |
| `node_red/build_flow.py` | 產生橋接分頁 `owner_reports_flow.json` |
| `node_red/build_p4_patch.py` | 把語音紀錄需要的修改寫進健康監測流程檔（`--apply-to`）：P4 儲存改合併、次數顯示、「← 飼主語音紀錄」 |
| `tests/` | pytest 86：分類規則（論文 E1 的種子測試集）、明確模式、存檔／去重／軟刪除／匯出、MQTT 路由、低耦合檢查 |

---

## MQTT 介面（`CAT_MONITORING_VOICE_TOPIC_PREFIX`，預設 `cat/voice`）

| 方向 | topic | payload |
|---|---|---|
| 收 | `cat/voice/utterance` | `{"id","text","mode"?（"explicit"）,"ts"?（毫秒或秒）,"deviceId"?,"source"?}` |
| 發 | `cat/voice/result` | `{"id","ok","matched","cancelled","mode","reason","reports":[{"report_id","kind","category","label","day","day_offset","matched","related_behaviors"}],"reply","confirm_key"}` |
| 發（retained） | `cat/voice/reports/recent` | 最近 30 筆＋今天各類筆數＋類別清單（給之後的工具用） |
| 發（retained） | `cat/voice/status` | `{"online":true/false}`（離線由 MQTT 遺囑代發） |
| 收 | `cat/voice/reports/delete` | `{"report_id", "restore"?}`（軟刪除／復原） |
| 收 | `cat/voice/reports/get` | 任何內容 → 重發最近紀錄 |

---

## 安裝與執行

1. **紀錄服務**（保持開著）：最方便是設定視窗「📡 IoT 子系統」分頁的「🎙 飼主語音紀錄 ▶ 啟動」（背景執行；**關掉設定視窗時會一起強制關閉**）；或手動：
   ```bash
   cd paper/cat_monitoring_system
   C:\Users\homec\anaconda3\envs\yolo_new\python.exe -m iot.voice
   ```
2. **Node-RED（1880）**：
   - 匯入 `node_red/owner_reports_flow.json`（橋接分頁），選「**取代**」，然後部署。
   - 橋接分頁的節點 ID 跟 1880 相同（09-27 起）：
     - `python -m iot.voice.node_red.build_flow` 產生檔案時，會照 `node_red/owner_reports_ids_1880.json` 換成 1880 上的 ID；
     - 1880 的 ID 又變了（例如用了「匯入複本」）→ 跑 `python -m iot.voice.node_red.build_flow --sync-ids`：
       從 1880 GET /flows 讀（Node-RED 關著就讀 `~/.node-red/flows.json`），依「類型＋名稱」一對一配對，重寫對照表再產生；
       有同名的節點配不起來就直接停，不會亂猜。
   - 橋接交給 P4 是**依名稱**呼叫（動態 link call「← 飼主語音紀錄」），不依賴健康監測流程的 ID。
   - 健康監測（個體化基線）流程是 `C:\ai_project\node-red-flows\cat_health_v4_flow.json`：節點 ID 跟 1880 相同，改它 → 匯入選「**取代**」。
     它已經包含 P4 的修改和「← 飼主語音紀錄」。要重新寫進去：`python -m iot.voice.node_red.build_p4_patch --apply-to <流程檔>`
     （依名稱找節點，已有就沿用 ID，重跑不重複）。
3. 語音辨識 v7 的流程（修補 26）與韌體 7.2.0：見 `C:\arduino\語音辨識-v7\v7說明.md`。

### 設定（環境變數 `CAT_MONITORING_VOICE_*`）

| 環境變數 | 預設 | 說明 |
|---|---|---|
| `_MQTT_HOST` / `_MQTT_PORT` | `192.168.0.171` / `1883` | broker（跟 Node-RED 同一台） |
| `_TOPIC_PREFIX` | `cat/voice` | topic 前綴 |
| `_DB_PATH` | `iot/voice/data/owner_reports.db` | SQLite 路徑（`data/` 不進版控） |
| `_CAT_ALIASES` | `貓,猫,咪` | 一般模式的主詞規則（明確模式不用） |
| `_LOG_UNMATCHED` | `1` | 一般模式沒分類的句子也存（明確模式一律存） |
| `_RECENT_LIMIT` | `30` | 最近紀錄幾筆 |

---

## 測試

```bash
pytest paper/cat_monitoring_system/iot/voice/tests
```

不需要 broker、paho、cv2、torch。端對端（ESP32 模擬 → Whisper → 語音分頁 → 橋接 → MQTT → 本服務 → P4 畫面）在
`C:\arduino\語音辨識-v7\tests\owner_report_e2e_test.js`（19 項）。

---

## 已知限制

- 分類是規則式（關鍵字樣式＋否定詞），可解釋、可重現，但口語變化大時會記成「其他」；觸發後的每一句都存在 `utterances`，
  可以回頭補樣式（加進 `tests/` 當回歸測試）。
- 記錯了目前沒有語音或 P4 上的刪除（v7.1 的獨立儀表板已拿掉）；說完觸發句後可以說「取消」。資料庫支援軟刪除（`cat/voice/reports/delete`），
  之後可以加「刪除剛剛那筆紀錄」。
- P4 的「記錄今日事件」改成合併之後，就沒辦法用它拿掉同一天已有的標籤（原本是用覆蓋的方式改）。
- 只接一般模式（ESP32 語音）。自由對話模式、網頁麥克風尚未接上。

## 09-27 晚：依名稱呼叫、P4 語音清單
- 橋接分頁交給 P4 改用**動態 link call** 依名稱呼叫「← 飼主語音紀錄」；健康監測那邊由「↩ 回到飼主語音紀錄」（link out 回傳）回應。
  - 健康監測流程重新匯入、節點 ID 被 Node-RED 換掉也接得上。
  - 原本依 ID 的 link out，在使用者重新匯入後就斷了。
- P4 近期事件：語音紀錄不再寫進備註（備註最多 300 字，同一天記很多筆時前面的會被截掉），
  改成那一天下面一個可展開的「🎙 語音紀錄 N 筆」（時間、類別、原句；每天最多留 200 筆）。
- v7.3（09-27）：每一天存 `voice`（語音紀錄）＋`manual`（按一次「記錄今日事件」＝一筆）兩份清單，
  `counts` 每次寫入都依這兩份**重新數**（不再 +1 累加），所以「× N」一定跟紀錄對得上；舊資料寫入時或部署時自動補齊。
  手動記錄存完由「使用者設定管理器」輸出 3 直接更新 P4。見 `C:\arduino\語音辨識-v7\v7說明.md` 的 v7.3 一節。
