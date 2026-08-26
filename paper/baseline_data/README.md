# baseline_data/

個體化基線相關的落地資料檔案，2026-08-26 從 `C:\a\`（跟這個 git 專案完全分開的路徑）搬進來，目的是可遷移性——換一台機器、重新 clone 這個 repo，這幾個檔案的路徑不需要再手動改設定。詳細背景見
[`docs/資料層架構現況與統一管理評估.md`](../docs/資料層架構現況與統一管理評估.md) 第十四、十五節。

| 檔案 | 誰讀寫 | 對應設定 |
|---|---|---|
| `tracker_state.json` | 只有 Python 端（`cat_monitoring_system/trackers/behavior_tracker.py`） | `config.py` 的 `LoggingConfig.TRACKER_STATE_PATH` |
| `daily_history.db`（SQLite） | 只有 Python 端（`cat_monitoring_system/analytics/daily_store.py`） | `config.py` 的 `LoggingConfig.DAILY_HISTORY_DB_PATH` |
| `global.json` | Node-RED 即時讀寫（`gfile`），Python 端唯讀工具也讀這裡——**但要等環境變數生效** | 見下方說明 |

## `global.json`：2026-08-27 起改用環境變數驅動，需要重開機/登出重登才會真正切換

真正的即時寫入者是 Node-RED（`C:\Users\homec\.node-red\settings.js` 的
`functionGlobalContext.gfile`）。2026-08-27 起，這個路徑改成讀環境變數
`CAT_MONITORING_NODERED_GLOBAL_CONTEXT_PATH`（Python 端 `NodeRedConfig.GLOBAL_CONTEXT_PATH`
讀的是**同一個**環境變數名稱，兩邊自動保持一致，不用各自維護一份路徑）：

```
CAT_MONITORING_NODERED_GLOBAL_CONTEXT_PATH=C:\ai_project\paper\baseline_data\global.json
```

這個環境變數已經用 `[Environment]::SetEnvironmentVariable(...,'User')` 設定好，但 Windows
使用者環境變數要「新登入的 session」才讀得到——**在你登出重登或重開機之前，Node-RED
（跟任何目前開著的終端機）都還會用 settings.js／config.py 各自的 fallback（都是舊路徑
`C:\a\global.json`），此時這裡的 `global.json` 不會被更新，是搬遷當下複製的一份快照**。

- 登出重登／重開機後啟動 Node-RED：`gfile` 會改讀寫這裡的 `global.json`，才算真正切換完成。
- 在那之前：兩邊 fallback 刻意都設成同一個舊路徑 `C:\a\global.json`（不是新路徑），所以
  不會發生「一邊已經切新路徑、另一邊還在舊路徑」的分岔，只是還沒真的搬而已。
- 想確認有沒有切換成功：`_tools/1_read_context.py` 印出來的內容如果有比 `C:\a\global.json`
  更新的資料，就代表 Node-RED 已經改讀這裡了。
