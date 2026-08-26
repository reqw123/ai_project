# baseline_data/

個體化基線相關的落地資料檔案，2026-08-26 從 `C:\a\`（跟這個 git 專案完全分開的路徑）搬進來，目的是可遷移性——換一台機器、重新 clone 這個 repo，這幾個檔案的路徑不需要再手動改設定。詳細背景見
[`docs/資料層架構現況與統一管理評估.md`](../docs/資料層架構現況與統一管理評估.md) 第十四節。

| 檔案 | 誰讀寫 | 對應設定 |
|---|---|---|
| `tracker_state.json` | 只有 Python 端（`cat_monitoring_system/trackers/behavior_tracker.py`） | `config.py` 的 `LoggingConfig.TRACKER_STATE_PATH` |
| `daily_history.db`（SQLite） | 只有 Python 端（`cat_monitoring_system/analytics/daily_store.py`） | `config.py` 的 `LoggingConfig.DAILY_HISTORY_DB_PATH` |
| `global.json` | **不是即時來源，僅供離線查閱的一次性快照** | 見下方說明 |

## `global.json` 為什麼是快照、不是即時資料

真正的即時寫入者是 Node-RED（路徑寫死在 `C:\Users\homec\.node-red\settings.js` 的
`functionGlobalContext.gfile` 裡，不受這個 git 專案版本控制），Node-RED 目前仍持續寫回
`C:\a\global.json`，**不是**這裡的副本。`config.py` 的 `NodeRedConfig.GLOBAL_CONTEXT_PATH`
也仍然指向 `C:\a\global.json`，Python 端的唯讀工具（`_tools/1_read_context.py`、
`_tools/2_backfill_daily_store.py`、`node_red_tests/fetch_real_history.py`）讀的都是那一份，
不是這裡的快照。

如果之後要讓 `global.json` 也真正搬進專案、跟著 repo 走，需要同時修改
`C:\Users\homec\.node-red\settings.js` 的 `gfile` 路徑常數（以及已部署的 `flows.json`，若其中
有節點直接寫死路徑），屬於更大範圍的 Node-RED 端遷移工程，目前刻意未執行。
