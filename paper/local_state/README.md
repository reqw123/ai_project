# local_state/ — 每台機器各自的本機狀態（不進版控）

這個資料夾放「跟這台機器 / 這個使用者綁定、換一台電腦就該重來」的狀態檔，
**整個資料夾在 `.gitignore` 裡**（只有這份 `README.md` 例外，進版控是為了讓
這個慣例被看得到）。

跟 `paper/runtime_settings.current.json` 之類的**系統設定**分開——那些是
`settings_manager.FIELD_SCHEMA` 管的、會影響 `main.py` 執行結果、要進版控、
有測試與文件綁定的東西；這裡的檔案純粹是介面便利性，壞掉 / 不存在一律
當「沒有紀錄」處理，絕不影響程式啟動。

## 目前的檔案

| 檔案 | 寫入者 | 內容 |
|---|---|---|
| `gui_state.json` | `settings_gui/ui_state.py` | 設定視窗的 GUI 偏好。目前兩個 key：<br>• `recent_video_paths`：「🎬 影片路徑（選填）」下拉記住的最近 10 筆影片檔／資料夾路徑（每筆 `{"path", "locked"}`）。<br>• `tool_script_order`：「🧩 獨立腳本工具」下拉的手動排列順序（每筆 `{"key", "locked"}`，`key` = 相對 `tools/` 的路徑）。<br>兩者的 `locked` 項在各自的「⚙ 管理」視窗中位置完全固定：不能上移下移、擋刪除、不被新項目擠掉。 |

## 新增這類檔案的規約

- 一律走 fail-safe：讀不到 / 解析失敗回傳空值，不 raise。
- 寫入走「temp file + `os.replace`」原子寫，並在寫入前 `mkdir(parents=True, exist_ok=True)`。
- 檔名用小寫，不要用 `.` 開頭的隱藏檔（放在這個資料夾裡就夠明確了）。
