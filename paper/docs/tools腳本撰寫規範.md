# 📐 tools/ 獨立腳本撰寫規範

> [!IMPORTANT]
> **本文件目的**
>
> 規範「往 `paper/tools/`（含 `train_data/` 子資料夾）新增獨立腳本」時必須遵守的寫法，
> 讓腳本從**設定視窗的「獨立腳本工具」面板**啟動時，終端訊息能**即時**顯示在模擬終端、
> `input()` 能在面板輸入、按「停止」能乾淨結束。
>
> **適用對象：** 所有會出現在設定視窗下拉選單的腳本（`tools/` 底下、檔名不以 `_` 開頭的 `.py`）。
>
> **跟其他文件的分工：** 各腳本「做什麼、怎麼跑」見 [`獨立運行腳本索引.md`](獨立運行腳本索引.md)；
> 本文件只管「新腳本要怎麼寫，才能跟設定視窗的終端面板正常配合」。

---

## 🚀 30 秒快速了解

- 設定視窗啟動腳本時，**stdout／stderr 會接到 pipe**（不是真終端機），stdin 也是 pipe。
- 設定視窗端**已經處理好**：不緩衝輸出、UTF-8 編碼、stdin 導向輸入框、面板收合時也能輸入。
- 腳本端**只要不破壞這些**：用 `print()`、別換掉 `sys.stdout`、別依賴 `isatty()`、別用 `tqdm`／`\r`、
  迴圈要有結束條件、每個互動動作印一行說明。

---

## 🧭 快速導航

1. [設定視窗端已經替你做好的事](#sec-1)
2. [新腳本必須遵守的規則](#sec-2)
3. [新腳本範本（可直接複製）](#sec-3)
4. [上線前檢查清單](#sec-4)
5. [驗證方法](#sec-5)
6. [實測過的坑（為什麼有這些規則）](#sec-6)

---

<a id="sec-1"></a>

## 1️⃣ 設定視窗端已經替你做好的事

實作見 `settings_gui/process_manager.py` 的 `start_tool()` 與 `settings_gui/console_panel.py`。

| 項目 | 設定視窗做的事 | 你不用再做 |
|---|---|---|
| 輸出不延遲 | 子行程環境變數 `PYTHONUNBUFFERED=1` | 不必到處加 `flush=True`／`python -u` |
| 中文／emoji 不亂碼 | `PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1` | 不必自己處理編碼 |
| 工作目錄 | `cwd` 設為**腳本自己所在的資料夾** | — |
| 輸入 | stdin 接 pipe；面板底部「輸入」列送出的文字會寫進 `input()`，並在面板回顯 `> 文字` | 不必自己做輸入介面 |
| 面板收合時 | 輸入列仍可用，標題列會顯示「最後一行輸出」預覽（看得到腳本在問什麼） | — |
| 輸出太多 | 每 80 ms 合併一次顯示、最多顯示最後 2000 行；面板只保留最新約 4000～5000 行 | — |
| 停止 | 先送 `CTRL_BREAK_EVENT`，**4 秒**內沒結束就強制終止，再等 2 秒確認 | — |

> [!NOTE]
> 一次只能執行一個腳本（或 main.py）。改了 `process_manager.py`／`console_panel.py` 後，
> **設定視窗必須重開**才會載入新版（模組在啟動時就載入記憶體了）。

---

<a id="sec-2"></a>

## 2️⃣ 新腳本必須遵守的規則

### ✅ 輸出

1. **用 `print()`，或預設的 `logging`（走 stderr，會被合併進同一個面板）。**
2. **不要換掉 `sys.stdout`／`sys.stderr`**（`sys.stdout = ...`、`contextlib.redirect_stdout`、
   `io.TextIOWrapper(sys.stdout.buffer, ...)`）。真的要包，必須帶 `line_buffering=True`，
   否則緩衝會回來，面板又變成「不會顯示」。
3. **不要把 `logging` 只導向檔案**（`FileHandler`／`basicConfig(filename=...)` 且沒有 `StreamHandler`），
   否則面板什麼都看不到；要寫檔就同時保留一個 console handler。
4. **不要用 `tqdm` 或 `\r` 原地刷新。** 面板不處理 `\r`，每次刷新會擠成同一行越拉越長。
   進度改成「每 N% 或每幾秒印一行」，例如 `print(f"進度: {done}/{total} ({pct:.1f}%)")`。
5. **每個互動動作印一行說明**：切換影片、暫停／繼續、切換模式、按鍵操作。
   OpenCV 視窗裡按了鍵，終端要有一行回應，使用者才知道有沒有生效。
6. **切換影片／處理新檔案時，最後一行印出目前檔名**（範例：`▶ 目前影片：cat5.mp4`）。

### ✅ 輸入

7. **不要依賴 `isatty()`。** 從設定視窗啟動時 stdin 是 pipe，`isatty()` 為 `False`，但**是可以讀的**。
   用 `input()`，並處理 `EOFError` 給預設值（參考 `1_run_video_inference.py` 的 `resolve_run_mode()`）。
8. 提示字直接放在 `input("提示: ")`，面板會即時顯示（沒有換行的提示字也看得到）。

### ✅ 行程與視窗

9. **開子行程時**：不要用 `CREATE_NEW_CONSOLE`／`DETACHED_PROCESS`／`pythonw`，
   不要用 `capture_output` 把子行程輸出吞掉（沿用預設繼承 stdout，子行程的訊息才會出現在面板）。
10. **迴圈一定要有結束條件。** 尤其是「清單輪播」類迴圈：全部項目都無效（路徑是資料夾、不存在、打不開）
    時必須自己停下並印出原因，不能無限空轉狂噴訊息。
11. **收尾要快。** 「停止」= `CTRL_BREAK_EVENT`，Python 不會把它轉成 `KeyboardInterrupt`，多半直接被結束。
    需要保存的結果（CSV、圖表）要在處理過程中就落盤，不要全部留到最後。

### ✅ 路徑與設定

12. **路徑用 `Path(__file__)` 推算**，不要依賴目前工作目錄。
13. **要 import 核心模組時，兩個路徑都要加進 `sys.path`**（見下方範本）。
14. **需要跟主系統一致的參數不要寫死**：YOLO 輸入尺寸讀 `YOLOConfig.IMAGE_SIZE`
    （`from config import YOLOConfig`），推論精度用 FP16（CUDA 上 `quantize=16`，見 `KeypointDetector`）。

### ✅ 預覽視窗大小（有開影片視窗的腳本）

15. **視窗大小一律用同一套寫法、同樣的變數名**，放在腳本開頭的設定區，要換大小只改 `DISPLAY_RESOLUTION`：

    ```python
    # 預覽視窗解析度：改這個變數即可（"720p" → 1280x720、"1080p" → 1920x1080）。
    # 只決定視窗大小，不影響偵測／推論吃的原始畫面。
    DISPLAY_RESOLUTION = "720p"
    _DISPLAY_RESOLUTION_PRESETS = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
    }
    import os as _os
    _env_resolution = _os.getenv("DISPLAY_RESOLUTION", "").strip()  # 設定視窗「⚙ 額外設定」可覆寫上面的預設；空白＝沿用預設
    if _env_resolution:
        if _env_resolution in _DISPLAY_RESOLUTION_PRESETS:
            DISPLAY_RESOLUTION = _env_resolution
        else:
            print(f"⚠ 環境變數 DISPLAY_RESOLUTION={_env_resolution!r} 無效（只接受 {list(_DISPLAY_RESOLUTION_PRESETS)}），沿用預設 {DISPLAY_RESOLUTION}")
    DISPLAY_SIZE = _DISPLAY_RESOLUTION_PRESETS[DISPLAY_RESOLUTION]  # 視窗顯示解析度（寬, 高）
    DISPLAY_FULLSCREEN = (DISPLAY_RESOLUTION == "1080p")  # 1080p ＝ 全螢幕；720p 維持一般視窗
    ```

    建立視窗時，`DISPLAY_FULLSCREEN` 為 True 就補一段（放在 `resizeWindow` 之後）：

    ```python
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)      # 全螢幕需要 WINDOW_NORMAL，不能用 AUTOSIZE
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])
    if DISPLAY_FULLSCREEN:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    ```

    - 全螢幕沒有標題列：原本寫在視窗標題裡的操作說明要改印在終端；全螢幕時 `cv2.resizeWindow()` 會讓視窗跳出全螢幕，
      有縮放視窗功能的（例如 +/- 鍵）要在全螢幕時略過。
    - 左右並排的多視窗腳本無法用單一全螢幕：改成拿掉標題列與邊框、緊貼鋪滿螢幕（範例：
      `2_run_dual_model_compare.py` 的 `_tile_borderless_windows()`）。

    - **檔內的 `DISPLAY_RESOLUTION` 是預設值；環境變數 `DISPLAY_RESOLUTION` 優先。** 設定視窗的
      「⚙ 額外設定」按鈕選了 720p／1080p，執行腳本時就會用環境變數傳進來；選「使用腳本內設定」則不傳。
      放在 `DISPLAY_SIZE` 計算之前，並印警告處理無效值，照上面原樣複製即可。

    - 程式其餘部分只讀 `DISPLAY_SIZE`，不要再各自寫死 `1280`、`720`、`(1080, 720)` 之類的數字，
      也不要另取 `PREVIEW_DISPLAY_SIZE`、`DISPLAY_W/H`、`MAX_W/MAX_H` 這類別名。
    - **只縮放「給人看的那份畫面」**：偵測、姿態、分類一律吃原始解析度；骨架與文字等疊圖在縮放之後才畫，
      線條才會銳利（範例：`1_run_video_inference.py` 的 `resize_with_letterbox()`）。
    - **左右並排兩個畫面的腳本**（例如 `2_run_dual_model_compare.py`）：解析度指「兩個畫面合計」，
      每個畫面寬度減半：`DISPLAY_SIZE = (_pane_w // 2, _pane_h)`，這樣 1080p 剛好放得進 1080p 螢幕。
    - 影像標註類工具（視窗大小要跟原始畫面像素一一對應，例如 `train_data/0_dataset_collect.py`）不適用。

### ✅ 檔名與登錄

16. **檔名不要以 `_` 開頭**，否則不會出現在下拉選單（`_` 開頭視為內部模組，例如 `_rename_gui.py`）。
17. **在 [`獨立運行腳本索引.md`](獨立運行腳本索引.md) 的第 5 節（`tools/`）或第 6 節（`train_data/`）補一列說明**，
    否則腳本能執行，但說明卡片會顯示「找不到功能說明」。**表格格式與標題有程式解析**，改之前務必讀該文件開頭的
    IMPORTANT 區塊：列格式 `` | `檔名.py` | 功能說明 | 使用方法 | ``，功能欄不能含 `|`，
    章節標題（`## 5. paper/tools/`、`## 6. paper/tools/train_data/`）不能改。

---

<a id="sec-3"></a>

## 3️⃣ 新腳本範本（可直接複製）

```python
"""一句話說明這支腳本做什麼（同一句話也寫進 docs/獨立運行腳本索引.md）"""
import sys
from pathlib import Path

# 核心模組（models／detectors／utils…）與 config.py 各在不同資料夾，兩個都要加
sys.path.insert(0, str(Path(__file__).parent.parent / "cat_monitoring_system"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # config.py 在 paper/ 根目錄
# 放在 train_data/ 子資料夾時，改成 .parent.parent.parent

from config import YOLOConfig  # noqa: E402

YOLO_IMGSZ = YOLOConfig.IMAGE_SIZE  # 跟主系統同步，不要寫死 640


def resolve_run_mode():
    """不依賴 isatty()；stdin 讀不到（EOF）時給預設值，不會卡住。"""
    print("\n請選擇執行模式:\n  1) 只生成統計（不開視窗）\n  2) 開視窗測試")
    while True:
        try:
            choice = input("輸入模式 (1/2, 預設=2): ").strip()
        except EOFError:
            print("\n未輸入模式，預設使用模式 2")
            return 2
        if choice in ("", "2"):
            return 2
        if choice == "1":
            return 1
        print(f"⚠ 輸入無效「{choice}」，請輸入 1 或 2")


def main():
    mode = resolve_run_mode()
    items = [...]  # 待處理清單
    valid_streak_fail = 0
    for i, item in enumerate(items, 1):
        if not is_usable(item):                       # 路徑無效：印原因、計數，全部無效就結束
            print(f"❌ 略過: {item}")
            valid_streak_fail += 1
            if valid_streak_fail >= len(items):
                print("❌ 全部項目都無法處理，結束")
                return
            continue
        valid_streak_fail = 0
        print(f"▶ 目前項目 [{i}/{len(items)}]：{Path(item).name}")   # 最後一行印檔名
        ...  # 處理；進度每 N% 印一行，不要用 tqdm


if __name__ == "__main__":
    main()
```

---

<a id="sec-4"></a>

## 4️⃣ 上線前檢查清單

- [ ] 檔名不以 `_` 開頭，放在 `tools/`（或 `train_data/`）
- [ ] 只用 `print()`／預設 `logging`，沒有換掉 `sys.stdout`／`sys.stderr`
- [ ] 沒有 `tqdm`、沒有 `\r` 原地刷新
- [ ] 沒有 `isatty()` 判斷；`input()` 有處理 `EOFError`
- [ ] 每個按鍵／模式切換有印一行；切換檔案時最後一行印檔名
- [ ] 所有迴圈都有結束條件（含「全部無效」）
- [ ] 子行程沒用 `CREATE_NEW_CONSOLE`／`DETACHED_PROCESS`／`capture_output`
- [ ] 需要保存的結果在處理過程中就落盤（停止時不會遺失）
- [ ] 路徑以 `Path(__file__)` 推算；`sys.path` 加了 `cat_monitoring_system` 與 `paper/`
- [ ] YOLO 尺寸讀 `YOLOConfig.IMAGE_SIZE`，推論精度 FP16
- [ ] 有預覽視窗的腳本，視窗大小用統一的 `DISPLAY_RESOLUTION`／`_DISPLAY_RESOLUTION_PRESETS`／`DISPLAY_SIZE`
- [ ] 已在 `獨立運行腳本索引.md` 第 5／6 節補一列說明（格式照該文件規定）
- [ ] 從設定視窗實際啟動一次，確認面板 3～5 秒內出現第一行輸出

---

<a id="sec-5"></a>

## 5️⃣ 驗證方法

**最直接**：開設定視窗 → 選腳本 → 「▶ 執行所選腳本」，看面板是否在幾秒內出現輸出、
`input()` 提示是否出現、在輸入列打字後腳本是否收到。面板收合時也應該能輸入。

**不開視窗的快速檢查**（模擬設定視窗的啟動方式，量「多久收到第一批輸出」）：

```python
import subprocess, sys, os, time
env = os.environ.copy()
env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")  # 跟 start_tool() 一致
p = subprocess.Popen([sys.executable, r"C:\ai_project\paper\tools\你的腳本.py"],
                     cwd=r"C:\ai_project\paper\tools", env=env, stdin=subprocess.PIPE,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
t0 = time.time(); print(p.stdout.readline(), f"← {time.time()-t0:.1f}s 後收到第一行")
p.kill()
```

> [!WARNING]
> 這個檢查一定要帶 `PYTHONUNBUFFERED=1`。**不帶的話**，只要腳本開頭輸出不到 ~8 KB，
> 你會看到「什麼都沒收到」——這正是沒有這個環境變數時面板的實際狀況。

---

<a id="sec-6"></a>

## 6️⃣ 實測過的坑（為什麼有這些規則）

| 現象 | 原因 | 對應規則 |
|---|---|---|
| 有 GUI 視窗的腳本，面板一個字都沒有 | pipe 預設區塊緩衝（約 8 KB），腳本開頭只印幾百字元，一直卡在緩衝區 | 設定視窗已加 `PYTHONUNBUFFERED=1`；腳本別換掉 `sys.stdout` |
| 按「停止」後面板還輸出了很久 | 腳本無限迴圈狂噴訊息（清單全是資料夾／無效路徑，`"next"` 繞圈），舊訊息積壓在佇列 | 規則 10；面板現在合併顯示、最多 2000 行，停止後約 3 秒內跟上 |
| 進度條擠成一長行 | `tqdm` 用 `\r` 原地刷新，面板不處理 `\r` | 規則 4 |
| 面板收合就不能輸入 | 輸入列原本在會被隱藏的區塊裡 | 已改為收合也保留輸入列＋標題列顯示最後一行輸出 |
| 腳本被誤判「非互動」直接跳過提問 | 用 `isatty()` 判斷，但 pipe 的 stdin 是可讀的 | 規則 7 |

已實測、輸出會即時同步的有 GUI 視窗腳本：`1_run_video_inference`、`1_measure_ear_distance_single_video`、
`2_run_dual_model_compare`、`test_bbox_area_ratio`、`test_bone_length_stability` 等
（各於啟動後 3～5 秒內出現首批輸出，含 `input()` 提示字）。

---

## 📎 延伸閱讀

- [`獨立運行腳本索引.md`](獨立運行腳本索引.md) — 各腳本用途與操作方式（第 5／6 節有程式解析，改格式前先讀開頭說明）
- `settings_gui/process_manager.py` — `start_tool()`／`request_shutdown_and_wait()`
- `settings_gui/console_panel.py` — 面板輸出、輸入、收合行為
