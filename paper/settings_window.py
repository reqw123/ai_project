"""貓咪監測系統 — 獨立設定管理視窗（tkinter GUI）。

管的是 runtime_settings.current.json（執行期 JSON 覆寫層），不是 config.py 原始碼，也
不是 stgcn_config.yaml——本視窗完全不會寫入這兩者。欄位、驗證規則、環境變數
名稱、對應的 config.py class attribute，全部來自 ``settings_manager.FIELD_SCHEMA``
（單一事實來源）；新增一個可調整欄位只需要在該檔案的 FIELD_SCHEMA 加一筆，並在
config.py 對應屬性外包一層 ``_runtime_default()``——不需要碰這支 GUI 的程式碼，
分頁與欄位列會自動長出來。

視覺風格延續 ``cat_monitoring_system/analytics/manage_baseline_history.py``
（同一組顏色常數、Microsoft JhengHei 字型、頂部深色標題列、Canvas+Scrollbar 捲動）。

執行方式：

    python settings_window.py

存檔後不會立即套用到正在跑的 main.py（第一版刻意不做熱重載，避免推論中的模型／
執行緒／tracker 狀態不一致）——存檔訊息會明確提示「重新啟動主程式後生效」。
"""

import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import unicodedata
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

_SCRIPT_DIR = Path(__file__).resolve().parent  # paper/，config.py 與 settings_manager.py 所在處
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# main.py 用 "./logs"、"./output" 這類相對路徑當預設值（見 config.py ModelPaths），
# 專案裡實際的 paper/logs、paper/output 目錄也是這樣建出來的——代表 main.py 一直
# 以來都是以 paper/ 當工作目錄執行，這裡啟動子行程時比照辦理，維持行為一致。
_MAIN_PY_PATH = _SCRIPT_DIR / "cat_monitoring_system" / "main.py"

import config  # noqa: E402
import settings_manager  # noqa: E402
from settings_manager import FIELD_SCHEMA, TAB_ORDER, _MISSING, _get_nested, _redact, _set_nested  # noqa: E402

# ── 視覺樣式：沿用 analytics/manage_baseline_history.py 的同一組常數 ─────────

_FONT_FAMILY = "Microsoft JhengHei"

COLOR_BG_MAIN = "#eef2f6"
COLOR_HEADER_BG = "#2c3e50"
COLOR_HEADER_FG = "#ffffff"
COLOR_HEADER_SUB_FG = "#b7c4cf"
COLOR_INFO_BG = "#e8f1fb"
COLOR_INFO_FG = "#2c4053"
COLOR_INFO_BORDER = "#c7dbf0"
COLOR_TAB_BG = "#ffffff"
COLOR_LABEL_FG = "#20303f"
COLOR_HINT_FG = "#6b7c8c"
COLOR_ERROR_FG = "#c0392b"
COLOR_WARNING_FG = "#b36b00"
COLOR_SUCCESS_FG = "#1a7a1a"

BTN_PRIMARY_BG = "#27ae60"
BTN_PRIMARY_ACTIVE = "#219150"
BTN_SECONDARY_BG = "#5d7285"
BTN_SECONDARY_ACTIVE = "#4a5d6e"
BTN_WARN_BG = "#e67e22"
BTN_WARN_ACTIVE = "#cf711d"
BTN_NEUTRAL_BG = "#95a5a6"
BTN_NEUTRAL_ACTIVE = "#7f8c8d"

BADGE_ENV_BG = "#f5b7b1"
BADGE_ENV_FG = "#78281f"
BADGE_JSON_BG = "#a9cce3"
BADGE_JSON_FG = "#1b4f72"
BADGE_DEFAULT_BG = "#d5dbdb"
BADGE_DEFAULT_FG = "#4d5656"
BADGE_FORM_BG = "#f9e79f"
BADGE_FORM_FG = "#7d6608"

# 底部「main.py 終端機輸出」面板：刻意用深色終端機配色跟上方淡色表單拉開視覺區隔，
# 一眼就能認出這塊是「輸出訊息」而不是可編輯欄位。
COLOR_CONSOLE_BG = "#1e1e1e"
COLOR_CONSOLE_FG = "#d4d4d4"
COLOR_CONSOLE_MUTED_FG = "#7f8c8d"
COLOR_CONSOLE_GRIP_BG = "#3a3f44"  # 可拖拉調整高度的把手，比面板底色略亮，暗示「這裡能拖」
COLOR_TOOL_LISTBOX_BG = "#eaf4fc"  # 獨立腳本工具下拉選單展開後的淡藍色底，掃視一長串腳本名稱更輕鬆
COLOR_TOOL_DESC_BG = "#eaf4fc"  # 常駐說明卡片底色，跟下拉選單同一色系，視覺上關聯在一起
COLOR_TOOL_DESC_BORDER = "#aed6f1"
COLOR_TOOL_DESC_ACCENT = "#1b4f72"  # 卡片左側色條 + 呼應下拉選單選取列的深藍
COLOR_TOOL_DESC_FG = "#1b2631"  # 深色文字，淡藍底上要維持可讀性

CONSOLE_DEFAULT_HEIGHT = 260  # 預設展開高度
CONSOLE_MIN_HEIGHT = 50  # 拖拉能縮到的最小高度，比收合狀態(34px)略高，仍看得到一點點輸出內容
CONSOLE_COLLAPSED_HEIGHT = 34  # 內縮後只剩標題列那一條水平線的高度
CONSOLE_MAX_HEIGHT_RESERVE = 200  # 標題列高度還量不到時的退回值（拖到最高＝扣掉標題列高度）
CONSOLE_DEFAULT_HEIGHT_FRACTION = 0.75  # 終端機面板「初始」高度＝視窗高度的這個比例

CONSOLE_FONT_FAMILY = "Consolas"
CONSOLE_DEFAULT_FONT_SIZE = 20  # 原本 10 的 2 倍；Ctrl+0／Ctrl+numpad0 重設回這個大小
CONSOLE_MIN_FONT_SIZE = 6
CONSOLE_MAX_FONT_SIZE = 28

# 布林旗標（開關型設定）的視覺樣式——刻意跟數字/字串欄位的樸素外觀拉開差異，
# 讓「這是一個開關」在掃視整頁時能立刻被認出來，不用逐行讀 label 文字。
FLAG_ON_BG = "#27ae60"
FLAG_ON_FG = "#ffffff"
FLAG_OFF_BG = "#aeb6bf"
FLAG_OFF_FG = "#2c3e50"
FLAG_ROW_BG = "#f4faf6"  # 布林欄位那一整列的底色，跟其他欄位列的白底做出區隔

# 每個分頁一個代表色：分頁切換鈕本身直接用這個顏色上色（不用 ttk.Notebook——
# 它在 Windows 原生佈景主題下無法讓每個分頁按鈕有不同底色，所以分頁列改成自己
# 刻的一排 tk.Button，才能做到「每個分頁按鈕都套用自己的配色」），分頁內容區
# 頂部也會有一條同色的色條 + 色塊 emoji，兩處呼應。
TAB_COLORS = {
    "模型與輸入來源":          ("🟦", "#2874a6"),
    "YOLO 推論":               ("🟩", "#1e8449"),
    "ST-GCN 推論":             ("🟨", "#b7950b"),
    "異常偵測與骨架品質":       ("🟧", "#ca6f1e"),
    "執行模式與排程":           ("🟥", "#c0392b"),
    "Flask 與 Node-RED":       ("🟪", "#7d3c98"),
    "行為追蹤與警報門檻":       ("🟫", "#8b5a2b"),
    "貓咪身份驗證":             ("⬛", "#34495e"),
    "日誌、CSV、資料庫與輸出路徑": ("⬜", "#616a6b"),
    "視覺化與串流顯示":         ("🔷", "#148f77"),
    "進階設定":                 ("🔶", "#af601a"),
}


def _styled_button(parent, text, command, bg, active_bg, fg="#ffffff", font=None):
    return tk.Button(
        parent, text=text, command=command, bg=bg, fg=fg,
        activebackground=active_bg, activeforeground=fg, relief="flat",
        font=font or (_FONT_FAMILY, 12, "bold"), padx=14, pady=6, cursor="hand2",
    )


def _lighten(hex_color: str, factor: float) -> str:
    """把顏色往白色混合 factor 比例（0~1，越大越淡）；分頁未選取時用淡版底色，
    選取時用原色，兩種狀態都看得出屬於哪個分頁、又能分辨目前選到哪一個。"""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _display_width(text: str) -> int:
    """算字串的「顯示寬度」（全形/中文字元算 2 個半形字寬，其餘算 1）——用在等寬
    字型的清單裡要用空白對齊多欄內容時，len() 對中文字元會低估實際佔用的寬度。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


# 依 value_type 對應 config.py 現有的 _env_* 解析函式，讓「顯示某欄位目前受環境
# 變數控制時的解析結果」跟 config.py 實際計算 class attribute 時用的邏輯一致，
# 不在這裡另外重新實作一份可能跟主邏輯不同步的 parser。
def _parse_env_value(env_var: str, value_type: str):
    if value_type == "bool":
        return config._env_bool(env_var, False)
    if value_type == "int":
        return config._env_int(env_var, 0)
    if value_type == "float":
        return config._env_float(env_var, 0.0)
    if value_type == "video_input":
        return config._env_video_input(env_var, "")
    if value_type == "size":
        return config._env_size(env_var, None)
    return config._env_str(env_var, "")


class _ScrollableTab(tk.Frame):
    """單一分頁的可捲動容器（欄位多，單頁會爆版），滑鼠滾輪僅在游標停在該分頁時作用。"""

    def __init__(self, parent):
        super().__init__(parent, bg=COLOR_TAB_BG)
        # width/height=1：Canvas 沒指定尺寸時 Tk 預設請求約 200x150，這個「地板」會讓
        # 底下終端機面板拖到最高時卡在提早的高度（content_area 撐住不縮），設成 1 只是
        # 蓋掉這個預設請求值，實際渲染時仍會被 pack(fill="both", expand=True) 撐滿可用空間。
        canvas = tk.Canvas(self, bg=COLOR_TAB_BG, highlightthickness=0, width=1, height=1)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.body = tk.Frame(canvas, bg=COLOR_TAB_BG)
        self.body.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        _body_window = canvas.create_window((0, 0), window=self.body, anchor="nw")
        # 把 body 的寬度綁定到 Canvas 目前的可見寬度：沒有這行，body 的寬度完全由
        # 「目前顯示內容裡最寬的那一行」決定——像影像來源欄位切換「本機檔案」／
        # 「攝影機索引」／「URL」三種模式時，各模式底下的控制項自然寬度不一樣，
        # 會讓整個分頁內容的寬度跟著切換模式忽寬忽窄，使用者看到的就是輸入框、
        # 按鈕的位置跟著跳動。改成每次 Canvas 尺寸變動（含初次繪製、視窗縮放）
        # 都同步把 body 這個 window item 的寬度釘死成 Canvas 當下的寬度，內容寬度
        # 就不會再反過來牽動外層版面。
        canvas.bind(
            "<Configure>", lambda e: canvas.itemconfig(_body_window, width=e.width)
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_enter(_e):
            canvas.bind_all(
                "<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            )

        def _on_leave(_e):
            canvas.unbind_all("<MouseWheel>")

        self.bind("<Enter>", _on_enter)
        self.bind("<Leave>", _on_leave)


class SettingsWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("貓咪監測系統 — 設定管理")
        self.configure(bg=COLOR_BG_MAIN)
        self.minsize(1000, 900)  # 下方新增終端機輸出面板固定高度，原本 640 會把分頁表單擠壓到快捲不動
        self._size_window_near_fullscreen()

        # 所有字體統一放大 1.2 倍（基準字級 × _FONT_SCALE，四捨五入）。
        self._font_title = tkfont.Font(family=_FONT_FAMILY, size=22, weight="bold")
        self._font_subtitle = tkfont.Font(family=_FONT_FAMILY, size=12)
        self._font_info = tkfont.Font(family=_FONT_FAMILY, size=11)
        self._font_label = tkfont.Font(family=_FONT_FAMILY, size=13)
        self._font_label_bold = tkfont.Font(family=_FONT_FAMILY, size=13, weight="bold")
        self._font_hint = tkfont.Font(family=_FONT_FAMILY, size=11)
        self._font_banner = tkfont.Font(family=_FONT_FAMILY, size=16, weight="bold")
        self._font_tabbtn = tkfont.Font(family=_FONT_FAMILY, size=13, weight="bold")

        # 每個 json_key -> {"var":..., "widget":..., "badge_var":..., "field":..., ...}
        self._field_widgets = {}
        # 每個 json_key -> app 啟動當下的「生效值」（getattr config.<class>.<attr> 現讀一次快照），
        # 供「載入目前設定」在 env/json 都沒設定時當作預設值錨點，不在別處重複硬編碼字面值。
        self._baseline_effective = {
            f["json_key"]: getattr(getattr(config, f["attr"][0]), f["attr"][1])
            for f in FIELD_SCHEMA
        }
        # self._main_process 是「目前這個視窗啟動的外部行程」共用欄位——main.py 跟下面
        # 新增的「獨立腳本工具」共用同一個欄位、同一套啟動/關閉/終端機輸出機制，因為
        # 同一時間本來就只允許跑一個（避免兩支行程搶同一份 runtime_settings.current.json／模型
        # 顯存，也让終端機輸出不會兩邊來源混在一起分不清楚）。_active_process_label／
        # _active_kind 記錄「目前（或最後一次）跑的是什麼」，供狀態列文字與按鈕互斥判斷用。
        self._main_process = None  # None＝本視窗目前未啟動任何行程
        self._active_process_label = None  # 例："main.py" 或某個腳本的檔名；None＝從未啟動過
        self._active_kind = None  # "main" | "tool" | None
        self._tool_script_var = tk.StringVar(value="")  # 「獨立腳本工具」下拉/瀏覽選中的 .py 路徑
        # main.py 的 stdout/stderr 用背景執行緒逐行讀取後丟進這個 queue，再由主執行緒
        # 透過 self.after() 輪詢取出、寫進右側終端機面板——tkinter 元件只能在主執行緒
        # 操作，讀取行程輸出又是阻塞式的 readline()，所以不能省略這層 queue 直接跨執行緒改 Text。
        self._log_queue = queue.Queue()
        self._log_reader_thread = None
        self._log_line_count = 0
        # 關閉本視窗（不管是按右上角 X 還是下方「關閉」按鈕）視同 main.py 關閉請求，
        # 避免不小心關掉設定視窗後，main.py 還在背景跑、卻再也找不到入口能停止它。
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

        self._build_header()
        self._build_process_bar()
        self._build_info_bar()
        self._build_middle_area()
        self._build_bottom_bar()

        self._populate_from_effective_state()
        self._poll_main_process()

        # 到這裡整個視窗的固定佔用區塊（標題/流程列/獨立腳本工具列/資訊列/分頁按鈕列/
        # 底部按鈕列）都已經建好，現在才量得到它們的真實高度——用這個真實高度把終端機
        # 面板的「初始」高度設成視窗高度的 CONSOLE_DEFAULT_HEIGHT_FRACTION（預設
        # 75%）。終端機優先、預設就佔大部分畫面是刻意的設計，分頁表單內容區被壓縮到
        # 只剩一小截、甚至看不到欄位是可接受的結果；用比例而非寫死像素值，是因為不同
        # 螢幕解析度／系統 DPI 縮放下可用空間差異很大，比例才會等比縮放。
        self.update_idletasks()
        self._apply_sane_console_default_height()
        # 終端機面板在 _build_middle_area() 就建立了，早於 _build_bottom_bar()；Tk 同層
        # 元件預設的堆疊順序是「後建立的蓋在先建立的上面」，所以底部按鈕列這時候會蓋在
        # 終端機面板之上，跟「終端機蓋過其他內容」的設計相反。_place_console() 每次呼叫
        # 都會 lift() 一次，但 _apply_sane_console_default_height() 只在高度真的改變時
        # 才會呼叫它，剛好沒改變的話就不會補這次 lift()——這裡直接無條件再 lift() 一次，
        # 確保不管前面有沒有觸發 place，終端機面板一定疊在所有其他元件最上層。
        self._console_container.lift()

    def _size_window_near_fullscreen(self):
        """開窗時盡量佔滿螢幕（接近全屏，但保留標題列/工作列，不用真正的無邊框全螢幕），
        欄位多、分頁多，太小的視窗會讓使用者一直捲動找欄位。Windows 上優先用系統原生的
        「最大化」狀態；非 Windows 或該狀態不支援時，退回用螢幕尺寸算一個大視窗並置中。
        """
        try:
            self.state("zoomed")
            return
        except tk.TclError:
            pass
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = int(screen_w * 0.92), int(screen_h * 0.88)
        x, y = (screen_w - w) // 2, (screen_h - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    # ── 版面 ─────────────────────────────────────────────────────────

    def _build_header(self):
        header = tk.Frame(self, bg=COLOR_HEADER_BG)
        header.pack(fill="x")
        self._header_frame = header

        title_row = tk.Frame(header, bg=COLOR_HEADER_BG)
        title_row.pack(fill="x", padx=18, pady=(14, 0))
        tk.Label(
            title_row, text="⚙️ 貓咪監測系統 — 設定管理", bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG,
            font=self._font_title, anchor="w",
        ).pack(side="left")
        self._mtime_var = tk.StringVar(value="")
        tk.Label(
            title_row, textvariable=self._mtime_var, bg=COLOR_HEADER_BG, fg=COLOR_HEADER_SUB_FG,
            font=self._font_title, anchor="e",
        ).pack(side="right")

        tk.Label(
            header,
            text="管理 runtime_settings.current.json（執行期覆寫）；不會修改 config.py 原始碼，"
            "也不會碰 ST-GCN 訓練設定（stgcn_config.yaml）",
            bg=COLOR_HEADER_BG, fg=COLOR_HEADER_SUB_FG, font=self._font_subtitle, anchor="w",
        ).pack(fill="x", padx=18, pady=(2, 14))

    def _build_process_bar(self):
        """main.py 啟動／關閉／縮小本視窗——放在標題正下方、永遠可見（不用捲動就找得到），
        對應「專案規模大、常常找不到 main.py 入口」的痛點：設定存好後直接在這裡啟動，
        不用再去檔案總管或終端機找路徑。第二排是「獨立腳本工具」：下拉選單列出
        cat_monitoring_system/tools/ 底下的 .py，也可以「瀏覽...」選任意 .py（例如
        專案裡其他資料夾的除錯/評估腳本），跟 main.py 共用同一套啟動/關閉/終端機輸出
        機制、同一時間只能跑一個（見 __init__ 裡 self._main_process 的說明）。"""
        bar = tk.Frame(self, bg=COLOR_HEADER_BG)
        bar.pack(fill="x")
        self._process_bar_frame = bar
        inner = tk.Frame(bar, bg=COLOR_HEADER_BG)
        inner.pack(fill="x", padx=18, pady=(0, 6))

        self._process_status_var = tk.StringVar(value="🖥️ 尚未啟動任何程式")
        tk.Label(
            inner, textvariable=self._process_status_var, bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG,
            font=self._font_label_bold, anchor="w",
        ).pack(side="left")

        _styled_button(
            inner, "🗕 縮小視窗", self._on_minimize, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
            font=self._font_hint,
        ).pack(side="right", padx=(8, 0))
        self._stop_main_btn = _styled_button(
            inner, "⏹ 關閉 main.py", self._on_stop_main, BTN_WARN_BG, BTN_WARN_ACTIVE,
            font=self._font_hint,
        )
        self._stop_main_btn.pack(side="right", padx=(8, 0))
        self._start_main_btn = _styled_button(
            inner, "▶ 啟動 main.py", self._on_start_main, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE,
            font=self._font_hint,
        )
        self._start_main_btn.pack(side="right", padx=(8, 0))

        # 「獨立腳本工具」自成一個外框，跟上面 main.py 那排用一條分隔線隔開，
        # 字級／間距刻意比其他按鈕列大一號——下拉選單是這裡的主角（要能一眼看清楚
        # 選的是哪支腳本），分兩排排版，不跟「瀏覽...」「執行」「停止」擠在同一行。
        # 整塊上下方向的間距（frame pady、按鈕內距、提示文字 pady）都比第一版縮到
        # 約 0.8 倍，避免這塊占掉太多垂直空間、把下面的分頁表單／終端機面板往下擠。
        tool_outer = tk.Frame(bar, bg=COLOR_HEADER_BG, highlightbackground=COLOR_HEADER_SUB_FG, highlightthickness=1)
        tool_outer.pack(fill="x", padx=18, pady=(2, 6))

        tool_row1 = tk.Frame(tool_outer, bg=COLOR_HEADER_BG)
        tool_row1.pack(fill="x", padx=10, pady=(5, 2))
        tk.Label(
            tool_row1, text="🧩 獨立腳本工具", bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG,
            font=self._font_label_bold, anchor="w",
        ).pack(side="left")

        tool_row2 = tk.Frame(tool_outer, bg=COLOR_HEADER_BG)
        tool_row2.pack(fill="x", padx=10, pady=(0, 6))
        self._tool_script_map = self._discover_tool_scripts()  # 顯示名稱（含流水號）→ 完整路徑
        # 下拉選單本身字級加大到跟表單欄位同級（原本用 _font_hint 太小，字擠在一起
        # 分不清楚選的是哪支）；展開後的清單（popdown listbox）字級另外設、還要放大
        # 到 Combobox 本身字級的 1.5 倍——這是使用者明確要的：清單一次列出一堆腳本
        # 名稱，字大一號＋淡藍底色掃視起來更輕鬆，不用瞇眼睛找。popdown listbox 是
        # ttk 內部另外生的元件，不會自動跟著 Combobox 本身的 font/顏色走，只能用
        # option_add() 這種全域樣式規則設，沒有直接的 widget 參數可以配置。字型改用
        # 跟終端機面板同一款等寬字（CONSOLE_FONT_FAMILY），因為 _discover_tool_scripts()
        # 用補空白對齊流水號，非等寬字型（例如中文字型）每個字元寬度不一，補再多空白
        # 也對不齊；等寬字下用字元數計算才是準的。
        self._tool_listbox_font = tkfont.Font(
            family=CONSOLE_FONT_FAMILY, size=round(self._font_label.cget("size") * 1.5)
        )
        combo = ttk.Combobox(
            tool_row2, textvariable=self._tool_script_var,
            values=list(self._tool_script_map.keys()), font=self._tool_listbox_font, height=16,
        )
        combo.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=2)
        self.option_add("*TCombobox*Listbox.font", self._tool_listbox_font)
        self.option_add("*TCombobox*Listbox.background", COLOR_TOOL_LISTBOX_BG)
        self.option_add("*TCombobox*Listbox.foreground", COLOR_LABEL_FG)
        self.option_add("*TCombobox*Listbox.selectBackground", TAB_COLORS["模型與輸入來源"][1])
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self._tool_combo = combo

        # 即時篩選：邊打字邊把清單縮小到「顯示名稱含有目前輸入內容」的腳本（不分大小
        # 寫）。Ctrl+F 是進入點——把焦點切到這顆下拉選單、清空目前內容準備輸入；
        # <KeyRelease> 則是每打一個字就重新篩選一次 values。
        #
        # 這裡刻意不會在打字過程中自動展開/刷新 popdown 清單——試過用
        # combo.event_generate("<Down>")（模擬按下方向鍵）或 ttk::combobox::Post
        # （ttk 內部「顯示 popdown」指令）都一樣：不管哪一種，ttk combobox 都會在
        # 顯示 popdown 的同時把鍵盤焦點轉移到清單本身（讓使用者能用方向鍵選取項目），
        # 就算馬上呼叫 combo.focus_set() 搶回來也沒用，因為那個焦點轉移是 ttk 內部
        # 排到之後才執行的，會蓋掉搶回來的結果——這正是「打 Ctrl+F 之後打不了字」的
        # 根本原因：打第一個字，篩選函式就自動彈出清單，焦點被清單搶走，後面打的字
        # 全部進不了輸入框。改成單純更新 combo["values"]（不主動彈出），篩選結果會
        # 先安靜地縮小，游標／焦點全程留在輸入框，可以正常一路打完整個查詢字串；
        # 想看篩選後的清單，打完字自己按一次下拉箭頭或 ↓ 鍵展開即可。
        all_display_names = list(self._tool_script_map.keys())
        # 導覽鍵不觸發重新篩選，否則按 ↓ 選清單裡的項目會把該項目文字填回輸入框、
        # 又被當成新的篩選字串重新篩一次，跟使用者原本想「往下移動選取」的意圖對不上。
        _nav_keysyms = {"Up", "Down", "Return", "KP_Enter", "Escape", "Tab"}

        def _refresh_tool_combo_filter(event=None):
            if event is not None and event.keysym in _nav_keysyms:
                return
            typed = combo.get().strip().lower()
            if typed:
                # 搜尋範圍不只比對檔名，也比對 self._tool_script_desc_map 的功能說明
                # 文字——記不住確切檔名、只記得「大概是做什麼的」時一樣找得到（例如
                # 打「比較」能找到 eval_pose_compare.py，即使檔名本身沒有這兩個字）。
                combo["values"] = [
                    n for n in all_display_names
                    if typed in n.lower() or typed in self._tool_script_desc_map.get(n, "").lower()
                ]
            else:
                combo["values"] = all_display_names

        combo.bind("<KeyRelease>", _refresh_tool_combo_filter)

        # 常駐說明卡片：選定（或篩選/打字剛好完全對到）某支腳本時，這裡會顯示
        # docs/獨立運行腳本索引.md 記錄的功能說明——常忘記腳本名稱或功能時不用切去
        # 開文件對照，選了就看得到。trace_add 綁在 StringVar 上，不管是滑鼠選清單、
        # 「瀏覽...」選檔案、還是打字打到完全符合，都會觸發更新。
        #
        # 特地做成一張淡藍底、左側色條、大圖示的卡片——跟深色的流程列/獨立腳本工具
        # 外框拉開視覺層次，讓人一眼就注意到「這裡有東西」，不是埋在一堆按鈕文字裡
        # 容易被忽略的一行小字；圖示會依狀態換（📖 找到說明／💡 尚未選擇／⚠️ 找不到
        # 說明），不是每次都同一個圖示，掃視時更容易分辨目前是哪種狀態。
        desc_card = tk.Frame(
            tool_outer, bg=COLOR_TOOL_DESC_BG,
            highlightbackground=COLOR_TOOL_DESC_BORDER, highlightthickness=1,
        )
        desc_card.pack(fill="x", padx=10, pady=(2, 6))
        tk.Frame(desc_card, bg=COLOR_TOOL_DESC_ACCENT, width=5).pack(side="left", fill="y")

        self._tool_desc_icon_var = tk.StringVar(value="💡")
        tk.Label(
            desc_card, textvariable=self._tool_desc_icon_var, bg=COLOR_TOOL_DESC_BG,
            font=("Segoe UI Emoji", 20),
        ).pack(side="left", padx=(12, 10), pady=10)

        self._tool_desc_var = tk.StringVar(value="")
        tk.Label(
            desc_card, textvariable=self._tool_desc_var, bg=COLOR_TOOL_DESC_BG, fg=COLOR_TOOL_DESC_FG,
            font=self._font_label, anchor="w", justify="left", wraplength=1780,
        ).pack(side="left", fill="x", expand=True, padx=(0, 12), pady=10)

        def _on_tool_script_var_change(*_a):
            # 這個 trace 綁在 self._tool_script_var 上，Combobox 打字時每敲一個字都會
            # 觸發（textvariable 本來就跟 Entry 內容即時同步，不是只有選定/送出時才變）
            # ——之前的版本只判斷「完全對到清單裡某個顯示名稱」，打字打到一半、還沒
            # 選定完成時一律顯示「⚠️ 找不到說明」，看起來像是搜尋沒作用；拿掉了打字時
            # 自動彈出清單那個機制（見上面 _refresh_tool_combo_filter 的說明，那樣做
            # 會搶走鍵盤焦點導致打不了字）之後，這個誤導感更明顯——不打開清單，畫面
            # 上完全沒有任何回饋。這裡改成：打完整符合就顯示說明；沒完整符合但還有
            # 部分符合（檔名或功能說明含有目前打的字）就顯示「符合幾支＋預覽名稱」，
            # 讓使用者不用手動展開清單也能立刻看到搜尋有沒有效果；真的一支都不符合
            # 才顯示找不到。
            raw = self._tool_script_var.get()
            desc = self._tool_script_desc_map.get(raw)
            if desc:
                self._tool_desc_icon_var.set("📖")
                self._tool_desc_var.set(desc)
                return
            stripped = raw.strip()
            if not stripped:
                self._tool_desc_icon_var.set("💡")
                self._tool_desc_var.set("從下拉選單選一支腳本後，這裡會顯示它的功能說明"
                                         "（來源：docs/獨立運行腳本索引.md）")
                return
            typed = stripped.lower()
            matches = [
                n for n in all_display_names
                if typed in n.lower() or typed in self._tool_script_desc_map.get(n, "").lower()
            ]
            if matches:
                self._tool_desc_icon_var.set("🔍")
                preview = "、".join(m.split("#")[0].strip() for m in matches[:3])
                more = f" 等共 {len(matches)} 支" if len(matches) > 3 else ""
                self._tool_desc_var.set(
                    f"符合「{stripped}」：{preview}{more} —— 按 ↓ 或點下拉箭頭展開清單挑選"
                )
            else:
                self._tool_desc_icon_var.set("⚠️")
                self._tool_desc_var.set(
                    f"找不到符合「{stripped}」的腳本或功能說明"
                    "（如果這是「瀏覽...」選的清單外路徑，這是正常的，不影響執行）"
                )

        self._tool_script_var.trace_add("write", _on_tool_script_var_change)
        _on_tool_script_var_change()

        def _focus_tool_combo_search(_event=None):
            combo["values"] = all_display_names
            combo.focus_set()
            combo.delete(0, "end")
            return "break"

        self.bind_all("<Control-f>", _focus_tool_combo_search)

        browse_btn = _styled_button(
            tool_row2, "瀏覽...", self._on_browse_tool_script, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
            font=self._font_label,
        )
        browse_btn.pack(side="left", padx=(0, 8))
        open_file_btn = _styled_button(
            tool_row2, "開啟檔案", self._on_open_tool_script_file, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
            font=self._font_label,
        )
        open_file_btn.pack(side="left", padx=(0, 8))
        self._start_tool_btn = _styled_button(
            tool_row2, "▶ 執行所選腳本", self._on_start_tool, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE,
            font=self._font_label,
        )
        self._start_tool_btn.pack(side="left", padx=(0, 8))
        self._stop_tool_btn = _styled_button(
            tool_row2, "⏹ 停止腳本", self._on_stop_tool, BTN_WARN_BG, BTN_WARN_ACTIVE,
            font=self._font_label,
        )
        self._stop_tool_btn.pack(side="left")
        for _btn in (browse_btn, open_file_btn, self._start_tool_btn, self._stop_tool_btn):
            _btn.config(pady=4)  # _styled_button 預設 pady=6，這裡縮小到約 0.8 倍跟整塊區域一致

        tk.Label(
            tool_outer,
            text="下拉選單只列出 cat_monitoring_system/tools/ 底下的腳本（依資料夾/檔名排序，"
            "子資料夾會顯示成「資料夾名/檔名.py」）；其他位置的腳本（例如 cat_pose/ 底下）用「瀏覽...」選取。"
            "這些獨立工具大多需要先打開改寫死在檔案開頭的路徑/參數再執行，"
            "「開啟檔案」用 VS Code（找不到的話退回記事本）開啟目前選定腳本的原始碼。",
            bg=COLOR_HEADER_BG, fg=COLOR_HEADER_SUB_FG, font=self._font_hint,
            anchor="w", justify="left", wraplength=1400,
        ).pack(fill="x", padx=10, pady=(0, 4))

        self._update_process_buttons_state()

    def _on_minimize(self):
        self.iconify()

    def _discover_tool_scripts(self):
        """掃 cat_monitoring_system/tools/ 底下（含子資料夾，例如 train_data/）所有 .py，
        回傳 {顯示名稱: 完整路徑} 給下拉選單用——顯示名稱用「相對於 tools/ 的路徑」
        （例如 "eval_pose_compare.py"、"train_data/0_dataset_collect.py"），比起完整
        絕對路徑短很多，才看得清楚選的是哪一支；「瀏覽...」則可另外選這個清單以外的
        任意 .py（例如 C:\\ai_project\\cat_pose\\ 底下那些獨立工具）。

        每個顯示名稱後面補空白對齊、再接兩位數流水號（例如 "01"、"02"...），流水號
        統一補到清單裡「最寬」檔名之後那一欄——用等寬字型（見 combo 那邊的
        _tool_listbox_font）搭配這個補空白對齊，數字才會排成一直線。這裡用「顯示
        寬度」（_display_width，全形/中文字元算 2 個半形字寬）而不是單純字元數
        （len()）來算要補幾格空白：tools/ 底下不是所有腳本都是純 ASCII 檔名（例如
        `影片拼接.py`），中文字元在等寬字型下通常還是佔兩個半形字的寬度，直接用
        len() 對這種檔名補的空白數會不夠、流水號對不齊。流水號同時也是 Ctrl+F
        打字篩選時，使用者一眼確認「這是清單第幾支」的依據。"""
        self._tool_script_desc_map = {}  # 顯示名稱（含流水號）→ 功能說明，給常駐說明列／Ctrl+F 用
        # 一定要在下面 tools_dir 不存在時的早退之前設好：_on_tool_script_var_change()
        # 在 _build_process_bar() 尾端會無條件呼叫一次，若 tools_dir 剛好讀不到（例如
        # 資料夾正在被同步/搬移）就會早退不往下跑，這個屬性沒設到就直接 AttributeError。
        tools_dir = _SCRIPT_DIR / "cat_monitoring_system" / "tools"
        if not tools_dir.exists():
            return {}
        # 排除底線開頭的檔名（例如 _smoothing_eval_common.py）——這是 Python 慣例的
        # 「內部共用模組」命名，只被其他腳本 import，本身沒有 if __name__ == "__main__"
        # 可執行區塊，選了直接執行只會靜靜跑完 import、什麼事都沒發生，對使用者來說
        # 像是壞掉了；docs/獨立運行腳本索引.md 也不會有這種模組的條目（本來就不是
        # 獨立腳本），列進下拉選單只會製造一個沒有說明、點了也沒反應的選項。
        paths = sorted(
            p for p in tools_dir.rglob("*.py")
            if "__pycache__" not in p.parts and not p.name.startswith("_")
        )
        names = [str(p.relative_to(tools_dir)).replace("\\", "/") for p in paths]
        widths = [_display_width(n) for n in names]
        pad_width = max(widths, default=0) + 4
        plain_descriptions = self._load_tool_script_descriptions()  # {相對路徑: 功能說明}
        mapping = {}
        for idx, (name, width, p) in enumerate(zip(names, widths, paths), start=1):
            display = f"{name}{' ' * (pad_width - width)}#{idx:02d}"
            mapping[display] = str(p)
            self._tool_script_desc_map[display] = plain_descriptions.get(name, "")
        return mapping

    def _load_tool_script_descriptions(self):
        """解析 docs/獨立運行腳本索引.md 裡「5. paper/cat_monitoring_system/tools/」
        與「6. .../train_data/」兩節的表格，取出每支腳本的「功能」欄位文字，回傳
        {相對於 tools/ 的路徑: 功能說明}——這份文件本來就是這個專案既有、持續維護
        的腳本索引，不重新生成一份新的說明，直接沿用。表格格式是
        `| \\`檔名\\` | 功能說明 | 使用方法 |`，用簡單的逐行 regex 讀，不依賴任何
        Markdown 套件。找不到檔案或格式對不上時安靜回傳空字典，不影響下拉選單本身
        正常運作——這份說明是加值資訊，不是必要功能，讀取失敗不該讓整個 GUI 掛掉。"""
        md_path = _SCRIPT_DIR / "docs" / "獨立運行腳本索引.md"
        descriptions = {}
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError:
            return descriptions

        row_re = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|")

        def _parse_section(section_text, prefix=""):
            for line in section_text.splitlines():
                m = row_re.match(line.strip())
                if m:
                    descriptions[f"{prefix}{m.group(1)}"] = m.group(2)

        # 第 5 節表格內的路徑就是相對於 tools/ 的檔名（例如 "eval_pose_compare.py"），
        # 直接對得上 _discover_tool_scripts() 算出來的 name；第 6 節（train_data/
        # 子資料夾）表格內路徑沒帶 "train_data/" 前綴，這裡要另外補上才對得起來。
        #
        # 章節邊界的標題比對一定要用 re.M 的 ^ 錨定在「行首」，不能只單純用
        # re.search 找字串出現的位置——這份文件如果哪裡用一般文字或行內程式碼提到
        # 這兩個標題字串（例如說明格式規則時舉例引用），沒有錨定行首的話會誤認那裡
        # 就是章節開頭，抓到完全不對的範圍，導致整份說明解析失敗（此教訓來自實際
        # 踩過一次：文件裡新增的格式規範說明段落引用了標題全文，行首錨定修好前，
        # 27 支腳本的說明一次全部消失）。
        sec5 = re.search(r"^## 5\. paper/cat_monitoring_system/tools/.*?(?=^## \d|\Z)", text, re.S | re.M)
        if sec5:
            _parse_section(sec5.group(0))
        sec6 = re.search(
            r"^## 6\. paper/cat_monitoring_system/tools/train_data/.*?(?=^## \d|\Z)", text, re.S | re.M
        )
        if sec6:
            _parse_section(sec6.group(0), prefix="train_data/")
        return descriptions

    def _on_window_close(self):
        """視窗關閉的唯一入口（右上角 X 與下方「關閉」按鈕都走這裡）：main.py 或
        獨立腳本工具若還在本視窗啟動的範圍內執行中，視同一併請求關閉，並實際等到
        確認結束才讓視窗消失，不是送出信號就放著不管——避免「視窗關掉了，行程其實
        還在跑」的情況。"""
        if self._main_process is not None and self._main_process.poll() is None:
            if not messagebox.askyesno(
                "關閉設定視窗",
                f"{self._active_process_label} 目前正在執行中（PID {self._main_process.pid}）。\n\n"
                "關閉本視窗會一併送出關閉信號，避免關掉視窗後失去控制、"
                "無法再停止正在執行的程式。\n\n是否繼續？",
            ):
                return
            self._process_status_var.set(f"🖥️ 正在關閉 {self._active_process_label}…")
            self.update_idletasks()
            if not self._request_main_shutdown():
                messagebox.showwarning(
                    "關閉設定視窗",
                    f"{self._active_process_label} 似乎沒有在預期時間內結束，請自行檢查工作管理員確認狀態。",
                )
        self.destroy()

    def _on_start_main(self):
        if self._main_process is not None and self._main_process.poll() is None:
            messagebox.showinfo(
                "啟動 main.py",
                f"{self._active_process_label} 執行中（PID {self._main_process.pid}），"
                "請先停止後再啟動 main.py（同一時間只能執行一個）。",
            )
            return
        if not _MAIN_PY_PATH.exists():
            messagebox.showerror("啟動 main.py", f"找不到 main.py：{_MAIN_PY_PATH}")
            return
        if not messagebox.askyesno(
            "啟動 main.py",
            "即將啟動 main.py，套用目前 runtime_settings.current.json／環境變數的設定內容。\n\n"
            "若您在下方設定表單中有修改但尚未按「儲存設定」，這些修改不會套用到這次啟動。\n\n"
            "是否繼續？",
        ):
            return
        try:
            # stdout/stderr 導向 pipe 讓下方終端機面板即時讀取，不再依賴「本視窗是否
            # 在終端機裡啟動」——雙擊執行 settings_window.py（沒有終端機視窗）時，
            # main.py 的輸出過去完全看不到，這是這裡改用 PIPE 的主要動機。
            # stdin 也導向 pipe：不設的話子行程預設會繼承本視窗自己的 stdin（如果是被
            # cmd/bat 啟動的話），main.py 若用到 input() 之類的互動輸入，畫面會停在
            # 一個使用者根本看不到、也打不到字的地方，變成「卡住卻不知道在等什麼」。
            # 改成 PIPE 後，下方終端機面板新增的輸入框才能把文字寫進子行程的 stdin。
            # PYTHONIOENCODING/PYTHONUTF8：main.py 印的訊息大量使用中文與 emoji，子行程
            # 若沿用系統預設的 ANSI 編碼（如 cp950）寫出 stdout，跟這裡用 utf-8 解碼會對不上、
            # 印出亂碼，所以強制子行程一律以 UTF-8 寫出。
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["PYTHONUTF8"] = "1"
            popen_kwargs = {
                "cwd": str(_SCRIPT_DIR),
                "env": child_env,
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
            }
            if os.name == "nt":
                # 只用 CREATE_NEW_PROCESS_GROUP（不加 CREATE_NEW_CONSOLE）：
                # Windows 的 GenerateConsoleCtrlEvent（CTRL_BREAK_EVENT 底層機制）只能送到
                # 「跟呼叫端共用同一個主控台」的行程群組——先前版本額外加了
                # CREATE_NEW_CONSOLE 讓 main.py 開獨立主控台視窗顯示輸出，結果反而讓
                # 「關閉 main.py」按鈕送出的 CTRL_BREAK_EVENT 永遠送不到（不同主控台，
                # 對方收不到信號），這是關閉沒有生效的根本原因。改成不開新主控台，
                # 輸出改走 stdout=PIPE 由右側面板顯示，兩者互不衝突，關閉功能維持生效。
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self._main_process = subprocess.Popen([sys.executable, str(_MAIN_PY_PATH)], **popen_kwargs)
        except OSError as e:
            messagebox.showerror("啟動 main.py", f"啟動失敗：{e}")
            return
        self._active_process_label = "main.py"
        self._active_kind = "main"
        self._update_process_buttons_state()
        self._on_clear_console()
        self._console_append(f"— main.py 已啟動（PID {self._main_process.pid}） —\n", tag="muted")
        self._start_log_reader(self._main_process)
        self._bring_child_window_to_front(self._main_process)
        messagebox.showinfo(
            "啟動 main.py",
            f"main.py 已啟動（PID {self._main_process.pid}）。\n\n"
            "輸出會即時顯示在下方「終端機輸出」面板中。",
        )

    def _on_stop_main(self):
        if self._active_kind != "main" or self._main_process is None or self._main_process.poll() is not None:
            messagebox.showinfo("關閉 main.py", "目前沒有偵測到由本視窗啟動、仍在執行中的 main.py。")
            self._update_process_buttons_state()
            return
        self._process_status_var.set("🖥️ 正在關閉 main.py…")
        self.update_idletasks()
        stopped = self._request_main_shutdown()
        self._update_process_buttons_state()
        if stopped:
            messagebox.showinfo("關閉 main.py", "main.py 已確認結束。")
        else:
            messagebox.showwarning(
                "關閉 main.py",
                "已送出關閉信號並嘗試強制結束，但仍無法確認 main.py 已停止，"
                "請自行檢查工作管理員確認狀態。",
            )

    def _on_browse_tool_script(self):
        initial_dir = str(_SCRIPT_DIR / "cat_monitoring_system" / "tools")
        path = filedialog.askopenfilename(
            title="選擇要執行的 Python 腳本",
            initialdir=initial_dir if Path(initial_dir).exists() else str(_SCRIPT_DIR),
            filetypes=[("Python 腳本", "*.py"), ("所有檔案", "*.*")],
        )
        if path:
            self._tool_script_var.set(path)

    def _on_open_tool_script_file(self):
        """用文字編輯器開啟目前選定的腳本原始碼——這些獨立工具大多是「先打開改
        檔案開頭寫死的常數（模型路徑、RUN_MODE 之類），再執行」的用法，開啟檔案
        是執行前的常見前置動作，跟「瀏覽...」選檔案、「▶ 執行所選腳本」是互補的。

        刻意不用 os.startfile()（作業系統預設關聯程式）——這台電腦不確定 .py 的
        預設關聯是編輯器還是直接用 Python 執行，萬一是後者，一顆寫著「開啟檔案」
        的按鈕卻把腳本執行了會很意外。改成明確找文字編輯器開：優先用 VS Code
        （這個專案本來就是用 VS Code 開發，指令列有裝的話直接找得到），找不到就
        退回 Windows 內建、保證存在、也保證不會執行 .py 的記事本。
        """
        raw = self._tool_script_var.get().strip()
        if not raw:
            messagebox.showwarning("開啟檔案", "請先從下拉選單選擇，或按「瀏覽...」挑一個 .py 腳本。")
            return
        script_path = self._tool_script_map.get(raw, raw)
        script_file = Path(script_path)
        if not script_file.exists():
            messagebox.showerror("開啟檔案", f"找不到檔案：\n{script_path}")
            return
        # notepad.exe 用完整路徑而不是靠 PATH 解析裸檔名——實測發現某些啟動環境
        # （例如透過 Anaconda 環境的 python.exe 執行時）PATH 裡不見得含
        # C:\Windows\System32，單寫 "notepad.exe" 會直接 FileNotFoundError；
        # %SystemRoot% 這個環境變數則不受 PATH 影響，永遠指向 Windows 安裝目錄。
        fallback_notepad = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "notepad.exe")
        editor = shutil.which("code") or fallback_notepad
        try:
            subprocess.Popen([editor, str(script_file)])
        except OSError as e:
            messagebox.showerror("開啟檔案", f"開啟失敗：{e}")

    def _on_start_tool(self):
        raw = self._tool_script_var.get().strip()
        if not raw:
            messagebox.showwarning("執行腳本", "請先從下拉選單選擇，或按「瀏覽...」挑一個 .py 腳本。")
            return
        # 下拉選單顯示的是相對路徑（例如 "eval_pose_compare.py"），要透過
        # self._tool_script_map 換回完整路徑；「瀏覽...」或手動輸入的則已經是完整
        # 路徑，不在 map 裡，.get(raw, raw) 這種情況直接把 raw 原樣當完整路徑用。
        script_path = self._tool_script_map.get(raw, raw)
        script_file = Path(script_path)
        if not script_file.exists():
            messagebox.showerror("執行腳本", f"找不到檔案：\n{script_path}")
            return
        if self._main_process is not None and self._main_process.poll() is None:
            messagebox.showinfo(
                "執行腳本",
                f"{self._active_process_label} 執行中（PID {self._main_process.pid}），"
                "請先停止後再執行其他腳本（同一時間只能執行一個）。",
            )
            return
        if not messagebox.askyesno(
            "執行腳本",
            f"即將執行：{script_file.name}\n\n完整路徑：{script_path}\n\n"
            "這是一支獨立腳本工具，本視窗不會檢查或修改它內部寫死的路徑/參數，"
            "請自行確認內容符合你要的設定。\n\n是否繼續？",
        ):
            return
        try:
            # 跟 _on_start_main 共用同一套 PIPE + UTF-8 強制編碼的邏輯，理由同上；
            # cwd 用腳本自己所在的資料夾，比照「假裝 cd 進去該資料夾再執行」的直覺，
            # 因為這些獨立腳本大多是各自獨立維護、原本就預期在自己的資料夾下執行。
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["PYTHONUTF8"] = "1"
            popen_kwargs = {
                "cwd": str(script_file.parent),
                "env": child_env,
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self._main_process = subprocess.Popen([sys.executable, str(script_file)], **popen_kwargs)
        except OSError as e:
            messagebox.showerror("執行腳本", f"啟動失敗：{e}")
            return
        self._active_process_label = script_file.name
        self._active_kind = "tool"
        self._update_process_buttons_state()
        self._on_clear_console()
        self._console_append(f"— {script_file.name} 已啟動（PID {self._main_process.pid}） —\n", tag="muted")
        self._start_log_reader(self._main_process)
        self._bring_child_window_to_front(self._main_process)

    def _on_stop_tool(self):
        if self._active_kind != "tool" or self._main_process is None or self._main_process.poll() is not None:
            messagebox.showinfo("停止腳本", "目前沒有偵測到由本視窗啟動、仍在執行中的腳本。")
            self._update_process_buttons_state()
            return
        label = self._active_process_label
        self._process_status_var.set(f"🖥️ 正在停止 {label}…")
        self.update_idletasks()
        stopped = self._request_main_shutdown()
        self._update_process_buttons_state()
        if stopped:
            messagebox.showinfo("停止腳本", f"{label} 已確認結束。")
        else:
            messagebox.showwarning(
                "停止腳本",
                f"已送出關閉信號並嘗試強制結束，但仍無法確認 {label} 已停止，"
                "請自行檢查工作管理員確認狀態。",
            )

    def _request_main_shutdown(self, graceful_timeout=4.0, force_timeout=2.0) -> bool:
        """請求 self._main_process（可能是 main.py，也可能是獨立腳本工具）結束，
        回傳 True 代表最終確認已經不在執行。

        兩段式：先送 CTRL_BREAK_EVENT（main.py 有 _install_hard_shutdown_on_ctrl_c()
        接管 SIGBREAK，會先嘗試清理 CSV/segment logger/Node-RED session 再結束，本身
        內建 3 秒 watchdog；一般獨立腳本沒有這層特別處理，收到 CTRL_BREAK_EVENT 多半
        直接被 Windows 結束掉，效果上也是「停下來」），等 graceful_timeout 秒讓它有
        機會走完清理流程；逾時仍在跑，才 terminate() 強制砍斷，再等 force_timeout 秒
        確認真的死透。「關閉 main.py」「停止腳本」按鈕跟「關掉設定視窗」共用這個函式，
        確保三個入口都是真的確認生效、不是送出信號就假設成功。
        """
        if self._main_process is None or self._main_process.poll() is not None:
            return True
        try:
            if os.name == "nt":
                self._main_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self._main_process.send_signal(signal.SIGINT)
        except OSError:
            pass

        deadline = time.monotonic() + graceful_timeout
        while time.monotonic() < deadline:
            if self._main_process.poll() is not None:
                return True
            time.sleep(0.2)

        if self._main_process.poll() is None:
            try:
                self._main_process.terminate()
            except OSError:
                pass
            deadline = time.monotonic() + force_timeout
            while time.monotonic() < deadline:
                if self._main_process.poll() is not None:
                    return True
                time.sleep(0.2)

        return self._main_process.poll() is not None

    def _update_process_buttons_state(self):
        """main.py 與獨立腳本工具共用 self._main_process 這一個欄位（同一時間只能跑一個），
        所以兩組「啟動」按鈕永遠一起致能/禁用；「停止」按鈕則依 self._active_kind 只讓
        對應那一顆生效，避免「明明在跑腳本，卻按了寫著『關閉 main.py』的按鈕」這種文字
        跟實際動作對不上的情況。"""
        running = self._main_process is not None and self._main_process.poll() is None
        if running:
            self._process_status_var.set(f"🖥️ {self._active_process_label} 執行中（PID {self._main_process.pid}）")
            self._start_main_btn.config(state="disabled")
            self._start_tool_btn.config(state="disabled")
            self._stop_main_btn.config(state="normal" if self._active_kind == "main" else "disabled")
            self._stop_tool_btn.config(state="normal" if self._active_kind == "tool" else "disabled")
        else:
            was_started = self._active_process_label is not None
            self._process_status_var.set(
                f"🖥️ {self._active_process_label} 已結束" if was_started else "🖥️ 尚未啟動任何程式"
            )
            self._start_main_btn.config(state="normal")
            self._start_tool_btn.config(state="normal")
            self._stop_main_btn.config(state="disabled")
            self._stop_tool_btn.config(state="disabled")

        # 終端機面板的輸入列在本函式第一次被呼叫時（_build_process_bar 尾端）還沒
        # 建出來（_build_middle_area／_build_console_panel 是之後才跑的），用
        # getattr 擋一下，之後每次呼叫這兩個 widget 就一定存在了。
        stdin_entry = getattr(self, "_stdin_entry", None)
        send_stdin_btn = getattr(self, "_send_stdin_btn", None)
        if stdin_entry is not None and send_stdin_btn is not None:
            state = "normal" if running else "disabled"
            stdin_entry.config(state=state)
            send_stdin_btn.config(state=state)

    def _poll_main_process(self):
        """每 2 秒檢查一次子行程是否還活著——不管是 main.py 還是獨立腳本工具，都有可能
        不是被「關閉／停止」按鈕結束的（使用者直接把主控台視窗叉掉、或程式自己崩潰），
        這裡確保按鈕狀態不會卡住。"""
        if self._main_process is not None:
            self._update_process_buttons_state()
        self.after(2000, self._poll_main_process)

    def _build_info_bar(self):
        wrap = tk.Frame(self, bg=COLOR_BG_MAIN)
        wrap.pack(fill="x", padx=16, pady=(12, 6))
        self._info_bar_frame = wrap
        box = tk.Frame(wrap, bg=COLOR_INFO_BG, highlightbackground=COLOR_INFO_BORDER, highlightthickness=1)
        box.pack(fill="x")
        self._info_var = tk.StringVar(value="")
        tk.Label(
            box, textvariable=self._info_var, bg=COLOR_INFO_BG, fg=COLOR_INFO_FG,
            font=self._font_info, anchor="w", justify="left",
        ).pack(fill="x", padx=12, pady=8)
        self._refresh_top_info()

    def _refresh_top_info(self):
        path = settings_manager.get_runtime_settings_path()
        if settings_manager.runtime_settings_exists():
            mtime = settings_manager.get_last_modified()
            mtime_str = mtime.strftime("%Y-%m-%d %H:%M:%S") if mtime else "未知"
            self._mtime_var.set(f"🕒 最後修改：{mtime_str}")
            self._info_var.set(f"📄 runtime_settings.current.json：{path}")
        else:
            self._mtime_var.set("🕒 最後修改：（尚未建立設定檔）")
            self._info_var.set(
                f"📄 runtime_settings.current.json：{path}（尚未建立）\n"
                "⚠ 目前僅使用環境變數／config.py 內建預設值；按「儲存設定」後才會建立此檔案。"
            )

    def _build_middle_area(self):
        """分頁設定表單維持完整版面（永遠是滿版 pack，不因為終端機而縮水）。終端機
        輸出面板改用 place() 做成一塊貼齊視窗左右邊界與底部、可自由調整高度的浮動
        面板——不參與 pack 版面協商，所以拉高時是直接「疊在」表單上面蓋過去，不是
        跟表單擠位置；可以一路拉到貼齊標題列下緣（見 _console_max_height()），此時
        流程列／獨立腳本工具列／資訊列／分頁按鈕列／表單內容／底部按鈕列都會被蓋住，
        這是刻意允許的效果，使用者要操作這些時把終端機拖小或收合即可隨時蓋回去。"""
        middle = tk.Frame(self, bg=COLOR_BG_MAIN)
        middle.pack(fill="both", expand=True)
        self._build_tabs(middle)

        console_container = tk.Frame(self, bg=COLOR_CONSOLE_BG, height=CONSOLE_DEFAULT_HEIGHT)
        console_container.pack_propagate(False)  # 固定高度，不因裡面的 Text 內容而被撐大
        self._build_console_panel(console_container)
        self._place_console(CONSOLE_DEFAULT_HEIGHT)

    def _place_console(self, height):
        """終端機面板固定用 place() 貼齊視窗左右邊界，高度由參數決定——這一個函式是
        所有「改變終端機高度」操作（拖拉／收合展開／初始套用比例）共用的唯一進出口，
        確保每次呼叫方式都一致，也確保面板永遠疊在其他內容之上（每次呼叫都 lift()
        一次，不受其他元件建立順序影響）。

        底部不是貼齊視窗最底端（那樣會蓋到「儲存設定」那排按鈕，按不到），而是貼齊
        底部按鈕列的「上緣」——用 self._bottom_bar_frame.winfo_y()（它在 self 座標系
        下的 y 位置，因為底部按鈕列本來就是 self 的直接子元件）當作面板下緣的錨點，
        往上量 height 這麼高。_build_bottom_bar() 還沒建立、拿不到這個參照時，先退回
        貼齊視窗底部（僅發生在建構過程最初那一次呼叫，稍後 _bottom_bar_frame 建好後
        的下一次呼叫就會修正到正確位置）。"""
        bottom_bar = getattr(self, "_bottom_bar_frame", None)
        floor_y = bottom_bar.winfo_y() if bottom_bar is not None else self.winfo_height()
        self._console_container.place(x=0, y=floor_y, anchor="sw", relwidth=1.0, height=height)
        self._console_container.lift()

    def _build_console_panel(self, parent):
        """終端機面板可拖拉調整高度（頂部把手）、也可用箭頭按鈕一鍵內縮成一條水平線
        （收合時只留標題列，箭頭仍看得到，再點一次箭頭恢復收合前的高度）。"""
        self._console_container = parent
        self._console_collapsed = False
        self._console_expanded_height = CONSOLE_DEFAULT_HEIGHT

        # 拖拉把手：一條比面板底色略亮的細線，滑鼠移上去會變成上下箭頭游標，
        # 拖動時即時調整面板高度——收合時這條把手會跟著隱藏（沒有東西可以拖）。
        grip = tk.Frame(parent, bg=COLOR_CONSOLE_GRIP_BG, height=5, cursor="sb_v_double_arrow")
        grip.pack(fill="x", side="top")
        grip.bind("<ButtonPress-1>", self._on_console_drag_start)
        grip.bind("<B1-Motion>", self._on_console_drag_motion)
        self._console_grip = grip

        header = tk.Frame(parent, bg=COLOR_HEADER_BG, cursor="sb_v_double_arrow")
        header.pack(fill="x", side="top")
        header.bind("<ButtonPress-1>", self._on_console_drag_start)
        header.bind("<B1-Motion>", self._on_console_drag_motion)
        self._console_header = header

        self._console_toggle_btn = tk.Button(
            header, text="▼", command=self._toggle_console_collapse, bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG,
            activebackground=COLOR_HEADER_BG, activeforeground=COLOR_HEADER_FG, relief="flat", bd=0,
            font=self._font_label_bold, cursor="hand2", padx=8,
        )
        self._console_toggle_btn.pack(side="left")
        tk.Label(
            header, text="🖥️ 終端機輸出（main.py／獨立腳本工具共用；可拖拉頂端調整高度）",
            bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG, font=self._font_label_bold, anchor="w",
        ).pack(side="left", padx=(0, 10), pady=6)

        # 工具列 + Text 輸出區包成一個子容器，收合時整包 pack_forget()，
        # 展開時整包用 before/after 對齊 grip／header 之間的正確順序重新插回。
        body = tk.Frame(parent, bg=COLOR_CONSOLE_BG)
        body.pack(fill="both", expand=True, side="top")
        self._console_body = body

        toolbar = tk.Frame(body, bg=COLOR_CONSOLE_BG)
        toolbar.pack(fill="x", padx=8, pady=(6, 4))
        self._autoscroll_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            toolbar, text="自動捲動", variable=self._autoscroll_var, bg=COLOR_CONSOLE_BG,
            fg=COLOR_CONSOLE_FG, selectcolor=COLOR_CONSOLE_BG, activebackground=COLOR_CONSOLE_BG,
            activeforeground=COLOR_CONSOLE_FG, font=self._font_hint, bd=0, highlightthickness=0,
        ).pack(side="left")
        _styled_button(
            toolbar, "清除", self._on_clear_console, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
            font=self._font_hint,
        ).pack(side="right")
        _styled_button(
            toolbar, "另存為檔案...", self._on_save_console, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
            font=self._font_hint,
        ).pack(side="right", padx=(0, 6))

        text_frame = tk.Frame(body, bg=COLOR_CONSOLE_BG)
        text_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._console_font_size = CONSOLE_DEFAULT_FONT_SIZE
        # height=1：Text 元件不指定 height 時預設請求 24 行的高度，字級越大這個「預設
        # 請求」就越誇張（20pt 時逼近 780px），會讓 pack 版面協商誤以為這個元件的最小
        # 需求就是那麼高，擠壓到排在它後面的「輸入」列（見 _on_send_stdin 那一列）幾乎
        # 沒有空間、被壓成 1px 看不見。設成 height=1 只是蓋掉這個預設請求值，實際渲染
        # 時仍會被 pack(fill="both", expand=True) 撐滿可用空間，跟先前修 Canvas
        # 預設地板過大是同一類問題、同一種修法。
        self._console_text = tk.Text(
            text_frame, bg=COLOR_CONSOLE_BG, fg=COLOR_CONSOLE_FG, insertbackground=COLOR_CONSOLE_FG,
            font=(CONSOLE_FONT_FAMILY, self._console_font_size), wrap="word", state="disabled",
            relief="flat", padx=6, pady=4, height=1,
        )
        console_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self._console_text.yview)
        self._console_text.configure(yscrollcommand=console_scroll.set)
        self._console_text.pack(side="left", fill="both", expand=True)
        console_scroll.pack(side="right", fill="y")
        self._console_text.tag_configure("muted", foreground=COLOR_CONSOLE_MUTED_FG)

        # Ctrl+/- 縮放終端機文字大小，原理同 VS Code：用 bind_all（不是只綁在
        # self._console_text 上）讓快捷鍵不管目前焦點在視窗裡哪個元件都會生效，不用
        # 特地點進輸出區才能用。"+" 鍵在大多數鍵盤佈局要按 Shift，實際收到的事件是
        # <Control-plus>；沒按 Shift 直接按實體 "=" 鍵送出的是 <Control-equal>，
        # 兩種都綁，數字鍵盤的 +/- 也一併綁（<Control-KP_Add>/<Control-KP_Subtract>）。
        # Ctrl+0／Ctrl+數字鍵盤0 重設回預設字級，同樣比照 VS Code 的習慣。
        for seq in ("<Control-plus>", "<Control-equal>", "<Control-KP_Add>"):
            self.bind_all(seq, lambda _e: self._adjust_console_font_size(1))
        for seq in ("<Control-minus>", "<Control-KP_Subtract>"):
            self.bind_all(seq, lambda _e: self._adjust_console_font_size(-1))
        for seq in ("<Control-0>", "<Control-KP_0>"):
            self.bind_all(seq, lambda _e: self._reset_console_font_size())

        # 輸入列：main.py／獨立腳本工具如果跑到 input() 之類需要互動輸入的地方
        # （例如某些工具腳本開場問「請選擇執行模式 1/2」），行程會停在那裡等，光看
        # 輸出面板完全看不出來、也沒地方能回應——這一列讓使用者能直接把文字送進
        # 子行程的 stdin。沒有行程在跑時停用，避免誤按送出目標不存在的輸入。
        input_row = tk.Frame(body, bg=COLOR_CONSOLE_BG)
        input_row.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(
            input_row, text="輸入：", bg=COLOR_CONSOLE_BG, fg=COLOR_CONSOLE_FG,
            font=("Consolas", 10, "bold"),
        ).pack(side="left")
        self._stdin_var = tk.StringVar()
        self._stdin_entry = tk.Entry(
            input_row, textvariable=self._stdin_var, font=("Consolas", 10),
            bg="#2a2d31", fg=COLOR_CONSOLE_FG, insertbackground=COLOR_CONSOLE_FG,
            relief="flat", disabledbackground="#242628", state="disabled",
        )
        self._stdin_entry.pack(side="left", fill="x", expand=True, padx=(6, 6), ipady=3)
        self._stdin_entry.bind("<Return>", lambda _e: self._on_send_stdin())
        self._send_stdin_btn = _styled_button(
            input_row, "傳送", self._on_send_stdin, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE,
            font=self._font_hint,
        )
        self._send_stdin_btn.config(state="disabled")
        self._send_stdin_btn.pack(side="left")
        tk.Label(
            input_row, text="（沒有行程在跑時停用；main.py／腳本停在等待輸入時，在這裡打字後按 Enter 或「傳送」）",
            bg=COLOR_CONSOLE_BG, fg=COLOR_CONSOLE_MUTED_FG, font=self._font_hint,
        ).pack(side="left", padx=(8, 0))

        self._console_append(
            "（尚未啟動任何程式；按上方「▶ 啟動 main.py」或選好腳本後按「▶ 執行所選腳本」，輸出會即時顯示在這裡）\n",
            tag="muted",
        )
        self.after(80, self._drain_log_queue)

    def _on_send_stdin(self):
        if self._main_process is None or self._main_process.poll() is not None or self._main_process.stdin is None:
            return
        text = self._stdin_var.get()
        try:
            self._main_process.stdin.write(text + "\n")
            self._main_process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            self._console_append(f"\n[傳送輸入失敗：{e}]\n", tag="muted")
            return
        self._console_append(f"> {text}\n")
        self._console_text.see("end")  # 使用者剛互動過，不管「自動捲動」有沒有勾都捲到底比較符合直覺
        self._stdin_var.set("")

    def _console_window_height(self):
        """視窗目前的真實高度；還沒完成第一次幾何配置（量到 <=1）時退回用螢幕高度頂著。"""
        win_h = self.winfo_height()
        return win_h if win_h > 1 else self.winfo_screenheight()

    def _console_max_height(self):
        """拖拉能撐到的最大高度：貼齊視窗底部往上量，最多到「標題列」下緣為止——
        終端機面板是浮動疊層（place()，見 _build_middle_area），不是跟表單搶版面，
        所以能一路蓋過流程列／獨立腳本工具列／資訊列／分頁按鈕列／表單內容／底部
        按鈕列，唯獨標題本身（貓咪監測系統 — 設定管理）永遠留在最上面看得到。
        header 高度在視窗還沒完成第一次幾何配置前可能量到 0，此時退回用
        CONSOLE_MAX_HEIGHT_RESERVE 這個粗估值頂著，避免除出負數或撞到下限。"""
        win_h = self._console_window_height()
        header_h = getattr(self, "_header_frame", None)
        header_h = header_h.winfo_height() if header_h is not None else 0
        reserve = header_h if header_h > 0 else CONSOLE_MAX_HEIGHT_RESERVE
        return max(CONSOLE_MIN_HEIGHT, win_h - reserve)

    def _apply_sane_console_default_height(self):
        """視窗剛建好、量得到真實幾何尺寸時呼叫一次：先把「展開時要多高」定案成視窗
        高度的 CONSOLE_DEFAULT_HEIGHT_FRACTION（預設 75%），但畫面上一開始是收合
        狀態——只留標題那一條水平線，不會一開機就蓋住表單/按鈕，使用者要看終端機
        輸出時自己點箭頭展開，展開後直接是這個算好的 75% 高度，不用手動拖。用
        「視窗高度的比例」而不是寫死的像素值，是因為不同螢幕解析度／系統 DPI 縮放
        下，視窗實際可用的垂直空間差異很大，寫死像素值在小螢幕上可能整個蓋過頭、
        在大螢幕上又顯得太小，比例才會等比縮放。"""
        win_h = self._console_window_height()
        header_h = getattr(self, "_header_frame", None)
        if header_h is None or header_h.winfo_height() <= 0:
            return  # 幾何配置還不可信，維持建構時給的 CONSOLE_DEFAULT_HEIGHT，不冒然套用
        target = round(win_h * CONSOLE_DEFAULT_HEIGHT_FRACTION)
        sane = max(CONSOLE_MIN_HEIGHT, min(target, self._console_max_height()))
        self._place_console(sane)
        self.update_idletasks()  # 讓下面 _toggle_console_collapse() 收合時，能正確量到 sane 這個高度存起來
        if not self._console_collapsed:
            self._toggle_console_collapse()

    def _on_console_drag_start(self, event):
        if self._console_collapsed:
            return
        self._console_drag_start_y = event.y_root
        self._console_drag_start_height = self._console_container.winfo_height()

    def _on_console_drag_motion(self, event):
        if self._console_collapsed:
            return
        delta = self._console_drag_start_y - event.y_root  # 往上拖＝正值＝面板變高
        new_height = self._console_drag_start_height + delta
        new_height = max(CONSOLE_MIN_HEIGHT, min(new_height, self._console_max_height()))
        self._place_console(new_height)
        self._console_expanded_height = new_height

    def _toggle_console_collapse(self):
        """收合：記住目前高度，只留 grip 上方那條 header 水平線（箭頭仍看得到）。
        展開：用 before=/after= 把 grip、body 依原本順序插回 header 上下兩側，
        並還原收合前的高度。"""
        self._console_collapsed = not self._console_collapsed
        if self._console_collapsed:
            self._console_expanded_height = self._console_container.winfo_height()
            self._console_grip.pack_forget()
            self._console_body.pack_forget()
            self._place_console(CONSOLE_COLLAPSED_HEIGHT)
            self._console_toggle_btn.config(text="▲")
        else:
            self._console_grip.pack(fill="x", side="top", before=self._console_header)
            self._console_body.pack(fill="both", expand=True, side="top")
            self._place_console(self._console_expanded_height)
            self._console_toggle_btn.config(text="▼")

    def _console_append(self, text, tag=None):
        self._console_text.configure(state="normal")
        self._console_text.insert("end", text, (tag,) if tag else ())
        # 長時間跑下來輸出量可能很大，Text 內容超過上限就砍掉前面舊的部分，
        # 避免記憶體無限增長——保留「最新」的訊息比保留最舊的更有用。
        self._log_line_count += text.count("\n")
        if self._log_line_count > 5000:
            self._console_text.delete("1.0", f"{self._log_line_count - 4000}.0")
            self._log_line_count = 4000
        if self._autoscroll_var.get():
            self._console_text.see("end")
        self._console_text.configure(state="disabled")

    def _on_clear_console(self):
        self._console_text.configure(state="normal")
        self._console_text.delete("1.0", "end")
        self._console_text.configure(state="disabled")
        self._log_line_count = 0

    def _adjust_console_font_size(self, delta):
        new_size = max(CONSOLE_MIN_FONT_SIZE, min(CONSOLE_MAX_FONT_SIZE, self._console_font_size + delta))
        if new_size == self._console_font_size:
            return
        self._console_font_size = new_size
        self._console_text.configure(font=(CONSOLE_FONT_FAMILY, self._console_font_size))

    def _reset_console_font_size(self):
        if self._console_font_size == CONSOLE_DEFAULT_FONT_SIZE:
            return
        self._console_font_size = CONSOLE_DEFAULT_FONT_SIZE
        self._console_text.configure(font=(CONSOLE_FONT_FAMILY, self._console_font_size))

    def _on_save_console(self):
        path = filedialog.asksaveasfilename(
            title="另存主控台輸出", defaultextension=".txt",
            initialfile="main_py_console.txt", filetypes=[("文字檔", "*.txt"), ("所有檔案", "*.*")],
        )
        if not path:
            return
        try:
            Path(path).write_text(self._console_text.get("1.0", "end"), encoding="utf-8")
        except OSError as e:
            messagebox.showerror("另存主控台輸出", f"儲存失敗：{e}")
            return
        messagebox.showinfo("另存主控台輸出", f"已儲存到：\n{path}")

    def _start_log_reader(self, process):
        """背景執行緒逐行讀取子行程的 stdout（stderr 已合併進去），讀到 EOF
        （行程結束、pipe 關閉）就丟一個 None 進 queue 當結束訊號。"""

        def _reader():
            try:
                for line in iter(process.stdout.readline, ""):
                    self._log_queue.put(line)
            except (OSError, ValueError):
                pass
            finally:
                self._log_queue.put(None)

        self._log_reader_thread = threading.Thread(target=_reader, daemon=True)
        self._log_reader_thread.start()

    def _bring_child_window_to_front(self, process):
        """背景執行緒輪詢，等子行程（main.py／獨立腳本）自己開出的 GUI 視窗出現後
        自動拉到最上層、取得焦點，不用手動 Alt+Tab 切過去。

        這些腳本大多要先載入 YOLO/ST-GCN 模型才會真的開窗，時間不固定（幾秒到
        十幾秒），所以用輪詢而不是啟動後立刻找一次。找視窗只認「屬於這個 PID
        的可見頂層視窗」，不比對標題文字——各腳本視窗標題五花八門（tkinter 預設
        標題、cv2.imshow 的檔名...），比對 PID 才不用每支腳本都額外維護一份標題
        清單，且對之後新增的腳本自動適用。

        需要 pywin32（win32gui/win32process/win32con）；環境沒裝的話直接跳過，
        不影響腳本本身執行、也不彈錯誤訊息——這只是「省得手動切視窗」的錦上添花
        功能，缺了頂多要自己切一下，不該讓整個啟動流程失敗。
        """
        try:
            import win32api
            import win32con
            import win32gui
            import win32process
        except ImportError:
            return

        def _worker():
            deadline = time.time() + 20.0  # 模型載入可能要幾秒到十幾秒，給足時間再放棄
            target_hwnd = None
            while time.time() < deadline:
                if process.poll() is not None:
                    return  # 行程提早結束（例如啟動失敗），沒視窗好找
                found = []

                def _enum_handler(hwnd, _extra):
                    if not win32gui.IsWindowVisible(hwnd):
                        return
                    if win32gui.GetParent(hwnd) != 0:
                        return  # 只找頂層視窗，排除子控制項
                    if not win32gui.GetWindowText(hwnd):
                        return  # 排除沒標題的隱藏輔助視窗
                    _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                    if found_pid == process.pid:
                        found.append(hwnd)

                try:
                    win32gui.EnumWindows(_enum_handler, None)
                except Exception:
                    pass
                if found:
                    target_hwnd = found[0]
                    break
                time.sleep(0.3)

            if target_hwnd is None:
                return

            try:
                if win32gui.IsIconic(target_hwnd):
                    win32gui.ShowWindow(target_hwnd, win32con.SW_RESTORE)
                # Windows 預設會擋「非目前使用中程式」呼叫 SetForegroundWindow（避免
                # 程式亂搶焦點）。這裡用常見的繞法：先把自己這條執行緒的輸入佇列跟
                # 目前前景視窗的執行緒 attach 在一起，讓系統把接下來的
                # SetForegroundWindow 視為「使用者自己切換」而放行，做完再 detach。
                fg_hwnd = win32gui.GetForegroundWindow()
                current_thread_id = win32api.GetCurrentThreadId()
                fg_thread_id = win32process.GetWindowThreadProcessId(fg_hwnd)[0] if fg_hwnd else 0
                attached = False
                if fg_thread_id and fg_thread_id != current_thread_id:
                    attached = win32process.AttachThreadInput(current_thread_id, fg_thread_id, True)
                try:
                    win32gui.SetForegroundWindow(target_hwnd)
                finally:
                    if attached:
                        win32process.AttachThreadInput(current_thread_id, fg_thread_id, False)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _drain_log_queue(self):
        drained = 0
        try:
            while drained < 500:  # 單次 tick 最多處理 500 行，避免瞬間大量輸出卡住 GUI 主執行緒
                line = self._log_queue.get_nowait()
                if line is None:
                    self._console_append(f"\n— {self._active_process_label} 行程已結束 —\n", tag="muted")
                else:
                    self._console_append(line)
                drained += 1
        except queue.Empty:
            pass
        self.after(80, self._drain_log_queue)

    def _build_tabs(self, parent):
        """分頁列刻意不用 ttk.Notebook——它在 Windows 原生佈景主題下無法讓每個分頁
        按鈕套用不同底色。改用一排自製的 tk.Button 當分頁切換鈕，每顆都直接用
        TAB_COLORS 上色：選取中＝原色底＋白字，未選取＝同色系的淡色底＋原色字
        （用 _lighten() 混白計算），兩種狀態都看得出屬於哪個分頁。
        """
        fields_by_tab = {}
        for field in FIELD_SCHEMA:
            fields_by_tab.setdefault(field["tab"], []).append(field)

        # 分頁按鈕列改成可橫向捲動：分頁一多，按鈕排成一列會超出視窗寬度，用
        # Canvas + 橫向 ttk.Scrollbar 包住，滑桿可以拉、滑鼠停在按鈕列上滾輪也能橫向捲。
        tab_bar_outer = tk.Frame(parent, bg=COLOR_BG_MAIN)
        tab_bar_outer.pack(fill="x", padx=16, pady=(0, 4))
        self._tab_bar_outer = tab_bar_outer

        tab_canvas = tk.Canvas(tab_bar_outer, bg=COLOR_BG_MAIN, highlightthickness=0)
        tab_hscroll = ttk.Scrollbar(tab_bar_outer, orient="horizontal", command=tab_canvas.xview)
        tab_bar = tk.Frame(tab_canvas, bg=COLOR_BG_MAIN)
        tab_bar.bind(
            "<Configure>", lambda e: tab_canvas.configure(scrollregion=tab_canvas.bbox("all"))
        )
        tab_canvas.create_window((0, 0), window=tab_bar, anchor="nw")
        tab_canvas.configure(xscrollcommand=tab_hscroll.set)
        tab_canvas.pack(side="top", fill="x")
        tab_hscroll.pack(side="top", fill="x")

        def _on_tabbar_wheel(event):
            tab_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

        tab_canvas.bind("<Enter>", lambda e: tab_canvas.bind_all("<MouseWheel>", _on_tabbar_wheel))
        tab_canvas.bind("<Leave>", lambda e: tab_canvas.unbind_all("<MouseWheel>"))

        content_area = tk.Frame(parent, bg=COLOR_TAB_BG)
        content_area.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self._content_area = content_area

        self._tab_buttons = {}
        self._tab_frames = {}
        self._tab_accents = {}

        for tab_name in TAB_ORDER:
            emoji, accent = TAB_COLORS.get(tab_name, ("⬜", COLOR_HEADER_BG))
            self._tab_accents[tab_name] = accent

            btn = tk.Button(
                tab_bar, text=f"{emoji} {tab_name}", font=self._font_tabbtn,
                bd=0, relief="flat", cursor="hand2", padx=12, pady=8,
                command=lambda t=tab_name: self._select_tab(t),
            )
            btn.pack(side="left", padx=(0, 4), pady=4)
            self._tab_buttons[tab_name] = btn

            tab = _ScrollableTab(content_area)
            self._tab_frames[tab_name] = tab

            banner = tk.Frame(tab.body, bg=accent)
            banner.pack(fill="x")
            tk.Label(
                banner, text=f"{emoji} {tab_name}", bg=accent, fg="#ffffff",
                font=self._font_banner, anchor="w",
            ).pack(fill="x", padx=14, pady=8)

            for field in fields_by_tab.get(tab_name, []):
                self._build_field_row(tab.body, field, accent)
            if tab_name == "ST-GCN 推論":
                tk.Label(
                    tab.body,
                    text="ℹ️ 訓練用參數（SEQUENCE_LENGTH／FEATURE_MODE／NUM_CLASSES）由"
                    " stgcn_config.yaml 管理，不在此設定視窗顯示或覆寫。",
                    bg=COLOR_TAB_BG, fg=COLOR_HINT_FG, font=self._font_hint,
                    anchor="w", justify="left", wraplength=1200,
                ).pack(fill="x", padx=14, pady=(6, 10))

        # Canvas 不會像 Frame 一樣自動長到內容的高度，按鈕全部排好後量出實際需要的
        # 高度再回填，分頁按鈕列才會剛好一行高，不會被裁切也不會留多餘空白。
        self.update_idletasks()
        tab_canvas.configure(height=tab_bar.winfo_reqheight())

        self._select_tab(TAB_ORDER[0])

    def _select_tab(self, tab_name):
        for name, frame in self._tab_frames.items():
            if name == tab_name:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()
        for name, btn in self._tab_buttons.items():
            accent = self._tab_accents[name]
            if name == tab_name:
                btn.config(bg=accent, fg="#ffffff", activebackground=accent, activeforeground="#ffffff")
            else:
                light = _lighten(accent, 0.72)
                btn.config(bg=light, fg=accent, activebackground=light, activeforeground=accent)

    def _build_bottom_bar(self):
        bottom = tk.Frame(self, bg=COLOR_BG_MAIN)
        bottom.pack(fill="x", padx=16, pady=(0, 12))
        self._bottom_bar_frame = bottom
        _styled_button(bottom, "載入目前設定", self._on_load_current, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE).pack(side="left")
        _styled_button(bottom, "還原 GUI 預設值", self._on_restore_defaults, BTN_WARN_BG, BTN_WARN_ACTIVE).pack(side="left", padx=(8, 0))
        _styled_button(bottom, "匯出設定", self._on_export, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE).pack(side="left", padx=(8, 0))
        _styled_button(bottom, "匯入設定", self._on_import, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE).pack(side="left", padx=(8, 0))
        _styled_button(bottom, "關閉", self._on_window_close, BTN_NEUTRAL_BG, BTN_NEUTRAL_ACTIVE).pack(side="right")
        _styled_button(bottom, "儲存設定", self._on_save, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE).pack(side="right", padx=(0, 8))

    # ── 欄位列渲染 ────────────────────────────────────────────────────

    def _build_field_row(self, parent, field, accent=COLOR_HEADER_BG):
        key = field["json_key"]
        vt = field["value_type"]
        # 布林旗標整列換底色 + 換成方形切換鈕，跟數字/字串欄位的樸素外觀拉開差異，
        # 掃視整頁時能直接認出「這是開關」；其餘欄位維持白底，只在最左側留一條
        # 該分頁代表色的細長色條，捲動時仍能辨認目前在哪個分頁。
        row_bg = FLAG_ROW_BG if vt == "bool" else COLOR_TAB_BG

        container = tk.Frame(parent, bg=COLOR_TAB_BG)
        container.pack(fill="x", pady=(5, 0))
        tk.Frame(container, bg=accent, width=4).pack(side="left", fill="y")

        row = tk.Frame(container, bg=row_bg)
        row.pack(side="left", fill="both", expand=True, padx=(10, 14))

        tk.Label(
            row, text=field["label"], bg=row_bg, fg=COLOR_LABEL_FG,
            font=self._font_label_bold if vt == "bool" else self._font_label,
            anchor="nw", justify="left", width=30,
        ).pack(side="left", anchor="n")

        control = tk.Frame(row, bg=row_bg)
        control.pack(side="left", fill="x", expand=True, padx=(8, 8))

        badge_var = tk.StringVar(value="")
        badge = tk.Label(row, textvariable=badge_var, font=self._font_hint, padx=6, pady=2)
        badge.pack(side="right", anchor="n")

        info = {"field": field, "badge_var": badge_var, "badge_widget": badge}

        if vt == "bool":
            var = tk.BooleanVar()
            toggle_btn = tk.Checkbutton(
                control, variable=var, indicatoron=False, onvalue=True, offvalue=False,
                font=self._font_label_bold, width=9, bd=0, relief="flat", cursor="hand2",
                highlightthickness=0, padx=8, pady=4,
            )
            toggle_btn.pack(side="left")

            def _refresh_toggle_look(*_a, btn=toggle_btn, v=var):
                if v.get():
                    btn.config(text="✅ 開啟", bg=FLAG_ON_BG, fg=FLAG_ON_FG,
                               activebackground=FLAG_ON_BG, activeforeground=FLAG_ON_FG, selectcolor=FLAG_ON_BG)
                else:
                    btn.config(text="⬜ 關閉", bg=FLAG_OFF_BG, fg=FLAG_OFF_FG,
                               activebackground=FLAG_OFF_BG, activeforeground=FLAG_OFF_FG, selectcolor=FLAG_OFF_BG)

            var.trace_add("write", _refresh_toggle_look)
            _refresh_toggle_look()
            info["var"] = var
        elif vt in ("int", "float", "str"):
            var = tk.StringVar()
            tk.Entry(control, textvariable=var, font=self._font_label).pack(side="left", fill="x", expand=True)
            info["var"] = var
        elif vt == "hhmm":
            var = tk.StringVar()
            tk.Entry(control, textvariable=var, font=self._font_label, width=10).pack(side="left")
            tk.Label(control, text="  例：06:00，留空代表不啟用", bg=COLOR_TAB_BG, fg=COLOR_HINT_FG, font=self._font_hint).pack(side="left")
            info["var"] = var
        elif vt == "enum":
            var = tk.StringVar()
            ttk.Combobox(control, textvariable=var, values=field.get("choices", []), state="readonly", width=18).pack(side="left")
            info["var"] = var
        elif vt == "file":
            var = tk.StringVar()
            tk.Entry(control, textvariable=var, font=self._font_label).pack(side="left", fill="x", expand=True)
            browse_filter = field.get("browse_filter")

            def _browse(v=var, bf=browse_filter):
                filetypes = [bf, ("所有檔案", "*.*")] if bf else [("所有檔案", "*.*")]
                path = filedialog.askopenfilename(title="選擇檔案", filetypes=filetypes)
                if path:
                    v.set(path)

            _styled_button(control, "瀏覽...", _browse, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, font=self._font_hint).pack(side="left", padx=(6, 0))
            info["var"] = var
        elif vt == "folder":
            var = tk.StringVar()
            tk.Entry(control, textvariable=var, font=self._font_label).pack(side="left", fill="x", expand=True)

            def _browse(v=var):
                path = filedialog.askdirectory(title="選擇資料夾")
                if path:
                    v.set(path)

            _styled_button(control, "選擇資料夾...", _browse, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, font=self._font_hint).pack(side="left", padx=(6, 0))
            info["var"] = var
        elif vt == "video_input":
            # 影像來源本質上是三種完全不同格式的東西（本機路徑／攝影機 index／URL），
            # 硬塞進同一個 Entry 只能靠「使用者自己記得格式」手動切換，容易打錯也不
            # 直觀。改成「模式選擇（Radiobutton）＋依模式顯示對應控制項」：切換模式時
            # 舊模式的值還留著（不會被清掉），方便在幾種來源之間來回試而不用重打。
            mode_var = tk.StringVar(value="file")
            path_var = tk.StringVar()
            camera_var = tk.StringVar(value="0")
            url_var = tk.StringVar()

            mode_row = tk.Frame(control, bg=row_bg)
            mode_row.pack(side="top", fill="x")
            for mode_value, mode_label in (
                ("file", "📁 本機影片檔案"), ("camera", "📷 攝影機索引"), ("url", "🌐 RTSP/HTTP 串流網址"),
            ):
                tk.Radiobutton(
                    mode_row, text=mode_label, variable=mode_var, value=mode_value,
                    bg=row_bg, activebackground=row_bg, font=self._font_hint,
                    command=lambda: info["apply_video_mode"](),
                ).pack(side="left", padx=(0, 12))

            sub_row = tk.Frame(control, bg=row_bg)
            sub_row.pack(side="top", fill="x", pady=(4, 0))

            file_row = tk.Frame(sub_row, bg=row_bg)
            tk.Entry(file_row, textvariable=path_var, font=self._font_label).pack(
                side="left", fill="x", expand=True
            )

            def _browse_video(v=path_var):
                path = filedialog.askopenfilename(
                    title="選擇影片檔案",
                    filetypes=[
                        ("影片檔案", "*.mp4;*.avi;*.mov;*.mkv;*.wmv;*.flv"),
                        ("所有檔案", "*.*"),
                    ],
                )
                if path:
                    v.set(path)

            _styled_button(
                file_row, "瀏覽...", _browse_video, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
                font=self._font_hint,
            ).pack(side="left", padx=(6, 0))

            camera_row = tk.Frame(sub_row, bg=row_bg)
            tk.Label(camera_row, text="攝影機索引：", bg=row_bg, fg=COLOR_LABEL_FG, font=self._font_hint).pack(side="left")
            tk.Spinbox(
                camera_row, from_=0, to=20, textvariable=camera_var, width=5, font=self._font_label,
            ).pack(side="left")
            tk.Label(
                camera_row, text="（USB/內建攝影機通常從 0 開始；多台裝置依序 1、2…）",
                bg=row_bg, fg=COLOR_HINT_FG, font=self._font_hint,
            ).pack(side="left", padx=(8, 0))

            url_row = tk.Frame(sub_row, bg=row_bg)
            tk.Entry(url_row, textvariable=url_var, font=self._font_label).pack(
                side="left", fill="x", expand=True
            )

            info["var_by_mode"] = {"file": path_var, "camera": camera_var, "url": url_var}
            info["mode_var"] = mode_var

            def _apply_video_mode(*_a):
                for r in (file_row, camera_row, url_row):
                    r.pack_forget()
                {"file": file_row, "camera": camera_row, "url": url_row}[mode_var.get()].pack(fill="x")
                _refresh_video_hint()

            hint_var = tk.StringVar(value="")
            info["detect_var"] = hint_var

            def _refresh_video_hint(*_a):
                mode = mode_var.get()
                if mode == "file":
                    hint_var.set("將以本機影片檔案路徑啟動，例：C:\\videos\\cat.mp4")
                elif mode == "camera":
                    hint_var.set(f"將以攝影機索引 {camera_var.get() or 0} 啟動（cv2.VideoCapture({camera_var.get() or 0})）")
                else:
                    hint_var.set(
                        "例：rtsp://使用者:密碼@192.168.0.192:554/stream1（IP Cam/RTSP）　或　"
                        "http://192.168.0.50:81/stream（ESP32-CAM 等 MJPEG 串流）"
                    )

            info["apply_video_mode"] = _apply_video_mode
            camera_var.trace_add("write", _refresh_video_hint)
            _apply_video_mode()

            tk.Label(
                parent, textvariable=hint_var, bg=COLOR_TAB_BG, fg=COLOR_HINT_FG,
                font=self._font_hint, anchor="w", justify="left", wraplength=1200,
            ).pack(fill="x", padx=14, pady=(2, 0))
        elif vt == "size":
            enabled_var = tk.BooleanVar()
            width_var = tk.StringVar()
            height_var = tk.StringVar()
            w_entry = tk.Entry(control, textvariable=width_var, font=self._font_label, width=6)
            h_entry = tk.Entry(control, textvariable=height_var, font=self._font_label, width=6)

            def _toggle():
                state = "disabled" if enabled_var.get() else "normal"
                w_entry.config(state=state)
                h_entry.config(state=state)

            tk.Checkbutton(
                control, text="維持原始解析度", variable=enabled_var, bg=COLOR_TAB_BG,
                activebackground=COLOR_TAB_BG, command=_toggle,
            ).pack(side="left")
            tk.Label(control, text="  寬:", bg=COLOR_TAB_BG, font=self._font_hint).pack(side="left")
            w_entry.pack(side="left")
            tk.Label(control, text=" 高:", bg=COLOR_TAB_BG, font=self._font_hint).pack(side="left")
            h_entry.pack(side="left")
            info["enabled_var"] = enabled_var
            info["width_var"] = width_var
            info["height_var"] = height_var
            info["toggle"] = _toggle

        env_note_var = tk.StringVar(value="")
        tk.Label(
            parent, textvariable=env_note_var, bg=COLOR_TAB_BG, fg=COLOR_WARNING_FG,
            font=self._font_hint, anchor="w", justify="left", wraplength=880,
        ).pack(fill="x", padx=14, pady=(0, 6))
        info["env_note_var"] = env_note_var

        self._field_widgets[key] = info

    # ── 讀取／寫入表單值 ─────────────────────────────────────────────

    def _set_field_value(self, key, value):
        info = self._field_widgets[key]
        vt = info["field"]["value_type"]
        if vt == "bool":
            info["var"].set(bool(value) if value is not None else False)
        elif vt == "video_input":
            # 依值的型別/格式自動判斷該切到哪個模式：int → 攝影機索引；
            # rtsp/http(s):// 開頭 → URL；其餘一律當成本機路徑。三個模式各自的
            # StringVar 都會寫入對應值，即使目前顯示的不是那個模式，之後切換
            # 過去時舊值還在，不會憑空消失。
            if isinstance(value, int):
                info["mode_var"].set("camera")
                info["var_by_mode"]["camera"].set(str(value))
            elif isinstance(value, str) and value.lower().startswith(("rtsp://", "http://", "https://")):
                info["mode_var"].set("url")
                info["var_by_mode"]["url"].set(value)
            else:
                info["mode_var"].set("file")
                info["var_by_mode"]["file"].set("" if value is None else str(value))
            info["apply_video_mode"]()
        elif vt == "size":
            if value is None:
                info["enabled_var"].set(True)
                info["width_var"].set("")
                info["height_var"].set("")
            elif isinstance(value, dict):
                info["enabled_var"].set(False)
                info["width_var"].set(str(value.get("width", "")))
                info["height_var"].set(str(value.get("height", "")))
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                info["enabled_var"].set(False)
                info["width_var"].set(str(value[0]))
                info["height_var"].set(str(value[1]))
            else:
                info["enabled_var"].set(True)
                info["width_var"].set("")
                info["height_var"].set("")
            info["toggle"]()
        else:
            info["var"].set("" if value is None else str(value))

    def _get_field_value(self, key):
        """回傳 (value, error_or_None)：把表單目前輸入轉成 JSON 可用型別。"""
        info = self._field_widgets[key]
        field = info["field"]
        vt = field["value_type"]
        label = field["label"]
        if vt == "bool":
            return info["var"].get(), None
        if vt == "int":
            text = info["var"].get().strip()
            try:
                return int(text), None
            except ValueError:
                return None, f"{label}：必須是整數"
        if vt == "float":
            text = info["var"].get().strip()
            try:
                return float(text), None
            except ValueError:
                return None, f"{label}：必須是數字"
        if vt in ("str", "file", "folder", "enum"):
            return info["var"].get(), None
        if vt == "hhmm":
            return info["var"].get().strip(), None
        if vt == "video_input":
            mode = info["mode_var"].get()
            if mode == "camera":
                text = info["var_by_mode"]["camera"].get().strip()
                try:
                    return int(text), None
                except ValueError:
                    return None, f"{label}：攝影機索引必須是整數"
            if mode == "url":
                url = info["var_by_mode"]["url"].get().strip()
                if url == "":
                    return None, f"{label}：RTSP/HTTP 網址不可為空"
                return url, None
            path = info["var_by_mode"]["file"].get().strip()
            if path == "":
                return None, f"{label}：本機影片路徑不可為空"
            return path, None
        if vt == "size":
            if info["enabled_var"].get():
                return None, None
            w_text = info["width_var"].get().strip()
            h_text = info["height_var"].get().strip()
            try:
                return {"width": int(w_text), "height": int(h_text)}, None
            except ValueError:
                return None, f"{label}：寬高必須是整數"
        return None, None

    def _collect_form_data(self):
        data = {}
        errors = []
        for field in FIELD_SCHEMA:
            value, err = self._get_field_value(field["json_key"])
            if err:
                errors.append(err)
                continue
            _set_nested(data, field["json_key"], value)
        return data, errors

    def _resolve_field_display(self, field):
        """回傳 (value, source)：source ∈ {"env","json","default"}。"""
        key = field["json_key"]
        env_var = field["env_var"]
        if env_var and os.getenv(env_var) is not None:
            return _parse_env_value(env_var, field["value_type"]), "env"
        json_data = settings_manager.load_runtime_settings()
        raw = _get_nested(json_data, key)
        if raw is not _MISSING:
            return raw, "json"
        return self._baseline_effective.get(key), "default"

    def _apply_source(self, key, source):
        info = self._field_widgets[key]
        field = info["field"]
        badge_var = info["badge_var"]
        badge_widget = info["badge_widget"]
        if source == "env":
            badge_var.set("[環境變數]")
            badge_widget.config(bg=BADGE_ENV_BG, fg=BADGE_ENV_FG)
            info["env_note_var"].set(
                f"⚠ 此欄位目前受環境變數 {field['env_var']} 控制；儲存 JSON 不會立即生效，"
                "需先取消該環境變數才會改用 runtime_settings.current.json 的值。"
            )
        elif source == "json":
            badge_var.set("[JSON]")
            badge_widget.config(bg=BADGE_JSON_BG, fg=BADGE_JSON_FG)
            info["env_note_var"].set("")
        elif source == "form":
            badge_var.set("[表單暫存，尚未儲存]")
            badge_widget.config(bg=BADGE_FORM_BG, fg=BADGE_FORM_FG)
            info["env_note_var"].set("")
        else:
            badge_var.set("[預設值]")
            badge_widget.config(bg=BADGE_DEFAULT_BG, fg=BADGE_DEFAULT_FG)
            info["env_note_var"].set("")

    def _populate_from_effective_state(self):
        for field in FIELD_SCHEMA:
            key = field["json_key"]
            value, source = self._resolve_field_display(field)
            self._set_field_value(key, value)
            self._apply_source(key, source)

    def _populate_form(self, data_dict, source_label="form"):
        for field in FIELD_SCHEMA:
            key = field["json_key"]
            value = _get_nested(data_dict, key)
            if value is _MISSING:
                continue
            self._set_field_value(key, value)
            self._apply_source(key, source_label)

    # ── 按鈕事件 ─────────────────────────────────────────────────────

    def _on_load_current(self):
        settings_manager.reload_runtime_settings()
        self._populate_from_effective_state()
        self._refresh_top_info()
        messagebox.showinfo("載入目前設定", "已重新讀取環境變數／runtime_settings.current.json／內建預設值。")

    def _on_save(self):
        data, errors = self._collect_form_data()
        if errors:
            messagebox.showerror("儲存設定", "以下欄位輸入格式有誤，請修正後再試：\n\n" + "\n".join(errors))
            return
        ok, errs, warnings = settings_manager.save_runtime_settings(data)
        if not ok:
            messagebox.showerror("儲存設定", "驗證未通過，設定未儲存：\n\n" + "\n".join(errs))
            return
        msg = "設定已儲存；重新啟動主程式後生效。"
        if warnings:
            msg += "\n\n提醒：\n" + "\n".join(warnings)
        messagebox.showinfo("儲存設定", msg)
        self._refresh_top_info()
        self._populate_from_effective_state()

    def _on_restore_defaults(self):
        if not messagebox.askyesno(
            "還原 GUI 預設值",
            "將表單所有可管理欄位還原為內建預設值（不含 ST-GCN 訓練設定，本來就不在此設定視窗管理），"
            "是否繼續？\n\n此動作僅套用到表單，仍需按「儲存設定」才會寫入 runtime_settings.current.json。",
        ):
            return
        defaults = settings_manager.restore_defaults()
        if not defaults:
            messagebox.showerror("還原 GUI 預設值", "default_runtime_settings.json 讀取失敗或內容為空。")
            return
        self._populate_form(defaults, source_label="form")
        messagebox.showinfo("還原 GUI 預設值", "已載入 GUI 預設值到表單，請按「儲存設定」以正式套用。")

    def _on_export(self):
        data, errors = self._collect_form_data()
        if errors:
            messagebox.showerror("匯出設定", "以下欄位輸入格式有誤，請修正後再試：\n\n" + "\n".join(errors))
            return
        path = filedialog.asksaveasfilename(
            title="匯出設定", defaultextension=".json",
            initialfile="runtime_settings_export.json", filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        ok, errs, warnings = settings_manager.export_settings(data, path)
        if not ok:
            messagebox.showerror("匯出設定", "驗證未通過，未匯出：\n\n" + "\n".join(errs))
            return
        msg = f"已匯出到：\n{path}"
        if warnings:
            msg += "\n\n提醒：\n" + "\n".join(warnings)
        messagebox.showinfo("匯出設定", msg)

    def _on_import(self):
        path = filedialog.askopenfilename(title="匯入設定", filetypes=[("JSON", "*.json")])
        if not path:
            return
        ok, data, errors, warnings = settings_manager.import_settings(path)
        if errors:
            messagebox.showerror("匯入設定", "檔案驗證失敗，未套用：\n\n" + "\n".join(str(e) for e in errors))
            return
        current, _ = self._collect_form_data()
        changes = settings_manager.diff_settings(current, data)
        if not changes:
            messagebox.showinfo("匯入設定", "與目前表單內容相同，沒有變更。")
            return
        if self._show_diff_dialog(changes):
            self._populate_form(data, source_label="form")
            messagebox.showinfo("匯入設定", "已套用到表單，請按「儲存設定」以正式寫入 runtime_settings.current.json。")

    def _show_diff_dialog(self, changes) -> bool:
        dialog = tk.Toplevel(self)
        dialog.title("匯入設定變更預覽")
        dialog.geometry("860x580")
        dialog.configure(bg=COLOR_BG_MAIN)
        dialog.transient(self)
        dialog.grab_set()

        tk.Label(
            dialog, text=f"共 {len(changes)} 項欄位將被變更：", bg=COLOR_BG_MAIN,
            font=self._font_label_bold, anchor="w",
        ).pack(fill="x", padx=14, pady=(12, 4))

        text_frame = tk.Frame(dialog, bg=COLOR_BG_MAIN)
        text_frame.pack(fill="both", expand=True, padx=14, pady=4)
        text = tk.Text(text_frame, font=self._font_info, wrap="word")
        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for _key, label, old_v, new_v in changes:
            text.insert("end", f"{label}\n    舊：{_redact(old_v)}\n    新：{_redact(new_v)}\n\n")
        text.configure(state="disabled")

        result = {"apply": False}
        btn_row = tk.Frame(dialog, bg=COLOR_BG_MAIN)
        btn_row.pack(fill="x", padx=14, pady=(4, 12))

        def _apply():
            result["apply"] = True
            dialog.destroy()

        _styled_button(btn_row, "取消", dialog.destroy, BTN_NEUTRAL_BG, BTN_NEUTRAL_ACTIVE).pack(side="right")
        _styled_button(btn_row, "套用到表單", _apply, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE).pack(side="right", padx=(0, 8))

        self.wait_window(dialog)
        return result["apply"]


if __name__ == "__main__":
    SettingsWindow().mainloop()
