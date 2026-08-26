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
import re
import shutil
import subprocess
import sys
from datetime import datetime
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
from settings_gui.console_panel import ConsolePanel  # noqa: E402
from settings_gui.process_manager import ProcessManager  # noqa: E402
from settings_gui.field_search import FieldSearchBar  # noqa: E402
from settings_gui import tab_docs_panel  # noqa: E402
from settings_gui.style import (  # noqa: E402
    BTN_PRIMARY_BG,
    BTN_PRIMARY_ACTIVE,
    BTN_SECONDARY_BG,
    BTN_SECONDARY_ACTIVE,
    BTN_INFO_BG,
    BTN_INFO_ACTIVE,
    BTN_INFO_FG,
    BTN_WARN_FG,
    COLOR_CONSOLE_BG,
    COLOR_HEADER_BG,
    COLOR_HEADER_FG,
    COLOR_TOOL_DESC_BG,
    COLOR_TOOL_DESC_BORDER,
    COLOR_TOOL_DESC_ACCENT,
    COLOR_TOOL_DESC_FG,
    CONSOLE_DEFAULT_HEIGHT,
    CONSOLE_FONT_FAMILY,
    SPACE_XS,
    SPACE_SM,
    SPACE_MD,
    SPACE_LG,
    _display_width,
    _styled_button,
)

# ── 視覺樣式：沿用 analytics/manage_baseline_history.py 的同一組常數 ─────────
# （終端機面板／子行程管理相關的常數與 _styled_button 已搬到 settings_gui/style.py，
#  上面用 import 拿回來；以下留著的是只有本檔案自己會用到的樣式常數。）

_FONT_FAMILY = "Microsoft JhengHei"

COLOR_BG_MAIN = "#eef2f6"
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

BTN_WARN_BG = "#e67e22"
BTN_WARN_ACTIVE = "#cf711d"
# 原本這裡還有一組 BTN_NEUTRAL_*（純灰）給「取消」「關閉」這類收尾動作用，跟
# BTN_SECONDARY_*（藍灰）語意上重疊，2026-08 版面美化時合併掉了——那兩個按鈕
# 現在改用 BTN_SECONDARY_BG 但加 `outline=True`（見 settings_gui/style.py 的
# _styled_button），用「有沒有實心填色」而不是另開一個新色相來分辨重要性。

BADGE_ENV_BG = "#f5b7b1"
BADGE_ENV_FG = "#78281f"
BADGE_JSON_BG = "#a9cce3"
BADGE_JSON_FG = "#1b4f72"
BADGE_DEFAULT_BG = "#d5dbdb"
BADGE_DEFAULT_FG = "#4d5656"
BADGE_FORM_BG = "#f9e79f"
BADGE_FORM_FG = "#7d6608"

COLOR_TOOL_LISTBOX_BG = "#eaf4fc"  # 獨立腳本工具下拉選單展開後的淡藍色底，掃視一長串腳本名稱更輕鬆
# COLOR_TOOL_DESC_*（常駐說明卡片配色）搬到 settings_gui/style.py 了——分頁右欄的
# tab_docs_panel.py 也要用同一組配色，放在共用模組才不會 import 回這支檔案造成循環。

# 布林旗標（開關型設定）的視覺樣式——刻意跟數字/字串欄位的樸素外觀拉開差異，
# 讓「這是一個開關」在掃視整頁時能立刻被認出來，不用逐行讀 label 文字。
FLAG_ON_BG = "#27ae60"
FLAG_ON_FG = "#ffffff"
FLAG_OFF_BG = "#aeb6bf"
FLAG_OFF_FG = "#2c3e50"
FLAG_ROW_BG = "#f4faf6"  # 布林欄位那一整列的底色，跟其他欄位列的白底做出區隔

# 全域欄位搜尋命中時，欄位列的高亮外框色——跟其餘配色（藍/綠/橘/紅系）都不撞，
# 一眼認出「這是搜尋結果」，不會誤認成布林開關或徽章的既有配色語意。
SEARCH_HIGHLIGHT_BORDER = "#e91e8c"

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


def _lighten(hex_color: str, factor: float) -> str:
    """把顏色往白色混合 factor 比例（0~1，越大越淡）；分頁未選取時用淡版底色，
    選取時用原色，兩種狀態都看得出屬於哪個分頁、又能分辨目前選到哪一個。"""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


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
        self.canvas = canvas  # 存成屬性，讓外部（例如欄位搜尋跳轉後的捲動定位）能直接操作
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
        self._apply_ttk_style()

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
        # main.py 跟下面新增的「獨立腳本工具」共用同一套啟動/關閉/終端機輸出機制
        # （ProcessManager，見 settings_gui/process_manager.py），因為同一時間本來就
        # 只允許跑一個（避免兩支行程搶同一份 runtime_settings.current.json／模型顯存，
        # 也讓終端機輸出不會兩邊來源混在一起分不清楚）。on_state_change 掛
        # _update_process_buttons_state，讓 ProcessManager 每次狀態改變（啟動/關閉/
        # 輪詢發現行程已死）都會回頭更新這裡的按鈕與狀態列文字。ProcessManager 建構
        # 時 console（ConsolePanel）還沒蓋出來（_build_middle_area() 比 _build_process_bar()
        # 晚），所以 .console 是事後在 _build_middle_area() 裡才指派。
        self._process_manager = ProcessManager(self, on_state_change=self._update_process_buttons_state)
        self._tool_script_var = tk.StringVar(value="")  # 「獨立腳本工具」下拉/瀏覽選中的 .py 路徑
        # 關閉本視窗（不管是按右上角 X 還是下方「關閉」按鈕）視同 main.py 關閉請求，
        # 避免不小心關掉設定視窗後，main.py 還在背景跑、卻再也找不到入口能停止它。
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

        self._build_header()
        self._build_process_bar()
        self._build_info_bar()
        self._build_middle_area()
        self._build_bottom_bar()

        self._populate_from_effective_state()
        # 裝在初次載入「之後」——避免初次載入時逐欄位 set() 觸發下面這個同步邏輯，
        # 誤判成「使用者剛剛改了 Host/Port」而動到端點欄位（見方法內註解）。
        self._wire_nodered_endpoint_autosync()
        self._process_manager.poll()

        # 到這裡整個視窗的固定佔用區塊（標題/流程列/獨立腳本工具列/資訊列/分頁按鈕列/
        # 底部按鈕列）都已經建好，現在才量得到它們的真實高度——用這個真實高度把終端機
        # 面板的「初始」高度設成視窗高度的 CONSOLE_DEFAULT_HEIGHT_FRACTION（預設
        # 75%）。終端機優先、預設就佔大部分畫面是刻意的設計，分頁表單內容區被壓縮到
        # 只剩一小截、甚至看不到欄位是可接受的結果；用比例而非寫死像素值，是因為不同
        # 螢幕解析度／系統 DPI 縮放下可用空間差異很大，比例才會等比縮放。
        self.update_idletasks()
        self._console_panel.apply_sane_default_height()
        # 終端機面板在 _build_middle_area() 就建立了，早於 _build_bottom_bar()；Tk 同層
        # 元件預設的堆疊順序是「後建立的蓋在先建立的上面」，所以底部按鈕列這時候會蓋在
        # 終端機面板之上，跟「終端機蓋過其他內容」的設計相反。ConsolePanel.place() 每次
        # 呼叫都會 lift() 一次，但 apply_sane_default_height() 只在高度真的改變時才會
        # 呼叫它，剛好沒改變的話就不會補這次 lift()——這裡直接無條件再 lift() 一次，
        # 確保不管前面有沒有觸發 place，終端機面板一定疊在所有其他元件最上層。
        self._console_panel.lift()

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

    def _apply_ttk_style(self):
        """ttk.Scrollbar／ttk.Combobox 這類元件預設吃 Windows 原生佈景主題，很多
        顏色設定（尤其捲動軸的滑塊/滑軌顏色）在原生佈景下會被系統忽略，實際跑
        起來滑塊幾乎跟滑軌同色、幾乎看不到有一條線可以拖——分頁按鈕列的橫向捲動
        軸、分頁右欄新加的橫向捲動軸、每個分頁內容區的垂直捲動軸都是這個問題。
        切換成 'clam' 佈景：這是 Tk 內建、純用 Tk 自己畫（不呼叫系統原生繪製）的
        佈景，顏色設定才會真的生效，滑塊改成藍灰色（跟按鈕次要色同一組），滑軌
        用主背景色，兩者對比夠明顯，一眼就看得到那條線、也看得出滑塊在哪一段。
        """
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            return  # 這個環境沒有 'clam' 佈景可用，維持系統預設，不強求

        # 'clam' 佈景的滑塊（thumb）子元件預設還是會用 lightcolor/darkcolor 畫一圈
        # 立體斜角（跟滑軌對比出「整條灰灰的、中間一小段顏色不一樣」的斑駁感），
        # 而且預設會在滑塊中間畫一個 grip（一小撮凸起的裝飾點，顏色又是另外算的）
        # ——這正是「灰色中間小段白白的」的來源：thumb 的底色雖然設定了，但
        # lightcolor/darkcolor 沒有一起蓋掉、grip 也沒有關掉，兩個各自預設的
        # 顏色疊在一起，看起來就是一條灰撲撲的軌道中間卡一小段對不上色的東西。
        # 修法：lightcolor/darkcolor 都設成跟滑塊底色同一色（斜角消失，變成
        # 扁平色塊）、gripcount=0（完全不畫那個裝飾點）。Horizontal／Vertical
        # 兩個方向的樣式分開明講設定（不只設共用的 "TScrollbar"）：不同 ttk 版本
        # 對子樣式是否會自動繼承共用樣式的顏色設定不完全可靠，明講兩份才保證
        # 兩個方向的捲動軸都吃到一樣的扁平配色。
        for orient_style in ("Horizontal.TScrollbar", "Vertical.TScrollbar"):
            style.configure(
                orient_style, background=BTN_SECONDARY_BG, troughcolor=COLOR_BG_MAIN,
                bordercolor=COLOR_BG_MAIN, lightcolor=BTN_SECONDARY_BG,
                darkcolor=BTN_SECONDARY_BG, arrowcolor="#ffffff", relief="flat",
                gripcount=0,
            )
            style.map(orient_style, background=[("active", BTN_SECONDARY_ACTIVE)])

        # 分頁右欄「欄位說明」面板的橫向捲動軸——之前為了跟使用者一起確認拉桿
        # 行為，暫時用很搶眼的桃紅色跟其他捲動軸區隔開來，方便指認「就是這一條」。
        # 拉桿行為已經確認沒問題，定案改成跟這塊面板本身配色一致的深藍
        # （COLOR_TOOL_DESC_ACCENT，跟卡片左側色條、標題文字同一個顏色）——這條
        # 捲軸本來就只服務「欄位說明文件」這塊面板，用同一色系而不是另外挑一個
        # 不相干的顏色，一眼就看得出「這是說明文件面板的一部分」，也不會像桃紅色
        # 那樣在一堆藍灰色系按鈕/元件裡顯得突兀。
        style.configure(
            "DocsPanel.Horizontal.TScrollbar", background=COLOR_TOOL_DESC_ACCENT,
            troughcolor=COLOR_BG_MAIN, bordercolor=COLOR_BG_MAIN,
            lightcolor=COLOR_TOOL_DESC_ACCENT, darkcolor=COLOR_TOOL_DESC_ACCENT, gripcount=0,
            arrowcolor="#ffffff", relief="flat",
        )
        style.map("DocsPanel.Horizontal.TScrollbar", background=[("active", "#2e6da4")])

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
        機制、同一時間只能跑一個（見 __init__ 裡 self._process_manager 的說明）。"""
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
            font=self._font_hint, outline=True,
        ).pack(side="right", padx=(SPACE_SM, 0))
        self._stop_main_btn = _styled_button(
            inner, "⏹ 關閉 main.py", self._on_stop_main, BTN_INFO_BG, BTN_INFO_ACTIVE,
            # 這顆跟下面「停止腳本」原本共用橘色警告色，改成淡藍色後跟「還原 GUI
            # 預設值」那顆分開，不再是同一組配色。BTN_INFO_BG 是刻意偏亮的淡藍，
            # 白字對比不夠（第一版用過，使用者回報字會糊），改配深藏青字
            # （BTN_INFO_FG），沿用粗體字型（筆畫較寬，蓋色面積更多，字比較扎實，
            # 維持跟其他按鈕一致的粗體視覺語言）。
            fg=BTN_INFO_FG, font=(_FONT_FAMILY, 11, "bold"),
        )
        self._stop_main_btn.pack(side="right", padx=(SPACE_SM, 0))
        self._start_main_btn = _styled_button(
            inner, "▶ 啟動 main.py", self._on_start_main, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE,
            font=self._font_hint,
        )
        self._start_main_btn.pack(side="right", padx=(SPACE_SM, 0))

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

        # 影片路徑覆寫（選填）：填了就在「▶ 執行所選腳本」啟動子行程時，額外塞一個
        # TEST_VIDEO_PATH 環境變數進去（見 _on_start_tool）。只對有讀這個環境變數的
        # 腳本有效（目前是 1_run_video_inference.py／2_run_dual_model_compare.py／
        # 1_measure_ear_distance_single_video.py 這三支），其餘腳本會安靜忽略、跟沒填
        # 一樣——本視窗本來就不檢查每支腳本內部邏輯（見上面「瀏覽...」按鈕旁的提示文字），
        # 這個欄位延續同一個原則，不對「填了但腳本不支援」的情況另外提示或報錯。
        tool_row_video = tk.Frame(tool_outer, bg=COLOR_HEADER_BG)
        tool_row_video.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(
            tool_row_video, text="🎬 影片路徑（選填）:", bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG,
            font=self._font_hint,
        ).pack(side="left")
        self._tool_video_path_var = tk.StringVar(value="")
        tk.Entry(
            tool_row_video, textvariable=self._tool_video_path_var, font=self._font_hint,
        ).pack(side="left", fill="x", expand=True, padx=(6, 8))
        # 原本是單一「瀏覽...」按鈕彈出選單選「檔案」或「資料夾」——彈出選單本身
        # 是原生元件，不管怎麼配色都不會有實心按鈕那種立體感/一致外觀（見
        # _on_browse_tool_video 原本的說明）。改成直接放兩顆並排的小按鈕，兩個
        # 選項都看得到、不用多一次點擊才知道有哪些選項，也才能套用跟其他按鈕
        # 一致的樣式系統。
        _styled_button(
            tool_row_video, "🎬 選擇影片", self._pick_tool_video_file, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
            font=self._font_hint, compact=True,
        ).pack(side="left", padx=(0, SPACE_XS))
        _styled_button(
            tool_row_video, "📁 選擇資料夾", self._pick_tool_video_folder, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
            font=self._font_hint, compact=True,
        ).pack(side="left")

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
            font=self._font_label, compact=True,
        )
        browse_btn.pack(side="left", padx=(0, SPACE_SM))
        open_file_btn = _styled_button(
            tool_row2, "開啟檔案", self._on_open_tool_script_file, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
            font=self._font_label, compact=True,
        )
        open_file_btn.pack(side="left", padx=(0, SPACE_SM))
        self._start_tool_btn = _styled_button(
            tool_row2, "▶ 執行所選腳本", self._on_start_tool, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE,
            font=self._font_label, compact=True,
        )
        self._start_tool_btn.pack(side="left", padx=(0, SPACE_SM))
        self._stop_tool_btn = _styled_button(
            tool_row2, "⏹ 停止腳本", self._on_stop_tool, BTN_INFO_BG, BTN_INFO_ACTIVE,
            fg=BTN_INFO_FG, font=self._font_label_bold, compact=True,
        )
        self._stop_tool_btn.pack(side="left")

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
        pm = self._process_manager
        if pm.is_running:
            if not messagebox.askyesno(
                "關閉設定視窗",
                f"{pm.active_label} 目前正在執行中（PID {pm.process.pid}）。\n\n"
                "關閉本視窗會一併送出關閉信號，避免關掉視窗後失去控制、"
                "無法再停止正在執行的程式。\n\n是否繼續？",
            ):
                return
            self._process_status_var.set(f"🖥️ 正在關閉 {pm.active_label}…")
            self.update_idletasks()
            if not pm.request_shutdown_and_wait():
                messagebox.showwarning(
                    "關閉設定視窗",
                    f"{pm.active_label} 似乎沒有在預期時間內結束，請自行檢查工作管理員確認狀態。",
                )
        self.destroy()

    def _on_start_main(self):
        self._process_manager.start_main(_MAIN_PY_PATH, _SCRIPT_DIR)

    def _on_stop_main(self):
        self._process_manager.stop_main()

    def _on_browse_tool_script(self):
        initial_dir = str(_SCRIPT_DIR / "cat_monitoring_system" / "tools")
        path = filedialog.askopenfilename(
            title="選擇要執行的 Python 腳本",
            initialdir=initial_dir if Path(initial_dir).exists() else str(_SCRIPT_DIR),
            filetypes=[("Python 腳本", "*.py"), ("所有檔案", "*.*")],
        )
        if path:
            self._tool_script_var.set(path)

    def _pick_tool_video_file(self):
        """跟 _pick_tool_video_folder 並排成兩顆按鈕（見 _build_process_bar 的
        tool_row_video）——原本是單一按鈕彈出 tk.Menu 選單問要選檔案還是資料夾，
        改成直接給兩顆按鈕，兩個選項一眼就看得到，也不用忍受原生選單元件沒辦法
        套用一致樣式的問題（TEST_VIDEO_PATH 腳本端本來就同時吃檔案跟資料夾，
        見上面的提示文字）。"""
        path = filedialog.askopenfilename(
            title="選擇影片檔案",
            filetypes=[
                ("影片檔案", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.mpg *.mpeg *.webm"),
                ("所有檔案", "*.*"),
            ],
        )
        if path:
            self._tool_video_path_var.set(path)

    def _pick_tool_video_folder(self):
        path = filedialog.askdirectory(title="選擇影片資料夾")
        if path:
            self._tool_video_path_var.set(path)

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
        video_path = self._tool_video_path_var.get().strip()
        extra_env = {"TEST_VIDEO_PATH": video_path} if video_path else None
        self._process_manager.start_tool(script_file, extra_env=extra_env)

    def _on_stop_tool(self):
        self._process_manager.stop_tool()

    def _update_process_buttons_state(self):
        """main.py 與獨立腳本工具共用 ProcessManager 的 self.process 這一個欄位（同一
        時間只能跑一個），所以兩組「啟動」按鈕永遠一起致能/禁用；「停止」按鈕則依
        pm.active_kind 只讓對應那一顆生效，避免「明明在跑腳本，卻按了寫著『關閉
        main.py』的按鈕」這種文字跟實際動作對不上的情況。這個函式是 ProcessManager
        的 on_state_change callback，行程狀態一有變化（啟動/關閉/輪詢發現已死）就會
        被呼叫。"""
        pm = self._process_manager
        running = pm.is_running
        if running:
            self._process_status_var.set(f"🖥️ {pm.active_label} 執行中（PID {pm.process.pid}）")
            self._start_main_btn.config(state="disabled")
            self._start_tool_btn.config(state="disabled")
            self._stop_main_btn.config(state="normal" if pm.active_kind == "main" else "disabled")
            self._stop_tool_btn.config(state="normal" if pm.active_kind == "tool" else "disabled")
        else:
            was_started = pm.active_label is not None
            self._process_status_var.set(
                f"🖥️ {pm.active_label} 已結束" if was_started else "🖥️ 尚未啟動任何程式"
            )
            self._start_main_btn.config(state="normal")
            self._start_tool_btn.config(state="normal")
            self._stop_main_btn.config(state="disabled")
            self._stop_tool_btn.config(state="disabled")

        # 終端機面板的輸入列在本函式第一次被呼叫時（_build_process_bar 尾端）還沒
        # 建出來（_build_middle_area 是之後才跑的），用 getattr 擋一下，之後每次
        # 呼叫這個 panel 就一定存在了。
        console = getattr(self, "_console_panel", None)
        if console is not None:
            console.set_input_enabled(running)

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
        輸出面板（ConsolePanel，見 settings_gui/console_panel.py）改用 place() 做成
        一塊貼齊視窗左右邊界與底部、可自由調整高度的浮動面板——不參與 pack 版面
        協商，所以拉高時是直接「疊在」表單上面蓋過去，不是跟表單擠位置；可以一路
        拉到貼齊標題列下緣（見 ConsolePanel.max_height()），此時流程列／獨立腳本
        工具列／資訊列／分頁按鈕列／表單內容／底部按鈕列都會被蓋住，這是刻意允許
        的效果，使用者要操作這些時把終端機拖小或收合即可隨時蓋回去。"""
        middle = tk.Frame(self, bg=COLOR_BG_MAIN)
        middle.pack(fill="both", expand=True)
        self._build_tabs(middle)

        console_container = tk.Frame(self, bg=COLOR_CONSOLE_BG, height=CONSOLE_DEFAULT_HEIGHT)
        console_container.pack_propagate(False)  # 固定高度，不因裡面的 Text 內容而被撐大
        self._console_panel = ConsolePanel(self, console_container)
        # ProcessManager 建構時 ConsolePanel 還沒蓋出來（見 __init__ 開頭說明），
        # 兩邊互相需要對方：ProcessManager 啟動子行程後要往終端機寫訊息，
        # ConsolePanel 的輸入列要把文字送進子行程 stdin——都在這裡補上。
        self._process_manager.console = self._console_panel
        self._console_panel.set_stdin_handler(self._process_manager.send_stdin)
        # 終端機面板每次改變高度/位置都要讓分頁右欄那條浮動橫向捲軸（見
        # _reposition_active_docs_hscroll）跟著重新貼齊「終端機正上方」，在第一次
        # place() 之前就先掛好這個回呼，之後不管是這裡、拖拉、收合展開、或
        # __init__ 最後的 apply_sane_default_height() 觸發的每一次 place()，
        # 都會自動補這次重新定位，不用另外在每個呼叫點各自記得呼叫一次。
        self._console_panel.set_on_resize(self._reposition_active_docs_hscroll)
        self._console_panel.place(CONSOLE_DEFAULT_HEIGHT)

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
        # 全域欄位搜尋欄常駐貼在同一列的右側（tab_bar_row 裡跟 tab_canvas 並排，
        # 搜尋欄本身不隨分頁按鈕橫向捲動）；橫向捲軸 tab_hscroll 維持在 tab_bar_row
        # 下面單獨一列，寬度還是貼齊整個 tab_bar_outer，不受搜尋欄影響。
        tab_bar_outer = tk.Frame(parent, bg=COLOR_BG_MAIN)
        tab_bar_outer.pack(fill="x", padx=16, pady=(0, 4))
        self._tab_bar_outer = tab_bar_outer

        tab_bar_row = tk.Frame(tab_bar_outer, bg=COLOR_BG_MAIN)
        tab_bar_row.pack(fill="x", side="top")

        tab_canvas = tk.Canvas(tab_bar_row, bg=COLOR_BG_MAIN, highlightthickness=0)
        tab_hscroll = ttk.Scrollbar(tab_bar_outer, orient="horizontal", command=tab_canvas.xview)
        tab_bar = tk.Frame(tab_canvas, bg=COLOR_BG_MAIN)
        tab_bar.bind(
            "<Configure>", lambda e: tab_canvas.configure(scrollregion=tab_canvas.bbox("all"))
        )
        tab_canvas.create_window((0, 0), window=tab_bar, anchor="nw")
        tab_canvas.configure(xscrollcommand=tab_hscroll.set)
        self._field_search = FieldSearchBar(self, tab_bar_row, bg=COLOR_BG_MAIN)
        tab_canvas.pack(side="left", fill="x", expand=True)
        tab_hscroll.pack(side="top", fill="x")

        def _on_tabbar_wheel(event):
            tab_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

        tab_canvas.bind("<Enter>", lambda e: tab_canvas.bind_all("<MouseWheel>", _on_tabbar_wheel))
        tab_canvas.bind("<Leave>", lambda e: tab_canvas.unbind_all("<MouseWheel>"))

        content_area = tk.Frame(parent, bg=COLOR_TAB_BG)
        content_area.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self._content_area = content_area
        # 視窗被縮放時，content_area／right_col 的實際寬度會跟著變，浮動的說明
        # 文件橫向捲軸（見 _reposition_active_docs_hscroll）要重新量一次位置，
        # 不然縮放後會跟右欄對不齊。終端機面板尺寸改變不會觸發這個事件（place()
        # 是獨立於 pack/grid 版面協商之外的疊加層，不會連帶讓 content_area 觸發
        # <Configure>），那邊另外用 ConsolePanel.set_on_resize() 掛回呼處理。
        content_area.bind("<Configure>", lambda _e: self._reposition_active_docs_hscroll())

        self._tab_buttons = {}
        self._tab_frames = {}
        self._tab_accents = {}
        # 每個分頁右半邊的容器：塞的是 tab_docs_panel 畫出來的欄位說明文件卡片
        # （見下方迴圈）；分頁內容（banner + 欄位列）只塞進左半邊的 left_col，
        # 兩欄用 grid + uniform 群組強制等寬（不會因為左邊欄位內容變寬/變窄而跟著晃動）。
        self._tab_right_columns = {}
        # 每個分頁對應的 tab_docs_panel.render() 回傳值裡的 "resync"：一個「把
        # 橫向捲軸歸零、重新同步」的 callable，在 _select_tab() 真正切換到該分頁、
        # 即將顯示的當下呼叫（見 tab_docs_panel.render() 的說明）。
        self._tab_docs_resync = {}
        # 每個分頁對應的橫向捲軸元件本身（tab_docs_panel.render() 回傳值裡的
        # "hscroll"）。這條捲軸不跟著分頁內容排版，是浮動貼在「終端機面板正
        # 上方、跟右欄同寬」的固定位置（見 _reposition_active_docs_hscroll()），
        # 同一時間只有目前選取中分頁的那一條會被 place() 出來，其餘用
        # place_forget() 收起。
        self._tab_docs_hscroll = {}
        # 目前 place() 顯示中的那一條 hscroll（None＝目前分頁沒有可用的說明文件、
        # 沒有東西可以顯示）；_select_tab() 切換分頁、視窗/終端機尺寸改變時都要
        # 重新定位它，見 _reposition_active_docs_hscroll()。
        self._active_docs_hscroll = None
        # 目前搜尋高亮住的欄位（json_key 清單）；換一次搜尋或清空搜尋都要先清掉
        # 上一次的高亮，狀態存在這裡，見 _highlight_fields()。
        self._highlighted_field_keys = []

        # 分頁右欄的說明文件內容只需要解析一次（不是每個分頁各解析一次整份文件），
        # 迴圈外先呼叫一次 tab_docs_panel.parse()，迴圈內用 .get(tab_name, []) 取用。
        # 內容是「這個分頁對應哪個模組/核心函式」（見
        # docs/設定分頁模組與核心函式對照表.md），不是逐欄位的 JSON 路徑對照表
        # （那份是 docs/設定視窗欄位對照表.md，這裡沒有用到）。
        tab_docs = tab_docs_panel.parse(_SCRIPT_DIR / "docs" / "設定分頁模組與核心函式對照表.md")

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

            # 分頁內容區分成左右兩欄：左欄放 banner + 所有欄位列（原本會撐滿整個
            # tab.body 寬度），右欄目前刻意留空。columnconfigure 用同一個 uniform
            # 群組名稱("tab_half")讓兩欄強制等寬（各佔 50%），不受左欄內容實際
            # 需要的寬度影響——如果不用 uniform，Tk 的 pack/grid 預設是依內容需求
            # 分配寬度，欄位多的分頁左欄會比欄位少的分頁寬，兩欄就不會對齊。
            columns = tk.Frame(tab.body, bg=COLOR_TAB_BG)
            columns.pack(fill="both", expand=True)
            columns.columnconfigure(0, weight=1, uniform="tab_half")
            columns.columnconfigure(1, weight=1, uniform="tab_half")
            columns.rowconfigure(0, weight=1)

            left_col = tk.Frame(columns, bg=COLOR_TAB_BG)
            left_col.grid(row=0, column=0, sticky="nsew")
            right_col = tk.Frame(columns, bg=COLOR_TAB_BG)
            right_col.grid(row=0, column=1, sticky="nsew", padx=(14, 0))
            self._tab_right_columns[tab_name] = right_col
            docs_result = tab_docs_panel.render(
                right_col, tab_docs.get(tab_name, []), self, tab_name, content_area
            )
            self._tab_docs_resync[tab_name] = docs_result["resync"]
            self._tab_docs_hscroll[tab_name] = docs_result["hscroll"]

            banner = tk.Frame(left_col, bg=accent)
            banner.pack(fill="x")
            tk.Label(
                banner, text=f"{emoji} {tab_name}", bg=accent, fg="#ffffff",
                font=self._font_banner, anchor="w",
            ).pack(fill="x", padx=14, pady=8)

            for field in fields_by_tab.get(tab_name, []):
                self._build_field_row(left_col, field, accent)
            if tab_name == "ST-GCN 推論":
                tk.Label(
                    left_col,
                    text="ℹ️ 訓練用參數（SEQUENCE_LENGTH／FEATURE_MODE／NUM_CLASSES）由"
                    " stgcn_config.yaml 管理，不在此設定視窗顯示或覆寫。",
                    bg=COLOR_TAB_BG, fg=COLOR_HINT_FG, font=self._font_hint,
                    anchor="w", justify="left", wraplength=750,
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

        # 右欄說明文件面板的橫向捲軸要在這個分頁「真正被畫到畫面上」之後才重新
        # 歸零／同步一次（用 after_idle 排到 frame.pack() 造成的版面更新處理完
        # 之後才執行）：分頁在還沒被選取之前是 pack_forget 狀態、沒有被映射到
        # 畫面上，這段期間對 ttk.Scrollbar 呼叫 .set() 更新的只是內部狀態，
        # 不保證元件之後顯示出來時會照著重繪滑塊——實測會出現滑塊視覺位置卡在
        # 建構當下（版面還沒定案）算出來的錯誤比例，一直沒跟著之後的真實版面
        # 更新過來。改成分頁被選取時才呼叫，確保這一次呼叫發生在元件保證會被
        # 畫出來的狀態下。
        resync = self._tab_docs_resync.get(tab_name)
        if resync is not None:
            self.after_idle(resync)

        # 說明文件面板的橫向捲軸不跟著分頁內容排版（見 tab_docs_panel.render()
        # 的說明），同一時間只顯示「目前分頁」那一條，切分頁時要把上一條收起、
        # 換成這一條，並重新定位到「終端機面板正上方」。跟 resync 一樣要排到
        # after_idle：這裡也依賴 frame.pack() 造成的版面更新先跑完，才量得到
        # 這個分頁的 right_col 真實寬度/位置。
        old_hscroll = self._active_docs_hscroll
        if old_hscroll is not None:
            old_hscroll.place_forget()
        self._active_docs_hscroll = self._tab_docs_hscroll.get(tab_name)
        self.after_idle(self._reposition_active_docs_hscroll)

    def _reposition_active_docs_hscroll(self):
        """把目前分頁的說明文件橫向捲軸（`self._active_docs_hscroll`）用
        `place()` 固定貼在「終端機輸出面板正上方、跟右欄同寬」的位置——這條
        捲軸的母元件是 `content_area`（見 tab_docs_panel.render() 呼叫端傳入的
        `hscroll_master`），不是分頁內容本身，所以位置要靠這裡手動算，不會隨著
        分頁版面自動排好。

        呼叫時機（凡是「這條捲軸該出現在哪裡」可能改變的時候都要呼叫一次）：
        - `_select_tab()`：換了分頁，換了另一條 hscroll、換了另一個 right_col。
        - `content_area` 尺寸改變（視窗縮放）：見下面的 `<Configure>` 綁定。
        - 終端機面板尺寸/位置改變（拖拉／收合展開／初始套用比例）：見
          `ConsolePanel.set_on_resize()` 掛的回呼，接到這裡。

        還沒建到終端機面板（`_console_panel`）或底部按鈕列存在之前（建構過程
        最初 `_select_tab(TAB_ORDER[0])` 那一次呼叫）沒有基準點可以定位，先跳過
        ——終端機面板真正 place() 出來的那一刻（`_build_middle_area()` 最後）
        會經由 `set_on_resize` 回頭補呼叫一次，屆時基準點就都齊了。

        `getattr(..., None)` 不直接用 `self._active_docs_hscroll`：這個方法也被
        `content_area` 的 `<Configure>` 事件回呼掛著，理論上要等 `_build_tabs()`
        把這個屬性設好之後事件才會真的被處理，但用 getattr 多一層防呆不用去
        依賴這個時序假設一定成立。"""
        hscroll = getattr(self, "_active_docs_hscroll", None)
        if hscroll is None:
            return
        console = getattr(self, "_console_panel", None)
        if console is None:
            return
        tab_name = next(
            (name for name, h in self._tab_docs_hscroll.items() if h is hscroll), None
        )
        right_col = self._tab_right_columns.get(tab_name) if tab_name is not None else None
        if right_col is None:
            return

        self._content_area.update_idletasks()
        # x/width 用 right_col 目前的實際畫面位置/寬度（跟分頁哪一欄對齊）；
        # y 用終端機面板容器的畫面頂端（跟終端機正上方對齊，不管終端機目前是
        # 展開、收合還是被拖到多高）。兩邊都是螢幕絕對座標（winfo_rootx/rooty），
        # 換算成 content_area 座標系底下的相對值，因為 place() 的 x/y 是相對母
        # 元件（這裡是 content_area）算的，不是相對整個螢幕。
        origin_x = self._content_area.winfo_rootx()
        origin_y = self._content_area.winfo_rooty()
        x = right_col.winfo_rootx() - origin_x
        width = right_col.winfo_width()
        y = console.container.winfo_rooty() - origin_y
        hscroll.place(x=x, y=y, anchor="sw", width=width)
        hscroll.lift()

    def _set_tab_match_badges(self, matches_by_tab):
        """依序幫每個分頁按鈕的文字補上/拿掉「(N)」符合數量後綴——
        FieldSearchBar（settings_gui/field_search.py）搜尋結果的窄接口回呼。
        matches_by_tab 是 {分頁名稱: 符合欄位數}，沒有出現在裡面的分頁一律視為 0
        （拿掉後綴，恢復成原本沒有搜尋時的按鈕文字）。"""
        for tab_name, btn in self._tab_buttons.items():
            emoji, _ = TAB_COLORS.get(tab_name, ("⬜", COLOR_HEADER_BG))
            count = matches_by_tab.get(tab_name, 0)
            suffix = f" ({count})" if count else ""
            btn.config(text=f"{emoji} {tab_name}{suffix}")

    def _highlight_fields(self, json_keys):
        """幫指定欄位的外層 container 加高亮外框，並先清掉上一次的高亮——
        FieldSearchBar 每次重新搜尋都會呼叫這個方法（空清單＝單純清掉舊高亮，
        對應搜尋欄被清空的情況）。"""
        for key in self._highlighted_field_keys:
            info = self._field_widgets.get(key)
            if info is not None:
                info["container"].config(highlightthickness=0)
                info["accent_strip"].config(bg=info["accent_color"], width=4)
        self._highlighted_field_keys = list(json_keys)
        for key in json_keys:
            info = self._field_widgets.get(key)
            if info is not None:
                # 只改左側色條的顏色/寬度不夠明顯（原本每列本來就有一條分頁代表色的
                # 細條，改個顏色不容易注意到）；同時加粗外框＋把色條加寬變成高亮色，
                # 兩個訊號疊加才夠顯眼，一眼就能在一整頁欄位裡找到搜尋命中的是哪一列。
                info["container"].config(
                    highlightbackground=SEARCH_HIGHLIGHT_BORDER,
                    highlightcolor=SEARCH_HIGHLIGHT_BORDER,
                    highlightthickness=3,
                )
                info["accent_strip"].config(bg=SEARCH_HIGHLIGHT_BORDER, width=10)

    def _scroll_field_into_view(self, tab_name, json_key):
        """把指定欄位捲進該分頁的可視範圍——算它的 container 相對 tab.body 頂端的
        垂直位置，換算成 ConsolePanel／_ScrollableTab 共用的 canvas.yview_moveto()
        要的 0~1 比例。"""
        info = self._field_widgets.get(json_key)
        tab = self._tab_frames.get(tab_name)
        if info is None or tab is None:
            return
        tab.canvas.update_idletasks()
        body_height = tab.body.winfo_height()
        if body_height <= 0:
            return
        container = info["container"]
        # container 的父層是 left_col、left_col 的父層是 columns、columns 的父層
        # 才是 tab.body（見 _build_tabs 的三層 grid/pack 結構），三層都從 (0,0)
        # 起始排版，直接加總三層 winfo_y() 就是 container 相對 tab.body 頂端的
        # 實際 y 座標。
        y = container.winfo_y() + container.master.winfo_y() + container.master.master.winfo_y()
        fraction = max(0.0, min(1.0, y / body_height))
        tab.canvas.yview_moveto(fraction)

    def _build_bottom_bar(self):
        bottom = tk.Frame(self, bg=COLOR_BG_MAIN)
        bottom.pack(fill="x", padx=16, pady=(0, 12))
        self._bottom_bar_frame = bottom
        _styled_button(bottom, "載入目前設定", self._on_load_current, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE).pack(side="left")
        _styled_button(
            bottom, "還原 GUI 預設值", self._on_restore_defaults, BTN_WARN_BG, BTN_WARN_ACTIVE, fg=BTN_WARN_FG,
        ).pack(side="left", padx=(SPACE_SM, 0))
        _styled_button(bottom, "匯出設定", self._on_export, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE).pack(side="left", padx=(SPACE_SM, 0))
        _styled_button(bottom, "匯入設定", self._on_import, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE).pack(side="left", padx=(SPACE_SM, 0))
        _styled_button(bottom, "關閉", self._on_window_close, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, outline=True).pack(side="right")
        _styled_button(bottom, "儲存設定", self._on_save, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE).pack(side="right", padx=(0, SPACE_SM))

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
        accent_strip = tk.Frame(container, bg=accent, width=4)
        accent_strip.pack(side="left", fill="y")

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

        info = {
            "field": field, "badge_var": badge_var, "badge_widget": badge,
            "container": container, "accent_strip": accent_strip, "accent_color": accent,
        }

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
            # validate == "output_path" 的欄位（tracker_state_path／daily_history_db_path／
            # global_context_path／csv_path／segments_csv_path）是「程式自己會寫出去的檔案」，
            # 不是「選一個已存在的檔案來讀」——用 askopenfilename 會在檔案還不存在時（例如
            # 搬到新機器、新 clone、或使用者想改指到一個全新的位置）沒辦法順利選定，只能整段
            # 手動打字。2026-08-26 發現：改用 asksaveasfilename（跟下面「匯出設定」按鈕一致），
            # 讓使用者可以瀏覽到想要的資料夾、直接打檔名建立新路徑；其餘 value_type == "file"
            # 欄位（例如 cat_identity 的特徵基準檔，validate == "optional_file_warn"）語意是
            # 「讀取既有檔案」，維持 askopenfilename 不變。
            is_output_path = field.get("validate") == "output_path"

            def _browse(v=var, bf=browse_filter, is_output=is_output_path):
                filetypes = [bf, ("所有檔案", "*.*")] if bf else [("所有檔案", "*.*")]
                if is_output:
                    path = filedialog.asksaveasfilename(
                        title="選擇路徑", filetypes=filetypes, confirmoverwrite=False,
                    )
                else:
                    path = filedialog.askopenfilename(title="選擇檔案", filetypes=filetypes)
                if path:
                    v.set(path)

            _styled_button(
                control, "瀏覽...", _browse, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
                font=self._font_hint, compact=True,
            ).pack(side="left", padx=(SPACE_SM, 0))
            info["var"] = var
        elif vt == "folder":
            var = tk.StringVar()
            tk.Entry(control, textvariable=var, font=self._font_label).pack(side="left", fill="x", expand=True)

            def _browse(v=var):
                path = filedialog.askdirectory(title="選擇資料夾")
                if path:
                    v.set(path)

            _styled_button(
                control, "選擇資料夾...", _browse, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
                font=self._font_hint, compact=True,
            ).pack(side="left", padx=(SPACE_SM, 0))
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
                font=self._font_hint, compact=True,
            ).pack(side="left", padx=(SPACE_SM, 0))

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
                font=self._font_hint, anchor="w", justify="left", wraplength=750,
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
            font=self._font_hint, anchor="w", justify="left", wraplength=750,
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

    def _wire_nodered_endpoint_autosync(self):
        """Host/Port 改變時，「進階設定」的 3 個 Node-RED 端點欄位如果目前的值
        還是「跟改變前的 Host/Port 組出來的樣子一模一樣」，就視為使用者從沒動過
        （只是承接 config.py 原本 f"http://{HOST}:{PORT}/..." 那個自動推導預設
        值），跟著同步更新成新的 Host/Port；一旦某個端點的值不符合這個樣式，代表
        使用者刻意手動改成別的網址（例如故意指向另一台機器），就不再自動覆蓋。

        背景：`advanced.nodered_endpoint_notify`／`_result`／`_result_v2` 這 3
        個欄位一旦存過一次 `runtime_settings.current.json`（幾乎必然，因為
        `_on_save()` 每次都整包寫回全部欄位），`config.py` 的
        `_runtime_default()` 就會一路優先讀 JSON 裡的字面值，config.py 那行
        `f"http://{HOST}:{PORT}/..."` 動態組字串永遠不會再被執行到——換 Host/
        Port 後這 3 個端點會「看起來像自動、實際上是凍結的舊值」。這裡把這段
        同步邏輯補回 GUI 層，讓「沒被使用者特別改過」的端點欄位可以繼續跟著
        Host/Port 走，同時不破壞刻意覆寫的情況（`docs/設定視窗欄位對照表.md`
        對照表裡這 3 個端點原本就設計成「可個別覆寫成完全不同的網址」）。
        """
        host_info = self._field_widgets.get("nodered.host")
        port_info = self._field_widgets.get("nodered.port")
        if not host_info or not port_info:
            return
        endpoint_suffixes = {
            "advanced.nodered_endpoint_notify": "python_online",
            "advanced.nodered_endpoint_result": "yolo_result",
            "advanced.nodered_endpoint_result_v2": "yolo_result_v2",
        }
        endpoint_vars = {
            key: self._field_widgets[key]["var"]
            for key in endpoint_suffixes
            if key in self._field_widgets
        }
        if not endpoint_vars:
            return

        last = {"host": host_info["var"].get(), "port": port_info["var"].get()}

        def _on_host_or_port_change(*_a):
            new_host = host_info["var"].get()
            new_port = port_info["var"].get()
            old_host, old_port = last["host"], last["port"]
            if new_host != old_host or new_port != old_port:
                for key, suffix in endpoint_suffixes.items():
                    var = endpoint_vars.get(key)
                    if var is None:
                        continue
                    expected_old = f"http://{old_host}:{old_port}/{suffix}"
                    if var.get() == expected_old:
                        var.set(f"http://{new_host}:{new_port}/{suffix}")
            last["host"], last["port"] = new_host, new_port

        host_info["var"].trace_add("write", _on_host_or_port_change)
        port_info["var"].trace_add("write", _on_host_or_port_change)

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
        # 預設檔名帶上「匯出當下」的時間戳記（YYYYMMDD_HHMMSS，跟專案裡
        # eval_results/ 底下既有的時間戳記資料夾同一種格式，不用另外發明新格式）
        # 當流水編號：每次匯出檔名自然不同，不會互相覆蓋，檔名本身照字面排序就是
        # 照時間排序，方便直接比對不同時間點匯出的版本差異。使用者仍可以在存檔
        # 對話框裡自己改檔名，這裡只是給一個不用手動想名字的預設值。
        default_name = f"runtime_settings_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = filedialog.asksaveasfilename(
            title="匯出設定", defaultextension=".json",
            initialfile=default_name, filetypes=[("JSON", "*.json")],
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

        _styled_button(btn_row, "取消", dialog.destroy, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, outline=True).pack(side="right")
        _styled_button(btn_row, "套用到表單", _apply, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE).pack(side="right", padx=(0, SPACE_SM))

        self.wait_window(dialog)
        return result["apply"]


if __name__ == "__main__":
    SettingsWindow().mainloop()
