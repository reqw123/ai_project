"""`settings_window.py` 與 `settings_gui` 底下各模組共用的樣式常數／小工具。

只搬了「其他 settings_gui 模組也會用到」的那一小撮常數過來（標題列配色、終端機
配色、按鈕配色、終端機尺寸與字級參數、說明卡片配色、間距比例），加上
`_styled_button()`——其餘只有 `settings_window.py` 自己用到的樣式常數（分頁代表色、
徽章配色、布林旗標配色等）仍留在 `settings_window.py` 裡，沒有必要搬。

**2026-08 版面美化（第二版）**：使用者回饋「很多重複顏色且黯淡無味」。改動：
① `_styled_button()` 從 `relief="raised"` 立體邊框改**扁平**（`relief="flat"`），
互動回饋改用滑鼠進出時切換底色；② 按鈕色相從「幾乎全擠在藍灰」拆成按語意分開
的 7 組（見下方 `BTN_*` 區塊：PRIMARY／SECONDARY／MANAGE／INFO／DANGER／WARN／
QUIET），飽和度整體拉高；③ `outline=True` 不再是幽靈按鈕，改指向 `BTN_QUIET_*`
（扁平淺灰的「安靜」層級）。第一版（4 組→3 組、raised 立體按鈕、幽靈 outline）
的作法已整個被取代。
"""

import tkinter as tk
import unicodedata

# ── 標題列配色（settings_window.py 的標題列／流程列，跟終端機面板的標題列共用同一組）──
COLOR_HEADER_BG = "#2c3e50"
COLOR_HEADER_FG = "#ffffff"

# ── 間距比例（4 的倍數）：按鈕內距、按鈕之間的間隔、卡片內距全部從這幾個值挑，
# 不再各自寫死不同的數字——網頁前端常見的 4/8/12/16px 節奏，套用在 Tkinter 的
# padx/pady 上一樣適用。
SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 16

# ── 按鈕配色（2026-08 第二版：扁平、全部填色）─────────────────────────────
# 使用者回饋「很多重複顏色且黯淡無味」——改掉：① 拿掉 relief=raised 的立體邊框
# 改扁平；② 原本幾乎所有非主要按鈕都擠在同一個藍灰色，現在按語意拆成幾個彼此
# 分得開的色相；③ 飽和度整體拉高。每組都給 BG（常態）+ ACTIVE（hover／按下，
# 比常態深一階，扁平按鈕沒有立體感，就靠這個切換做互動回饋）。
#
#   PRIMARY   綠   正面且唯一的主要動作（儲存／啟動／執行／套用）
#   SECONDARY 鋼藍 一般功能性動作（瀏覽／開啟檔案／匯出入／上移下移…）
#   MANAGE    靛藍 管理／設定類（各處的「⚙ 管理」），跟一般次要動作再拉開一個色相
#   INFO      青   中斷正在跑的行程（關閉 main.py／停止腳本／停止）
#   DANGER    紅   破壞性動作（刪除）
#   WARN      琥珀 要留意的中斷性動作（還原 GUI 預設值／全部排除）
#   QUIET     淺灰 收尾／取消／關閉視窗／縮小視窗——`outline=True` 走這組，仍是
#                  明確的實心按鈕，只是刻意不搶眼（取代舊版看起來像 disabled 的幽靈按鈕）
BTN_PRIMARY_BG = "#16a34a"
BTN_PRIMARY_ACTIVE = "#15803d"
BTN_SECONDARY_BG = "#3b6fb5"
BTN_SECONDARY_ACTIVE = "#335f9c"
BTN_MANAGE_BG = "#5b6bd6"
BTN_MANAGE_ACTIVE = "#4c5cc4"
BTN_INFO_BG = "#0891b2"
BTN_INFO_ACTIVE = "#0e7490"
BTN_INFO_FG = "#ffffff"
BTN_DANGER_BG = "#dc2626"
BTN_DANGER_ACTIVE = "#b91c1c"
BTN_WARN_BG = "#d97706"
BTN_WARN_ACTIVE = "#b45309"
BTN_WARN_FG = "#3a1f00"  # 琥珀底配白字對比不足（未達 WCAG AA），改深棕字
BTN_QUIET_BG = "#e2e8f0"
BTN_QUIET_ACTIVE = "#cbd5e1"
BTN_QUIET_FG = "#334155"

# 底部「main.py 終端機輸出」面板：刻意用深色終端機配色跟上方淡色表單拉開視覺區隔，
# 一眼就能認出這塊是「輸出訊息」而不是可編輯欄位。
COLOR_CONSOLE_BG = "#1e1e1e"
COLOR_CONSOLE_FG = "#d4d4d4"
COLOR_CONSOLE_MUTED_FG = "#7f8c8d"
COLOR_CONSOLE_GRIP_BG = "#3a3f44"  # 可拖拉調整高度的把手，比面板底色略亮，暗示「這裡能拖」

CONSOLE_DEFAULT_HEIGHT = 260  # 預設展開高度
CONSOLE_MIN_HEIGHT = 50  # 拖拉能縮到的最小高度，比收合狀態(34px)略高，仍看得到一點點輸出內容
CONSOLE_COLLAPSED_HEIGHT = 34  # 內縮後只剩標題列那一條水平線的高度
CONSOLE_MAX_HEIGHT_RESERVE = 200  # 標題列高度還量不到時的退回值（拖到最高＝扣掉標題列高度）
CONSOLE_DEFAULT_HEIGHT_FRACTION = 0.75  # 終端機面板「初始」高度＝視窗高度的這個比例

CONSOLE_FONT_FAMILY = "Consolas"
CONSOLE_DEFAULT_FONT_SIZE = 20  # 原本 10 的 2 倍；Ctrl+0／Ctrl+numpad0 重設回這個大小
CONSOLE_MIN_FONT_SIZE = 6
CONSOLE_MAX_FONT_SIZE = 28

# 常駐說明卡片配色（獨立腳本工具的功能說明卡片、分頁右欄的欄位說明文件面板共用）。
COLOR_TOOL_DESC_BG = "#eaf4fc"
COLOR_TOOL_DESC_BORDER = "#aed6f1"
COLOR_TOOL_DESC_ACCENT = "#1b4f72"  # 卡片左側色條
COLOR_TOOL_DESC_FG = "#1b2631"  # 深色文字，淡藍底上要維持可讀性


def _display_width(text: str) -> int:
    """算字串的「顯示寬度」（全形/中文字元算 2 個半形字寬，其餘算 1）——用在等寬
    字型的清單裡要用空白對齊多欄內容時，len() 對中文字元會低估實際佔用的寬度；
    也用來估算 tk.Entry 的 `width=`（字元數單位，基準是拉丁字元寬度）該給多少，
    中文字元佔用的實際像素寬度大約是拉丁字元的兩倍，用這個函式換算比直接用
    len() 準確。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _styled_button(parent, text, command, bg, active_bg, fg="#ffffff", font=None,
                    outline=False, compact=False):
    """通用按鈕產生器——扁平（`relief="flat"`、無立體邊框），互動回饋靠滑鼠
    進出時把底色從 `bg` 切到 `active_bg`（比常態深一階），按下時 Tk 也會用
    `activebackground` 再壓一次。

    - `outline=True`：不是真的畫外框，而是「安靜」層級——一律走 `BTN_QUIET_*`
      這組扁平淺灰（忽略傳進來的 `bg`/`active_bg`）。給收尾動作用（取消、關閉、
      縮小視窗），仍是看得出來、點得下去的實心按鈕，只是刻意不搶眼。
    - `compact=True`：內距改用較小的間距（`SPACE_SM`/`SPACE_XS`），給空間較擠的
      地方用（例如緊貼在輸入框旁邊的「瀏覽...」按鈕）。

    邊框用 1px `highlightthickness` 畫（顏色＝`active_bg`，比底色深一階），讓按鈕
    在跟自己顏色接近的背景上仍有清楚的邊界；`relief` 全程 flat，沒有 raised
    那種 Win95 浮凸感。
    """
    padx = SPACE_SM if compact else SPACE_MD
    pady = SPACE_XS if compact else SPACE_SM
    if outline:
        bg, active_bg, fg = BTN_QUIET_BG, BTN_QUIET_ACTIVE, BTN_QUIET_FG
    border = active_bg

    btn = tk.Button(
        parent, text=text, command=command, bg=bg, fg=fg,
        activebackground=active_bg, activeforeground=fg,
        disabledforeground="#ccd3dc",
        relief="flat", bd=0,
        highlightthickness=1, highlightbackground=border, highlightcolor=border,
        font=font or ("Microsoft JhengHei", 12, "bold"),
        padx=padx, pady=pady, cursor="hand2",
    )

    def _hover(enter):
        # 進入時只在「可按」狀態才變深；離開時一律還原成常態底色——即使中途被
        # 設成 disabled，滑鼠移開後也不會卡在 hover 的深色上。
        if enter and str(btn["state"]) == "disabled":
            return
        btn.configure(bg=active_bg if enter else bg)

    btn.bind("<Enter>", lambda _e: _hover(True))
    btn.bind("<Leave>", lambda _e: _hover(False))
    return btn
