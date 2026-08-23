"""`settings_window.py` 與 `settings_gui` 底下各模組共用的樣式常數／小工具。

只搬了「其他 settings_gui 模組也會用到」的那一小撮常數過來（標題列配色、終端機
配色、按鈕配色、終端機尺寸與字級參數、說明卡片配色、間距比例），加上
`_styled_button()`——其餘只有 `settings_window.py` 自己用到的樣式常數（分頁代表色、
徽章配色、布林旗標配色等）仍留在 `settings_window.py` 裡，沒有必要搬。

**2026-08 版面美化**：按鈕顏色從原本 4 組（主要綠／次要藍灰／警告橘／中性灰）簡化
成 3 組——原本的「次要」跟「中性」語意上都是「不是主要也不是警告」，合併成一組
`BTN_SECONDARY_*`（沿用原本次要色的藍灰色調，比純灰更有辨識度、也更貼近整體深藍
色系的標題列）；`settings_window.py` 原本各自獨立定義的 `BTN_NEUTRAL_*` 已移除，
所有原本用中性灰的按鈕改用這裡的 `BTN_SECONDARY_*`。同時新增間距比例常數
（4 的倍數）跟 `_styled_button()` 的 `outline`/`compact` 兩個選項，取代原本「有的
按鈕用預設 padding、有的手動 `.config(pady=4)` 覆寫」這種各自為政的做法。
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

# ── 按鈕配色：主要（綠，儲存/啟動/執行這類正面且唯一的主要動作）／次要（藍灰，
# 其餘功能性動作跟中性/取消動作共用同一色相，用 outline 選項而非另開新顏色區分
# 「次要」跟「更不重要」）／警告（橘，還原預設值/關閉正在跑的行程這類要留意的
# 中斷性動作）——三組，不是四組。
BTN_PRIMARY_BG = "#27ae60"
BTN_PRIMARY_ACTIVE = "#219150"
BTN_SECONDARY_BG = "#5d7285"
BTN_SECONDARY_ACTIVE = "#4a5d6e"
# 橘色（BTN_WARN_BG，settings_window.py 裡定義）配白字對比不夠，字看起來會糊/淡
# （#e67e22 對純白文字的對比度偏低，接近但不到 WCAG AA 門檻）。這幾個警告色按鈕
# 改用深棕色文字，跟橘色底的對比明顯提升，其餘（綠/藍灰）底色夠深，白字對比
# 已經足夠，不用跟著改。
BTN_WARN_FG = "#3a1f00"
_GHOST_HOVER_BG = "#eef1f5"  # outline 按鈕 hover 時的淡灰底，不管邊框用哪個顏色都套同一個 hover 底色

# 淡藍色：目前只有「關閉 main.py」「停止腳本」這兩顆中斷性動作按鈕在用——原本
# 跟「還原 GUI 預設值」共用橘色警告色，使用者指定要跟那顆分開、改成淡藍色。跟
# BTN_SECONDARY 的藍灰色刻意拉開一點飽和度差異，不會被誤認成同一組次要按鈕。
#
# 第一版（#3b82c4 配白字）使用者回報字會糊——算過對比度才知道原因：這個亮度的
# 藍不上不下，中等亮度的底色配黑字或白字對比都不夠好（不是隨便挑深/淺文字色
# 就能解決，底色本身要挑到明顯偏亮或偏暗的一端）。改成真正的淡（偏亮）藍當底色，
# 配深藍字（跟橘色底配深棕字是同一個邏輯，但這次底色改亮，字就要改深，方向
# 相反）——量過對比度：白底 #eef2f6 至 BTN_INFO_BG 的相對亮度落在偏亮那一端，
# BTN_INFO_FG 這個深藏青跟它的對比度約 6.3，比第一版的白字（約 4.1，未達 WCAG
# AA 的 4.5 門檻）明顯好上不少。
BTN_INFO_BG = "#8ec4e8"
BTN_INFO_ACTIVE = "#6badd9"
BTN_INFO_FG = "#0d3a5c"

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
    """通用按鈕產生器。

    - `outline=True`：畫成「幽靈按鈕」——底色跟 `parent` 背景融為一體、只有邊框跟
      文字用 `bg` 這個顏色（`active_bg` 這時候不會用到）。給整排按鈕裡「最不重要」
      的收尾動作用（取消、關閉），不用為了再降一級重要性又發明一個新色相，靠
      「有沒有實心填色」這個更輕的視覺差異就夠分辨了。
    - `compact=True`：內距改用較小的間距（`SPACE_SM`/`SPACE_XS`），給空間較擠的
      地方用（例如緊貼在輸入框旁邊的「瀏覽...」按鈕），跟一般按鈕
      （`SPACE_MD`/`SPACE_SM`）拉開一個明確的兩級尺寸系統，不再各自寫死數字。

    實心按鈕（`outline=False`，多數情況）用 `relief="raised"` + `bd=2` 做出立體感
    （亮邊在左上、暗邊在右下，按下去時 Tk 會自動反過來變凹陷，有按壓回饋）——
    Tkinter 沒有真的陰影可以畫，這是這個工具箱能做到「有立體感」最直接的方式。
    outline 按鈕維持純邊框、不加浮凸，因為它本來就是刻意做得比較「輕」的次要
    按鈕，浮凸感會跟它「不重要」的視覺定位互相矛盾。
    """
    padx = SPACE_SM if compact else SPACE_MD
    pady = SPACE_XS if compact else SPACE_SM
    if outline:
        return tk.Button(
            parent, text=text, command=command, bg=parent.cget("bg"), fg=bg,
            activebackground=_GHOST_HOVER_BG, activeforeground=bg,
            relief="solid", bd=1, highlightthickness=0,
            font=font or ("Microsoft JhengHei", 12, "bold"), padx=padx, pady=pady, cursor="hand2",
        )
    return tk.Button(
        parent, text=text, command=command, bg=bg, fg=fg,
        activebackground=active_bg, activeforeground=fg, relief="raised", bd=2,
        font=font or ("Microsoft JhengHei", 12, "bold"), padx=padx, pady=pady, cursor="hand2",
    )
