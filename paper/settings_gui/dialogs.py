"""自訂訊息彈窗——取代 `tkinter.messagebox` 的原生（灰底、系統字型、跟本視窗其餘
畫面風格完全對不上）樣式。本模組是「訊息彈窗」這件事唯一負責的地方：`settings_window.py`
裡所有原本呼叫 `messagebox.showinfo/showwarning/showerror/askyesno` 的地方，一律改成
呼叫這裡的 `show_info/show_warning/show_error/ask_yesno`，介面刻意設計成跟
`messagebox` 對應函式相同的呼叫方式（多一個 `parent` 放第一個參數），逐一替換時
不用重新設計呼叫端的邏輯。

視覺風格沿用 `settings_gui/style.py` 的配色系統（標題列深藍、主要動作綠色、次要
動作藍灰）與 `settings_gui/widgets.py` 的 `_styled_button()`（膠囊圓角按鈕）；
不足的部分（訊息本文配色、各種類的強調色）另外在這裡定義一組小型、本模組專用的
色盤——沒有搬進 `style.py` 共用，因為這幾個顏色目前只有這裡用得到（跟 `style.py`
開頭註解「只搬其他模組也會用到的常數過來」的原則一致）。

彈跳視窗保留 OS 原生的視窗框（可拖曳、右上角 X 可關、工作列行為正常），只有
「裡面畫的內容」換成自訂樣式——不做成完全無邊框視窗，避免額外重做拖曳/最小化
這類原生視窗管理員本來就處理好的行為，得不償失。

用法（都是同步呼叫，跟 messagebox 一樣，函式內部會擋住直到使用者關閉彈窗）：

    from settings_gui import dialogs

    dialogs.show_info(self, "儲存設定", "已儲存。")
    dialogs.show_warning(self, "執行腳本", "請先選擇腳本。")
    dialogs.show_error(self, "執行腳本", f"找不到檔案：\\n{path}")
    if not dialogs.ask_yesno(self, "關閉視窗", "確定要關閉嗎？"):
        return
"""

import tkinter as tk
from tkinter import font as tkfont

from settings_gui.style import (
    BTN_PRIMARY_BG,
    BTN_PRIMARY_ACTIVE,
    BTN_SECONDARY_BG,
    BTN_SECONDARY_ACTIVE,
    SPACE_SM,
    SPACE_MD,
    SPACE_LG,
)
from settings_gui.widgets import _styled_button

_FONT_FAMILY = "Microsoft JhengHei"
_BODY_BG = "#ffffff"
_TEXT_FG = "#20303f"
_MESSAGE_WRAPLENGTH = 420  # 跟訊息內容的字級／視窗寬度抓一個閱讀起來舒服的折行寬度

# 警告用的橘色跟白字對比不夠（跟 style.py 裡 BTN_WARN_FG 的理由相同），這裡沿用
# 深棕字搭配；error/info/question 三種底色夠深，白字/深藍字對比已經足夠。
_KIND_STYLE = {
    "info":     {"icon": "ℹ️", "accent": "#2c4053", "strip": "#5d7285"},
    "warning":  {"icon": "⚠️", "accent": "#b36b00", "strip": "#e67e22"},
    "error":    {"icon": "❌", "accent": "#c0392b", "strip": "#c0392b"},
    "question": {"icon": "❓", "accent": "#1b4f72", "strip": "#1b4f72"},
}

_BTN_WARN_BG = "#e67e22"
_BTN_WARN_ACTIVE = "#cf711d"
_BTN_WARN_FG = "#3a1f00"
_BTN_ERROR_BG = "#c0392b"
_BTN_ERROR_ACTIVE = "#a93226"


def _run_dialog(parent, title, message, kind, buttons, default_value):
    """畫出一個彈跳視窗、擋住直到使用者關閉，回傳被按下的按鈕對應的值；用 X 關掉
    或按 Escape 一律視同取消，回傳 None（`ask_yesno` 會把 None 轉成 False）。

    buttons: [(標籤文字, 回傳值, 底色, 按下時底色, 文字色), ...]，視覺上「由左到右」
    剛好是傳入順序的反過來——清單第一個元素會被排到最右邊，最後一個排到最左邊
    （見迴圈裡的說明），所以「否／取消」放清單第一個、「是／確定」放最後一個。
    """
    style = _KIND_STYLE[kind]
    result = {"value": None}

    top = tk.Toplevel(parent)
    top.withdraw()  # 內容還沒排版定案前先不顯示，避免先在左上角閃一下再跳到置中位置
    top.title(title)
    top.configure(bg=_BODY_BG)
    top.resizable(False, False)
    top.transient(parent)

    font_title = tkfont.Font(family=_FONT_FAMILY, size=13, weight="bold")
    font_body = tkfont.Font(family=_FONT_FAMILY, size=12)

    # 頂部強調色細條——一眼就能從顏色分辨「這是警告還是錯誤還是單純詢問」，
    # 不用先讀完文字才知道嚴重程度。
    tk.Frame(top, bg=style["strip"], height=4).pack(fill="x")

    header = tk.Frame(top, bg=_BODY_BG)
    header.pack(fill="x", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM))
    tk.Label(
        header, text=style["icon"], bg=_BODY_BG, font=("Segoe UI Emoji", 26),
    ).pack(side="left", padx=(0, SPACE_MD))
    tk.Label(
        header, text=title, bg=_BODY_BG, fg=style["accent"], font=font_title,
        anchor="w", justify="left", wraplength=_MESSAGE_WRAPLENGTH,
    ).pack(side="left", fill="x", expand=True)

    body = tk.Frame(top, bg=_BODY_BG)
    body.pack(fill="both", expand=True, padx=SPACE_LG, pady=(0, SPACE_MD))
    tk.Label(
        body, text=message, bg=_BODY_BG, fg=_TEXT_FG, font=font_body,
        anchor="w", justify="left", wraplength=_MESSAGE_WRAPLENGTH,
    ).pack(fill="both", expand=True)

    btn_row = tk.Frame(top, bg=_BODY_BG)
    btn_row.pack(fill="x", padx=SPACE_LG, pady=(0, SPACE_LG))

    def _finish(value):
        result["value"] = value
        top.destroy()

    # 每顆按鈕都用 side="right" pack：清單裡第一個元素最先被排到最右邊，後面的
    # 元素依序排到它左邊，所以視覺上「清單第一個在右、最後一個在左」——呼叫端
    # 把「否／取消」放在清單第一個、「是／確定」放在最後一個，畫出來就是
    # Windows 慣例的「確定／是 在左、取消／否 在右」。
    for label, value, bg, active_bg, fg in buttons:
        _styled_button(
            btn_row, label, lambda v=value: _finish(v), bg, active_bg, fg=fg, font=font_body,
        ).pack(side="right", padx=(SPACE_SM, 0))

    top.protocol("WM_DELETE_WINDOW", lambda: _finish(None))
    top.bind("<Escape>", lambda _e: _finish(None))
    top.bind("<Return>", lambda _e: _finish(default_value))

    top.update_idletasks()
    w, h = top.winfo_reqwidth(), top.winfo_reqheight()
    px, py = parent.winfo_rootx(), parent.winfo_rooty()
    pw, ph = parent.winfo_width(), parent.winfo_height()
    x = px + max(0, (pw - w) // 2)
    y = py + max(0, (ph - h) // 3)  # 略偏上半部，比正中央更符合對話框慣例
    top.geometry(f"{w}x{h}+{x}+{y}")
    top.deiconify()

    top.grab_set()
    top.focus_set()
    top.wait_window()
    return result["value"]


def show_info(parent, title, message):
    """對應 `messagebox.showinfo`；單顆「確定」按鈕，無回傳值。"""
    _run_dialog(
        parent, title, message, kind="info",
        buttons=[("確定", True, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE, "#ffffff")],
        default_value=True,
    )


def show_warning(parent, title, message):
    """對應 `messagebox.showwarning`；單顆「確定」按鈕，無回傳值。"""
    _run_dialog(
        parent, title, message, kind="warning",
        buttons=[("確定", True, _BTN_WARN_BG, _BTN_WARN_ACTIVE, _BTN_WARN_FG)],
        default_value=True,
    )


def show_error(parent, title, message):
    """對應 `messagebox.showerror`；單顆「確定」按鈕，無回傳值。"""
    _run_dialog(
        parent, title, message, kind="error",
        buttons=[("確定", True, _BTN_ERROR_BG, _BTN_ERROR_ACTIVE, "#ffffff")],
        default_value=True,
    )


def ask_yesno(parent, title, message) -> bool:
    """對應 `messagebox.askyesno`；「否」「是」兩顆按鈕，回傳 True/False。
    用 X 關閉或按 Escape 視同「否」。"""
    result = _run_dialog(
        parent, title, message, kind="question",
        buttons=[
            ("否", False, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, "#ffffff"),
            ("是", True, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE, "#ffffff"),
        ],
        default_value=True,
    )
    return bool(result)
