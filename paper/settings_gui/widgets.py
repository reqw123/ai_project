"""自製的 Tkinter Canvas 手繪元件（圓角按鈕／圓角狀態徽章），取代 Tkinter 原生
`tk.Button`／`tk.Label` 畫不出來的視覺效果。

**從 style.py 搬出來（2026-09）**：這兩個元件原本寫在 `style.py`，但 `style.py`
自己的定位是「共用樣式常數」——純資料，不該混進兩個完整的自製 widget 實作（各自
有事件綁定、繪圖邏輯、`config()` 覆寫，合計曾經佔 `style.py` 六成以上的行數）。
專案裡其他每一支「有畫 widget」的檔案（`console_panel.py`／`dialogs.py`／
`image_popup.py`／`tool_order.py`）都是一支檔案負責一個 widget，這裡比照辦理，把
「畫圓角圖形」這件事獨立成自己的模組；顏色/間距常數（`BTN_PRIMARY_BG`／
`SPACE_SM` 之類）仍然從 `settings_gui.style` import，不重複定義。

**2026-09 按鈕改款：立體浮雕 → 平面＋膠囊圓角**：使用者不喜歡原本 `relief="raised"`
＋`bd=2` 那種「框線明顯、正正方方」的立體感。Tkinter 的 `tk.Button` 沒有真的圓角可
畫（`relief`/`bd` 只能做方形的浮雕/平面/凹陷），所以改成 `_PillButton`——一個繼承
`tk.Canvas`、自己用 `create_polygon(..., smooth=True)` 手繪圓角矩形（半徑吃滿整個
高度＝膠囊形狀）再疊一層置中文字的自製按鈕元件。`_styled_button()` 的函式簽章、
回傳值支援的操作（`.pack()`／`.config(text=...)`／`.config(state=...)`）都刻意跟
原本的 `tk.Button` 版本相容，全站 36 處呼叫點不用改一行；已知不相容之處：不支援
`.cget()` 讀回 `text`/`state`（專案裡目前沒有地方這樣用），也沒有實作 `.config()`
之外的其他 `tk.Button` 專屬 option（`command` 只能在建立時給，事後不能換）。

**2026-09 欄位來源徽章改款**：設定分頁每個欄位列最右邊「[JSON]」「[環境變數]」
這種來源標籤，原本是純文字 `tk.Label` 硬加方括號充當「這是個標籤」的視覺提示，
使用者覺得不好看。改成 `_StatusBadge`——沿用 `_PillButton` 的圓角矩形手繪技巧，
但形狀是固定小圓角（不是膠囊）、不接受點擊，改綁 `textvariable` 自動跟著內容
變動重繪；呼叫端原本 `badge_var.set(...)` 的文字不再需要自己包方括號（圓角色塊
本身就是視覺框線，方括號變成多餘的 ASCII 裝飾）。
"""

import tkinter as tk
from tkinter import font as tkfont

from settings_gui.style import SPACE_XS, SPACE_SM, SPACE_MD, _GHOST_HOVER_BG


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(round(c)))) for c in rgb))


def _blend(hex_a, hex_b, t):
    """兩個顏色的線性內插：t=0 回傳 hex_a、t=1 回傳 hex_b。用來算 disabled 狀態的
    「淡化」顏色（往背景色混）跟 pressed 狀態的「壓暗」顏色（往黑色混），不用另外
    為每個色階手動配一個新的十六進位色碼。"""
    a, b = _hex_to_rgb(hex_a), _hex_to_rgb(hex_b)
    return _rgb_to_hex(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))


def _rounded_rect_points(x1, y1, x2, y2, r):
    """`create_polygon(..., smooth=True)` 要的控制點清單——在四個角各放兩個點，
    中間拉平滑曲線，是 Tkinter 畫圓角矩形的標準土法煉鋼技巧。`_PillButton` 跟
    `_StatusBadge` 共用同一份，不重複寫兩次。"""
    return [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]


class _PillButton(tk.Canvas):
    """平面＋膠囊圓角按鈕——取代原本 `relief="raised"` 的立體浮雕框線樣式。

    Tkinter 的 `tk.Button` 畫不出圓角（`relief`/`bd` 只有方形的浮雕/平面/凹陷可選），
    所以這裡繼承 `tk.Canvas`，自己用 `create_polygon(..., smooth=True)`
    （在四個角的控制點之間拉一條平滑曲線，是 Tkinter 畫「圓角矩形」的標準土法煉鋼
    技巧）手繪一個圓角半徑等於整個高度一半的膠囊形狀，再疊一層置中文字上去。

    盡量相容 `tk.Button` 在本專案裡實際用到的操作——`.pack()`/`.grid()` 直接繼承
    自 `tk.Canvas` 就有；`.config(text=...)` 換文字、`.config(state="disabled"/
    "normal")` 切換可否點擊都有實作並會觸發重繪。`command` 只能在建立時給、事後
    不能換，`.cget()` 也沒有覆寫成看得懂 `text`/`state`——這兩點目前專案裡都沒有
    地方會用到，等真的需要再補。
    """

    def __init__(self, parent, text, command, bg, active_bg, fg="#ffffff", font=None,
                 outline=False, compact=False):
        self._bg = bg
        self._active_bg = active_bg
        self._fg = fg
        self._outline = outline
        self._parent_bg = parent.cget("bg")
        self._command = command
        self._text = text
        self._font = font or ("Microsoft JhengHei", 12, "bold")
        self._state = "normal"
        self._hover = False
        self._pressed = False

        measurer = tkfont.Font(font=self._font)
        text_w = measurer.measure(text)
        text_h = measurer.metrics("linespace")
        pady = SPACE_XS if compact else SPACE_SM
        # 膠囊形狀兩端是半圓，水平內距要比一般矩形按鈕多留一些，文字才不會看起來
        # 被兩端的弧線「吃」到——用縱向內距的高度換算，圓角越高、多留的量跟著增加。
        padx = (SPACE_SM if compact else SPACE_MD) + pady
        width = text_w + padx * 2
        height = text_h + pady * 2
        self._radius = height / 2

        super().__init__(
            parent, width=width, height=height, bg=self._parent_bg,
            highlightthickness=0, bd=0, cursor="hand2",
        )
        self._redraw()

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    # ── 事件 ─────────────────────────────────────────────────────────
    def _on_enter(self, _event=None):
        if self._state == "disabled":
            return
        self._hover = True
        self._redraw()

    def _on_leave(self, _event=None):
        self._hover = False
        self._pressed = False
        self._redraw()

    def _on_press(self, _event=None):
        if self._state == "disabled":
            return
        self._pressed = True
        self._redraw()

    def _on_release(self, _event=None):
        if self._state == "disabled":
            return
        was_pressed = self._pressed
        self._pressed = False
        self._redraw()
        # 只有「按下」跟「放開」都發生在按鈕上才算一次點擊（跟原生 tk.Button 一致）
        # ——避免在按鈕上按下、拖到別處放開卻仍觸發的誤按。
        if was_pressed and self._command is not None:
            self._command()

    # ── 畫圖 ─────────────────────────────────────────────────────────
    def _redraw(self):
        self.delete("all")
        w, h = int(self["width"]), int(self["height"])
        r = self._radius
        points = _rounded_rect_points(1, 1, w - 1, h - 1, r)

        if self._outline:
            border_color = self._bg
            if self._state == "disabled":
                border_color = _blend(border_color, self._parent_bg, 0.6)
                text_color = border_color
                fill = self._parent_bg
            else:
                fill = _GHOST_HOVER_BG if self._hover else self._parent_bg
                text_color = self._bg
            self.create_polygon(
                points, smooth=True, fill=fill, outline=border_color, width=1.4,
            )
        else:
            fill = self._bg
            if self._state == "disabled":
                fill = _blend(self._bg, self._parent_bg, 0.55)
                text_color = "#9aa7b0"
            else:
                if self._pressed:
                    fill = _blend(self._active_bg, "#000000", 0.12)
                elif self._hover:
                    fill = self._active_bg
                text_color = self._fg
            self.create_polygon(points, smooth=True, fill=fill, outline="")

        self.create_text(w / 2, h / 2, text=self._text, fill=text_color, font=self._font)
        self.configure(cursor="arrow" if self._state == "disabled" else "hand2")

    # ── 對外相容介面 ─────────────────────────────────────────────────
    def config(self, **kwargs):
        redraw = False
        if "text" in kwargs:
            self._text = kwargs.pop("text")
            redraw = True
        if "state" in kwargs:
            self._state = kwargs.pop("state")
            redraw = True
        if kwargs:
            super().config(**kwargs)
        if redraw:
            self._redraw()

    configure = config


def _styled_button(parent, text, command, bg, active_bg, fg="#ffffff", font=None,
                    outline=False, compact=False):
    """通用按鈕產生器，回傳一個 `_PillButton`（平面＋膠囊圓角，見該類別的說明）。

    - `outline=True`：畫成「幽靈按鈕」——底色跟 `parent` 背景融為一體、只有邊框跟
      文字用 `bg` 這個顏色（`active_bg` 這時候不會用到）。給整排按鈕裡「最不重要」
      的收尾動作用（取消、關閉），不用為了再降一級重要性又發明一個新色相，靠
      「有沒有實心填色」這個更輕的視覺差異就夠分辨了。
    - `compact=True`：內距改用較小的間距（`SPACE_SM`/`SPACE_XS`），給空間較擠的
      地方用（例如緊貼在輸入框旁邊的「瀏覽...」按鈕），跟一般按鈕
      （`SPACE_MD`/`SPACE_SM`）拉開一個明確的兩級尺寸系統，不再各自寫死數字。
    """
    return _PillButton(
        parent, text, command, bg, active_bg, fg=fg, font=font,
        outline=outline, compact=compact,
    )


class _StatusBadge(tk.Canvas):
    """小圓角狀態徽章（例如設定分頁裡每個欄位列最右邊的「來源標籤」：目前值是
    來自環境變數、runtime_settings.current.json、表單暫存還是內建預設值）。跟
    `_PillButton` 系出同門——一樣是用 `_rounded_rect_points()` 手繪圓角矩形，
    但這裡純顯示、不接受點擊，形狀也收斂成小圓角矩形而非膠囊（半徑固定一個較
    小的值），視覺上明確是「附加在欄位旁的標籤」，跟主要操作用的膠囊按鈕做出
    區隔，不會兩種圓角元件混在一起分不出誰是誰。

    改綁 `textvariable`：文字內容用 `trace_add("write", ...)` 跟著 `StringVar`
    自動更新並重繪，呼叫端不用自己處理「文字變了要重畫」這件事，也不用在
    widget 被銷毀時記得解除 trace——這裡用 `<Destroy>` 事件自動清掉。
    """

    _RADIUS = 6  # 固定用小圓角矩形，不像 _PillButton 用「半高」做成膠囊——見上方說明

    def __init__(self, parent, textvariable, bg, fg, font=None, padx=None, pady=None):
        self._parent_bg = parent.cget("bg")
        self._font = font or ("Microsoft JhengHei", 10)
        self._bg = bg
        self._fg = fg
        self._padx = SPACE_SM if padx is None else padx
        self._pady = 2 if pady is None else pady
        self._var = textvariable
        self._text = textvariable.get()

        super().__init__(parent, highlightthickness=0, bd=0, bg=self._parent_bg)
        self._redraw()

        self._trace_id = textvariable.trace_add("write", self._on_var_change)
        self.bind("<Destroy>", self._on_destroy)

    def _on_var_change(self, *_a):
        self._text = self._var.get()
        self._redraw()

    def _on_destroy(self, _event=None):
        try:
            self._var.trace_remove("write", self._trace_id)
        except Exception:
            pass

    def _redraw(self):
        self.delete("all")
        measurer = tkfont.Font(font=self._font)
        text_w = measurer.measure(self._text)
        text_h = measurer.metrics("linespace")
        w = text_w + self._padx * 2
        h = text_h + self._pady * 2
        self.configure(width=w, height=h)
        if not self._text:
            # 沒有來源標籤可顯示（例如 source 判斷邏輯以外的情況）時，畫一個
            # 0 大小的空 canvas 而不是留一個實色小方塊——跟「本來就沒有徽章」
            # 視覺上一致。
            return
        points = _rounded_rect_points(0, 0, max(w, 1), max(h, 1), self._RADIUS)
        self.create_polygon(points, smooth=True, fill=self._bg, outline="")
        self.create_text(w / 2, h / 2, text=self._text, fill=self._fg, font=self._font)

    def config(self, **kwargs):
        redraw = False
        if "bg" in kwargs:
            self._bg = kwargs.pop("bg")
            redraw = True
        if "fg" in kwargs:
            self._fg = kwargs.pop("fg")
            redraw = True
        if kwargs:
            super().config(**kwargs)
        if redraw:
            self._redraw()

    configure = config


def _styled_badge(parent, textvariable, bg, fg, font=None):
    """建立一個 `_StatusBadge`（小圓角狀態徽章，見該類別說明）。`bg`/`fg` 是初始
    顏色，之後可以用 `.config(bg=..., fg=...)` 動態換色（例如同一個欄位的來源
    從「JSON」變成「環境變數」時，顏色要跟著換但文字/形狀邏輯不變）。"""
    return _StatusBadge(parent, textvariable, bg, fg, font=font)
