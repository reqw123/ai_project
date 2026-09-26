"""終端機面板的小型虛擬鍵盤。

按標題列的「⌨」鈕，在按鈕正上方彈出；按鍵會把字元打進終端機輸入框（插在游標位置，
跟實體鍵盤打字一樣），「↵」＝在輸入框按 Enter（送出給子行程）。

刻意用「疊在主視窗上的 Frame」（`place()` + `lift()`）而不是另開 Toplevel：
- 一定在主視窗範圍內（使用者要求），位置用 `_clamp()` 夾在視窗邊界裡，視窗縮放時重新定位；
- 不會搶走主視窗焦點、不會跑到別的螢幕或被其他視窗蓋住。
按鍵是 `tk.Button`（Windows 上滑鼠點擊不會把焦點移到按鈕），輸入框的游標位置得以保留。
"""

import tkinter as tk

_BG = "#262a2f"
_BORDER = "#5d7285"
_KEY_BG = "#3a4048"
_KEY_ACTIVE = "#4f5761"
_KEY_FG = "#f2f4f6"
_FN_BG = "#5d7285"          # 功能鍵（⌫ ⇧ 清除 關閉）
_FN_ACTIVE = "#4a5d6e"
_ENTER_BG = "#27ae60"
_ENTER_ACTIVE = "#219150"
_SHIFT_ON_BG = "#e67e22"    # Shift 開啟時：跟「等待輸入」提醒同一個琥珀色

KEY_SIZE = 30               # 一般按鍵邊長（px）
_GAP = 3
_MARGIN = 4                 # 鍵盤與視窗邊緣／錨點按鈕之間至少留的距離

# 每列：(顯示／輸入的字元 or 特殊鍵代號, 寬度倍數)
_ROWS = [
    [("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1), ("6", 1), ("7", 1), ("8", 1), ("9", 1), ("0", 1),
     ("BACKSPACE", 1.6)],
    [("q", 1), ("w", 1), ("e", 1), ("r", 1), ("t", 1), ("y", 1), ("u", 1), ("i", 1), ("o", 1), ("p", 1),
     ("-", 1), ("_", 1)],
    [("a", 1), ("s", 1), ("d", 1), ("f", 1), ("g", 1), ("h", 1), ("j", 1), ("k", 1), ("l", 1), (":", 1),
     ("ENTER", 1.6)],
    [("SHIFT", 1.6), ("z", 1), ("x", 1), ("c", 1), ("v", 1), ("b", 1), ("n", 1), ("m", 1), (".", 1), (",", 1),
     ("/", 1), ("\\", 1)],
    [("CLEAR", 1.6), ("SPACE", 6), ("(", 1), (")", 1), ("CLOSE", 1.6)],
]
_LABELS = {"BACKSPACE": "⌫", "ENTER": "↵ 送出", "SHIFT": "⇧", "SPACE": "空白", "CLEAR": "清除", "CLOSE": "✕"}
# Shift 開啟時非字母鍵改輸入的符號
_SHIFTED = {"1": "!", "2": "@", "3": "#", "4": "$", "5": "%", "6": "^", "7": "&", "8": "*", "9": "(",
            "0": ")", "-": "+", "_": "=", ":": ";", ".": ">", ",": "<", "/": "?", "\\": "|",
            "(": "[", ")": "]"}


class VirtualKeyboard:
    """window：要疊上去的主視窗（Tk/Toplevel）；entry：打字目標的 tk.Entry；
    anchor：彈出位置的基準 widget（鍵盤右下角對齊它的右上方）；
    on_enter：按「↵」時呼叫（通常是 ConsolePanel.on_send_stdin）；
    on_visibility(visible)：顯示／隱藏時通知（讓呼叫端更新「⌨」鈕的外觀）。"""

    def __init__(self, window, entry, anchor, on_enter, on_visibility=None, font_family="Consolas"):
        self.window = window
        self.entry = entry
        self.anchor = anchor
        self.on_enter = on_enter
        self.on_visibility = on_visibility
        self.shift = False
        self._font = (font_family, 10, "bold")
        self._px = tk.PhotoImage(width=1, height=1)  # 讓 tk.Button 的 width/height 以像素計
        self._char_btns = {}   # 原始字元 → Button（Shift 切換時換顯示文字）
        self._shift_btn = None
        self.frame = None
        self._configure_bind = None

    # ── 顯示／隱藏 ───────────────────────────────────────────────────

    @property
    def visible(self):
        return self.frame is not None and self.frame.winfo_ismapped()

    def toggle(self):
        if self.visible:
            self.hide()
        else:
            self.show()

    def show(self):
        if self.frame is None:
            self._build()
        self.frame.place(x=0, y=0)
        self.frame.lift()
        self.window.update_idletasks()
        self.reposition()
        if self._configure_bind is None:
            self._configure_bind = self.window.bind("<Configure>", self._on_window_configure, add="+")
        if self.on_visibility:
            self.on_visibility(True)

    def hide(self):
        if self.frame is not None and self.frame.winfo_ismapped():
            self.frame.place_forget()
            if self.on_visibility:
                self.on_visibility(False)

    def _on_window_configure(self, event):
        if event.widget is self.window and self.visible:
            self.reposition()

    @staticmethod
    def _clamp(anchor_right, anchor_top, kb_w, kb_h, win_w, win_h, margin=_MARGIN):
        """鍵盤左上角座標：右緣對齊錨點右緣、底緣在錨點上方 margin；超出視窗就往內推。"""
        x = anchor_right - kb_w
        y = anchor_top - margin - kb_h
        x = max(margin, min(x, win_w - kb_w - margin))
        y = max(margin, min(y, win_h - kb_h - margin))
        return x, y

    def reposition(self):
        if self.frame is None:
            return
        try:
            self.window.update_idletasks()
            ax = self.anchor.winfo_rootx() - self.window.winfo_rootx()
            ay = self.anchor.winfo_rooty() - self.window.winfo_rooty()
            x, y = self._clamp(
                ax + self.anchor.winfo_width(), ay,
                self.frame.winfo_reqwidth(), self.frame.winfo_reqheight(),
                self.window.winfo_width(), self.window.winfo_height(),
            )
            self.frame.place(x=x, y=y)
            self.frame.lift()
        except tk.TclError:
            pass  # 視窗已關閉

    # ── 建立按鍵 ─────────────────────────────────────────────────────

    def _build(self):
        self.frame = tk.Frame(
            self.window, bg=_BG, highlightthickness=1, highlightbackground=_BORDER, padx=6, pady=6,
        )
        tk.Label(
            self.frame, text="⌨ 虛擬鍵盤：按鍵會打進下方終端機輸入框，「↵ 送出」＝按 Enter（Esc 關閉）",
            bg=_BG, fg="#aab4be", font=("Microsoft JhengHei", 9), anchor="w",
        ).pack(fill="x", pady=(0, 4))
        for row in _ROWS:
            line = tk.Frame(self.frame, bg=_BG)
            line.pack(anchor="w", pady=(0, _GAP))
            for key, mult in row:
                self._make_key(line, key, mult).pack(side="left", padx=(0, _GAP))
        self.frame.bind("<Escape>", lambda _e: self.hide())
        self.entry.bind("<Escape>", lambda _e: self.hide(), add="+")

    def _make_key(self, parent, key, mult):
        width = int(KEY_SIZE * mult + _GAP * (mult - 1))
        bg, active = _KEY_BG, _KEY_ACTIVE
        if key == "ENTER":
            bg, active = _ENTER_BG, _ENTER_ACTIVE
        elif key in ("BACKSPACE", "SHIFT", "CLEAR", "CLOSE"):
            bg, active = _FN_BG, _FN_ACTIVE
        btn = tk.Button(
            parent, text=_LABELS.get(key, key), image=self._px, compound="center",
            width=width, height=KEY_SIZE, command=lambda k=key: self.press(k),
            bg=bg, fg=_KEY_FG, activebackground=active, activeforeground=_KEY_FG,
            relief="flat", bd=0, highlightthickness=0, cursor="hand2",
            font=self._font if key not in _LABELS else ("Microsoft JhengHei", 9, "bold"),
            takefocus=0,
        )
        if key == "SHIFT":
            self._shift_btn = btn
        elif key not in _LABELS:
            self._char_btns[key] = btn
        return btn

    # ── 按鍵動作 ─────────────────────────────────────────────────────

    def char_for(self, key):
        """目前 Shift 狀態下，這顆字元鍵實際輸入的字元。"""
        if not self.shift:
            return key
        return key.upper() if key.isalpha() else _SHIFTED.get(key, key)

    def press(self, key):
        e = self.entry
        try:
            if str(e["state"]) == "disabled":
                return
            if key == "ENTER":
                self.on_enter()
            elif key == "BACKSPACE":
                if e.selection_present():
                    e.delete("sel.first", "sel.last")
                else:
                    i = e.index("insert")
                    if i > 0:
                        e.delete(i - 1)
            elif key == "CLEAR":
                e.delete(0, "end")
            elif key == "SHIFT":
                self.set_shift(not self.shift)
            elif key == "CLOSE":
                self.hide()
            else:
                ch = " " if key == "SPACE" else self.char_for(key)
                if e.selection_present():
                    e.delete("sel.first", "sel.last")
                e.insert("insert", ch)
                e.xview("insert")
                if self.shift:
                    self.set_shift(False)  # 單次 Shift：打完一個字自動放開，跟手機鍵盤一樣
        except tk.TclError:
            pass

    def set_shift(self, on):
        self.shift = on
        for key, btn in self._char_btns.items():
            btn.config(text=self.char_for(key))
        if self._shift_btn is not None:
            self._shift_btn.config(bg=_SHIFT_ON_BG if on else _FN_BG)
