"""全域欄位搜尋列：跨 11 個分頁比對欄位「顯示標籤／JSON key／環境變數名稱／說明文字（hint）」，
停止打字約 300ms 後（debounce）自動跳到第一個符合欄位所在的分頁並捲動定位。只要
搜尋框內還有字元，「所有」分頁裡的符合欄位都會一直保持高亮外框（切到哪個分頁就
看得到那個分頁被框起來的欄位），並在各分頁按鈕上顯示符合數量徽章；清空搜尋框才
會一次拿掉全部高亮與徽章。

這個 class 只管：搜尋欄本身的 Entry 元件、debounce 計時、比對 FIELD_SCHEMA 算出
「符合的分頁與數量」＋「第一個符合結果」——實際畫面效果（切換分頁、幫欄位加
高亮外框、把分頁按鈕文字加上數量後綴、捲動定位）都透過呼叫 `window` 上四個窄
接口方法完成（`_select_tab`／`_set_tab_match_badges`／`_highlight_fields`／
`_scroll_field_into_view`），不直接碰 Tkinter 版面細節，理由跟 ConsolePanel／
ProcessManager 一致：這個 class 不知道也不需要知道分頁按鈕、欄位列實際長什麼樣。
"""

import tkinter as tk

from settings_manager import FIELD_SCHEMA, TAB_ORDER

_DEBOUNCE_MS = 300


class FieldSearchBar:
    def __init__(self, window, parent, bg):
        self.window = window
        self.var = tk.StringVar()
        self._debounce_job = None
        self._build(parent, bg)

    def _build(self, parent, bg):
        tk.Label(
            parent, text="🔍", bg=bg, font=self.window._font_label,
        ).pack(side="right", padx=(0, 4), pady=4)
        entry = tk.Entry(parent, textvariable=self.var, font=self.window._font_label, width=22)
        entry.pack(side="right", padx=(6, 2), pady=4, ipady=2)
        entry.bind("<Escape>", lambda _e: self.var.set(""))
        self.var.trace_add("write", self._on_change)

    def _on_change(self, *_a):
        if self._debounce_job is not None:
            self.window.after_cancel(self._debounce_job)
        self._debounce_job = self.window.after(_DEBOUNCE_MS, self._run_search)

    def _run_search(self):
        self._debounce_job = None
        query = self.var.get().strip().lower()
        if not query:
            self.window._set_tab_match_badges({})
            self.window._highlight_fields([])
            return

        matches_by_tab: dict[str, list[str]] = {}
        for field in FIELD_SCHEMA:
            haystack = (
                f"{field['label']} {field['json_key']} "
                f"{field.get('env_var') or ''} {field.get('hint') or ''}"
            ).lower()
            if query in haystack:
                matches_by_tab.setdefault(field["tab"], []).append(field["json_key"])

        self.window._set_tab_match_badges({tab: len(keys) for tab, keys in matches_by_tab.items()})

        if not matches_by_tab:
            self.window._highlight_fields([])
            return

        # 依 TAB_ORDER（分頁按鈕實際排列順序）找第一個有符合結果的分頁，
        # 「第一個」的定義才會跟使用者眼睛看到的分頁順序一致。
        first_tab = next(tab for tab in TAB_ORDER if tab in matches_by_tab)
        matched_keys = matches_by_tab[first_tab]
        self.window._select_tab(first_tab)
        # 高亮「所有」分頁的符合欄位，不只自動跳過去的第一個分頁——已建好的分頁
        # 直接套上（切分頁只是 pack_forget，widget 不會被銷毀），還沒建的分頁
        # （設定視窗的分頁是延遲建立的）由 _ensure_tab_built() 建好時補套，所以跨
        # 分頁高亮可以一路保持到搜尋框被清空為止，使用者切到任一有徽章的分頁都
        # 看得到命中的欄位被框起來，不會「一離開第一個分頁就沒框了」。
        all_matched_keys = [k for keys in matches_by_tab.values() for k in keys]
        self.window._highlight_fields(all_matched_keys)
        self.window._scroll_field_into_view(first_tab, matched_keys[0])
