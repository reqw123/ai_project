"""設定視窗「📡 IoT 子系統」分頁：各感測器、飼主語音紀錄的啟動口（畫面；邏輯在 ``iot_services.py``）。

這個分頁沒有設定欄位（不在 settings_manager 的 TAB_ORDER／FIELD_SCHEMA 裡），settings_window.py 只在
「建分頁內容」時把整頁交給 ``build()``，關閉視窗時呼叫 ``shutdown_services()``。
每個服務一張卡片：狀態、▶ 啟動、■ 停止、📄 紀錄；下方是選到的服務的紀錄（字體可放大縮小、高度可拖拉）。

版面（09-27 改）：整個分頁固定填滿「分頁頂端 → 全域終端機上緣」，自己不跟著分頁捲動——
上半是卡片區（內容多時自己捲），下半是紀錄框，紀錄框的標題列就是拖拉把手（往上拖＝變高，
跟全域終端機一樣）。全域終端機變高時先縮卡片區，紀錄框不會被它蓋住。

09-27 起：關掉設定視窗時，執行中的 IoT 服務一律強制關閉（使用者要求）。
09-27 起：工具列「⚙ 參數設定」覆寫 iot/config.py 的參數（``iot_config_dialog.py``／``iot_config_overrides.py``）。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

from settings_gui import dialogs, iot_config_dialog, iot_services
from settings_gui import ui_state as _ui_state
from settings_gui.style import (
    BTN_PRIMARY_ACTIVE, BTN_PRIMARY_BG, BTN_SECONDARY_ACTIVE, BTN_SECONDARY_BG,
    COLOR_CONSOLE_BG, COLOR_CONSOLE_FG, SPACE_LG, SPACE_MD, SPACE_SM, SPACE_XS,
)
from settings_gui.widgets import _styled_button

TAB_NAME = "IoT 子系統"
TAB_EMOJI = "📡"   # Tk 顯示不了帶異體字選擇符（U+FE0F）的 emoji，選不需要它的
TAB_ACCENT = "#0e6655"

_FONT = "Microsoft JhengHei"
_BG = "#f3f6f8"            # 分頁底色（淡灰藍，卡片是白的才浮得出來）
_CARD_BG = "#ffffff"
_CARD_BORDER = "#dbe3e9"
_FG = "#1f2d3a"
_HINT = "#5f7182"
_MONO_BG = "#eef2f5"
_STOP_BG, _STOP_ACTIVE = "#c0392b", "#a93226"
_PILL = {   # 狀態膠囊：底色、字色
    "on": ("#e3f5e9", "#1e7b45"), "off": ("#edf0f3", "#6c7a86"),
    "ok": ("#e3f5e9", "#1e7b45"), "bad": ("#fdecea", "#b03a2e"),
}
_SERVICE_COLOR = {"env": "#16a085", "motion": "#2e86c1", "weight": "#d68910", "bodytemp": "#cb4335", "voice": "#7d3c98"}
# 深色終端機上的字色：連線檢查報告（全域終端機）、紀錄框裡的 WARNING／ERROR 行
_REPORT_COLORS = {"iot_ok": "#5dd39e", "iot_warn": "#f5c451", "iot_bad": "#ff6b6b", "iot_info": "#9fb3c8"}
# 整合檢視：每行前面「[服務]」的字色（深色終端機底；刻意不用紅色，紅色留給 WARNING 行）
_MERGED_PREFIX_COLOR = {"env": "#48c9b0", "bodytemp": "#f0b27a", "motion": "#5dade2", "weight": "#f7dc6f",
                        "voice": "#c39bd3"}
_LOG_WARN_MARKS = ("[WARNING]", "[ERROR]", "[CRITICAL]", "Traceback")
_REFRESH_MS = 2000    # 服務狀態、broker（要掃行程，比較慢）
_LOG_POLL_MS = 500     # 紀錄框：看紀錄檔有沒有變（只 stat）

# 紀錄面板：字體大小、高度（記在 settings_gui/ui_state.json，下次開視窗沿用）
_LOG_FONT_DEFAULT, _LOG_FONT_MIN, _LOG_FONT_MAX = 12, 7, 30
# 紀錄框高度記「佔分頁可用高度的比例」：全域終端機變高、分頁變矮時，卡片區和紀錄框一起等比例縮。
# 比例可以超過 1：紀錄框往上拉超過分頁時會蓋住卡片、分頁列，最高到跟全域終端機一樣的上限（09-27 使用者要求）
_LOG_RATIO_DEFAULT, _LOG_RATIO_MIN, _LOG_RATIO_MAX = 0.45, 0.10, 10.0
_LOG_HEIGHT_MIN = 100  # 紀錄框（含標題列）最矮
_PAGE_MIN = 200        # 全域終端機拉到很高時，分頁最少還是這麼高（再矮就讓終端機蓋住）
_GRIP_BG, _GRIP_HOVER = "#dde5eb", "#c8d4dd"


def _state_num(key, default, lo, hi, cast=int):
    try:
        return max(lo, min(hi, cast(_ui_state.get(key, str(default)))))
    except (TypeError, ValueError):
        return default


class IotTab:
    def __init__(self, parent, window):
        self.window = window
        self.selected = iot_services.SERVICES[0].key
        self._status_vars = {}
        self._status_labels = {}
        self._cards = {}
        self._ip_vars = {}      # 卡片標題列的裝置 IP 膠囊
        self._log_cleared = {}  # 服務 → 紀錄框按「清空」時的紀錄檔位置（只清畫面，這次開視窗有效）
        self._merged = _ui_state.get("iot_log_merged", "0") == "1"   # 整合檢視：全部服務依時間合併（記住上次）
        self._ip_labels = {}
        self._busy = False
        self._shown_log = None
        f = lambda size, weight="normal": tkfont.Font(root=parent, family=_FONT, size=size, weight=weight)   # noqa: E731
        self._f_title = f(17, "bold")
        self._f_card = f(15, "bold")
        self._f_body, self._f_pill, self._f_btn = f(12), f(12, "bold"), f(12, "bold")
        self._f_mono = tkfont.Font(root=parent, family="Consolas", size=11)
        self._log_font = tkfont.Font(root=parent, family="Consolas",
                                     size=_state_num("iot_log_font", _LOG_FONT_DEFAULT, _LOG_FONT_MIN, _LOG_FONT_MAX))

        # 整頁高度由 _fit 設成「看得到的高度」（不讓外層分頁捲動），裡面的東西不能把它撐大
        root = tk.Frame(parent, bg=_BG, height=600)
        root.pack(fill="both", expand=True)
        root.pack_propagate(False)
        self.root = root
        self._outer = parent.master if isinstance(parent.master, tk.Canvas) else None   # 分頁的捲動 Canvas

        # 分頁內容區很矮（視窗上方的標題、流程列、分頁列就佔掉一半），不放大標題橫幅，空間留給卡片和紀錄
        tk.Frame(root, bg=TAB_ACCENT, height=4).pack(fill="x")

        # ── 工具列：標題、broker 狀態＋全部啟動／全部停止（固定在上面，不跟卡片區捲）──
        top = tk.Frame(root, bg=_BG)
        top.pack(fill="x", padx=SPACE_LG, pady=(SPACE_SM, SPACE_SM))
        tk.Label(top, text=f"{TAB_EMOJI} {TAB_NAME}", bg=_BG, fg=TAB_ACCENT, font=self._f_title).pack(side="left", padx=(0, SPACE_LG))
        tk.Label(top, text="目前連上的 MQTT", bg=_BG, fg=_HINT, font=self._f_body).pack(side="left")
        self._broker_var = tk.StringVar(value="檢查中…")
        self._broker_label = tk.Label(top, textvariable=self._broker_var, font=self._f_pill, padx=12, pady=4,
                                      bg=_PILL["off"][0], fg=_PILL["off"][1])
        self._broker_label.pack(side="left", padx=(SPACE_SM, 0))
        # 各 ESP32 的 IP 顯示在各自卡片的標題列；這裡只在出現「認不得的裝置」時提示（主機名稱不是 cat-…：
        # 還沒燒新韌體、或不是這個系統的裝置）。
        self._unknown_var = tk.StringVar(value="")
        tk.Label(top, textvariable=self._unknown_var, bg=_BG, fg="#9a5b0b", font=self._f_body).pack(
            side="left", padx=(SPACE_MD, 0))
        # 全部都在跑 →「全部啟動」變灰；全部都停了 →「全部停止」變灰（_apply 依狀態切換）
        self._btn_stop_all = _styled_button(top, "■ 全部停止", self._stop_all, _STOP_BG, _STOP_ACTIVE, font=self._f_btn)
        self._btn_stop_all.pack(side="right", padx=(SPACE_SM, 0))
        self._btn_start_all = _styled_button(top, "▶ 全部啟動", self._start_all, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE, font=self._f_btn)
        self._btn_start_all.pack(side="right", padx=(SPACE_SM, 0))
        _styled_button(top, "⟳ 重新整理", self._manual_refresh, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, font=self._f_btn).pack(side="right")
        _styled_button(top, "⚙ 參數設定", self._open_config, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
                       font=self._f_btn).pack(side="right", padx=(0, SPACE_SM))

        # 資料庫：路徑、大小、保留天數（09-27 使用者要求寫在設定視窗；定時清理見 iot/config.py DATA_RETENTION_DAYS）
        dbrow = tk.Frame(root, bg=_BG)
        dbrow.pack(fill="x", padx=SPACE_LG, pady=(0, SPACE_SM))
        self._db_var = tk.StringVar()
        tk.Label(dbrow, textvariable=self._db_var, bg=_BG, fg=_HINT, font=self._f_body, anchor="w").pack(side="left")
        _styled_button(dbrow, "📂 開啟資料夾", self._open_db_dir, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
                       font=self._f_btn, compact=True).pack(side="left", padx=(SPACE_SM, 0))
        self._update_db_info()

        # 紀錄框先 pack（貼底）、卡片區最後 pack：空間不夠時先縮卡片區
        self._build_log_panel(root, parent)
        area = self._build_scroll_area(root)

        # ── 服務卡片（兩欄）──
        grid = tk.Frame(area, bg=_BG)
        grid.pack(fill="x", padx=SPACE_LG - SPACE_SM // 2)
        for i in range(2):
            grid.columnconfigure(i, weight=1, uniform="iot_cards")
        for i, svc in enumerate(iot_services.SERVICES):
            self._build_card(grid, svc).grid(row=i // 2, column=i % 2, sticky="nsew", padx=SPACE_SM // 2, pady=SPACE_SM // 2)

        # ── 說明卡（放卡片後面：分頁很矮時先看到卡片）──
        note = tk.Frame(area, bg="#e6f4f1")
        note.pack(fill="x", padx=SPACE_LG, pady=(SPACE_SM, SPACE_SM))
        tk.Frame(note, bg=TAB_ACCENT, width=5).pack(side="left", fill="y")
        note_text = tk.Label(
            note, bg="#e6f4f1", fg="#24463f", font=self._f_body, anchor="w", justify="left",
            text="• 感測器與飼主語音紀錄的啟動口：每個服務各自一個背景行程，可以跟 main.py 同時執行。\n"
                 "• 關掉設定視窗時，執行中的 IoT 服務會一起強制關閉（在 cmd 手動執行的 python -m iot／python -m iot.voice 也會）。\n"
                 "• 資料都經過 MQTT broker；broker 連不上時服務會一直重試。紀錄檔在 paper\\logs\\iot\\。\n"
                 "• ⚙ 參數設定：覆寫 iot/config.py 的告警門檻、broker 等參數（空白＝預設值），重新啟動感測器服務後生效。\n"
                 "• 下方紀錄框：拖標題列調整高度（雙擊回預設）；A＋／A－、Ctrl＋滾輪調整字級；"
                 "「清空」只清畫面、不刪紀錄檔。")
        note_text.pack(fill="x", padx=SPACE_MD, pady=SPACE_SM)
        note.bind("<Configure>", lambda e: note_text.configure(wraplength=max(200, e.width - 40)))
        self._hook_area_wheel(area)

        # 整頁高度跟著「分頁可見高度」與「全域終端機高度」走
        self._fit_pending = False
        if self._outer is not None:
            self._outer.bind("<Configure>", lambda e: self._schedule_fit(), add="+")
        console = getattr(getattr(window, "_console_panel", None), "container", None)
        if console is not None:
            console.bind("<Configure>", lambda e: self._schedule_fit(), add="+")
        root.bind("<Map>", lambda e: self._schedule_fit(), add="+")

        self._update_zoom_label()
        self._select(self.selected, merged=self._merged)
        self.refresh()
        self._tick()
        self._log_sig = None
        self._log_tick()

    # ── 版面：卡片區（自己捲）、紀錄框（標題列拖拉）──────────────────────
    def _build_scroll_area(self, root):
        wrap = tk.Frame(root, bg=_BG)
        wrap.pack(side="top", fill="both", expand=True)
        canvas = tk.Canvas(wrap, bg=_BG, highlightthickness=0, width=1, height=1)
        sb = tk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        area = tk.Frame(canvas, bg=_BG)
        item = canvas.create_window((0, 0), window=area, anchor="nw")
        area.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(item, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self._area_canvas = canvas
        return area

    def _hook_area_wheel(self, area):
        """卡片區的滾輪只捲卡片區。外層分頁用 bind_all 綁滾輪，這裡在每個子元件的 bindtags
        插一個自己的標籤、處理完回傳 break，就不會再走到 bind_all。"""
        tag = f"IotWheel{id(self)}"
        self.root.bind_class(tag, "<MouseWheel>", self._on_area_wheel)

        def walk(w):
            tags = list(w.bindtags())
            if tag not in tags:
                w.bindtags(tuple(tags[:-1] + [tag, tags[-1]]))   # 放在 "all" 前面
            for c in w.winfo_children():
                walk(c)
        walk(self._area_canvas)

    def _on_area_wheel(self, e):
        c = self._area_canvas
        if c.yview() != (0.0, 1.0):   # 內容比可見範圍高才捲
            c.yview_scroll(int(-1 * (e.delta / 120)), "units")
        return "break"

    def _build_log_panel(self, root, parent):
        # 紀錄框浮在設定視窗上（跟全域終端機一樣用 place），才能往上拉超過分頁範圍、最高到跟全域終端機一樣的上限。
        # 分頁裡留一塊同高的空位（_log_slot），卡片區排在它上面；拉得比分頁還高時就蓋住卡片、分頁列
        # （跟全域終端機一樣）。切到別的分頁時藏起來（_place_log／<Unmap>）。
        self._log_slot = tk.Frame(root, bg=_BG, height=_LOG_HEIGHT_MIN)
        self._log_slot.pack(side="bottom", fill="x")
        self._log_slot.pack_propagate(False)
        host = self.window if isinstance(self.window, (tk.Tk, tk.Toplevel)) else root
        self._log_panel = panel = tk.Frame(host, bg=_BG, height=_LOG_HEIGHT_MIN)
        panel.pack_propagate(False)   # 高度由 _fit／拖拉決定，不被內容撐大
        self._log_height = _LOG_HEIGHT_MIN
        self._log_ratio = _state_num("iot_log_ratio", _LOG_RATIO_DEFAULT, _LOG_RATIO_MIN, _LOG_RATIO_MAX, float)
        self._drag_h0 = None
        root.bind("<Unmap>", lambda e: panel.place_forget(), add="+")

        # 標題列＝拖拉把手（整條都能抓；按鈕除外）。一行就好：上緣一條細色條＋標題＋按鈕
        f = lambda size, weight="normal": tkfont.Font(root=parent, family=_FONT, size=size, weight=weight)   # noqa: E731
        f_title, f_small, f_btn = f(12, "bold"), f(10), f(10, "bold")
        head = tk.Frame(panel, bg=_GRIP_BG, cursor="sb_v_double_arrow")
        head.pack(side="top", fill="x")
        line = tk.Frame(head, bg=TAB_ACCENT, height=3, cursor="sb_v_double_arrow")
        line.pack(side="top", fill="x")
        row = tk.Frame(head, bg=_GRIP_BG, cursor="sb_v_double_arrow")
        row.pack(side="top", fill="x", padx=SPACE_SM, pady=2)
        self._log_title = tk.StringVar()
        title = tk.Label(row, textvariable=self._log_title, bg=_GRIP_BG, fg=_FG, font=f_title, anchor="w",
                         cursor="sb_v_double_arrow")
        title.pack(side="left")
        hint = tk.Label(row, text="⇕ 拖這條調整高度", bg=_GRIP_BG, fg=_HINT, font=f_small, cursor="sb_v_double_arrow")
        hint.pack(side="left", padx=(SPACE_SM, 0))
        _styled_button(row, "用記事本開啟", self._open_log, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, font=f_btn,
                       compact=True).pack(side="right")
        # 整合檢視：全部服務的紀錄依時間合併成一條，每行前面標服務名稱（09-27 使用者要求）
        self._merge_btn = _styled_button(row, "🔀 整合檢視", self._toggle_merged, BTN_SECONDARY_BG,
                                         BTN_SECONDARY_ACTIVE, font=f_btn, compact=True)
        self._merge_btn.pack(side="right", padx=(0, SPACE_XS))
        # 清空：只清畫面（記住紀錄檔目前位置，之後只顯示新寫入的），不刪檔案（09-27 使用者要求）
        _styled_button(row, "清空", self._clear_log, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, font=f_btn,
                       compact=True).pack(side="right", padx=(0, SPACE_XS))
        self._zoom_var = tk.StringVar()
        for text, cmd in (("A+", lambda: self._zoom(+1)), ("A−", lambda: self._zoom(-1))):
            _styled_button(row, text, cmd, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, font=f_btn,
                           compact=True).pack(side="right", padx=(0, SPACE_XS))
        zl = tk.Label(row, textvariable=self._zoom_var, bg=_GRIP_BG, fg=_HINT, font=f_small, cursor="hand2")
        zl.pack(side="right", padx=(0, SPACE_SM))
        zl.bind("<Button-1>", lambda e: self._zoom(0))   # 點字級＝回預設大小
        grips = (head, line, row, title, hint)
        for w in grips:
            w.bind("<ButtonPress-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)
            w.bind("<ButtonRelease-1>", self._drag_end)
            w.bind("<Double-Button-1>", lambda e: self._reset_height())
            w.bind("<Enter>", lambda e: self._grip_color(_GRIP_HOVER))
            w.bind("<Leave>", lambda e: self._grip_color(_GRIP_BG))
        self._grips = grips

        self._log_box = tk.Frame(panel, bg=COLOR_CONSOLE_BG)
        self._log_box.pack(side="top", fill="both", expand=True)
        self._log = tk.Text(self._log_box, bg=COLOR_CONSOLE_BG, fg=COLOR_CONSOLE_FG, font=self._log_font, relief="flat",
                            wrap="none", padx=SPACE_MD, pady=SPACE_SM, insertbackground=COLOR_CONSOLE_FG, height=1)
        self._log.tag_configure("warn_line", foreground=_REPORT_COLORS["iot_bad"])
        for key, color in _MERGED_PREFIX_COLOR.items():   # 整合檢視的「[服務]」前綴，蓋在紅字上面
            self._log.tag_configure(f"svc_{key}", foreground=color)
            self._log.tag_raise(f"svc_{key}")
        # 跟隨最新：使用者自己往上捲（滾輪、捲軸）才停止，捲回最底又恢復；紀錄框大小改變時照樣停在最底
        self._follow = True
        sb = tk.Scrollbar(self._log_box, command=lambda *a: (self._log.yview(*a), self._update_follow()))
        hb = tk.Scrollbar(self._log_box, orient="horizontal", command=self._log.xview)
        self._log.configure(yscrollcommand=sb.set, xscrollcommand=hb.set, state="disabled")
        sb.pack(side="right", fill="y")
        hb.pack(side="bottom", fill="x")
        self._log.pack(side="left", fill="both", expand=True)
        # 紀錄框自己的滾輪：只捲紀錄、不捲整個分頁；Ctrl＋滾輪／Ctrl＋＋－0 放大縮小
        self._log.bind("<MouseWheel>", self._on_log_wheel)
        self._log.bind("<Configure>", lambda e: self._follow and self._to_end())
        self._log.bind("<Button-1>", lambda e: self._log.focus_set())
        for seq, d in (("<Control-plus>", 1), ("<Control-equal>", 1), ("<Control-KP_Add>", 1),
                       ("<Control-minus>", -1), ("<Control-KP_Subtract>", -1), ("<Control-0>", 0), ("<Control-KP_0>", 0)):
            self._log.bind(seq, lambda e, d=d: (self._zoom(d), "break")[1])

    def _grip_color(self, color):
        for w in self._grips[:1] + self._grips[2:]:   # 上緣色條（_grips[1]）不變色
            w.configure(bg=color)

    # 整頁可見高度：外層 Canvas 的高度，扣掉被全域終端機（浮在上面）蓋住的部分
    def _visible_height(self) -> int:
        if self._outer is None:
            return int(self.root.cget("height"))
        try:
            top, h = self._outer.winfo_rooty(), self._outer.winfo_height()
            covered = 0
            box = getattr(getattr(self.window, "_console_panel", None), "container", None)
            if box is not None and box.winfo_ismapped():
                covered = max(0, top + h - box.winfo_rooty())
        except tk.TclError:
            return int(self.root.cget("height"))
        return max(_PAGE_MIN, h - covered)

    def _schedule_fit(self):
        if not self._fit_pending:
            self._fit_pending = True
            self.root.after_idle(self._fit)

    def _fit(self):
        self._fit_pending = False
        if not self.root.winfo_exists() or not self.root.winfo_ismapped():
            return
        h = self._visible_height()
        if int(self.root.cget("height")) != h:
            self.root.configure(height=h)
        if self._outer is not None:
            self._outer.yview_moveto(0)   # 這個分頁不捲外層
        self.root.update_idletasks()      # 先把新高度排好，_place_log 才量得到正確位置
        self._set_log_height(self._space() * self._log_ratio)

    def _avail(self) -> int:
        """分頁裡卡片區＋紀錄框空位可分的高度（整頁扣掉上方固定的色條、工具列）。"""
        fixed = sum(w.winfo_reqheight() for w in self.root.pack_slaves()
                    if w is not self._log_slot and w is not self._area_canvas.master)
        return max(1, int(self.root.cget("height")) - fixed - 2 * SPACE_SM)

    def _space(self) -> int:
        """記比例用的基準高度＝分頁裡紀錄框最多能佔的高度（紀錄框下方留 SPACE_SM）。"""
        return max(1, self._avail() - SPACE_SM)

    def _win_y(self, w) -> int:
        return w.winfo_rooty() - self.window.winfo_rooty()

    def _log_bottom(self) -> int:
        """紀錄框下緣（設定視窗座標）：分頁底往上留 SPACE_SM，而且**不低於全域終端機上緣**。
        09-27 修：全域終端機拉高時分頁高度被 _PAGE_MIN 撐住，分頁底會跑到終端機裡面，
        紀錄框（lift 在最上層）就蓋住終端機、滾輪變成在捲 IoT 紀錄。"""
        bottom = self._win_y(self.root) + int(self.root.cget("height")) - SPACE_SM
        box = getattr(getattr(self.window, "_console_panel", None), "container", None)
        try:
            if box is not None and box.winfo_ismapped():
                bottom = min(bottom, self._win_y(box) - SPACE_SM)
        except tk.TclError:
            pass
        return bottom

    def _log_top_limit(self) -> int:
        """紀錄框上緣最高到哪（設定視窗座標）：跟全域終端機拖拉的上限一樣
        （底部按鈕列上緣往上量 console.max_height()，大約是視窗標題列下緣）。"""
        console = getattr(self.window, "_console_panel", None)
        bar = getattr(self.window, "_bottom_bar_frame", None)
        try:
            if console is not None and bar is not None:
                return max(0, bar.winfo_y() - console.max_height())
        except (tk.TclError, AttributeError):
            pass
        return self._win_y(self.root)

    def _floating(self) -> bool:
        return self._log_panel.master is not self.root and self.root.winfo_ismapped()

    def _log_height_max(self) -> int:
        if not self._floating():
            return max(_LOG_HEIGHT_MIN, self._space())
        return max(_LOG_HEIGHT_MIN, self._log_bottom() - self._log_top_limit())

    def _set_log_height(self, h) -> int:
        h = max(_LOG_HEIGHT_MIN, min(self._log_height_max(), round(h)))
        self._log_height = h
        if int(self._log_panel.cget("height")) != h:
            self._log_panel.configure(height=h)
        slot = max(1, min(h + SPACE_SM, self._avail()))   # 分頁裡的空位：放不下的部分就往上蓋住卡片、分頁列
        if int(self._log_slot.cget("height")) != slot:
            self._log_slot.configure(height=slot)
        self._place_log()
        return h

    def _place_log(self):
        """紀錄框浮在設定視窗上：下緣貼分頁底（全域終端機上緣），往上長 _log_height。分頁看不到就藏起來。"""
        panel = self._log_panel
        if panel.master is self.root:
            return
        if not self.root.winfo_ismapped():
            panel.place_forget()
            return
        if self._log_bottom() - self._log_top_limit() < _LOG_HEIGHT_MIN:
            panel.place_forget()   # 全域終端機拉得很高、上方放不下紀錄框：先藏起來，不去蓋終端機
            return
        x = self.root.winfo_rootx() - self.window.winfo_rootx() + SPACE_LG
        panel.place(x=x, y=self._log_bottom(), anchor="sw", width=max(1, self.root.winfo_width() - 2 * SPACE_LG),
                    height=self._log_height)
        panel.lift()

    # ── 服務卡片 ─────────────────────────────────────────────────────────
    def _build_card(self, parent, svc):
        color = _SERVICE_COLOR.get(svc.key, TAB_ACCENT)
        card = tk.Frame(parent, bg=_CARD_BG, highlightthickness=1, highlightbackground=_CARD_BORDER)
        tk.Frame(card, bg=color, width=6).pack(side="left", fill="y")
        body = tk.Frame(card, bg=_CARD_BG)
        body.pack(side="left", fill="both", expand=True, padx=SPACE_MD, pady=SPACE_MD)
        head = tk.Frame(body, bg=_CARD_BG)
        head.pack(fill="x")
        tk.Label(head, text=svc.title, bg=_CARD_BG, fg=_FG, font=self._f_card, anchor="w").pack(side="left")
        ip_var = tk.StringVar(value="")   # 這個種類的 ESP32 IP（_apply 填；沒資料時藏起來）
        self._ip_vars[svc.key] = ip_var
        self._ip_labels[svc.key] = tk.Label(head, textvariable=ip_var, font=self._f_pill, padx=10, pady=3,
                                            bg=_PILL["off"][0], fg=_PILL["off"][1])
        var = tk.StringVar(value="…")
        lab = tk.Label(head, textvariable=var, font=self._f_pill, padx=10, pady=3, bg=_PILL["off"][0], fg=_PILL["off"][1])
        lab.pack(side="right")
        self._status_vars[svc.key], self._status_labels[svc.key] = var, lab
        desc = tk.Label(body, text=svc.desc, bg=_CARD_BG, fg=_HINT, font=self._f_body, anchor="w", justify="left", wraplength=480)
        desc.pack(fill="x", pady=(SPACE_SM, SPACE_SM))
        body.bind("<Configure>", lambda e, d=desc: d.configure(wraplength=max(200, e.width - 8)))
        # 4 個感測器是同一支 python -m iot、各開一個行程，只差 KINDS 環境變數；語音紀錄是另一支程式（iot.voice）
        cmd = f"python -m {svc.module}" + (f"  · KINDS={svc.kind}" if svc.kind else "")
        tk.Label(body, text=cmd, bg=_MONO_BG, fg="#3d5366", font=self._f_mono, anchor="w", padx=8, pady=3).pack(anchor="w")
        btns = tk.Frame(body, bg=_CARD_BG)
        btns.pack(fill="x", pady=(SPACE_MD, 0))
        _styled_button(btns, "▶ 啟動", lambda k=svc.key: self._start(k), BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE, font=self._f_btn).pack(side="left")
        _styled_button(btns, "■ 停止", lambda k=svc.key: self._stop(k), _STOP_BG, _STOP_ACTIVE, font=self._f_btn).pack(side="left", padx=SPACE_SM)
        _styled_button(btns, "📄 紀錄", lambda k=svc.key: self._select(k, merged=False), BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
                       font=self._f_btn).pack(side="left")
        self._cards[svc.key] = card
        return card

    # ── 紀錄面板：字體、高度 ──────────────────────────────────────────────
    def _update_zoom_label(self):
        self._zoom_var.set(f"字級 {self._log_font.cget('size')}")

    def _zoom(self, delta):
        size = _LOG_FONT_DEFAULT if delta == 0 else self._log_font.cget("size") + delta
        size = max(_LOG_FONT_MIN, min(_LOG_FONT_MAX, size))
        self._log_font.configure(size=size)
        self._update_zoom_label()
        _ui_state.update(iot_log_font=str(size))

    def _on_log_wheel(self, e):
        if e.state & 0x0004:   # Ctrl＋滾輪
            self._zoom(1 if e.delta > 0 else -1)
        else:
            self._log.yview_scroll(int(-1 * (e.delta / 120)), "units")
            self._update_follow()
        return "break"

    def _to_end(self):
        """捲到最新一行。see("end") 在紀錄框還很窄時會順便往右捲（每行開頭被切掉），所以橫向拉回最左邊。"""
        self._log.see("end")
        self._log.xview_moveto(0)

    def _update_follow(self):
        self._follow = self._log.yview()[1] >= 0.999

    # 拖紀錄框標題列：往上拖＝變高（跟全域終端機一樣，把手在上緣，永遠看得到）
    def _drag_start(self, e):
        self._drag_y0 = e.y_root
        self._drag_h0 = int(self._log_panel.cget("height"))

    def _drag_move(self, e):
        if self._drag_h0 is None:
            return
        h = self._set_log_height(self._drag_h0 + (self._drag_y0 - e.y_root))
        self._log_ratio = max(_LOG_RATIO_MIN, min(_LOG_RATIO_MAX, h / self._space()))

    def _drag_end(self, _e):
        if self._drag_h0 is None:
            return
        self._drag_h0 = None
        _ui_state.update(iot_log_ratio=f"{self._log_ratio:.3f}")

    def _reset_height(self):
        """雙擊標題列：回預設高度（可用高度的 45%）。"""
        self._log_ratio = _LOG_RATIO_DEFAULT
        self._set_log_height(self._space() * self._log_ratio)
        _ui_state.update(iot_log_ratio=f"{self._log_ratio:.3f}")

    # ── 選服務、顯示紀錄 ─────────────────────────────────────────────────
    def _select(self, key, merged=None):
        """看某個服務的紀錄。merged=True＝整合檢視（全部服務依時間合併）、False＝單一服務、
        None＝維持目前的檢視方式（啟動／停止服務時用，整合檢視中不會被切回單一服務）。"""
        self.selected = key
        if merged is None:
            merged = self._merged
        if merged != self._merged:
            self._merged = merged
            _ui_state.update(iot_log_merged="1" if merged else "0")
        svc = iot_services.get(key)
        if merged:
            self._log_title.set("📄  全部 IoT 服務的紀錄（依時間合併）")
        else:
            self._log_title.set(f"📄  {svc.title} 的紀錄（{svc.log_path.name}）")
        self._merge_btn.config(text="↩ 回到單一服務" if merged else "🔀 整合檢視")
        for k, card in self._cards.items():
            on = not merged and k == key
            color = _SERVICE_COLOR.get(k, TAB_ACCENT)
            card.configure(highlightbackground=color if on else _CARD_BORDER, highlightthickness=2 if on else 1)
        self._shown_log = None
        self._log_sig = None
        self._follow = True   # 換服務：從最新的幾行開始看
        self._show_log()

    def _toggle_merged(self):
        self._select(self.selected, merged=not self._merged)

    def _merged_rows(self) -> list[tuple[str, str]]:
        """整合檢視的內容：[(服務, 顯示的一行)]，每行前面加「[服務名稱]」。"""
        short = {s.key: s.title.split(" ", 1)[-1] for s in iot_services.SERVICES}
        rows = iot_services.merged_tail(since=self._log_cleared)
        if not rows:
            return [("", "（還沒有任何紀錄，或全部都清空了：等待新的紀錄……）")]
        return [(k, f"[{short.get(k, k)}] {line}") for k, line in rows]

    def _show_log(self):
        if self._merged:
            rows = self._merged_rows()
            text = "\n".join(line for _k, line in rows)
        else:
            rows = None
            text = iot_services.tail(self.selected, since=self._log_cleared.get(self.selected, 0))
        if text == self._shown_log:   # 沒變就不重畫（不打斷選取、不跳動）
            return
        self._shown_log = text
        first = self._log.yview()[0]
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.insert("end", text)
        # WARNING／ERROR 行（沒收到資料、跟 broker 斷線、連不上……）用紅字
        for n, line in enumerate(text.split("\n"), start=1):
            if any(m in line for m in _LOG_WARN_MARKS):
                self._log.tag_add("warn_line", f"{n}.0", f"{n}.end")
        # 整合檢視：每行前面的「[服務]」用該服務的顏色
        for n, (key, line) in enumerate(rows or (), start=1):
            if key and line.startswith("["):
                self._log.tag_add(f"svc_{key}", f"{n}.0", f"{n}.{line.index(']') + 1}")
        self._log.configure(state="disabled")
        if self._follow:
            self._to_end()
        else:   # 使用者正在往上看：新內容進來也不跳走
            self._log.yview_moveto(first)

    # ── 狀態 ─────────────────────────────────────────────────────────────
    def _manual_refresh(self):
        """「⟳ 重新整理」按鈕：先清掉 ESP32 主機名稱的 DNS 快取（平常快取 5 分鐘），燒錄改名後按一下就重新查；
        再做一次完整連線檢查（broker、Node-RED、每個服務、每張卡片的裝置、未辨識裝置），結果印到底部的
        全域終端機，有問題的用紅字（09-27 使用者要求）。自動更新（每 2 秒）走 refresh()，不清快取、不印。"""
        iot_services.clear_dns_cache()
        self.refresh()
        box = {}

        def work():   # 背景執行緒只量、不碰 Tk
            try:
                box["lines"] = iot_services.check_connections()
            except Exception as e:  # noqa: BLE001
                box["lines"] = [("bad", f"✕ IoT 連線檢查失敗：{e}")]

        threading.Thread(target=work, daemon=True, name="iot-tab-check").start()
        self.root.after(100, lambda: self._print_report(box))

    def _print_report(self, box):
        if not self.root.winfo_exists():
            return
        if "lines" not in box:
            self.root.after(100, lambda: self._print_report(box))
            return
        console = getattr(self.window, "_console_panel", None)
        if console is None:
            return
        for tag, color in _REPORT_COLORS.items():
            console.text.tag_configure(tag, foreground=color)
        if getattr(console, "_collapsed", False):   # 使用者按了按鈕就是要看報告：收合著就展開
            console.toggle_collapse()
        console.append("\n")
        for level, text in box["lines"]:
            console.append(text + "\n", f"iot_{level}")

    def refresh(self):
        """狀態、broker 在背景執行緒量（掃行程、連 broker 都可能要幾百毫秒），量完回主執行緒更新畫面。"""
        if self._busy:
            return
        self._busy = True
        box = {}

        def work():   # 背景執行緒只量、不碰 Tk（Tkinter 不是執行緒安全的）；結果由主執行緒的 _collect 取走
            try:
                statuses = iot_services.scan()
                box["result"] = (statuses, iot_services.broker_reachable(), iot_services.mqtt_links(statuses))
            except Exception as e:  # noqa: BLE001
                box["result"] = (None, (False, str(e)), None)

        threading.Thread(target=work, daemon=True, name="iot-tab-refresh").start()
        self.root.after(100, lambda: self._collect(box))

    def _collect(self, box):
        if not self.root.winfo_exists():
            return
        if "result" not in box:
            self.root.after(100, lambda: self._collect(box))
            return
        self._apply(*box["result"])

    def _apply(self, statuses, broker, links=None):
        self._busy = False
        self._update_db_info()
        ok, where = broker
        running = any(st.running for st in (statuses or {}).values())
        # broker 膠囊（09-27 使用者要求）：顯示服務「目前實際連上」的 broker 位址（讀 TCP 連線表），不是設定值連不連得到。
        # 服務都沒在跑＝沒有連線，灰色附上設定值；設定值連不上才用紅色提醒
        if links is not None and links.service_targets:
            text, state = "● " + "、".join(links.service_targets), "ok"
        elif running and links is not None:
            text, state = f"✕ 尚未連上（設定 {where}，服務重試中）", "bad"
        elif ok:
            text, state = f"— 服務未啟動（設定 {where}）", "off"
        else:
            text, state = f"✕ 服務未啟動，設定 {where} 也連不上", "bad"
        self._broker_var.set(text)
        self._broker_label.configure(bg=_PILL[state][0], fg=_PILL[state][1])
        # 各卡片的裝置 IP（主機名稱對應，見 iot_services.mqtt_links）
        for key, var in self._ip_vars.items():
            if links is None:
                text, state = "", "off"
            elif links.devices is None:
                text, state = "broker 不在本機，看不到 IP", "off"
            elif links.ips_for(key):
                text, state = "IP " + "、".join(links.ips_for(key)), "ok"
            else:
                text, state = "✕ 裝置未連上", "bad"
            # 狀態沒變就什麼都不動（以前每 2 秒都重新 pack 一次，整列重排、畫面微微閃動）
            lab = self._ip_labels[key]
            if var.get() == text and lab.cget("bg") == _PILL[state][0] and bool(lab.winfo_manager()) == bool(text):
                continue
            var.set(text)
            lab.configure(bg=_PILL[state][0], fg=_PILL[state][1])
            if text and not lab.winfo_manager():
                lab.pack(side="left", padx=(SPACE_SM, 0))
            elif not text and lab.winfo_manager():
                lab.pack_forget()
        unknown = links.unknown() if links is not None else []
        shown = "、".join(f"{d.ip}（{d.hostname or '查不到名稱'}）" for d in unknown[:2])
        more = f" 等 {len(unknown)} 台" if len(unknown) > 2 else ""
        self._unknown_var.set(f"⚠ 未辨識裝置：{shown}{more}" if unknown else "")
        for key, st in (statuses or {}).items():
            if key not in self._status_vars:
                continue
            self._status_vars[key].set(("● " if st.running else "○ ") + st.text())
            bg, fg = _PILL["on" if st.running else "off"]
            self._status_labels[key].configure(bg=bg, fg=fg)
        if statuses:
            running = [st.running for st in statuses.values()]
            self._btn_start_all.config(state="disabled" if all(running) else "normal")
            self._btn_stop_all.config(state="normal" if any(running) else "disabled")
        self._show_log()

    def _tick(self):
        """這個分頁看得到的時候每 2 秒更新；看不到就只排下一次、不做事。"""
        if not self.root.winfo_exists():
            return
        if self.root.winfo_ismapped():
            self._schedule_fit()
            self.refresh()
        self.root.after(_REFRESH_MS, self._tick)

    def _log_tick(self):
        """紀錄框即時同步：每 0.5 秒看一下紀錄檔的大小／修改時間（只 stat，不讀檔），有變才重讀。
        服務用 PYTHONUNBUFFERED 啟動、輸出直接寫進紀錄檔，所以新的一行大約 0.5 秒內就會出現。"""
        if not self.root.winfo_exists():
            return
        if self.root.winfo_ismapped():
            keys = [s.key for s in iot_services.SERVICES] if self._merged else [self.selected]
            sig = [self._merged]
            for k in keys:   # 整合檢視：任何一個服務的紀錄檔有變就重讀
                try:
                    st = iot_services.get(k).log_path.stat()
                    sig.append((k, st.st_size, st.st_mtime_ns))
                except OSError:
                    sig.append((k, None, None))
            sig = tuple(sig)
            if sig != self._log_sig:
                self._log_sig = sig
                self._show_log()
        self.root.after(_LOG_POLL_MS, self._log_tick)

    # ── 動作 ─────────────────────────────────────────────────────────────
    def _start(self, key):
        ok, msg = iot_services.start(key)
        self._select(key)
        if not ok:
            dialogs.show_info(self.window, "啟動 IoT 服務", msg)
            return
        # 啟動後馬上結束（缺套件、設定錯）→ 2 秒後檢查，告訴使用者看紀錄
        self.root.after(2000, lambda: self._check_started(key))
        self.refresh()

    def _check_started(self, key):
        if not iot_services.scan()[key].running:
            dialogs.show_warning(self.window, "啟動 IoT 服務",
                                 f"{iot_services.get(key).title} 啟動後馬上結束了，請看下方紀錄的錯誤訊息。")
        self.refresh()

    def _stop(self, key):
        st = iot_services.scan()[key]
        if not st.running:
            dialogs.show_info(self.window, "停止 IoT 服務", f"{iot_services.get(key).title} 沒有在執行。")
            return
        extra = ""
        if len(st.shared_kinds) > 1:
            extra = "\n\n這個行程同時處理：" + "、".join(iot_services.get(k).title for k in st.shared_kinds) + "，會一起停止。"
        if not st.managed:
            extra += "\n\n這個服務是在設定視窗外面啟動的（例如 cmd）。"
        if not dialogs.ask_yesno(self.window, "停止 IoT 服務", f"要停止 {iot_services.get(key).title}（PID {st.pid}）嗎？{extra}"):
            return
        ok, msg = iot_services.stop(key, pid=st.pid)
        if not ok:
            dialogs.show_error(self.window, "停止 IoT 服務", msg)
        self.refresh()

    def _start_all(self):
        msgs = [iot_services.start(s.key)[1] for s in iot_services.SERVICES if not iot_services.scan()[s.key].running]
        self.refresh()
        dialogs.show_info(self.window, "全部啟動", "\n".join(msgs) if msgs else "全部都已經在執行。")

    def _stop_all(self):
        running = {st.pid for st in iot_services.scan().values() if st.running}
        if not running:
            dialogs.show_info(self.window, "全部停止", "沒有在執行的 IoT 服務。")
            return
        if not dialogs.ask_yesno(self.window, "全部停止", f"要停止全部 IoT 服務嗎？（{len(running)} 個行程）"):
            return
        iot_services.stop_all("由設定視窗「全部停止」停止")
        self.refresh()

    def _open_config(self):
        iot_config_dialog.open_dialog(self.window)

    def _update_db_info(self):
        path, size, days = iot_services.db_info()
        keep = f"原始讀數保留 {days:g} 天（自動清理）" if days > 0 else "不自動清理（保留天數＝0）"
        self._db_var.set(f"🗄 資料庫：{path}　{size / 1024 / 1024:.1f} MB　·　{keep}")

    def _open_db_dir(self):
        folder = iot_services.db_info()[0].parent
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    def _clear_log(self):
        keys = [s.key for s in iot_services.SERVICES] if self._merged else [self.selected]
        for k in keys:   # 整合檢視：全部服務一起清
            self._log_cleared[k] = iot_services.log_size(k)
        self._shown_log = None
        self._follow = True
        self._show_log()

    def _open_log(self):
        if self._merged:   # 整合檢視：開紀錄資料夾（每個服務各一個 .log）
            iot_services.LOG_DIR.mkdir(parents=True, exist_ok=True)
            os.startfile(str(iot_services.LOG_DIR))
            return
        path = iot_services.get(self.selected).log_path
        if not path.exists():
            dialogs.show_info(self.window, "紀錄檔", "還沒有紀錄檔：這個服務還沒從設定視窗啟動過。")
            return
        # 一定用記事本開（.log 的預設程式不一定是記事本）。紀錄檔是 UTF-8：從設定視窗啟動時會在開頭補 BOM；
        # 還在跑、沒有 BOM 的舊紀錄檔（不能動它）就開一份加了 BOM 的快照，記事本才不會把中文顯示成亂碼
        target = path
        try:
            data = path.read_bytes()
            if not data.startswith(b"\xef\xbb\xbf"):
                target = Path(tempfile.gettempdir()) / f"iot_{self.selected}_快照.log"
                target.write_bytes(b"\xef\xbb\xbf" + data)
        except OSError:
            target = path
        subprocess.Popen(["notepad.exe", str(target)])


def build(parent, window) -> IotTab:
    """settings_window.py 建這個分頁時呼叫；parent＝分頁的內容區。"""
    return IotTab(parent, window)


def running_titles() -> list[str]:
    """目前在執行的 IoT 服務名稱（settings_window.py 關閉視窗時用）。"""
    return iot_services.running_titles()


def shutdown_services() -> list[int]:
    """關閉設定視窗時呼叫：強制停止全部 IoT 服務（09-27 使用者要求）。回傳停掉的 PID。"""
    return iot_services.stop_all("設定視窗關閉，一併停止")
