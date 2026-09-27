"""設定視窗「📡 IoT 子系統」分頁的「⚙ 參數設定」對話框：覆寫 ``iot/config.py`` 的告警門檻等參數（畫面；邏輯在
``iot_config_overrides.py``）。

版面（09-27 美化）：上方是深綠標題列（右側即時顯示「已覆寫 N 項」）；每個分組一張白卡片（上緣分組色條、
標題旁顯示該組改了幾項），螢幕夠寬就排成兩欄。每個參數一列：名稱＋說明（兩行）｜輸入框＋單位｜預設值膠囊｜↺。
**空白＝用預設值**；有填的輸入框底色變黃、↺ 亮起可單獨清掉。
儲存後若有感測器服務在跑，問要不要馬上重新啟動（config 只在啟動時讀一次）。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont

from settings_gui import dialogs, iot_config_overrides as ov, iot_services
from settings_gui.style import (
    BTN_PRIMARY_ACTIVE, BTN_PRIMARY_BG, BTN_SECONDARY_ACTIVE, BTN_SECONDARY_BG, SPACE_LG, SPACE_MD, SPACE_SM, SPACE_XS,
)
from settings_gui.widgets import _styled_button

_FONT = "Microsoft JhengHei"
_BG = "#f3f6f8"
_CARD_BG = "#ffffff"
_CARD_BORDER = "#dbe3e9"
_ROW_SEP = "#edf1f4"
_FG = "#1f2d3a"
_HINT = "#5f7182"
_ACCENT = "#0e6655"
_HEAD_SUB = "#cfe8e1"
_MONO_BG, _MONO_FG = "#eef2f5", "#3d5366"
_ENTRY_BG, _ENTRY_CHANGED = "#ffffff", "#fff6d5"
_BADGE_ON, _BADGE_OFF = ("#fdebd0", "#9a5b0b"), ("#edf0f3", "#6c7a86")
_TOTAL_ON, _TOTAL_OFF = ("#fdebd0", "#9a5b0b"), ("#e8f6f1", _ACCENT)
_RESET_ON, _RESET_OFF = "#b9770e", "#c9d2da"
# 分組色條：依 FIELDS 裡分組出現的順序（環境、體表、秤重、告警通知、感測器偵測、終端列印、MQTT、其他），
# 跟 IoT 分頁的服務卡片同色系
_GROUP_COLORS = ("#16a085", "#cb4335", "#d68910", "#8e44ad", "#0e6655", "#34495e", "#2e86c1", "#6c7a86")
_TWO_COL_MIN_SCREEN = 1400   # 螢幕寬度夠才排兩欄


class IotConfigDialog:
    def __init__(self, window):
        self.window = window
        self.defaults = ov.read_defaults()
        saved = ov.load()
        self.vars: dict[str, tk.StringVar] = {}
        self.entries: dict[str, tk.Entry] = {}
        self._resets: dict[str, tk.Label] = {}
        self._group_of: dict[str, str] = {}
        self._group_badges: dict[str, tk.Label] = {}

        top = self.top = tk.Toplevel(window)
        top.title("IoT 參數設定（覆寫 iot/config.py）")
        top.configure(bg=_BG)
        top.transient(window)
        f = lambda size, weight="normal": tkfont.Font(root=top, family=_FONT, size=size, weight=weight)   # noqa: E731
        f_title, f_group, f_label = f(16, "bold"), f(13, "bold"), f(12, "bold")
        f_body, f_small, f_btn = f(12), f(10), f(12, "bold")
        f_mono = tkfont.Font(root=top, family="Consolas", size=10)

        # ── 標題列 ──
        head = tk.Frame(top, bg=_ACCENT)
        head.pack(fill="x")
        inner = tk.Frame(head, bg=_ACCENT)
        inner.pack(fill="x", padx=SPACE_LG, pady=SPACE_MD)
        self._total_var = tk.StringVar()
        self._total_badge = tk.Label(inner, textvariable=self._total_var, font=f_btn, padx=12, pady=4)
        self._total_badge.pack(side="right", anchor="n")
        tk.Label(inner, text="⚙ IoT 感測器參數", bg=_ACCENT, fg="#ffffff", font=f_title).pack(anchor="w")
        tk.Label(inner, bg=_ACCENT, fg=_HEAD_SUB, font=f_body, justify="left", anchor="w",
                 text="空白＝用 iot/config.py 的預設值；有填的欄位（黃底）會在啟動感測器服務時以環境變數覆寫。\n"
                      "改完要重新啟動感測器服務才會生效。飼主語音紀錄不受影響。\n"
                      "只有從設定視窗 ▶ 啟動才會套用；在 cmd 手動執行 python -m iot 不會（要自己設 "
                      "CAT_MONITORING_IOT_* 環境變數）。").pack(anchor="w", pady=(SPACE_XS, 0))

        # 底部按鈕先 pack（貼底），中間捲動區最後 pack：視窗矮時先縮捲動區
        bar = tk.Frame(top, bg=_BG)
        bar.pack(side="bottom", fill="x", padx=SPACE_LG, pady=SPACE_MD)
        tk.Frame(top, bg=_CARD_BORDER, height=1).pack(side="bottom", fill="x")
        _styled_button(bar, "取消", top.destroy, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, font=f_btn,
                       outline=True).pack(side="right")
        _styled_button(bar, "💾 儲存", self._save, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE, font=f_btn).pack(
            side="right", padx=(0, SPACE_SM))
        _styled_button(bar, "全部清空（回預設）", self._clear_all, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
                       font=f_btn).pack(side="left")

        wrap = tk.Frame(top, bg=_BG)
        wrap.pack(fill="both", expand=True, padx=(SPACE_LG, 0), pady=(SPACE_MD, 0))
        canvas = tk.Canvas(wrap, bg=_BG, highlightthickness=0)
        sb = tk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        body = tk.Frame(canvas, bg=_BG)
        canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self._canvas = canvas
        # 子元件的 bindtags 都含這個 Toplevel：在這裡處理完回 break，就不會走到設定視窗 bind_all 的滾輪
        top.bind("<MouseWheel>", self._on_wheel)

        # ── 分組卡片 ──
        groups: dict[str, list] = {}
        for fld in ov.FIELDS:
            groups.setdefault(fld.group, []).append(fld)
        ncol = 2 if top.winfo_screenwidth() >= _TWO_COL_MIN_SCREEN else 1
        for c in range(ncol):
            body.grid_columnconfigure(c, weight=1, uniform="card")
        fonts = (f_group, f_label, f_body, f_small, f_mono)
        for i, (group, fields) in enumerate(groups.items()):
            card = self._build_card(body, group, fields, _GROUP_COLORS[i % len(_GROUP_COLORS)], fonts, saved)
            last_col = i % ncol == ncol - 1
            card.grid(row=i // ncol, column=i % ncol, sticky="nsew",
                      padx=(0, SPACE_MD if last_col else SPACE_SM), pady=(0, SPACE_MD))
        for env in self.vars:
            self._mark(env)

        top.update_idletasks()
        w = min(body.winfo_reqwidth() + SPACE_LG * 2 + 20, int(top.winfo_screenwidth() * 0.92))
        h = min(body.winfo_reqheight() + head.winfo_reqheight() + bar.winfo_reqheight() + 60,
                int(top.winfo_screenheight() * 0.85))
        top.geometry(f"{w}x{h}+{window.winfo_rootx() + 40}+{max(0, window.winfo_rooty() + 20)}")
        top.grab_set()
        top.focus_set()

    def _build_card(self, parent, group, fields, color, fonts, saved):
        f_group, f_label, f_body, f_small, f_mono = fonts
        card = tk.Frame(parent, bg=_CARD_BG, highlightthickness=1, highlightbackground=_CARD_BORDER)
        tk.Frame(card, bg=color, height=4).pack(fill="x")
        hd = tk.Frame(card, bg=_CARD_BG)
        hd.pack(fill="x", padx=SPACE_MD, pady=(SPACE_SM, SPACE_XS))
        tk.Label(hd, text=group, bg=_CARD_BG, fg=color, font=f_group, anchor="w").pack(side="left")
        badge = tk.Label(hd, font=f_small, padx=8, pady=1)
        badge.pack(side="right")
        self._group_badges[group] = badge

        grid = tk.Frame(card, bg=_CARD_BG)
        grid.pack(fill="both", expand=True, padx=SPACE_MD, pady=(0, SPACE_SM))
        grid.grid_columnconfigure(0, weight=1)
        r = 0
        for n, fld in enumerate(fields):
            if n:
                tk.Frame(grid, bg=_ROW_SEP, height=1).grid(row=r, column=0, columnspan=4, sticky="ew")
                r += 1
            left = tk.Frame(grid, bg=_CARD_BG)
            left.grid(row=r, column=0, sticky="w", pady=SPACE_XS)
            tk.Label(left, text=fld.label, bg=_CARD_BG, fg=_FG, font=f_label, anchor="w").pack(anchor="w")
            if fld.hint:
                tk.Label(left, text=fld.hint, bg=_CARD_BG, fg=_HINT, font=f_small, anchor="w").pack(anchor="w")

            mid = tk.Frame(grid, bg=_CARD_BG)
            mid.grid(row=r, column=1, sticky="w", padx=(SPACE_MD, SPACE_SM))
            var = tk.StringVar(value=saved.get(fld.env, ""))
            ent = tk.Entry(mid, textvariable=var, width=16, font=f_body, relief="flat", bd=0,
                           highlightthickness=1, highlightbackground=_CARD_BORDER, highlightcolor=_ACCENT,
                           show="*" if fld.env.endswith("PASSWORD") else "")
            ent.pack(side="left", ipady=3, ipadx=4)
            tk.Label(mid, text=fld.unit, bg=_CARD_BG, fg=_HINT, font=f_body, anchor="w", width=4).pack(
                side="left", padx=(SPACE_XS, 0))

            d = self.defaults.get(fld.env)
            dtext = "（無）" if d is None else ("（空白）" if d.text == "" else d.text)
            if fld.env.endswith("PASSWORD") and d is not None and d.text:
                dtext = "***"
            tk.Label(grid, text=f"預設 {dtext}", bg=_MONO_BG, fg=_MONO_FG, font=f_mono, padx=6, pady=1).grid(
                row=r, column=2, sticky="w", padx=(0, SPACE_SM))
            reset = tk.Label(grid, text="↺", bg=_CARD_BG, fg=_RESET_OFF, font=f_label, cursor="hand2")
            reset.grid(row=r, column=3, sticky="e")
            reset.bind("<Button-1>", lambda _e, v=var: v.set(""))

            var.trace_add("write", lambda *_a, e=fld.env: self._mark(e))
            self.vars[fld.env], self.entries[fld.env] = var, ent
            self._resets[fld.env], self._group_of[fld.env] = reset, group
            r += 1
        return card

    def _on_wheel(self, e):
        c = self._canvas
        if c.yview() != (0.0, 1.0):
            c.yview_scroll(int(-1 * (e.delta / 120)), "units")
        return "break"

    def _changed(self, env) -> bool:
        return bool(self.vars[env].get().strip())

    def _mark(self, env):
        on = self._changed(env)
        self.entries[env].configure(bg=_ENTRY_CHANGED if on else _ENTRY_BG)
        self._resets[env].configure(fg=_RESET_ON if on else _RESET_OFF)
        group = self._group_of[env]
        n = sum(1 for e, g in self._group_of.items() if g == group and self._changed(e))
        bg, fg = _BADGE_ON if n else _BADGE_OFF
        self._group_badges[group].configure(text=f"已改 {n} 項" if n else "全部預設", bg=bg, fg=fg)
        total = sum(1 for e in self.vars if self._changed(e))
        bg, fg = _TOTAL_ON if total else _TOTAL_OFF
        self._total_var.set(f"已覆寫 {total} 項" if total else "全部使用預設值")
        self._total_badge.configure(bg=bg, fg=fg)

    def _clear_all(self):
        for var in self.vars.values():
            var.set("")

    def values(self) -> dict[str, str]:
        return {env: var.get() for env, var in self.vars.items()}

    def _save(self):
        clean, errors = ov.validate(self.values(), self.defaults)
        if errors:
            dialogs.show_error(self.top, "IoT 參數設定", "有欄位不正確，沒有儲存：\n\n" + "\n".join(errors))
            return
        try:
            ov.save(clean)
        except OSError as e:
            dialogs.show_error(self.top, "IoT 參數設定", f"寫入失敗：{e}")
            return
        running = [s.title for s in iot_services.SERVICES if s.kind and iot_services.scan()[s.key].running]
        self.top.destroy()
        if not running:
            dialogs.show_info(self.window, "IoT 參數設定",
                              f"已儲存（{len(clean)} 個參數覆寫）。下次啟動感測器服務時生效。")
            return
        if dialogs.ask_yesno(self.window, "IoT 參數設定",
                             f"已儲存（{len(clean)} 個參數覆寫）。\n\n執行中的感測器服務：{'、'.join(running)}\n"
                             "要現在重新啟動讓新參數生效嗎？"):
            msgs = iot_services.restart_sensors()
            dialogs.show_info(self.window, "重新啟動感測器服務", "\n".join(msgs) or "沒有需要重新啟動的服務。")


def open_dialog(window) -> IotConfigDialog:
    return IotConfigDialog(window)
