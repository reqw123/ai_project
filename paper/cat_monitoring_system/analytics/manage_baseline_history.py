"""個體化基線歷史資料管理工具（獨立視窗，GUI）。

放在跟基線計算模組（``baseline.py``／``daily_store.py``／``config.py``）
同一個目錄，因為這支工具本質上就是那幾個模組的操作介面，不是通用性
的診斷腳本——跟 ``plugins/lick_stage/`` 系列「每個子系統自成一體」的
擺放慣例一致。

用途：列出 Python 端獨立收集的多天歷史（``analytics/daily_store.py``），
讓使用者逐筆勾選「採用」或「排除」，按下「套用變更」後：

1. 把排除狀態寫進 ``daily_store`` 的 ``excluded_dates`` 表（``set_excluded()``）；
2. **立即用新的排除清單重新算一次基線**（``analytics.baseline.compute_baseline``）
   並把結果顯示出來——這一步是為了直接證明排除設定真的有生效，不是「存了但
   不知道有沒有用」。

跟這份資料的其他兩個讀取路徑共用同一份 ``excluded_dates`` 表，套用後立即對
兩邊都生效，不需要重啟任何東西：

- ``dashboard/refresher.py`` 背景排程（每 ``BaselineDashboardConfig.
  RECOMPUTE_INTERVAL_SEC`` 秒重算一次 ``/dashboard/baseline`` 頁面）下一次
  執行就會讀到新的排除清單。
- ``server/routes.py`` 的 ``POST /api/deviation``（Node-RED 用來比對新舊
  引擎）在請求沒帶 ``excluded_dates`` 時，一樣預設讀這份清單。

**只是排除，不是刪除**：跟 Node-RED 自己的 ``v2_excluded_dates`` 語意一致——
資料還在 ``daily_history`` 表裡，只是被排除的那幾天不會被基線計算採用。
不需要攝影機/Flask/Node-RED 任何一個在跑，純粹讀寫 SQLite。

執行方式：

    python cat_monitoring_system/analytics/manage_baseline_history.py
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox, ttk

_SCRIPT_DIR = Path(__file__).resolve().parent  # cat_monitoring_system/analytics/
_CAT_MONITORING_SYSTEM_DIR = _SCRIPT_DIR.parent
_PAPER_DIR = _CAT_MONITORING_SYSTEM_DIR.parent
for _p in (str(_CAT_MONITORING_SYSTEM_DIR), str(_PAPER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analytics import daily_store  # noqa: E402
from analytics.baseline import InsufficientDataError, compute_baseline  # noqa: E402
from analytics.config import BaselineConfig  # noqa: E402


# ── 視覺樣式：集中定義顏色/字型，改配色只需要動這裡 ─────────────────────────

_FONT_FAMILY = "Microsoft JhengHei"

COLOR_BG_MAIN = "#eef2f6"  # 整體視窗底色
COLOR_HEADER_BG = "#2c3e50"  # 頂部標題列（深藍灰）
COLOR_HEADER_FG = "#ffffff"
COLOR_HEADER_SUB_FG = "#b7c4cf"
COLOR_INFO_BG = "#e8f1fb"  # 說明文字底色（淺藍）
COLOR_INFO_FG = "#2c4053"
COLOR_INFO_BORDER = "#c7dbf0"
COLOR_SUMMARY_FG = "#1f2f3d"
COLOR_COLHEAD_BG = "#d7e3f0"  # 表頭列
COLOR_COLHEAD_FG = "#1f2f3d"
COLOR_ROW_INCLUDED_EVEN = "#ffffff"
COLOR_ROW_INCLUDED_ODD = "#f4f9f6"
COLOR_ROW_EXCLUDED = "#fdecea"  # 排除中的列：淡紅底
COLOR_ROW_FG_INCLUDED = "#20303f"
COLOR_ROW_FG_EXCLUDED = "#a8433f"
COLOR_ROW_NOTE_FG = "#b36b00"  # 監測時數不足警示
COLOR_SUCCESS_FG = "#1a7a1a"
COLOR_WARNING_FG = "#b36b00"
COLOR_ERROR_FG = "#c0392b"
COLOR_BORDER = "#c7d0da"

# 配色跟主設定視窗（settings_gui/style.py 的 2026-08 第二版）對齊：扁平、飽和、
# 按語意分色。這支是獨立執行的工具（sys.path 上沒有 paper/），不方便直接 import
# settings_gui.style，所以在這裡放一份對應的常數。
BTN_PRIMARY_BG = "#16a34a"  # 套用變更（主要動作，綠）
BTN_PRIMARY_ACTIVE = "#15803d"
BTN_SECONDARY_BG = "#3b6fb5"  # 一般動作（全部採用／重新整理，鋼藍）
BTN_SECONDARY_ACTIVE = "#335f9c"
BTN_WARN_BG = "#d97706"  # 全部排除（琥珀，提醒這是大範圍動作）
BTN_WARN_ACTIVE = "#b45309"
BTN_NEUTRAL_BG = "#e2e8f0"  # 關閉（扁平淺灰的「安靜」層級）
BTN_NEUTRAL_ACTIVE = "#cbd5e1"
BTN_NEUTRAL_FG = "#334155"


def _styled_button(parent, text, command, bg, active_bg, fg="#ffffff", font=None):
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        activebackground=active_bg,
        activeforeground=fg,
        disabledforeground="#9aa7b5",
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=active_bg,
        highlightcolor=active_bg,
        font=font or (_FONT_FAMILY, 10, "bold"),
        padx=14,
        pady=6,
        cursor="hand2",
    )

    def _hover(enter):
        if enter and str(btn["state"]) == "disabled":
            return
        btn.configure(bg=active_bg if enter else bg)

    btn.bind("<Enter>", lambda _e: _hover(True))
    btn.bind("<Leave>", lambda _e: _hover(False))
    return btn


class BaselineHistoryManager(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("個體化基線歷史資料管理")
        self.geometry("1000x640")
        self.minsize(860, 480)
        self.configure(bg=COLOR_BG_MAIN)

        self._font_row = tkfont.Font(family=_FONT_FAMILY, size=12)
        self._font_row_bold = tkfont.Font(family=_FONT_FAMILY, size=12, weight="bold")
        self._font_note = tkfont.Font(family=_FONT_FAMILY, size=9)
        self._font_colhead = tkfont.Font(family=_FONT_FAMILY, size=11, weight="bold")

        self._row_vars = {}  # date_str -> tk.BooleanVar（True=採用, False=排除）
        self._row_widgets = {}  # date_str -> {"frame":..., "labels":[...]}（即時上色用）
        self._original_excluded = set()  # 目前已持久化的排除清單，用來算 diff

        self._build_layout()
        self.reload()

    # ── UI 佈局 ──────────────────────────────────────────────────────────

    def _build_layout(self):
        # 1) 頂部標題區（深色底，跟其他區塊明確區隔）
        header = tk.Frame(self, bg=COLOR_HEADER_BG)
        header.pack(fill="x")
        tk.Label(
            header,
            text="🐱 個體化基線歷史資料管理",
            bg=COLOR_HEADER_BG,
            fg=COLOR_HEADER_FG,
            font=(_FONT_FAMILY, 18, "bold"),
            anchor="w",
        ).pack(fill="x", padx=18, pady=(14, 0))
        tk.Label(
            header,
            text="Python 端獨立收集的多天歷史（analytics/daily_store.py）——"
            "在這裡採用/排除的設定，會直接影響個體化基線計算結果",
            bg=COLOR_HEADER_BG,
            fg=COLOR_HEADER_SUB_FG,
            font=(_FONT_FAMILY, 10),
            anchor="w",
        ).pack(fill="x", padx=18, pady=(2, 14))

        # 2) 說明區（淺藍底的提示框，跟標題區、清單區都區分開）
        info_wrap = tk.Frame(self, bg=COLOR_BG_MAIN)
        info_wrap.pack(fill="x", padx=16, pady=(12, 6))
        info_box = tk.Frame(
            info_wrap,
            bg=COLOR_INFO_BG,
            highlightbackground=COLOR_INFO_BORDER,
            highlightthickness=1,
        )
        info_box.pack(fill="x")
        tk.Label(
            info_box,
            text="ℹ️  打勾＝採用（納入基線計算）；取消打勾＝排除（保留原始資料，"
            "只是不採用）。排除不會刪除任何資料，隨時可以再勾回來。",
            bg=COLOR_INFO_BG,
            fg=COLOR_INFO_FG,
            font=(_FONT_FAMILY, 9),
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=12, pady=8)

        self._summary_var = tk.StringVar(value="載入中...")
        tk.Label(
            self,
            textvariable=self._summary_var,
            bg=COLOR_BG_MAIN,
            fg=COLOR_SUMMARY_FG,
            font=(_FONT_FAMILY, 10, "bold"),
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=18, pady=(4, 2))

        # 即時預估：勾選狀態每變一次就重算一次，套用前就能看到會不會低於
        # BaselineConfig.MIN_BASELINE_DAYS_DEFAULT，不用等按下套用才知道。
        self._estimate_var = tk.StringVar(value="")
        self._estimate_label = tk.Label(
            self,
            textvariable=self._estimate_var,
            bg=COLOR_BG_MAIN,
            font=(_FONT_FAMILY, 10, "bold"),
            anchor="w",
            justify="left",
        )
        self._estimate_label.pack(fill="x", padx=18, pady=(0, 8))

        # 3) 表頭列（跟資料列的底色明確不同）
        self._columns = [
            ("採用", 6),
            ("日期", 12),
            ("監測時數(hr)", 12),
            ("walk(s)", 9),
            ("lick(s)", 9),
            ("scratch(s)", 10),
            ("shake(次)", 9),
            ("備註", 30),
        ]
        header_row = tk.Frame(self, bg=COLOR_COLHEAD_BG)
        header_row.pack(fill="x", padx=16)
        for text, width in self._columns:
            tk.Label(
                header_row,
                text=text,
                width=width,
                bg=COLOR_COLHEAD_BG,
                fg=COLOR_COLHEAD_FG,
                font=self._font_colhead,
                anchor="w",
            ).pack(side="left", padx=2, pady=8)

        # 4) 可捲動的歷史資料清單（資料變多時滑鼠滾輪往下捲）
        body_wrap = tk.Frame(self, bg=COLOR_BG_MAIN)
        body_wrap.pack(fill="both", expand=True, padx=16, pady=(0, 4))

        canvas = tk.Canvas(
            body_wrap, bg=COLOR_ROW_INCLUDED_EVEN, highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(body_wrap, orient="vertical", command=canvas.yview)
        self._rows_frame = tk.Frame(canvas, bg=COLOR_ROW_INCLUDED_EVEN)
        self._rows_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._rows_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self._canvas = canvas

        # 5) 底部操作列
        bottom = tk.Frame(self, bg=COLOR_BG_MAIN)
        bottom.pack(fill="x", padx=16, pady=(6, 4))
        _styled_button(
            bottom, "全部採用", self._select_all, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE
        ).pack(side="left")
        _styled_button(
            bottom, "全部排除", self._deselect_all, BTN_WARN_BG, BTN_WARN_ACTIVE
        ).pack(side="left", padx=(8, 0))
        _styled_button(
            bottom, "重新整理", self.reload, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE
        ).pack(side="left", padx=(8, 0))
        _styled_button(
            bottom, "關閉", self.destroy, BTN_NEUTRAL_BG, BTN_NEUTRAL_ACTIVE, fg=BTN_NEUTRAL_FG
        ).pack(side="right")
        _styled_button(
            bottom,
            "套用變更並重新計算基線",
            self._apply_changes,
            BTN_PRIMARY_BG,
            BTN_PRIMARY_ACTIVE,
            font=(_FONT_FAMILY, 11, "bold"),
        ).pack(side="right", padx=(0, 8))

        # 6) 套用結果訊息（顏色依成功/警告動態變化，見 _recompute_and_show）
        result_wrap = tk.Frame(self, bg=COLOR_BG_MAIN)
        result_wrap.pack(fill="x", padx=18, pady=(0, 12))
        self._result_var = tk.StringVar(value="")
        self._result_label = tk.Label(
            result_wrap,
            textvariable=self._result_var,
            bg=COLOR_BG_MAIN,
            fg=COLOR_SUCCESS_FG,
            font=(_FONT_FAMILY, 10, "bold"),
            wraplength=940,
            justify="left",
            anchor="w",
        )
        self._result_label.pack(fill="x")

    # ── 資料載入 / 顯示 ──────────────────────────────────────────────────

    def reload(self):
        for child in self._rows_frame.winfo_children():
            child.destroy()
        self._row_vars.clear()
        self._row_widgets.clear()
        self._result_var.set("")

        records = daily_store.load_history()
        self._original_excluded = set(daily_store.load_excluded_dates())
        self._records_by_date = {r.day.isoformat(): r for r in records}

        if not records:
            tk.Label(
                self._rows_frame,
                text="目前 daily_store 沒有任何歷史紀錄"
                "（behavior_tracker 每天跨日時才會寫入一筆，或用"
                " _tools/2_backfill_daily_store.py 從 Node-RED 既有歷史一次性搬遷）。",
                bg=COLOR_ROW_INCLUDED_EVEN,
                fg="#8a97a3",
                font=self._font_row,
            ).pack(anchor="w", pady=24, padx=8)
        else:
            for i, record in enumerate(
                sorted(records, key=lambda r: r.day, reverse=True)
            ):
                self._add_row(record, zebra=(i % 2 == 0))

        self._update_summary(records)
        self._update_live_estimate()

    def _add_row(self, record, zebra):
        date_str = record.day.isoformat()
        included = date_str not in self._original_excluded
        var = tk.BooleanVar(value=included)
        self._row_vars[date_str] = var

        base_bg_included = COLOR_ROW_INCLUDED_EVEN if zebra else COLOR_ROW_INCLUDED_ODD

        row = tk.Frame(self._rows_frame, bg=base_bg_included)
        row.pack(fill="x")

        chk = tk.Checkbutton(
            row,
            variable=var,
            bg=base_bg_included,
            activebackground=base_bg_included,
            selectcolor="#ffffff",
            font=self._font_row,
        )
        chk.pack(side="left", padx=(4, 2), pady=6)

        labels = []
        cells = [
            (date_str, 12, self._font_row_bold),
            (f"{record.monitoring_seconds / 3600:.2f}", 12, self._font_row),
            (f"{record.walk_time:.0f}", 9, self._font_row),
            (f"{record.lick_time:.0f}", 9, self._font_row),
            (f"{record.scratch_time:.0f}", 10, self._font_row),
            (f"{record.shake_count}", 9, self._font_row),
        ]
        for text, width, font in cells:
            lbl = tk.Label(
                row,
                text=text,
                width=width,
                bg=base_bg_included,
                fg=COLOR_ROW_FG_INCLUDED,
                font=font,
                anchor="w",
            )
            lbl.pack(side="left", padx=2, pady=6)
            labels.append(lbl)

        note = ""
        if record.monitoring_seconds < BaselineConfig.MIN_DAILY_MONITORING_SEC_DEFAULT:
            note = "⚠ 監測時數不足門檻，即使勾選採用也不會被算入"
        note_lbl = tk.Label(
            row,
            text=note,
            width=30,
            bg=base_bg_included,
            fg=COLOR_ROW_NOTE_FG,
            font=self._font_note,
            anchor="w",
            justify="left",
        )
        note_lbl.pack(side="left", padx=2, pady=6)
        labels.append(note_lbl)

        self._row_widgets[date_str] = {
            "frame": row,
            "checkbutton": chk,
            "labels": labels,
            "base_bg": base_bg_included,
        }

        def _on_toggle(*_args, _date_str=date_str):
            self._recolor_row(_date_str)
            self._update_live_estimate()

        var.trace_add("write", _on_toggle)
        self._recolor_row(date_str)  # 套用初始狀態的顏色（排除中的天一開始就要標紅）

    def _recolor_row(self, date_str):
        """依目前勾選狀態即時上色：採用＝正常底色，排除＝淡紅底＋紅字，
        不用等按下「套用變更」才看得出差異。"""
        widgets = self._row_widgets.get(date_str)
        var = self._row_vars.get(date_str)
        if widgets is None or var is None:
            return
        included = var.get()
        bg = widgets["base_bg"] if included else COLOR_ROW_EXCLUDED
        fg = COLOR_ROW_FG_INCLUDED if included else COLOR_ROW_FG_EXCLUDED

        widgets["frame"].configure(bg=bg)
        widgets["checkbutton"].configure(bg=bg, activebackground=bg)
        for lbl in widgets["labels"]:
            lbl.configure(bg=bg, fg=fg if lbl is not widgets["labels"][-1] else COLOR_ROW_NOTE_FG)

    def _update_summary(self, records):
        total = len(records)
        excluded_now = len(self._original_excluded)
        self._summary_var.set(
            f"📋 daily_store 共 {total} 天歷史，目前排除 {excluded_now} 天　"
            f"（基線至少需要 {BaselineConfig.MIN_BASELINE_DAYS_DEFAULT} 天、"
            f"最多採用最近 {BaselineConfig.MAX_BASELINE_DAYS_DEFAULT} 天，"
            f"單日監測時數需 ≥ {BaselineConfig.MIN_DAILY_MONITORING_SEC_DEFAULT / 3600:.1f} 小時才算有效）"
        )

    def _predict_valid_counts(self):
        """依目前勾選狀態（不管有沒有按套用）試算：
        - ``after``：排除目前取消勾選的天之後，還剩幾天符合「單日監測時數
          達門檻」；這是 compute_baseline() 實際會採用的天數上限。
        - ``total``：完全不排除任何一天時，符合門檻的天數——
          compute_baseline() 內建一個安全網：排除後不足最低天數時，會直接
          忽略部分排除設定、改用這個 total 兜底（見 baseline.py
          「Fall back to including excluded days...」），所以 total 才是
          「這批資料理論上最多能不能湊到最低天數」的真正答案。
        """
        min_sec = BaselineConfig.MIN_DAILY_MONITORING_SEC_DEFAULT
        after = 0
        total = 0
        for date_str, record in self._records_by_date.items():
            if record.monitoring_seconds < min_sec:
                continue
            total += 1
            var = self._row_vars.get(date_str)
            if var is not None and var.get():
                after += 1
        return after, total

    def _update_live_estimate(self):
        """勾選狀態一變就重算一次，套用前就能看到會不會低於
        BaselineConfig.MIN_BASELINE_DAYS_DEFAULT——這是「阻擋」機制的第一層：
        讓使用者在按下套用之前就先看到後果，不用等套用完才發現。"""
        min_days = BaselineConfig.MIN_BASELINE_DAYS_DEFAULT
        after, total = self._predict_valid_counts()
        if after >= min_days:
            self._estimate_var.set(f"✅ 依目前勾選狀態，預估可採用 {after} 天（≥ 最低需求 {min_days} 天）")
            self._estimate_label.configure(fg=COLOR_SUCCESS_FG)
        elif total >= min_days:
            self._estimate_var.set(
                f"⚠ 依目前勾選狀態，預估可採用天數只剩 {after} 天（低於最低需求 {min_days} 天）——"
                f"套用後 compute_baseline 會自動忽略部分排除設定、改用全部符合門檻的 {total} 天，"
                "你排除的天不會真正生效"
            )
            self._estimate_label.configure(fg=COLOR_WARNING_FG)
        else:
            self._estimate_var.set(
                f"⚠ 即使不排除任何一天，符合監測時數門檻的也只有 {total} 天（低於最低需求 {min_days} 天）"
                "，目前無論怎麼排除都算不出基線，需要更多天數的資料"
            )
            self._estimate_label.configure(fg=COLOR_ERROR_FG)

    # ── 快捷全選/全不選 ──────────────────────────────────────────────────

    def _select_all(self):
        for var in self._row_vars.values():
            var.set(True)

    def _deselect_all(self):
        for var in self._row_vars.values():
            var.set(False)

    # ── 套用變更 ─────────────────────────────────────────────────────────

    def _apply_changes(self):
        # 第二層阻擋：真的要落地寫入之前，先攔一次會讓有效天數低於
        # BaselineConfig.MIN_BASELINE_DAYS_DEFAULT 的情況，讓使用者明確
        # 確認要不要繼續（而不是套用完才在結果訊息裡發現）。
        min_days = BaselineConfig.MIN_BASELINE_DAYS_DEFAULT
        after, total = self._predict_valid_counts()
        if after < min_days:
            if total >= min_days:
                proceed = messagebox.askyesno(
                    "排除天數過多",
                    f"依目前勾選狀態，預估可採用天數只剩 {after} 天，"
                    f"低於最低需求 {min_days} 天。\n\n"
                    f"套用後 compute_baseline 會自動忽略部分排除設定、"
                    f"改用全部符合監測時數門檻的 {total} 天來計算，"
                    "你排除的天不會真正生效。\n\n是否仍要套用？",
                    icon="warning",
                )
            else:
                proceed = messagebox.askyesno(
                    "資料天數不足",
                    f"即使不排除任何一天，符合監測時數門檻的也只有 {total} 天，"
                    f"低於最低需求 {min_days} 天，目前無論怎麼排除都算不出基線。"
                    "\n\n是否仍要套用這次的排除設定？",
                    icon="warning",
                )
            if not proceed:
                return

        changed = []
        for date_str, var in self._row_vars.items():
            now_included = var.get()
            was_included = date_str not in self._original_excluded
            if now_included != was_included:
                daily_store.set_excluded(date_str, excluded=not now_included)
                changed.append((date_str, "採用" if now_included else "排除"))

        if not changed:
            messagebox.showinfo("套用變更", "沒有偵測到任何變更，排除清單維持原樣。")
        else:
            summary = "\n".join(f"  {d} → {s}" for d, s in changed)
            print(f"[manage_baseline_history] 已套用 {len(changed)} 筆變更：\n{summary}")

        # 不論有沒有變更，都用「目前實際持久化的狀態」重新算一次基線並顯示
        # 結果——這是使用者要求的「確實套用到基線設定」的直接證明，而不是
        # 「存了就假設有生效」。
        self._recompute_and_show(changed)
        self.reload()

    def _recompute_and_show(self, changed):
        # 這裡只餵給 compute_baseline()（只用得到最近 MAX_BASELINE_DAYS_
        # DEFAULT 天），不像 reload() 要顯示完整歷史列表，所以可以套用
        # limit_days（見 analytics/config.py 的 HISTORY_LOAD_LIMIT_DAYS）。
        history = daily_store.load_history(limit_days=BaselineConfig.HISTORY_LOAD_LIMIT_DAYS)
        excluded_dates = daily_store.load_excluded_dates()
        try:
            baseline = compute_baseline(history, excluded_dates=excluded_dates)
        except InsufficientDataError as e:
            msg = (
                f"⚠ 已套用 {len(changed)} 筆變更，但目前有效天數不足"
                f"（{e.current_days}／需要 {e.required_days} 天），暫時無法算出基線。"
            )
            self._result_label.configure(fg=COLOR_WARNING_FG)
            self._result_var.set(msg)
            messagebox.showwarning("套用變更", msg)
            return

        warn = ""
        if baseline.sanity_warnings:
            warn = "\n⚠ " + "；".join(baseline.sanity_warnings)
        msg = (
            f"✅ 已套用 {len(changed)} 筆變更並重新計算基線："
            f"採用 {baseline.days_count} 天，信心等級「{baseline.confidence}」。"
            f"{warn}"
        )
        self._result_label.configure(
            fg=COLOR_SUCCESS_FG if not warn else COLOR_WARNING_FG
        )
        self._result_var.set(msg)
        messagebox.showinfo("套用變更", msg)


def main():
    app = BaselineHistoryManager()
    app.mainloop()


if __name__ == "__main__":
    main()
