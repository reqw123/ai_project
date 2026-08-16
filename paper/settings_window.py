"""貓咪監測系統 — 獨立設定管理視窗（tkinter GUI）。

管的是 runtime_settings.json（執行期 JSON 覆寫層），不是 config.py 原始碼，也
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
import signal
import subprocess
import sys
import time
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

# ── 視覺樣式：沿用 analytics/manage_baseline_history.py 的同一組常數 ─────────

_FONT_FAMILY = "Microsoft JhengHei"

COLOR_BG_MAIN = "#eef2f6"
COLOR_HEADER_BG = "#2c3e50"
COLOR_HEADER_FG = "#ffffff"
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

BTN_PRIMARY_BG = "#27ae60"
BTN_PRIMARY_ACTIVE = "#219150"
BTN_SECONDARY_BG = "#5d7285"
BTN_SECONDARY_ACTIVE = "#4a5d6e"
BTN_WARN_BG = "#e67e22"
BTN_WARN_ACTIVE = "#cf711d"
BTN_NEUTRAL_BG = "#95a5a6"
BTN_NEUTRAL_ACTIVE = "#7f8c8d"

BADGE_ENV_BG = "#f5b7b1"
BADGE_ENV_FG = "#78281f"
BADGE_JSON_BG = "#a9cce3"
BADGE_JSON_FG = "#1b4f72"
BADGE_DEFAULT_BG = "#d5dbdb"
BADGE_DEFAULT_FG = "#4d5656"
BADGE_FORM_BG = "#f9e79f"
BADGE_FORM_FG = "#7d6608"

# 布林旗標（開關型設定）的視覺樣式——刻意跟數字/字串欄位的樸素外觀拉開差異，
# 讓「這是一個開關」在掃視整頁時能立刻被認出來，不用逐行讀 label 文字。
FLAG_ON_BG = "#27ae60"
FLAG_ON_FG = "#ffffff"
FLAG_OFF_BG = "#aeb6bf"
FLAG_OFF_FG = "#2c3e50"
FLAG_ROW_BG = "#f4faf6"  # 布林欄位那一整列的底色，跟其他欄位列的白底做出區隔

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


def _styled_button(parent, text, command, bg, active_bg, fg="#ffffff", font=None):
    return tk.Button(
        parent, text=text, command=command, bg=bg, fg=fg,
        activebackground=active_bg, activeforeground=fg, relief="flat",
        font=font or (_FONT_FAMILY, 12, "bold"), padx=14, pady=6, cursor="hand2",
    )


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
        canvas = tk.Canvas(self, bg=COLOR_TAB_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.body = tk.Frame(canvas, bg=COLOR_TAB_BG)
        self.body.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.body, anchor="nw")
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
        self.minsize(1000, 640)
        self._size_window_near_fullscreen()

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
        self._main_process = None  # 由本視窗啟動的 main.py 子行程；None＝本視窗未啟動過
        # 關閉本視窗（不管是按右上角 X 還是下方「關閉」按鈕）視同 main.py 關閉請求，
        # 避免不小心關掉設定視窗後，main.py 還在背景跑、卻再也找不到入口能停止它。
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

        self._build_header()
        self._build_process_bar()
        self._build_info_bar()
        self._build_tabs()
        self._build_bottom_bar()

        self._populate_from_effective_state()
        self._poll_main_process()

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

    # ── 版面 ─────────────────────────────────────────────────────────

    def _build_header(self):
        header = tk.Frame(self, bg=COLOR_HEADER_BG)
        header.pack(fill="x")

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
            text="管理 runtime_settings.json（執行期覆寫）；不會修改 config.py 原始碼，"
            "也不會碰 ST-GCN 訓練設定（stgcn_config.yaml）",
            bg=COLOR_HEADER_BG, fg=COLOR_HEADER_SUB_FG, font=self._font_subtitle, anchor="w",
        ).pack(fill="x", padx=18, pady=(2, 14))

    def _build_process_bar(self):
        """main.py 啟動／關閉／縮小本視窗——放在標題正下方、永遠可見（不用捲動就找得到），
        對應「專案規模大、常常找不到 main.py 入口」的痛點：設定存好後直接在這裡啟動，
        不用再去檔案總管或終端機找路徑。"""
        bar = tk.Frame(self, bg=COLOR_HEADER_BG)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=COLOR_HEADER_BG)
        inner.pack(fill="x", padx=18, pady=(0, 10))

        self._process_status_var = tk.StringVar(value="🖥️ main.py 尚未啟動")
        tk.Label(
            inner, textvariable=self._process_status_var, bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG,
            font=self._font_label_bold, anchor="w",
        ).pack(side="left")

        _styled_button(
            inner, "🗕 縮小視窗", self._on_minimize, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
            font=self._font_hint,
        ).pack(side="right", padx=(8, 0))
        self._stop_main_btn = _styled_button(
            inner, "⏹ 關閉 main.py", self._on_stop_main, BTN_WARN_BG, BTN_WARN_ACTIVE,
            font=self._font_hint,
        )
        self._stop_main_btn.pack(side="right", padx=(8, 0))
        self._start_main_btn = _styled_button(
            inner, "▶ 啟動 main.py", self._on_start_main, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE,
            font=self._font_hint,
        )
        self._start_main_btn.pack(side="right", padx=(8, 0))

        self._update_process_buttons_state()

    def _on_minimize(self):
        self.iconify()

    def _on_window_close(self):
        """視窗關閉的唯一入口（右上角 X 與下方「關閉」按鈕都走這裡）：main.py 若還在
        本視窗啟動的範圍內執行中，視同一併請求關閉，並實際等到確認結束才讓視窗消失，
        不是送出信號就放著不管——避免「視窗關掉了，但 main.py 其實還在跑」的情況。"""
        if self._main_process is not None and self._main_process.poll() is None:
            if not messagebox.askyesno(
                "關閉設定視窗",
                f"main.py 目前正在執行中（PID {self._main_process.pid}）。\n\n"
                "關閉本視窗會一併送出關閉信號給 main.py，避免關掉視窗後失去控制、"
                "無法再停止正在執行的系統。\n\n是否繼續？",
            ):
                return
            self._process_status_var.set("🖥️ 正在關閉 main.py…")
            self.update_idletasks()
            if not self._request_main_shutdown():
                messagebox.showwarning(
                    "關閉設定視窗",
                    "main.py 似乎沒有在預期時間內結束，請自行檢查工作管理員確認狀態。",
                )
        self.destroy()

    def _on_start_main(self):
        if self._main_process is not None and self._main_process.poll() is None:
            messagebox.showinfo(
                "啟動 main.py", f"main.py 已經在執行中（PID {self._main_process.pid}）。"
            )
            return
        if not _MAIN_PY_PATH.exists():
            messagebox.showerror("啟動 main.py", f"找不到 main.py：{_MAIN_PY_PATH}")
            return
        if not messagebox.askyesno(
            "啟動 main.py",
            "即將啟動 main.py，套用目前 runtime_settings.json／環境變數的設定內容。\n\n"
            "若您在下方設定表單中有修改但尚未按「儲存設定」，這些修改不會套用到這次啟動。\n\n"
            "是否繼續？",
        ):
            return
        try:
            popen_kwargs = {"cwd": str(_SCRIPT_DIR)}
            if os.name == "nt":
                # 只用 CREATE_NEW_PROCESS_GROUP（不加 CREATE_NEW_CONSOLE）：
                # Windows 的 GenerateConsoleCtrlEvent（CTRL_BREAK_EVENT 底層機制）只能送到
                # 「跟呼叫端共用同一個主控台」的行程群組——先前版本額外加了
                # CREATE_NEW_CONSOLE 讓 main.py 開獨立主控台視窗顯示輸出，結果反而讓
                # 「關閉 main.py」按鈕送出的 CTRL_BREAK_EVENT 永遠送不到（不同主控台，
                # 對方收不到信號），這是關閉沒有生效的根本原因。改成不開新主控台，
                # main.py 的輸出會直接印在啟動本視窗那個終端機裡，換取關閉功能能真正生效。
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self._main_process = subprocess.Popen([sys.executable, str(_MAIN_PY_PATH)], **popen_kwargs)
        except OSError as e:
            messagebox.showerror("啟動 main.py", f"啟動失敗：{e}")
            return
        self._update_process_buttons_state()
        messagebox.showinfo(
            "啟動 main.py",
            f"main.py 已啟動（PID {self._main_process.pid}）。\n\n"
            "輸出會印在啟動本視窗的終端機裡（若您是雙擊執行、沒有終端機視窗，可能看不到輸出，"
            "但不影響 main.py 運作）。",
        )

    def _on_stop_main(self):
        if self._main_process is None or self._main_process.poll() is not None:
            messagebox.showinfo("關閉 main.py", "目前沒有偵測到由本視窗啟動、仍在執行中的 main.py。")
            self._update_process_buttons_state()
            return
        self._process_status_var.set("🖥️ 正在關閉 main.py…")
        self.update_idletasks()
        stopped = self._request_main_shutdown()
        self._update_process_buttons_state()
        if stopped:
            messagebox.showinfo("關閉 main.py", "main.py 已確認結束。")
        else:
            messagebox.showwarning(
                "關閉 main.py",
                "已送出關閉信號並嘗試強制結束，但仍無法確認 main.py 已停止，"
                "請自行檢查工作管理員確認狀態。",
            )

    def _request_main_shutdown(self, graceful_timeout=4.0, force_timeout=2.0) -> bool:
        """請求 main.py 結束，回傳 True 代表最終確認已經不在執行。

        兩段式：先送 CTRL_BREAK_EVENT（main.py 的 _install_hard_shutdown_on_ctrl_c()
        接管 SIGBREAK，會先嘗試清理 CSV/segment logger/Node-RED session 再結束，
        本身內建 3 秒 watchdog），等 graceful_timeout 秒讓它有機會走完清理流程；
        逾時仍在跑，才 terminate() 強制砍斷，再等 force_timeout 秒確認真的死透。
        「關閉 main.py」按鈕跟「關掉設定視窗」共用這個函式，確保兩個入口都是
        真的確認生效、不是送出信號就假設成功。
        """
        if self._main_process is None or self._main_process.poll() is not None:
            return True
        try:
            if os.name == "nt":
                self._main_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self._main_process.send_signal(signal.SIGINT)
        except OSError:
            pass

        deadline = time.monotonic() + graceful_timeout
        while time.monotonic() < deadline:
            if self._main_process.poll() is not None:
                return True
            time.sleep(0.2)

        if self._main_process.poll() is None:
            try:
                self._main_process.terminate()
            except OSError:
                pass
            deadline = time.monotonic() + force_timeout
            while time.monotonic() < deadline:
                if self._main_process.poll() is not None:
                    return True
                time.sleep(0.2)

        return self._main_process.poll() is not None

    def _update_process_buttons_state(self):
        running = self._main_process is not None and self._main_process.poll() is None
        if running:
            self._process_status_var.set(f"🖥️ main.py 執行中（PID {self._main_process.pid}）")
            self._start_main_btn.config(state="disabled")
            self._stop_main_btn.config(state="normal")
        else:
            was_started = self._main_process is not None
            self._process_status_var.set("🖥️ main.py 已結束" if was_started else "🖥️ main.py 尚未啟動")
            self._start_main_btn.config(state="normal")
            self._stop_main_btn.config(state="disabled")

    def _poll_main_process(self):
        """每 2 秒檢查一次子行程是否還活著——main.py 有可能不是被「關閉」按鈕結束的
        （使用者直接把主控台視窗叉掉、或程式自己崩潰），這裡確保按鈕狀態不會卡住。"""
        if self._main_process is not None:
            self._update_process_buttons_state()
        self.after(2000, self._poll_main_process)

    def _build_info_bar(self):
        wrap = tk.Frame(self, bg=COLOR_BG_MAIN)
        wrap.pack(fill="x", padx=16, pady=(12, 6))
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
            self._info_var.set(f"📄 runtime_settings.json：{path}")
        else:
            self._mtime_var.set("🕒 最後修改：（尚未建立設定檔）")
            self._info_var.set(
                f"📄 runtime_settings.json：{path}（尚未建立）\n"
                "⚠ 目前僅使用環境變數／config.py 內建預設值；按「儲存設定」後才會建立此檔案。"
            )

    def _build_tabs(self):
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
        tab_bar_outer = tk.Frame(self, bg=COLOR_BG_MAIN)
        tab_bar_outer.pack(fill="x", padx=16, pady=(0, 4))

        tab_canvas = tk.Canvas(tab_bar_outer, bg=COLOR_BG_MAIN, highlightthickness=0)
        tab_hscroll = ttk.Scrollbar(tab_bar_outer, orient="horizontal", command=tab_canvas.xview)
        tab_bar = tk.Frame(tab_canvas, bg=COLOR_BG_MAIN)
        tab_bar.bind(
            "<Configure>", lambda e: tab_canvas.configure(scrollregion=tab_canvas.bbox("all"))
        )
        tab_canvas.create_window((0, 0), window=tab_bar, anchor="nw")
        tab_canvas.configure(xscrollcommand=tab_hscroll.set)
        tab_canvas.pack(side="top", fill="x")
        tab_hscroll.pack(side="top", fill="x")

        def _on_tabbar_wheel(event):
            tab_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

        tab_canvas.bind("<Enter>", lambda e: tab_canvas.bind_all("<MouseWheel>", _on_tabbar_wheel))
        tab_canvas.bind("<Leave>", lambda e: tab_canvas.unbind_all("<MouseWheel>"))

        content_area = tk.Frame(self, bg=COLOR_TAB_BG)
        content_area.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        self._tab_buttons = {}
        self._tab_frames = {}
        self._tab_accents = {}

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

            banner = tk.Frame(tab.body, bg=accent)
            banner.pack(fill="x")
            tk.Label(
                banner, text=f"{emoji} {tab_name}", bg=accent, fg="#ffffff",
                font=self._font_banner, anchor="w",
            ).pack(fill="x", padx=14, pady=8)

            for field in fields_by_tab.get(tab_name, []):
                self._build_field_row(tab.body, field, accent)
            if tab_name == "ST-GCN 推論":
                tk.Label(
                    tab.body,
                    text="ℹ️ 訓練用參數（SEQUENCE_LENGTH／FEATURE_MODE／NUM_CLASSES）由"
                    " stgcn_config.yaml 管理，不在此設定視窗顯示或覆寫。",
                    bg=COLOR_TAB_BG, fg=COLOR_HINT_FG, font=self._font_hint,
                    anchor="w", justify="left", wraplength=1200,
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

    def _build_bottom_bar(self):
        bottom = tk.Frame(self, bg=COLOR_BG_MAIN)
        bottom.pack(fill="x", padx=16, pady=(0, 12))
        _styled_button(bottom, "載入目前設定", self._on_load_current, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE).pack(side="left")
        _styled_button(bottom, "還原 GUI 預設值", self._on_restore_defaults, BTN_WARN_BG, BTN_WARN_ACTIVE).pack(side="left", padx=(8, 0))
        _styled_button(bottom, "匯出設定", self._on_export, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE).pack(side="left", padx=(8, 0))
        _styled_button(bottom, "匯入設定", self._on_import, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE).pack(side="left", padx=(8, 0))
        _styled_button(bottom, "關閉", self._on_window_close, BTN_NEUTRAL_BG, BTN_NEUTRAL_ACTIVE).pack(side="right")
        _styled_button(bottom, "儲存設定", self._on_save, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE).pack(side="right", padx=(0, 8))

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
        tk.Frame(container, bg=accent, width=4).pack(side="left", fill="y")

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

        info = {"field": field, "badge_var": badge_var, "badge_widget": badge}

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

            def _browse(v=var, bf=browse_filter):
                filetypes = [bf, ("所有檔案", "*.*")] if bf else [("所有檔案", "*.*")]
                path = filedialog.askopenfilename(title="選擇檔案", filetypes=filetypes)
                if path:
                    v.set(path)

            _styled_button(control, "瀏覽...", _browse, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, font=self._font_hint).pack(side="left", padx=(6, 0))
            info["var"] = var
        elif vt == "folder":
            var = tk.StringVar()
            tk.Entry(control, textvariable=var, font=self._font_label).pack(side="left", fill="x", expand=True)

            def _browse(v=var):
                path = filedialog.askdirectory(title="選擇資料夾")
                if path:
                    v.set(path)

            _styled_button(control, "選擇資料夾...", _browse, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, font=self._font_hint).pack(side="left", padx=(6, 0))
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
                font=self._font_hint,
            ).pack(side="left", padx=(6, 0))

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
                font=self._font_hint, anchor="w", justify="left", wraplength=1200,
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
            font=self._font_hint, anchor="w", justify="left", wraplength=880,
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
                "需先取消該環境變數才會改用 runtime_settings.json 的值。"
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
        messagebox.showinfo("載入目前設定", "已重新讀取環境變數／runtime_settings.json／內建預設值。")

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
            "是否繼續？\n\n此動作僅套用到表單，仍需按「儲存設定」才會寫入 runtime_settings.json。",
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
        path = filedialog.asksaveasfilename(
            title="匯出設定", defaultextension=".json",
            initialfile="runtime_settings_export.json", filetypes=[("JSON", "*.json")],
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
            messagebox.showinfo("匯入設定", "已套用到表單，請按「儲存設定」以正式寫入 runtime_settings.json。")

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

        _styled_button(btn_row, "取消", dialog.destroy, BTN_NEUTRAL_BG, BTN_NEUTRAL_ACTIVE).pack(side="right")
        _styled_button(btn_row, "套用到表單", _apply, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE).pack(side="right", padx=(0, 8))

        self.wait_window(dialog)
        return result["apply"]


if __name__ == "__main__":
    SettingsWindow().mainloop()
