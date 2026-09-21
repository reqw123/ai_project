# -*- coding: utf-8 -*-
"""
==================================================================
滑鼠座標控制實驗 (Mouse Coordinate Control Lab)
------------------------------------------------------------------
用途：
    這是一個學習用的 Windows 桌面自動化實驗程式。
    透過 pyautogui 控制滑鼠，並用 Tkinter 建立操作介面。

功能：
    1. 即時顯示滑鼠座標 (X, Y)
    2. 記錄目前滑鼠位置成一個座標點（支援按鈕點擊 或 全域快捷鍵 F8 觸發）
    3. 可累積多個座標點，形成一條「移動路徑」
    4. 播放時依序移動到每個座標、停留、左鍵點擊（支援按鈕 或 快捷鍵 F9 觸發）
    5. 提供速度控制（移動快慢）
    6. 提供停止按鈕（支援按鈕 或 快捷鍵 F10 觸發，可隨時中斷播放）
    7. 啟用 PyAutoGUI 的 FAILSAFE 機制
       （滑鼠快速移到螢幕左上角 (0,0) 會立即拋出例外、中止程式操作）
    8. 全域快捷鍵：即使目前焦點不在本程式視窗上（例如滑鼠正停在瀏覽器/
       遊戲畫面上），也能透過鍵盤觸發記錄/播放/停止，
       原理是透過 pynput 在系統層級安裝「低階鍵盤鉤子」(Low-Level Keyboard Hook)。
    9. 每個座標點可額外附加「按鍵/組合鍵」設定，例如 enter、ctrl+c、alt+tab，
       播放時會在該座標「點擊」之後，額外模擬這個按鍵動作。
    10. 排程模式：可設定「特定時間」自動開始播放，並指定「重複執行次數」，
        程式會在背景倒數，時間一到就依序執行完整軌跡，可重複多次。
    11. 每個座標點可獨立選擇「滑鼠動作」：左鍵點擊、右鍵點擊、或不點擊，
        搭配「按鍵/組合鍵」欄位，可以組合出只點擊、只按鍵、兩者都要、
        或兩者都不觸發（純粹移動滑鼠）等各種情境。
    12. 快速連按：可獨立觸發（不需記錄座標，直接在「目前滑鼠位置」以指定
        每秒點擊次數連續點擊指定次數），也可以「插入」成事件清單裡的一步，
        安插在任兩個座標點中間，播放到該步時就在當下位置原地連續點擊。
    13. 時段循環排程：設定開始時間、結束時間與循環間隔，在指定時段內
        每隔 N 分鐘完整播放一次事件清單；支援跨午夜，並可在啟動程式時若已
        位於有效時段內立即執行一次。
    14. 計時持續執行：以當下為基準，執行指定總時長，每隔指定時間播放一次；
        總時長與間隔皆可精確填入「時 / 分 / 秒」，狀態顯示獨立於時段循環排程。
    15. 每筆座標事件各自保存移動時間、動作前等待、點擊/按鍵重複次數、
        重複間隔與完成後等待；滑鼠只移動一次，只有點擊或按鍵會重複。
    16. 支援純快捷鍵、整段文字輸入、事件啟用/停用/排序/複製，以及逐筆失敗策略。
    17. 執行前 3 秒安全倒數；流程可用 JSON 儲存、載入並自動備份，執行結果
        與錯誤會持久保存，也可匯出為 JSON。啟動時自動讀取預設流程 JSON
        （預設為程式同資料夾的 mouse_flow_default.json，可用「自動採用預設流程」
        按鈕改選路徑並持久保存），找不到只會提示，不影響使用。

快捷鍵一覽：
    F8  = 記錄目前滑鼠座標
    F9  = 開始播放
    F10 = 停止播放（同時也能取消排程中的倒數或重複執行、快速連按）
    F11 = 在目前滑鼠位置開始快速連按

注意事項：
    - 本程式僅供學習 Windows 自動化原理使用。
    - 請勿用於未經授權的帳號操作、遊戲外掛、繞過驗證機制等用途。
    - 執行前建議先在不重要的視窗/畫面上測試，避免誤觸重要按鈕。
    - 需要額外安裝：pip install pyautogui pynput pyperclip
==================================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pyautogui
import threading
import time
import copy
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

try:
    import pyperclip
except ImportError:
    pyperclip = None

from pynput import keyboard as pynput_keyboard

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

COLORS = {
    "app_bg": "#F1F5F9",
    "card": "#FFFFFF",
    "field": "#F8FAFC",
    "border": "#DCE3EC",
    "text": "#172033",
    "muted": "#64748B",
    "header": "#0F172A",
    "header_soft": "#1E293B",
    "primary": "#2563EB",
    "success": "#059669",
    "danger": "#DC2626",
    "warning": "#EA580C",
    "purple": "#7C3AED",
    "teal": "#0F766E",
    "secondary": "#475569",
}

WORKFLOW_FORMAT = "MouseFlowStudio"
WORKFLOW_VERSION = 4


class WorkflowFormatError(ValueError):
    """流程 JSON 格式不符；problems 為完整問題清單，summary 為單行摘要。"""

    MAX_LISTED = 8

    def __init__(self, problems):
        self.problems = list(problems)
        listed = self.problems[:self.MAX_LISTED]
        detail = "\n".join(f"・{problem}" for problem in listed)
        if len(self.problems) > len(listed):
            detail += f"\n…另有 {len(self.problems) - len(listed)} 項問題"
        self.summary = f"格式不符（共 {len(self.problems)} 項問題）：{self.problems[0]}"
        super().__init__(f"流程檔案格式不符，已阻擋匯入：\n{detail}")


class RoundedButton(tk.Canvas):
    def __init__(
        self, master, text, command=None, bg=COLORS["secondary"], fg="white",
        font=("Microsoft JhengHei", 10, "bold"), width=160, height=40,
        radius=12, hover_bg=None
    ):
        try:
            parent_bg = master.cget("bg")
        except tk.TclError:
            parent_bg = COLORS["app_bg"]

        super().__init__(
            master, width=width, height=height, bg=parent_bg,
            highlightthickness=0, bd=0, cursor="hand2", takefocus=True
        )
        self.button_text = text
        self.command = command
        self.normal_bg = bg
        self.hover_bg = hover_bg or self._shade(bg, 1.10)
        self.pressed_bg = self._shade(bg, 0.86)
        self.text_color = fg
        self.text_font = font
        self.radius = radius
        self.button_height = height
        self._current_bg = self.normal_bg
        self._pressed = False

        self.bind("<Configure>", lambda _event: self._draw())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Return>", lambda _event: self._invoke())
        self.bind("<space>", lambda _event: self._invoke())
        self._draw()

    @staticmethod
    def _shade(color, factor):
        if not isinstance(color, str) or len(color) != 7 or not color.startswith("#"):
            return color
        channels = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
        if factor >= 1:
            channels = [int(value + (255 - value) * (factor - 1)) for value in channels]
        else:
            channels = [int(value * factor) for value in channels]
        channels = [max(0, min(255, value)) for value in channels]
        return "#" + "".join(f"{value:02X}" for value in channels)

    def _draw(self):
        self.delete("all")
        width = max(2, self.winfo_width())
        height = max(2, self.winfo_height())
        radius = min(self.radius, width / 2, height / 2)

        points = [
            radius, 1, width - radius, 1,
            width - 1, 1, width - 1, radius,
            width - 1, height - radius, width - 1, height - 1,
            width - radius, height - 1, radius, height - 1,
            1, height - 1, 1, height - radius,
            1, radius, 1, 1,
        ]
        self.create_polygon(
            points, smooth=True, splinesteps=24,
            fill=self._current_bg, outline=self._current_bg
        )
        self.create_text(
            width / 2, height / 2,
            text=self.button_text, fill=self.text_color, font=self.text_font
        )

    def _on_enter(self, _event):
        if not self._pressed:
            self._current_bg = self.hover_bg
            self._draw()

    def _on_leave(self, _event):
        self._pressed = False
        self._current_bg = self.normal_bg
        self._draw()

    def _on_press(self, _event):
        self.focus_set()
        self._pressed = True
        self._current_bg = self.pressed_bg
        self._draw()

    def _on_release(self, event):
        was_pressed = self._pressed
        self._pressed = False
        inside = 0 <= event.x <= self.winfo_width() and 0 <= event.y <= self.winfo_height()
        self._current_bg = self.hover_bg if inside else self.normal_bg
        self._draw()
        if was_pressed and inside:
            self._invoke()

    def _invoke(self):
        if self.command:
            self.command()


class MouseCoordinateLab:

    # 循環類排程的模式名稱：window = 排程 B，timer = 排程 C
    CYCLE_LABELS = {"window": "時段循環排程", "timer": "計時排程"}

    def __init__(self, root):
        self.root = root
        self.root.title("Mouse Flow Studio｜滑鼠自動化工作台")
        self.root.geometry("680x850")
        self.root.minsize(580, 640)
        self.root.resizable(True, True)
        self.root.configure(bg=COLORS["app_bg"])
        self.root.option_add("*Font", ("Microsoft JhengHei", 10))

        self.recorded_points = []
        self.is_playing = False
        self.stop_requested = False
        self.is_scheduling = False
        self.scheduled_tasks = []
        # 排程 B（時段循環）與排程 C（計時持續）各自擁有獨立的狀態，避免顯示互相混淆
        self.cycle_states = {
            mode: {
                "active": False, "start": None, "end": None,
                "next_run": None, "run_count": 0, "last_state": "尚未啟動",
            }
            for mode in self.CYCLE_LABELS
        }
        self.execution_logs = []
        self.log_lock = threading.Lock()
        self.autosave_suspended = False
        self.app_data_dir = self._resolve_app_data_dir()
        self.autosave_path = self.app_data_dir / "mouse_flow_autosave.json"
        self.settings_path = self.app_data_dir / "mouse_flow_settings.json"
        self.default_workflow_path = self._load_default_workflow_setting()
        self.execution_log_path = self.app_data_dir / "execution_history.jsonl"

        self._build_ui()
        self._load_recent_execution_logs()
        self._load_default_workflow_on_startup()
        self._update_current_position()
        self._update_schedule_countdown()
        self._update_cycle_schedule_displays()
        self._start_hotkey_listener()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        status_bar = tk.Frame(self.root, bg=COLORS["header"], padx=14, pady=9)
        status_bar.pack(side="bottom", fill="x")

        self.label_status_dot = tk.Label(
            status_bar, text="●", bg=COLORS["header"], fg="#34D399",
            font=("Arial", 12, "bold")
        )
        self.label_status_dot.pack(side="left", padx=(0, 8))

        self.label_status = tk.Label(
            status_bar,
            text="狀態：待命中｜F8 記錄・F9 播放・F10 停止・F11 快速連按",
            bg=COLORS["header"], fg="#E2E8F0", anchor="w",
            justify="left", font=("Microsoft JhengHei", 9)
        )
        self.label_status.pack(side="left", fill="x", expand=True)

        footer_stop = RoundedButton(
            status_bar, text="■  立即停止  F10", command=self.stop_playback,
            bg=COLORS["danger"], width=145, height=34, radius=10,
            font=("Microsoft JhengHei", 9, "bold")
        )
        footer_stop.pack(side="right", padx=(10, 0))

        content_shell = tk.Frame(self.root, bg=COLORS["app_bg"])
        content_shell.pack(side="top", fill="both", expand=True)

        canvas = tk.Canvas(content_shell, bg=COLORS["app_bg"], highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)

        outer_scrollbar = tk.Scrollbar(content_shell, orient="vertical", command=canvas.yview)
        outer_scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=outer_scrollbar.set)

        self.main_frame = tk.Frame(canvas, bg=COLORS["app_bg"])
        canvas_window = canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_resize(event):
            canvas.itemconfig(canvas_window, width=event.width)

        self.main_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_resize)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        header = tk.Frame(self.main_frame, bg=COLORS["header"], padx=22, pady=20)
        header.pack(fill="x", padx=0, pady=(0, 12))

        tk.Label(
            header, text="MOUSE FLOW STUDIO", bg=COLORS["header"], fg="#60A5FA",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w")
        tk.Label(
            header, text="滑鼠自動化工作台", bg=COLORS["header"], fg="white",
            font=("Microsoft JhengHei", 21, "bold")
        ).pack(anchor="w", pady=(3, 2))
        tk.Label(
            header,
            text="記錄座標、組合點擊流程，並用多種排程模式自動執行。",
            bg=COLORS["header"], fg="#CBD5E1",
            font=("Microsoft JhengHei", 10)
        ).pack(anchor="w")

        shortcut_row = tk.Frame(header, bg=COLORS["header"])
        shortcut_row.pack(fill="x", pady=(14, 0))
        for shortcut, description in (
            ("F8", "記錄座標"), ("F9", "立即播放"),
            ("F10", "安全停止"), ("F11", "快速連按")
        ):
            chip = tk.Frame(shortcut_row, bg=COLORS["header_soft"], padx=9, pady=6)
            chip.pack(side="left", padx=(0, 7))
            tk.Label(
                chip, text=shortcut, bg=COLORS["header_soft"], fg="#93C5FD",
                font=("Consolas", 9, "bold")
            ).pack(side="left")
            tk.Label(
                chip, text=f"  {description}", bg=COLORS["header_soft"], fg="#E2E8F0",
                font=("Microsoft JhengHei", 9)
            ).pack(side="left")

        guide = tk.Frame(
            self.main_frame, bg="#EAF2FF", padx=16, pady=14,
            highlightthickness=1, highlightbackground="#BFDBFE"
        )
        guide.pack(fill="x", padx=18, pady=(0, 10))
        tk.Label(
            guide, text="快速上手｜照著 4 步驟完成第一個流程",
            bg="#EAF2FF", fg="#1E40AF", font=("Microsoft JhengHei", 11, "bold")
        ).pack(anchor="w", pady=(0, 8))

        guide_steps = (
            ("1", "設定動作", "選擇操作、時間與重複次數"),
            ("2", "記錄位置", "滑鼠移到目標後按 F8"),
            ("3", "確認流程", "依清單順序檢查 A、B 等事件"),
            ("4", "開始執行", "按 F9，或設定下方排程"),
        )
        for number, title, description in guide_steps:
            row = tk.Frame(guide, bg="#EAF2FF")
            row.pack(fill="x", pady=2)
            tk.Label(
                row, text=number, width=2, bg=COLORS["primary"], fg="white",
                font=("Segoe UI", 9, "bold")
            ).pack(side="left", padx=(0, 9))
            tk.Label(
                row, text=f"{title}｜", bg="#EAF2FF", fg=COLORS["text"],
                font=("Microsoft JhengHei", 9, "bold")
            ).pack(side="left")
            tk.Label(
                row, text=description, bg="#EAF2FF", fg=COLORS["muted"],
                font=("Microsoft JhengHei", 9)
            ).pack(side="left")

        workflow_card = tk.LabelFrame(
            self.main_frame, text="流程檔案｜JSON 儲存與自動備份", padx=12, pady=12
        )
        workflow_card.pack(fill="x", padx=18, pady=6)
        workflow_card.configure(fg=COLORS["primary"])

        workflow_buttons = tk.Frame(workflow_card)
        workflow_buttons.pack(fill="x")

        workflow_button_specs = [
            ("儲存流程 JSON", self.save_workflow, COLORS["primary"]),
            ("載入流程 JSON", self.load_workflow, COLORS["teal"]),
            ("還原自動備份", self.restore_autosave, COLORS["secondary"]),
            ("自動採用預設流程", self.choose_default_workflow, COLORS["warning"]),
        ]
        for column, (label, handler, color) in enumerate(workflow_button_specs):
            # uniform 讓四欄等寬，固定間距讓四顆按鈕大小與對齊一致
            workflow_buttons.columnconfigure(column, weight=1, uniform="workflow_buttons")
            RoundedButton(
                workflow_buttons, text=label, command=handler, bg=color, width=100, height=40
            ).grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 4, 0 if column == 3 else 4))

        self.label_autosave_status = tk.Label(
            workflow_card,
            text="自動備份：新增、修改、排序或刪除事件後會立即更新",
            fg=COLORS["muted"], justify="left", font=("Microsoft JhengHei", 9)
        )
        self.label_autosave_status.pack(anchor="w", pady=(6, 0))

        self.label_default_flow_status = tk.Label(
            workflow_card,
            text=f"預設流程：{self.default_workflow_path}",
            fg=COLORS["muted"], justify="left", wraplength=760,
            font=("Microsoft JhengHei", 9)
        )
        self.label_default_flow_status.pack(anchor="w", pady=(2, 0))

        frame_pos = tk.LabelFrame(self.main_frame, text="1  即時滑鼠座標", padx=12, pady=12)
        frame_pos.pack(fill="x", padx=18, pady=6)
        frame_pos.configure(fg=COLORS["primary"])

        self.label_current_pos = tk.Label(
            frame_pos, text="X: ---, Y: ---", fg=COLORS["primary"],
            font=("Consolas", 20, "bold")
        )
        self.label_current_pos.pack()

        frame_record = tk.LabelFrame(self.main_frame, text="2  記錄一個動作", padx=12, pady=12)
        frame_record.pack(fill="x", padx=18, pady=6)
        frame_record.configure(fg=COLORS["success"])

        tk.Label(
            frame_record,
            text="先設定這個座標要做的滑鼠／鍵盤動作，再將游標移到目標位置按 F8。",
            fg=COLORS["muted"], justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(0, 5))

        frame_click_type = tk.Frame(frame_record)
        frame_click_type.pack(fill="x", pady=(8, 0))
        tk.Label(frame_click_type, text="滑鼠動作：").pack(side="left")
        self.click_type_var = tk.StringVar(value="left")
        tk.Radiobutton(frame_click_type, text="左鍵點擊", variable=self.click_type_var, value="left").pack(side="left")
        tk.Radiobutton(frame_click_type, text="右鍵點擊", variable=self.click_type_var, value="right").pack(side="left")
        tk.Radiobutton(frame_click_type, text="不點擊", variable=self.click_type_var, value="none").pack(side="left")

        tk.Label(
            frame_record,
            text="說明：選「不點擊」時，這個座標點只會移動滑鼠過去，不會觸發任何點擊，\n"
                 "適合搭配下面的按鍵欄位，做出「只按鍵、不點滑鼠」的動作。",
            fg="#555555", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(3, 0))

        frame_key = tk.Frame(frame_record)
        frame_key.pack(fill="x", pady=(8, 0))
        tk.Label(frame_key, text="按鍵/組合鍵（選填，例：enter、ctrl+c、alt+tab）：").pack(anchor="w")

        frame_key_input_row = tk.Frame(frame_key)
        frame_key_input_row.pack(fill="x")
        self.entry_key = tk.Entry(frame_key_input_row, font=("Consolas", 11))
        self.entry_key.pack(side="left", fill="x", expand=True)

        btn_key_reference = RoundedButton(
            frame_key_input_row, text="按鍵對照表", command=self.show_key_reference,
            bg=COLORS["secondary"], width=120, height=34, radius=10,
            font=("Microsoft JhengHei", 9, "bold")
        )
        btn_key_reference.pack(side="left", padx=(5, 0))

        tk.Label(
            frame_record,
            text="說明：多個鍵用「+」分隔即為組合鍵；只填一個字則是單鍵。\n"
                 "播放時會在該座標「點擊」完成後，接著模擬這個按鍵。\n"
                 "留空表示這個座標點只有滑鼠點擊，不觸發任何按鍵。",
            fg="#555555", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(3, 0))

        frame_text_input = tk.Frame(frame_record)
        frame_text_input.pack(fill="x", pady=(9, 0))
        tk.Label(frame_text_input, text="文字內容（建立純文字輸入事件時使用，支援中文）：").pack(anchor="w")
        self.text_event_input = tk.Text(frame_text_input, height=3, font=("Microsoft JhengHei", 10), wrap="word")
        self.text_event_input.pack(fill="x", pady=(3, 0))
        tk.Label(
            frame_text_input,
            text="文字事件不移動滑鼠，會在目前取得焦點的輸入欄位貼上這段文字。",
            fg=COLORS["muted"], justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(3, 0))

        event_timing_card = tk.Frame(
            frame_record, bg="#F0FDF4", padx=12, pady=10,
            highlightthickness=1, highlightbackground="#BBF7D0"
        )
        event_timing_card.pack(fill="x", pady=(10, 0))

        tk.Label(
            event_timing_card, text="此事件的時間與重複設定",
            bg="#F0FDF4", fg="#166534", font=("Microsoft JhengHei", 10, "bold")
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 7))

        tk.Label(event_timing_card, text="移動到座標（秒）：", bg="#F0FDF4").grid(row=1, column=0, sticky="w", pady=3)
        self.spin_event_move_duration = tk.Spinbox(event_timing_card, from_=0, to=60, increment=0.1, width=8, font=("Consolas", 10))
        self.spin_event_move_duration.grid(row=1, column=1, sticky="ew", padx=(5, 14), pady=3)
        self._replace_spinbox_value(self.spin_event_move_duration, 0.5)

        tk.Label(event_timing_card, text="到達後等待（秒）：", bg="#F0FDF4").grid(row=1, column=2, sticky="w", pady=3)
        self.spin_event_action_delay = tk.Spinbox(event_timing_card, from_=0, to=60, increment=0.1, width=8, font=("Consolas", 10))
        self.spin_event_action_delay.grid(row=1, column=3, sticky="ew", padx=(5, 0), pady=3)
        self._replace_spinbox_value(self.spin_event_action_delay, 0.5)

        tk.Label(event_timing_card, text="動作重複次數：", bg="#F0FDF4").grid(row=2, column=0, sticky="w", pady=3)
        self.spin_event_repeat_count = tk.Spinbox(event_timing_card, from_=1, to=9999, increment=1, width=8, font=("Consolas", 10))
        self.spin_event_repeat_count.grid(row=2, column=1, sticky="ew", padx=(5, 14), pady=3)
        self._replace_spinbox_value(self.spin_event_repeat_count, 1)

        tk.Label(event_timing_card, text="每次重複間隔（秒）：", bg="#F0FDF4").grid(row=2, column=2, sticky="w", pady=3)
        self.spin_event_repeat_interval = tk.Spinbox(event_timing_card, from_=0, to=3600, increment=0.1, width=8, font=("Consolas", 10))
        self.spin_event_repeat_interval.grid(row=2, column=3, sticky="ew", padx=(5, 0), pady=3)
        self._replace_spinbox_value(self.spin_event_repeat_interval, 1.0)

        tk.Label(event_timing_card, text="完成後等待（秒）：", bg="#F0FDF4").grid(row=3, column=0, sticky="w", pady=3)
        self.spin_event_after_wait = tk.Spinbox(event_timing_card, from_=0, to=3600, increment=0.1, width=8, font=("Consolas", 10))
        self.spin_event_after_wait.grid(row=3, column=1, sticky="ew", padx=(5, 14), pady=3)
        self._replace_spinbox_value(self.spin_event_after_wait, 0.5)

        tk.Label(event_timing_card, text="失敗時：", bg="#F0FDF4").grid(row=4, column=0, sticky="w", pady=3)
        failure_policy_frame = tk.Frame(event_timing_card, bg="#F0FDF4")
        failure_policy_frame.grid(row=4, column=1, columnspan=3, sticky="w", pady=3)
        self.failure_policy_var = tk.StringVar(value="stop")
        tk.Radiobutton(failure_policy_frame, text="停止流程", variable=self.failure_policy_var, value="stop").pack(side="left")
        tk.Radiobutton(failure_policy_frame, text="跳過事件", variable=self.failure_policy_var, value="skip").pack(side="left")
        tk.Radiobutton(failure_policy_frame, text="重試", variable=self.failure_policy_var, value="retry").pack(side="left")
        tk.Label(failure_policy_frame, text="次數：").pack(side="left", padx=(6, 0))
        self.spin_event_retry_count = tk.Spinbox(failure_policy_frame, from_=1, to=20, increment=1, width=5, font=("Consolas", 10))
        self.spin_event_retry_count.pack(side="left")
        self._replace_spinbox_value(self.spin_event_retry_count, 2)

        event_timing_card.columnconfigure(1, weight=1)
        event_timing_card.columnconfigure(3, weight=1)

        tk.Label(
            event_timing_card,
            text="執行順序：移動一次 → 到達後等待 → 重複動作 → 完成後等待 → 下一事件\n"
                 "重試次數是失敗後的額外嘗試次數，並會從該事件開頭重新執行。",
            bg="#F0FDF4", fg="#166534", justify="left", font=("Microsoft JhengHei", 9)
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(7, 0))

        btn_record = RoundedButton(
            frame_record, text="＋  記錄目前滑鼠位置   F8", command=self.record_point,
            bg=COLORS["success"], height=44, font=("Microsoft JhengHei", 11, "bold")
        )
        btn_record.pack(fill="x", pady=(10, 0))

        special_event_buttons = tk.Frame(frame_record)
        special_event_buttons.pack(fill="x", pady=(6, 0))
        btn_add_key_event = RoundedButton(
            special_event_buttons, text="＋  新增純快捷鍵事件", command=self.add_key_only_event,
            bg=COLORS["primary"], height=40, font=("Microsoft JhengHei", 10, "bold")
        )
        btn_add_key_event.pack(side="left", fill="x", expand=True, padx=(0, 3))
        btn_add_text_event = RoundedButton(
            special_event_buttons, text="＋  新增文字輸入事件", command=self.add_text_event,
            bg=COLORS["purple"], height=40, font=("Microsoft JhengHei", 10, "bold")
        )
        btn_add_text_event.pack(side="left", fill="x", expand=True, padx=(3, 0))

        tk.Label(
            frame_record,
            text="F8 會新增座標事件；純快捷鍵與文字事件不記錄座標，也不移動滑鼠。",
            fg=COLORS["success"], justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(4, 0))

        frame_list = tk.LabelFrame(
            self.main_frame, text="3  事件流程｜由上往下依序執行", padx=12, pady=12
        )
        frame_list.pack(fill="both", expand=True, padx=18, pady=6)
        frame_list.configure(fg=COLORS["purple"])

        frame_list_body = tk.Frame(frame_list)
        frame_list_body.pack(fill="both", expand=True)
        self.listbox_points = tk.Listbox(frame_list_body, font=("Consolas", 10), height=8)
        self.listbox_points.grid(row=0, column=0, sticky="nsew")

        scrollbar = tk.Scrollbar(frame_list_body, orient="vertical")
        scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar = tk.Scrollbar(frame_list_body, orient="horizontal")
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

        self.listbox_points.config(yscrollcommand=scrollbar.set, xscrollcommand=horizontal_scrollbar.set)
        scrollbar.config(command=self.listbox_points.yview)
        horizontal_scrollbar.config(command=self.listbox_points.xview)
        frame_list_body.rowconfigure(0, weight=1)
        frame_list_body.columnconfigure(0, weight=1)

        tk.Label(
            self.main_frame,
            text="排列提示｜先選取清單中的某一項，再新增動作，就會插入在該項目後方；\n"
                 "未選取時會加到最下方。要修改既有事件，請先選取，再按「載入」→ 修改 →「更新」。",
            fg="#555555", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", padx=22, pady=(2, 0))

        frame_list_btns = tk.Frame(self.main_frame)
        frame_list_btns.pack(fill="x", padx=18, pady=(6, 2))

        btn_load_event = RoundedButton(
            frame_list_btns, text="載入選取事件設定", command=self.load_selected_event,
            bg=COLORS["primary"], height=38
        )
        btn_load_event.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 5))

        btn_update_event = RoundedButton(
            frame_list_btns, text="更新選取事件", command=self.update_selected_event,
            bg=COLORS["teal"], height=38
        )
        btn_update_event.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 5))

        btn_move_up = RoundedButton(
            frame_list_btns, text="↑  向上移動", command=lambda: self.move_selected_event(-1),
            bg=COLORS["secondary"], height=38
        )
        btn_move_up.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(0, 5))

        btn_move_down = RoundedButton(
            frame_list_btns, text="↓  向下移動", command=lambda: self.move_selected_event(1),
            bg=COLORS["secondary"], height=38
        )
        btn_move_down.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(0, 5))

        btn_duplicate_event = RoundedButton(
            frame_list_btns, text="複製選取事件", command=self.duplicate_selected_event,
            bg=COLORS["purple"], height=38
        )
        btn_duplicate_event.grid(row=2, column=0, sticky="ew", padx=(0, 4), pady=(0, 5))

        btn_toggle_event = RoundedButton(
            frame_list_btns, text="啟用／停用事件", command=self.toggle_selected_event,
            bg=COLORS["warning"], height=38
        )
        btn_toggle_event.grid(row=2, column=1, sticky="ew", padx=(4, 0), pady=(0, 5))

        btn_delete = RoundedButton(
            frame_list_btns, text="刪除選取項目", command=self.delete_selected_point,
            bg=COLORS["danger"], height=38
        )
        btn_delete.grid(row=3, column=0, sticky="ew", padx=(0, 4))

        btn_clear = RoundedButton(
            frame_list_btns, text="清空全部項目", command=self.clear_all_points,
            bg=COLORS["secondary"], height=38
        )
        btn_clear.grid(row=3, column=1, sticky="ew", padx=(4, 0))

        frame_list_btns.columnconfigure(0, weight=1)
        frame_list_btns.columnconfigure(1, weight=1)

        self.scale_speed = tk.DoubleVar(value=0.5)
        self.scale_click_delay = tk.DoubleVar(value=0.5)
        self.scale_wait = tk.DoubleVar(value=0.5)

        frame_control = tk.LabelFrame(
            self.main_frame, text="4  執行完整事件流程", padx=12, pady=12
        )
        frame_control.pack(fill="x", padx=18, pady=8)
        frame_control.configure(fg=COLORS["primary"])

        control_buttons = tk.Frame(frame_control)
        control_buttons.pack(fill="x")

        self.btn_play = RoundedButton(
            control_buttons, text="▶  立即播放流程   F9", command=self.start_playback,
            bg=COLORS["primary"], height=48, radius=14, font=("Microsoft JhengHei", 12, "bold")
        )
        self.btn_play.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.btn_stop = RoundedButton(
            control_buttons, text="■  停止所有動作   F10", command=self.stop_playback,
            bg=COLORS["danger"], height=48, radius=14, font=("Microsoft JhengHei", 12, "bold")
        )
        self.btn_stop.pack(side="left", expand=True, fill="x")

        tk.Label(
            frame_control,
            text="按下播放後會先倒數 3 秒；倒數期間也能按 F10 取消。",
            fg=COLORS["muted"], font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(7, 0))

        frame_logs = tk.LabelFrame(self.main_frame, text="執行結果與錯誤紀錄", padx=12, pady=12)
        frame_logs.pack(fill="x", padx=18, pady=6)
        frame_logs.configure(fg=COLORS["secondary"])

        log_list_frame = tk.Frame(frame_logs)
        log_list_frame.pack(fill="both", expand=True)
        self.listbox_logs = tk.Listbox(log_list_frame, font=("Consolas", 9), height=6)
        self.listbox_logs.pack(side="left", fill="both", expand=True)
        log_scrollbar = tk.Scrollbar(log_list_frame, orient="vertical")
        log_scrollbar.pack(side="right", fill="y")
        self.listbox_logs.config(yscrollcommand=log_scrollbar.set)
        log_scrollbar.config(command=self.listbox_logs.yview)

        log_buttons = tk.Frame(frame_logs)
        log_buttons.pack(fill="x", pady=(6, 0))
        btn_export_logs = RoundedButton(
            log_buttons, text="匯出執行紀錄", command=self.export_execution_logs,
            bg=COLORS["primary"], height=36, font=("Microsoft JhengHei", 9, "bold")
        )
        btn_export_logs.pack(side="left", fill="x", expand=True, padx=(0, 4))
        btn_clear_logs = RoundedButton(
            log_buttons, text="清除執行紀錄", command=self.clear_execution_logs,
            bg=COLORS["danger"], height=36, font=("Microsoft JhengHei", 9, "bold")
        )
        btn_clear_logs.pack(side="left", fill="x", expand=True, padx=(4, 0))

        tk.Label(
            frame_logs,
            text="紀錄會自動保存在本機，包含時間、成功、跳過、重試與錯誤結果。",
            fg=COLORS["muted"], font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(5, 0))

        frame_rapid = tk.LabelFrame(self.main_frame, text="進階功能｜快速連按", padx=12, pady=12)
        frame_rapid.pack(fill="x", padx=18, pady=6)
        frame_rapid.configure(fg=COLORS["purple"])

        tk.Label(
            frame_rapid,
            text="用法一：先把滑鼠移到目標位置，按下方按鈕或 F11，\n"
                 "安全倒數 3 秒後在目前位置連續點擊（跟事件清單無關）。\n"
                 "用法二：按「插入快速連按事件」，把這裡的設定變成清單中的一步，\n"
                 "播放時執行到這一步才會觸發，可以插在任兩個座標點中間。",
            fg="#555555", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(0, 8))

        frame_rapid_button_type = tk.Frame(frame_rapid)
        frame_rapid_button_type.pack(fill="x")
        tk.Label(frame_rapid_button_type, text="連按按鍵：").pack(side="left")
        self.rapid_click_button_var = tk.StringVar(value="left")
        tk.Radiobutton(frame_rapid_button_type, text="左鍵", variable=self.rapid_click_button_var, value="left").pack(side="left")
        tk.Radiobutton(frame_rapid_button_type, text="右鍵", variable=self.rapid_click_button_var, value="right").pack(side="left")

        tk.Label(frame_rapid, text="每秒點擊次數（次/秒，數值越大越快）：").pack(anchor="w", pady=(8, 0))
        self.scale_rapid_rate = tk.Scale(frame_rapid, from_=1, to=50, resolution=1, orient="horizontal")
        self.scale_rapid_rate.set(10)
        self.scale_rapid_rate.pack(fill="x")

        frame_rapid_count = tk.Frame(frame_rapid)
        frame_rapid_count.pack(fill="x", pady=(8, 0))
        tk.Label(frame_rapid_count, text="總點擊次數：").pack(side="left")
        self.spin_rapid_count = tk.Spinbox(frame_rapid_count, from_=1, to=9999, width=6, font=("Consolas", 11))
        self.spin_rapid_count.delete(0, "end")
        self.spin_rapid_count.insert(0, "20")
        self.spin_rapid_count.pack(side="left", padx=(5, 0))

        self.btn_rapid_click = RoundedButton(
            frame_rapid, text="⚡  倒數 3 秒後連按   F11", command=self.start_rapid_click,
            bg=COLORS["purple"], height=42, font=("Microsoft JhengHei", 10, "bold")
        )
        self.btn_rapid_click.pack(fill="x", pady=(8, 0))

        self.btn_insert_rapid_click = RoundedButton(
            frame_rapid, text="＋  將連按設定加入事件流程", command=self.insert_rapid_click_event,
            bg="#5B21B6", height=42, font=("Microsoft JhengHei", 10, "bold")
        )
        self.btn_insert_rapid_click.pack(fill="x", pady=(5, 0))

        tk.Label(
            frame_rapid,
            text="提示：快速連按進行中，一樣可以按「⏹ 停止 (F10)」按鈕或快捷鍵中止。",
            fg="#555555", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(3, 0))

        schedule_guide = tk.Frame(
            self.main_frame, bg="#FFF7ED", padx=14, pady=12,
            highlightthickness=1, highlightbackground="#FED7AA"
        )
        schedule_guide.pack(fill="x", padx=18, pady=(12, 5))
        tk.Label(
            schedule_guide, text="如何選擇排程？", bg="#FFF7ED", fg="#9A3412",
            font=("Microsoft JhengHei", 10, "bold")
        ).pack(anchor="w")
        tk.Label(
            schedule_guide,
            text="排程 A：在某個時間執行固定次數。\n排程 B：在開始～結束時段內，每隔 N 分鐘持續執行。\n排程 C：從現在開始計時持續執行指定時長。",
            bg="#FFF7ED", fg="#7C2D12", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(3, 0))

        frame_schedule = tk.LabelFrame(
            self.main_frame, text="排程 A｜指定時間與執行次數", padx=12, pady=12
        )
        frame_schedule.pack(fill="x", padx=18, pady=6)
        frame_schedule.configure(fg=COLORS["warning"])

        self.label_schedule_countdown = tk.Label(
            frame_schedule, text="距離最近排程：尚無排程任務",
            font=("Consolas", 12, "bold"), fg="#E65100"
        )
        self.label_schedule_countdown.pack(anchor="w", pady=(0, 8))

        tk.Label(frame_schedule, text="執行時間（24小時制，格式 HH:MM 或 HH:MM:SS）：").pack(anchor="w")
        self.entry_schedule_time = tk.Entry(frame_schedule, font=("Consolas", 11))
        self.entry_schedule_time.pack(fill="x")

        tk.Label(
            frame_schedule,
            text="若填寫的時間已經過了（例如現在是 15:00，卻填 14:30），\n"
                 "會視為「明天的這個時間」自動執行，不會立刻觸發。",
            fg="#555555", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(2, 8))

        frame_repeat = tk.Frame(frame_schedule)
        frame_repeat.pack(fill="x")
        tk.Label(frame_repeat, text="完整事件流程重複次數：").pack(side="left")
        self.spin_repeat = tk.Spinbox(frame_repeat, from_=1, to=999, width=6, font=("Consolas", 11))
        self.spin_repeat.delete(0, "end")
        self.spin_repeat.insert(0, "1")
        self.spin_repeat.pack(side="left", padx=(5, 0))

        tk.Label(frame_schedule, text="每次完整流程之間的間隔秒數（可精細到 0.1 秒）：").pack(anchor="w", pady=(8, 0))
        self.scale_interval = tk.Scale(frame_schedule, from_=0, to=60, resolution=0.1, orient="horizontal")
        self.scale_interval.set(2)
        self.scale_interval.pack(fill="x")

        self.btn_add_task = RoundedButton(
            frame_schedule, text="＋  將此設定加入排程佇列", command=self.add_schedule_task,
            bg=COLORS["warning"], height=42, font=("Microsoft JhengHei", 10, "bold")
        )
        self.btn_add_task.pack(fill="x", pady=(8, 0))

        tk.Label(frame_schedule, text="排程任務佇列（依執行時間排序）：").pack(anchor="w", pady=(10, 0))

        frame_schedule_list = tk.Frame(frame_schedule)
        frame_schedule_list.pack(fill="both", expand=True)
        self.listbox_schedule = tk.Listbox(frame_schedule_list, font=("Consolas", 10), height=5)
        self.listbox_schedule.pack(side="left", fill="both", expand=True)

        schedule_scrollbar = tk.Scrollbar(frame_schedule_list, orient="vertical")
        schedule_scrollbar.pack(side="right", fill="y")
        self.listbox_schedule.config(yscrollcommand=schedule_scrollbar.set)
        schedule_scrollbar.config(command=self.listbox_schedule.yview)

        frame_schedule_btns = tk.Frame(frame_schedule)
        frame_schedule_btns.pack(fill="x", pady=(5, 0))
        self.btn_delete_task = RoundedButton(
            frame_schedule_btns, text="刪除選取任務", command=self.delete_selected_task,
            bg=COLORS["danger"], height=38
        )
        self.btn_delete_task.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.btn_clear_tasks = RoundedButton(
            frame_schedule_btns, text="清空全部任務", command=self.clear_all_tasks,
            bg=COLORS["secondary"], height=38
        )
        self.btn_clear_tasks.pack(side="left", expand=True, fill="x")

        self.btn_schedule = RoundedButton(
            frame_schedule, text="▶  啟動排程佇列", command=self.start_schedule_queue,
            bg=COLORS["success"], height=44, font=("Microsoft JhengHei", 11, "bold")
        )
        self.btn_schedule.pack(fill="x", pady=(10, 0))

        tk.Label(
            frame_schedule,
            text="提示：每輪會在排定時間前 3 秒顯示安全倒數；等待、倒數或執行中都可按 F10 中止。\n"
                 "執行中無法新增或刪除任務，需先停止。",
            fg="#555555", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(3, 0))

        # -------------------- 區塊五之三：開始～結束時段循環排程 --------------------
        frame_window_schedule = tk.LabelFrame(
            self.main_frame, text="排程 B｜開始～結束時段循環", padx=12, pady=12
        )
        frame_window_schedule.pack(fill="x", padx=18, pady=6)
        frame_window_schedule.configure(fg=COLORS["teal"])

        self.label_window_schedule = tk.Label(
            frame_window_schedule,
            text="時段排程：尚未啟動",
            font=("Consolas", 11, "bold"), fg="#00695C",
            justify="left", anchor="w"
        )
        self.label_window_schedule.pack(fill="x", pady=(0, 8))

        frame_window_times = tk.Frame(frame_window_schedule)
        frame_window_times.pack(fill="x")

        # 時/分欄位輸入限制：只能是最多 2 位數字，"：" 固定為 Label 不可編輯
        digits_only_vcmd = (self.root.register(self._validate_hm_digits), "%P")

        # 開始時間 (HH:MM)
        tk.Label(frame_window_times, text="開始時間：").grid(row=0, column=0, sticky="w")
        start_frame = tk.Frame(frame_window_times, bg=COLORS["card"])
        start_frame.grid(row=0, column=1, sticky="w", padx=(3, 12))
        self.entry_start_h = tk.Entry(
            start_frame, width=3, font=("Consolas", 11), justify="center",
            validate="key", validatecommand=digits_only_vcmd
        )
        self.entry_start_h.pack(side="left")
        tk.Label(start_frame, text=":", font=("Consolas", 11, "bold"), bg=COLORS["card"]).pack(side="left")
        self.entry_start_m = tk.Entry(
            start_frame, width=3, font=("Consolas", 11), justify="center",
            validate="key", validatecommand=digits_only_vcmd
        )
        self.entry_start_m.pack(side="left")

        # 結束時間 (HH:MM)
        tk.Label(frame_window_times, text="結束時間：").grid(row=0, column=2, sticky="w")
        end_frame = tk.Frame(frame_window_times, bg=COLORS["card"])
        end_frame.grid(row=0, column=3, sticky="w", padx=(3, 0))
        self.entry_end_h = tk.Entry(
            end_frame, width=3, font=("Consolas", 11), justify="center",
            validate="key", validatecommand=digits_only_vcmd
        )
        self.entry_end_h.pack(side="left")
        tk.Label(end_frame, text=":", font=("Consolas", 11, "bold"), bg=COLORS["card"]).pack(side="left")
        self.entry_end_m = tk.Entry(
            end_frame, width=3, font=("Consolas", 11), justify="center",
            validate="key", validatecommand=digits_only_vcmd
        )
        self.entry_end_m.pack(side="left")

        # 預填「一分鐘後開始、一小時後結束」
        default_start = datetime.now() + timedelta(minutes=1)
        default_end = default_start + timedelta(hours=1)
        self.entry_start_h.insert(0, default_start.strftime("%H"))
        self.entry_start_m.insert(0, default_start.strftime("%M"))
        self.entry_end_h.insert(0, default_end.strftime("%H"))
        self.entry_end_m.insert(0, default_end.strftime("%M"))

        frame_window_times.columnconfigure(1, weight=1)
        frame_window_times.columnconfigure(3, weight=1)

        tk.Label(
            frame_window_schedule,
            text="若結束時間早於開始時間，會自動視為跨午夜（例如 22:00～06:00）。",
            fg="#555555", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(3, 8))

        frame_window_interval = tk.Frame(frame_window_schedule)
        frame_window_interval.pack(fill="x")
        tk.Label(frame_window_interval, text="每隔幾分鐘執行一次：").pack(side="left")
        self.spin_window_interval = tk.Spinbox(
            frame_window_interval, from_=1, to=1440, increment=1,
            width=8, font=("Consolas", 11)
        )
        self.spin_window_interval.delete(0, "end")
        self.spin_window_interval.insert(0, "10")
        self.spin_window_interval.pack(side="left", padx=(5, 0))

        self.window_run_now_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            frame_window_schedule,
            text="若目前已在有效時段內，啟動後立即執行一次",
            variable=self.window_run_now_var
        ).pack(anchor="w", pady=(8, 0))

        tk.Label(
            frame_window_schedule,
            text="每次觸發都會完整播放上方事件清單一次。\n"
                 "到達結束時間會停止目前流程；長時間的單次滑鼠移動最多會延遲到該次移動結束。",
            fg="#555555", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(3, 8))

        self.btn_window_schedule = RoundedButton(
            frame_window_schedule,
            text="▶  啟動時段循環排程",
            command=self.start_window_schedule,
            bg=COLORS["teal"], height=44,
            font=("Microsoft JhengHei", 11, "bold")
        )
        self.btn_window_schedule.pack(fill="x")

        # -------------------- 區塊五之四：計時持續執行排程 --------------------
        frame_timer_schedule = tk.LabelFrame(
            self.main_frame, text="排程 C｜計時持續執行", padx=12, pady=12
        )
        frame_timer_schedule.pack(fill="x", padx=18, pady=6)
        frame_timer_schedule.configure(fg=COLORS["purple"])

        self.label_timer_schedule = tk.Label(
            frame_timer_schedule,
            text="計時排程：尚未啟動",
            font=("Consolas", 11, "bold"), fg=COLORS["purple"],
            justify="left", anchor="w"
        )
        self.label_timer_schedule.pack(fill="x", pady=(0, 8))

        frame_timer_inputs = tk.Frame(frame_timer_schedule)
        frame_timer_inputs.pack(fill="x")

        # 時/分/秒欄位：只能輸入最多 2 位數字，單位文字固定為 Label 不可編輯
        timer_digits_vcmd = (self.root.register(self._validate_hm_digits), "%P")

        def build_hms_row(row, caption, default_h, default_m, default_s):
            tk.Label(frame_timer_inputs, text=caption).grid(row=row, column=0, sticky="w", pady=3)
            hms_frame = tk.Frame(frame_timer_inputs, bg=COLORS["card"])
            hms_frame.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=3)
            entries = []
            for unit, default in (("時", default_h), ("分", default_m), ("秒", default_s)):
                entry = tk.Entry(
                    hms_frame, width=3, font=("Consolas", 11), justify="center",
                    validate="key", validatecommand=timer_digits_vcmd
                )
                entry.insert(0, default)
                entry.pack(side="left")
                tk.Label(hms_frame, text=f" {unit} ", bg=COLORS["card"]).pack(side="left")
                entries.append(entry)
            return entries

        self.entry_timer_total_h, self.entry_timer_total_m, self.entry_timer_total_s = build_hms_row(
            0, "執行總時長：", "01", "00", "00"
        )
        self.entry_timer_interval_h, self.entry_timer_interval_m, self.entry_timer_interval_s = build_hms_row(
            1, "每次間隔：", "00", "10", "00"
        )

        tk.Label(
            frame_timer_schedule,
            text="時、分、秒皆可填入（分、秒需在 0～59，留空視為 0）。\n"
                 "啟動後立即開始第一輪，並在總時長內每隔指定時間循環執行；\n"
                 "每輪開始前會有 3 秒安全倒數，間隔至少 1 秒，總時長需大於 3 秒。",
            fg="#555555", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(5, 8))

        self.btn_timer_schedule = RoundedButton(
            frame_timer_schedule,
            text="▶  以當下時間為基準開始執行",
            command=self.start_timer_schedule,
            bg=COLORS["purple"], height=44,
            font=("Microsoft JhengHei", 11, "bold")
        )
        self.btn_timer_schedule.pack(fill="x")

        safety_tip = tk.Frame(
            self.main_frame, bg="#FFF1F2", padx=14, pady=12,
            highlightthickness=1, highlightbackground="#FECDD3"
        )
        safety_tip.pack(fill="x", padx=18, pady=(12, 18))
        tk.Label(
            safety_tip, text="安全提醒", bg="#FFF1F2", fg="#BE123C",
            font=("Microsoft JhengHei", 10, "bold")
        ).pack(anchor="w")
        tk.Label(
            safety_tip,
            text="任何時候都可按 F10 停止。若程式失控，將滑鼠快速移到螢幕左上角 (0, 0)，\n"
                 "PyAutoGUI FAILSAFE 會在下一個滑鼠動作前強制中止。",
            bg="#FFF1F2", fg="#881337", justify="left",
            font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(3, 0))

        self._apply_modern_widget_styles(self.main_frame)

    def _apply_modern_widget_styles(self, parent):
        system_backgrounds = {"SystemButtonFace", "#d9d9d9", "#D9D9D9"}

        for widget in parent.winfo_children():
            try:
                parent_bg = widget.master.cget("bg")
            except tk.TclError:
                parent_bg = COLORS["app_bg"]

            if isinstance(widget, RoundedButton):
                widget.configure(bg=parent_bg)
                widget._draw()
            elif isinstance(widget, tk.LabelFrame):
                widget.configure(
                    bg=COLORS["card"], bd=0, relief="flat",
                    highlightthickness=1, highlightbackground=COLORS["border"],
                    font=("Microsoft JhengHei", 11, "bold"), labelanchor="nw"
                )
            elif isinstance(widget, tk.Frame):
                current_bg = widget.cget("bg")
                if current_bg in system_backgrounds:
                    widget.configure(bg=parent_bg)
            elif isinstance(widget, tk.Label):
                widget.configure(bg=parent_bg)
                if widget.cget("fg") in {"SystemButtonText", "#000000", "black"}:
                    widget.configure(fg=COLORS["text"])
            elif isinstance(widget, (tk.Radiobutton, tk.Checkbutton)):
                widget.configure(
                    bg=parent_bg, fg=COLORS["text"], activebackground=parent_bg,
                    activeforeground=COLORS["text"], selectcolor=parent_bg,
                    highlightthickness=0, cursor="hand2"
                )
            elif isinstance(widget, (tk.Entry, tk.Spinbox)):
                widget.configure(
                    bg=COLORS["field"], fg=COLORS["text"], insertbackground=COLORS["text"],
                    relief="flat", bd=0, highlightthickness=1,
                    highlightbackground=COLORS["border"], highlightcolor=COLORS["primary"]
                )
            elif isinstance(widget, tk.Text):
                widget.configure(
                    bg=COLORS["field"], fg=COLORS["text"], insertbackground=COLORS["text"],
                    relief="flat", bd=0, highlightthickness=1,
                    highlightbackground=COLORS["border"], highlightcolor=COLORS["primary"],
                    selectbackground=COLORS["primary"]
                )
            elif isinstance(widget, tk.Listbox):
                widget.configure(
                    bg=COLORS["field"], fg=COLORS["text"],
                    selectbackground=COLORS["primary"], selectforeground="white",
                    relief="flat", bd=0, highlightthickness=1,
                    highlightbackground=COLORS["border"], activestyle="none"
                )
            elif isinstance(widget, tk.Scale):
                widget.configure(
                    bg=parent_bg, fg=COLORS["text"], activebackground=COLORS["primary"],
                    troughcolor="#CBD5E1", highlightthickness=0, bd=0
                )
            elif isinstance(widget, tk.Scrollbar):
                widget.configure(
                    bg="#CBD5E1", activebackground="#94A3B8",
                    troughcolor=COLORS["field"], bd=0, relief="flat"
                )

            self._apply_modern_widget_styles(widget)

    def _update_current_position(self):
        x, y = pyautogui.position()
        self.label_current_pos.config(text=f"X: {x}, Y: {y}")
        self.root.after(50, self._update_current_position)

    def _update_schedule_countdown(self):
        if not self.scheduled_tasks:
            self.label_schedule_countdown.config(text="距離最近排程：尚無排程任務")
        else:
            nearest_task = min(self.scheduled_tasks, key=lambda t: t["target_dt"])
            remaining = (nearest_task["target_dt"] - datetime.now()).total_seconds()

            if remaining > 0:
                total_seconds = int(remaining)
                hours, rem = divmod(total_seconds, 3600)
                minutes, seconds = divmod(rem, 60)
                self.label_schedule_countdown.config(
                    text=f"距離最近排程還有 {hours:02d}:{minutes:02d}:{seconds:02d}"
                         f"（將於 {nearest_task['target_dt'].strftime('%H:%M:%S')} 執行）"
                )
            else:
                if self.is_scheduling:
                    self.label_schedule_countdown.config(text="最近的排程任務正在執行中...")
                else:
                    self.label_schedule_countdown.config(
                        text="⚠ 任務時間已到，但尚未啟動！請按下「▶ 啟動排程佇列」"
                    )
        self.root.after(500, self._update_schedule_countdown)

    @staticmethod
    def _format_hms(total_seconds):
        total_seconds = max(0, int(total_seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _build_cycle_display_text(self, mode):
        """組出排程 B / C 各自的狀態文字；兩者資料完全獨立，不會互相顯示。"""
        label = self.CYCLE_LABELS[mode]
        state = self.cycle_states[mode]
        if not state["active"]:
            return f"{label}：{state['last_state']}"

        now = datetime.now()
        start_dt = state["start"]
        end_dt = state["end"]
        next_run = state["next_run"]
        remain_caption = "時段剩餘" if mode == "window" else "總時長剩餘"

        if start_dt is None or end_dt is None:
            return f"{label}：正在準備..."
        if now < start_dt:
            return (
                f"{label}：等待開始 {start_dt.strftime('%Y-%m-%d %H:%M:%S')} "
                f"（倒數 {self._format_hms((start_dt - now).total_seconds())}）"
            )
        if now >= end_dt:
            return f"{label}：已到達結束時間，正在停止..."

        remaining = self._format_hms((end_dt - now).total_seconds())
        done = f"已執行 {state['run_count']} 次"
        if next_run is None:
            return f"{label}：流程執行中｜{done}｜{remain_caption} {remaining}"
        return (
            f"{label}：下次執行倒數 {self._format_hms((next_run - now).total_seconds())}｜"
            f"{remain_caption} {remaining}｜{done}"
        )

    def _update_cycle_schedule_displays(self):
        self.label_window_schedule.config(text=self._build_cycle_display_text("window"))
        self.label_timer_schedule.config(text=self._build_cycle_display_text("timer"))
        self.root.after(500, self._update_cycle_schedule_displays)

    def show_key_reference(self):
        window = tk.Toplevel(self.root)
        window.title("按鍵名稱對照表｜Mouse Flow Studio")
        window.geometry("500x620")
        window.minsize(420, 480)
        window.configure(bg=COLORS["app_bg"])
        window.bind("<Escape>", lambda _e: window.destroy())

        reference_header = tk.Frame(window, bg=COLORS["header"], padx=18, pady=14)
        reference_header.pack(fill="x")
        tk.Label(
            reference_header, text="按鍵與組合鍵對照表", bg=COLORS["header"], fg="white",
            font=("Microsoft JhengHei", 16, "bold")
        ).pack(anchor="w")
        tk.Label(
            reference_header,
            text="將下方名稱填入「按鍵／組合鍵」欄位；多鍵請用 + 分隔。",
            bg=COLORS["header"], fg="#CBD5E1", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(2, 0))

        reference_footer = tk.Frame(window, bg=COLORS["app_bg"], padx=14, pady=10)
        reference_footer.pack(side="bottom", fill="x")
        close_button = RoundedButton(
            reference_footer, text="關閉對照表", command=window.destroy,
            bg=COLORS["secondary"], height=38
        )
        close_button.pack(fill="x")

        reference_body = tk.Frame(window, bg=COLORS["card"], padx=14, pady=14)
        reference_body.pack(fill="both", expand=True, padx=14, pady=(14, 0))

        text_widget = tk.Text(
            reference_body, font=("Consolas", 10), wrap="word",
            bg=COLORS["field"], fg=COLORS["text"], relief="flat", bd=0,
            padx=12, pady=12, selectbackground=COLORS["primary"]
        )
        text_widget.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(reference_body, orient="vertical", command=text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        text_widget.config(yscrollcommand=scrollbar.set)

        reference_text = """
【數字鍵】直接打數字本身即可
0 1 2 3 4 5 6 7 8 9

【英文字母】直接打字母本身即可（大小寫效果相同，皆為小寫按鍵）
a b c d e f g h i j k l m n o p q r s t u v w x y z

【常用符號】直接打符號本身，pyautogui 會自動處理是否需要按 Shift
!  "  #  $  %  &  '  (  )  *  +  ,  -  .  /
:  ;  <  =  >  ?  @  [  \\  ]  ^  _  `
{  |  }  ~

【功能鍵】
f1  f2  f3  f4  f5  f6  f7  f8  f9  f10  f11  f12
f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24

【方向鍵】
up      （上）
down    （下）
left    （左）
right   （右）

【常用特殊鍵】
enter / return   （Enter 鍵）
esc / escape     （Esc 鍵）
tab              （Tab 鍵）
space            （空白鍵）
backspace        （倒退鍵）
delete / del     （Delete 鍵）
home             （Home 鍵）
end              （End 鍵）
pageup / pgup    （Page Up）
pagedown / pgdn  （Page Down）
insert           （Insert 鍵）
capslock         （大寫鎖定）
printscreen      （螢幕截圖 PrtScn）

【修飾鍵（通常搭配組合鍵使用，如 ctrl+c）】
ctrl  / ctrlleft  / ctrlright
alt   / altleft   / altright
shift / shiftleft / shiftright
win   / winleft   / winright   （Windows 鍵）

【多媒體鍵】
volumeup  volumedown  volumemute
playpause  nexttrack  prevtrack

------------------------------------------------------------
組合鍵寫法：用「+」分隔多個鍵名，例如：
    ctrl+c        複製
    ctrl+v        貼上
    ctrl+z        復原
    ctrl+s        儲存
    alt+tab       切換視窗
    ctrl+shift+esc  開啟工作管理員
    win+d         顯示桌面

提醒：符號鍵的對應是基於美式鍵盤佈局，
     若 Windows 目前輸入法/鍵盤佈局不是美式，實際打出的符號可能不同，
     建議先用單一符號測試過一次再放進正式的自動化流程。
"""
        text_widget.insert("1.0", reference_text.strip())
        text_widget.config(state="disabled")

    @staticmethod
    def _replace_spinbox_value(spinbox, value):
        spinbox.delete(0, "end")
        spinbox.insert(0, str(value))

    @staticmethod
    def _validate_hm_digits(proposed_value):
        """排程 B 時/分、排程 C 時/分/秒欄位的輸入限制：只允許空字串或最多 2 位數字，
        擋掉字母、符號與超過 2 位的輸入，"：" 分隔符維持 Label 唯讀。"""
        return proposed_value == "" or (proposed_value.isdigit() and len(proposed_value) <= 2)

    def _read_event_timing_settings(self):
        try:
            move_duration = float(self.spin_event_move_duration.get())
            action_delay = float(self.spin_event_action_delay.get())
            repeat_count = int(self.spin_event_repeat_count.get())
            repeat_interval = float(self.spin_event_repeat_interval.get())
            after_wait = float(self.spin_event_after_wait.get())
            retry_count = int(self.spin_event_retry_count.get())

            if move_duration < 0 or action_delay < 0 or repeat_interval < 0 or after_wait < 0:
                raise ValueError
            if repeat_count < 1 or retry_count < 1:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror(
                "事件設定錯誤",
                "時間與間隔必須是大於等於 0 的數字；動作次數與重試次數必須是大於等於 1 的整數。"
            )
            return None

        return {
            "move_duration": move_duration,
            "action_delay": action_delay,
            "repeat_count": repeat_count,
            "repeat_interval": repeat_interval,
            "after_wait": after_wait,
            "failure_policy": self.failure_policy_var.get(),
            "retry_count": retry_count,
        }

    def _load_timing_into_editor(self, event):
        self._replace_spinbox_value(self.spin_event_move_duration, event.get("move_duration", 0.5))
        self._replace_spinbox_value(self.spin_event_action_delay, event.get("action_delay", 0.5))
        self._replace_spinbox_value(self.spin_event_repeat_count, event.get("repeat_count", 1))
        self._replace_spinbox_value(self.spin_event_repeat_interval, event.get("repeat_interval", 1.0))
        self._replace_spinbox_value(self.spin_event_after_wait, event.get("after_wait", 0.5))
        self.failure_policy_var.set(event.get("failure_policy", "stop"))
        self._replace_spinbox_value(self.spin_event_retry_count, event.get("retry_count", 2))

    def load_selected_event(self):
        selection = self.listbox_points.curselection()
        if not selection:
            messagebox.showinfo("提示", "請先在事件流程中選取要載入的項目")
            return

        index = selection[0]
        event = self.recorded_points[index]
        event_type = event["type"]
        self._load_timing_into_editor(event)

        if event_type == "point":
            self.click_type_var.set(event.get("click_type", "left"))
            self.entry_key.delete(0, "end")
            if event.get("key"):
                self.entry_key.insert(0, event["key"])
        elif event_type == "key":
            self.click_type_var.set("none")
            self.entry_key.delete(0, "end")
            self.entry_key.insert(0, event.get("key", ""))
        elif event_type == "text":
            self.text_event_input.delete("1.0", "end")
            self.text_event_input.insert("1.0", event.get("text", ""))
        elif event_type == "rapid_click":
            self.rapid_click_button_var.set(event.get("button_type", "left"))
            self.scale_rapid_rate.set(event.get("rate", 10))
            self._replace_spinbox_value(self.spin_rapid_count, event.get("count", 20))

        self._set_status(f"已載入事件 {index + 1}（{event_type}），可修改後按「更新選取事件」")

    def update_selected_event(self):
        if self.is_playing:
            messagebox.showinfo("提示", "流程執行中無法修改事件，請先停止")
            return

        selection = self.listbox_points.curselection()
        if not selection:
            messagebox.showinfo("提示", "請先在事件流程中選取要更新的項目")
            return

        index = selection[0]
        event = self.recorded_points[index]
        event_type = event["type"]

        if event_type in ("point", "key", "text"):
            timing = self._read_event_timing_settings()
            if timing is None:
                return
            if event_type == "point":
                key_text = self.entry_key.get().strip()
                event.update({
                    "click_type": self.click_type_var.get(),
                    "key": key_text if key_text else None,
                    **timing,
                })
            elif event_type == "key":
                key_text = self.entry_key.get().strip()
                if not key_text:
                    messagebox.showwarning("提示", "純快捷鍵事件不可留空")
                    return
                event.update({"key": key_text, **timing})
            else:
                text_value = self.text_event_input.get("1.0", "end-1c")
                if not text_value.strip():
                    messagebox.showwarning("提示", "文字輸入事件不可留空")
                    return
                event.update({"text": text_value, **timing})
        else:
            try:
                click_count = int(self.spin_rapid_count.get())
                after_wait = float(self.spin_event_after_wait.get())
                retry_count = int(self.spin_event_retry_count.get())
                if click_count < 1 or after_wait < 0 or retry_count < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("錯誤", "連按次數必須大於等於 1，完成後等待不可小於 0")
                return

            event.update({
                "rate": self.scale_rapid_rate.get(),
                "count": click_count,
                "button_type": self.rapid_click_button_var.get(),
                "after_wait": after_wait,
                "failure_policy": self.failure_policy_var.get(),
                "retry_count": retry_count,
            })

        self._refresh_listbox_labels()
        self.listbox_points.selection_set(index)
        self.listbox_points.see(index)
        self._set_status(f"已更新事件 {index + 1} 的設定 ✅")
        self._autosave_workflow("更新事件")

    def record_point(self):
        if self.is_playing:
            # F8 是全域快捷鍵，執行中誤按不要跳視窗打斷自動化，只在狀態列提示
            self._set_status("流程執行中無法記錄新座標，請先按 F10 停止")
            return
        timing = self._read_event_timing_settings()
        if timing is None:
            return

        x, y = pyautogui.position()
        click_type = self.click_type_var.get()
        key_text = self.entry_key.get().strip()
        key_value = key_text if key_text else None

        event = {
            "type": "point", "x": x, "y": y,
            "click_type": click_type, "key": key_value,
            "enabled": True,
            **timing,
        }

        selection = self.listbox_points.curselection()
        if selection:
            insert_index = selection[0] + 1
            self.recorded_points.insert(insert_index, event)
        else:
            self.recorded_points.append(event)

        self._refresh_listbox_labels()

        click_label = {"left": "左鍵點擊", "right": "右鍵點擊", "none": "不點擊"}[click_type]
        key_status = f"＋按鍵 {key_value}" if key_value else ""
        self._set_status(
            f"已記錄：({x}, {y})，{click_label}{key_status}，"
            f"重複 {timing['repeat_count']} 次／間隔 {timing['repeat_interval']:g} 秒"
        )
        self._autosave_workflow("新增座標事件")

    def _insert_event_after_selection(self, event):
        selection = self.listbox_points.curselection()
        if selection:
            insert_index = selection[0] + 1
            self.recorded_points.insert(insert_index, event)
        else:
            insert_index = len(self.recorded_points)
            self.recorded_points.append(event)
        self._refresh_listbox_labels()
        self.listbox_points.selection_set(insert_index)
        self.listbox_points.see(insert_index)
        return insert_index

    def add_key_only_event(self):
        if self.is_playing:
            messagebox.showinfo("提示", "流程執行中無法新增事件，請先停止")
            return
        key_text = self.entry_key.get().strip()
        if not key_text:
            messagebox.showwarning("提示", "請先在「按鍵／組合鍵」欄位輸入快捷鍵")
            return
        timing = self._read_event_timing_settings()
        if timing is None:
            return
        event = {"type": "key", "key": key_text, "enabled": True, **timing}
        index = self._insert_event_after_selection(event)
        self._set_status(f"已新增純快捷鍵事件 {index + 1}：{key_text} ×{timing['repeat_count']}")
        self._autosave_workflow("新增快捷鍵事件")

    def add_text_event(self):
        if self.is_playing:
            messagebox.showinfo("提示", "流程執行中無法新增事件，請先停止")
            return
        text_value = self.text_event_input.get("1.0", "end-1c")
        if not text_value.strip():
            messagebox.showwarning("提示", "請先輸入要寫入目標欄位的文字內容")
            return
        timing = self._read_event_timing_settings()
        if timing is None:
            return
        event = {"type": "text", "text": text_value, "enabled": True, **timing}
        index = self._insert_event_after_selection(event)
        self._set_status(f"已新增文字輸入事件 {index + 1}：{len(text_value)} 個字元 ×{timing['repeat_count']}")
        self._autosave_workflow("新增文字輸入事件")

    @staticmethod
    def _format_event_label(index, event):
        state_mark = "●" if event.get("enabled", True) else "⏸"
        failure_policy = event.get("failure_policy", "stop")
        if failure_policy == "retry":
            failure_label = f"重試{event.get('retry_count', 2)}次"
        else:
            failure_label = {"stop": "失敗停止", "skip": "失敗跳過"}.get(failure_policy, "失敗停止")

        if event["type"] == "rapid_click":
            button_label = "左鍵" if event["button_type"] == "left" else "右鍵"
            return (
                f"{state_mark} 事件 {index}｜⚡ {button_label}連按 ×{event['count']} "
                f"@ {event['rate']:g}/秒｜後等 {event.get('after_wait', 0.5):g}s｜{failure_label}"
            )

        repeat_count = event.get("repeat_count", 1)
        repeat_interval = event.get("repeat_interval", 1.0)
        action_delay = event.get("action_delay", 0.5)
        after_wait = event.get("after_wait", 0.5)
        timing_label = (
            f"×{repeat_count}（間隔 {repeat_interval:g}s）｜"
            f"前等 {action_delay:g}s・後等 {after_wait:g}s｜{failure_label}"
        )

        if event["type"] == "key":
            return f"{state_mark} 事件 {index}｜⌨ 純快捷鍵 {event.get('key', '')} {timing_label}"

        if event["type"] == "text":
            preview = event.get("text", "").replace("\r", " ").replace("\n", " ")
            if len(preview) > 18:
                preview = preview[:18] + "…"
            return f"{state_mark} 事件 {index}｜✎ 文字「{preview}」 {timing_label}"

        x, y = event["x"], event["y"]
        click_type = event.get("click_type", "left")
        key_value = event.get("key")
        click_label = {"left": "左鍵", "right": "右鍵", "none": "不點擊"}.get(click_type, "左鍵")
        move_duration = event.get("move_duration", 0.5)
        label = f"{state_mark} 點 {index}｜({x}, {y}) {click_label}"
        if key_value:
            label += f"＋{key_value}"
        label += f" {timing_label}｜移動 {move_duration:g}s"
        return label

    def delete_selected_point(self):
        if self.is_playing:
            messagebox.showinfo("提示", "流程執行中無法刪除事件，請先停止")
            return
        selection = self.listbox_points.curselection()
        if not selection:
            messagebox.showinfo("提示", "請先在清單中選取要刪除的項目")
            return

        index = selection[0]
        del self.recorded_points[index]
        self._refresh_listbox_labels()
        self._set_status("已刪除選取的項目")
        self._autosave_workflow("刪除事件")

    def clear_all_points(self):
        if self.is_playing:
            messagebox.showinfo("提示", "流程執行中無法清空事件，請先停止")
            return
        if not self.recorded_points:
            return
        if not messagebox.askyesno("確認清空", "確定要刪除事件流程中的全部項目嗎？"):
            return
        self.recorded_points.clear()
        self.listbox_points.delete(0, tk.END)
        self._set_status("已清空所有項目")
        self._autosave_workflow("清空事件")

    def move_selected_event(self, direction):
        if self.is_playing:
            messagebox.showinfo("提示", "流程執行中無法調整順序，請先停止")
            return
        selection = self.listbox_points.curselection()
        if not selection:
            messagebox.showinfo("提示", "請先選取要移動的事件")
            return

        old_index = selection[0]
        new_index = old_index + direction
        if new_index < 0 or new_index >= len(self.recorded_points):
            self._set_status("事件已位於最上方或最下方")
            return

        self.recorded_points[old_index], self.recorded_points[new_index] = (
            self.recorded_points[new_index], self.recorded_points[old_index]
        )
        self._refresh_listbox_labels()
        self.listbox_points.selection_set(new_index)
        self.listbox_points.see(new_index)
        self._set_status(f"已將事件移動到第 {new_index + 1} 位")
        self._autosave_workflow("調整事件順序")

    def duplicate_selected_event(self):
        if self.is_playing:
            messagebox.showinfo("提示", "流程執行中無法複製事件，請先停止")
            return
        selection = self.listbox_points.curselection()
        if not selection:
            messagebox.showinfo("提示", "請先選取要複製的事件")
            return

        source_index = selection[0]
        new_index = source_index + 1
        self.recorded_points.insert(new_index, copy.deepcopy(self.recorded_points[source_index]))
        self._refresh_listbox_labels()
        self.listbox_points.selection_set(new_index)
        self.listbox_points.see(new_index)
        self._set_status(f"已複製事件，新事件位於第 {new_index + 1} 位")
        self._autosave_workflow("複製事件")

    def toggle_selected_event(self):
        if self.is_playing:
            messagebox.showinfo("提示", "流程執行中無法切換事件狀態，請先停止")
            return
        selection = self.listbox_points.curselection()
        if not selection:
            messagebox.showinfo("提示", "請先選取要啟用或停用的事件")
            return

        index = selection[0]
        event = self.recorded_points[index]
        event["enabled"] = not event.get("enabled", True)
        state_text = "啟用" if event["enabled"] else "停用"
        self._refresh_listbox_labels()
        self.listbox_points.selection_set(index)
        self.listbox_points.see(index)
        self._set_status(f"已{state_text}事件 {index + 1}")
        self._autosave_workflow(f"{state_text}事件")

    def _refresh_listbox_labels(self):
        self.listbox_points.delete(0, tk.END)
        for i, event in enumerate(self.recorded_points, start=1):
            self.listbox_points.insert(tk.END, self._format_event_label(i, event))
            if not event.get("enabled", True):
                self.listbox_points.itemconfig(tk.END, fg="#94A3B8")

    def insert_rapid_click_event(self):
        if self.is_playing:
            messagebox.showinfo("提示", "流程執行中無法新增事件，請先停止")
            return
        rate = self.scale_rapid_rate.get()
        try:
            click_count = int(self.spin_rapid_count.get())
            if click_count < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("錯誤", "總點擊次數必須是大於等於 1 的整數")
            return

        try:
            after_wait = float(self.spin_event_after_wait.get())
            retry_count = int(self.spin_event_retry_count.get())
            if after_wait < 0 or retry_count < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("錯誤", "完成後等待不可小於 0，重試次數必須大於等於 1")
            return

        button_type = self.rapid_click_button_var.get()
        event = {
            "type": "rapid_click", "rate": rate, "count": click_count,
            "button_type": button_type, "after_wait": after_wait,
            "failure_policy": self.failure_policy_var.get(), "retry_count": retry_count,
            "enabled": True,
        }

        selection = self.listbox_points.curselection()
        if selection:
            insert_index = selection[0] + 1
            self.recorded_points.insert(insert_index, event)
        else:
            self.recorded_points.append(event)

        self._refresh_listbox_labels()
        button_label = "左鍵" if button_type == "left" else "右鍵"
        self._set_status(f"已插入快速連按事件：{button_label}，每秒 {rate:g} 次，共 {click_count} 次，完成後等待 {after_wait:g} 秒")
        self._autosave_workflow("新增快速連按事件")

    def start_playback(self):
        if self.is_playing:
            messagebox.showinfo("提示", "目前正在播放中")
            return
        if not self.recorded_points:
            messagebox.showwarning("提示", "事件清單是空的，請先新增至少一個事件")
            return

        self.is_playing = True
        self.stop_requested = False
        self._set_status("播放中...（可按下停止按鈕中斷）")

        speed = self.scale_speed.get()
        click_delay = self.scale_click_delay.get()
        wait_time = self.scale_wait.get()

        thread = threading.Thread(
            target=self._playback_worker,
            args=(speed, click_delay, wait_time),
            daemon=True
        )
        thread.start()

    def _interruptible_sleep(self, seconds, deadline=None):
        end_monotonic = time.monotonic() + max(0.0, seconds)
        while True:
            if self.stop_requested:
                return "stopped"
            if deadline is not None and datetime.now() >= deadline:
                return "deadline"
            remaining = end_monotonic - time.monotonic()
            if remaining <= 0:
                return "completed"
            time.sleep(min(0.1, remaining))

    def _run_start_countdown(self, context="流程", seconds=3, deadline=None):
        for remaining in range(seconds, 0, -1):
            if self.stop_requested:
                self._set_status("執行前倒數已取消")
                return "stopped"
            if deadline is not None and datetime.now() >= deadline:
                self._set_status("倒數期間已到達結束時間")
                return "deadline"
            self._set_status(f"{context}將在 {remaining} 秒後開始，可按 F10 取消")
            wait_result = self._interruptible_sleep(1.0, deadline)
            if wait_result != "completed":
                return wait_result
        return "completed"

    @staticmethod
    def _event_description(event, index=None):
        prefix = f"事件 {index}" if index is not None else "事件"
        event_type = event.get("type")
        if event_type == "point":
            return f"{prefix} 座標 ({event.get('x')}, {event.get('y')})"
        if event_type == "rapid_click":
            return f"{prefix} 快速連按 ×{event.get('count', 1)}"
        if event_type == "key":
            return f"{prefix} 快捷鍵 {event.get('key', '')}"
        if event_type == "text":
            return f"{prefix} 輸入文字（{len(event.get('text', ''))} 字）"
        return f"{prefix} 未知類型"

    @staticmethod
    def _execute_key_value(key_value):
        key_parts = [part.strip().lower() for part in str(key_value).split("+") if part.strip()]
        if not key_parts:
            raise ValueError("快捷鍵內容不可為空")
        if len(key_parts) == 1:
            pyautogui.press(key_parts[0])
        else:
            pyautogui.hotkey(*key_parts)

    @staticmethod
    def _input_text_value(text_value):
        text_value = str(text_value)
        if not text_value:
            raise ValueError("輸入文字不可為空")
        if pyperclip is not None:
            previous_clipboard = None
            clipboard_read = False
            copied = False
            try:
                previous_clipboard = pyperclip.paste()
                clipboard_read = True
            except Exception:
                pass
            try:
                pyperclip.copy(text_value)
                copied = True
                pyautogui.hotkey("ctrl", "v")
                time.sleep(0.05)
                return
            except Exception:
                pass
            finally:
                if copied and clipboard_read:
                    try:
                        pyperclip.copy(previous_clipboard)
                    except Exception:
                        pass
        if text_value.isascii():
            pyautogui.write(text_value, interval=0.01)
            return
        raise RuntimeError("中文輸入需要可用的剪貼簿模組，請安裝 pyperclip 後再試")

    def _execute_event_core(self, index, event, speed, click_delay, wait_time, deadline=None):
        event_type = event.get("type")
        after_wait = event.get("after_wait", wait_time)

        if event_type == "rapid_click":
            self._set_status(f"第 {index} 項：快速連按 {event.get('count', 1)} 次")
            result = self._execute_rapid_click(
                event.get("rate", 10),
                event.get("count", 1),
                event.get("button_type", "left"),
                deadline=deadline,
            )
            if result != "completed":
                return result
            return self._interruptible_sleep(after_wait, deadline)

        if event_type not in {"point", "key", "text"}:
            raise ValueError(f"不支援的事件類型：{event_type}")

        if event_type == "point":
            x, y = event["x"], event["y"]
            self._set_status(f"第 {index} 項：移動到 ({x}, {y})")
            current_x, current_y = pyautogui.position()
            if (current_x, current_y) != (x, y):
                pyautogui.moveTo(x, y, duration=event.get("move_duration", speed))

        if self.stop_requested:
            return "stopped"
        if deadline is not None and datetime.now() >= deadline:
            return "deadline"

        wait_result = self._interruptible_sleep(event.get("action_delay", click_delay), deadline)
        if wait_result != "completed":
            return wait_result

        repeat_count = max(1, int(event.get("repeat_count", 1)))
        repeat_interval = max(0.0, float(event.get("repeat_interval", 1.0)))
        for repeat_index in range(1, repeat_count + 1):
            if self.stop_requested:
                return "stopped"
            if deadline is not None and datetime.now() >= deadline:
                return "deadline"

            self._set_status(f"第 {index} 項：執行動作 {repeat_index}/{repeat_count}")
            if event_type == "point":
                click_type = event.get("click_type", "left")
                if click_type in {"left", "right"}:
                    pyautogui.click(button=click_type)
                if event.get("key"):
                    self._execute_key_value(event["key"])
            elif event_type == "key":
                self._execute_key_value(event.get("key"))
            else:
                self._input_text_value(event.get("text", ""))

            if repeat_index < repeat_count:
                wait_result = self._interruptible_sleep(repeat_interval, deadline)
                if wait_result != "completed":
                    return wait_result

        return self._interruptible_sleep(after_wait, deadline)

    def _execute_event_with_policy(self, index, event, speed, click_delay, wait_time, deadline=None):
        policy = event.get("failure_policy", "stop")
        retry_count = max(1, int(event.get("retry_count", 2)))
        description = self._event_description(event, index)
        attempt = 0

        while True:
            attempt += 1
            try:
                result = self._execute_event_core(index, event, speed, click_delay, wait_time, deadline)
                if result == "completed":
                    self._add_execution_log("success", f"{description} 執行成功")
                return result
            except pyautogui.FailSafeException:
                raise
            except Exception as error:
                error_text = f"{type(error).__name__}: {error}"
                if policy == "skip":
                    self._add_execution_log("warning", f"{description} 失敗後已跳過：{error_text}")
                    self._set_status(f"{description} 失敗，已依設定跳過")
                    return "skipped"
                if policy == "retry" and attempt <= retry_count:
                    self._add_execution_log("warning", f"{description} 失敗，準備重試 {attempt}/{retry_count}：{error_text}")
                    self._set_status(f"{description} 失敗，1 秒後重試 {attempt}/{retry_count}")
                    wait_result = self._interruptible_sleep(1.0, deadline)
                    if wait_result != "completed":
                        return wait_result
                    continue
                self._add_execution_log("error", f"{description} 執行失敗：{error_text}")
                self._set_status(f"{description} 執行失敗，流程已停止")
                return "failed"

    def _run_trajectory_loop(self, speed, click_delay, wait_time, deadline=None):
        for index, event in enumerate(self.recorded_points, start=1):
            if self.stop_requested:
                self._set_status("播放已被使用者中止")
                return "stopped"
            if deadline is not None and datetime.now() >= deadline:
                self._set_status("已到達結束時間，停止目前流程")
                return "deadline"
            if not event.get("enabled", True):
                self._add_execution_log("info", f"{self._event_description(event, index)} 已停用，略過")
                continue
            result = self._execute_event_with_policy(index, event, speed, click_delay, wait_time, deadline)
            if result not in {"completed", "skipped"}:
                return result
        return "completed"

    def _playback_worker(self, speed, click_delay, wait_time):
        try:
            self._add_execution_log("info", "手動播放已啟動，進入 3 秒安全倒數")
            countdown_result = self._run_start_countdown("手動播放")
            if countdown_result != "completed":
                self._add_execution_log("warning", "手動播放在倒數期間取消")
                return

            result = self._run_trajectory_loop(speed, click_delay, wait_time)
            if result == "completed":
                self._set_status("所有事件播放完成 ✅")
                self._add_execution_log("success", "手動播放完整流程成功")
            elif result == "failed":
                self._add_execution_log("error", "手動播放因事件失敗而停止")
            elif result == "stopped":
                self._add_execution_log("warning", "手動播放已由使用者停止")

        except pyautogui.FailSafeException:
            self._set_status("⚠ 觸發 FAILSAFE 安全機制！已立即停止所有滑鼠操作")
            self._add_execution_log("error", "手動播放觸發 FAILSAFE，已立即停止")
        except Exception as error:
            self._set_status(f"播放發生未預期錯誤：{error}")
            self._add_execution_log("error", f"手動播放未預期錯誤：{type(error).__name__}: {error}")
        finally:
            self.is_playing = False

    def stop_playback(self):
        if not self.is_playing:
            self._set_status("目前沒有正在播放、排程或連按中的動作")
            return
        self.stop_requested = True
        self._set_status("已送出停止指令，等待目前動作中止...")
        self._add_execution_log("warning", "使用者送出停止指令")

    def start_rapid_click(self):
        if self.is_playing:
            messagebox.showinfo("提示", "目前已有播放、排程或連按正在進行中，請先停止")
            return
        rate = self.scale_rapid_rate.get()
        try:
            click_count = int(self.spin_rapid_count.get())
            if click_count < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("錯誤", "總點擊次數必須是大於等於 1 的整數")
            return
        button_type = self.rapid_click_button_var.get()
        self.is_playing = True
        self.stop_requested = False
        button_label = "左鍵" if button_type == "left" else "右鍵"
        self._set_status(f"快速連按已啟動：{button_label}，每秒 {rate} 次，共 {click_count} 次")

        thread = threading.Thread(
            target=self._rapid_click_worker,
            args=(rate, click_count, button_type),
            daemon=True
        )
        thread.start()

    def _rapid_click_worker(self, rate, click_count, button_type):
        try:
            self._add_execution_log("info", "獨立快速連按已啟動，進入 3 秒安全倒數")
            countdown_result = self._run_start_countdown("快速連按")
            if countdown_result != "completed":
                self._add_execution_log("warning", "快速連按在倒數期間取消")
                return

            result = self._execute_rapid_click(rate, click_count, button_type)
            if result == "completed":
                self._set_status(f"快速連按完成，共點擊 {click_count} 次 ✅")
                self._add_execution_log("success", f"快速連按成功，共 {click_count} 次")
            elif result == "stopped":
                self._add_execution_log("warning", "快速連按已由使用者停止")

        except pyautogui.FailSafeException:
            self._set_status("⚠ 觸發 FAILSAFE 安全機制！快速連按已立即停止")
            self._add_execution_log("error", "快速連按觸發 FAILSAFE，已立即停止")
        except Exception as error:
            self._set_status(f"快速連按失敗：{error}")
            self._add_execution_log("error", f"快速連按失敗：{type(error).__name__}: {error}")
        finally:
            self.is_playing = False

    def _execute_rapid_click(self, rate, click_count, button_type, deadline=None):
        interval = 1.0 / rate if rate > 0 else 0
        for i in range(1, click_count + 1):
            if self.stop_requested:
                self._set_status("快速連按已被使用者中止")
                return "stopped"
            if deadline is not None and datetime.now() >= deadline:
                self._set_status("已到達結束時間，快速連按已停止")
                return "deadline"
            pyautogui.click(button=button_type)
            self._set_status(f"快速連按中：第 {i}/{click_count} 次")
            if i < click_count:
                wait_result = self._interruptible_sleep(interval, deadline)
                if wait_result != "completed":
                    return wait_result
        return "completed"

    @staticmethod
    def _parse_clock_time(time_text):
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(time_text, fmt).time()
            except ValueError:
                continue
        return None

    def _parse_schedule_time(self, time_text):
        target_time = self._parse_clock_time(time_text)
        if target_time is None:
            return None
        now = datetime.now()
        target_dt = datetime.combine(now.date(), target_time)
        if target_dt <= now:
            target_dt += timedelta(days=1)
        return target_dt

    def add_schedule_task(self):
        if self.is_playing:
            messagebox.showinfo("提示", "排程佇列執行中，無法新增任務，請先按停止")
            return
        time_text = self.entry_schedule_time.get().strip()
        target_dt = self._parse_schedule_time(time_text)
        if target_dt is None:
            messagebox.showerror("錯誤", "時間格式錯誤，請輸入 HH:MM 或 HH:MM:SS")
            return
        try:
            repeat_count = int(self.spin_repeat.get())
            if repeat_count < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("錯誤", "重複執行次數必須是大於等於 1 的整數")
            return

        interval = self.scale_interval.get()
        task = {"target_dt": target_dt, "repeat_count": repeat_count, "interval": interval}
        self.scheduled_tasks.append(task)
        self._refresh_schedule_listbox()
        self._set_status(
            f"已新增排程任務：{target_dt.strftime('%Y-%m-%d %H:%M:%S')}，"
            f"重複 {repeat_count} 次（間隔 {interval:.1f} 秒）"
        )
        self._autosave_workflow("新增排程任務")

    def _refresh_schedule_listbox(self):
        self.listbox_schedule.delete(0, tk.END)
        sorted_tasks = sorted(self.scheduled_tasks, key=lambda t: t["target_dt"])
        for i, task in enumerate(sorted_tasks, start=1):
            self.listbox_schedule.insert(
                tk.END,
                f"任務 {i}：{task['target_dt'].strftime('%Y-%m-%d %H:%M:%S')}，"
                f"完整流程 ×{task['repeat_count']}（間隔 {task['interval']:.1f} 秒）"
            )

    def delete_selected_task(self):
        if self.is_playing:
            messagebox.showinfo("提示", "排程佇列執行中，無法刪除任務，請先按停止")
            return
        selection = self.listbox_schedule.curselection()
        if not selection:
            messagebox.showinfo("提示", "請先在排程任務清單中選取要刪除的任務")
            return
        sorted_tasks = sorted(self.scheduled_tasks, key=lambda t: t["target_dt"])
        target_task = sorted_tasks[selection[0]]
        self.scheduled_tasks.remove(target_task)
        self._refresh_schedule_listbox()
        self._set_status("已刪除選取的排程任務")
        self._autosave_workflow("刪除排程任務")

    def clear_all_tasks(self):
        if self.is_playing:
            messagebox.showinfo("提示", "排程佇列執行中，無法清空任務，請先按停止")
            return
        if not self.scheduled_tasks:
            return
        self.scheduled_tasks.clear()
        self._refresh_schedule_listbox()
        self._set_status("已清空所有排程任務")
        self._autosave_workflow("清空排程任務")

    def start_schedule_queue(self):
        if self.is_playing:
            messagebox.showinfo("提示", "目前已有動作進行中，請先停止")
            return
        if not self.scheduled_tasks:
            messagebox.showwarning("提示", "排程佇列是空的，請先新增任務")
            return
        if not self.recorded_points:
            messagebox.showwarning("提示", "事件清單是空的，請先新增事件")
            return

        self.is_playing = True
        self.is_scheduling = True
        self.stop_requested = False
        self._set_status(f"排程佇列已啟動，共有 {len(self.scheduled_tasks)} 個任務等待執行")
        speed = self.scale_speed.get()
        click_delay = self.scale_click_delay.get()
        wait_time = self.scale_wait.get()
        thread = threading.Thread(
            target=self._multi_schedule_worker,
            args=(speed, click_delay, wait_time),
            daemon=True
        )
        thread.start()

    def _multi_schedule_worker(self, speed, click_delay, wait_time):
        try:
            while self.scheduled_tasks:
                if self.stop_requested:
                    self._set_status("排程佇列已被使用者取消")
                    return

                task = min(self.scheduled_tasks, key=lambda t: t["target_dt"])
                target_dt = task["target_dt"]
                repeat_count = task["repeat_count"]
                interval = task["interval"]

                while True:
                    if self.stop_requested:
                        self._set_status("排程佇列已被使用者取消")
                        return
                    now = datetime.now()
                    remaining = (target_dt - now).total_seconds()
                    if remaining <= 3:
                        break
                    total_seconds = int(remaining)
                    hours, rem = divmod(total_seconds, 3600)
                    minutes, seconds = divmod(rem, 60)
                    self._set_status(
                        f"排程等待中，下個任務將於 {target_dt.strftime('%H:%M:%S')} 執行 "
                        f"（倒數 {hours:02d}:{minutes:02d}:{seconds:02d}）"
                    )
                    wait_result = self._interruptible_sleep(min(1.0, max(0.0, remaining - 3.0)))
                    if wait_result == "stopped":
                        self._set_status("排程佇列已被使用者取消")
                        return

                task_stopped = False
                self._add_execution_log("info", f"排程任務開始：{target_dt.strftime('%Y-%m-%d %H:%M:%S')}，共 {repeat_count} 輪")
                for rep in range(1, repeat_count + 1):
                    if self.stop_requested:
                        self._set_status("排程佇列已被使用者中止")
                        task_stopped = True
                        break

                    countdown_result = self._run_start_countdown(f"排程第 {rep}/{repeat_count} 輪")
                    if countdown_result != "completed":
                        task_stopped = True
                        break

                    self._set_status(f"執行任務中：第 {rep}/{repeat_count} 次")
                    result = self._run_trajectory_loop(speed, click_delay, wait_time)

                    if result in {"stopped", "failed"}:
                        task_stopped = True
                        break

                    self._add_execution_log("success", f"排程任務第 {rep}/{repeat_count} 輪完成")

                    if rep < repeat_count and interval > 0:
                        wait_result = self._interruptible_sleep(interval)
                        if wait_result == "stopped":
                            self._set_status("排程佇列已被使用者中止")
                            task_stopped = True
                            break

                if task_stopped:
                    return

                if task in self.scheduled_tasks:
                    self.scheduled_tasks.remove(task)
                self.root.after(0, self._refresh_schedule_listbox)
                self.root.after(0, self._autosave_workflow, "排程任務完成")
                self._add_execution_log("success", f"排程任務已完成並移出佇列：{target_dt.strftime('%H:%M:%S')}")

            self._set_status("所有排程任務皆已完成 ✅")
            self._add_execution_log("success", "排程佇列全部完成")

        except pyautogui.FailSafeException:
            self._set_status("⚠ 觸發 FAILSAFE 安全機制！排程佇列已立即中止")
            self._add_execution_log("error", "排程佇列觸發 FAILSAFE，已立即中止")
        except Exception as error:
            self._set_status(f"排程佇列發生未預期錯誤：{error}")
            self._add_execution_log("error", f"排程佇列錯誤：{type(error).__name__}: {error}")
        finally:
            self.is_playing = False
            self.is_scheduling = False

    @staticmethod
    def _calculate_window_bounds(start_time, end_time, now=None):
        now = now or datetime.now()
        candidate_windows = []
        for day_offset in (-1, 0, 1):
            start_date = now.date() + timedelta(days=day_offset)
            start_dt = datetime.combine(start_date, start_time)
            end_dt = datetime.combine(start_date, end_time)
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)
            candidate_windows.append((start_dt, end_dt))

        active_windows = [window for window in candidate_windows if window[0] <= now < window[1]]
        if active_windows:
            return max(active_windows, key=lambda window: window[0])

        upcoming_windows = [window for window in candidate_windows if window[0] > now]
        if upcoming_windows:
            return min(upcoming_windows, key=lambda window: window[0])

        start_dt = datetime.combine(now.date() + timedelta(days=1), start_time)
        end_dt = datetime.combine(now.date() + timedelta(days=1), end_time)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)
        return start_dt, end_dt

    def start_window_schedule(self):
        if self.is_playing:
            messagebox.showinfo("提示", "目前已有動作進行中，請先停止")
            return
        if not self.recorded_points:
            messagebox.showwarning("提示", "事件清單是空的，請先新增事件")
            return

        start_text = f"{self.entry_start_h.get().strip()}:{self.entry_start_m.get().strip()}"
        end_text = f"{self.entry_end_h.get().strip()}:{self.entry_end_m.get().strip()}"
        start_time = self._parse_clock_time(start_text)
        end_time = self._parse_clock_time(end_text)
        
        if start_time is None or end_time is None:
            messagebox.showerror("錯誤", "時與分輸入錯誤，請確認皆為合法數字（例如 09:00）")
            return

        try:
            interval_minutes = int(self.spin_window_interval.get())
            if interval_minutes <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("錯誤", "執行間隔必須是大於 0 的整數分鐘，例如 1 或 10")
            return

        now = datetime.now()
        start_dt, end_dt = self._calculate_window_bounds(start_time, end_time, now)
        interval_seconds = interval_minutes * 60.0

        if start_dt <= now < end_dt and self.window_run_now_var.get():
            first_run_dt = now
        elif now <= start_dt:
            first_run_dt = start_dt
        else:
            elapsed = (now - start_dt).total_seconds()
            next_step = int(elapsed // interval_seconds) + 1
            first_run_dt = start_dt + timedelta(seconds=next_step * interval_seconds)

        self.cycle_states["window"].update(
            active=True, start=start_dt, end=end_dt, next_run=first_run_dt,
            run_count=0, last_state="等待中",
        )

        self.is_playing = True
        self.stop_requested = False

        speed = self.scale_speed.get()
        click_delay = self.scale_click_delay.get()
        wait_time = self.scale_wait.get()

        self._set_status(
            f"時段循環排程已啟動：{start_dt.strftime('%Y-%m-%d %H:%M:%S')} ～ "
            f"{end_dt.strftime('%Y-%m-%d %H:%M:%S')}，每 {interval_minutes:g} 分鐘執行一次"
        )
        thread = threading.Thread(
            target=self._cycle_schedule_worker,
            args=("window", first_run_dt, end_dt, interval_seconds, speed, click_delay, wait_time),
            daemon=True
        )
        thread.start()

    @staticmethod
    def _read_hms_entries(hour_entry, minute_entry, second_entry):
        """讀取時/分/秒欄位並換算成總秒數；留空視為 0，格式不合法回傳 None。"""
        values = []
        for entry, upper_limit in ((hour_entry, None), (minute_entry, 59), (second_entry, 59)):
            text = entry.get().strip()
            if text == "":
                values.append(0)
                continue
            if not text.isdigit():
                return None
            number = int(text)
            if upper_limit is not None and number > upper_limit:
                return None
            values.append(number)
        hours, minutes, seconds = values
        return hours * 3600 + minutes * 60 + seconds

    @staticmethod
    def _write_hms_entries(hour_entry, minute_entry, second_entry, total_seconds):
        total_seconds = max(0, int(total_seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        for entry, value in ((hour_entry, hours), (minute_entry, minutes), (second_entry, seconds)):
            entry.delete(0, "end")
            entry.insert(0, f"{value:02d}")

    def start_timer_schedule(self):
        if self.is_playing:
            messagebox.showinfo("提示", "目前已有動作進行中，請先停止")
            return
        if not self.recorded_points:
            messagebox.showwarning("提示", "事件清單是空的，請先新增事件")
            return

        total_seconds = self._read_hms_entries(
            self.entry_timer_total_h, self.entry_timer_total_m, self.entry_timer_total_s
        )
        interval_seconds = self._read_hms_entries(
            self.entry_timer_interval_h, self.entry_timer_interval_m, self.entry_timer_interval_s
        )
        if total_seconds is None or interval_seconds is None:
            messagebox.showerror("錯誤", "時、分、秒必須是數字，且分與秒需在 0～59 之間")
            return
        if interval_seconds < 1:
            messagebox.showerror("錯誤", "每次間隔至少需要 1 秒")
            return
        if total_seconds <= 3:
            messagebox.showerror("錯誤", "執行總時長需大於 3 秒（每輪開始前有 3 秒安全倒數）")
            return

        now = datetime.now()
        end_dt = now + timedelta(seconds=total_seconds)
        total_text = self._format_hms(total_seconds)
        interval_text = self._format_hms(interval_seconds)

        self.cycle_states["timer"].update(
            active=True, start=now, end=end_dt, next_run=now,
            run_count=0, last_state="等待中",
        )

        self.is_playing = True
        self.stop_requested = False

        speed = self.scale_speed.get()
        click_delay = self.scale_click_delay.get()
        wait_time = self.scale_wait.get()

        self._set_status(
            f"計時排程已啟動：持續執行 {total_text}（至 {end_dt.strftime('%H:%M:%S')}），"
            f"每隔 {interval_text} 一次"
        )
        self._add_execution_log("info", f"計時排程設定：總時長 {total_text}，間隔 {interval_text}")
        thread = threading.Thread(
            target=self._cycle_schedule_worker,
            args=("timer", now, end_dt, float(interval_seconds), speed, click_delay, wait_time),
            daemon=True
        )
        thread.start()

    def _cycle_schedule_worker(self, mode, first_run_dt, end_dt, interval_seconds, speed, click_delay, wait_time):
        """排程 B（mode="window"）與排程 C（mode="timer"）共用的循環執行核心。"""
        name = self.CYCLE_LABELS[mode]
        state = self.cycle_states[mode]
        run_count = 0
        next_run = first_run_dt
        final_state = "已完成"

        try:
            self._add_execution_log("info", f"{name}開始，結束時間 {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            while True:
                if self.stop_requested:
                    final_state = f"已由使用者停止（已執行 {run_count} 次）"
                    self._set_status(f"{name}已被使用者取消")
                    return

                now = datetime.now()
                if now >= end_dt or next_run >= end_dt:
                    final_state = f"已到達結束時間（共執行 {run_count} 次）"
                    self._set_status(f"{name}完成，共執行 {run_count} 次 ✅")
                    return

                state["next_run"] = next_run
                countdown_start = next_run - timedelta(seconds=3)
                if now < countdown_start:
                    wait_result = self._interruptible_sleep((countdown_start - now).total_seconds(), deadline=end_dt)
                    if wait_result == "stopped":
                        final_state = f"已由使用者停止（已執行 {run_count} 次）"
                        self._set_status(f"{name}已被使用者取消")
                        return
                    if wait_result == "deadline":
                        final_state = f"已到達結束時間（共執行 {run_count} 次）"
                        self._set_status(f"{name}完成，共執行 {run_count} 次 ✅")
                        return

                if datetime.now() >= end_dt:
                    final_state = f"已到達結束時間（共執行 {run_count} 次）"
                    self._set_status(f"{name}完成，共執行 {run_count} 次 ✅")
                    return

                state["next_run"] = None
                countdown_result = self._run_start_countdown(f"{name}第 {run_count + 1} 輪", deadline=end_dt)
                if countdown_result == "stopped":
                    final_state = f"已由使用者停止（已執行 {run_count} 次）"
                    return
                if countdown_result == "deadline":
                    final_state = f"倒數期間到達結束時間（共執行 {run_count} 次）"
                    return

                self._set_status(f"{name}：正在執行第 {run_count + 1} 次完整事件清單")
                result = self._run_trajectory_loop(speed, click_delay, wait_time, deadline=end_dt)

                if result == "stopped":
                    final_state = f"已由使用者停止（已執行 {run_count} 次）"
                    return
                if result == "deadline":
                    final_state = f"結束時間到，已中止當次流程（先前完成 {run_count} 次）"
                    self._set_status(f"已到達結束時間，{name}已停止")
                    return
                if result == "failed":
                    final_state = f"事件失敗，已停止（先前完成 {run_count} 次）"
                    self._set_status(f"{name}因事件失敗而停止")
                    return

                run_count += 1
                state["run_count"] = run_count
                self._add_execution_log("success", f"{name}第 {run_count} 輪完成")

                next_run += timedelta(seconds=interval_seconds)
                now = datetime.now()
                while next_run <= now:
                    next_run += timedelta(seconds=interval_seconds)

        except pyautogui.FailSafeException:
            final_state = f"FAILSAFE 已中止（完成 {run_count} 次）"
            self._set_status(f"⚠ 觸發 FAILSAFE 安全機制！{name}已立即中止")
            self._add_execution_log("error", f"{name}觸發 FAILSAFE，已立即中止")
        except Exception as error:
            final_state = f"未預期錯誤（完成 {run_count} 次）"
            self._set_status(f"{name}發生錯誤：{error}")
            self._add_execution_log("error", f"{name}錯誤：{type(error).__name__}: {error}")
        finally:
            state["last_state"] = final_state
            state["next_run"] = None
            state["active"] = False
            self.is_playing = False
            level = "success" if "到達結束時間" in final_state or final_state == "已完成" else "warning"
            self._add_execution_log("info" if level == "success" else level, f"{name}結束：{final_state}")

    def _start_hotkey_listener(self):
        self.hotkey_listener = pynput_keyboard.Listener(on_press=self._on_key_press)
        self.hotkey_listener.start()

    def _on_key_press(self, key):
        try:
            if key == pynput_keyboard.Key.f8:
                self.root.after(0, self.record_point)
            elif key == pynput_keyboard.Key.f9:
                self.root.after(0, self.start_playback)
            elif key == pynput_keyboard.Key.f10:
                self.root.after(0, self.stop_playback)
            elif key == pynput_keyboard.Key.f11:
                self.root.after(0, self.start_rapid_click)
        except Exception:
            pass

    def _on_close(self):
        try:
            self._autosave_workflow("關閉程式")
        except Exception:
            pass
        self.stop_requested = True
        if hasattr(self, "hotkey_listener"):
            self.hotkey_listener.stop()
        self.root.destroy()

    @staticmethod
    def _resolve_app_data_dir():
        candidates = []
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(Path(local_app_data) / "MouseFlowStudio")
        candidates.append(Path(__file__).resolve().parent / "mouse_flow_data")

        for candidate in candidates:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            except OSError:
                continue

        fallback = Path.cwd() / "mouse_flow_data"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    @staticmethod
    def _normalize_nonnegative_number(value, default):
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_positive_integer(value, default):
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    def _normalize_loaded_event(self, raw_event):
        if not isinstance(raw_event, dict):
            return None
        event_type = raw_event.get("type")
        if event_type not in {"point", "rapid_click", "key", "text"}:
            return None
        failure_policy = raw_event.get("failure_policy", "stop")
        if failure_policy not in {"stop", "skip", "retry"}:
            failure_policy = "stop"

        common = {
            "enabled": bool(raw_event.get("enabled", True)),
            "action_delay": self._normalize_nonnegative_number(raw_event.get("action_delay"), 0.5),
            "repeat_count": self._normalize_positive_integer(raw_event.get("repeat_count"), 1),
            "repeat_interval": self._normalize_nonnegative_number(raw_event.get("repeat_interval"), 1.0),
            "after_wait": self._normalize_nonnegative_number(raw_event.get("after_wait"), 0.5),
            "failure_policy": failure_policy,
            "retry_count": self._normalize_positive_integer(raw_event.get("retry_count"), 2),
        }

        if event_type == "point":
            try:
                x, y = int(raw_event["x"]), int(raw_event["y"])
            except (KeyError, TypeError, ValueError):
                return None
            click_type = raw_event.get("click_type", "left")
            if click_type not in {"left", "right", "none"}:
                click_type = "left"
            key_value = raw_event.get("key")
            return {
                "type": "point", "x": x, "y": y,
                "click_type": click_type,
                "key": str(key_value) if key_value else None,
                "move_duration": self._normalize_nonnegative_number(raw_event.get("move_duration"), 0.5),
                **common,
            }

        if event_type == "rapid_click":
            rate = self._normalize_nonnegative_number(raw_event.get("rate"), 10)
            if rate <= 0:
                rate = 10
            button_type = raw_event.get("button_type", "left")
            if button_type not in {"left", "right"}:
                button_type = "left"
            return {
                "type": "rapid_click", "rate": rate,
                "count": self._normalize_positive_integer(raw_event.get("count"), 20),
                "button_type": button_type,
                "enabled": common["enabled"],
                "after_wait": common["after_wait"],
                "failure_policy": common["failure_policy"],
                "retry_count": common["retry_count"],
            }

        if event_type == "key":
            key_value = str(raw_event.get("key", "")).strip()
            if not key_value:
                return None
            return {"type": "key", "key": key_value, "move_duration": 0.0, **common}

        text_value = str(raw_event.get("text", ""))
        if not text_value:
            return None
        return {"type": "text", "text": text_value, "move_duration": 0.0, **common}

    @staticmethod
    def _validate_workflow_data(data):
        """嚴格檢查流程 JSON 結構，不符合就丟出 WorkflowFormatError，不做任何修正。"""
        problems = []

        def is_number(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) \
                and value == value and value not in (float("inf"), float("-inf"))

        def is_integer(value):
            return (isinstance(value, int) and not isinstance(value, bool)) or \
                (isinstance(value, float) and is_number(value) and value.is_integer())

        def check_number(where, item, key, minimum=0.0, required=False, integer=False):
            if key not in item:
                if required:
                    problems.append(f"{where} 缺少必要欄位「{key}」")
                return
            value = item[key]
            ok = is_integer(value) if integer else is_number(value)
            if not ok:
                problems.append(f"{where} 的「{key}」必須是{'整數' if integer else '數字'}（目前：{value!r}）")
            elif value < minimum:
                problems.append(f"{where} 的「{key}」不可小於 {minimum:g}（目前：{value!r}）")

        def check_choice(where, item, key, choices):
            if key in item and item[key] not in choices:
                problems.append(f"{where} 的「{key}」必須是 {'/'.join(sorted(choices))} 其中之一（目前：{item[key]!r}）")

        if not isinstance(data, dict):
            raise WorkflowFormatError(["JSON 根節點必須是物件（{ }）"])

        if data.get("format") != WORKFLOW_FORMAT:
            problems.append(f"「format」必須是 \"{WORKFLOW_FORMAT}\"（目前：{data.get('format')!r}），這不是本程式儲存的流程檔")
        version = data.get("version")
        if not is_integer(version) or not 1 <= version <= WORKFLOW_VERSION:
            problems.append(f"「version」必須是 1～{WORKFLOW_VERSION} 的整數（目前：{version!r}）")

        events = data.get("events")
        if not isinstance(events, list):
            problems.append("「events」必須是陣列（清單）")
            events = []

        for number, event in enumerate(events, start=1):
            where = f"events[{number}]"
            if not isinstance(event, dict):
                problems.append(f"{where} 必須是物件")
                continue
            event_type = event.get("type")
            if event_type not in {"point", "rapid_click", "key", "text"}:
                problems.append(f"{where} 的「type」必須是 point/rapid_click/key/text 其中之一（目前：{event_type!r}）")
                continue
            where = f"{where}（{event_type}）"

            if "enabled" in event and not isinstance(event["enabled"], bool):
                problems.append(f"{where} 的「enabled」必須是 true 或 false")
            check_choice(where, event, "failure_policy", {"stop", "skip", "retry"})
            for key in ("action_delay", "repeat_interval", "after_wait", "move_duration"):
                check_number(where, event, key)
            for key in ("repeat_count", "retry_count"):
                check_number(where, event, key, minimum=1, integer=True)

            if event_type == "point":
                check_number(where, event, "x", minimum=-100000, required=True, integer=True)
                check_number(where, event, "y", minimum=-100000, required=True, integer=True)
                check_choice(where, event, "click_type", {"left", "right", "none"})
                if event.get("key") is not None and not isinstance(event["key"], str):
                    problems.append(f"{where} 的「key」必須是文字或 null")
            elif event_type == "rapid_click":
                check_number(where, event, "rate", minimum=0.001, required=True)
                check_number(where, event, "count", minimum=1, required=True, integer=True)
                check_choice(where, event, "button_type", {"left", "right"})
            elif event_type == "key":
                if not isinstance(event.get("key"), str) or not event["key"].strip():
                    problems.append(f"{where} 的「key」必須是非空白文字")
            else:
                if not isinstance(event.get("text"), str) or not event["text"]:
                    problems.append(f"{where} 的「text」必須是非空文字")

        tasks = data.get("scheduled_tasks", [])
        if not isinstance(tasks, list):
            problems.append("「scheduled_tasks」必須是陣列（清單）")
            tasks = []
        for number, task in enumerate(tasks, start=1):
            where = f"scheduled_tasks[{number}]"
            if not isinstance(task, dict):
                problems.append(f"{where} 必須是物件")
                continue
            target = task.get("target_dt")
            try:
                if not isinstance(target, str):
                    raise ValueError
                datetime.fromisoformat(target)
            except ValueError:
                problems.append(f"{where} 的「target_dt」必須是 ISO 日期時間文字（目前：{target!r}）")
            check_number(where, task, "repeat_count", minimum=1, required=True, integer=True)
            check_number(where, task, "interval", required=True)

        if "settings" in data and not isinstance(data["settings"], dict):
            problems.append("「settings」必須是物件")

        if problems:
            raise WorkflowFormatError(problems)

    def _read_workflow_file(self, path):
        """讀取並驗證流程檔，回傳可安全套用的資料；失敗時丟出 OSError/ValueError 系列例外。"""
        with Path(path).open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
        self._validate_workflow_data(data)
        return data

    def _create_workflow_payload(self):
        scheduled_tasks = [
            {
                "target_dt": task["target_dt"].isoformat(),
                "repeat_count": task["repeat_count"],
                "interval": task["interval"],
            }
            for task in self.scheduled_tasks
        ]
        settings = {}
        try:
            settings = {
                "window_start_h": self.entry_start_h.get().strip(),
                "window_start_m": self.entry_start_m.get().strip(),
                "window_end_h": self.entry_end_h.get().strip(),
                "window_end_m": self.entry_end_m.get().strip(),
                "window_interval_minutes": self.spin_window_interval.get(),
                "window_run_now": bool(self.window_run_now_var.get()),
                "timer_total_h": self.entry_timer_total_h.get().strip(),
                "timer_total_m": self.entry_timer_total_m.get().strip(),
                "timer_total_s": self.entry_timer_total_s.get().strip(),
                "timer_interval_h": self.entry_timer_interval_h.get().strip(),
                "timer_interval_m": self.entry_timer_interval_m.get().strip(),
                "timer_interval_s": self.entry_timer_interval_s.get().strip(),
                "rapid_rate": self.scale_rapid_rate.get(),
                "rapid_count": self.spin_rapid_count.get(),
                "rapid_button": self.rapid_click_button_var.get(),
            }
        except (AttributeError, tk.TclError):
            pass

        return {
            "format": WORKFLOW_FORMAT,
            "version": WORKFLOW_VERSION,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "events": copy.deepcopy(self.recorded_points),
            "scheduled_tasks": scheduled_tasks,
            "settings": settings,
        }

    @staticmethod
    def _write_json_atomic(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)

    def _apply_workflow_data(self, data):
        if not isinstance(data, dict):
            raise ValueError("JSON 根節點必須是物件")

        normalized_events = []
        for raw_event in data.get("events", []):
            event = self._normalize_loaded_event(raw_event)
            if event is not None:
                normalized_events.append(event)

        scheduled_tasks = []
        for raw_task in data.get("scheduled_tasks", []):
            if not isinstance(raw_task, dict):
                continue
            try:
                target_dt = datetime.fromisoformat(raw_task["target_dt"])
                repeat_count = max(1, int(raw_task.get("repeat_count", 1)))
                interval = max(0.0, float(raw_task.get("interval", 0)))
            except (KeyError, TypeError, ValueError):
                continue
            scheduled_tasks.append({
                "target_dt": target_dt,
                "repeat_count": repeat_count,
                "interval": interval,
            })

        self.recorded_points = normalized_events
        self.scheduled_tasks = scheduled_tasks
        self._refresh_listbox_labels()
        self._refresh_schedule_listbox()

        settings = data.get("settings", {})
        if isinstance(settings, dict):
            if settings.get("window_start_h"):
                self.entry_start_h.delete(0, "end")
                self.entry_start_h.insert(0, settings["window_start_h"])
                self.entry_start_m.delete(0, "end")
                self.entry_start_m.insert(0, settings["window_start_m"])
                self.entry_end_h.delete(0, "end")
                self.entry_end_h.insert(0, settings["window_end_h"])
                self.entry_end_m.delete(0, "end")
                self.entry_end_m.insert(0, settings["window_end_m"])
            elif settings.get("window_start"): 
                try:
                    h, m = settings["window_start"].split(":")[:2]
                    self.entry_start_h.delete(0, "end")
                    self.entry_start_h.insert(0, h)
                    self.entry_start_m.delete(0, "end")
                    self.entry_start_m.insert(0, m)
                    eh, em = settings["window_end"].split(":")[:2]
                    self.entry_end_h.delete(0, "end")
                    self.entry_end_h.insert(0, eh)
                    self.entry_end_m.delete(0, "end")
                    self.entry_end_m.insert(0, em)
                except Exception:
                    pass

            self._apply_timer_settings(settings)

            if settings.get("window_interval_minutes") is not None:
                self._replace_spinbox_value(self.spin_window_interval, settings["window_interval_minutes"])
            if settings.get("window_run_now") is not None:
                self.window_run_now_var.set(bool(settings["window_run_now"]))
            if settings.get("rapid_rate") is not None:
                self.scale_rapid_rate.set(settings["rapid_rate"])
            if settings.get("rapid_count") is not None:
                self._replace_spinbox_value(self.spin_rapid_count, settings["rapid_count"])
            if settings.get("rapid_button") in {"left", "right"}:
                self.rapid_click_button_var.set(settings["rapid_button"])

        return len(normalized_events), len(scheduled_tasks)

    def _apply_timer_settings(self, settings):
        """還原排程 C 的時/分/秒；同時相容舊版以「分鐘」儲存的 timer_total / timer_interval。"""
        groups = (
            ("timer_total", (self.entry_timer_total_h, self.entry_timer_total_m, self.entry_timer_total_s)),
            ("timer_interval", (self.entry_timer_interval_h, self.entry_timer_interval_m, self.entry_timer_interval_s)),
        )
        for prefix, entries in groups:
            keys = [f"{prefix}_h", f"{prefix}_m", f"{prefix}_s"]
            if any(key in settings for key in keys):
                for entry, key in zip(entries, keys):
                    text = str(settings.get(key, "")).strip()
                    if text.isdigit() and len(text) <= 2:
                        entry.delete(0, "end")
                        entry.insert(0, text)
            elif settings.get(prefix) is not None:
                try:
                    legacy_minutes = int(float(settings[prefix]))
                except (TypeError, ValueError):
                    continue
                self._write_hms_entries(*entries, legacy_minutes * 60)

    def _load_workflow_path(self, path, show_message=True, autosave=True):
        path = Path(path)
        data = self._read_workflow_file(path)

        self.autosave_suspended = True
        try:
            event_count, task_count = self._apply_workflow_data(data)
        finally:
            self.autosave_suspended = False

        self._set_status(f"已載入流程：{event_count} 個事件、{task_count} 個排程任務")
        self._add_execution_log("info", f"載入流程檔案：{path.name}（{event_count} 個事件）")
        if autosave:
            self._autosave_workflow("載入流程")
        if show_message:
            messagebox.showinfo("載入完成", f"已載入 {event_count} 個事件與 {task_count} 個排程任務。")

    def save_workflow(self):
        path = filedialog.asksaveasfilename(
            title="儲存滑鼠事件流程",
            defaultextension=".json",
            filetypes=[("Mouse Flow JSON", "*.json"), ("所有檔案", "*.*")],
            initialfile=f"mouse_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        if not path:
            return
        try:
            self._write_json_atomic(Path(path), self._create_workflow_payload())
            self._set_status(f"流程已儲存：{Path(path).name}")
            self._add_execution_log("success", f"手動儲存流程：{Path(path).name}")
        except (OSError, TypeError, ValueError) as error:
            messagebox.showerror("儲存失敗", str(error))
            self._add_execution_log("error", f"流程儲存失敗：{error}")

    def load_workflow(self):
        if self.is_playing:
            messagebox.showinfo("提示", "流程執行中無法載入檔案，請先停止")
            return
        path = filedialog.askopenfilename(
            title="載入滑鼠事件流程",
            filetypes=[("Mouse Flow JSON", "*.json"), ("所有檔案", "*.*")],
        )
        if not path:
            return
        try:
            self._load_workflow_path(path, show_message=True)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            messagebox.showerror("載入失敗", f"無法讀取這份流程檔案：\n{error}")
            self._add_execution_log("error", f"流程載入失敗：{error}")

    def _autosave_workflow(self, reason="更新流程"):
        if self.autosave_suspended:
            return
        try:
            self._write_json_atomic(self.autosave_path, self._create_workflow_payload())
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.label_autosave_status.config(text=f"自動備份：{timestamp} 已完成（{reason}）")
        except (OSError, TypeError, ValueError, tk.TclError) as error:
            try:
                self.label_autosave_status.config(text=f"自動備份失敗：{error}")
            except tk.TclError:
                pass

    def _load_default_workflow_setting(self):
        fallback = Path(__file__).resolve().parent / "mouse_flow_default.json"
        try:
            with self.settings_path.open("r", encoding="utf-8") as file:
                saved = json.load(file).get("default_workflow_path")
            if isinstance(saved, str) and saved.strip():
                return Path(saved)
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        return fallback

    def _save_default_workflow_setting(self, path):
        try:
            settings = {}
            try:
                with self.settings_path.open("r", encoding="utf-8") as file:
                    loaded = json.load(file)
                if isinstance(loaded, dict):
                    settings = loaded
            except (OSError, json.JSONDecodeError):
                pass
            settings["default_workflow_path"] = str(path)
            self._write_json_atomic(self.settings_path, settings)
            return True
        except (OSError, TypeError, ValueError) as error:
            self._add_execution_log("error", f"預設流程路徑儲存失敗：{error}")
            return False

    def choose_default_workflow(self):
        current = self.default_workflow_path
        path = filedialog.askopenfilename(
            title="選擇啟動時自動採用的預設流程 JSON",
            initialdir=str(current.parent) if current.parent.is_dir() else None,
            initialfile=current.name,
            filetypes=[("Mouse Flow JSON", "*.json"), ("所有檔案", "*.*")],
        )
        if not path:
            return

        try:
            self._read_workflow_file(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("無法採用為預設流程", f"{Path(path).name} 不符合流程格式，未變更預設路徑。\n\n{error}")
            self._add_execution_log("error", f"拒絕採用預設流程（{Path(path).name}）：{getattr(error, 'summary', error)}")
            return

        self.default_workflow_path = Path(path)
        saved = self._save_default_workflow_setting(self.default_workflow_path)
        self._add_execution_log("info", f"預設流程路徑已設定：{path}")

        if self.is_playing:
            self.label_default_flow_status.config(
                text=f"預設流程已設為：{path}（流程執行中，下次啟動才會載入）", fg=COLORS["muted"]
            )
        else:
            self._load_default_workflow_on_startup(autosave=True)
        if not saved:
            messagebox.showwarning("提示", "預設流程路徑無法寫入設定檔，下次啟動可能不會沿用。")

    def _load_default_workflow_on_startup(self, autosave=False):
        path = self.default_workflow_path
        if not path.is_file():
            message = f"找不到預設流程檔（{path}），請自行匯入 JSON 或規劃新流程"
            self.label_default_flow_status.config(text=f"⚠ {message}", fg=COLORS["warning"])
            self._add_execution_log("warning", message)
            return
        try:
            # 啟動時不覆寫自動備份，讓「還原自動備份」仍能取回上次的工作內容
            self._load_workflow_path(path, show_message=False, autosave=autosave)
            self.label_default_flow_status.config(
                text=f"已載入預設流程：{path}", fg=COLORS["muted"]
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            reason = getattr(error, "summary", error)
            message = f"預設流程檔未載入（{path.name}）：{reason}。請自行匯入 JSON 或規劃新流程"
            self.label_default_flow_status.config(text=f"⚠ {message}", fg=COLORS["warning"])
            self._add_execution_log("error", message)

    def restore_autosave(self):
        if self.is_playing:
            messagebox.showinfo("提示", "流程執行中無法還原備份，請先停止")
            return
        if not self.autosave_path.exists():
            messagebox.showinfo("自動備份", "目前尚未建立自動備份。")
            return
        try:
            self._load_workflow_path(self.autosave_path, show_message=True)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            messagebox.showerror("還原失敗", str(error))

    def _load_recent_execution_logs(self):
        loaded = []
        try:
            with self.execution_log_path.open("r", encoding="utf-8") as file:
                lines = file.readlines()[-500:]
            for line in lines:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict) and entry.get("timestamp") and entry.get("message"):
                    loaded.append(entry)
        except OSError:
            pass

        with self.log_lock:
            self.execution_logs = loaded[-500:]
        self._refresh_execution_log_listbox()

    def _refresh_execution_log_listbox(self):
        try:
            self.listbox_logs.delete(0, tk.END)
            level_labels = {
                "success": "成功",
                "error": "錯誤",
                "warning": "注意",
                "info": "資訊",
            }
            level_colors = {
                "success": "#047857",
                "error": "#BE123C",
                "warning": "#B45309",
                "info": "#475569",
            }
            with self.log_lock:
                visible_logs = list(reversed(self.execution_logs[-200:]))
            for entry in visible_logs:
                timestamp = str(entry.get("timestamp", ""))
                clock = timestamp[11:19] if len(timestamp) >= 19 else timestamp
                level = entry.get("level", "info")
                self.listbox_logs.insert(
                    tk.END,
                    f"{clock}｜{level_labels.get(level, level)}｜{entry.get('message', '')}",
                )
                self.listbox_logs.itemconfig(tk.END, fg=level_colors.get(level, level_colors["info"]))
        except (AttributeError, tk.TclError):
            pass

    def _add_execution_log(self, level, message):
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "level": level if level in {"success", "error", "warning", "info"} else "info",
            "message": str(message),
        }
        with self.log_lock:
            self.execution_logs.append(entry)
            if len(self.execution_logs) > 500:
                del self.execution_logs[:-500]
            try:
                self.execution_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.execution_log_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError:
                pass

        if threading.current_thread() is threading.main_thread():
            self._refresh_execution_log_listbox()
        else:
            try:
                self.root.after(0, self._refresh_execution_log_listbox)
            except (AttributeError, RuntimeError, tk.TclError):
                pass

    def export_execution_logs(self):
        path = filedialog.asksaveasfilename(
            title="匯出執行紀錄",
            defaultextension=".json",
            filetypes=[("JSON 紀錄", "*.json"), ("所有檔案", "*.*")],
            initialfile=f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        if not path:
            return
        try:
            with self.log_lock:
                logs = copy.deepcopy(self.execution_logs)
            self._write_json_atomic(
                Path(path),
                {
                    "format": "MouseFlowStudioExecutionLog",
                    "exported_at": datetime.now().isoformat(timespec="seconds"),
                    "entries": logs,
                },
            )
            self._set_status(f"執行紀錄已匯出：{Path(path).name}")
            self._add_execution_log("success", f"執行紀錄已匯出：{Path(path).name}")
        except (OSError, TypeError, ValueError) as error:
            messagebox.showerror("匯出失敗", str(error))

    def clear_execution_logs(self):
        if not messagebox.askyesno("清除執行紀錄", "確定要清除全部執行結果與錯誤紀錄嗎？"):
            return
        with self.log_lock:
            self.execution_logs.clear()
            try:
                self.execution_log_path.unlink(missing_ok=True)
            except OSError as error:
                messagebox.showerror("清除失敗", str(error))
                return
        self._refresh_execution_log_listbox()
        self._set_status("執行紀錄已清除")

    def _set_status(self, text):
        def update_label():
            try:
                self.label_status.config(text=f"狀態：{text}")
                lowered = text.lower()
                if any(word in lowered for word in ("failsafe", "錯誤", "失敗", "中止", "停止")):
                    dot_color = "#FB7185"
                elif any(word in lowered for word in ("播放中", "等待", "執行", "啟動", "連按中")):
                    dot_color = "#FBBF24"
                else:
                    dot_color = "#34D399"
                self.label_status_dot.config(fg=dot_color)
            except tk.TclError:
                pass

        if threading.current_thread() is threading.main_thread():
            update_label()
        else:
            try:
                self.root.after(0, update_label)
            except (RuntimeError, tk.TclError):
                pass


if __name__ == "__main__":
    root = tk.Tk()
    app = MouseCoordinateLab(root)
    root.mainloop()