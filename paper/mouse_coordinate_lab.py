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
    14. 每筆座標事件各自保存移動時間、動作前等待、點擊/按鍵重複次數、
        重複間隔與完成後等待；滑鼠只移動一次，只有點擊或按鍵會重複。
    15. 支援純快捷鍵、整段文字輸入、事件啟用/停用/排序/複製，以及逐筆失敗策略。
    16. 執行前 3 秒安全倒數；流程可用 JSON 儲存、載入並自動備份，執行結果
        與錯誤會持久保存，也可匯出為 JSON。

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

import tkinter as tk                # Tkinter：Python 內建的 GUI 函式庫，用來畫視窗、按鈕、標籤
from tkinter import ttk, messagebox, filedialog # filedialog 用於流程 JSON 的儲存與載入
import pyautogui                    # pyautogui：跨平台的滑鼠/鍵盤自動化函式庫
import threading                    # threading：用來讓「播放座標」在背景執行緒執行，避免卡住 GUI
import time                         # time：用來做「停留幾秒」的延遲
import copy                         # 複製事件時建立完全獨立的資料副本
import json                         # 流程儲存、載入與執行紀錄使用 JSON 格式
import os                           # 取得 Windows LOCALAPPDATA 資料夾
from pathlib import Path            # 跨平台處理流程檔案與備份路徑
from datetime import datetime, timedelta  # 用來計算「排程目標時間」與倒數剩餘時間

try:
    import pyperclip                # 透過剪貼簿可靠輸入中文等 Unicode 文字
except ImportError:
    pyperclip = None

# pynput：用來監聽「全域鍵盤事件」的函式庫。
# 與 Tkinter 內建的 bind() 不同，pynput 在 Windows 上是透過 ctypes
# 呼叫 SetWindowsHookEx(WH_KEYBOARD_LL, ...) 安裝「系統層級」的鍵盤鉤子，
# 所以即使焦點不在本程式視窗上，也能攔截到按鍵事件。
from pynput import keyboard as pynput_keyboard


# ------------------------------------------------------------------
# 【安全機制設定】PyAutoGUI 的 FAILSAFE
# ------------------------------------------------------------------
# FAILSAFE = True 是 pyautogui 內建的緊急停止機制。
# 只要偵測到滑鼠目前座標在螢幕「左上角 (0,0)」，
# pyautogui 下一次呼叫移動/點擊函式時就會拋出 pyautogui.FailSafeException，
# 讓程式立刻中止動作，避免自動化程式失控。
#
# 使用方式：操作過程中若想緊急停止，只要把滑鼠用力甩到螢幕左上角即可。
pyautogui.FAILSAFE = True

# PAUSE 是「每一次 pyautogui 動作之間」自動插入的延遲秒數（秒）。
# 設定一個小延遲可以讓系統有時間處理每個輸入事件，避免動作太快系統來不及反應。
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


class RoundedButton(tk.Canvas):
    """純 Tkinter 圓角按鈕，提供懸停、按下與鍵盤操作回饋。"""

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
        """將 #RRGGBB 顏色依 factor 調亮或調暗。"""
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
    """
    整個實驗程式的主要類別。
    把「介面」與「邏輯」都包在這個類別裡，方便管理狀態（座標清單、是否正在播放等）。
    """

    def __init__(self, root):
        # root 是 Tkinter 的主視窗物件 (tk.Tk())
        self.root = root
        self.root.title("Mouse Flow Studio｜滑鼠自動化工作台")
        self.root.geometry("680x820")
        self.root.minsize(580, 640)
        self.root.resizable(True, True)  # 內容變多了，改為允許使用者自行調整視窗大小
        self.root.configure(bg=COLORS["app_bg"])
        self.root.option_add("*Font", ("Microsoft JhengHei", 10))

        # ------------------------------------------------------------
        # 【狀態變數】用來記錄程式目前的運作狀態
        # ------------------------------------------------------------
        self.recorded_points = []   # 儲存所有已記錄的座標點，格式：[{"x":x,"y":y,"key":key_or_None}, ...]
        self.is_playing = False     # 是否正在播放中（True 表示正在自動移動滑鼠，涵蓋一般播放與排程執行）
        self.stop_requested = False # 使用者是否按下了「停止」按鈕（用來通知背景執行緒中斷，也用來取消排程）
        self.is_scheduling = False  # 是否有排程正在等待/執行中（用來避免重複設定排程）
        self.scheduled_tasks = []   # 排程任務佇列，格式：[{"target_dt":..., "repeat_count":..., "interval":...}, ...]
        self.is_window_scheduling = False  # 是否正在執行「開始～結束」時段循環排程
        self.window_start_dt = None         # 本次時段循環排程的實際開始日期時間
        self.window_end_dt = None           # 本次時段循環排程的實際結束日期時間
        self.window_next_run = None         # 下一次預計執行事件清單的日期時間
        self.window_last_state = "尚未啟動"
        self.execution_logs = []            # 介面顯示的近期執行結果與錯誤紀錄
        self.log_lock = threading.Lock()     # 保護背景執行緒同時寫入紀錄檔
        self.autosave_suspended = False      # 載入資料期間暫停重複觸發自動備份
        self.app_data_dir = self._resolve_app_data_dir()
        self.autosave_path = self.app_data_dir / "mouse_flow_autosave.json"
        self.execution_log_path = self.app_data_dir / "execution_history.jsonl"

        # 建立所有畫面元件（按鈕、標籤等）
        self._build_ui()

        # 載入最近的執行紀錄與上次自動備份，不必每次重新建立流程。
        self._load_recent_execution_logs()
        self._load_autosave_on_startup()

        # 啟動「即時座標更新」的迴圈，每隔一段時間更新一次畫面上的座標顯示
        self._update_current_position()

        # 啟動「排程倒數」的即時更新迴圈——不管排程佇列有沒有按下啟動，
        # 只要佇列裡有任務，就持續顯示「距離最近一筆排程還剩多少時間」。
        self._update_schedule_countdown()

        # 啟動「開始～結束」時段循環排程的顯示更新迴圈。
        self._update_window_schedule_display()

        # 啟動全域快捷鍵監聽（F8 記錄 / F9 播放 / F10 停止 / F11 快速連按）
        self._start_hotkey_listener()

        # 當使用者按下視窗右上角「X」關閉視窗時，先呼叫 self._on_close
        # 確保背景的鍵盤監聽執行緒也會被正確關閉，不留下殘留的系統鉤子。
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    # 介面建立區
    # ==================================================================
    def _build_ui(self):
        """
        建立所有畫面元件：標籤、按鈕、清單框、滑桿等。

        整體版面用「Canvas + Scrollbar」包起來，而不是直接把元件塞進 self.root。
        原因：功能一直增加（記錄、播放、排程...），總高度已經超過螢幕能顯示的範圍，
             如果元件直接塞進固定大小的視窗，超出視窗高度的部分會被裁切、
             完全看不到也點不到（這正是排程模式按鈕消失不見的原因）。
             改用可捲動的畫布後，不管視窗多小、之後又加了多少新功能，
             使用者都能透過滾動捲軸看到並操作到最下面的元件。
        """
        # 固定在視窗底部的狀態列。即使使用者捲到很長的排程區塊，
        # 目前狀態與紅色停止按鈕仍會一直看得見。
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

        # Canvas：一塊可以「局部繪製、局部捲動」的畫布區域
        canvas = tk.Canvas(content_shell, bg=COLORS["app_bg"], highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)

        # 垂直捲軸，綁定到 canvas 的上下捲動
        outer_scrollbar = tk.Scrollbar(content_shell, orient="vertical", command=canvas.yview)
        outer_scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=outer_scrollbar.set)

        # self.main_frame 是實際承載所有 LabelFrame/元件的容器，
        # 它被放進 canvas 裡面，而不是直接放進 root。
        self.main_frame = tk.Frame(canvas, bg=COLORS["app_bg"])
        canvas_window = canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        def _on_frame_configure(event):
            # 每當 main_frame 內容改變大小（例如新增座標點），
            # 就重新計算 canvas 的「可捲動範圍」，確保捲軸長度正確。
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_resize(event):
            # 當視窗寬度改變時，讓 main_frame 的寬度跟著 canvas 一樣寬，
            # 避免內部元件的 fill="x" 失去效果（維持版面隨視窗縮放）。
            canvas.itemconfig(canvas_window, width=event.width)

        self.main_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_resize)

        def _on_mousewheel(event):
            # Windows 滑鼠滾輪的 event.delta 是 120 的倍數，
            # 除以 120 再乘上負號，讓「往上滾」對應「往上捲動」。
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # 只在滑鼠停留在這個視窗範圍內時才綁定滾輪事件，
        # 避免影響到其他同時開啟的視窗（例如按鍵對照表）。
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # -------------------- 頂部品牌區與操作引導 --------------------
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
            text="記錄座標、組合點擊流程，並用單次排程或時段循環自動執行。",
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

        btn_save_workflow = RoundedButton(
            workflow_buttons, text="儲存流程 JSON", command=self.save_workflow,
            bg=COLORS["primary"], height=40
        )
        btn_save_workflow.pack(side="left", fill="x", expand=True, padx=(0, 4))

        btn_load_workflow = RoundedButton(
            workflow_buttons, text="載入流程 JSON", command=self.load_workflow,
            bg=COLORS["teal"], height=40
        )
        btn_load_workflow.pack(side="left", fill="x", expand=True, padx=4)

        btn_restore_autosave = RoundedButton(
            workflow_buttons, text="還原自動備份", command=self.restore_autosave,
            bg=COLORS["secondary"], height=40
        )
        btn_restore_autosave.pack(side="left", fill="x", expand=True, padx=(4, 0))

        self.label_autosave_status = tk.Label(
            workflow_card,
            text="自動備份：新增、修改、排序或刪除事件後會立即更新",
            fg=COLORS["muted"], justify="left", font=("Microsoft JhengHei", 9)
        )
        self.label_autosave_status.pack(anchor="w", pady=(6, 0))

        # -------------------- 區塊一：即時滑鼠座標 --------------------
        frame_pos = tk.LabelFrame(self.main_frame, text="1  即時滑鼠座標", padx=12, pady=12)
        frame_pos.pack(fill="x", padx=18, pady=6)
        frame_pos.configure(fg=COLORS["primary"])

        # 這個 Label 會被 _update_current_position() 持續更新內容
        self.label_current_pos = tk.Label(
            frame_pos, text="X: ---, Y: ---", fg=COLORS["primary"],
            font=("Consolas", 20, "bold")
        )
        self.label_current_pos.pack()

        # -------------------- 區塊二：記錄座標 --------------------
        frame_record = tk.LabelFrame(self.main_frame, text="2  記錄一個動作", padx=12, pady=12)
        frame_record.pack(fill="x", padx=18, pady=6)
        frame_record.configure(fg=COLORS["success"])

        tk.Label(
            frame_record,
            text="先設定這個座標要做的滑鼠／鍵盤動作，再將游標移到目標位置按 F8。",
            fg=COLORS["muted"], justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(0, 5))

        # ---- 滑鼠動作設定 ----
        # 讓使用者為「下一個要記錄的座標點」選擇滑鼠要做什麼：左鍵點擊、右鍵點擊、
        # 或完全不點擊。設計成跟下面的「按鍵/組合鍵」互相獨立，
        # 這樣就能組合出各種情境：只點擊、只按鍵、點擊+按鍵、甚至兩者都不觸發
        # （例如只是想單純移動滑鼠到某處，不做任何動作）。
        #
        # 用 tk.StringVar 搭配三個 Radiobutton 是 Tkinter 裡「單選」的標準做法——
        # 三個 Radiobutton 共用同一個 StringVar，選中誰，變數的值就變成誰的 value，
        # 跟網頁表單的 <input type="radio" name="..."> 是一樣的概念。
        frame_click_type = tk.Frame(frame_record)
        frame_click_type.pack(fill="x", pady=(8, 0))

        tk.Label(frame_click_type, text="滑鼠動作：").pack(side="left")

        self.click_type_var = tk.StringVar(value="left")  # 預設為左鍵點擊，維持原本的行為

        tk.Radiobutton(
            frame_click_type, text="左鍵點擊", variable=self.click_type_var, value="left"
        ).pack(side="left")
        tk.Radiobutton(
            frame_click_type, text="右鍵點擊", variable=self.click_type_var, value="right"
        ).pack(side="left")
        tk.Radiobutton(
            frame_click_type, text="不點擊", variable=self.click_type_var, value="none"
        ).pack(side="left")

        tk.Label(
            frame_record,
            text="說明：選「不點擊」時，這個座標點只會移動滑鼠過去，不會觸發任何點擊，\n"
                 "適合搭配下面的按鍵欄位，做出「只按鍵、不點滑鼠」的動作。",
            fg="#555555", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(3, 0))

        # ---- 按鍵/組合鍵設定（選填） ----
        # 這個輸入框讓使用者可以為「下一個要記錄的座標點」
        # 附加一個按鍵動作，例如：enter、tab、esc、ctrl+c、alt+tab。
        # 記錄時（按 F8 或按鈕）會一併讀取這裡目前的內容，跟座標存在一起。
        frame_key = tk.Frame(frame_record)
        frame_key.pack(fill="x", pady=(8, 0))

        tk.Label(frame_key, text="按鍵/組合鍵（選填，例：enter、ctrl+c、alt+tab）：").pack(anchor="w")

        # 輸入框 + 對照表按鈕 放在同一列
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
        tk.Label(
            frame_text_input,
            text="文字內容（建立純文字輸入事件時使用，支援中文）："
        ).pack(anchor="w")
        self.text_event_input = tk.Text(
            frame_text_input, height=3, font=("Microsoft JhengHei", 10), wrap="word"
        )
        self.text_event_input.pack(fill="x", pady=(3, 0))
        tk.Label(
            frame_text_input,
            text="文字事件不移動滑鼠，會在目前取得焦點的輸入欄位貼上這段文字。",
            fg=COLORS["muted"], justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(3, 0))

        # ---- 單一事件專屬的時間與重複設定 ----
        # 按下 F8 時會把這些數值連同座標一起保存，因此事件 A、B、C 可以各自
        # 使用完全不同的移動時間、重複次數與等待節奏。
        event_timing_card = tk.Frame(
            frame_record, bg="#F0FDF4", padx=12, pady=10,
            highlightthickness=1, highlightbackground="#BBF7D0"
        )
        event_timing_card.pack(fill="x", pady=(10, 0))

        tk.Label(
            event_timing_card, text="此事件的時間與重複設定",
            bg="#F0FDF4", fg="#166534", font=("Microsoft JhengHei", 10, "bold")
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 7))

        tk.Label(event_timing_card, text="移動到座標（秒）：", bg="#F0FDF4").grid(
            row=1, column=0, sticky="w", pady=3
        )
        self.spin_event_move_duration = tk.Spinbox(
            event_timing_card, from_=0, to=60, increment=0.1, width=8,
            font=("Consolas", 10)
        )
        self.spin_event_move_duration.grid(row=1, column=1, sticky="ew", padx=(5, 14), pady=3)
        self._replace_spinbox_value(self.spin_event_move_duration, 0.5)

        tk.Label(event_timing_card, text="到達後等待（秒）：", bg="#F0FDF4").grid(
            row=1, column=2, sticky="w", pady=3
        )
        self.spin_event_action_delay = tk.Spinbox(
            event_timing_card, from_=0, to=60, increment=0.1, width=8,
            font=("Consolas", 10)
        )
        self.spin_event_action_delay.grid(row=1, column=3, sticky="ew", padx=(5, 0), pady=3)
        self._replace_spinbox_value(self.spin_event_action_delay, 0.5)

        tk.Label(event_timing_card, text="動作重複次數：", bg="#F0FDF4").grid(
            row=2, column=0, sticky="w", pady=3
        )
        self.spin_event_repeat_count = tk.Spinbox(
            event_timing_card, from_=1, to=9999, increment=1, width=8,
            font=("Consolas", 10)
        )
        self.spin_event_repeat_count.grid(row=2, column=1, sticky="ew", padx=(5, 14), pady=3)
        self._replace_spinbox_value(self.spin_event_repeat_count, 1)

        tk.Label(event_timing_card, text="每次重複間隔（秒）：", bg="#F0FDF4").grid(
            row=2, column=2, sticky="w", pady=3
        )
        self.spin_event_repeat_interval = tk.Spinbox(
            event_timing_card, from_=0, to=3600, increment=0.1, width=8,
            font=("Consolas", 10)
        )
        self.spin_event_repeat_interval.grid(row=2, column=3, sticky="ew", padx=(5, 0), pady=3)
        self._replace_spinbox_value(self.spin_event_repeat_interval, 1.0)

        tk.Label(event_timing_card, text="完成後等待（秒）：", bg="#F0FDF4").grid(
            row=3, column=0, sticky="w", pady=3
        )
        self.spin_event_after_wait = tk.Spinbox(
            event_timing_card, from_=0, to=3600, increment=0.1, width=8,
            font=("Consolas", 10)
        )
        self.spin_event_after_wait.grid(row=3, column=1, sticky="ew", padx=(5, 14), pady=3)
        self._replace_spinbox_value(self.spin_event_after_wait, 0.5)

        tk.Label(event_timing_card, text="失敗時：", bg="#F0FDF4").grid(
            row=4, column=0, sticky="w", pady=3
        )
        failure_policy_frame = tk.Frame(event_timing_card, bg="#F0FDF4")
        failure_policy_frame.grid(row=4, column=1, columnspan=3, sticky="w", pady=3)
        self.failure_policy_var = tk.StringVar(value="stop")
        tk.Radiobutton(
            failure_policy_frame, text="停止流程", variable=self.failure_policy_var, value="stop"
        ).pack(side="left")
        tk.Radiobutton(
            failure_policy_frame, text="跳過事件", variable=self.failure_policy_var, value="skip"
        ).pack(side="left")
        tk.Radiobutton(
            failure_policy_frame, text="重試", variable=self.failure_policy_var, value="retry"
        ).pack(side="left")
        tk.Label(failure_policy_frame, text="次數：").pack(side="left", padx=(6, 0))
        self.spin_event_retry_count = tk.Spinbox(
            failure_policy_frame, from_=1, to=20, increment=1, width=5,
            font=("Consolas", 10)
        )
        self.spin_event_retry_count.pack(side="left")
        self._replace_spinbox_value(self.spin_event_retry_count, 2)

        event_timing_card.columnconfigure(1, weight=1)
        event_timing_card.columnconfigure(3, weight=1)

        tk.Label(
            event_timing_card,
            text="執行順序：移動一次 → 到達後等待 → 重複動作 → 完成後等待 → 下一事件\n"
                 "重試次數是失敗後的額外嘗試次數，並會從該事件開頭重新執行。",
            bg="#F0FDF4", fg="#166534", justify="left",
            font=("Microsoft JhengHei", 9)
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(7, 0))

        # 按下此按鈕時，會呼叫 self.record_point 把「目前滑鼠位置」加入清單。
        # F8 是全域快捷鍵，不需要先把焦點切回本程式視窗。
        btn_record = RoundedButton(
            frame_record, text="＋  記錄目前滑鼠位置   F8", command=self.record_point,
            bg=COLORS["success"], height=44,
            font=("Microsoft JhengHei", 11, "bold")
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

        # -------------------- 區塊三：事件清單（座標點 + 快速連按事件） --------------------
        # 這個清單現在裝的不只是座標點，還可以插入「快速連按事件」，
        # 兩種項目依清單順序依序執行——這樣就能做出
        # 「移動到A點點擊 -> 原地快速連按5次 -> 移動到B點點擊」這種混合流程。
        frame_list = tk.LabelFrame(
            self.main_frame, text="3  事件流程｜由上往下依序執行", padx=12, pady=12
        )
        frame_list.pack(fill="both", expand=True, padx=18, pady=6)
        frame_list.configure(fg=COLORS["purple"])

        # 每筆事件會顯示自己的完整時間摘要，因此同時提供垂直與水平捲軸。
        frame_list_body = tk.Frame(frame_list)
        frame_list_body.pack(fill="both", expand=True)

        self.listbox_points = tk.Listbox(frame_list_body, font=("Consolas", 10), height=8)
        self.listbox_points.grid(row=0, column=0, sticky="nsew")

        scrollbar = tk.Scrollbar(frame_list_body, orient="vertical")
        scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar = tk.Scrollbar(frame_list_body, orient="horizontal")
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

        self.listbox_points.config(
            yscrollcommand=scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set
        )
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

        # 載入、更新、刪除與清空事件的操作列。
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

        # 舊版事件沒有逐筆時間欄位時使用的相容預設值。
        # 新記錄的事件會保存自己的完整時間設定，不再由播放當下的全域滑桿決定。
        self.scale_speed = tk.DoubleVar(value=0.5)
        self.scale_click_delay = tk.DoubleVar(value=0.5)
        self.scale_wait = tk.DoubleVar(value=0.5)

        # -------------------- 區塊五：播放 / 停止控制 --------------------
        frame_control = tk.LabelFrame(
            self.main_frame, text="4  執行完整事件流程", padx=12, pady=12
        )
        frame_control.pack(fill="x", padx=18, pady=8)
        frame_control.configure(fg=COLORS["primary"])

        control_buttons = tk.Frame(frame_control)
        control_buttons.pack(fill="x")

        self.btn_play = RoundedButton(
            control_buttons, text="▶  立即播放流程   F9", command=self.start_playback,
            bg=COLORS["primary"], height=48, radius=14,
            font=("Microsoft JhengHei", 12, "bold")
        )
        self.btn_play.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.btn_stop = RoundedButton(
            control_buttons, text="■  停止所有動作   F10", command=self.stop_playback,
            bg=COLORS["danger"], height=48, radius=14,
            font=("Microsoft JhengHei", 12, "bold")
        )
        self.btn_stop.pack(side="left", expand=True, fill="x")

        tk.Label(
            frame_control,
            text="按下播放後會先倒數 3 秒；倒數期間也能按 F10 取消。",
            fg=COLORS["muted"], font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(7, 0))

        frame_logs = tk.LabelFrame(
            self.main_frame, text="執行結果與錯誤紀錄", padx=12, pady=12
        )
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

        # -------------------- 區塊五之一：快速連按 --------------------
        # 這裡的設定同時支援兩種用法：
        #   1. 「⚡ 開始快速連按」：立刻在目前滑鼠位置執行一次（跟事件清單無關）
        #   2. 「➕ 插入快速連按事件」：把這裡的設定包成一筆「事件」，
        #      插進上面的事件清單裡，播放時會在清單跑到這一筆時，
        #      在「當下滑鼠所在位置」（也就是清單中前一步移動到的位置）執行，
        #      執行完才會繼續往清單下一筆走——這樣就能做出
        #      「移動到A點點擊 -> 原地快速連按5次 -> 移動到B點點擊」這種混合流程。
        frame_rapid = tk.LabelFrame(
            self.main_frame, text="進階功能｜快速連按", padx=12, pady=12
        )
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

        # ---- 連按按鍵：左鍵 / 右鍵 ----
        frame_rapid_button_type = tk.Frame(frame_rapid)
        frame_rapid_button_type.pack(fill="x")
        tk.Label(frame_rapid_button_type, text="連按按鍵：").pack(side="left")
        self.rapid_click_button_var = tk.StringVar(value="left")
        tk.Radiobutton(
            frame_rapid_button_type, text="左鍵", variable=self.rapid_click_button_var, value="left"
        ).pack(side="left")
        tk.Radiobutton(
            frame_rapid_button_type, text="右鍵", variable=self.rapid_click_button_var, value="right"
        ).pack(side="left")

        # ---- 每秒點擊次數 ----
        # 這裡的「速率」定義為：一秒鐘要點擊幾次。數值越大點越快，
        # 但實際能達到的速率仍受限於系統處理輸入事件的速度，
        # 太高的數值（例如 50 次/秒）在某些應用程式上不一定能完整反應每一次點擊。
        tk.Label(frame_rapid, text="每秒點擊次數（次/秒，數值越大越快）：").pack(anchor="w", pady=(8, 0))
        self.scale_rapid_rate = tk.Scale(
            frame_rapid, from_=1, to=50, resolution=1, orient="horizontal"
        )
        self.scale_rapid_rate.set(10)  # 預設每秒點擊 10 次
        self.scale_rapid_rate.pack(fill="x")

        # ---- 點擊次數 ----
        frame_rapid_count = tk.Frame(frame_rapid)
        frame_rapid_count.pack(fill="x", pady=(8, 0))
        tk.Label(frame_rapid_count, text="總點擊次數：").pack(side="left")
        self.spin_rapid_count = tk.Spinbox(
            frame_rapid_count, from_=1, to=9999, width=6, font=("Consolas", 11)
        )
        self.spin_rapid_count.delete(0, "end")
        self.spin_rapid_count.insert(0, "20")  # 預設連按 20 次
        self.spin_rapid_count.pack(side="left", padx=(5, 0))

        self.btn_rapid_click = RoundedButton(
            frame_rapid, text="⚡  倒數 3 秒後連按   F11", command=self.start_rapid_click,
            bg=COLORS["purple"], height=42,
            font=("Microsoft JhengHei", 10, "bold")
        )
        self.btn_rapid_click.pack(fill="x", pady=(8, 0))

        self.btn_insert_rapid_click = RoundedButton(
            frame_rapid, text="＋  將連按設定加入事件流程", command=self.insert_rapid_click_event,
            bg="#5B21B6", height=42,
            font=("Microsoft JhengHei", 10, "bold")
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
            text="排程 A：在某個時間執行固定次數。　排程 B：在開始～結束時段內，每隔 N 分鐘持續執行。",
            bg="#FFF7ED", fg="#7C2D12", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(3, 0))

        # -------------------- 區塊五之二：排程模式（多任務佇列） --------------------
        # 讓使用者設定「特定時間」+「重複執行次數」+「重複間隔」組成一筆排程任務，
        # 可以一次新增多筆，全部排進佇列，啟動後會依時間先後自動依序執行。
        # 例如：任務一 14:30 執行 3 次、任務二 16:00 執行 5 次——
        # 佇列會先等到 14:30 執行完任務一，接著繼續等到 16:00 執行任務二。
        frame_schedule = tk.LabelFrame(
            self.main_frame, text="排程 A｜指定時間與執行次數", padx=12, pady=12
        )
        frame_schedule.pack(fill="x", padx=18, pady=6)
        frame_schedule.configure(fg=COLORS["warning"])

        # ---- 距離最近一筆排程的即時倒數 ----
        # 這個標籤由 _update_schedule_countdown() 持續更新，
        # 不管佇列有沒有按下「啟動排程佇列」，只要清單裡有任務就會顯示倒數，
        # 方便使用者隨時瞄一眼還剩多久，不用等到真正啟動才看得到。
        self.label_schedule_countdown = tk.Label(
            frame_schedule, text="距離最近排程：尚無排程任務",
            font=("Consolas", 12, "bold"), fg="#E65100"
        )
        self.label_schedule_countdown.pack(anchor="w", pady=(0, 8))

        # ---- 目標執行時間 ----
        tk.Label(frame_schedule, text="執行時間（24小時制，格式 HH:MM 或 HH:MM:SS）：").pack(anchor="w")
        self.entry_schedule_time = tk.Entry(frame_schedule, font=("Consolas", 11))
        self.entry_schedule_time.pack(fill="x")

        tk.Label(
            frame_schedule,
            text="若填寫的時間已經過了（例如現在是 15:00，卻填 14:30），\n"
                 "會視為「明天的這個時間」自動執行，不會立刻觸發。",
            fg="#555555", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(2, 8))

        # ---- 重複執行次數 ----
        frame_repeat = tk.Frame(frame_schedule)
        frame_repeat.pack(fill="x")
        tk.Label(frame_repeat, text="完整事件流程重複次數：").pack(side="left")
        # Spinbox：專門用來輸入「整數次數」的元件，有上下箭頭可以調整數值，
        # 也能直接打字輸入，比 Scale 滑桿更適合「次數」這種離散整數的情境。
        self.spin_repeat = tk.Spinbox(frame_repeat, from_=1, to=999, width=6, font=("Consolas", 11))
        self.spin_repeat.delete(0, "end")
        self.spin_repeat.insert(0, "1")  # 預設執行 1 次
        self.spin_repeat.pack(side="left", padx=(5, 0))

        # ---- 每次重複之間的間隔秒數 ----
        # 解析度改成 0.1 秒（原本是 1 秒），可以設定更精細的重複間隔，例如 0.3 秒。
        tk.Label(frame_schedule, text="每次完整流程之間的間隔秒數（可精細到 0.1 秒）：").pack(anchor="w", pady=(8, 0))
        self.scale_interval = tk.Scale(
            frame_schedule, from_=0, to=60, resolution=0.1, orient="horizontal"
        )
        self.scale_interval.set(2)  # 預設每次重複之間間隔 2 秒
        self.scale_interval.pack(fill="x")

        # ---- 新增到排程佇列按鈕 ----
        # 按下這顆按鈕只是把「目前設定的時間/次數/間隔」包成一筆任務，加進佇列清單，
        # 並不會立刻開始倒數——真正開始執行要按下面的「▶ 啟動排程佇列」。
        self.btn_add_task = RoundedButton(
            frame_schedule, text="＋  將此設定加入排程佇列", command=self.add_schedule_task,
            bg=COLORS["warning"], height=42,
            font=("Microsoft JhengHei", 10, "bold")
        )
        self.btn_add_task.pack(fill="x", pady=(8, 0))

        # ---- 排程任務佇列清單 ----
        tk.Label(frame_schedule, text="排程任務佇列（依執行時間排序）：").pack(anchor="w", pady=(10, 0))

        frame_schedule_list = tk.Frame(frame_schedule)
        frame_schedule_list.pack(fill="both", expand=True)

        self.listbox_schedule = tk.Listbox(frame_schedule_list, font=("Consolas", 10), height=5)
        self.listbox_schedule.pack(side="left", fill="both", expand=True)

        schedule_scrollbar = tk.Scrollbar(frame_schedule_list, orient="vertical")
        schedule_scrollbar.pack(side="right", fill="y")
        self.listbox_schedule.config(yscrollcommand=schedule_scrollbar.set)
        schedule_scrollbar.config(command=self.listbox_schedule.yview)

        # 刪除選取任務 / 清空全部任務 的按鈕列
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

        # ---- 啟動排程佇列按鈕 ----
        # 注意：這裡沒有另外做「取消排程」按鈕 —— 因為佇列執行期間
        # self.is_playing 會被設成 True，此時既有的「⏹ 停止 (F10)」
        # 按鈕與快捷鍵就能直接取消倒數中或執行中的整個佇列，邏輯是共用的。
        self.btn_schedule = RoundedButton(
            frame_schedule, text="▶  啟動排程佇列", command=self.start_schedule_queue,
            bg=COLORS["success"], height=44,
            font=("Microsoft JhengHei", 11, "bold")
        )
        self.btn_schedule.pack(fill="x", pady=(10, 0))

        tk.Label(
            frame_schedule,
            text="提示：每輪會在排定時間前 3 秒顯示安全倒數；等待、倒數或執行中都可按 F10 中止。\n"
                 "執行中無法新增或刪除任務，需先停止。",
            fg="#555555", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(3, 0))

        # -------------------- 區塊五之三：開始～結束時段循環排程 --------------------
        # 與上方「特定時間 + 固定重複次數」的任務佇列不同，這個模式用一段
        # 明確的有效時段控制自動化：從開始時間起，每隔 N 分鐘完整播放一次
        # 事件清單，直到結束時間。若結束時間早於或等於開始時間，代表跨午夜。
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

        tk.Label(frame_window_times, text="開始時間：").grid(row=0, column=0, sticky="w")
        self.entry_window_start = tk.Entry(frame_window_times, width=12, font=("Consolas", 11))
        self.entry_window_start.grid(row=0, column=1, sticky="ew", padx=(3, 12))

        tk.Label(frame_window_times, text="結束時間：").grid(row=0, column=2, sticky="w")
        self.entry_window_end = tk.Entry(frame_window_times, width=12, font=("Consolas", 11))
        self.entry_window_end.grid(row=0, column=3, sticky="ew", padx=(3, 0))

        frame_window_times.columnconfigure(1, weight=1)
        frame_window_times.columnconfigure(3, weight=1)

        # 預填「一分鐘後開始、一小時後結束」，開啟程式即可直接調整。
        default_start = datetime.now() + timedelta(minutes=1)
        default_end = default_start + timedelta(hours=1)
        self.entry_window_start.insert(0, default_start.strftime("%H:%M"))
        self.entry_window_end.insert(0, default_end.strftime("%H:%M"))

        tk.Label(
            frame_window_schedule,
            text="格式：HH:MM 或 HH:MM:SS。若結束時間早於開始時間，會自動視為跨午夜，\n"
                 "例如 22:00～06:00。",
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
                 "例如事件清單為 A 點、B 點，就會每隔指定分鐘執行一次 A → B。\n"
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

        tk.Label(
            frame_window_schedule,
            text="每輪執行前會先倒數 3 秒；等待、倒數或執行中皆可按 F10 取消。",
            fg="#555555", justify="left", font=("Microsoft JhengHei", 9)
        ).pack(anchor="w", pady=(3, 0))

        safety_tip = tk.Frame(
            self.main_frame, bg="#FFF1F2", padx=14, pady=12,
            highlightthickness=1, highlightbackground="#FECDD3"
        )
        safety_tip.pack(fill="x", padx=18, pady=(6, 18))
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
        """統一套用卡片、輸入框、清單、滑桿與選項元件的現代化樣式。"""
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

            elif isinstance(widget, tk.Radiobutton):
                widget.configure(
                    bg=parent_bg, fg=COLORS["text"], activebackground=parent_bg,
                    activeforeground=COLORS["text"], selectcolor=parent_bg,
                    highlightthickness=0, cursor="hand2"
                )

            elif isinstance(widget, tk.Checkbutton):
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

    # ==================================================================
    # 座標即時更新
    # ==================================================================
    def _update_current_position(self):
        """
        使用 pyautogui.position() 取得目前滑鼠座標，
        並更新到畫面上的標籤。
        使用 self.root.after() 讓這個函式每 50 毫秒自動呼叫自己一次，
        形成一個「非阻塞式」的更新迴圈（不會卡住 Tkinter 的主事件迴圈）。
        """
        x, y = pyautogui.position()  # 取得目前滑鼠在螢幕上的絕對座標 (x, y)
        self.label_current_pos.config(text=f"X: {x}, Y: {y}")

        # 50 毫秒後再次呼叫自己，達成「即時更新」的效果
        self.root.after(50, self._update_current_position)

    def _update_schedule_countdown(self):
        """
        持續更新「距離最近排程還剩多少時間」的顯示。

        跟 _update_current_position() 是同樣的設計模式：用 self.root.after()
        讓函式每隔一段時間自動呼叫自己一次，形成非阻塞式的即時更新迴圈。

        佇列裡只要有任務，不管排程有沒有按下「啟動」都會持續顯示倒數，
        讓使用者能先確認時間設定得對不對；但「時間已到之後」要顯示成
        「執行中」還是「尚未啟動，時間已過」，就必須靠 self.is_playing
        來分辨——否則會誤導使用者以為背景真的在跑，但其實根本沒有執行緒在動作。
        """
        if not self.scheduled_tasks:
            self.label_schedule_countdown.config(text="距離最近排程：尚無排程任務")
        else:
            # 從佇列中找出「執行時間最早」的任務，這就是目前最接近觸發的排程
            nearest_task = min(self.scheduled_tasks, key=lambda t: t["target_dt"])
            remaining = (nearest_task["target_dt"] - datetime.now()).total_seconds()

            if remaining > 0:
                # 格式化剩餘秒數為「時:分:秒」
                total_seconds = int(remaining)
                hours, rem = divmod(total_seconds, 3600)
                minutes, seconds = divmod(rem, 60)
                self.label_schedule_countdown.config(
                    text=f"距離最近排程還有 {hours:02d}:{minutes:02d}:{seconds:02d}"
                         f"（將於 {nearest_task['target_dt'].strftime('%H:%M:%S')} 執行）"
                )
            else:
                # remaining <= 0 只代表「這筆任務的目標時間已經到了」，
                # 不代表背景真的有執行緒在跑——如果使用者根本還沒按「啟動排程佇列」，
                # 或是佇列已經被停止/跑完，時間到了也不會有任何動作發生。
                # 這裡務必用 self.is_playing 分辨這兩種情況，
                # 否則會讓使用者誤以為「顯示執行中 = 滑鼠正在動作」，
                # 但其實背景根本沒有執行緒在運作，看起來就像「排程卡住沒反應」。
                if self.is_scheduling:
                    self.label_schedule_countdown.config(text="最近的排程任務正在執行中...")
                else:
                    self.label_schedule_countdown.config(
                        text="⚠ 任務時間已到，但尚未啟動！請按下「▶ 啟動排程佇列」"
                    )

        # 每 500 毫秒更新一次，數字用秒為單位，這個更新頻率已經足夠流暢
        self.root.after(500, self._update_schedule_countdown)

    def _update_window_schedule_display(self):
        """每 500 毫秒更新一次「開始～結束時段循環排程」的狀態。"""
        if not self.is_window_scheduling:
            self.label_window_schedule.config(text=f"時段排程：{self.window_last_state}")
            self.root.after(500, self._update_window_schedule_display)
            return

        now = datetime.now()
        start_dt = self.window_start_dt
        end_dt = self.window_end_dt
        next_run = self.window_next_run

        if start_dt is None or end_dt is None:
            display_text = "時段排程：正在準備..."
        elif now < start_dt:
            wait_seconds = max(0, int((start_dt - now).total_seconds()))
            hours, remainder = divmod(wait_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            display_text = (
                f"等待開始：{start_dt.strftime('%Y-%m-%d %H:%M:%S')} "
                f"（倒數 {hours:02d}:{minutes:02d}:{seconds:02d}）"
            )
        elif now >= end_dt:
            display_text = "已到達結束時間，正在停止..."
        elif next_run is None:
            display_text = f"有效時段內，流程執行中；結束：{end_dt.strftime('%H:%M:%S')}"
        else:
            next_seconds = max(0, int((next_run - now).total_seconds()))
            remaining_seconds = max(0, int((end_dt - now).total_seconds()))
            next_h, next_remainder = divmod(next_seconds, 3600)
            next_m, next_s = divmod(next_remainder, 60)
            end_h, end_remainder = divmod(remaining_seconds, 3600)
            end_m, end_s = divmod(end_remainder, 60)
            display_text = (
                f"下次執行倒數 {next_h:02d}:{next_m:02d}:{next_s:02d}｜"
                f"時段剩餘 {end_h:02d}:{end_m:02d}:{end_s:02d}"
            )

        self.label_window_schedule.config(text=display_text)
        self.root.after(500, self._update_window_schedule_display)

    # ==================================================================
    # 記錄座標相關功能
    # ==================================================================
    def show_key_reference(self):
        """
        開啟一個獨立的小視窗 (Toplevel)，顯示 pyautogui 支援的按鍵名稱對照表，
        依「數字」「英文字母」「常用符號」「功能鍵」「方向鍵」「特殊/修飾鍵」分類，
        方便使用者填寫上方的「按鍵/組合鍵」輸入框時直接查閱，不用去翻官方文件。

        注意：這份清單是 pyautogui 內建對應「美式鍵盤佈局 (US QWERTY)」的鍵名，
             符號鍵是否正確，會受 Windows 系統目前的鍵盤佈局影響。
        """
        # tk.Toplevel：建立一個「額外的視窗」，獨立於主視窗之外，
        # 但仍然屬於同一個應用程式（關閉主視窗時，這種子視窗也會一併消失）。
        window = tk.Toplevel(self.root)
        window.title("按鍵名稱對照表｜Mouse Flow Studio")
        window.geometry("500x620")
        window.minsize(420, 480)
        window.configure(bg=COLORS["app_bg"])

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

        # Text 元件用來顯示大量文字內容，搭配捲軸方便瀏覽
        text_widget = tk.Text(
            reference_body, font=("Consolas", 10), wrap="word",
            bg=COLORS["field"], fg=COLORS["text"], relief="flat", bd=0,
            padx=12, pady=12, selectbackground=COLORS["primary"]
        )
        text_widget.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(reference_body, orient="vertical", command=text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        text_widget.config(yscrollcommand=scrollbar.set)

        # ------------------------------------------------------------
        # 按鍵名稱清單，依類別整理。
        # 這些名稱直接對應 pyautogui 內部的 KEYBOARD_KEYS 清單，
        # 填在「按鍵/組合鍵」輸入框裡時，打法要完全一致（英文小寫）。
        # ------------------------------------------------------------
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
        text_widget.config(state="disabled")  # 設為唯讀，避免使用者誤改內容

    @staticmethod
    def _replace_spinbox_value(spinbox, value):
        """安全地替換 Spinbox 目前顯示的值。"""
        spinbox.delete(0, "end")
        spinbox.insert(0, str(value))

    def _read_event_timing_settings(self):
        """讀取並驗證「此事件的時間與重複設定」。"""
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
        """將事件內保存的時間參數載入編輯欄位；舊事件使用相容預設值。"""
        self._replace_spinbox_value(
            self.spin_event_move_duration, event.get("move_duration", 0.5)
        )
        self._replace_spinbox_value(
            self.spin_event_action_delay, event.get("action_delay", 0.5)
        )
        self._replace_spinbox_value(
            self.spin_event_repeat_count, event.get("repeat_count", 1)
        )
        self._replace_spinbox_value(
            self.spin_event_repeat_interval, event.get("repeat_interval", 1.0)
        )
        self._replace_spinbox_value(
            self.spin_event_after_wait, event.get("after_wait", 0.5)
        )
        self.failure_policy_var.set(event.get("failure_policy", "stop"))
        self._replace_spinbox_value(
            self.spin_event_retry_count, event.get("retry_count", 2)
        )

    def load_selected_event(self):
        """把清單中選取事件的設定載入上方編輯區，方便檢查或修改。"""
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

        self._set_status(
            f"已載入事件 {index + 1}（{event_type}），可修改後按「更新選取事件」"
        )

    def update_selected_event(self):
        """使用目前表單值更新選取事件，但不變更原本記錄的座標。"""
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
        """
        將「目前滑鼠位置」加入事件清單，並顯示在 Listbox 中。

        資料結構說明：
            清單裡現在有兩種事件類型，用 "type" 欄位分辨：

            座標點事件：
                {"type": "point", "x": 100, "y": 200, "click_type": "left",
                 "key": "ctrl+c", "move_duration": 0.5, "action_delay": 0.5,
                 "repeat_count": 3, "repeat_interval": 1.0, "after_wait": 0.5}
            快速連按事件：
                {"type": "rapid_click", "rate": 10, "count": 20, "button_type": "left"}

        插入位置：如果 Listbox 裡目前有選取項目，新記錄的座標點會插入到
        「選取項目的下一個位置」，方便把新座標點安插在中間；
        沒有選取任何項目時，就加到清單最後面（原本的行為）。
        """
        timing = self._read_event_timing_settings()
        if timing is None:
            return

        x, y = pyautogui.position()  # 抓取目前座標

        # 讀取「滑鼠動作」單選鈕目前選中的值：left / right / none
        click_type = self.click_type_var.get()

        # 讀取「按鍵/組合鍵」輸入框目前的內容。
        # .strip() 去除頭尾空白；若使用者沒填，會是空字串 ""。
        key_text = self.entry_key.get().strip()
        key_value = key_text if key_text else None  # 空字串一律轉成 None，代表「沒有按鍵」

        event = {
            "type": "point", "x": x, "y": y,
            "click_type": click_type, "key": key_value,
            "enabled": True,
            **timing,
        }

        # 依「目前是否有選取項目」決定插入位置
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
        """依目前清單選取位置插入事件，未選取時加入清單底部。"""
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
        """新增不記錄座標、也不移動滑鼠的純快捷鍵事件。"""
        key_text = self.entry_key.get().strip()
        if not key_text:
            messagebox.showwarning("提示", "請先在「按鍵／組合鍵」欄位輸入快捷鍵")
            return

        timing = self._read_event_timing_settings()
        if timing is None:
            return

        event = {
            "type": "key", "key": key_text, "enabled": True,
            **timing,
        }
        index = self._insert_event_after_selection(event)
        self._set_status(
            f"已新增純快捷鍵事件 {index + 1}：{key_text} ×{timing['repeat_count']}"
        )
        self._autosave_workflow("新增快捷鍵事件")

    def add_text_event(self):
        """新增不移動滑鼠、在目前焦點位置輸入指定文字的事件。"""
        text_value = self.text_event_input.get("1.0", "end-1c")
        if not text_value.strip():
            messagebox.showwarning("提示", "請先輸入要寫入目標欄位的文字內容")
            return

        timing = self._read_event_timing_settings()
        if timing is None:
            return

        event = {
            "type": "text", "text": text_value, "enabled": True,
            **timing,
        }
        index = self._insert_event_after_selection(event)
        self._set_status(
            f"已新增文字輸入事件 {index + 1}：{len(text_value)} 個字元 ×{timing['repeat_count']}"
        )
        self._autosave_workflow("新增文字輸入事件")

    @staticmethod
    def _format_event_label(index, event):
        """
        統一產生 Listbox 中每一列的顯示文字，依事件類型分別格式化，
        獨立成一個函式方便 record_point() / insert_rapid_click_event() /
        _refresh_listbox_labels() 共用，避免各處格式寫法不一致。
        """
        state_mark = "●" if event.get("enabled", True) else "⏸"
        failure_policy = event.get("failure_policy", "stop")
        if failure_policy == "retry":
            failure_label = f"重試{event.get('retry_count', 2)}次"
        else:
            failure_label = {"stop": "失敗停止", "skip": "失敗跳過"}.get(
                failure_policy, "失敗停止"
            )

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

        # event["type"] == "point"
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
        """
        刪除使用者在 Listbox 中選取的項目（座標點或快速連按事件皆可）。
        """
        if self.is_playing:
            messagebox.showinfo("提示", "流程執行中無法刪除事件，請先停止")
            return
        selection = self.listbox_points.curselection()  # 取得目前選取的項目 index（tuple）
        if not selection:
            messagebox.showinfo("提示", "請先在清單中選取要刪除的項目")
            return

        index = selection[0]  # curselection 回傳 tuple，取第一個即可（單選模式）
        del self.recorded_points[index]        # 從資料清單中移除
        self._refresh_listbox_labels()         # 重新編號並重畫剩餘的項目
        self._set_status("已刪除選取的項目")
        self._autosave_workflow("刪除事件")

    def clear_all_points(self):
        """
        清空所有已記錄的事件（座標點與快速連按事件都會被清空）。
        """
        if self.is_playing:
            messagebox.showinfo("提示", "流程執行中無法清空事件，請先停止")
            return
        if not self.recorded_points:
            return
        if not messagebox.askyesno("確認清空", "確定要刪除事件流程中的全部項目嗎？"):
            return
        self.recorded_points.clear()
        self.listbox_points.delete(0, tk.END)  # 清空 Listbox 所有項目
        self._set_status("已清空所有項目")
        self._autosave_workflow("清空事件")

    def move_selected_event(self, direction):
        """將選取事件向上或向下移動一格。"""
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
        """複製選取事件並插入到原事件下一格。"""
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
        """切換選取事件的啟用狀態；停用事件播放時會直接略過。"""
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
        """
        新增/刪除/插入事件後，重新產生 Listbox 的顯示文字，
        讓「點 1、事件 2、點 3...」的編號保持連續正確。
        """
        self.listbox_points.delete(0, tk.END)
        for i, event in enumerate(self.recorded_points, start=1):
            self.listbox_points.insert(tk.END, self._format_event_label(i, event))
            if not event.get("enabled", True):
                self.listbox_points.itemconfig(tk.END, fg="#94A3B8")

    def insert_rapid_click_event(self):
        """
        按下「➕ 插入快速連按事件到清單」時呼叫。

        把目前「快速連按」區塊設定的速率/次數/按鍵，包成一筆「快速連按事件」，
        插入到事件清單裡——插入位置規則跟 record_point() 一致：
        Listbox 有選取項目就插到選取項目的下一個位置，沒有就加到最後面。

        這個事件在播放時，會在「當下滑鼠所在位置」
        （也就是清單中前一步移動到的位置）連續點擊，
        執行完才會繼續往清單下一筆走，藉此達成「插在任兩個座標點中間」的效果。
        """
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

        button_type = self.rapid_click_button_var.get()  # "left" 或 "right"

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
        self._set_status(
            f"已插入快速連按事件：{button_label}，每秒 {rate:g} 次，"
            f"共 {click_count} 次，完成後等待 {after_wait:g} 秒"
        )
        self._autosave_workflow("新增快速連按事件")

    # ==================================================================
    # 播放（自動移動 + 點擊）相關功能
    # ==================================================================
    def start_playback(self):
        """
        按下「開始播放」時呼叫。
        真正的移動/點擊邏輯放在背景執行緒 (threading.Thread) 執行，
        原因：如果直接在主執行緒跑迴圈，Tkinter 的視窗畫面會「卡住」無法更新，
             使用者也按不到「停止」按鈕。
        """
        if self.is_playing:
            messagebox.showinfo("提示", "目前正在播放中")
            return

        if not self.recorded_points:
            messagebox.showwarning("提示", "事件清單是空的，請先新增至少一個事件")
            return

        self.is_playing = True
        self.stop_requested = False
        self._set_status("播放中...（可按下停止按鈕中斷）")

        # Tkinter 元件只能在主執行緒讀取；先取得播放設定，再傳給背景執行緒。
        speed = self.scale_speed.get()
        click_delay = self.scale_click_delay.get()
        wait_time = self.scale_wait.get()

        # 建立一個背景執行緒來跑 _playback_worker，
        # daemon=True 表示：若主視窗被關閉，這個執行緒也會跟著結束，不會卡住程式退出。
        thread = threading.Thread(
            target=self._playback_worker,
            args=(speed, click_delay, wait_time),
            daemon=True
        )
        thread.start()

    def _interruptible_sleep(self, seconds, deadline=None):
        """
        可被 F10 或時段結束時間中斷的等待。

        一般 time.sleep(60) 會讓停止按鈕最多延遲 60 秒才生效；這裡把等待切成
        最多 0.1 秒的小段，讓停止指令與時段截止都能快速被偵測。
        """
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

    def _run_trajectory_loop_legacy(self, speed, click_delay, wait_time, deadline=None):
        """
        依序執行事件清單裡的每一筆事件——清單裡有兩種事件類型：
            "point"       —— 移動一次 -> 動作前等待 -> 重複點擊/按鍵 -> 完成後等待
            "rapid_click" —— 在「當下滑鼠所在位置」原地連續點擊指定次數 -> 停留

        這段邏輯被抽成獨立函式，是因為「一般播放」(F9/按鈕) 跟「排程模式」
        都需要跑同一套動作——排程模式甚至要重複跑好幾遍——
        把核心邏輯抽出來共用，可以避免兩個地方各寫一份、日後改一次卻忘記改另一邊。

        speed / click_delay / wait_time 只作為舊版事件缺少逐筆欄位時的相容預設值；
        新事件會使用自己保存的 move_duration、action_delay、repeat_count、
        repeat_interval 與 after_wait。

        回傳值：
            "completed" —— 所有事件都正常跑完
            "stopped"   —— 途中被使用者按下停止而中斷
            "deadline"  —— 到達時段循環排程的結束時間

        注意：這個函式本身「不」處理 pyautogui.FailSafeException，
             刻意讓例外往外拋出，由呼叫端（_playback_worker / _multi_schedule_worker）
             自行決定發生 FAILSAFE 時要怎麼處理（例如排程模式遇到 FAILSAFE
             應該整個排程都中止，而不是只中止當次重複）。
        """
        for i, event in enumerate(self.recorded_points, start=1):
            # 每次動作前先檢查使用者是否按下了「停止」
            if self.stop_requested:
                self._set_status("播放已被使用者中止")
                return "stopped"
            if deadline is not None and datetime.now() >= deadline:
                self._set_status("已到達結束時間，停止目前流程")
                return "deadline"

            # ------------------------------------------------------
            # 快速連按事件：不移動滑鼠，直接在目前位置連續點擊。
            # 呼叫跟「⚡ 開始快速連按」共用的 _execute_rapid_click()，
            # 避免同一套連按邏輯在兩個地方各寫一份。
            # ------------------------------------------------------
            if event["type"] == "rapid_click":
                self._set_status(f"第 {i} 項：執行快速連按（共 {event['count']} 次）...")
                result = self._execute_rapid_click(
                    event["rate"], event["count"], event["button_type"], deadline=deadline
                )
                if result != "completed":
                    return result

                rapid_after_wait = event.get("after_wait", wait_time)
                wait_result = self._interruptible_sleep(rapid_after_wait, deadline)
                if wait_result != "completed":
                    return wait_result
                self._set_status(f"已完成第 {i} 項的快速連按")
                continue  # 跳過下面「座標點」專屬的邏輯，直接處理下一筆事件

            # ------------------------------------------------------
            # 座標點事件（event["type"] == "point"）
            # ------------------------------------------------------
            x, y = event["x"], event["y"]
            click_type = event.get("click_type", "left")  # 用 .get() 給預設值，避免舊資料沒有這個欄位時出錯
            key_value = event.get("key")
            move_duration = event.get("move_duration", speed)
            action_delay = event.get("action_delay", click_delay)
            repeat_count = event.get("repeat_count", 1)
            repeat_interval = event.get("repeat_interval", 1.0)
            after_wait = event.get("after_wait", wait_time)

            self._set_status(f"移動到第 {i} 點：({x}, {y}) ...")

            # ------------------------------------------------------
            # pyautogui.moveTo(x, y, duration)：
            #   將滑鼠從目前位置「平滑移動」到指定座標 (x, y)。
            #   duration 參數控制移動所花費的時間（秒），
            #   數值越大，移動軌跡越像人手操作的漸進移動；
            #   數值為 0 則是瞬間跳到該座標。
            #
            # 底層原理：pyautogui 會把 (目前座標) 到 (目標座標) 之間
            #   拆成很多小步驟，每一小步呼叫一次 Win32 的 SetCursorPos，
            #   搭配極短暫的 sleep，讓畫面上看起來像是平滑移動。
            #
            # 重點修正：duration 是「這次移動固定要花的時間」，
            #   跟「移動距離」完全無關——就算目標座標跟目前滑鼠位置一模一樣
            #   （例如連續兩個點都是同一個座標，只是想在原地重複點擊/按鍵），
            #   pyautogui 還是會乖乖把 duration 秒的時間全部睡完，才繼續往下走，
            #   這正是「座標沒變、間隔卻還是被移動速度拖慢」的原因。
            #   這裡先查詢目前滑鼠實際位置，如果跟目標座標相同就完全跳過
            #   moveTo()，不套用任何動畫時間，直接進入下一步。
            # ------------------------------------------------------
            current_x, current_y = pyautogui.position()
            if (current_x, current_y) != (x, y):
                pyautogui.moveTo(x, y, duration=move_duration)
            # 座標相同時，滑鼠本來就已經在那裡了，不需要移動也不需要套用 duration

            # 再次檢查是否被要求停止（因為移動需要時間，移動完後可能使用者已按停止）
            if self.stop_requested:
                self._set_status("播放已被使用者中止")
                return "stopped"
            if deadline is not None and datetime.now() >= deadline:
                self._set_status("已到達結束時間，停止目前流程")
                return "deadline"

            # 定位完成後，先等待「點擊觸發間隔」（預設 0.5 秒，可自行調整）。
            # 這個等待跟下面的 wait_time 是分開的兩個獨立設定，
            # 前者控制「多快觸發動作」，後者控制「動作完後停留多久才前進下一點」。
            # 就算這個點選擇「不點擊」，還是保留這段等待，讓「定位 -> 動作」的節奏維持一致
            # （例如只想按鍵不點擊時，通常也需要先讓目標視窗有時間反應游標移動過去這件事）。
            wait_result = self._interruptible_sleep(action_delay, deadline)
            if wait_result != "completed":
                return wait_result

            # 滑鼠只移動一次；以下迴圈只重複此座標綁定的「點擊／按鍵」。
            # 同時設定滑鼠與鍵盤時，每一輪都是先點擊、再送出按鍵。
            for repeat_index in range(1, repeat_count + 1):
                if self.stop_requested:
                    self._set_status("播放已被使用者中止")
                    return "stopped"
                if deadline is not None and datetime.now() >= deadline:
                    self._set_status("已到達結束時間，停止目前流程")
                    return "deadline"

                self._set_status(
                    f"第 {i} 點：執行點擊／按鍵 {repeat_index}/{repeat_count} 次"
                )

                if click_type in ("left", "right"):
                    pyautogui.click(button=click_type)

                if key_value:
                    try:
                        key_parts = [
                            key.strip().lower()
                            for key in key_value.split("+")
                            if key.strip()
                        ]
                        if len(key_parts) == 1:
                            pyautogui.press(key_parts[0])
                        elif len(key_parts) > 1:
                            pyautogui.hotkey(*key_parts)
                    except Exception as key_err:
                        # 單次按鍵名稱錯誤不會中斷整條流程，後續事件仍可繼續。
                        self._set_status(f"第 {i} 點按鍵模擬失敗：{key_err}")

                if repeat_index < repeat_count:
                    repeat_wait_result = self._interruptible_sleep(
                        repeat_interval, deadline
                    )
                    if repeat_wait_result != "completed":
                        return repeat_wait_result

            # 此事件的所有重複動作完成後，再依自己的 after_wait 前往下一事件。
            wait_result = self._interruptible_sleep(after_wait, deadline)
            if wait_result != "completed":
                return wait_result

            self._set_status(f"已完成第 {i} 點的動作")

        # for 迴圈正常跑完（沒有中途 return "stopped"），代表整條軌跡都執行完成
        return "completed"

    def _run_start_countdown(self, context="流程", seconds=3, deadline=None):
        """在任何實際輸入事件前顯示可中止的安全倒數。"""
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
        """產生安全的事件摘要；文字事件只顯示字數，不把內容寫入紀錄。"""
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
        """執行單鍵或以加號分隔的組合快捷鍵。"""
        key_parts = [part.strip().lower() for part in str(key_value).split("+") if part.strip()]
        if not key_parts:
            raise ValueError("快捷鍵內容不可為空")
        if len(key_parts) == 1:
            pyautogui.press(key_parts[0])
        else:
            pyautogui.hotkey(*key_parts)

    @staticmethod
    def _input_text_value(text_value):
        """輸入一段文字；含中文時優先透過剪貼簿貼上。"""
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
                # 某些 Linux/遠端環境雖安裝 pyperclip，卻沒有可用的剪貼簿後端。
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
        """執行單一事件一次；事件內的 repeat_count 只重複動作本身。"""
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

        wait_result = self._interruptible_sleep(
            event.get("action_delay", click_delay), deadline
        )
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

    def _execute_event_with_policy(
        self, index, event, speed, click_delay, wait_time, deadline=None
    ):
        """套用單筆事件的失敗策略：停止、跳過，或重試指定次數。"""
        policy = event.get("failure_policy", "stop")
        retry_count = max(1, int(event.get("retry_count", 2)))
        description = self._event_description(event, index)
        attempt = 0

        while True:
            attempt += 1
            try:
                result = self._execute_event_core(
                    index, event, speed, click_delay, wait_time, deadline
                )
                if result == "completed":
                    self._add_execution_log("success", f"{description} 執行成功")
                return result
            except pyautogui.FailSafeException:
                raise
            except Exception as error:
                error_text = f"{type(error).__name__}: {error}"
                if policy == "skip":
                    self._add_execution_log(
                        "warning", f"{description} 失敗後已跳過：{error_text}"
                    )
                    self._set_status(f"{description} 失敗，已依設定跳過")
                    return "skipped"

                if policy == "retry" and attempt <= retry_count:
                    self._add_execution_log(
                        "warning",
                        f"{description} 失敗，準備重試 {attempt}/{retry_count}：{error_text}",
                    )
                    self._set_status(f"{description} 失敗，1 秒後重試 {attempt}/{retry_count}")
                    wait_result = self._interruptible_sleep(1.0, deadline)
                    if wait_result != "completed":
                        return wait_result
                    continue

                self._add_execution_log("error", f"{description} 執行失敗：{error_text}")
                self._set_status(f"{description} 執行失敗，流程已停止")
                return "failed"

    def _run_trajectory_loop(self, speed, click_delay, wait_time, deadline=None):
        """依清單順序執行啟用事件，並套用每筆事件自己的時間與失敗策略。"""
        for index, event in enumerate(self.recorded_points, start=1):
            if self.stop_requested:
                self._set_status("播放已被使用者中止")
                return "stopped"
            if deadline is not None and datetime.now() >= deadline:
                self._set_status("已到達結束時間，停止目前流程")
                return "deadline"
            if not event.get("enabled", True):
                self._add_execution_log(
                    "info", f"{self._event_description(event, index)} 已停用，略過"
                )
                continue

            result = self._execute_event_with_policy(
                index, event, speed, click_delay, wait_time, deadline
            )
            if result not in {"completed", "skipped"}:
                return result

        return "completed"

    def _playback_worker(self, speed, click_delay, wait_time):
        """
        「一般播放」(F9 / 按鈕觸發) 的執行緒進入點。
        只負責：設定狀態 -> 呼叫共用的 _run_trajectory_loop 跑一次 -> 處理例外 -> 收尾。
        這個函式在背景執行緒中執行，不會阻塞 GUI。
        """
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
            # ----------------------------------------------------------
            # 【安全機制觸發】
            # 當使用者把滑鼠快速移到螢幕左上角 (0,0) 時，
            # pyautogui 偵測到這個位置，下一次呼叫 moveTo/click 等函式
            # 就會拋出 FailSafeException，中斷所有後續動作。
            # ----------------------------------------------------------
            self._set_status("⚠ 觸發 FAILSAFE 安全機制！已立即停止所有滑鼠操作")
            self._add_execution_log("error", "手動播放觸發 FAILSAFE，已立即停止")

        except Exception as error:
            self._set_status(f"播放發生未預期錯誤：{error}")
            self._add_execution_log(
                "error", f"手動播放未預期錯誤：{type(error).__name__}: {error}"
            )

        finally:
            # 不論播放是正常結束、被使用者停止、還是被 FAILSAFE 中斷，
            # 都要把狀態重置回「未播放」，讓使用者可以再次按下開始播放。
            self.is_playing = False

    def stop_playback(self):
        """
        按下「停止」按鈕時呼叫。
        只是把 stop_requested 設成 True，
        真正的中止動作是由 _run_trajectory_loop / _multi_schedule_worker /
        _rapid_click_worker 裡的迴圈自行檢查後跳出。
        （這是執行緒安全的做法，不直接強制殺掉執行緒）

        這個函式同時服務四種情境：
            1. 一般播放中 —— 中斷目前正在跑的軌跡
            2. 排程等待或執行中 —— 取消倒數，或中斷正在重複執行的軌跡
            3. 快速連按中 —— 中斷連續點擊
            4. 開始～結束時段循環排程 —— 中斷等待或目前正在執行的流程
        四者都是靠 self.is_playing / self.stop_requested 這兩個共用旗標判斷，
        所以不需要另外幫每種功能各寫一個「停止」函式。
        """
        if not self.is_playing:
            self._set_status("目前沒有正在播放、排程或連按中的動作")
            return
        self.stop_requested = True
        self._set_status("已送出停止指令，等待目前動作中止...")
        self._add_execution_log("warning", "使用者送出停止指令")

    # ==================================================================
    # 快速連按（在目前滑鼠位置，以指定速率連續點擊）
    # ==================================================================
    def start_rapid_click(self):
        """
        按下「⚡ 開始快速連按」(或 F11) 時呼叫。
        跟「播放」不同的地方：這裡完全不使用 self.recorded_points，
        只是在「當下滑鼠所在的位置」原地連續點擊，不會移動滑鼠。
        """
        if self.is_playing:
            messagebox.showinfo("提示", "目前已有播放、排程或連按正在進行中，請先停止")
            return

        rate = self.scale_rapid_rate.get()  # 每秒點擊次數

        try:
            click_count = int(self.spin_rapid_count.get())
            if click_count < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("錯誤", "總點擊次數必須是大於等於 1 的整數")
            return

        button_type = self.rapid_click_button_var.get()  # "left" 或 "right"

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
        """
        「⚡ 開始快速連按」(F11 / 按鈕，獨立觸發) 的執行緒進入點。
        只負責：呼叫共用的 _execute_rapid_click() 執行一次 -> 處理例外 -> 收尾。
        跟 _playback_worker 是同樣的分工模式：核心邏輯共用，這裡只管背景執行緒的生命週期。
        """
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
            self._add_execution_log(
                "error", f"快速連按失敗：{type(error).__name__}: {error}"
            )

        finally:
            self.is_playing = False

    def _execute_rapid_click(self, rate, click_count, button_type, deadline=None):
        """
        快速連按的核心邏輯，供兩個地方共用：
            1. _rapid_click_worker —— 獨立觸發（F11 / 按鈕），立刻在目前滑鼠位置執行
            2. _run_trajectory_loop —— 事件清單裡的「快速連按事件」，
               在清單跑到這一筆時，於當下滑鼠位置執行

        原理很單純：不呼叫 pyautogui.moveTo()（滑鼠完全不動），
        只是照著「每秒 rate 次」換算出的間隔時間，重複呼叫 pyautogui.click()。

        例如 rate=10（每秒 10 次），換算下來每次點擊間隔 1/10 = 0.1 秒。
        這個間隔是「兩次點擊之間」的休息時間，不是點擊本身花的時間——
        點擊動作（按下+放開）本身非常快，可以忽略不計。

        注意：實際能達到的點擊速率仍然受限於：
            1. Windows 處理輸入事件的速度
            2. 目標應用程式讀取/回應點擊事件的速度
        設定 50 次/秒不代表目標程式一定來得及反應每一次點擊。

        回傳值：
            "completed" —— click_count 次全部正常跑完
            "stopped"   —— 途中被使用者按下停止而中斷

        注意：這個函式本身「不」處理 pyautogui.FailSafeException，
             刻意讓例外往外拋出，交由呼叫端（_rapid_click_worker /
             _playback_worker / _multi_schedule_worker）統一處理，
             這跟 _run_trajectory_loop 是同樣的設計原則。
        """
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

            # 最後一次點擊完就不用再等了，直接結束
            if i < click_count:
                wait_result = self._interruptible_sleep(interval, deadline)
                if wait_result != "completed":
                    return wait_result

        return "completed"

    # ==================================================================
    # 排程模式（多任務佇列：設定特定時間 + 重複執行次數 + 間隔秒數）
    # ==================================================================
    @staticmethod
    def _parse_clock_time(time_text):
        """解析 HH:MM 或 HH:MM:SS，成功時回傳 datetime.time，失敗回傳 None。"""
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(time_text, fmt).time()
            except ValueError:
                continue
        return None

    def _parse_schedule_time(self, time_text):
        """
        共用的時間字串解析函式，回傳計算好的目標 datetime，
        或者在格式錯誤時回傳 None（呼叫端自行決定要不要跳出錯誤訊息）。

        依「冒號數量」判斷使用者填的是 HH:MM 還是 HH:MM:SS，兩種格式都接受；
        若解析出的時間已經過去，自動視為「明天的這個時間」。
        """
        target_time = self._parse_clock_time(time_text)
        if target_time is None:
            return None

        now = datetime.now()
        target_dt = datetime.combine(now.date(), target_time)
        if target_dt <= now:
            target_dt += timedelta(days=1)
        return target_dt

    def add_schedule_task(self):
        """
        按下「➕ 新增排程任務」時呼叫。
        只負責把目前設定的「時間 / 重複次數 / 間隔秒數」包成一筆任務，
        加進 self.scheduled_tasks 佇列並更新畫面清單——不會立刻開始倒數。
        """
        if self.is_playing:
            messagebox.showinfo("提示", "排程佇列執行中，無法新增任務，請先按停止")
            return

        time_text = self.entry_schedule_time.get().strip()
        target_dt = self._parse_schedule_time(time_text)
        if target_dt is None:
            messagebox.showerror(
                "錯誤", "時間格式錯誤，請輸入 HH:MM 或 HH:MM:SS，例如 14:30 或 14:30:00"
            )
            return

        try:
            repeat_count = int(self.spin_repeat.get())
            if repeat_count < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("錯誤", "重複執行次數必須是大於等於 1 的整數")
            return

        interval = self.scale_interval.get()  # 每次重複之間的間隔秒數（0.1 秒為單位）

        task = {"target_dt": target_dt, "repeat_count": repeat_count, "interval": interval}
        self.scheduled_tasks.append(task)
        self._refresh_schedule_listbox()

        self._set_status(
            f"已新增排程任務：{target_dt.strftime('%Y-%m-%d %H:%M:%S')}，"
            f"重複 {repeat_count} 次（間隔 {interval:.1f} 秒）"
        )
        self._autosave_workflow("新增排程任務")

    def _refresh_schedule_listbox(self):
        """
        依「執行時間」由早到晚排序後，重新畫出整個排程任務清單。
        任何新增/刪除/清空/執行完成的動作之後都要呼叫這個函式來同步畫面。
        """
        self.listbox_schedule.delete(0, tk.END)
        sorted_tasks = sorted(self.scheduled_tasks, key=lambda t: t["target_dt"])
        for i, task in enumerate(sorted_tasks, start=1):
            self.listbox_schedule.insert(
                tk.END,
                f"任務 {i}：{task['target_dt'].strftime('%Y-%m-%d %H:%M:%S')}，"
                f"完整流程 ×{task['repeat_count']}（間隔 {task['interval']:.1f} 秒）"
            )

    def delete_selected_task(self):
        """
        刪除使用者在排程佇列清單中選取的任務。
        """
        if self.is_playing:
            messagebox.showinfo("提示", "排程佇列執行中，無法刪除任務，請先按停止")
            return

        selection = self.listbox_schedule.curselection()
        if not selection:
            messagebox.showinfo("提示", "請先在排程任務清單中選取要刪除的任務")
            return

        # 清單顯示是「依時間排序過」的順序，所以要用同樣排序過的清單來對應被選到的任務，
        # 而不是直接用 self.scheduled_tasks 原本新增時的順序（兩者可能不一致）。
        sorted_tasks = sorted(self.scheduled_tasks, key=lambda t: t["target_dt"])
        target_task = sorted_tasks[selection[0]]
        self.scheduled_tasks.remove(target_task)
        self._refresh_schedule_listbox()
        self._set_status("已刪除選取的排程任務")
        self._autosave_workflow("刪除排程任務")

    def clear_all_tasks(self):
        """
        清空所有排程任務。
        """
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
        """
        按下「▶ 啟動排程佇列」時呼叫。
        開一個背景執行緒 (_multi_schedule_worker)，依序處理佇列裡的每一筆任務。
        """
        if self.is_playing:
            messagebox.showinfo("提示", "目前已有播放或排程正在進行中，請先停止後再啟動")
            return

        if not self.scheduled_tasks:
            messagebox.showwarning("提示", "排程佇列是空的，請先新增至少一筆排程任務")
            return

        if not self.recorded_points:
            messagebox.showwarning("提示", "事件清單是空的，請先新增至少一個事件")
            return

        self.is_playing = True
        self.is_scheduling = True
        self.stop_requested = False

        self._set_status(f"排程佇列已啟動，共有 {len(self.scheduled_tasks)} 個任務等待執行")

        # 先在 Tkinter 主執行緒讀取設定，再交給背景執行緒使用。
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
        """
        排程佇列的背景執行緒進入點。
        每一輪都從目前佇列中挑出「執行時間最早」的一筆任務來處理，分兩階段：
            階段一：倒數等待，直到系統時間到達該任務的 target_dt
            階段二：重複呼叫 _run_trajectory_loop() repeat_count 次，每次之間停留 interval 秒
        該任務處理完（或被中止）後就從佇列移除，接著繼續處理下一筆，
        直到佇列清空，或使用者按下停止為止。

        全程都會頻繁檢查 self.stop_requested，讓使用者隨時可以用
        「⏹ 停止 (F10)」中斷倒數或執行。
        """
        try:
            while self.scheduled_tasks:
                if self.stop_requested:
                    self._set_status("排程佇列已被使用者取消")
                    return

                # 每一輪都重新挑選「執行時間最早」的任務，
                # 這樣就算執行過程中前面任務拖得比較久，也一定會照時間順序處理。
                task = min(self.scheduled_tasks, key=lambda t: t["target_dt"])
                target_dt = task["target_dt"]
                repeat_count = task["repeat_count"]
                interval = task["interval"]

                # ---------------- 階段一：倒數等待 ----------------
                while True:
                    if self.stop_requested:
                        self._set_status("排程佇列已被使用者取消")
                        return

                    now = datetime.now()
                    remaining = (target_dt - now).total_seconds()
                    if remaining <= 3:
                        # 提前三秒進入安全倒數，讓實際輸入動作盡量貼齊排定時間。
                        break

                    total_seconds = int(remaining)
                    hours, rem = divmod(total_seconds, 3600)
                    minutes, seconds = divmod(rem, 60)
                    self._set_status(
                        f"排程等待中，下一個任務將於 {target_dt.strftime('%H:%M:%S')} 執行"
                        f"（倒數 {hours:02d}:{minutes:02d}:{seconds:02d}，"
                        f"佇列剩餘 {len(self.scheduled_tasks)} 個任務）"
                    )
                    wait_result = self._interruptible_sleep(
                        min(1.0, max(0.0, remaining - 3.0))
                    )
                    if wait_result == "stopped":
                        self._set_status("排程佇列已被使用者取消")
                        return

                # ---------------- 階段二：重複執行 repeat_count 次 ----------------
                task_stopped = False
                self._add_execution_log(
                    "info",
                    f"排程任務開始：{target_dt.strftime('%Y-%m-%d %H:%M:%S')}，共 {repeat_count} 輪",
                )
                for rep in range(1, repeat_count + 1):
                    if self.stop_requested:
                        self._set_status("排程佇列已被使用者中止")
                        task_stopped = True
                        break

                    countdown_result = self._run_start_countdown(
                        f"排程第 {rep}/{repeat_count} 輪"
                    )
                    if countdown_result != "completed":
                        task_stopped = True
                        break

                    self._set_status(f"執行任務中：第 {rep}/{repeat_count} 次")
                    result = self._run_trajectory_loop(speed, click_delay, wait_time)

                    if result in {"stopped", "failed"}:
                        task_stopped = True
                        break

                    self._add_execution_log(
                        "success", f"排程任務第 {rep}/{repeat_count} 輪完成"
                    )

                    if rep < repeat_count and interval > 0:
                        wait_result = self._interruptible_sleep(interval)
                        if wait_result == "stopped":
                            self._set_status("排程佇列已被使用者中止")
                            task_stopped = True
                            break

                if task_stopped:
                    return  # 使用者中止，整個佇列都不再繼續

                # 這筆任務正常執行完成，立刻從佇列移除（同步進行，不能用 root.after 排隊）。
                #
                # 重要：這裡故意不用 self.root.after(0, self._remove_completed_task, task)——
                # after() 只是把移除動作排進主執行緒的事件佇列，不會馬上執行。
                # 如果這裡讓背景執行緒不等待就直接跳回 while 迴圈重新挑任務，
                # 主執行緒可能還沒處理完那個排隊的移除動作，
                # 導致這一筆「其實已經執行完」的任務還留在 self.scheduled_tasks 裡，
                # 又被 min() 挑出來重複執行一次——這就是排程會卡住重複執行的原因。
                #
                # 修法：list.remove() 這種單純的串列操作在 CPython 裡受 GIL 保護，
                # 直接在背景執行緒同步執行是安全的；只有「更新 Listbox 畫面」這種
                # 真正牽涉 Tkinter 元件的動作，才需要透過 root.after() 排回主執行緒。
                if task in self.scheduled_tasks:
                    self.scheduled_tasks.remove(task)
                self.root.after(0, self._refresh_schedule_listbox)
                self.root.after(0, self._autosave_workflow, "排程任務完成")
                self._add_execution_log(
                    "success", f"排程任務已完成並移出佇列：{target_dt.strftime('%H:%M:%S')}"
                )

            self._set_status("所有排程任務皆已完成 ✅")
            self._add_execution_log("success", "排程佇列全部完成")

        except pyautogui.FailSafeException:
            # 排程執行期間若觸發 FAILSAFE，整個佇列直接中止，不會再繼續下一筆任務
            self._set_status("⚠ 觸發 FAILSAFE 安全機制！排程佇列已立即中止")
            self._add_execution_log("error", "排程佇列觸發 FAILSAFE，已立即中止")

        except Exception as error:
            self._set_status(f"排程佇列發生未預期錯誤：{error}")
            self._add_execution_log(
                "error", f"排程佇列錯誤：{type(error).__name__}: {error}"
            )

        finally:
            self.is_playing = False
            self.is_scheduling = False

    # ==================================================================
    # 開始～結束時段循環排程
    # ==================================================================
    @staticmethod
    def _calculate_window_bounds(start_time, end_time, now=None):
        """
        將只有時、分、秒的開始/結束時間換算成實際日期時間。

        - 現在已位於有效時段內：使用目前這一段時窗。
        - 今日時段尚未開始：使用今天的時窗。
        - 今日時段已結束：使用明天的時窗。
        - end <= start：視為跨午夜，例如 22:00～06:00。
        """
        now = now or datetime.now()
        candidate_windows = []

        # 同時檢查昨天、今天、明天，才能正確辨識凌晨仍位於
        # 「昨天 22:00 ～ 今天 06:00」這類跨午夜時段內的情況。
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

        # 理論上前面的三天候選一定會含有下一個時窗；保留此分支作防禦性處理。
        start_dt = datetime.combine(now.date() + timedelta(days=1), start_time)
        end_dt = datetime.combine(now.date() + timedelta(days=1), end_time)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)
        return start_dt, end_dt

    def start_window_schedule(self):
        """驗證設定並啟動「開始～結束」時段循環排程。"""
        if self.is_playing:
            messagebox.showinfo("提示", "目前已有播放、排程或連按正在進行中，請先停止")
            return

        if not self.recorded_points:
            messagebox.showwarning("提示", "事件清單是空的，請先新增至少一個事件")
            return

        start_text = self.entry_window_start.get().strip()
        end_text = self.entry_window_end.get().strip()
        start_time = self._parse_clock_time(start_text)
        end_time = self._parse_clock_time(end_text)
        if start_time is None or end_time is None:
            messagebox.showerror(
                "錯誤",
                "開始與結束時間必須是 HH:MM 或 HH:MM:SS，例如 09:00、18:30:00"
            )
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

        # 若啟動時已位於有效時段內，可選擇立刻跑第一次；否則等待下一個
        # 與開始時間對齊的間隔點。預設勾選「立即執行」，操作上最直覺。
        if start_dt <= now < end_dt and self.window_run_now_var.get():
            first_run_dt = now
        elif now <= start_dt:
            first_run_dt = start_dt
        else:
            elapsed = (now - start_dt).total_seconds()
            next_step = int(elapsed // interval_seconds) + 1
            first_run_dt = start_dt + timedelta(seconds=next_step * interval_seconds)

        self.window_start_dt = start_dt
        self.window_end_dt = end_dt
        self.window_next_run = first_run_dt
        self.window_last_state = "等待中"
        self.is_playing = True
        self.is_window_scheduling = True
        self.stop_requested = False

        speed = self.scale_speed.get()
        click_delay = self.scale_click_delay.get()
        wait_time = self.scale_wait.get()

        self._set_status(
            f"時段循環排程已啟動：{start_dt.strftime('%Y-%m-%d %H:%M:%S')} ～ "
            f"{end_dt.strftime('%Y-%m-%d %H:%M:%S')}，每 {interval_minutes:g} 分鐘執行一次"
        )

        thread = threading.Thread(
            target=self._window_schedule_worker,
            args=(first_run_dt, end_dt, interval_seconds, speed, click_delay, wait_time),
            daemon=True
        )
        thread.start()

    def _window_schedule_worker(
        self, first_run_dt, end_dt, interval_seconds, speed, click_delay, wait_time
    ):
        """
        在有效時段內按固定間隔完整播放事件清單。

        下一輪以「每輪預定開始時間」計算，不會因流程耗時逐輪漂移；如果單次流程
        比間隔更久，會略過已錯過的觸發點，不會同時啟動多條重疊流程。
        """
        run_count = 0
        next_run = first_run_dt
        final_state = "已完成"

        try:
            self._add_execution_log(
                "info", f"時段循環排程開始，結束時間 {end_dt.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            while True:
                if self.stop_requested:
                    final_state = f"已由使用者停止（已執行 {run_count} 次）"
                    self._set_status("時段循環排程已被使用者取消")
                    return

                now = datetime.now()
                if now >= end_dt or next_run >= end_dt:
                    final_state = f"已到達結束時間（共執行 {run_count} 次）"
                    self._set_status(f"時段循環排程完成，共執行 {run_count} 次 ✅")
                    return

                self.window_next_run = next_run

                countdown_start = next_run - timedelta(seconds=3)
                if now < countdown_start:
                    wait_result = self._interruptible_sleep(
                        (countdown_start - now).total_seconds(), deadline=end_dt
                    )
                    if wait_result == "stopped":
                        final_state = f"已由使用者停止（已執行 {run_count} 次）"
                        self._set_status("時段循環排程已被使用者取消")
                        return
                    if wait_result == "deadline":
                        final_state = f"已到達結束時間（共執行 {run_count} 次）"
                        self._set_status(f"時段循環排程完成，共執行 {run_count} 次 ✅")
                        return

                if datetime.now() >= end_dt:
                    final_state = f"已到達結束時間（共執行 {run_count} 次）"
                    self._set_status(f"時段循環排程完成，共執行 {run_count} 次 ✅")
                    return

                self.window_next_run = None
                countdown_result = self._run_start_countdown(
                    f"時段排程第 {run_count + 1} 輪", deadline=end_dt
                )
                if countdown_result == "stopped":
                    final_state = f"已由使用者停止（已執行 {run_count} 次）"
                    return
                if countdown_result == "deadline":
                    final_state = f"倒數期間到達結束時間（共執行 {run_count} 次）"
                    return

                self._set_status(f"時段循環排程：正在執行第 {run_count + 1} 次完整事件清單")
                result = self._run_trajectory_loop(
                    speed, click_delay, wait_time, deadline=end_dt
                )

                if result == "stopped":
                    final_state = f"已由使用者停止（已執行 {run_count} 次）"
                    return
                if result == "deadline":
                    final_state = f"結束時間到，已中止當次流程（先前完成 {run_count} 次）"
                    self._set_status("已到達結束時間，時段循環排程已停止")
                    return
                if result == "failed":
                    final_state = f"事件失敗，已停止（先前完成 {run_count} 次）"
                    self._set_status("時段循環排程因事件失敗而停止")
                    return

                run_count += 1
                self._add_execution_log(
                    "success", f"時段循環排程第 {run_count} 輪完成"
                )

                # 從原預定時間推算下一輪；若流程太久錯過一個以上間隔，直接略過。
                next_run += timedelta(seconds=interval_seconds)
                now = datetime.now()
                while next_run <= now:
                    next_run += timedelta(seconds=interval_seconds)

        except pyautogui.FailSafeException:
            final_state = f"FAILSAFE 已中止（完成 {run_count} 次）"
            self._set_status("⚠ 觸發 FAILSAFE 安全機制！時段循環排程已立即中止")
            self._add_execution_log("error", "時段循環排程觸發 FAILSAFE，已立即中止")

        except Exception as error:
            final_state = f"未預期錯誤（完成 {run_count} 次）"
            self._set_status(f"時段循環排程發生錯誤：{error}")
            self._add_execution_log(
                "error", f"時段循環排程錯誤：{type(error).__name__}: {error}"
            )

        finally:
            self.window_last_state = final_state
            self.window_next_run = None
            self.is_window_scheduling = False
            self.is_playing = False
            level = "success" if "到達結束時間" in final_state or final_state == "已完成" else "warning"
            self._add_execution_log("info" if level == "success" else level, f"時段循環排程結束：{final_state}")

    # ==================================================================
    # 全域快捷鍵（F8 記錄 / F9 播放 / F10 停止 / F11 快速連按）
    # ==================================================================
    def _start_hotkey_listener(self):
        """
        啟動一個「背景執行緒」，用 pynput 監聽全系統的鍵盤事件。

        原理說明：
            pynput.keyboard.Listener 在 Windows 上內部會呼叫
            ctypes -> user32.dll -> SetWindowsHookEx(WH_KEYBOARD_LL, callback, ...)，
            這是一種「低階鍵盤鉤子」，會安裝在系統的鍵盤事件處理流程中。
            每當任何按鍵被按下（不管焦點在哪個視窗），
            Windows 都會先呼叫我們註冊的 callback 函式，
            所以即使我們的 Tkinter 視窗完全沒有焦點，也能收到按鍵事件。

            這跟 Tkinter 的 root.bind("<F8>", ...) 完全不同 ——
            bind() 只有在「這個 Tkinter 視窗」是目前作業系統的焦點視窗時才有效。
        """
        # Listener 建立時傳入 on_press callback，每次「按下」按鍵都會呼叫一次
        self.hotkey_listener = pynput_keyboard.Listener(on_press=self._on_key_press)

        # Listener 內部本身就是一個獨立的背景執行緒在運作（daemon 執行緒），
        # 呼叫 .start() 之後主程式會立刻繼續往下執行，不會被卡住。
        self.hotkey_listener.start()

    def _on_key_press(self, key):
        """
        每當偵測到「任何一個按鍵被按下」時，pynput 都會呼叫這個函式。
        參數 key 是 pynput 定義的按鍵物件，例如 Key.f8、Key.f9 等特殊鍵，
        一般文字鍵則會是 KeyCode 物件（例如按 'a' 會是 KeyCode(char='a')）。

        重要：這個函式是在 pynput 自己的背景執行緒中被呼叫，
             而 Tkinter 的元件（Label、Button...）只能在「主執行緒」安全地更新，
             所以這裡不能直接呼叫 self.record_point() 等函式，
             而是要透過 self.root.after(0, func) 把工作「排程」回主執行緒執行。
             這是所有 GUI 框架（Tkinter、Qt、WinForms...）共通的「執行緒安全」規則。
        """
        try:
            if key == pynput_keyboard.Key.f8:
                # F8：記錄目前滑鼠座標
                self.root.after(0, self.record_point)

            elif key == pynput_keyboard.Key.f9:
                # F9：開始播放
                self.root.after(0, self.start_playback)

            elif key == pynput_keyboard.Key.f10:
                # F10：停止播放
                self.root.after(0, self.stop_playback)

            elif key == pynput_keyboard.Key.f11:
                # F11：在目前滑鼠位置開始快速連按
                self.root.after(0, self.start_rapid_click)

        except Exception:
            # 這裡刻意攔截所有例外（而不是只攔截 RuntimeError）。
            #
            # 原因：這個函式是 pynput 背景監聽執行緒的 callback，
            # 如果裡面丟出「任何」沒被攔截的例外，會導致 pynput 的監聽執行緒
            # 整個當掉、停止運作——後果是 F8/F9/F10/F11 全部一起失效，
            # 而且畫面上不會有任何錯誤訊息，使用者只會覺得「快捷鍵怎麼忽然不能用了」。
            #
            # 只攔截 RuntimeError（例如視窗關閉時 self.root 已被銷毀）不夠周全，
            # 任何其他型別的例外（例如某個元件狀態不如預期）都可能讓監聽提前終止。
            # 這裡選擇「寧可吞掉這次按鍵、保住監聽執行緒繼續運作」，
            # 也不要讓一次例外拖垮之後所有的快捷鍵功能。
            pass

    def _on_close(self):
        """
        使用者關閉視窗時呼叫。
        必須先停止 pynput 的鍵盤監聽（移除系統鉤子），
        否則背景執行緒會一直存在，導致程式無法正常結束。
        """
        try:
            self._autosave_workflow("關閉程式")
        except Exception:
            pass

        self.stop_requested = True   # 保險起見，順便通知播放迴圈中止

        if hasattr(self, "hotkey_listener"):
            self.hotkey_listener.stop()  # 停止 pynput 監聽、移除鍵盤鉤子

        self.root.destroy()  # 關閉 Tkinter 視窗，結束 mainloop()

    # ==================================================================
    # 小工具
    # ==================================================================
    @staticmethod
    def _resolve_app_data_dir():
        """取得可寫入的應用程式資料夾，Windows 優先使用 LOCALAPPDATA。"""
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
        """驗證並升級從 JSON 載入的事件，避免損壞資料讓播放執行緒崩潰。"""
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
            "action_delay": self._normalize_nonnegative_number(
                raw_event.get("action_delay"), 0.5
            ),
            "repeat_count": self._normalize_positive_integer(
                raw_event.get("repeat_count"), 1
            ),
            "repeat_interval": self._normalize_nonnegative_number(
                raw_event.get("repeat_interval"), 1.0
            ),
            "after_wait": self._normalize_nonnegative_number(
                raw_event.get("after_wait"), 0.5
            ),
            "failure_policy": failure_policy,
            "retry_count": self._normalize_positive_integer(
                raw_event.get("retry_count"), 2
            ),
        }

        if event_type == "point":
            try:
                x = int(raw_event["x"])
                y = int(raw_event["y"])
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
                "move_duration": self._normalize_nonnegative_number(
                    raw_event.get("move_duration"), 0.5
                ),
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

    def _create_workflow_payload(self):
        """建立可寫入 JSON 的完整流程資料。"""
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
                "window_start": self.entry_window_start.get().strip(),
                "window_end": self.entry_window_end.get().strip(),
                "window_interval_minutes": self.spin_window_interval.get(),
                "window_run_now": bool(self.window_run_now_var.get()),
                "rapid_rate": self.scale_rapid_rate.get(),
                "rapid_count": self.spin_rapid_count.get(),
                "rapid_button": self.rapid_click_button_var.get(),
            }
        except (AttributeError, tk.TclError):
            pass

        return {
            "format": "MouseFlowStudio",
            "version": 3,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "events": copy.deepcopy(self.recorded_points),
            "scheduled_tasks": scheduled_tasks,
            "settings": settings,
        }

    @staticmethod
    def _write_json_atomic(path, payload):
        """先寫入暫存檔再取代正式檔，避免程式中斷造成半份 JSON。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)

    def _apply_workflow_data(self, data):
        """將已驗證的 JSON 工作流程套用到介面。"""
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
            if settings.get("window_start"):
                self.entry_window_start.delete(0, "end")
                self.entry_window_start.insert(0, settings["window_start"])
            if settings.get("window_end"):
                self.entry_window_end.delete(0, "end")
                self.entry_window_end.insert(0, settings["window_end"])
            if settings.get("window_interval_minutes") is not None:
                self._replace_spinbox_value(
                    self.spin_window_interval, settings["window_interval_minutes"]
                )
            if settings.get("window_run_now") is not None:
                self.window_run_now_var.set(bool(settings["window_run_now"]))
            if settings.get("rapid_rate") is not None:
                self.scale_rapid_rate.set(settings["rapid_rate"])
            if settings.get("rapid_count") is not None:
                self._replace_spinbox_value(self.spin_rapid_count, settings["rapid_count"])
            if settings.get("rapid_button") in {"left", "right"}:
                self.rapid_click_button_var.set(settings["rapid_button"])

        return len(normalized_events), len(scheduled_tasks)

    def _load_workflow_path(self, path, show_message=True):
        path = Path(path)
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.autosave_suspended = True
        try:
            event_count, task_count = self._apply_workflow_data(data)
        finally:
            self.autosave_suspended = False

        self._set_status(f"已載入流程：{event_count} 個事件、{task_count} 個排程任務")
        self._add_execution_log(
            "info", f"載入流程檔案：{path.name}（{event_count} 個事件）"
        )
        self._autosave_workflow("載入流程")
        if show_message:
            messagebox.showinfo(
                "載入完成", f"已載入 {event_count} 個事件與 {task_count} 個排程任務。"
            )

    def save_workflow(self):
        """讓使用者選擇位置，將完整流程另存為 JSON。"""
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
        """從使用者選擇的 JSON 載入完整流程。"""
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
        """任何流程異動後自動覆寫備份檔。"""
        if self.autosave_suspended:
            return
        try:
            self._write_json_atomic(self.autosave_path, self._create_workflow_payload())
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.label_autosave_status.config(
                text=f"自動備份：{timestamp} 已完成（{reason}）"
            )
        except (OSError, TypeError, ValueError, tk.TclError) as error:
            try:
                self.label_autosave_status.config(text=f"自動備份失敗：{error}")
            except tk.TclError:
                pass

    def _load_autosave_on_startup(self):
        """啟動時自動恢復上次工作內容。"""
        if not self.autosave_path.exists():
            return
        try:
            self._load_workflow_path(self.autosave_path, show_message=False)
            self.label_autosave_status.config(text="自動備份：已恢復上次工作內容")
        except (OSError, json.JSONDecodeError, ValueError) as error:
            self.label_autosave_status.config(text=f"自動備份無法恢復：{error}")
            self._add_execution_log("error", f"自動備份恢復失敗：{error}")

    def restore_autosave(self):
        """手動還原自動備份檔。"""
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
        """啟動時讀取最近的執行紀錄，讓錯誤歷程不會因關閉程式消失。"""
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
        """由新到舊更新畫面上的執行紀錄。"""
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
                self.listbox_logs.itemconfig(
                    tk.END, fg=level_colors.get(level, level_colors["info"])
                )
        except (AttributeError, tk.TclError):
            pass

    def _add_execution_log(self, level, message):
        """新增一筆記憶體與 JSONL 執行紀錄，並安全地更新 Tkinter 清單。"""
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
                # 寫入歷程失敗不能反過來中斷滑鼠流程。
                pass

        if threading.current_thread() is threading.main_thread():
            self._refresh_execution_log_listbox()
        else:
            try:
                self.root.after(0, self._refresh_execution_log_listbox)
            except (AttributeError, RuntimeError, tk.TclError):
                pass

    def export_execution_logs(self):
        """將目前保存的歷程紀錄匯出為易讀 JSON。"""
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
        """經確認後清除畫面與本機保存的全部執行紀錄。"""
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
        """
        統一更新畫面底部的狀態文字。
        背景播放執行緒不可直接操作 Tkinter 元件，因此統一排回主執行緒更新。
        """
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
                # 視窗正好已被關閉時，不需要再更新狀態。
                pass

        if threading.current_thread() is threading.main_thread():
            update_label()
        else:
            try:
                self.root.after(0, update_label)
            except (RuntimeError, tk.TclError):
                pass


# ======================================================================
# 程式進入點
# ======================================================================
if __name__ == "__main__":
    root = tk.Tk()                 # 建立 Tkinter 主視窗物件
    app = MouseCoordinateLab(root) # 建立我們的應用程式實例，把主視窗傳進去
    root.mainloop()                # 進入 Tkinter 的事件迴圈，開始監聽使用者操作（按鈕點擊等）
                                    # 這行會一直阻塞，直到使用者關閉視窗為止
