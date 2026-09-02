"""終端機輸出面板：main.py／獨立腳本工具共用的 stdout/stderr 顯示 + stdin 輸入。

從 settings_window.py 搬出來的獨立元件——這塊本來就跟表單狀態（FIELD_SCHEMA／
self._field_widgets）沒有耦合，只需要：
  - `window`：拿它的 `.after()`／`.bind_all()`／`.winfo_height()`／
    `.winfo_screenheight()`／`.update_idletasks()`，以及它的 `_header_frame`／
    `_bottom_bar_frame`（算面板該貼齊哪個 y 座標、最高能拉多高）跟字型物件
    （`_font_label_bold`／`_font_hint`）；
  - `container`：呼叫端已經用 place() 定位過的外層 Frame，這裡只管裡面的內容。

跟子行程（main.py／獨立腳本）的關聯，透過兩個窄接口完成，不直接持有
subprocess.Popen 物件：
  - `start_log_reader(process, label)`：process_manager.py 啟動子行程成功後呼叫，
    開一條背景執行緒把 stdout 逐 byte（非逐行，見該函式說明）讀進來顯示；`label`
    只在子行程結束時印「— {label} 行程已結束 —」用。
  - `likely_waiting_for_input()` / `seconds_idle()`：讓呼叫端（settings_window.py
    的 `_update_process_buttons_state`）判斷子行程是否疑似卡在 input() 之類的
    互動輸入等待中，用於在狀態列文字加上提醒。
  - `set_stdin_handler(fn)`：process_manager.py 把它自己的 `send_stdin` 方法
    傳進來，使用者在輸入列按 Enter/傳送時，這個 class 只呼叫 `fn(text)`，
    不自己碰 subprocess 物件。
"""

import codecs
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from settings_gui.style import (
    BTN_PRIMARY_BG,
    BTN_PRIMARY_ACTIVE,
    BTN_SECONDARY_BG,
    BTN_SECONDARY_ACTIVE,
    COLOR_CONSOLE_BG,
    COLOR_CONSOLE_FG,
    COLOR_CONSOLE_GRIP_BG,
    COLOR_CONSOLE_MUTED_FG,
    COLOR_HEADER_BG,
    COLOR_HEADER_FG,
    CONSOLE_COLLAPSED_HEIGHT,
    CONSOLE_DEFAULT_FONT_SIZE,
    CONSOLE_DEFAULT_HEIGHT,
    CONSOLE_DEFAULT_HEIGHT_FRACTION,
    CONSOLE_FONT_FAMILY,
    CONSOLE_MAX_FONT_SIZE,
    CONSOLE_MAX_HEIGHT_RESERVE,
    CONSOLE_MIN_FONT_SIZE,
    CONSOLE_MIN_HEIGHT,
    SPACE_SM,
    _styled_button,
)

# log_queue 裡代表「某一條 log reader 讀到 EOF（行程結束、pipe 關閉）」的標記，
# 跟著一個「代次序號」一起放進 queue。_drain_log_queue 只認「目前這一代」的 EOF，
# 前一支行程殘留在 queue 裡、還沒被讀掉的 EOF 會被靜默丟棄（否則會在下一支行程剛
# 啟動時憑空冒出一行「— … 行程已結束 —」）。
_EOF = object()


class ConsolePanel:
    def __init__(self, window, container):
        self.window = window
        self.container = container
        self._collapsed = False
        self._expanded_height = CONSOLE_DEFAULT_HEIGHT
        # 終端機面板每次改變高度/位置（拖拉／收合展開／初始套用比例）都要通知外部
        # ——目前唯一的用途是讓分頁右欄「欄位說明」面板那條浮動橫向拉桿（見
        # settings_gui/tab_docs_panel.py、settings_window.py 的
        # _reposition_active_docs_hscroll）跟著重新貼齊「終端機面板正上方」，不然
        # 終端機被拖大/拖小之後，那條拉桿的位置就會跟終端機新的邊界對不上。
        self._on_resize = None

        self.log_queue = queue.Queue()
        self._log_reader_thread = None
        self._log_line_count = 0
        self._reader_label = None  # 目前這條 log reader 對應哪個行程的顯示名稱
        self._reader_gen = 0       # log reader 代次：切行程時 +1，用來忽略舊行程殘留的 EOF 標記
        self._drain_job = None     # _drain_log_queue 的 after id，關窗前 stop() 取消

        # 供「疑似卡在 input() 等待輸入」判斷用（見 likely_waiting_for_input()）：
        # 記錄最後一次收到輸出的時間，以及那次輸出是否以換行結尾。
        self._last_output_monotonic = None
        self._last_chunk_ends_newline = True

        self._stdin_handler = None  # process_manager.py 事後用 set_stdin_handler() 掛上來

        self._drag_start_y = None
        self._drag_start_height = None

        self._build()

    # ── 版面 ─────────────────────────────────────────────────────────

    def _build(self):
        """終端機面板可拖拉調整高度（頂部把手）、也可用箭頭按鈕一鍵內縮成一條水平線
        （收合時只留標題列，箭頭仍看得到，再點一次箭頭恢復收合前的高度）。"""
        parent = self.container

        # 拖拉把手：一條比面板底色略亮的細線，滑鼠移上去會變成上下箭頭游標，
        # 拖動時即時調整面板高度——收合時這條把手會跟著隱藏（沒有東西可以拖）。
        grip = tk.Frame(parent, bg=COLOR_CONSOLE_GRIP_BG, height=5, cursor="sb_v_double_arrow")
        grip.pack(fill="x", side="top")
        grip.bind("<ButtonPress-1>", self._on_drag_start)
        grip.bind("<B1-Motion>", self._on_drag_motion)
        self._grip = grip

        header = tk.Frame(parent, bg=COLOR_HEADER_BG, cursor="sb_v_double_arrow")
        header.pack(fill="x", side="top")
        header.bind("<ButtonPress-1>", self._on_drag_start)
        header.bind("<B1-Motion>", self._on_drag_motion)
        self._header = header

        self._toggle_btn = tk.Button(
            header, text="▼", command=self.toggle_collapse, bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG,
            activebackground=COLOR_HEADER_BG, activeforeground=COLOR_HEADER_FG, relief="flat", bd=0,
            font=self.window._font_label_bold, cursor="hand2", padx=8,
        )
        self._toggle_btn.pack(side="left")
        tk.Label(
            header, text="🖥️ 終端機輸出（main.py／獨立腳本工具共用；可拖拉頂端調整高度）",
            bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG, font=self.window._font_label_bold, anchor="w",
        ).pack(side="left", padx=(0, 10), pady=6)

        # 工具列 + Text 輸出區包成一個子容器，收合時整包 pack_forget()，
        # 展開時整包用 before/after 對齊 grip／header 之間的正確順序重新插回。
        body = tk.Frame(parent, bg=COLOR_CONSOLE_BG)
        body.pack(fill="both", expand=True, side="top")
        self._body = body

        toolbar = tk.Frame(body, bg=COLOR_CONSOLE_BG)
        toolbar.pack(fill="x", padx=8, pady=(6, 4))
        self.autoscroll_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            toolbar, text="自動捲動", variable=self.autoscroll_var, bg=COLOR_CONSOLE_BG,
            fg=COLOR_CONSOLE_FG, selectcolor=COLOR_CONSOLE_BG, activebackground=COLOR_CONSOLE_BG,
            activeforeground=COLOR_CONSOLE_FG, font=self.window._font_hint, bd=0, highlightthickness=0,
        ).pack(side="left")
        _styled_button(
            toolbar, "清除", self.clear, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
            font=self.window._font_hint, compact=True,
        ).pack(side="right")
        _styled_button(
            toolbar, "另存為檔案...", self.save_to_file, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
            font=self.window._font_hint, compact=True,
        ).pack(side="right", padx=(0, SPACE_SM))

        text_frame = tk.Frame(body, bg=COLOR_CONSOLE_BG)
        text_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.font_size = CONSOLE_DEFAULT_FONT_SIZE
        # height=1：Text 元件不指定 height 時預設請求 24 行的高度，字級越大這個「預設
        # 請求」就越誇張（20pt 時逼近 780px），會讓 pack 版面協商誤以為這個元件的最小
        # 需求就是那麼高，擠壓到排在它後面的「輸入」列（見 on_send_stdin 那一列）幾乎
        # 沒有空間、被壓成 1px 看不見。設成 height=1 只是蓋掉這個預設請求值，實際渲染
        # 時仍會被 pack(fill="both", expand=True) 撐滿可用空間，跟先前修 Canvas
        # 預設地板過大是同一類問題、同一種修法。
        self.text = tk.Text(
            text_frame, bg=COLOR_CONSOLE_BG, fg=COLOR_CONSOLE_FG, insertbackground=COLOR_CONSOLE_FG,
            font=(CONSOLE_FONT_FAMILY, self.font_size), wrap="word", state="disabled",
            relief="flat", padx=6, pady=4, height=1,
        )
        console_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=console_scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        console_scroll.pack(side="right", fill="y")
        self.text.tag_configure("muted", foreground=COLOR_CONSOLE_MUTED_FG)

        # Ctrl+/- 縮放終端機文字大小，原理同 VS Code：用 bind_all（不是只綁在
        # self.text 上）讓快捷鍵不管目前焦點在視窗裡哪個元件都會生效，不用
        # 特地點進輸出區才能用。"+" 鍵在大多數鍵盤佈局要按 Shift，實際收到的事件是
        # <Control-plus>；沒按 Shift 直接按實體 "=" 鍵送出的是 <Control-equal>，
        # 兩種都綁，數字鍵盤的 +/- 也一併綁（<Control-KP_Add>/<Control-KP_Subtract>）。
        # Ctrl+0／Ctrl+數字鍵盤0 重設回預設字級，同樣比照 VS Code 的習慣。
        for seq in ("<Control-plus>", "<Control-equal>", "<Control-KP_Add>"):
            self.window.bind_all(seq, lambda _e: self._adjust_font_size(1))
        for seq in ("<Control-minus>", "<Control-KP_Subtract>"):
            self.window.bind_all(seq, lambda _e: self._adjust_font_size(-1))
        for seq in ("<Control-0>", "<Control-KP_0>"):
            self.window.bind_all(seq, lambda _e: self.reset_font_size())

        # 輸入列：main.py／獨立腳本工具如果跑到 input() 之類需要互動輸入的地方
        # （例如某些工具腳本開場問「請選擇執行模式 1/2」），行程會停在那裡等，光看
        # 輸出面板完全看不出來、也沒地方能回應——這一列讓使用者能直接把文字送進
        # 子行程的 stdin。沒有行程在跑時停用，避免誤按送出目標不存在的輸入。
        input_row = tk.Frame(body, bg=COLOR_CONSOLE_BG)
        input_row.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(
            input_row, text="輸入：", bg=COLOR_CONSOLE_BG, fg=COLOR_CONSOLE_FG,
            font=("Consolas", 10, "bold"),
        ).pack(side="left")
        self.stdin_var = tk.StringVar()
        self.stdin_entry = tk.Entry(
            input_row, textvariable=self.stdin_var, font=("Consolas", 10),
            bg="#2a2d31", fg=COLOR_CONSOLE_FG, insertbackground=COLOR_CONSOLE_FG,
            relief="flat", disabledbackground="#242628", state="disabled",
        )
        self.stdin_entry.pack(side="left", fill="x", expand=True, padx=(6, 6), ipady=3)
        self.stdin_entry.bind("<Return>", lambda _e: self.on_send_stdin())
        # 全域 Ctrl+F（settings_window bind_all）會把焦點搶去「獨立腳本工具」下拉；
        # 在這個輸入框裡打字時攔下來，不然打到 f 就跳走。
        self.stdin_entry.bind("<Control-f>", lambda _e: "break")
        self.send_stdin_btn = _styled_button(
            input_row, "傳送", self.on_send_stdin, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE,
            font=self.window._font_hint, compact=True,
        )
        self.send_stdin_btn.config(state="disabled")
        self.send_stdin_btn.pack(side="left")
        tk.Label(
            input_row, text="（沒有行程在跑時停用；main.py／腳本停在等待輸入時，在這裡打字後按 Enter 或「傳送」）",
            bg=COLOR_CONSOLE_BG, fg=COLOR_CONSOLE_MUTED_FG, font=self.window._font_hint,
        ).pack(side="left", padx=(8, 0))

        self.append(
            "（尚未啟動任何程式；按上方「▶ 啟動 main.py」或選好腳本後按「▶ 執行所選腳本」，輸出會即時顯示在這裡）\n",
            tag="muted",
        )
        self._drain_job = self.window.after(80, self._drain_log_queue)

    # ── 位置／高度 ───────────────────────────────────────────────────

    def place(self, height):
        """終端機面板固定用 place() 貼齊視窗左右邊界，高度由參數決定——這一個函式是
        所有「改變終端機高度」操作（拖拉／收合展開／初始套用比例）共用的唯一進出口，
        確保每次呼叫方式都一致，也確保面板永遠疊在其他內容之上（每次呼叫都 lift()
        一次，不受其他元件建立順序影響）。

        底部不是貼齊視窗最底端（那樣會蓋到「儲存設定」那排按鈕，按不到），而是貼齊
        底部按鈕列的「上緣」——用 window._bottom_bar_frame.winfo_y()（它在 window 座標系
        下的 y 位置，因為底部按鈕列本來就是 window 的直接子元件）當作面板下緣的錨點，
        往上量 height 這麼高。window._build_bottom_bar() 還沒建立、拿不到這個參照時，
        先退回貼齊視窗底部（僅發生在建構過程最初那一次呼叫，稍後 _bottom_bar_frame
        建好後的下一次呼叫就會修正到正確位置）。"""
        bottom_bar = getattr(self.window, "_bottom_bar_frame", None)
        floor_y = bottom_bar.winfo_y() if bottom_bar is not None else self.window.winfo_height()
        self.container.place(x=0, y=floor_y, anchor="sw", relwidth=1.0, height=height)
        self.container.lift()
        if self._on_resize is not None:
            self._on_resize()

    def set_on_resize(self, callback):
        """註冊一個「終端機面板位置/高度改變後」要執行的回呼，見 __init__ 裡
        `_on_resize` 的說明。"""
        self._on_resize = callback

    def stop(self):
        """視窗關閉前呼叫：停掉 log 汲取迴圈，避免關窗後 _drain_log_queue 再對
        已銷毀的 Text widget 操作而丟 TclError。"""
        if self._drain_job is not None:
            try:
                self.window.after_cancel(self._drain_job)
            except tk.TclError:
                pass
            self._drain_job = None

    def lift(self):
        self.container.lift()

    def _window_height(self):
        """視窗目前的真實高度；還沒完成第一次幾何配置（量到 <=1）時退回用螢幕高度頂著。"""
        win_h = self.window.winfo_height()
        return win_h if win_h > 1 else self.window.winfo_screenheight()

    def max_height(self):
        """拖拉能撐到的最大高度：貼齊視窗底部往上量，最多到「標題列」下緣為止——
        終端機面板是浮動疊層（place()，見 settings_window.py 的 _build_middle_area），
        不是跟表單搶版面，所以能一路蓋過流程列／獨立腳本工具列／資訊列／分頁按鈕列／
        表單內容／底部按鈕列，唯獨標題本身（貓咪監測系統 — 設定管理）永遠留在最上面
        看得到。header 高度在視窗還沒完成第一次幾何配置前可能量到 0，此時退回用
        CONSOLE_MAX_HEIGHT_RESERVE 這個粗估值頂著，避免除出負數或撞到下限。"""
        win_h = self._window_height()
        header_frame = getattr(self.window, "_header_frame", None)
        header_h = header_frame.winfo_height() if header_frame is not None else 0
        reserve = header_h if header_h > 0 else CONSOLE_MAX_HEIGHT_RESERVE
        return max(CONSOLE_MIN_HEIGHT, win_h - reserve)

    def apply_sane_default_height(self):
        """視窗剛建好、量得到真實幾何尺寸時呼叫一次：先把「展開時要多高」定案成視窗
        高度的 CONSOLE_DEFAULT_HEIGHT_FRACTION（預設 75%），但畫面上一開始是收合
        狀態——只留標題那一條水平線，不會一開機就蓋住表單/按鈕，使用者要看終端機
        輸出時自己點箭頭展開，展開後直接是這個算好的 75% 高度，不用手動拖。用
        「視窗高度的比例」而不是寫死的像素值，是因為不同螢幕解析度／系統 DPI 縮放
        下，視窗實際可用的垂直空間差異很大，比例才會等比縮放。"""
        win_h = self._window_height()
        header_frame = getattr(self.window, "_header_frame", None)
        if header_frame is None or header_frame.winfo_height() <= 0:
            return  # 幾何配置還不可信，維持建構時給的預設高度，不冒然套用
        target = round(win_h * CONSOLE_DEFAULT_HEIGHT_FRACTION)
        sane = max(CONSOLE_MIN_HEIGHT, min(target, self.max_height()))
        self.place(sane)
        self.window.update_idletasks()  # 讓下面 toggle_collapse() 收合時，能正確量到 sane 這個高度存起來
        if not self._collapsed:
            self.toggle_collapse()

    def _on_drag_start(self, event):
        if self._collapsed:
            return
        self._drag_start_y = event.y_root
        self._drag_start_height = self.container.winfo_height()

    def _on_drag_motion(self, event):
        if self._collapsed:
            return
        delta = self._drag_start_y - event.y_root  # 往上拖＝正值＝面板變高
        new_height = self._drag_start_height + delta
        new_height = max(CONSOLE_MIN_HEIGHT, min(new_height, self.max_height()))
        self.place(new_height)
        self._expanded_height = new_height

    def toggle_collapse(self):
        """收合：記住目前高度，只留 grip 上方那條 header 水平線（箭頭仍看得到）。
        展開：用 before=/after= 把 grip、body 依原本順序插回 header 上下兩側，
        並還原收合前的高度。"""
        self._collapsed = not self._collapsed
        if self._collapsed:
            self._expanded_height = self.container.winfo_height()
            self._grip.pack_forget()
            self._body.pack_forget()
            self.place(CONSOLE_COLLAPSED_HEIGHT)
            self._toggle_btn.config(text="▲")
        else:
            self._grip.pack(fill="x", side="top", before=self._header)
            self._body.pack(fill="both", expand=True, side="top")
            self.place(self._expanded_height)
            self._toggle_btn.config(text="▼")

    # ── 輸出內容 ─────────────────────────────────────────────────────

    def append(self, text, tag=None):
        self.text.configure(state="normal")
        self.text.insert("end", text, (tag,) if tag else ())
        # 長時間跑下來輸出量可能很大，Text 內容超過上限就砍掉前面舊的部分，
        # 避免記憶體無限增長——保留「最新」的訊息比保留最舊的更有用。
        self._log_line_count += text.count("\n")
        if self._log_line_count > 5000:
            self.text.delete("1.0", f"{self._log_line_count - 4000}.0")
            self._log_line_count = 4000
        if self.autoscroll_var.get():
            self.text.see("end")
        self.text.configure(state="disabled")
        if text:
            self._last_output_monotonic = time.monotonic()
            self._last_chunk_ends_newline = text.endswith("\n")

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self._log_line_count = 0

    def seconds_idle(self):
        """距離最後一次輸出過了幾秒；本次行程還沒有任何輸出時回傳 None。"""
        if self._last_output_monotonic is None:
            return None
        return time.monotonic() - self._last_output_monotonic

    def likely_waiting_for_input(self, idle_threshold=3.0):
        """粗略判斷子行程是否卡在 input() 之類的互動輸入等待中。

        核心訊號是「最後一次輸出沒有以換行結尾」：Python 內建 input() 印出提示字
        （例如「請選擇執行模式 (1/2): 」）後一定會立刻 flush stdout，但提示字本身
        沒有換行符；配合 start_log_reader() 改用逐 byte 讀取（見該函式說明，不再
        用 readline() 等一整行），這個提示字會確實出現在終端機面板裡，且是「目前
        最後一段輸出」。單看這個訊號還不夠——才剛印出提示字的下一瞬間也是這個
        狀態，所以要求同時「已經一段時間沒有新輸出」（idle_threshold 秒）才視為
        疑似卡住。只看「太久沒輸出」則會誤判成模型載入、大量幀運算等正常的沉默期
        （這類輸出正常都會以換行收尾，不會誤觸發）。

        已知取捨：用 \\r 覆寫同一行的進度顯示（例如 tqdm）若更新間隔恰好超過
        idle_threshold，仍可能被誤判成疑似等待輸入。
        """
        idle = self.seconds_idle()
        if idle is None or self._last_chunk_ends_newline:
            return False
        return idle >= idle_threshold

    def _adjust_font_size(self, delta):
        new_size = max(CONSOLE_MIN_FONT_SIZE, min(CONSOLE_MAX_FONT_SIZE, self.font_size + delta))
        if new_size == self.font_size:
            return
        self.font_size = new_size
        self.text.configure(font=(CONSOLE_FONT_FAMILY, self.font_size))

    def reset_font_size(self):
        if self.font_size == CONSOLE_DEFAULT_FONT_SIZE:
            return
        self.font_size = CONSOLE_DEFAULT_FONT_SIZE
        self.text.configure(font=(CONSOLE_FONT_FAMILY, self.font_size))

    def save_to_file(self):
        path = filedialog.asksaveasfilename(
            title="另存主控台輸出", defaultextension=".txt",
            initialfile="main_py_console.txt", filetypes=[("文字檔", "*.txt"), ("所有檔案", "*.*")],
        )
        if not path:
            return
        try:
            Path(path).write_text(self.text.get("1.0", "end"), encoding="utf-8")
        except OSError as e:
            messagebox.showerror("另存主控台輸出", f"儲存失敗：{e}")
            return
        messagebox.showinfo("另存主控台輸出", f"已儲存到：\n{path}")

    # ── 子行程輸出接進來 ─────────────────────────────────────────────

    def start_log_reader(self, process, label):
        """背景執行緒讀取子行程的 stdout（stderr 已合併進去），讀到 EOF（行程結束、
        pipe 關閉）就丟一個 None 進 queue 當結束訊號。`label` 只在結束時印
        「— {label} 行程已結束 —」用，呼叫端（process_manager.py）啟動行程當下就
        知道是 main.py 還是哪支腳本，這裡不用另外去猜。

        刻意不用 readline() 逐「行」讀：Python 的 input() 印出提示字（例如「請選擇
        執行模式 (1/2): 」）之後一定會立刻 flush，但提示字本身沒有換行符——用
        readline() 的話，這一段已經 flush 出來的文字會卡在 parent 端的緩衝區裡，
        要等到子行程收到輸入、印出下一段「有換行」的內容才會一起冒出來，導致使用者
        在終端機面板上完全看不到提示字，只覺得畫面卡住卻不知道在等什麼（這也是
        likely_waiting_for_input() 判斷「疑似卡住」的前提：提示字要先能顯示出來）。
        改成直接對 stdout 的底層 file descriptor 做 os.read()：只要子行程 flush 過，
        即使沒有換行也會立刻讀得到、立刻顯示。多位元組 UTF-8 字元可能被切在兩次
        os.read() 中間，用 incremental decoder 而非每個 chunk 各自 decode，避免中文
        字被切開後 decode 出亂碼。
        """
        self._reader_label = label
        self._reader_gen += 1
        gen = self._reader_gen  # 這條 reader 的代次；EOF 標記帶上它，_drain 只認最新一代

        def _reader():
            try:
                fd = process.stdout.fileno()
            except (AttributeError, ValueError, OSError):
                fd = None

            if fd is None:
                # 拿不到底層 fd 時退回原本逐行讀取的作法，至少維持基本可用。
                try:
                    for line in iter(process.stdout.readline, ""):
                        self.log_queue.put(line)
                except (OSError, ValueError):
                    pass
                finally:
                    self.log_queue.put((_EOF, gen))
                return

            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            try:
                while True:
                    try:
                        chunk = os.read(fd, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    text = decoder.decode(chunk)
                    if text:
                        self.log_queue.put(text)
            except ValueError:
                pass
            finally:
                self.log_queue.put((_EOF, gen))

        self._log_reader_thread = threading.Thread(target=_reader, daemon=True)
        self._log_reader_thread.start()

    def _drain_log_queue(self):
        self._drain_job = None
        try:
            if not self.window.winfo_exists():
                return
        except tk.TclError:
            return
        drained = 0
        try:
            while drained < 500:  # 單次 tick 最多處理 500 個 chunk，避免瞬間大量輸出卡住 GUI 主執行緒
                chunk = self.log_queue.get_nowait()
                if isinstance(chunk, tuple) and chunk and chunk[0] is _EOF:
                    # 只認「目前這一代」reader 的 EOF；舊行程殘留的直接丟棄，
                    # 不然會在下一支行程剛啟動時憑空冒出一行「— … 行程已結束 —」。
                    if chunk[1] == self._reader_gen:
                        self.append(f"\n— {self._reader_label} 行程已結束 —\n", tag="muted")
                else:
                    self.append(chunk)
                drained += 1
        except queue.Empty:
            pass
        self._drain_job = self.window.after(80, self._drain_log_queue)

    # ── stdin 輸入 ───────────────────────────────────────────────────

    def set_stdin_handler(self, handler):
        """handler(text) -> (ok: bool, error: str | None)，由 process_manager.py
        的 ProcessManager.send_stdin 提供——這個 class 完全不碰 subprocess 物件，
        只負責畫面跟呼叫這個 callback。"""
        self._stdin_handler = handler

    def set_input_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.stdin_entry.config(state=state)
        self.send_stdin_btn.config(state=state)

    def on_send_stdin(self):
        if self._stdin_handler is None:
            return
        text = self.stdin_var.get()
        ok, err = self._stdin_handler(text)
        if err is not None:
            self.append(f"\n[傳送輸入失敗：{err}]\n", tag="muted")
            return
        if not ok:
            return  # 沒有行程在跑（guard 失敗），原本就靜默不處理
        self.append(f"> {text}\n")
        self.text.see("end")  # 使用者剛互動過，不管「自動捲動」有沒有勾都捲到底比較符合直覺
        self.stdin_var.set("")
