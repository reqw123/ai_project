"""main.py／獨立腳本工具共用的子行程生命週期管理：啟動、關閉（CTRL_BREAK 優雅關閉
+ 逾時強制 terminate）、輪詢存活狀態、把子行程視窗拉到前景。

從 settings_window.py 搬出來的獨立元件——同一時間只允許一個行程在跑
（`self.process` 這一個唯一欄位），main.py 跟任何一支獨立腳本工具共用同一套
機制，理由是避免兩支行程搶同一份 runtime_settings.current.json／模型顯存，
也讓終端機輸出不會兩邊來源混在一起分不清楚。

這個 class 不自己畫按鈕/狀態列文字，只管行程本身；呼叫端（settings_window.py）
透過 `on_state_change` callback 得知「行程啟動了／關閉了／輪詢發現行程已死」，
自行決定按鈕/狀態列要怎麼更新（見 SettingsWindow._update_process_buttons_state）。
輸出/清除終端機、把子行程 stdout 接進終端機面板，都透過建構後才掛上來的
`self.console`（ConsolePanel 實例）完成——建構當下 ConsolePanel 通常還沒建好
（_build_process_bar() 早於 _build_middle_area()），所以用屬性事後指派，不當
建構參數。
"""

import os
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox


def any_running(*managers):
    """回傳第一個 is_running 的 ProcessManager（沒有就回 None）。
    設定視窗與身分訓練視窗各自有一顆 ProcessManager，啟動前用這個互相檢查，
    避免 main.py 與 CNN 訓練同時搶 GPU/VRAM。傳入的 None 會被略過。"""
    for m in managers:
        if m is not None and m.is_running:
            return m
    return None


class ProcessManager:
    def __init__(self, window, on_state_change):
        self.window = window
        self.on_state_change = on_state_change
        # 事後指派（見本檔案開頭說明），呼叫 start_main()/start_tool() 前一定已經設好。
        self.console = None

        self.process = None
        self.active_label = None  # 例："main.py" 或某個腳本的檔名；None＝從未啟動過
        self.active_kind = None  # "main" | "tool" | None
        self._poll_job = None  # poll() 的 after id，關窗前 stop_poll() 取消
        self._notified_exit = False  # 這支行程結束後，on_state_change 是否已回報過一次

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    # ── 啟動 ─────────────────────────────────────────────────────────

    def start_main(self, main_py_path, cwd):
        if self.is_running:
            messagebox.showinfo(
                "啟動 main.py",
                f"{self.active_label} 執行中（PID {self.process.pid}），"
                "請先停止後再啟動 main.py（同一時間只能執行一個）。",
            )
            return
        if not Path(main_py_path).exists():
            messagebox.showerror("啟動 main.py", f"找不到 main.py：{main_py_path}")
            return
        if not messagebox.askyesno(
            "啟動 main.py",
            "即將啟動 main.py，套用目前 runtime_settings.current.json／環境變數的設定內容。\n\n"
            "若您在下方設定表單中有修改但尚未按「儲存設定」，這些修改不會套用到這次啟動。\n\n"
            "是否繼續？",
        ):
            return
        try:
            # stdout/stderr 導向 pipe 讓下方終端機面板即時讀取，不再依賴「本視窗是否
            # 在終端機裡啟動」——雙擊執行 settings_window.py（沒有終端機視窗）時，
            # main.py 的輸出過去完全看不到，這是這裡改用 PIPE 的主要動機。
            # stdin 也導向 pipe：不設的話子行程預設會繼承本視窗自己的 stdin（如果是被
            # cmd/bat 啟動的話），main.py 若用到 input() 之類的互動輸入，畫面會停在
            # 一個使用者根本看不到、也打不到字的地方，變成「卡住卻不知道在等什麼」。
            # 改成 PIPE 後，下方終端機面板新增的輸入框才能把文字寫進子行程的 stdin。
            # PYTHONIOENCODING/PYTHONUTF8：main.py 印的訊息大量使用中文與 emoji，子行程
            # 若沿用系統預設的 ANSI 編碼（如 cp950）寫出 stdout，跟這裡用 utf-8 解碼會對不上、
            # 印出亂碼，所以強制子行程一律以 UTF-8 寫出。
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["PYTHONUTF8"] = "1"
            popen_kwargs = {
                "cwd": str(cwd),
                "env": child_env,
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
            }
            if os.name == "nt":
                # 只用 CREATE_NEW_PROCESS_GROUP（不加 CREATE_NEW_CONSOLE）：
                # Windows 的 GenerateConsoleCtrlEvent（CTRL_BREAK_EVENT 底層機制）只能送到
                # 「跟呼叫端共用同一個主控台」的行程群組——先前版本額外加了
                # CREATE_NEW_CONSOLE 讓 main.py 開獨立主控台視窗顯示輸出，結果反而讓
                # 「關閉 main.py」按鈕送出的 CTRL_BREAK_EVENT 永遠送不到（不同主控台，
                # 對方收不到信號），這是關閉沒有生效的根本原因。改成不開新主控台，
                # 輸出改走 stdout=PIPE 由右側面板顯示，兩者互不衝突，關閉功能維持生效。
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self.process = subprocess.Popen([sys.executable, str(main_py_path)], **popen_kwargs)
        except OSError as e:
            messagebox.showerror("啟動 main.py", f"啟動失敗：{e}")
            return
        self._notified_exit = False
        self.active_label = "main.py"
        self.active_kind = "main"
        self.on_state_change()
        self.console.clear()
        self.console.append(f"— main.py 已啟動（PID {self.process.pid}） —\n", tag="muted")
        self.console.start_log_reader(self.process, self.active_label)
        self._bring_child_window_to_front(self.process)
        messagebox.showinfo(
            "啟動 main.py",
            f"main.py 已啟動（PID {self.process.pid}）。\n\n"
            "輸出會即時顯示在下方「終端機輸出」面板中。",
        )

    def start_tool(self, script_path, extra_env=None):
        """呼叫端（settings_window.py）已經確認過 script_path 存在，這裡不重複檢查。
        extra_env 是額外要塞進子行程環境變數的 dict（例如 TEST_VIDEO_PATH），沒有
        就傳 None——這裡完全不知道「影片路徑欄位」這種 UI 概念，只管把收到的
        環境變數原樣塞給子行程。"""
        script_file = Path(script_path)
        if self.is_running:
            messagebox.showinfo(
                "執行腳本",
                f"{self.active_label} 執行中（PID {self.process.pid}），"
                "請先停止後再執行其他腳本（同一時間只能執行一個）。",
            )
            return
        if not messagebox.askyesno(
            "執行腳本",
            f"即將執行：{script_file.name}\n\n完整路徑：{script_path}\n\n"
            "這是一支獨立腳本工具，本視窗不會檢查或修改它內部寫死的路徑/參數，"
            "請自行確認內容符合你要的設定。\n\n是否繼續？",
        ):
            return
        try:
            # 跟 start_main 共用同一套 PIPE + UTF-8 強制編碼的邏輯，理由同上；
            # cwd 用腳本自己所在的資料夾，比照「假裝 cd 進去該資料夾再執行」的直覺，
            # 因為這些獨立腳本大多是各自獨立維護、原本就預期在自己的資料夾下執行。
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["PYTHONUTF8"] = "1"
            if extra_env:
                child_env.update(extra_env)
            popen_kwargs = {
                "cwd": str(script_file.parent),
                "env": child_env,
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self.process = subprocess.Popen([sys.executable, str(script_file)], **popen_kwargs)
        except OSError as e:
            messagebox.showerror("執行腳本", f"啟動失敗：{e}")
            return
        self._notified_exit = False
        self.active_label = script_file.name
        self.active_kind = "tool"
        self.on_state_change()
        self.console.clear()
        self.console.append(f"— {script_file.name} 已啟動（PID {self.process.pid}） —\n", tag="muted")
        self.console.start_log_reader(self.process, self.active_label)
        self._bring_child_window_to_front(self.process)

    def start_tool_quiet(self, script_path, extra_env=None, label=None, clear_console=True):
        """跟 start_tool 一樣啟動一支獨立腳本，但**不彈任何確認/完成對話框**——
        給「多步驟串接」的呼叫端用（例如 identity_trainer_window.py 的
        建立資料集→訓練 兩段式流程，確認訊息由呼叫端自己統一出一次就好）。
        回傳 (ok: bool, error: str|None)；ok=False 且 error=None 代表「已有行程在跑」。
        clear_console=False 時不清空終端機（串接的第二步想保留第一步的輸出）。
        """
        script_file = Path(script_path)
        if self.is_running:
            return (False, None)
        if not script_file.exists():
            return (False, f"找不到檔案：{script_path}")
        try:
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["PYTHONUTF8"] = "1"
            if extra_env:
                child_env.update({k: str(v) for k, v in extra_env.items()})
            popen_kwargs = {
                "cwd": str(script_file.parent),
                "env": child_env,
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            self.process = subprocess.Popen([sys.executable, str(script_file)], **popen_kwargs)
        except OSError as e:
            return (False, f"啟動失敗：{e}")
        self._notified_exit = False
        self.active_label = label or script_file.name
        self.active_kind = "tool"
        self.on_state_change()
        if clear_console:
            self.console.clear()
        self.console.append(f"— {self.active_label} 已啟動（PID {self.process.pid}） —\n", tag="muted")
        self.console.start_log_reader(self.process, self.active_label)
        self._bring_child_window_to_front(self.process)
        return (True, None)

    # ── 關閉 ─────────────────────────────────────────────────────────

    def stop_main(self):
        if self.active_kind != "main" or not self.is_running:
            messagebox.showinfo("關閉 main.py", "目前沒有偵測到由本視窗啟動、仍在執行中的 main.py。")
            self.on_state_change()
            return
        self.window._process_status_var.set("🖥️ 正在關閉 main.py…")
        self.window.update_idletasks()
        stopped = self.request_shutdown_and_wait()
        self.on_state_change()
        if stopped:
            messagebox.showinfo("關閉 main.py", "main.py 已確認結束。")
        else:
            messagebox.showwarning(
                "關閉 main.py",
                "已送出關閉信號並嘗試強制結束，但仍無法確認 main.py 已停止，"
                "請自行檢查工作管理員確認狀態。",
            )

    def stop_tool(self):
        if self.active_kind != "tool" or not self.is_running:
            messagebox.showinfo("停止腳本", "目前沒有偵測到由本視窗啟動、仍在執行中的腳本。")
            self.on_state_change()
            return
        label = self.active_label
        self.window._process_status_var.set(f"🖥️ 正在停止 {label}…")
        self.window.update_idletasks()
        stopped = self.request_shutdown_and_wait()
        self.on_state_change()
        if stopped:
            messagebox.showinfo("停止腳本", f"{label} 已確認結束。")
        else:
            messagebox.showwarning(
                "停止腳本",
                f"已送出關閉信號並嘗試強制結束，但仍無法確認 {label} 已停止，"
                "請自行檢查工作管理員確認狀態。",
            )

    def request_shutdown_and_wait(self, graceful_timeout=4.0, force_timeout=2.0) -> bool:
        """請求 self.process（可能是 main.py，也可能是獨立腳本工具）結束，
        回傳 True 代表最終確認已經不在執行。

        兩段式：先送 CTRL_BREAK_EVENT（main.py 有 _install_hard_shutdown_on_ctrl_c()
        接管 SIGBREAK，會先嘗試清理 CSV/segment logger/Node-RED session 再結束，本身
        內建 3 秒 watchdog；一般獨立腳本沒有這層特別處理，收到 CTRL_BREAK_EVENT 多半
        直接被 Windows 結束掉，效果上也是「停下來」），等 graceful_timeout 秒讓它有
        機會走完清理流程；逾時仍在跑，才 terminate() 強制砍斷，再等 force_timeout 秒
        確認真的死透。「關閉 main.py」「停止腳本」按鈕跟「關掉設定視窗」共用這個函式，
        確保三個入口都是真的確認生效、不是送出信號就假設成功。
        """
        if self.process is None or self.process.poll() is not None:
            return True
        try:
            if os.name == "nt":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.send_signal(signal.SIGINT)
        except OSError:
            pass

        deadline = time.monotonic() + graceful_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                return True
            time.sleep(0.2)

        if self.process.poll() is None:
            try:
                self.process.terminate()
            except OSError:
                pass
            deadline = time.monotonic() + force_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    return True
                time.sleep(0.2)

        return self.process.poll() is not None

    # ── stdin ────────────────────────────────────────────────────────

    def send_stdin(self, text):
        """回傳 (ok, error)：error 非 None 代表寫入失敗（console_panel.py 會顯示這個
        訊息）；ok=False 且 error=None 代表「目前沒有行程在跑」，維持原本靜默不處理
        的行為（不彈錯誤，只是沒反應）。"""
        if self.process is None or self.process.poll() is not None or self.process.stdin is None:
            return False, None
        try:
            self.process.stdin.write(text + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            return False, str(e)
        return True, None

    # ── 輪詢／視窗前景化 ─────────────────────────────────────────────

    def poll(self):
        """每 2 秒檢查一次子行程是否還活著——不管是 main.py 還是獨立腳本工具，都有可能
        不是被「關閉／停止」按鈕結束的（使用者直接把主控台視窗叉掉、或程式自己崩潰），
        這裡確保按鈕狀態不會卡住。

        視窗已銷毀就不再重排（避免對死掉的 widget 呼叫 .after() 丟 TclError）；
        行程已結束且已回報過一次之後，也不再每 2 秒重複呼叫 on_state_change（否則
        狀態列／按鈕會被無限重寫、也讓「已結束」狀態沒辦法被別的訊息取代）。"""
        self._poll_job = None
        try:
            if not self.window.winfo_exists():
                return
        except tk.TclError:
            return
        if self.process is not None:
            ended = self.process.poll() is not None
            if not ended or not self._notified_exit:
                self.on_state_change()
            if ended:
                self._notified_exit = True
        self._poll_job = self.window.after(2000, self.poll)

    def stop_poll(self):
        """視窗關閉前呼叫：停掉存活輪詢的 after 迴圈。"""
        if self._poll_job is not None:
            try:
                self.window.after_cancel(self._poll_job)
            except tk.TclError:
                pass
            self._poll_job = None

    def _bring_child_window_to_front(self, process):
        """背景執行緒輪詢，等子行程（main.py／獨立腳本）自己開出的 GUI 視窗出現後
        自動拉到最上層、取得焦點，不用手動 Alt+Tab 切過去。

        這些腳本大多要先載入 YOLO/ST-GCN 模型才會真的開窗，時間不固定（幾秒到
        十幾秒），所以用輪詢而不是啟動後立刻找一次。找視窗只認「屬於這個 PID
        的可見頂層視窗」，不比對標題文字——各腳本視窗標題五花八門（tkinter 預設
        標題、cv2.imshow 的檔名...），比對 PID 才不用每支腳本都額外維護一份標題
        清單，且對之後新增的腳本自動適用。

        需要 pywin32（win32gui/win32process/win32con）；環境沒裝的話直接跳過，
        不影響腳本本身執行、也不彈錯誤訊息（不該讓整個啟動流程失敗）——但會在終端機
        印一行警告，提醒「這只是少了自動置頂的錦上添花功能」，不用真的去查程式碼
        才知道發生什麼事。
        """
        try:
            import win32api
            import win32con
            import win32gui
            import win32process
        except ModuleNotFoundError:
            print(
                "[settings_gui] 未安裝 pywin32，略過「子行程視窗自動置頂」功能"
                "（不影響腳本本身執行）。如需此功能，請在目前這個 Python 環境安裝：\n"
                f'  "{sys.executable}" -m pip install pywin32',
                file=sys.stderr,
            )
            return
        except ImportError as e:
            # pip 有裝但 import 失敗，最常見是 DLL 載入失敗（pywin32 的安裝後腳本
            # pywin32_postinstall.py 沒跑過：pythoncom3xx.dll／pywintypes3xx.dll
            # 只會落在 site-packages\pywin32_system32，但 Python 3.8+ 載入 .pyd 擴充
            # 模組時只搜尋 python.exe 自己所在的資料夾／System32／明確用
            # os.add_dll_directory() 註冊過的路徑，不會自動搜尋那個資料夾，也不會搜尋
            # .pyd 自己所在的資料夾——這台機器先前就遇過同樣的狀況，靠補跑
            # postinstall 腳本把 DLL 複製到 python.exe 旁邊解決）。跟上面
            # ModuleNotFoundError 分開處理是因為修法不同：前者要重新安裝，這裡只要
            # 補跑 postinstall。
            postinstall = Path(sys.executable).parent / "Scripts" / "pywin32_postinstall.py"
            print(
                "[settings_gui] pywin32 已安裝但載入失敗，略過「子行程視窗自動置頂」功能"
                f"（不影響腳本本身執行）。錯誤訊息：{e}\n"
                "  最常見原因是 pywin32 的安裝後腳本沒跑過，修復方式：\n"
                f'  "{sys.executable}" -m pip install --force-reinstall pywin32\n'
                f'  "{sys.executable}" "{postinstall}" -install',
                file=sys.stderr,
            )
            return

        def _worker():
            deadline = time.time() + 20.0  # 模型載入可能要幾秒到十幾秒，給足時間再放棄
            target_hwnd = None
            while time.time() < deadline:
                if process.poll() is not None:
                    return  # 行程提早結束（例如啟動失敗），沒視窗好找
                found = []

                def _enum_handler(hwnd, _extra):
                    if not win32gui.IsWindowVisible(hwnd):
                        return
                    if win32gui.GetParent(hwnd) != 0:
                        return  # 只找頂層視窗，排除子控制項
                    if not win32gui.GetWindowText(hwnd):
                        return  # 排除沒標題的隱藏輔助視窗
                    _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                    if found_pid == process.pid:
                        found.append(hwnd)

                try:
                    win32gui.EnumWindows(_enum_handler, None)
                except Exception:
                    pass
                if found:
                    target_hwnd = found[0]
                    break
                time.sleep(0.3)

            if target_hwnd is None:
                return

            try:
                if win32gui.IsIconic(target_hwnd):
                    win32gui.ShowWindow(target_hwnd, win32con.SW_RESTORE)
                # Windows 預設會擋「非目前使用中程式」呼叫 SetForegroundWindow（避免
                # 程式亂搶焦點）。這裡用常見的繞法：先把自己這條執行緒的輸入佇列跟
                # 目前前景視窗的執行緒 attach 在一起，讓系統把接下來的
                # SetForegroundWindow 視為「使用者自己切換」而放行，做完再 detach。
                fg_hwnd = win32gui.GetForegroundWindow()
                current_thread_id = win32api.GetCurrentThreadId()
                fg_thread_id = win32process.GetWindowThreadProcessId(fg_hwnd)[0] if fg_hwnd else 0
                attached = False
                if fg_thread_id and fg_thread_id != current_thread_id:
                    attached = win32process.AttachThreadInput(current_thread_id, fg_thread_id, True)
                try:
                    win32gui.SetForegroundWindow(target_hwnd)
                finally:
                    if attached:
                        win32process.AttachThreadInput(current_thread_id, fg_thread_id, False)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()
