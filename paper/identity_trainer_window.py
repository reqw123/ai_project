"""貓咪身分辨識模型訓練 GUI（獨立視窗，從 settings_window.py 開一顆按鈕進入）。

把 tools/cat_identity/ 底下三支腳本包成按鈕操作，給「要訓練自己的貓 vs 其他貓
辨識模型」的一般使用者用，不用打開 .py 改開頭寫死的路徑常數：

  步驟 1  選「我的貓的影片資料夾」＋「其他貓的影片資料夾」，替我的貓取一個英文顯示名
  步驟 2  按「開始」→ 依序跑 1_build_dataset.py（YOLO 抽裁切圖）再跑 2_train.py（訓練 CNN）
  步驟 3  從下拉挑一個訓練好的模型 → 用影片測試 / 一鍵設為監控系統使用的模型

與 settings_window.py 共用 settings_gui/ 的樣式與 ProcessManager（子行程生命週期）。
本視窗有自己的一顆 ProcessManager 與精簡終端機；啟動任何工作前會檢查父視窗的
main.py 是否在跑（反之 settings_window 啟動 main.py 前也會檢查本視窗），避免搶 GPU。
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import queue
import codecs
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

try:
    import winsound  # Windows 完成音效；沒有就退回 self.bell()
except ImportError:
    winsound = None

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# _TrainerConsole.log_queue 裡代表「這一代 log reader 讀到 EOF」的標記（配合代次序號，
# 忽略前一支行程殘留、還沒讀掉的 EOF——見 console_panel._EOF 的同樣說明）。
_TC_EOF = object()

_PAPER_DIR = Path(__file__).resolve().parent
if str(_PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(_PAPER_DIR))

import settings_manager  # noqa: E402
from settings_gui.process_manager import ProcessManager, any_running  # noqa: E402
from settings_gui.style import (  # noqa: E402
    BTN_INFO_BG, BTN_INFO_ACTIVE, BTN_INFO_FG,
    BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE,
    BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
    BTN_DANGER_BG, BTN_DANGER_ACTIVE,
    COLOR_CONSOLE_BG, COLOR_CONSOLE_FG, COLOR_CONSOLE_MUTED_FG,
    COLOR_HEADER_BG, COLOR_HEADER_FG,
    CONSOLE_FONT_FAMILY,
    SPACE_MD, SPACE_SM,
    _styled_button,
)

_PROJECT_ROOT = _PAPER_DIR.parent                         # C:\ai_project
_CAT_IDENTITY_DIR = _PAPER_DIR / "cat_monitoring_system" / "tools" / "cat_identity"
BUILD_SCRIPT = _CAT_IDENTITY_DIR / "1_build_dataset.py"
TRAIN_SCRIPT = _CAT_IDENTITY_DIR / "2_train.py"
INFER_SCRIPT = _CAT_IDENTITY_DIR / "3_infer_video.py"
IDENTITY_MODELS_DIR = _PROJECT_ROOT / "identity_models"
LATEST_MODEL_PATH = IDENTITY_MODELS_DIR / "latest.pt"
# 每隻要訓練的貓有各自的資料集資料夾 datasets/<英文名>/，避免「之前訓練過 A、
# 現在改訓練 B」時新舊裁切圖混進同一個 crops/ 造成模型分不清。同一個名字再訓練
# 一次 = 累積更多資料（適合「同一隻貓補影片重訓」）。
DATASETS_DIR = _PAPER_DIR / "cat_monitoring_system" / "tools" / "train_data" / "cat_identity" / "datasets"

_FONT_FAMILY = "Microsoft JhengHei"

# 訓練一律用 2_train.py / 1_build_dataset.py 自己的預設值（不再由 GUI 提供「訓練強度」
# 選項）：epoch 上限交給 2_train.py 的 EARLY_STOPPING_PATIENCE 收斂就停，取樣張數用
# 1_build_dataset.py 的 MAX_CROPS_PER_CLASS 預設。這裡只留一個常數給進度條估總 epoch 用。
DEFAULT_EPOCHS = 60   # 對應 2_train.py 的 EPOCHS 預設值（進度估算用，非硬性上限）

NAME_MAX_LEN = 10
_NAME_OK_RE = re.compile(r"^[A-Za-z0-9 _-]{1,%d}$" % NAME_MAX_LEN)

# 跟 tools/cat_identity/1_build_dataset.py 的 SUPPORTED_VIDEO_EXTS 一致
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"}
_EXTS_HINT = " ".join(sorted(VIDEO_EXTS))


def _slug(name):
    """英文顯示名 → 檔案系統安全的資料夾名（也當資料集鍵）。空的話用 'target'。"""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip().lower()).strip("_-")
    return s or "target"


def _scan_videos(path):
    """遞迴掃資料夾（跟 1_build_dataset.py 的 resolve_video_paths 一樣的規則）。
    回傳 (影片數, 非影片檔數, 前幾個影片檔名)；path 不是資料夾回傳 None。"""
    p = Path(path).expanduser()
    if not p.is_dir():
        return None
    vids, others, sample = 0, 0, []
    try:
        for f in p.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix.lower() in VIDEO_EXTS:
                vids += 1
                if len(sample) < 3:
                    sample.append(f.name)
            else:
                others += 1
            if vids + others > 20000:      # 安全上限，避免掃到超大目錄卡住
                break
    except OSError:
        pass
    return vids, others, sample

_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*/\s*(\d+)"
    r"(?:.*?train\s+loss\s+[\d.]+\s+acc\s+([\d.]+))?"
    r"(?:.*?val\s+loss\s+[\d.]+\s+acc\s+([\d.]+))?"
)
_BUILD_CLASS_RE = re.compile(r"類別「(.+?)」：(\d+) 支影片")
_BUILD_VID_DONE_RE = re.compile(r"讀 \d+ 幀 -> 取樣|已處理過，跳過|無法開啟")
_BEST_RE = re.compile(r"新的最佳 val accuracy[:：]\s*([\d.]+)")


def _fmt_hms(sec):
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════════════════
#  精簡終端機（實作 ProcessManager 需要的 clear / append / start_log_reader）
# ══════════════════════════════════════════════════════════════════════
class _TrainerConsole:
    def __init__(self, parent, window, on_text=None):
        self.window = window
        self.on_text = on_text          # 每段輸出文字的回呼（給進度解析用）
        self.log_queue = queue.Queue()
        self._reader_thread = None
        self._reader_gen = 0            # log reader 代次：切行程 +1，忽略舊行程殘留的 EOF
        self._drain_job = None          # _drain 的 after id，關窗前 stop() 取消

        wrap = tk.Frame(parent, bg=COLOR_CONSOLE_BG)
        wrap.pack(fill="both", expand=True)
        self._font = tkfont.Font(family=CONSOLE_FONT_FAMILY, size=11)
        self.text = tk.Text(
            wrap, bg=COLOR_CONSOLE_BG, fg=COLOR_CONSOLE_FG, insertbackground=COLOR_CONSOLE_FG,
            font=self._font, wrap="word", bd=0, highlightthickness=0, state="disabled",
        )
        sb = tk.Scrollbar(wrap, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self.text.tag_configure("muted", foreground=COLOR_CONSOLE_MUTED_FG)
        self.text.tag_configure("progress", foreground="#7ee787")
        self.text.tag_configure("done", foreground="#58d0ff",
                                font=(CONSOLE_FONT_FAMILY, 12, "bold"))
        self._drain_job = self.window.after(80, self._drain)

    _MAX_LINES = 1500          # 終端機保留的最大行數，超過就從頭砍（避免文字爆量卡住 GUI）

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def tail(self, n_lines=18):
        try:
            txt = self.text.get(f"end-{int(n_lines) + 1}l", "end-1c")
        except tk.TclError:
            return ""
        return txt.strip() or "（終端機沒有輸出）"

    def _trim(self):
        try:
            last = int(self.text.index("end-1c").split(".")[0])
        except (ValueError, tk.TclError):
            return
        if last > self._MAX_LINES:
            self.text.delete("1.0", f"{last - self._MAX_LINES}.0")

    def append(self, text, tag=None):
        # 子行程的 \r（tqdm 進度條）會讓文字無限長 → 一律轉成換行，交給 _trim 控制總量
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        self.text.configure(state="normal")
        self.text.insert("end", text, (tag,) if tag else ())
        self._trim()
        self.text.see("end")
        self.text.configure(state="disabled")

    def start_log_reader(self, process, label):
        self._reader_gen += 1
        gen = self._reader_gen

        def _reader():
            try:
                fd = process.stdout.fileno()
            except (AttributeError, ValueError, OSError):
                fd = None
            if fd is None:
                try:
                    for line in iter(process.stdout.readline, ""):
                        self.log_queue.put(line)
                except (OSError, ValueError):
                    pass
                finally:
                    self.log_queue.put((_TC_EOF, gen))
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
                    piece = decoder.decode(chunk)
                    if piece:
                        self.log_queue.put(piece)
            except ValueError:
                pass
            finally:
                self.log_queue.put((_TC_EOF, gen))

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

    def _drain(self):
        self._drain_job = None
        try:
            if not self.window.winfo_exists():
                return
        except tk.TclError:
            return
        # 一次 tick 把佇列裡的東西全部收成一段，合併寫入（避免每個 chunk 各自
        # insert + see + trim，高輸出量下那樣會把主執行緒卡到「沒有回應」）。
        parts, ended_gen, size = [], None, 0
        try:
            while size < 65536:                 # 每個 tick 最多收 ~64KB，其餘留到下一次
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple) and item and item[0] is _TC_EOF:
                    ended_gen = item[1]
                else:
                    parts.append(item)
                    size += len(item)
        except queue.Empty:
            pass
        if parts:
            batch = "".join(parts)
            self.append(batch)
            if self.on_text:
                try:
                    self.on_text(batch)
                except Exception:
                    pass
        # 只認「目前這一代」reader 的 EOF；舊行程殘留的忽略，不然 build→train 串接
        # 時會在 train 一開始就冒出一行「— 行程已結束 —」。
        if ended_gen is not None and ended_gen == self._reader_gen:
            self.append("\n— 行程已結束 —\n", tag="muted")
        self._drain_job = self.window.after(80, self._drain)

    def stop(self):
        """視窗關閉前呼叫：停掉汲取迴圈，避免關窗後對已銷毀的 Text 操作丟 TclError。"""
        if self._drain_job is not None:
            try:
                self.window.after_cancel(self._drain_job)
            except tk.TclError:
                pass
            self._drain_job = None


# ══════════════════════════════════════════════════════════════════════
#  主視窗
# ══════════════════════════════════════════════════════════════════════
class IdentityTrainerWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.master_window = master
        self.title("貓咪身分辨識模型訓練")
        self.configure(bg="#f4f6f8")
        self.geometry("1080x860")
        self.minsize(920, 640)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ProcessManager 會用到的屬性
        self._process_status_var = tk.StringVar(value="")
        self._font_label_bold = tkfont.Font(family=_FONT_FAMILY, size=12, weight="bold")
        self._font_hint = tkfont.Font(family=_FONT_FAMILY, size=10)
        self._interp_ok = None   # None＝Python 環境檢查中；檢查完才會變 True / False
        self._closing = False    # _on_close 進行中：擋掉背景執行緒回呼再碰已銷毀的 widget

        self._pm = ProcessManager(self, on_state_change=self._on_proc_state)
        self._prev_running = False
        self._chain = None            # None / ["build","train"] / ["infer"]
        self._chain_idx = 0
        self._pending_train_env = {}
        self._pending_total_epochs = DEFAULT_EPOCHS
        self._line_buf = ""

        # 「訓練中」視覺化 / 特效用的狀態
        self._spin_job = None
        self._spin_i = 0
        self._tick_job = None
        self._train_start = None
        self._cur_epoch = 0
        self._cur_total_epochs = 0
        self._epoch_hist = []         # [(epoch, train_acc, val_acc), ...]
        self._build_class = ""
        self._build_total = 0
        self._build_done = 0
        self._pulse_job = None
        self._pulse_on = False

        self._target_dir_var = tk.StringVar()
        self._other_dir_var = tk.StringVar()
        self._dir_status = {}         # var 物件 -> 該行的狀態 Label
        self._name_var = tk.StringVar(value="mimi")
        self._rebuild_var = tk.BooleanVar(value=False)
        self._model_var = tk.StringVar()
        self._model_map = {}          # 顯示字串 -> .pt Path
        self._model_meta = {}         # 顯示字串 -> run_meta.json dict

        self._build_ui()
        self._console = _TrainerConsole(self._console_holder, self, on_text=self._on_console_text)
        self._pm.console = self._console

        self._check_interpreter()
        self._refresh_models()
        self._update_dataset_hint()
        self._pm.poll()
        self._update_buttons()

    # ── 版面 ──────────────────────────────────────────────────────────
    def _build_ui(self):
        header = tk.Frame(self, bg=COLOR_HEADER_BG)
        header.pack(fill="x")
        tk.Label(
            header, text="🐱 貓咪身分辨識模型訓練", bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG,
            font=tkfont.Font(family=_FONT_FAMILY, size=15, weight="bold"), anchor="w",
        ).pack(side="left", padx=16, pady=10)

        self._banner = tk.Label(
            self, text="", bg="#c0392b", fg="#ffffff", font=self._font_hint,
            anchor="w", justify="left", padx=12, pady=6, wraplength=940,
        )   # 直譯器檢查失敗時才 pack 出來

        body = tk.Frame(self, bg="#f4f6f8")
        body.pack(fill="x", padx=SPACE_MD, pady=SPACE_SM)

        # 步驟 1
        s1 = tk.LabelFrame(body, text=" 步驟 1　準備影片 ", bg="#f4f6f8",
                           font=self._font_label_bold, padx=SPACE_MD, pady=SPACE_SM)
        s1.pack(fill="x", pady=(0, SPACE_SM))
        tk.Label(
            s1, bg="#f4f6f8", fg="#5d7285", font=self._font_hint, anchor="w", justify="left",
            wraplength=900,
            text=("每個資料夾放「這一類貓」的影片（一支幾十秒即可，建議 3 支以上；子資料夾會一起掃）。"
                  f"支援 {_EXTS_HINT}。不要放圖片、也不要把兩隻貓混在同一夾。"),
        ).pack(fill="x", pady=(0, 4))
        self._dir_row(s1, "我的貓的影片資料夾", self._target_dir_var)
        self._dir_row(s1, "其他貓的影片資料夾", self._other_dir_var)
        nrow = tk.Frame(s1, bg="#f4f6f8")
        nrow.pack(fill="x", pady=3)
        tk.Label(nrow, text="我的貓的顯示名稱（英文）", bg="#f4f6f8", width=20, anchor="w",
                 font=self._font_hint).pack(side="left")
        _vcmd = (self.register(
            lambda p: bool(re.fullmatch(r"[A-Za-z0-9 _-]{0,%d}" % NAME_MAX_LEN, p))), "%P")
        tk.Entry(nrow, textvariable=self._name_var, width=14,
                 validate="key", validatecommand=_vcmd).pack(side="left", padx=(0, 8))
        tk.Label(nrow, text=f"英數/底線/連字號，最多 {NAME_MAX_LEN} 字；也是這隻貓的資料集名稱，換名字＝換一隻貓。",
                 bg="#f4f6f8", fg="#7f8c8d", font=self._font_hint).pack(side="left")
        drow = tk.Frame(s1, bg="#f4f6f8")
        drow.pack(fill="x", pady=(0, 2))
        tk.Label(drow, text="", bg="#f4f6f8", width=20).pack(side="left")
        self._dataset_hint = tk.Label(drow, text="", bg="#f4f6f8", fg="#2c7", font=self._font_hint,
                                      anchor="w", justify="left")
        self._dataset_hint.pack(side="left", fill="x")
        self._name_var.trace_add("write", lambda *_a: self._update_dataset_hint())

        # 步驟 2
        s2 = tk.LabelFrame(body, text=" 步驟 2　建立資料集並訓練 ", bg="#f4f6f8",
                           font=self._font_label_bold, padx=SPACE_MD, pady=SPACE_SM)
        s2.pack(fill="x", pady=(0, SPACE_SM))
        irow = tk.Frame(s2, bg="#f4f6f8")
        irow.pack(fill="x", pady=3)
        tk.Label(irow, text="資料集", bg="#f4f6f8", width=20, anchor="w",
                 font=self._font_hint).pack(side="left")
        tk.Checkbutton(irow, text="清掉這隻貓的舊裁切圖重建（不勾＝在既有資料上累積）",
                       variable=self._rebuild_var, bg="#f4f6f8",
                       font=self._font_hint).pack(side="left")

        brow = tk.Frame(s2, bg="#f4f6f8")
        brow.pack(fill="x", pady=(6, 3))
        self._start_btn = _styled_button(brow, "▶ 開始（建立資料集 → 訓練）", self._on_start,
                                         BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE, font=self._font_hint)
        self._start_btn.pack(side="left")
        self._stop_btn = _styled_button(brow, "⏹ 停止", self._on_stop,
                                        BTN_INFO_BG, BTN_INFO_ACTIVE, fg=BTN_INFO_FG, font=self._font_hint)
        self._stop_btn.pack(side="left", padx=SPACE_SM)

        # ── 「訓練中」視覺化面板 ──────────────────────────────────────
        style = ttk.Style(self)
        try:
            style.theme_use(style.theme_use())  # 確保有主題可 configure
        except tk.TclError:
            pass
        style.configure("Trainer.Horizontal.TProgressbar", thickness=16,
                        troughcolor="#dfe6ec", background=BTN_PRIMARY_BG)

        self._viz = tk.Frame(s2, bg="#eef3f7", bd=1, relief="solid")
        self._viz.pack(fill="x", pady=(8, 0))

        # 階段指示（① 建立資料集 → ② 訓練 → ③ 完成）
        self._phase_row = tk.Frame(self._viz, bg="#eef3f7")
        self._phase_row.pack(fill="x", padx=10, pady=(8, 4))
        self._phase_labels = {}
        for key, txt in (("build", "① 建立資料集"), ("train", "② 訓練模型"), ("done", "③ 完成")):
            lb = tk.Label(self._phase_row, text=txt, bg="#eef3f7", fg="#9aa7b0",
                          font=tkfont.Font(family=_FONT_FAMILY, size=10, weight="bold"),
                          padx=8, pady=2)
            lb.pack(side="left", padx=(0, 4))
            self._phase_labels[key] = lb

        srow = tk.Frame(self._viz, bg="#eef3f7")
        srow.pack(fill="x", padx=10, pady=(0, 4))
        self._spin_lbl = tk.Label(srow, text="", bg="#eef3f7", fg=BTN_PRIMARY_BG,
                                  font=tkfont.Font(family=CONSOLE_FONT_FAMILY, size=13, weight="bold"))
        self._spin_lbl.pack(side="left")
        self._status_lbl = tk.Label(srow, textvariable=self._process_status_var, bg="#eef3f7",
                                    fg="#2c3e50", anchor="w",
                                    font=tkfont.Font(family=_FONT_FAMILY, size=11, weight="bold"))
        self._status_lbl.pack(side="left", padx=(6, 0))
        self._epoch_lbl = tk.Label(srow, text="", bg="#eef3f7", fg="#1b4f72",
                                   font=tkfont.Font(family=CONSOLE_FONT_FAMILY, size=13, weight="bold"))
        self._epoch_lbl.pack(side="right")

        pgrow = tk.Frame(self._viz, bg="#eef3f7")
        pgrow.pack(fill="x", padx=10, pady=(0, 4))
        self._progress = ttk.Progressbar(pgrow, mode="determinate", length=360,
                                         style="Trainer.Horizontal.TProgressbar")
        self._progress.pack(side="left", padx=(0, 10))
        self._pct_lbl = tk.Label(pgrow, text="", bg="#eef3f7", fg="#5d7285",
                                 font=tkfont.Font(family=CONSOLE_FONT_FAMILY, size=10))
        self._pct_lbl.pack(side="left")

        # 準確率即時曲線（訓練階段才顯示）
        self._curve = tk.Canvas(self._viz, height=110, bg="#ffffff", highlightthickness=1,
                                highlightbackground="#cbd6df")
        self._curve.pack(fill="x", padx=10, pady=(2, 10))
        self._curve_hint = self._curve.create_text(
            8, 8, anchor="nw", fill="#9aa7b0", font=self._font_hint,
            text="訓練開始後這裡會畫出每個 epoch 的準確率變化",
        )
        self._curve.bind("<Configure>", lambda _e: self._epoch_hist and self._draw_curve())
        self._set_phase(None)

        # 步驟 3
        s3 = tk.LabelFrame(body, text=" 步驟 3　選擇要用的模型 ", bg="#f4f6f8",
                           font=self._font_label_bold, padx=SPACE_MD, pady=SPACE_SM)
        s3.pack(fill="x")
        mrow = tk.Frame(s3, bg="#f4f6f8")
        mrow.pack(fill="x", pady=3)
        self._model_combo = ttk.Combobox(mrow, textvariable=self._model_var, state="readonly",
                                         font=self._font_hint, height=16)
        self._model_combo.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._model_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_model_detail())
        _styled_button(mrow, "↻", self._refresh_models, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
                       font=self._font_hint, compact=True).pack(side="left")
        # 所選模型的完整資訊（下拉字串刻意留短、不塞細節，避免視窗變窄時被裁掉）
        self._model_detail = tk.Label(s3, text="", bg="#eef3f7", fg="#1b4f72", anchor="w",
                                      justify="left", font=self._font_hint, padx=8, pady=4)
        self._model_detail.pack(fill="x", pady=(2, 2))
        self._model_detail.bind(
            "<Configure>",
            lambda e: e.widget.configure(wraplength=max(e.width - 20, 200)))

        arow = tk.Frame(s3, bg="#f4f6f8")
        arow.pack(fill="x", pady=(6, 0))
        self._test_btn = _styled_button(arow, "🎬 用影片測試所選模型", self._on_test_model,
                                        BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, font=self._font_hint)
        self._test_btn.pack(side="left")
        self._apply_btn = _styled_button(arow, "✓ 設為監控系統使用的模型", self._on_apply_model,
                                         BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE, font=self._font_hint)
        self._apply_btn.pack(side="left", padx=SPACE_SM)
        self._del_btn = _styled_button(arow, "🗑 刪除所選模型", self._on_delete_model,
                                       BTN_DANGER_BG, BTN_DANGER_ACTIVE, font=self._font_hint,
                                       compact=True)
        self._del_btn.pack(side="right")

        # 終端機
        cframe = tk.Frame(self, bg=COLOR_CONSOLE_BG)
        cframe.pack(fill="both", expand=True, padx=SPACE_MD, pady=(SPACE_SM, 0))
        tk.Label(cframe, text="終端機輸出", bg=COLOR_HEADER_BG, fg=COLOR_HEADER_FG,
                 font=self._font_hint, anchor="w").pack(fill="x")
        self._console_holder = tk.Frame(cframe, bg=COLOR_CONSOLE_BG)
        self._console_holder.pack(fill="both", expand=True)

        bottom = tk.Frame(self, bg="#f4f6f8")
        bottom.pack(fill="x", padx=SPACE_MD, pady=SPACE_SM)
        _styled_button(bottom, "關閉", self._on_close, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
                       outline=True, font=self._font_hint).pack(side="right")

    def _dataset_dir(self):
        return DATASETS_DIR / _slug(self._name_var.get())

    def _crop_counts(self, dataset_dir):
        """回傳 (目標貓張數, 他貓張數)。資料夾不存在算 0。"""
        crops = dataset_dir / "crops"
        counts = []
        for cls in ("目標貓", "他貓"):
            sub = crops / cls
            counts.append(sum(1 for _ in sub.glob("*.jpg")) if sub.is_dir() else 0)
        return tuple(counts)

    def _update_dataset_hint(self):
        d = self._dataset_dir()
        tgt, oth = self._crop_counts(d)
        total = tgt + oth
        if total == 0:
            self._dataset_hint.configure(text=f"「{d.name}」尚無資料集，這次會新建。", fg="#1e8449")
            return
        # 缺一邊 / 兩邊差距很大 → 標橘色提醒；否則綠色
        balanced = tgt > 0 and oth > 0 and min(tgt, oth) / max(tgt, oth) >= 0.4
        self._dataset_hint.configure(
            text=(f"「{d.name}」目前資料集：目標貓 {tgt} 張、他貓 {oth} 張（共 {total}）。"
                  f"　{'不勾「清掉重建」的話這次會往上累積。' if balanced else '⚠ 兩類數量不平衡，建議清掉重建或補影片。'}"),
            fg="#1e8449" if balanced else "#b9770e",
        )

    def _dir_row(self, parent, label, var):
        row = tk.Frame(parent, bg="#f4f6f8")
        row.pack(fill="x", pady=(3, 0))
        tk.Label(row, text=label, bg="#f4f6f8", width=20, anchor="w",
                 font=self._font_hint).pack(side="left")
        tk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=(0, 6))
        _styled_button(row, "瀏覽…", lambda: self._pick_dir(var), BTN_SECONDARY_BG,
                       BTN_SECONDARY_ACTIVE, font=self._font_hint, compact=True).pack(side="left")
        srow = tk.Frame(parent, bg="#f4f6f8")
        srow.pack(fill="x", pady=(0, 2))
        tk.Label(srow, text="", bg="#f4f6f8", width=20).pack(side="left")
        status = tk.Label(srow, text="", bg="#f4f6f8", font=self._font_hint, anchor="w", justify="left")
        status.pack(side="left", fill="x")
        self._dir_status[str(var)] = status
        var.trace_add("write", lambda *_a, v=var: self._refresh_dir_status(v))

    def _pick_dir(self, var):
        d = filedialog.askdirectory(title="選擇影片資料夾", parent=self)
        if d:
            var.set(d)

    def _refresh_dir_status(self, var):
        lbl = self._dir_status.get(str(var))
        if lbl is None:
            return
        path = var.get().strip()
        if not path:
            lbl.configure(text="", fg="#5d7285")
            return
        res = _scan_videos(path)
        if res is None:
            lbl.configure(text="✗ 這個路徑不是資料夾（或不存在）", fg="#c0392b")
            return
        vids, others, sample = res
        if vids == 0:
            lbl.configure(
                text=f"✗ 資料夾裡找不到任何影片檔（支援 {_EXTS_HINT}）—— 確認選到「裝影片的那一層」了嗎？",
                fg="#c0392b",
            )
            return
        eg = "，例如 " + "、".join(sample) if sample else ""
        extra = f"；另有 {others} 個非影片檔（會自動略過）" if others else ""
        if vids < 3:
            lbl.configure(text=f"⚠ 只找到 {vids} 支影片{eg}{extra} —— 建議至少 3 支不同影片，模型才穩",
                          fg="#b9770e")
        else:
            lbl.configure(text=f"✓ 找到 {vids} 支影片{eg}{extra}", fg="#1e8449")

    # ── 「訓練中」視覺化 / 特效 ──────────────────────────────────────
    def _set_phase(self, phase):
        """phase = None / "build" / "train" / "done"，把階段指示燈點到對應那顆。"""
        order = ["build", "train", "done"]
        done_idx = order.index(phase) if phase in order else -1
        for i, key in enumerate(order):
            lb = self._phase_labels[key]
            if phase is None:
                lb.configure(bg="#eef3f7", fg="#9aa7b0")
            elif i < done_idx:
                lb.configure(bg="#d5f0e0", fg="#1e8449")          # 已完成
            elif i == done_idx:
                accent = "#27ae60" if phase == "done" else "#2c7be5"
                lb.configure(bg=accent, fg="#ffffff")             # 進行中 / 完成
            else:
                lb.configure(bg="#eef3f7", fg="#9aa7b0")          # 還沒到

    def _start_spinner(self):
        if self._spin_job is not None:
            return

        def _spin():
            self._spin_i = (self._spin_i + 1) % len(_SPINNER_FRAMES)
            self._spin_lbl.configure(text=_SPINNER_FRAMES[self._spin_i])
            # 狀態文字輕微呼吸（兩段綠來回），營造「還在動」的感覺
            self._pulse_on = not self._pulse_on
            self._status_lbl.configure(fg="#1e8449" if self._pulse_on else "#2c3e50")
            self._spin_job = self.after(110, _spin)

        _spin()

    def _stop_spinner(self):
        if self._spin_job is not None:
            self.after_cancel(self._spin_job)
            self._spin_job = None
        self._spin_lbl.configure(text="")
        self._status_lbl.configure(fg="#2c3e50")

    def _update_elapsed_label(self):
        if self._train_start is None:
            return
        el = time.monotonic() - self._train_start
        cur, tot = self._cur_epoch, self._cur_total_epochs
        eta = ""
        if cur > 0 and tot > 0:
            eta = f"　~剩 {_fmt_hms(el / cur * (tot - cur))}"
        pct = int(cur / tot * 100) if tot else 0
        ep = f"epoch {cur}/{tot}　" if tot else ""
        self._pct_lbl.configure(text=f"{ep}{pct:3d}%　已用時 {_fmt_hms(el)}{eta}")

    def _start_elapsed_tick(self):
        self._train_start = time.monotonic()

        def _tick():
            if self._train_start is None:
                return
            self._update_elapsed_label()
            self._tick_job = self.after(1000, _tick)

        _tick()

    def _stop_elapsed_tick(self):
        if self._tick_job is not None:
            self.after_cancel(self._tick_job)
            self._tick_job = None
        self._train_start = None

    def _draw_curve(self):
        c = self._curve
        c.delete("plot")
        hist = self._epoch_hist
        if len(hist) < 1:
            return
        if self._curve_hint is not None:
            c.delete(self._curve_hint)
            self._curve_hint = None
        w = max(c.winfo_width(), 200)
        h = 110
        pad_l, pad_r, pad_t, pad_b = 34, 8, 10, 18
        x0, x1 = pad_l, w - pad_r
        y0, y1 = pad_t, h - pad_b
        # 座標軸 + 50/75/100% 參考線
        for frac, lab in ((1.0, "100"), (0.75, "75"), (0.5, "50")):
            yy = y1 - frac * (y1 - y0)
            c.create_line(x0, yy, x1, yy, fill="#e4eaef", tags="plot")
            c.create_text(x0 - 4, yy, text=lab, anchor="e", fill="#9aa7b0",
                          font=self._font_hint, tags="plot")
        tot = max(self._cur_total_epochs, len(hist), 2)

        def _pt(ep, acc):
            x = x0 + (ep - 1) / max(tot - 1, 1) * (x1 - x0)
            acc = min(max(acc, 0.4), 1.0)
            y = y1 - (acc - 0.4) / 0.6 * (y1 - y0)
            return x, y

        for idx, col in ((1, "#f39c12"), (2, "#27ae60")):  # hist row = (epoch, train_acc, val_acc)
            pts = [_pt(hh[0], hh[idx]) for hh in hist if hh[idx] is not None]
            if len(pts) >= 2:
                flat = [v for xy in pts for v in xy]
                c.create_line(*flat, fill=col, width=2, tags="plot", smooth=True)
            for x, y in pts[-1:]:
                c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=col, outline="", tags="plot")
        # 圖例
        c.create_text(x1, y0, text="● val", anchor="ne", fill="#27ae60",
                      font=self._font_hint, tags="plot")
        c.create_text(x1 - 42, y0, text="● train", anchor="ne", fill="#f39c12",
                      font=self._font_hint, tags="plot")

    def _notify_done(self, title, body, ok=True):
        """訓練 / 工作結束時提醒使用者：音效 + 視窗拉到前景 + 對話框。"""
        try:
            if winsound is not None:
                winsound.MessageBeep(winsound.MB_ICONASTERISK if ok else winsound.MB_ICONHAND)
            else:
                self.bell()
        except Exception:
            pass
        try:
            self.deiconify()
            self.lift()
            self.attributes("-topmost", True)
            self.after(1200, self._drop_topmost)  # 具名回呼，會先檢查視窗還在不在
            self.focus_force()
        except tk.TclError:
            pass
        (messagebox.showinfo if ok else messagebox.showwarning)(title, body, parent=self)

    def _drop_topmost(self):
        try:
            if self.winfo_exists():
                self.attributes("-topmost", False)
        except tk.TclError:
            pass

    def _reset_viz(self):
        self._stop_spinner()
        self._stop_elapsed_tick()
        self._progress.stop()
        self._progress.configure(mode="determinate", value=0)
        self._pct_lbl.configure(text="")
        self._epoch_lbl.configure(text="")
        self._cur_epoch = 0
        self._cur_total_epochs = 0
        self._epoch_hist = []
        self._build_class = ""
        self._build_total = 0
        self._build_done = 0
        self._curve.delete("plot")

    # ── 直譯器檢查（背景執行緒，避免 import torch 卡住開窗）────────────
    def _check_interpreter(self):
        self._interp_ok = None  # 檢查未完成前一律當「還不能執行」：Start / Test 按鈕停用
        self._interp_result = None  # 背景執行緒只寫這個純旗標（True/False），不碰任何 tk 物件
        self._process_status_var.set("正在檢查 Python 環境…")
        self._update_buttons()

        def _worker():
            try:
                r = subprocess.run(
                    [sys.executable, "-c", "import torch, torchvision, ultralytics"],
                    capture_output=True, timeout=180,
                )
                self._interp_result = (r.returncode == 0)
            except Exception:
                self._interp_result = False

        threading.Thread(target=_worker, daemon=True).start()
        # 從「主執行緒」用 after 輪詢那個旗標——跨執行緒呼叫 self.after() 在部分
        # Tcl 版本會丟 RuntimeError（"main thread is not in main loop"），旗標一律
        # 拿不到、按鈕永遠卡在停用。改由主執行緒自己輪詢就完全沒有這個問題。
        self._poll_interp_result()

    def _poll_interp_result(self):
        if self._closing:
            return
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        if self._interp_result is not None:
            self._apply_interpreter_result(self._interp_result)
            return
        self.after(200, self._poll_interp_result)

    def _apply_interpreter_result(self, ok):
        if self._closing:
            return
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        self._interp_ok = ok
        self._process_status_var.set("" if ok else "缺少 torch / ultralytics")
        if not ok:
            self._banner.configure(text=(
                "⚠ 目前這個 Python 沒有 torch / torchvision / ultralytics，無法訓練或推論。\n"
                "請改用專案 conda 環境啟動設定視窗，例如：\n"
                r'  & "C:\Users\lynnc\anaconda3\envs\yolo\python.exe" settings_window.py'
            ))
            self._banner.pack(fill="x", after=self.winfo_children()[0])
        self._update_buttons()

    # ── 開始 / 停止 ──────────────────────────────────────────────────
    def _busy_guard(self):
        """回傳 True 代表現在不能啟動新工作（並已彈訊息）。"""
        if self._interp_ok is None:
            messagebox.showinfo(
                "請稍候",
                "正在檢查這個 Python 環境是否有 torch / ultralytics，等幾秒檢查完再試。",
                parent=self,
            )
            return True
        if not self._interp_ok:
            messagebox.showwarning("無法執行", "缺少 torch / ultralytics，請看視窗頂端的紅色提示。", parent=self)
            return True
        blocker = any_running(getattr(self.master_window, "_process_manager", None), self._pm)
        if blocker is not None:
            messagebox.showinfo(
                "有工作正在執行",
                f"「{blocker.active_label}」正在執行中（PID {blocker.process.pid}），"
                "請先等它結束或停止後再操作（同一時間只能跑一個，避免搶 GPU）。",
                parent=self,
            )
            return True
        return False

    def _on_start(self):
        if self._busy_guard():
            return
        target_dir = self._target_dir_var.get().strip()
        other_dir = self._other_dir_var.get().strip()
        if not (target_dir and other_dir):
            messagebox.showwarning("尚未選資料夾", "請先選「我的貓」和「其他貓」的影片資料夾。", parent=self)
            return

        # 嚴格檢查兩個資料夾：必須存在、且裡面（含子資料夾）掃得到影片檔
        problems, low_count = [], []
        scans = {}
        for label, d in (("我的貓", target_dir), ("其他貓", other_dir)):
            res = _scan_videos(d)
            scans[label] = res
            if res is None:
                problems.append(f"・「{label}」的路徑不是資料夾（或不存在）：\n    {d}")
            elif res[0] == 0:
                problems.append(f"・「{label}」資料夾裡（含子資料夾）找不到任何影片檔：\n    {d}")
            elif res[0] < 3:
                low_count.append(f"{label} {res[0]} 支")
        try:
            same = Path(target_dir).resolve() == Path(other_dir).resolve()
        except OSError:
            same = target_dir == other_dir
        if same:
            problems.append("・「我的貓」和「其他貓」不能是同一個資料夾。")

        if problems:
            messagebox.showerror(
                "資料夾不符合要求",
                "以下問題要先處理才能開始：\n\n" + "\n\n".join(problems) + "\n\n"
                "── 應該準備的資料 ──\n"
                f"　• 兩個資料夾，一個放「我的貓」的影片、一個放「其他貓」的影片\n"
                f"　• 影片格式：{_EXTS_HINT}（子資料夾也會一起掃）\n"
                f"　• 每類建議 3 支以上不同影片，一支幾十秒即可\n"
                f"　• 不要放圖片，也不要把兩隻貓混在同一夾",
                parent=self,
            )
            return
        only_one = any(r and r[0] == 1 for r in scans.values())
        if low_count:
            extra = ""
            if only_one:
                extra = ("\n\n⚠ 有一類只有 1 支影片：無法做「影片層級」驗證（同一支影片的相鄰"
                         "影格會同時進訓練和測試），出來的準確率會虛高、看不出真實效果。"
                         "訓練仍會跑（自動退回影格層級切分），但強烈建議每類至少 2～3 支"
                         "不同影片再訓練。")
            if not messagebox.askyesno(
                "影片數量偏少",
                f"目前 {'、'.join(low_count)}。影片太少時模型可能學不到穩定特徵、"
                f"容易把兩隻貓認錯。{extra}\n\n仍要用現有影片繼續嗎？",
                parent=self,
            ):
                return

        raw_name = self._name_var.get().strip()
        if not _NAME_OK_RE.match(raw_name):
            messagebox.showwarning(
                "名稱格式",
                f"顯示名稱請用英文字母、數字、空白、- 或 _，長度 1～{NAME_MAX_LEN} 字。", parent=self)
            return
        display_name = _slug(raw_name)
        dataset_dir = self._dataset_dir()
        exists = (dataset_dir / "crops").is_dir() and any((dataset_dir / "crops").rglob("*.jpg"))

        mode_line = (
            f"  0. 清掉「{dataset_dir.name}」既有裁切圖後重建\n" if (exists and self._rebuild_var.get())
            else (f"  0. 在「{dataset_dir.name}」既有裁切圖上累積\n" if exists
                  else f"  0. 新建資料集「{dataset_dir.name}」\n")
        )
        if not messagebox.askyesno(
            "開始訓練",
            f"即將依序執行：\n"
            f"{mode_line}"
            f"  1. 建立資料集（YOLO 逐幀抽裁切圖）\n"
            f"  2. 訓練身分 CNN（顯示名：{display_name}）\n\n"
            f"整個過程可能數分鐘到數十分鐘，過程中請勿啟動 main.py。是否開始？",
            parent=self,
        ):
            return

        if exists and self._rebuild_var.get():
            if not messagebox.askyesno(
                "重新建立資料集",
                f"會先刪除這隻貓既有的裁切圖與 manifest：\n{dataset_dir}\n\n"
                f"（其他貓的資料集資料夾不受影響）確定？", parent=self,
            ):
                return
            for sub in ("crops", "frames"):
                shutil.rmtree(dataset_dir / sub, ignore_errors=True)
            (dataset_dir / "_manifest.csv").unlink(missing_ok=True)

        dataset_dir.mkdir(parents=True, exist_ok=True)
        build_env = {
            "TEST_VIDEO_PATH_TARGET": target_dir,
            "TEST_VIDEO_PATH_OTHER": other_dir,
            "CAT_IDENTITY_OUTPUT_ROOT": str(dataset_dir),
        }
        self._pending_train_env = {
            "CAT_IDENTITY_TARGET_DISPLAY_NAME": display_name,
            "CAT_IDENTITY_DATASET_PATH": str(dataset_dir / "crops"),
        }
        # 進度條估總 epoch 用：2_train.py 的 EPOCHS 預設（實際多半會提早 early stop）
        self._pending_total_epochs = DEFAULT_EPOCHS

        self._chain = ["build", "train"]
        self._chain_idx = 0
        self._line_buf = ""
        self._reset_viz()
        self._set_phase("build")
        self._start_spinner()
        self._progress.configure(mode="indeterminate")
        self._progress.start(18)
        self._process_status_var.set("建立資料集中…")
        ok, err = self._pm.start_tool_quiet(BUILD_SCRIPT, extra_env=build_env, label="建立資料集")
        if not ok:
            self._chain = None
            self._reset_viz()
            self._set_phase(None)
            self._process_status_var.set("")
            messagebox.showerror("啟動失敗", err or "無法啟動建立資料集腳本。", parent=self)
        self._update_buttons()

    def _start_train_step(self):
        self._chain_idx = 1
        self._set_phase("train")
        self._start_spinner()
        self._epoch_hist = []
        self._cur_epoch = 0
        self._cur_total_epochs = self._pending_total_epochs
        self._start_elapsed_tick()
        # 第一個 epoch 還沒跑完（2_train.py 要跑完 train+val 才印 Epoch 行），
        # 先用已知的上限把「目前/總共」顯示出來，不用乾等
        self._epoch_lbl.configure(text=f"Epoch 1 / {self._cur_total_epochs}")
        self._process_status_var.set(f"訓練中… 第 1 / {self._cur_total_epochs} 個 epoch（進行中）")
        self._progress.configure(mode="determinate", maximum=self._cur_total_epochs, value=0)
        self._pct_lbl.configure(text=f"0 / {self._cur_total_epochs}　準備中…")
        self._draw_curve()
        ok, err = self._pm.start_tool_quiet(
            TRAIN_SCRIPT, extra_env=self._pending_train_env, label="訓練模型", clear_console=False,
        )
        if not ok:
            self._chain = None
            self._reset_viz()
            self._set_phase(None)
            self._process_status_var.set("")
            messagebox.showerror("啟動失敗", err or "無法啟動訓練腳本。", parent=self)
        self._update_buttons()

    def _on_stop(self):
        self._chain = None
        self._reset_viz()
        self._set_phase(None)
        self._process_status_var.set("已停止")
        if self._pm.is_running:
            self._pm.stop_tool()
        else:
            messagebox.showinfo("停止", "目前沒有正在執行的工作。", parent=self)
        self._update_buttons()

    # ── 子行程狀態變化 ───────────────────────────────────────────────
    def _on_proc_state(self):
        now = self._pm.is_running
        if self._prev_running and not now:
            self._prev_running = False        # 先落地，避免 _on_step_finished 內啟動下一步時重入判斷出錯
            self._on_step_finished()
        # 用「現在」的真實狀態收尾（_on_step_finished 可能已啟動下一步 → is_running 又變 True）
        self._prev_running = self._pm.is_running
        self._update_buttons()

    def _on_step_finished(self):
        rc = self._pm.process.poll() if self._pm.process is not None else None
        self._stop_spinner()
        self._progress.stop()
        if self._chain is None:
            self._stop_elapsed_tick()
            self._progress.configure(mode="determinate", value=0)
            self._process_status_var.set("已停止")
            return
        step = self._chain[self._chain_idx] if self._chain_idx < len(self._chain) else None
        if rc != 0:
            self._chain = None
            self._reset_viz()
            self._set_phase(None)
            label = {"build": "建立資料集", "train": "訓練", "infer": "影片測試"}.get(step, step or "")
            self._process_status_var.set(f"✗ {label} 失敗（結束碼 {rc}）")
            tail = self._console.tail(18)
            self._notify_done(
                f"{label}失敗",
                f"「{label}」沒有正常結束（結束碼 {rc}）。\n\n終端機最後幾行：\n"
                f"{'─' * 40}\n{tail}\n{'─' * 40}",
                ok=False,
            )
            return
        if step == "build":
            self._update_dataset_hint()
            self._process_status_var.set("資料集完成，開始訓練…")
            self._start_train_step()
        elif step == "train":
            self._chain = None
            self._stop_elapsed_tick()
            self._set_phase("done")
            self._progress.configure(mode="determinate", maximum=100, value=100)
            self._pct_lbl.configure(text="100%　完成")
            if self._cur_epoch:
                self._epoch_lbl.configure(
                    text=f"Epoch {self._cur_epoch} / {self._cur_total_epochs}（結束）")
            self._process_status_var.set("✓ 訓練完成")
            self._update_dataset_hint()
            self._refresh_models()
            if self._model_combo["values"]:
                self._model_var.set(self._model_combo["values"][0])
            self._announce_training_done()
        else:
            self._chain = None
            self._stop_elapsed_tick()
            self._progress.configure(mode="determinate", value=0)
            self._process_status_var.set("完成")

    def _announce_training_done(self):
        """讀最新一次訓練的 run_meta.json，組一段有結果數字的完成通知。"""
        acc_s = epochs_s = time_s = name_s = "?"
        try:
            latest = max(IDENTITY_MODELS_DIR.glob("run_*/run_meta.json"),
                         key=lambda p: p.stat().st_mtime)
            meta = json.loads(latest.read_text(encoding="utf-8"))
            acc = (meta.get("test_metrics") or {}).get("overall_accuracy")
            acc_s = f"{acc:.1%}" if isinstance(acc, (int, float)) else "?"
            epochs_s = f"{meta.get('best_epoch', '?')} / 上限 {meta.get('epochs_cap', '?')}"
            time_s = meta.get("total_time", "?")
            name_s = meta.get("target_display_name", "?")
        except (ValueError, OSError):
            pass
        self._console.append(
            f"\n🎉 訓練完成！顯示名 {name_s}｜測試準確率 {acc_s}｜最佳 epoch {epochs_s}｜耗時 {time_s}\n",
            tag="done",
        )
        self._notify_done(
            "✓ 訓練完成",
            f"身分辨識模型訓練完成！\n\n"
            f"　顯示名稱：{name_s}\n"
            f"　測試準確率：{acc_s}\n"
            f"　最佳 epoch：{epochs_s}\n"
            f"　訓練耗時：{time_s}\n\n"
            f"已加入步驟 3 清單（最新一筆已自動選取）。\n"
            f"要讓監控系統改用它，按「✓ 設為監控系統使用的模型」。",
            ok=True,
        )

    def _on_console_text(self, piece):
        # \r（tqdm）也當行結尾切；沒有換行時 buffer 也要設上限，別讓它無限長
        self._line_buf += piece.replace("\r", "\n")
        if len(self._line_buf) > 16384:
            self._line_buf = self._line_buf[-8192:]
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            in_build = (self._chain and self._chain_idx < len(self._chain)
                        and self._chain[self._chain_idx] == "build")
            if in_build:
                mc = _BUILD_CLASS_RE.search(line)
                if mc:
                    self._build_class = mc.group(1)
                    self._build_total = int(mc.group(2))
                    self._build_done = 0
                    if self._progress["mode"] != "determinate":
                        self._progress.stop()
                    self._progress.configure(mode="determinate", maximum=max(self._build_total, 1), value=0)
                    self._process_status_var.set(f"抽圖中… 「{self._build_class}」0/{self._build_total} 支")
                    continue
                if "已達每類上限" in line and self._build_total:
                    # 這一類達到取樣張數上限、剩下的影片會被整批跳過（不再有逐支的
                    # 處理行），進度條直接補到滿，不然會卡在中間到下一階段才跳掉。
                    self._build_done = self._build_total
                    self._progress.configure(value=self._build_total)
                    self._process_status_var.set(
                        f"抽圖中… 「{self._build_class}」已達張數上限，跳過其餘影片")
                    continue
                if _BUILD_VID_DONE_RE.search(line) and self._build_total:
                    self._build_done = min(self._build_done + 1, self._build_total)
                    self._progress.configure(value=self._build_done)
                    self._process_status_var.set(
                        f"抽圖中… 「{self._build_class}」{self._build_done}/{self._build_total} 支")
                    continue
            m = _EPOCH_RE.search(line)
            if m:
                self._on_epoch_line(m)
                continue
            mb = _BEST_RE.search(line)
            if mb:
                self._process_status_var.set(
                    f"訓練中… epoch {self._cur_epoch}/{self._cur_total_epochs}"
                    f"　★ 目前最佳 val {float(mb.group(1)):.1%}")

    def _on_epoch_line(self, m):
        cur, total = int(m.group(1)), int(m.group(2))
        tr_acc = float(m.group(3)) if m.group(3) else None
        va_acc = float(m.group(4)) if m.group(4) else None
        self._cur_epoch, self._cur_total_epochs = cur, total
        self._epoch_hist.append((cur, tr_acc, va_acc))

        if self._progress["mode"] != "determinate":
            self._progress.stop()
        self._progress.configure(mode="determinate", maximum=total, value=cur)
        pct = int(cur / total * 100) if total else 0
        va_s = f"　val {va_acc:.1%}" if va_acc is not None else ""
        # 下一個 epoch 已經在跑了，所以顯示 min(cur+1, total)「進行中」
        running = min(cur + 1, total)
        self._epoch_lbl.configure(text=f"Epoch {running} / {total}")
        self._process_status_var.set(
            f"訓練中… 完成 {cur}/{total} 個 epoch{va_s}（第 {running} 個進行中）")
        self._update_elapsed_label()
        self._draw_curve()

        # 終端機底部補一行人看得懂的進度
        bar_n = 22
        filled = round(pct / 100 * bar_n)
        bar = "█" * filled + "░" * (bar_n - filled)
        el = time.monotonic() - self._train_start if self._train_start else 0
        eta = ""
        if cur:
            eta = f"　~剩 {_fmt_hms(el / cur * (total - cur))}"
        self._console.append(
            f"  ▸ 進度 {cur:>3}/{total}  ▕{bar}▏ {pct:3d}%{va_s}　已用時 {_fmt_hms(el)}{eta}\n",
            tag="progress",
        )

    # ── 模型清單 ─────────────────────────────────────────────────────
    def _refresh_models(self):
        self._model_map.clear()
        self._model_meta = {}
        entries = []
        if IDENTITY_MODELS_DIR.is_dir():
            for pt in sorted(IDENTITY_MODELS_DIR.glob("run_*/*.pt"), reverse=True):
                if pt.name.endswith("_last.pt"):
                    continue
                meta = {}
                mp = pt.parent / "run_meta.json"
                if mp.exists():
                    try:
                        meta = json.loads(mp.read_text(encoding="utf-8"))
                    except (ValueError, OSError):
                        meta = {}
                acc = (meta.get("test_metrics") or {}).get("overall_accuracy")
                acc_s = f"{acc:.0%}" if isinstance(acc, (int, float)) else "?"
                name = meta.get("target_display_name") or "target"
                cc = meta.get("class_counts") or {}
                tgt, oth = cc.get("目標貓"), cc.get("他貓")
                cc_s = (f" ｜ 目標貓 {tgt} / 他貓 {oth}"
                        if isinstance(tgt, int) and isinstance(oth, int) else " ｜ 無資料記錄")
                disp = f"{meta.get('run_id', pt.parent.name)} ｜ {name} ｜ 準確率 {acc_s}{cc_s}"
                self._model_map[disp] = pt
                self._model_meta[disp] = meta
                entries.append(disp)
        self._model_combo["values"] = entries
        if entries and self._model_var.get() not in self._model_map:
            self._model_var.set(entries[0])
        elif not entries:
            self._model_var.set("")
        self._update_model_detail()

    def _update_model_detail(self):
        meta = getattr(self, "_model_meta", {}).get(self._model_var.get())
        if not meta:
            self._model_detail.configure(text="")
            return
        cc = meta.get("class_counts") or {}
        tgt, oth = cc.get("目標貓"), cc.get("他貓")
        if isinstance(tgt, int) and isinstance(oth, int):
            line1 = f"訓練資料　目標貓 {tgt} 張・他貓 {oth} 張（共 {tgt + oth}）"
        else:
            line1 = "訓練資料　（此模型無 run_meta 記錄）"
        bits = []
        sp = meta.get("split_sizes") or {}
        if sp:
            bits.append(f"切分 {sp.get('train','?')}/{sp.get('val','?')}/{sp.get('test','?')}")
        if meta.get("best_epoch"):
            bits.append(f"最佳 epoch {meta['best_epoch']}/{meta.get('epochs_cap','?')}")
        if meta.get("total_time"):
            bits.append(f"耗時 {meta['total_time']}")
        acc = (meta.get("test_metrics") or {}).get("overall_accuracy")
        if isinstance(acc, (int, float)):
            bits.append(f"測試準確率 {acc:.1%}")
        line2 = "　·　".join(bits)
        self._model_detail.configure(text=line1 + ("\n" + line2 if line2 else ""))

    def _selected_model_path(self):
        return self._model_map.get(self._model_var.get())

    def _selected_model_display_name(self, pt):
        """所選模型的自訂顯示名稱（訓練時的 CAT_IDENTITY_TARGET_DISPLAY_NAME）：
        優先用 _refresh_models() 已載入的 run_meta，缺了再直接讀 run_meta.json。
        兩者都拿不到就回傳空字串。"""
        meta = (getattr(self, "_model_meta", {}) or {}).get(self._model_var.get()) or {}
        name = str(meta.get("target_display_name") or "").strip()
        if not name:
            mp = pt.parent / "run_meta.json"
            if mp.exists():
                try:
                    name = str(json.loads(mp.read_text(encoding="utf-8"))
                               .get("target_display_name") or "").strip()
                except (OSError, ValueError):
                    name = ""
        return name

    def _on_test_model(self):
        if self._busy_guard():
            return
        pt = self._selected_model_path()
        if pt is None:
            messagebox.showwarning("尚未選模型", "請先從步驟 3 的下拉選一個模型。", parent=self)
            return
        d = filedialog.askdirectory(title="選一個要測試的影片資料夾（整支影片，不要用訓練過的）", parent=self)
        if not d:
            return
        self._chain = ["infer"]
        self._chain_idx = 0
        self._reset_viz()
        self._set_phase(None)
        self._start_spinner()
        self._process_status_var.set("推論測試中…（會開一個影片預覽視窗）")
        self._progress.configure(mode="indeterminate")
        self._progress.start(18)
        ok, err = self._pm.start_tool_quiet(
            INFER_SCRIPT,
            extra_env={"IDENTITY_MODEL_PATH_OVERRIDE": str(pt), "TEST_VIDEO_PATH": d},
            label="影片測試",
        )
        if not ok:
            self._chain = None
            self._reset_viz()
            messagebox.showerror("啟動失敗", err or "無法啟動推論腳本。", parent=self)
        self._update_buttons()

    def _on_apply_model(self):
        pt = self._selected_model_path()
        if pt is None:
            messagebox.showwarning("尚未選模型", "請先從步驟 3 的下拉選一個模型。", parent=self)
            return
        if any_running(getattr(self.master_window, "_process_manager", None), self._pm) is not None:
            messagebox.showinfo("請稍候", "有工作正在執行，請等它結束再設定使用中的模型。", parent=self)
            return
        try:
            shutil.copy2(pt, LATEST_MODEL_PATH)
            rel = pt.relative_to(IDENTITY_MODELS_DIR)
            (IDENTITY_MODELS_DIR / "latest.txt").write_text(str(rel) + "\n", encoding="utf-8")
        except OSError as e:
            messagebox.showerror("失敗", f"複製模型檔失敗：{e}", parent=self)
            return

        data = settings_manager.load_runtime_settings(force_reload=True) or settings_manager.restore_defaults()
        data = json.loads(json.dumps(data))  # deep copy
        data.setdefault("cat_identity", {})
        data["cat_identity"]["identity_model_path"] = str(LATEST_MODEL_PATH)
        data["cat_identity"]["target_cat_class"] = "目標貓"
        # 「貓咪 ID」（設定視窗那個唯讀欄位）跟著這個模型的自訂名稱走
        disp_name = self._selected_model_display_name(pt)
        if disp_name:
            data["cat_identity"]["cat_id"] = disp_name
        ok, errors, warnings = settings_manager.save_runtime_settings(data)
        if not ok:
            messagebox.showerror("寫入設定失敗", "runtime_settings.current.json 驗證未通過：\n" + "\n".join(errors), parent=self)
            return

        # 設定視窗還開著的話，把它那個唯讀的「身分辨識 CNN 模型檔」欄位同步刷新
        sync = getattr(self.master_window, "_sync_identity_model_field", None)
        if callable(sync):
            try:
                sync()
            except Exception:
                pass

        enabled = bool(data.get("cat_identity", {}).get("enable_identity_verification"))
        extra = "" if enabled else "\n\n⚠ 目前「啟用身分驗證」是關閉的，記得到設定視窗打開才會生效。"
        note = f"\n\n設定視窗的「貓咪 ID」已同步為「{disp_name}」。" if disp_name else ""
        messagebox.showinfo(
            "已套用",
            f"已把這個模型設為監控系統使用的模型：\n{self._model_var.get()}\n\n"
            f"（已複製成 latest.pt，下次啟動 main.py 生效）{note}{extra}",
            parent=self,
        )

    def _on_delete_model(self):
        pt = self._selected_model_path()
        if pt is None:
            messagebox.showwarning("尚未選模型", "請先從步驟 3 的下拉選一個模型。", parent=self)
            return
        if any_running(getattr(self.master_window, "_process_manager", None), self._pm) is not None:
            messagebox.showinfo("請稍候", "有工作正在執行，請等它結束再刪除模型。", parent=self)
            return

        run_dir = pt.parent
        # 安全檢查：只允許刪 identity_models/run_*/ 底下的資料夾
        try:
            in_models_dir = run_dir.parent.resolve() == IDENTITY_MODELS_DIR.resolve()
        except OSError:
            in_models_dir = False
        if not (in_models_dir and run_dir.name.startswith("run_")):
            messagebox.showerror("無法刪除", f"這個路徑不在預期的模型資料夾結構內：\n{run_dir}", parent=self)
            return

        # 是不是監控系統「直接指定」的那一個？（指到 latest.pt 的話刪 run 夾不影響執行）
        direct_active = False
        try:
            cur = settings_manager.load_runtime_settings(force_reload=True) or {}
            cur_path = (cur.get("cat_identity") or {}).get("identity_model_path", "")
            direct_active = bool(cur_path) and Path(cur_path).resolve() == pt.resolve()
        except OSError:
            pass
        feeds_latest = False
        try:
            txt = (IDENTITY_MODELS_DIR / "latest.txt").read_text(encoding="utf-8").strip()
            feeds_latest = txt.replace("\\", "/").endswith(f"{run_dir.name}/{pt.name}")
        except OSError:
            pass

        files = sorted(f.name for f in run_dir.iterdir() if f.is_file())
        warn = ""
        if direct_active:
            warn = ("\n\n🚫 這是監控系統「目前直接指定使用」的模型，刪掉後 main.py 會載入失敗、"
                    "身分驗證自動停用。建議先用別的模型按「設為監控系統使用的模型」再刪。")
        elif feeds_latest:
            warn = ("\n\nℹ 這是 latest.pt 的來源。latest.pt 是獨立複本，刪掉這個資料夾不影響"
                    "監控系統執行，只是 latest.txt 的來源記錄會失效。")

        if not messagebox.askyesno(
            "刪除模型",
            f"要永久刪除這個訓練結果資料夾嗎？\n\n{run_dir}\n"
            f"（{len(files)} 個檔案：{', '.join(files[:6])}{' …' if len(files) > 6 else ''}）\n\n"
            f"這會一併刪掉權重、混淆矩陣、訓練曲線、metrics。無法復原。{warn}",
            icon="warning", parent=self,
        ):
            return

        try:
            shutil.rmtree(run_dir)
        except OSError as e:
            messagebox.showerror("刪除失敗", f"{e}", parent=self)
            return
        self._console.append(f"\n🗑 已刪除模型資料夾：{run_dir}\n", tag="muted")
        self._refresh_models()

    # ── 按鈕狀態 / 關閉 ──────────────────────────────────────────────
    def _update_buttons(self):
        running = self._pm.is_running
        interp_ready = self._interp_ok is True   # None（檢查中）也算還不能執行
        self._start_btn.configure(state="disabled" if running or not interp_ready else "normal")
        self._stop_btn.configure(state="normal" if running else "disabled")
        self._test_btn.configure(state="disabled" if running or not interp_ready else "normal")
        self._apply_btn.configure(state="disabled" if running else "normal")
        self._del_btn.configure(state="disabled" if running else "normal")

    def _on_close(self):
        if self._pm.is_running:
            if not messagebox.askyesno(
                "關閉視窗",
                f"「{self._pm.active_label}」還在執行中，關閉視窗會一併送出停止信號。是否繼續？",
                parent=self,
            ):
                return
            self._chain = None
            self._pm.request_shutdown_and_wait()
        # 先把背景迴圈全部停掉，再 destroy——否則 poll()／_drain／環境檢查執行緒的
        # 回呼會在視窗銷毀後再觸發一次，對死掉的 widget 操作丟 TclError。
        self._closing = True
        self._chain = None
        self._pm.stop_poll()
        self._console.stop()
        self._stop_spinner()
        self._stop_elapsed_tick()
        if getattr(self.master_window, "_identity_trainer_win", None) is self:
            self.master_window._identity_trainer_win = None
        self.destroy()
