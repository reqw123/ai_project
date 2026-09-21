"""設定視窗「⚙ 額外設定」：集中管理要額外塞給獨立腳本工具的環境變數。

用途：有些設定（例如預覽視窗要 720p 還是 1080p）原本得打開每支腳本、改檔案開頭的常數
才能換。這裡改成在設定視窗選一次，按「▶ 執行所選腳本」時，設定視窗把它們當環境變數
塞進子行程；腳本讀到環境變數就覆寫檔內的預設值，沒有設（＝「使用腳本內設定」）就完全照舊。

新增一個額外環境變數只要兩件事：
  1. 在下面的 `FIELDS` 加一筆（環境變數名、顯示名稱、型別、說明）——「額外設定」視窗、
     按鈕上的摘要、啟動時的傳遞、終端機面板的提示行都是照 `FIELDS` 自動產生，不用再改別處。
  2. 在要吃這個變數的腳本裡 `os.getenv("環境變數名")` 讀進來（例如 `DISPLAY_RESOLUTION`
     的讀法，見 docs/tools腳本撰寫規範.md 第 15 條）。

環境變數名盡量沿用 `paper/config.py` 已經定義的名稱（`CAT_MONITORING_*`）：這樣腳本不論是自己讀環境變數
覆寫檔內常數，還是直接吃 `config.py`（`YOLOConfig`／`ModelPaths`…），都會拿到同一個值。

型別目前支援：
  · "choice"：從固定選項挑一個（`choices` 是 [(值, 顯示文字), ...]，值 "" ＝ 不覆寫）
  · "text"：自由輸入一行文字（留空 ＝ 不覆寫）
  · "file"：選檔案路徑（輸入框＋「瀏覽...」，`filetypes`／`initialdir` 控制對話框；留空 ＝ 不覆寫）
  · "float"：自由輸入數值，`min`／`max` 限制範圍（留空 ＝ 不覆寫）；存檔時範圍不對會擋下來
  · "stepper"：不能打字，只能用大的 ▲／▼ 按鈕（可按住連續調整）、鍵盤 ↑／↓ 或滾輪在 `min`～`max`、每次 `step` 的固定數值間調整，
    第一格永遠是「不覆寫」（信心門檻這類 0.1～1.0、每次 0.01 的數值用這個，避免打錯字或填出範圍外的數字）；
    數值一律補到 step 的小數位數（step=0.01 → "0.10"、"0.11"、…、"1.00"，共 91 格）

儲存位置：`settings_gui/ui_state.json`（純介面便利記憶，跟「上次選的腳本」同一層，鍵名
`extra_env.<環境變數名>`）；不寫進 runtime_settings.current.json，也不影響 main.py。
目前只傳給「獨立腳本工具」（`process_manager.start_tool`），不傳給 main.py。
讀寫都是 best-effort：檔案不存在／壞掉／寫不進去都安靜降級成「全部不覆寫」。
"""

import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from settings_gui import dialogs
from settings_gui import ui_state as _ui_state
from settings_gui.style import (
    BTN_PRIMARY_ACTIVE,
    BTN_PRIMARY_BG,
    BTN_SECONDARY_ACTIVE,
    BTN_SECONDARY_BG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
)
from settings_gui.widgets import _styled_button

_KEY_PREFIX = "extra_env."
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # C:\ai_project（paper/settings_gui 的上兩層）

# 「不覆寫」在資料層是空字串 ""，但 Radiobutton 不能直接拿 "" 當選項值：Tk 把「變數值 == tristatevalue
# （預設就是 ""）」視為三態，此時同一組的每一顆單選鈕都會畫成「已選」，看起來像同時勾了好幾個。
# 所以對話框裡用這個非空的代號當「不覆寫」那顆的值，存檔／載入時再跟 "" 互轉。
NO_OVERRIDE = "__no_override__"
NO_OVERRIDE_TEXT = "不覆寫"  # stepper 輸入框裡「不覆寫」那一格顯示的文字（第一格）

# ── 額外環境變數清單：之後要管理新的環境變數，就在這裡加一筆 ──────────────────
FIELDS = [
    {
        "env": "DISPLAY_RESOLUTION",
        "label": "預覽視窗解析度",
        "summary": "決定影片預覽視窗的大小：720p＝一般視窗，1080p＝全螢幕。",
        "type": "choice",
        "choices": [
            ("", "使用腳本內設定（不覆寫）"),
            ("720p", "720p（1280 × 720）"),
            ("1080p", "1080p（1920 × 1080，全螢幕）"),
        ],
        "short": {"720p": "720p", "1080p": "1080p"},  # 按鈕上的摘要用的短字
        "hint": "有開影片預覽視窗的獨立腳本會讀這個環境變數決定視窗大小：720p＝一般視窗，1080p＝全螢幕"
                "（鋪滿 1920×1080 螢幕，按 ESC 結束腳本）。只影響視窗大小，不影響偵測／推論吃的原始畫面。"
                "選「使用腳本內設定」則沿用各腳本檔案開頭的 DISPLAY_RESOLUTION。",
    },
    {
        # 對應 config.py：ModelPaths.STGCN_MODEL（環境變數 CAT_MONITORING_STGCN_MODEL）
        "env": "CAT_MONITORING_STGCN_MODEL",
        "label": "ST-GCN 模型檔案（行為分類）",
        "summary": "行為分類用的 ST-GCN 權重檔（.pth）；留空＝使用各腳本檔內預設。",
        "type": "file",
        "filetypes": [("PyTorch 權重", "*.pth *.pt"), ("所有檔案", "*.*")],
        "initialdir": str(_PROJECT_ROOT / "stgcn_models"),
        "hint": "行為分類用的 ST-GCN 權重（.pth），對應 config.py 的 ModelPaths.STGCN_MODEL。"
                "目前 1_classify_and_sort_videos／1_run_video_inference／1_visualize_three_normalizations／"
                "test_bone_length_stability 會讀；2_run_dual_model_compare 有自己的 A／B 兩組模型，不受影響。"
                "留空＝使用各腳本檔內預設。",
    },
    {
        # 對應 config.py：YOLOConfig.CONFIDENCE_THRESHOLD（環境變數 CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD）
        "env": "CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD",
        "label": "YOLO 偵測框（bbox）信心門檻",
        "summary": "低於此值的偵測框（bbox）直接丟掉。",
        "half": True,
        "type": "stepper",
        "min": 0.1,
        "max": 1.0,
        "step": 0.01,
        "hint": "這是「偵測框（bbox）」的信心門檻（YOLO predict 的 conf）：低於此值的整隻貓偵測框直接丟掉，"
                "值越高越嚴格、越容易漏偵測。不是關鍵點（kp）的信心門檻——那是下面另一個設定。"
                "點 ▲／▼ 調整 0.1～1.0（每次 0.01）；「不覆寫」＝使用各腳本檔內預設（多數是 0.5，1_skeleton_visualizer 是 0.8）。",
    },
    {
        # 對應 config.py：AnomalyDetectionConfig.KP_CONF_THRES（環境變數 CAT_MONITORING_KP_CONF_THRES）
        "env": "CAT_MONITORING_KP_CONF_THRES",
        "label": "關鍵點（kp）信心門檻",
        "summary": "低於此值的關鍵點（kp）不畫、不算位移。",
        "half": True,
        "type": "stepper",
        "min": 0.1,
        "max": 1.0,
        "step": 0.01,
        "hint": "這是「關鍵點（kp）」的信心門檻：信心低於此值的關鍵點不畫、也不拿去算位移。不是偵測框（bbox）的門檻——"
                "那是上面另一個設定。目前 1_classify_and_sort_images／1_measure_ear_distance_single_video／"
                "1_run_video_inference／1_skeleton_visualizer／1_visualize_interpolation／1_visualize_three_normalizations／"
                "2_run_dual_model_compare／test_bone_length_stability／test_pose_jitter_analysis 的骨架顯示門檻"
                "（DRAW_KP_CONF_THRESHOLD）與 test_anomaly_detection 的 KP_CONF_THRES 會讀。"
                "點 ▲／▼ 調整 0.1～1.0（每次 0.01）；「不覆寫」＝使用各腳本檔內預設（0.25～0.7 不等）。",
    },
]

# 對話框配色：沿用主視窗的色系（深藍標題列、淺色內容、白色卡片、綠色＝主要動作／已覆寫）
_HEADER_BG = "#2c3e50"
_HEADER_SUB_FG = "#cfd9e2"
_BODY_BG = "#eef2f6"
_CARD_BG = "#ffffff"
_CARD_BORDER = "#d5dde5"
_TEXT_FG = "#20303f"
_MUTED_FG = "#5a6b7b"
_LINK_FG = "#2874a6"
_STRIP_IDLE = "#cfd8e1"     # 卡片左側色條：不覆寫
_STRIP_ACTIVE = "#27ae60"   # 卡片左側色條：已覆寫
_CHIP_BG = "#eef2f6"
_CHIP_FG = "#5a6b7b"
_SEG_BG = "#e3e9ef"         # 分段按鈕（選項）
_SEG_HOVER = "#d3dce6"
_SEG_ACTIVE = "#2c3e50"
_SEG_BORDER = "#c3ccd4"
_ENTRY_BORDER = "#c3ccd4"    # 輸入框邊框（focus 時變綠）


def _ui_family(root):
    """對話框用的字型：Noto Sans TC 有裝就用它（全形標點字寬窄、筆畫均勻，中英混排整齊），沒有就退回微軟正黑體。"""
    from tkinter import font as _tkfont

    families = set(_tkfont.families(root))
    for name in ("Noto Sans TC", "Microsoft JhengHei"):
        if name in families:
            return name
    return "Microsoft JhengHei"


def _key(env):
    return _KEY_PREFIX + env


def _parse_float(field, value):
    """float 型別的解析：回傳數值；不是數字或超出 min／max 回傳 None。"""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):  # nan／inf 的大小比較一律是 False，不擋的話會直接通過範圍檢查
        return None
    lo, hi = field.get("min"), field.get("max")
    if (lo is not None and num < lo) or (hi is not None and num > hi):
        return None
    return num


def stepper_values(field):
    """stepper 的所有可選值（字串），例如 min=0.1、max=1.0、step=0.1 → ["0.1", "0.2", ..., "1.0"]。"""
    lo, hi, step = field["min"], field["max"], field["step"]
    decimals = max(0, len(str(step).split(".")[1])) if "." in str(step) else 0
    count = int(round((hi - lo) / step)) + 1
    return [f"{lo + i * step:.{decimals}f}" for i in range(count)]


def _normalize_stepper(field, value):
    """把 "0.50"、" 0.5 " 這類寫法統一成可選值裡的 "0.5"；不在可選值內回傳 None。"""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    values = stepper_values(field)
    for v in values:
        if abs(float(v) - num) < 1e-9:
            return v
    return None


def _is_valid(field, value):
    if not isinstance(value, str):
        return False
    if field["type"] == "choice":
        return value in {v for v, _label in field["choices"]}
    if field["type"] == "float":
        return _parse_float(field, value) is not None
    if field["type"] == "stepper":
        return _normalize_stepper(field, value) is not None
    return True


def load():
    """回傳 {環境變數名: 目前值}；值為 "" 代表不覆寫。壞值／不在選項內／超出範圍的值一律當成 ""。"""
    out = {}
    for f in FIELDS:
        v = _ui_state.get(_key(f["env"]), "") or ""
        if f["type"] == "stepper":
            v = _normalize_stepper(f, v) or ""
        out[f["env"]] = v if _is_valid(f, v) else ""
    return out


def save(values):
    """把 {環境變數名: 值} 存起來；空字串／非法值會刪掉該鍵（＝不覆寫）。"""
    updates = {}
    for f in FIELDS:
        v = (values.get(f["env"], "") or "").strip()
        if f["type"] == "stepper" and v:
            v = _normalize_stepper(f, v) or ""
        updates[_key(f["env"])] = v if v and _is_valid(f, v) else None
    _ui_state.update(**updates)


def collect_for_launch():
    """啟動腳本時要額外塞進子行程環境的 {環境變數名: 值}（只含有設定的項目）。"""
    return {env: v for env, v in load().items() if v}


def describe_for_launch():
    """終端機面板啟動時印的一行提示（例如「額外環境變數：DISPLAY_RESOLUTION=1080p」）；沒有則回傳 ""。"""
    active = collect_for_launch()
    if not active:
        return ""
    return "額外環境變數：" + "、".join(f"{k}={v}" for k, v in active.items())


def _short_text(field, value):
    """按鈕摘要用的短字：file 只留檔名，choice 用 short 對照，其餘原值。"""
    if field["type"] == "file":
        return Path(value).name
    return field.get("short", {}).get(value, value)


def summary():
    """按鈕上的短摘要：沒有覆寫 → ""；只有一項 → 該項的短字；多項 → 「N 項」。"""
    values = load()
    active = [(f, values[f["env"]]) for f in FIELDS if values[f["env"]]]
    if not active:
        return ""
    if len(active) == 1:
        f, v = active[0]
        return _short_text(f, v)[:12]
    return f"{len(active)} 項"


class _Stepper(tk.Frame):
    """「唯讀數值框 ＋ 大的 ▲／▼ 按鈕」：取代原生 Spinbox（Windows 上它的上下箭頭只有十幾像素寬，很難點）。

    - 數值框唯讀，不能打字；有焦點時可按鍵盤 ↑／↓（每次一格）、PageUp／PageDown（一次十格），滑鼠移到數值框上滾輪也能調整。
    - ▲／▼ 是大按鈕（`tk.Button`），按住不放會自動連續調整（repeatdelay／repeatinterval）。
    - `values` 的第一格是「不覆寫」；到頭就停住（不循環），避免從 1.00 一路按回「不覆寫」而誤觸。
    """

    _BTN_FONT = ("Segoe UI Symbol", 15, "bold")
    PAGE = 10  # PageUp／PageDown 一次跳幾格

    def __init__(self, parent, var, values, bg=_CARD_BG, family="Microsoft JhengHei"):
        super().__init__(parent, bg=bg)
        self.var = var
        self.values = list(values)
        self.entry = tk.Entry(
            self, textvariable=var, state="readonly", readonlybackground="#ffffff", width=7, justify="center",
            font=("Consolas", 17, "bold"), fg=_MUTED_FG, relief="flat", bd=0, takefocus=True,
            highlightthickness=2, highlightbackground=_ENTRY_BORDER, highlightcolor=BTN_PRIMARY_BG,
        )
        self.entry.pack(side="left", ipady=4)
        # 「不覆寫」時數值框用淡一點的字，有設定值時才用深色，一眼分得出目前有沒有值
        var.trace_add("write", self._paint_value)
        self._paint_value()
        btn_kw = dict(
            font=self._BTN_FONT, width=3, bg=BTN_SECONDARY_BG, fg="#ffffff", activebackground=BTN_SECONDARY_ACTIVE,
            activeforeground="#ffffff", relief="flat", bd=0, cursor="hand2", takefocus=False,
            repeatdelay=400, repeatinterval=90,
        )
        self.up_btn = tk.Button(self, text="▲", command=self.step_up, **btn_kw)
        self.up_btn.pack(side="left", padx=(SPACE_SM, 2), fill="y")
        self.down_btn = tk.Button(self, text="▼", command=self.step_down, **btn_kw)
        self.down_btn.pack(side="left", fill="y")
        self.entry.bind("<Up>", lambda _e: self._key_step(1))
        self.entry.bind("<Down>", lambda _e: self._key_step(-1))
        self.entry.bind("<Prior>", lambda _e: self._key_step(self.PAGE))  # PageUp
        self.entry.bind("<Next>", lambda _e: self._key_step(-self.PAGE))  # PageDown
        self.entry.bind("<MouseWheel>", self._on_wheel)
        self.entry.bind("<Button-1>", lambda _e: self.entry.focus_set())
        self.entry.bind("<Enter>", lambda _e: self.entry.focus_set())

    def _paint_value(self, *_):
        self.entry.config(fg=_MUTED_FG if self.var.get() == NO_OVERRIDE_TEXT else _TEXT_FG)

    def _index(self):
        try:
            return self.values.index(self.var.get())
        except ValueError:
            return 0

    def _set_index(self, i):
        self.var.set(self.values[max(0, min(len(self.values) - 1, i))])

    def step_up(self):
        self._set_index(self._index() + 1)

    def step_down(self):
        self._set_index(self._index() - 1)

    def _key_step(self, direction):
        self._set_index(self._index() + direction)
        return "break"

    def _on_wheel(self, event):
        self._set_index(self._index() + (1 if event.delta > 0 else -1))
        return "break"

    def get(self):
        return self.var.get()


def _is_active(field, value):
    """這個欄位目前是不是「有覆寫」（不是「不覆寫」／空白）。"""
    v = (value or "").strip()
    return bool(v) and v not in (NO_OVERRIDE, NO_OVERRIDE_TEXT)


class _Card(tk.Frame):
    """一個設定項目的卡片：左側狀態色條（綠＝已覆寫、灰＝不覆寫）、標題＋「已覆寫／不覆寫」徽章、
    環境變數名（小字等寬）、一行摘要、控制項區（`self.control`）、可展開的詳細說明。"""

    def __init__(self, parent, family, field):
        super().__init__(parent, bg=_CARD_BG, highlightbackground=_CARD_BORDER, highlightthickness=1)
        self.strip = tk.Frame(self, bg=_STRIP_IDLE, width=5)
        self.strip.pack(side="left", fill="y")
        body = tk.Frame(self, bg=_CARD_BG)
        body.pack(side="left", fill="both", expand=True, padx=(14, 14), pady=(8, 10))

        head = tk.Frame(body, bg=_CARD_BG)
        head.pack(fill="x")
        tk.Label(head, text=field["label"], bg=_CARD_BG, fg=_TEXT_FG, font=(family, 13, "bold"), anchor="w").pack(side="left")
        self.badge = tk.Label(head, text="不覆寫", bg=_CARD_BG, fg=_MUTED_FG, font=(family, 10, "bold"), padx=8, pady=1)
        self.badge.pack(side="right")

        tk.Label(
            body, text=field["env"], bg=_CHIP_BG, fg=_CHIP_FG, font=("Consolas", 9), padx=6, pady=1, anchor="w",
        ).pack(anchor="w", pady=(4, 0))

        self.summary = tk.Label(
            body, text=field.get("summary", ""), bg=_CARD_BG, fg=_MUTED_FG, font=(family, 11),
            anchor="w", justify="left", wraplength=320,
        )
        if field.get("summary"):
            self.summary.pack(fill="x", pady=(4, 0))

        self.control = tk.Frame(body, bg=_CARD_BG)
        self.control.pack(fill="x", pady=(8, 0))

        self.detail = tk.Label(
            body, text=field.get("hint", ""), bg=_CARD_BG, fg=_MUTED_FG, font=(family, 10),
            anchor="w", justify="left", wraplength=320,
        )
        self._detail_open = False
        self.toggle = tk.Label(body, text="ⓘ 詳細說明 ▸", bg=_CARD_BG, fg=_LINK_FG, font=(family, 10), cursor="hand2")
        if field.get("hint"):
            self.toggle.pack(anchor="w", pady=(6, 0))
            self.toggle.bind("<Button-1>", lambda _e: self._toggle_detail())
        # 換行寬度跟著卡片寬度走
        body.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        wrap = max(180, event.width - 8)
        self.summary.config(wraplength=wrap)
        self.detail.config(wraplength=wrap)

    def _toggle_detail(self):
        self._detail_open = not self._detail_open
        if self._detail_open:
            self.detail.pack(fill="x", pady=(4, 0))
            self.toggle.config(text="ⓘ 詳細說明 ▾")
        else:
            self.detail.pack_forget()
            self.toggle.config(text="ⓘ 詳細說明 ▸")

    def set_active(self, active):
        if active:
            self.strip.config(bg=_STRIP_ACTIVE)
            self.badge.config(text="已覆寫", bg=_STRIP_ACTIVE, fg="#ffffff")
        else:
            self.strip.config(bg=_STRIP_IDLE)
            self.badge.config(text="不覆寫", bg=_CARD_BG, fg=_MUTED_FG)


def open_dialog(parent, on_change=None):
    """彈出「額外設定」視窗。按「✓ 儲存」存檔後呼叫 on_change()（給呼叫端更新按鈕摘要）。"""
    dlg = tk.Toplevel(parent)
    dlg.title("額外設定")
    dlg.transient(parent)
    dlg.grab_set()
    dlg.configure(bg=_BODY_BG)
    dlg.resizable(False, False)
    family = _ui_family(dlg)

    # ── 標題列（深藍，跟主視窗的標題列同色系）──
    header = tk.Frame(dlg, bg=_HEADER_BG)
    header.pack(fill="x")
    tk.Label(
        header, text="⚙  額外設定", bg=_HEADER_BG, fg="#ffffff", font=(family, 17, "bold"), anchor="w",
    ).pack(fill="x", padx=20, pady=(12, 0))
    tk.Label(
        header,
        text="按「執行所選腳本」時傳給腳本的環境變數，腳本讀到就覆寫檔內預設；「不覆寫」＝照腳本檔內的寫法。"
             "只對獨立腳本工具生效，不影響 main.py。",
        bg=_HEADER_BG, fg=_HEADER_SUB_FG, font=(family, 11), anchor="w", justify="left", wraplength=740,
    ).pack(fill="x", padx=20, pady=(2, 10))

    current = load()
    vars_by_env = {}
    cards = {}

    def _refresh_state(*_):
        active = 0
        for f in FIELDS:
            on = _is_active(f, vars_by_env[f["env"]].get())
            cards[f["env"]].set_active(on)
            active += 1 if on else 0
        summary_var.set(f"已覆寫 {active} 項" if active else "目前沒有覆寫任何設定（全部使用腳本內設定）")

    body = tk.Frame(dlg, bg=_BODY_BG)
    body.pack(fill="both", expand=True, padx=16, pady=(12, 0))
    summary_var = tk.StringVar(value="")

    def _make_control(card, f, var):
        """依型別把控制項放進卡片的 control 區。"""
        ctl = card.control
        if f["type"] == "choice":
            seg = tk.Frame(ctl, bg=_SEG_BORDER, padx=1, pady=1)
            seg.pack(anchor="w")
            radios = []
            for value, text in f["choices"]:
                rb = tk.Radiobutton(
                    seg, text=text, variable=var, value=(value or NO_OVERRIDE), indicatoron=False,
                    font=(family, 12, "bold"), relief="flat", bd=0, padx=16, pady=7, cursor="hand2",
                    highlightthickness=0, takefocus=True,
                )
                rb.pack(side="left", padx=(0, 1))
                radios.append(rb)

            def _paint(*_):
                for rb in radios:
                    selected = var.get() == rb.cget("value")
                    bg = _SEG_ACTIVE if selected else _SEG_BG
                    fg = "#ffffff" if selected else _TEXT_FG
                    rb.config(
                        bg=bg, fg=fg, selectcolor=bg, activebackground=(_SEG_ACTIVE if selected else _SEG_HOVER),
                        activeforeground=fg,
                    )

            def _hover(rb, on):
                if var.get() != rb.cget("value"):
                    rb.config(bg=_SEG_HOVER if on else _SEG_BG, selectcolor=_SEG_HOVER if on else _SEG_BG)

            for rb in radios:
                rb.bind("<Enter>", lambda _e, r=rb: _hover(r, True))
                rb.bind("<Leave>", lambda _e, r=rb: _hover(r, False))
            var.trace_add("write", _paint)
            _paint()
        elif f["type"] == "file":
            def _browse(v=var, ff=f):
                path = filedialog.askopenfilename(
                    parent=dlg, title=f"選擇：{ff['label']}", filetypes=ff.get("filetypes", [("所有檔案", "*.*")]),
                    initialdir=ff.get("initialdir") or None,
                )
                if path:
                    v.set(path)

            row = tk.Frame(ctl, bg=_CARD_BG)
            row.pack(fill="x")
            tk.Entry(
                row, textvariable=var, font=("Consolas", 11), fg=_TEXT_FG, relief="flat", bd=0,
                highlightthickness=2, highlightbackground=_ENTRY_BORDER, highlightcolor=BTN_PRIMARY_BG,
            ).pack(side="left", fill="x", expand=True, ipady=6)
            _styled_button(
                row, "瀏覽...", _browse, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
                font=(family, 11, "bold"), compact=True,
            ).pack(side="left", padx=(SPACE_SM, 0))
        elif f["type"] == "stepper":
            _Stepper(ctl, var, (NO_OVERRIDE_TEXT, *stepper_values(f)), family=family).pack(anchor="w")
            tk.Label(
                ctl, text=f"▲▼ / ↑↓ / 滾輪調整　PageUp・PageDown 一次 {_Stepper.PAGE} 格\n範圍 {f['min']}～{f['max']}，每次 {f['step']}",
                bg=_CARD_BG, fg=_MUTED_FG, font=(family, 10), anchor="w", justify="left",
            ).pack(anchor="w", pady=(8, 0))
        else:  # text / float
            tk.Entry(
                ctl, textvariable=var, font=("Consolas", 12), fg=_TEXT_FG, relief="flat", bd=0,
                highlightthickness=2, highlightbackground=_ENTRY_BORDER, highlightcolor=BTN_PRIMARY_BG,
            ).pack(fill="x", ipady=6)

    # 連續的 half 欄位並排成一列（兩欄等寬），其餘一列一張卡片
    i = 0
    while i < len(FIELDS):
        f = FIELDS[i]
        group = [f]
        if f.get("half"):
            while i + len(group) < len(FIELDS) and FIELDS[i + len(group)].get("half") and len(group) < 2:
                group.append(FIELDS[i + len(group)])
        row = tk.Frame(body, bg=_BODY_BG)
        row.pack(fill="x", pady=(0, 8 if i + len(group) < len(FIELDS) else 0))
        for col, gf in enumerate(group):
            row.columnconfigure(col, weight=1, uniform="cards")
            card = _Card(row, family, gf)
            card.grid(row=0, column=col, sticky="nsew", padx=(0, 10) if col < len(group) - 1 else (0, 0))
            is_choice = gf["type"] == "choice"
            is_stepper = gf["type"] == "stepper"
            default_text = NO_OVERRIDE if is_choice else (NO_OVERRIDE_TEXT if is_stepper else "")
            var = tk.StringVar(value=(current.get(gf["env"], "") or default_text))
            vars_by_env[gf["env"]] = var
            cards[gf["env"]] = card
            _make_control(card, gf, var)
            var.trace_add("write", _refresh_state)
        i += len(group)

    # ── 底部按鈕列 ──
    footer = tk.Frame(dlg, bg=_CARD_BG, highlightbackground=_CARD_BORDER, highlightthickness=1)
    footer.pack(fill="x", side="bottom", pady=(12, 0))
    bar = tk.Frame(footer, bg=_CARD_BG)
    bar.pack(fill="x", padx=16, pady=10)
    btn_font = (family, 12, "bold")

    def _reset_all():
        for f in FIELDS:
            vars_by_env[f["env"]].set(
                NO_OVERRIDE if f["type"] == "choice" else (NO_OVERRIDE_TEXT if f["type"] == "stepper" else "")
            )

    def _cancel(_e=None):
        dlg.destroy()

    def _collect_values():
        return {
            env: ("" if var.get() in (NO_OVERRIDE, NO_OVERRIDE_TEXT) else var.get()) for env, var in vars_by_env.items()
        }

    def _save(_e=None):
        values = _collect_values()
        for f in FIELDS:
            v = (values.get(f["env"], "") or "").strip()
            if not v:
                continue
            if f["type"] == "float" and _parse_float(f, v) is None:
                lo, hi = f.get("min"), f.get("max")
                dialogs.show_warning(dlg, "額外設定", f"「{f['label']}」要填 {lo}～{hi} 之間的數字，目前是：{v}")
                return
            if f["type"] == "file" and not Path(v).is_file():
                if not dialogs.ask_yesno(
                    dlg, "額外設定", f"「{f['label']}」指定的檔案好像不存在：\n{v}\n\n仍要儲存嗎？",
                ):
                    return
        save(values)
        dlg.destroy()
        if on_change is not None:
            on_change()

    _styled_button(
        bar, "↺ 全部不覆寫", _reset_all, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, font=btn_font, compact=True,
    ).pack(side="left")
    tk.Label(bar, textvariable=summary_var, bg=_CARD_BG, fg=_MUTED_FG, font=(family, 11)).pack(side="left", padx=(14, 0))
    _styled_button(
        bar, "取消", _cancel, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE, outline=True, font=btn_font, compact=True,
    ).pack(side="right")
    _styled_button(
        bar, "✓ 儲存", _save, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE, font=btn_font, compact=True,
    ).pack(side="right", padx=(0, SPACE_SM))

    dlg.bind("<Escape>", _cancel)
    _refresh_state()
    dlg.update_idletasks()
    dlg.update_idletasks()  # 卡片的換行寬度要等第一輪版面完成後才會更新，再跑一次高度才會是最終值
    try:
        w, h = max(dlg.winfo_reqwidth(), 780), dlg.winfo_reqheight()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        dlg.geometry(f"{w}x{h}+{px + max(0, (pw - w) // 2)}+{py + max(0, (ph - h) // 2)}")
    except tk.TclError:
        pass
    dlg.wait_window()
