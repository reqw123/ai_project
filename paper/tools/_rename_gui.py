"""
分類歸檔後的「批次序號命名」視窗（選用的獨立擴充功能）
=======================================================
五個行為資料夾（walk/lick/scratch/shake/stop）各自列出目前的檔案，選一個類別、
指定起始序號，就能把該類別選取的檔案依序改名成「<行為>_<序號><原副檔名>」
（例如 walk_1.mp4、walk_2.mp4）。直接關閉視窗＝跳過這一步，不會動任何檔案。

【隔離設計：這支檔案壞掉不能影響分類腳本】
- 1_classify_and_sort_videos.py／1_classify_and_sort_images.py 完全不 import 這支檔案，
  歸檔完成後只用 subprocess 執行 `python _rename_gui.py`，參數以 JSON 經 stdin 傳入：
  {"dest_folders": {"walk": "資料夾路徑", ...}, "exts": [".mp4", ...], "fresh_files": ["路徑", ...]}
- 因此這支檔案的語法錯誤、檔案遺失、執行時例外、甚至整個行程當掉，都只會讓這個子行程
  失敗；分類腳本只印一行警告，已完成的分類與歸檔不受影響。
- 已知限制：在設定視窗按「停止」只會終止分類腳本本身，已經開著的命名視窗不會被一併
  關閉（手動關掉即可，無害）。

底線開頭的檔名是「內部輔助」慣例，settings_window.py 的腳本下拉選單會排除。

防止覆蓋其他來源影片的機制（這是設計這支工具的主要原因）：
- 起始序號預設是「該資料夾內已存在的 <行為>_N 最大值 + 1」，不是固定從 1 開始。
- 只要目標檔名在資料夾裡已存在、而且那個檔案不在這次選取範圍內，視窗會標紅並
  停用「執行命名」按鈕，不會讓你按下去。
- 實際改名分兩階段（先全部改成暫名、再改成最終名），所以「選取範圍內互相換位」
  （例如 walk_2 想改成 walk_1、同時 walk_1 也在選取中）不會中途撞名；任何一步失敗
  會盡力把已改的名字還原。
- 以 os.rename 改名（Windows 上目標已存在會直接報錯，不會靜默覆蓋），再加一次
  明確的 lexists 檢查，跨平台行為一致。
"""
import json
import os
import re
import sys
import uuid
from pathlib import Path


# ==================== 純邏輯（不依賴 tkinter，可單獨測試）====================
def _natural_key(name):
    """自然排序：walk_2 排在 walk_10 前面。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_class_files(folder, exts):
    """回傳資料夾內（單層）副檔名在 exts 內的檔案，依檔名自然排序。資料夾不存在回傳 []。"""
    p = Path(folder)
    if not p.is_dir():
        return []
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]
    return sorted(files, key=lambda f: _natural_key(f.name))


def next_free_index(folder, prefix, exts):
    """資料夾內已存在的「<prefix>_<數字>」檔名的最大數字 + 1；一個都沒有就回傳 1。"""
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$", re.IGNORECASE)
    used = []
    for f in list_class_files(folder, exts):
        m = pattern.match(f.stem)
        if m:
            used.append(int(m.group(1)))
    return max(used, default=0) + 1


def plan_renames(folder, selected, prefix, start):
    """規劃改名。selected 是要改名的檔案（Path list，維持傳入順序編號）。

    回傳 (plan, conflicts)：
    - plan：[(來源 Path, 目標 Path), ...]，目標名稱 = <prefix>_<start+i><原副檔名>
    - conflicts：目標檔名已存在於資料夾、且該檔不在 selected 內的目標名稱清單
      （在 selected 內的不算衝突——兩階段改名會處理互相換位）
    """
    folder = Path(folder)
    selected_names = {f.name.lower() for f in selected}
    existing_names = {p.name.lower() for p in folder.iterdir()} if folder.is_dir() else set()

    plan, conflicts = [], []
    for i, src in enumerate(selected):
        dst = folder / f"{prefix}_{start + i}{src.suffix}"
        plan.append((src, dst))
        low = dst.name.lower()
        if low in existing_names and low not in selected_names:
            conflicts.append(dst.name)
    return plan, conflicts


def execute_renames(plan):
    """執行 plan_renames() 的結果，回傳實際改名的檔案數。

    兩階段：先全部改成唯一暫名，再改成最終名。失敗時（例如檔案被別的程式開著）
    把已經改過的名字盡力還原，然後把原例外往上丟。"""
    moves = [(s, d) for s, d in plan if s.name != d.name]
    staged = []    # [(暫名, 原名, 最終名)]
    finished = []  # 已經改成最終名的 staged 項目
    try:
        for src, dst in moves:
            temp = src.with_name(f".__renaming_{uuid.uuid4().hex}{src.suffix}")
            os.rename(src, temp)
            staged.append((temp, src, dst))
        for item in staged:
            temp, _src, dst = item
            if os.path.lexists(dst):  # 不靠 os.rename 在各平台的覆蓋語意，明確擋掉
                raise FileExistsError(f"目標檔名已存在：{dst}")
            os.rename(temp, dst)
            finished.append(item)
    except OSError:
        for item in staged:
            temp, src, dst = item
            current = dst if item in finished else temp
            try:
                os.rename(current, src)
            except OSError:
                pass
        raise
    return len(moves)


# ==================== GUI ====================
# 配色沿用 1_classify_and_sort_images.py 分類視窗的深色石板色系，讓「分類 → 命名」兩階段是同一套介面。
# 這裡只放外觀常數；本模組刻意不 import 專案其他檔案（隔離設計），色碼直接寫在這裡。
_C_BG = "#111827"        # 視窗底色
_C_CARD = "#1F2937"      # 卡片／控制列底色
_C_BORDER = "#374151"    # 卡片外框（未作用中）
_C_LIST_BG = "#0B1220"   # 清單／預覽區底色
_C_FG = "#F9FAFB"        # 主要文字
_C_TEXT = "#D1D5DB"      # 清單一般文字
_C_MUTED = "#9CA3AF"     # 次要文字
_C_DIM = "#6B7280"       # 停用文字
_C_WARN = "#F87171"      # 警告紅
_C_OK = "#4ADE80"        # 成功綠
_C_PRIMARY = "#16A34A"   # 主要按鈕（執行命名）
_C_PRIMARY_HOVER = "#22C55E"
_C_SECONDARY = "#374151"  # 次要按鈕
_C_SECONDARY_HOVER = "#4B5563"
_FONT = "Microsoft JhengHei UI"
# 每個行為類別的代表色（亮色系，深色底上清楚）；沒列到的類別依順序輪流取用 _FALLBACK_ACCENTS
_ACCENTS = {
    "walk": "#60A5FA", "lick": "#F472B6", "scratch": "#FBBF24", "shake": "#A78BFA", "stop": "#34D399",
}
_FALLBACK_ACCENTS = ["#60A5FA", "#F472B6", "#FBBF24", "#A78BFA", "#34D399", "#F87171", "#22D3EE"]


def open_rename_dialog(dest_folders, exts, fresh_files=()):
    """開啟批次序號命名視窗，阻塞到使用者關閉視窗為止。

    dest_folders：{行為名稱: 資料夾路徑}（維持傳入順序）
    exts：允許的影片副檔名集合（小寫、含點）
    fresh_files：這次執行剛歸檔進來的檔案路徑，清單裡會加「＊」標記，方便跟其他來源
                 已在資料夾裡的檔案區分（只是標記，改不改名由使用者選取決定）
    沒有顯示環境（tk 開不起來）時印出訊息後直接略過。"""
    import tkinter as tk
    from tkinter import ttk, messagebox

    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(f"⚠ 無法開啟批次命名視窗（{e}），略過此步驟")
        return

    fresh = {str(Path(p)).lower() for p in fresh_files}
    root.title("批次序號命名 — 直接關閉視窗＝跳過")
    root.geometry("1280x760")
    root.minsize(900, 600)
    root.configure(bg=_C_BG)

    names = list(dest_folders)
    accents = {
        n: _ACCENTS.get(n.lower(), _FALLBACK_ACCENTS[i % len(_FALLBACK_ACCENTS)]) for i, n in enumerate(names)
    }
    files = {n: [] for n in names}
    listboxes, cards, badges, name_labels, empty_labels = {}, {}, {}, {}, {}
    cat_var = tk.StringVar(value="")
    active = {"name": None}  # 目前作用中的類別；不能用 cat_var 判斷「是否剛切換」——cat_var 會先被設成新值
    start_var = tk.StringVar(value="1")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")  # clam 才能自訂捲軸顏色；預設的 Windows 原生主題會忽略配色
    except tk.TclError:
        pass
    style.configure(
        "Dark.Vertical.TScrollbar", background=_C_BORDER, troughcolor=_C_LIST_BG, bordercolor=_C_LIST_BG,
        arrowcolor=_C_MUTED, lightcolor=_C_BORDER, darkcolor=_C_BORDER, relief="flat",
    )
    style.map("Dark.Vertical.TScrollbar", background=[("active", _C_SECONDARY_HOVER)])

    # ---------- 標題列 ----------
    header = tk.Frame(root, bg=_C_BG)
    header.pack(fill="x", padx=18, pady=(16, 8))
    title_row = tk.Frame(header, bg=_C_BG)
    title_row.pack(fill="x")
    tk.Label(
        title_row, text="批次序號命名", font=(_FONT, 16, "bold"), fg=_C_FG, bg=_C_BG,
    ).pack(side="left")
    legend = tk.Label(
        title_row, text="＊ 本次剛歸檔", font=(_FONT, 9, "bold"), fg=_C_LIST_BG, bg=_C_MUTED, padx=8, pady=2,
    )
    legend.pack(side="right")
    tk.Label(
        header,
        text="點類別標題決定要命名哪個類別（有＊的檔案就預設只選它們，沒有＊則全選），"
             "也可在清單裡用 Ctrl／Shift 自行多選。只有選取的檔案會被改名，依清單順序編號。",
        font=(_FONT, 9), fg=_C_MUTED, bg=_C_BG, anchor="w", justify="left", wraplength=1200,
    ).pack(fill="x", pady=(4, 0))
    tk.Label(
        header, text="直接關閉視窗＝跳過，不會動任何檔案。", font=(_FONT, 9), fg=_C_DIM, bg=_C_BG, anchor="w",
    ).pack(fill="x")

    # ---------- 五張類別卡片 ----------
    columns = tk.Frame(root, bg=_C_BG)
    columns.pack(fill="both", expand=True, padx=18, pady=6)
    columns.rowconfigure(0, weight=1)
    for c in range(len(names)):
        columns.columnconfigure(c, weight=1, uniform="cat")

    def display(path):
        mark = "＊ " if str(path).lower() in fresh else "    "
        return f"{mark}{path.name}"

    def restyle_cards():
        """作用中的卡片外框與標題亮起類別色，其餘維持低調。"""
        for n in names:
            on = active["name"] == n
            cards[n].configure(highlightbackground=accents[n] if on else _C_BORDER)
            name_labels[n].configure(fg=accents[n] if on else _C_FG)

    def reload_lists():
        for n in names:
            files[n] = list_class_files(dest_folders[n], exts)
            lb = listboxes[n]
            selected_before = set(lb.curselection())
            lb.delete(0, "end")
            for idx, f in enumerate(files[n]):
                lb.insert("end", display(f))
                if str(f).lower() in fresh:
                    lb.itemconfig(idx, fg=accents[n])  # 本次剛歸檔的檔案用類別色，一眼分得出來
            for i in selected_before:
                if i < len(files[n]):
                    lb.selection_set(i)
            badges[n].config(text=str(len(files[n])))
            if files[n]:
                empty_labels[n].place_forget()
            else:
                empty_labels[n].place(relx=0.5, rely=0.35, anchor="center")

    def selected_files():
        n = cat_var.get()
        if not n:
            return []
        return [files[n][i] for i in listboxes[n].curselection() if i < len(files[n])]

    for c, n in enumerate(names):
        accent = accents[n]
        card = tk.Frame(
            columns, bg=_C_CARD, highlightthickness=2, highlightbackground=_C_BORDER, highlightcolor=_C_BORDER,
        )
        card.grid(row=0, column=c, sticky="nsew", padx=(0 if c == 0 else 6, 0 if c == len(names) - 1 else 6))
        cards[n] = card
        tk.Frame(card, bg=accent, height=4).pack(fill="x")

        head = tk.Frame(card, bg=_C_CARD, cursor="hand2")
        head.pack(fill="x", padx=12, pady=(10, 8))
        name_lbl = tk.Label(head, text=n, font=(_FONT, 12, "bold"), fg=_C_FG, bg=_C_CARD, cursor="hand2")
        name_lbl._is_card_header = True  # 給自動化測試辨識用（可點的類別標題）
        name_lbl.pack(side="left")
        badge = tk.Label(
            head, text="0", font=(_FONT, 9, "bold"), fg=_C_LIST_BG, bg=accent, padx=8, pady=1, cursor="hand2",
        )
        badge.pack(side="right")
        name_labels[n], badges[n] = name_lbl, badge
        for w in (head, name_lbl, badge):
            w.bind("<Button-1>", lambda _e, n=n: activate(n, "default"))

        holder = tk.Frame(card, bg=_C_LIST_BG)
        holder.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        sb = ttk.Scrollbar(holder, orient="vertical", style="Dark.Vertical.TScrollbar")
        lb = tk.Listbox(
            holder, selectmode="extended", exportselection=False, activestyle="none",
            bg=_C_LIST_BG, fg=_C_TEXT, font=(_FONT, 10), relief="flat", borderwidth=0, highlightthickness=0,
            selectbackground=accent, selectforeground=_C_LIST_BG, selectborderwidth=0,
            yscrollcommand=sb.set,
        )
        sb.config(command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)
        lb.bind("<<ListboxSelect>>", lambda _e, n=n: activate(n, "keep"))
        listboxes[n] = lb
        empty_labels[n] = tk.Label(holder, text="（沒有檔案）", font=(_FONT, 9), fg=_C_DIM, bg=_C_LIST_BG)

    # ---------- 控制列 ----------
    controls = tk.Frame(root, bg=_C_CARD, highlightthickness=1, highlightbackground=_C_BORDER)
    controls.pack(fill="x", padx=18, pady=6)
    inner = tk.Frame(controls, bg=_C_CARD)
    inner.pack(fill="x", padx=14, pady=10)

    def make_button(parent, text, command, bg, hover, fg=_C_FG, bold=False):
        btn = tk.Button(
            parent, text=text, command=command, font=(_FONT, 10, "bold" if bold else "normal"),
            fg=fg, bg=bg, activeforeground=fg, activebackground=hover, relief="flat", bd=0,
            padx=14, pady=6, cursor="hand2", disabledforeground=_C_DIM,
        )
        btn._normal_bg, btn._hover_bg = bg, hover
        btn.bind("<Enter>", lambda _e, b=btn: b.config(bg=b._hover_bg) if str(b.cget("state")) == "normal" else None)
        btn.bind("<Leave>", lambda _e, b=btn: b.config(bg=b._normal_bg) if str(b.cget("state")) == "normal" else None)
        return btn

    tk.Label(inner, text="起始序號", font=(_FONT, 10), fg=_C_MUTED, bg=_C_CARD).pack(side="left")
    spin = tk.Spinbox(
        inner, from_=1, to=999999, width=8, textvariable=start_var, font=(_FONT, 11),
        bg=_C_LIST_BG, fg=_C_FG, insertbackground=_C_FG, buttonbackground=_C_SECONDARY,
        relief="flat", highlightthickness=1, highlightbackground=_C_BORDER, highlightcolor=_C_MUTED,
    )
    spin.pack(side="left", padx=(8, 12), ipady=3)

    def set_rename_enabled(enabled):
        if enabled:
            rename_btn.config(state="normal", bg=rename_btn._normal_bg)
        else:
            rename_btn.config(state="disabled", bg=_C_SECONDARY)

    rename_btn = make_button(
        inner, "執行命名", lambda: do_rename(), _C_PRIMARY, _C_PRIMARY_HOVER, bold=True,
    )
    rename_btn.pack(side="right")
    auto_btn = make_button(inner, "自動帶入下一個空序號", lambda: auto_start(), _C_SECONDARY, _C_SECONDARY_HOVER)
    auto_btn.pack(side="left", padx=(0, 8))
    all_btn = make_button(
        inner, "全選目前類別",
        lambda: cat_var.get() and activate(cat_var.get(), "all"), _C_SECONDARY, _C_SECONDARY_HOVER,
    )
    all_btn.pack(side="left")

    # ---------- 預覽區 ----------
    preview_card = tk.Frame(root, bg=_C_CARD, highlightthickness=1, highlightbackground=_C_BORDER)
    preview_card.pack(fill="x", padx=18, pady=6)
    tk.Label(
        preview_card, text="預覽", font=(_FONT, 9, "bold"), fg=_C_MUTED, bg=_C_CARD, anchor="w",
    ).pack(fill="x", padx=14, pady=(8, 4))
    preview_body = tk.Frame(preview_card, bg=_C_LIST_BG)
    preview_body.pack(fill="x", padx=10, pady=(0, 10))
    preview_sb = ttk.Scrollbar(preview_body, orient="vertical", style="Dark.Vertical.TScrollbar")
    preview = tk.Text(
        preview_body, height=9, state="disabled", wrap="none", font=("Consolas", 10),
        bg=_C_LIST_BG, fg=_C_FG, relief="flat", borderwidth=0, highlightthickness=0, padx=12, pady=8,
        yscrollcommand=preview_sb.set, cursor="arrow",
    )
    preview_sb.config(command=preview.yview)
    preview_sb.pack(side="right", fill="y")
    preview.pack(side="left", fill="x", expand=True)
    preview.tag_config("plain", foreground=_C_FG)
    preview.tag_config("head", foreground=_C_FG, font=("Consolas", 10, "bold"))
    preview.tag_config("muted", foreground=_C_MUTED)
    preview.tag_config("old", foreground=_C_MUTED)
    preview.tag_config("arrow", foreground=_C_DIM)
    preview.tag_config("new", foreground=_C_OK)
    preview.tag_config("noop", foreground=_C_DIM)
    preview.tag_config("warn", foreground=_C_WARN)

    status_var = tk.StringVar(value="")
    tk.Label(
        root, textvariable=status_var, font=(_FONT, 9), fg=_C_OK, bg=_C_BG, anchor="w",
    ).pack(fill="x", padx=20, pady=(2, 12))

    def set_preview(lines, warn=False):
        """lines 的每個元素是純字串，或 [(文字, tag), ...] 的分段清單（舊名／箭頭／新名分色用）。"""
        preview.config(state="normal")
        preview.delete("1.0", "end")
        for i, line in enumerate(lines):
            segments = line if isinstance(line, list) else [(line, "warn" if warn else "muted")]
            for text, tag in segments:
                preview.insert("end", text, tag)
            if i < len(lines) - 1:
                preview.insert("end", "\n")
        preview.config(state="disabled")

    def refresh_preview(*_):
        n = cat_var.get()
        sel = selected_files()
        set_rename_enabled(False)
        if not n:
            set_preview(["請先選擇要命名的類別。"])
            return
        if not sel:
            set_preview([f"[{n}] 尚未選取任何檔案。"])
            return
        try:
            start = int(start_var.get())
            if start < 1:
                raise ValueError
        except ValueError:
            set_preview(["起始序號必須是 1 以上的整數。"], warn=True)
            return
        plan, conflicts = plan_renames(dest_folders[n], sel, n, start)
        if conflicts:
            shown = "、".join(conflicts[:8]) + (" …" if len(conflicts) > 8 else "")
            set_preview([
                f"⚠ {len(conflicts)} 個目標檔名已被資料夾內「未選取」的檔案佔用：{shown}",
                "請調高起始序號（可按「自動帶入下一個空序號」），或把佔用該檔名的檔案一併選取。",
            ], warn=True)
            return
        preview.tag_config("new", foreground=accents[n])  # 新檔名用該類別的代表色
        lines = [[(f"[{n}] 將改名 {sum(1 for s, d in plan if s.name != d.name)} 個檔案（起始序號 {start}）：", "head")]]
        for src, dst in plan:
            line = [("  ", "plain"), (src.name, "old"), ("  →  ", "arrow"), (dst.name, "new")]
            if src.name == dst.name:
                line.append(("（名稱相同，略過）", "noop"))
            lines.append(line)
        set_preview(lines)
        set_rename_enabled(True)

    def activate(n, mode):
        """切換要命名的類別：清掉其他類別的選取，起始序號帶入該資料夾的下一個空序號。

        mode："default"＝有「＊」（本次剛歸檔）的檔案就只選它們、沒有就全選（避免預設就把
        其他來源的舊檔一起重新編號）；"all"＝全選；"keep"＝維持使用者在清單裡點選的結果。"""
        changed = active["name"] != n
        active["name"] = n
        cat_var.set(n)
        for other in names:
            if other != n:
                listboxes[other].selection_clear(0, "end")
        if mode == "all":
            listboxes[n].selection_set(0, "end")
        elif mode == "default":
            listboxes[n].selection_clear(0, "end")
            fresh_idx = [i for i, f in enumerate(files[n]) if str(f).lower() in fresh]
            if fresh_idx:
                for i in fresh_idx:
                    listboxes[n].selection_set(i)
            else:
                listboxes[n].selection_set(0, "end")
        if changed:
            start_var.set(str(next_free_index(dest_folders[n], n, exts)))
        restyle_cards()
        refresh_preview()

    def auto_start():
        n = cat_var.get()
        if n:
            start_var.set(str(next_free_index(dest_folders[n], n, exts)))
            refresh_preview()

    def do_rename():
        n = cat_var.get()
        sel = selected_files()
        try:
            start = int(start_var.get())
        except ValueError:
            return
        plan, conflicts = plan_renames(dest_folders[n], sel, n, start)
        if conflicts:
            refresh_preview()
            return
        count = sum(1 for s, d in plan if s.name != d.name)
        if count == 0:
            messagebox.showinfo("批次序號命名", "選取的檔案名稱已經符合，沒有需要改名的檔案。", parent=root)
            return
        if not messagebox.askyesno(
            "確認改名",
            f"將把 [{n}] 資料夾內選取的 {count} 個檔案改名為 "
            f"{plan[0][1].name} … {plan[-1][1].name}。\n\n此動作無法一鍵復原，確定執行？",
            parent=root,
        ):
            return
        try:
            done = execute_renames(plan)
        except OSError as e:
            messagebox.showerror("改名失敗", f"改名過程發生錯誤，已盡力還原：\n{e}", parent=root)
            reload_lists()
            refresh_preview()
            return
        for src, _dst in plan:
            fresh.discard(str(src).lower())
        for i in listboxes[n].curselection():
            listboxes[n].selection_clear(i)
        reload_lists()
        start_var.set(str(next_free_index(dest_folders[n], n, exts)))
        status_var.set(f"✓ [{n}] 已改名 {done} 個檔案。可繼續選其他類別，或直接關閉視窗結束。")
        refresh_preview()

    spin.config(command=refresh_preview)
    spin.bind("<KeyRelease>", refresh_preview)

    reload_lists()
    refresh_preview()
    root.lift()
    root.attributes("-topmost", True)  # 由設定視窗子行程啟動時常被蓋在後面，先浮到最上層再放開
    root.after(400, lambda: root.attributes("-topmost", False))
    root.mainloop()


# ==================== 獨立子行程入口 ====================
def _main():
    """從 stdin 讀 JSON 參數後開視窗。回傳行程結束碼：0 正常、2 參數錯誤。"""
    if sys.stdin is None or sys.stdin.isatty():
        print("用法：本檔案由分類腳本以獨立子行程啟動，參數以 JSON 經 stdin 傳入"
              "（dest_folders／exts／fresh_files），不需要手動執行。", file=sys.stderr)
        return 2
    try:
        payload = json.loads(sys.stdin.read())
        dest_folders = {str(k): str(v) for k, v in dict(payload["dest_folders"]).items()}
        exts = {str(e).lower() for e in payload["exts"]}
        fresh_files = [str(p) for p in payload.get("fresh_files", [])]
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"⚠ 批次命名視窗：參數格式錯誤（{type(e).__name__}: {e}）", file=sys.stderr)
        return 2
    open_rename_dialog(dest_folders, exts, fresh_files)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
