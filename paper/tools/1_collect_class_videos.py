"""
同類行為影片聚集工具（GUI）
=======================================================
你會不斷地從各個來源分類出 walk/lick/scratch/shake/stop 五個行為資料夾（例如
1_classify_and_sort_videos.py 會在每個來源底下產生 class/<行為>/）。這支工具把「多個資料夾」裡
「同一個行為」的影片，一次聚集到你指定的單一輸出資料夾。執行後跳出視窗，全部在視窗裡操作：

  ① 選一個行為類別（一次只能選一類，不可混類別）
  ② 新增一個或多個輸入資料夾：按「新增資料夾」跳出可 Ctrl／Shift 一次多選的選擇視窗，或直接從檔案總管
     把資料夾（可一次拖多個）拖進視窗。每個資料夾都會「遞迴」往下查找名稱正好等於所選類別（小寫）的資料夾
     （資料夾本身也算，所以直接選到 …\\walk 也可以）
  ③ 選輸出資料夾：影片直接收進這個資料夾（一次只有一類，不再多包一層）
  ④ 選項：複製／搬移、保留原檔名／統一改成 <行為>_<序號>、是否略過內容完全相同的檔案
  → 先按「預覽」看會做什麼，確認後按「開始收納」；預覽顯示的就是實際會做的事。

整合前的全數驗證（預覽與開始收納都會做）：你選的「每一個」輸入資料夾，底下都必須至少找得到一個
名稱為小寫類別名（例如 walk）的資料夾，缺少任何一個就跳警告並拒絕整合，一個檔案都不會動。
也可以只選一個外層資料夾、讓它包住不同層的多個 walk 資料夾——那個外層資料夾找得到就通過，
底下所有層的 walk 都會被收進來。

同名檔案絕不覆蓋：不同資料夾裡的同名檔案會全部收進輸出資料夾，撞名的自動加 _2、_3 後綴
（命名選「統一改成 <行為>_<序號>」時，序號接在輸出資料夾既有最大序號之後）。
複製時先寫成 .part 暫存檔、大小核對無誤才改成正式檔名，中途中斷不會留下半支影片。

「略過內容完全相同的檔案」預設勾選：用「內容」（先比檔案大小、大小相同才算雜湊）判斷，
所以同一支影片不管檔名有沒有改過、從哪個資料夾來，第二次都不會被重複收進去，可以放心反覆執行。
取消勾選則連內容相同的也全部收（改名保留）。

每次有實際動作的執行，會在輸出資料夾留一份 collect_manifest_<時間>.csv，記錄每支影片
「從哪裡來、放到哪裡、做了什麼」，改名後仍可回溯來源。

只用標準函式庫，不載入 YOLO／torch：python 1_collect_class_videos.py
"""
import csv
import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# ==================== 預設值（視窗開啟時的初始選項，都可以在視窗裡改）====================
DEFAULT_MODE = "copy"      # "copy"＝複製（保留來源，預設，安全）／"move"＝搬移（來源檔被搬走）
DEFAULT_NAMING = "keep"    # "keep"＝保留原檔名（撞名加 _2 後綴）／"sequential"＝統一改成 <行為>_<序號>
DEFAULT_DEDUPE = True      # 略過「內容完全相同」的檔案（避免重跑時重複收進來）
MAX_SCAN_DEPTH = 32        # 遞迴查找的層數上限：實質上不限，只是防止 Windows 目錄連結造成的無窮迴圈；
                           # 不能設小——walk 藏得比上限深就會被誤判成「缺少」而拒絕整合
REQUIRE_LOWERCASE_FOLDER_NAME = True  # True＝資料夾名稱必須完全等於小寫類別名（Walk／WALK 不算）；False＝不分大小寫
# 常用資料夾捷徑（按鈕標籤, 路徑）：主視窗與多選視窗都會出現一鍵跳過去的按鈕，要加新的常用位置在這裡加一行
QUICK_FOLDERS = [("NVIDIA", str(Path.home() / "Videos" / "NVIDIA"))]

# 須與 cat_monitoring_system/utils/constants.py 的 BEHAVIOR_CLASSES 一致（tests 有檢查，不一致會失敗）；
# 這裡刻意不 import 專案模組，讓這支保持只靠標準函式庫、啟動快。
BEHAVIOR_CLASSES = ("walk", "lick", "scratch", "shake", "stop")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"}


# ==================== 純邏輯（不依賴 tkinter，可單獨測試）====================
@dataclass
class Item:
    behavior: str
    src: Path
    dst: Optional[Path]   # 略過的項目為 None
    action: str           # "copy"（要複製／搬移）或 "skip_duplicate"（內容已存在）
    note: str = ""


def _natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _is_inside(path, folder):
    """path 是否等於 folder 或位在 folder 底下（避免把輸出資料夾自己當成來源，聚集到自己身上）。"""
    try:
        p, f = Path(path).resolve(), Path(folder).resolve()
    except OSError:
        return False
    return p == f or f in p.parents


def find_class_dirs(folder, behavior, max_depth=MAX_SCAN_DEPTH, case_sensitive=REQUIRE_LOWERCASE_FOLDER_NAME):
    """在 folder 底下（含 folder 本身）遞迴找出名稱正好等於 behavior 的資料夾，依路徑自然排序。

    只比對完整資料夾名稱（walk 不會誤抓 walk_old），所以不會把別的類別混進來。
    case_sensitive=True 時名稱必須完全等於 behavior（小寫）；False 才不分大小寫。"""
    root = Path(folder)
    if not root.is_dir():
        return []
    if case_sensitive:
        matches = lambda name: name == behavior  # noqa: E731
    else:
        matches = lambda name: name.lower() == behavior.lower()  # noqa: E731
    found = [root] if matches(root.name) else []
    base_depth = len(root.parts)
    for dirpath, dirnames, _files in os.walk(root):
        if len(Path(dirpath).parts) - base_depth >= max_depth:
            dirnames[:] = []  # 已達最大深度，不再往下
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "$"))]  # 略過 .git、$RECYCLE.BIN 之類
        found.extend(Path(dirpath) / d for d in dirnames if matches(d))
    return sorted(found, key=lambda p: _natural_key(str(p)))


def unique_dirs(dirs):
    """依「解析後的實際路徑」去重（保留第一次出現的順序），同一個資料夾被多個輸入指到只算一次。"""
    seen, out = set(), []
    for d in dirs:
        try:
            key = str(Path(d).resolve()).lower()
        except OSError:
            key = str(d).lower()
        if key not in seen:
            seen.add(key)
            out.append(Path(d))
    return out


@dataclass
class CoverageReport:
    """整合前的全數驗證結果：每一個輸入資料夾底下是否都找得到目標類別資料夾。"""
    behavior: str
    case_sensitive: bool
    hits: dict          # {輸入資料夾: [找到的該類別資料夾, ...]}（依輸入順序）
    missing: list       # 底下找不到該類別資料夾的輸入資料夾（含根本不存在的資料夾）
    near_misses: dict   # {缺少的輸入資料夾: [大小寫不符的資料夾名稱, ...]}，只是給警告訊息當提示

    @property
    def ok(self):
        return not self.missing

    def class_dirs(self):
        """所有輸入資料夾找到的該類別資料夾（去重後）——這才是真正要收納的來源。"""
        return unique_dirs(d for found in self.hits.values() for d in found)


def check_coverage(folders, behavior, case_sensitive=REQUIRE_LOWERCASE_FOLDER_NAME):
    """逐一檢查「每一個」輸入資料夾底下（遞迴、含自己）是否至少有一個 behavior 資料夾。

    一個外層資料夾包住多個不同層的 behavior 資料夾也算通過（該資料夾找得到至少一個即可）。"""
    hits, missing, near_misses = {}, [], {}
    for folder in folders:
        folder = Path(folder)
        found = find_class_dirs(folder, behavior, case_sensitive=case_sensitive)
        hits[folder] = found
        if found:
            continue
        missing.append(folder)
        if case_sensitive and folder.is_dir():
            variants = find_class_dirs(folder, behavior, case_sensitive=False)
            if variants:
                near_misses[folder] = sorted({v.name for v in variants})
    return CoverageReport(behavior, case_sensitive, hits, missing, near_misses)


def format_missing_message(report, limit=10):
    """拒絕整合時顯示給使用者的警告文字。"""
    rule = "需為小寫、遞迴查找" if report.case_sensitive else "遞迴查找、不分大小寫"
    lines = [f"以下 {len(report.missing)} 個資料夾底下找不到名稱為「{report.behavior}」的資料夾（{rule}）："]
    for folder in report.missing[:limit]:
        if not folder.is_dir():
            extra = "（資料夾不存在）"
        elif folder in report.near_misses:
            extra = f"（有找到大小寫不符的：{'、'.join(report.near_misses[folder])}）"
        else:
            extra = ""
        lines.append(f"  • {folder}{extra}")
    if len(report.missing) > limit:
        lines.append(f"  …其餘 {len(report.missing) - limit} 個")
    lines += ["", "已拒絕整合，沒有動任何檔案。請移除這些資料夾，或改選正確的類別／資料夾後再試。"]
    return "\n".join(lines)


def list_videos(folder):
    p = Path(folder)
    if not p.is_dir():
        return []
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
    return sorted(files, key=lambda f: _natural_key(f.name))


def _digest(path, _chunk=1024 * 1024):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            block = f.read(_chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class _ContentIndex:
    """判斷某支影片的「內容」是否已經在一組檔案裡：先比大小（便宜），大小相同才算雜湊（昂貴）。"""

    def __init__(self):
        self._by_size = {}
        self._digests = {}

    def add(self, path):
        self._by_size.setdefault(path.stat().st_size, []).append(path)

    def _digest_of(self, path):
        if path not in self._digests:
            self._digests[path] = _digest(path)
        return self._digests[path]

    def find_duplicate(self, path):
        candidates = self._by_size.get(path.stat().st_size)
        if not candidates:
            return None
        mine = self._digest_of(path)
        for c in candidates:
            if self._digest_of(c) == mine:
                return c
        return None


def _max_serial(existing, behavior):
    pattern = re.compile(rf"^{re.escape(behavior)}_(\d+)$", re.IGNORECASE)
    used = [int(m.group(1)) for p in existing if (m := pattern.match(p.stem))]
    return max(used, default=0)


def _free_name(name, taken):
    """name 沒被佔用就原樣回傳，否則加 _2、_3… 後綴直到沒被佔用（比對不分大小寫）。"""
    if name.lower() not in taken:
        return name
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 2
    while f"{stem}_{n}{suffix}".lower() in taken:
        n += 1
    return f"{stem}_{n}{suffix}"


def plan_collect(class_dirs, dest_dir, behavior, naming="keep", dedupe=True):
    """規劃要做的事，不動任何檔案（預覽與實際執行共用同一份計畫，預覽結果才可信）。

    class_dirs：已找到的「該行為」資料夾清單（來源）；位在 dest_dir 內的會被排除。
    同名檔案一律不覆蓋：撞名（含輸出資料夾裡原有的檔案、以及這次其他來源已排定的檔名）自動改名。"""
    dest_dir = Path(dest_dir)
    existing = list(dest_dir.iterdir()) if dest_dir.is_dir() else []
    taken = {p.name.lower() for p in existing}  # 含 .part 等所有檔名，避免撞名
    index = _ContentIndex()
    if dedupe:
        for p in existing:
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                index.add(p)
    next_serial = _max_serial(existing, behavior) + 1

    items = []
    for class_dir in class_dirs:
        if _is_inside(class_dir, dest_dir):
            continue
        for video in list_videos(class_dir):
            if dedupe:
                dup = index.find_duplicate(video)
                if dup is not None:
                    items.append(Item(behavior, video, None, "skip_duplicate", f"內容與 {dup} 相同"))
                    continue
            if naming == "sequential":
                name = f"{behavior}_{next_serial}{video.suffix.lower()}"
                next_serial += 1
                name = _free_name(name, taken)
            else:
                name = _free_name(video.name, taken)
            taken.add(name.lower())
            if dedupe:
                index.add(video)  # 之後來源裡內容相同的影片會被認出是重複
            note = "" if name == video.name else f"改名為 {name}"
            items.append(Item(behavior, video, dest_dir / name, "copy", note))
    return items


def execute_item(item, mode):
    """複製（或搬移）一支影片，不覆蓋。失敗時丟 OSError，並清掉自己留下的暫存檔。"""
    dst = item.dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(dst):
        raise FileExistsError(f"目的地已存在，為避免覆蓋而略過：{dst}")

    if mode == "move":
        try:
            os.rename(item.src, dst)  # 同磁碟機直接改路徑，瞬間完成、不會複製大檔
            return
        except OSError:
            pass  # 跨磁碟機等情況，改走下面「複製 → 核對 → 刪來源」

    tmp = dst.with_name(dst.name + ".part")
    try:
        shutil.copy2(item.src, tmp)
        if tmp.stat().st_size != item.src.stat().st_size:
            raise OSError(f"複製後大小不符：{item.src}")
        os.rename(tmp, dst)
    except OSError:
        if os.path.lexists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    if mode == "move":
        os.remove(item.src)


def write_manifest(dest_dir, rows):
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    path = Path(dest_dir) / f"collect_manifest_{stamp}.csv"
    n = 2
    while path.exists():  # 同一秒內連跑也不覆蓋前一份
        path = Path(dest_dir) / f"collect_manifest_{stamp}_{n}.csv"
        n += 1
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["行為", "動作", "來源", "目的地", "備註"])
        writer.writerows(rows)
    return path


# ==================== GUI ====================
# 配色沿用 1_classify_and_sort_images.py 分類視窗與 _rename_gui.py 命名視窗的深色石板色系。
_C_BG = "#111827"
_C_CARD = "#1F2937"
_C_BORDER = "#374151"
_C_LIST_BG = "#0B1220"
_C_FG = "#F9FAFB"
_C_TEXT = "#D1D5DB"
_C_MUTED = "#9CA3AF"
_C_DIM = "#6B7280"
_C_WARN = "#F87171"
_C_OK = "#4ADE80"
_C_PRIMARY = "#16A34A"
_C_PRIMARY_HOVER = "#22C55E"
_C_SECONDARY = "#374151"
_C_SECONDARY_HOVER = "#4B5563"
_C_DANGER = "#B91C1C"
_C_DANGER_HOVER = "#DC2626"
_FONT = "Microsoft JhengHei UI"
_ACCENTS = {"walk": "#60A5FA", "lick": "#F472B6", "scratch": "#FBBF24", "shake": "#A78BFA", "stop": "#34D399"}


def _make_button(parent, text, command, bg=_C_SECONDARY, hover=_C_SECONDARY_HOVER, bold=False):
    import tkinter as tk

    btn = tk.Button(
        parent, text=text, command=command, font=(_FONT, 10, "bold" if bold else "normal"),
        fg=_C_FG, bg=bg, activeforeground=_C_FG, activebackground=hover, relief="flat", bd=0,
        padx=14, pady=6, cursor="hand2", disabledforeground=_C_DIM,
    )
    btn._normal_bg, btn._hover_bg = bg, hover
    btn.bind("<Enter>", lambda _e, b=btn: b.config(bg=b._hover_bg) if str(b.cget("state")) == "normal" else None)
    btn.bind("<Leave>", lambda _e, b=btn: b.config(bg=b._normal_bg) if str(b.cget("state")) == "normal" else None)
    return btn


def _setup_styles(style):
    """設定 ttk 元件的深色樣式（捲軸、進度條、資料夾樹），主視窗與資料夾選擇視窗共用。"""
    style.configure(
        "Dark.Vertical.TScrollbar", background=_C_BORDER, troughcolor=_C_LIST_BG, bordercolor=_C_LIST_BG,
        arrowcolor=_C_MUTED, lightcolor=_C_BORDER, darkcolor=_C_BORDER, relief="flat",
    )
    style.map("Dark.Vertical.TScrollbar", background=[("active", _C_SECONDARY_HOVER)])
    style.configure(
        "Green.Horizontal.TProgressbar", troughcolor=_C_LIST_BG, background=_C_PRIMARY,
        bordercolor=_C_LIST_BG, lightcolor=_C_PRIMARY, darkcolor=_C_PRIMARY,
    )
    style.configure(
        "Dark.Treeview", background=_C_LIST_BG, fieldbackground=_C_LIST_BG, foreground=_C_TEXT,
        borderwidth=0, rowheight=26, font=(_FONT, 10),
    )
    style.map("Dark.Treeview", background=[("selected", "#2563EB")], foreground=[("selected", "#FFFFFF")])
    style.layout("Dark.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])  # 去掉外框與對焦虛線


def _list_drives():
    """Windows 的磁碟機根目錄清單（例如 ['C:\\', 'D:\\']）；其他平台回傳 ['/']。"""
    if os.name != "nt":
        return ["/"]
    try:
        import ctypes

        mask = ctypes.windll.kernel32.GetLogicalDrives()
        drives = [f"{chr(65 + i)}:\\" for i in range(26) if mask & (1 << i)]
        if drives:
            return drives
    except Exception:  # noqa: BLE001
        pass
    return [f"{c}:\\" for c in "CDEFGH" if os.path.exists(f"{c}:\\")]


def _list_subdirs(path):
    """path 底下的子資料夾名稱（自然排序）；略過 . 與 $ 開頭的隱藏／系統資料夾，讀不到就當作沒有。"""
    names = []
    try:
        with os.scandir(path) as entries:
            for e in entries:
                try:
                    if e.is_dir(follow_symlinks=False) and not e.name.startswith((".", "$")):
                        names.append(e.name)
                except OSError:
                    continue
    except OSError:
        return []
    return sorted(names, key=_natural_key)


_DUMMY = "::dummy::"  # 樹狀清單裡「尚未展開」節點的佔位子節點後綴（冒號不可能出現在 Windows 路徑名稱裡）


def pick_folders(parent, initialdir=None):
    """可「一次選多個資料夾」的選擇視窗（tkinter 內建的系統對話框只能單選）。

    左鍵點選、Ctrl／Shift 多選；雙擊或點箭頭展開。回傳選到的資料夾路徑 list（依路徑自然排序），
    取消或直接關閉視窗回傳 []。純 tkinter 實作，不需要額外套件。"""
    import tkinter as tk
    from tkinter import ttk

    dlg = tk.Toplevel(parent)
    dlg.title("選擇資料夾（可多選）")
    dlg.minsize(560, 440)
    dlg.configure(bg=_C_BG)
    dlg.transient(parent)
    parent.update_idletasks()
    x = parent.winfo_rootx() + max(20, (parent.winfo_width() - 780) // 2)
    y = parent.winfo_rooty() + 40
    dlg.geometry(f"780x620+{x}+{y}")
    _setup_styles(ttk.Style(dlg))

    result = []
    path_var = tk.StringVar()
    msg_var = tk.StringVar(value="")
    count_var = tk.StringVar(value="尚未選取")

    top = tk.Frame(dlg, bg=_C_BG)
    top.pack(fill="x", padx=14, pady=(14, 4))
    tk.Label(top, text="位置", font=(_FONT, 10), fg=_C_MUTED, bg=_C_BG).pack(side="left")
    entry = tk.Entry(
        top, textvariable=path_var, font=(_FONT, 10), bg=_C_LIST_BG, fg=_C_FG, insertbackground=_C_FG,
        relief="flat", highlightthickness=1, highlightbackground=_C_BORDER, highlightcolor=_C_MUTED,
    )
    entry.pack(side="left", fill="x", expand=True, padx=8, ipady=4)
    tk.Label(
        dlg, text="按住 Ctrl 或 Shift 可一次選取多個資料夾；也可以直接輸入路徑後按「前往」。",
        font=(_FONT, 9), fg=_C_MUTED, bg=_C_BG, anchor="w",
    ).pack(fill="x", padx=16, pady=(0, 4))
    quick = tk.Frame(dlg, bg=_C_BG)  # 常用位置捷徑列，按鈕在 reveal 定義好之後才建立
    quick.pack(fill="x", padx=14, pady=(0, 2))

    body = tk.Frame(dlg, bg=_C_LIST_BG)
    body.pack(fill="both", expand=True, padx=14, pady=6)
    sb = ttk.Scrollbar(body, orient="vertical", style="Dark.Vertical.TScrollbar")
    tree = ttk.Treeview(body, show="tree", selectmode="extended", style="Dark.Treeview", yscrollcommand=sb.set)
    sb.config(command=tree.yview)
    sb.pack(side="right", fill="y")
    tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)

    def add_node(parent_iid, path, text):
        tree.insert(parent_iid, "end", iid=path, text=text)
        tree.insert(path, "end", iid=path + _DUMMY, text="…")  # 先放佔位子節點，樹才會顯示展開箭頭；展開時才真的去讀

    def populate(iid):
        kids = tree.get_children(iid)
        if len(kids) == 1 and kids[0].endswith(_DUMMY):
            tree.delete(kids[0])
            for name in _list_subdirs(iid):
                add_node(iid, os.path.join(iid, name), name)

    def reveal(path):
        """展開到 path 並選取它；找不到（不存在或不在樹裡）回傳 False。"""
        p = os.path.abspath(path)
        if os.name == "nt" and len(p) > 1 and p[1] == ":":
            p = p[0].upper() + p[1:]
        if not os.path.isdir(p):
            return False
        target = Path(p)
        for ancestor in list(reversed(target.parents)) + [target]:
            iid = str(ancestor)
            if not tree.exists(iid):
                return False
            populate(iid)
            tree.item(iid, open=True)  # 連目標本身也展開，開啟後就能直接挑它底下的子資料夾
        tree.selection_set(str(target))
        tree.focus(str(target))
        tree.see(str(target))
        path_var.set(str(target))
        return True

    def real_selection():
        return [i for i in tree.selection() if not i.endswith(_DUMMY) and os.path.isdir(i)]

    def on_select(_e=None):
        n = len(real_selection())
        count_var.set(f"已選取 {n} 個資料夾" if n else "尚未選取")
        msg_var.set("")

    def on_open(_e=None):
        iid = tree.focus()
        if iid:
            populate(iid)

    def goto(_e=None):
        if not reveal(path_var.get().strip()):
            msg_var.set("找不到這個資料夾")
        return "break"

    def go_up():
        sel = real_selection() or ([tree.focus()] if tree.focus() else [])
        if sel:
            parent_dir = os.path.dirname(sel[0].rstrip("\\/"))
            if parent_dir and not reveal(parent_dir):
                msg_var.set("已經是最上層")

    def ok(_e=None):
        result.extend(sorted(set(real_selection()), key=_natural_key))
        dlg.destroy()

    def cancel(_e=None):
        dlg.destroy()

    def jump_to(path):
        if not reveal(path):
            msg_var.set(f"找不到 {path}")

    _make_button(top, "前往", goto).pack(side="left")
    _make_button(top, "上一層", go_up).pack(side="left", padx=(6, 0))
    if QUICK_FOLDERS:
        tk.Label(quick, text="常用位置", font=(_FONT, 9), fg=_C_MUTED, bg=_C_BG).pack(side="left", padx=(2, 8))
        for label, qpath in QUICK_FOLDERS:
            _make_button(quick, f"⚡ {label}", lambda p=qpath: jump_to(p)).pack(side="left", padx=(0, 6))

    bottom = tk.Frame(dlg, bg=_C_BG)
    bottom.pack(fill="x", padx=14, pady=(4, 14))
    tk.Label(bottom, textvariable=count_var, font=(_FONT, 10, "bold"), fg=_C_OK, bg=_C_BG).pack(side="left")
    tk.Label(bottom, textvariable=msg_var, font=(_FONT, 9), fg=_C_WARN, bg=_C_BG).pack(side="left", padx=12)
    ok_btn = _make_button(bottom, "確定", ok, _C_PRIMARY, _C_PRIMARY_HOVER, bold=True)
    ok_btn.pack(side="right")
    _make_button(bottom, "取消", cancel).pack(side="right", padx=(0, 8))

    for drive in _list_drives():
        add_node("", drive, drive)
    tree.bind("<<TreeviewOpen>>", on_open)
    tree.bind("<<TreeviewSelect>>", on_select)
    tree.bind("<Double-Button-1>", lambda e: None)  # 雙擊維持預設的展開／收合
    entry.bind("<Return>", goto)
    dlg.bind("<Escape>", cancel)
    dlg.protocol("WM_DELETE_WINDOW", cancel)
    dlg.bind("<Control-Return>", ok)

    start = initialdir if initialdir and os.path.isdir(str(initialdir)) else str(Path.home())
    reveal(start)
    try:
        dlg.wait_visibility()
        dlg.grab_set()
    except tk.TclError:
        pass
    tree.focus_set()
    parent.wait_window(dlg)
    return result


def enable_folder_drop(widget, on_drop):
    """Windows：讓視窗接受從檔案總管拖進來的檔案／資料夾。純 ctypes、不需要額外套件。

    on_drop(paths) 會在 UI 執行緒被呼叫，paths 是 list[str]。回傳是否成功啟用；任何一步出錯都只回傳
    False（拖曳功能停用，其餘功能不受影響），不會丟例外。非 Windows 直接回傳 False。"""
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32, shell32 = ctypes.windll.user32, ctypes.windll.shell32
        WM_DROPFILES, WM_COPYGLOBALDATA, GWL_WNDPROC, MSGFLT_ALLOW = 0x0233, 0x0049, -4, 1
        LRESULT = ctypes.c_ssize_t
        WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

        set_long = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW  # 32 位元 Python 沒有 ...Ptr 版
        set_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
        set_long.restype = ctypes.c_void_p
        user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.CallWindowProcW.restype = LRESULT
        shell32.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]
        shell32.DragQueryFileW.argtypes = [ctypes.c_void_p, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
        shell32.DragQueryFileW.restype = wintypes.UINT
        shell32.DragFinish.argtypes = [ctypes.c_void_p]

        widget.update_idletasks()
        hwnd = widget.winfo_id()
        original = {"proc": None}

        def wndproc(h, msg, wparam, lparam):
            if msg == WM_DROPFILES:
                try:
                    count = shell32.DragQueryFileW(wparam, 0xFFFFFFFF, None, 0)
                    paths = []
                    for i in range(count):
                        length = shell32.DragQueryFileW(wparam, i, None, 0)
                        buf = ctypes.create_unicode_buffer(length + 1)
                        shell32.DragQueryFileW(wparam, i, buf, length + 1)
                        paths.append(buf.value)
                    on_drop(paths)
                except Exception:  # noqa: BLE001 — 視窗程序裡絕不能讓例外逃出去
                    pass
                finally:
                    shell32.DragFinish(wparam)
                return 0
            return user32.CallWindowProcW(original["proc"], h, msg, wparam, lparam)

        new_proc = WNDPROC(wndproc)
        widget._folder_drop_proc = new_proc  # 一定要留著參照，否則被回收後視窗一收到訊息就會當掉
        original["proc"] = set_long(hwnd, GWL_WNDPROC, ctypes.cast(new_proc, ctypes.c_void_p))
        shell32.DragAcceptFiles(hwnd, True)
        try:  # 以系統管理員身分執行時，UIPI 預設會擋掉從一般檔案總管拖進來的訊息，這裡放行
            user32.ChangeWindowMessageFilterEx.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.DWORD, ctypes.c_void_p]
            for m in (WM_DROPFILES, WM_COPYGLOBALDATA):
                user32.ChangeWindowMessageFilterEx(hwnd, m, MSGFLT_ALLOW, None)
        except Exception:  # noqa: BLE001
            pass
        return bool(original["proc"])
    except Exception:  # noqa: BLE001
        return False


def run_gui():
    import queue
    import threading
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(f"❌ 無法開啟視窗（{e}）")
        return

    root.title("同類行為影片聚集")
    root.geometry("1120x980")
    root.minsize(920, 700)
    root.configure(bg=_C_BG)

    input_dirs = []                                  # 輸入資料夾（依加入順序）
    state = {"plan": None, "sig": None, "busy": False, "dest": None, "last_dir": None}
    q = queue.Queue()                                # 背景執行緒 → 主執行緒的訊息
    cancel = threading.Event()

    behavior_var = tk.StringVar(value="")
    output_var = tk.StringVar(value="")
    mode_var = tk.StringVar(value=DEFAULT_MODE)
    naming_var = tk.StringVar(value=DEFAULT_NAMING)
    dedupe_var = tk.BooleanVar(value=DEFAULT_DEDUPE)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    _setup_styles(style)

    # ---------- 小工具 ----------
    make_button = _make_button

    def set_enabled(btn, enabled):
        if enabled:
            btn.config(state="normal", bg=btn._normal_bg)
        else:
            btn.config(state="disabled", bg=_C_SECONDARY)

    def make_card(step, title, hint=None):
        card = tk.Frame(root, bg=_C_CARD, highlightthickness=1, highlightbackground=_C_BORDER)
        card.pack(fill="x", padx=18, pady=3)
        head = tk.Frame(card, bg=_C_CARD)
        head.pack(fill="x", padx=14, pady=(8, 2))
        tk.Label(
            head, text=step, font=(_FONT, 10, "bold"), fg=_C_LIST_BG, bg=_C_MUTED, padx=8, pady=1,
        ).pack(side="left")
        tk.Label(head, text=title, font=(_FONT, 11, "bold"), fg=_C_FG, bg=_C_CARD).pack(side="left", padx=(8, 0))
        if hint:
            tk.Label(
                card, text=hint, font=(_FONT, 9), fg=_C_MUTED, bg=_C_CARD, anchor="w", justify="left", wraplength=1040,
            ).pack(fill="x", padx=14)
        body = tk.Frame(card, bg=_C_CARD)
        body.pack(fill="x", padx=14, pady=(4, 8))
        return body

    # ---------- 標題 ----------
    header = tk.Frame(root, bg=_C_BG)
    header.pack(fill="x", padx=18, pady=(12, 4))
    tk.Label(header, text="同類行為影片聚集", font=(_FONT, 16, "bold"), fg=_C_FG, bg=_C_BG).pack(anchor="w")
    tk.Label(
        header, text="把多個資料夾裡「同一個行為」的影片收進同一個輸出資料夾。同名檔案絕不覆蓋，會自動改名一起收進去。",
        font=(_FONT, 9), fg=_C_MUTED, bg=_C_BG, anchor="w",
    ).pack(fill="x", pady=(2, 0))

    # ---------- ① 行為類別 ----------
    body1 = make_card("1", "選擇行為類別", "一次只能選一類（不可混類別）。之後只會查找、收納這一類的影片。")
    class_buttons = {}
    for b in BEHAVIOR_CLASSES:
        rb = tk.Radiobutton(
            body1, text=b, value=b, variable=behavior_var, indicatoron=False, font=(_FONT, 11, "bold"),
            fg=_C_FG, bg=_C_LIST_BG, selectcolor=_ACCENTS[b], activebackground=_C_SECONDARY_HOVER,
            activeforeground=_C_FG, relief="flat", bd=0, padx=22, pady=8, cursor="hand2", highlightthickness=0,
        )
        rb.pack(side="left", padx=(0, 8))
        class_buttons[b] = rb

    def restyle_class_buttons(*_):
        for b, rb in class_buttons.items():
            rb.config(fg=_C_LIST_BG if behavior_var.get() == b else _C_FG)

    # ---------- ② 輸入資料夾 ----------
    body2 = make_card(
        "2", "新增輸入資料夾（可一次選多個，或直接拖曳進來）",
        "按「新增資料夾」用Ctrl／Shift一次選多個，或從檔案總管把資料夾（可一次拖多個）拖進這個視窗。"
        "每個輸入資料夾底下（遞迴查找，含資料夾本身）都必須至少有一個名稱為小寫類別名的資料夾，"
        "缺少任何一個就會警告並拒絕整合；也可以只選一個外層資料夾，讓它包住不同層的多個資料夾。"
        "例如類別選 walk：選到 …\\istock 會找出 …\\istock\\class\\walk；直接選到 …\\walk 也可以。",
    )
    list_holder = tk.Frame(body2, bg=_C_LIST_BG)
    list_holder.pack(side="left", fill="both", expand=True)
    in_sb = ttk.Scrollbar(list_holder, orient="vertical", style="Dark.Vertical.TScrollbar")
    in_list = tk.Listbox(
        list_holder, height=4, selectmode="extended", exportselection=False, activestyle="none",
        bg=_C_LIST_BG, fg=_C_TEXT, font=(_FONT, 10), relief="flat", borderwidth=0, highlightthickness=0,
        selectbackground=_C_SECONDARY_HOVER, selectforeground=_C_FG, yscrollcommand=in_sb.set,
    )
    in_sb.config(command=in_list.yview)
    in_sb.pack(side="right", fill="y")
    in_list.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=6)
    drop_hint = tk.Label(
        list_holder, text="把資料夾從檔案總管拖曳到這裡（可一次拖多個），或按右邊「新增資料夾」",
        font=(_FONT, 10), fg=_C_DIM, bg=_C_LIST_BG,
    )
    in_btns = tk.Frame(body2, bg=_C_CARD)
    in_btns.pack(side="left", fill="y", padx=(10, 0))

    # ---------- ③ 輸出資料夾 ----------
    body3 = make_card("3", "選擇輸出資料夾", "影片直接收進這個資料夾（不會再多包一層）。資料夾不存在會自動建立。")
    out_entry = tk.Entry(
        body3, textvariable=output_var, font=(_FONT, 10), bg=_C_LIST_BG, fg=_C_FG, insertbackground=_C_FG,
        relief="flat", highlightthickness=1, highlightbackground=_C_BORDER, highlightcolor=_C_MUTED,
    )
    out_entry.pack(side="left", fill="x", expand=True, ipady=6)

    # ---------- ④ 選項 ----------
    body4 = make_card("4", "選項")

    def option_group(label, var, choices):
        g = tk.Frame(body4, bg=_C_CARD)
        g.pack(fill="x", pady=2)
        tk.Label(g, text=label, font=(_FONT, 10), fg=_C_MUTED, bg=_C_CARD, width=10, anchor="w").pack(side="left")
        for text, value in choices:
            tk.Radiobutton(
                g, text=text, value=value, variable=var, font=(_FONT, 10), fg=_C_FG, bg=_C_CARD,
                selectcolor=_C_LIST_BG, activebackground=_C_CARD, activeforeground=_C_FG, highlightthickness=0,
                cursor="hand2",
            ).pack(side="left", padx=(0, 16))

    option_group("處理方式", mode_var, [("複製（保留原檔）", "copy"), ("搬移（原檔會被搬走）", "move")])
    option_group("命名", naming_var, [("保留原檔名（撞名自動加 _2）", "keep"), ("統一改成 <行為>_<序號>", "sequential")])
    dedupe_row = tk.Frame(body4, bg=_C_CARD)
    dedupe_row.pack(fill="x", pady=2)
    tk.Label(dedupe_row, text="重複檔案", font=(_FONT, 10), fg=_C_MUTED, bg=_C_CARD, width=10, anchor="w").pack(side="left")
    tk.Checkbutton(
        dedupe_row, text="略過內容完全相同的檔案（避免重跑時重複收進來；取消勾選則全部收、撞名改名）",
        variable=dedupe_var, font=(_FONT, 10), fg=_C_FG, bg=_C_CARD, selectcolor=_C_LIST_BG,
        activebackground=_C_CARD, activeforeground=_C_FG, highlightthickness=0, cursor="hand2",
    ).pack(side="left")

    # ---------- 動作列 ----------
    actions = tk.Frame(root, bg=_C_BG)
    actions.pack(fill="x", padx=18, pady=(8, 4))
    preview_btn = make_button(actions, "預覽", lambda: on_preview())
    preview_btn.pack(side="left")
    start_btn = make_button(actions, "開始收納", lambda: on_start(), _C_PRIMARY, _C_PRIMARY_HOVER, bold=True)
    start_btn.pack(side="left", padx=(8, 0))
    cancel_btn = make_button(actions, "取消", lambda: on_cancel(), _C_DANGER, _C_DANGER_HOVER)
    cancel_btn.pack(side="left", padx=(8, 0))
    progress = ttk.Progressbar(actions, mode="determinate", style="Green.Horizontal.TProgressbar")
    progress.pack(side="left", fill="x", expand=True, padx=(16, 0))
    set_enabled(start_btn, False)
    set_enabled(cancel_btn, False)

    # ---------- 紀錄／預覽區 ----------
    log_card = tk.Frame(root, bg=_C_CARD, highlightthickness=1, highlightbackground=_C_BORDER)
    log_card.pack(fill="both", expand=True, padx=18, pady=(4, 4))
    tk.Label(log_card, text="預覽／紀錄", font=(_FONT, 9, "bold"), fg=_C_MUTED, bg=_C_CARD, anchor="w").pack(
        fill="x", padx=14, pady=(8, 4)
    )
    log_body = tk.Frame(log_card, bg=_C_LIST_BG)
    log_body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    log_sb = ttk.Scrollbar(log_body, orient="vertical", style="Dark.Vertical.TScrollbar")
    log = tk.Text(
        log_body, height=10, state="disabled", wrap="none", font=("Consolas", 10), bg=_C_LIST_BG, fg=_C_FG,
        relief="flat", borderwidth=0, highlightthickness=0, padx=12, pady=8, yscrollcommand=log_sb.set, cursor="arrow",
    )
    log_sb.config(command=log.yview)
    log_sb.pack(side="right", fill="y")
    log.pack(side="left", fill="both", expand=True)
    for tag, color in (("plain", _C_FG), ("muted", _C_MUTED), ("dim", _C_DIM), ("ok", _C_OK), ("warn", _C_WARN),
                       ("head", _C_FG), ("info", "#93C5FD")):
        log.tag_config(tag, foreground=color)
    log.tag_config("head", font=("Consolas", 10, "bold"))

    status_var = tk.StringVar(value="")
    status_label = tk.Label(root, textvariable=status_var, font=(_FONT, 9), fg=_C_OK, bg=_C_BG, anchor="w")
    status_label.pack(fill="x", padx=20, pady=(0, 10))
    # 「✗」開頭的訊息（例如拒絕整合）用紅色，其餘維持綠色，避免拒絕訊息看起來像成功
    status_var.trace_add("write", lambda *_: status_label.config(fg=_C_WARN if status_var.get().startswith("✗") else _C_OK))

    def append_log(text, tag="plain"):
        log.config(state="normal")
        log.insert("end", text + "\n", tag)
        log.see("end")
        log.config(state="disabled")

    def clear_log():
        log.config(state="normal")
        log.delete("1.0", "end")
        log.config(state="disabled")

    # ---------- 狀態管理 ----------
    def signature():
        return (behavior_var.get(), tuple(str(d).lower() for d in input_dirs), output_var.get().strip(),
                naming_var.get(), dedupe_var.get())

    def invalidate(*_):
        """任何影響計畫的設定被改動，就作廢先前的預覽——「開始收納」必須對應一份最新的預覽。"""
        if state["busy"]:
            return
        state["plan"] = None
        set_enabled(start_btn, False)

    input_widgets = []

    def set_busy(busy):
        state["busy"] = busy
        for w in input_widgets:
            w.config(state="disabled" if busy else "normal")
        set_enabled(preview_btn, not busy)
        if busy:
            set_enabled(start_btn, False)
        set_enabled(cancel_btn, busy)

    def refresh_input_list():
        in_list.delete(0, "end")
        for d in input_dirs:
            in_list.insert("end", str(d))
        if input_dirs:
            drop_hint.place_forget()
        else:
            drop_hint.place(relx=0.5, rely=0.5, anchor="center")
        invalidate()

    # ---------- 輸入資料夾操作 ----------
    def add_paths(paths):
        """把一批路徑加進輸入清單（按鈕多選與拖曳共用）：略過不是資料夾的、已經加入過的，並回報結果。"""
        existing = {str(d).lower() for d in input_dirs}
        added = duplicated = skipped = 0
        for raw in paths:
            p = Path(os.path.normpath(raw))
            if not p.is_dir():
                skipped += 1
            elif str(p).lower() in existing:
                duplicated += 1
            else:
                existing.add(str(p).lower())
                input_dirs.append(p)
                state["last_dir"] = str(p.parent)
                added += 1
        if added:
            refresh_input_list()
        parts = [f"已加入 {added} 個資料夾"]
        if duplicated:
            parts.append(f"{duplicated} 個已經在清單裡")
        if skipped:
            parts.append(f"{skipped} 個不是資料夾而略過")
        status_var.set("，".join(parts) + "。")

    def add_folder(start_dir=None):
        try:
            chosen = pick_folders(root, start_dir or state["last_dir"])
        except Exception:  # noqa: BLE001 — 多選視窗萬一壞掉，退回系統的單選對話框，工具本身照常能用
            single = filedialog.askdirectory(title="選擇輸入資料夾", mustexist=True, parent=root)
            chosen = [single] if single else []
        if chosen:
            add_paths(chosen)

    def remove_selected():
        for i in reversed(in_list.curselection()):
            del input_dirs[i]
        refresh_input_list()

    def clear_inputs():
        input_dirs.clear()
        refresh_input_list()

    def browse_output():
        chosen = filedialog.askdirectory(title="選擇輸出資料夾", parent=root)
        if chosen:
            output_var.set(os.path.normpath(chosen))

    add_btn = make_button(in_btns, "＋ 新增資料夾（可多選）…", add_folder)
    add_btn.pack(fill="x", pady=(6, 4))
    quick_btns = []
    for label, qpath in QUICK_FOLDERS:
        qb = make_button(in_btns, f"⚡ {label}（一鍵開啟多選）", lambda p=qpath: add_folder(p))
        qb.pack(fill="x", pady=4)
        quick_btns.append(qb)
    rm_btn = make_button(in_btns, "－ 移除選取", remove_selected)
    rm_btn.pack(fill="x", pady=4)
    clr_btn = make_button(in_btns, "清空", clear_inputs)
    clr_btn.pack(fill="x", pady=4)
    browse_btn = make_button(body3, "瀏覽…", browse_output)
    browse_btn.pack(side="left", padx=(8, 0))
    input_widgets.extend([add_btn, *quick_btns, rm_btn, clr_btn, browse_btn, out_entry])

    # ---------- 預覽 ----------
    def validate():
        behavior = behavior_var.get()
        if behavior not in BEHAVIOR_CLASSES:
            messagebox.showwarning("尚未選擇類別", "請先在 ① 選擇要收納的行為類別。", parent=root)
            return None
        if not input_dirs:
            messagebox.showwarning("尚未新增輸入資料夾", "請先在 ② 新增至少一個輸入資料夾。", parent=root)
            return None
        dest_text = output_var.get().strip()
        if not dest_text:
            messagebox.showwarning("尚未選擇輸出資料夾", "請先在 ③ 選擇輸出資料夾。", parent=root)
            return None
        return behavior, list(input_dirs), Path(dest_text), naming_var.get(), dedupe_var.get(), signature()

    def preview_worker(behavior, dirs, dest, naming, dedupe, sig):
        try:
            q.put(("log", f"=== 預覽：類別 {behavior}，輸出 → {dest}", "head"))
            # 整合前先逐一驗證：每個輸入資料夾底下都要找得到該類別資料夾，缺一個就拒絕
            report = check_coverage(dirs, behavior)
            for folder, hits in report.hits.items():
                if hits:
                    q.put(("log", f"✓ {folder}　→　找到 {len(hits)} 個「{behavior}」資料夾", "info"))
                elif not folder.is_dir():
                    q.put(("log", f"✗ {folder}　→　資料夾不存在", "warn"))
                else:
                    q.put(("log", f"✗ {folder}　→　找不到「{behavior}」資料夾", "warn"))
            if not report.ok:
                q.put(("blocked", format_missing_message(report)))
                return
            class_dirs = report.class_dirs()
            usable = []
            for d in class_dirs:
                inside = _is_inside(d, dest)
                n = len(list_videos(d))
                q.put(("log", f"  {'（位於輸出資料夾內，略過）' if inside else '・'} {d}　[{n} 支影片]",
                       "dim" if inside else "muted"))
                if not inside:
                    usable.append(d)
            if not usable:
                q.put(("plan_done", None, sig,
                       f"找到的「{behavior}」資料夾全都位在輸出資料夾內，沒有可收納的來源。請改選不同的輸出資料夾。"))
                return
            items = plan_collect(usable, dest, behavior, naming, dedupe)
            q.put(("plan_done", items, sig, None))
        except Exception as exc:  # noqa: BLE001 — 背景執行緒內任何錯誤都要回報到畫面，不能悄悄消失
            q.put(("plan_done", None, sig, f"預覽失敗：{type(exc).__name__}: {exc}"))

    def on_preview():
        args = validate()
        if args is None:
            return
        behavior, dirs, dest, naming, dedupe, sig = args
        clear_log()
        status_var.set("正在查找與規劃…（影片很多或大小相同時，比對內容會花一點時間）")
        progress.config(value=0)
        cancel.clear()
        set_busy(True)
        state["plan"] = None
        threading.Thread(target=preview_worker, args=(behavior, dirs, dest, naming, dedupe, sig), daemon=True).start()

    def on_plan_done(items, sig, message):
        set_busy(False)
        if signature() != sig:  # 預覽途中使用者又改了設定，這份結果已經過期
            status_var.set("設定在預覽期間被更改，請重新按「預覽」。")
            return
        if items is None:
            append_log(f"⚠ {message}", "warn")
            status_var.set("")
            return
        to_copy = [it for it in items if it.action == "copy"]
        dups = [it for it in items if it.action == "skip_duplicate"]
        verb = "搬移" if mode_var.get() == "move" else "複製"
        shown = 400
        append_log("", "plain")
        for it in items[:shown]:
            if it.action == "skip_duplicate":
                append_log(f"  = {it.src.name}　已收納過（{it.note}），略過", "dim")
            else:
                label = f"{it.src.name}  →  {it.dst.name}" if it.note else it.src.name
                append_log(f"  ○ 將{verb}：{label}", "plain")
        if len(items) > shown:
            append_log(f"  …（其餘 {len(items) - shown} 項未逐一列出）", "dim")
        append_log("", "plain")
        append_log(f"預覽結果：將{verb} {len(to_copy)} 支、內容重複略過 {len(dups)} 支。"
                   + ("　確認無誤後按「開始收納」。" if to_copy else "　沒有需要收納的檔案。"), "head")
        renamed = sum(1 for it in to_copy if it.note)
        if renamed:
            append_log(f"其中 {renamed} 支因撞名會自動改名，不會覆蓋任何檔案。", "muted")
        state["plan"], state["sig"], state["dest"] = items, sig, Path(output_var.get().strip())
        set_enabled(start_btn, bool(to_copy))
        status_var.set("")

    # ---------- 執行 ----------
    def run_worker(items, mode, dest, behavior, dirs):
        rows, ok_count, failed, cancelled = [], 0, 0, False
        to_copy = [it for it in items if it.action == "copy"]
        verb = "搬移" if mode == "move" else "複製"
        try:
            # 預覽之後檔案系統可能又變動過（資料夾被搬走／改名），動任何檔案之前再驗證一次
            report = check_coverage(dirs, behavior)
            if not report.ok:
                q.put(("blocked", format_missing_message(report)))
                return
            dest.mkdir(parents=True, exist_ok=True)
            done = 0
            for it in items:
                if it.action == "skip_duplicate":
                    rows.append([it.behavior, "skip_duplicate", str(it.src), "", it.note])
                    continue
                if cancel.is_set():
                    cancelled = True
                    q.put(("log", "⚠ 已取消，剩餘項目未處理。", "warn"))
                    break
                label = f"{it.src.name}  →  {it.dst.name}" if it.note else it.src.name
                try:
                    execute_item(it, mode)
                except OSError as e:
                    failed += 1
                    q.put(("log", f"  ✗ {verb}失敗：{it.src.name}（{e}）", "warn"))
                    rows.append([it.behavior, "failed", str(it.src), str(it.dst), str(e)])
                else:
                    ok_count += 1
                    q.put(("log", f"  ✓ 已{verb}：{label}", "ok"))
                    rows.append([it.behavior, mode, str(it.src), str(it.dst), it.note])
                done += 1
                q.put(("progress", done, len(to_copy)))
            manifest = None
            if any(r[1] != "skip_duplicate" for r in rows):
                manifest = write_manifest(dest, rows)
            q.put(("run_done", ok_count, failed, cancelled, manifest, verb))
        except Exception as exc:  # noqa: BLE001
            q.put(("log", f"⚠ 執行中斷：{type(exc).__name__}: {exc}", "warn"))
            q.put(("run_done", ok_count, failed, True, None, verb))

    def on_start():
        items = state["plan"]
        if not items or state["sig"] != signature():
            status_var.set("請先按「預覽」取得最新的計畫。")
            return
        mode = mode_var.get()
        to_copy = [it for it in items if it.action == "copy"]
        dest = state["dest"]
        if mode == "move":
            message = (f"將「搬移」{len(to_copy)} 支影片到\n{dest}\n\n來源資料夾裡的原檔會被搬走。"
                       "\n同名檔案不會覆蓋（會自動改名）。\n\n確定開始？")
        else:
            message = (f"將複製 {len(to_copy)} 支影片到\n{dest}\n\n來源檔案不會被更動；"
                       "同名檔案不會覆蓋（會自動改名）。\n\n確定開始？")
        if not messagebox.askyesno("確認收納", message, parent=root):
            return
        cancel.clear()
        set_busy(True)
        append_log("", "plain")
        append_log(f"=== 開始{'搬移' if mode == 'move' else '複製'}", "head")
        status_var.set("執行中…（可按「取消」，已完成的檔案不會被還原）")
        progress.config(value=0, maximum=max(len(to_copy), 1))
        threading.Thread(
            target=run_worker, args=(items, mode, dest, behavior_var.get(), list(input_dirs)), daemon=True,
        ).start()

    def on_blocked(message):
        """驗證不通過：顯示警告並拒絕整合（此時還沒動任何檔案）。"""
        set_busy(False)
        state["plan"] = None
        set_enabled(start_btn, False)
        append_log("", "plain")
        for line in message.splitlines():
            append_log(line, "warn" if line.startswith(("以下", "  •", "已拒絕")) else "plain")
        status_var.set("✗ 已拒絕整合：有資料夾底下找不到該類別資料夾。")
        messagebox.showwarning("拒絕整合", message, parent=root)

    def on_cancel():
        cancel.set()
        status_var.set("正在取消…目前這支處理完就會停止。")

    def on_run_done(ok_count, failed, cancelled, manifest, verb):
        set_busy(False)
        state["plan"] = None  # 檔案已變動，舊預覽作廢
        set_enabled(start_btn, False)
        append_log("", "plain")
        append_log(f"完成：已{verb} {ok_count} 支、失敗 {failed} 支" + ("（已中途取消）" if cancelled else ""), "head")
        if manifest:
            append_log(f"紀錄檔：{manifest}", "muted")
        status_var.set(f"✓ 已{verb} {ok_count} 支。要再收納別的資料夾／類別，直接改設定後按「預覽」。")

    def poll():
        try:
            while True:
                msg = q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    append_log(msg[1], msg[2])
                elif kind == "progress":
                    progress.config(value=msg[1], maximum=max(msg[2], 1))
                elif kind == "plan_done":
                    on_plan_done(msg[1], msg[2], msg[3])
                elif kind == "blocked":
                    on_blocked(msg[1])
                elif kind == "drop":
                    if state["busy"]:
                        status_var.set("處理中，暫時無法新增資料夾；請等完成或按「取消」後再拖曳。")
                    else:
                        add_paths(msg[1])
                elif kind == "run_done":
                    on_run_done(*msg[1:])
        except queue.Empty:
            pass
        root.after(80, poll)

    def on_close():
        if state["busy"]:
            messagebox.showinfo("處理中", "正在處理中，請先按「取消」或等待完成再關閉視窗。", parent=root)
            return
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    for var in (behavior_var, output_var, naming_var, dedupe_var):
        var.trace_add("write", invalidate)
    behavior_var.trace_add("write", restyle_class_buttons)
    restyle_class_buttons()
    append_log("① 選類別　② 新增輸入資料夾（可多選，或直接從檔案總管拖進來）　③ 選輸出資料夾　→　按「預覽」。", "muted")
    refresh_input_list()  # 清單一開始是空的，顯示「拖曳到這裡」的提示
    if not enable_folder_drop(root, lambda paths: q.put(("drop", paths))):
        append_log("（這個環境無法啟用拖曳功能，請改用「新增資料夾」按鈕。）", "dim")
    poll()
    root.lift()
    root.attributes("-topmost", True)
    root.after(400, lambda: root.attributes("-topmost", False))
    root.mainloop()


if __name__ == "__main__":
    run_gui()
