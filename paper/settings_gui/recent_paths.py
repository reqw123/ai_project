"""「🎬 影片路徑（選填）」欄位的最近使用清單＋管理對話框＋檔案拖放。

- 清單存在 `settings_gui/ui_state.json`（跟「上次選的腳本」同一份純 UI 便利記憶）：
    · `recent_video_paths`  依下拉選單顯示順序排列的路徑清單
    · `recent_video_locked` 被鎖定（釘選）的路徑
- 記錄時機由呼叫端決定（settings_window.py：執行腳本時、按「選擇影片／資料夾」、
  拖放進輸入框時各記一次）。新路徑插在「未鎖定項目」的最前面，已經在清單裡的
  會被提到最前面；未鎖定項目最多留 MAX_RECENT 筆，多的從最舊的丟掉。
- 鎖定的項目位置完全固定：新路徑不會把它擠掉、不能上移下移、不能刪除（要先解鎖），
  路徑暫時找不到時也照樣留著（例如外接硬碟沒插）。
- 刪除只是「不再記住」，不會動到磁碟上的檔案。
- 路徑比對不分大小寫、`\\` 與 `/` 視為相同（Windows 路徑語意）；存檔一律用 `/`，
  跟 filedialog 回傳的格式一致。

讀寫都是 best-effort：ui_state.json 壞掉／寫不進去都安靜降級，不影響 GUI。
"""

import os
import tkinter as tk
from tkinter import font as tkfont, messagebox

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
from settings_gui.tool_order import (
    _DIALOG_BG,
    _LISTBOX_BG,
    _LISTBOX_SEL_BG,
    _LOCKED_BG,
    _LOCKED_FG,
    _LOCKED_SEL_BG,
    _TEXT_FG,
)
from settings_gui.widgets import _styled_button

MAX_RECENT = 10            # 未鎖定項目最多記幾筆；鎖定的不算在內、也永遠不會被擠掉

_KEY_PATHS = "recent_video_paths"
_KEY_LOCKED = "recent_video_locked"

_MISSING_FG = "#b03a2e"          # 路徑找不到：暗紅字
_DANGER_BG = "#e74c3c"
_DANGER_ACTIVE = "#c0392b"


# ── 資料層 ────────────────────────────────────────────────────────────


def normalize(path):
    """去頭尾空白／引號（從檔案總管「複製路徑」會帶雙引號）→ normpath → 統一用 "/"。
    空字串回傳 ""。"""
    if not isinstance(path, str):
        return ""
    s = path.strip().strip('"').strip()
    if not s:
        return ""
    return os.path.normpath(s).replace("\\", "/")


def _key(path):
    return os.path.normcase(normalize(path))


def _clean(value):
    """list → 正規化、去重（不分大小寫）、丟掉非字串/空白項。"""
    if not isinstance(value, list):
        return []
    seen = set()
    out = []
    for x in value:
        p = normalize(x)
        k = _key(p)
        if p and k not in seen:
            seen.add(k)
            out.append(p)
    return out


def load():
    """回傳 (paths, locked)：paths 是顯示順序，locked 是鎖定路徑的 list（都在 paths 裡）。
    從沒用過這個功能（沒有 recent_video_paths）時，用舊版「上次填的影片路徑」當第一筆種子。"""
    data = _ui_state.load()
    if _KEY_PATHS in data:
        paths = _clean(data.get(_KEY_PATHS))
    else:
        paths = _clean([data.get("last_tool_video_path", "")])
    keys = {_key(p) for p in paths}
    locked = [p for p in _clean(data.get(_KEY_LOCKED)) if _key(p) in keys]
    return paths, locked


def save(paths, locked):
    paths = _clean(list(paths))
    keys = {_key(p) for p in paths}
    locked = [p for p in _clean(list(locked)) if _key(p) in keys]
    _ui_state.update(**{_KEY_PATHS: paths, _KEY_LOCKED: locked or None})


def is_locked(path, locked):
    return _key(path) in {_key(p) for p in locked}


def _place(old, locked, unlocked):
    """把鎖定項目放回它在 old 裡的原位置，其餘位置依序填 unlocked。
    清單變短、鎖定項目的原位置已經超出長度時，依原本順序接在最後面。"""
    locked_keys = {_key(p) for p in locked}
    locked_at = {i: p for i, p in enumerate(old) if _key(p) in locked_keys}
    n = len(locked_at) + len(unlocked)
    free = iter(unlocked)
    out = []
    for i in range(n):
        if i in locked_at:
            out.append(locked_at.pop(i))
            continue
        nxt = next(free, None)
        if nxt is None:
            # 未鎖定的用完了，拿後面位置的鎖定項目往前補
            nxt = locked_at.pop(min(locked_at))
        out.append(nxt)
    out.extend(locked_at[i] for i in sorted(locked_at))
    return out


def add(paths, locked, new, max_recent=MAX_RECENT):
    """純函式：回傳加入 new 之後的新 paths（不動傳入的 list）。
    new 已鎖定 → 位置不變；已存在（未鎖定）→ 提到未鎖定的最前面；未鎖定只留 max_recent 筆。"""
    new = normalize(new)
    if not new or is_locked(new, locked):
        return list(paths)
    locked_keys = {_key(p) for p in locked}
    new_key = _key(new)
    unlocked = [p for p in paths if _key(p) not in locked_keys and _key(p) != new_key]
    unlocked = ([new] + unlocked)[:max_recent]
    return _place(paths, locked, unlocked)


def record(path):
    """把 path 記進最近清單並存檔；回傳新的 paths。空路徑什麼都不做。"""
    paths, locked = load()
    new_paths = add(paths, locked, path)
    if new_paths != paths:
        save(new_paths, locked)
    return new_paths


def move(paths, locked, index, step):
    """純函式：把 index 那一項往 step 方向（-1 上 / +1 下）跟「下一個未鎖定項目」互換，
    中間經過的鎖定項目原地不動。回傳 (new_paths, new_index)；不能移動時 new_index 為 None。"""
    if not (0 <= index < len(paths)) or is_locked(paths[index], locked):
        return list(paths), None
    j = index + step
    while 0 <= j < len(paths) and is_locked(paths[j], locked):
        j += step
    if not (0 <= j < len(paths)):
        return list(paths), None
    out = list(paths)
    out[index], out[j] = out[j], out[index]
    return out, j


def remove(paths, locked, index):
    """純函式：刪除 index 那一項（鎖定的不能刪，原樣回傳），其餘鎖定項目盡量留在原位。"""
    if not (0 <= index < len(paths)) or is_locked(paths[index], locked):
        return list(paths)
    target = _key(paths[index])
    unlocked = [p for p in paths if not is_locked(p, locked) and _key(p) != target]
    return _place(paths, locked, unlocked)


# ── 管理對話框 ─────────────────────────────────────────────────────────


def open_dialog(parent, on_pick=None):
    """彈出「管理最近影片路徑」對話框（modal）。按「完成」才存檔；「取消」／Esc 放棄這次的修改。
    on_pick(path)：在清單上雙擊某一項（或按「套用到欄位」）時呼叫——先存檔、關閉對話框，
    再把該路徑交給呼叫端填進輸入框。"""
    paths, locked = load()
    paths = list(paths)
    locked = list(locked)

    dlg = tk.Toplevel(parent)
    dlg.title("管理最近影片路徑")
    dlg.transient(parent)
    dlg.grab_set()
    dlg.configure(bg=_DIALOG_BG)

    lb_font = tkfont.Font(family="標楷體", size=13, weight="bold")
    btn_font = ("Microsoft JhengHei", 11, "bold")

    tk.Label(
        dlg,
        text="上移 / 下移調整下拉選單的顯示順序（也可按 Alt+↑ / Alt+↓）；雙擊某一項＝套用到輸入框。\n"
             "🔒 鎖定的項目位置完全固定——不會被新路徑擠掉、不能上移下移、也不能刪除（要先解鎖），"
             "而且路徑暫時找不到時仍會留在下拉選單裡。\n"
             f"未鎖定的最多記住最近 {MAX_RECENT} 筆。刪除只是不再記住，不會動到磁碟上的檔案。",
        bg=_DIALOG_BG, fg=_TEXT_FG, font=("Microsoft JhengHei", 10),
        justify="left", anchor="w",
    ).pack(fill="x", padx=SPACE_MD, pady=(SPACE_MD, SPACE_SM))

    body = tk.Frame(dlg, bg=_DIALOG_BG)
    body.pack(fill="both", expand=True, padx=SPACE_MD, pady=(0, SPACE_SM))

    lb = tk.Listbox(
        body, selectmode="browse", activestyle="none", font=lb_font,
        bg=_LISTBOX_BG, fg=_TEXT_FG,
        selectbackground=_LISTBOX_SEL_BG, selectforeground="#ffffff",
        width=80, height=14, highlightthickness=1, relief="solid", bd=1,
        exportselection=False,
    )
    lb.pack(side="left", fill="both", expand=True)
    sb = tk.Scrollbar(body, command=lb.yview)
    sb.pack(side="left", fill="y")
    lb.config(yscrollcommand=sb.set)

    empty_hint = tk.Label(
        lb, text="（目前沒有記住任何路徑——執行腳本、選擇影片／資料夾或拖放進輸入框後會自動記住）",
        bg=_LISTBOX_BG, fg="#7f8c8d", font=("Microsoft JhengHei", 10),
    )

    def _fmt(path, pos):
        lock = is_locked(path, locked)
        text = f"{pos:>2}. {'🔒 ' if lock else '    '}{path}"
        if not os.path.exists(path):
            text += "  （找不到）"
        if lock:
            text += "  【鎖定】"
        return text

    def _current_index():
        cur = lb.curselection()
        return cur[0] if cur and cur[0] < len(paths) else None

    def _redraw(keep=None):
        if keep is None:
            keep = _current_index()
        lb.delete(0, "end")
        for idx, p in enumerate(paths, start=1):
            lb.insert("end", _fmt(p, idx))
            k = idx - 1
            if is_locked(p, locked):
                lb.itemconfig(
                    k, background=_LOCKED_BG, foreground=_LOCKED_FG,
                    selectbackground=_LOCKED_SEL_BG, selectforeground="#ffffff",
                )
            elif not os.path.exists(p):
                lb.itemconfig(k, foreground=_MISSING_FG)
        if paths:
            empty_hint.place_forget()
            k = max(0, min(keep if keep is not None else 0, len(paths) - 1))
            lb.selection_set(k)
            lb.activate(k)
            lb.see(k)
        else:
            empty_hint.place(relx=0.5, rely=0.5, anchor="center")
        _update_lock_btn()

    def _move(step):
        i = _current_index()
        if i is None:
            return "break"
        if is_locked(paths[i], locked):
            messagebox.showinfo("排序", "這一項已鎖定，位置固定不能移動。\n請先按「🔓 解鎖」。", parent=dlg)
            return "break"
        new_paths, j = move(paths, locked, i, step)
        if j is not None:
            paths[:] = new_paths
            _redraw(keep=j)
        return "break"

    def _toggle_lock():
        i = _current_index()
        if i is None:
            return
        p = paths[i]
        if is_locked(p, locked):
            locked[:] = [q for q in locked if _key(q) != _key(p)]
        else:
            locked.append(p)
        _redraw(keep=i)

    def _delete(_e=None):
        i = _current_index()
        if i is None:
            return
        if is_locked(paths[i], locked):
            messagebox.showinfo("刪除", "這一項已鎖定，不能刪除。\n請先按「🔓 解鎖」。", parent=dlg)
            return
        paths[:] = remove(paths, locked, i)
        _redraw(keep=i)

    def _finish(_e=None):
        save(paths, locked)
        dlg.destroy()

    def _cancel(_e=None):
        dlg.destroy()

    def _apply(_e=None):
        i = _current_index()
        if i is None:
            return
        p = paths[i]
        _finish()
        if on_pick is not None:
            on_pick(p)

    btns = tk.Frame(dlg, bg=_DIALOG_BG)
    btns.pack(fill="x", padx=SPACE_MD, pady=(0, SPACE_SM))
    _styled_button(
        btns, "↑ 上移", lambda: _move(-1), BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
        font=btn_font, compact=True,
    ).pack(side="left", padx=(0, SPACE_XS))
    _styled_button(
        btns, "↓ 下移", lambda: _move(1), BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
        font=btn_font, compact=True,
    ).pack(side="left", padx=(0, SPACE_XS))
    lock_btn = _styled_button(
        btns, "🔒 鎖定", _toggle_lock, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
        font=btn_font, compact=True,
    )
    lock_btn.pack(side="left", padx=(0, SPACE_XS))
    _styled_button(
        btns, "🗑 刪除", _delete, _DANGER_BG, _DANGER_ACTIVE,
        font=btn_font, compact=True,
    ).pack(side="left", padx=(0, SPACE_XS))
    if on_pick is not None:
        _styled_button(
            btns, "📥 套用到輸入框", _apply, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
            font=btn_font, compact=True,
        ).pack(side="left", padx=(SPACE_MD, 0))

    def _update_lock_btn(*_):
        i = _current_index()
        lock_btn.config(text="🔓 解鎖" if i is not None and is_locked(paths[i], locked) else "🔒 鎖定")

    for _seq in ("<<ListboxSelect>>", "<ButtonRelease-1>", "<KeyRelease-Up>", "<KeyRelease-Down>"):
        lb.bind(_seq, _update_lock_btn, add="+")
    if on_pick is not None:
        lb.bind("<Double-Button-1>", _apply)
        lb.bind("<Return>", _apply)
    lb.bind("<Delete>", _delete)
    dlg.bind("<Alt-Up>", lambda _e: _move(-1))
    dlg.bind("<Alt-Down>", lambda _e: _move(1))
    dlg.bind("<Escape>", _cancel)
    dlg.protocol("WM_DELETE_WINDOW", _cancel)

    bar = tk.Frame(dlg, bg=_DIALOG_BG)
    bar.pack(fill="x", padx=SPACE_MD, pady=(0, SPACE_MD))
    _styled_button(
        bar, "取消", _cancel, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
        outline=True, font=btn_font, compact=True,
    ).pack(side="right")
    _styled_button(
        bar, "✓ 完成", _finish, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE,
        font=btn_font, compact=True,
    ).pack(side="right", padx=(0, SPACE_SM))

    _redraw(keep=0)
    lb.focus_set()

    dlg.update_idletasks()
    try:
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        dlg.geometry(f"+{px + max(0, (pw - w) // 2)}+{py + max(0, (ph - h) // 2)}")
    except tk.TclError:
        pass

    dlg.wait_window()


# ── 檔案拖放 ───────────────────────────────────────────────────────────


def enable_file_drop(widgets, on_drop):
    """讓 widgets 接受從檔案總管拖放進來的檔案／資料夾；拖進來時呼叫 on_drop(path)
    （多個只取第一個）。靠 tkinterdnd2（yolo_new 環境已安裝）：import 時它會把拖放
    方法掛到所有 tkinter widget 上，再用 _require() 把 tkdnd 擴充載入「現有的」Tk
    解譯器——不用把主視窗改成 TkinterDnD.Tk。套件不存在或載入失敗就回傳 False、
    安靜略過（拖放是加值功能，手動選檔／貼路徑照常可用）。"""
    widgets = [w for w in widgets if w is not None]
    if not widgets:
        return False
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        TkinterDnD._require(widgets[0].winfo_toplevel())
    except Exception:
        return False

    def _handler(event):
        try:
            items = event.widget.tk.splitlist(event.data)
        except tk.TclError:
            items = [event.data]
        path = normalize(items[0]) if items else ""
        if path:
            on_drop(path)
        return event.action

    ok = False
    for w in widgets:
        try:
            w.drop_target_register(DND_FILES)
            w.dnd_bind("<<Drop>>", _handler)
            ok = True
        except (tk.TclError, AttributeError):
            pass
    return ok
