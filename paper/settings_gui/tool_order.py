"""「獨立腳本工具」下拉選單的自訂排序。

- 順序／鎖定狀態存在 `settings_gui/tool_order.json`。相容兩種格式：
    · 純陣列  `["a.py", "train_data/z.py"]`（舊格式，只有順序）
    · 物件    `{"order": [...], "locked": [...]}`（有鎖定項時用這個）
  路徑都是「相對於 cat_monitoring_system/tools/」的字串，依想要的顯示順序排列。
- `apply_order()`：有列到的腳本照 json 裡的順序排在最前面；其餘沒列到的
  （之後新增、還沒排過的）一律接在後面、依相對路徑字串排序。json 裡列到但實際
  檔案已不存在的項目會自動略過，不需要手動清理。
- `open_dialog()`：彈一個用 ↑/↓ 調整順序的小視窗，每列右側顯示目前位置的流水號，
  可對個別項目「上鎖」（鎖定的項目位置固定、不能移動、其他項目也不能跨越它）。
  按「儲存並套用」時回呼 `on_apply(new_relnames, locked_relnames)`，由呼叫端
  （settings_window.py）負責存檔＋重建下拉選單。

這是純 GUI 顯示偏好，不是 main.py 的執行期設定，所以獨立成一個檔案，不塞進
runtime_settings.current.json。刪掉 tool_order.json（或在對話框裡按「依檔名排序」
且沒有任何鎖定再儲存）＝回到「全部依檔名排序」，之後新增的腳本也會自動照檔名融入。
"""

import json
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont, messagebox

from settings_gui.style import (
    BTN_PRIMARY_ACTIVE,
    BTN_PRIMARY_BG,
    BTN_SECONDARY_ACTIVE,
    BTN_SECONDARY_BG,
    CONSOLE_FONT_FAMILY,
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
)
from settings_gui.widgets import _styled_button

_ORDER_PATH = Path(__file__).resolve().parent / "tool_order.json"

_DIALOG_BG = "#f4f6f8"
_LISTBOX_BG = "#eaf4fc"          # 跟 settings_window 下拉清單同一個淡藍底
_LISTBOX_SEL_BG = "#1b4f72"
_TEXT_FG = "#1b2631"
_LOCKED_BG = "#f5e0c3"           # 鎖定列：琥珀色底
_LOCKED_FG = "#7a6a55"           # 鎖定列：偏灰文字
_LOCKED_SEL_BG = "#c9a978"


def _clean_list(value):
    if not isinstance(value, list):
        return []
    seen = set()
    out = []
    for x in value:
        if not isinstance(x, str):
            continue
        s = x.replace("\\", "/").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _read_raw():
    try:
        data = json.loads(_ORDER_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(data, list):
        return {"order": data}
    if isinstance(data, dict):
        return data
    return {}


def load_order():
    """回傳依序排列、去重後的相對路徑清單；讀不到或格式不對回傳 []。"""
    return _clean_list(_read_raw().get("order", []))


def load_locked():
    """回傳被鎖定的相對路徑清單（順序不重要）；讀不到回傳 []。"""
    return _clean_list(_read_raw().get("locked", []))


def save_order(relnames, locked=None):
    """把順序（與鎖定清單）寫回 tool_order.json；回傳 (ok, error_message)。
    兩者都空 → 直接刪掉檔案，回到「全部依檔名排序」的預設狀態。"""
    order = list(relnames)
    locked = sorted(set(locked or []) & set(order))
    if not order and not locked:
        try:
            _ORDER_PATH.unlink(missing_ok=True)
            return True, None
        except OSError as e:
            return False, str(e)
    payload = {"order": order}
    if locked:
        payload["locked"] = locked
    try:
        _ORDER_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return True, None
    except OSError as e:
        return False, str(e)


def apply_order(items):
    """items：[(relname, path), ...]。依 tool_order.json 重排——有列到的照清單順序在前，
    其餘依 relname 字串排序接在後面。回傳重排後的新 list（不動傳入的 list）。"""
    order = load_order()
    rank = {name: i for i, name in enumerate(order)}
    return sorted(items, key=lambda it: (rank.get(it[0], len(order)), it[0]))


def open_dialog(parent, current_relnames, on_apply, locked_relnames=None):
    """彈出排序小視窗。

    current_relnames：目前生效的順序（relname 清單，通常來自 apply_order 的結果）。
    locked_relnames：初始鎖定的 relname（沒有就 None）。
    on_apply(new_relnames, locked_relnames)：按「儲存並套用」時呼叫；若調整後的順序
        剛好等於純檔名排序、又沒有任何鎖定，會傳入 ([], [])（呼叫端據此刪掉 json）。
        回傳 False 代表失敗，對話框留著讓使用者重試；其餘（True/None）視為成功並關閉。
    """
    order = list(current_relnames)
    locked = set(locked_relnames or []) & set(order)
    if not order:
        messagebox.showinfo("自訂工具排序", "目前沒有可排序的腳本。", parent=parent)
        return

    dlg = tk.Toplevel(parent)
    dlg.title("自訂工具排序")
    dlg.transient(parent)
    dlg.grab_set()
    dlg.configure(bg=_DIALOG_BG)

    lb_font = tkfont.Font(family=CONSOLE_FONT_FAMILY, size=13)
    btn_font = ("Microsoft JhengHei", 11, "bold")

    tk.Label(
        dlg,
        text="用 ↑ / ↓ 調整「🧩 獨立腳本工具」下拉選單的顯示順序（也可按 Alt+↑ / Alt+↓）。\n"
             "每列最左邊是目前位置的流水號。鎖定的項目（琥珀色）位置固定、不能移動，"
             "其他項目也不能跨越它——先解鎖才能調整。\n"
             "沒排到的腳本會自動接在最後、依檔名排序；之後新增的腳本也一樣。",
        bg=_DIALOG_BG, fg=_TEXT_FG, font=("Microsoft JhengHei", 10),
        justify="left", anchor="w",
    ).pack(fill="x", padx=SPACE_MD, pady=(SPACE_MD, SPACE_SM))

    body = tk.Frame(dlg, bg=_DIALOG_BG)
    body.pack(fill="both", expand=True, padx=SPACE_MD, pady=(0, SPACE_SM))

    lb = tk.Listbox(
        body, selectmode="browse", activestyle="none", font=lb_font,
        bg=_LISTBOX_BG, fg=_TEXT_FG,
        selectbackground=_LISTBOX_SEL_BG, selectforeground="#ffffff",
        width=64, height=22, highlightthickness=1, relief="solid", bd=1,
        exportselection=False,
    )
    lb.pack(side="left", fill="both", expand=True)
    sb = tk.Scrollbar(body, command=lb.yview)
    sb.pack(side="left", fill="y")
    lb.config(yscrollcommand=sb.set)

    btns = tk.Frame(body, bg=_DIALOG_BG)
    btns.pack(side="left", fill="y", padx=(SPACE_SM, 0))

    def _fmt(relname, pos):
        # 流水號放最左邊固定欄：中英混合檔名在 Tk listbox 裡靠補空白對不齊右側流水號。
        return f"#{pos:02d}  {relname}"

    def _current_relname():
        cur = lb.curselection()
        return order[cur[0]] if cur and cur[0] < len(order) else None

    def _redraw(keep=None):
        sel = keep if keep is not None else _current_relname()
        lb.delete(0, "end")
        for idx, r in enumerate(order, start=1):
            lb.insert("end", _fmt(r, idx))
            if r in locked:
                lb.itemconfig(
                    idx - 1, background=_LOCKED_BG, foreground=_LOCKED_FG,
                    selectbackground=_LOCKED_SEL_BG, selectforeground="#ffffff",
                )
        if sel in order:
            k = order.index(sel)
            lb.selection_set(k)
            lb.activate(k)
            lb.see(k)
        elif order:
            lb.selection_set(0)
            lb.activate(0)
        _update_lock_btn()

    def _try_move(step, silent=False):
        r = _current_relname()
        if r is None:
            return False
        if r in locked:
            if not silent:
                messagebox.showinfo(
                    "排序", f"「{r}」已鎖定，無法調整排序。\n請先按「🔓 解鎖」再移動。",
                    parent=dlg,
                )
            return False
        i = order.index(r)
        j = i + step
        if j < 0 or j >= len(order):
            return False
        if order[j] in locked:
            if not silent:
                where = "上方" if step < 0 else "下方"
                messagebox.showinfo(
                    "排序", f"{where}的「{order[j]}」已鎖定，不能跨越。\n請先解鎖那一項。",
                    parent=dlg,
                )
            return False
        order[i], order[j] = order[j], order[i]
        _redraw(keep=r)
        return True

    def _move(step):
        _try_move(step)
        return "break"

    def _move_to(where):
        r = _current_relname()
        if r is None:
            return
        if r in locked:
            messagebox.showinfo("排序", f"「{r}」已鎖定，無法調整排序。", parent=dlg)
            return
        step = -1 if where == "top" else 1
        while _try_move(step, silent=True):
            pass

    def _toggle_lock():
        r = _current_relname()
        if r is None:
            return
        locked.discard(r) if r in locked else locked.add(r)
        _redraw(keep=r)

    def _sort_by_name():
        free = iter(sorted(r for r in order if r not in locked))
        order[:] = [r if r in locked else next(free) for r in order]
        _redraw()

    for label, cmd in (
        ("⤒ 置頂", lambda: _move_to("top")),
        ("↑ 上移", lambda: _move(-1)),
        ("↓ 下移", lambda: _move(1)),
        ("⤓ 置底", lambda: _move_to("bottom")),
    ):
        _styled_button(
            btns, label, cmd, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
            font=btn_font, compact=True,
        ).pack(fill="x", pady=(0, SPACE_XS))

    lock_btn = _styled_button(
        btns, "🔒 鎖定", _toggle_lock, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
        font=btn_font, compact=True,
    )
    lock_btn.pack(fill="x", pady=(SPACE_MD, 0))

    _styled_button(
        btns, "↺ 依檔名排序", _sort_by_name, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
        font=btn_font, compact=True,
    ).pack(fill="x", pady=(SPACE_XS, 0))

    def _update_lock_btn(*_):
        r = _current_relname()
        lock_btn.config(text="🔓 解鎖" if r in locked else "🔒 鎖定")

    # <<ListboxSelect>> 是主要事件；再補滑鼠放開／方向鍵放開，避免某些平台上
    # 用方向鍵移動選取時按鈕文字沒跟著更新。
    for _seq in ("<<ListboxSelect>>", "<ButtonRelease-1>", "<KeyRelease-Up>", "<KeyRelease-Down>"):
        lb.bind(_seq, _update_lock_btn, add="+")
    dlg.bind("<Alt-Up>", lambda _e: _move(-1))
    dlg.bind("<Alt-Down>", lambda _e: _move(1))

    bar = tk.Frame(dlg, bg=_DIALOG_BG)
    bar.pack(fill="x", padx=SPACE_MD, pady=(0, SPACE_MD))

    def _cancel(_e=None):
        dlg.destroy()

    def _save(_e=None):
        if order == sorted(order) and not locked:
            result = on_apply([], [])
        else:
            result = on_apply(list(order), sorted(locked))
        if result is not False:
            dlg.destroy()

    _styled_button(
        bar, "取消", _cancel, BTN_SECONDARY_BG, BTN_SECONDARY_ACTIVE,
        outline=True, font=btn_font, compact=True,
    ).pack(side="right")
    _styled_button(
        bar, "✓ 儲存並套用", _save, BTN_PRIMARY_BG, BTN_PRIMARY_ACTIVE,
        font=btn_font, compact=True,
    ).pack(side="right", padx=(0, SPACE_SM))

    dlg.bind("<Escape>", _cancel)

    _redraw()
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
