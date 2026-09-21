"""「獨立腳本工具」下拉選單的自訂排序。

- 順序／鎖定狀態存在 `settings_gui/tool_order.json`。相容兩種格式：
    · 純陣列  `["a.py", "train_data/z.py"]`（舊格式，只有順序）
    · 物件    `{"order": [...], "locked": [...]}`（有鎖定項時用這個）
  路徑都是「相對於 cat_monitoring_system/tools/」的字串，依想要的顯示順序排列。
- `apply_order()`：有列到的腳本照 json 裡的順序排在最前面；其餘沒列到的
  （之後新增、還沒排過的）一律接在後面、依相對路徑字串排序。json 裡列到但實際
  檔案已不存在的項目會自動略過，不需要手動清理。
- 備註：物件格式多一個 `"notes": {"相對路徑": "簡短備註"}`，給每支腳本一句 ≤ NOTE_MAX_LEN 字的
  提醒（會顯示在下拉選單、每列名稱後面）。路徑改名後舊備註就對不上（跟順序同樣以相對路徑為 key）。
- `open_dialog()`：彈一個用 ↑/↓ 調整順序的小視窗，每列右側顯示目前位置的流水號，
  可對個別項目「上鎖」（鎖定的項目位置固定、不能移動、其他項目也不能跨越它），
  清單下方的「備註」輸入框編輯目前選取那支腳本的備註（Enter＝儲存並跳到下一支）。
  按「儲存並套用」時回呼 `on_apply(new_relnames, locked_relnames, notes)`，由呼叫端
  （settings_window.py）負責存檔＋重建下拉選單。

這是純 GUI 顯示偏好，不是 main.py 的執行期設定，所以獨立成一個檔案，不塞進
runtime_settings.current.json。刪掉 tool_order.json（或在對話框裡按「依檔名排序」
且沒有任何鎖定、沒有任何備註再儲存）＝回到「全部依檔名排序」，之後新增的腳本也會自動照檔名融入。
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

NOTE_MAX_LEN = 12          # 備註字數上限（設計上只是「簡單造詞提醒」，約 10 字以內）
NOTE_SEP = "  ── "        # 下拉選單裡「名稱」與「備註」之間的分隔符

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


def _clean_notes(value):
    """{相對路徑: 備註} → 清掉非字串、空白備註，路徑統一成 "/"、備註去頭尾空白。"""
    if not isinstance(value, dict):
        return {}
    out = {}
    for k, v in value.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        key = k.replace("\\", "/").strip()
        note = v.strip()
        if key and note:
            out[key] = note
    return out


def load_notes():
    """回傳 {相對路徑: 備註}；讀不到或格式不對回傳 {}。"""
    return _clean_notes(_read_raw().get("notes", {}))


def with_note(text, note):
    """下拉選單顯示用：`#05  名稱` + 分隔符 + 備註；沒有備註就原樣回傳。"""
    return f"{text}{NOTE_SEP}{note}" if note else text


def strip_note(text):
    """with_note() 的反向：去掉備註，只留 `#05  名稱`（選定後輸入框、路徑查找用的 key）。"""
    return text.split(NOTE_SEP, 1)[0]


def save_order(relnames, locked=None, notes=None):
    """把順序、鎖定清單與備註寫回 tool_order.json；回傳 (ok, error_message)。
    三者都空 → 直接刪掉檔案，回到「全部依檔名排序」的預設狀態。
    notes 是「完整」的 {相對路徑: 備註}（包含目前不在清單裡的腳本的舊備註，由呼叫端原樣帶著）。"""
    order = list(relnames)
    locked = sorted(set(locked or []) & set(order))
    notes = _clean_notes(notes or {})
    if not order and not locked and not notes:
        try:
            _ORDER_PATH.unlink(missing_ok=True)
            return True, None
        except OSError as e:
            return False, str(e)
    payload = {}
    if order or not notes:
        payload["order"] = order
    if locked:
        payload["locked"] = locked
    if notes:
        payload["notes"] = notes
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


def open_dialog(parent, current_relnames, on_apply, locked_relnames=None, notes_seed=None):
    """彈出排序小視窗。

    current_relnames：目前生效的順序（relname 清單，通常來自 apply_order 的結果）。
    locked_relnames：初始鎖定的 relname（沒有就 None）。
    notes_seed：目前已存的備註 {relname: 備註}（沒有就 None）。可能含有「目前不在清單裡」的腳本
        （例如檔案暫時被搬走）的舊備註，對話框原封不動帶著、儲存時一起回傳，不會被悄悄丟掉。
    on_apply(new_relnames, locked_relnames, notes)：按「儲存並套用」時呼叫；若調整後的順序
        剛好等於純檔名排序、又沒有任何鎖定，new_relnames／locked_relnames 會傳入 []、[]
        （呼叫端據此決定要不要刪掉 json——備註還在就不能刪）。
        回傳 False 代表失敗，對話框留著讓使用者重試；其餘（True/None）視為成功並關閉。
    """
    order = list(current_relnames)
    locked = set(locked_relnames or []) & set(order)
    notes = dict(notes_seed or {})
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
             "沒排到的腳本會自動接在最後、依檔名排序；之後新增的腳本也一樣。\n"
             f"下方「備註」可替選取的腳本寫一句簡短提醒（{NOTE_MAX_LEN} 字內），會顯示在下拉選單、"
             "名稱後面；在備註欄按 Enter 會儲存並跳到下一支。",
        bg=_DIALOG_BG, fg=_TEXT_FG, font=("Microsoft JhengHei", 10),
        justify="left", anchor="w",
    ).pack(fill="x", padx=SPACE_MD, pady=(SPACE_MD, SPACE_SM))

    body = tk.Frame(dlg, bg=_DIALOG_BG)
    body.pack(fill="both", expand=True, padx=SPACE_MD, pady=(0, SPACE_SM))

    lb = tk.Listbox(
        body, selectmode="browse", activestyle="none", font=lb_font,
        bg=_LISTBOX_BG, fg=_TEXT_FG,
        selectbackground=_LISTBOX_SEL_BG, selectforeground="#ffffff",
        width=76, height=20, highlightthickness=1, relief="solid", bd=1,
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
        return with_note(f"#{pos:02d}  {relname}", notes.get(relname, ""))

    def _current_relname():
        cur = lb.curselection()
        return order[cur[0]] if cur and cur[0] < len(order) else None

    def _redraw(keep=None):
        sel = keep if keep is not None else _current_relname()
        lb.delete(0, "end")
        for idx, r in enumerate(order, start=1):
            lb.insert("end", _fmt(r, idx))
            _style_locked_row(idx - 1)
        if sel in order:
            k = order.index(sel)
            lb.selection_set(k)
            lb.activate(k)
            lb.see(k)
        elif order:
            lb.selection_set(0)
            lb.activate(0)
        _on_selection_changed()

    def _style_locked_row(k):
        if order[k] in locked:
            lb.itemconfig(
                k, background=_LOCKED_BG, foreground=_LOCKED_FG,
                selectbackground=_LOCKED_SEL_BG, selectforeground="#ffffff",
            )

    def _refresh_row(k):
        """只重畫第 k 列（編輯備註時每打一個字都會呼叫，不要整個清單重建）。"""
        lb.delete(k)
        lb.insert(k, _fmt(order[k], k + 1))
        _style_locked_row(k)
        lb.selection_set(k)
        lb.activate(k)

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

    def _on_selection_changed(*_):
        _update_lock_btn()
        _load_note_into_entry()

    # <<ListboxSelect>> 是主要事件；再補滑鼠放開／方向鍵放開，避免某些平台上
    # 用方向鍵移動選取時按鈕文字沒跟著更新。
    for _seq in ("<<ListboxSelect>>", "<ButtonRelease-1>", "<KeyRelease-Up>", "<KeyRelease-Down>"):
        lb.bind(_seq, _on_selection_changed, add="+")
    dlg.bind("<Alt-Up>", lambda _e: _move(-1))
    dlg.bind("<Alt-Down>", lambda _e: _move(1))

    # ── 備註輸入列：編輯目前選取那支腳本的備註 ──
    note_row = tk.Frame(dlg, bg=_DIALOG_BG)
    note_row.pack(fill="x", padx=SPACE_MD, pady=(0, SPACE_SM))
    tk.Label(
        note_row, text="備註：", bg=_DIALOG_BG, fg=_TEXT_FG, font=("Microsoft JhengHei", 11, "bold"),
    ).pack(side="left")
    note_var = tk.StringVar()
    note_entry = tk.Entry(note_row, textvariable=note_var, font=("Microsoft JhengHei", 12), relief="solid", bd=1)
    note_entry.pack(side="left", fill="x", expand=True, padx=(SPACE_XS, SPACE_SM), ipady=3)
    note_count = tk.Label(note_row, text=f"0/{NOTE_MAX_LEN}", bg=_DIALOG_BG, fg=_LOCKED_FG, font=("Consolas", 10))
    note_count.pack(side="left")
    tk.Label(
        note_row, text="  Enter＝儲存並跳下一支　↑/↓＝換列", bg=_DIALOG_BG, fg=_LOCKED_FG,
        font=("Microsoft JhengHei", 9),
    ).pack(side="left")

    _loading = [False]  # 程式自己更新輸入框內容時（換列、截斷），不要當成使用者編輯

    def _set_note_var(text):
        _loading[0] = True
        try:
            note_var.set(text)
        finally:
            _loading[0] = False
        note_count.config(text=f"{len(text)}/{NOTE_MAX_LEN}")

    def _load_note_into_entry():
        r = _current_relname()
        _set_note_var(notes.get(r, "") if r else "")

    def _on_note_edit(*_):
        if _loading[0]:
            return
        r = _current_relname()
        if r is None:
            return
        text = note_var.get()
        if len(text) > NOTE_MAX_LEN:
            text = text[:NOTE_MAX_LEN]
            _set_note_var(text)
        note_count.config(text=f"{len(text)}/{NOTE_MAX_LEN}")
        if text.strip():
            notes[r] = text.strip()
        else:
            notes.pop(r, None)
        _refresh_row(order.index(r))

    note_var.trace_add("write", _on_note_edit)

    def _select_index(k):
        k = max(0, min(k, len(order) - 1))
        lb.selection_clear(0, "end")
        lb.selection_set(k)
        lb.activate(k)
        lb.see(k)
        _on_selection_changed()

    def _current_index():
        cur = lb.curselection()
        return cur[0] if cur else 0

    def _note_go(step):
        _select_index(_current_index() + step)
        note_entry.focus_set()
        note_entry.select_range(0, "end")
        return "break"

    note_entry.bind("<Return>", lambda _e: _note_go(1))
    note_entry.bind("<Down>", lambda _e: _note_go(1))
    note_entry.bind("<Up>", lambda _e: _note_go(-1))
    # Tk 會把「多按了修飾鍵」的事件也配對給沒帶修飾鍵的綁定：不明確綁 Alt+↑/↓，Alt+↓ 會被上面的
    # <Down> 吃掉（回傳 "break" 連視窗層級的「移動順序」綁定都收不到）。Alt 版更具體，會優先生效。
    note_entry.bind("<Alt-Up>", lambda _e: _move(-1))
    note_entry.bind("<Alt-Down>", lambda _e: _move(1))

    bar = tk.Frame(dlg, bg=_DIALOG_BG)
    bar.pack(fill="x", padx=SPACE_MD, pady=(0, SPACE_MD))

    def _cancel(_e=None):
        dlg.destroy()

    def _save(_e=None):
        if order == sorted(order) and not locked:
            result = on_apply([], [], dict(notes))
        else:
            result = on_apply(list(order), sorted(locked), dict(notes))
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
