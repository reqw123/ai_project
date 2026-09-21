"""
Unit Test：settings_gui/tool_order.py 的「腳本備註」資料層與排序對話框

備註跟排序／鎖定存在同一份 tool_order.json，重點在：舊格式（純陣列、只有 order／locked）
仍讀得進來、備註不會因為「順序剛好等於檔名排序」而被連檔案一起刪掉、目前不在清單裡的
腳本的舊備註會被原樣保留、選單顯示字串與 key 能互相還原。對話框用真的 Tk 視窗驅動，
沒有顯示環境時自動略過。全用 tmp_path，不碰真實的 tool_order.json。
"""

import json
import tkinter as tk

import pytest

from settings_gui import tool_order


@pytest.fixture
def order_path(tmp_path, monkeypatch):
    p = tmp_path / "tool_order.json"
    monkeypatch.setattr(tool_order, "_ORDER_PATH", p)
    return p


def _read(p):
    return json.loads(p.read_text(encoding="utf-8"))


# ── 資料層 ────────────────────────────────────────────────────────────


def test_load_notes_missing_file_is_empty(order_path):
    assert tool_order.load_notes() == {}


def test_load_notes_cleans_bad_entries(order_path):
    order_path.write_text(
        json.dumps({"order": ["a.py"], "notes": {"a.py": "  影片推論 ", "b.py": "", "c.py": 3, "d\\e.py": "子資料夾"}}),
        encoding="utf-8",
    )
    assert tool_order.load_notes() == {"a.py": "影片推論", "d/e.py": "子資料夾"}


def test_old_array_format_still_loads(order_path):
    order_path.write_text(json.dumps(["b.py", "a.py"]), encoding="utf-8")
    assert tool_order.load_order() == ["b.py", "a.py"]
    assert tool_order.load_notes() == {}


def test_save_with_notes_roundtrip(order_path):
    ok, err = tool_order.save_order(["b.py", "a.py"], ["a.py"], {"a.py": "耳距", "b.py": "比較"})
    assert ok and err is None
    assert _read(order_path) == {"order": ["b.py", "a.py"], "locked": ["a.py"], "notes": {"a.py": "耳距", "b.py": "比較"}}
    assert tool_order.load_order() == ["b.py", "a.py"]
    assert tool_order.load_locked() == ["a.py"]
    assert tool_order.load_notes() == {"a.py": "耳距", "b.py": "比較"}


def test_save_without_notes_keeps_previous_file_shape(order_path):
    tool_order.save_order(["b.py", "a.py"], None)
    assert _read(order_path) == {"order": ["b.py", "a.py"]}


def test_default_order_with_notes_does_not_delete_file(order_path):
    """順序剛好等於檔名排序（對話框傳 [] 、[]）但還有備註 → 檔案要留著，備註不能丟。"""
    ok, _ = tool_order.save_order([], [], {"a.py": "耳距"})
    assert ok and order_path.exists()
    assert tool_order.load_notes() == {"a.py": "耳距"}
    assert tool_order.load_order() == []


def test_everything_empty_deletes_file(order_path):
    tool_order.save_order(["b.py"], None, {"b.py": "x"})
    assert order_path.exists()
    ok, _ = tool_order.save_order([], [], {})
    assert ok and not order_path.exists()


def test_notes_for_missing_scripts_are_kept_verbatim(order_path):
    """備註是「完整字典」，含目前不在清單裡的腳本；save_order 不會依 order 過濾掉它們。"""
    tool_order.save_order(["a.py"], None, {"a.py": "現有", "gone.py": "暫時不在"})
    assert tool_order.load_notes() == {"a.py": "現有", "gone.py": "暫時不在"}


def test_with_note_and_strip_note_are_inverse():
    display = "#05  train_data/0_dataset_collect.py"
    assert tool_order.with_note(display, "") == display
    shown = tool_order.with_note(display, "收集資料")
    assert shown.startswith(display) and shown.endswith("收集資料") and shown != display
    assert tool_order.strip_note(shown) == display
    assert tool_order.strip_note(display) == display  # 沒備註時原樣


# ── 對話框（真的 Tk 視窗）────────────────────────────────────────────


@pytest.fixture(scope="module")
def tk_root():
    # module 範圍共用一個 Tk：同一個行程反覆建立／銷毀 Tk() 在 Windows 上偶爾會失敗
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("沒有可用的顯示環境")
    # 不能 withdraw：對話框是這個視窗的 transient，母視窗隱藏時對話框也跟著隱藏、收不到（合成的）鍵盤事件
    root.geometry("40x40+0+0")
    root.update()
    yield root
    root.destroy()


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _drive_dialog(root, script, order, locked=None, notes=None):
    """開對話框，等它建好後呼叫 script(dlg, listbox, entry, save_btn_cb)；回傳 on_apply 收到的參數。"""
    got = {}
    errors = []

    def on_apply(new_order, new_locked, new_notes):
        got["args"] = (new_order, new_locked, new_notes)
        return True

    def driver():
        dlg = None
        try:
            dlg = [w for w in _walk(root) if isinstance(w, tk.Toplevel)][-1]
            widgets = list(_walk(dlg))
            lb = next(w for w in widgets if isinstance(w, tk.Listbox))
            entry = [w for w in widgets if isinstance(w, tk.Entry)][-1]  # 備註輸入框是對話框裡唯一的 Entry
            # 「✓ 儲存並套用」是自訂 Canvas 按鈕：直接呼叫它的 command
            save = next(w for w in widgets if getattr(w, "_text", "") == "✓ 儲存並套用")
            dlg.focus_force()  # event_generate 的鍵盤事件只會送給「有焦點」的元件
            entry.focus_force()
            dlg.update()
            script(dlg, lb, entry, lambda: save._command())
        except BaseException as e:  # noqa: BLE001 — 例外在 Tk callback 裡會被吞掉，這裡收集後由測試本體重新拋出
            errors.append(e)
        finally:
            # 不管成功失敗一律關掉對話框，否則 open_dialog() 的 wait_window() 會永遠卡住
            if dlg is not None and dlg.winfo_exists():
                dlg.destroy()

    root.after(150, driver)
    tool_order.open_dialog(root, order, on_apply, locked, notes_seed=notes)
    if errors:
        raise errors[0]
    return got.get("args")


def _type(entry, text):
    entry.delete(0, "end")
    entry.insert(0, text)  # Entry 的 textvariable trace 會被觸發


def test_dialog_edit_note_and_save(tk_root):
    def script(dlg, lb, entry, save):
        assert lb.get(0).startswith("#01  a.py")
        lb.selection_clear(0, "end"); lb.selection_set(0); lb.event_generate("<<ListboxSelect>>")
        _type(entry, "影片推論")
        assert "影片推論" in lb.get(0)  # 列即時更新
        save()

    args = _drive_dialog(tk_root, script, ["a.py", "b.py"])
    assert args == ([], [], {"a.py": "影片推論"})  # 順序等於檔名排序、無鎖定 → 傳 []、[]，備註照帶


def test_dialog_note_length_is_capped(tk_root):
    def script(dlg, lb, entry, save):
        _type(entry, "一二三四五六七八九十甲乙丙丁戊")
        assert len(entry.get()) == tool_order.NOTE_MAX_LEN
        save()

    args = _drive_dialog(tk_root, script, ["a.py"])
    assert len(args[2]["a.py"]) == tool_order.NOTE_MAX_LEN


def test_dialog_clearing_note_removes_it(tk_root):
    def script(dlg, lb, entry, save):
        assert entry.get() == "舊備註"  # 選取第一列時載入既有備註
        _type(entry, "")
        save()

    args = _drive_dialog(tk_root, script, ["a.py"], notes={"a.py": "舊備註"})
    assert args[2] == {}


def test_dialog_keeps_notes_of_scripts_not_in_list(tk_root):
    def script(dlg, lb, entry, save):
        _type(entry, "新的")
        save()

    args = _drive_dialog(tk_root, script, ["a.py"], notes={"gone.py": "暫時不在"})
    assert args[2] == {"a.py": "新的", "gone.py": "暫時不在"}


def test_dialog_enter_saves_and_moves_to_next_row(tk_root):
    def script(dlg, lb, entry, save):
        _type(entry, "第一")
        entry.event_generate("<Return>")
        assert lb.curselection() == (1,)
        assert entry.get() == ""  # 第二列沒有備註
        _type(entry, "第二")
        save()

    args = _drive_dialog(tk_root, script, ["a.py", "b.py"])
    assert args[2] == {"a.py": "第一", "b.py": "第二"}


def test_dialog_moving_a_row_keeps_its_note(tk_root):
    def script(dlg, lb, entry, save):
        _type(entry, "耳距")                       # a.py 的備註
        dlg.event_generate("<Alt-Down>")           # a.py 下移一格
        assert lb.get(1).startswith("#02  a.py") and "耳距" in lb.get(1)
        save()

    args = _drive_dialog(tk_root, script, ["a.py", "b.py"])
    assert args[0] == ["b.py", "a.py"] and args[2] == {"a.py": "耳距"}
