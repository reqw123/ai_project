"""
Unit Test：settings_gui/virtual_keyboard.py 的終端機虛擬鍵盤

重點在：彈出位置一定在主視窗範圍內（貼近錨點按鈕上方，超出就往內推）、按鍵打進輸入框的
游標位置、Shift 單次生效、⌫／清除／↵ 送出／✕ 關閉、輸入框停用時按鍵無效。
用真的 Tk 視窗驅動，沒有顯示環境時自動略過。
"""

import tkinter as tk

import pytest

from conftest import make_tk_root
from settings_gui.virtual_keyboard import VirtualKeyboard


# ── 位置計算（純函式）─────────────────────────────────────────────────


def test_clamp_prefers_right_aligned_above_anchor():
    x, y = VirtualKeyboard._clamp(anchor_right=900, anchor_top=500, kb_w=400, kb_h=200, win_w=1000, win_h=700)
    assert (x, y) == (500, 296)  # 右緣對齊錨點、底緣在錨點上方 4px


def test_clamp_pushes_back_inside_window():
    # 錨點太靠左、太靠上：往右、往下推回視窗內
    x, y = VirtualKeyboard._clamp(anchor_right=100, anchor_top=50, kb_w=400, kb_h=200, win_w=1000, win_h=700)
    assert (x, y) == (4, 4)
    # 錨點超出右邊：往左推
    x, _ = VirtualKeyboard._clamp(anchor_right=1200, anchor_top=500, kb_w=400, kb_h=200, win_w=1000, win_h=700)
    assert x == 1000 - 400 - 4


# ── 真的視窗 ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tk_root():
    root = make_tk_root()
    yield root
    root.destroy()


@pytest.fixture
def kb(tk_root):
    win = tk.Toplevel(tk_root)
    win.geometry("900x600+0+0")
    entry = tk.Entry(win)
    entry.pack(side="bottom", fill="x")
    anchor = tk.Button(win, text="⌨")
    anchor.place(relx=1.0, rely=1.0, x=-10, y=-40, anchor="se")
    sent = []
    k = VirtualKeyboard(win, entry, anchor, on_enter=lambda: sent.append(entry.get()))
    k.sent = sent
    win.update()
    yield k
    win.destroy()


def test_show_places_keyboard_inside_window_above_anchor(kb):
    kb.show()
    kb.window.update()
    assert kb.visible
    f, w = kb.frame, kb.window
    x, y = f.winfo_x(), f.winfo_y()
    assert x >= 0 and y >= 0
    assert x + f.winfo_width() <= w.winfo_width()
    assert y + f.winfo_height() <= w.winfo_height()
    anchor_top = kb.anchor.winfo_rooty() - w.winfo_rooty()
    assert y + f.winfo_height() <= anchor_top  # 在按鈕上方，不蓋住按鈕


def test_toggle_and_close_key(kb):
    kb.toggle()
    assert kb.visible
    kb.toggle()
    assert not kb.visible
    kb.show()
    kb.press("CLOSE")
    assert not kb.visible


def test_typing_inserts_at_cursor(kb):
    kb.entry.insert(0, "ac")
    kb.entry.icursor(1)
    kb.press("b")
    assert kb.entry.get() == "abc"
    kb.press("SPACE")
    assert kb.entry.get() == "ab c"


def test_shift_is_one_shot_and_maps_symbols(kb):
    kb.show()
    kb.press("SHIFT")
    assert kb.shift
    kb.press("y")
    assert kb.entry.get() == "Y"
    assert not kb.shift  # 打完一個字自動放開
    kb.press("SHIFT")
    kb.press("1")
    kb.press("1")
    assert kb.entry.get() == "Y!1"


def test_backspace_and_clear(kb):
    kb.entry.insert(0, "hello")
    kb.press("BACKSPACE")
    assert kb.entry.get() == "hell"
    kb.entry.select_range(0, 2)
    kb.press("BACKSPACE")
    assert kb.entry.get() == "ll"
    kb.press("CLEAR")
    assert kb.entry.get() == ""


def test_enter_calls_on_enter(kb):
    kb.press("2")
    kb.press("ENTER")
    assert kb.sent == ["2"]


def test_disabled_entry_ignores_keys(kb):
    kb.entry.config(state="disabled")
    kb.press("1")
    kb.press("ENTER")
    assert kb.entry.get() == "" and kb.sent == []


def test_repositions_when_window_resized(kb):
    kb.show()
    kb.window.geometry("700x500")
    kb.window.update()
    kb.window.update()
    f, w = kb.frame, kb.window
    assert f.winfo_x() + f.winfo_width() <= w.winfo_width()
    assert f.winfo_y() + f.winfo_height() <= w.winfo_height()
