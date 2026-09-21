"""
Unit Test：settings_gui/console_panel.py 的「啟動腳本後把焦點鎖到終端機輸入框」

只測焦點邏輯：用真的 Tk 視窗，但不建整個 ConsolePanel／設定視窗，
而是用 __new__ 造一個只帶 window、stdin_entry 的殼（焦點邏輯只用到這兩個屬性）。
"""

import time
import tkinter as tk

import pytest

from settings_gui.console_panel import ConsolePanel


@pytest.fixture(scope="module")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("沒有可用的顯示環境")
    root.geometry("300x120+0+0")
    root.update()
    yield root
    root.destroy()


@pytest.fixture
def panel(tk_root):
    win = tk.Toplevel(tk_root)
    win.geometry("300x120+0+0")
    other = tk.Entry(win)  # 代表「本視窗的其他控制項」
    other.pack()
    entry = tk.Entry(win, state="disabled")
    entry.pack()
    win.update()
    p = ConsolePanel.__new__(ConsolePanel)
    p.window = win
    p.stdin_entry = entry
    p.other = other
    yield p
    win.destroy()


def _pump(root, ms):
    end = time.time() + ms / 1000
    while time.time() < end:
        root.update()
        time.sleep(0.01)


def _focus_is_entry(p):
    return p.window.focus_get() is p.stdin_entry


def test_disabled_input_is_not_focused(tk_root, panel):
    assert panel.focus_input(force=True) is False
    assert not _focus_is_entry(panel)


def test_focus_input_puts_focus_and_cursor_at_end(tk_root, panel):
    panel.stdin_entry.config(state="normal")
    panel.stdin_entry.insert(0, "abc")
    assert panel.focus_input(force=True) is True
    tk_root.update()
    assert _focus_is_entry(panel)
    assert panel.stdin_entry.index("insert") == 3


def test_focus_soon_takes_focus_from_a_widget_that_stole_it(tk_root, panel):
    """啟動後別的東西（例如預覽視窗跳出來）把焦點搶走，重試的時間點內要搶回輸入框。"""
    panel.stdin_entry.config(state="normal")
    panel.window.focus_force()
    panel.window.update()
    panel.window.focus_set()  # 焦點落在視窗本身（＝啟動確認對話框關掉後的狀態）
    panel.focus_input_soon()
    _pump(tk_root, 200)
    assert _focus_is_entry(panel)


def test_focus_soon_respects_a_control_the_user_clicked(tk_root, panel):
    """使用者已經自己把焦點放到別的控制項，就不能硬搶回來。"""
    panel.stdin_entry.config(state="normal")
    panel.window.focus_force()
    panel.other.focus_force()
    panel.window.update()
    assert panel.window.focus_get() is panel.other
    panel.focus_input_soon()
    _pump(tk_root, 1800)  # 涵蓋所有重試時間點
    assert panel.window.focus_get() is panel.other


def test_focus_soon_survives_window_being_closed(tk_root, panel):
    panel.stdin_entry.config(state="normal")
    panel.focus_input_soon()
    panel.window.destroy()
    _pump(tk_root, 200)  # 已排程的 after 觸發時視窗不在了，不應丟例外
    # fixture 收尾會再 destroy 一次，這裡重建一個空殼避免 TclError
    panel.window = tk.Toplevel(tk_root)


def test_retry_schedule_stops_early_enough_not_to_steal_the_preview_window():
    assert max(ConsolePanel.FOCUS_RETRY_DELAYS_MS) <= 2000
    assert ConsolePanel.FOCUS_RETRY_DELAYS_MS[0] == 0
