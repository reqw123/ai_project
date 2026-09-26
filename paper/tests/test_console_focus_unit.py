"""
Unit Test：settings_gui/console_panel.py 的終端機輸入焦點與「疑似在等輸入」提醒

啟動腳本時刻意「不」自動搶焦點（多數工具不會 input()，搶了反而跟腳本的預覽視窗互搶）；
改成真的在等輸入時，輸入列邊框閃爍＋提示文字換色，使用者按 Ctrl+I 自己過去。
用真的 Tk 視窗，但不建整個 ConsolePanel／設定視窗，而是用 __new__ 造一個只帶
這些邏輯會用到的屬性的殼。沒有顯示環境時自動略過。
"""

import time
import tkinter as tk

import pytest

from conftest import make_tk_root
from settings_gui.console_panel import ConsolePanel
from settings_gui.style import COLOR_CONSOLE_BG


@pytest.fixture(scope="module")
def tk_root():
    root = make_tk_root()
    root.geometry("300x120+0+0")
    root.update()
    yield root
    root.destroy()


@pytest.fixture
def panel(tk_root):
    win = tk.Toplevel(tk_root)
    win.geometry("300x160+0+0")
    other = tk.Entry(win)  # 代表「本視窗的其他控制項」
    other.pack()
    row = tk.Frame(win, highlightthickness=2, highlightbackground=COLOR_CONSOLE_BG)
    row.pack()
    entry = tk.Entry(row, state="disabled")
    entry.pack()
    hint = tk.Label(row, text=ConsolePanel._INPUT_HINT_IDLE)
    hint.pack()
    btn = tk.Button(row, state="disabled")
    quick = [tk.Button(row, text=t, state="disabled") for t in ("2", "1")]
    win.update()
    p = ConsolePanel.__new__(ConsolePanel)
    p.window = win
    p.stdin_entry = entry
    p.send_stdin_btn = btn
    p._quick_answer_btns = quick
    p._input_row = row
    p._input_hint_label = hint
    p._input_alert_since = None
    p._last_output_monotonic = None
    p._last_chunk_ends_newline = True
    p._last_line_is_progress = False
    p.other = other
    entry.bind("<FocusIn>", lambda _e: p._stop_input_alert(), add="+")
    yield p
    win.destroy()


def _focus_is_entry(p):
    return p.window.focus_get() is p.stdin_entry


def _simulate_prompt(p, idle_sec=5.0, progress=False):
    """模擬子行程印出沒有換行的提示字（或 \\r 進度列）後，已經 idle_sec 秒沒有新輸出。"""
    p.set_input_enabled(True)
    p._last_chunk_ends_newline = False
    p._last_line_is_progress = progress
    p._last_output_monotonic = time.monotonic() - idle_sec


def _alerting(p):
    return p._input_alert_since is not None


# ── 焦點 ──────────────────────────────────────────────────────────────


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


def test_ctrl_i_shortcut_jumps_to_input(tk_root, panel):
    panel.set_input_enabled(True)
    panel.window.focus_force()
    panel.other.focus_force()
    tk_root.update()
    assert panel._on_focus_input_shortcut() == "break"
    tk_root.update()
    assert _focus_is_entry(panel)


def test_no_auto_focus_api_left():
    """啟動後自動搶焦點的機制已移除，避免之後又被誤接回去。"""
    assert not hasattr(ConsolePanel, "focus_input_soon")
    assert not hasattr(ConsolePanel, "FOCUS_RETRY_DELAYS_MS")


# ── 等輸入提醒 ────────────────────────────────────────────────────────


def test_alert_turns_on_when_waiting_for_input(tk_root, panel):
    panel.other.focus_force()
    tk_root.update()
    _simulate_prompt(panel)
    panel._update_input_alert()
    assert _alerting(panel)
    assert panel._input_hint_label.cget("text") == ConsolePanel._INPUT_HINT_ALERT
    assert panel._input_row.cget("highlightbackground") == ConsolePanel._INPUT_ALERT_COLOR


def test_alert_ignores_progress_bar_line(tk_root, panel):
    _simulate_prompt(panel, progress=True)
    panel._update_input_alert()
    assert not _alerting(panel)


def test_alert_waits_for_idle_threshold(tk_root, panel):
    _simulate_prompt(panel, idle_sec=0.5)  # 提示字剛印出來，還沒超過門檻
    panel._update_input_alert()
    assert not _alerting(panel)


def test_alert_stops_when_user_focuses_input(tk_root, panel):
    panel.other.focus_force()
    tk_root.update()
    _simulate_prompt(panel)
    panel._update_input_alert()
    assert _alerting(panel)
    panel.focus_input(force=True)
    tk_root.update()
    assert not _alerting(panel)
    assert panel._input_hint_label.cget("text") == ConsolePanel._INPUT_HINT_IDLE
    assert panel._input_row.cget("highlightbackground") == COLOR_CONSOLE_BG


def test_alert_stops_when_output_resumes(tk_root, panel):
    _simulate_prompt(panel)
    panel._update_input_alert()
    assert _alerting(panel)
    panel._last_chunk_ends_newline = True  # 使用者回答後腳本繼續印出完整的行
    panel._last_output_monotonic = time.monotonic()
    panel._update_input_alert()
    assert not _alerting(panel)


def test_alert_stops_when_process_ends(tk_root, panel):
    _simulate_prompt(panel)
    panel._update_input_alert()
    assert _alerting(panel)
    panel.set_input_enabled(False)
    assert not _alerting(panel)


def test_alert_blinks_then_stays_lit(tk_root, panel):
    _simulate_prompt(panel)
    panel._update_input_alert()
    period = ConsolePanel._INPUT_ALERT_BLINK_PERIOD
    panel._input_alert_since = time.monotonic() - period * 1.5  # 閃爍期間的「暗」半拍
    panel._update_input_alert()
    assert panel._input_row.cget("highlightbackground") == COLOR_CONSOLE_BG
    panel._input_alert_since = time.monotonic() - ConsolePanel._INPUT_ALERT_BLINK_SEC - period * 1.5
    panel._update_input_alert()  # 超過閃爍期就一直亮著
    assert panel._input_row.cget("highlightbackground") == ConsolePanel._INPUT_ALERT_COLOR


# ── 標題列「1」「2」快速回答鈕 ────────────────────────────────────────


def test_quick_answer_buttons_follow_input_enabled(tk_root, panel):
    assert all(str(b["state"]) == "disabled" for b in panel._quick_answer_btns)
    panel.set_input_enabled(True)
    assert all(str(b["state"]) == "normal" for b in panel._quick_answer_btns)
    panel.set_input_enabled(False)
    assert all(str(b["state"]) == "disabled" for b in panel._quick_answer_btns)


def test_quick_answer_buttons_turn_amber_during_alert(tk_root, panel):
    panel.other.focus_force()
    tk_root.update()
    _simulate_prompt(panel)
    panel._update_input_alert()
    assert all(b.cget("bg") == ConsolePanel._INPUT_ALERT_COLOR for b in panel._quick_answer_btns)
    panel.set_input_enabled(False)
    assert all(b.cget("bg") != ConsolePanel._INPUT_ALERT_COLOR for b in panel._quick_answer_btns)
