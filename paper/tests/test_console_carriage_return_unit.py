"""
Unit Test：settings_gui/console_panel.py 的 ConsolePanel.append() 對 "\\r" 的處理

子行程輸出是原始位元組解碼，Windows 的一般換行以 "\\r\\n" 進來，單獨的 "\\r" 是 tqdm
進度條「回到行首重畫」。重點在：進度條原地覆寫同一行（不再接成一條超長的行，那會讓
Tk 每次插入都重排整行、把主執行緒卡住）、"\\r\\n" 當一般換行、被 os.read 切開的
"\\r" + "\\n" 仍當換行。用真的 tk.Text 驅動，沒有顯示環境時自動略過。
"""

import tkinter as tk
from types import SimpleNamespace

import pytest

from conftest import make_tk_root
from settings_gui.console_panel import ConsolePanel


@pytest.fixture(scope="module")
def tk_root():
    root = make_tk_root()
    yield root
    root.destroy()


@pytest.fixture
def panel(tk_root):
    text = tk.Text(tk_root)
    stub = SimpleNamespace(
        text=text,
        autoscroll_var=tk.BooleanVar(tk_root, value=True),
        _preview_var=tk.StringVar(tk_root),
        _log_line_count=0,
        _cr_pending=False,
        _last_line_is_progress=False,
        _last_output_monotonic=None,
        _last_chunk_ends_newline=True,
    )
    stub.append = lambda text, tag=None: ConsolePanel.append(stub, text, tag)
    stub.content = lambda: stub.text.get("1.0", "end-1c")
    yield stub
    text.destroy()


def test_plain_crlf_becomes_newline(panel):
    panel.append("a\r\nb\r\n")
    assert panel.content() == "a\nb\n"
    assert panel._log_line_count == 2


def test_progress_bar_overwrites_same_line(panel):
    panel.append("開始訓練\r\n")
    for i in range(1, 6):
        panel.append(f"\rTraining: {i * 20}%")
    assert panel.content() == "開始訓練\nTraining: 100%"
    panel.append("\r\n下一行\r\n")
    assert panel.content() == "開始訓練\nTraining: 100%\n下一行\n"


def test_multiple_updates_in_one_chunk_keep_last(panel):
    panel.append("\rStep 1\rStep 2\rStep 3\r\nDone\r\n")
    assert panel.content() == "Step 3\nDone\n"


def test_crlf_split_across_chunks_is_newline(panel):
    panel.append("line1\r")
    panel.append("\nline2\r\n")
    assert panel.content() == "line1\nline2\n"


def test_trailing_cr_then_progress_overwrites(panel):
    panel.append("50%\r")
    panel.append("60%")
    assert panel.content() == "60%"


def test_progress_line_flag_for_input_detection(panel):
    """最後一行是 \\r 進度列 → 標成進度（不算等輸入）；之後印出沒換行的提示字 → 不是進度。"""
    panel.append("\rTraining: 40%")
    assert panel._last_line_is_progress
    panel.append("\r\n請選擇模式 (1/2): ")
    assert not panel._last_line_is_progress
    panel.append("50%\r")
    assert panel._last_line_is_progress


def test_overwrite_does_not_touch_previous_lines(panel):
    panel.append("keep me\r\nold progress")
    panel.append("\rnew progress")
    assert panel.content() == "keep me\nnew progress"


# ── 標題列「1」「2」快速回答鈕：共用 _send_stdin_text ─────────────────────


def test_quick_answer_sends_and_echoes_without_touching_entry(panel):
    sent = []
    panel._stdin_handler = lambda text: (sent.append(text) or True, None)
    panel.stdin_var = tk.StringVar(panel.text, value="正在打的字")
    assert ConsolePanel._send_stdin_text(panel, "2") is True
    assert sent == ["2"]
    assert panel.content().endswith("> 2\n")
    assert panel.stdin_var.get() == "正在打的字"  # 快速回答不清掉輸入框


def test_quick_answer_without_running_process_is_silent(panel):
    panel._stdin_handler = lambda text: (False, None)
    assert ConsolePanel._send_stdin_text(panel, "1") is False
    assert panel.content() == ""
