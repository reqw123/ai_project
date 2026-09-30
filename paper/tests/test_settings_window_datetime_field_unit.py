"""
Unit Test：settings_window.py 的挖空式日期時間欄位（影片開始錄影時間）與分頁底部捲動空白

1. 「影片開始錄影時間」是 [年]年[月]月[日]日 [時]:[分] 五格：寫入值會拆進五格、
   打數字會組回 "YYYY-MM-DD HH:MM"，沒填齊時存檔驗證會擋下。
2. 終端機面板浮在表單上、蓋住捲動區最下面一截：每個分頁的捲動範圍要多留那段高度，
   捲到最底時最後一列（本分頁就是這個欄位）要完整出現在終端機上方。

用真的 SettingsWindow（會短暫出現一個視窗）；save_runtime_settings 換成直接失敗，
不會改到真正的設定檔。沒有顯示環境時跳過。
"""

import tkinter as tk

import pytest

import settings_manager
import settings_window as sw

KEY = "run_mode.video_start_time"


@pytest.fixture
def window(monkeypatch, capfd):
    def _refuse_save(*_a, **_k):
        raise AssertionError("測試不應該寫入設定檔")

    monkeypatch.setattr(settings_manager, "save_runtime_settings", _refuse_save)
    try:
        with capfd.disabled():  # 見 test_settings_window_lazy_tabs_unit.py 的說明
            w = sw.SettingsWindow()
    except tk.TclError:
        pytest.skip("沒有可用的顯示環境")
    w._select_tab("執行模式與排程")
    w.update()
    yield w
    w.destroy()


@pytest.mark.parametrize("value, parts, logical", [
    ("2026-09-30 18:00", ["2026", "09", "30", "18", "00"], "2026-09-30 18:00"),
    ("2026-9-3 8:05", ["2026", "09", "03", "08", "05"], "2026-09-03 08:05"),
    ("2026-09-30T18:00:30", ["2026", "09", "30", "18", "00"], "2026-09-30 18:00"),
    ("", [""] * 5, ""),
    ("亂打的", [""] * 5, ""),
])
def test_value_is_split_into_five_boxes(window, value, parts, logical):
    window._set_field_value(KEY, value)
    assert [p.get() for p in window._field(KEY)["_dt_parts"]] == parts
    assert window._get_field_value(KEY) == (logical, None)


def test_typed_digits_become_a_valid_value(window):
    for p, t in zip(window._field(KEY)["_dt_parts"], ["2026", "9", "30", "18", "0"]):
        p.set(t)
    value, _ = window._get_field_value(KEY)
    assert value == "2026-09-30 18:00"
    assert settings_manager._validate_video_start(value, "x") is None


def test_partially_filled_is_rejected_on_save(window):
    for p, t in zip(window._field(KEY)["_dt_parts"], ["2026", "09", "", "18", "00"]):
        p.set(t)
    value, _ = window._get_field_value(KEY)
    assert settings_manager._validate_video_start(value, "x") is not None


def test_last_row_scrolls_above_the_console_panel(window):
    tab = window._tab_frames["執行模式與排程"]
    window.update()
    tab.canvas.yview_moveto(1.0)
    window.update()
    field_bottom = window._field(KEY)["container"].winfo_rooty() + window._field(KEY)["container"].winfo_height()
    console_top = window._console_panel.container.winfo_rooty()
    assert field_bottom <= console_top
