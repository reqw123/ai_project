"""
Unit Test：settings_window.py 的分頁延遲建立

開啟時只建第一個分頁，其餘分頁在「被切換到／程式要讀寫它的欄位／背景補建」時
才建。這裡驗證兩件事：
1. 維護規則：self._field_widgets 只能由幾個固定方法直接存取，其餘一律透過
   self._field()／self._field_if_built()——否則讀寫到還沒建的分頁欄位會 KeyError，
   而且只在「背景還沒建完就操作」時才出現，平常很難發現。
2. 背景還沒建完時的操作（存檔收集表單、Node-RED 端點連動、搜尋高亮、切分頁）
   結果跟全部分頁都建好時一致。

用真的 SettingsWindow（會短暫出現一個視窗）；只讀取目前設定，不會存檔——
save_runtime_settings 被換成直接失敗，萬一測試誤觸存檔也不會改到真正的設定檔。
沒有顯示環境時跳過。
"""

import ast
import tkinter as tk
from pathlib import Path

import pytest

import settings_manager
import settings_window as sw
from settings_manager import FIELD_SCHEMA, TAB_ORDER

_SOURCE = Path(sw.__file__).read_text(encoding="utf-8")

# 可以直接存取 self._field_widgets 的方法：宣告、寫入、兩個存取函式本身
_ALLOWED_DIRECT_ACCESS = {"__init__", "_build_field_row", "_field", "_field_if_built"}


def test_field_widgets_only_accessed_through_accessors():
    offenders = []

    def visit(node, func_name):
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else func_name
            if (
                isinstance(child, ast.Attribute)
                and child.attr == "_field_widgets"
                and isinstance(child.value, ast.Name)
                and child.value.id == "self"
                and func_name not in _ALLOWED_DIRECT_ACCESS
            ):
                offenders.append(f"第 {child.lineno} 行（{func_name}）")
            visit(child, name)

    visit(ast.parse(_SOURCE), None)
    assert not offenders, (
        "分頁是延遲建立的，讀寫欄位請改用 self._field(key)（還沒建就先建）或 "
        "self._field_if_built(key)（還沒建回傳 None），不要直接存取 self._field_widgets：\n"
        + "\n".join(offenders)
    )


@pytest.fixture
def window(monkeypatch, capfd):
    def _refuse_save(*_a, **_k):
        raise AssertionError("測試不應該寫入設定檔")

    monkeypatch.setattr(settings_manager, "save_runtime_settings", _refuse_save)
    try:
        # Windows 上 pytest 攔截標準輸出入時，Tcl 初始化偶爾會讀不到 init.tcl
        # （TclError "couldn't read file ... init.tcl: No error"）；建立視窗的
        # 當下暫時關掉攔截就不會發生。
        with capfd.disabled():
            w = sw.SettingsWindow()
    except tk.TclError:
        pytest.skip("沒有可用的顯示環境")
    # 不呼叫 w.update()：背景補建是 after() 計時器，不跑事件迴圈就不會觸發，
    # 測試才能停在「只建了第一頁」的狀態。
    yield w
    w.destroy()


def test_only_first_tab_built_on_open(window):
    assert window._tabs_built == {TAB_ORDER[0]}
    first_tab_keys = {f["json_key"] for f in FIELD_SCHEMA if f["tab"] == TAB_ORDER[0]}
    assert set(window._field_widgets) == first_tab_keys


def test_collect_form_before_background_equals_fully_built(window):
    data_lazy, errors_lazy = window._collect_form_data()  # 大部分分頁還沒建 → 當場建
    assert window._tabs_built == set(TAB_ORDER)

    # 全部分頁都建好後，把所有欄位重新填成目前生效值再收集一次，要跟上面一致
    window._populate_from_effective_state()
    data_full, errors_full = window._collect_form_data()
    assert data_lazy == data_full
    assert errors_lazy == errors_full


def test_select_unbuilt_tab_builds_it(window):
    tab = TAB_ORDER[-1]
    assert tab not in window._tabs_built
    window._select_tab(tab)
    assert tab in window._tabs_built
    assert window._active_tab == tab


def test_background_build_finishes_all_tabs(window):
    window._build_pending_tabs_in_background()
    for _ in range(500):
        if window._tabs_built == set(TAB_ORDER):
            break
        window.update()
        window.after(10)
    assert window._tabs_built == set(TAB_ORDER)


def test_nodered_endpoints_follow_host_when_advanced_tab_not_built(window):
    window._select_tab("Flask 與 Node-RED")
    assert "進階設定" not in window._tabs_built

    host = window._field("nodered.host")["var"]
    port = window._field("nodered.port")["var"].get()
    old_host = host.get()
    suffixes = {
        "advanced.nodered_endpoint_notify": "python_online",
        "advanced.nodered_endpoint_result": "yolo_result",
        "advanced.nodered_endpoint_result_v2": "yolo_result_v2",
    }
    by_key = {f["json_key"]: f for f in FIELD_SCHEMA}
    before = {k: str(window._resolve_field_display(by_key[k])[0]) for k in suffixes}

    host.set("10.9.8.7")

    for key, suffix in suffixes.items():
        now = window._field(key)["var"].get()
        if before[key] == f"http://{old_host}:{port}/{suffix}":
            assert now == f"http://10.9.8.7:{port}/{suffix}", key   # 沒被手改過 → 跟著同步
        else:
            assert now == before[key], key                           # 刻意改過 → 不動


def test_search_highlight_applies_to_tab_built_later(window):
    later_tab = "行為追蹤與警報門檻"
    assert later_tab not in window._tabs_built
    first_key = next(f["json_key"] for f in FIELD_SCHEMA if f["tab"] == TAB_ORDER[0])
    later_key = next(f["json_key"] for f in FIELD_SCHEMA if f["tab"] == later_tab)

    window._highlight_fields([first_key, later_key])
    assert later_tab not in window._tabs_built  # 高亮本身不會觸發建立
    window._ensure_tab_built(later_tab)

    for key in (first_key, later_key):
        assert int(window._field(key)["container"].cget("highlightthickness")) == 3, key

    window._highlight_fields([])
    assert int(window._field(later_key)["container"].cget("highlightthickness")) == 0
