"""
Unit Test：設定視窗的「📡 IoT 子系統」分頁（settings_gui/iot_tab.py 掛進 settings_window.py 的方式）

用真的 SettingsWindow（會短暫出現一個視窗）；不存檔。會啟動的只有暫存資料夾裡的**假服務**
（服務清單換掉，不會碰到使用者真的 python -m iot／iot.voice）；ui_state.json 的寫入也換成不寫檔。
沒有顯示環境時跳過。
"""

import time
import tkinter as tk

import pytest

import settings_manager
import settings_window as sw
from settings_gui import iot_services, iot_tab
from settings_manager import TAB_ORDER


@pytest.fixture
def window(monkeypatch, capfd):
    def _refuse_save(*_a, **_k):
        raise AssertionError("測試不應該寫入設定檔")

    monkeypatch.setattr(settings_manager, "save_runtime_settings", _refuse_save)
    monkeypatch.setattr(iot_tab._ui_state, "update", lambda **k: None)   # 字級、高度不要寫進真的 ui_state.json
    try:
        with capfd.disabled():
            w = sw.SettingsWindow()
    except tk.TclError:
        pytest.skip("沒有可用的顯示環境")
    yield w
    try:
        w.destroy()
    except tk.TclError:   # 測試裡已經關掉了
        pass


@pytest.fixture
def fake_services(tmp_path, monkeypatch):
    """服務清單換成兩個假服務（只會睡覺的小模組），狀態檔、紀錄檔都在暫存資料夾。"""
    mods = tmp_path / "mods"
    mods.mkdir()
    (mods / "fake_close_svc.py").write_text("import time\nwhile True:\n    time.sleep(0.2)\n", encoding="utf-8")
    env = {"PYTHONPATH": str(mods)}
    services = (iot_services.Service(key="fa", title="🎙 假甲", desc="", module="fake_close_svc", env=env),)
    monkeypatch.setattr(iot_services, "SERVICES", services)
    monkeypatch.setattr(iot_services, "_BY_KEY", {s.key: s for s in services})
    monkeypatch.setattr(iot_services, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(iot_services, "STATE_FILE", tmp_path / "state.json")
    yield services
    for s in services:
        st = iot_services.scan()[s.key]
        if st.running:
            iot_services.stop(s.key, pid=st.pid)


def test_closing_window_force_stops_iot_services(fake_services, window):
    ok, msg = iot_services.start("fa")
    assert ok, msg
    pid = iot_services.scan()["fa"].pid
    assert iot_tab.running_titles() == ["🎙 假甲"]
    window._on_window_close()                        # 右上角 X／「關閉」按鈕都走這裡
    assert not iot_services.scan()["fa"].running     # 09-27：關掉設定視窗 → IoT 服務一起強制關閉
    assert "設定視窗關閉，一併停止" in iot_services.tail("fa") and f"PID {pid}" in iot_services.tail("fa")


def test_log_zoom_and_drag(window):
    tab = iot_tab.IotTab(tk.Frame(window), window)
    size = tab._log_font.cget("size")
    tab._zoom(+1)
    assert tab._log_font.cget("size") == size + 1 and "字級" in tab._zoom_var.get()
    tab._zoom(0)
    assert tab._log_font.cget("size") == iot_tab._LOG_FONT_DEFAULT
    for _ in range(50):
        tab._zoom(-1)
    assert tab._log_font.cget("size") == iot_tab._LOG_FONT_MIN          # 有下限

def test_log_drag_and_fit_above_console(window):
    """09-27 改版：紀錄框標題列往上拖＝變高；整頁固定在全域終端機上方，終端機變高時紀錄框不會被蓋住。"""
    class E:   # 假的拖拉事件
        def __init__(self, y): self.y_root = y
    window.geometry("1400x1000+10+10")
    window._select_tab(iot_tab.TAB_NAME)
    window.update()
    body = window._tab_frames[iot_tab.TAB_NAME].body
    tab = next(o for o in _iot_tabs() if o.root.master is body)   # settings_window 建的那個分頁物件
    console = window._console_panel

    def settle():
        for _ in range(5):
            window.update()

    def log_bottom():
        return tab._log_panel.winfo_rooty() + tab._log_panel.winfo_height()

    for h in (120, 200):   # 終端機高度改變 → 紀錄框下緣一直在終端機上緣之上
        if console._collapsed:
            console.toggle_collapse()
        console.place(h)
        settle()
        assert log_bottom() <= console.container.winfo_rooty() + 1, h
    console.toggle_collapse()   # 收合全域終端機，拖拉空間大一點
    settle()
    start = iot_tab._LOG_HEIGHT_MIN
    tab._set_log_height(start)
    assert tab._log_height_max() >= start + 60, "測試視窗太矮"
    tab._drag_start(E(500))
    tab._drag_move(E(440))                                                 # 往上拖 60 → 高 60
    assert int(tab._log_panel.cget("height")) == start + 60
    tab._drag_move(E(5000))
    assert int(tab._log_panel.cget("height")) == iot_tab._LOG_HEIGHT_MIN   # 有下限
    tab._drag_move(E(-5000))
    assert int(tab._log_panel.cget("height")) == tab._log_height_max()     # 有上限
    settle()
    # 上限跟全域終端機一樣：上緣最高到「底部按鈕列上緣 − console.max_height()」（超過分頁，蓋住卡片和分頁列）
    top = tab._log_panel.winfo_rooty() - window.winfo_rooty()
    assert abs(top - (window._bottom_bar_frame.winfo_y() - console.max_height())) <= 1
    assert tab._log_height_max() > tab._space()
    assert log_bottom() <= console.container.winfo_rooty() + 1
    tab._drag_move(E(440))
    tab._drag_end(E(440))
    ratio = tab._log_ratio
    assert abs(ratio - (start + 60) / tab._space()) < 0.01                # 記的是比例
    # 全域終端機展開 → 分頁變矮，紀錄框依比例一起縮、仍在終端機上方；收回去 → 長回原來高度
    console.toggle_collapse()
    console.place(200)
    settle()
    assert log_bottom() <= console.container.winfo_rooty() + 1
    assert tab._log_ratio == ratio
    console.toggle_collapse()
    settle()
    assert int(tab._log_panel.cget("height")) == start + 60
    tab._reset_height()                                                    # 雙擊標題列：回預設比例
    assert tab._log_ratio == iot_tab._LOG_RATIO_DEFAULT
    assert window._tab_frames[iot_tab.TAB_NAME].canvas.yview()[0] == 0.0   # 外層分頁不捲
    # 紀錄框浮在視窗上：切到別的分頁要藏起來，切回來再出現
    window._select_tab(TAB_ORDER[0])
    settle()
    assert not tab._log_panel.winfo_ismapped()
    window._select_tab(iot_tab.TAB_NAME)
    settle()
    assert tab._log_panel.winfo_ismapped()


def test_log_view_syncs_within_a_second(fake_services, window):
    """紀錄檔多一行 → 紀錄框 1 秒內顯示（每 0.5 秒 stat 一次，不用等 2 秒的狀態掃描）。"""
    window._select_tab(iot_tab.TAB_NAME)
    window.update()
    body = window._tab_frames[iot_tab.TAB_NAME].body
    tab = next(o for o in _iot_tabs() if o.root.master is body)
    tab._select("fa")
    log = iot_services.get("fa").log_path
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write("2026-09-27 04:10:00 [INFO] 體表溫度 36.5°C\n")
    t0 = time.time()
    while time.time() - t0 < 1.5 and "體表溫度 36.5°C" not in tab._log.get("1.0", "end"):
        window.update()
        time.sleep(0.02)
    assert "體表溫度 36.5°C" in tab._log.get("1.0", "end")
    assert time.time() - t0 < 1.0


def _iot_tabs():
    import gc
    return [o for o in gc.get_objects() if isinstance(o, iot_tab.IotTab)]


def test_iot_tab_is_after_settings_tabs_and_not_prebuilt(window):
    names = list(window._tab_buttons)
    assert names[:len(TAB_ORDER)] == list(TAB_ORDER) and names[-1] == iot_tab.TAB_NAME
    assert iot_tab.TAB_NAME not in TAB_ORDER                      # 不是設定分頁（沒有欄位）
    assert iot_tab.TAB_NAME not in window._tabs_built             # 點到才建


def test_start_stop_all_buttons_grey_out(window):
    window._select_tab(iot_tab.TAB_NAME)
    window.update()
    tab = iot_tab.IotTab(tk.Frame(window), window)   # 另建一個分頁物件直接餵假狀態（不掃真的行程）
    run = {s.key: iot_services.Status(True, 1) for s in iot_services.SERVICES}
    stop = {s.key: iot_services.Status(False) for s in iot_services.SERVICES}
    tab._apply(run, (True, "x"))
    assert tab._btn_start_all._state == "disabled" and tab._btn_stop_all._state == "normal"   # 全部在跑：全部啟動變灰
    tab._apply(stop, (True, "x"))
    assert tab._btn_start_all._state == "normal" and tab._btn_stop_all._state == "disabled"   # 全部停了：全部停止變灰
    half = dict(stop, voice=iot_services.Status(True, 1))
    tab._apply(half, (True, "x"))
    assert tab._btn_start_all._state == "normal" and tab._btn_stop_all._state == "normal"


def test_select_iot_tab_builds_cards(window):
    window._select_tab(iot_tab.TAB_NAME)
    window.update()
    assert iot_tab.TAB_NAME in window._tabs_built
    texts = []

    def walk(w):
        for c in w.winfo_children():
            if isinstance(c, tk.Label):
                texts.append(str(c.cget("text")))
            walk(c)

    walk(window._tab_frames[iot_tab.TAB_NAME].body)
    for s in iot_services.SERVICES:
        assert s.title in texts, f"少了「{s.title}」卡片"
    assert any("python -m iot.voice" in t for t in texts)
    # 切回設定分頁再切回來不會重建
    window._select_tab(TAB_ORDER[0])
    window._select_tab(iot_tab.TAB_NAME)


def test_config_dialog_save_and_validate(window, tmp_path, monkeypatch):
    """⚙ 參數設定：填值存檔（空白不存）、下限≥上限時擋下不存。"""
    from settings_gui import iot_config_dialog, iot_config_overrides as ov
    monkeypatch.setattr(ov, "OVERRIDES_FILE", tmp_path / "ov.json")
    shown = []
    monkeypatch.setattr(iot_config_dialog.dialogs, "show_info", lambda *a: shown.append(("info", a[2])))
    monkeypatch.setattr(iot_config_dialog.dialogs, "show_error", lambda *a: shown.append(("error", a[2])))
    monkeypatch.setattr(iot_config_dialog.iot_services, "SERVICES", ())   # 沒有執行中的感測器 → 不問重新啟動

    dlg = iot_config_dialog.open_dialog(window)
    assert set(dlg.vars) == {f.env for f in ov.FIELDS}
    dlg.vars["CAT_MONITORING_IOT_TEMP_MIN_C"].set("30")                  # 預設上限 28 → 擋下
    dlg._save()
    assert shown[-1][0] == "error" and not ov.OVERRIDES_FILE.exists()
    dlg.vars["CAT_MONITORING_IOT_TEMP_MAX_C"].set("33")
    dlg._save()
    assert shown[-1][0] == "info"
    assert ov.load() == {"CAT_MONITORING_IOT_TEMP_MIN_C": "30", "CAT_MONITORING_IOT_TEMP_MAX_C": "33"}

    dlg = iot_config_dialog.open_dialog(window)                        # 重開：帶出已存的值，黃底
    assert dlg.vars["CAT_MONITORING_IOT_TEMP_MAX_C"].get() == "33"
    assert dlg.entries["CAT_MONITORING_IOT_TEMP_MAX_C"].cget("bg") == iot_config_dialog._ENTRY_CHANGED
    dlg._clear_all()
    dlg._save()
    assert ov.load() == {}


def test_log_panel_never_covers_global_console(window):
    """09-27：全域終端機拉高時，IoT 紀錄框（浮在最上層）不能蓋住它；上方放不下就藏起來。"""
    window._select_tab(iot_tab.TAB_NAME)
    console = window._console_panel
    if console._collapsed:
        console.toggle_collapse()
    panels = lambda: [p for p in window.place_slaves() if p is not console.container]   # noqa: E731
    for h in (150, 450, 700, 2000, 150):
        console.place(h)
        for _ in range(5):
            window.update()
        top = console.container.winfo_rooty()
        for p in panels():
            if p.winfo_ismapped():
                assert p.winfo_rooty() + p.winfo_height() <= top, f"終端機高 {h}：IoT 紀錄框蓋住終端機"
    assert any(p.winfo_ismapped() for p in panels())   # 終端機縮回來，紀錄框要再出現


def test_merged_log_view(fake_services, window, tmp_path, monkeypatch):
    """🔀 整合檢視：標題、每行前面標服務名稱；卡片的「📄 紀錄」切回單一服務。"""
    monkeypatch.setattr(iot_tab._ui_state, "get", lambda key, default=None: default)
    p = iot_services.get("fa").log_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("2026-09-27 10:00:01,000 [WARNING] iot.runner: 沒收到資料\n", encoding="utf-8")
    tab = iot_tab.IotTab(tk.Frame(window), window)
    tab._toggle_merged()
    assert tab._merged and "全部 IoT 服務" in tab._log_title.get()
    text = tab._log.get("1.0", "end")
    assert "[假甲] 2026-09-27 10:00:01,000 [WARNING]" in text
    assert "warn_line" in tab._log.tag_names("1.10")                  # WARNING 行還是紅字
    tab._select("fa", merged=False)
    assert not tab._merged and "假甲 的紀錄" in tab._log_title.get()
    assert tab._log.get("1.0", "end").startswith("2026-09-27")        # 單一服務：沒有前綴
