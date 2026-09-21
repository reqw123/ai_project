"""
Unit Test：settings_gui/extra_env.py（設定視窗「⚙ 額外設定」的環境變數管理）

重點：沒設定＝完全不傳（維持各腳本自己的預設）、非法值不會被傳出去、存檔後能讀回、
按鈕摘要文字正確、對話框存檔會呼叫 on_change。狀態存在 ui_state.json，測試一律改指到 tmp_path，
不碰真實的 settings_gui/ui_state.json。
"""

import tkinter as tk

import pytest

from settings_gui import extra_env, ui_state


@pytest.fixture
def state_path(tmp_path, monkeypatch):
    p = tmp_path / "ui_state.json"
    monkeypatch.setattr(ui_state, "_STATE_PATH", p)
    return p


# ── 資料層 ────────────────────────────────────────────────────────────


def test_default_is_no_override(state_path):
    assert extra_env.load() == {f["env"]: "" for f in extra_env.FIELDS}
    assert {"DISPLAY_RESOLUTION", "CAT_MONITORING_STGCN_MODEL", "CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD"} <= set(extra_env.load())
    assert extra_env.collect_for_launch() == {}
    assert extra_env.summary() == ""
    assert extra_env.describe_for_launch() == ""


def test_save_and_load_roundtrip(state_path):
    extra_env.save({"DISPLAY_RESOLUTION": "1080p"})
    assert extra_env.load()["DISPLAY_RESOLUTION"] == "1080p"
    assert extra_env.collect_for_launch() == {"DISPLAY_RESOLUTION": "1080p"}
    assert extra_env.summary() == "1080p"
    assert extra_env.describe_for_launch() == "額外環境變數：DISPLAY_RESOLUTION=1080p"


def test_saving_empty_removes_the_override(state_path):
    extra_env.save({"DISPLAY_RESOLUTION": "720p"})
    extra_env.save({"DISPLAY_RESOLUTION": ""})
    assert extra_env.collect_for_launch() == {}
    assert "extra_env.DISPLAY_RESOLUTION" not in ui_state.load()


def test_invalid_choice_is_never_passed_on(state_path):
    extra_env.save({"DISPLAY_RESOLUTION": "4k"})  # 不在選項內：存檔時就丟掉
    assert extra_env.collect_for_launch() == {}
    # 檔案被手動改成非法值：讀出來也當成不覆寫
    ui_state.update(**{"extra_env.DISPLAY_RESOLUTION": "banana"})
    assert extra_env.load()["DISPLAY_RESOLUTION"] == ""
    assert extra_env.collect_for_launch() == {}


def test_does_not_disturb_other_ui_state(state_path):
    ui_state.update(last_tool_script="C:/x/a.py")
    extra_env.save({"DISPLAY_RESOLUTION": "720p"})
    assert ui_state.get("last_tool_script") == "C:/x/a.py"


def test_text_field_and_multi_summary(state_path, monkeypatch):
    fields = list(extra_env.FIELDS) + [{"env": "MY_TEXT", "label": "自訂", "type": "text", "hint": ""}]
    monkeypatch.setattr(extra_env, "FIELDS", fields)
    extra_env.save({"DISPLAY_RESOLUTION": "1080p", "MY_TEXT": "  hello  "})
    assert extra_env.collect_for_launch() == {"DISPLAY_RESOLUTION": "1080p", "MY_TEXT": "hello"}
    assert extra_env.summary() == "2 項"


# ── 對話框（真的 Tk 視窗）────────────────────────────────────────────


@pytest.fixture(scope="module")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("沒有可用的顯示環境")
    root.geometry("40x40+0+0")
    root.update()
    yield root
    root.destroy()


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _drive(root, script, on_change):
    errors = []

    def driver():
        dlg = None
        try:
            dlg = [w for w in _walk(root) if isinstance(w, tk.Toplevel)][-1]
            radios = {w.cget("value"): w for w in _walk(dlg) if isinstance(w, tk.Radiobutton)}
            buttons = {getattr(w, "_text", ""): w for w in _walk(dlg) if hasattr(w, "_command")}
            script(radios, buttons)
        except BaseException as e:  # noqa: BLE001 — Tk callback 裡的例外會被吞掉，收集後由測試本體重拋
            errors.append(e)
        finally:
            if dlg is not None and dlg.winfo_exists():
                dlg.destroy()

    root.after(150, driver)
    extra_env.open_dialog(root, on_change=on_change)
    if errors:
        raise errors[0]


def test_dialog_save_persists_and_notifies(tk_root, state_path):
    called = []

    def script(radios, buttons):
        assert set(radios) == {extra_env.NO_OVERRIDE, "720p", "1080p"}
        radios["1080p"].invoke()
        buttons["✓ 儲存"]._command()

    _drive(tk_root, script, lambda: called.append(True))
    assert extra_env.collect_for_launch() == {"DISPLAY_RESOLUTION": "1080p"}
    assert called == [True]


def test_dialog_cancel_changes_nothing(tk_root, state_path):
    called = []

    def script(radios, buttons):
        radios["720p"].invoke()
        buttons["取消"]._command()

    _drive(tk_root, script, lambda: called.append(True))
    assert extra_env.collect_for_launch() == {}
    assert called == []


def test_dialog_preselects_saved_value_and_reset_clears(tk_root, state_path):
    extra_env.save({"DISPLAY_RESOLUTION": "720p"})

    def script(radios, buttons):
        assert radios["720p"].getvar(radios["720p"].cget("variable")) == "720p"  # 預先選到已存的值
        buttons["↺ 全部不覆寫"]._command()
        buttons["✓ 儲存"]._command()

    _drive(tk_root, script, None)
    assert extra_env.collect_for_launch() == {}


def test_dialog_default_selects_exactly_one_no_override(tk_root, state_path):
    """回歸：「不覆寫」用空字串當單選鈕的值時，Tk 會進入三態、同組每顆都畫成已選（看起來像同時勾了好幾個）。
    現在預設應該只有「不覆寫」那一顆被選中，而且變數值不能等於 tristatevalue。"""
    def script(radios, buttons):
        anyr = next(iter(radios.values()))
        var = anyr.cget("variable")
        value = tk_root.getvar(var)
        assert value == extra_env.NO_OVERRIDE
        assert value != anyr.cget("tristatevalue")  # 不能落在三態值上
        selected = [v for v, r in radios.items() if tk_root.getvar(r.cget("variable")) == r.cget("value")]
        assert selected == [extra_env.NO_OVERRIDE]
        # 改選 1080p 後仍然只有一顆
        radios["1080p"].invoke()
        selected = [v for v, r in radios.items() if tk_root.getvar(r.cget("variable")) == r.cget("value")]
        assert selected == ["1080p"]
        buttons["取消"]._command()

    _drive(tk_root, script, None)


# ── STGCN 模型路徑（file）與 YOLO bbox／關鍵點 kp 信心門檻（stepper）──────────────

YOLO_ENV = "CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD"
KP_ENV = "CAT_MONITORING_KP_CONF_THRES"
MODEL_ENV = "CAT_MONITORING_STGCN_MODEL"


def test_confidence_fields_are_steppers_from_0_1_to_1_0_by_0_01():
    by_env = {f["env"]: f for f in extra_env.FIELDS}
    expected = [f"{0.10 + i * 0.01:.2f}" for i in range(91)]  # 0.10、0.11、…、1.00
    for env in (YOLO_ENV, KP_ENV):
        f = by_env[env]
        assert f["type"] == "stepper" and f["step"] == 0.01
        assert extra_env.stepper_values(f) == expected
        assert expected[0] == "0.10" and expected[-1] == "1.00" and len(expected) == 91


def test_stepper_accepts_only_the_listed_steps(state_path):
    for ok, stored in (("0.1", "0.10"), ("0.5", "0.50"), ("0.65", "0.65"), ("0.62", "0.62"), ("1.0", "1.00"), ("0.85", "0.85")):
        extra_env.save({YOLO_ENV: ok})
        assert extra_env.collect_for_launch() == {YOLO_ENV: stored}, ok
    for bad in ("0", "0.0", "0.05", "0.099", "0.625", "1.01", "1.1", "1.5", "-0.1", "abc", "nan", "inf"):
        extra_env.save({YOLO_ENV: bad})
        assert extra_env.collect_for_launch() == {}, bad
    # 檔案被手動改成非法值：讀出來也當成不覆寫
    ui_state.update(**{"extra_env." + YOLO_ENV: "0.625"})
    assert extra_env.load()[YOLO_ENV] == ""


def test_stepper_normalizes_equivalent_spellings(state_path):
    extra_env.save({KP_ENV: " 0.5 "})
    assert extra_env.collect_for_launch() == {KP_ENV: "0.50"}
    extra_env.save({KP_ENV: "1"})
    assert extra_env.collect_for_launch() == {KP_ENV: "1.00"}
    assert extra_env.summary() == "1.00"


def test_bbox_and_kp_are_independent_settings(state_path):
    extra_env.save({YOLO_ENV: "0.3", KP_ENV: "0.8"})
    assert extra_env.collect_for_launch() == {YOLO_ENV: "0.30", KP_ENV: "0.80"}
    assert extra_env.summary() == "2 項"
    extra_env.save({YOLO_ENV: "", KP_ENV: "0.8"})
    assert extra_env.collect_for_launch() == {KP_ENV: "0.80"}


def test_generic_float_type_still_validates_range(state_path, monkeypatch):
    fields = list(extra_env.FIELDS) + [{"env": "MY_FLOAT", "label": "數值", "type": "float", "min": 0.0, "max": 2.0, "hint": ""}]
    monkeypatch.setattr(extra_env, "FIELDS", fields)
    extra_env.save({"MY_FLOAT": "1.25"})
    assert extra_env.collect_for_launch() == {"MY_FLOAT": "1.25"}
    for bad in ("2.5", "-1", "abc", "nan"):
        extra_env.save({"MY_FLOAT": bad})
        assert extra_env.collect_for_launch() == {}, bad


def test_file_field_summary_shows_file_name_only(state_path):
    extra_env.save({MODEL_ENV: "C:/ai_project/stgcn_models/run_124/124_best_model.pth"})
    assert extra_env.collect_for_launch()[MODEL_ENV].endswith("124_best_model.pth")
    assert extra_env.summary() == "124_best_mod"  # 只留檔名，且截到 12 字


def test_field_hints_make_bbox_vs_kp_explicit():
    """使用者要求：清楚註明是 bbox 還是 kp，並且環境變數名對應 config.py。"""
    by_env = {f["env"]: f for f in extra_env.FIELDS}
    bbox, kp = by_env[YOLO_ENV], by_env[KP_ENV]
    assert "bbox" in bbox["label"] and "bbox" in bbox["hint"] and "kp" in bbox["hint"]
    assert "kp" in kp["label"] and "kp" in kp["hint"] and "bbox" in kp["hint"]
    import config
    src = open(config.__file__, encoding="utf-8").read()
    for env in (YOLO_ENV, KP_ENV, MODEL_ENV):
        assert f'"{env}"' in src  # 名稱真的存在於 config.py，沒有打錯


def _drive_dialog_widgets(root, script):
    """開對話框後呼叫 script(dlg, entries, spinboxes, buttons)；例外收集後由測試本體重拋，對話框一律關閉。"""
    errors = []

    def driver():
        dlg = None
        try:
            dlg = [w for w in _walk(root) if isinstance(w, tk.Toplevel)][-1]
            widgets = list(_walk(dlg))
            steppers = [w for w in widgets if isinstance(w, extra_env._Stepper)]
            # 一般輸入框：排除 stepper 裡面那個唯讀數值框
            entries = [w for w in widgets if isinstance(w, tk.Entry) and not isinstance(w.master, extra_env._Stepper)]
            spinboxes = steppers
            buttons = {getattr(w, "_text", ""): w for w in widgets if hasattr(w, "_command")}
            script(dlg, entries, spinboxes, buttons)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            if dlg is not None and dlg.winfo_exists():
                dlg.destroy()

    root.after(150, driver)
    extra_env.open_dialog(root)
    if errors:
        raise errors[0]


def test_dialog_spinboxes_are_readonly_and_start_at_no_override(tk_root, state_path):
    def script(dlg, entries, spinboxes, buttons):
        assert len(spinboxes) == 2  # bbox、kp 各一個
        for sp in spinboxes:
            assert str(sp.entry.cget("state")) == "readonly"  # 不能打字
            assert sp.get() == extra_env.NO_OVERRIDE_TEXT  # 預設就是「不覆寫」
            assert sp.values == [extra_env.NO_OVERRIDE_TEXT] + [f"{0.10 + i * 0.01:.2f}" for i in range(91)]
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


def test_dialog_arrow_keys_step_the_value(tk_root, state_path):
    def script(dlg, entries, spinboxes, buttons):
        sp = spinboxes[0]
        sp.entry.focus_force()
        dlg.update()
        sp.entry.event_generate("<Up>")
        dlg.update()
        assert sp.get() == "0.10"         # 從「不覆寫」按 ↑ → 0.10
        for _ in range(4):
            sp.entry.event_generate("<Up>")
            dlg.update()
        assert sp.get() == "0.14"         # 每次 0.01：0.11、0.12、0.13、0.14
        sp.entry.event_generate("<Down>")
        dlg.update()
        assert sp.get() == "0.13"
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


def test_dialog_typing_is_blocked_on_spinbox(tk_root, state_path):
    def script(dlg, entries, spinboxes, buttons):
        sp = spinboxes[0]
        sp.entry.focus_force()
        dlg.update()
        sp.entry.event_generate("<KeyPress>", keysym="7")
        dlg.update()
        assert sp.get() == extra_env.NO_OVERRIDE_TEXT  # 打字沒有作用
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


def test_dialog_saves_stepped_confidences_and_model_file(tk_root, state_path, tmp_path):
    model = tmp_path / "x_best_model.pth"
    model.write_bytes(b"0")

    def script(dlg, entries, spinboxes, buttons):
        entries[0].delete(0, "end")
        entries[0].insert(0, str(model))
        for _ in range(3):
            spinboxes[0].up_btn.invoke()      # bbox：不覆寫 → 0.10 → 0.11 → 0.12
        for _ in range(8):
            spinboxes[1].up_btn.invoke()      # kp：→ 0.10 + 7 × 0.01 = 0.17
        buttons["✓ 儲存"]._command()

    _drive_dialog_widgets(tk_root, script)
    assert extra_env.collect_for_launch() == {YOLO_ENV: "0.12", KP_ENV: "0.17", MODEL_ENV: str(model)}


def test_dialog_reset_all_puts_spinboxes_back_to_no_override(tk_root, state_path):
    extra_env.save({YOLO_ENV: "0.6", KP_ENV: "0.2"})

    def script(dlg, entries, spinboxes, buttons):
        assert [sp.get() for sp in spinboxes] == ["0.60", "0.20"]  # 預先帶入已存的值
        buttons["↺ 全部不覆寫"]._command()
        assert [sp.get() for sp in spinboxes] == [extra_env.NO_OVERRIDE_TEXT] * 2
        buttons["✓ 儲存"]._command()

    _drive_dialog_widgets(tk_root, script)
    assert extra_env.collect_for_launch() == {}


def test_dialog_missing_model_file_asks_and_can_be_declined(tk_root, state_path, monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(extra_env.dialogs, "ask_yesno", lambda *a, **k: asked.append(a[2]) or False)

    def script(dlg, entries, spinboxes, buttons):
        entries[0].insert(0, str(tmp_path / "nope.pth"))
        buttons["✓ 儲存"]._command()

    _drive_dialog_widgets(tk_root, script)
    assert asked and extra_env.collect_for_launch() == {}


def test_dialog_generic_float_field_rejects_out_of_range(tk_root, state_path, monkeypatch):
    fields = list(extra_env.FIELDS) + [{"env": "MY_FLOAT", "label": "數值", "type": "float", "min": 0.0, "max": 2.0, "hint": ""}]
    monkeypatch.setattr(extra_env, "FIELDS", fields)
    warned = []
    monkeypatch.setattr(extra_env.dialogs, "show_warning", lambda *a, **k: warned.append(a[2]))

    def script(dlg, entries, spinboxes, buttons):
        entries[-1].insert(0, "9")   # 最後一個 Entry 是自訂的 float 欄位
        buttons["✓ 儲存"]._command()

    _drive_dialog_widgets(tk_root, script)
    assert warned and extra_env.collect_for_launch() == {}


def test_stepper_buttons_are_large_enough_to_click(tk_root, state_path):
    """回歸：原生 Spinbox 的上下箭頭只有十幾像素，太小不好點。現在 ▲／▼ 要夠大。"""
    def script(dlg, entries, spinboxes, buttons):
        dlg.update_idletasks()
        for sp in spinboxes:
            for btn in (sp.up_btn, sp.down_btn):
                assert btn.winfo_reqwidth() >= 40 and btn.winfo_reqheight() >= 32, (btn.winfo_reqwidth(), btn.winfo_reqheight())
                assert float(btn.cget("repeatinterval")) > 0  # 按住不放會連續調整
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


def test_stepper_stops_at_both_ends_and_wheel_works(tk_root, state_path):
    def script(dlg, entries, spinboxes, buttons):
        sp = spinboxes[0]
        sp.entry.focus_force()
        dlg.update()
        for _ in range(3):
            sp.down_btn.invoke()               # 已經是第一格（不覆寫）：往下不會再動
        assert sp.get() == extra_env.NO_OVERRIDE_TEXT
        for _ in range(100):
            sp.up_btn.invoke()                 # 一路往上：停在 1.00，不會循環回「不覆寫」
        assert sp.get() == "1.00"
        sp.entry.event_generate("<MouseWheel>", delta=-120)
        dlg.update()
        assert sp.get() == "0.99"              # 滾輪往下 = 減一格
        sp.entry.event_generate("<MouseWheel>", delta=120)
        dlg.update()
        assert sp.get() == "1.00"
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


def test_stepper_page_keys_jump_ten_steps(tk_root, state_path):
    def script(dlg, entries, spinboxes, buttons):
        sp = spinboxes[0]
        sp.entry.focus_force()
        dlg.update()
        sp.entry.event_generate("<Prior>")     # PageUp
        dlg.update()
        assert sp.get() == "0.19"              # 不覆寫 → 第 10 格 = 0.10 + 9 × 0.01
        sp.entry.event_generate("<Prior>")
        dlg.update()
        assert sp.get() == "0.29"
        sp.entry.event_generate("<Next>")      # PageDown
        dlg.update()
        assert sp.get() == "0.19"
        for _ in range(3):
            sp.entry.event_generate("<Next>")  # 往下到頭停在「不覆寫」
            dlg.update()
        assert sp.get() == extra_env.NO_OVERRIDE_TEXT
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


# ── 版面設計（卡片、狀態徽章、詳細說明）──────────────────────────────


def _cards(dlg):
    return [w for w in _walk(dlg) if isinstance(w, extra_env._Card)]


def test_dialog_has_one_card_per_field_with_env_chip_and_summary(tk_root, state_path):
    def script(dlg, entries, spinboxes, buttons):
        cards = _cards(dlg)
        assert len(cards) == len(extra_env.FIELDS)
        texts = {w.cget("text") for c in cards for w in _walk(c) if isinstance(w, tk.Label)}
        for f in extra_env.FIELDS:
            assert f["label"] in texts and f["env"] in texts and f["summary"] in texts  # 標題、環境變數名、一行摘要
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


def test_confidence_cards_sit_side_by_side(tk_root, state_path):
    def script(dlg, entries, spinboxes, buttons):
        by_env = {c.master.winfo_id(): c for c in _cards(dlg)}
        cards = _cards(dlg)
        bbox = next(c for c in cards if c.winfo_children() and any(isinstance(w, extra_env._Stepper) for w in _walk(c)))
        dlg.update_idletasks()
        steppers = [c for c in cards if any(isinstance(w, extra_env._Stepper) for w in _walk(c))]
        assert len(steppers) == 2
        a, b = steppers
        assert a.winfo_y() == b.winfo_y() and a.winfo_x() < b.winfo_x()  # 同一列、左右並排
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


def test_badge_and_footer_summary_follow_the_values(tk_root, state_path):
    def script(dlg, entries, spinboxes, buttons):
        cards = _cards(dlg)
        badges = lambda: [c.badge.cget("text") for c in cards]  # noqa: E731
        assert set(badges()) == {"不覆寫"}
        footer_text = lambda: next(  # noqa: E731
            w.cget("text") for w in _walk(dlg) if isinstance(w, tk.Label) and ("覆寫" in w.cget("text")) and ("項" in w.cget("text") or "目前沒有" in w.cget("text"))
        )
        assert "目前沒有覆寫" in footer_text()
        spinboxes[0].up_btn.invoke()
        dlg.update()
        assert badges().count("已覆寫") == 1 and "已覆寫 1 項" in footer_text()
        spinboxes[1].up_btn.invoke()
        dlg.update()
        assert badges().count("已覆寫") == 2 and "已覆寫 2 項" in footer_text()
        buttons["↺ 全部不覆寫"]._command()
        dlg.update()
        assert set(badges()) == {"不覆寫"}
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


def test_detail_text_is_collapsed_until_toggled(tk_root, state_path):
    def script(dlg, entries, spinboxes, buttons):
        card = _cards(dlg)[0]
        assert not card.detail.winfo_manager()           # 預設收起（詳細說明很長，會把視窗撐得很高）
        card.toggle.event_generate("<Button-1>")
        dlg.update()
        assert card.detail.winfo_manager() and card.detail.content() == extra_env.FIELDS[0]["hint"]
        card.toggle.event_generate("<Button-1>")
        dlg.update()
        assert not card.detail.winfo_manager()
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


def test_dialog_height_fits_a_1080p_screen(tk_root, state_path):
    """視窗不能高到放不進 1080p 螢幕（含標題列與工作列）。"""
    def script(dlg, entries, spinboxes, buttons):
        dlg.update_idletasks()
        assert dlg.winfo_height() <= 880, dlg.winfo_height()
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


# ── 詳細說明的滾動容器（文字太長時不被截斷）──────────────────────────


def _stepper_cards(dlg):
    return [c for c in _cards(dlg) if any(isinstance(w, extra_env._Stepper) for w in _walk(c))]


def _open_detail(dlg, card):
    card.toggle.event_generate("<Button-1>")
    dlg.update()
    dlg.update()


def test_long_detail_gets_a_scrollbar_and_keeps_all_text(tk_root, state_path):
    def script(dlg, entries, spinboxes, buttons):
        for card in _stepper_cards(dlg):  # bbox／kp 的說明最長
            _open_detail(dlg, card)
            d = card.detail
            assert d.content() == card.detail.text.get("1.0", "end-1c")
            assert d.text.cget("state") == "disabled"                     # 唯讀
            lines = d.text.count("1.0", "end-1c", "displaylines")[0]
            shown = int(d.text.cget("height"))
            assert shown <= extra_env._ScrollText.MAX_LINES
            if lines > extra_env._ScrollText.MAX_LINES:
                assert d._bar_shown and d.bar.winfo_ismapped()            # 超過就出現捲軸
            else:
                assert not d._bar_shown                                   # 沒超過就不多此一舉
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


def test_scrolling_reaches_the_end_of_the_text(tk_root, state_path):
    """最後一句話（cat_pose／自動標註工具那段）必須能捲得到，不是被裁掉。"""
    def script(dlg, entries, spinboxes, buttons):
        card = _stepper_cards(dlg)[0]
        _open_detail(dlg, card)
        d = card.detail
        assert d._bar_shown, "這段說明夠長，應該要有捲軸"
        assert d.text.yview()[0] == 0.0 and d.text.yview()[1] < 1.0     # 一開始只看到前面
        for _ in range(50):
            d.text.event_generate("<MouseWheel>", delta=-120)
            dlg.update()
        assert d.text.yview()[1] == 1.0                                 # 滾到底
        assert d.text.bbox("end-2c") is not None                        # 最後一個字元真的在可視範圍內
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


def test_short_detail_needs_no_scrollbar_and_shrinks_to_fit(tk_root, state_path):
    def script(dlg, entries, spinboxes, buttons):
        card = next(c for c in _cards(dlg) if c not in _stepper_cards(dlg))
        _open_detail(dlg, card)
        d = card.detail
        lines = d.text.count("1.0", "end-1c", "displaylines")[0]
        assert int(d.text.cget("height")) == min(lines, extra_env._ScrollText.MAX_LINES)
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


def test_dialog_grows_to_fit_and_shrinks_back_when_detail_toggled(tk_root, state_path):
    """展開後視窗要跟著變高（否則下面的按鈕列會被擠出視窗），收起後回到原高度。"""
    def script(dlg, entries, spinboxes, buttons):
        dlg.update()
        h0 = dlg.winfo_height()
        card = _stepper_cards(dlg)[0]
        _open_detail(dlg, card)
        h1 = dlg.winfo_height()
        assert h1 > h0
        footer_btn = buttons["✓ 儲存"]
        assert footer_btn.winfo_rooty() + footer_btn.winfo_height() <= dlg.winfo_rooty() + dlg.winfo_height()  # 儲存鈕仍在視窗內
        assert h1 <= dlg.winfo_screenheight()
        _open_detail(dlg, card)  # 再按一次收起
        assert h0 - 60 <= dlg.winfo_height() <= h0 + 2 and dlg.winfo_height() < h1  # 回到（或略小於）原高度：第一次開啟時的高度是估算值
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)


def test_both_details_open_still_fit_the_screen_and_show_buttons(tk_root, state_path):
    def script(dlg, entries, spinboxes, buttons):
        for card in _stepper_cards(dlg):
            _open_detail(dlg, card)
        assert dlg.winfo_height() <= dlg.winfo_screenheight()
        btn = buttons["✓ 儲存"]
        assert btn.winfo_ismapped()
        assert btn.winfo_rooty() + btn.winfo_height() <= dlg.winfo_rooty() + dlg.winfo_height()
        buttons["取消"]._command()

    _drive_dialog_widgets(tk_root, script)
