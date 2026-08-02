"""
ImprovedBehaviorTracker Regression Test（Characterization Test 風格）

跟 `processors/tests` 的 FrameProcessor 測試不同：BehaviorTracker 完全不依賴
GPU/YOLO/ST-GCN，是純粹的狀態機（只讀寫 `self.*` 與 `config.py` 的常數），
因此可以把驅動它的「時間」也完全控制住，讓整個測試變成 100% 確定性
（不像 FrameProcessor 那樣需要浮點誤差容忍）。

做法：monkeypatch `trackers.behavior_tracker` 模組內的 `time`/`datetime`
名稱，換成一個可以手動推進的假時鐘（`_scenario_utils.FakeClock`），再依序
呼叫固定腳本（`_scenario_utils.SCRIPT`）的 `.update(behavior_id, activity_value)`
序列，最後用「手算應得結果」逐項斷言——不是像 FrameProcessor 那樣先跑一次
「相信目前輸出」再凍結快照，這裡因為邏輯簡單到可以人工驗證，所以直接寫成
明確的期望值，正確性更有保障。

腳本情境與手算結果見 `_scenario_utils.py` 的 `SCRIPT` 常數與其註解。

驗證涵蓋 6 項需求：
1. 行為切換（walk→lick→scratch→stop→NOT_VISIBLE→LOW_CONF→walk）
2. duration（behavior_time 累積，含「定期累積」+「結算補最後一段」兩種路徑）
3. count（behavior_count 每次事件結束才 +1）
4. transition（transition_matrix，含跨過 NOT_VISIBLE/LOW_CONF 空窗後的轉移）
5. alert（get_alerts()：搔抓時間/舔舐時間/低活動度三種各自觸發與不觸發的情境）
6. state save/load（含「同一天可還原」與「跨天不還原」兩種情境）
"""

import json
from datetime import datetime

import pytest

from config import LoggingConfig
from trackers import behavior_tracker as behavior_tracker_module
from trackers.behavior_tracker import ImprovedBehaviorTracker

from _scenario_utils import SNAPSHOT_PATH, FakeClock, full_state_snapshot, run_script


@pytest.fixture
def fake_clock():
    return FakeClock(datetime(2026, 1, 1, 8, 0, 0))


@pytest.fixture
def isolated_tracker(tmp_path, monkeypatch, fake_clock):
    """建立一個時間完全受控、不會碰到正式 tracker_state.json 的 tracker。"""
    monkeypatch.setattr(
        LoggingConfig, "TRACKER_STATE_PATH", str(tmp_path / "test_tracker_state.json")
    )
    monkeypatch.setattr(behavior_tracker_module.time, "time", fake_clock.time)
    monkeypatch.setattr(behavior_tracker_module, "datetime", fake_clock)
    tracker = ImprovedBehaviorTracker()
    return tracker


# ── 1+2+3+4：行為切換 / duration / count / transition ──────────────────────


def test_scripted_scenario_behavior_time_duration(isolated_tracker, fake_clock):
    """behavior_time 累積：定期累積 + 結算補最後一段，總和需等於整段事件的真實時長。"""
    run_script(isolated_tracker, fake_clock)
    assert isolated_tracker.behavior_time["walk"] == pytest.approx(5.0)
    assert isolated_tracker.behavior_time["lick"] == pytest.approx(15.0)
    assert isolated_tracker.behavior_time["scratch"] == pytest.approx(12.0)
    assert isolated_tracker.behavior_time["stop"] == pytest.approx(1.0)
    assert isolated_tracker.behavior_time["shake"] == pytest.approx(0.0)


def test_scripted_scenario_behavior_count(isolated_tracker, fake_clock):
    """behavior_count：每個行為在腳本中各自完整結算一次事件，count 應各為 1。"""
    run_script(isolated_tracker, fake_clock)
    assert isolated_tracker.behavior_count["walk"] == 1
    assert isolated_tracker.behavior_count["lick"] == 1
    assert isolated_tracker.behavior_count["scratch"] == 1
    assert isolated_tracker.behavior_count["stop"] == 1
    assert isolated_tracker.behavior_count["shake"] == 0


def test_scripted_scenario_transition_matrix(isolated_tracker, fake_clock):
    """transition_matrix：包含跨過 NOT_VISIBLE/LOW_CONF 空窗後仍正確記錄的轉移（stop->walk）。"""
    run_script(isolated_tracker, fake_clock)
    assert isolated_tracker.transition_matrix == {
        "walk->lick": 1,
        "lick->scratch": 1,
        "scratch->stop": 1,
        "stop->walk": 1,
    }


def test_scripted_scenario_not_detected_and_low_conf_time(isolated_tracker, fake_clock):
    """not_detected_time / low_conf_time / low_conf_count 各自獨立累積，互不干擾。"""
    run_script(isolated_tracker, fake_clock)
    assert isolated_tracker.not_detected_time == pytest.approx(1.0)
    assert isolated_tracker.low_conf_time == pytest.approx(3.0)
    assert isolated_tracker.low_conf_count == 1  # 連續兩次 LOW_CONF 只算同一個事件


def test_scripted_scenario_current_behavior_after_script(isolated_tracker, fake_clock):
    """腳本結束時最後一個行為（walk）尚未結算，應仍是 current_behavior。"""
    run_script(isolated_tracker, fake_clock)
    assert isolated_tracker.current_behavior == "walk"
    assert isolated_tracker.current_gcn_id == 0


# ── 5：alert ────────────────────────────────────────────────────────────────


def test_scripted_scenario_alerts(isolated_tracker, fake_clock):
    """三種各自觸發／不觸發的警報情境：
    - scratch_time=12.0 > SCRATCH_ALERT_TIME_SECONDS(10.0) → 觸發「搔抓時間異常」
    - lick_time=15.0 > LICK_ALERT_TIME_SECONDS(10.0) → 觸發「舔舐時間較長」
    - walk_time=5.0 < LOW_ACTIVITY_TIME_THRESHOLD_SECONDS(20.0) 且 total_time>0 → 觸發「活動度過低」
    - shake_count=0 遠低於 SHAKE_ALERT_COUNT_THRESHOLD(10) → 不觸發甩頭警報
    - stop_time=1.0 遠低於 STOP_ALERT_TIME_SECONDS(300.0) → 不觸發靜止警報
    """
    run_script(isolated_tracker, fake_clock)
    alerts = isolated_tracker.get_alerts()
    titles = {a["title"] for a in alerts}
    assert "搔抓時間異常" in titles
    assert "舔舐時間較長" in titles
    assert "活動度過低" in titles
    assert "甩頭動作頻繁" not in titles
    assert "長時間靜止不動" not in titles
    assert len(alerts) == 3


# ── 6：state save/load ───────────────────────────────────────────────────────


def test_save_state_then_load_restores_same_day(tmp_path, monkeypatch, fake_clock):
    """同一天內：save_state() 寫出的內容，新建的 tracker 實例應該完整還原。"""
    state_path = tmp_path / "test_tracker_state.json"
    monkeypatch.setattr(LoggingConfig, "TRACKER_STATE_PATH", str(state_path))
    monkeypatch.setattr(behavior_tracker_module.time, "time", fake_clock.time)
    monkeypatch.setattr(behavior_tracker_module, "datetime", fake_clock)

    original = ImprovedBehaviorTracker()
    run_script(original, fake_clock)
    original.save_state()

    assert state_path.exists(), "save_state() 應該要寫出檔案"
    with open(state_path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["behavior_time"]["walk"] == pytest.approx(5.0)
    assert saved["behavior_count"]["scratch"] == 1
    assert saved["transition_matrix"] == {
        "walk->lick": 1,
        "lick->scratch": 1,
        "scratch->stop": 1,
        "stop->walk": 1,
    }

    # 用同一天（fake_clock 沒有推進日期）建立新的 tracker，__init__ 會自動 load_state()
    restored = ImprovedBehaviorTracker()
    assert restored.behavior_time == original.behavior_time
    assert restored.behavior_count == original.behavior_count
    assert restored.transition_matrix == original.transition_matrix
    assert restored.low_conf_time == pytest.approx(original.low_conf_time)
    assert restored.not_detected_time == pytest.approx(original.not_detected_time)


def test_load_state_ignored_when_saved_on_a_different_day(tmp_path, monkeypatch, fake_clock):
    """跨天：存檔日期跟「現在」不同天時，新 tracker 不應該還原舊資料（避免跨日資料污染）。"""
    state_path = tmp_path / "test_tracker_state.json"
    monkeypatch.setattr(LoggingConfig, "TRACKER_STATE_PATH", str(state_path))
    monkeypatch.setattr(behavior_tracker_module.time, "time", fake_clock.time)
    monkeypatch.setattr(behavior_tracker_module, "datetime", fake_clock)

    original = ImprovedBehaviorTracker()
    run_script(original, fake_clock)
    original.save_state()

    # 把假時鐘推到隔天，再建立新 tracker
    fake_clock.advance(24 * 3600)
    next_day_tracker = ImprovedBehaviorTracker()
    assert next_day_tracker.behavior_time["walk"] == 0.0
    assert next_day_tracker.behavior_count["scratch"] == 0
    assert next_day_tracker.transition_matrix == {}


# ── 額外：把完整腳本情境的最終狀態凍結成快照（多一層回歸保護） ──────────────


def test_scripted_scenario_matches_frozen_snapshot(isolated_tracker, fake_clock):
    """把整段腳本跑完的完整狀態，跟已凍結的基準快照比對（regression 保護網）。

    刻意不在測試裡自動產生快照（即使目前的輸出已經由上面幾個明確斷言驗證過
    是對的）：跟 `processors/tests` 的原則一致——快照的產生永遠是獨立、手動、
    需要人確認的步驟，測試本身只負責「比對」，不負責「建立基準」，否則快照
    檔案一旦不小心被刪除，測試會靜默地重新產生並永遠通過，喪失偵測回歸的
    能力。基準快照由 `generate_behavior_tracker_snapshot.py` 產生。
    """
    if not SNAPSHOT_PATH.exists():
        pytest.fail(
            f"找不到基準快照 {SNAPSHOT_PATH}。\n"
            "請先手動執行一次：\n"
            "    python trackers/tests/generate_behavior_tracker_snapshot.py\n"
            "確認輸出合理後，把產生的快照檔一併提交，測試才有比較基準。"
        )

    run_script(isolated_tracker, fake_clock)
    current = full_state_snapshot(isolated_tracker)

    with open(SNAPSHOT_PATH, encoding="utf-8") as f:
        baseline = json.load(f)
    assert current == baseline
