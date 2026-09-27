"""已連上 broker 但沒收到感測資料：連上後首次檢查、多久沒資料算異常（timeout）、異常持續中重印（repeat）。"""

from iot.ingest.no_data_watchdog import NoDataWatchdog


def _wd(kinds=("bodytemp",), repeat=10):
    return NoDataWatchdog(kinds, "cat/iot", "192.168.0.171:1883", first_check_sec=5, timeout_sec=60,
                          repeat_sec=repeat)


def _run(wd, start, end):
    """每秒 tick 一次，回傳 {秒: 訊息清單}（只留有訊息的）。"""
    return {t: m for t in range(start, end) if (m := wd.tick(t))}


def test_no_check_before_connected():
    wd = _wd()
    assert wd.tick(1000) == []


def test_first_check_then_repeat_while_silent():
    wd = _wd()
    wd.on_connected(0)
    out = _run(wd, 0, 31)
    assert sorted(out) == [5, 15, 25]                        # 5 秒首次檢查，之後每 10 秒重印
    assert "bodytemp 已經 5 秒沒有資料" in out[5][0] and "192.168.0.171:1883" in out[5][0]
    assert "（持續中）" not in out[5][0] and out[15][0].startswith("（持續中）")
    assert "檢查感測器接線" in out[5][0]                         # 提示兩種可能，不是只叫人查 WiFi


def test_recovery_then_timeout():
    wd = _wd()
    wd.on_connected(0)
    wd.tick(5)
    wd.on_message("cat/iot/bodytemp/petbox", now=6)
    assert _run(wd, 7, 66) == {}                             # 收到資料後：60 秒內都正常
    out = _run(wd, 66, 80)
    assert sorted(out) == [66, 76] and "已經 60 秒沒有資料" in out[66][0]   # 6 秒收到 → 66 秒算異常


def test_recovery_message_once():
    wd = _wd()
    wd.on_connected(0)
    wd.tick(5)
    assert "bodytemp" in wd.on_message("cat/iot/bodytemp/petbox", now=6)
    assert wd.on_message("cat/iot/bodytemp/petbox", now=7) is None


def test_data_before_first_check_no_warning():
    wd = _wd(("env", "bodytemp"))
    wd.on_connected(0)
    wd.on_message("cat/iot/env/petbox", now=2)
    out = wd.tick(5)
    assert len(out) == 1 and "bodytemp" in out[0]


def test_repeat_zero_prints_once():
    wd = _wd(repeat=0)
    wd.on_connected(0)
    assert sorted(_run(wd, 0, 200)) == [5]


def test_other_topics_ignored():
    wd = _wd()
    wd.on_connected(0)
    assert wd.on_message("cat/iot/derived/bodytemp", now=1) is None
    assert wd.on_message("other/bodytemp/x", now=1) is None
    assert wd.tick(5) != []


def test_disconnect_pauses_and_reconnect_restarts():
    wd = _wd()
    wd.on_connected(0)
    wd.on_disconnected()
    assert _run(wd, 0, 100) == {}
    wd.on_connected(200)
    assert wd.tick(204) == []
    assert "已經 5 秒沒有資料" in wd.tick(205)[0]
