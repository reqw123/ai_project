"""
Unit Test：settings_gui/iot_services.py（設定視窗「📡 IoT 子系統」分頁的啟動／停止／狀態）

全部用**假服務**（暫存資料夾裡的小模組，只會印心跳、睡覺）：不會動到使用者正在跑的
python -m iot／python -m iot.voice，也不會連 MQTT broker。狀態檔、紀錄檔也都換到暫存資料夾。
"""

import socket
import subprocess
import sys
import time

import pytest

from settings_gui import iot_services as svc

_FAKE = "import time, sys\nprint('fake service up', sys.argv, flush=True)\nwhile True:\n    time.sleep(0.2)\n"


@pytest.fixture
def fake(tmp_path, monkeypatch):
    mods = tmp_path / "mods"
    mods.mkdir()
    (mods / "fake_voice_svc.py").write_text(_FAKE, encoding="utf-8")
    (mods / "fake_hub_svc.py").write_text(_FAKE, encoding="utf-8")
    env = {"PYTHONPATH": str(mods)}
    services = (
        svc.Service(key="va", title="🎙 假語音", desc="", module="fake_voice_svc", env=env),
        svc.Service(key="ka", title="🌡️ 假甲", desc="", module="fake_hub_svc", kind="ka", env={**env, svc._KINDS_ENV: "ka"}),
        svc.Service(key="kb", title="📍 假乙", desc="", module="fake_hub_svc", kind="kb", env={**env, svc._KINDS_ENV: "kb"}),
    )
    monkeypatch.setattr(svc, "SERVICES", services)
    monkeypatch.setattr(svc, "_BY_KEY", {s.key: s for s in services})
    monkeypatch.setattr(svc, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(svc, "STATE_FILE", tmp_path / "state.json")
    started = []
    yield {"mods": mods, "started": started}
    for s in services:   # 收尾：只停這個測試的假服務
        st = svc.scan()[s.key]
        if st.running:
            svc.stop(s.key, pid=st.pid)
    for p in started:
        if p.poll() is None:
            p.kill()


def _wait(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def test_start_scan_tail_stop(fake):
    assert not svc.scan()["va"].running
    ok, msg = svc.start("va")
    assert ok, msg
    st = svc.scan()["va"]
    assert st.running and st.managed and st.pid > 0
    assert _wait(lambda: "fake service up" in svc.tail("va"))
    assert "由設定視窗啟動" in svc.tail("va")
    ok, _ = svc.start("va")
    assert not ok                                    # 已經在跑：不重複啟動
    ok, msg = svc.stop("va", pid=st.pid)
    assert ok, msg
    assert not svc.scan()["va"].running
    assert "由設定視窗停止" in svc.tail("va")
    assert svc._load_state() == {}                    # 狀態檔清掉


def test_each_sensor_is_its_own_process(fake):
    assert svc.start("ka")[0] and svc.start("kb")[0]
    a, b = svc.scan()["ka"], svc.scan()["kb"]
    assert a.running and b.running and a.pid != b.pid
    assert a.shared_kinds == ("ka",) and b.shared_kinds == ("kb",)   # 各自只處理自己那一種
    svc.stop("ka", pid=a.pid)
    assert not svc.scan()["ka"].running and svc.scan()["kb"].running  # 停一個不影響另一個


def test_external_all_kinds_hub_is_recognized(fake):
    # 在 cmd 手動執行「python -m 模組」、沒設 KINDS＝全部感測器一個行程
    env = {k: v for k, v in __import__("os").environ.items() if k != svc._KINDS_ENV}
    env["PYTHONPATH"] = str(fake["mods"])
    p = subprocess.Popen([sys.executable, "-m", "fake_hub_svc"], env=env, stdout=subprocess.DEVNULL)
    fake["started"].append(p)
    assert _wait(lambda: svc.scan()["ka"].running)
    a, b = svc.scan()["ka"], svc.scan()["kb"]
    assert a.pid == b.pid == p.pid and not a.managed and set(a.shared_kinds) == {"ka", "kb"}
    assert "外部啟動" in a.text() and "同一個行程" in a.text()
    assert not svc.start("ka")[0]                     # 已經被那個行程處理：不重複啟動（避免兩個行程都處理同一種）
    ok, msg = svc.stop("ka", pid=p.pid + 99999)
    assert not ok and "不是要停的" in msg             # PID 對不上：不停
    assert svc.stop("kb", pid=p.pid)[0]
    assert p.wait(5) is not None
    assert not svc.scan()["ka"].running and not svc.scan()["kb"].running


def test_tail_without_log(fake):
    assert "還沒有紀錄檔" in svc.tail("va")   # 紀錄檔在暫存資料夾（不看真的 paper/logs/iot）


def test_real_catalog():
    keys = [s.key for s in svc.SERVICES]
    assert keys == ["env", "bodytemp", "motion", "weight", "voice"]   # 環境＋體表溫度同一列、語音最後
    for s in svc.SERVICES:
        assert (s.module == "iot" and s.env == {svc._KINDS_ENV: s.kind}) or (s.module == "iot.voice" and not s.kind)
    assert (svc.CAT_DIR / "iot" / "__main__.py").exists() and (svc.CAT_DIR / "iot" / "voice" / "__main__.py").exists()


def test_broker_reachable():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert svc.broker_reachable("127.0.0.1", port)[0]
    finally:
        srv.close()
    assert not svc.broker_reachable("127.0.0.1", port, timeout=0.3)[0]


def _conn(status, laddr, raddr, pid):
    from types import SimpleNamespace as N
    addr = lambda a: N(ip=a[0], port=a[1]) if a else ()   # noqa: E731
    return N(status=status, laddr=addr(laddr), raddr=addr(raddr), pid=pid)


def test_links_from_connection_table():
    """服務實際連到的 broker 位址；連進本機 broker 的裝置用主機名稱對應到卡片（排除本機 IP）。"""
    local = {"192.168.0.10", "192.168.0.171", "127.0.0.1"}
    names = {"192.168.0.148": "cat-petbox-carrier", "192.168.0.20": "esp32-DA98B8",
             "192.168.0.30": "cat-voice", "192.168.0.40": "my-phone"}
    conns = [
        _conn("LISTEN", ("0.0.0.0", 1883), None, 1),
        _conn("ESTABLISHED", ("192.168.0.171", 51886), ("192.168.0.171", 1883), 100),   # 服務 → broker
        _conn("ESTABLISHED", ("192.168.0.171", 1883), ("192.168.0.171", 51886), 1),     # broker 那一端（本機）
        _conn("ESTABLISHED", ("192.168.0.171", 1883), ("192.168.0.148", 53604), 1),     # 外出包
        _conn("ESTABLISHED", ("192.168.0.171", 1883), ("192.168.0.20", 40000), 1),      # 還沒燒新韌體
        _conn("ESTABLISHED", ("192.168.0.171", 1880), ("192.168.0.30", 40001), 2),      # 語音終端 → Node-RED
        _conn("ESTABLISHED", ("192.168.0.171", 1880), ("192.168.0.40", 40002), 2),      # 手機開 Node-RED 頁面：不算
        _conn("ESTABLISHED", ("192.168.0.10", 51835), ("192.168.0.10", 1883), 999),     # 別的程式（Node-RED）
    ]
    links = svc._links_from(conns, {100}, 1883, local, resolve=lambda ip: names.get(ip, ""))
    assert links.service_targets == ("192.168.0.171:1883",)
    assert [d.ip for d in links.devices] == ["192.168.0.20", "192.168.0.30", "192.168.0.148"]
    assert links.ips_for("env") == links.ips_for("bodytemp") == ["192.168.0.148"]   # 外出包兩張卡片都有
    assert links.ips_for("voice") == ["192.168.0.30"] and links.ips_for("motion") == []
    assert [d.ip for d in links.unknown()] == ["192.168.0.20"]


def test_device_keys_from_hostname():
    assert svc.device_keys("cat-weight-food-bowl") == ("weight",)
    assert svc.device_keys("CAT-Petbox-carrier") == ("env", "bodytemp")
    assert svc.device_keys("cat-room-living-room") == ("env", "motion")   # esp32_room：兩張卡片都顯示
    assert svc.device_keys("esp32-DA98B8") == () and svc.device_keys("") == () and svc.device_keys("cat-xyz") == ()


def test_links_from_broker_not_local():
    conns = [_conn("ESTABLISHED", ("192.168.0.10", 5000), ("10.0.0.5", 1883), 100)]
    links = svc._links_from(conns, {100}, 1883, {"192.168.0.10"}, resolve=lambda ip: "")
    assert links.service_targets == ("10.0.0.5:1883",) and links.devices is None


def test_clear_dns_cache(monkeypatch):
    """主機名稱快取 5 分鐘；「⟳ 重新整理」清快取後立刻重新查（ESP32 燒錄改名後不用等）。"""
    names = iter(["esp32-DA98B8", "cat-petbox-carrier"])
    monkeypatch.setattr(svc.socket, "gethostbyaddr", lambda ip: (next(names) + ".home", [], [ip]))
    svc.clear_dns_cache()
    assert svc._reverse_dns("10.9.9.9") == "esp32-DA98B8"
    assert svc._reverse_dns("10.9.9.9") == "esp32-DA98B8"          # 快取：不會再查
    svc.clear_dns_cache()
    assert svc._reverse_dns("10.9.9.9") == "cat-petbox-carrier"
    svc.clear_dns_cache()


def test_connection_report_levels(monkeypatch):
    """⟳ 重新整理的連線檢查：服務沒連上、裝置沒連線 → 紅（bad）；未啟動、未辨識 → 黃（warn）；網卡提示。"""
    services = (
        svc.Service(key="env", title="🌡 環境", desc="", module="iot", kind="env"),
        svc.Service(key="motion", title="👣 移動", desc="", module="iot", kind="motion"),
        svc.Service(key="weight", title="⚖ 秤重", desc="", module="iot", kind="weight"),
    )
    monkeypatch.setattr(svc, "SERVICES", services)
    statuses = {"env": svc.Status(True, 11), "motion": svc.Status(True, 22), "weight": svc.Status(False)}
    links = svc.MqttLinks(("192.168.0.171:1883",),
                          (svc.Device("192.168.0.148", "cat-petbox-carrier", ("env", "bodytemp")),
                           svc.Device("192.168.0.20", "esp32-DA98B8", ())),
                          connected_pids=(11,), node_red=False)
    lines = svc.connection_report(statuses, (True, "192.168.0.171:1883"), links, {"192.168.0.171": "Wi-Fi"})
    text = {t: lv for lv, t in lines}
    assert any("Wi-Fi" in t and lv == "ok" for t, lv in text.items())
    assert any("Node-RED" in t and lv == "bad" for t, lv in text.items())
    assert any(t.startswith("● 🌡 環境") and lv == "ok" for t, lv in text.items())
    assert any(t.startswith("✕ 👣 移動") and "沒連上 broker" in t and lv == "bad" for t, lv in text.items())
    assert any(t.startswith("✕ ⚖ 秤重") and "服務未啟動" in t for t, lv in text.items())   # 未啟動＋裝置沒連線 → 取較嚴重的
    assert any("未辨識裝置 192.168.0.20" in t and lv == "warn" for t, lv in text.items())
    assert lines[-1][0] == "bad" and "錯誤" in lines[-1][1]
    bad = svc.connection_report(statuses, (False, "10.0.0.1:1883"), None, {})
    assert bad[1][0] == "bad" and "連不上" in bad[1][1]


def test_tail_since_offset_for_clear(fake):
    """紀錄框「清空」：只顯示清空之後的新紀錄，檔案不動；檔案被輪替變小就當作沒清過。"""
    path = svc.get("va").log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("舊的一行\n舊的第二行\n", encoding="utf-8")
    mark = svc.log_size("va")
    assert "已清空" in svc.tail("va", since=mark)
    with open(path, "a", encoding="utf-8") as f:
        f.write("新的一行\n")
    assert svc.tail("va", since=mark) == "新的一行"
    assert "舊的一行" in svc.tail("va")                  # 不清的話全部都在（檔案沒被刪）
    path.write_text("輪替後\n", encoding="utf-8")        # 比清空位置還小 → 當作沒清過
    assert svc.tail("va", since=mark) == "輪替後"


def test_merge_logs_by_time():
    """整合檢視：依時間合併；沒有時間的行（Traceback）跟著上一行；設定視窗的 ===== 行也有時間。"""
    a = ("2026-09-27 10:00:01,000 [INFO] a1\n"
         "2026-09-27 10:00:03,500 [WARNING] a2\n"
         "Traceback (most recent call last):\n"
         "  File x\n")
    b = ("===== 2026-09-27 10:00:00 由設定視窗啟動 =====\n"
         "2026-09-27 10:00:02,000 [INFO] b1\n"
         "2026-09-27 10:00:03,500 [INFO] b2\n")
    rows = svc.merge_logs({"a": a, "b": b})
    assert [line.split(" ")[-1] if "=====" not in line else "start" for _k, line in rows] == \
        ["start", "a1", "b1", "a2", "last):", "x", "b2"]
    assert [k for k, _ in rows] == ["b", "a", "b", "a", "a", "a", "b"]


def test_merged_tail_skips_missing_and_cleared(fake):
    for key, line in (("va", "2026-09-27 10:00:02,000 [INFO] 語音"), ("ka", "2026-09-27 10:00:01,000 [INFO] 甲")):
        p = svc.get(key).log_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(line + "\n", encoding="utf-8")
    assert [k for k, _ in svc.merged_tail()] == ["ka", "va"]          # kb 沒有紀錄檔：跳過
    assert [k for k, _ in svc.merged_tail(since={"ka": svc.log_size("ka")})] == ["va"]   # ka 清空了
