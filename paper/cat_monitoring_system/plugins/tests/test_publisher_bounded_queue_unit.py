"""第一階段（M1 傳輸收尾）：NodeRedPublisher / ZoneHttpPublisher 有界
latest-only 佇列 + raise_for_status + 指標（說明書「Python Publisher」）。

Node-RED 離線時記憶體不得持續成長：舊的未送出 payload 直接被覆蓋並計入
dropped，而不是無上限地堆進 ThreadPoolExecutor。
"""

import threading
import time

import pytest

pytest.importorskip("cv2", reason="plugins/lick_stage 需要 cv2")

import plugins.lick_stage.publisher as publisher_module
from plugins.lick_stage.publisher import NodeRedPublisher


class _SlowOKClient:
    """post() 會 block 一小段時間後回傳 200，用來製造「worker 忙碌」的窗口。"""

    def __init__(self, delay=0.05):
        self.delay = delay
        self.calls = 0
        self._lock = threading.Lock()

    def post(self, *a, **kw):
        with self._lock:
            self.calls += 1
        time.sleep(self.delay)
        return self

    def raise_for_status(self):
        return None

    def close(self):
        return None


def test_latest_only_queue_drops_intermediate_payloads(monkeypatch):
    client = _SlowOKClient(delay=0.05)
    monkeypatch.setattr(publisher_module, "_HAS_REQUESTS", True)

    pub = NodeRedPublisher(url="http://x/y", timeout=0.5)
    monkeypatch.setattr(pub, "_session", client)

    # 快速塞 20 筆：worker 一次只處理一筆，其餘應被後來的覆蓋
    for i in range(20):
        pub.publish({"seq": i})
    time.sleep(0.4)
    pub.close()

    s = pub.stats()
    # 送出的遠少於 20；被丟棄的 + 送出的 應約等於投遞數
    assert s["publish_ok"] >= 1
    assert s["dropped_payload"] >= 10
    assert client.calls <= 6  # 有界：不會把 20 筆全送


def test_stats_report_success_and_last_ts(monkeypatch):
    client = _SlowOKClient(delay=0.0)
    monkeypatch.setattr(publisher_module, "_HAS_REQUESTS", True)
    pub = NodeRedPublisher(url="http://x/y", timeout=0.5)
    monkeypatch.setattr(pub, "_session", client)

    pub.publish({"a": 1})
    time.sleep(0.1)
    pub.close()

    s = pub.stats()
    assert s["publish_ok"] >= 1
    assert s["last_success_ts"] > 0.0
    assert s["publish_fail"] == 0


def test_raise_for_status_failure_is_counted(monkeypatch):
    class _HTTP500:
        def post(self, *a, **kw):
            return self

        def raise_for_status(self):
            raise RuntimeError("HTTP 500")

        def close(self):
            pass

    monkeypatch.setattr(publisher_module, "_HAS_REQUESTS", True)
    pub = NodeRedPublisher(url="http://x/y", timeout=0.5)
    monkeypatch.setattr(pub, "_session", _HTTP500())
    pub._post({"a": 1})  # 直接呼叫，繞過 worker
    pub.close()
    assert pub.stats()["publish_fail"] == 1
