"""
本機錄影（影片時鐘模式）播完後自動結束行程（2026-09-30）

錄影推論是為了收集那一天的基線資料：影片播完＝資料收齊，行程直接結束，確保統計停在
影片結尾。這裡用假的 FrameProcessor 驅動 SharedFrameStreamer._update_frame()，不載入
模型；另外驗證 NodeRedClient.close(wait=...) 不會把最後一筆推送丟掉。
"""

import threading
import time

import pytest

from communication import nodered_client
from server import streaming


class _FakeProcessor:
    """read_raw_frame() 一律回傳播畢，記錄收尾流程被呼叫的順序。"""

    def __init__(self, uses_media_clock):
        self.uses_media_clock = uses_media_clock
        self.source_fps = 30.0
        self.calls = []

    def is_stream_source(self):
        return False

    def read_raw_frame(self):
        return False, None

    def finish_plugin_sessions(self):
        self.calls.append("finish_plugin_sessions")

    def finish_media_day(self):
        self.calls.append("finish_media_day")

    def cleanup(self, flush_timeout=0.0):
        self.calls.append(("cleanup", flush_timeout))

    def run_summary_line(self):
        self.calls.append("run_summary_line")
        return "📊 此次運行總監測時長：0 小時 0 分 3 秒"


def _run_update_loop(processor, monkeypatch):
    exited = threading.Event()
    monkeypatch.setattr(streaming, "_exit_process", exited.set)
    streamer = streaming.SharedFrameStreamer.__new__(streaming.SharedFrameStreamer)
    streamer.frame_processor = processor
    streamer.running = True
    streamer.paused = False
    streamer.finished = False
    worker = threading.Thread(target=streamer._update_frame, daemon=True)
    worker.start()
    exited.wait(timeout=1.0)
    streamer.running = False  # 非影片時鐘模式不會自己結束，測完手動停下
    worker.join(timeout=2.0)
    return streamer, exited.is_set()


def test_media_clock_video_end_writes_day_then_exits(monkeypatch):
    processor = _FakeProcessor(uses_media_clock=True)
    streamer, exited = _run_update_loop(processor, monkeypatch)

    assert exited
    assert streamer.finished
    # 先結算外掛、寫進多天歷史，再清理（等 Node-RED 最後一筆送出），最後一行印總監測時長後結束
    assert processor.calls == [
        "finish_plugin_sessions",
        "finish_media_day",
        ("cleanup", streaming._MEDIA_EXIT_FLUSH_SECONDS),
        "run_summary_line",
    ]


def test_video_without_media_clock_stays_idle_after_end(monkeypatch):
    processor = _FakeProcessor(uses_media_clock=False)
    streamer, exited = _run_update_loop(processor, monkeypatch)

    assert not exited
    assert streamer.finished
    assert processor.calls == ["finish_plugin_sessions", "finish_media_day"]


def test_exit_still_happens_when_cleanup_fails(monkeypatch):
    processor = _FakeProcessor(uses_media_clock=True)

    def _boom(flush_timeout=0.0):
        raise RuntimeError("cleanup failed")

    processor.cleanup = _boom
    _, exited = _run_update_loop(processor, monkeypatch)
    assert exited


@pytest.fixture
def slow_post(monkeypatch):
    """模擬 Node-RED 每次 POST 要 0.2 秒，記錄實際送出的 payload。"""
    sent = []

    def _post(url, json=None, timeout=None):
        time.sleep(0.2)
        sent.append(json)

    monkeypatch.setattr(nodered_client.requests, "post", _post)
    return sent


def test_close_with_wait_delivers_last_queued_payload(slow_post):
    worker = nodered_client._EndpointWorker("http://x/y", "t")
    worker.put({"n": 1})
    time.sleep(0.05)  # worker 正在送第 1 筆
    worker.put({"n": 2})  # 最後一筆排在佇列裡
    worker.close(wait=2.0)
    assert slow_post == [{"n": 1}, {"n": 2}]
    assert not worker._thread.is_alive()


def test_close_without_wait_keeps_fast_drop_behavior(slow_post):
    worker = nodered_client._EndpointWorker("http://x/y", "t")
    worker.put({"n": 1})
    time.sleep(0.05)
    worker.put({"n": 2})
    worker.close()  # 預設不等：停止訊號取代排隊中的資料（Ctrl+C 快速結束）
    worker._thread.join(timeout=2.0)
    assert slow_post == [{"n": 1}]
