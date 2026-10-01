"""
FrameProcessor 本機錄影改用影片時間（2026-09-30，docs/錄影推論改用影片時間-待辦.md）

不載入 YOLO／ST-GCN：用 FrameProcessor.__new__ 建一個空殼，只放 _setup_media_clock()
等方法會用到的屬性，驗證「哪些來源會切到影片時間」與影片時鐘的換算。
"""

from datetime import datetime

import cv2
import pytest

from config import RunModeConfig
from processors.frame_processor import FrameProcessor

START = datetime(2026, 9, 30, 18, 0, 0)


class _FakeCap:
    def __init__(self, pos_ms=0.0):
        self.pos_ms = pos_ms

    def get(self, prop):
        return self.pos_ms if prop == cv2.CAP_PROP_POS_MSEC else 0.0


class _FakeTracker:
    def __init__(self):
        self.clock = None
        self.finished = False

    def enable_media_clock(self, fn):
        self.clock = fn

    def finish_media_day(self):
        self.finished = True
        return START.date()


def _shell(grabber=None):
    fp = FrameProcessor.__new__(FrameProcessor)
    fp._grabber = grabber
    fp.tracker = _FakeTracker()
    fp.cap = _FakeCap()
    fp.frame_idx = 0
    fp._plugin_source_fps = 30.0
    fp._media_clock_start = None
    fp.nodered = None
    return fp


@pytest.fixture
def video_file(tmp_path):
    p = tmp_path / "recording.mp4"
    p.write_bytes(b"x")
    return str(p)


def test_local_file_with_start_time_uses_video_clock(video_file, monkeypatch):
    monkeypatch.setattr(RunModeConfig, "VIDEO_START_DATETIME", START)
    fp = _shell()
    fp._setup_media_clock(video_file)
    assert fp.tracker.clock is not None

    fp.cap.pos_ms = 90 * 60 * 1000.0  # 影片播到 1 小時 30 分
    assert fp.tracker.clock() == pytest.approx(START.timestamp() + 5400)
    assert fp._clock_now() == datetime(2026, 9, 30, 19, 30, 0)


def test_falls_back_to_frame_index_when_decoder_has_no_position(video_file, monkeypatch):
    monkeypatch.setattr(RunModeConfig, "VIDEO_START_DATETIME", START)
    fp = _shell()
    fp._setup_media_clock(video_file)
    fp.cap.pos_ms = 0.0
    fp.frame_idx = 300  # 30fps → 10 秒
    assert fp.tracker.clock() == pytest.approx(START.timestamp() + 10)


def test_local_file_without_start_time_keeps_wall_clock_and_warns(video_file, monkeypatch, capsys):
    monkeypatch.setattr(RunModeConfig, "VIDEO_START_DATETIME", None)
    fp = _shell()
    fp._setup_media_clock(video_file)
    assert fp.tracker.clock is None
    assert "沒有設定「影片開始錄影時間」" in capsys.readouterr().out


@pytest.mark.parametrize("source", ["http://192.168.0.10:81/stream", 0, "not_a_file.mp4"])
def test_live_sources_never_use_video_clock(source, monkeypatch):
    monkeypatch.setattr(RunModeConfig, "VIDEO_START_DATETIME", START)
    fp = _shell(grabber=object() if isinstance(source, str) and source.startswith("http") else None)
    fp._setup_media_clock(source)
    assert fp.tracker.clock is None


def test_finish_media_day_only_in_video_clock_mode(video_file, monkeypatch):
    fp = _shell()
    fp.finish_media_day()  # 電腦時鐘模式：什麼都不做
    assert fp.tracker.finished is False

    monkeypatch.setattr(RunModeConfig, "VIDEO_START_DATETIME", START)
    fp._setup_media_clock(video_file)
    fp.finish_media_day()
    assert fp.tracker.finished is True


def test_finish_media_day_marks_last_push_as_media_finished(video_file, monkeypatch):
    """影片播完的最後一筆推送帶 media_finished，儀表板收到就停在真實值、不再補算顯示。"""
    sent = []

    class _FakeNodeRed:
        def send_data(self, payload):
            sent.append(payload)

    fp = _shell()
    fp.nodered = _FakeNodeRed()
    fp._display_behavior_id = 0
    fp._display_confidence = 0.9
    fp._build_nodered_payload = lambda bid, conf: {"today_stats": {"total_uptime": 12.3}}
    monkeypatch.setattr(RunModeConfig, "VIDEO_START_DATETIME", START)
    fp._setup_media_clock(video_file)
    fp.finish_media_day()
    assert sent == [{"today_stats": {"total_uptime": 12.3}, "media_finished": True}]
    assert fp.tracker.finished is True


def test_config_parser_and_settings_validator_accept_same_formats():
    from config import _parse_video_start
    import settings_manager as sm

    for ok in ("2026-09-30 18:00", "2026-09-30 18:00:05", "2026-09-30T18:00"):
        assert _parse_video_start(ok) is not None
        assert sm._validate_video_start(ok, "x") is None
    assert _parse_video_start("") is None and sm._validate_video_start("", "x") is None
    for bad in ("18:00", "2026/09/30 18:00", "2026-09-30"):
        assert _parse_video_start(bad) is None
        assert sm._validate_video_start(bad, "x") is not None


def test_uses_media_clock_reflects_setup():
    fp = _shell()
    assert fp.uses_media_clock is False
    fp._media_clock_start = START.timestamp()
    assert fp.uses_media_clock is True


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0.0, "0 小時 0 分 0 秒"),
        (3.1, "0 小時 0 分 3 秒"),
        (3725.6, "1 小時 2 分 6 秒"),
        (21645.0, "6 小時 0 分 45 秒"),
    ],
)
def test_run_summary_line_formats_total_uptime(seconds, expected):
    fp = _shell()
    fp.tracker.run_seconds = seconds
    assert fp.run_summary_line() == f"📊 此次運行總監測時長：{expected}"
