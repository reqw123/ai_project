"""
API Regression Test：server/routes.py 的 Flask 端點

最終階段（API Regression Test）。驗證的是「API 行為本身」是否維持一致
（HTTP Status、JSON Schema、Response Keys、錯誤處理），刻意不驗證任何
演算法/推論結果的正確性（ST-GCN 分類、行為追蹤邏輯、基線/偏差統計數字
等已在前面的 Characterization Test / Unit Test 階段涵蓋）。

策略：
- 用 Flask 測試用戶端（test_client），不啟動真正的 HTTP server。
- `server.routes` 模組層級的 `frame_processor`/`frame_streamer` 是共用
  全域變數，`_ensure_processor_started()` 在沒有預先設定好的情況下會
  嘗試建立真正的 GPU pipeline（FrameProcessor + SharedFrameStreamer +
  真實攝影機/影片來源）。為了讓測試完全不依賴 GPU/YOLO/真實攝影機，也
  不受目前系統時間是否落在 RunModeConfig 排程視窗內影響，這裡把
  `_ensure_processor_started` monkeypatch 成 no-op，並把
  `frame_processor`/`frame_streamer` 直接換成假物件（Fake），只實作
  路由邏輯實際會用到的屬性/方法。
- `pytest.importorskip("cv2", ...)`：因為 `server/routes.py` 檔案層級
  import cv2/flask/torch 才能載入（見 test_routes_pure_functions_unit.py
  開頭說明）。
"""

from datetime import date, datetime, timedelta

import numpy as np
import pytest

pytest.importorskip(
    "cv2",
    reason="server/routes.py 檔案層級 import cv2/flask/torch 才能載入，此環境未安裝",
)

from flask import Flask

import server.routes as routes_module
from config import ModelPaths as _ModelPaths


class FakeFrameProcessor:
    """只實作 routes.py 實際會用到的屬性/方法，不啟動任何真正的推論。"""

    def __init__(self):
        self.overlay = True
        self.show_skeleton = True
        self.show_label = True
        self.show_bbox = True
        self.history_records = []

    def get_behavior_history_records(self, limit):
        return self.history_records[-limit:]


class FakeFrameStreamer:
    def __init__(self):
        self.paused = False
        self.jpeg_bytes = None
        self.clip_frames = []
        self.acquire_calls = 0
        self.release_calls = 0

    def acquire_client(self):
        self.acquire_calls += 1

    def release_client(self):
        self.release_calls += 1

    def get_jpeg(self):
        return self.jpeg_bytes

    def get_clip_frames(self):
        return self.clip_frames


@pytest.fixture
def fake_processor():
    return FakeFrameProcessor()


@pytest.fixture
def fake_streamer():
    return FakeFrameStreamer()


@pytest.fixture
def client(monkeypatch, fake_processor, fake_streamer):
    """建立 Flask 測試用戶端；`_ensure_processor_started()` 換成 no-op，
    避免任何路徑意外觸發真正的 GPU pipeline 建置。"""
    monkeypatch.setattr(routes_module, "_ensure_processor_started", lambda: None)
    monkeypatch.setattr(routes_module, "frame_processor", fake_processor)
    monkeypatch.setattr(routes_module, "frame_streamer", fake_streamer)

    app = Flask(__name__)
    routes_module.register_routes(app)
    app.testing = True
    return app.test_client()


# ============================================================================
# GET /
# ============================================================================


class TestIndexRoute:
    def test_returns_200_with_html_mimetype(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.mimetype == "text/html"

    def test_body_contains_stream_link(self, client):
        resp = client.get("/")
        assert b"/stream" in resp.data


# ============================================================================
# GET /snapshot
# ============================================================================


class TestSnapshotRoute:
    def test_no_streamer_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(routes_module, "frame_streamer", None)
        resp = client.get("/snapshot")
        assert resp.status_code == 503
        assert resp.mimetype == "image/jpeg"

    def test_streamer_with_jpeg_ready_returns_200_with_jpeg_bytes(
        self, client, fake_streamer
    ):
        fake_streamer.jpeg_bytes = b"\xff\xd8fake-jpeg-bytes\xff\xd9"
        resp = client.get("/snapshot")
        assert resp.status_code == 200
        assert resp.mimetype == "image/jpeg"
        assert resp.data == fake_streamer.jpeg_bytes

    def test_acquires_and_releases_client_slot(self, client, fake_streamer):
        fake_streamer.jpeg_bytes = b"data"
        client.get("/snapshot")
        assert fake_streamer.acquire_calls == 1
        assert fake_streamer.release_calls == 1


# ============================================================================
# GET /stream （只測 503 no-streamer 分支；正常分支是無窮 MJPEG generator，
# 不適合、也不需要在 Regression Test 裡完整消費）
# ============================================================================


class TestStreamRoute:
    def test_no_streamer_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(routes_module, "frame_streamer", None)
        resp = client.get("/stream")
        assert resp.status_code == 503
        assert resp.mimetype == "text/plain"


# ============================================================================
# GET /video_clip
# ============================================================================


class TestVideoClipRoute:
    def test_no_frames_available_returns_503_with_error_key(self, client):
        resp = client.get("/video_clip")
        assert resp.status_code == 503
        assert "error" in resp.get_json()

    def test_valid_frames_returns_200_with_expected_schema(
        self, client, fake_streamer, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(_ModelPaths, "OUTPUT_DIR", str(tmp_path))
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        fake_streamer.clip_frames = [frame, frame, frame]

        resp = client.get("/video_clip")
        assert resp.status_code == 200
        body = resp.get_json()
        for key in ("path", "frames", "duration", "ts", "thumbnail"):
            assert key in body
        assert body["frames"] == 3
        assert body["thumbnail"].startswith("data:image/jpeg;base64,")


# ============================================================================
# GET /api/behavior_history
# ============================================================================


def _mk_record(behavior_id, timestamp, duration, activity=0, behavior_fallback="unknown"):
    return {
        "gcn_behavior_id": behavior_id,
        "behavior": behavior_fallback,
        "timestamp": timestamp,
        "duration": duration,
        "activity": activity,
    }


class TestBehaviorHistoryRoute:
    def test_default_limit_returns_all_records_newest_first(self, client, fake_processor):
        t0 = datetime(2026, 7, 1, 8, 0, 0)
        fake_processor.history_records = [
            _mk_record(0, t0, 10.0),
            _mk_record(1, t0 + timedelta(seconds=10), 5.0),
        ]
        resp = client.get("/api/behavior_history")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["count"] == 2
        # 路由用 reversed(records)，最新的排在最前面
        assert body["segments"][0]["behavior_id"] == 1
        assert body["segments"][1]["behavior_id"] == 0

    def test_known_behavior_id_uses_chinese_text_map(self, client, fake_processor):
        fake_processor.history_records = [_mk_record(0, datetime(2026, 7, 1), 1.0)]
        resp = client.get("/api/behavior_history")
        body = resp.get_json()
        assert body["segments"][0]["behavior"] == "走動"

    def test_unknown_behavior_id_falls_back_to_provided_behavior_field(
        self, client, fake_processor
    ):
        fake_processor.history_records = [
            _mk_record(99, datetime(2026, 7, 1), 1.0, behavior_fallback="custom_label")
        ]
        resp = client.get("/api/behavior_history")
        body = resp.get_json()
        assert body["segments"][0]["behavior"] == "custom_label"

    def test_segment_schema_has_expected_keys(self, client, fake_processor):
        fake_processor.history_records = [
            _mk_record(0, datetime(2026, 7, 1, 12, 30, 45), 3.14, activity=5)
        ]
        resp = client.get("/api/behavior_history")
        seg = resp.get_json()["segments"][0]
        assert seg["behavior_id"] == 0
        assert seg["timestamp"] == "2026-07-01 12:30:45"
        assert seg["duration_sec"] == pytest.approx(3.1)
        assert seg["activity"] == 5

    def test_limit_query_param_restricts_count(self, client, fake_processor):
        t0 = datetime(2026, 7, 1)
        fake_processor.history_records = [
            _mk_record(0, t0 + timedelta(seconds=i), 1.0) for i in range(5)
        ]
        resp = client.get("/api/behavior_history?limit=2")
        body = resp.get_json()
        assert body["count"] == 2

    def test_limit_below_one_is_clamped_to_one(self, client, fake_processor):
        t0 = datetime(2026, 7, 1)
        fake_processor.history_records = [
            _mk_record(0, t0 + timedelta(seconds=i), 1.0) for i in range(3)
        ]
        resp = client.get("/api/behavior_history?limit=0")
        assert resp.get_json()["count"] == 1

    def test_limit_above_1000_does_not_error(self, client, fake_processor):
        """夾鉗上限為 1000，這裡只驗證超大 limit 不會出錯，不驗證夾鉗值本身
        （資料本身只有 1 筆，回傳數量自然受資料量限制，非夾鉗邏輯限制）。"""
        fake_processor.history_records = [_mk_record(0, datetime(2026, 7, 1), 1.0)]
        resp = client.get("/api/behavior_history?limit=999999")
        assert resp.status_code == 200
        assert resp.get_json()["count"] == 1

    def test_non_numeric_limit_falls_back_to_default_200(self, client, fake_processor):
        fake_processor.history_records = [
            _mk_record(0, datetime(2026, 7, 1) + timedelta(seconds=i), 1.0)
            for i in range(3)
        ]
        resp = client.get("/api/behavior_history?limit=not_a_number")
        assert resp.status_code == 200
        assert resp.get_json()["count"] == 3

    def test_empty_history_returns_empty_segments(self, client, fake_processor):
        resp = client.get("/api/behavior_history")
        assert resp.get_json() == {"count": 0, "segments": []}


# ============================================================================
# POST /api/deviation
# ============================================================================


def _mk_history(n=7):
    return [
        {
            "date": (date(2026, 1, 1) + timedelta(days=i)).isoformat(),
            "monitoring_seconds": 7200,
            "walk_time": 100.0,
            "walk_count": 5,
            "lick_time": 50.0,
            "lick_count": 2,
        }
        for i in range(n)
    ]


class TestDeviationRoute:
    def test_valid_request_returns_200_with_full_schema(self, client):
        resp = client.post(
            "/api/deviation",
            json={
                "daily_history": _mk_history(7),
                "today": {"walk_time": 100.0, "lick_count": 2},
            },
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ok"
        for key in ("baseline", "deviation", "fusion"):
            assert key in body
            assert isinstance(body[key], dict)

    def test_insufficient_history_returns_200_with_insufficient_data_status(self, client):
        resp = client.post(
            "/api/deviation",
            json={"daily_history": _mk_history(2), "today": {"walk_time": 100.0}},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "insufficient_data"
        assert "current_days" in body
        assert "required_days" in body

    def test_missing_daily_history_returns_400(self, client):
        resp = client.post("/api/deviation", json={"today": {}})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_missing_today_returns_400(self, client):
        resp = client.post("/api/deviation", json={"daily_history": _mk_history(7)})
        assert resp.status_code == 400

    def test_non_list_daily_history_returns_400(self, client):
        resp = client.post(
            "/api/deviation", json={"daily_history": "not-a-list", "today": {}}
        )
        assert resp.status_code == 400

    def test_malformed_date_in_history_returns_400(self, client):
        history = _mk_history(7)
        history[0]["date"] = "2026/01/01"
        resp = client.post(
            "/api/deviation",
            json={"daily_history": history, "today": {"walk_time": 1.0}},
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_non_integer_min_baseline_days_returns_400(self, client):
        resp = client.post(
            "/api/deviation",
            json={
                "daily_history": _mk_history(7),
                "today": {"walk_time": 1.0},
                "min_baseline_days": "not_a_number",
            },
        )
        assert resp.status_code == 400

    def test_non_numeric_class_c_score_returns_400(self, client):
        resp = client.post(
            "/api/deviation",
            json={
                "daily_history": _mk_history(7),
                "today": {"walk_time": 1.0},
                "class_c_score": "not_a_number",
            },
        )
        assert resp.status_code == 400

    def test_empty_body_returns_400(self, client):
        resp = client.post("/api/deviation", data="", content_type="application/json")
        assert resp.status_code == 400

    def test_custom_min_baseline_days_is_honored(self, client):
        resp = client.post(
            "/api/deviation",
            json={
                "daily_history": _mk_history(3),
                "today": {"walk_time": 100.0},
                "min_baseline_days": 3,
            },
        )
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"


# ============================================================================
# /api/overlay
# ============================================================================


class TestOverlayRoute:
    def test_options_request_returns_200_with_cors_headers(self, client):
        resp = client.options("/api/overlay")
        assert resp.status_code == 200
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    def test_get_returns_current_flag_state(self, client, fake_processor):
        fake_processor.show_skeleton = False
        resp = client.get("/api/overlay")
        assert resp.status_code == 200
        assert resp.get_json() == {
            "master": True,
            "skeleton": False,
            "label": True,
            "bbox": True,
        }
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    def test_post_missing_key_returns_400(self, client):
        resp = client.post("/api/overlay", json={"value": True})
        assert resp.status_code == 400

    def test_post_valid_key_and_value_updates_state(self, client, fake_processor):
        resp = client.post("/api/overlay", json={"key": "skeleton", "value": False})
        assert resp.status_code == 200
        assert fake_processor.show_skeleton is False
        body = resp.get_json()
        assert body["ok"] is True
        assert body["state"]["skeleton"] is False

    def test_post_master_true_resets_all_sub_flags(self, client, fake_processor):
        fake_processor.show_skeleton = False
        fake_processor.show_label = False
        fake_processor.show_bbox = False
        resp = client.post("/api/overlay", json={"key": "master", "value": True})
        assert resp.status_code == 200
        assert fake_processor.show_skeleton is True
        assert fake_processor.show_label is True
        assert fake_processor.show_bbox is True

    def test_post_master_false_clears_all_sub_flags(self, client, fake_processor):
        resp = client.post("/api/overlay", json={"key": "master", "value": False})
        assert resp.status_code == 200
        assert fake_processor.show_skeleton is False
        assert fake_processor.show_label is False
        assert fake_processor.show_bbox is False

    def test_post_unknown_key_returns_400(self, client):
        resp = client.post("/api/overlay", json={"key": "nonexistent", "value": True})
        assert resp.status_code == 400

    def test_post_non_bool_value_without_toggle_action_returns_400(self, client):
        resp = client.post("/api/overlay", json={"key": "skeleton", "value": "yes"})
        assert resp.status_code == 400

    def test_post_toggle_action_flips_current_value(self, client, fake_processor):
        fake_processor.show_label = True
        resp = client.post("/api/overlay", json={"key": "label", "action": "toggle"})
        assert resp.status_code == 200
        assert fake_processor.show_label is False
        assert resp.get_json()["value"] is False

    def test_post_toggle_with_unknown_key_returns_400(self, client):
        resp = client.post(
            "/api/overlay", json={"key": "nonexistent", "action": "toggle"}
        )
        assert resp.status_code == 400
