"""dashboard/views.py 的路由測試。

`dashboard.views` 本身不 import cv2/torch/ultralytics（跟 server/routes.py
不一樣），所以這裡直接組一個最小的 Flask app 掛上 blueprint 測試，不需要
`pytest.importorskip("cv2", ...)`，任何環境都能跑。「旗標關閉時
flask_app.create_app() 真的不會註冊這個 blueprint」這件事因為要經過
server/routes.py（它才需要 cv2），另外寫在 test_flag_integration.py。
"""

import pytest
from flask import Flask

from dashboard import cache
from dashboard.views import bp


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(bp)
    app.testing = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_cache():
    cache.clear()
    yield
    cache.clear()


def test_latest_endpoint_returns_not_yet_computed_before_any_data(client):
    resp = client.get("/api/deviation/latest")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "not_yet_computed"}


def test_latest_endpoint_returns_cached_payload_once_available(client):
    cache.set_latest(
        {
            "status": "ok",
            "baseline": {"days_count": 10, "required_days": 7, "confidence": "Medium"},
            "deviation": {"metrics": {}},
            "fusion": {"score": 12.3, "level": "Normal"},
        }
    )
    resp = client.get("/api/deviation/latest")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["fusion"]["level"] == "Normal"
    assert "cached_at" in body


def test_dashboard_page_returns_html(client):
    resp = client.get("/dashboard/baseline")
    assert resp.status_code == 200
    assert resp.mimetype == "text/html"
    body = resp.get_data(as_text=True)
    assert "個體化基線儀表板" in body
    # 輪詢間隔要被實際帶入頁面，不是留著沒替換的樣板佔位字串
    assert "__POLL_MS__" not in body


def test_dashboard_page_does_not_claim_period_coverage():
    """迴歸測試：這個頁面刻意不畫「時段涵蓋率」，因為 Python 版 baseline.py
    沒有這個資料（那是 Node-RED 舊引擎才有算的 periods 概念，見
    analytics/README.md「還沒做的事」）。不小心複製 P4 面板整段內容進來、
    畫出假資料，是這個頁面最容易犯的錯，所以特別鎖住。"""
    from dashboard.views import _render_page

    html = _render_page()
    assert "時段涵蓋率" not in html
