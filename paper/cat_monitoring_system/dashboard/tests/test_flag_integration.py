"""驗證 config.BaselineDashboardConfig.ENABLED 真的能決定
flask_app.create_app() 要不要把 dashboard blueprint 掛上去。

`server.flask_app` 匯入 `server.routes`，後者檔案層級 import cv2/torch，
所以這個測試檔案需要 cv2 才能載入——跟
server/tests/test_routes_api_regression.py 是同一個既有限制，此環境沒裝
cv2 時會被跳過，在裝好完整依賴的機器上才會真的執行。
"""

import pytest

pytest.importorskip(
    "cv2",
    reason="server/flask_app.py -> server/routes.py 檔案層級 import cv2/torch 才能載入，此環境未安裝",
)

import config as _config_module
import server.flask_app as flask_app_module


def _reload_with_flag(monkeypatch, enabled: bool):
    """在乾淨狀態下重建 app：切換旗標、清掉 dashboard 相關的 import 快取，
    確保每次都是真的重新走一次 create_app() 的條件式 import 邏輯。"""
    monkeypatch.setattr(_config_module.BaselineDashboardConfig, "ENABLED", enabled)
    import sys

    for mod_name in list(sys.modules):
        if mod_name == "dashboard" or mod_name.startswith("dashboard."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    return flask_app_module.create_app()


def test_blueprint_not_registered_when_disabled(monkeypatch):
    app = _reload_with_flag(monkeypatch, enabled=False)
    with app.test_client() as client:
        resp = client.get("/dashboard/baseline")
        assert resp.status_code == 404
        resp2 = client.get("/api/deviation/latest")
        assert resp2.status_code == 404
    import sys

    assert "dashboard.views" not in sys.modules, (
        "旗標關閉時 dashboard.views 不應該被匯入到記憶體裡"
    )


def test_blueprint_registered_when_enabled(monkeypatch):
    app = _reload_with_flag(monkeypatch, enabled=True)
    with app.test_client() as client:
        resp = client.get("/dashboard/baseline")
        assert resp.status_code == 200
        resp2 = client.get("/api/deviation/latest")
        assert resp2.status_code == 200
