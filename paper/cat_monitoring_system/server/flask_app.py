"""
Flask 應用工廠模組：create_app() 建立並回傳 Flask 實例
"""

from flask import Flask

from server.routes import register_routes  # noqa: E402 -- 匯入時會順便把 project root 加進 sys.path
from config import BaselineDashboardConfig


def create_app():
    """建立 Flask 應用實例並註冊所有路由。"""
    app = Flask(__name__)
    register_routes(app)
    if BaselineDashboardConfig.ENABLED:
        # 旗標關閉時完全不匯入 dashboard 套件（不佔用任何資源），
        # 見 config.BaselineDashboardConfig、dashboard/__init__.py。
        from dashboard.views import bp as _baseline_dashboard_bp

        app.register_blueprint(_baseline_dashboard_bp)
    return app
