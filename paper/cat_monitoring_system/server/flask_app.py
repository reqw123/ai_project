"""
Flask 應用工廠模組：create_app() 建立並回傳 Flask 實例
"""

from flask import Flask

from server.routes import register_routes


def create_app():
    """建立 Flask 應用實例並註冊所有路由。"""
    app = Flask(__name__)
    register_routes(app)
    return app
