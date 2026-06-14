"""
Flask 應用工廠模組：create_app() 建立並回傳 Flask 實例
"""
from flask import Flask
from server.routes import register_routes

def create_app():
    app = Flask(__name__)
    register_routes(app)
    return app
