"""
Flask 應用主入口
"""
from flask import Flask
from server.routes import register_routes

def create_app():
    app = Flask(__name__)
    register_routes(app)
    return app
