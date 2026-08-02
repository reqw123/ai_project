@echo off
cd /d "%~dp0"
python fetch_real_history.py
node render_baseline_ui_preview.js
start baseline_ui_preview.html
pause
