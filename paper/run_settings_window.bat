@echo off
setlocal
chcp 65001 >nul
title Cat Monitoring System - Settings Window

rem Fixed to the Anaconda "yolo_new" venv's python.exe. Not using "conda activate"
rem here because a plain double-clicked batch file has not run conda's shell hook,
rem so activate would usually fail or not be found.
set "PY=C:\Users\homec\anaconda3\envs\yolo_new\python.exe"

rem %~dp0 = the folder this batch file itself lives in (trailing backslash included).
rem settings_window.py is always resolved relative to the batch file's own location,
rem never hardcoded as an absolute path -- if the whole paper/ folder gets moved or
rem renamed, this still works as long as both files stay in the same folder together.
set "SCRIPT_DIR=%~dp0"
set "SCRIPT=%SCRIPT_DIR%settings_window.py"
set "PYTHONIOENCODING=utf-8"

if not exist "%PY%" (
    echo [ERROR] Anaconda "yolo_new" python.exe not found:
    echo   %PY%
    echo Make sure Anaconda is installed and the yolo_new env exists.
    echo If your env lives elsewhere, edit the PY variable in this file.
    pause
    exit /b 1
)

if not exist "%SCRIPT%" (
    echo [ERROR] settings_window.py not found:
    echo   %SCRIPT%
    echo This batch file must sit in the same folder as settings_window.py.
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"
"%PY%" -X utf8 "%SCRIPT%"

if errorlevel 1 (
    echo.
    echo [settings_window.py exited with a non-zero code -- scroll up for details]
    pause
)
