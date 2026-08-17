@echo off
setlocal
chcp 65001 >nul
title Create Desktop Shortcut - Cat Monitoring Settings Window

rem The target batch file is always resolved relative to this file's own folder
rem (%~dp0), never hardcoded as an absolute string. That means this script works
rem no matter which machine/clone it runs on -- if the whole paper/ folder moves,
rem just re-run this file and it rebuilds the shortcut pointing at the new spot.
rem Note: a Windows .lnk file itself always stores an absolute target path -- that
rem is a hard limit of the shortcut format, not something this script can avoid.
rem What is "relative" here is where THIS SCRIPT gets that absolute path FROM
rem (%~dp0) rather than a path baked into the script as a literal string.
set "TARGET=%~dp0run_settings_window.bat"
set "WORKDIR=%~dp0"
set "SHORTCUT_NAME=Cat Monitoring Settings.lnk"

if not exist "%TARGET%" (
    echo [ERROR] run_settings_window.bat not found:
    echo   %TARGET%
    echo Make sure run_settings_window.bat sits in the same folder as this file.
    pause
    exit /b 1
)

for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "(New-Object -ComObject WScript.Shell).SpecialFolders('Desktop')"`) do set "DESKTOP=%%D"

if "%DESKTOP%"=="" (
    echo [ERROR] Could not resolve the Desktop folder.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ws = New-Object -ComObject WScript.Shell;" ^
    "$lnk = $ws.CreateShortcut('%DESKTOP%\%SHORTCUT_NAME%');" ^
    "$lnk.TargetPath = '%TARGET%';" ^
    "$lnk.WorkingDirectory = '%WORKDIR%';" ^
    "$lnk.IconLocation = 'C:\Windows\System32\imageres.dll,109';" ^
    "$lnk.Description = 'Cat Monitoring System - Settings Window (launches settings_window.py via Anaconda yolo_new env)';" ^
    "$lnk.Save()"

if errorlevel 1 (
    echo [ERROR] Failed to create the desktop shortcut.
    pause
    exit /b 1
)

echo.
echo Desktop shortcut created: %SHORTCUT_NAME%
echo Target: %TARGET%
echo.
pause
