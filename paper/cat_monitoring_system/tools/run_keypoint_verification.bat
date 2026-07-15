@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PY=C:\Users\homec\anaconda3\envs\yolo_new\python.exe
set SCRIPT=%~dp01_check_keypoint_importance.py

echo ============================================================
echo [1/4] Ground-truth only (no model): scratch vs stop
echo ============================================================
"%PY%" -X utf8 "%SCRIPT%" --mode groundtruth --class_a scratch --class_b stop

echo.
echo ============================================================
echo [2/4] Ground-truth only (no model): lick vs stop
echo ============================================================
"%PY%" -X utf8 "%SCRIPT%" --mode groundtruth --class_a lick --class_b stop

if "%~1"=="" (
    echo.
    echo ============================================================
    echo [3/4][4/4] Model-conditioned run: skipped
    echo   To also run the model-conditioned comparison, pass a checkpoint path:
    echo   run_keypoint_verification.bat "C:\path\to\best_model.pth"
    echo ============================================================
    goto :done
)

echo.
echo ============================================================
echo [3/4] Model-conditioned (correct predictions only): scratch vs stop
echo ============================================================
"%PY%" -X utf8 "%SCRIPT%" --mode model --model_path "%~1" --class_a scratch --class_b stop

echo.
echo ============================================================
echo [4/4] Model-conditioned (correct predictions only): lick vs stop
echo ============================================================
"%PY%" -X utf8 "%SCRIPT%" --mode model --model_path "%~1" --class_a lick --class_b stop

:done
echo.
echo Done.
pause
