@echo off
cd /d "%~dp0\.."
echo Starting Real-Time Visual Servoing PID Aim Controller...
uv run roll/control/run_aim_loop.py
pause
