@echo off
cd /d "%~dp0\.."
echo Starting RocoClicker Visual Servoing PID Aim Controller GUI...
uv run roll/control/aim_gui.py
pause
