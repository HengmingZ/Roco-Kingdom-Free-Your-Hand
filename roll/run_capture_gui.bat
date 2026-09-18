@echo off
cd /d "%~dp0\.."
echo Starting Screen Capture GUI...
uv run roll/capture/capture_gui.py
pause
