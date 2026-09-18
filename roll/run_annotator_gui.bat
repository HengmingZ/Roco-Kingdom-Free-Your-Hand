@echo off
cd /d "%~dp0\.."
echo Starting Single-Object YOLO Annotator GUI...
uv run roll/annotation/annotator_gui.py
pause
