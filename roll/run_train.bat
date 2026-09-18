@echo off
cd /d "%~dp0\.."
echo Starting YOLOv8m Model Training...
uv run roll/model/train.py --epochs 50 --batch 8
pause
