@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
echo [*] 正在检查并从 GitHub Release 拉取最佳模型权重 (best.pt)...
uv run roll/download_weights.py %*
pause
