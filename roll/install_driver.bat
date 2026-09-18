@echo off
setlocal
cd /d  %~dp0

echo ========================================================
echo   Interception Driver Installer
echo ========================================================
echo.

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [Prompt] Requesting Administrator Privileges...
    powershell -NoProfile -Command Start-Process -FilePath %~f0 -Verb RunAs
    exit /b
)

set INSTALLER=%~dp0driver_installer\install-interception.exe
if not exist %INSTALLER% (
    set INSTALLER=%~dp0third\Interception\command line installer\install-interception.exe
)

if not exist %INSTALLER% (
    color 0C
    echo [Error] Installer not found: %INSTALLER%
    pause
    exit /b 1
)

color 0A
echo [1/2] Installing Interception kernel driver...
%INSTALLER% /install

echo.
echo ========================================================
echo [2/2] Done!
echo [NOTICE] You MUST REBOOT your computer for it to take effect!
echo After reboot, run: run_clicker.bat
echo ========================================================
echo.
pause