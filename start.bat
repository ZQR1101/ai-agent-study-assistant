@echo off
setlocal
cd /d "%~dp0"

rem Double-click entry point. PowerShell does the environment checks and starts both services.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" %*
if errorlevel 1 (
  echo.
  echo Startup failed. See the message above for details.
  pause
)
