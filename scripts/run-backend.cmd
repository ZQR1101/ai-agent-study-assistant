@echo off
title AI Study Assistant - Backend
cd /d "%~dp0.."

"%AI_STUDY_PYTHON%" -m uvicorn backend.server:app --reload --host 127.0.0.1 --port 8000
if errorlevel 1 (
  echo.
  echo Backend stopped with an error.
  pause
)
