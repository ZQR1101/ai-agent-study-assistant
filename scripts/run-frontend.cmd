@echo off
title AI Study Assistant - Frontend
cd /d "%~dp0.."

"%AI_STUDY_NODE%" "%~dp0..\node_modules\vite\bin\vite.js" --host 127.0.0.1 --port 5500
if errorlevel 1 (
  echo.
  echo Frontend stopped with an error.
  pause
)
