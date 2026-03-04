@echo off
title RAG Assistant - Backend
echo.
echo  RAG Documentation Assistant v2 - Backend
echo  =========================================
echo.
if not exist ".env" (
    echo [ERROR] .env file not found!
    echo Copy .env.example to .env and add your API key.
    pause
    exit /b 1
)
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)
echo Starting backend on http://localhost:8000
echo API docs: http://localhost:8000/docs
echo.
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
pause
