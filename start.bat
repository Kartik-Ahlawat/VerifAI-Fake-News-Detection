@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo Fake News Detector - Windows Demo Launcher
echo ============================================
echo.

if not exist ".venv\" (
  echo [1/5] Creating virtual environment...
  py -3 -m venv .venv
  if errorlevel 1 python -m venv .venv
  if errorlevel 1 (
    echo Could not create .venv using 'py -3' or 'python'.
    echo Install Python 3.10+ from https://python.org and re-run.
    pause
    exit /b 1
  )
)

echo [2/5] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo Failed to activate .venv.
  pause
  exit /b 1
)

echo [3/5] Installing project package...
python -m pip install --disable-pip-version-check -e .
if errorlevel 1 (
  echo Package install failed.
  pause
  exit /b 1
)

if not exist "models\fake_news_model.json" (
  echo [4/5] Model not found. Training from sample data...
  python -m fake_news_detector.train --data data/sample_news.csv --model-out models/fake_news_model.json
  if errorlevel 1 (
    echo Model training failed.
    pause
    exit /b 1
  )
) else (
  echo [4/5] Model found: models\fake_news_model.json
)

echo [5/5] Starting demo server...
echo.
echo Open this URL in browser:
echo   http://127.0.0.1:8765/
echo.
echo Press Ctrl+C to stop the server.
echo.
start "" "http://127.0.0.1:8765/"
python -m fake_news_detector.api --model models/fake_news_model.json --host 127.0.0.1 --port 8765 --web-dir web

pause
