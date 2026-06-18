@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo ================================================
echo   Fake News Detector  ^|  AI-Powered Demo
echo ================================================
echo.

REM ── Load GEMINI_API_KEY from .env ─────────────────
if exist ".env" (
  for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
    if /i "%%a"=="GEMINI_API_KEY" set "GEMINI_API_KEY=%%b"
  )
)
if defined GEMINI_API_KEY (
  echo  [OK] Gemini API key loaded.
) else (
  echo  [!] No GEMINI_API_KEY found in .env - running in local-only mode.
)
echo.

REM ── Virtual environment ───────────────────────────
if not exist ".venv\" (
  echo [1/4] Setting up Python environment...
  py -3 -m venv .venv
  if errorlevel 1 python -m venv .venv
  if errorlevel 1 (
    echo ERROR: Could not create .venv. Install Python 3.10+ and re-run.
    pause
    exit /b 1
  )
) else (
  echo [1/4] Python environment ready.
)

REM ── Activate + install ────────────────────────────
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo ERROR: Failed to activate virtual environment.
  pause
  exit /b 1
)

echo [2/4] Installing package (first run only)...
python -m pip install --disable-pip-version-check -q -e .
if errorlevel 1 (
  echo ERROR: Package install failed.
  pause
  exit /b 1
)

REM ── Train model if not present ────────────────────
if not exist "models\fake_news_model.json" (
  echo [3/4] Training model on demo dataset...
  python -m fake_news_detector.train --data data/expanded_demo_news.csv --model-out models/fake_news_model.json
  if errorlevel 1 (
    echo ERROR: Model training failed.
    pause
    exit /b 1
  )
) else (
  echo [3/4] Trained model found.
)

REM ── Launch ────────────────────────────────────────
echo [4/4] Starting server...
echo.
echo  -----------------------------------------------
echo   Open this in your browser:
echo   http://127.0.0.1:8765/
echo  -----------------------------------------------
echo.
echo  Press Ctrl+C to stop the server when done.
echo.
start "" "http://127.0.0.1:8765/"
python -m fake_news_detector.api --model models/fake_news_model.json --host 127.0.0.1 --port 8765 --web-dir web
pause
