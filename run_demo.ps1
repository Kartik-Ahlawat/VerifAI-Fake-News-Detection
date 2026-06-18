$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

Write-Host '============================================'
Write-Host 'Fake News Detector - PowerShell Demo Launcher'
Write-Host '============================================'

if (-not (Test-Path '.venv')) {
  Write-Host '[1/5] Creating virtual environment...'
  try {
    py -3 -m venv .venv
  } catch {
    python -m venv .venv
  }
}

Write-Host '[2/5] Activating virtual environment...'
& .\.venv\Scripts\Activate.ps1

Write-Host '[3/5] Installing project package...'
python -m pip install --disable-pip-version-check -e .

if (-not (Test-Path 'models\fake_news_model.json')) {
  Write-Host '[4/5] Model not found. Training from sample data...'
  python -m fake_news_detector.train --data data/sample_news.csv --model-out models/fake_news_model.json
} else {
  Write-Host '[4/5] Model found: models\fake_news_model.json'
}

Write-Host '[5/5] Starting demo server...'
Start-Process 'http://127.0.0.1:8765/'
python -m fake_news_detector.api --model models/fake_news_model.json --host 127.0.0.1 --port 8765 --web-dir web
