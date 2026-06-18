
# AI Fake News Detection System

A complete fake-news detection system built with **pure Python AI** (enhanced Multinomial Naive Bayes + improved NLP preprocessing), so it runs without external packages.

## Features
- Trainable AI text classifier (`Multinomial Naive Bayes`)
- Enhanced preprocessing: URL/email/number normalization, stemming, optional bigrams
- Vocabulary filtering (`min_token_freq`, `max_doc_freq_ratio`) for better generalization
- Multi-dataset training (`--data file1.csv file2.csv` or directory)
- Built-in support for common fake-news dataset formats:
  - `label,text` CSV
  - `label,title,text` CSV
  - `Fake.csv` + `True.csv` split files (label inferred from filename)
- Evaluation metrics (accuracy, precision, recall, F1)
- Optional stratified cross-validation
- CLI prediction for single or batch text inputs
- HTTP API for real-time prediction (`/health`, `/predict`)
- Datasets included for quick start

## Project Structure
```
fake-news-detection-ai/
├── data/
│   ├── sample_news.csv
│   └── expanded_demo_news.csv
├── models/
├── web/
│   └── index.html
├── src/fake_news_detector/
│   ├── api.py
│   ├── data.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
└── tests/
```

## Train the Model
Small sample training:
```bash
cd /home/somil/fake-news-detection-ai
PYTHONPATH=src python3 -m fake_news_detector.train \
  --data data/sample_news.csv \
  --model-out models/fake_news_model.json
```

Larger demo training (better baseline):
```bash
cd /home/somil/fake-news-detection-ai
PYTHONPATH=src python3 -m fake_news_detector.train \
  --data data/expanded_demo_news.csv \
  --model-out models/fake_news_model.json \
  --cv-folds 5
```

Training with common real dataset files (`Fake.csv` + `True.csv`):
```bash
cd /home/somil/fake-news-detection-ai
PYTHONPATH=src python3 -m fake_news_detector.train \
  --data data/Fake.csv data/True.csv \
  --model-out models/fake_news_model.json \
  --cv-folds 5 \
  --min-token-freq 3 \
  --max-doc-freq-ratio 0.85
```

Useful tuning flags:
- `--disable-bigrams` (if your dataset is very noisy)
- `--disable-stemming` (if stemming hurts your domain text)
- `--no-deduplicate` (keep exact duplicate rows)

## Predict from CLI
Single text:
```bash
PYTHONPATH=src python3 -m fake_news_detector.predict \
  --model models/fake_news_model.json \
  --text "Breaking claim says all banks will shut forever next week."
```

Hybrid local + Claude API (optional):
```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key"
PYTHONPATH=src python3 -m fake_news_detector.predict \
  --model models/fake_news_model.json \
  --text "Breaking claim says all banks will shut forever next week." \
  --use-anthropic \
  --anthropic-model claude-3-5-haiku-latest \
  --llm-weight 0.35
```

Batch mode (one article per line in file):
```bash
PYTHONPATH=src python3 -m fake_news_detector.predict \
  --model models/fake_news_model.json \
  --input-file data/sample_inputs.txt
```

## Run API
```bash
PYTHONPATH=src python3 -m fake_news_detector.api \
  --model models/fake_news_model.json \
  --host 127.0.0.1 \
  --port 8000
```

`--allow-origin` is `restrict` by default (localhost origins only).

Optional API key protection (recommended even for demos):
```bash
export FAKE_NEWS_API_KEY="change_this_for_demo"
PYTHONPATH=src python3 -m fake_news_detector.api \
  --model models/fake_news_model.json \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key-env FAKE_NEWS_API_KEY
```

Enable hybrid LLM predictions in API (optional):
```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key"
PYTHONPATH=src python3 -m fake_news_detector.api \
  --model models/fake_news_model.json \
  --host 127.0.0.1 \
  --port 8000 \
  --use-llm-default \
  --anthropic-model claude-3-5-haiku-latest \
  --llm-weight 0.35
```

You can also keep `--use-llm-default` off and pass `\"use_llm\": true` per `/predict` request.

Example request:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Officials released audited economic figures in parliament today."}'
```

Request with explicit LLM usage:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Officials released audited economic figures in parliament today.","use_llm":true}'
```

## Frontend Demo UI
Start a simple static server for the web app:
```bash
cd /home/somil/fake-news-detection-ai/web
python3 -m http.server 5500
```

Then open:
- `http://127.0.0.1:5500`

In the UI:
- keep API base URL as `http://127.0.0.1:8000`
- click `Check API`
- paste text and click `Detect Fake News`

## Easy Windows Demo (For Presentation)
Use this when sharing with someone on a Windows laptop.

1. Send the full project folder (zip it first).
2. Ask them to install Python 3.10+ from `python.org` (with `Add Python to PATH` checked).
3. They should extract the folder and run `run_demo.cmd` (recommended).
4. If `run_demo.cmd` is blocked, run `run_demo.ps1` from PowerShell:
   - Open PowerShell in project folder
   - Run: `powershell -ExecutionPolicy Bypass -File .\\run_demo.ps1`
5. The launcher will automatically:
   - create `.venv`
   - install the package
   - train a model if missing
   - start the API + web UI at `http://127.0.0.1:8765/`

Professor demo flow:
- Open `http://127.0.0.1:8765/`
- Paste 1 suspicious-looking text and 1 official-looking text
- Show prediction, confidence, and fake/real probability bars
- Keep terminal open during demo

Troubleshooting:
- If `py` is not found, install Python and reopen terminal.
- If script execution is blocked in PowerShell, use:
  `Set-ExecutionPolicy -Scope Process Bypass`
## Expected Input Format for Custom Data
Supported formats:
1. Single-file labeled CSV:
   - `label` (`fake`/`real`, or `0`/`1`)
   - `text`
2. Single-file labeled CSV with title:
   - `label`
   - `title`
   - `text`
3. Split files:
   - `Fake.csv` and `True.csv` with at least `text` or `title+text`
   - labels are inferred from filename

## Run Tests
```bash
cd /home/somil/fake-news-detection-ai
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Notes
- `expanded_demo_news.csv` is larger and useful for demo/testing, but it is template-generated.
- For real-world reliability, train on a larger verified real dataset.
- Keep API keys in environment variables only; never hardcode them in files.
- Model bundles are serialized with JSON format `fake_news_detector_v1`.
- Public non-local HTTP binding now requires explicit `--allow-public-http` and API key auth; for production, place this API behind a TLS reverse proxy.
