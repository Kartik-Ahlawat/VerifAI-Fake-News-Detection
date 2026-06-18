#!/usr/bin/env bash
set -e

# Load environment variables from .env
if [ -f "$(dirname "$0")/.env" ]; then
    export $(grep -v '^#' "$(dirname "$0")/.env" | xargs)
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY not set. Add it to .env file."
    exit 1
fi

echo "Starting Fake News Detector API on port 8765..."
python3 -m fake_news_detector.api \
    --model models/fake_news_model.json \
    --port 8765 \
    --gemini-key-env GEMINI_API_KEY \
    --llm-weight 0.4
