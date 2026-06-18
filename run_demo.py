from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WINDOWS = sys.platform.startswith("win")
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
MODEL_PATH = ROOT / "models" / "fake_news_model.json"


def load_dotenv() -> None:
    """Load key=value pairs from .env into os.environ (does not overwrite existing vars)."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def run(cmd: list[str]) -> None:
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, env=os.environ)


def main() -> None:
    print()
    print("=" * 50)
    print("  Fake News Detector  |  AI-Powered Demo")
    print("=" * 50)
    print()

    load_dotenv()
    if os.environ.get("GEMINI_API_KEY"):
        print(" [OK] Gemini API key loaded.")
    else:
        print(" [!]  No GEMINI_API_KEY found — running in local-only mode.")
    print()

    if not VENV_PYTHON.exists():
        print("[1/4] Setting up Python environment...")
        run([sys.executable, "-m", "venv", ".venv"])
    else:
        print("[1/4] Python environment ready.")

    print("[2/4] Installing package (first run only)...")
    run([str(VENV_PYTHON), "-m", "pip", "install", "--disable-pip-version-check", "-q", "-e", "."])

    if not MODEL_PATH.exists():
        print("[3/4] Training model on demo dataset...")
        run(
            [
                str(VENV_PYTHON),
                "-m",
                "fake_news_detector.train",
                "--data",
                "data/expanded_demo_news.csv",
                "--model-out",
                "models/fake_news_model.json",
            ]
        )
    else:
        print("[3/4] Trained model found.")

    print("[4/4] Starting server...")
    print()
    print("  " + "-" * 45)
    print("   Open this in your browser:")
    print("   http://127.0.0.1:8765/")
    print("  " + "-" * 45)
    print()
    print("  Press Ctrl+C to stop the server when done.")
    print()
    webbrowser.open("http://127.0.0.1:8765/")
    run(
        [
            str(VENV_PYTHON),
            "-m",
            "fake_news_detector.api",
            "--model",
            "models/fake_news_model.json",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
            "--web-dir",
            "web",
        ]
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nLauncher failed with exit code {exc.returncode}.")
        sys.exit(exc.returncode)
