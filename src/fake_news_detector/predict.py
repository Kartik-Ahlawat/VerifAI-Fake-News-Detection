"""CLI to run predictions with a trained fake news model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODEL_PATH
from .inference import PredictionOptions, predict_with_optional_llm
from .llm import DEFAULT_GEMINI_MODEL
from .model import load_model_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict fake/real news from text")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained model bundle")
    parser.add_argument("--text", help="Single text to classify")
    parser.add_argument("--input-file", help="Path to file with one text per line")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini API for hybrid prediction")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name")
    parser.add_argument(
        "--gemini-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable name containing Gemini API key",
    )
    parser.add_argument(
        "--llm-weight",
        type=float,
        default=0.35,
        help="Hybrid fusion weight for LLM result (0 to 1)",
    )
    return parser


def load_inputs(text: str | None, input_file: str | None) -> list[str]:
    if bool(text) == bool(input_file):
        raise ValueError("Provide exactly one of --text or --input-file")

    if text:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("--text cannot be empty")
        return [cleaned]

    path = Path(input_file or "")
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError("Input file has no non-empty lines")
    return lines


def main() -> None:
    args = build_parser().parse_args()
    inputs = load_inputs(args.text, args.input_file)

    if args.llm_weight < 0 or args.llm_weight > 1:
        raise ValueError("--llm-weight must be between 0 and 1")

    gemini_key = os.getenv(args.gemini_key_env, "").strip()
    use_gemini = bool(gemini_key) or args.use_gemini
    if args.use_gemini and not gemini_key:
        raise ValueError(
            f"--use-gemini requires {args.gemini_key_env} environment variable to be set"
        )

    bundle = load_model_bundle(args.model)
    model = bundle["model"]

    results: list[dict[str, Any]] = []
    for text in inputs:
        result = predict_with_optional_llm(
            model=model,
            text=text,
            options=PredictionOptions(
                # Automatically enable Gemini when API key is present.
                use_llm=use_gemini,
                gemini_api_key=gemini_key,
                gemini_model=args.gemini_model,
                llm_weight=args.llm_weight,
            ),
        )

        result["text"] = text
        results.append(result)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    for index, item in enumerate(results, start=1):
        print(
            f"[{index}] Label: {item['label']} | Confidence: {item['confidence']:.4f} "
            f"| Method: {item.get('method', 'local_model')}"
        )
        print(f"    Fake: {item['probabilities']['fake']:.4f} | Real: {item['probabilities']['real']:.4f}")
        if "llm_error" in item:
            print(f"    LLM Warning: {item['llm_error']}")
        print(f"    Text: {item['text']}")


if __name__ == "__main__":
    main()
