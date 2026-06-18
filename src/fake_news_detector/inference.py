"""Shared inference and request-validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .llm import classify_with_gemini, combine_local_and_llm
from .model import predict_news

MAX_TEXT_LENGTH = 50_000

LlmClassifier = Callable[..., dict[str, Any]]


@dataclass(slots=True)
class PredictionOptions:
    use_llm: bool = False
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    llm_weight: float = 0.35


def validate_prediction_text(text: Any, min_length: int = 5) -> str:
    """Validate and normalize free-text prediction input."""
    if not isinstance(text, str) or len(text.strip()) < min_length:
        raise ValueError(f"Field 'text' must be a string with at least {min_length} characters")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Field 'text' exceeds maximum length of {MAX_TEXT_LENGTH} characters")
    return text


def predict_with_optional_llm(
    model: Any,
    text: str,
    options: PredictionOptions,
    llm_classifier: LlmClassifier = classify_with_gemini,
) -> dict[str, Any]:
    """Run local prediction and optionally fuse with LLM result."""
    local_result = predict_news(model, text)
    result = dict(local_result)
    result["method"] = "local_model"

    if options.use_llm and options.gemini_api_key:
        try:
            llm_result = llm_classifier(
                text=text,
                api_key=options.gemini_api_key,
                model=options.gemini_model,
            )
            result = combine_local_and_llm(
                local_result=local_result,
                llm_result=llm_result,
                llm_weight=options.llm_weight,
            )
        except RuntimeError as exc:
            # Known API errors are surfaced while preserving local result.
            result["llm_error"] = str(exc)
            result["llm_error_type"] = "api_error"
        except Exception:  # noqa: BLE001
            result["llm_error"] = "Unexpected error during LLM call"
            result["llm_error_type"] = "unexpected"

    return result
