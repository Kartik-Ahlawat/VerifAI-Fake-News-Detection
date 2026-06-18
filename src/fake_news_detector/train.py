"""CLI to train and evaluate fake news detection model."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import DEFAULT_MODEL_PATH
from .data import deduplicate_dataset, load_datasets
from .model import build_model, evaluate_model, save_model_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fake news detection model")
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="CSV files or directories containing CSV files",
    )
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH, help="Path to save trained model")
    parser.add_argument("--text-col", default="text", help="Primary text/body column name")
    parser.add_argument("--title-col", default="title", help="Optional title column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--no-deduplicate", action="store_true", help="Keep duplicate rows")

    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction to reserve for evaluation")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds (0 to disable)")

    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing factor")
    parser.add_argument("--min-token-freq", type=int, default=2, help="Minimum token frequency to keep")
    parser.add_argument(
        "--max-doc-freq-ratio",
        type=float,
        default=0.9,
        help="Drop tokens appearing in more than this document ratio",
    )
    parser.add_argument("--disable-bigrams", action="store_true", help="Disable bigram features")
    parser.add_argument("--disable-stemming", action="store_true", help="Disable stemming")
    return parser


def stratified_split(
    texts: list[str],
    labels: list[int],
    test_size: float,
    random_state: int,
) -> tuple[list[str], list[str], list[int], list[int]]:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if not (0.05 <= test_size <= 0.5):
        raise ValueError("--test-size must be between 0.05 and 0.5")

    rng = random.Random(random_state)
    indices_by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        indices_by_class.setdefault(label, []).append(idx)

    if len(indices_by_class) < 2:
        raise ValueError("Dataset must contain at least two classes")

    train_indices: list[int] = []
    test_indices: list[int] = []

    for class_indices in indices_by_class.values():
        if len(class_indices) < 2:
            raise ValueError("Each class needs at least 2 samples for train/test split")

        shuffled = class_indices[:]
        rng.shuffle(shuffled)

        test_count = max(1, int(round(len(shuffled) * test_size)))
        if test_count >= len(shuffled):
            test_count = len(shuffled) - 1

        test_indices.extend(shuffled[:test_count])
        train_indices.extend(shuffled[test_count:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    x_train = [texts[i] for i in train_indices]
    y_train = [labels[i] for i in train_indices]
    x_test = [texts[i] for i in test_indices]
    y_test = [labels[i] for i in test_indices]

    if not x_train or not x_test:
        raise ValueError("Invalid split produced empty train or test set")

    return x_train, x_test, y_train, y_test


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stratified_folds(labels: list[int], folds: int, random_state: int) -> list[list[int]]:
    rng = random.Random(random_state)

    indices_by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        indices_by_class.setdefault(label, []).append(idx)

    fold_indices: list[list[int]] = [[] for _ in range(folds)]
    for class_indices in indices_by_class.values():
        shuffled = class_indices[:]
        rng.shuffle(shuffled)
        for i, idx in enumerate(shuffled):
            fold_indices[i % folds].append(idx)

    for chunk in fold_indices:
        rng.shuffle(chunk)
    return fold_indices


def run_cross_validation(
    texts: list[str],
    labels: list[int],
    folds: int,
    random_state: int,
    model_options: dict[str, Any],
) -> dict[str, Any] | None:
    if folds < 2:
        return None

    class_counts = Counter(labels)
    min_class_count = min(class_counts.values())
    actual_folds = min(folds, min_class_count)
    if actual_folds < 2:
        return None

    fold_splits = _stratified_folds(labels, actual_folds, random_state)

    accuracies: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    all_indices = set(range(len(labels)))
    for fold_id, test_indices in enumerate(fold_splits, start=1):
        train_indices = sorted(all_indices - set(test_indices))
        if not train_indices or not test_indices:
            continue

        x_train = [texts[i] for i in train_indices]
        y_train = [labels[i] for i in train_indices]
        x_test = [texts[i] for i in test_indices]
        y_test = [labels[i] for i in test_indices]

        model = build_model(**model_options)
        model.fit(x_train, y_train)

        metrics = evaluate_model(model, x_test, y_test)
        accuracies.append(metrics["accuracy"])
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1_scores.append(metrics["f1"])

    if not accuracies:
        return None

    return {
        "folds": actual_folds,
        "accuracy_mean": _mean(accuracies),
        "precision_mean": _mean(precisions),
        "recall_mean": _mean(recalls),
        "f1_mean": _mean(f1_scores),
    }


def main() -> None:
    args = build_parser().parse_args()

    if args.min_token_freq < 1:
        raise ValueError("--min-token-freq must be >= 1")
    if not (0.5 <= args.max_doc_freq_ratio <= 1.0):
        raise ValueError("--max-doc-freq-ratio must be between 0.5 and 1.0")

    dataset = load_datasets(
        paths=args.data,
        text_col=args.text_col,
        label_col=args.label_col,
        title_col=args.title_col,
    )
    if not args.no_deduplicate:
        dataset = deduplicate_dataset(dataset)

    if dataset.size < 20:
        raise ValueError("Need at least 20 rows to train and evaluate reliably")

    class_counts = Counter(dataset.labels)
    if len(class_counts) < 2:
        raise ValueError("Dataset must contain both fake and real samples")

    model_options: dict[str, Any] = {
        "alpha": args.alpha,
        "use_bigrams": not args.disable_bigrams,
        "use_stemming": not args.disable_stemming,
        "min_token_freq": args.min_token_freq,
        "max_doc_freq_ratio": args.max_doc_freq_ratio,
    }

    cv_summary = run_cross_validation(
        texts=dataset.texts,
        labels=dataset.labels,
        folds=args.cv_folds,
        random_state=args.random_state,
        model_options=model_options,
    )

    x_train, x_test, y_train, y_test = stratified_split(
        texts=dataset.texts,
        labels=dataset.labels,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = build_model(**model_options)
    model.fit(x_train, y_train)

    metrics = evaluate_model(model, x_test, y_test)

    bundle = {
        "model": model,
        "metrics": metrics,
        "meta": {
            "model_version": __version__,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_data": [str(Path(path).resolve()) for path in args.data],
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "class_distribution": dict(class_counts),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "alpha": args.alpha,
            "min_token_freq": args.min_token_freq,
            "max_doc_freq_ratio": args.max_doc_freq_ratio,
            "use_bigrams": not args.disable_bigrams,
            "use_stemming": not args.disable_stemming,
            "cv_folds": cv_summary["folds"] if cv_summary else 0,
            "duplicates_removed": not args.no_deduplicate,
            "dataset_size": dataset.size,
        },
        "cross_validation": cv_summary,
    }

    saved_path = save_model_bundle(bundle, args.model_out)
    summary = {
        "saved_model": str(saved_path.resolve()),
        "metrics": {
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
        },
        "cross_validation": (
            {
                "folds": cv_summary["folds"],
                "accuracy_mean": round(cv_summary["accuracy_mean"], 4),
                "precision_mean": round(cv_summary["precision_mean"], 4),
                "recall_mean": round(cv_summary["recall_mean"], 4),
                "f1_mean": round(cv_summary["f1_mean"], 4),
            }
            if cv_summary
            else None
        ),
        "meta": bundle["meta"],
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
