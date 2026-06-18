"""Dataset loading and validation helpers."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import LABEL_TO_ID

WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(slots=True)
class Dataset:
    texts: list[str]
    labels: list[int]

    @property
    def size(self) -> int:
        return len(self.texts)


def normalize_label(value: object) -> int:
    """Normalize supported label strings into binary IDs."""
    key = str(value).strip().lower()
    if key not in LABEL_TO_ID:
        supported = ", ".join(sorted(set(LABEL_TO_ID)))
        raise ValueError(f"Unsupported label '{value}'. Use one of: {supported}")
    return LABEL_TO_ID[key]


def _clean_text(text: str) -> str:
    cleaned = WHITESPACE_PATTERN.sub(" ", text.strip())
    return cleaned


def _resolve_column(fieldnames: list[str], preferred_name: str) -> str | None:
    lookup = {name.strip().lower(): name for name in fieldnames}
    return lookup.get(preferred_name.strip().lower())


def _infer_label_from_filename(path: Path) -> int | None:
    name = path.name.lower()
    if "fake" in name or "false" in name:
        return 0
    if "true" in name or "real" in name:
        return 1
    return None


def _join_non_empty(parts: Iterable[str]) -> str:
    values = [value.strip() for value in parts if value and value.strip()]
    return " ".join(values)


def load_dataset(
    csv_path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
    title_col: str = "title",
    explicit_label: int | None = None,
) -> Dataset:
    """Load dataset from CSV.

    Supported formats:
    - label + text
    - label + title + text
    - Fake/True split files where label is inferred from filename
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset CSV has no header: {path}")

        source_fields = list(reader.fieldnames)
        resolved_text_col = _resolve_column(source_fields, text_col)
        resolved_title_col = _resolve_column(source_fields, title_col)
        resolved_label_col = _resolve_column(source_fields, label_col)

        if resolved_text_col is None and resolved_title_col is None:
            raise ValueError(
                f"Missing text fields in {path}. Expected at least one of: {text_col}, {title_col}"
            )

        inferred_label = explicit_label if explicit_label is not None else _infer_label_from_filename(path)
        if resolved_label_col is None and inferred_label is None:
            raise ValueError(
                f"No label column '{label_col}' in {path}, and filename did not imply class "
                "(expected Fake/True naming)"
            )

        texts: list[str] = []
        labels: list[int] = []

        for line_number, row in enumerate(reader, start=2):
            title = (row.get(resolved_title_col) or "") if resolved_title_col else ""
            body = (row.get(resolved_text_col) or "") if resolved_text_col else ""
            text = _clean_text(_join_non_empty([title, body]))
            if not text:
                continue

            if resolved_label_col:
                raw_label = (row.get(resolved_label_col) or "").strip()
                if raw_label:
                    try:
                        label = normalize_label(raw_label)
                    except ValueError as exc:
                        raise ValueError(f"{exc} (line {line_number} in {path})") from exc
                else:
                    if inferred_label is None:
                        continue
                    label = inferred_label
            else:
                label = inferred_label

            if label is None:
                continue

            texts.append(text)
            labels.append(label)

    if not texts:
        raise ValueError(f"Dataset is empty after cleaning rows: {path}")

    return Dataset(texts=texts, labels=labels)


def _expand_input_paths(paths: Iterable[str | Path]) -> list[Path]:
    expanded: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.csv")))
        else:
            expanded.append(path)

    unique_ordered: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_ordered.append(path)
    return unique_ordered


def load_datasets(
    paths: list[str | Path],
    text_col: str = "text",
    label_col: str = "label",
    title_col: str = "title",
) -> Dataset:
    """Load and concatenate multiple dataset files/directories."""
    if not paths:
        raise ValueError("At least one dataset path is required")

    resolved_paths = _expand_input_paths(paths)
    if not resolved_paths:
        raise ValueError("No CSV files found in provided dataset paths")

    texts: list[str] = []
    labels: list[int] = []
    for path in resolved_paths:
        dataset = load_dataset(
            csv_path=path,
            text_col=text_col,
            label_col=label_col,
            title_col=title_col,
        )
        texts.extend(dataset.texts)
        labels.extend(dataset.labels)

    return Dataset(texts=texts, labels=labels)


def deduplicate_dataset(dataset: Dataset) -> Dataset:
    """Remove exact duplicate text+label pairs while preserving order."""
    seen: set[tuple[str, int]] = set()
    unique_texts: list[str] = []
    unique_labels: list[int] = []

    for text, label in zip(dataset.texts, dataset.labels):
        key = (_clean_text(text).lower(), label)
        if key in seen:
            continue
        seen.add(key)
        unique_texts.append(text)
        unique_labels.append(label)

    return Dataset(texts=unique_texts, labels=unique_labels)
