"""Pure-Python model building, persistence, and inference utilities."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import ID_TO_LABEL

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")
NON_WORD_PATTERN = re.compile(r"[^a-z0-9_<>'\s]")
SPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"<url>|<email>|<num>|[a-z][a-z']+")
SPECIAL_TOKENS = {"<url>", "<email>", "<num>"}

# Compact stop-word list for signal-to-noise improvements.
STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "by",
    "that",
    "this",
    "it",
    "as",
    "after",
    "before",
    "about",
    "into",
    "over",
    "under",
    "their",
    "his",
    "her",
    "its",
    "they",
    "them",
    "will",
    "would",
    "could",
    "should",
    "can",
    "may",
    "might",
    "not",
    "than",
    "then",
    "such",
    "just",
    "also",
    "very",
    "more",
    "most",
    "many",
    "much",
    "some",
    "other",
    "our",
    "your",
    "you",
    "we",
    "i",
}


def normalize_text(text: str) -> str:
    """Normalize noisy patterns while preserving semantic markers."""
    normalized = text.lower()
    normalized = URL_PATTERN.sub(" <url> ", normalized)
    normalized = EMAIL_PATTERN.sub(" <email> ", normalized)
    normalized = NUMBER_PATTERN.sub(" <num> ", normalized)
    normalized = NON_WORD_PATTERN.sub(" ", normalized)
    normalized = SPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def _light_stem(token: str) -> str:
    if token in SPECIAL_TOKENS:
        return token

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("ly"):
        return token[:-2]
    if len(token) > 5 and token.endswith("ment"):
        return token[:-4]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize(text: str, use_bigrams: bool = True, use_stemming: bool = True) -> list[str]:
    """Tokenize input text into normalized unigram/bigram terms."""
    normalized = normalize_text(text)
    raw_tokens = TOKEN_PATTERN.findall(normalized)

    tokens: list[str] = []
    for token in raw_tokens:
        if token not in SPECIAL_TOKENS and token in STOP_WORDS:
            continue

        if use_stemming:
            token = _light_stem(token)

        if token not in SPECIAL_TOKENS and len(token) < 3:
            continue

        tokens.append(token)

    if use_bigrams and len(tokens) > 1:
        bigrams = [f"{left}__{right}" for left, right in zip(tokens, tokens[1:])]
        tokens.extend(bigrams)

    return tokens


@dataclass(slots=True)
class NaiveBayesTextClassifier:
    """Multinomial Naive Bayes text classifier with enhanced preprocessing."""

    alpha: float = 1.0
    use_bigrams: bool = True
    use_stemming: bool = True
    min_token_freq: int = 2
    max_doc_freq_ratio: float = 0.9

    class_counts: Counter[int] = field(default_factory=Counter)
    word_counts: dict[int, Counter[str]] = field(default_factory=dict)
    total_words: Counter[int] = field(default_factory=Counter)
    vocabulary: set[str] = field(default_factory=set)
    trained: bool = False

    def _tokenize(self, text: str) -> list[str]:
        return tokenize(
            text,
            use_bigrams=self.use_bigrams,
            use_stemming=self.use_stemming,
        )

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> "NaiveBayesTextClassifier":
        text_list = list(texts)
        label_list = list(labels)

        if len(text_list) != len(label_list):
            raise ValueError("texts and labels must have the same length")
        if not text_list:
            raise ValueError("Cannot train on empty data")

        self.class_counts = Counter(label_list)
        if len(self.class_counts) < 2:
            raise ValueError("Need at least two classes to train")

        docs_tokens: list[list[str]] = [self._tokenize(text) for text in text_list]

        token_freq = Counter()
        doc_freq = Counter()
        for tokens in docs_tokens:
            token_freq.update(tokens)
            doc_freq.update(set(tokens))

        doc_count = len(docs_tokens)
        max_doc_count = max(1, int(round(self.max_doc_freq_ratio * doc_count)))

        vocabulary: set[str] = set()
        for token, freq in token_freq.items():
            if freq < self.min_token_freq:
                continue
            if doc_freq[token] > max_doc_count:
                continue
            vocabulary.add(token)

        # Fallback: if pruning is too aggressive, keep most frequent tokens.
        if not vocabulary:
            vocabulary = {token for token, _ in token_freq.most_common(200)}

        self.word_counts = {label_id: Counter() for label_id in self.class_counts}
        self.total_words = Counter()
        self.vocabulary = vocabulary

        for tokens, label in zip(docs_tokens, label_list):
            filtered = [token for token in tokens if token in self.vocabulary]
            if not filtered:
                continue
            self.word_counts[label].update(filtered)
            self.total_words[label] += len(filtered)

        if not self.vocabulary:
            raise ValueError("No usable tokens found in training data")

        self.trained = True
        return self

    def predict_proba(self, texts: Iterable[str]) -> list[dict[int, float]]:
        self._ensure_trained()
        results: list[dict[int, float]] = []

        total_samples = sum(self.class_counts.values())
        class_count = len(self.class_counts)
        vocab_size = max(1, len(self.vocabulary))

        for text in texts:
            tokens = [token for token in self._tokenize(text) if token in self.vocabulary]
            token_counts = Counter(tokens)

            log_prob_by_class: dict[int, float] = {}
            for class_id in self.class_counts:
                prior = (self.class_counts[class_id] + self.alpha) / (
                    total_samples + (self.alpha * class_count)
                )
                log_prob = math.log(prior)

                denominator = self.total_words[class_id] + (self.alpha * vocab_size)
                for token, count in token_counts.items():
                    numerator = self.word_counts[class_id].get(token, 0) + self.alpha
                    log_prob += count * math.log(numerator / denominator)

                log_prob_by_class[class_id] = log_prob

            # Softmax normalization for stable probabilities.
            max_log = max(log_prob_by_class.values())
            exp_scores = {
                class_id: math.exp(score - max_log)
                for class_id, score in log_prob_by_class.items()
            }
            total_score = sum(exp_scores.values())
            probabilities = {
                class_id: score / total_score
                for class_id, score in exp_scores.items()
            }
            results.append(probabilities)

        return results

    def predict(self, texts: Iterable[str]) -> list[int]:
        probabilities = self.predict_proba(texts)
        return [max(prob.items(), key=lambda item: item[1])[0] for prob in probabilities]

    def _ensure_trained(self) -> None:
        if not self.trained:
            raise ValueError("Model is not trained")


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _class_metrics(y_true: list[int], y_pred: list[int], class_id: int) -> dict[str, float | int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == class_id and p == class_id)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != class_id and p == class_id)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == class_id and p != class_id)
    support = sum(1 for t in y_true if t == class_id)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "support": support,
    }


def _confusion_matrix_binary(y_true: list[int], y_pred: list[int], positive_class: int = 1) -> dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p == positive_class)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != positive_class and p != positive_class)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_class and p == positive_class)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p != positive_class)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def build_model(
    alpha: float = 1.0,
    use_bigrams: bool = True,
    use_stemming: bool = True,
    min_token_freq: int = 2,
    max_doc_freq_ratio: float = 0.9,
) -> NaiveBayesTextClassifier:
    return NaiveBayesTextClassifier(
        alpha=alpha,
        use_bigrams=use_bigrams,
        use_stemming=use_stemming,
        min_token_freq=min_token_freq,
        max_doc_freq_ratio=max_doc_freq_ratio,
    )


def evaluate_model(model: NaiveBayesTextClassifier, x_test: list[str], y_test: list[int]) -> dict[str, Any]:
    y_pred = model.predict(x_test)

    accuracy = _safe_div(
        sum(1 for t, p in zip(y_test, y_pred) if t == p),
        len(y_test),
    )

    positive_class = 1
    tp = sum(1 for t, p in zip(y_test, y_pred) if t == positive_class and p == positive_class)
    fp = sum(1 for t, p in zip(y_test, y_pred) if t != positive_class and p == positive_class)
    fn = sum(1 for t, p in zip(y_test, y_pred) if t == positive_class and p != positive_class)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    report = {
        ID_TO_LABEL[0]: _class_metrics(y_test, y_pred, 0),
        ID_TO_LABEL[1]: _class_metrics(y_test, y_pred, 1),
        "accuracy": accuracy,
    }

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": _confusion_matrix_binary(y_test, y_pred, positive_class=1),
    }


def _model_to_dict(model: NaiveBayesTextClassifier) -> dict[str, Any]:
    """Serialize a NaiveBayesTextClassifier to a JSON-safe dict."""
    return {
        "__format__": "fake_news_detector_v1",
        "alpha": model.alpha,
        "use_bigrams": model.use_bigrams,
        "use_stemming": model.use_stemming,
        "min_token_freq": model.min_token_freq,
        "max_doc_freq_ratio": model.max_doc_freq_ratio,
        "class_counts": {str(k): v for k, v in model.class_counts.items()},
        "word_counts": {
            str(class_id): dict(counts)
            for class_id, counts in model.word_counts.items()
        },
        "total_words": {str(k): v for k, v in model.total_words.items()},
        "vocabulary": sorted(model.vocabulary),
        "trained": model.trained,
    }


def _model_from_dict(data: dict[str, Any]) -> NaiveBayesTextClassifier:
    """Reconstruct a NaiveBayesTextClassifier from a serialized dict."""
    model = NaiveBayesTextClassifier(
        alpha=float(data["alpha"]),
        use_bigrams=bool(data["use_bigrams"]),
        use_stemming=bool(data["use_stemming"]),
        min_token_freq=int(data["min_token_freq"]),
        max_doc_freq_ratio=float(data["max_doc_freq_ratio"]),
    )
    model.class_counts = Counter({int(k): v for k, v in data["class_counts"].items()})
    model.word_counts = {
        int(class_id): Counter(counts)
        for class_id, counts in data["word_counts"].items()
    }
    model.total_words = Counter({int(k): v for k, v in data["total_words"].items()})
    model.vocabulary = set(data["vocabulary"])
    model.trained = bool(data["trained"])
    return model


def save_model_bundle(bundle: dict[str, Any], output_path: str | Path) -> Path:
    """Save model bundle to disk using JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(bundle)
    if isinstance(serializable.get("model"), NaiveBayesTextClassifier):
        serializable["model"] = _model_to_dict(serializable["model"])
    serializable["__format__"] = "fake_news_detector_v1"
    with path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file)
    return path


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load model bundle from disk."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with model_path.open("r", encoding="utf-8") as file:
        bundle = json.load(file)

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Invalid model bundle format")

    if bundle.get("__format__") != "fake_news_detector_v1":
        raise ValueError("Model bundle is corrupt or from an incompatible version")

    model_data = bundle["model"]
    if not isinstance(model_data, dict):
        raise ValueError("Model bundle is corrupt or from an incompatible version")

    required_fields = {"alpha", "vocabulary", "trained", "class_counts", "word_counts"}
    if not required_fields.issubset(model_data.keys()):
        raise ValueError("Model bundle is corrupt or from an incompatible version")

    bundle["model"] = _model_from_dict(model_data)

    if not isinstance(bundle["model"], NaiveBayesTextClassifier):
        raise ValueError("Model bundle is corrupt or from an incompatible version")

    return bundle


def predict_news(model: NaiveBayesTextClassifier, text: str) -> dict[str, Any]:
    """Predict whether a text looks fake or real."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input text cannot be empty")

    probs = model.predict_proba([cleaned])[0]
    fake_prob = float(probs.get(0, 0.0))
    real_prob = float(probs.get(1, 0.0))

    # Avoid optimistic bias on exact ties; default to fake for 50/50 uncertainty.
    label_id = 1 if real_prob > fake_prob else 0
    return {
        "label": ID_TO_LABEL[label_id],
        "confidence": max(fake_prob, real_prob),
        "probabilities": {
            ID_TO_LABEL[0]: fake_prob,
            ID_TO_LABEL[1]: real_prob,
        },
    }
