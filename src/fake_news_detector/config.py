"""Central configuration/constants for the fake news detector."""

from __future__ import annotations

DEFAULT_MODEL_PATH = "models/fake_news_model.json"

LABEL_TO_ID = {
    "fake": 0,
    "false": 0,
    "0": 0,
    "real": 1,
    "true": 1,
    "1": 1,
}

ID_TO_LABEL = {
    0: "fake",
    1: "real",
}
