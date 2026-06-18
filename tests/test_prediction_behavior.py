from __future__ import annotations

import unittest

from fake_news_detector.model import predict_news


class _TieModel:
    def predict_proba(self, texts):  # type: ignore[no-untyped-def]
        return [{0: 0.5, 1: 0.5}]


class PredictionBehaviorTests(unittest.TestCase):
    def test_predict_news_tie_defaults_to_fake(self) -> None:
        result = predict_news(_TieModel(), "Some claim text")
        self.assertEqual(result["label"], "fake")
        self.assertEqual(result["confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()
