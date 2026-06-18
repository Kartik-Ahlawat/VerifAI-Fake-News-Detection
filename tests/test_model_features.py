from __future__ import annotations

import unittest

from fake_news_detector.model import tokenize


class TokenizerFeatureTests(unittest.TestCase):
    def test_tokenize_adds_special_tokens_and_bigrams(self) -> None:
        text = "Visit https://example.com and read the report 2026 now"
        tokens = tokenize(text, use_bigrams=True, use_stemming=False)

        self.assertIn("<url>", tokens)
        self.assertIn("<num>", tokens)
        self.assertTrue(any("__" in token for token in tokens))


if __name__ == "__main__":
    unittest.main()
