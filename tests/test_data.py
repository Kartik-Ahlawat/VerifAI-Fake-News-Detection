from __future__ import annotations

import unittest

from fake_news_detector.data import normalize_label


class NormalizeLabelTests(unittest.TestCase):
    def test_normalize_label_variants(self) -> None:
        self.assertEqual(normalize_label("fake"), 0)
        self.assertEqual(normalize_label("FALSE"), 0)
        self.assertEqual(normalize_label("0"), 0)
        self.assertEqual(normalize_label("real"), 1)
        self.assertEqual(normalize_label("True"), 1)
        self.assertEqual(normalize_label("1"), 1)

    def test_normalize_label_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            normalize_label("unknown")


if __name__ == "__main__":
    unittest.main()
