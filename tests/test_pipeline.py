from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from fake_news_detector.data import load_dataset
from fake_news_detector.model import build_model, load_model_bundle, predict_news, save_model_bundle
from fake_news_detector.train import stratified_split


class PipelineTests(unittest.TestCase):
    def test_train_save_load_predict_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_path = temp_path / "train.csv"

            rows = [
                ["fake", "Rumor says the sun will not rise tomorrow according to a hidden report."],
                ["fake", "Forwarded post claims water can charge a phone battery instantly."],
                ["real", "The government released audited quarterly growth numbers in parliament."],
                ["real", "Public hospital announced new OPD timing through official circular."],
                ["fake", "Clickbait video says gravity was turned off in one city district."],
                ["real", "Railway ministry published timetable changes on the official portal."],
                ["fake", "Anonymous page says all exams are canceled forever from this year."],
                ["real", "University posted exam schedule and fee deadlines on its website."],
            ]

            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["label", "text"])
                writer.writerows(rows)

            dataset = load_dataset(csv_path)
            x_train, x_test, y_train, _ = stratified_split(
                texts=dataset.texts,
                labels=dataset.labels,
                test_size=0.25,
                random_state=42,
            )

            model = build_model(alpha=1.0)
            model.fit(x_train, y_train)

            bundle_path = temp_path / "model.pkl"
            save_model_bundle({"model": model, "meta": {}}, bundle_path)
            bundle = load_model_bundle(bundle_path)

            result = predict_news(bundle["model"], x_test[0])
            self.assertIn(result["label"], {"fake", "real"})
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)
            self.assertEqual(set(result["probabilities"].keys()), {"fake", "real"})


if __name__ == "__main__":
    unittest.main()
