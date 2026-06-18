from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from fake_news_detector.data import deduplicate_dataset, load_datasets


class DataLoadingTests(unittest.TestCase):
    def test_load_datasets_with_inferred_labels_from_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_csv = temp_path / "Fake.csv"
            true_csv = temp_path / "True.csv"

            with fake_csv.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["title", "text", "subject"])
                writer.writerow(["Rumor", "A viral post claims banks will close forever.", "politics"])

            with true_csv.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["title", "text", "subject"])
                writer.writerow(["Official", "Ministry published audited budget report today.", "news"])

            dataset = load_datasets([str(fake_csv), str(true_csv)])
            self.assertEqual(dataset.size, 2)
            self.assertEqual(sorted(dataset.labels), [0, 1])

    def test_deduplicate_dataset_removes_exact_duplicates(self) -> None:
        from fake_news_detector.data import Dataset

        dataset = Dataset(
            texts=[
                "Official report confirms policy change.",
                "Official report confirms policy change.",
                "Viral message claims impossible event.",
            ],
            labels=[1, 1, 0],
        )

        deduped = deduplicate_dataset(dataset)
        self.assertEqual(deduped.size, 2)


if __name__ == "__main__":
    unittest.main()
