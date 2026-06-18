from __future__ import annotations

import unittest

from fake_news_detector.inference import PredictionOptions, predict_with_optional_llm, validate_prediction_text
from fake_news_detector.model import build_model


class InferenceValidationTests(unittest.TestCase):
    def test_validate_prediction_text_rejects_short_or_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            validate_prediction_text(None)
        with self.assertRaises(ValueError):
            validate_prediction_text("  hi ")

    def test_validate_prediction_text_rejects_oversized_text(self) -> None:
        with self.assertRaises(ValueError):
            validate_prediction_text("x" * 50001)


class HumanFlowPredictionTests(unittest.TestCase):
    def _trained_model(self):
        texts = [
            "Breaking secret rumor says all ATMs will stop forever next Monday.",
            "Forwarded message claims moonlight can cure every disease instantly.",
            "Conspiracy page says election date is canceled without any notice.",
            "Government press release published audited budget statements today.",
            "University website posted official exam timetable and admit card dates.",
            "Transport department announced revised bus routes in the public notice.",
            "Viral thread says gravity will be switched off for one hour tonight.",
            "Hospital circular confirms free vaccination camp schedule for citizens.",
        ]
        labels = [0, 0, 0, 1, 1, 1, 0, 1]
        model = build_model(alpha=1.0, min_token_freq=1)
        model.fit(texts, labels)
        return model

    def test_local_prediction_behaves_like_expected_user_reading(self) -> None:
        model = self._trained_model()

        rumor = predict_with_optional_llm(
            model=model,
            text="A viral post says all schools are permanently shut from tomorrow.",
            options=PredictionOptions(use_llm=False),
        )
        official = predict_with_optional_llm(
            model=model,
            text="Official ministry notice confirms updated school reopening schedule.",
            options=PredictionOptions(use_llm=False),
        )

        self.assertEqual(rumor["label"], "fake")
        self.assertEqual(official["label"], "real")
        self.assertEqual(rumor["method"], "local_model")

    def test_llm_failure_falls_back_to_local_prediction(self) -> None:
        model = self._trained_model()

        def failing_llm_classifier(**kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("Gemini timeout")

        result = predict_with_optional_llm(
            model=model,
            text="Official bulletin confirms new metro timings from next week.",
            options=PredictionOptions(
                use_llm=True,
                gemini_api_key="test-key",
                gemini_model="gemini-2.5-flash",
                llm_weight=0.35,
            ),
            llm_classifier=failing_llm_classifier,
        )

        self.assertEqual(result["method"], "local_model")
        self.assertEqual(result["llm_error_type"], "api_error")
        self.assertIn("Gemini timeout", result["llm_error"])


if __name__ == "__main__":
    unittest.main()
