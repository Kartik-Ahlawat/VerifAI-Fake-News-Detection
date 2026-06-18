from __future__ import annotations

import unittest

from fake_news_detector.llm import combine_local_and_llm, llm_result_to_probabilities


class LlmFusionTests(unittest.TestCase):
    def test_llm_result_to_probabilities(self) -> None:
        fake_probs = llm_result_to_probabilities({"label": "fake", "confidence": 0.8})
        self.assertAlmostEqual(fake_probs["fake"], 0.8)
        self.assertAlmostEqual(fake_probs["real"], 0.2)

        real_probs = llm_result_to_probabilities({"label": "real", "confidence": 0.7})
        self.assertAlmostEqual(real_probs["fake"], 0.3)
        self.assertAlmostEqual(real_probs["real"], 0.7)

    def test_combine_local_and_llm(self) -> None:
        local = {
            "label": "real",
            "confidence": 0.9,
            "probabilities": {"fake": 0.1, "real": 0.9},
        }
        llm = {
            "label": "fake",
            "confidence": 0.9,
            "reason": "Highly sensational and unverified claim style.",
            "provider": "gemini",
            "model": "gemini-2.0-flash",
        }

        combined = combine_local_and_llm(local_result=local, llm_result=llm, llm_weight=0.5)
        self.assertIn(combined["label"], {"fake", "real"})
        self.assertAlmostEqual(
            combined["probabilities"]["fake"] + combined["probabilities"]["real"],
            1.0,
            places=6,
        )
        self.assertEqual(combined["method"], "hybrid_local_plus_gemini")

    def test_high_confidence_fake_llm_overrides_local_label(self) -> None:
        local = {
            "label": "real",
            "confidence": 0.95,
            "probabilities": {"fake": 0.05, "real": 0.95},
        }
        llm = {
            "label": "fake",
            "confidence": 0.92,
            "reason": "Claim is physically implausible and non-verifiable.",
            "provider": "gemini",
            "model": "gemini-2.5-flash",
        }

        combined = combine_local_and_llm(local_result=local, llm_result=llm, llm_weight=0.35)
        self.assertEqual(combined["label"], "fake")
        self.assertEqual(combined.get("safety_override"), "llm_high_confidence_fake")
        self.assertAlmostEqual(combined["probabilities"]["fake"], 0.92)
        self.assertAlmostEqual(combined["probabilities"]["real"], 0.08)


if __name__ == "__main__":
    unittest.main()
