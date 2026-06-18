from __future__ import annotations

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fake_news_detector.api import make_handler
from fake_news_detector.model import build_model


class ApiIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        texts = [
            "Viral post claims all airports are closing forever tomorrow.",
            "Anonymous message says gravity will stop for one hour tonight.",
            "Official notice confirms updated airport security rules this week.",
            "Transport ministry published revised train timetable on portal.",
        ]
        labels = [0, 0, 1, 1]

        model = build_model(alpha=1.0, min_token_freq=1)
        model.fit(texts, labels)

        handler = make_handler(
            model_bundle={"model": model, "metrics": {}, "meta": {}},
            model_path="models/test_model.json",
            allow_origin="restrict",
            gemini_api_key="",
            gemini_model="gemini-2.5-flash",
            llm_weight=0.35,
            use_llm_default=False,
        )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_health_endpoint_returns_ok(self) -> None:
        with urlopen(f"{self.base_url}/health", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(response.status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["model_loaded"])
        self.assertIn("version", payload)

    def test_predict_endpoint_returns_label_and_probabilities(self) -> None:
        body = json.dumps(
            {"text": "Official railway bulletin confirms new platform timings."}
        ).encode("utf-8")

        request = Request(
            f"{self.base_url}/predict",
            method="POST",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(response.status, 200)
        self.assertIn(payload["label"], {"fake", "real"})
        self.assertIn("probabilities", payload)
        self.assertEqual(set(payload["probabilities"].keys()), {"fake", "real"})

    def test_predict_rejects_invalid_content_type(self) -> None:
        body = json.dumps({"text": "Some article text for testing."}).encode("utf-8")
        request = Request(
            f"{self.base_url}/predict",
            method="POST",
            data=body,
            headers={"Content-Type": "text/plain"},
        )

        with self.assertRaises(HTTPError) as context:
            urlopen(request, timeout=2)

        self.assertEqual(context.exception.code, 415)


if __name__ == "__main__":
    unittest.main()
