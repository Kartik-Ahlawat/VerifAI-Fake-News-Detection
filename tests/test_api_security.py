from __future__ import annotations

import unittest

from fake_news_detector.api import _parse_content_length, _validate_runtime_security


class ApiSecurityTests(unittest.TestCase):
    def test_content_length_validation(self) -> None:
        self.assertEqual(_parse_content_length("12"), 12)

        with self.assertRaises(ValueError):
            _parse_content_length("abc")
        with self.assertRaises(ValueError):
            _parse_content_length("0")

    def test_public_host_requires_explicit_flags_and_auth(self) -> None:
        with self.assertRaises(ValueError):
            _validate_runtime_security(
                host="0.0.0.0",
                allow_origin="restrict",
                server_api_key="secret",
                allow_public_http=False,
            )

        with self.assertRaises(ValueError):
            _validate_runtime_security(
                host="0.0.0.0",
                allow_origin="restrict",
                server_api_key="",
                allow_public_http=True,
            )

    def test_wildcard_cors_requires_auth(self) -> None:
        with self.assertRaises(ValueError):
            _validate_runtime_security(
                host="127.0.0.1",
                allow_origin="*",
                server_api_key="",
                allow_public_http=False,
            )

        _validate_runtime_security(
            host="127.0.0.1",
            allow_origin="*",
            server_api_key="secret",
            allow_public_http=False,
        )


if __name__ == "__main__":
    unittest.main()
