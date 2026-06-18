"""HTTP API server exposing fake news prediction endpoints."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Type
from urllib.parse import urlparse

from . import __version__
from .config import DEFAULT_MODEL_PATH
from .inference import PredictionOptions, predict_with_optional_llm, validate_prediction_text
from .llm import DEFAULT_GEMINI_MODEL
from .model import load_model_bundle

MAX_BODY_BYTES = 512 * 1024  # 512 KB
RL_MAX_REQUESTS = 60
RL_WINDOW_SECONDS = 60.0
READ_TIMEOUT_SECONDS = 10.0

_rl_lock = threading.Lock()
_rl_counts: dict[str, list[float]] = {}

_LOCAL_ORIGINS = {
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://localhost",
    "http://localhost:8000",
}


def _sanitize_error(msg: str) -> str:
    return re.sub(r'/[^\s"\']+', '[path]', msg)


def _is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def _validate_runtime_security(
    host: str,
    allow_origin: str,
    server_api_key: str,
    allow_public_http: bool,
) -> None:
    if allow_origin not in {"restrict", "*"}:
        raise ValueError("--allow-origin must be either 'restrict' or '*'")

    if allow_origin == "*" and not server_api_key:
        raise ValueError("Wildcard CORS requires API key authentication")

    if _is_loopback_host(host):
        return

    if not allow_public_http:
        raise ValueError(
            "Refusing non-loopback host without explicit --allow-public-http. "
            "Use a TLS reverse proxy in front of this service."
        )

    if not server_api_key:
        print("Warning: Running without API key authentication")

    if allow_origin == "*":
        raise ValueError("Refusing wildcard CORS on non-loopback host")


def _parse_content_length(value: str) -> int:
    try:
        content_length = int(value)
    except ValueError as exc:
        raise ValueError("Invalid Content-Length header") from exc
    if content_length <= 0:
        raise ValueError("Request body is required")
    if content_length > MAX_BODY_BYTES:
        raise ValueError("Request body too large")
    return content_length


def _check_rate_limit(ip: str) -> tuple[bool, int]:
    """Returns (allowed, remaining_requests_in_window)."""
    now = time.monotonic()
    with _rl_lock:
        timestamps = _rl_counts.get(ip, [])
        timestamps = [t for t in timestamps if now - t < RL_WINDOW_SECONDS]
        if len(timestamps) >= RL_MAX_REQUESTS:
            # Prune: store only if non-empty to avoid unbounded dict growth.
            if timestamps:
                _rl_counts[ip] = timestamps
            elif ip in _rl_counts:
                del _rl_counts[ip]
            return False, 0
        timestamps.append(now)
        _rl_counts[ip] = timestamps
        return True, max(0, RL_MAX_REQUESTS - len(timestamps))


class PredictionHandler(BaseHTTPRequestHandler):
    model_bundle: dict[str, Any] | None = None
    model_path: str = ""
    allow_origin: str = "restrict"
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    llm_weight: float = 0.35
    use_llm_default: bool = False
    server_api_key: str = ""
    web_html_path: str = ""
    expose_health_details: bool = False

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(READ_TIMEOUT_SECONDS)

    def _path(self) -> str:
        return urlparse(self.path).path

    def _send_cors_headers(self) -> None:
        if self.allow_origin == "*":
            self.send_header("Access-Control-Allow-Origin", "*")
        else:
            origin = self.headers.get("Origin", "")
            if origin in _LOCAL_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")

    def _send_security_headers(self, is_html: bool = False) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        if is_html:
            # Allow inline styles/scripts (used by the UI) and Google Fonts.
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src https://fonts.gstatic.com; "
                "script-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src *",
            )

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self._send_security_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _authenticate(self) -> bool:
        if not self.server_api_key:
            return True
        key_header = self.headers.get("X-API-Key", "")
        auth_header = self.headers.get("Authorization", "")
        provided = key_header or (auth_header[7:] if auth_header.startswith("Bearer ") else "")
        if provided and hmac.compare_digest(provided, self.server_api_key):
            return True
        self._send_json(401, {"error": "Unauthorized"})
        return False

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._send_cors_headers()
        self._send_security_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self._path() == "/":
            html_path = Path(self.web_html_path) if self.web_html_path else None
            if html_path and html_path.exists():
                body = html_path.read_bytes()
                self.send_response(200)
                self._send_security_headers(is_html=True)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self._send_json(404, {"error": "Web UI not found"})
            return

        if self._path() != "/health":
            self._send_json(404, {"error": "Not found"})
            return

        if not self._authenticate():
            return

        metrics = {}
        meta = {}
        if self.model_bundle:
            raw_metrics = self.model_bundle.get("metrics", {})
            raw_meta = self.model_bundle.get("meta", {})
            if isinstance(raw_metrics, dict):
                # Keep health payload compact for frontend cards.
                metrics = {
                    "accuracy": raw_metrics.get("accuracy"),
                    "precision": raw_metrics.get("precision"),
                    "recall": raw_metrics.get("recall"),
                    "f1": raw_metrics.get("f1"),
                }
            if isinstance(raw_meta, dict):
                meta = {
                    "model_version": raw_meta.get("model_version"),
                    "created_at_utc": raw_meta.get("created_at_utc"),
                    "train_samples": raw_meta.get("train_samples"),
                    "test_samples": raw_meta.get("test_samples"),
                    "test_size": raw_meta.get("test_size"),
                    "alpha": raw_meta.get("alpha"),
                }

        self._send_json(
            200,
            {
                "status": "ok" if self.model_bundle else "loading",
                "model_loaded": self.model_bundle is not None,
                "model_path": self.model_path if self.expose_health_details else "hidden",
                "version": __version__,
                "model_metrics": metrics,
                "model_meta": meta,
                "llm_available": bool(self.gemini_api_key),
                "llm_default_enabled": self.use_llm_default,
                "llm_model": self.gemini_model,
                "llm_weight": self.llm_weight,
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self._path() != "/predict":
            self._send_json(404, {"error": "Not found"})
            return

        if not self._authenticate():
            return

        ct = self.headers.get("Content-Type", "")
        if not ct.startswith("application/json"):
            self._send_json(415, {"error": "Content-Type must be application/json"})
            return

        ip = self.client_address[0]
        allowed, remaining = _check_rate_limit(ip)
        if not allowed:
            self._send_json(429, {"error": "Too many requests"}, extra_headers={
                "X-RateLimit-Limit": str(RL_MAX_REQUESTS),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Window": str(int(RL_WINDOW_SECONDS)),
            })
            return

        try:
            content_length = _parse_content_length(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            status = 413 if str(exc) == "Request body too large" else 400
            self._send_json(status, {"error": str(exc)})
            return

        try:
            raw_body = self.rfile.read(content_length)
        except socket.timeout:
            self._send_json(408, {"error": "Request body read timeout"})
            return
        if len(raw_body) != content_length:
            self._send_json(400, {"error": "Incomplete request body"})
            return
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "Invalid JSON payload"})
            return

        try:
            text = validate_prediction_text(payload.get("text"))
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        if self.model_bundle is None:
            self._send_json(503, {"error": "Model not loaded"})
            return

        try:
            result = predict_with_optional_llm(
                model=self.model_bundle["model"],
                text=text,
                options=PredictionOptions(
                    # Always use Gemini when the server key is configured.
                    use_llm=bool(self.gemini_api_key),
                    gemini_api_key=self.gemini_api_key,
                    gemini_model=self.gemini_model,
                    llm_weight=self.llm_weight,
                ),
            )
        except ValueError as exc:
            self._send_json(400, {"error": _sanitize_error(str(exc))})
            return
        except Exception:  # noqa: BLE001
            self._send_json(500, {"error": "Internal server error"})
            return

        self._send_json(200, result, extra_headers={
            "X-RateLimit-Limit": str(RL_MAX_REQUESTS),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Window": str(int(RL_WINDOW_SECONDS)),
        })

    def version_string(self) -> str:
        # Hide Python/BaseHTTP version info from Server response header.
        return "fake-news-api"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep API output clean in terminal.
        return


def make_handler(
    model_bundle: dict[str, Any],
    model_path: str | Path,
    allow_origin: str,
    gemini_api_key: str,
    gemini_model: str,
    llm_weight: float,
    use_llm_default: bool,
    server_api_key: str = "",
    web_html_path: str = "",
    expose_health_details: bool = False,
) -> Type[PredictionHandler]:
    class _Handler(PredictionHandler):
        pass

    _Handler.model_bundle = model_bundle
    _Handler.model_path = str(Path(model_path).resolve())
    _Handler.allow_origin = allow_origin
    _Handler.gemini_api_key = gemini_api_key
    _Handler.gemini_model = gemini_model
    _Handler.llm_weight = llm_weight
    _Handler.use_llm_default = use_llm_default
    _Handler.server_api_key = server_api_key
    _Handler.web_html_path = web_html_path
    _Handler.expose_health_details = expose_health_details
    return _Handler


def run_server(
    host: str,
    port: int,
    model_path: str,
    allow_origin: str,
    gemini_api_key: str,
    gemini_model: str,
    llm_weight: float,
    use_llm_default: bool,
    server_api_key: str = "",
    web_html_path: str = "",
    expose_health_details: bool = False,
    allow_public_http: bool = False,
) -> None:
    _validate_runtime_security(
        host=host,
        allow_origin=allow_origin,
        server_api_key=server_api_key,
        allow_public_http=allow_public_http,
    )
    model_bundle = load_model_bundle(model_path)
    handler = make_handler(
        model_bundle=model_bundle,
        model_path=model_path,
        allow_origin=allow_origin,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        llm_weight=llm_weight,
        use_llm_default=use_llm_default,
        server_api_key=server_api_key,
        web_html_path=web_html_path,
        expose_health_details=expose_health_details,
    )
    server = ThreadingHTTPServer((host, port), handler)

    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"WARNING: API bound to {host} without TLS. "
            "API keys in headers are transmitted in plaintext."
        )

    if web_html_path:
        print(f"Open http://{host}:{port}/ in your browser")
    print(f"API endpoints: GET /health, POST /predict")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fake news detection HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--allow-origin",
        default="restrict",
        help="CORS origin policy: 'restrict' (default, localhost only) or '*' (allow all)",
    )
    parser.add_argument(
        "--gemini-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable name containing Gemini API key",
    )
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name (default: gemini-2.5-flash)")
    parser.add_argument("--llm-weight", type=float, default=0.35, help="Hybrid LLM fusion weight (0-1)")
    parser.add_argument("--use-llm-default", action="store_true", help="Enable LLM by default for API calls")
    parser.add_argument(
        "--api-key-env",
        default="",
        help="Environment variable name containing server API key (disabled by default)",
    )
    parser.add_argument(
        "--allow-public-http",
        action="store_true",
        help="Allow non-loopback binding over plain HTTP (unsafe; use only behind trusted TLS reverse proxy)",
    )
    parser.add_argument(
        "--expose-health-details",
        action="store_true",
        help="Expose absolute model path in /health response",
    )
    parser.add_argument(
        "--web-dir",
        default="",
        help="Directory containing web/index.html to serve on GET / (auto-detected if omitted)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.llm_weight < 0 or args.llm_weight > 1:
        raise ValueError("--llm-weight must be between 0 and 1")

    gemini_key = os.getenv(args.gemini_key_env, "").strip()

    server_api_key = ""
    if args.api_key_env:
        server_api_key = os.getenv(args.api_key_env, "").strip()
        if not server_api_key:
            raise ValueError(f"Environment variable {args.api_key_env} is empty")

    _validate_runtime_security(
        host=args.host,
        allow_origin=args.allow_origin,
        server_api_key=server_api_key,
        allow_public_http=args.allow_public_http,
    )

    # Auto-detect web/index.html relative to model or cwd.
    web_html_path = ""
    if args.web_dir:
        candidate = Path(args.web_dir) / "index.html"
        if candidate.exists():
            web_html_path = str(candidate)
    else:
        for candidate in [
            Path(args.model).parent.parent / "web" / "index.html",
            Path.cwd() / "web" / "index.html",
        ]:
            if candidate.exists():
                web_html_path = str(candidate)
                break

    run_server(
        host=args.host,
        port=args.port,
        model_path=args.model,
        allow_origin=args.allow_origin,
        gemini_api_key=gemini_key,
        gemini_model=args.gemini_model,
        llm_weight=args.llm_weight,
        use_llm_default=bool(gemini_key),
        server_api_key=server_api_key,
        web_html_path=web_html_path,
        expose_health_details=args.expose_health_details,
        allow_public_http=args.allow_public_http,
    )


if __name__ == "__main__":
    main()
