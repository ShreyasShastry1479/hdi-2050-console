"""Vercel Function proxy for HDI-GPT.

The OpenAI-compatible API key remains in Vercel environment variables and is
never exposed to browser JavaScript.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL = os.environ.get("HDI_GPT_MODEL", "gpt-4o-mini")
BASE_URL = os.environ.get("HDI_GPT_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MAX_REQUEST_BYTES = 1_000_000


class handler(BaseHTTPRequestHandler):
    def _headers(self, status: int, length: int = 0) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(length))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._headers(status, len(body))
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._headers(204)

    def do_POST(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HDI_GPT_API_KEY")
        if not api_key:
            self._json(503, {"error": "HDI-GPT API mode is not configured."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._json(413, {"error": "Invalid or oversized request body."})
                return
            incoming = json.loads(self.rfile.read(length))
            payload = {
                "model": MODEL,
                "messages": incoming.get("messages", []),
                "tools": incoming.get("tools", []),
                "tool_choice": "auto",
                "temperature": min(1.0, max(0.0, float(incoming.get("temperature", 0.25)))),
            }
            upstream = Request(
                f"{BASE_URL}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urlopen(upstream, timeout=45) as response:
                self._json(response.status, json.loads(response.read().decode("utf-8")))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            self._json(error.code, {"error": detail})
        except (URLError, TimeoutError, json.JSONDecodeError, TypeError, ValueError) as error:
            self._json(502, {"error": str(error)})
