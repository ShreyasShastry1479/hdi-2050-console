"""Local static server plus secure HDI-GPT API proxy.

Run from the project root:

    $env:OPENAI_API_KEY="sk-..."
    python hdi_gpt_server.py

Optional environment variables:
    HDI_GPT_MODEL       default: gpt-4o-mini
    HDI_GPT_BASE_URL    default: https://api.openai.com/v1
    HDI_GPT_PORT        default: 8766

OpenRouter/OpenAI-compatible example:
    $env:HDI_GPT_BASE_URL="https://openrouter.ai/api/v1"
    $env:HDI_GPT_MODEL="openai/gpt-4o-mini"
    $env:OPENAI_API_KEY="..."

Do not put API keys in web/hdi-2050-dashboard.html. The browser posts dashboard
context to /api/hdi-gpt, and this local server attaches the API key.
"""

from __future__ import annotations

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL = os.environ.get("HDI_GPT_MODEL", "gpt-4o-mini")
BASE_URL = os.environ.get("HDI_GPT_BASE_URL", "https://api.openai.com/v1").rstrip("/")
PORT = int(os.environ.get("HDI_GPT_PORT", "8766"))


class HDIGPTHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/api/hdi-gpt":
            self.send_error(404, "Unknown endpoint")
            return
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HDI_GPT_API_KEY")
        if not api_key:
            self._json_response(
                200,
                {
                    "reply": (
                        "HDI-GPT API mode is not configured yet. Set OPENAI_API_KEY "
                        "in the local server environment, then restart hdi_gpt_server.py."
                    ),
                    "tool_calls": [],
                },
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            incoming = json.loads(self.rfile.read(length) or b"{}")
            payload = {
                "model": incoming.get("model") or MODEL,
                "messages": incoming.get("messages", []),
                "tools": incoming.get("tools", []),
                "tool_choice": "auto",
                "temperature": incoming.get("temperature", 0.3),
            }
            request = Request(
                f"{BASE_URL}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urlopen(request, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))
            self._json_response(200, data)
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            self._json_response(error.code, {"error": body})
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as error:
            self._json_response(502, {"error": str(error)})

    def _json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("localhost", PORT), HDIGPTHandler)
    print(f"HDI-GPT server running at http://localhost:{PORT}/web/hdi-2050-dashboard.html")
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("HDI_GPT_API_KEY"):
        print("API key detected. HDI-GPT API-backed chat is enabled.")
    else:
        print("No API key detected. Set OPENAI_API_KEY or HDI_GPT_API_KEY in this shell to enable API-backed chat.")
    server.serve_forever()
