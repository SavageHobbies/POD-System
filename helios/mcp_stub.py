from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Tuple

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787


def _json_response(handler: BaseHTTPRequestHandler, status: int, data: Any) -> None:
    body = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class MCPHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (stdlib signature)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return _json_response(self, 400, {"error": "invalid_json"})

        if self.path == "/tools/zeitgeist_trend_finder":
            # Ignore payload; return deterministic sample
            return _json_response(
                self,
                200,
                {
                    "trend_name": "retro_minimalist_memes",
                    "keywords": ["retro", "minimalist", "meme", "tee"],
                    "opportunity_score": 8.7,
                    "velocity": "rising",
                    "urgency_level": "immediate",
                    "commercial_indicators": {
                        "social_mentions_24h": 15400,
                        "search_growth_7d": "+340%",
                        "influencer_adoption": "early_mainstream",
                        "product_categories": ["apparel", "accessories"],
                    },
                    "timing_analysis": {
                        "predicted_peak": "14-21_days",
                        "saturation_risk": "medium",
                        "entry_window": "7_days_optimal",
                    },
                    "confidence_level": 0.87,
                },
            )

        return _json_response(self, 404, {"error": "not_found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        # Quieter server: override default stderr logging
        return


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> Tuple[str, int]:
    server = ThreadingHTTPServer((host, port), MCPHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return host, port
