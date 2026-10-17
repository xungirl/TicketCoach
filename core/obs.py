"""
Observability: one JSON object per log line on stdout.

Cloud Run ships stdout to Cloud Logging, which parses JSON lines and maps
"severity" / "message" natively, so every extra field (request_id, model,
latency_ms, tokens...) becomes filterable, e.g.
    jsonPayload.message="llm_call_failed" AND jsonPayload.model="qwen-plus"
"""

import json
import logging
import os
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
        }
        payload.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging() -> None:
    """Idempotent: route the app's loggers to stdout as JSON lines."""
    root = logging.getLogger("ticketcoach")
    if getattr(root, "_json_configured", False):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    root.propagate = False
    root._json_configured = True
