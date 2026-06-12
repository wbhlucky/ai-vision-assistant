from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

# Per-request request ID, safe across concurrent coroutines
_request_id: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    return _request_id.get()


def set_request_id(rid: str) -> ContextVar.token:
    return _request_id.set(rid)


def reset_request_id(token: ContextVar.token) -> None:
    _request_id.reset(token)


def new_request_id() -> str:
    return uuid.uuid4().hex


class RequestIdFilter(logging.Filter):
    """Inject the current request ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


def configure_logging(level: str = "INFO") -> None:
    # Ensure UTF-8 output on Windows consoles
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s request_id=%(request_id)s %(name)s - %(message)s",
    )

    rid_filter = RequestIdFilter()
    root = logging.getLogger()
    root.addFilter(rid_filter)
    for handler in root.handlers:
        handler.addFilter(rid_filter)
