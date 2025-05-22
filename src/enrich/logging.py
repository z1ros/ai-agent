"""Structured logging with request correlation.

Emits one JSON object per line so logs are queryable in any aggregator without
regex parsing. A correlation ID is carried in a :class:`~contextvars.ContextVar`,
which means it propagates through async call stacks automatically: every log
line produced while handling one request shares an ID, without threading it
through every function signature.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# LogRecord attributes that are always present. Anything else on the record
# came from an `extra=` argument and belongs in the structured output.
_STANDARD_FIELDS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


def get_correlation_id() -> str | None:
    """Return the correlation ID for the current context, if one is set."""
    return _correlation_id.get()


def set_correlation_id(value: str | None = None) -> str:
    """Set the correlation ID for the current context, generating one if needed."""
    resolved = value or uuid.uuid4().hex[:16]
    _correlation_id.set(resolved)
    return resolved


@contextmanager
def correlation_context(value: str | None = None) -> Iterator[str]:
    """Scope a correlation ID to a block, restoring the previous value after."""
    token = _correlation_id.set(value or uuid.uuid4().hex[:16])
    try:
        yield _correlation_id.get()  # type: ignore[misc]
    finally:
        _correlation_id.reset(token)


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        correlation_id = get_correlation_id()
        if correlation_id:
            payload["correlation_id"] = correlation_id

        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable output for local development."""

    def format(self, record: logging.LogRecord) -> str:
        correlation_id = get_correlation_id()
        prefix = f"[{correlation_id}] " if correlation_id else ""
        base = (
            f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} "
            f"{prefix}{record.name}: {record.getMessage()}"
        )

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_FIELDS and not key.startswith("_")
        }
        if extras:
            base += " " + " ".join(f"{k}={v}" for k, v in extras.items())

        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)

        return base


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install the root handler. Safe to call more than once."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())

    root.addHandler(handler)
    root.setLevel(level.upper())

    # These are chatty at INFO and drown out application logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
