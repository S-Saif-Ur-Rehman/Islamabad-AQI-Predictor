"""Safe helpers for cleaner runtime behavior."""
from __future__ import annotations

import functools
import logging
import warnings
from typing import Any, Callable, TypeVar

from urllib3.exceptions import InsecureRequestWarning

F = TypeVar("F", bound=Callable[..., Any])


def suppress_known_runtime_warnings() -> None:
    """Hide noisy warnings that are not useful in normal app use."""
    warnings.filterwarnings("ignore", category=InsecureRequestWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"google\._upb\._message.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"google\.protobuf.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*custom tp_new.*")

    for logger_name in ("werkzeug", "urllib3", "hopsworks", "tensorflow", "google"):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        logging.getLogger(logger_name).propagate = False


def sanitize_exception_message(exc: BaseException | None, fallback: str = "request failed") -> str:
    """Return a safe, simple message for users."""
    if exc is None:
        return fallback

    raw = str(exc)
    if not raw:
        return fallback

    blocked_tokens = ("token=", "appid=", "api_key", "Authorization", "Bearer ", "https://", "http://")
    if any(token in raw.lower() for token in blocked_tokens):
        return fallback

    return raw[:250] if len(raw) > 250 else raw


def safe_failure(default_return: Any, logger: logging.Logger | None = None):
    """Return a default value if a function raises an unexpected error."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - behavior is exercised through integration tests
                if logger is not None:
                    logger.exception("Safe failure triggered for %s", func.__name__)
                return default_return

        return wrapper  # type: ignore[return-value]

    return decorator
