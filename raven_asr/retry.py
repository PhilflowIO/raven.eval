"""Retry decorator for transient ASR-adapter failures.

Wraps async functions with exponential backoff for:
* httpx 429 / 5xx responses (honors ``Retry-After`` header when present)
* httpx network errors (timeouts, connection resets)
* Modal ``ResourceExhaustedError`` (queue full / GPU contention)

Logs every retry at WARNING with attempt number + sleep time, re-raises
after ``max_retries`` exhaustion. Kept dependency-light (no tenacity) so
the harness stays self-contained.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import httpx

logger = logging.getLogger("raven_asr.retry")

P = ParamSpec("P")
T = TypeVar("T")

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _is_modal_exhaustion(exc: BaseException) -> bool:
    """Best-effort detection of Modal queue/resource backpressure.

    Modal raises various exception types across SDK versions
    (``ResourceExhaustedError``, ``FunctionTimeoutError``, gRPC ``RESOURCE_EXHAUSTED``).
    Match by class-name so we don't hard-import modal.exception just for this.
    """
    name = type(exc).__name__
    return name in {
        "ResourceExhaustedError",
        "FunctionTimeoutError",
        "InternalFailure",
    }


def _parse_retry_after(value: str | None) -> float | None:
    """Parse ``Retry-After`` header (seconds form only — HTTP-date is rare on APIs)."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def with_retry(
    max_retries: int = 5,
    base_backoff: float = 1.0,
    max_backoff: float = 60.0,
    jitter: float = 0.25,
) -> Callable[
    [Callable[P, Coroutine[Any, Any, T]]],
    Callable[P, Coroutine[Any, Any, T]],
]:
    """Decorator factory: retry async fn on transient errors.

    Args:
        max_retries: total attempts after the first failure (so 5 = 1 try + 5 retries).
        base_backoff: initial sleep, doubled each attempt.
        max_backoff: hard cap on sleep time per attempt.
        jitter: multiplicative jitter range (0.25 = ±25%).
    """

    def decorator(
        fn: Callable[P, Coroutine[Any, Any, T]],
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            attempt = 0
            while True:
                try:
                    return await fn(*args, **kwargs)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code not in RETRYABLE_STATUS:
                        raise
                    if attempt >= max_retries:
                        raise
                    wait = _parse_retry_after(exc.response.headers.get("Retry-After"))
                    if wait is None:
                        wait = min(max_backoff, base_backoff * (2 ** attempt))
                    wait = wait * (1.0 + random.uniform(-jitter, jitter))
                    logger.warning(
                        "retry %d/%d after HTTP %d: sleeping %.2fs (%s)",
                        attempt + 1,
                        max_retries,
                        exc.response.status_code,
                        wait,
                        getattr(fn, "__qualname__", fn.__name__),
                    )
                    await asyncio.sleep(wait)
                except httpx.HTTPError as exc:  # NetworkError, TimeoutException, etc.
                    if attempt >= max_retries:
                        raise
                    wait = min(max_backoff, base_backoff * (2 ** attempt))
                    wait = wait * (1.0 + random.uniform(-jitter, jitter))
                    logger.warning(
                        "retry %d/%d after network error %s: sleeping %.2fs (%s)",
                        attempt + 1,
                        max_retries,
                        type(exc).__name__,
                        wait,
                        getattr(fn, "__qualname__", fn.__name__),
                    )
                    await asyncio.sleep(wait)
                except Exception as exc:
                    if not _is_modal_exhaustion(exc):
                        raise
                    if attempt >= max_retries:
                        raise
                    wait = min(max_backoff, base_backoff * (2 ** attempt))
                    wait = wait * (1.0 + random.uniform(-jitter, jitter))
                    logger.warning(
                        "retry %d/%d after Modal %s: sleeping %.2fs (%s)",
                        attempt + 1,
                        max_retries,
                        type(exc).__name__,
                        wait,
                        getattr(fn, "__qualname__", fn.__name__),
                    )
                    await asyncio.sleep(wait)
                attempt += 1

        return wrapper

    return decorator
