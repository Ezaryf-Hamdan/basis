"""Retryability classification and backoff.

``is_retryable_bedrock_error`` is lifted from ``model_config`` essentially
as-is - the set of Bedrock error codes it treats as retryable is operational
knowledge worth keeping, including the subtle ``ValidationException`` case
where the message mentions a model (an unavailable model id looks like a
validation failure, not a service error).

Added: actual backoff. The original walked its fallback chain immediately on a
retryable error::

    for model in chain:
        try:
            return await loop.run_in_executor(...)
        except Exception as e:
            if not is_retryable_bedrock_error(e): raise
            last_err = e
            continue

so a ``ThrottlingException`` moved straight to the next model with no delay,
and once the chain was exhausted the whole call failed. Under throttling that
is the wrong shape twice over: it converts a transient capacity problem into a
permanent failure, and it burns the fallback chain in microseconds. Retrying
the same model with jittered exponential backoff before falling back is both
cheaper and more likely to succeed.
"""
from __future__ import annotations

import random
from collections.abc import Iterator

__all__ = [
    "RETRYABLE_CODES",
    "backoff_delays",
    "is_retryable",
]

#: Lifted verbatim from ``model_config.RETRYABLE_BEDROCK_CODES``.
RETRYABLE_CODES = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "ModelNotReadyException",
        "ModelTimeoutException",
        # Added: Bedrock returns these under load and both are transient.
        "InternalServerException",
        "TooManyRequestsException",
    }
)


def is_retryable(err: BaseException) -> bool:
    """True if the error should be retried or fall back to the next model.

    Imports botocore lazily so this module is usable without the ``bedrock``
    extra - a consumer classifying errors from its own provider gets the
    generic paths without needing boto installed.
    """
    try:
        from botocore.exceptions import ClientError, ReadTimeoutError
        from botocore.exceptions import ConnectionError as BotoConnectionError
    except ImportError:
        return False

    if isinstance(err, (ReadTimeoutError, BotoConnectionError)):
        return True

    if isinstance(err, ClientError):
        error = err.response.get("Error", {}) if hasattr(err, "response") else {}
        code = error.get("Code", "")
        if code in RETRYABLE_CODES:
            return True
        if code == "ValidationException":
            # An unavailable or mis-provisioned model id surfaces as a
            # validation error; falling back to the next model is the right
            # response, while a genuine schema violation is not retryable.
            message = str(error.get("Message", "")).lower()
            return "model" in message
        return False

    return False


def backoff_delays(
    attempts: int,
    *,
    base: float = 0.5,
    cap: float = 8.0,
    jitter: bool = True,
) -> Iterator[float]:
    """Jittered exponential backoff delays, in seconds.

    Full jitter (uniform over [0, computed]) rather than a fixed multiplier:
    when several concurrent jobs are throttled together, equal-delay retries
    re-collide. Bulk fan-out in the source repo makes that the normal case, not
    the edge case.
    """
    for i in range(attempts):
        delay = min(cap, base * (2**i))
        yield random.uniform(0, delay) if jitter else delay
