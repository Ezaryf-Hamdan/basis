"""Lazily-resolved settings.

Lifted from AILedSDLC/agent/config.py, with one structural change that matters:
that module raised ``RuntimeError`` from module scope when DATABASE_URL was
unset::

    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is required")

The same five lines appear in config.py, model_config.py, memory.py and
personas.py. Import-time raises make a shared package unusable: a consumer that
only wants tool governance or PII redaction cannot import anything without a
Postgres URL, and no module can be imported in a unit test without one either.
AILedSDLC has a whole test (test_database_url_failclosed.py) devoted to this
behaviour, so the fail-closed intent is deliberate and is preserved - it just
happens at first use rather than at import.

Feature flags are deliberately NOT lifted. config.py carried
FSD_AGENTIC_AUTHORING, FITGAP_AGENTIC_CLASSIFY and CODER_SPECIALIST_ENABLED,
all of which name SAP delivery features. Those belong to the application, not
the substrate.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from .errors import ConfigurationError

__all__ = ["Settings", "reset_settings", "settings"]


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError("%s must be an integer, got %r" % (name, raw)) from exc


@dataclass(frozen=True)
class Settings:
    """Resolved once per process, on first access.

    Nothing here raises on construction. ``require_database_url`` raises when a
    caller actually needs the DB, which keeps the fail-closed guarantee while
    letting the DB-free half of the package import cleanly.
    """

    database_url: str | None = None

    # Connection pool. AILedSDLC opened a fresh psycopg2 connection per call in
    # every DB module - model_config._query_one does it four times to resolve a
    # single task_key. Two separate bugs were filed about the fallout (#200,
    # #367: "psycopg2's `with connection` commits but does NOT close ->
    # backend/pool exhaustion"). A pool is the fix.
    pool_min_size: int = 1
    pool_max_size: int = 10
    pool_timeout: float = 30.0
    pool_reconnect_timeout: float = 10.0

    # Model resolution cache TTL in seconds. model_config.get_model() read the
    # DB on EVERY call ("Reads DB on every call (no cache)"), so one job that
    # invoked N task keys did N round trips for configuration that changes
    # roughly never. 0 disables caching and restores the original behaviour.
    model_config_ttl_seconds: int = 30

    default_model_id: str = "us.anthropic.claude-opus-5"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = 1024
    aws_region: str = "us-east-1"

    service_name: str = "basis"
    service_version: str = "0.1.0"
    otel_endpoint: str | None = None
    # Print spans to stdout. Off by default: a library that writes to stdout
    # unasked corrupts a caller's output and crashes at interpreter shutdown
    # when the batch processor flushes into a closed stream.
    otel_console: bool = False

    # Callback API the agent tier reports activity to (AILedSDLC: Fastify).
    activity_base_url: str | None = None
    activity_timeout_seconds: float = 5.0

    redact_pii: bool = True
    max_span_content_chars: int = 8000

    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=os.environ.get("DATABASE_URL") or None,
            pool_min_size=_int("BASIS_POOL_MIN_SIZE", 1),
            pool_max_size=_int("BASIS_POOL_MAX_SIZE", 10),
            pool_timeout=float(os.environ.get("BASIS_POOL_TIMEOUT", "30")),
            pool_reconnect_timeout=float(
                os.environ.get("BASIS_POOL_RECONNECT_TIMEOUT", "10")
            ),
            model_config_ttl_seconds=_int("BASIS_MODEL_CONFIG_TTL", 30),
            default_model_id=os.environ.get(
                "BASIS_DEFAULT_MODEL_ID", "us.anthropic.claude-opus-5"
            ),
            embedding_model_id=os.environ.get(
                "BASIS_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
            ),
            embedding_dimensions=_int("BASIS_EMBEDDING_DIMENSIONS", 1024),
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            service_name=os.environ.get("BASIS_SERVICE_NAME", "basis"),
            service_version=os.environ.get("BASIS_SERVICE_VERSION", "0.1.0"),
            otel_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or None,
            otel_console=_flag("BASIS_OTEL_CONSOLE", False),
            activity_base_url=os.environ.get("BASIS_ACTIVITY_BASE_URL") or None,
            activity_timeout_seconds=float(
                os.environ.get("BASIS_ACTIVITY_TIMEOUT", "5")
            ),
            redact_pii=_flag("BASIS_REDACT_PII", True),
            max_span_content_chars=_int("BASIS_MAX_SPAN_CONTENT_CHARS", 8000),
        )

    def require_database_url(self) -> str:
        """The fail-closed check, moved from import time to use time."""
        if not self.database_url:
            raise ConfigurationError(
                "DATABASE_URL is required for this operation but is not set"
            )
        return self.database_url


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Process-wide settings, resolved on first call."""
    return Settings.from_env()


def reset_settings() -> None:
    """Drop the cached Settings so the next call re-reads the environment.

    For tests, and for a process that rotates configuration.
    """
    settings.cache_clear()
