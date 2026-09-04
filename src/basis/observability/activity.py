"""Activity log writer.

Lifted from ``activity_logger.py``. The original posts agent activity to the
Node API with the OTel trace/span ids attached, which is the right design - the
Node tier owns the row and the RBAC check on it - so the shape is kept.

Changes:

  * Identity comes from the RunContext instead of a module-level contextvar
    lookup (``from config import get_service_token, get_delegated_token``). The
    original's fallback chain was ``delegated -> explicit -> transport token``,
    which means a run whose delegated token was never bound silently logged
    under the service identity - the attribution bug the surrounding code was
    written to prevent. Here a run with no credential is reported as such and
    the caller decides.

  * ``httpx`` is optional. If the ``http`` extra is absent, logging degrades to
    a debug line rather than an ImportError at module load.

  * The client is created once per logger rather than per call. The original
    opened a fresh ``httpx.AsyncClient`` inside every ``log()``, so each
    activity row paid a TCP + TLS handshake.

Fire-and-forget semantics are preserved deliberately: a failed activity write
must never break the run that produced it.
"""
from __future__ import annotations

import logging
from typing import Any

from ..context import RunContext, current_context
from ..settings import settings
from .redaction import redact, truncate
from .tracing import current_trace_ids

__all__ = ["ActivityLogger"]

log = logging.getLogger(__name__)


class ActivityLogger:
    """Posts activity events for a run to the platform API."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        ctx: RunContext | None = None,
        timeout: float | None = None,
    ):
        cfg = settings()
        self.base_url = (base_url or cfg.activity_base_url or "").rstrip("/")
        self._ctx = ctx
        self._timeout = timeout if timeout is not None else cfg.activity_timeout_seconds
        self._client: Any = None

    def _context(self) -> RunContext | None:
        return self._ctx if self._ctx is not None else current_context()

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError:
                return None
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def log(
        self,
        activity_type: str,
        *,
        tool_name: str | None = None,
        input_summary: str | None = None,
        output_summary: str | None = None,
        duration_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Record one activity event. Returns True if it was accepted.

        Never raises. Returns False on any failure so a caller that wants to
        count dropped events can.
        """
        ctx = self._context()
        if ctx is None:
            log.debug("activity %s dropped: no RunContext bound", activity_type)
            return False
        if not self.base_url:
            log.debug("activity %s dropped: no activity_base_url configured", activity_type)
            return False

        project_id = ctx.project_id
        if not project_id:
            log.debug("activity %s dropped: run is not project-scoped", activity_type)
            return False

        trace_id, span_id = current_trace_ids()

        body: dict[str, Any] = {
            "activity_type": activity_type,
            "tenant_id": ctx.tenant_id,
            "run_id": ctx.run_id,
        }
        for key, value in (
            ("session_id", ctx.session_id),
            ("job_id", ctx.job_id),
            ("persona_id", ctx.persona_id),
            ("task_key", ctx.task_key),
            ("tool_name", tool_name),
            ("trace_id", trace_id),
            ("span_id", span_id),
        ):
            if value:
                body[key] = value

        # Summaries can contain client document text; redact and cap them the
        # same way span content is treated.
        if input_summary:
            body["input_summary"] = truncate(redact(input_summary), 2000)
        if output_summary:
            body["output_summary"] = truncate(redact(output_summary), 2000)
        if duration_ms is not None:
            body["duration_ms"] = int(duration_ms)
        if metadata:
            body["metadata"] = metadata

        token = ctx.token
        if not token:
            log.warning(
                "activity %s dropped: run %s has no credential; refusing to fall "
                "back to a service identity",
                activity_type,
                ctx.run_id,
            )
            return False

        client = await self._get_client()
        if client is None:
            log.debug("activity %s dropped: httpx not installed", activity_type)
            return False

        url = "%s/projects/%s/agent/activity" % (self.base_url, project_id)
        try:
            resp = await client.post(
                url,
                json=body,
                headers={
                    "Authorization": "Bearer %s" % token,
                    "Content-Type": "application/json",
                },
            )
        except Exception as exc:
            log.warning("activity %s post failed: %s", activity_type, exc)
            return False

        if resp.status_code >= 400:
            log.warning(
                "activity %s rejected: %s %s",
                activity_type,
                resp.status_code,
                resp.text[:200],
            )
            return False
        return True
