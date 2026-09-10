"""Short-term and long-term memory.

Lifted from ``memory_service.MemoryService`` plus the useful parts of
``long_term_memory.LongTermMemory``. The two overlapped in the source: both
wrote ``agent_memories``, with different column sets and different recall
strategies, and neither could usefully see the other's rows.

The SQL now lives in a repository (`storage.postgres.PgMemoryRepository`), so
this class holds the *policy* - what a scope means, how relevance and
importance are blended, when access is recorded - and none of the storage.
That split is what lets the same behaviour run against
`storage.inmemory.InMemoryMemoryRepository` in a test, and against something
that is not Postgres in a consumer that does not have Postgres.

Behavioural fixes carried over from the source:

  * **Recall actually uses the embeddings.** ``memory_service.recall_long_term``
    stored an embedding on write and then ordered by
    ``importance DESC, created_at DESC`` - the vector was never read, so recall
    returned the *most important* memories rather than the most *relevant*
    ones. The parallel ``long_term_memory.recall`` did search content, but with
    ``content ILIKE %word%`` OR'd across query words, under a docstring reading
    "TODO: Upgrade to embedding-based search with Bedrock Titan". The embedding
    column, the Titan call and pgvector all already existed; nothing joined
    them up.
  * **Every predicate is tenant-scoped**, which is what makes cross-client
    recall impossible rather than merely unlikely.
  * **The N+1 access update is gone.** ``LongTermMemory.recall`` called
    ``update_access(id)`` in a Python loop, each opening its own connection, so
    a top-5 recall cost six connections.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from ..storage import MemoryRepository
from .scope import MemoryScope

__all__ = ["MemoryService"]

log = logging.getLogger(__name__)


class MemoryService:
    """Per-(tenant, user, project, persona) scoped memory."""

    def __init__(
        self,
        repo: MemoryRepository | None = None,
        *,
        dsn: str | None = None,
    ):
        if repo is None:
            from ..storage.postgres import PgMemoryRepository

            repo = PgMemoryRepository(dsn=dsn)
        self._repo = repo

    # ── short term ─────────────────────────────────────────────────────────

    def write_short_term(
        self,
        scope: MemoryScope,
        note_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        ttl_seconds: int | None = None,
    ) -> str:
        """Append a scratch note for the current job.

        ``ttl_seconds`` sets an expiry so old notes are invisible after that
        window without requiring an explicit ``clear_short_term`` call. Useful
        for notes that are only relevant within a single turn.
        """
        expires_at = (
            datetime.now(UTC) + timedelta(seconds=ttl_seconds)
            if ttl_seconds is not None
            else None
        )
        return self._repo.write_short_term(
            scope, note_type, content, metadata or {}, expires_at=expires_at
        )

    def read_short_term(self, scope: MemoryScope) -> list[dict[str, Any]]:
        """Scratch notes for this (job, persona), oldest first."""
        return self._repo.read_short_term(scope)

    def clear_short_term(self, scope: MemoryScope) -> int:
        """Drop this job's scratch notes, e.g. after consolidation.

        New. The source never deleted short-term notes, so
        ``agent_short_term_memory`` grew without bound - every job's scratch pad
        retained forever, after its content had already been distilled into
        long-term memory.
        """
        return self._repo.clear_short_term(scope)

    # ── long term ──────────────────────────────────────────────────────────

    def write_long_term(
        self,
        scope: MemoryScope,
        memory_type: str,
        content: str,
        *,
        importance: float = 0.5,
        embedding: Sequence[float] | None = None,
        source: str = "consolidation",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist a durable memory."""
        return self._repo.write_long_term(
            scope,
            memory_type=memory_type,
            content=content,
            importance=importance,
            embedding=embedding,
            source=source,
            metadata=metadata or {},
        )

    def recall(
        self,
        scope: MemoryScope,
        *,
        query_embedding: Sequence[float] | None = None,
        query_text: str | None = None,
        limit: int = 10,
        min_importance: float = 0.0,
        importance_weight: float = 0.25,
        record_access: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories.

        With ``query_embedding``, ranks by cosine distance blended with
        importance. When ``query_text`` is also supplied, vector and full-text
        arms are fused via Reciprocal Rank Fusion — the same approach used by
        knowledge retrieval. Providing both produces the best recall on short,
        keyword-heavy queries that cosine distance alone misses.

        ``distance`` and ``score`` are returned so a caller can threshold or
        rerank.
        """
        rows = self._repo.recall(
            scope,
            query_embedding=query_embedding,
            query_text=query_text,
            limit=limit,
            min_importance=min_importance,
            importance_weight=importance_weight,
        )

        if rows and record_access:
            try:
                self._repo.record_access(scope, [r["id"] for r in rows])
            except Exception as exc:
                log.warning("memory access bookkeeping failed: %s", exc)
        return rows

    def prune_short_term(self, scope: MemoryScope) -> int:
        """Delete expired short-term notes for this tenant.

        Idempotent. Run periodically (e.g. at job start) to keep the table
        from accumulating notes whose TTL has passed.
        """
        return self._repo.prune_short_term(tenant_id=scope.tenant_id)

    def prune_expired(self, scope: MemoryScope, *, retain_days: int = 90) -> int:
        """Delete long-term memories older than ``retain_days`` for this tenant.

        Does not delete memories regardless of age — ``retain_days`` is a
        floor, not a ceiling. Run as a maintenance task, not on every request.
        """
        return self._repo.prune_expired(tenant_id=scope.tenant_id, retain_days=retain_days)
