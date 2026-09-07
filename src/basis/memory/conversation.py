"""Conversation persistence.

Lifted from ``memory.py`` (``save_conversation`` / ``load_conversation``). The
SQL moved to a repository; the corrections stayed.

**The load predicate was wrong for a session store.** The original::

    SELECT messages FROM agent_conversations
    WHERE session_id = %s AND user_id = %s
    ORDER BY updated_at DESC LIMIT 1

but the write was ``ON CONFLICT (session_id) DO UPDATE``, so ``session_id`` was
unique and there was never more than one row to order. The ``ORDER BY ... LIMIT 1``
implied rows could duplicate, which they could not.

More consequentially, a unique constraint on ``session_id`` alone makes a
session id global: with tenancy added it has to become
``(tenant_id, session_id)`` or one tenant's session id can collide with
another's and the upsert overwrites it. That is in migration 0001.

**``json.dumps(None)`` writes the string ``"null"``.** The original passed
``context`` straight to ``json.dumps``, so "no context" was stored as a JSON
null rather than SQL NULL.
"""
from __future__ import annotations

from typing import Any

from ..context import RunContext, require_context
from ..storage import ConversationRepository

__all__ = [
    "ConversationStore",
    "delete_conversation",
    "load_conversation",
    "save_conversation",
]


class ConversationStore:
    """Session transcripts, scoped to a tenant."""

    def __init__(
        self,
        repo: ConversationRepository | None = None,
        *,
        dsn: str | None = None,
    ):
        if repo is None:
            from ..storage.postgres import PgConversationRepository

            repo = PgConversationRepository(dsn=dsn)
        self._repo = repo

    def save(
        self,
        messages: list[dict[str, Any]],
        *,
        ctx: RunContext | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        run = ctx if ctx is not None else require_context()
        if not run.session_id:
            raise ValueError("run %s has no session_id" % run.run_id)
        self._repo.save(run, messages, context)

    def load(self, *, ctx: RunContext | None = None) -> list[dict[str, Any]]:
        run = ctx if ctx is not None else require_context()
        if not run.session_id:
            return []
        return self._repo.load(run)

    def delete(self, *, ctx: RunContext | None = None) -> int:
        """Remove a transcript. Needed for any retention or erasure policy."""
        run = ctx if ctx is not None else require_context()
        if not run.session_id:
            return 0
        return self._repo.delete(run)


# Module-level convenience wrappers, matching the shape of the lifted
# functions so a port from `memory.save_conversation(...)` is a one-line change.


def save_conversation(
    messages: list[dict[str, Any]],
    *,
    ctx: RunContext | None = None,
    context: dict[str, Any] | None = None,
    dsn: str | None = None,
) -> None:
    ConversationStore(dsn=dsn).save(messages, ctx=ctx, context=context)


def load_conversation(
    *, ctx: RunContext | None = None, dsn: str | None = None
) -> list[dict[str, Any]]:
    return ConversationStore(dsn=dsn).load(ctx=ctx)


def delete_conversation(
    *, ctx: RunContext | None = None, dsn: str | None = None
) -> int:
    return ConversationStore(dsn=dsn).delete(ctx=ctx)
