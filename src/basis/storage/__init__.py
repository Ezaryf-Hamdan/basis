"""Storage ports: persistence protocols, one per function that needs it.

The first cut of basis put SQL directly in the service classes, which made the
package Postgres-only with no seam - eight modules contained `::vector`,
`<=>`, `tsvector`, `jsonb`, `ON CONFLICT`, `FOR UPDATE`, recursive CTEs and
hardcoded table names. "Portable to any AI platform" was not true; "portable to
any Postgres-backed AI platform" was.

These protocols are the seam. A function depends on a repository interface, and
a repository implementation owns the storage decisions.

Two implementations ship:

  * ``storage.postgres`` - the real one. Keeps every Postgres feature that is
    load-bearing, notably pgvector similarity and the generated tsvector for
    hybrid retrieval. Table names come from a ``TableMap``, so it can be
    pointed at an existing schema.
  * ``storage.inmemory`` - complete, not a stub. It exists for three reasons:
    it lets the whole package be tested without a database, it proves the ports
    are honest (an interface with one implementation is an assumption, not an
    abstraction), and it is a legitimate choice for a single-process consumer
    that does not need durability.

Deliberately *not* abstracted: vector search semantics. `search_chunks` returns
ranked results and a backend is expected to do that however it can. Pretending
a key-value store can do cosine similarity would be a worse lie than admitting
the port needs a capable backend.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from ..context import RunContext

__all__ = [
    "ArtifactRepository",
    "ChunkRepository",
    "ConversationRepository",
    "MemoryRepository",
    "PersonaRepository",
    "TableMap",
    "TaskModelRepository",
    "ToolRepository",
    "WorkflowRepository",
]


class TableMap:
    """Table names a storage backend should use.

    Hardcoded names were the second portability blocker: dropping basis onto a
    schema that already had `documents` or `audit_log` meant writing an adapter
    even when the shapes matched. Override what differs, inherit the rest.
    """

    memories = "agent_memories"
    short_term = "agent_short_term_memory"
    conversations = "agent_conversations"
    tools = "agent_tools"
    persona_tools = "agent_persona_tools"
    tool_invocations = "agent_tool_invocations"
    personas = "agent_personas"
    task_models = "agent_task_models"
    workflow_runs = "basis_workflow_runs"
    workflow_steps = "basis_workflow_steps"
    artifact_versions = "basis_artifact_versions"
    artifact_lineage = "basis_artifact_lineage"
    artifact_baselines = "basis_artifact_baselines"
    artifact_baseline_members = "basis_artifact_baseline_members"
    chunks = "basis_chunks"

    def __init__(self, **overrides: str):
        unknown = set(overrides) - {
            k for k in dir(type(self)) if not k.startswith("_")
        }
        if unknown:
            raise ValueError(
                "TableMap has no such table(s): %s" % ", ".join(sorted(unknown))
            )
        for key, value in overrides.items():
            setattr(self, key, value)


# ── memory ─────────────────────────────────────────────────────────────────


@runtime_checkable
class MemoryRepository(Protocol):
    """Short-term scratch notes and long-term memories."""

    def write_short_term(
        self,
        scope: Any,
        note_type: str,
        content: str,
        metadata: Mapping[str, Any],
        *,
        expires_at: Any = None,
    ) -> str: ...

    def read_short_term(self, scope: Any) -> list[dict[str, Any]]: ...

    def clear_short_term(self, scope: Any) -> int: ...

    def write_long_term(
        self,
        scope: Any,
        *,
        memory_type: str,
        content: str,
        importance: float,
        embedding: Sequence[float] | None,
        source: str,
        metadata: Mapping[str, Any],
    ) -> str: ...

    def recall(
        self,
        scope: Any,
        *,
        query_embedding: Sequence[float] | None,
        limit: int,
        min_importance: float,
        importance_weight: float,
        query_text: str | None = None,
    ) -> list[dict[str, Any]]: ...

    def record_access(self, scope: Any, memory_ids: Sequence[str]) -> None: ...

    def prune_short_term(self, *, tenant_id: str) -> int: ...

    def prune_expired(self, *, tenant_id: str, retain_days: int) -> int: ...


@runtime_checkable
class ConversationRepository(Protocol):
    def save(
        self,
        ctx: RunContext,
        messages: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any] | None,
    ) -> None: ...

    def load(self, ctx: RunContext) -> list[dict[str, Any]]: ...

    def delete(self, ctx: RunContext) -> int: ...


# ── tools ──────────────────────────────────────────────────────────────────


@runtime_checkable
class ToolRepository(Protocol):
    def upsert_catalog(
        self, tenant_id: str, tools: Sequence[Mapping[str, Any]]
    ) -> int: ...

    def grants_for(
        self, *, tenant_id: str, persona_id: str | None
    ) -> frozenset[str] | None: ...

    def record_invocation(self, row: Mapping[str, Any]) -> None: ...

    def prune_invocations(self, *, tenant_id: str, retain_days: int) -> int: ...


# ── workflow ───────────────────────────────────────────────────────────────


@runtime_checkable
class WorkflowRepository(Protocol):
    """Checkpoint persistence. Same shape as `workflow.state.CheckpointStore`,
    restated here so `workflow` depends on `storage` rather than the reverse."""

    def create(self, record: Any) -> None: ...

    def load(self, run_id: str, *, tenant_id: str) -> Any | None: ...

    def save_step(self, record: Any, step_id: str) -> None: ...

    def save_run(self, record: Any) -> None: ...


# ── artifacts ──────────────────────────────────────────────────────────────


@runtime_checkable
class ArtifactRepository(Protocol):
    def create_version(
        self,
        ctx: RunContext,
        *,
        artifact_key: str,
        kind: str,
        content: Any,
        content_hash: str,
        title: str | None,
        state: str,
        metadata: Mapping[str, Any],
        lineage: Sequence[tuple[str, str]],
    ) -> dict[str, Any]: ...

    def head(self, ctx: RunContext, artifact_key: str) -> dict[str, Any] | None: ...

    def history(self, ctx: RunContext, artifact_key: str) -> list[dict[str, Any]]: ...

    def transition(
        self, ctx: RunContext, version_id: str, *, to_state: str, allowed_from: Sequence[str]
    ) -> str: ...

    def sources_of(self, ctx: RunContext, version_id: str) -> list[dict[str, Any]]: ...

    def impact_of(
        self,
        ctx: RunContext,
        version_id: str,
        *,
        kinds: Sequence[str],
        max_depth: int,
    ) -> list[dict[str, Any]]: ...

    def create_baseline(
        self, ctx: RunContext, name: str, *, states: Sequence[str]
    ) -> int: ...


# ── knowledge ──────────────────────────────────────────────────────────────


@runtime_checkable
class ChunkRepository(Protocol):
    """Chunk storage and retrieval.

    ``search_vector`` and ``search_text`` are separate because fusion happens
    above them - a backend that can only do one still works, it just gets
    single-arm results.
    """

    def add_chunks(self, chunks: Sequence[Mapping[str, Any]]) -> int: ...

    def delete_document(self, *, tenant_id: str | None, document_id: str) -> int: ...

    def search_vector(
        self, *, scope: Mapping[str, Any], embedding: Sequence[float], limit: int,
        max_distance: float | None,
    ) -> list[dict[str, Any]]: ...

    def search_text(
        self, *, scope: Mapping[str, Any], text: str, limit: int
    ) -> list[dict[str, Any]]: ...


# ── models / personas config ───────────────────────────────────────────────


@runtime_checkable
class TaskModelRepository(Protocol):
    def resolve(
        self, *, tenant_id: str, task_key: str, role: str, workstream: str | None
    ) -> dict[str, Any] | None: ...


@runtime_checkable
class PersonaRepository(Protocol):
    def all(self, *, tenant_id: str) -> list[dict[str, Any]]: ...

    def by_id(self, *, tenant_id: str, persona_id: str) -> dict[str, Any] | None: ...
