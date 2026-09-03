"""Postgres implementations of every storage port.

All the SQL that used to sit inside the service classes lives here. Moving it
was mechanical, but it changes two things that matter:

  * every table name now comes from a ``TableMap``, so basis can be pointed at
    an existing schema without an adapter,
  * the services no longer contain SQL, so a non-Postgres backend is a matter
    of writing a sibling module rather than editing eight files.

Postgres features kept deliberately, because they are load-bearing rather than
incidental: pgvector `<=>` cosine distance (the whole retrieval story),
generated `tsvector` columns (a denormalized index the application has to
maintain is one that goes stale), `FOR UPDATE` on version heads (the only thing
stopping two writers creating the same version number), recursive CTEs (lineage
traversal), and `ON CONFLICT` upserts.
"""
from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .. import db
from ..context import RunContext
from . import TableMap

if TYPE_CHECKING:  # annotations only - see PgWorkflowRepository.load
    from ..workflow.state import RunRecord

__all__ = [
    "PgArtifactRepository",
    "PgChunkRepository",
    "PgConversationRepository",
    "PgMemoryRepository",
    "PgPersonaRepository",
    "PgTaskModelRepository",
    "PgToolRepository",
    "PgWorkflowRepository",
]

log = logging.getLogger(__name__)


def _vector_literal(embedding: Sequence[float]) -> str:
    """pgvector text form, 6 decimals - the format the lifted code used."""
    return "[" + ",".join("%.6f" % x for x in embedding) + "]"


class _PgBase:
    def __init__(self, *, dsn: str | None = None, tables: TableMap | None = None):
        self._dsn = dsn
        self.t = tables or TableMap()


# ── memory ─────────────────────────────────────────────────────────────────


class PgMemoryRepository(_PgBase):
    def write_short_term(
        self, scope: Any, note_type: str, content: str, metadata: Mapping[str, Any]
    ) -> str:
        note_id = str(uuid.uuid4())
        db.execute(
            f"""
            INSERT INTO {self.t.short_term}
              (id, tenant_id, job_id, run_id, persona_id, user_id, project_id,
               note_type, content, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                note_id,
                scope.tenant_id,
                scope.require_job(),
                scope.run_id,
                scope.persona_id,
                scope.user_id,
                scope.project_id,
                note_type,
                content,
                json.dumps(dict(metadata or {})),
            ),
            dsn=self._dsn,
        )
        return note_id

    def read_short_term(self, scope: Any) -> list[dict[str, Any]]:
        return db.query_all(
            f"""
            SELECT id, note_type, content, metadata, created_at
            FROM {self.t.short_term}
            WHERE tenant_id = %s AND job_id = %s AND persona_id = %s
            ORDER BY created_at
            """,
            (scope.tenant_id, scope.require_job(), scope.persona_id),
            dsn=self._dsn,
        )

    def clear_short_term(self, scope: Any) -> int:
        return db.execute(
            f"DELETE FROM {self.t.short_term} "
            "WHERE tenant_id = %s AND job_id = %s AND persona_id = %s",
            (scope.tenant_id, scope.require_job(), scope.persona_id),
            dsn=self._dsn,
        )

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
    ) -> str:
        mem_id = str(uuid.uuid4())
        db.execute(
            f"""
            INSERT INTO {self.t.memories}
              (id, tenant_id, user_id, project_id, persona_id, job_id, run_id,
               memory_type, content, embedding, importance, source,
               access_count, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, 0, %s::jsonb)
            """,
            (
                mem_id,
                scope.tenant_id,
                scope.user_id,
                scope.project_id,
                scope.persona_id,
                scope.job_id,
                scope.run_id,
                memory_type,
                content,
                _vector_literal(embedding) if embedding is not None else None,
                importance,
                source,
                json.dumps(dict(metadata or {})),
            ),
            dsn=self._dsn,
        )
        return mem_id

    def recall(
        self,
        scope: Any,
        *,
        query_embedding: Sequence[float] | None,
        limit: int,
        min_importance: float,
        importance_weight: float,
    ) -> list[dict[str, Any]]:
        if query_embedding is not None:
            return db.query_all(
                f"""
                SELECT id, memory_type, content, importance, access_count,
                       created_at, metadata,
                       (embedding <=> %(qvec)s::vector) AS distance,
                       (1.0 - (embedding <=> %(qvec)s::vector))
                         * (1.0 - %(iw)s)
                         + importance * %(iw)s AS score
                FROM {self.t.memories}
                WHERE tenant_id  = %(tenant)s
                  AND user_id    = %(user)s
                  AND project_id = %(project)s
                  AND persona_id = %(persona)s
                  AND importance >= %(min_importance)s
                  AND embedding IS NOT NULL
                ORDER BY score DESC
                LIMIT %(limit)s
                """,
                {
                    "qvec": _vector_literal(query_embedding),
                    "iw": importance_weight,
                    "tenant": scope.tenant_id,
                    "user": scope.user_id,
                    "project": scope.project_id,
                    "persona": scope.persona_id,
                    "min_importance": min_importance,
                    "limit": limit,
                },
                dsn=self._dsn,
            )

        return db.query_all(
            f"""
            SELECT id, memory_type, content, importance, access_count,
                   created_at, metadata, NULL AS distance, importance AS score
            FROM {self.t.memories}
            WHERE tenant_id  = %s AND user_id = %s
              AND project_id = %s AND persona_id = %s
              AND importance >= %s
            ORDER BY importance DESC, created_at DESC
            LIMIT %s
            """,
            (
                scope.tenant_id,
                scope.user_id,
                scope.project_id,
                scope.persona_id,
                min_importance,
                limit,
            ),
            dsn=self._dsn,
        )

    def record_access(self, scope: Any, memory_ids: Sequence[str]) -> None:
        if not memory_ids:
            return
        # One statement, not the per-row loop with its own connection that
        # `LongTermMemory.recall` used.
        db.execute(
            f"""
            UPDATE {self.t.memories}
            SET access_count = access_count + 1, last_accessed_at = NOW()
            WHERE tenant_id = %s AND id = ANY(%s)
            """,
            (scope.tenant_id, list(memory_ids)),
            dsn=self._dsn,
        )


class PgConversationRepository(_PgBase):
    def save(
        self,
        ctx: RunContext,
        messages: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any] | None,
    ) -> None:
        db.execute(
            f"""
            INSERT INTO {self.t.conversations}
              (tenant_id, session_id, user_id, project_id, persona_id, messages,
               context, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, NOW())
            ON CONFLICT (tenant_id, session_id) DO UPDATE SET
              messages   = EXCLUDED.messages,
              context    = EXCLUDED.context,
              persona_id = EXCLUDED.persona_id,
              updated_at = NOW()
            """,
            (
                ctx.tenant_id,
                ctx.session_id,
                ctx.user_id,
                ctx.project_id,
                ctx.persona_id,
                json.dumps(list(messages)),
                # Not json.dumps(None): that writes the string "null".
                json.dumps(dict(context)) if context is not None else None,
            ),
            dsn=self._dsn,
        )

    def load(self, ctx: RunContext) -> list[dict[str, Any]]:
        row = db.query_one(
            f"SELECT messages FROM {self.t.conversations} "
            "WHERE tenant_id = %s AND session_id = %s AND user_id = %s",
            (ctx.tenant_id, ctx.session_id, ctx.user_id),
            dsn=self._dsn,
        )
        if not row or not row.get("messages"):
            return []
        messages = row["messages"]
        if isinstance(messages, str):
            return json.loads(messages)
        return messages if isinstance(messages, list) else []

    def delete(self, ctx: RunContext) -> int:
        return db.execute(
            f"DELETE FROM {self.t.conversations} "
            "WHERE tenant_id = %s AND session_id = %s AND user_id = %s",
            (ctx.tenant_id, ctx.session_id, ctx.user_id),
            dsn=self._dsn,
        )


# ── tools ──────────────────────────────────────────────────────────────────


class PgToolRepository(_PgBase):
    def upsert_catalog(
        self, tenant_id: str, tools: Sequence[Mapping[str, Any]]
    ) -> int:
        seen: list[str] = []
        with db.cursor(self._dsn, dict_rows=False) as cur:
            for tool in tools:
                name = tool.get("name")
                if not name:
                    continue
                seen.append(name)
                cur.execute(
                    f"""
                    INSERT INTO {self.t.tools}
                      (id, tenant_id, tool_name, description, schema, effect,
                       source, is_active, last_seen_at)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s, 'mcp', TRUE, NOW())
                    ON CONFLICT (tenant_id, tool_name) DO UPDATE SET
                      description  = EXCLUDED.description,
                      schema       = EXCLUDED.schema,
                      effect       = EXCLUDED.effect,
                      is_active    = TRUE,
                      last_seen_at = NOW()
                    """,
                    (
                        str(uuid.uuid4()),
                        tenant_id,
                        name,
                        tool.get("description") or "",
                        json.dumps(tool.get("schema") or {}),
                        tool.get("effect") or "unknown",
                    ),
                )
            # Deactivate anything the server no longer advertises, so a removed
            # tool stops being grantable without a migration.
            if seen:
                cur.execute(
                    f"UPDATE {self.t.tools} SET is_active = FALSE "
                    "WHERE tenant_id = %s AND source = 'mcp' AND tool_name <> ALL(%s)",
                    (tenant_id, seen),
                )
            else:
                cur.execute(
                    f"UPDATE {self.t.tools} SET is_active = FALSE "
                    "WHERE tenant_id = %s AND source = 'mcp'",
                    (tenant_id,),
                )
        return len(seen)

    def grants_for(
        self, *, tenant_id: str, persona_id: str | None
    ) -> frozenset[str] | None:
        # None means "no persona narrowing"; an empty set means "granted
        # nothing". That distinction is the fail-closed part.
        if not persona_id:
            return None
        rows = db.query_all(
            f"""
            SELECT t.tool_name
            FROM {self.t.persona_tools} pt
            JOIN {self.t.tools} t ON t.id = pt.tool_id
            WHERE pt.persona_id = %s AND pt.tenant_id = %s
              AND t.tenant_id = %s AND pt.enabled = TRUE AND t.is_active = TRUE
            """,
            (persona_id, tenant_id, tenant_id),
            dsn=self._dsn,
        )
        return frozenset(r["tool_name"] for r in rows)

    def record_invocation(self, row: Mapping[str, Any]) -> None:
        db.execute(
            f"""
            INSERT INTO {self.t.tool_invocations}
              (id, tenant_id, project_id, run_id, job_id, persona_id,
               principal_id, principal_kind, tool_name, effect, allowed,
               denied_reason, duration_ms, error, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """,
            (
                str(uuid.uuid4()),
                row["tenant_id"],
                row.get("project_id"),
                row["run_id"],
                row.get("job_id"),
                row.get("persona_id"),
                row.get("principal_id"),
                row.get("principal_kind"),
                row["tool_name"],
                row.get("effect"),
                row.get("allowed", True),
                row.get("denied_reason"),
                row.get("duration_ms"),
                row.get("error"),
            ),
            dsn=self._dsn,
        )

    def prune_invocations(self, *, tenant_id: str, retain_days: int) -> int:
        # Retention by age, per tenant - not a global row-count FIFO, which
        # let a busy tenant evict a quiet tenant's audit history.
        return db.execute(
            f"""
            DELETE FROM {self.t.tool_invocations}
            WHERE tenant_id = %s AND created_at < NOW() - MAKE_INTERVAL(days => %s)
            """,
            (tenant_id, retain_days),
            dsn=self._dsn,
        )


# ── artifacts ──────────────────────────────────────────────────────────────


class PgArtifactRepository(_PgBase):
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
    ) -> dict[str, Any]:
        new_id = str(uuid.uuid4())
        with db.cursor(self._dsn) as cur:
            # FOR UPDATE serializes concurrent version creation on this key -
            # without it two writers both compute "version 3".
            cur.execute(
                f"""
                SELECT id, version, state, content_hash
                FROM {self.t.artifact_versions}
                WHERE tenant_id = %s AND artifact_key = %s
                ORDER BY version DESC LIMIT 1
                FOR UPDATE
                """,
                (ctx.tenant_id, artifact_key),
            )
            head = cur.fetchone()

            if head and head["content_hash"] == content_hash:
                return {"duplicate": True, "version": head["version"]}

            next_version = (head["version"] + 1) if head else 1
            supersedes_id = str(head["id"]) if head else None

            if head:
                cur.execute(
                    f"UPDATE {self.t.artifact_versions} SET state = 'superseded' "
                    "WHERE id = %s AND tenant_id = %s",
                    (head["id"], ctx.tenant_id),
                )

            cur.execute(
                f"""
                INSERT INTO {self.t.artifact_versions}
                  (id, tenant_id, project_id, artifact_key, kind, version,
                   title, content, content_hash, state, supersedes_id,
                   created_by, run_id, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s,
                        %s::jsonb, NOW())
                RETURNING created_at
                """,
                (
                    new_id,
                    ctx.tenant_id,
                    ctx.project_id,
                    artifact_key,
                    kind,
                    next_version,
                    title,
                    json.dumps(content, default=str),
                    content_hash,
                    state,
                    supersedes_id,
                    ctx.user_id,
                    ctx.run_id,
                    json.dumps(dict(metadata or {})),
                ),
            )
            created_at = cur.fetchone()["created_at"]

            edges = list(lineage)
            if supersedes_id:
                edges.append((supersedes_id, "revises"))
            for source_id, link in edges:
                cur.execute(
                    f"""
                    INSERT INTO {self.t.artifact_lineage}
                      (id, tenant_id, target_id, source_id, kind, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (target_id, source_id, kind) DO NOTHING
                    """,
                    (str(uuid.uuid4()), ctx.tenant_id, new_id, source_id, link),
                )

        return {
            "duplicate": False,
            "id": new_id,
            "version": next_version,
            "supersedes_id": supersedes_id,
            "created_at": created_at,
        }

    def head(self, ctx: RunContext, artifact_key: str) -> dict[str, Any] | None:
        return db.query_one(
            f"SELECT * FROM {self.t.artifact_versions} "
            "WHERE tenant_id = %s AND artifact_key = %s ORDER BY version DESC LIMIT 1",
            (ctx.tenant_id, artifact_key),
            dsn=self._dsn,
        )

    def history(self, ctx: RunContext, artifact_key: str) -> list[dict[str, Any]]:
        return db.query_all(
            f"""
            SELECT id, version, state, content_hash, title, created_by,
                   approved_by, created_at
            FROM {self.t.artifact_versions}
            WHERE tenant_id = %s AND artifact_key = %s
            ORDER BY version DESC
            """,
            (ctx.tenant_id, artifact_key),
            dsn=self._dsn,
        )

    def transition(
        self,
        ctx: RunContext,
        version_id: str,
        *,
        to_state: str,
        allowed_from: Sequence[str],
    ) -> str:
        with db.cursor(self._dsn) as cur:
            cur.execute(
                f"SELECT state FROM {self.t.artifact_versions} "
                "WHERE id = %s AND tenant_id = %s FOR UPDATE",
                (version_id, ctx.tenant_id),
            )
            row = cur.fetchone()
            if not row:
                return ""
            current = row["state"]
            if current not in allowed_from:
                return current
            cur.execute(
                f"""
                UPDATE {self.t.artifact_versions}
                SET state = %s,
                    approved_by = CASE WHEN %s THEN %s ELSE approved_by END,
                    approved_at = CASE WHEN %s THEN NOW() ELSE approved_at END
                WHERE id = %s AND tenant_id = %s
                """,
                (
                    to_state,
                    to_state == "approved",
                    ctx.user_id,
                    to_state == "approved",
                    version_id,
                    ctx.tenant_id,
                ),
            )
        return to_state

    def sources_of(self, ctx: RunContext, version_id: str) -> list[dict[str, Any]]:
        return db.query_all(
            f"""
            SELECT l.kind, v.id, v.artifact_key, v.kind AS artifact_kind,
                   v.version, v.state
            FROM {self.t.artifact_lineage} l
            JOIN {self.t.artifact_versions} v ON v.id = l.source_id
            WHERE l.target_id = %s AND l.tenant_id = %s
            """,
            (version_id, ctx.tenant_id),
            dsn=self._dsn,
        )

    def impact_of(
        self,
        ctx: RunContext,
        version_id: str,
        *,
        kinds: Sequence[str],
        max_depth: int,
    ) -> list[dict[str, Any]]:
        # Depth-capped recursive walk: a lineage graph accumulated over a long
        # engagement gets deep, and an uncapped recursion is how a read query
        # takes the database down.
        return db.query_all(
            f"""
            WITH RECURSIVE downstream(id, depth) AS (
                SELECT l.target_id, 1
                FROM {self.t.artifact_lineage} l
                WHERE l.source_id = %(root)s AND l.tenant_id = %(tenant)s
                  AND l.kind = ANY(%(kinds)s)
                UNION
                SELECT l.target_id, d.depth + 1
                FROM {self.t.artifact_lineage} l
                JOIN downstream d ON l.source_id = d.id
                WHERE l.tenant_id = %(tenant)s AND l.kind = ANY(%(kinds)s)
                  AND d.depth < %(max_depth)s
            )
            SELECT v.id, v.artifact_key, v.kind, v.version, v.state, v.title,
                   MIN(d.depth) AS depth
            FROM downstream d
            JOIN {self.t.artifact_versions} v ON v.id = d.id
            WHERE v.tenant_id = %(tenant)s
            GROUP BY v.id, v.artifact_key, v.kind, v.version, v.state, v.title
            ORDER BY depth, v.artifact_key
            """,
            {
                "root": version_id,
                "tenant": ctx.tenant_id,
                "kinds": list(kinds),
                "max_depth": max_depth,
            },
            dsn=self._dsn,
        )

    def create_baseline(
        self, ctx: RunContext, name: str, *, states: Sequence[str]
    ) -> int:
        baseline_id = str(uuid.uuid4())
        with db.cursor(self._dsn, dict_rows=False) as cur:
            cur.execute(
                f"""
                INSERT INTO {self.t.artifact_baselines}
                  (id, tenant_id, project_id, name, created_by, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                """,
                (baseline_id, ctx.tenant_id, ctx.project_id, name, ctx.user_id),
            )
            cur.execute(
                f"""
                INSERT INTO {self.t.artifact_baseline_members}
                  (baseline_id, tenant_id, version_id)
                SELECT %s, %s, v.id
                FROM {self.t.artifact_versions} v
                WHERE v.tenant_id = %s
                  AND (%s::uuid IS NULL OR v.project_id = %s)
                  AND v.state = ANY(%s)
                  AND NOT EXISTS (
                      SELECT 1 FROM {self.t.artifact_versions} newer
                      WHERE newer.tenant_id = v.tenant_id
                        AND newer.artifact_key = v.artifact_key
                        AND newer.version > v.version
                  )
                """,
                (
                    baseline_id,
                    ctx.tenant_id,
                    ctx.tenant_id,
                    ctx.project_id,
                    ctx.project_id,
                    list(states),
                ),
            )
            return cur.rowcount


# ── knowledge ──────────────────────────────────────────────────────────────


class PgChunkRepository(_PgBase):
    def add_chunks(self, chunks: Sequence[Mapping[str, Any]]) -> int:
        if not chunks:
            return 0
        with db.cursor(self._dsn, dict_rows=False) as cur:
            for chunk in chunks:
                embedding = chunk.get("embedding")
                cur.execute(
                    f"""
                    INSERT INTO {self.t.chunks}
                      (id, tenant_id, project_id, collection_id, document_id,
                       corpus, ordinal, title, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb)
                    """,
                    (
                        chunk.get("id") or str(uuid.uuid4()),
                        chunk.get("tenant_id"),
                        chunk.get("project_id"),
                        chunk.get("collection_id"),
                        chunk.get("document_id"),
                        chunk["corpus"],
                        chunk.get("ordinal", 0),
                        chunk.get("title"),
                        chunk["content"],
                        _vector_literal(embedding) if embedding else None,
                        json.dumps(dict(chunk.get("metadata") or {})),
                    ),
                )
        return len(chunks)

    def delete_document(self, *, tenant_id: str | None, document_id: str) -> int:
        # Re-ingestion replaces a document's chunks wholesale rather than
        # trying to diff them.
        return db.execute(
            f"DELETE FROM {self.t.chunks} "
            "WHERE document_id = %s AND (tenant_id = %s OR (%s::uuid IS NULL "
            "AND tenant_id IS NULL))",
            (document_id, tenant_id, tenant_id),
            dsn=self._dsn,
        )

    def search_vector(
        self,
        *,
        scope: Mapping[str, Any],
        embedding: Sequence[float],
        limit: int,
        max_distance: float | None,
    ) -> list[dict[str, Any]]:
        params = dict(scope["params"])
        params["qvec"] = _vector_literal(embedding)
        params["limit"] = limit
        extra = ""
        if max_distance is not None:
            params["max_distance"] = max_distance
            extra = " AND (embedding <=> %(qvec)s::vector) <= %(max_distance)s"

        return db.query_all(
            f"""
            SELECT id, content, corpus, document_id, collection_id, ordinal,
                   title, metadata,
                   (embedding <=> %(qvec)s::vector) AS distance
            FROM {self.t.chunks}
            WHERE {scope['sql']} AND embedding IS NOT NULL{extra}
            ORDER BY embedding <=> %(qvec)s::vector
            LIMIT %(limit)s
            """,
            params,
            dsn=self._dsn,
        )

    def search_text(
        self, *, scope: Mapping[str, Any], text: str, limit: int
    ) -> list[dict[str, Any]]:
        params = dict(scope["params"])
        params["q"] = text
        params["limit"] = limit
        # websearch_to_tsquery, not plainto_tsquery: it tolerates operators and
        # quoted phrases instead of erroring, and this text comes from a chat box.
        return db.query_all(
            f"""
            SELECT id, content, corpus, document_id, collection_id, ordinal,
                   title, metadata,
                   ts_rank(fts, websearch_to_tsquery('english', %(q)s)) AS text_rank
            FROM {self.t.chunks}
            WHERE {scope['sql']}
              AND fts @@ websearch_to_tsquery('english', %(q)s)
            ORDER BY text_rank DESC
            LIMIT %(limit)s
            """,
            params,
            dsn=self._dsn,
        )


# ── config ─────────────────────────────────────────────────────────────────


class PgTaskModelRepository(_PgBase):
    def resolve(
        self, *, tenant_id: str, task_key: str, role: str, workstream: str | None
    ) -> dict[str, Any] | None:
        # One query replacing the six the lifted code could issue: get_model,
        # get_system_prompt and get_fallback_chain each ran their own two-step
        # cascade with a fresh connection per step.
        return db.query_one(
            f"""
            WITH candidates AS (
                SELECT 1 AS rank, 0 AS tiebreak, '' AS name, 'task_model' AS source,
                       model_id, model_config, system_prompt,
                       fallback_model_ids, fallback_model_config
                FROM {self.t.task_models}
                WHERE tenant_id = %(tenant)s AND task_key = %(task_key)s
                UNION ALL
                SELECT 2, 0, name, 'persona_workstream',
                       model_id, model_config, system_prompt,
                       fallback_model_ids, fallback_model_config
                FROM {self.t.personas}
                WHERE tenant_id = %(tenant)s AND is_active = TRUE
                  AND role = %(role)s AND workstream = %(workstream)s
                  AND %(workstream)s IS NOT NULL
                UNION ALL
                -- `tiebreak` makes the role-only branch deterministic. A role
                -- with both a generic persona (workstream IS NULL) and one or
                -- more workstream-specific ones matched all of them here, with
                -- no ORDER BY - so `resolve("consultant")` could return the
                -- finance persona. The generic one is the correct answer for a
                -- role-only key; `name` breaks any remaining tie so the result
                -- is stable across calls rather than dependent on heap order.
                SELECT 3, (workstream IS NOT NULL)::int, name, 'persona_role',
                       model_id, model_config, system_prompt,
                       fallback_model_ids, fallback_model_config
                FROM {self.t.personas}
                WHERE tenant_id = %(tenant)s AND is_active = TRUE
                  AND role = %(role)s
            )
            SELECT * FROM candidates ORDER BY rank, tiebreak, name LIMIT 1
            """,
            {
                "tenant": tenant_id,
                "task_key": task_key,
                "role": role,
                "workstream": workstream,
            },
            dsn=self._dsn,
        )


_PERSONA_COLUMNS = (
    "id, name, role, workstream, system_prompt, model_id, model_config, is_active"
)


class PgPersonaRepository(_PgBase):
    def all(self, *, tenant_id: str) -> list[dict[str, Any]]:
        return db.query_all(
            f"SELECT {_PERSONA_COLUMNS} FROM {self.t.personas} "
            "WHERE tenant_id = %s AND is_active = TRUE ORDER BY role, workstream",
            (tenant_id,),
            dsn=self._dsn,
        )

    def by_id(self, *, tenant_id: str, persona_id: str) -> dict[str, Any] | None:
        return db.query_one(
            f"SELECT {_PERSONA_COLUMNS} FROM {self.t.personas} "
            "WHERE id = %s AND tenant_id = %s",
            (persona_id, tenant_id),
            dsn=self._dsn,
        )


# ── workflow ───────────────────────────────────────────────────────────────


class PgWorkflowRepository(_PgBase):
    """Durable workflow checkpoints.

    Satisfies both `storage.WorkflowRepository` and
    `workflow.state.CheckpointStore` - they are the same four methods, stated
    in two places so neither package has to import the other.
    """

    def create(self, record: RunRecord) -> None:
        db.execute(
            f"""
            INSERT INTO {self.t.workflow_runs}
              (id, tenant_id, project_id, workflow_name, workflow_version,
               status, principal_id, payload, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW(), NOW())
            """,
            (
                record.run_id,
                record.tenant_id,
                record.project_id,
                record.workflow_name,
                record.workflow_version,
                record.status.value,
                record.principal_id,
                json.dumps(dict(record.payload)),
            ),
            dsn=self._dsn,
        )

    def load(self, run_id: str, *, tenant_id: str) -> Any | None:
        # Imported here, not at module scope. `storage` sits *below* the eight
        # functions and must not depend upward on them - but a repository
        # legitimately needs to rebuild the aggregate it persists. A scoped
        # import keeps the layering honest and avoids an import cycle with
        # `workflow.state`, which reaches this class the same way.
        from ..workflow.state import RunRecord, RunStatus, StepRecord, StepStatus

        row = db.query_one(
            f"""
            SELECT id, tenant_id, project_id, workflow_name, workflow_version,
                   status, principal_id, payload, error, created_at, updated_at
            FROM {self.t.workflow_runs}
            WHERE id = %s AND tenant_id = %s
            """,
            (run_id, tenant_id),
            dsn=self._dsn,
        )
        if not row:
            return None

        record = RunRecord(
            run_id=str(row["id"]),
            tenant_id=str(row["tenant_id"]),
            workflow_name=row["workflow_name"],
            workflow_version=row["workflow_version"],
            status=RunStatus(row["status"]),
            project_id=str(row["project_id"]) if row["project_id"] else None,
            principal_id=row["principal_id"],
            payload=row["payload"] or {},
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

        for srow in db.query_all(
            f"""
            SELECT step_id, status, attempts, output, error, approved_by,
                   started_at, finished_at
            FROM {self.t.workflow_steps}
            WHERE run_id = %s AND tenant_id = %s
            """,
            (run_id, tenant_id),
            dsn=self._dsn,
        ):
            output = srow["output"]
            record.steps[srow["step_id"]] = StepRecord(
                step_id=srow["step_id"],
                status=StepStatus(srow["status"]),
                attempts=srow["attempts"],
                # Stored under a wrapper key so a None output round-trips as
                # None rather than as a JSON null indistinguishable from
                # "never set".
                output=(output or {}).get("value") if output else None,
                error=srow["error"],
                approved_by=srow["approved_by"],
                started_at=srow["started_at"],
                finished_at=srow["finished_at"],
            )
        return record

    def save_step(self, record: RunRecord, step_id: str) -> None:
        step = record.step(step_id)
        db.execute(
            f"""
            INSERT INTO {self.t.workflow_steps}
              (run_id, tenant_id, step_id, status, attempts, output, error,
               approved_by, started_at, finished_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
            ON CONFLICT (run_id, step_id) DO UPDATE SET
              status       = EXCLUDED.status,
              attempts     = EXCLUDED.attempts,
              output       = EXCLUDED.output,
              error        = EXCLUDED.error,
              approved_by  = EXCLUDED.approved_by,
              started_at   = EXCLUDED.started_at,
              finished_at  = EXCLUDED.finished_at
            """,
            (
                record.run_id,
                record.tenant_id,
                step_id,
                step.status.value,
                step.attempts,
                json.dumps({"value": _jsonable(step.output)}),
                step.error,
                step.approved_by,
                step.started_at,
                step.finished_at,
            ),
            dsn=self._dsn,
        )

    def save_run(self, record: RunRecord) -> None:
        record.updated_at = datetime.now(UTC)
        db.execute(
            f"""
            UPDATE {self.t.workflow_runs}
            SET status = %s, error = %s, updated_at = NOW()
            WHERE id = %s AND tenant_id = %s
            """,
            (record.status.value, record.error, record.run_id, record.tenant_id),
            dsn=self._dsn,
        )


def _jsonable(value: Any) -> Any:
    """Coerce a step output into something json.dumps accepts.

    A step returns whatever its agent returned, which is not guaranteed to be
    JSON-serializable. Failing the checkpoint write - and therefore losing a
    completed step - because an output held a dataclass would be a bad trade,
    so an unserializable value is stored as its repr with a marker.
    """
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return {"__unserializable__": repr(value)[:4000]}
