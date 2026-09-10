"""In-memory implementations of every storage port.

Not stubs. These are complete enough to run the whole package, and they exist
for three reasons:

1. **They prove the ports are honest.** An interface with a single
   implementation is an assumption dressed as an abstraction - the shape
   inevitably leaks whatever that one backend happens to do. Writing a second
   backend is what forced `search_vector` / `search_text` to be separate, and
   what surfaced that `grants_for` returning `None` versus an empty set is a
   semantic distinction rather than an accident.

2. **They make the package testable without a database.** Every behavioural
   test - version supersession, lineage traversal, RRF fusion, recall ranking -
   runs against these in milliseconds.

3. **They are a legitimate deployment choice.** A single-process consumer that
   does not need durability should not be forced to run Postgres.

Cosine similarity is computed in Python here. That is correct but O(n) per
query, so it is fine for tests and small corpora and wrong for a real one -
stated plainly rather than left for someone to discover under load.
"""
from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ..context import RunContext

__all__ = [
    "InMemoryArtifactRepository",
    "InMemoryChunkRepository",
    "InMemoryConversationRepository",
    "InMemoryMemoryRepository",
    "InMemoryPersonaRepository",
    "InMemoryTaskModelRepository",
    "InMemoryToolRepository",
]


def _now() -> datetime:
    return datetime.now(UTC)


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """1 - cosine similarity, matching pgvector's `<=>` operator."""
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - (dot / (na * nb))


# ── memory ─────────────────────────────────────────────────────────────────


class InMemoryMemoryRepository:
    #: No I/O. See `concurrency.maybe_offload`.
    blocking = False

    def __init__(self) -> None:
        self.short: list[dict[str, Any]] = []
        self.long: list[dict[str, Any]] = []

    def write_short_term(
        self,
        scope: Any,
        note_type: str,
        content: str,
        metadata: Mapping[str, Any],
        *,
        expires_at: Any = None,
    ) -> str:

        note_id = str(uuid.uuid4())
        self.short.append(
            {
                "id": note_id,
                "tenant_id": scope.tenant_id,
                "job_id": scope.require_job(),
                "persona_id": scope.persona_id,
                "user_id": scope.user_id,
                "project_id": scope.project_id,
                "note_type": note_type,
                "content": content,
                "metadata": dict(metadata or {}),
                "created_at": _now(),
                "expires_at": expires_at,
            }
        )
        return note_id

    def _short_matches(self, scope: Any) -> list[dict[str, Any]]:
        now = _now()
        job = scope.require_job()
        return [
            n
            for n in self.short
            if n["tenant_id"] == scope.tenant_id
            and n["job_id"] == job
            and n["persona_id"] == scope.persona_id
            and (n.get("expires_at") is None or n["expires_at"] > now)
        ]

    def read_short_term(self, scope: Any) -> list[dict[str, Any]]:
        rows = self._short_matches(scope)
        rows.sort(key=lambda n: n["created_at"])
        return [dict(n) for n in rows]

    def clear_short_term(self, scope: Any) -> int:
        doomed = {n["id"] for n in self._short_matches(scope)}
        before = len(self.short)
        self.short = [n for n in self.short if n["id"] not in doomed]
        return before - len(self.short)

    def prune_short_term(self, *, tenant_id: str) -> int:
        now = _now()
        before = len(self.short)
        self.short = [
            n for n in self.short
            if not (n["tenant_id"] == tenant_id
                    and n.get("expires_at") is not None
                    and n["expires_at"] < now)
        ]
        return before - len(self.short)

    def prune_expired(self, *, tenant_id: str, retain_days: int) -> int:
        from datetime import timedelta

        cutoff = _now() - timedelta(days=retain_days)
        before = len(self.long)
        self.long = [
            m for m in self.long
            if not (m["tenant_id"] == tenant_id and m["created_at"] < cutoff)
        ]
        return before - len(self.long)

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
        self.long.append(
            {
                "id": mem_id,
                "tenant_id": scope.tenant_id,
                "user_id": scope.user_id,
                "project_id": scope.project_id,
                "persona_id": scope.persona_id,
                "memory_type": memory_type,
                "content": content,
                "embedding": list(embedding) if embedding else None,
                "importance": importance,
                "source": source,
                "access_count": 0,
                "metadata": dict(metadata or {}),
                "created_at": _now(),
            }
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
        query_text: str | None = None,
    ) -> list[dict[str, Any]]:
        # Tenant predicate applied identically to the SQL arm — loose filtering
        # here would hide isolation bugs the Postgres backend catches.
        rows = [
            m
            for m in self.long
            if m["tenant_id"] == scope.tenant_id
            and m["user_id"] == scope.user_id
            and m["project_id"] == scope.project_id
            and m["persona_id"] == scope.persona_id
            and m["importance"] >= min_importance
        ]

        if query_embedding is not None and query_text:
            # RRF: vector arm + word-overlap text arm
            words = set(query_text.lower().split())
            k = 60

            vec_scored = []
            for m in rows:
                if not m["embedding"]:
                    continue
                d = _cosine_distance(query_embedding, m["embedding"])
                vec_scored.append((m["id"], d))
            vec_scored.sort(key=lambda x: x[1])
            vec_rank = {mid: i + 1 for i, (mid, _) in enumerate(vec_scored)}

            txt_scored = []
            for m in rows:
                overlap = len(words & set(m["content"].lower().split()))
                if overlap:
                    txt_scored.append((m["id"], overlap))
            txt_scored.sort(key=lambda x: -x[1])
            txt_rank = {mid: i + 1 for i, (mid, _) in enumerate(txt_scored)}

            all_ids = {m["id"] for m in rows if m["id"] in vec_rank or m["id"] in txt_rank}
            rrf: list[tuple[Any, float]] = []
            for mid in all_ids:
                score = 0.0
                if mid in vec_rank:
                    score += 1.0 / (k + vec_rank[mid])
                if mid in txt_rank:
                    score += 1.0 / (k + txt_rank[mid])
                rrf.append((mid, score))
            rrf.sort(key=lambda x: -x[1])

            by_id = {m["id"]: m for m in rows}
            out = []
            for mid, score in rrf[:limit]:
                row = dict(by_id[mid])
                row["distance"] = None
                row["score"] = score
                out.append(row)
            return out

        if query_embedding is not None:
            scored = []
            for m in rows:
                if not m["embedding"]:
                    continue
                distance = _cosine_distance(query_embedding, m["embedding"])
                score = (
                    (1.0 - distance) * (1.0 - importance_weight)
                    + m["importance"] * importance_weight
                )
                out = dict(m)
                out["distance"] = distance
                out["score"] = score
                scored.append(out)
            scored.sort(key=lambda m: -m["score"])
            return scored[:limit]

        rows = sorted(rows, key=lambda m: (-m["importance"], m["created_at"]))
        out_rows = []
        for m in rows[:limit]:
            row = dict(m)
            row["distance"] = None
            row["score"] = m["importance"]
            out_rows.append(row)
        return out_rows

    def record_access(self, scope: Any, memory_ids: Sequence[str]) -> None:
        wanted = set(memory_ids)
        for m in self.long:
            if m["id"] in wanted and m["tenant_id"] == scope.tenant_id:
                m["access_count"] += 1
                m["last_accessed_at"] = _now()


class InMemoryConversationRepository:
    #: No I/O. See `concurrency.maybe_offload`.
    blocking = False

    def __init__(self) -> None:
        # Keyed by (tenant, session) - matching the composite unique index the
        # Postgres schema needs so session ids cannot collide across tenants.
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

    def save(
        self,
        ctx: RunContext,
        messages: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any] | None,
    ) -> None:
        self.rows[(ctx.tenant_id, ctx.session_id or "")] = {
            "user_id": ctx.user_id,
            "project_id": ctx.project_id,
            "persona_id": ctx.persona_id,
            "messages": [dict(m) for m in messages],
            "context": dict(context) if context is not None else None,
            "updated_at": _now(),
        }

    def load(self, ctx: RunContext) -> list[dict[str, Any]]:
        row = self.rows.get((ctx.tenant_id, ctx.session_id or ""))
        if not row or row["user_id"] != ctx.user_id:
            return []
        return [dict(m) for m in row["messages"]]

    def delete(self, ctx: RunContext) -> int:
        key = (ctx.tenant_id, ctx.session_id or "")
        row = self.rows.get(key)
        if not row or row["user_id"] != ctx.user_id:
            return 0
        del self.rows[key]
        return 1


# ── tools ──────────────────────────────────────────────────────────────────


class InMemoryToolRepository:
    #: No I/O. See `concurrency.maybe_offload`.
    blocking = False

    def __init__(self) -> None:
        self.catalog: dict[tuple[str, str], dict[str, Any]] = {}
        self.grants: dict[tuple[str, str], set[str]] = {}
        self.invocations: list[dict[str, Any]] = []

    def upsert_catalog(
        self, tenant_id: str, tools: Sequence[Mapping[str, Any]]
    ) -> int:
        seen = set()
        for tool in tools:
            name = tool.get("name")
            if not name:
                continue
            seen.add(name)
            self.catalog[(tenant_id, name)] = {
                **dict(tool),
                "is_active": True,
                "last_seen_at": _now(),
            }
        for (tid, name), row in self.catalog.items():
            if tid == tenant_id and name not in seen:
                row["is_active"] = False
        return len(seen)

    def grant(self, *, tenant_id: str, persona_id: str, tool_names: Sequence[str]) -> None:
        self.grants.setdefault((tenant_id, persona_id), set()).update(tool_names)

    def grants_for(
        self, *, tenant_id: str, persona_id: str | None
    ) -> frozenset[str] | None:
        if not persona_id:
            return None
        granted = self.grants.get((tenant_id, persona_id), set())
        active = {
            name
            for (tid, name), row in self.catalog.items()
            if tid == tenant_id and row.get("is_active")
        }
        # Intersect with the active catalog, as the SQL join does.
        return frozenset(granted & active) if active else frozenset()

    def record_invocation(self, row: Mapping[str, Any]) -> None:
        self.invocations.append({**dict(row), "created_at": _now()})

    def prune_invocations(self, *, tenant_id: str, retain_days: int) -> int:
        cutoff = _now().timestamp() - retain_days * 86400
        before = len(self.invocations)
        self.invocations = [
            i
            for i in self.invocations
            if not (
                i.get("tenant_id") == tenant_id
                and i["created_at"].timestamp() < cutoff
            )
        ]
        return before - len(self.invocations)


# ── artifacts ──────────────────────────────────────────────────────────────


class InMemoryArtifactRepository:
    #: No I/O. See `concurrency.maybe_offload`.
    blocking = False

    def __init__(self) -> None:
        self.versions: list[dict[str, Any]] = []
        self.lineage: list[dict[str, Any]] = []
        self.baselines: dict[str, list[str]] = {}

    def _heads(self, tenant_id: str, artifact_key: str) -> list[dict[str, Any]]:
        rows = [
            v
            for v in self.versions
            if v["tenant_id"] == tenant_id and v["artifact_key"] == artifact_key
        ]
        rows.sort(key=lambda v: -v["version"])
        return rows

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
        existing = self._heads(ctx.tenant_id, artifact_key)
        head = existing[0] if existing else None

        if head and head["content_hash"] == content_hash:
            return {"duplicate": True, "version": head["version"]}

        next_version = (head["version"] + 1) if head else 1
        supersedes_id = head["id"] if head else None
        if head:
            head["state"] = "superseded"

        new_id = str(uuid.uuid4())
        self.versions.append(
            {
                "id": new_id,
                "tenant_id": ctx.tenant_id,
                "project_id": ctx.project_id,
                "artifact_key": artifact_key,
                "kind": kind,
                "version": next_version,
                "title": title,
                "content": content,
                "content_hash": content_hash,
                "state": state,
                "supersedes_id": supersedes_id,
                "created_by": ctx.user_id,
                "approved_by": None,
                "run_id": ctx.run_id,
                "metadata": dict(metadata or {}),
                "created_at": _now(),
            }
        )

        edges = list(lineage)
        if supersedes_id:
            edges.append((supersedes_id, "revises"))
        for source_id, link in edges:
            if not any(
                e["target_id"] == new_id
                and e["source_id"] == source_id
                and e["kind"] == link
                for e in self.lineage
            ):
                self.lineage.append(
                    {
                        "tenant_id": ctx.tenant_id,
                        "target_id": new_id,
                        "source_id": source_id,
                        "kind": link,
                    }
                )

        return {
            "duplicate": False,
            "id": new_id,
            "version": next_version,
            "supersedes_id": supersedes_id,
            "created_at": _now(),
        }

    def head(self, ctx: RunContext, artifact_key: str) -> dict[str, Any] | None:
        rows = self._heads(ctx.tenant_id, artifact_key)
        return dict(rows[0]) if rows else None

    def history(self, ctx: RunContext, artifact_key: str) -> list[dict[str, Any]]:
        return [dict(v) for v in self._heads(ctx.tenant_id, artifact_key)]

    def _by_id(self, tenant_id: str, version_id: str) -> dict[str, Any] | None:
        for v in self.versions:
            if v["id"] == version_id and v["tenant_id"] == tenant_id:
                return v
        return None

    def transition(
        self,
        ctx: RunContext,
        version_id: str,
        *,
        to_state: str,
        allowed_from: Sequence[str],
    ) -> str:
        row = self._by_id(ctx.tenant_id, version_id)
        if row is None:
            return ""
        if row["state"] not in allowed_from:
            return row["state"]
        row["state"] = to_state
        if to_state == "approved":
            row["approved_by"] = ctx.user_id
            row["approved_at"] = _now()
        return to_state

    def sources_of(self, ctx: RunContext, version_id: str) -> list[dict[str, Any]]:
        out = []
        for edge in self.lineage:
            if edge["target_id"] != version_id or edge["tenant_id"] != ctx.tenant_id:
                continue
            src = self._by_id(ctx.tenant_id, edge["source_id"])
            if src:
                out.append(
                    {
                        "kind": edge["kind"],
                        "id": src["id"],
                        "artifact_key": src["artifact_key"],
                        "artifact_kind": src["kind"],
                        "version": src["version"],
                        "state": src["state"],
                    }
                )
        return out

    def impact_of(
        self,
        ctx: RunContext,
        version_id: str,
        *,
        kinds: Sequence[str],
        max_depth: int,
    ) -> list[dict[str, Any]]:
        """BFS downstream, mirroring the recursive CTE including its depth cap."""
        wanted = set(kinds)
        depths: dict[str, int] = {}
        frontier = [(version_id, 0)]

        while frontier:
            current, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for edge in self.lineage:
                if (
                    edge["source_id"] != current
                    or edge["tenant_id"] != ctx.tenant_id
                    or edge["kind"] not in wanted
                ):
                    continue
                target = edge["target_id"]
                new_depth = depth + 1
                if target not in depths or new_depth < depths[target]:
                    depths[target] = new_depth
                    frontier.append((target, new_depth))

        out = []
        for vid, depth in depths.items():
            row = self._by_id(ctx.tenant_id, vid)
            if row:
                out.append(
                    {
                        "id": row["id"],
                        "artifact_key": row["artifact_key"],
                        "kind": row["kind"],
                        "version": row["version"],
                        "state": row["state"],
                        "title": row["title"],
                        "depth": depth,
                    }
                )
        out.sort(key=lambda r: (r["depth"], r["artifact_key"]))
        return out

    def create_baseline(
        self, ctx: RunContext, name: str, *, states: Sequence[str]
    ) -> int:
        allowed = set(states)
        captured = []
        seen_keys = set()
        for v in sorted(self.versions, key=lambda v: -v["version"]):
            if v["tenant_id"] != ctx.tenant_id:
                continue
            if ctx.project_id and v["project_id"] != ctx.project_id:
                continue
            if v["artifact_key"] in seen_keys:
                continue
            seen_keys.add(v["artifact_key"])
            if v["state"] in allowed:
                captured.append(v["id"])
        self.baselines[name] = captured
        return len(captured)


# ── knowledge ──────────────────────────────────────────────────────────────


class InMemoryChunkRepository:
    """Chunk store with Python-side cosine similarity.

    O(n) per query by construction. Correct, and unsuitable for a real corpus -
    use `PgChunkRepository` with an HNSW index for that.
    """
    #: No I/O. See `concurrency.maybe_offload`.
    blocking = False


    def __init__(self) -> None:
        self.chunks: list[dict[str, Any]] = []

    def add_chunks(self, chunks: Sequence[Mapping[str, Any]]) -> int:
        for chunk in chunks:
            row = dict(chunk)
            row.setdefault("id", str(uuid.uuid4()))
            row.setdefault("ordinal", 0)
            row.setdefault("metadata", {})
            self.chunks.append(row)
        return len(chunks)

    def delete_document(self, *, tenant_id: str | None, document_id: str) -> int:
        before = len(self.chunks)
        self.chunks = [
            c
            for c in self.chunks
            if not (
                c.get("document_id") == document_id
                and c.get("tenant_id") == tenant_id
            )
        ]
        return before - len(self.chunks)

    def _in_scope(self, chunk: Mapping[str, Any], scope: Mapping[str, Any]) -> bool:
        """Evaluate the same corpus rules the SQL encodes."""
        params = scope["params"]
        for corpus in scope["corpora"]:
            if corpus == "client_project":
                if (
                    chunk.get("corpus") == "client_project"
                    and chunk.get("tenant_id") == params.get("tenant")
                    and chunk.get("project_id") == params.get("project")
                ):
                    break
            elif corpus == "org_assets":
                if (
                    chunk.get("corpus") == "org_assets"
                    and chunk.get("tenant_id") == params.get("tenant")
                ):
                    break
            elif corpus == "public_domain" and chunk.get("corpus") == "public_domain":
                break
        else:
            return False

        collections = params.get("collections")
        return not (collections and chunk.get("collection_id") not in collections)

    def search_vector(
        self,
        *,
        scope: Mapping[str, Any],
        embedding: Sequence[float],
        limit: int,
        max_distance: float | None,
    ) -> list[dict[str, Any]]:
        hits = []
        for chunk in self.chunks:
            if not chunk.get("embedding") or not self._in_scope(chunk, scope):
                continue
            distance = _cosine_distance(embedding, chunk["embedding"])
            if max_distance is not None and distance > max_distance:
                continue
            hits.append({**chunk, "distance": distance})
        hits.sort(key=lambda c: c["distance"])
        return hits[:limit]

    def search_text(
        self, *, scope: Mapping[str, Any], text: str, limit: int
    ) -> list[dict[str, Any]]:
        """Naive term overlap standing in for `ts_rank`.

        Not a tsvector: no stemming, no stop words, no weighting. It ranks in
        roughly the right order for a test and should not be mistaken for
        Postgres full-text search.
        """
        terms = [t.lower() for t in text.split() if len(t) > 2]
        if not terms:
            return []
        hits = []
        for chunk in self.chunks:
            if not self._in_scope(chunk, scope):
                continue
            body = str(chunk.get("content", "")).lower()
            matched = sum(1 for t in terms if t in body)
            if matched:
                hits.append({**chunk, "text_rank": matched / len(terms)})
        hits.sort(key=lambda c: -c["text_rank"])
        return hits[:limit]


# ── config ─────────────────────────────────────────────────────────────────


class InMemoryTaskModelRepository:
    #: No I/O. See `concurrency.maybe_offload`.
    blocking = False

    def __init__(self, rows: Mapping[tuple[str, str], Mapping[str, Any]] | None = None):
        self.rows: dict[tuple[str, str], dict[str, Any]] = {
            k: dict(v) for k, v in (rows or {}).items()
        }

    def set(self, tenant_id: str, task_key: str, **row: Any) -> None:
        self.rows[(tenant_id, task_key)] = dict(row)

    def resolve(
        self, *, tenant_id: str, task_key: str, role: str, workstream: str | None
    ) -> dict[str, Any] | None:
        row = self.rows.get((tenant_id, task_key))
        if row is None:
            return None
        return {
            "source": "inmemory",
            "model_id": row.get("model_id"),
            "model_config": row.get("model_config") or {},
            "system_prompt": row.get("system_prompt"),
            "fallback_model_ids": row.get("fallback_model_ids") or [],
            "fallback_model_config": row.get("fallback_model_config") or {},
        }


class InMemoryPersonaRepository:
    #: No I/O. See `concurrency.maybe_offload`.
    blocking = False

    def __init__(self, personas: Sequence[Mapping[str, Any]] = ()):
        self.personas = [dict(p) for p in personas]

    def all(self, *, tenant_id: str) -> list[dict[str, Any]]:
        return [
            dict(p)
            for p in self.personas
            if p.get("tenant_id") == tenant_id and p.get("is_active", True)
        ]

    def by_id(self, *, tenant_id: str, persona_id: str) -> dict[str, Any] | None:
        for p in self.personas:
            if p.get("id") == persona_id and p.get("tenant_id") == tenant_id:
                return dict(p)
        return None
