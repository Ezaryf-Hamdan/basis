"""Integration tests against a live Postgres with pgvector.

These exercise the SQL that unit tests cannot: pgvector cosine distance,
generated tsvector columns, `FOR UPDATE` on version heads, the recursive
lineage CTE, `ON CONFLICT` upserts, and `set_config` session variables.

Run with:

    INTEGRATION_TESTS=1 python -m pytest tests/test_integration_postgres.py

Apply `migrations/0000_basis_baseline.sql` then `0002_basis_platform.sql` to the
target database first.
"""
from __future__ import annotations

import uuid

import pytest

from basis import bind, db
from basis.artifacts import ArtifactState, ArtifactStore, LineageKind
from basis.artifacts.store import ArtifactConflict
from basis.embeddings import HashEmbedder
from basis.knowledge import Corpus, HybridRetriever, RetrievalQuery
from basis.memory import MemoryScope, MemoryService
from basis.memory.conversation import ConversationStore
from basis.models.resolver import clear_cache, resolve
from basis.personas import PersonaStore
from basis.storage.postgres import (
    PgArtifactRepository,
    PgChunkRepository,
    PgConversationRepository,
    PgMemoryRepository,
    PgPersonaRepository,
    PgTaskModelRepository,
    PgToolRepository,
    PgWorkflowRepository,
)
from basis.tools import ToolPolicy, ToolRegistry
from basis.workflow import (
    RunStatus,
    Step,
    StepKind,
    StepStatus,
    Workflow,
    WorkflowEngine,
)
from basis.workflow.state import new_run_record

pytestmark = pytest.mark.integration

#: The vector column is VECTOR(1024); the embedder must match or the cast fails.
EMBED_DIM = 1024


def _scope(ctx) -> MemoryScope:
    return MemoryScope.from_context(ctx)


# ═══════════════════════════════════════════════════════════════════════════
# db.py — pooling and session variables
# ═══════════════════════════════════════════════════════════════════════════


def test_pool_reuses_connections(pg):
    """The whole point of the pool: N queries must not mean N backends."""
    before = db.query_one(
        "SELECT count(*) AS n FROM pg_stat_activity WHERE datname = current_database()",
        dsn=pg,
    )["n"]
    for _ in range(20):
        db.query_one("SELECT 1 AS x", dsn=pg)
    after = db.query_one(
        "SELECT count(*) AS n FROM pg_stat_activity WHERE datname = current_database()",
        dsn=pg,
    )["n"]
    assert after - before <= 2


def test_session_vars_are_applied_and_transaction_local(pg):
    """`set_config(..., is_local => true)` is what makes basis work against an
    RLS-enforced schema like ai-core's."""
    with db.session_vars(**{"app.workspace_id": "ws-42"}):
        row = db.query_one(
            "SELECT current_setting('app.workspace_id', true) AS v", dsn=pg
        )
        assert row["v"] == "ws-42"

    # Outside the block the setting must be gone, or it would leak to the next
    # borrower of the pooled connection.
    row = db.query_one(
        "SELECT current_setting('app.workspace_id', true) AS v", dsn=pg
    )
    assert row["v"] in (None, "")


def test_session_vars_nest(pg):
    with db.session_vars(**{"app.a": "1"}):
        with db.session_vars(**{"app.b": "2"}):
            row = db.query_one(
                "SELECT current_setting('app.a', true) AS a, "
                "current_setting('app.b', true) AS b",
                dsn=pg,
            )
            assert (row["a"], row["b"]) == ("1", "2")


def test_transaction_rolls_back_on_error(pg):
    key = str(uuid.uuid4())
    with pytest.raises(RuntimeError):
        with db.cursor(pg, dict_rows=False) as cur:
            cur.execute(
                "INSERT INTO agent_task_models (tenant_id, task_key, model_id) "
                "VALUES (%s, %s, %s)",
                ("aaaaaaaa-0000-0000-0000-000000000001", key, "m"),
            )
            raise RuntimeError("boom")
    assert db.query_one(
        "SELECT 1 AS x FROM agent_task_models WHERE task_key = %s", (key,), dsn=pg
    ) is None


# ═══════════════════════════════════════════════════════════════════════════
# MEMORY — real pgvector recall
# ═══════════════════════════════════════════════════════════════════════════


def test_short_term_roundtrip(pg, ctx):
    svc = MemoryService(PgMemoryRepository(dsn=pg))
    scope = _scope(ctx)
    svc.write_short_term(scope, "observation", "two ledgers in scope")
    svc.write_short_term(scope, "risk", "FX cutover unclear")
    rows = svc.read_short_term(scope)
    assert [r["note_type"] for r in rows] == ["observation", "risk"]
    assert svc.clear_short_term(scope) == 2


def test_long_term_vector_recall_ranks_by_similarity(pg, ctx):
    """The behaviour the source stored embeddings for and never used."""
    embedder = HashEmbedder(dimensions=EMBED_DIM)
    svc = MemoryService(PgMemoryRepository(dsn=pg))
    scope = _scope(ctx)

    target = "OB09 holds the exchange rate configuration"
    noise = "plant maintenance is explicitly out of scope"
    # The relevant memory is deliberately LOW importance and the irrelevant one
    # HIGH, so importance-only ordering would get this backwards.
    svc.write_long_term(
        scope, "fact", target, importance=0.2, embedding=embedder.embed(target)
    )
    svc.write_long_term(
        scope, "fact", noise, importance=0.95, embedding=embedder.embed(noise)
    )

    rows = svc.recall(scope, query_embedding=embedder.embed(target), limit=5)
    assert rows[0]["content"] == target
    assert rows[0]["distance"] < rows[1]["distance"]
    assert rows[0]["score"] > rows[1]["score"]


def test_recall_without_embedding_falls_back_to_importance(pg, ctx):
    svc = MemoryService(PgMemoryRepository(dsn=pg))
    scope = _scope(ctx)
    svc.write_long_term(scope, "fact", "low", importance=0.1)
    svc.write_long_term(scope, "fact", "high", importance=0.9)
    rows = svc.recall(scope, limit=5)
    assert [r["content"] for r in rows] == ["high", "low"]
    assert rows[0]["distance"] is None


def test_memory_written_without_embedding_is_still_stored(pg, ctx):
    """A throttled embedding endpoint must not lose the memory."""
    svc = MemoryService(PgMemoryRepository(dsn=pg))
    scope = _scope(ctx)
    svc.write_long_term(scope, "fact", "no vector", importance=0.5, embedding=None)
    assert len(svc.recall(scope)) == 1


def test_recall_is_tenant_isolated_in_sql(pg, ctx, other_ctx):
    embedder = HashEmbedder(dimensions=EMBED_DIM)
    svc = MemoryService(PgMemoryRepository(dsn=pg))
    text = "shared phrasing across both tenants"

    svc.write_long_term(
        _scope(ctx), "fact", text, importance=0.9, embedding=embedder.embed(text)
    )
    svc.write_long_term(
        _scope(other_ctx), "fact", text, importance=0.9, embedding=embedder.embed(text)
    )

    rows = svc.recall(_scope(ctx), query_embedding=embedder.embed(text), limit=10)
    assert len(rows) == 1


def test_access_count_increments_in_one_statement(pg, ctx):
    svc = MemoryService(PgMemoryRepository(dsn=pg))
    scope = _scope(ctx)
    svc.write_long_term(scope, "fact", "a", importance=0.5)
    svc.write_long_term(scope, "fact", "b", importance=0.5)
    svc.recall(scope)
    rows = db.query_all(
        "SELECT access_count, last_accessed_at FROM agent_memories", dsn=pg
    )
    assert all(r["access_count"] == 1 for r in rows)
    assert all(r["last_accessed_at"] is not None for r in rows)


# ═══════════════════════════════════════════════════════════════════════════
# CONVERSATIONS — upsert and the composite key
# ═══════════════════════════════════════════════════════════════════════════


def test_conversation_upsert_and_tenant_key(pg, ctx, other_ctx):
    store = ConversationStore(PgConversationRepository(dsn=pg))

    with bind(ctx):
        store.save([{"role": "user", "content": "one"}])
        store.save([{"role": "user", "content": "two"}])  # upsert, not insert
        assert store.load() == [{"role": "user", "content": "two"}]

    # Same session_id, different tenant. Would overwrite under the source's
    # global UNIQUE(session_id).
    with bind(other_ctx):
        assert store.load() == []
        store.save([{"role": "user", "content": "other tenant"}])

    with bind(ctx):
        assert store.load() == [{"role": "user", "content": "two"}]

    assert db.query_one("SELECT count(*) AS n FROM agent_conversations", dsn=pg)["n"] == 2


def test_conversation_none_context_is_sql_null(pg, ctx):
    store = ConversationStore(PgConversationRepository(dsn=pg))
    with bind(ctx):
        store.save([{"role": "user", "content": "x"}], context=None)
    row = db.query_one("SELECT context FROM agent_conversations", dsn=pg)
    # Not the JSON string "null", which is what json.dumps(None) produced.
    assert row["context"] is None


def test_conversation_delete(pg, ctx):
    store = ConversationStore(PgConversationRepository(dsn=pg))
    with bind(ctx):
        store.save([{"role": "user", "content": "x"}])
        assert store.delete() == 1
        assert store.load() == []


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — catalog sync, grants, audit
# ═══════════════════════════════════════════════════════════════════════════


def test_catalog_sync_upserts_then_deactivates(pg, ctx):
    repo = PgToolRepository(dsn=pg)
    registry = ToolRegistry(ToolPolicy(), repo=repo)

    assert registry.sync_from_mcp(
        [{"name": "get_a"}, {"name": "get_b"}, {"name": "delete_c"}],
        tenant_id=ctx.tenant_id,
    ) == 3

    # Re-sync without delete_c: it must be deactivated, not deleted.
    registry.sync_from_mcp(
        [{"name": "get_a"}, {"name": "get_b"}], tenant_id=ctx.tenant_id
    )
    rows = {
        r["tool_name"]: r["is_active"]
        for r in db.query_all("SELECT tool_name, is_active FROM agent_tools", dsn=pg)
    }
    assert rows == {"get_a": True, "get_b": True, "delete_c": False}


def test_catalog_sync_persists_effect(pg, ctx):
    repo = PgToolRepository(dsn=pg)
    ToolRegistry(ToolPolicy(), repo=repo).sync_from_mcp(
        [{"name": "get_a"}, {"name": "delete_b"}], tenant_id=ctx.tenant_id
    )
    rows = {
        r["tool_name"]: r["effect"]
        for r in db.query_all("SELECT tool_name, effect FROM agent_tools", dsn=pg)
    }
    assert rows == {"get_a": "read", "delete_b": "write"}


def test_grants_join_filters_inactive_tools(pg, ctx):
    repo = PgToolRepository(dsn=pg)
    registry = ToolRegistry(ToolPolicy(), repo=repo)
    registry.sync_from_mcp(
        [{"name": "get_a"}, {"name": "get_b"}], tenant_id=ctx.tenant_id
    )

    db.execute(
        "INSERT INTO agent_personas (id, tenant_id, name, role) "
        "VALUES (%s, %s, 'analyst', 'analyst')",
        (ctx.persona_id, ctx.tenant_id),
        dsn=pg,
    )
    db.execute(
        """
        INSERT INTO agent_persona_tools (tenant_id, persona_id, tool_id)
        SELECT %s, %s, id FROM agent_tools WHERE tool_name = 'get_a'
        """,
        (ctx.tenant_id, ctx.persona_id),
        dsn=pg,
    )

    assert repo.grants_for(tenant_id=ctx.tenant_id, persona_id=ctx.persona_id) == frozenset(
        {"get_a"}
    )
    # Deactivating the tool removes the grant without touching the grant row.
    db.execute("UPDATE agent_tools SET is_active = FALSE", dsn=pg)
    assert repo.grants_for(tenant_id=ctx.tenant_id, persona_id=ctx.persona_id) == frozenset()


def test_no_persona_means_no_narrowing(pg, ctx):
    """None and an empty set mean different things - the fail-closed detail."""
    repo = PgToolRepository(dsn=pg)
    assert repo.grants_for(tenant_id=ctx.tenant_id, persona_id=None) is None


def test_invocation_audit_written_and_pruned(pg, ctx):
    repo = PgToolRepository(dsn=pg)
    registry = ToolRegistry(ToolPolicy(), repo=repo, grants_provider=_NoGrants())

    with bind(ctx):
        registry.invoke("get_thing", lambda: "ok")
        from basis.errors import ToolDenied

        with pytest.raises(ToolDenied):
            registry.check("delete_thing")

    rows = db.query_all(
        "SELECT tool_name, allowed, effect, denied_reason FROM agent_tool_invocations "
        "ORDER BY created_at",
        dsn=pg,
    )
    assert [(r["tool_name"], r["allowed"]) for r in rows] == [
        ("get_thing", True),
        ("delete_thing", False),
    ]
    assert rows[1]["denied_reason"]

    # Nothing is old enough to prune yet.
    assert repo.prune_invocations(tenant_id=ctx.tenant_id, retain_days=1) == 0
    # Backdate and prune.
    db.execute(
        "UPDATE agent_tool_invocations SET created_at = NOW() - INTERVAL '10 days'",
        dsn=pg,
    )
    assert repo.prune_invocations(tenant_id=ctx.tenant_id, retain_days=1) == 2


class _NoGrants:
    """Bypasses persona grants so the policy alone decides."""

    def grants_for(self, *, tenant_id, persona_id):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# ARTIFACTS — FOR UPDATE, state machine, recursive lineage CTE
# ═══════════════════════════════════════════════════════════════════════════


def test_version_supersession(pg, ctx):
    store = ArtifactStore(PgArtifactRepository(dsn=pg))
    with bind(ctx):
        v1 = store.create_version("FSD-1", "fsd", {"body": "first"})
        v2 = store.create_version("FSD-1", "fsd", {"body": "second"})

    assert (v1.version, v2.version) == (1, 2)
    assert v2.supersedes_id == v1.id
    states = {
        r["version"]: r["state"]
        for r in db.query_all(
            "SELECT version, state FROM basis_artifact_versions", dsn=pg
        )
    }
    assert states == {1: "superseded", 2: "draft"}


def test_duplicate_content_conflicts(pg, ctx):
    store = ArtifactStore(PgArtifactRepository(dsn=pg))
    with bind(ctx):
        store.create_version("FSD-1", "fsd", {"body": "same"})
        with pytest.raises(ArtifactConflict):
            store.create_version("FSD-1", "fsd", {"body": "same"})
    assert db.query_one(
        "SELECT count(*) AS n FROM basis_artifact_versions", dsn=pg
    )["n"] == 1


def test_unique_constraint_blocks_duplicate_version_numbers(pg, ctx):
    """The backstop behind FOR UPDATE."""
    store = ArtifactStore(PgArtifactRepository(dsn=pg))
    with bind(ctx):
        store.create_version("A", "doc", {"n": 1})
    from psycopg.errors import UniqueViolation

    with pytest.raises(UniqueViolation):
        db.execute(
            """
            INSERT INTO basis_artifact_versions
              (id, tenant_id, artifact_key, kind, version, content_hash, state)
            VALUES (%s, %s, 'A', 'doc', 1, 'deadbeef', 'draft')
            """,
            (str(uuid.uuid4()), ctx.tenant_id),
            dsn=pg,
        )


def test_state_machine_enforced_against_db(pg, ctx):
    store = ArtifactStore(PgArtifactRepository(dsn=pg))
    with bind(ctx):
        v = store.create_version("FSD-1", "fsd", {"n": 1})
        with pytest.raises(ArtifactConflict):
            store.transition(v.id, ArtifactState.APPROVED)  # skips review
        store.transition(v.id, ArtifactState.IN_REVIEW)
        store.transition(v.id, ArtifactState.APPROVED)

    row = db.query_one(
        "SELECT state, approved_by, approved_at FROM basis_artifact_versions", dsn=pg
    )
    assert row["state"] == "approved"
    assert row["approved_by"] == ctx.user_id
    assert row["approved_at"] is not None


def test_drift_detection_against_db(pg, ctx):
    store = ArtifactStore(PgArtifactRepository(dsn=pg))
    with bind(ctx):
        v = store.create_version("REQ-1", "requirement", {"text": "original"})
        store.transition(v.id, ArtifactState.IN_REVIEW)
        store.transition(v.id, ArtifactState.APPROVED)

        assert not store.check_drift("REQ-1", {"text": "original"}).changed
        drift = store.check_drift("REQ-1", {"text": "edited externally"})
        assert drift.changed and drift.invalidates_approval


def test_recursive_lineage_cte(pg, ctx):
    """The impact-analysis query, which has never executed before."""
    store = ArtifactStore(PgArtifactRepository(dsn=pg))
    with bind(ctx):
        req = store.create_version("REQ-1", "requirement", {"n": 1})
        fsd = store.create_version(
            "FSD-1", "fsd", {"n": 2},
            derived_from=[(req.id, LineageKind.DERIVED_FROM)],
        )
        obj = store.create_version(
            "OBJ-1", "build_object", {"n": 3},
            derived_from=[(fsd.id, LineageKind.DERIVED_FROM)],
        )
        test = store.create_version(
            "TEST-1", "test_script", {"n": 4},
            derived_from=[(obj.id, LineageKind.DERIVED_FROM)],
        )

        impact = store.impact_of(req.id)
        assert [(r["artifact_key"], r["depth"]) for r in impact] == [
            ("FSD-1", 1),
            ("OBJ-1", 2),
            ("TEST-1", 3),
        ]
        assert {r["id"] for r in impact} == {fsd.id, obj.id, test.id}

        # Depth cap must actually bind.
        assert len(store.impact_of(req.id, max_depth=2)) == 2

        # Provenance in the other direction.
        assert [s["artifact_key"] for s in store.sources_of(fsd.id)] == ["REQ-1"]


def test_lineage_diamond_reports_shortest_depth(pg, ctx):
    """Two paths to the same node: MIN(depth) must win, and DISTINCT must not
    return it twice."""
    store = ArtifactStore(PgArtifactRepository(dsn=pg))
    with bind(ctx):
        root = store.create_version("ROOT", "doc", {"n": 0})
        left = store.create_version(
            "LEFT", "doc", {"n": 1}, derived_from=[(root.id, LineageKind.DERIVED_FROM)]
        )
        sink = store.create_version(
            "SINK", "doc", {"n": 3},
            derived_from=[
                (root.id, LineageKind.DERIVED_FROM),   # depth 1
                (left.id, LineageKind.DERIVED_FROM),   # depth 2
            ],
        )
        impact = store.impact_of(root.id)

    rows = [r for r in impact if r["artifact_key"] == "SINK"]
    assert len(rows) == 1
    assert rows[0]["depth"] == 1
    assert sink.id == rows[0]["id"]


def test_impact_excludes_revisions(pg, ctx):
    store = ArtifactStore(PgArtifactRepository(dsn=pg))
    with bind(ctx):
        v1 = store.create_version("DOC-1", "doc", {"n": 1})
        store.create_version("DOC-1", "doc", {"n": 2})
        assert store.impact_of(v1.id) == []
        assert len(store.impact_of(v1.id, exclude_revisions=False)) == 1


def test_baseline_captures_approved_heads_only(pg, ctx):
    store = ArtifactStore(PgArtifactRepository(dsn=pg))
    with bind(ctx):
        a = store.create_version("A", "doc", {"n": 1})
        store.transition(a.id, ArtifactState.IN_REVIEW)
        store.transition(a.id, ArtifactState.APPROVED)
        store.create_version("B", "doc", {"n": 2})  # draft

        assert store.create_baseline("design-freeze") == 1

    members = db.query_all(
        """
        SELECT v.artifact_key
        FROM basis_artifact_baseline_members m
        JOIN basis_artifact_versions v ON v.id = m.version_id
        """,
        dsn=pg,
    )
    assert [m["artifact_key"] for m in members] == ["A"]


def test_baseline_excludes_superseded_versions(pg, ctx):
    """A baseline captures heads, not every approved row in history."""
    store = ArtifactStore(PgArtifactRepository(dsn=pg))
    with bind(ctx):
        v1 = store.create_version("A", "doc", {"n": 1})
        store.transition(v1.id, ArtifactState.IN_REVIEW)
        store.transition(v1.id, ArtifactState.APPROVED)
        v2 = store.create_version("A", "doc", {"n": 2})
        store.transition(v2.id, ArtifactState.IN_REVIEW)
        store.transition(v2.id, ArtifactState.APPROVED)

        assert store.create_baseline("freeze") == 1

    row = db.query_one(
        """
        SELECT v.version
        FROM basis_artifact_baseline_members m
        JOIN basis_artifact_versions v ON v.id = m.version_id
        """,
        dsn=pg,
    )
    assert row["version"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE — pgvector + generated tsvector hybrid search
# ═══════════════════════════════════════════════════════════════════════════


def _seed(repo, ctx, other_ctx, embedder):
    def chunk(cid, tenant, project, corpus, content):
        return {
            "id": cid,
            "tenant_id": tenant,
            "project_id": project,
            "corpus": corpus,
            "content": content,
            "embedding": embedder.embed(content),
        }

    repo.add_chunks([
        chunk(str(uuid.uuid4()), ctx.tenant_id, ctx.project_id, "client_project",
              "OB09 holds the exchange rate configuration"),
        chunk(str(uuid.uuid4()), ctx.tenant_id, ctx.project_id, "client_project",
              "plant maintenance is out of scope"),
        chunk(str(uuid.uuid4()), ctx.tenant_id, None, "org_assets",
              "reusable exchange rate accelerator"),
        chunk(str(uuid.uuid4()), other_ctx.tenant_id, other_ctx.project_id,
              "client_project", "exchange rate secrets of another client"),
        chunk(str(uuid.uuid4()), None, None, "public_domain",
              "SAP exchange rate fundamentals"),
    ])


def test_generated_tsvector_column_populates(pg, ctx):
    repo = PgChunkRepository(dsn=pg)
    repo.add_chunks([
        {
            "id": str(uuid.uuid4()),
            "tenant_id": ctx.tenant_id,
            "project_id": ctx.project_id,
            "corpus": "client_project",
            "content": "exchange rate configuration",
        }
    ])
    row = db.query_one("SELECT fts::text AS f FROM basis_chunks", dsn=pg)
    # Generated, so the application never writes it and it cannot go stale.
    assert "exchang" in row["f"]


def test_chunk_scope_check_constraint_is_enforced(pg, ctx):
    """Fail-closed at the schema level: a client_project chunk without a
    project cannot exist, so a buggy ingester cannot create a leaky row."""
    from psycopg.errors import CheckViolation

    repo = PgChunkRepository(dsn=pg)
    with pytest.raises(CheckViolation):
        repo.add_chunks([
            {
                "id": str(uuid.uuid4()),
                "tenant_id": ctx.tenant_id,
                "project_id": None,
                "corpus": "client_project",
                "content": "no project",
            }
        ])


def test_hybrid_search_never_crosses_tenants(pg, ctx, other_ctx):
    embedder = HashEmbedder(dimensions=EMBED_DIM)
    repo = PgChunkRepository(dsn=pg)
    _seed(repo, ctx, other_ctx, embedder)
    retriever = HybridRetriever(repo)

    with bind(ctx):
        hits = retriever.search(
            RetrievalQuery(text="exchange rate", limit=10)
        )
    assert hits
    assert all("another client" not in h.content for h in hits)


def test_hybrid_search_spans_corpora(pg, ctx, other_ctx):
    embedder = HashEmbedder(dimensions=EMBED_DIM)
    repo = PgChunkRepository(dsn=pg)
    _seed(repo, ctx, other_ctx, embedder)
    retriever = HybridRetriever(repo)

    with bind(ctx):
        hits = retriever.search(
            RetrievalQuery(
                text="exchange rate",
                corpora=(
                    Corpus.CLIENT_PROJECT,
                    Corpus.ORG_ASSETS,
                    Corpus.PUBLIC_DOMAIN,
                ),
                limit=10,
            )
        )
    contents = " | ".join(h.content for h in hits)
    assert "OB09" in contents
    assert "accelerator" in contents
    assert "fundamentals" in contents
    assert "another client" not in contents


def test_hybrid_fusion_ranks_double_matches_first(pg, ctx, other_ctx):
    embedder = HashEmbedder(dimensions=EMBED_DIM)
    repo = PgChunkRepository(dsn=pg)
    _seed(repo, ctx, other_ctx, embedder)
    retriever = HybridRetriever(repo)

    text = "OB09 holds the exchange rate configuration"
    with bind(ctx):
        hits = retriever.search(
            RetrievalQuery(text=text, embedding=embedder.embed(text), limit=5)
        )
    assert hits[0].content == text
    assert hits[0].distance is not None
    assert hits[0].text_rank is not None


def test_vector_arm_honours_min_similarity(pg, ctx, other_ctx):
    embedder = HashEmbedder(dimensions=EMBED_DIM)
    repo = PgChunkRepository(dsn=pg)
    _seed(repo, ctx, other_ctx, embedder)
    retriever = HybridRetriever(repo)

    with bind(ctx):
        strict = retriever.search(
            RetrievalQuery(
                text="zzz-no-lexical-match-zzz",
                embedding=embedder.embed("completely unrelated query text"),
                min_similarity=0.99,
                limit=10,
            )
        )
    assert strict == []


def test_replace_document_is_wholesale(pg, ctx):
    repo = PgChunkRepository(dsn=pg)
    retriever = HybridRetriever(repo)
    doc = str(uuid.uuid4())
    repo.add_chunks([
        {"id": str(uuid.uuid4()), "tenant_id": ctx.tenant_id,
         "project_id": ctx.project_id, "document_id": doc,
         "corpus": "client_project", "content": "stale text", "ordinal": 0},
    ])
    retriever.replace_document(
        document_id=doc,
        tenant_id=ctx.tenant_id,
        chunks=[
            {"id": str(uuid.uuid4()), "tenant_id": ctx.tenant_id,
             "project_id": ctx.project_id, "document_id": doc,
             "corpus": "client_project", "content": "fresh text", "ordinal": 0},
        ],
    )
    rows = db.query_all("SELECT content FROM basis_chunks", dsn=pg)
    assert [r["content"] for r in rows] == ["fresh text"]


# ═══════════════════════════════════════════════════════════════════════════
# MODEL / PERSONA RESOLUTION — the UNION cascade
# ═══════════════════════════════════════════════════════════════════════════


def test_resolution_cascade_prefers_task_model(pg, ctx):
    clear_cache()
    repo = PgTaskModelRepository(dsn=pg)
    db.execute(
        "INSERT INTO agent_task_models (tenant_id, task_key, model_id, system_prompt) "
        "VALUES (%s, 'classify', 'model-from-task', 'task prompt')",
        (ctx.tenant_id,),
        dsn=pg,
    )
    db.execute(
        "INSERT INTO agent_personas (id, tenant_id, name, role, model_id) "
        "VALUES (%s, %s, 'classify', 'classify', 'model-from-persona')",
        (str(uuid.uuid4()), ctx.tenant_id),
        dsn=pg,
    )
    task = resolve("classify", tenant_id=ctx.tenant_id, repo=repo, use_cache=False)
    assert task.model_id == "model-from-task"
    assert task.source == "task_model"
    assert task.system_prompt == "task prompt"


def test_resolution_falls_through_to_workstream_then_role(pg, ctx):
    clear_cache()
    repo = PgTaskModelRepository(dsn=pg)
    db.execute(
        "INSERT INTO agent_personas (id, tenant_id, name, role, workstream, model_id) "
        "VALUES (%s, %s, 'fc-fin', 'consultant', 'finance', 'model-finance')",
        (str(uuid.uuid4()), ctx.tenant_id),
        dsn=pg,
    )
    db.execute(
        "INSERT INTO agent_personas (id, tenant_id, name, role, model_id) "
        "VALUES (%s, %s, 'fc', 'consultant', 'model-generic')",
        (str(uuid.uuid4()), ctx.tenant_id),
        dsn=pg,
    )

    specific = resolve(
        "consultant:finance", tenant_id=ctx.tenant_id, repo=repo, use_cache=False
    )
    assert specific.model_id == "model-finance"
    assert specific.source == "persona_workstream"

    generic = resolve("consultant", tenant_id=ctx.tenant_id, repo=repo, use_cache=False)
    assert generic.model_id == "model-generic"
    assert generic.source == "persona_role"


def test_resolution_is_tenant_scoped(pg, ctx, other_ctx):
    clear_cache()
    repo = PgTaskModelRepository(dsn=pg)
    db.execute(
        "INSERT INTO agent_task_models (tenant_id, task_key, model_id) "
        "VALUES (%s, 'classify', 'tenant-a-model')",
        (ctx.tenant_id,),
        dsn=pg,
    )
    # The other tenant must NOT inherit it.
    other = resolve(
        "classify", tenant_id=other_ctx.tenant_id, repo=repo, use_cache=False
    )
    assert other.source == "default"


def test_resolution_returns_fallback_chain(pg, ctx):
    clear_cache()
    repo = PgTaskModelRepository(dsn=pg)
    db.execute(
        """
        INSERT INTO agent_task_models
          (tenant_id, task_key, model_id, fallback_model_ids)
        VALUES (%s, 'classify', 'primary', ARRAY['fb1','fb2'])
        """,
        (ctx.tenant_id,),
        dsn=pg,
    )
    task = resolve("classify", tenant_id=ctx.tenant_id, repo=repo, use_cache=False)
    assert task.fallback_model_ids == ("fb1", "fb2")
    assert [m for m, _ in task.chain()] == ["primary", "fb1", "fb2"]


def test_persona_store_against_db(pg, ctx, other_ctx):
    for tenant, name in ((ctx.tenant_id, "acme-analyst"), (other_ctx.tenant_id, "globex-analyst")):
        db.execute(
            "INSERT INTO agent_personas (id, tenant_id, name, role) "
            "VALUES (%s, %s, %s, 'analyst')",
            (str(uuid.uuid4()), tenant, name),
            dsn=pg,
        )
    store = PersonaStore(PgPersonaRepository(dsn=pg), ttl_seconds=0)
    assert [p["name"] for p in store.all(tenant_id=ctx.tenant_id)] == ["acme-analyst"]
    assert [p["name"] for p in store.all(tenant_id=other_ctx.tenant_id)] == [
        "globex-analyst"
    ]


def test_inactive_personas_are_excluded(pg, ctx):
    db.execute(
        "INSERT INTO agent_personas (id, tenant_id, name, role, is_active) "
        "VALUES (%s, %s, 'retired', 'analyst', FALSE)",
        (str(uuid.uuid4()), ctx.tenant_id),
        dsn=pg,
    )
    store = PersonaStore(PgPersonaRepository(dsn=pg), ttl_seconds=0)
    assert store.all(tenant_id=ctx.tenant_id) == []


# ═══════════════════════════════════════════════════════════════════════════
# WORKFLOW — durable checkpoints and resume
# ═══════════════════════════════════════════════════════════════════════════


def test_checkpoint_roundtrip(pg, ctx):
    repo = PgWorkflowRepository(dsn=pg)
    record = new_run_record(ctx, "wf", 1, {"input": "x"})
    repo.create(record)

    record.step("s1").status = StepStatus.COMPLETE
    record.step("s1").output = {"result": 42}
    record.step("s1").attempts = 2
    repo.save_step(record, "s1")
    record.status = RunStatus.COMPLETE
    repo.save_run(record)

    loaded = repo.load(record.run_id, tenant_id=ctx.tenant_id)
    assert loaded is not None
    assert loaded.status is RunStatus.COMPLETE
    assert loaded.payload == {"input": "x"}
    assert loaded.step("s1").status is StepStatus.COMPLETE
    assert loaded.step("s1").output == {"result": 42}
    assert loaded.step("s1").attempts == 2


def test_checkpoint_none_output_roundtrips_as_none(pg, ctx):
    """The {"value": ...} wrapper exists so a None output is distinguishable
    from "never set"."""
    repo = PgWorkflowRepository(dsn=pg)
    record = new_run_record(ctx, "wf", 1, {})
    repo.create(record)
    record.step("s1").status = StepStatus.COMPLETE
    record.step("s1").output = None
    repo.save_step(record, "s1")

    loaded = repo.load(record.run_id, tenant_id=ctx.tenant_id)
    assert loaded.step("s1").output is None
    assert loaded.step("s1").status is StepStatus.COMPLETE


def test_checkpoint_unserializable_output_does_not_lose_the_step(pg, ctx):
    class Opaque:
        pass

    repo = PgWorkflowRepository(dsn=pg)
    record = new_run_record(ctx, "wf", 1, {})
    repo.create(record)
    record.step("s1").status = StepStatus.COMPLETE
    record.step("s1").output = Opaque()
    repo.save_step(record, "s1")

    loaded = repo.load(record.run_id, tenant_id=ctx.tenant_id)
    assert loaded.step("s1").status is StepStatus.COMPLETE
    assert "__unserializable__" in loaded.step("s1").output


def test_checkpoint_is_tenant_scoped(pg, ctx, other_ctx):
    repo = PgWorkflowRepository(dsn=pg)
    record = new_run_record(ctx, "wf", 1, {})
    repo.create(record)
    assert repo.load(record.run_id, tenant_id=other_ctx.tenant_id) is None


async def test_durable_run_survives_a_new_engine(pg, ctx):
    """The property job_executor could not have: resume in a different process.

    Two engines, two stores, one database - the second engine picks up a run it
    never started.
    """
    calls = []

    async def step(step_ctx, task):
        calls.append(task.upstream)
        return "done"

    wf = Workflow(
        "durable",
        [
            Step(id="prepare", kind=StepKind.TASK, task=step),
            Step(id="gate", kind=StepKind.APPROVAL, depends_on=("prepare",)),
            Step(id="finish", kind=StepKind.TASK, task=step, depends_on=("gate",)),
        ],
    )

    engine_a = WorkflowEngine(store=PgWorkflowRepository(dsn=pg))
    with bind(ctx):
        record = await engine_a.start(wf, ctx)
    assert record.status is RunStatus.WAITING_APPROVAL

    # A fresh engine and store, as a different process would have.
    engine_b = WorkflowEngine(store=PgWorkflowRepository(dsn=pg))
    with bind(ctx):
        resumed = await engine_b.approve(wf, ctx, record.run_id, "gate")

    assert resumed.status is RunStatus.COMPLETE
    # 'prepare' ran once, before the pause; it is not re-run on resume.
    assert len(calls) == 2
    assert calls[1] == {"gate": {"approved": True, "by": ctx.user_id}}


async def test_failed_run_persists_skipped_dependents(pg, ctx):
    async def boom(step_ctx, task):
        raise RuntimeError("nope")

    async def ok(step_ctx, task):
        return "ok"

    wf = Workflow(
        "failing",
        [
            Step(id="bad", kind=StepKind.TASK, task=boom),
            Step(id="after", kind=StepKind.TASK, task=ok, depends_on=("bad",)),
        ],
    )
    store = PgWorkflowRepository(dsn=pg)
    engine = WorkflowEngine(store=store)

    from basis.workflow import WorkflowRunError

    with bind(ctx):
        with pytest.raises(WorkflowRunError):
            await engine.start(wf, ctx)

    row = db.query_one(
        "SELECT id, status FROM basis_workflow_runs", dsn=pg
    )
    assert row["status"] == "failed"
    steps = {
        r["step_id"]: r["status"]
        for r in db.query_all("SELECT step_id, status FROM basis_workflow_steps", dsn=pg)
    }
    assert steps == {"bad": "failed", "after": "skipped"}
