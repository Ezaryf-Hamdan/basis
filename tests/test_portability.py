"""Tests that the ports are real.

Two things get proved here, and neither could be proved before the refactor:

1. **The storage ports are honest.** Version supersession, lineage traversal,
   drift detection, hybrid retrieval and memory recall all run against
   `storage.inmemory` with no database. If a service still held SQL, these
   tests could not exist.

2. **The functions do not depend on each other.** `test_no_cross_function_imports`
   parses the import graph and fails if a function grows a top-level import of
   a sibling. That is the only way this property stays true - it is exactly the
   kind of thing that decays silently.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from basis import Principal, RunContext, bind
from basis.artifacts import ArtifactState, ArtifactStore, LineageKind
from basis.artifacts.store import ArtifactConflict
from basis.embeddings import HashEmbedder, NullEmbedder
from basis.knowledge import Corpus, HybridRetriever, RetrievalQuery
from basis.memory import MemoryScope, MemoryService
from basis.memory.conversation import ConversationStore
from basis.personas import PersonaStore
from basis.storage import TableMap
from basis.storage.inmemory import (
    InMemoryArtifactRepository,
    InMemoryChunkRepository,
    InMemoryConversationRepository,
    InMemoryMemoryRepository,
    InMemoryPersonaRepository,
    InMemoryToolRepository,
)
from basis.tools import ToolPolicy, ToolRegistry

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "basis"

FUNCTIONS = {
    "models",
    "tools",
    "memory",
    "observability",
    "personas",
    "embeddings",
    "agents",
    "workflow",
    "artifacts",
    "knowledge",
}


def _ctx(tenant="t1", user="u1", project="p1", **kw):
    return RunContext(
        principal=Principal(subject_id=user, tenant_id=tenant),
        project_id=project,
        **kw,
    )


def _scope(tenant="t1", user="u1", project="p1", persona="per1", job="j1"):
    return MemoryScope(
        tenant_id=tenant,
        user_id=user,
        project_id=project,
        persona_id=persona,
        job_id=job,
    )


# ═══════════════════════════════════════════════════════════════════════════
# DECOUPLING
# ═══════════════════════════════════════════════════════════════════════════


def _cross_function_edges(top_level_only: bool = True) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = {}
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        owner = rel.parts[0] if len(rel.parts) > 1 else rel.stem
        if owner not in FUNCTIONS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))

        in_function: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.ImportFrom):
                        in_function.add(id(inner))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue
            if top_level_only and id(node) in in_function:
                continue
            target = (node.module or "").split(".")[0]
            if target in FUNCTIONS and target != owner:
                edges.setdefault(owner, set()).add(target)
    return edges


def test_no_cross_function_imports():
    """No function may import a sibling at module level.

    Cross-function collaboration goes through `basis.ports`. If this fails,
    the named edge needs a protocol in ports.py rather than a direct import -
    that is what broke the models <-> observability cycle.
    """
    edges = _cross_function_edges()
    assert edges == {}, "cross-function coupling reappeared: %s" % {
        k: sorted(v) for k, v in edges.items()
    }


def test_ports_module_depends_on_nothing():
    """`ports` must not import any function package.

    A port module that grows a dependency stops being a port - it becomes just
    another layer with an opinion about who is downstream.
    """
    tree = ast.parse((SRC / "ports.py").read_text(encoding="utf-8"))
    offenders = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").split(".")[0] in FUNCTIONS
    ]
    assert offenders == [], "ports.py imports function packages: %s" % offenders


def test_no_sql_outside_storage():
    """SQL belongs to the storage layer (and to the adapters, which target a
    specific consumer's schema by definition)."""
    markers = ("INSERT INTO", "DELETE FROM", "SELECT id", "SELECT *", "UPDATE ")
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith(("storage/", "adapters/")) or rel == "db.py":
            continue
        source = path.read_text(encoding="utf-8")
        # Strip docstrings: several modules quote the original SQL to explain
        # what was wrong with it, which is documentation, not a dependency.
        tree = ast.parse(source)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                doc = ast.get_docstring(node)
                if doc:
                    docstrings.add(doc)
        code = source
        for doc in docstrings:
            code = code.replace(doc, "")
        if any(m in code for m in markers):
            offenders.append(rel)
    assert offenders == [], "SQL found outside the storage layer: %s" % offenders


def test_port_implementations_satisfy_protocols():
    from basis.agents import AgentRegistry, AgentRunner, AgentRunnerInvoker
    from basis.models import CatalogPriceBook, ModelGateway, OllamaProvider
    from basis.models.providers.base import ModelProvider
    from basis.observability.tracing import Tracer
    from basis.ports import (
        AgentInvoker,
        Embedder,
        ModelClient,
        PriceBook,
        TracerPort,
    )

    assert isinstance(Tracer(), TracerPort)
    assert isinstance(CatalogPriceBook(), PriceBook)
    assert isinstance(OllamaProvider(), ModelProvider)
    assert isinstance(HashEmbedder(), Embedder)
    assert isinstance(NullEmbedder(), Embedder)
    assert isinstance(
        AgentRunnerInvoker(AgentRunner(AgentRegistry()), lambda c: None), AgentInvoker
    )
    # ModelGateway must satisfy ModelClient without knowing the protocol exists.
    assert isinstance(ModelGateway(providers=[OllamaProvider()]), ModelClient)


def test_table_map_rejects_unknown_tables():
    TableMap(chunks="my_chunks")  # known
    with pytest.raises(ValueError):
        TableMap(not_a_table="x")


def test_table_map_overrides_apply():
    assert TableMap(memories="acme_memories").memories == "acme_memories"
    # Unrelated names keep their defaults.
    assert TableMap(memories="acme_memories").chunks == "basis_chunks"


# ═══════════════════════════════════════════════════════════════════════════
# MEMORY against the in-memory backend
# ═══════════════════════════════════════════════════════════════════════════


def test_short_term_roundtrip_and_clear():
    svc = MemoryService(InMemoryMemoryRepository())
    scope = _scope()
    svc.write_short_term(scope, "observation", "the client uses two ledgers")
    svc.write_short_term(scope, "observation", "FX posts nightly")
    assert len(svc.read_short_term(scope)) == 2
    assert svc.clear_short_term(scope) == 2
    assert svc.read_short_term(scope) == []


def test_short_term_is_job_scoped():
    repo = InMemoryMemoryRepository()
    svc = MemoryService(repo)
    svc.write_short_term(_scope(job="j1"), "note", "one")
    svc.write_short_term(_scope(job="j2"), "note", "two")
    assert len(svc.read_short_term(_scope(job="j1"))) == 1


def test_recall_without_embedding_orders_by_importance():
    svc = MemoryService(InMemoryMemoryRepository())
    scope = _scope()
    svc.write_long_term(scope, "fact", "low", importance=0.2)
    svc.write_long_term(scope, "fact", "high", importance=0.9)
    rows = svc.recall(scope)
    assert [r["content"] for r in rows] == ["high", "low"]


def test_recall_with_embedding_uses_similarity():
    """The behaviour the source stored embeddings for and never used."""
    embedder = HashEmbedder()
    svc = MemoryService(InMemoryMemoryRepository())
    scope = _scope()

    target = "exchange rate configuration lives in OB09"
    svc.write_long_term(
        scope, "fact", target, importance=0.3, embedding=embedder.embed(target)
    )
    other = "the plant maintenance module is out of scope"
    svc.write_long_term(
        scope, "fact", other, importance=0.9, embedding=embedder.embed(other)
    )

    # Querying with the exact text of the LOW-importance memory must surface it
    # first, which importance-only ordering could never do.
    rows = svc.recall(scope, query_embedding=embedder.embed(target))
    assert rows[0]["content"] == target
    assert rows[0]["distance"] < rows[1]["distance"]


def test_recall_respects_min_importance():
    svc = MemoryService(InMemoryMemoryRepository())
    scope = _scope()
    svc.write_long_term(scope, "fact", "weak", importance=0.1)
    assert svc.recall(scope, min_importance=0.5) == []


def test_recall_is_tenant_isolated():
    repo = InMemoryMemoryRepository()
    svc = MemoryService(repo)
    svc.write_long_term(_scope(tenant="acme"), "fact", "acme secret", importance=0.9)
    svc.write_long_term(_scope(tenant="globex"), "fact", "globex secret", importance=0.9)

    rows = svc.recall(_scope(tenant="acme"))
    assert [r["content"] for r in rows] == ["acme secret"]


def test_recall_records_access():
    repo = InMemoryMemoryRepository()
    svc = MemoryService(repo)
    scope = _scope()
    svc.write_long_term(scope, "fact", "x", importance=0.5)
    svc.recall(scope)
    svc.recall(scope)
    assert repo.long[0]["access_count"] == 2


def test_conversation_roundtrip_and_tenant_isolation():
    repo = InMemoryConversationRepository()
    store = ConversationStore(repo)
    messages = [{"role": "user", "content": "hi"}]

    a = _ctx(tenant="acme", session_id="s1")
    b = _ctx(tenant="globex", session_id="s1")  # same session id, other tenant

    with bind(a):
        store.save(messages)
    with bind(b):
        # Would collide if session_id were globally unique - which it was in the
        # source schema.
        assert store.load() == []
        store.save([{"role": "user", "content": "other"}])
    with bind(a):
        assert store.load() == messages


def test_conversation_none_context_is_not_the_string_null():
    repo = InMemoryConversationRepository()
    store = ConversationStore(repo)
    ctx = _ctx(session_id="s1")
    with bind(ctx):
        store.save([{"role": "user", "content": "x"}])
    assert repo.rows[("t1", "s1")]["context"] is None


# ═══════════════════════════════════════════════════════════════════════════
# ARTIFACTS against the in-memory backend
# ═══════════════════════════════════════════════════════════════════════════


def test_versions_increment_and_supersede():
    store = ArtifactStore(InMemoryArtifactRepository())
    ctx = _ctx()
    with bind(ctx):
        v1 = store.create_version("FSD-1", "fsd", {"body": "first"})
        v2 = store.create_version("FSD-1", "fsd", {"body": "second"})

        assert (v1.version, v2.version) == (1, 2)
        assert v2.supersedes_id == v1.id
        assert store.head("FSD-1")["version"] == 2
        # The predecessor is marked superseded, not left as draft.
        assert store.history("FSD-1")[1]["state"] == ArtifactState.SUPERSEDED.value


def test_identical_content_is_a_conflict_not_a_new_version():
    store = ArtifactStore(InMemoryArtifactRepository())
    with bind(_ctx()):
        store.create_version("FSD-1", "fsd", {"body": "same"})
        with pytest.raises(ArtifactConflict):
            store.create_version("FSD-1", "fsd", {"body": "same"})


def test_state_machine_blocks_draft_to_approved():
    store = ArtifactStore(InMemoryArtifactRepository())
    with bind(_ctx()):
        v = store.create_version("FSD-1", "fsd", {"body": "x"})
        with pytest.raises(ArtifactConflict):
            store.transition(v.id, ArtifactState.APPROVED)
        store.transition(v.id, ArtifactState.IN_REVIEW)
        assert store.transition(v.id, ArtifactState.APPROVED) is ArtifactState.APPROVED


def test_service_principal_cannot_approve_artifact():
    from basis.errors import AuthorizationDenied

    store = ArtifactStore(InMemoryArtifactRepository())
    human = _ctx()
    with bind(human):
        v = store.create_version("FSD-1", "fsd", {"body": "x"})
        store.transition(v.id, ArtifactState.IN_REVIEW)

    robot = RunContext(
        principal=Principal.service("svc", "t1"), project_id="p1"
    )
    with bind(robot):
        with pytest.raises(AuthorizationDenied):
            store.transition(v.id, ArtifactState.APPROVED)


def test_drift_detects_external_edit_and_flags_approved():
    store = ArtifactStore(InMemoryArtifactRepository())
    with bind(_ctx()):
        v = store.create_version("REQ-1", "requirement", {"text": "original"})
        store.transition(v.id, ArtifactState.IN_REVIEW)
        store.transition(v.id, ArtifactState.APPROVED)

        clean = store.check_drift("REQ-1", {"text": "original"})
        assert not clean.changed and not clean.needs_reingestion

        edited = store.check_drift("REQ-1", {"text": "edited in Word"})
        assert edited.changed
        assert edited.needs_reingestion
        # The signal that matters: the baseline no longer reflects reality.
        assert edited.invalidates_approval


def test_drift_on_unknown_artifact_reports_change():
    store = ArtifactStore(InMemoryArtifactRepository())
    with bind(_ctx()):
        report = store.check_drift("NEW-1", {"a": 1})
    assert report.changed and report.stored_version is None


def test_impact_analysis_traverses_lineage_transitively():
    """The capability the source could not have: 'this changed, what is stale?'"""
    store = ArtifactStore(InMemoryArtifactRepository())
    with bind(_ctx()):
        req = store.create_version("REQ-1", "requirement", {"text": "r"})
        fsd = store.create_version(
            "FSD-1", "fsd", {"text": "f"},
            derived_from=[(req.id, LineageKind.DERIVED_FROM)],
        )
        code = store.create_version(
            "OBJ-1", "build_object", {"text": "o"},
            derived_from=[(fsd.id, LineageKind.DERIVED_FROM)],
        )

        impact = store.impact_of(req.id)
        by_key = {r["artifact_key"]: r["depth"] for r in impact}
        assert by_key == {"FSD-1": 1, "OBJ-1": 2}
        assert [r["artifact_key"] for r in impact] == ["FSD-1", "OBJ-1"]
        assert {r["id"] for r in impact} == {fsd.id, code.id}


def test_impact_excludes_revisions_by_default():
    store = ArtifactStore(InMemoryArtifactRepository())
    with bind(_ctx()):
        v1 = store.create_version("DOC-1", "doc", {"n": 1})
        store.create_version("DOC-1", "doc", {"n": 2})
        # v2 revises v1, but a new version of a document is not "impacted by"
        # its own predecessor in the sense a user means.
        assert store.impact_of(v1.id) == []
        assert len(store.impact_of(v1.id, exclude_revisions=False)) == 1


def test_impact_respects_depth_cap():
    store = ArtifactStore(InMemoryArtifactRepository())
    with bind(_ctx()):
        prev = store.create_version("A-0", "doc", {"n": 0})
        for i in range(1, 6):
            prev = store.create_version(
                "A-%d" % i, "doc", {"n": i},
                derived_from=[(prev.id, LineageKind.DERIVED_FROM)],
            )
        root = store.history("A-0")[0]["id"]
        assert len(store.impact_of(root, max_depth=2)) == 2


def test_sources_of_gives_provenance():
    store = ArtifactStore(InMemoryArtifactRepository())
    with bind(_ctx()):
        r1 = store.create_version("REQ-1", "requirement", {"n": 1})
        r2 = store.create_version("REQ-2", "requirement", {"n": 2})
        fsd = store.create_version(
            "FSD-1", "fsd", {"n": 3},
            derived_from=[
                (r1.id, LineageKind.DERIVED_FROM),
                (r2.id, LineageKind.DERIVED_FROM),
            ],
        )
        sources = store.sources_of(fsd.id)
    assert {s["artifact_key"] for s in sources} == {"REQ-1", "REQ-2"}


def test_baseline_captures_only_approved_heads():
    repo = InMemoryArtifactRepository()
    store = ArtifactStore(repo)
    with bind(_ctx()):
        approved = store.create_version("A", "doc", {"n": 1})
        store.transition(approved.id, ArtifactState.IN_REVIEW)
        store.transition(approved.id, ArtifactState.APPROVED)
        store.create_version("B", "doc", {"n": 2})  # stays draft

        assert store.create_baseline("design-freeze") == 1


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE against the in-memory backend
# ═══════════════════════════════════════════════════════════════════════════


def _seed_chunks(repo, embedder):
    repo.add_chunks([
        {
            "id": "c1", "tenant_id": "acme", "project_id": "p1",
            "corpus": "client_project", "content": "OB09 holds exchange rate config",
            "embedding": embedder.embed("OB09 holds exchange rate config"),
        },
        {
            "id": "c2", "tenant_id": "acme", "project_id": "p1",
            "corpus": "client_project", "content": "plant maintenance is out of scope",
            "embedding": embedder.embed("plant maintenance is out of scope"),
        },
        {
            "id": "c3", "tenant_id": "globex", "project_id": "p9",
            "corpus": "client_project", "content": "globex confidential rates",
            "embedding": embedder.embed("globex confidential rates"),
        },
        {
            "id": "c4", "tenant_id": None, "project_id": None,
            "corpus": "public_domain", "content": "SAP exchange rate basics",
            "embedding": embedder.embed("SAP exchange rate basics"),
        },
    ])


def test_retrieval_never_crosses_tenants():
    """The corpus predicate is a permission boundary, not a filing convenience."""
    embedder = HashEmbedder()
    repo = InMemoryChunkRepository()
    _seed_chunks(repo, embedder)
    retriever = HybridRetriever(repo)

    with bind(_ctx(tenant="acme", project="p1")):
        hits = retriever.search(RetrievalQuery(text="rates", limit=10))
    assert "c3" not in {h.id for h in hits}


def test_retrieval_can_span_project_and_public_corpora():
    embedder = HashEmbedder()
    repo = InMemoryChunkRepository()
    _seed_chunks(repo, embedder)
    retriever = HybridRetriever(repo)

    with bind(_ctx(tenant="acme", project="p1")):
        hits = retriever.search(
            RetrievalQuery(
                text="exchange rate",
                corpora=(Corpus.CLIENT_PROJECT, Corpus.PUBLIC_DOMAIN),
                limit=10,
            )
        )
    ids = {h.id for h in hits}
    assert "c1" in ids and "c4" in ids
    assert "c3" not in ids


def test_project_scoped_corpus_requires_a_project():
    from basis.errors import TenantIsolationError

    retriever = HybridRetriever(InMemoryChunkRepository())
    with bind(_ctx(project=None)):
        with pytest.raises(TenantIsolationError):
            retriever.search(RetrievalQuery(text="x"))


def test_hybrid_search_fuses_both_arms():
    embedder = HashEmbedder()
    repo = InMemoryChunkRepository()
    _seed_chunks(repo, embedder)
    retriever = HybridRetriever(repo)

    query_text = "OB09 holds exchange rate config"
    with bind(_ctx(tenant="acme", project="p1")):
        hits = retriever.search(
            RetrievalQuery(
                text=query_text, embedding=embedder.embed(query_text), limit=5
            )
        )
    # c1 matches both arms, so fusion must rank it first.
    assert hits[0].id == "c1"
    assert hits[0].distance is not None
    assert hits[0].text_rank is not None


def test_replace_document_drops_stale_chunks():
    repo = InMemoryChunkRepository()
    retriever = HybridRetriever(repo)
    repo.add_chunks([
        {"id": "old", "tenant_id": "acme", "project_id": "p1", "document_id": "d1",
         "corpus": "client_project", "content": "stale text"},
    ])
    retriever.replace_document(
        document_id="d1",
        tenant_id="acme",
        chunks=[
            {"id": "new", "tenant_id": "acme", "project_id": "p1", "document_id": "d1",
             "corpus": "client_project", "content": "fresh text"},
        ],
    )
    assert [c["id"] for c in repo.chunks] == ["new"]


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS + PERSONAS against the in-memory backend
# ═══════════════════════════════════════════════════════════════════════════


def test_catalog_sync_deactivates_removed_tools():
    repo = InMemoryToolRepository()
    registry = ToolRegistry(ToolPolicy(), repo=repo)
    assert registry.sync_from_mcp(
        [{"name": "get_a"}, {"name": "get_b"}], tenant_id="t1"
    ) == 2
    registry.sync_from_mcp([{"name": "get_a"}], tenant_id="t1")
    assert repo.catalog[("t1", "get_b")]["is_active"] is False


def test_grants_narrow_the_catalog():
    repo = InMemoryToolRepository()
    registry = ToolRegistry(ToolPolicy(), repo=repo)
    registry.sync_from_mcp([{"name": "get_a"}, {"name": "get_b"}], tenant_id="t1")
    repo.grant(tenant_id="t1", persona_id="per1", tool_names=["get_a"])

    tools = [{"name": "get_a"}, {"name": "get_b"}]
    with bind(_ctx(persona_id="per1")):
        kept = registry.tools_for(tools)
    assert [t["name"] for t in kept] == ["get_a"]


def test_persona_with_no_grants_gets_nothing():
    """Fail-closed: an empty grant set is not the same as no grant set."""
    repo = InMemoryToolRepository()
    registry = ToolRegistry(ToolPolicy(), repo=repo)
    registry.sync_from_mcp([{"name": "get_a"}], tenant_id="t1")

    with bind(_ctx(persona_id="per_ungranted")):
        assert registry.tools_for([{"name": "get_a"}]) == []


def test_denied_tool_is_audited():
    from basis.errors import ToolDenied

    repo = InMemoryToolRepository()
    registry = ToolRegistry(ToolPolicy(), repo=repo)
    with bind(_ctx()):
        with pytest.raises(ToolDenied):
            registry.check("delete_everything")
    assert repo.invocations[-1]["allowed"] is False
    assert repo.invocations[-1]["tool_name"] == "delete_everything"


def test_invoke_records_success():
    repo = InMemoryToolRepository()
    registry = ToolRegistry(ToolPolicy(), repo=repo)
    with bind(_ctx()):
        assert registry.invoke("get_thing", lambda: "value") == "value"
    assert repo.invocations[-1]["allowed"] is True
    assert repo.invocations[-1]["effect"] == "read"


def test_persona_cache_is_tenant_keyed():
    """The source's module-global cache leaked the first tenant's personas to
    every subsequent tenant in the process."""
    repo = InMemoryPersonaRepository([
        {"id": "a", "tenant_id": "acme", "name": "acme-analyst",
         "role": "analyst", "workstream": None, "is_active": True},
        {"id": "g", "tenant_id": "globex", "name": "globex-analyst",
         "role": "analyst", "workstream": None, "is_active": True},
    ])
    store = PersonaStore(repo)
    assert store.all(tenant_id="acme")[0]["name"] == "acme-analyst"
    assert store.all(tenant_id="globex")[0]["name"] == "globex-analyst"


def test_persona_store_returns_copies():
    repo = InMemoryPersonaRepository([
        {"id": "a", "tenant_id": "t1", "name": "n", "role": "r",
         "workstream": None, "is_active": True},
    ])
    store = PersonaStore(repo)
    first = store.all(tenant_id="t1")
    first[0]["name"] = "mutated"
    assert store.all(tenant_id="t1")[0]["name"] == "n"


def test_persona_by_id_is_tenant_scoped():
    from basis.errors import ConfigurationError

    repo = InMemoryPersonaRepository([
        {"id": "a", "tenant_id": "acme", "name": "n", "role": "r",
         "workstream": None, "is_active": True},
    ])
    store = PersonaStore(repo)
    assert store.by_id("a", tenant_id="acme")["name"] == "n"
    with pytest.raises(ConfigurationError):
        store.by_id("a", tenant_id="globex")


def test_require_role_raises_instead_of_substituting():
    """The source returned a hardcoded {"name": "Assistant"} persona when none
    matched, which silently swaps a useless persona for a missing config."""
    from basis.errors import ConfigurationError

    store = PersonaStore(InMemoryPersonaRepository([]))
    with pytest.raises(ConfigurationError):
        store.require_role("solution_architect", tenant_id="t1")


# ═══════════════════════════════════════════════════════════════════════════
# EMBEDDERS
# ═══════════════════════════════════════════════════════════════════════════


def test_hash_embedder_is_deterministic_and_normalized():
    import math

    e = HashEmbedder(dimensions=16)
    a, b = e.embed("same text"), e.embed("same text")
    assert a == b
    assert len(a) == 16
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-9)


def test_null_embedder_yields_no_vectors():
    e = NullEmbedder()
    assert e.embed("anything") == []
    assert e.embed_many(["a", "b"]) == [[], []]
    assert e.dimensions == 0


# ═══════════════════════════════════════════════════════════════════════════
# ASYNC SAFETY
# ═══════════════════════════════════════════════════════════════════════════

#: Methods that perform blocking I/O. Calling one bare inside a coroutine
#: stalls the event loop.
_BLOCKING_METHODS = frozenset({
    "save_step", "save_run", "create", "load",
    "read_short_term", "write_short_term", "write_long_term", "clear_short_term",
    "recall", "search", "head", "create_version", "grants_for",
    "prune_invocations", "query_one", "query_all", "execute",
    "embed", "embed_many",
})


def test_no_blocking_io_inside_coroutines():
    """Every blocking store/model call reached from `async def` must be offloaded.

    This started as a real defect: the workflow engine checkpointed twice per
    step with a synchronous round trip from inside a coroutine, so a durable
    run serialized the whole event loop. The unit tests could not catch it
    because `InMemoryCheckpointStore` does no I/O - which is precisely why this
    check is static rather than behavioural.

    Wrap with `concurrency.offload` (always) or `maybe_offload` (skips the
    thread hop when the target sets `blocking = False`).
    """
    offenders: list[str] = []

    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.AsyncFunctionDef):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                if getattr(node.func, "attr", None) not in _BLOCKING_METHODS:
                    continue
                offenders.append(
                    "%s:%s -> %s"
                    % (path.relative_to(SRC).as_posix(), fn.name, ast.unparse(node)[:80])
                )

    assert offenders == [], (
        "blocking I/O called bare inside a coroutine:\n  " + "\n  ".join(offenders)
    )


def test_inmemory_backends_declare_non_blocking():
    """`maybe_offload` defaults to offloading, so a backend that does no I/O
    must opt out explicitly or it pays a pointless thread hop per call."""
    from basis.storage import inmemory
    from basis.workflow import InMemoryCheckpointStore

    classes = [
        getattr(inmemory, n) for n in inmemory.__all__
    ] + [InMemoryCheckpointStore]
    missing = [c.__name__ for c in classes if getattr(c, "blocking", True) is not False]
    assert missing == [], "missing `blocking = False`: %s" % missing


def test_postgres_backends_are_blocking_by_default():
    """The safe default must hold for the backends that really do I/O."""
    from basis.storage import postgres

    for name in postgres.__all__:
        cls = getattr(postgres, name)
        assert getattr(cls, "blocking", True) is True, (
            "%s must not claim to be non-blocking" % name
        )
