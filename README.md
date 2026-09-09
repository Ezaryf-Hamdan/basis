# basis

A reusable substrate for building an AI platform.

Eight functions an AI platform needs. Four were already working inside
`AILedSDLC` and were lifted out; four did not exist there and were built from
scratch. `ai-core` is the first consumer, and nothing here is specific to it.

```python
from basis import Principal, RunContext, bind
from basis.models import ModelGateway
from basis.tools import ToolPolicy, ToolRegistry

ctx = RunContext(
    principal=Principal(subject_id=user_id, tenant_id=tenant_id, token=jwt),
    project_id=project_id,
    persona_id=persona_id,
    task_key="classify_requirement",
)

with bind(ctx):
    response = await gateway.invoke(
        "classify_requirement",
        user_prompt=text,
    )
```

## What this is, and what it is not

This package is the part of the AILedSDLC Python tier that is **not** about SAP
delivery. Roughly a third of `agent/`'s 31k non-test lines were generic
platform concerns tangled together with the domain; those concerns are here,
untangled. The SAP generators — FSD authoring, BPD trees, RICEFW config, KDD,
build objects, fit/gap — stay in the application. They are 80% of the source by
volume and none of it belongs in a shared package.

Four of the eight functions had nothing liftable in Python and were built here
instead — see [Corrections to the inventory](#corrections-to-the-inventory) for
where the POC deck's read of the source differs from what the code actually
says.

## Install

```bash
pip install -e .              # core: run context, tool policy, memory, redaction
pip install -e ".[bedrock]"   # + Strands/Bedrock model provider and Titan embeddings
pip install -e ".[otel]"      # + OpenTelemetry tracing
pip install -e ".[all]"
```

The core deliberately depends only on `psycopg`. A consumer that wants tool
governance or PII redaction should not have to install boto3, Strands and an
OTel exporter to get them — and in the source repo it did, because everything
imported everything.

`psycopg` 3 rather than `psycopg2`: `ai-core/parser` runs Python **3.14**, where
`psycopg2-binary` has no wheels. The SQL is unaffected (both use `%s`
placeholders); `RealDictCursor` becomes `row_factory=dict_row`.

## Layout

Eight functions a platform needs. Four lifted from AILedSDLC, four built from
scratch because they were not there.

| Function | Module | LOC | Source |
|---|---|---|---|
| models | [models/](src/basis/models/) | 1304 | **lifted** — `model_config.py`, `llm_router.py` |
| tools | [tools/](src/basis/tools/) | 675 | **lifted** — `tool_access.py`, `tool_call_capture.py` |
| observability | [observability/](src/basis/observability/) | 684 | **lifted** — `otel_setup.py`, `activity_logger.py` |
| agents | [agents/](src/basis/agents/) + [personas/](src/basis/personas/) | 825 | **lifted** personas; **built** the Agent contract |
| security | [context.py](src/basis/context.py) | 250 | **built** — auth stays in Node by design |
| knowledge | [knowledge/](src/basis/knowledge/) | 515 | **built** — only `chunk_text` was liftable |
| workflow | [workflow/](src/basis/workflow/) | 985 | **built** — nothing existed |
| artifacts | [artifacts/](src/basis/artifacts/) | 420 | **built** — nothing existed |

Supporting: [memory/](src/basis/memory/) 606 (consolidated from four
overlapping modules), [embeddings/](src/basis/embeddings/) 253,
[storage/](src/basis/storage/) 1864 (two backends),
[ports.py](src/basis/ports.py) 248, [adapters/](src/basis/adapters/) 386, and
the core: [context.py](src/basis/context.py) /
[db.py](src/basis/db.py) / [settings.py](src/basis/settings.py) /
[errors.py](src/basis/errors.py) / [concurrency.py](src/basis/concurrency.py).

**~10,000 LOC across 53 modules. 155 unit tests need nothing; 47 more run against live Postgres.**

### Portability

Two rules, both enforced by tests rather than convention.

**No function imports another function.** Cross-function collaboration goes
through [ports.py](src/basis/ports.py) — structural `Protocol`s, so an
implementation satisfies one without subclassing or registering. Before this
there were six top-level edges and a `models` ↔ `observability` **cycle** that
survived only because one import sat inside a function body.

```
                 BEFORE                          AFTER
agents      ──▶ models                   agents      ──▶ ports
memory      ──▶ models, embeddings       memory      ──▶ ports
personas    ──▶ models                   personas    ──▶ ports
workflow    ──▶ agents, observability    workflow    ──▶ ports
models      ──▶ observability  ◀─┐       models      ──▶ ports
observability ──▶ models       ──┘ CYCLE observability ──▶ ports
```

`test_no_cross_function_imports` parses the AST and fails if an edge reappears.
`test_ports_module_depends_on_nothing` keeps ports from becoming another layer.
(`adapters/` is exempt by design — its whole job is to know a consumer.)

**No service contains SQL.** Storage sits behind repository protocols in
[storage/](src/basis/storage/), with two complete backends:

| | Backend | Use |
|---|---|---|
| `storage.postgres` | pgvector + tsvector + recursive CTEs | production |
| `storage.inmemory` | Python-side cosine, term overlap | tests, single-process, and **proof the ports are honest** |

The in-memory backend is not a stub. Writing it is what forced `search_vector`
and `search_text` apart, and what surfaced that `grants_for` returning `None`
versus an empty set is a semantic distinction rather than an accident. An
interface with one implementation is an assumption wearing an abstraction's
clothes.

Table names come from a [`TableMap`](src/basis/storage/__init__.py), so basis
can be pointed at an existing schema without an adapter.

**What is still Postgres-shaped, deliberately:** `storage.postgres` uses
pgvector `<=>`, generated `tsvector` columns, `FOR UPDATE` on version heads and
recursive CTEs for lineage. Those are load-bearing, not incidental — abstracting
them away would cost the best retrieval and the only real concurrency guarantee
in the package. A non-Postgres consumer writes a sibling repository module; it
does not edit eight services.

### Two load-bearing design decisions

## The schema is not backward compatible

`migrations/0001_basis_schema.sql` adds `tenant_id` to six tables and **changes
three unique constraints** (`agent_task_models.task_key`,
`agent_tools.tool_name`, `agent_conversations.session_id` all become
tenant-scoped). It also adds `agent_tool_invocations`, which is new.

Set `BASIS_BACKFILL_TENANT` at the top of the file before applying it to a
database with existing rows. Applying this to AILedSDLC's database is a
migration with a backfill, not a no-op — that is the cost of the isolation
guarantee, and it is unavoidable because the tenant concept genuinely did not
exist.

## Bugs found in the lifted code

These are defects in the source, carried into the notes so they do not get
re-introduced. Each is fixed here; each is still live in `AILedSDLC/agent`.

**1. Sampling parameters are sent to models that reject them.** `model_config._build_model`
passes `temperature` and `top_p` whenever the task's `model_config` row sets
them, and `validate_model_config` accepts both as valid. But `temperature`,
`top_p` and `top_k` were **removed** on Claude Sonnet 5 and the Opus 4.7/4.8/5
family — they return a 400. The repo's own default is
`us.anthropic.claude-sonnet-5`. Any task key with a tuned temperature fails on
the default model. It is data-dependent, which is why it can sit unnoticed.
Fixed in `models/catalog.accepts_sampling` + `providers/bedrock`; caught at
config time by `resolver.validate_model_config`.

**2. `asyncio.run` inside a sync function called from async code.**
`memory_consolidator._invoke_consolidator` wraps `invoke_with_fallback` in
`asyncio.run()`. Called from a running loop that raises
`RuntimeError: asyncio.run() cannot be called from a running event loop`. It
only works today because consolidation runs in a worker thread with no loop of
its own. `memory.consolidator.consolidate` is a coroutine.

**3. The persona cache leaks across tenants.** `personas._cached_personas` is a
module-level global with no TTL and no tenant key. The first tenant to call
`load_personas()` populates it for every subsequent tenant in the process. It
also returns the mutable cached list, so any caller mutating a persona dict
corrupts it for everyone. Fixed in `personas.store.PersonaStore`.

**4. Stored embeddings are never searched.** `memory_service.write_long_term`
takes an embedding, casts it `::vector`, and stores it. `recall_long_term` then
orders by `importance DESC, created_at DESC` — the vector is never read. The
parallel `long_term_memory.recall` does search content, but with
`content ILIKE %word%` OR'd across query words, under a docstring reading
"TODO: Upgrade to embedding-based search with Bedrock Titan". The embedding
column, the Titan call and pgvector were all already present; nothing joined
them up. `memory.service.recall` does cosine-distance search blended with
importance, and returns the distance so a reranker can be built on it.

**5. Fallback with no backoff.** `invoke_with_fallback` walks the whole model
chain on a `ThrottlingException` with zero delay, then fails. A transient
capacity limit becomes a permanent error, and the chain burns in microseconds.
`models/retry.backoff_delays` retries each model with jittered exponential
backoff before demoting.

**6. Token usage is discarded.** `invoke_with_fallback` returns
`(text, model_id)` and throws away the usage block. This is the mechanical
reason per-client cost attribution does not exist — not a missing dashboard, a
missing return value. `ModelResponse.usage` carries it and
`observability.genai.record_usage` writes it to a tenant-tagged span.

**7. Connection churn.** Every DB module opened a fresh connection per call;
`model_config` issued up to six to resolve one task key fully (four cascades
across three functions, each with its own `_query_one`). The repo has two
filed bugs about the fallout (#200, #367 — psycopg2's `with connection`
commits but does not close). `db.py` pools, and `resolver` does it in one
query with a TTL cache.

**8. `json.dumps(None)` writes the string `"null"`.** `memory.save_conversation`
passes `context` straight to `json.dumps`, so "no context" is stored as a JSON
null rather than SQL NULL.

**9. Global FIFO audit pruning.** `tool_call_capture._prune_tool_call_rows` runs
`DELETE ... WHERE id NOT IN (SELECT id ... ORDER BY created_at DESC LIMIT 10000)`
— a full re-scan and re-sort per call, holding a long lock, and global, so a
busy tenant evicts a quiet tenant's audit history. Replaced with per-tenant
retention by age.

**10. Only the first tool-result block is captured.**
`tool_call_capture.extract_tool_calls` reads `raw_content[0].get('text','')`,
so a tool returning several content blocks — which MCP permits — loses
everything after the first from the audit record.

## Corrections to the inventory

The deck's service-by-service read was built from slides, not code. Reading the
code changes four rows.

**`knowledge` is not liftable — it is Node.** The deck rates this the strongest
candidate ("LIFT", high confidence). There is no vector retrieval in
`AILedSDLC/agent` at all: `document_ingest.py` contains zero references to
pgvector, embeddings or similarity search — it is LLM-based document *analysis*
and decomposition, not RAG. The actual pipeline is in
`server/src/modules/precedents/`, `rag-eval/` and `search/`, in TypeScript, and
`fsd_content_generator.py:518` reaches it over REST (`POST /precedents/search`).
The only embedding code in Python was `memory_consolidator._embed`, for memory
consolidation. So `knowledge` sits with `security`: real, working, and in the
wrong language. That moves roughly a week of the estimate and removes the
deck's single highest-confidence lift.

**`models` is richer than "a thin wrapper".** The deck prices a greenfield
rebuild at ~1 week and infers there is "almost certainly no policy routing".
There is: DB-driven per-task model resolution with a persona cascade, ordered
fallback chains, Bedrock retryability classification, and prompt caching with
the correct Converse `cachePoint` shape. What is genuinely absent is per-tenant
isolation, redaction and usage recording. The estimate is low.

**`agents` — the coupling diagnosis is right, the volume is worse.** Agent
Reusability 5/10 and "sequencing almost certainly lives inside the agents" are
both correct. `job_executor.py` is **235 KB / 4,396 lines** in one module, with
a 27-entry job-type dispatch, and it is where orchestration lives. Nothing in
it is liftable as-is.

**`observability` — the OTel that exists is better than credited.**
`otel_setup.py` already has PII redaction, gen_ai prompt/completion capture,
and the CloudWatch-specific insight that Transaction Search strips span events
but preserves attributes. That is real operational knowledge and it lifts. The
deck's conclusion still holds — no span carried a tenant, because no tenant
existed — but the instrumentation was not starting from zero.

Unchanged from the deck: `workflow`, `artifacts` — nothing to lift, build from
scratch. `security` stays in Node by design (ADR-0002); `basis.context` is the
Python-side counterpart it needs.

## Testing

```bash
python -m pytest -q                          # 155 unit tests, no DB or AWS
INTEGRATION_TESTS=1 python -m pytest -m integration   # 47 against live Postgres
```

**202 tests. Verified on Python 3.13.14 and 3.14.4, against Postgres 18.6 with
pgvector 0.8.6.** 3.11 and 3.12 are declared supported and covered by the CI
matrix, but have not been run locally.

Integration tests need `migrations/0000` + `0002` applied and
`BASIS_TEST_DSN` set. They truncate between tests rather than recreating the
schema.

Four checks enforce the architecture rather than describing it:

| Test | Fails if |
|---|---|
| `test_no_cross_function_imports` | a function imports a sibling at module level |
| `test_ports_module_depends_on_nothing` | `ports.py` grows a dependency |
| `test_no_sql_outside_storage` | SQL appears outside `storage/` or `adapters/` |
| `test_no_blocking_io_inside_coroutines` | a blocking store call is awaited bare in `async def` |

`ruff check` and `mypy src/basis` are both clean.

## What is still missing

| Gap | Why it is open |
|---|---|
| **No provider has made a live call** | `BedrockProvider` needs AWS credentials; `OllamaProvider` needs a running endpoint. Both are structurally exercised; neither has talked to a real model. |
| **`job_executor.py` untangled** | 4,396 lines of AILedSDLC orchestration still needs porting onto `basis.workflow`. The engine exists; the migration of the 27 job types does not. |
| **Ingestion pipeline** | `HybridRetriever.add` stores chunks and `chunk_text` splits text, but nothing wires chunk → embed → store into one call. |
| **Python 3.11 / 3.12** | Declared and CI-covered, not locally verified. |
| **Retention beyond audit rows** | `prune_invocations` exists. Nothing prunes artefacts, chunks or workflow runs, and there is no per-tenant erasure path. |
| **Exactly-once side effects** | Out of scope by design — steps must be idempotent. Needs a transactional outbox or Temporal. |

## Wiring it into ai-core

`ai-core`'s Python surface today is `parser/` — a FastAPI document parser on
3.14. It has no agent tier yet, so `basis` is additive:

```python
from basis import Principal, RunContext, bind
from basis.models import ModelGateway
from basis.tools import ToolPolicy, ToolRegistry

# ai-core's own taxonomy, not AILedSDLC's.
POLICY = ToolPolicy(
    allow_exact=frozenset({"propose_change"}),
    known_scopes=frozenset({"analyze", "generate"}),
)

gateway = ModelGateway()
registry = ToolRegistry(POLICY)

with bind(RunContext(principal=..., project_id=..., task_key="analyze")):
    tools = registry.tools_for(mcp_tools)
    result = await gateway.invoke("analyze", user_prompt=text, tools=tools)
```

The Node/Next tier keeps issuing and verifying JWTs; hand the verified claims
to `Principal.from_claims` and Python never parses a token.
