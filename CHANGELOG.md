# Changelog

Semantic versioning. Until 1.0.0 the minor version may break compatibility;
each such change is listed under **Breaking** with the migration.

## [Unreleased]

## [0.1.0] - 2026-09-08

First release. Eight platform functions: four lifted out of `AILedSDLC/agent`,
four built from scratch.

### Added
- `context` — `RunContext` / `Principal`, the tenancy spine (new; the source
  had no tenant concept in Python at all).
- `models` — `ModelGateway` with per-tenant task resolution, ordered fallback
  chains, jittered retry, prompt caching, and token/cost recording. Two
  providers: Bedrock (via Strands) and an OpenAI-compatible one covering
  Ollama / vLLM / SGLang.
- `tools` — fail-closed `ToolPolicy` (keep-list, never a denylist), per-tenant
  grants, and an invocation audit trail.
- `observability` — tracing with tenant/project/run attributes, PII and
  credential redaction, `gen_ai` span conventions, cost estimation.
- `agents` — `Agent` contract that structurally cannot orchestrate, plus
  `AgentRunner` for delegation with cycle and round bounds.
- `workflow` — DAG engine with definition-time validation, layered parallelism,
  durable checkpoints, resume, and human approval gates.
- `artifacts` — immutable versions, lineage edges, approval state machine,
  baselines, and content-hash drift detection for external edits.
- `knowledge` — corpus-scoped hybrid retrieval (pgvector + tsvector) with
  Reciprocal Rank Fusion reranking.
- `memory` — short-term scratch notes, long-term memories with working vector
  recall, and LLM consolidation.
- `storage` — repository protocols with two complete backends (`postgres`,
  `inmemory`) and a configurable `TableMap`.
- `ports` — the protocols the functions use to reach each other.
- `adapters.aicore` — maps basis onto ai-core's `workspace_id` / RLS schema.
- Migrations: `0000` greenfield baseline, `0001` AILedSDLC retrofit,
  `0002` workflow / artifacts / knowledge.

### Fixed (defects found in the lifted source, all still live there)
- Sampling parameters (`temperature` / `top_p` / `top_k`) sent to Claude
  Sonnet 5 and the Opus 4.7+ family, which reject them with a 400 — including
  on the source's own default model.
- `asyncio.run()` called from inside a synchronous function reachable from a
  running event loop (`memory_consolidator`).
- Process-global persona cache with no tenant key, leaking the first tenant's
  personas to every subsequent tenant.
- Embeddings stored but never searched; recall ordered by importance only.
- Model fallback with no backoff, turning a throttle into a hard failure.
- Token usage discarded, making per-client cost attribution impossible.
- Connection churn — up to six connections to resolve one task key.
- `json.dumps(None)` writing the string `"null"` instead of SQL NULL.
- Global FIFO audit pruning, letting one tenant evict another's audit trail.
- Only the first content block of a multi-block tool result captured.

### Fixed (found by running the suite against live Postgres)
- psycopg 3 returns `uuid.UUID`; the domain uses `str`. Ids from
  `impact_of()` did not compare equal to `ArtifactVersion.id`. Normalized at
  the `db` boundary.
- The persona resolution cascade had no ordering, so a role with both a
  generic and a workstream-specific persona resolved non-deterministically.
- Blocking store calls inside coroutines — 13 sites, 10 in the workflow
  engine. Every checkpoint stalled the event loop. Now offloaded, with a
  static test preventing regression.
