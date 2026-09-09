-- basis 0001: tenancy spine + tool invocation audit
--
-- This is the schema basis expects. It is NOT a no-op against AILedSDLC's
-- existing database: every table below gains a tenant_id, and three unique
-- constraints change shape as a result. Read the notes before applying.
--
-- Why the tenant column is not optional: the lifted queries filtered on
-- (user_id, project_id, persona_id) with no tenant predicate, and
-- agent_tools / agent_personas had no scoping at all. That is the concrete
-- form of the Data Isolation and Multi-Client Readiness gaps - a persona tuned
-- for one client applied to every client, and a memory recall could surface
-- another tenant's content if project ids collided.
--
-- Applying this to an existing AILedSDLC database requires a backfill value
-- for tenant_id. Set BASIS_BACKFILL_TENANT below to the single tenant that
-- the existing rows belong to before running.

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

-- Change this before applying to a database with existing rows.
\set BASIS_BACKFILL_TENANT '00000000-0000-0000-0000-000000000000'


-- ── agent_personas ──────────────────────────────────────────────────────────
ALTER TABLE agent_personas
    ADD COLUMN IF NOT EXISTS tenant_id UUID;

UPDATE agent_personas SET tenant_id = :'BASIS_BACKFILL_TENANT'
    WHERE tenant_id IS NULL;

ALTER TABLE agent_personas
    ALTER COLUMN tenant_id SET NOT NULL;

-- resolver.py matches on (tenant_id, role, workstream); is_active filters.
CREATE INDEX IF NOT EXISTS agent_personas_tenant_role_idx
    ON agent_personas (tenant_id, role, workstream)
    WHERE is_active;


-- ── agent_task_models ───────────────────────────────────────────────────────
ALTER TABLE agent_task_models
    ADD COLUMN IF NOT EXISTS tenant_id UUID;

UPDATE agent_task_models SET tenant_id = :'BASIS_BACKFILL_TENANT'
    WHERE tenant_id IS NULL;

ALTER TABLE agent_task_models
    ALTER COLUMN tenant_id SET NOT NULL;

-- task_key was globally unique. It must be unique per tenant instead, or one
-- client's task tuning overwrites another's.
ALTER TABLE agent_task_models
    DROP CONSTRAINT IF EXISTS agent_task_models_task_key_key;

CREATE UNIQUE INDEX IF NOT EXISTS agent_task_models_tenant_task_key
    ON agent_task_models (tenant_id, task_key);


-- ── agent_tools ─────────────────────────────────────────────────────────────
ALTER TABLE agent_tools
    ADD COLUMN IF NOT EXISTS tenant_id UUID,
    -- ToolPolicy.classify() result, persisted so an audit query can filter by
    -- effect without re-deriving it from the name.
    ADD COLUMN IF NOT EXISTS effect TEXT NOT NULL DEFAULT 'unknown';

UPDATE agent_tools SET tenant_id = :'BASIS_BACKFILL_TENANT'
    WHERE tenant_id IS NULL;

ALTER TABLE agent_tools
    ALTER COLUMN tenant_id SET NOT NULL;

-- registry.sync_from_mcp() upserts ON CONFLICT (tenant_id, tool_name).
ALTER TABLE agent_tools
    DROP CONSTRAINT IF EXISTS agent_tools_tool_name_key;

CREATE UNIQUE INDEX IF NOT EXISTS agent_tools_tenant_name_key
    ON agent_tools (tenant_id, tool_name);


-- ── agent_persona_tools ─────────────────────────────────────────────────────
-- The per-tenant grant table. Without tenant_id, enabling a tool for a persona
-- enabled it for every client that persona served.
ALTER TABLE agent_persona_tools
    ADD COLUMN IF NOT EXISTS tenant_id UUID;

UPDATE agent_persona_tools SET tenant_id = :'BASIS_BACKFILL_TENANT'
    WHERE tenant_id IS NULL;

ALTER TABLE agent_persona_tools
    ALTER COLUMN tenant_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS agent_persona_tools_lookup_idx
    ON agent_persona_tools (tenant_id, persona_id)
    WHERE enabled;


-- ── agent_memories ──────────────────────────────────────────────────────────
ALTER TABLE agent_memories
    ADD COLUMN IF NOT EXISTS tenant_id UUID,
    ADD COLUMN IF NOT EXISTS run_id UUID,
    ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ;

UPDATE agent_memories SET tenant_id = :'BASIS_BACKFILL_TENANT'
    WHERE tenant_id IS NULL;

ALTER TABLE agent_memories
    ALTER COLUMN tenant_id SET NOT NULL;

-- The recall predicate, in the order the query filters.
CREATE INDEX IF NOT EXISTS agent_memories_scope_idx
    ON agent_memories (tenant_id, user_id, project_id, persona_id, importance DESC);

-- Vector recall. This index is why MemoryService.recall can rank by cosine
-- distance at all - the source repo stored embeddings with no ANN index and
-- never queried them, so nothing revealed the omission.
-- vector_cosine_ops matches the <=> operator used in service.py.
CREATE INDEX IF NOT EXISTS agent_memories_embedding_idx
    ON agent_memories USING hnsw (embedding vector_cosine_ops);


-- ── agent_short_term_memory ─────────────────────────────────────────────────
ALTER TABLE agent_short_term_memory
    ADD COLUMN IF NOT EXISTS tenant_id UUID,
    ADD COLUMN IF NOT EXISTS run_id UUID;

UPDATE agent_short_term_memory SET tenant_id = :'BASIS_BACKFILL_TENANT'
    WHERE tenant_id IS NULL;

ALTER TABLE agent_short_term_memory
    ALTER COLUMN tenant_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS agent_short_term_memory_job_idx
    ON agent_short_term_memory (tenant_id, job_id, persona_id, created_at);


-- ── agent_conversations ─────────────────────────────────────────────────────
ALTER TABLE agent_conversations
    ADD COLUMN IF NOT EXISTS tenant_id UUID,
    ADD COLUMN IF NOT EXISTS project_id UUID;

UPDATE agent_conversations SET tenant_id = :'BASIS_BACKFILL_TENANT'
    WHERE tenant_id IS NULL;

ALTER TABLE agent_conversations
    ALTER COLUMN tenant_id SET NOT NULL;

-- session_id was globally unique, so a session id from one tenant could
-- collide with another's and the upsert in conversation.py would overwrite it.
ALTER TABLE agent_conversations
    DROP CONSTRAINT IF EXISTS agent_conversations_session_id_key;

CREATE UNIQUE INDEX IF NOT EXISTS agent_conversations_tenant_session_key
    ON agent_conversations (tenant_id, session_id);


-- ── agent_tool_invocations (new) ────────────────────────────────────────────
-- The audit record that did not exist. tool_access.py filtered the catalog but
-- never recorded that a tool ran, so there was no answer to "what did the
-- agent do on this client's data".
CREATE TABLE IF NOT EXISTS agent_tool_invocations (
    id              UUID PRIMARY KEY,
    tenant_id       UUID        NOT NULL,
    project_id      UUID,
    run_id          UUID        NOT NULL,
    job_id          UUID,
    persona_id      UUID,
    principal_id    TEXT        NOT NULL,
    principal_kind  TEXT        NOT NULL,
    tool_name       TEXT        NOT NULL,
    effect          TEXT        NOT NULL,
    allowed         BOOLEAN     NOT NULL,
    denied_reason   TEXT,
    duration_ms     INTEGER,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS agent_tool_invocations_tenant_time_idx
    ON agent_tool_invocations (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS agent_tool_invocations_run_idx
    ON agent_tool_invocations (run_id);

-- Denials are the rows an operator actually goes looking for.
CREATE INDEX IF NOT EXISTS agent_tool_invocations_denied_idx
    ON agent_tool_invocations (tenant_id, created_at DESC)
    WHERE NOT allowed;

COMMIT;
