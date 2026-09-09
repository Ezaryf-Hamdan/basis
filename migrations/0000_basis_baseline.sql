-- basis 0000: greenfield baseline
--
-- Creates every table basis needs from nothing. This is the migration a NEW
-- consumer runs; `0001_ailedsdlc_retrofit.sql` is the alternative path for a
-- database that already has AILedSDLC's `agent_*` tables and needs them
-- altered instead.
--
-- Apply order:
--   greenfield          ->  0000, then 0002
--   existing AILedSDLC  ->  0001, then 0002   (skip 0000)
--
-- Every table is tenant-scoped from the start. That is the whole point: the
-- source schema had no tenant column anywhere, which is why per-client
-- isolation and cost attribution were not enforceable rather than merely
-- unenforced.

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;


-- ═══════════════════════════════════════════════════════════════════════════
-- MODEL / PERSONA CONFIGURATION
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS agent_personas (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id             UUID        NOT NULL,
    name                  TEXT        NOT NULL,
    role                  TEXT        NOT NULL,
    -- Optional sub-specialisation within a role. NULL means the role-level
    -- persona, which is the fallback in the resolution cascade.
    workstream            TEXT,
    system_prompt         TEXT,
    model_id              TEXT,
    model_config          JSONB       NOT NULL DEFAULT '{}',
    fallback_model_ids    TEXT[]      NOT NULL DEFAULT '{}',
    fallback_model_config JSONB       NOT NULL DEFAULT '{}',
    is_active             BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- The resolver matches on (tenant, role, workstream) with is_active as a
-- filter, so the partial index mirrors the query exactly.
CREATE INDEX IF NOT EXISTS agent_personas_tenant_role_idx
    ON agent_personas (tenant_id, role, workstream)
    WHERE is_active;


CREATE TABLE IF NOT EXISTS agent_task_models (
    tenant_id             UUID        NOT NULL,
    task_key              TEXT        NOT NULL,
    model_id              TEXT,
    model_config          JSONB       NOT NULL DEFAULT '{}',
    system_prompt         TEXT,
    fallback_model_ids    TEXT[]      NOT NULL DEFAULT '{}',
    fallback_model_config JSONB       NOT NULL DEFAULT '{}',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Per tenant, not global. A client tuning a task onto a cheaper model must
    -- not change it for every other client - which is exactly what the
    -- source's global UNIQUE(task_key) did.
    PRIMARY KEY (tenant_id, task_key)
);


-- ═══════════════════════════════════════════════════════════════════════════
-- TOOLS
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS agent_tools (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id    UUID        NOT NULL,
    tool_name    TEXT        NOT NULL,
    description  TEXT        NOT NULL DEFAULT '',
    schema       JSONB       NOT NULL DEFAULT '{}',
    -- ToolPolicy.classify() result, persisted so an audit query can filter by
    -- effect without re-deriving it from the name.
    effect       TEXT        NOT NULL DEFAULT 'unknown',
    source       TEXT        NOT NULL DEFAULT 'mcp',
    is_active    BOOLEAN     NOT NULL DEFAULT TRUE,
    last_seen_at TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- registry.sync_from_mcp upserts ON CONFLICT (tenant_id, tool_name).
    UNIQUE (tenant_id, tool_name)
);


CREATE TABLE IF NOT EXISTS agent_persona_tools (
    tenant_id  UUID    NOT NULL,
    persona_id UUID    NOT NULL REFERENCES agent_personas(id) ON DELETE CASCADE,
    tool_id    UUID    NOT NULL REFERENCES agent_tools(id) ON DELETE CASCADE,
    enabled    BOOLEAN NOT NULL DEFAULT TRUE,

    PRIMARY KEY (tenant_id, persona_id, tool_id)
);

CREATE INDEX IF NOT EXISTS agent_persona_tools_lookup_idx
    ON agent_persona_tools (tenant_id, persona_id)
    WHERE enabled;


-- The audit record that did not exist in the source. tool_access.py filtered
-- the catalog but never recorded that a tool ran, so there was no answer to
-- "what did the agent do on this client's data".
CREATE TABLE IF NOT EXISTS agent_tool_invocations (
    id             UUID PRIMARY KEY,
    tenant_id      UUID        NOT NULL,
    project_id     UUID,
    run_id         UUID        NOT NULL,
    job_id         UUID,
    persona_id     UUID,
    principal_id   TEXT        NOT NULL,
    principal_kind TEXT        NOT NULL,
    tool_name      TEXT        NOT NULL,
    effect         TEXT        NOT NULL,
    allowed        BOOLEAN     NOT NULL,
    denied_reason  TEXT,
    duration_ms    INTEGER,
    error          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS agent_tool_invocations_tenant_time_idx
    ON agent_tool_invocations (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS agent_tool_invocations_run_idx
    ON agent_tool_invocations (run_id);

-- Denials are the rows an operator actually goes looking for.
CREATE INDEX IF NOT EXISTS agent_tool_invocations_denied_idx
    ON agent_tool_invocations (tenant_id, created_at DESC)
    WHERE NOT allowed;


-- ═══════════════════════════════════════════════════════════════════════════
-- MEMORY
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS agent_short_term_memory (
    id         UUID PRIMARY KEY,
    tenant_id  UUID        NOT NULL,
    job_id     UUID        NOT NULL,
    run_id     UUID,
    persona_id UUID        NOT NULL,
    user_id    TEXT        NOT NULL,
    project_id UUID        NOT NULL,
    note_type  TEXT        NOT NULL,
    content    TEXT        NOT NULL,
    metadata   JSONB       NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS agent_short_term_memory_job_idx
    ON agent_short_term_memory (tenant_id, job_id, persona_id, created_at);


CREATE TABLE IF NOT EXISTS agent_memories (
    id               UUID PRIMARY KEY,
    tenant_id        UUID        NOT NULL,
    user_id          TEXT        NOT NULL,
    project_id       UUID        NOT NULL,
    persona_id       UUID        NOT NULL,
    job_id           UUID,
    run_id           UUID,
    memory_type      TEXT        NOT NULL,
    content          TEXT        NOT NULL,
    -- Nullable: a memory written while the embedding endpoint was unavailable
    -- is still readable via the importance-ordered fallback path, and losing it
    -- entirely would be worse.
    embedding        VECTOR(1024),
    importance       REAL        NOT NULL DEFAULT 0.5,
    source           TEXT        NOT NULL DEFAULT 'consolidation',
    access_count     INTEGER     NOT NULL DEFAULT 0,
    metadata         JSONB       NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS agent_memories_scope_idx
    ON agent_memories (tenant_id, user_id, project_id, persona_id, importance DESC);

-- Vector recall. This index is why MemoryService.recall can rank by cosine
-- distance at all - the source stored embeddings with no ANN index and never
-- queried them, so nothing revealed the omission.
-- vector_cosine_ops matches the <=> operator; a different opclass is silently
-- not used.
CREATE INDEX IF NOT EXISTS agent_memories_embedding_idx
    ON agent_memories USING hnsw (embedding vector_cosine_ops);


CREATE TABLE IF NOT EXISTS agent_conversations (
    tenant_id  UUID        NOT NULL,
    session_id TEXT        NOT NULL,
    user_id    TEXT        NOT NULL,
    project_id UUID,
    persona_id UUID,
    messages   JSONB       NOT NULL DEFAULT '[]',
    context    JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Composite, not session_id alone: a global session id lets one tenant's
    -- id collide with another's, and the upsert would overwrite it.
    PRIMARY KEY (tenant_id, session_id)
);

COMMIT;
