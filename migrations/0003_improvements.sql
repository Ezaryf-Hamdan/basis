-- basis 0003: hardening and capability improvements
--
-- Four independent additions, each safe to apply incrementally:
--
--   1. Optimistic locking on basis_workflow_runs — prevents two engine
--      instances from silently overwriting each other's checkpoint. The
--      version column starts at 1 for all existing rows and is incremented
--      on every save_run call. A writer whose SELECT saw version N and then
--      issues UPDATE WHERE version = N gets 0 rows if another writer got there
--      first, and raises ConcurrentModificationError rather than silently
--      losing data.
--
--   2. Short-term memory TTL — expires_at lets a note be written with a
--      finite lifetime so it does not need an explicit clear_short_term call.
--      read_short_term filters expired notes; prune_short_term deletes them.
--      Existing rows have expires_at = NULL (never expires), preserving the
--      previous behaviour for callers that do not set a TTL.
--
--   3. Full-text search index on agent_memories — enables the hybrid
--      vector + text RRF recall path in PgMemoryRepository. The generated
--      tsvector is kept in sync by Postgres so it cannot drift out of date the
--      way an application-maintained column could.
--
--   4. Project-scoped tool grants — project_id on agent_persona_tools makes
--      a grant specific to a (tenant, persona, project) triple rather than
--      (tenant, persona) globally. A NULL project_id means "applies to all
--      projects for this persona", matching the previous behaviour so existing
--      rows are unaffected.
--
-- Apply order: 0000/0001 then 0002 then 0003.

BEGIN;

-- ═══════════════════════════════════════════════════════════════════════════
-- 1. Workflow optimistic locking
-- ═══════════════════════════════════════════════════════════════════════════

ALTER TABLE basis_workflow_runs
    ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;

-- Backfill existing rows at version 1 (already the DEFAULT, but explicit to
-- avoid a silent 0 if the column was added with DEFAULT 0 in error).
UPDATE basis_workflow_runs SET version = 1 WHERE version IS NULL OR version = 0;


-- ═══════════════════════════════════════════════════════════════════════════
-- 2. Short-term memory TTL
-- ═══════════════════════════════════════════════════════════════════════════

ALTER TABLE agent_short_term_memory
    ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;

-- Partial index: only rows that can expire, so the index stays small and the
-- planner uses it for the cleanup query.
CREATE INDEX IF NOT EXISTS agent_short_term_memory_expires_idx
    ON agent_short_term_memory (tenant_id, expires_at)
    WHERE expires_at IS NOT NULL;


-- ═══════════════════════════════════════════════════════════════════════════
-- 3. Full-text search on long-term memories
-- ═══════════════════════════════════════════════════════════════════════════

-- Generated tsvector — Postgres keeps it in sync. The `english` config covers
-- stemming and stop words; switch to `simple` if memories may contain
-- non-English content that should not be stemmed.
ALTER TABLE agent_memories
    ADD COLUMN IF NOT EXISTS tsv TSVECTOR
        GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(content, ''))
        ) STORED;

CREATE INDEX IF NOT EXISTS agent_memories_fts_idx
    ON agent_memories USING GIN (tsv);


-- ═══════════════════════════════════════════════════════════════════════════
-- 4. Project-scoped tool grants
-- ═══════════════════════════════════════════════════════════════════════════

ALTER TABLE agent_persona_tools
    ADD COLUMN IF NOT EXISTS project_id UUID;

-- Index for the scoped lookup: first try (persona, tenant, project), fall
-- back to (persona, tenant, NULL) in application code.
CREATE INDEX IF NOT EXISTS agent_persona_tools_project_idx
    ON agent_persona_tools (persona_id, tenant_id, project_id)
    WHERE project_id IS NOT NULL;

COMMIT;
