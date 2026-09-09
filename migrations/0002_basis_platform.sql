-- basis 0002: workflow, artifacts, knowledge
--
-- The four functions built from scratch, rather than lifted. Unlike 0001 this
-- creates only new `basis_*` tables and alters nothing, so it is safe to apply
-- to a database that already has other things in it.
--
-- Naming: every table is prefixed `basis_` precisely because a consumer may
-- already have `documents`, `chunks` or `audit_log` of its own. ai-core does
-- have all three, which is what the adapter in `basis.adapters.aicore` exists
-- to reconcile - it points basis at those tables instead of these.
--
-- Apply order: 0001 then 0002.

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;


-- ═══════════════════════════════════════════════════════════════════════════
-- WORKFLOW
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS basis_workflow_runs (
    id               UUID PRIMARY KEY,
    tenant_id        UUID        NOT NULL,
    project_id       UUID,
    workflow_name    TEXT        NOT NULL,
    workflow_version INTEGER     NOT NULL DEFAULT 1,
    -- pending | running | waiting_approval | complete | failed | rejected
    -- | cancelled. `failed` and `rejected` are separate on purpose: a model
    -- error and a human declining are different outcomes and reporting must
    -- not conflate them (AILedSDLC encoded this distinction in a marker string
    -- prefix parsed out of error_message).
    status           TEXT        NOT NULL DEFAULT 'pending',
    principal_id     TEXT,
    payload          JSONB       NOT NULL DEFAULT '{}',
    error            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS basis_workflow_runs_tenant_idx
    ON basis_workflow_runs (tenant_id, created_at DESC);

-- The resume query: find runs parked at a gate or interrupted mid-flight.
CREATE INDEX IF NOT EXISTS basis_workflow_runs_resumable_idx
    ON basis_workflow_runs (tenant_id, status)
    WHERE status IN ('running', 'waiting_approval', 'pending');


CREATE TABLE IF NOT EXISTS basis_workflow_steps (
    run_id      UUID        NOT NULL REFERENCES basis_workflow_runs(id) ON DELETE CASCADE,
    tenant_id   UUID        NOT NULL,
    step_id     TEXT        NOT NULL,
    status      TEXT        NOT NULL DEFAULT 'pending',
    attempts    INTEGER     NOT NULL DEFAULT 0,
    -- Wrapped as {"value": ...} so a NULL output round-trips as None rather
    -- than as a JSON null indistinguishable from "never set".
    output      JSONB,
    error       TEXT,
    approved_by TEXT,
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    PRIMARY KEY (run_id, step_id)
);

CREATE INDEX IF NOT EXISTS basis_workflow_steps_tenant_idx
    ON basis_workflow_steps (tenant_id, run_id);


-- ═══════════════════════════════════════════════════════════════════════════
-- ARTIFACTS
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS basis_artifact_versions (
    id            UUID PRIMARY KEY,
    tenant_id     UUID        NOT NULL,
    project_id    UUID,
    -- Stable logical identity across versions. (tenant_id, artifact_key)
    -- names the artefact; `version` names this revision of it.
    artifact_key  TEXT        NOT NULL,
    kind          TEXT        NOT NULL,
    version       INTEGER     NOT NULL,
    title         TEXT,
    content       JSONB,
    -- SHA-256 over canonicalized content. This column is what makes external
    -- edit detection possible: on re-ingestion, a differing hash means the
    -- artefact changed underneath the platform. Without it, a re-upload of
    -- the same file is indistinguishable from a substantive edit.
    content_hash  TEXT        NOT NULL,
    -- draft | in_review | approved | rejected | superseded
    state         TEXT        NOT NULL DEFAULT 'draft',
    supersedes_id UUID REFERENCES basis_artifact_versions(id) ON DELETE SET NULL,
    created_by    TEXT,
    approved_by   TEXT,
    approved_at   TIMESTAMPTZ,
    run_id        UUID,
    metadata      JSONB       NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Versions are immutable and densely numbered per artefact. This is the
    -- constraint that stops two concurrent writers both creating "version 3";
    -- the store also takes FOR UPDATE on the head row.
    UNIQUE (tenant_id, artifact_key, version)
);

CREATE INDEX IF NOT EXISTS basis_artifact_head_idx
    ON basis_artifact_versions (tenant_id, artifact_key, version DESC);

CREATE INDEX IF NOT EXISTS basis_artifact_project_kind_idx
    ON basis_artifact_versions (tenant_id, project_id, kind);

-- Drift lookups go by hash.
CREATE INDEX IF NOT EXISTS basis_artifact_hash_idx
    ON basis_artifact_versions (tenant_id, content_hash);

-- "What is awaiting review" is an operator's standing question.
CREATE INDEX IF NOT EXISTS basis_artifact_review_idx
    ON basis_artifact_versions (tenant_id, state)
    WHERE state IN ('draft', 'in_review');


-- Lineage is a separate edge table, NOT a parent pointer on the version row.
-- A version chain answers "what came before this"; lineage answers "what was
-- this derived from", which is a different graph - an FSD derived from three
-- requirements and a fit/gap has four lineage edges and one predecessor.
-- Impact analysis traverses this table, and it is why impact analysis was
-- impossible in either source system.
CREATE TABLE IF NOT EXISTS basis_artifact_lineage (
    id         UUID PRIMARY KEY,
    tenant_id  UUID        NOT NULL,
    target_id  UUID        NOT NULL REFERENCES basis_artifact_versions(id) ON DELETE CASCADE,
    source_id  UUID        NOT NULL REFERENCES basis_artifact_versions(id) ON DELETE CASCADE,
    -- derived_from | cites | revises | extracted_from
    kind       TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (target_id, source_id, kind)
);

-- Downstream traversal (impact analysis) walks source -> target.
CREATE INDEX IF NOT EXISTS basis_lineage_source_idx
    ON basis_artifact_lineage (source_id, kind);

-- Upstream traversal (provenance) walks target -> source.
CREATE INDEX IF NOT EXISTS basis_lineage_target_idx
    ON basis_artifact_lineage (target_id, kind);


-- A baseline is a named, frozen set of versions - what "approved as of the
-- design freeze" means. Without it, "approved" drifts as individual artefacts
-- are re-approved.
CREATE TABLE IF NOT EXISTS basis_artifact_baselines (
    id         UUID PRIMARY KEY,
    tenant_id  UUID        NOT NULL,
    project_id UUID,
    name       TEXT        NOT NULL,
    created_by TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (tenant_id, project_id, name)
);

CREATE TABLE IF NOT EXISTS basis_artifact_baseline_members (
    baseline_id UUID NOT NULL REFERENCES basis_artifact_baselines(id) ON DELETE CASCADE,
    tenant_id   UUID NOT NULL,
    version_id  UUID NOT NULL REFERENCES basis_artifact_versions(id) ON DELETE CASCADE,
    PRIMARY KEY (baseline_id, version_id)
);


-- ═══════════════════════════════════════════════════════════════════════════
-- KNOWLEDGE
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS basis_chunks (
    id            UUID PRIMARY KEY,
    -- NULL only for corpus='public_domain', which is readable by any tenant.
    -- Every other corpus is tenant-scoped and the CHECK below enforces it.
    tenant_id     UUID,
    project_id    UUID,
    collection_id UUID,
    document_id   UUID,
    -- client_project | org_assets | public_domain
    -- This is a permission boundary, not a filing convenience: client project
    -- content must never surface in another client's retrieval, so the corpus
    -- is part of every WHERE clause in retrieval.py.
    corpus        TEXT        NOT NULL,
    ordinal       INTEGER     NOT NULL DEFAULT 0,
    title         TEXT,
    content       TEXT        NOT NULL,
    embedding     VECTOR(1024),
    -- Generated, not application-maintained: a denormalized tsvector that the
    -- application has to remember to update is a tsvector that goes stale.
    fts           TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    metadata      JSONB       NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Fail-closed at the schema level: a tenant-scoped corpus cannot be
    -- written without a tenant, and a project corpus cannot be written
    -- without a project. Enforcing it here means a buggy ingester cannot
    -- create a row that leaks across tenants.
    CONSTRAINT basis_chunks_scope_ck CHECK (
        (corpus = 'public_domain'  AND tenant_id IS NULL)
     OR (corpus = 'org_assets'     AND tenant_id IS NOT NULL)
     OR (corpus = 'client_project' AND tenant_id IS NOT NULL AND project_id IS NOT NULL)
    )
);

-- The retrieval scope predicate, in the order the queries filter.
CREATE INDEX IF NOT EXISTS basis_chunks_scope_idx
    ON basis_chunks (corpus, tenant_id, project_id, collection_id);

-- Vector arm. vector_cosine_ops matches the <=> operator in retrieval.py;
-- an index built with a different opclass is silently not used.
CREATE INDEX IF NOT EXISTS basis_chunks_embedding_idx
    ON basis_chunks USING hnsw (embedding vector_cosine_ops);

-- Lexical arm. GIN over the generated column.
CREATE INDEX IF NOT EXISTS basis_chunks_fts_idx
    ON basis_chunks USING gin (fts);

-- Re-ingestion replaces a document's chunks wholesale.
CREATE INDEX IF NOT EXISTS basis_chunks_document_idx
    ON basis_chunks (document_id, ordinal);

COMMIT;
