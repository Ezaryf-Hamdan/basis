"""Shared fixtures.

The integration fixtures run against a real Postgres. They are skipped unless
``INTEGRATION_TESTS=1`` so the default suite stays hermetic, and they truncate
between tests rather than recreating the schema, which keeps the suite fast
enough to actually run.
"""
from __future__ import annotations

import os

import pytest

from basis import Principal, RunContext
from basis.settings import reset_settings

INTEGRATION = os.environ.get("INTEGRATION_TESTS") == "1"

DEFAULT_DSN = "postgresql://postgres:postgres@127.0.0.1:5432/basis_test"

#: Every table the integration tests write to. Truncated between tests.
_TABLES = (
    "agent_tool_invocations",
    "agent_persona_tools",
    "agent_tools",
    "agent_short_term_memory",
    "agent_memories",
    "agent_conversations",
    "agent_task_models",
    "agent_personas",
    "basis_workflow_steps",
    "basis_workflow_runs",
    "basis_artifact_baseline_members",
    "basis_artifact_baselines",
    "basis_artifact_lineage",
    "basis_artifact_versions",
    "basis_chunks",
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires a live Postgres with pgvector; set "
        "INTEGRATION_TESTS=1 to run",
    )


def pytest_collection_modifyitems(config, items):
    if INTEGRATION:
        return
    skip = pytest.mark.skip(reason="set INTEGRATION_TESTS=1 and provide BASIS_TEST_DSN")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def dsn() -> str:
    return os.environ.get("BASIS_TEST_DSN", DEFAULT_DSN)


@pytest.fixture(autouse=True)
def _clean_settings():
    """Settings are process-cached; drop them so env changes take effect."""
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def pg(dsn):
    """A clean database for one test.

    Truncates every basis table. ``CASCADE`` because the artifact and workflow
    tables have FK chains, and ``RESTART IDENTITY`` so serial columns do not
    drift across tests.
    """
    from basis import db

    db.close_pools()
    with db.cursor(dsn, dict_rows=False) as cur:
        cur.execute(
            "TRUNCATE %s RESTART IDENTITY CASCADE" % ", ".join(_TABLES)
        )
    yield dsn
    db.close_pools()


@pytest.fixture
def ctx() -> RunContext:
    """A run context with fixed uuids, so failures are reproducible."""
    return RunContext(
        principal=Principal(
            subject_id="11111111-1111-1111-1111-111111111111",
            tenant_id="aaaaaaaa-0000-0000-0000-000000000001",
        ),
        run_id="99999999-9999-9999-9999-999999999999",
        project_id="bbbbbbbb-0000-0000-0000-000000000001",
        persona_id="cccccccc-0000-0000-0000-000000000001",
        job_id="dddddddd-0000-0000-0000-000000000001",
        session_id="session-1",
    )


@pytest.fixture
def other_ctx() -> RunContext:
    """A second tenant, for isolation assertions."""
    return RunContext(
        principal=Principal(
            subject_id="22222222-2222-2222-2222-222222222222",
            tenant_id="aaaaaaaa-0000-0000-0000-000000000002",
        ),
        project_id="bbbbbbbb-0000-0000-0000-000000000002",
        persona_id="cccccccc-0000-0000-0000-000000000002",
        job_id="dddddddd-0000-0000-0000-000000000002",
        session_id="session-1",  # deliberately the same session id
    )
