"""Run state and the checkpoint store.

**Built from scratch.**

The state model is deliberately small: a run is a row, each step attempt is a
row, and the engine reconstructs where it got to by reading them. That is what
makes a run resumable after a process restart - the thing `job_executor` cannot
do, because its progress lives in Python locals.

Two design choices worth stating:

**Step results are persisted, not just statuses.** Resuming a run means
re-supplying the completed steps' outputs to their dependents. If only
statuses were stored, a resume would have to re-run completed steps to
reproduce their outputs - which for an LLM step means paying for it twice and
possibly getting a different answer. So `stage_results`-style output storage is
mandatory, not an optimization.

**Terminal states are explicit and separate.** `failed` and `rejected` are
distinct, because "the model errored" and "a human said no" are different
outcomes that need different handling and different reporting. AILedSDLC
collapsed several of these into `failed` with a marker string prefix parsed by
the UI; carrying it in a column is what a status column is for.
"""
from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from ..context import RunContext

__all__ = [
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "PostgresCheckpointStore",
    "RunRecord",
    "RunStatus",
    "StepRecord",
    "StepStatus",
]


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETE = "complete"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (
            RunStatus.COMPLETE,
            RunStatus.FAILED,
            RunStatus.REJECTED,
            RunStatus.CANCELLED,
        )


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"


@dataclass
class StepRecord:
    """One step's state within a run."""

    step_id: str
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    output: Any = None
    error: str | None = None
    approved_by: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def is_done(self) -> bool:
        return self.status in (
            StepStatus.COMPLETE,
            StepStatus.APPROVED,
            StepStatus.SKIPPED,
        )


@dataclass
class RunRecord:
    """A workflow run."""

    run_id: str
    tenant_id: str
    workflow_name: str
    workflow_version: int
    status: RunStatus = RunStatus.PENDING
    project_id: str | None = None
    principal_id: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    steps: dict[str, StepRecord] = field(default_factory=dict)
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    # Monotone counter incremented on every save_run. Used by PgWorkflowRepository
    # for optimistic locking: a concurrent writer that loaded an older version
    # loses the race and gets ConcurrentModificationError rather than silently
    # overwriting checkpoint data.
    version: int = 1

    def step(self, step_id: str) -> StepRecord:
        record = self.steps.get(step_id)
        if record is None:
            record = StepRecord(step_id=step_id)
            self.steps[step_id] = record
        return record

    def completed_outputs(self) -> dict[str, Any]:
        """Outputs of every finished step, keyed by step id.

        This is what gets handed to a step as its upstream input, and what
        makes resume possible without re-running finished work.
        """
        return {
            sid: rec.output for sid, rec in self.steps.items() if rec.is_done
        }

    def pending_approval(self) -> str | None:
        for sid, rec in self.steps.items():
            if rec.status is StepStatus.WAITING_APPROVAL:
                return sid
        return None


@runtime_checkable
class CheckpointStore(Protocol):
    """Persistence for run state.

    A Protocol rather than a base class, and that is load-bearing:
    `storage.postgres.PgWorkflowRepository` satisfies it *structurally*, without
    importing this module, which is what keeps the storage layer from depending
    upward on `workflow`. A consumer can equally back it with something it
    already has - ai-core's `pipeline_runs` with its `stage_results jsonb` maps
    onto this directly.
    """

    def create(self, record: RunRecord) -> None: ...

    def load(self, run_id: str, *, tenant_id: str) -> RunRecord | None: ...

    def save_step(self, record: RunRecord, step_id: str) -> None: ...

    def save_run(self, record: RunRecord) -> None: ...


class InMemoryCheckpointStore:
    """Non-durable store. For tests, and for a single-process run that does
    not need to survive a restart.

    It is offered explicitly rather than as a test fixture because "I do not
    need durability here" is a legitimate choice, and forcing a Postgres
    dependency on a caller who does not need it is how a package becomes
    unusable.
    """

    #: No I/O, so callers may skip the executor hop. See
    #: `concurrency.maybe_offload`.
    blocking = False

    def __init__(self) -> None:
        self._runs: dict[tuple[str, str], RunRecord] = {}

    def create(self, record: RunRecord) -> None:
        self._runs[(record.tenant_id, record.run_id)] = record

    def load(self, run_id: str, *, tenant_id: str) -> RunRecord | None:
        return self._runs.get((tenant_id, run_id))

    def save_step(self, record: RunRecord, step_id: str) -> None:
        self._runs[(record.tenant_id, record.run_id)] = record

    def save_run(self, record: RunRecord) -> None:
        self._runs[(record.tenant_id, record.run_id)] = record


#: The durable implementation lives in `storage.postgres` alongside every other
#: repository, so this module holds no SQL. Imported lazily by name to keep
#: `workflow` free of a storage dependency.
def PostgresCheckpointStore(dsn: str | None = None, tables: Any = None) -> CheckpointStore:
    """Durable checkpoint store on Postgres.

    A function rather than a class so the import stays lazy - a consumer using
    the in-memory store never loads psycopg.
    """
    from ..storage.postgres import PgWorkflowRepository

    return PgWorkflowRepository(dsn=dsn, tables=tables)


def new_run_record(
    ctx: RunContext, workflow_name: str, workflow_version: int, payload: Mapping[str, Any]
) -> RunRecord:
    """Build a fresh RunRecord from a RunContext."""
    return RunRecord(
        run_id=str(uuid.uuid4()),
        tenant_id=ctx.tenant_id,
        workflow_name=workflow_name,
        workflow_version=workflow_version,
        project_id=ctx.project_id,
        principal_id=ctx.user_id,
        payload=dict(payload),
        created_at=datetime.now(UTC),
    )
