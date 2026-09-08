"""The workflow engine.

**Built from scratch.**

Executes a `Workflow` layer by layer, checkpointing after every step, and
stops cleanly at an approval gate so the run can be resumed later - possibly in
a different process, days later, which is the whole point.

The execution contract:

  * A step runs only when every dependency is done. Layers come from the DAG
    (`graph.Workflow.layers`), so steps within a layer run concurrently up to
    ``max_parallel``.
  * Every step is checkpointed *before* and *after* it runs. The before-write
    is what makes a crashed run recoverable: on resume, a step left in
    ``RUNNING`` is known to have been interrupted rather than never started.
  * An approval step sets the run to ``WAITING_APPROVAL`` and returns. Nothing
    blocks, nothing polls, no thread is parked. `approve()` resumes it.
  * A step that exhausts its retries fails the run. Dependents are marked
    ``SKIPPED`` rather than left ``PENDING``, so "did not run because an
    upstream failed" is distinguishable from "not reached yet".

What this deliberately does not do: guarantee exactly-once side effects. If the
process dies after a step's side effect but before its checkpoint, a resume
re-runs that step. Steps should therefore be idempotent, and that requirement
is stated here rather than discovered later. Making it exactly-once needs a
transactional outbox or Temporal, which is the trade the deck already weighed.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ..concurrency import maybe_offload
from ..context import RunContext, bind
from ..errors import AuthorizationDenied, BasisError
from ..ports import AgentInvoker, TracerPort
from .graph import Step, StepInput, StepKind, Workflow
from .state import (
    CheckpointStore,
    InMemoryCheckpointStore,
    RunRecord,
    RunStatus,
    StepStatus,
    new_run_record,
)

__all__ = ["WorkflowEngine", "WorkflowRunError"]

log = logging.getLogger(__name__)


class WorkflowRunError(BasisError):
    """A run failed. Carries the step that caused it."""

    def __init__(self, run_id: str, step_id: str, message: str):
        self.run_id = run_id
        self.step_id = step_id
        super().__init__("run %s failed at step %r: %s" % (run_id, step_id, message))

    def marker(self) -> str:
        return "workflow_failed: %s" % self


def _now() -> datetime:
    return datetime.now(UTC)


class WorkflowEngine:
    """Runs workflows with durable checkpoints and human approval gates."""

    def __init__(
        self,
        *,
        invoker: AgentInvoker | None = None,
        store: CheckpointStore | None = None,
        max_parallel: int = 4,
        tracer: TracerPort | None = None,
    ):
        # An `AgentInvoker`, not an AgentRunner. The engine names only a
        # protocol, so an AGENT step could equally be served by a remote
        # worker or a queue consumer - `agents.AgentRunnerInvoker` is simply
        # the in-process implementation.
        self._invoker = invoker
        self._store = store or InMemoryCheckpointStore()
        self._max_parallel = max_parallel
        if tracer is None:
            from ..observability.tracing import Tracer

            tracer = Tracer()
        self._tracer = tracer

    # ── starting and resuming ──────────────────────────────────────────────

    async def start(
        self,
        workflow: Workflow,
        ctx: RunContext,
        payload: Mapping[str, Any] | None = None,
    ) -> RunRecord:
        """Begin a new run.

        Validates that every agent the workflow names is registered *before*
        creating the run, so a typo in a step fails immediately rather than
        halfway through, after billable work.
        """
        if self._invoker is not None:
            missing = workflow.missing_agents(self._invoker.known_agents())
            if missing:
                raise WorkflowDefinitionMissingAgents(workflow.name, missing)

        record = new_run_record(ctx, workflow.name, workflow.version, payload or {})
        await maybe_offload(self._store, self._store.create, record)
        return await self._drive(workflow, record, ctx)

    async def resume(
        self,
        workflow: Workflow,
        ctx: RunContext,
        run_id: str,
    ) -> RunRecord:
        """Continue an existing run from its checkpoints."""
        record = await maybe_offload(
            self._store, self._store.load, run_id, tenant_id=ctx.tenant_id
        )
        if record is None:
            raise KeyError(
                "no run %s for tenant %s" % (run_id, ctx.tenant_id)
            )
        if record.status.is_terminal:
            return record

        # A step interrupted mid-flight is retried, not assumed complete. It
        # has no output, so treating it as done would feed None downstream.
        for step_record in record.steps.values():
            if step_record.status is StepStatus.RUNNING:
                log.warning(
                    "run %s step %r was interrupted; retrying",
                    run_id,
                    step_record.step_id,
                )
                step_record.status = StepStatus.PENDING
                await maybe_offload(
                    self._store, self._store.save_step, record, step_record.step_id
                )

        return await self._drive(workflow, record, ctx)

    async def approve(
        self,
        workflow: Workflow,
        ctx: RunContext,
        run_id: str,
        step_id: str,
    ) -> RunRecord:
        """Satisfy an approval gate and continue."""
        record = await self._require_waiting(ctx, run_id, step_id)
        step = workflow.step(step_id)
        self._check_approver(ctx, step)

        step_record = record.step(step_id)
        step_record.status = StepStatus.APPROVED
        step_record.approved_by = ctx.user_id
        step_record.output = {"approved": True, "by": ctx.user_id}
        step_record.finished_at = _now()
        await maybe_offload(self._store, self._store.save_step, record, step_id)

        record.status = RunStatus.RUNNING
        await maybe_offload(self._store, self._store.save_run, record)
        return await self._drive(workflow, record, ctx)

    async def reject(
        self,
        workflow: Workflow,
        ctx: RunContext,
        run_id: str,
        step_id: str,
        *,
        reason: str = "",
    ) -> RunRecord:
        """Refuse an approval gate. Terminates the run as ``rejected``.

        A distinct terminal state from ``failed`` - a human declining is a
        business outcome, not an error, and reporting cannot conflate them.
        """
        record = await self._require_waiting(ctx, run_id, step_id)
        step = workflow.step(step_id)
        self._check_approver(ctx, step)

        step_record = record.step(step_id)
        step_record.status = StepStatus.REJECTED
        step_record.approved_by = ctx.user_id
        step_record.error = reason or "rejected"
        step_record.finished_at = _now()
        await maybe_offload(self._store, self._store.save_step, record, step_id)

        record.status = RunStatus.REJECTED
        record.error = reason or "rejected by %s" % ctx.user_id
        await maybe_offload(self._store, self._store.save_run, record)
        return record

    async def _require_waiting(
        self, ctx: RunContext, run_id: str, step_id: str
    ) -> RunRecord:
        record = await maybe_offload(
            self._store, self._store.load, run_id, tenant_id=ctx.tenant_id
        )
        if record is None:
            raise KeyError("no run %s for tenant %s" % (run_id, ctx.tenant_id))
        step_record = record.steps.get(step_id)
        if step_record is None or step_record.status is not StepStatus.WAITING_APPROVAL:
            raise WorkflowRunError(
                run_id, step_id, "step is not awaiting approval"
            )
        return record

    def _check_approver(self, ctx: RunContext, step: Step) -> None:
        """Enforce the step's approver roles.

        A service principal is refused outright: an approval gate exists to put
        a human in the loop, so letting an automation satisfy one defeats the
        control. This is the same reasoning as `ToolPolicy.allow_writes_for_service`.
        """
        if ctx.principal.is_service:
            raise AuthorizationDenied(
                "workflow_approval",
                "approve",
                "a service principal may not satisfy an approval gate",
            )
        if step.approver_roles and not (step.approver_roles & ctx.principal.roles):
            raise AuthorizationDenied(
                "workflow_approval",
                "approve",
                "approval of %r requires one of: %s"
                % (step.id, ", ".join(sorted(step.approver_roles))),
            )

    # ── the loop ───────────────────────────────────────────────────────────

    async def _drive(
        self,
        workflow: Workflow,
        record: RunRecord,
        ctx: RunContext,
    ) -> RunRecord:
        """Advance a run as far as it can go."""
        record.status = RunStatus.RUNNING
        await maybe_offload(self._store, self._store.save_run, record)

        with self._tracer.span(
            "workflow.run",
            ctx,
            **{
                "basis.workflow.name": workflow.name,
                "basis.workflow.version": workflow.version,
                "basis.workflow.run_id": record.run_id,
            },
        ):
            for layer in workflow.layers:
                runnable = [
                    workflow.step(sid)
                    for sid in layer
                    if not record.step(sid).is_done
                    and record.step(sid).status
                    not in (StepStatus.SKIPPED, StepStatus.REJECTED)
                ]
                if not runnable:
                    continue

                # An approval gate in this layer stops the run here. Steps
                # beside it in the same layer are left pending, not run - the
                # gate exists to hold the pipeline, and racing past its
                # siblings would partly defeat it.
                gates = [s for s in runnable if s.kind is StepKind.APPROVAL]
                if gates:
                    gate = gates[0]
                    gate_record = record.step(gate.id)
                    gate_record.status = StepStatus.WAITING_APPROVAL
                    gate_record.started_at = _now()
                    await maybe_offload(
                        self._store, self._store.save_step, record, gate.id
                    )
                    record.status = RunStatus.WAITING_APPROVAL
                    await maybe_offload(self._store, self._store.save_run, record)
                    log.info(
                        "run %s waiting for approval at step %r",
                        record.run_id,
                        gate.id,
                    )
                    return record

                results = await self._run_layer(runnable, record, ctx)

                failed = [(sid, err) for sid, err in results if err is not None]
                if failed:
                    step_id, err = failed[0]
                    record.status = RunStatus.FAILED
                    record.error = str(err)[:2000]
                    await self._mark_unreachable(workflow, record)
                    await maybe_offload(self._store, self._store.save_run, record)
                    raise WorkflowRunError(record.run_id, step_id, str(err))

            record.status = RunStatus.COMPLETE
            await maybe_offload(self._store, self._store.save_run, record)
            return record

    async def _run_layer(
        self,
        steps: Sequence[Step],
        record: RunRecord,
        ctx: RunContext,
    ) -> list[tuple[str, BaseException | None]]:
        """Run one layer with bounded concurrency."""
        sem = asyncio.Semaphore(self._max_parallel)

        async def _one(step: Step) -> tuple[str, BaseException | None]:
            async with sem:
                try:
                    await self._run_step(step, record, ctx)
                    return step.id, None
                except Exception as exc:
                    return step.id, exc

        return list(await asyncio.gather(*(_one(s) for s in steps)))

    async def _run_step(
        self,
        step: Step,
        record: RunRecord,
        ctx: RunContext,
    ) -> None:
        """Run one step, with its retry policy, checkpointing throughout."""
        step_record = record.step(step.id)
        upstream = {
            dep: record.step(dep).output
            for dep in step.depends_on
            if record.step(dep).is_done
        }

        task_input = StepInput(
            payload={**dict(record.payload), **dict(step.payload)},
            upstream=upstream,
            instructions=step.description,
        )

        # Each step gets its own child context, so its spans and any memory
        # writes carry a distinct run_id while inheriting the tenant.
        step_ctx = ctx.child(task_key=step.agent or step.id)

        last_error: BaseException | None = None
        attempt = 0

        while True:
            attempt += 1
            step_record.attempts = attempt
            step_record.status = StepStatus.RUNNING
            step_record.started_at = step_record.started_at or _now()
            # Checkpoint before running: a crash now leaves RUNNING behind,
            # which resume() recognises as interrupted.
            await maybe_offload(self._store, self._store.save_step, record, step.id)

            try:
                with self._tracer.span(
                    "workflow.step",
                    step_ctx,
                    **{
                        "basis.workflow.step_id": step.id,
                        "basis.workflow.step_kind": step.kind.value,
                        "basis.workflow.attempt": attempt,
                    },
                ):
                    output = await self._execute(step, step_ctx, task_input)
            except Exception as exc:
                last_error = exc
                if step.retry.should_retry(exc, attempt):
                    delay = step.retry.delay_for(attempt)
                    log.warning(
                        "step %r attempt %d failed (%s); retrying in %.1fs",
                        step.id,
                        attempt,
                        exc,
                        delay,
                    )
                    step_record.error = str(exc)[:2000]
                    await maybe_offload(
                        self._store, self._store.save_step, record, step.id
                    )
                    await asyncio.sleep(delay)
                    continue

                step_record.status = StepStatus.FAILED
                step_record.error = str(exc)[:2000]
                step_record.finished_at = _now()
                await maybe_offload(
                    self._store, self._store.save_step, record, step.id
                )
                raise

            step_record.status = StepStatus.COMPLETE
            step_record.output = output
            step_record.error = None
            step_record.finished_at = _now()
            await maybe_offload(self._store, self._store.save_step, record, step.id)
            return

        # Unreachable; the loop returns or raises.
        raise last_error  # pragma: no cover

    async def _execute(
        self,
        step: Step,
        step_ctx: RunContext,
        task_input: StepInput,
    ) -> Any:
        """Dispatch by step kind."""
        if step.kind is StepKind.AGENT:
            if self._invoker is None:
                raise WorkflowRunError(
                    "-", step.id, "AGENT step requires an AgentInvoker"
                )
            # Bound so the invoker's own model calls resolve against this
            # step's context rather than the parent run's.
            with bind(step_ctx):
                return await self._invoker.invoke_agent(
                    step.agent or "",
                    payload=task_input.payload,
                    upstream=task_input.upstream,
                    ctx=step_ctx,
                    instructions=task_input.instructions,
                )

        if step.kind is StepKind.TASK:
            fn = step.task
            with bind(step_ctx):
                outcome = fn(step_ctx, task_input)  # type: ignore[operator]
                if asyncio.iscoroutine(outcome):
                    return await outcome
                return outcome

        raise WorkflowRunError("-", step.id, "cannot execute kind %s" % step.kind)

    async def _mark_unreachable(self, workflow: Workflow, record: RunRecord) -> None:
        """Mark every step that can no longer run as SKIPPED.

        Leaving them PENDING would be ambiguous - a reader cannot tell "not
        reached yet" from "never will be". Walks the graph so transitive
        dependents are covered too.
        """
        changed = True
        while changed:
            changed = False
            for step_id, step in workflow.steps.items():
                rec = record.step(step_id)
                if rec.status not in (StepStatus.PENDING,):
                    continue
                blocked = any(
                    record.step(dep).status
                    in (StepStatus.FAILED, StepStatus.SKIPPED, StepStatus.REJECTED)
                    for dep in step.depends_on
                )
                if blocked:
                    rec.status = StepStatus.SKIPPED
                    await maybe_offload(
                        self._store, self._store.save_step, record, step_id
                    )
                    changed = True


class WorkflowDefinitionMissingAgents(BasisError):
    """A workflow names agents that are not registered."""

    def __init__(self, workflow_name: str, missing: frozenset[str]):
        self.missing = missing
        super().__init__(
            "workflow %r names unregistered agents: %s"
            % (workflow_name, ", ".join(sorted(missing)))
        )
