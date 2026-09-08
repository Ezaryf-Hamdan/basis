"""Workflow definition: steps, DAG validation, execution order.

**Built from scratch.** Nothing in AILedSDLC corresponds to this.
`job_executor.py` is 4,396 lines with a 27-entry `job_type` dispatch, and the
sequencing for each job type is written inline inside its handler. That means:
no durable execution, no checkpoints, no resume, no approval gates, and no way
to see a job's shape without reading its handler.

The deck recommends a lightweight custom DAG engine for MVP over Temporal, and
this is that engine. The deliberate scope limit: it persists state and resumes,
but it does **not** provide exactly-once side effects or distributed workers.
Those are the things Temporal exists for, and pretending a few hundred lines
supply them would be worse than not having them.

Validation happens at definition time, not run time. A workflow with a cycle,
a dangling dependency, or a duplicate step id fails when it is constructed -
so a malformed pipeline cannot reach production and fail on a customer's data.
Unlike agent delegations (which are dynamic and can only be bounded at run
time, see `agents/registry.py`), workflow edges are static and therefore
genuinely checkable.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..errors import BasisError

__all__ = [
    "RetryPolicy",
    "Step",
    "StepInput",
    "StepKind",
    "Workflow",
    "WorkflowDefinitionError",
]


class WorkflowDefinitionError(BasisError):
    """A workflow is structurally invalid. Raised at definition time."""


@dataclass(frozen=True)
class StepInput:
    """What a step is given to work on.

    Neutral by design. The first cut passed ``agents.AgentInput`` here, which
    made ``workflow`` import ``agents`` - so the engine could not drive a step
    that was not a basis Agent, and the two functions were welded together.
    ``upstream`` is keyed by step id, so a step never learns which agent (or
    remote worker, or plain function) produced its inputs.
    """

    payload: Mapping[str, Any] = field(default_factory=dict)
    upstream: Mapping[str, Any] = field(default_factory=dict)
    instructions: str = ""


class StepKind(str, Enum):
    """What a step does when the engine reaches it."""

    #: Run a registered agent.
    AGENT = "agent"
    #: Run a plain async callable. For glue that is not agent work - fetching,
    #: transforming, writing an artifact.
    TASK = "task"
    #: Stop and wait for a human decision. The run persists and resumes when
    #: `WorkflowEngine.approve` / `.reject` is called.
    APPROVAL = "approval"


@dataclass(frozen=True)
class RetryPolicy:
    """How to retry a failing step.

    Separate from the model-level retry in `models.retry`: that handles a
    throttled Bedrock call inside one invocation, this handles a step that
    failed for any reason. A step wrapping a model call can be retried by both,
    which is intended - the inner one absorbs transient capacity, the outer one
    absorbs everything else.
    """

    max_attempts: int = 1
    base_delay: float = 1.0
    max_delay: float = 30.0
    #: Exception types that should be retried. Empty means "retry any
    #: exception". Naming them is strongly preferred: retrying a
    #: ValidationError just burns budget to fail again.
    retry_on: tuple[type[BaseException], ...] = ()

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False
        if not self.retry_on:
            return True
        return isinstance(exc, self.retry_on)

    def delay_for(self, attempt: int) -> float:
        return min(self.max_delay, self.base_delay * (2 ** max(0, attempt - 1)))


NO_RETRY = RetryPolicy(max_attempts=1)


@dataclass(frozen=True)
class Step:
    """One node in a workflow.

    ``depends_on`` names the steps whose output this step needs. Their results
    arrive in `AgentInput.upstream` keyed by step id, so a step never knows
    *how* its inputs were produced - only that they are there.
    """

    id: str
    kind: StepKind = StepKind.AGENT
    #: Agent name for AGENT steps.
    agent: str | None = None
    #: Async callable ``(RunContext, Mapping[str, Any]) -> Any`` for TASK steps.
    task: object | None = None
    depends_on: tuple[str, ...] = ()
    retry: RetryPolicy = NO_RETRY
    #: Static payload merged into the step's input.
    payload: Mapping[str, object] = field(default_factory=dict)
    #: Who may satisfy an APPROVAL step. Empty means any authorized caller.
    approver_roles: frozenset[str] = field(default_factory=frozenset)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise WorkflowDefinitionError("step id must be non-empty")
        if self.kind is StepKind.AGENT and not self.agent:
            raise WorkflowDefinitionError(
                "step %r is an AGENT step and must name an agent" % self.id
            )
        if self.kind is StepKind.TASK and self.task is None:
            raise WorkflowDefinitionError(
                "step %r is a TASK step and must supply a callable" % self.id
            )
        if self.kind is StepKind.APPROVAL and (self.agent or self.task):
            raise WorkflowDefinitionError(
                "step %r is an APPROVAL step and must not carry an agent or task"
                % self.id
            )
        if self.id in self.depends_on:
            raise WorkflowDefinitionError("step %r depends on itself" % self.id)


class Workflow:
    """A validated DAG of steps."""

    def __init__(self, name: str, steps: Sequence[Step], *, version: int = 1):
        if not name:
            raise WorkflowDefinitionError("workflow name must be non-empty")
        if not steps:
            raise WorkflowDefinitionError("workflow %r has no steps" % name)

        self.name = name
        self.version = version
        self._steps: dict[str, Step] = {}

        for step in steps:
            if step.id in self._steps:
                raise WorkflowDefinitionError(
                    "workflow %r has duplicate step id %r" % (name, step.id)
                )
            self._steps[step.id] = step

        self._validate_edges()
        self._order = self._topological_layers()

    def _validate_edges(self) -> None:
        for step in self._steps.values():
            for dep in step.depends_on:
                if dep not in self._steps:
                    raise WorkflowDefinitionError(
                        "step %r depends on unknown step %r" % (step.id, dep)
                    )

    def _topological_layers(self) -> tuple[tuple[str, ...], ...]:
        """Group steps into layers that can run concurrently.

        Kahn's algorithm, kept layered rather than flat because the layering is
        the parallelism plan: everything in a layer has its dependencies
        satisfied, so the engine can run a layer with bounded concurrency. A
        flat topological order would serialize work that has no reason to be
        serial - and bulk fan-out is the normal case here.

        A leftover set at the end means a cycle, which is reported with the
        steps involved rather than a bare "cycle detected".
        """
        indegree = {sid: len(s.depends_on) for sid, s in self._steps.items()}
        dependents: dict[str, list[str]] = {sid: [] for sid in self._steps}
        for sid, step in self._steps.items():
            for dep in step.depends_on:
                dependents[dep].append(sid)

        layers: list[tuple[str, ...]] = []
        ready = sorted(sid for sid, deg in indegree.items() if deg == 0)
        if not ready:
            raise WorkflowDefinitionError(
                "workflow %r has no entry step; every step has a dependency, "
                "which means a cycle" % self.name
            )

        remaining = dict(indegree)
        while ready:
            layers.append(tuple(ready))
            next_ready: list[str] = []
            for sid in ready:
                remaining.pop(sid, None)
                for child in dependents[sid]:
                    remaining[child] -= 1
                    if remaining[child] == 0:
                        next_ready.append(child)
            ready = sorted(next_ready)

        if remaining:
            raise WorkflowDefinitionError(
                "workflow %r contains a cycle among steps: %s"
                % (self.name, sorted(remaining))
            )
        return tuple(layers)

    # ── accessors ──────────────────────────────────────────────────────────

    @property
    def layers(self) -> tuple[tuple[str, ...], ...]:
        """Steps grouped into concurrently-runnable layers, in order."""
        return self._order

    @property
    def steps(self) -> Mapping[str, Step]:
        return dict(self._steps)

    def step(self, step_id: str) -> Step:
        try:
            return self._steps[step_id]
        except KeyError:
            raise KeyError(
                "workflow %r has no step %r" % (self.name, step_id)
            ) from None

    def order(self) -> tuple[str, ...]:
        """A flat topological order, for display."""
        return tuple(sid for layer in self._order for sid in layer)

    def approval_steps(self) -> tuple[str, ...]:
        return tuple(
            sid
            for sid, s in self._steps.items()
            if s.kind is StepKind.APPROVAL
        )

    def agent_names(self) -> frozenset[str]:
        """Every agent this workflow needs. Use to validate against a registry
        before a run starts, rather than failing halfway through."""
        return frozenset(
            s.agent for s in self._steps.values() if s.agent
        )

    def missing_agents(self, available: Iterable[str]) -> frozenset[str]:
        return self.agent_names() - frozenset(available)

    def __repr__(self) -> str:
        return "<Workflow %r v%d steps=%d layers=%d>" % (
            self.name,
            self.version,
            len(self._steps),
            len(self._order),
        )
