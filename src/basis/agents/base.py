"""The Agent contract.

**Built from scratch.** AILedSDLC has agents but no agent abstraction: each of
`fitgap_agentic`, `coder_agents`, `config_area_generator`, `kdd_generator` and
a dozen more constructs a Strands `Agent` inline, with its own prompt
assembly, its own retrieval, its own persistence and its own sequencing. The
deck scores Agent Reusability 5/10 and reads the cause correctly - "prompts,
retrieval, orchestration and persistence tangled in one class".

The single most important thing this module does is make that tangle
**unrepresentable**, not merely discouraged.

Invariant 1 says sequencing does not live inside agents. The usual way to
"enforce" that is a comment, which lasts until the first deadline. Here an
`Agent` has no access to anything that could run another agent: it receives an
`AgentContext` carrying a model gateway and a tool catalog, and nothing else.
There is no registry handle, no executor, no workflow reference. An agent that
wants help must *return* a `Delegation` describing the help it needs, and the
workflow engine decides whether and how to satisfy it.

So the shapes are:

    Agent.run(input)  ->  AgentResult          # I finished
                      ->  AgentResult(delegations=[...])   # I need these first

An agent physically cannot call a peer, because it was never handed the means.
That is the difference between an invariant and a wish.
"""
from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..context import RunContext
from ..ports import Completion, ModelClient, RetrieverPort

__all__ = [
    "Agent",
    "AgentContext",
    "AgentInput",
    "AgentResult",
    "Delegation",
]


@dataclass(frozen=True)
class Delegation:
    """A request for another agent to do something first.

    Returned by an agent, satisfied by the orchestrator. Carries no callable
    and no reference to the target agent object - only its name and the input
    it should receive - so honouring a delegation is necessarily the caller's
    decision.
    """

    agent_name: str
    input: AgentInput
    reason: str = ""
    #: Where to put the result in this agent's next input. When the
    #: orchestrator re-runs the delegating agent, the delegate's output is
    #: placed under this key in ``AgentInput.upstream``.
    result_key: str = "delegate"


@dataclass(frozen=True)
class AgentInput:
    """What an agent is asked to work on.

    ``payload`` is the task-specific data. ``upstream`` holds results from
    steps or delegations that ran before this one - it is how data flows
    between agents without any agent knowing which agent produced it.
    """

    payload: Mapping[str, Any] = field(default_factory=dict)
    upstream: Mapping[str, Any] = field(default_factory=dict)
    instructions: str = ""

    def with_upstream(self, key: str, value: Any) -> AgentInput:
        merged = dict(self.upstream)
        merged[key] = value
        return AgentInput(
            payload=self.payload, upstream=merged, instructions=self.instructions
        )


@dataclass(frozen=True)
class AgentResult:
    """What an agent produced.

    ``delegations`` being non-empty means the agent did *not* finish and is
    asking for prerequisites. The orchestrator runs them and calls the agent
    again with their results in ``upstream``. ``complete`` distinguishes
    "finished with no output" from "not finished".
    """

    output: Any = None
    complete: bool = True
    delegations: Sequence[Delegation] = ()
    #: Notes worth remembering past this run. The orchestrator writes them to
    #: short-term memory, so an agent never touches the memory store itself.
    notes: Sequence[tuple[str, str]] = ()
    usage: Mapping[str, Any] = field(default_factory=dict)
    model_id: str | None = None

    @property
    def needs_delegation(self) -> bool:
        return bool(self.delegations)


@dataclass(frozen=True)
class AgentContext:
    """The only capabilities an agent is given.

    Deliberately minimal. Note what is *absent*: no `AgentRegistry`, no
    `WorkflowEngine`, no `MemoryService`, no database handle. An agent cannot
    sequence work, cannot write durable state, and cannot reach a peer,
    because none of those are reachable from here.

    ``tools`` is already filtered by `ToolRegistry.tools_for`, at the single
    seam, before the agent is constructed - the same discipline as
    `job_executor.py:3705`, which applied its policy once before any handler
    or sub-agent handoff.
    """

    run: RunContext
    #: A `ports.ModelClient`. `models.ModelGateway` satisfies it structurally,
    #: so nothing here imports the model layer.
    gateway: ModelClient
    tools: Sequence[Any] = ()
    #: Retrieval, when the agent is allowed it. A `ports.RetrieverPort`.
    retriever: RetrieverPort | None = None
    #: Resolved persona row, when the run has one.
    persona: Mapping[str, Any] | None = None

    async def retrieve(self, query: Any) -> list[Any]:
        """Run a retrieval, off the event loop.

        `knowledge.HybridRetriever.search` is synchronous - it has sync
        consumers - so an agent calling it directly from its coroutine would
        block the loop on a database round trip. This is the safe path, and the
        reason it exists on the context rather than being left to each agent.
        """
        if self.retriever is None:
            raise RuntimeError(
                "this agent was not given a retriever; knowledge access was "
                "not granted for this run"
            )
        from ..concurrency import offload

        return await offload(self.retriever.search, query, ctx=self.run)

    async def generate(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        task_key: str | None = None,
        use_tools: bool = True,
    ) -> Completion:
        """Call the model for this run.

        Routes through the gateway, so the call is traced, costed, retried and
        tenant-scoped without the agent doing anything. In AILedSDLC each agent
        site built its own `BedrockModel` and got none of that.
        """
        return await self.gateway.invoke(
            task_key or self.run.task_key or "default",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=self.tools if use_tools else (),
            ctx=self.run,
        )


class Agent(abc.ABC):
    """One unit of agent work.

    Subclass and implement `run`. Keep it to a single responsibility: an agent
    that finds itself wanting to call two other agents in order is a workflow,
    and belongs in `basis.workflow` instead.
    """

    #: Stable identifier, used by workflows and delegations to name this agent.
    name: str = ""

    #: Task key for model resolution. Falls back to ``name``.
    task_key: str = ""

    #: Human-readable, for operator UIs and audit records.
    description: str = ""

    def __init__(self, *, name: str | None = None, task_key: str | None = None):
        if name:
            self.name = name
        if task_key:
            self.task_key = task_key
        if not self.name:
            raise ValueError(
                "%s must set a class-level `name` or pass one to __init__"
                % type(self).__name__
            )

    def resolved_task_key(self) -> str:
        return self.task_key or self.name

    def system_prompt(self, ctx: AgentContext) -> str | None:
        """The agent's system prompt.

        Defaults to the persona's prompt when the run has one, else None so the
        gateway falls back to the prompt stored against the task key. Override
        for an agent whose prompt is computed.
        """
        if ctx.persona:
            prompt = ctx.persona.get("system_prompt")
            if prompt:
                return str(prompt)
        return None

    @abc.abstractmethod
    async def run(self, ctx: AgentContext, task: AgentInput) -> AgentResult:
        """Do the work. Return a result, or delegations if prerequisites are missing."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return "<%s name=%r>" % (type(self).__name__, self.name)
