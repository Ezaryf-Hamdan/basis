"""Agent registry and the runner that resolves delegations.

**Built from scratch.**

The registry is a name-to-agent map. It is held by the *orchestrator*, never
handed to an agent - that asymmetry is what keeps invariant 1 enforceable
(see `base.py`).

`AgentRunner` is the only thing that satisfies a `Delegation`. It runs an
agent, and if the agent came back asking for prerequisites it runs those and
calls the agent again with their results attached. Two bounds make that safe:

  * ``max_rounds`` caps how many times one agent may re-ask. Without it, an
    agent that always returns the same delegation loops forever. AILedSDLC's
    equivalent - inline sub-agent calls - had no such bound; a persona that
    consulted the solution architect on every turn would recurse until the
    stack or the budget gave out.
  * ``max_depth`` caps delegation nesting, so a chain A→B→C→A terminates.

Note that a cycle is *detected here*, at run time, rather than forbidden
statically. Delegations are dynamic by nature - an agent decides what it needs
from its input - so a static graph check cannot see them. Workflow steps, which
*are* static, get a real cycle check in `basis.workflow`.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ..errors import BasisError
from .base import Agent, AgentContext, AgentInput, AgentResult

__all__ = [
    "AgentRegistry",
    "AgentRunner",
    "AgentRunnerInvoker",
    "CallableAgent",
    "DelegationLimitExceeded",
]

log = logging.getLogger(__name__)


class DelegationLimitExceeded(BasisError):
    """An agent kept asking for prerequisites past the configured bound."""

    def __init__(self, agent_name: str, kind: str, limit: int):
        self.agent_name = agent_name
        super().__init__(
            "agent %r exceeded %s limit of %d; likely a delegation cycle"
            % (agent_name, kind, limit)
        )

    def marker(self) -> str:
        return "delegation_limit: %s" % self


class AgentRegistry:
    """Name to agent. Held by orchestrators only."""

    def __init__(self, agents: Sequence[Agent] = ()):
        self._agents: dict[str, Agent] = {}
        for agent in agents:
            self.register(agent)

    def register(self, agent: Agent) -> None:
        if agent.name in self._agents:
            raise ValueError("agent name %r is already registered" % agent.name)
        self._agents[agent.name] = agent

    def get(self, name: str) -> Agent:
        try:
            return self._agents[name]
        except KeyError:
            raise KeyError(
                "no agent named %r; registered: %s"
                % (name, sorted(self._agents))
            ) from None

    def names(self) -> list[str]:
        return sorted(self._agents)

    def __contains__(self, name: object) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)


class AgentRunner:
    """Runs agents and satisfies their delegations.

    This is the *only* component that may call more than one agent, which is
    the concrete meaning of "sequencing lives outside agents".
    """

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        max_rounds: int = 3,
        max_depth: int = 3,
    ):
        self.registry = registry
        self._max_rounds = max_rounds
        self._max_depth = max_depth

    async def run(
        self,
        agent_name: str,
        ctx: AgentContext,
        task: AgentInput,
        *,
        _depth: int = 0,
        _seen: tuple[str, ...] = (),
    ) -> AgentResult:
        """Run an agent to completion, resolving any delegations it asks for."""
        if _depth > self._max_depth:
            raise DelegationLimitExceeded(agent_name, "depth", self._max_depth)
        if agent_name in _seen:
            raise DelegationLimitExceeded(
                agent_name, "cycle (%s)" % " -> ".join((*_seen, agent_name)), _depth
            )

        agent = self.registry.get(agent_name)
        current = task

        for round_no in range(self._max_rounds + 1):
            result = await agent.run(ctx, current)

            if not result.needs_delegation:
                return result

            if round_no == self._max_rounds:
                raise DelegationLimitExceeded(agent_name, "rounds", self._max_rounds)

            log.debug(
                "agent %s round %d requested %d delegation(s)",
                agent_name,
                round_no,
                len(result.delegations),
            )

            # Delegations are run sequentially and their results attached. They
            # are not parallelized: an agent listing two prerequisites has not
            # said they are independent, and guessing wrong is a correctness
            # bug rather than a performance one. A workflow step is the place
            # to express real parallelism.
            for delegation in result.delegations:
                sub_result = await self.run(
                    delegation.agent_name,
                    ctx,
                    delegation.input,
                    _depth=_depth + 1,
                    _seen=(*_seen, agent_name),
                )
                current = current.with_upstream(
                    delegation.result_key, sub_result.output
                )

        # Unreachable: the loop either returns or raises.
        raise DelegationLimitExceeded(agent_name, "rounds", self._max_rounds)


class CallableAgent(Agent):
    """Wraps a plain async function as an agent.

    For the common case where an agent is one prompt and no branching, so a
    subclass would be four lines of boilerplate. Also makes tests readable.
    """

    def __init__(
        self,
        name: str,
        fn: Any,
        *,
        task_key: str | None = None,
        description: str = "",
    ):
        super().__init__(name=name, task_key=task_key)
        self._fn = fn
        self.description = description

    async def run(self, ctx: AgentContext, task: AgentInput) -> AgentResult:
        result = await self._fn(ctx, task)
        if isinstance(result, AgentResult):
            return result
        return AgentResult(output=result)


class AgentRunnerInvoker:
    """Adapts an `AgentRunner` to `ports.AgentInvoker`.

    This is the piece that lets `workflow` drive agents without importing
    `agents`. The dependency direction is deliberate: `agents` knows about the
    port and provides an implementation of it, so `workflow` names only the
    protocol. An AGENT step could therefore be served by something that is not
    a basis Agent at all - a remote worker, a queue consumer - by supplying a
    different implementation.
    """

    def __init__(self, runner: AgentRunner, context_factory: Any):
        """``context_factory(ctx) -> AgentContext``.

        A factory rather than a stored AgentContext because each step gets its
        own child RunContext, and an agent's model calls must resolve against
        that step's context rather than the parent run's.
        """
        self._runner = runner
        self._context_factory = context_factory

    def known_agents(self) -> list[str]:
        return self._runner.registry.names()

    async def invoke_agent(
        self,
        agent_name: str,
        *,
        payload: Any,
        upstream: Any,
        ctx: Any,
        instructions: str = "",
    ) -> Any:
        task = AgentInput(
            payload=dict(payload or {}),
            upstream=dict(upstream or {}),
            instructions=instructions,
        )
        result = await self._runner.run(
            agent_name, self._context_factory(ctx), task
        )
        return result.output
