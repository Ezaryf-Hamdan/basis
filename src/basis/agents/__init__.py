"""Agent contract, registry, and the runner that resolves delegations.

The asymmetry here is deliberate: an ``Agent`` receives an ``AgentContext``
(model gateway + filtered tools) and nothing that could run another agent. Only
``AgentRunner`` - held by an orchestrator - can. That is invariant 1 made
structural rather than advisory.
"""
from .base import Agent, AgentContext, AgentInput, AgentResult, Delegation
from .registry import (
    AgentRegistry,
    AgentRunner,
    AgentRunnerInvoker,
    CallableAgent,
    DelegationLimitExceeded,
)

__all__ = [
    "Agent",
    "AgentContext",
    "AgentInput",
    "AgentRegistry",
    "AgentResult",
    "AgentRunner",
    "AgentRunnerInvoker",
    "CallableAgent",
    "Delegation",
    "DelegationLimitExceeded",
]
