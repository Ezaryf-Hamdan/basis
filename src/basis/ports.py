"""Ports: the protocols the eight functions use to talk to each other.

Before this module the functions imported each other concretely:

    agents      -> models          (AgentContext held a ModelGateway)
    memory      -> models          (consolidator built a ModelGateway)
    personas    -> models          (LlmRouter held a ModelGateway)
    workflow    -> agents          (engine imported AgentRunner)
    models      -> observability   (gateway imported genai + tracing)
    observability -> models        (genai imported catalog.price_for)

The last two are a cycle. It did not break only because `genai` did its import
inside a function body - which is to say the design was already wrong and was
being held together by import laziness. That is exactly the kind of thing that
works until someone moves an import to the top of a file.

Every protocol here is structural (`typing.Protocol`), so an implementation
does not subclass anything and there is no registration step. `ModelGateway`
already satisfies `ModelClient` without being told about it.

This module imports nothing from any function package - only `context`. That
is enforced by a test, because a port module that grows a dependency stops
being a port.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .context import RunContext

__all__ = [
    "AgentInvoker",
    "AuditSink",
    "Completion",
    "Embedder",
    "ModelClient",
    "NullPriceBook",
    "NullSpan",
    "NullTracer",
    "PriceBook",
    "RetrieverPort",
    "SpanLike",
    "TracerPort",
]


# ── model results ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Completion:
    """A provider-neutral generation result.

    Lives here rather than in `models` so that `agents`, `memory` and
    `personas` can name the type they receive without importing the model
    layer. `models.providers.base.ModelResponse` is an alias of this.

    ``usage`` is the field the lifted code discarded - `invoke_with_fallback`
    returned only `(text, model_id)`, which is the mechanical reason per-client
    cost attribution did not exist.
    """

    text: str
    model_id: str
    usage: Mapping[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None
    latency_ms: float | None = None
    messages: Sequence[Any] = field(default_factory=tuple)


@runtime_checkable
class ModelClient(Protocol):
    """What everything except `models` needs from a model layer.

    Deliberately just this one method. An agent does not need to resolve task
    keys, build fallback chains or know what a provider is - it needs to send
    prompts and get text back.
    """

    async def invoke(
        self,
        task_key: str,
        *,
        user_prompt: str,
        system_prompt: str | None = ...,
        tools: Sequence[Any] = ...,
        ctx: RunContext | None = ...,
    ) -> Completion: ...


@runtime_checkable
class Embedder(Protocol):
    """Text to vector.

    This protocol is the fix for an inconsistency in the first cut: `models`
    got a proper provider boundary and embeddings got a concrete class
    (`TitanEmbedder`) imported directly by the consolidator. There was no
    reason for the asymmetry.
    """

    dimensions: int

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...


# ── observability ──────────────────────────────────────────────────────────


@runtime_checkable
class SpanLike(Protocol):
    """The subset of an OTel span that basis actually calls."""

    def set_attribute(self, key: str, value: Any) -> None: ...

    def add_event(self, name: str, attributes: Any = ...) -> None: ...


@runtime_checkable
class TracerPort(Protocol):
    """Starts spans. Implemented by `observability.tracing`.

    Taking this as a port is what lets `models` emit spans without importing
    `observability`, breaking the cycle described at the top of this module.
    """

    def span(
        self, name: str, ctx: RunContext | None = ..., **attributes: Any
    ) -> AbstractContextManager[SpanLike]: ...


@runtime_checkable
class PriceBook(Protocol):
    """Model pricing lookup, for cost attribution.

    The other half of the cycle: `observability` needed prices, which live in
    the model catalog. Now it asks a port instead, and a consumer billing
    through a reseller can supply its own rates - which they should, since the
    bundled catalog carries first-party list prices and Bedrock is
    partner-priced separately.
    """

    def price_for(self, model_id: str) -> tuple[float, float] | None: ...


@runtime_checkable
class AuditSink(Protocol):
    """Records that a tool ran. Implemented by `tools.registry` and by
    `adapters.aicore.AiCoreAuditSink`."""

    def record_invocation(
        self,
        tool_name: str,
        *,
        ctx: RunContext | None = ...,
        allowed: bool = ...,
        reason: str | None = ...,
        duration_ms: int | None = ...,
        error: str | None = ...,
        effect: str | None = ...,
    ) -> None: ...


# ── orchestration ──────────────────────────────────────────────────────────


@runtime_checkable
class AgentInvoker(Protocol):
    """What `workflow` needs in order to run an agent step.

    Only a name, a payload and upstream results go in; a plain value comes
    back. The workflow engine therefore does not import `agents` at all, and
    could drive something that is not a basis Agent - a remote worker, say.
    """

    async def invoke_agent(
        self,
        agent_name: str,
        *,
        payload: Mapping[str, Any],
        upstream: Mapping[str, Any],
        ctx: RunContext,
        instructions: str = ...,
    ) -> Any: ...

    def known_agents(self) -> Sequence[str]:
        """Names available, so a workflow can be validated before it runs."""
        ...


@runtime_checkable
class RetrieverPort(Protocol):
    """Knowledge retrieval, as an agent sees it."""

    def search(self, query: Any, *, ctx: RunContext | None = ...) -> list[Any]: ...


# ── null implementations ───────────────────────────────────────────────────
#
# Supplied so a caller can wire a function without its optional collaborators
# rather than passing None and having every call site guard for it. A no-op
# tracer is a better default than a None check in fifteen places.


class NullSpan:
    """Accepts and discards everything."""

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def add_event(self, name: str, attributes: Any = None) -> None:
        return None

    def record_exception(self, exc: BaseException) -> None:
        return None

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        return None


class NullTracer:
    """A tracer that produces no spans."""

    def span(
        self, name: str, ctx: RunContext | None = None, **attributes: Any
    ) -> AbstractContextManager[SpanLike]:
        from contextlib import nullcontext

        return nullcontext(NullSpan())


class NullPriceBook:
    """No prices known, so no cost is reported.

    Distinct from reporting zero: a missing price must not look like free.
    """

    def price_for(self, model_id: str) -> tuple[float, float] | None:
        return None
