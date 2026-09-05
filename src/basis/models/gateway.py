"""ModelGateway - the single door to any model.

Replaces ``model_config.invoke_with_fallback``. The original is the right idea
in the wrong shape: it resolved a task key, built a chain, and walked it on
retryable errors, but it returned ``BedrockModel`` objects to its callers, so
"the gateway" was really just a helper and every caller stayed coupled to
Bedrock. Four other modules (``reflection``, ``inter_agent``, ``llm_router``,
``memory_consolidator``) bypassed it entirely and constructed their own models
with hardcoded ids.

What the gateway now guarantees that the original did not:

  * **Callers never see a provider type.** They pass a task key and prompts and
    receive a ``ModelResponse``.
  * **Retry before fallback.** The original fell straight through the chain on
    a ``ThrottlingException`` with no delay, turning transient capacity limits
    into hard failures and exhausting the chain in microseconds. See
    ``retry.backoff_delays``.
  * **Every call is traced and costed.** The span carries the run's tenant and
    project (via ``tracing.span``), the prompt and completion are recorded
    redacted, and token usage becomes a cost estimate. The original discarded
    usage entirely, which is why cost-per-client was unanswerable.
  * **Fail-closed on tenancy.** Resolution is tenant-scoped, so a gateway call
    without a bound RunContext raises rather than silently resolving against
    another tenant's configuration.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from ..concurrency import run_in_context
from ..context import RunContext, require_context
from ..errors import ModelUnavailable
from ..ports import PriceBook, TracerPort
from . import retry
from .providers.base import ModelProvider, ModelRequest, ModelResponse
from .resolver import TaskModel, resolve

__all__ = ["ModelGateway"]

log = logging.getLogger(__name__)


class ModelGateway:
    """Resolves task keys and invokes models through registered providers."""

    def __init__(
        self,
        providers: Sequence[ModelProvider] | None = None,
        *,
        dsn: str | None = None,
        retries_per_model: int = 2,
        resolver: Any = None,
        tracer: TracerPort | None = None,
        price_book: PriceBook | None = None,
    ):
        self._providers: list[ModelProvider] = list(providers or [])
        self._dsn = dsn
        self._retries_per_model = retries_per_model
        # Injectable so a consumer can resolve task keys against its own
        # schema instead of basis's tables - see `adapters.aicore.AiCoreResolver`,
        # which reads ai-core's `model_roles`. Anything with a
        # `.resolve(task_key, tenant_id=...) -> TaskModel` satisfies this.
        self._resolver = resolver

        # Tracing and pricing arrive as ports rather than direct imports. That
        # is what breaks the models <-> observability cycle: `observability`
        # needs prices (which live in the model catalog) and `models` needs
        # spans. Both now ask an interface.
        if tracer is None:
            from ..observability.tracing import Tracer

            tracer = Tracer()
        self._tracer = tracer

        if price_book is None:
            from .catalog import CatalogPriceBook

            price_book = CatalogPriceBook()
        self._price_book = price_book

        if not self._providers:
            # Default to Bedrock when the extra is installed, so the common
            # case needs no wiring. Absent, the gateway is still constructible
            # and raises only when actually invoked.
            try:
                from .providers.bedrock import BedrockProvider

                self._providers.append(BedrockProvider())
            except ImportError:
                log.debug("no bedrock extra installed; register a provider explicitly")

    def register(self, provider: ModelProvider) -> None:
        self._providers.append(provider)

    def provider_for(self, model_id: str) -> ModelProvider:
        for provider in self._providers:
            if provider.supports(model_id):
                return provider
        raise ModelUnavailable(
            model_id, 0, RuntimeError("no registered provider serves %r" % model_id)
        )

    # ── resolution ─────────────────────────────────────────────────────────

    def resolve(self, task_key: str, *, ctx: RunContext | None = None) -> TaskModel:
        run = ctx if ctx is not None else require_context()
        if self._resolver is not None:
            return self._resolver.resolve(task_key, tenant_id=run.tenant_id)
        return resolve(task_key, tenant_id=run.tenant_id, dsn=self._dsn)

    # ── invocation ─────────────────────────────────────────────────────────

    async def invoke(
        self,
        task_key: str,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        tools: Sequence[Any] = (),
        ctx: RunContext | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Invoke the model configured for ``task_key``, with retry and fallback.

        ``system_prompt`` defaults to the prompt stored against the task key,
        matching ``get_system_prompt``'s role. Passing one explicitly wins.
        """
        run = ctx if ctx is not None else require_context()
        task = self.resolve(task_key, ctx=run)
        effective_system = (
            system_prompt if system_prompt is not None else task.system_prompt
        )

        chain = task.chain()
        last_error: BaseException | None = None
        attempts = 0

        with self._tracer.span(
            "gen_ai.invoke", run, **{"basis.task_key": task_key}
        ) as sp:
            sp.set_attribute("basis.model.resolution_source", task.source)
            sp.set_attribute("basis.model.chain_length", len(chain))

            for position, (model_id, model_cfg) in enumerate(chain):
                merged = dict(model_cfg)
                if overrides:
                    merged.update(overrides)
                request = _build_request(
                    model_id=model_id,
                    system_prompt=effective_system,
                    user_prompt=user_prompt,
                    tools=tools,
                    config=merged,
                )

                try:
                    provider = self.provider_for(model_id)
                except ModelUnavailable as exc:
                    last_error = exc
                    continue

                # Retry the same model before demoting to the next one: a
                # throttle is a capacity signal, not a model problem.
                for delay in _attempt_delays(self._retries_per_model):
                    if delay:
                        await asyncio.sleep(delay)
                    attempts += 1
                    try:
                        response = await run_in_context(provider.generate, request)
                    except Exception as exc:
                        if not retry.is_retryable(exc):
                            sp.set_attribute("basis.model.failed_model", model_id)
                            raise
                        last_error = exc
                        log.warning(
                            "retryable model error on %s (attempt %d): %s",
                            model_id,
                            attempts,
                            exc,
                        )
                        continue

                    sp.set_attribute("gen_ai.response.model", response.model_id)
                    sp.set_attribute("basis.model.chain_position", position)
                    sp.set_attribute("basis.model.attempts", attempts)
                    from ..observability import genai

                    genai.record_model_call(
                        sp,
                        model_id=response.model_id,
                        system_prompt=effective_system,
                        user_prompt=user_prompt,
                        completion=response.text,
                        usage=response.usage,
                        latency_ms=response.latency_ms,
                        price_book=self._price_book,
                    )
                    return response

            raise ModelUnavailable(task_key, attempts, last_error)

    async def invoke_text(
        self,
        task_key: str,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        tools: Sequence[Any] = (),
        ctx: RunContext | None = None,
    ) -> tuple[str, str]:
        """``(text, model_id)`` - the exact return shape of the lifted
        ``invoke_with_fallback``, so existing call sites port with a one-line change.
        """
        response = await self.invoke(
            task_key,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=tools,
            ctx=ctx,
        )
        return response.text, response.model_id


def _attempt_delays(retries: int) -> list[float]:
    """[0.0, then jittered backoff] - the first attempt is immediate."""
    return [0.0, *list(retry.backoff_delays(max(0, retries)))]


def _build_request(
    *,
    model_id: str,
    system_prompt: str | None,
    user_prompt: str,
    tools: Sequence[Any],
    config: dict[str, Any],
) -> ModelRequest:
    """Translate a stored ``model_config`` row into a provider request.

    Sampling parameters are passed through here and dropped by the provider
    when the target model rejects them, so a config row tuned for Haiku still
    works unchanged against a Sonnet 5 fallback.
    """
    return ModelRequest(
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=tuple(tools),
        max_tokens=config.get("max_tokens"),
        temperature=config.get("temperature"),
        top_p=config.get("top_p"),
        top_k=config.get("top_k"),
        stop_sequences=tuple(config.get("stop_sequences") or ()),
        streaming=config.get("streaming", True),
        enable_prompt_caching=config.get("enable_prompt_caching", True),
        read_timeout=config.get("read_timeout", 300),
    )
