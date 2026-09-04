"""gen_ai span conventions, including the token/cost attributes that were missing.

The prompt and completion recorders are lifted from ``otel_setup.py``, keeping
the insight that earned them their comment:

    "Uses span attributes instead of events because CloudWatch Transaction
    Search's `aws/spans` log group strips events but preserves attributes."

That is a real, hard-won operational detail, so the dual write (attribute *and*
event) is preserved exactly - attribute for CloudWatch, event for backends that
prefer events.

What is added is the half that did not exist. Slide 13 lists token usage and
model latency as ai-core gaps, and the deck's FinOps capability depends on
per-tenant cost attribution. ``record_usage`` writes the token counts and, when
a price is known, a computed cost - onto a span that (via ``tracing.span``)
already carries tenant_id and project_id. That combination is what makes
"cost per client" answerable.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..ports import PriceBook
from .redaction import redact, truncate

__all__ = [
    "record_completion",
    "record_model_call",
    "record_prompt",
    "record_usage",
]


def record_prompt(span: Any, system_prompt: str | None, user_prompt: str | None) -> None:
    """Attach the request messages to the span, redacted and capped."""
    payload = [
        {"role": "system", "content": truncate(redact(system_prompt))},
        {"role": "user", "content": truncate(redact(user_prompt))},
    ]
    encoded = json.dumps(payload)
    span.set_attribute("gen_ai.prompt", encoded)
    span.add_event("gen_ai.prompt", attributes={"gen_ai.prompt": encoded})


def record_completion(span: Any, completion_text: str | None) -> None:
    """Attach the response to the span, redacted and capped."""
    payload = [{"role": "assistant", "content": truncate(redact(completion_text))}]
    encoded = json.dumps(payload)
    span.set_attribute("gen_ai.completion", encoded)
    span.add_event("gen_ai.completion", attributes={"gen_ai.completion": encoded})


def record_usage(
    span: Any,
    *,
    model_id: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    latency_ms: float | None = None,
    prices: Mapping[str, tuple[float, float]] | None = None,
    price_book: PriceBook | None = None,
) -> None:
    """Record token counts, latency, and (when priced) cost on the span.

    ``prices`` maps a model id to (input_usd_per_mtok, output_usd_per_mtok).
    Defaults come from ``basis.models.catalog``, which carries first-party
    rates; Bedrock is partner-priced separately, so a deployment billing
    through Bedrock should pass its own table rather than trust the default.
    Cost is therefore reported as an estimate under an explicit attribute name.

    ``cache_read_tokens`` matters more than it looks: prompt caching is enabled
    by default in the lifted model layer, so a run whose cache is silently
    invalidated costs several times more with no other visible symptom. Without
    this attribute that regression is undetectable.
    """
    span.set_attribute("gen_ai.system", "aws.bedrock")
    span.set_attribute("gen_ai.request.model", model_id)

    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))
    if cache_read_tokens is not None:
        span.set_attribute("gen_ai.usage.cache_read_input_tokens", int(cache_read_tokens))
    if cache_write_tokens is not None:
        span.set_attribute(
            "gen_ai.usage.cache_creation_input_tokens", int(cache_write_tokens)
        )
    if latency_ms is not None:
        span.set_attribute("gen_ai.client.latency_ms", float(latency_ms))

    # Prices come from a port, not from the model catalog. Importing
    # `models.catalog` here created a models <-> observability cycle that only
    # survived because this import sat inside the function body.
    if prices is not None:
        price = prices.get(model_id)
    elif price_book is not None:
        price = price_book.price_for(model_id)
    else:
        price = None

    if price and input_tokens is not None and output_tokens is not None:
        in_rate, out_rate = price
        cost = (input_tokens / 1_000_000.0) * in_rate + (
            output_tokens / 1_000_000.0
        ) * out_rate
        span.set_attribute("basis.cost.estimated_usd", round(cost, 6))
        span.set_attribute("basis.cost.basis", "first_party_list_price")


def record_model_call(
    span: Any,
    *,
    model_id: str,
    system_prompt: str | None,
    user_prompt: str | None,
    completion: str | None,
    usage: Mapping[str, Any] | None = None,
    latency_ms: float | None = None,
    price_book: PriceBook | None = None,
) -> None:
    """One call that records everything about a model invocation.

    Convenience for the gateway, so a caller cannot record the prompt but
    forget the usage - which is how the original ended up with prompts in
    traces and no token accounting anywhere.
    """
    record_prompt(span, system_prompt, user_prompt)
    record_completion(span, completion)
    usage = usage or {}
    record_usage(
        span,
        model_id=model_id,
        input_tokens=usage.get("inputTokens") or usage.get("input_tokens"),
        output_tokens=usage.get("outputTokens") or usage.get("output_tokens"),
        cache_read_tokens=usage.get("cacheReadInputTokens")
        or usage.get("cache_read_input_tokens"),
        cache_write_tokens=usage.get("cacheWriteInputTokens")
        or usage.get("cache_creation_input_tokens"),
        latency_ms=latency_ms,
        price_book=price_book,
    )
