"""Model catalog: capabilities, caps, prices, and parameter restrictions.

Lifted from ``model_config._AVAILABLE_MODELS``, which carried id / name /
max_tokens_cap / supports_caching. Three things are added, one of which fixes a
live bug in the source.

**1. Sampling restrictions (the bug).** ``model_config._build_model`` does this::

    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg["temperature"]
    if cfg.get("top_p") is not None:
        kwargs["top_p"] = cfg["top_p"]

and ``validate_model_config`` accepts ``temperature`` in 0..1 and ``top_p`` in
0..1 as valid. But ``temperature``, ``top_p`` and ``top_k`` were **removed** on
Claude Sonnet 5 and the Opus 4.7+ / 5 family - passing any of them returns a
400. The repo's own default model is ``us.anthropic.claude-sonnet-5``, so any
task whose ``agent_task_models.model_config`` row sets a temperature fails on
the default model. It is a data-dependent failure, which is why it can sit
unnoticed: it only fires for task keys that happen to have tuned sampling.
``accepts_sampling`` records this per model and the provider drops the
parameters rather than sending a request that cannot succeed.

**2. Prices.** Needed for the cost attribution in
``observability.genai.record_usage``. These are Anthropic first-party list
rates; **Bedrock is partner-priced separately**, so treat a computed cost as an
estimate and override the table for a Bedrock-billed deployment.

**3. Opus 5.** The source catalog predates it and lists Opus 4.8 as the top
model. Opus 5 is the current flagship at the same $5/$25 rate, so it is added
and is the package default.

Ids here are Bedrock cross-region inference profile ids (the ``us.anthropic.``
form), because that is what ``strands.models.BedrockModel`` is given in the
lifted code. ``canonical_id`` maps them back to first-party ids for price
lookup.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_MODEL_ID",
    "MODELS",
    "CatalogPriceBook",
    "ModelSpec",
    "accepts_sampling",
    "available_models",
    "max_tokens_cap",
    "price_for",
    "spec_for",
    "supports_caching",
]

DEFAULT_MODEL_ID = "us.anthropic.claude-opus-5"


@dataclass(frozen=True)
class ModelSpec:
    """What the platform needs to know about a model before calling it."""

    id: str
    name: str
    canonical_id: str
    # None where the value is not established. The lifted catalog carried only
    # max_tokens_cap, so the Nova context windows are unverified rather than known.
    context_window: int | None
    max_tokens_cap: int
    supports_caching: bool
    accepts_sampling: bool
    input_usd_per_mtok: float | None = None
    output_usd_per_mtok: float | None = None
    # Values above this need streaming to avoid HTTP read timeouts. The lifted
    # code already defaults streaming=True for exactly this reason (a 6-minute
    # generate_wricefw_impl_plan job timed out at 120s on blocking converse).
    requires_streaming_above: int = 16000


#: Sampling parameters (temperature / top_p / top_k) were removed on the Claude
#: 5 family and on Opus 4.7+. Haiku 4.5 and the Nova models still accept them.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="us.anthropic.claude-opus-5",
        name="Claude Opus 5",
        canonical_id="claude-opus-5",
        context_window=1_000_000,
        max_tokens_cap=128_000,
        supports_caching=True,
        accepts_sampling=False,
        input_usd_per_mtok=5.00,
        output_usd_per_mtok=25.00,
    ),
    ModelSpec(
        id="us.anthropic.claude-opus-4-8",
        name="Claude Opus 4.8",
        canonical_id="claude-opus-4-8",
        context_window=1_000_000,
        max_tokens_cap=128_000,
        supports_caching=True,
        accepts_sampling=False,
        input_usd_per_mtok=5.00,
        output_usd_per_mtok=25.00,
    ),
    ModelSpec(
        id="us.anthropic.claude-sonnet-5",
        name="Claude Sonnet 5",
        canonical_id="claude-sonnet-5",
        context_window=1_000_000,
        max_tokens_cap=128_000,
        supports_caching=True,
        accepts_sampling=False,
        input_usd_per_mtok=2.00,
        output_usd_per_mtok=10.00,
    ),
    ModelSpec(
        id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        name="Claude Haiku 4.5",
        canonical_id="claude-haiku-4-5",
        context_window=200_000,
        max_tokens_cap=64_000,
        supports_caching=True,
        accepts_sampling=True,
        input_usd_per_mtok=1.00,
        output_usd_per_mtok=5.00,
    ),
    ModelSpec(
        id="us.amazon.nova-premier-v1:0",
        name="Amazon Nova Premier",
        canonical_id="nova-premier",
        context_window=None,
        max_tokens_cap=32_000,
        supports_caching=False,
        accepts_sampling=True,
    ),
    ModelSpec(
        id="us.amazon.nova-pro-v1:0",
        name="Amazon Nova Pro",
        canonical_id="nova-pro",
        context_window=None,
        max_tokens_cap=32_000,
        supports_caching=False,
        accepts_sampling=True,
    ),
    ModelSpec(
        id="us.amazon.nova-lite-v1:0",
        name="Amazon Nova Lite",
        canonical_id="nova-lite",
        context_window=None,
        max_tokens_cap=32_000,
        supports_caching=False,
        accepts_sampling=True,
    ),
)

_BY_ID = {m.id: m for m in MODELS}
_BY_CANONICAL = {m.canonical_id: m for m in MODELS}

# Prompt caching support, kept as a prefix check as well as a per-model flag,
# because a task row may name a model id the catalog has not been updated for.
_CACHING_SUPPORTED_PREFIXES = ("us.anthropic.claude", "anthropic.claude")

# Sampling-restricted families, for ids not in the catalog. Anything in the
# Claude 5 line or Opus 4.7+ rejects temperature/top_p/top_k.
_NO_SAMPLING_SUBSTRINGS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)


def spec_for(model_id: str) -> ModelSpec | None:
    """Catalog entry for a Bedrock or first-party id, if known."""
    return _BY_ID.get(model_id) or _BY_CANONICAL.get(model_id)


def price_for(model_id: str) -> tuple[float, float] | None:
    """(input, output) USD per million tokens, or None if unpriced."""
    spec = spec_for(model_id)
    if spec is None or spec.input_usd_per_mtok is None or spec.output_usd_per_mtok is None:
        return None
    return spec.input_usd_per_mtok, spec.output_usd_per_mtok


def accepts_sampling(model_id: str) -> bool:
    """Whether temperature / top_p / top_k may be sent to this model.

    Falls back to a substring check so an id absent from the catalog is still
    handled conservatively rather than being sent parameters that 400.
    """
    spec = spec_for(model_id)
    if spec is not None:
        return spec.accepts_sampling
    lowered = model_id.lower()
    return not any(s in lowered for s in _NO_SAMPLING_SUBSTRINGS)


def supports_caching(model_id: str) -> bool:
    """Whether this model supports Bedrock prompt caching.

    Preserves the prefix behaviour of ``model_config.supports_caching`` for
    unknown ids while preferring the catalog flag when there is one.
    """
    spec = spec_for(model_id)
    if spec is not None:
        return spec.supports_caching
    return model_id.startswith(_CACHING_SUPPORTED_PREFIXES)


def max_tokens_cap(model_id: str) -> int:
    """The output cap for a model, defaulting conservatively."""
    spec = spec_for(model_id)
    return spec.max_tokens_cap if spec else 64_000


class CatalogPriceBook:
    """``ports.PriceBook`` over this catalog.

    Rates are Anthropic first-party list prices. **Bedrock and Vertex are
    partner-priced separately**, so a deployment billed through either should
    supply its own price book rather than trust these numbers - which is
    exactly why pricing is a port and not a hardcoded lookup.
    """

    def __init__(self, overrides: dict[str, tuple[float, float]] | None = None):
        self._overrides = dict(overrides or {})

    def price_for(self, model_id: str) -> tuple[float, float] | None:
        if model_id in self._overrides:
            return self._overrides[model_id]
        return price_for(model_id)


def available_models() -> list[dict[str, object]]:
    """Catalog as dicts, in the shape ``model_config.available_models`` returned.

    Kept for drop-in compatibility with the admin UI that consumes it.
    """
    return [
        {
            "id": m.id,
            "name": m.name,
            "max_tokens_cap": m.max_tokens_cap,
            "supports_caching": m.supports_caching,
            "accepts_sampling": m.accepts_sampling,
            "context_window": m.context_window,
        }
        for m in MODELS
    ]
