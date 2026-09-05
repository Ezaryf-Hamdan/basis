"""Bedrock provider, via Strands.

Lifted from ``model_config._build_model`` / ``_invoke_one`` / ``enable_auto_cache``
/ ``wrap_with_cache``, behind the ``ModelProvider`` boundary. Strands and boto
are imported lazily so the package installs and imports without the ``bedrock``
extra.

Preserved from the source, with its reasoning intact:

  * ``streaming=True`` by default. The original comment explains why and it is
    a real production finding: "use converse_stream instead of converse so the
    first bytes arrive quickly and botocore's read_timeout never fires on
    long-running generation jobs (e.g. generate_wricefw_impl_plan takes ~6 min;
    blocking converse timed out at 120s)".
  * ``read_timeout=300`` with ``connect_timeout=10`` as the per-chunk safety net.
  * The Bedrock Converse cache-point shape - ``[{"text": ...},
    {"cachePoint": {"type": "default"}}]`` - which is genuinely different from
    the Anthropic-native ``cache_control`` shape. The original's comment
    calling that out has saved someone an afternoon.
  * Strands' auto cache strategy for tool-loop and multi-turn agents.

Fixed:

  * **Sampling parameters are dropped for models that reject them.** See
    ``catalog.accepts_sampling``. The original passed ``temperature`` and
    ``top_p`` through unconditionally, which returns a 400 on Sonnet 5 and the
    Opus 5 / 4.8 family - including the repo's own default model.
  * ``max_tokens`` is clamped to the model's cap instead of being sent as
    given, so a task row tuned for one model does not fail against another in
    the same fallback chain.
  * Usage and latency are returned rather than discarded.
"""
from __future__ import annotations

import time
from typing import Any

from ...settings import settings
from ..catalog import accepts_sampling, max_tokens_cap, supports_caching
from .base import ModelRequest, ModelResponse

__all__ = ["BedrockProvider", "wrap_with_cache"]


def wrap_with_cache(text: str, model_id: str, enable: bool) -> Any:
    """Wrap a system prompt for Bedrock Converse prompt caching.

    Lifted unchanged. Bedrock Converse uses content blocks with text +
    cachePoint, not the Anthropic-native ``cache_control`` shape, and the
    cachePoint block follows the text it should cache. Returns a plain string
    when caching is off or unsupported, so callers can pass the result straight
    through.
    """
    if not enable or not supports_caching(model_id):
        return text
    return [{"text": text}, {"cachePoint": {"type": "default"}}]


class BedrockProvider:
    """Serves Anthropic and Nova models on Bedrock through Strands."""

    name = "bedrock"

    def __init__(self, *, region: str | None = None):
        self._region = region or settings().aws_region

    def supports(self, model_id: str) -> bool:
        return model_id.startswith(
            ("us.anthropic.", "us.amazon.", "anthropic.", "amazon.", "eu.anthropic.", "apac.anthropic.")
        )

    # ── model construction ─────────────────────────────────────────────────

    def build_model(self, request: ModelRequest) -> Any:
        """Construct a Strands BedrockModel for this request."""
        from botocore.config import Config as BotoConfig
        from strands.models import BedrockModel

        model_id = request.model_id
        kwargs: dict[str, Any] = {
            "model_id": model_id,
            "streaming": request.streaming,
        }

        if request.max_tokens is not None:
            # Clamp rather than reject: a fallback chain mixes models with
            # different caps, and failing the request because the second model
            # has a smaller ceiling than the first is not useful behaviour.
            kwargs["max_tokens"] = min(request.max_tokens, max_tokens_cap(model_id))

        # The fix. temperature / top_p / top_k were removed on the Claude 5
        # family and Opus 4.7+; sending them returns a 400.
        if accepts_sampling(model_id):
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.top_p is not None:
                kwargs["top_p"] = request.top_p
            if request.top_k is not None:
                kwargs["top_k"] = request.top_k

        if request.stop_sequences:
            kwargs["stop_sequences"] = list(request.stop_sequences)

        kwargs["boto_client_config"] = BotoConfig(
            read_timeout=request.read_timeout,
            connect_timeout=10,
            retries={"max_attempts": 2},
            region_name=self._region,
        )
        kwargs.update(request.extra)

        model = BedrockModel(**kwargs)

        if request.enable_prompt_caching and supports_caching(model_id):
            model = self.enable_auto_cache(model)
        return model

    def enable_auto_cache(self, model: Any) -> Any:
        """Turn on Strands' native auto prompt caching.

        Lifted from ``enable_auto_cache``. Strands then injects a cachePoint at
        the end of the last user message on every request, so the stable prefix
        (system + tools + prior conversation) is written on turn 1 and read on
        subsequent tool-loop turns. A pure cache boundary, so outputs are
        unchanged; a no-op for models without caching support.
        """
        try:
            from strands.models.bedrock import CacheConfig

            model_id = model.config.get("model_id", "")
        except Exception:
            return model
        if supports_caching(model_id):
            try:
                model.update_config(cache_config=CacheConfig(strategy="auto"))
            except Exception:
                return model
        return model

    # ── invocation ─────────────────────────────────────────────────────────

    def generate(self, request: ModelRequest) -> ModelResponse:
        """One blocking generation. The gateway offloads and classifies errors."""
        from strands import Agent

        model = self.build_model(request)

        system_prompt = wrap_with_cache(
            request.system_prompt or "",
            request.model_id,
            request.enable_prompt_caching,
        )

        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=list(request.tools),
        )

        started = time.monotonic()
        result = agent(request.user_prompt)
        latency_ms = (time.monotonic() - started) * 1000.0

        return ModelResponse(
            text=str(result),
            model_id=request.model_id,
            usage=_extract_usage(result),
            stop_reason=_extract_stop_reason(result),
            latency_ms=latency_ms,
            messages=list(getattr(agent, "messages", []) or []),
        )


def _extract_usage(result: Any) -> dict[str, Any]:
    """Pull token counts off a Strands AgentResult.

    Defensive because the shape has moved between Strands versions and the
    lifted code never touched it, so there is no established accessor to
    follow. An empty dict means "unknown", which the cost recorder handles.
    """
    metrics = getattr(result, "metrics", None)
    usage = getattr(metrics, "accumulated_usage", None) if metrics else None
    if usage is None:
        usage = getattr(result, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    out: dict[str, Any] = {}
    for attr in (
        "inputTokens",
        "outputTokens",
        "totalTokens",
        "cacheReadInputTokens",
        "cacheWriteInputTokens",
    ):
        value = getattr(usage, attr, None)
        if value is not None:
            out[attr] = value
    return out


def _extract_stop_reason(result: Any) -> str | None:
    reason = getattr(result, "stop_reason", None)
    return str(reason) if reason else None
