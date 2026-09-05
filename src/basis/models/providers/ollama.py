"""Ollama / OpenAI-compatible provider.

A second `ModelProvider`, and the point of it is as much structural as
functional: a protocol with one implementation is an assumption wearing an
abstraction's clothes. Until this existed, `ModelProvider` was shaped entirely
by what Strands/Bedrock happened to do and there was no evidence the seam was
real.

Writing it surfaced three things the Bedrock-only design had baked in:

  * ``ModelRequest.enable_prompt_caching`` and ``read_timeout`` are
    Bedrock-specific. A provider must be free to ignore fields it has no
    concept of, so they are hints rather than instructions.
  * ``ModelResponse.messages`` assumed a Strands tool-loop transcript. A
    provider that does not run a tool loop returns an empty sequence, and
    `tools.capture.extract_tool_calls` already tolerates that.
  * Usage key casing differs per provider (Bedrock: ``inputTokens``, OpenAI:
    ``prompt_tokens``). `observability.genai.record_usage` reads both
    spellings, which was originally defensiveness and turns out to be
    necessary.

Targets the OpenAI-compatible `/v1/chat/completions` surface that Ollama, vLLM
and SGLang all expose, so one implementation covers the GPU-host stack rather
than three.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .base import ModelRequest, ModelResponse

__all__ = ["OllamaProvider"]

#: Prefixes this provider claims. Bedrock ids carry a region/vendor prefix
#: (`us.anthropic.`), so anything without one is assumed local.
_BEDROCK_PREFIXES = (
    "us.anthropic.",
    "us.amazon.",
    "eu.anthropic.",
    "apac.anthropic.",
    "anthropic.",
    "amazon.",
)


class OllamaProvider:
    """Serves models from an OpenAI-compatible endpoint.

    Uses ``urllib`` rather than adding an HTTP client to the core dependency
    set - this provider ships with no extra at all, so a consumer running
    local inference installs plain ``basis`` and nothing else.
    """

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        model_prefixes: tuple[str, ...] = (),
    ):
        self.base_url = (
            base_url or os.environ.get("BASIS_OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
        ).rstrip("/")
        self._api_key = api_key or os.environ.get("BASIS_OLLAMA_API_KEY")
        self._timeout = timeout
        # When set, only these prefixes are claimed. Use it when several local
        # providers coexist and each serves a distinct family.
        self._model_prefixes = model_prefixes

    def supports(self, model_id: str) -> bool:
        if self._model_prefixes:
            return model_id.startswith(self._model_prefixes)
        # Claim anything that is not obviously a Bedrock inference profile, so
        # the gateway's provider list can be ordered [Bedrock, Ollama] and each
        # picks up what it should.
        return not model_id.startswith(_BEDROCK_PREFIXES)

    def generate(self, request: ModelRequest) -> ModelResponse:
        """One blocking generation. The gateway offloads and classifies errors."""
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            # A cache-wrapped system prompt arrives as Bedrock content blocks;
            # flatten it, since this endpoint has no cachePoint concept.
            messages.append(
                {"role": "system", "content": _flatten(request.system_prompt)}
            )
        messages.append({"role": "user", "content": request.user_prompt})

        body: dict[str, Any] = {
            "model": request.model_id,
            "messages": messages,
            # Streaming is deliberately off here. The Bedrock provider streams
            # because botocore's read_timeout fires on long generations; this
            # path sets an explicit socket timeout instead, and a non-streaming
            # response is simpler to parse correctly.
            "stream": False,
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        # Local models do accept sampling parameters - the Claude 5 restriction
        # is a property of those models, not of every model, so this provider
        # passes them through.
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop_sequences:
            body["stop"] = list(request.stop_sequences)

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer %s" % self._api_key

        req = urllib.request.Request(
            "%s/v1/chat/completions" % self.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        started = time.monotonic()
        try:
            with urllib.request.urlopen(
                req, timeout=request.read_timeout or self._timeout
            ) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                "ollama %s returned %s: %s" % (self.base_url, exc.code, detail)
            ) from exc
        latency_ms = (time.monotonic() - started) * 1000.0

        choices = payload.get("choices") or []
        text = ""
        stop_reason = None
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
            stop_reason = choices[0].get("finish_reason")

        return ModelResponse(
            text=text,
            model_id=payload.get("model") or request.model_id,
            usage=dict(payload.get("usage") or {}),
            stop_reason=stop_reason,
            latency_ms=latency_ms,
            messages=(),
        )


def _flatten(system_prompt: Any) -> str:
    """Collapse Bedrock content blocks into plain text.

    `providers.bedrock.wrap_with_cache` may hand back
    ``[{"text": ...}, {"cachePoint": ...}]``. A provider with no caching
    concept must not send that structure verbatim.
    """
    if isinstance(system_prompt, str):
        return system_prompt
    if isinstance(system_prompt, (list, tuple)):
        return "\n".join(
            str(block["text"])
            for block in system_prompt
            if isinstance(block, dict) and "text" in block
        )
    return str(system_prompt)
