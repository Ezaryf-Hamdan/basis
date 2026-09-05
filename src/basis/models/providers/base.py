"""The provider boundary.

This is the refactor the deck asks for: "turning direct SDK calls into
``ModelProvider`` implementations behind ``ModelGateway``, so agents stop
importing ``boto3``."

In the lifted code there is no boundary. ``model_config`` imports
``strands.models.BedrockModel`` and ``botocore.config`` directly and returns a
``BedrockModel`` from ``get_model()``, so every caller is typed against
Bedrock. ``memory_consolidator._embed`` constructs a ``boto3`` client inline.
``reflection`` and ``inter_agent`` each build their own ``BedrockModel`` with a
hardcoded model id. There are at least four independent paths to the SDK.

The Protocol below is what the gateway depends on. ai-core matters here: per
the architecture it runs Ollama / vLLM / SGLang on a GPU host, so a second
provider is a near-term requirement rather than a hypothetical, and the
gateway must not be typed against Bedrock to accommodate it.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...ports import Completion

__all__ = ["ModelProvider", "ModelRequest", "ModelResponse"]


@dataclass(frozen=True)
class ModelRequest:
    """A provider-neutral generation request."""

    model_id: str
    system_prompt: str | None = None
    user_prompt: str = ""
    tools: Sequence[Any] = field(default_factory=tuple)
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: Sequence[str] = field(default_factory=tuple)
    streaming: bool = True
    enable_prompt_caching: bool = True
    read_timeout: float = 300.0
    # Free-form provider passthrough for options that do not generalize.
    extra: dict[str, Any] = field(default_factory=dict)


#: A provider-neutral generation result.
#:
#: Defined in ``basis.ports`` as ``Completion`` and aliased here. The alias
#: rather than a second dataclass is deliberate: ``agents``, ``memory`` and
#: ``personas`` all receive one of these, and if they had to import this module
#: to name the type they would be coupled to the model layer for a data class.
#:
#: ``usage`` is the field the lifted code never surfaced -
#: ``invoke_with_fallback`` returned ``(text, model_id)`` and discarded the
#: token counts, which is the mechanical reason per-client cost attribution did
#: not exist.
ModelResponse = Completion


@runtime_checkable
class ModelProvider(Protocol):
    """What the gateway requires of a backend.

    ``generate`` is synchronous and blocking. The gateway offloads it through
    ``concurrency.run_in_context``, which matches how the lifted code already
    works (Strands and boto3 are both blocking) and keeps providers simple.
    """

    name: str

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Run one generation. Raise on failure; the gateway classifies."""
        ...

    def supports(self, model_id: str) -> bool:
        """Whether this provider can serve the given model id."""
        ...
