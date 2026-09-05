"""Model providers.

`ModelProvider` is the protocol. `ollama` needs no extra dependency and is
re-exported; `bedrock` needs the ``bedrock`` extra and is imported lazily by
the gateway so it stays optional.
"""
from .base import ModelProvider, ModelRequest, ModelResponse
from .ollama import OllamaProvider

__all__ = [
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "OllamaProvider",
]
