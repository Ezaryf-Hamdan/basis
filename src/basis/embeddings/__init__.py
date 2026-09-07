"""Embedding providers.

`ports.Embedder` is the protocol. Implementations that need no cloud SDK live
in `base`; `titan` needs the ``bedrock`` extra and is imported directly so this
package imports without boto3.
"""
from .base import HashEmbedder, NullEmbedder, OpenAICompatibleEmbedder

__all__ = ["HashEmbedder", "NullEmbedder", "OpenAICompatibleEmbedder"]
