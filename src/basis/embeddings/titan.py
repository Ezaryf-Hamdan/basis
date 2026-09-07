"""Titan text embeddings.

Extracted from ``memory_consolidator._embed``, which was a private function
inside the consolidation module::

    def _embed(text: str) -> List[float]:
        import boto3
        client = boto3.client("bedrock-runtime")
        ...

Three problems with it there, all fixed here:

  * **It was unreachable.** Being private inside the consolidator meant the
    one working embedding path in the Python tier could not be used by
    anything else - which is part of why memory recall never used vectors and
    why retrieval lives in the Node tier instead.
  * **A client per call.** ``boto3.client()`` inside the function, called once
    per memory in a loop, so a consolidation writing five memories built five
    clients. The client is cached here.
  * **No batching.** Titan v2 accepts one input per request, but the loop had
    no concurrency either, so N memories meant N serial round trips.
    ``embed_many`` keeps it explicit and bounded.

Dimensions and normalization follow the original: 1024 dimensions, normalized,
matching the ``vector(1024)`` column.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from typing import Any

from ..settings import settings

__all__ = ["TitanEmbedder"]


class TitanEmbedder:
    """Bedrock Titan embeddings, with a reused client."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        dimensions: int | None = None,
        region: str | None = None,
    ):
        cfg = settings()
        self.model_id = model_id or cfg.embedding_model_id
        self.dimensions = dimensions or cfg.embedding_dimensions
        self._region = region or cfg.aws_region
        self._client: Any = None
        self._lock = threading.Lock()

    def _get_client(self) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import boto3

                    self._client = boto3.client(
                        "bedrock-runtime", region_name=self._region
                    )
        return self._client

    def embed(self, text: str) -> list[float]:
        """Embed one string. Blocking; offload via ``concurrency.run_in_context``."""
        if not text or not text.strip():
            raise ValueError("cannot embed empty text")
        body = json.dumps(
            {
                "inputText": text,
                "dimensions": self.dimensions,
                "normalize": True,
            }
        )
        resp = self._get_client().invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        payload = json.loads(resp["body"].read())
        return payload["embedding"]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed several strings, in order.

        Serial by design: Titan has no batch endpoint, and firing N concurrent
        InvokeModel calls is a reliable way to get throttled. Callers that need
        throughput should use ``concurrency.gather_bounded`` around ``embed``
        with a limit they have measured.
        """
        return [self.embed(t) for t in texts]
