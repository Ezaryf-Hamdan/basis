"""Embedder implementations and a local/OpenAI-compatible provider.

`ports.Embedder` is the protocol. This module holds the implementations that do
not need a cloud SDK, plus `NullEmbedder` for the case where a caller
deliberately wants no vectors.

The inconsistency this fixes: the first cut gave chat models a proper provider
boundary and gave embeddings a concrete `TitanEmbedder` that
`memory.consolidator` imported directly. So a consumer running local inference
either got an ImportError or an unintended Bedrock call, from a module that had
nothing to do with AWS. There was no reason for the asymmetry.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence

__all__ = ["HashEmbedder", "NullEmbedder", "OpenAICompatibleEmbedder"]


class OpenAICompatibleEmbedder:
    """Embeddings from an OpenAI-compatible ``/v1/embeddings`` endpoint.

    Covers Ollama, vLLM, TEI and OpenAI itself. Uses ``urllib`` so it needs no
    extra dependency.

    ``dimensions`` must match the vector column width. It is not discovered
    from the endpoint: a mismatch surfaces as a Postgres cast error deep inside
    an insert, and the resulting message names neither the model nor the
    expected width. Failing loudly at construction is kinder.
    """

    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.dimensions = dimensions
        self.base_url = (
            base_url
            or os.environ.get("BASIS_EMBEDDING_BASE_URL")
            or "http://127.0.0.1:11434"
        ).rstrip("/")
        self._api_key = api_key or os.environ.get("BASIS_EMBEDDING_API_KEY")
        self._timeout = timeout

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("cannot embed empty text")
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch in one request.

        Unlike Titan, this surface accepts a list - so batching is a real
        win here rather than a loop with a nicer signature.
        """
        if not texts:
            return []

        body = {"model": self.model, "input": list(texts)}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer %s" % self._api_key

        req = urllib.request.Request(
            "%s/v1/embeddings" % self.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                "embedding endpoint %s returned %s: %s"
                % (self.base_url, exc.code, detail)
            ) from exc

        # The API returns items with an `index`; do not assume input order.
        items = sorted(payload.get("data") or [], key=lambda d: d.get("index", 0))
        vectors = [list(item["embedding"]) for item in items]

        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError(
                    "model %s returned %d dimensions but this embedder is "
                    "configured for %d; the vector column width must match"
                    % (self.model, len(vector), self.dimensions)
                )
        return vectors


class NullEmbedder:
    """Produces no vectors.

    For a deployment that wants memory and knowledge without semantic search -
    keyword retrieval and importance-ordered recall both still work. Explicit,
    so "no embeddings" is a configured choice rather than a missing dependency.
    """

    dimensions = 0

    def embed(self, text: str) -> list[float]:
        return []

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [[] for _ in texts]


class HashEmbedder:
    """Deterministic pseudo-embeddings from a hash of the text.

    **Not semantic.** Two paraphrases get unrelated vectors. It exists so
    ranking, fusion and storage paths can be tested end to end without a model
    server, and because a deterministic vector makes a failing test
    reproducible in a way a real embedder does not.

    Never use this in a deployment: retrieval quality would be indistinguishable
    from random.
    """

    def __init__(self, dimensions: int = 32):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        import hashlib
        import math

        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        # Stretch the digest to the requested width, then L2-normalize so
        # cosine distance behaves the way pgvector expects.
        raw = [
            digest[i % len(digest)] / 255.0 - 0.5 for i in range(self.dimensions)
        ]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
