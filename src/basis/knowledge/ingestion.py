"""End-to-end ingestion: text → chunks → embeddings → retriever.

Without this, every consumer re-implements the same three-step loop. The loop
is not complicated but the choices inside it are subtle: chunk size interacts
with embedding model context windows, and re-ingestion of a document must
delete the previous chunks rather than accumulate duplicates.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from .chunking import CHUNK_OVERLAP, CHUNK_SIZE, MAX_CHUNKS, chunk_text

__all__ = ["ingest"]


class _Embedder(Protocol):
    def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...


class _Retriever(Protocol):
    def add(self, chunks: Sequence[Any]) -> int: ...

    def replace_document(
        self, *, document_id: str, tenant_id: str | None, chunks: Sequence[Any]
    ) -> int: ...


def ingest(
    text: str,
    *,
    embedder: _Embedder,
    retriever: _Retriever,
    tenant_id: str | None,
    project_id: str | None = None,
    corpus: str,
    document_id: str | None = None,
    collection_id: str | None = None,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    max_chunks: int | None = MAX_CHUNKS,
    replace: bool = True,
) -> int:
    """Chunk, embed and store a document in one call.

    ``replace=True`` (the default) replaces any existing chunks for
    ``document_id``, so re-ingestion is idempotent. Set it to ``False`` when
    adding new sections to a partially-ingested document, or when no
    ``document_id`` is available.

    ``document_id`` must be a UUID string - it maps to a ``uuid`` column. A
    natural key like ``"REQ-001"`` raises ``InvalidTextRepresentation``; put
    that in ``metadata`` or ``title`` instead.

    Returns the number of chunks stored.

    Example::

        n = ingest(
            full_text,
            embedder=embedder,
            retriever=HybridRetriever(PgChunkRepository(dsn=DSN)),
            tenant_id=ctx.tenant_id,
            project_id=ctx.project_id,
            corpus="client_project",
            document_id=str(uuid.uuid4()),
            metadata={"artifact_key": "REQ-001"},
        )
    """
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap, max_chunks=max_chunks)
    if not chunks:
        return 0

    # embed_many, not a loop over embed(). For OpenAI-compatible endpoints the
    # whole batch goes in one request, so a 100-chunk document costs 1 round
    # trip rather than 100. Titan has no batch endpoint and loops internally,
    # so this is never worse.
    embeddings = embedder.embed_many([c.text for c in chunks])

    rows = [
        {
            "tenant_id": tenant_id,
            "project_id": project_id,
            "corpus": corpus,
            "document_id": document_id,
            "collection_id": collection_id,
            "ordinal": c.ordinal,
            "title": title,
            "content": c.text,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        for c, embedding in zip(chunks, embeddings, strict=True)
    ]

    if replace and document_id is not None:
        # Wholesale replacement, not a diff - chunk boundaries move when text
        # changes, so a surviving stale chunk is worse than a re-embedded one.
        return retriever.replace_document(
            document_id=document_id, tenant_id=tenant_id, chunks=rows
        )
    return retriever.add(rows)
