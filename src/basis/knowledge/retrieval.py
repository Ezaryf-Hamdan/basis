"""Retrieval: corpora, hybrid search, and reranking.

**Built from scratch.** The deck rates `knowledge` the strongest lift
candidate; reading the code says otherwise. There is no vector retrieval in
AILedSDLC's Python tier at all - `document_ingest.py` contains no reference to
pgvector, embeddings or similarity search. The working pipeline is in the Node
tier (`server/src/modules/precedents/`, `rag-eval/`, `search/`, plus a
`bedrock-kb.service.js`), and Python reaches it over REST at
`fsd_content_generator.py:518`. So this is new code, not a port.

Three things it provides that neither system had in Python:

**Corpus scoping as a first-class predicate.** The three-corpus split the deck
identifies - client project, public domain, Accenture assets - is a
*permission* boundary, not a filing convenience. Client project content must
never surface in another client's retrieval. So `Corpus` is part of the query
and part of the SQL `WHERE`, and the tenant predicate is non-negotiable.

**Hybrid search.** Vector similarity alone misses exact terms - part numbers,
transaction codes, field names - which is precisely the vocabulary of SAP
delivery work. Postgres full-text search alone misses paraphrase. Combining
them with Reciprocal Rank Fusion gets both, and RRF needs no score
normalization between the two (a real problem, since cosine distance and
`ts_rank` are not on comparable scales).

**A reranker.** The deck notes "no reranker anywhere in either system", and
that is the single highest-leverage retrieval improvement available. The
`Reranker` protocol is the seam; `RRFReranker` is a dependency-free default
that already beats either ranking alone.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from ..context import RunContext, require_context
from ..errors import TenantIsolationError
from ..storage import ChunkRepository

__all__ = [
    "Corpus",
    "HybridRetriever",
    "PgVectorRetriever",
    "RRFReranker",
    "Reranker",
    "RetrievalQuery",
    "RetrievedChunk",
    "Retriever",
]

log = logging.getLogger(__name__)


class Corpus(str, Enum):
    """Which body of knowledge a query may read.

    Maps onto the deck's `KnowledgeBase` split. The names are deliberately
    about *ownership*, because that is what determines who may see them.
    """

    #: This client's own project material. Tenant- and project-scoped.
    CLIENT_PROJECT = "client_project"
    #: Reusable internal assets. Tenant-scoped, project-agnostic.
    ORG_ASSETS = "org_assets"
    #: Public reference material. Readable by any tenant.
    PUBLIC_DOMAIN = "public_domain"

    @property
    def is_tenant_scoped(self) -> bool:
        return self is not Corpus.PUBLIC_DOMAIN

    @property
    def is_project_scoped(self) -> bool:
        return self is Corpus.CLIENT_PROJECT


@dataclass(frozen=True)
class RetrievalQuery:
    """A retrieval request."""

    text: str
    corpora: tuple[Corpus, ...] = (Corpus.CLIENT_PROJECT,)
    embedding: Sequence[float] | None = None
    limit: int = 10
    #: How many candidates each arm fetches before fusion. Larger than
    #: ``limit`` on purpose: fusion can only reorder what it is given, so a
    #: candidate pool the size of the result set makes reranking pointless.
    candidates: int = 50
    min_similarity: float | None = None
    collection_ids: tuple[str, ...] = ()
    metadata_filter: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.corpora:
            raise ValueError("a retrieval query must name at least one corpus")
        if self.limit < 1:
            raise ValueError("limit must be positive")


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieval hit."""

    id: str
    content: str
    corpus: Corpus
    document_id: str | None = None
    collection_id: str | None = None
    ordinal: int | None = None
    title: str | None = None
    #: Cosine distance, when the vector arm matched. Lower is closer.
    distance: float | None = None
    #: Postgres ts_rank, when the lexical arm matched. Higher is better.
    text_rank: float | None = None
    #: Post-fusion score. Higher is better.
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def citation(self) -> str:
        """A stable citation string.

        The lifted `reflection.py` prompt shows the convention this platform
        already uses - ``[Source: Title - Section]`` - and preserving it means
        existing prompts keep working.
        """
        parts = [p for p in (self.title, self.document_id) if p]
        return "[Source: %s]" % (" - ".join(parts) or self.id)


@runtime_checkable
class Retriever(Protocol):
    """A source of chunks."""

    def search(
        self, query: RetrievalQuery, *, ctx: RunContext | None = None
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidates. May also drop them."""

    def rerank(
        self, query: RetrievalQuery, candidates: Sequence[RetrievedChunk]
    ) -> list[RetrievedChunk]: ...


class RRFReranker:
    """Reciprocal Rank Fusion over the vector and lexical rankings.

    ``score = sum over arms of 1 / (k + rank)``

    RRF is used rather than a weighted score sum because the two arms produce
    incomparable numbers: cosine distance is a bounded metric where lower is
    better, `ts_rank` is an unbounded relevance score where higher is better.
    Normalizing them against each other requires knowing their distributions,
    which vary per corpus and per query. Ranks have no such problem.

    ``k`` damps the influence of top positions; 60 is the value from the
    original RRF paper and behaves well without tuning.
    """

    def __init__(self, k: int = 60, *, weights: tuple[float, float] = (1.0, 1.0)):
        self.k = k
        self.vector_weight, self.text_weight = weights

    def rerank(
        self, query: RetrievalQuery, candidates: Sequence[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        by_vector = [c for c in candidates if c.distance is not None]
        by_vector.sort(key=lambda c: c.distance or 0.0)

        by_text = [c for c in candidates if c.text_rank is not None]
        by_text.sort(key=lambda c: -(c.text_rank or 0.0))

        scores: dict[str, float] = {c.id: 0.0 for c in candidates}
        for rank, chunk in enumerate(by_vector, start=1):
            scores[chunk.id] += self.vector_weight / (self.k + rank)
        for rank, chunk in enumerate(by_text, start=1):
            scores[chunk.id] += self.text_weight / (self.k + rank)

        fused = [
            RetrievedChunk(
                id=c.id,
                content=c.content,
                corpus=c.corpus,
                document_id=c.document_id,
                collection_id=c.collection_id,
                ordinal=c.ordinal,
                title=c.title,
                distance=c.distance,
                text_rank=c.text_rank,
                score=scores[c.id],
                metadata=c.metadata,
            )
            for c in candidates
        ]
        fused.sort(key=lambda c: -c.score)
        return fused[: query.limit]


class HybridRetriever:
    """Corpus-scoped hybrid retrieval over a `ChunkRepository`.

    The tenant and corpus predicates are built here and applied *inside* the
    repository query - not to the results afterwards. That distinction is the
    difference between isolation and a filter: post-filtering means the store
    read another tenant's rows and the application chose to discard them, which
    is not a guarantee.
    """

    def __init__(
        self,
        repo: ChunkRepository | None = None,
        *,
        dsn: str | None = None,
        reranker: Reranker | None = None,
        table: str | None = None,
    ):
        if repo is None:
            from ..storage import TableMap
            from ..storage.postgres import PgChunkRepository

            tables = TableMap(chunks=table) if table else None
            repo = PgChunkRepository(dsn=dsn, tables=tables)
        self._repo = repo
        self._reranker = reranker or RRFReranker()

    def add(self, chunks: Sequence[Mapping[str, Any]]) -> int:
        """Store chunks. Each needs at least ``corpus`` and ``content``."""
        return self._repo.add_chunks(list(chunks))

    def replace_document(
        self, *, document_id: str, tenant_id: str | None, chunks: Sequence[Mapping[str, Any]]
    ) -> int:
        """Re-ingestion: drop a document's chunks and write the new set.

        Wholesale replacement rather than a diff. Chunk boundaries move when
        text changes, so matching old chunks to new ones is guesswork - and a
        stale chunk that survives re-ingestion is worse than a re-embedded one.
        """
        self._repo.delete_document(tenant_id=tenant_id, document_id=document_id)
        return self._repo.add_chunks(list(chunks))

    def search(
        self, query: RetrievalQuery, *, ctx: RunContext | None = None
    ) -> list[RetrievedChunk]:
        run = ctx if ctx is not None else require_context()

        if any(c.is_project_scoped for c in query.corpora) and not run.project_id:
            raise TenantIsolationError(
                "query includes a project-scoped corpus but run %s has no "
                "project_id" % run.run_id
            )

        scope = self._scope(query, run)
        candidates: dict[str, RetrievedChunk] = {}

        if query.embedding is not None:
            max_distance = (
                1.0 - query.min_similarity
                if query.min_similarity is not None
                else None
            )
            for row in self._repo.search_vector(
                scope=scope,
                embedding=query.embedding,
                limit=query.candidates,
                max_distance=max_distance,
            ):
                chunk = self._to_chunk(row, distance=row.get("distance"))
                candidates[chunk.id] = chunk

        for row in self._repo.search_text(
            scope=scope, text=query.text, limit=query.candidates
        ):
            chunk = self._to_chunk(row, text_rank=row.get("text_rank"))
            existing = candidates.get(chunk.id)
            if existing is None:
                candidates[chunk.id] = chunk
            else:
                # Found by both arms: keep both signals so RRF credits it twice.
                candidates[chunk.id] = replace(
                    existing, text_rank=chunk.text_rank
                )

        if not candidates:
            return []
        return self._reranker.rerank(query, list(candidates.values()))

    # ── scope ──────────────────────────────────────────────────────────────

    @staticmethod
    def _scope(query: RetrievalQuery, run: RunContext) -> dict[str, Any]:
        """The corpus/tenant predicate, as SQL plus params plus a corpus list.

        One OR group per corpus, so a single query can span this client's
        project *and* public reference material without ever letting another
        tenant's client-project row through. The ``corpora`` list lets a
        non-SQL backend evaluate the same rules.
        """
        clauses: list[str] = []
        params: dict[str, Any] = {
            "tenant": run.tenant_id,
            "project": run.project_id,
        }

        for corpus in query.corpora:
            if corpus is Corpus.CLIENT_PROJECT:
                clauses.append(
                    "(corpus = 'client_project' AND tenant_id = %(tenant)s "
                    "AND project_id = %(project)s)"
                )
            elif corpus is Corpus.ORG_ASSETS:
                clauses.append(
                    "(corpus = 'org_assets' AND tenant_id = %(tenant)s)"
                )
            elif corpus is Corpus.PUBLIC_DOMAIN:
                clauses.append("(corpus = 'public_domain')")

        sql = "(" + " OR ".join(clauses) + ")"

        if query.collection_ids:
            sql += " AND collection_id = ANY(%(collections)s)"
            params["collections"] = list(query.collection_ids)

        return {
            "sql": sql,
            "params": params,
            "corpora": [c.value for c in query.corpora],
        }

    @staticmethod
    def _to_chunk(
        row: Mapping[str, Any],
        *,
        distance: float | None = None,
        text_rank: float | None = None,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            id=str(row["id"]),
            content=row["content"],
            corpus=Corpus(row["corpus"]),
            document_id=str(row["document_id"]) if row.get("document_id") else None,
            collection_id=(
                str(row["collection_id"]) if row.get("collection_id") else None
            ),
            ordinal=row.get("ordinal"),
            title=row.get("title"),
            distance=float(distance) if distance is not None else None,
            text_rank=float(text_rank) if text_rank is not None else None,
            metadata=dict(row.get("metadata") or {}),
        )


#: Previous name, kept so existing wiring keeps working. The class is no longer
#: Postgres-specific - it takes any `ChunkRepository` - so the name was wrong.
PgVectorRetriever = HybridRetriever
