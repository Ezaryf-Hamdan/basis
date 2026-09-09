"""Text chunking.

The one genuinely liftable piece of `document_ingest.py`: `chunk_text` plus its
`CHUNK_SIZE` / `CHUNK_OVERLAP` / `MAX_CHUNKS` constants. Everything else in
that 981-line module is SAP delivery-document analysis.

The lifted constants are kept because they encode a real decision - a 24,000
character window with 2,000 characters of overlap, capped at 20 chunks, was
sized against the same bound as the truncation it replaced. That is a tuned
value, not a guess, so it stays the default.

What is added: chunking on natural boundaries. The original sliced at a fixed
character offset, which cuts mid-word and mid-sentence. For a summarization
pass that is survivable; for embedding it is not, because a chunk beginning
mid-sentence embeds badly and retrieves badly. `chunk_text` therefore prefers
to break at a paragraph, then a sentence, then a word, and only slices blindly
if none of those exist within the window.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "MAX_CHUNKS",
    "Chunk",
    "chunk_text",
]

#: Lifted from document_ingest.py.
CHUNK_SIZE = 24_000  # max chars per window
CHUNK_OVERLAP = 2_000  # chars shared between consecutive windows
MAX_CHUNKS = 20

#: Break preferences, most to least desirable. A paragraph break is a real
#: semantic boundary; a bare space is a last resort before slicing.
_BOUNDARIES: Sequence[str] = ("\n\n", "\n", ". ", "? ", "! ", "; ", " ")


@dataclass(frozen=True)
class Chunk:
    """One window of text, with its position in the source."""

    ordinal: int
    text: str
    start: int
    end: int

    def __len__(self) -> int:
        return len(self.text)


def _find_break(text: str, window_end: int, floor: int) -> int:
    """Best boundary at or before ``window_end``, else ``window_end``.

    ``floor`` stops a pathological case: a window with its only boundary near
    the start would otherwise produce a tiny chunk and make no progress. If no
    boundary appears in the back half of the window, we slice.
    """
    for marker in _BOUNDARIES:
        idx = text.rfind(marker, floor, window_end)
        if idx > floor:
            return idx + len(marker)
    return window_end


def chunk_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    max_chunks: int | None = MAX_CHUNKS,
    respect_boundaries: bool = True,
) -> list[Chunk]:
    """Split text into overlapping windows.

    Returns [] for empty input, and a single chunk when the text fits - so a
    caller does not need to special-case short documents.
    """
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError(
            "overlap (%d) must be smaller than chunk_size (%d) or the window "
            "never advances" % (overlap, chunk_size)
        )

    if len(text) <= chunk_size:
        return [Chunk(ordinal=0, text=text, start=0, end=len(text))]

    chunks: list[Chunk] = []
    start = 0
    ordinal = 0

    while start < len(text):
        window_end = min(start + chunk_size, len(text))

        if window_end < len(text) and respect_boundaries:
            # Only look for a boundary in the back half, so a chunk is never
            # less than half the requested size.
            end = _find_break(text, window_end, start + chunk_size // 2)
        else:
            end = window_end

        piece = text[start:end]
        if piece.strip():
            chunks.append(
                Chunk(ordinal=ordinal, text=piece, start=start, end=end)
            )
            ordinal += 1

        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        if end >= len(text):
            break

        advance = end - overlap
        # Guarantee forward progress even if the boundary search returned
        # something close to `start`.
        start = advance if advance > start else end

    return chunks
