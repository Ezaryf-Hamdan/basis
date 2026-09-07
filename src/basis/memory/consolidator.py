"""End-of-run consolidation: scratch notes to durable memories.

Lifted from ``memory_consolidator.py``, which is well-structured - the prompt,
the JSON extraction with a fenced-block regex and a bracket-scan fallback, and
the "failures here are non-fatal" contract are all kept.

The significant change is that the original was not awaitable. ``_invoke_consolidator``
did this::

    def _invoke_consolidator(prompt: str) -> str:
        from model_config import invoke_with_fallback
        async def _run() -> str: ...
        return asyncio.run(_run())

``asyncio.run`` inside a synchronous function called from an async job raises
``RuntimeError: asyncio.run() cannot be called from a running event loop``.
The only way that shipped without firing is if consolidation ran exclusively
from a worker thread with no loop of its own - which the ``run_in_executor``
pattern elsewhere makes plausible, but it makes ``consolidate_job`` unsafe to
call from anywhere else, and it is a latent failure the moment someone awaits
it. ``consolidate`` here is a coroutine and awaits the gateway directly.

Also fixed: the per-memory ``_embed`` call was inside the write loop with its
own boto3 client construction. Embeddings are now taken once, from a shared
embedder.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from ..concurrency import offload
from ..context import RunContext, require_context
from ..ports import Embedder, ModelClient
from .scope import MemoryScope
from .service import MemoryService

__all__ = [
    "CONSOLIDATION_TASK_KEY",
    "build_consolidation_prompt",
    "consolidate",
    "group_notes_by_type",
    "parse_consolidation_response",
]

log = logging.getLogger(__name__)

CONSOLIDATION_TASK_KEY = "memory_consolidator"

_SYSTEM_PROMPT = (
    "You distill scratch notes into durable memories. Output JSON only."
)

# Lifted verbatim - the shape of this prompt (explicit types, an importance
# scale with anchors, and an instruction to discard noise with a 0-5 target)
# is doing real work and there is no reason to reword it.
_CONSOLIDATION_INSTRUCTIONS = """You are a memory consolidator for an AI agent persona.
Your job: read scratch notes from a single job and decide which observations are
worth remembering long-term.

Persona: {persona_name}

Scratch notes from this job:
{notes_block}

Output ONLY a JSON array of memories. Each memory has:
  memory_type: "fact" | "preference" | "decision" | "pattern"
  content:     concise statement (1-2 sentences) that will help future jobs
  importance:  float in [0,1]; 1.0 = critical for future work, 0.3 = mildly useful

Discard noise. Aim for 0-5 high-quality memories per job. If nothing is worth
keeping, return [].

Output format:
```json
[
  {{"memory_type":"fact","content":"...","importance":0.8}}
]
```"""

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)

_VALID_TYPES = frozenset({"fact", "preference", "decision", "pattern"})


def group_notes_by_type(
    notes: Sequence[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Bucket notes by ``note_type``. Lifted unchanged."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for note in notes:
        grouped.setdefault(note.get("note_type", "other"), []).append(note)
    return grouped


def build_consolidation_prompt(
    persona_name: str, notes: Sequence[dict[str, Any]]
) -> str:
    """Render the consolidation prompt, or "" when there is nothing to do."""
    if not notes:
        return ""
    lines = [
        "- [%s] %s" % (n.get("note_type", "?"), n.get("content", "")) for n in notes
    ]
    return _CONSOLIDATION_INSTRUCTIONS.format(
        persona_name=persona_name,
        notes_block="\n".join(lines),
    )


def parse_consolidation_response(raw: str) -> list[dict[str, Any]]:
    """Extract the memory array from a model response.

    Lifted, with the fenced-block-then-bracket-scan strategy intact. Added: the
    ``memory_type`` is validated against the four values the prompt asks for
    and the importance is coerced into [0,1]. The original accepted whatever
    string the model produced and passed it to an insert, so a hallucinated
    type became a permanent row.
    """
    if not raw:
        return []
    text = raw.strip()

    match = _JSON_BLOCK.search(text)
    if match:
        text = match.group(1)
    else:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1 or end < start:
            return []
        text = text[start : end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "memory_type" not in item or "content" not in item:
            continue
        memory_type = str(item["memory_type"]).strip().lower()
        if memory_type not in _VALID_TYPES:
            log.debug("discarding memory with unknown type %r", memory_type)
            continue
        content = str(item["content"]).strip()
        if not content:
            continue
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        out.append(
            {
                "memory_type": memory_type,
                "content": content,
                "importance": min(1.0, max(0.0, importance)),
            }
        )
    return out


async def consolidate(
    *,
    gateway: ModelClient,
    memory: MemoryService,
    persona_name: str,
    scope: MemoryScope | None = None,
    ctx: RunContext | None = None,
    embedder: Embedder | None = None,
    clear_notes: bool = False,
) -> int:
    """Read scratch notes, distill them, write durable memories.

    Returns the number of memories written. Best-effort throughout: the run has
    already produced its primary output and a missed consolidation corrupts
    nothing, which is the contract the original documented and is worth
    keeping. Failures are logged rather than raised.
    """
    run = ctx if ctx is not None else require_context()
    memory_scope = scope if scope is not None else MemoryScope.from_context(run)

    # Offloaded: `MemoryService` is synchronous by design (sync consumers
    # exist), so an async caller must not run its I/O on the event loop.
    notes = await offload(memory.read_short_term, memory_scope)
    if not notes:
        return 0

    prompt = build_consolidation_prompt(persona_name, notes)
    if not prompt:
        return 0

    try:
        response = await gateway.invoke(
            CONSOLIDATION_TASK_KEY,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=prompt,
            ctx=run,
        )
    except Exception as exc:
        log.warning("consolidator model call failed: %s", exc)
        return 0

    memories = parse_consolidation_response(response.text)
    if not memories:
        return 0

    # No implicit default. The first cut reached for TitanEmbedder here, which
    # silently bound this function to Bedrock - a caller running local
    # inference got an ImportError or an unwanted AWS call. An absent embedder
    # now means "store without vectors", which is a decision the caller made.
    if embedder is None:
        log.info("no embedder supplied; storing memories without vectors")

    written = 0
    for item in memories:
        embedding = None
        if embedder is not None:
            try:
                embedding = await offload(embedder.embed, item["content"])
            except Exception as exc:
                # Write it anyway. A memory without a vector is still readable
                # by the importance-ordered fallback path; losing it entirely
                # because the embedding endpoint was throttled is worse.
                log.warning("embedding failed, storing without vector: %s", exc)
        try:
            await offload(
                memory.write_long_term,
                memory_scope,
                memory_type=item["memory_type"],
                content=item["content"],
                importance=item["importance"],
                embedding=embedding,
                source="consolidation",
            )
            written += 1
        except Exception as exc:
            log.warning("failed to persist consolidated memory: %s", exc)

    if clear_notes and written:
        try:
            await offload(memory.clear_short_term, memory_scope)
        except Exception as exc:
            log.warning("failed to clear short-term notes: %s", exc)

    return written
