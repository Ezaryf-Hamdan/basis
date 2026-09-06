"""Tool-call extraction from an agent transcript.

Lifted from ``tool_call_capture.py``. ``extract_tool_calls`` is the most
directly reusable function in the source repo: it parses a Strands
``agent.messages`` transcript into ordered (seq, name, args, result) tuples by
pairing ``toolUse`` blocks in assistant turns with ``toolResult`` blocks in the
following user turns, and it is defensive about malformed blocks in exactly the
way that a transcript coming out of a model requires.

Kept unchanged:
  * the ``toolUseId`` pairing, which is what makes this correct under parallel
    tool use - sequence position alone would mis-pair,
  * per-block ``try/except`` with a warning rather than a raise, so one
    malformed block does not lose the whole audit trail,
  * the untruncated ``raw_result_text`` contract, with truncation left to the
    caller. The original documents this explicitly ("callers apply [:2000] and
    compute result_truncated") and it matters: the audit layer needs to know a
    result *was* truncated.

Changed:
  * A ``ToolCall`` dataclass replaces the 4-tuple. The tuple was unpacked
    positionally at every call site.
  * Multi-block results are joined rather than dropped. The original read only
    ``raw_content[0].get('text','')``, so a tool returning several content
    blocks - which MCP permits - had everything after the first silently lost
    from the audit record.
  * The FIFO pruner is rewritten. The original used
    ``DELETE WHERE id NOT IN (SELECT id ... ORDER BY created_at DESC LIMIT 10000)``,
    which re-scans and re-sorts the whole table on every call and holds a long
    lock. It is also global, so a busy tenant could evict a quiet tenant's
    audit history. The replacement prunes per tenant by timestamp cutoff.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..storage import ToolRepository

__all__ = ["ToolCall", "extract_tool_calls", "prune_invocations"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    """One completed tool call from a transcript."""

    seq: int
    name: str
    args: dict[str, Any]
    result_text: str

    def truncated(self, limit: int = 2000) -> tuple[str, bool]:
        """(text, was_truncated) - the contract the audit layer needs."""
        if len(self.result_text) <= limit:
            return self.result_text, False
        return self.result_text[:limit], True


def _result_text(raw_content: Any) -> str:
    """Join every text block in a toolResult, not just the first."""
    if not isinstance(raw_content, list):
        return ""
    parts: list[str] = []
    for block in raw_content:
        if isinstance(block, dict) and "text" in block:
            parts.append(str(block["text"]))
    return "\n".join(parts)


def extract_tool_calls(messages: Iterable[Any]) -> list[ToolCall]:
    """Parse an agent transcript into ordered tool calls.

    ``seq`` is the 0-indexed order in which each ``toolUse`` first appeared
    across all turns. Only calls that received a result are returned; an
    in-flight call at the end of a transcript is omitted, as in the original.
    """
    pending: dict[str, tuple[int, str, dict[str, Any]]] = {}
    completed: list[ToolCall] = []
    seq = 0

    for message in messages:
        try:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "")
            content = message.get("content", [])
            if not isinstance(content, list):
                continue

            if role == "assistant":
                for block in content:
                    try:
                        if not isinstance(block, dict) or "toolUse" not in block:
                            continue
                        tu = block["toolUse"]
                        pending[tu["toolUseId"]] = (seq, tu["name"], tu.get("input", {}))
                        seq += 1
                    except (KeyError, TypeError, IndexError) as exc:
                        log.warning("skip malformed toolUse block: %s", exc)

            elif role == "user":
                for block in content:
                    try:
                        if not isinstance(block, dict) or "toolResult" not in block:
                            continue
                        tr = block["toolResult"]
                        tool_use_id = tr["toolUseId"]
                        entry = pending.pop(tool_use_id, None)
                        if entry is None:
                            continue
                        entry_seq, name, args = entry
                        completed.append(
                            ToolCall(
                                seq=entry_seq,
                                name=name,
                                args=args,
                                result_text=_result_text(tr.get("content", [])),
                            )
                        )
                    except (KeyError, TypeError, IndexError, AttributeError) as exc:
                        log.warning("skip malformed toolResult block: %s", exc)

        except (KeyError, TypeError, AttributeError) as exc:
            log.warning("skip malformed message: %s", exc)

    completed.sort(key=lambda c: c.seq)
    return completed


def prune_invocations(
    *,
    tenant_id: str,
    retain_days: int = 90,
    dsn: str | None = None,
    repo: ToolRepository | None = None,
) -> int:
    """Delete tool-invocation audit rows older than ``retain_days`` for a tenant.

    Retention by age, per tenant, rather than the original's global 10,000-row
    FIFO. Two reasons: a row count is not a retention policy anyone can put in
    a contract, and a global cap lets one tenant's activity delete another's
    audit trail. Returns the number of rows removed.
    """
    if repo is None:
        from ..storage.postgres import PgToolRepository

        repo = PgToolRepository(dsn=dsn)
    return repo.prune_invocations(tenant_id=tenant_id, retain_days=retain_days)
