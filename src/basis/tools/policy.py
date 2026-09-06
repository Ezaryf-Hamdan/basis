"""Fail-closed tool policy.

This is the most valuable thing in the lifted repo and the part that most
needed generalizing. ``tool_access.py`` gets the security model right and then
hardcodes the SAP delivery domain into it.

What is kept, because it is correct and hard-won:

  * **Exclude by default.** The original's comment is the whole design:
    "EXCLUDE-BY-DEFAULT (Cassandra Finding 1): a tool is KEPT only if it is a
    pure read/fetch family OR an explicitly-allowed proposal/trigger tool.
    Every writer ... and any FUTURE writer - is excluded because it matches no
    keep-prefix and is not in the explicit keep-set. Never write this as a
    denylist." A denylist fails open on the next tool someone adds; a keep-list
    fails closed. That property is preserved exactly.

  * **Unknown scope gets the most restrictive policy.** ``tools_for_job_type``
    gave an unrecognised job type reads only. Same here.

  * **One enforcement seam.** The original applied the filter at a single point
    (job_executor.py:3705) "before any handler/sub-agent handoff ... so it
    reaches AuthoringSession + every nested inline sub-agent". Filtering at one
    seam rather than per-call-site is why it actually holds.

What changes:

  * The SAP-specific sets are gone. ``_KNOWN_JOB_TYPES`` listed 27 delivery job
    types ("classify_fitgap", "generate_fsd_content", ...) and
    ``CO_AUTHOR_KEEP_EXACT`` named three FSD tools. Those are application
    policy, so they become constructor arguments. The source repo needed a
    drift test to keep that frozenset in sync with a dispatch table elsewhere
    in the codebase - injecting it removes the drift, and the need for the test.

  * Effects are explicit. The deck notes the registry has "no read/write effect
    distinction". Prefix matching already encodes one implicitly, so
    ``classify`` makes it a first-class answer and audit records can record it.

  * Per-tenant grants are expressible. ``allowed_names`` from the DB is
    intersected with the policy rather than replacing it, so a tenant grant can
    narrow the keep-set but never widen it past what the policy permits.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "READ_ONLY",
    "ToolEffect",
    "ToolPolicy",
    "extract_tool_info",
]


class ToolEffect(str, Enum):
    """What a tool does to the world.

    ``SIGNAL`` is the category the original discovered but did not name: tools
    like ``propose_fsd_sections`` and ``trigger_fsd_generation`` were kept in
    the read-only set with the comment "(pure signal, no write)". They are not
    reads, but they do not mutate durable state either - they hand a decision
    back to the orchestrator. Naming the category is what lets a run allow them
    without allowing writes.
    """

    READ = "read"
    SIGNAL = "signal"
    WRITE = "write"
    UNKNOWN = "unknown"


# Prefix families, lifted from CO_AUTHOR_KEEP_PREFIXES plus the writer prefixes
# the original enumerated in comments but never encoded ("create_/update_/
# delete_/link_/unlink_/assign_/unassign_/set_project_*/send_notification").
# Encoding the writer side does not weaken the keep-list - classification is
# for audit and for error messages. Authorization still comes from the
# keep-list alone.
DEFAULT_READ_PREFIXES: frozenset[str] = frozenset(
    {"get_", "list_", "search_", "browse_", "fetch_", "read_", "describe_"}
)

DEFAULT_WRITE_PREFIXES: frozenset[str] = frozenset(
    {
        "create_",
        "update_",
        "delete_",
        "patch_",
        "put_",
        "link_",
        "unlink_",
        "assign_",
        "unassign_",
        "set_",
        "send_",
        "approve_",
        "reject_",
        "publish_",
        "upload_",
        "write_",
    }
)


def extract_tool_info(tool: Any) -> dict[str, Any]:
    """Normalize a Strands MCPAgentTool, an MCP dict, or a plain dict.

    Lifted from ``tool_access._extract`` unchanged in behaviour - it handles
    both ``tool_spec`` attribute objects and dict forms, and both
    ``inputSchema`` and ``input_schema`` spellings, because MCP servers and
    Strands disagree about the casing.
    """
    if isinstance(tool, dict):
        spec = tool.get("tool_spec") or tool
        return {
            "name": spec.get("name") or tool.get("tool_name"),
            "description": spec.get("description", "") or "",
            "schema": spec.get("inputSchema") or spec.get("input_schema") or {},
        }
    spec = getattr(tool, "tool_spec", None) or {}
    return {
        "name": spec.get("name") or getattr(tool, "tool_name", None),
        "description": spec.get("description", "") or "",
        "schema": spec.get("inputSchema") or spec.get("input_schema") or {},
    }


@dataclass(frozen=True)
class ToolPolicy:
    """Which tools a run may see, and what they are allowed to do.

    Construct one per application (or per run scope) and hand it to the
    registry. The defaults are read-only, which is the safe thing to get if you
    forget to configure it.
    """

    read_prefixes: frozenset[str] = DEFAULT_READ_PREFIXES
    write_prefixes: frozenset[str] = DEFAULT_WRITE_PREFIXES

    # Non-read tools allowed by exact name. Replaces CO_AUTHOR_KEEP_EXACT.
    allow_exact: frozenset[str] = field(default_factory=frozenset)

    # Scopes (job types, run kinds) for which ``allow_exact`` applies at all.
    # Replaces _KNOWN_JOB_TYPES. An empty set means "no scope is known", so
    # allow_exact never applies - reads only.
    known_scopes: frozenset[str] = field(default_factory=frozenset)

    # Tools declared as signalling rather than writing, for classification.
    signal_names: frozenset[str] = field(default_factory=frozenset)

    # When False (the default), a service principal gets the same fail-closed
    # treatment as anyone else. The original noted that writer tools "under the
    # job service token run god-mode"; refusing to special-case service
    # identities is how that stops being true.
    allow_writes_for_service: bool = False

    def classify(self, tool_name: str) -> ToolEffect:
        """Best-effort effect for a tool name. Never used for authorization."""
        if not tool_name:
            return ToolEffect.UNKNOWN
        if tool_name in self.signal_names:
            return ToolEffect.SIGNAL
        if any(tool_name.startswith(p) for p in self.read_prefixes):
            return ToolEffect.READ
        if any(tool_name.startswith(p) for p in self.write_prefixes):
            return ToolEffect.WRITE
        return ToolEffect.UNKNOWN

    def allows(
        self,
        tool_name: str,
        *,
        scope: str | None = None,
        granted: frozenset[str] | None = None,
    ) -> bool:
        """The authorization decision. Keep-list only, never a denylist.

        ``scope`` is the run's job type / kind. ``granted`` is an optional
        per-tenant or per-persona grant set from the DB; when present it can
        only narrow the result.
        """
        if not tool_name:
            return False

        if any(tool_name.startswith(p) for p in self.read_prefixes) or (scope is not None and scope in self.known_scopes and tool_name in self.allow_exact):
            permitted = True
        else:
            # No keep-prefix match and no explicit allowance for a known scope.
            # Anything unrecognised - including every future writer - lands here.
            permitted = False

        if not permitted:
            return False
        # A grant set can only narrow. `None` means no persona-level narrowing;
        # an empty set means "granted nothing", which denies.
        return not (granted is not None and tool_name not in granted)

    def filter(
        self,
        tools: Iterable[Any],
        *,
        scope: str | None = None,
        granted: frozenset[str] | None = None,
    ) -> list[Any]:
        """Reduce a tool catalog to what this policy permits.

        Replaces ``filter_tools`` / ``co_author_tools`` / ``tools_for_job_type``,
        which were three functions doing the same thing against three different
        hardcoded sets.
        """
        kept = []
        for tool in tools:
            name = extract_tool_info(tool).get("name") or ""
            if self.allows(name, scope=scope, granted=granted):
                kept.append(tool)
        return kept

    def denied_reason(
        self,
        tool_name: str,
        *,
        scope: str | None = None,
        granted: frozenset[str] | None = None,
    ) -> str | None:
        """Why a tool was refused, for the audit record and the error message."""
        if self.allows(tool_name, scope=scope, granted=granted):
            return None
        if granted is not None and tool_name not in granted:
            return "not granted to this tenant/persona"
        effect = self.classify(tool_name)
        if effect is ToolEffect.WRITE:
            return "write-effect tool is not reachable in this scope"
        if scope is not None and scope not in self.known_scopes:
            return "unknown scope %r gets read-only tools" % (scope,)
        return "no keep-list match (excluded by default)"


#: A policy that permits reads and nothing else. Useful as an explicit default
#: and as the floor for an unrecognised scope.
READ_ONLY = ToolPolicy()
