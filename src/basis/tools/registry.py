"""Tool catalog sync and the enforcement seam.

Lifted from ``tool_access.py``'s ``sync_tools_from_mcp`` / ``get_allowed_tools``,
with the pieces the deck flags as missing added: per-tenant grants, and an
audit record on invocation.

The sync logic is kept as-is because it is right: upsert everything the MCP
server currently advertises, then deactivate anything in the DB that was not
seen, so a tool removed upstream stops being grantable without anyone running a
migration.

The change is tenancy. ``agent_tools`` and ``agent_persona_tools`` had no
tenant column, so a grant was global: enabling a tool for a persona enabled it
for every client that persona served. ``migrations/0001_basis_schema.sql``
adds the column, and ``grants_for`` filters on it.

The SQL itself now lives in `storage.postgres.PgToolRepository`, so this class
holds only the policy decisions - which tools a run may see, and what gets
recorded when one runs.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

from ..context import RunContext, require_context
from ..errors import ToolDenied
from ..storage import ToolRepository
from .policy import ToolPolicy, extract_tool_info

__all__ = ["ToolRegistry"]

log = logging.getLogger(__name__)


class ToolRegistry:
    """The single seam through which a run obtains and invokes tools."""

    def __init__(
        self,
        policy: ToolPolicy,
        *,
        repo: ToolRepository | None = None,
        dsn: str | None = None,
        audit: bool = True,
        audit_sink: Any = None,
        grants_provider: Any = None,
    ):
        self.policy = policy
        if repo is None:
            from ..storage.postgres import PgToolRepository

            repo = PgToolRepository(dsn=dsn)
        self._repo = repo
        self._audit = audit
        # Injectable so a consumer can write audit rows into its own table
        # rather than basis's - see `adapters.aicore.AiCoreAuditSink`, which
        # targets ai-core's append-only `audit_log`.
        self._audit_sink = audit_sink
        # Injectable grant lookup, for a consumer whose tool permissions live
        # somewhere other than `agent_persona_tools`.
        self._grants_provider = grants_provider

    # ── catalog ────────────────────────────────────────────────────────────

    def sync_from_mcp(self, tools: Iterable[Any], *, tenant_id: str | None = None) -> int:
        """Mirror an MCP server's advertised catalog into ``agent_tools``.

        Returns the number of active tools observed. Behaviour matches
        ``sync_tools_from_mcp``: upsert each, then deactivate MCP-sourced rows
        that were not in this list.
        """
        tenant = tenant_id
        if tenant is None:
            ctx = require_context()
            tenant = ctx.tenant_id

        # Normalize here (policy owns tool-name semantics), persist there.
        rows = []
        for tool in tools:
            info = extract_tool_info(tool)
            if not info.get("name"):
                continue
            rows.append(
                {
                    "name": info["name"],
                    "description": info.get("description") or "",
                    "schema": info.get("schema") or {},
                    "effect": self.policy.classify(info["name"]).value,
                }
            )
        return self._repo.upsert_catalog(tenant, rows)

    def grants_for(
        self, *, tenant_id: str, persona_id: str | None
    ) -> frozenset[str] | None:
        """Tool names granted to a persona within a tenant.

        Returns None when the run has no persona, meaning "no persona-level
        narrowing" - the policy alone decides. Returns a (possibly empty) set
        when a persona is present, so a persona with no grants gets nothing
        rather than everything. That distinction is the fail-closed bit.
        """
        if self._grants_provider is not None:
            return self._grants_provider.grants_for(
                tenant_id=tenant_id, persona_id=persona_id
            )
        return self._repo.grants_for(tenant_id=tenant_id, persona_id=persona_id)

    # ── the seam ───────────────────────────────────────────────────────────

    def tools_for(
        self,
        tools: Iterable[Any],
        *,
        ctx: RunContext | None = None,
        scope: str | None = None,
        use_grants: bool = True,
    ) -> list[Any]:
        """The catalog a run is allowed to see.

        Call this once, at the boundary where work is handed to an agent, and
        pass the result down. Filtering here rather than at each call site is
        what made the original's guarantee hold across nested sub-agents.
        """
        run = ctx if ctx is not None else require_context()
        effective_scope = scope if scope is not None else run.task_key

        # Materialize first: `tools` may be a generator (MCPClient.list_tools_sync
        # returns a fresh iterator), and it is read twice below.
        candidates = list(tools)

        granted = None
        if use_grants:
            try:
                granted = self.grants_for(
                    tenant_id=run.tenant_id, persona_id=run.persona_id
                )
            except Exception as exc:
                # Fail closed: if grants cannot be read we do not fall back to
                # "policy only", because that would silently widen access.
                log.error("tool grant lookup failed for run %s: %s", run.run_id, exc)
                raise

        kept = self.policy.filter(candidates, scope=effective_scope, granted=granted)
        log.debug(
            "run %s scope=%s: %d of %d tools permitted",
            run.run_id,
            effective_scope,
            len(kept),
            len(candidates),
        )
        return kept

    def check(
        self,
        tool_name: str,
        *,
        ctx: RunContext | None = None,
        scope: str | None = None,
        use_grants: bool = True,
    ) -> None:
        """Assert a tool is permitted, or raise ToolDenied.

        A second line of defence behind ``tools_for``. If a tool is reached by
        some path that skipped the filter, this makes it loud. The original had
        no equivalent - a policy bypass would simply have worked.

        ``use_grants`` must match what was passed to ``tools_for`` for this
        run. They are separate arguments because they are separate calls, but
        disagreeing is always a bug: filtering with ``use_grants=False`` and
        then checking with grants on denies tools the agent was handed, which
        surfaces as a tool that exists but always fails.
        """
        run = ctx if ctx is not None else require_context()
        effective_scope = scope if scope is not None else run.task_key
        granted = (
            self.grants_for(tenant_id=run.tenant_id, persona_id=run.persona_id)
            if use_grants
            else None
        )
        reason = self.policy.denied_reason(
            tool_name, scope=effective_scope, granted=granted
        )
        if reason:
            self.record_invocation(
                tool_name, ctx=run, allowed=False, reason=reason, duration_ms=0
            )
            raise ToolDenied(tool_name, reason)

    def record_invocation(
        self,
        tool_name: str,
        *,
        ctx: RunContext | None = None,
        allowed: bool = True,
        reason: str | None = None,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        """Write the audit row for one tool invocation.

        The deck's read of MCP Reusability 6/10 - "the adapters exist, the
        governance doesn't ... no audit record on invocation" - is accurate:
        nothing in tool_access.py recorded that a tool ran. Without this row
        there is no answer to "what did the agent do on this client's data".
        """
        if not self._audit:
            return
        run = ctx if ctx is not None else require_context()

        if self._audit_sink is not None:
            self._audit_sink.record_invocation(
                tool_name,
                ctx=run,
                allowed=allowed,
                reason=reason,
                duration_ms=duration_ms,
                error=error,
                effect=self.policy.classify(tool_name).value,
            )
            return

        try:
            self._repo.record_invocation(
                {
                    "tenant_id": run.tenant_id,
                    "project_id": run.project_id,
                    "run_id": run.run_id,
                    "job_id": run.job_id,
                    "persona_id": run.persona_id,
                    "principal_id": run.user_id,
                    "principal_kind": run.principal.kind,
                    "tool_name": tool_name,
                    "effect": self.policy.classify(tool_name).value,
                    "allowed": allowed,
                    "denied_reason": reason,
                    "duration_ms": duration_ms,
                    "error": error,
                }
            )
        except Exception as exc:
            # An audit write must not break the run, but a silent drop is how
            # audit trails become untrustworthy - so it is logged at error.
            log.error("tool audit write failed for %s: %s", tool_name, exc)

    def invoke(
        self,
        tool_name: str,
        fn: Any,
        *args: Any,
        ctx: RunContext | None = None,
        scope: str | None = None,
        use_grants: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Check, run, and audit a tool call in one place.

        Pass the same ``use_grants`` value used for ``tools_for`` on this run.
        """
        run = ctx if ctx is not None else require_context()
        self.check(tool_name, ctx=run, scope=scope, use_grants=use_grants)
        started = time.monotonic()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self.record_invocation(
                tool_name,
                ctx=run,
                allowed=True,
                duration_ms=int((time.monotonic() - started) * 1000),
                error=str(exc)[:500],
            )
            raise
        self.record_invocation(
            tool_name,
            ctx=run,
            allowed=True,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        return result
