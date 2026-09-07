"""Persona store.

Lifted from ``personas.py``. The DB-driven persona shape (id, name, role,
workstream, system_prompt, model_id, model_config) is a good design and is
kept; the caching around it is replaced.

The original::

    _cached_personas: list[dict] | None = None

    def load_personas(force_refresh: bool = False) -> list[dict]:
        global _cached_personas
        if _cached_personas is not None and not force_refresh:
            return _cached_personas

Three problems. The cache is process-global with no TTL, so a persona edit in
the admin UI never takes effect until a restart and ``force_refresh`` has to be
threaded to every call site. It is not tenant-keyed, so the first tenant to
load personas populates the cache for every subsequent tenant - a cross-client
data leak through a cache, and the most direct one in the repo. And it returns
the mutable list it cached, so any caller mutating a persona dict corrupts it
for everyone.

Here the cache is TTL'd, keyed by tenant, and returns copies.

``get_solution_architect``'s hardcoded fallback persona is also gone. It
returned a dict with ``"name": "Assistant"`` and a generic prompt when no SA
existed, which silently substitutes a useless persona for a missing
configuration - a run that produces plausible output from the wrong persona is
worse than one that fails.
"""
from __future__ import annotations

import copy
import time
from typing import Any

from ..context import RunContext, require_context
from ..errors import ConfigurationError
from ..storage import PersonaRepository

__all__ = ["PersonaStore"]

class PersonaStore:
    """Tenant-scoped persona lookups with a short TTL cache."""

    def __init__(
        self,
        repo: PersonaRepository | None = None,
        *,
        dsn: str | None = None,
        ttl_seconds: float = 60.0,
    ):
        if repo is None:
            from ..storage.postgres import PgPersonaRepository

            repo = PgPersonaRepository(dsn=dsn)
        self._repo = repo
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    def invalidate(self, tenant_id: str | None = None) -> None:
        """Drop cached personas for a tenant, or for all tenants."""
        if tenant_id is None:
            self._cache.clear()
        else:
            self._cache.pop(tenant_id, None)

    def all(self, *, tenant_id: str) -> list[dict[str, Any]]:
        """Active personas for a tenant. Returns copies, not the cached list."""
        hit = self._cache.get(tenant_id)
        if hit and (time.monotonic() - hit[0]) < self._ttl:
            return copy.deepcopy(hit[1])

        rows = self._repo.all(tenant_id=tenant_id)
        self._cache[tenant_id] = (time.monotonic(), rows)
        return copy.deepcopy(rows)

    def by_id(self, persona_id: str, *, tenant_id: str) -> dict[str, Any]:
        """One persona by id, scoped to the tenant.

        The tenant predicate is what stops a persona_id from another tenant
        resolving - the original had no such lookup at all, so callers passed
        persona ids around unvalidated.
        """
        row = self._repo.by_id(tenant_id=tenant_id, persona_id=persona_id)
        if not row:
            raise ConfigurationError(
                "persona %s not found for tenant %s" % (persona_id, tenant_id)
            )
        return row

    def by_role(
        self, role: str, *, tenant_id: str, workstream: str | None = None
    ) -> dict[str, Any] | None:
        """Best match for a role, preferring an exact workstream match.

        Generalizes ``get_persona_for_workstream``, which hardcoded
        ``role == "functional_consultant"``.
        """
        personas = self.all(tenant_id=tenant_id)
        if workstream:
            for persona in personas:
                if persona["role"] == role and persona["workstream"] == workstream:
                    return persona
        for persona in personas:
            if persona["role"] == role and not persona["workstream"]:
                return persona
        for persona in personas:
            if persona["role"] == role:
                return persona
        return None

    def require_role(
        self, role: str, *, tenant_id: str, workstream: str | None = None
    ) -> dict[str, Any]:
        """``by_role``, but raise instead of returning a stand-in persona."""
        persona = self.by_role(role, tenant_id=tenant_id, workstream=workstream)
        if persona is None:
            raise ConfigurationError(
                "no active persona with role %r%s for tenant %s"
                % (
                    role,
                    " (workstream %s)" % workstream if workstream else "",
                    tenant_id,
                )
            )
        return persona

    def for_context(
        self, ctx: RunContext | None = None
    ) -> dict[str, Any]:
        """The persona named by the bound run."""
        run = ctx if ctx is not None else require_context()
        if not run.persona_id:
            raise ConfigurationError("run %s has no persona_id" % run.run_id)
        return self.by_id(run.persona_id, tenant_id=run.tenant_id)
