"""MemoryScope - the addressing key for every memory read and write.

Lifted from ``memory_service.MemoryScope``, which was already close to right:
a frozen dataclass carrying (user_id, project_id, persona_id, job_id) with a
``validate()`` that refuses a partial scope. That is three quarters of a
RunContext, and it is the clearest evidence in the source repo that the shape
of the tenancy problem was understood - it just stopped one level short.

The one addition is ``tenant_id``, and it is the whole point. Without it, two
clients served by the same persona on projects with colliding ids share a
memory namespace, and a recall for one can surface the other's content. The
deck scores Data Isolation 6/10; this is where the missing 4 lives.

``from_context`` is the intended constructor - deriving the scope from the
bound RunContext means a caller cannot assemble a scope with a tenant that does
not match the run.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..context import RunContext, require_context
from ..errors import TenantIsolationError

__all__ = ["MemoryScope"]


@dataclass(frozen=True)
class MemoryScope:
    """Identifies the memory namespace a run may read and write."""

    tenant_id: str
    user_id: str
    project_id: str
    persona_id: str
    job_id: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Refuse a partial scope.

        Same fail-closed contract as the original's ``validate()``, extended to
        cover the tenant. Raised at construction rather than at each method, so
        an incomplete scope cannot be passed around.
        """
        missing = [
            name
            for name, value in (
                ("tenant_id", self.tenant_id),
                ("user_id", self.user_id),
                ("project_id", self.project_id),
                ("persona_id", self.persona_id),
            )
            if not value
        ]
        if missing:
            raise TenantIsolationError(
                "MemoryScope requires %s" % ", ".join(missing)
            )

    def require_job(self) -> str:
        """Short-term memory is job-scoped; long-term is not.

        The original enforced this with an inline check in both
        ``write_short_term`` and ``read_short_term``.
        """
        if not self.job_id:
            raise TenantIsolationError(
                "this operation is job-scoped and the scope has no job_id"
            )
        return self.job_id

    @classmethod
    def from_context(cls, ctx: RunContext | None = None) -> MemoryScope:
        """Derive a scope from the bound run.

        Requires a project and a persona, and says which is missing - the
        original would have accepted a None persona_id and only failed later at
        the ``validate()`` call inside each write.
        """
        run = ctx if ctx is not None else require_context()
        if not run.persona_id:
            raise TenantIsolationError(
                "run %s has no persona_id; memory is persona-scoped" % run.run_id
            )
        return cls(
            tenant_id=run.tenant_id,
            user_id=run.user_id,
            project_id=run.require_project(),
            persona_id=run.persona_id,
            job_id=run.job_id,
            run_id=run.run_id,
        )
