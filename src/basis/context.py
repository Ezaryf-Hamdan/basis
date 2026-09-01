"""RunContext and Principal - the identity and tenancy spine.

This is the one module in basis that is genuinely NEW rather than lifted, and
everything else depends on it.

Why it did not exist before: AILedSDLC keeps authentication in Node
(Express/Drizzle), so the Python tier only ever received a bearer token. It
tracked that token in a single module-level contextvar
(``config._delegated_token_var``) and passed user_id / project_id / persona_id
around as loose string arguments. There was no tenant concept anywhere, which
is why per-tenant cost attribution and data isolation could not be enforced -
there was nothing to enforce them against.

RunContext replaces the loose arguments and the bare token contextvar with one
immutable object bound to the async context, so:

  * every DB predicate can be tenant-scoped from a single source,
  * every span can carry tenant_id / project_id / run_id,
  * a delegated credential travels with the run instead of in a global.

INVARIANT (inherited from AILedSDLC's INV-1, and widened): a RunContext is
built by trusted glue - the code that accepted the request and verified the
token - and NEVER from tool arguments or model output. Nothing in this module
parses a JWT; verification stays in the caller (Node today) and the verified
claims are handed in.
"""
from __future__ import annotations

import contextvars
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

from .errors import TenantIsolationError

__all__ = [
    "SERVICE_PRINCIPAL_KIND",
    "USER_PRINCIPAL_KIND",
    "Principal",
    "RunContext",
    "bind",
    "current_context",
    "require_context",
]

USER_PRINCIPAL_KIND = "user"
SERVICE_PRINCIPAL_KIND = "service"


@dataclass(frozen=True)
class Principal:
    """Who a run acts as.

    ``kind`` distinguishes a delegated human identity from a service identity.
    The distinction matters because AILedSDLC learned the hard way that a
    service token running a user's job is god-mode: comments in
    tool_access.py:82-90 describe writer tools that "under the job service
    token run god-mode". Making the kind explicit lets policy refuse
    user-attributable writes under a service principal.

    ``token`` is opaque here. basis never inspects or verifies it.
    """

    subject_id: str
    tenant_id: str
    kind: str = USER_PRINCIPAL_KIND
    email: str | None = None
    roles: frozenset[str] = field(default_factory=frozenset)
    token: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.subject_id:
            raise TenantIsolationError("Principal requires a subject_id")
        if not self.tenant_id:
            raise TenantIsolationError("Principal requires a tenant_id")
        if self.kind not in (USER_PRINCIPAL_KIND, SERVICE_PRINCIPAL_KIND):
            raise ValueError("unknown principal kind: %r" % (self.kind,))

    @property
    def is_service(self) -> bool:
        return self.kind == SERVICE_PRINCIPAL_KIND

    def has_role(self, role: str) -> bool:
        return role in self.roles

    @classmethod
    def service(
        cls, subject_id: str, tenant_id: str, token: str | None = None
    ) -> Principal:
        return cls(
            subject_id=subject_id,
            tenant_id=tenant_id,
            kind=SERVICE_PRINCIPAL_KIND,
            token=token,
        )

    @classmethod
    def from_claims(
        cls, claims: Mapping[str, Any], *, token: str | None = None
    ) -> Principal:
        """Build from already-verified JWT claims.

        Deliberately does not verify anything - the caller has done that. Claim
        names follow the shape AILedSDLC's Node tier already issues (``sub``,
        ``tenant_id``, ``email``, ``roles``).
        """
        roles_raw = claims.get("roles") or ()
        if isinstance(roles_raw, str):
            roles: Sequence[str] = [roles_raw]
        else:
            roles = [str(r) for r in roles_raw]
        return cls(
            subject_id=str(claims.get("sub") or ""),
            tenant_id=str(claims.get("tenant_id") or ""),
            kind=str(claims.get("kind") or USER_PRINCIPAL_KIND),
            email=(str(claims["email"]) if claims.get("email") else None),
            roles=frozenset(roles),
            token=token,
        )


@dataclass(frozen=True)
class RunContext:
    """Everything a single unit of agent work needs to know about itself.

    ``run_id`` is the correlation key: it is what ties spans, activity rows,
    tool-call audit records, and memory writes for one run together. It is
    generated here if not supplied, so a caller cannot forget it.

    ``project_id`` is optional because some runs are tenant-scoped but not
    project-scoped (e.g. a tenant-wide catalog sync). Memory operations require
    it and will say so.
    """

    principal: Principal
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str | None = None
    session_id: str | None = None
    job_id: str | None = None
    persona_id: str | None = None
    task_key: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    @property
    def tenant_id(self) -> str:
        return self.principal.tenant_id

    @property
    def user_id(self) -> str:
        return self.principal.subject_id

    @property
    def token(self) -> str | None:
        """The credential this run should present downstream.

        Replaces ``config.get_delegated_token()``. Same fail-closed intent: if
        the run has no credential, callers get None and must decide, rather
        than silently inheriting a process-wide service token.
        """
        return self.principal.token

    def require_project(self) -> str:
        if not self.project_id:
            raise TenantIsolationError(
                "run %s is not project-scoped; this operation requires a project_id"
                % (self.run_id,)
            )
        return self.project_id

    def child(self, **overrides: Any) -> RunContext:
        """Derive a sub-run that inherits tenancy but gets its own run_id.

        Used for sub-agent / delegated-tool work. The tenant cannot be changed
        through this path - that is the point.
        """
        new_principal = overrides.get("principal")
        if (
            isinstance(new_principal, Principal)
            and new_principal.tenant_id != self.tenant_id
        ):
            raise TenantIsolationError(
                "child run may not change tenant (%s -> %s)"
                % (self.tenant_id, new_principal.tenant_id)
            )
        overrides.setdefault("run_id", str(uuid.uuid4()))
        return replace(self, **overrides)

    def span_attributes(self) -> dict[str, str]:
        """The semantic-convention attributes every span in a run must carry.

        Slide 13's gap: ai-core has CloudWatch but no end-to-end tracing, and
        even where OTel exists it carries no tenant. Cost-per-client is
        impossible without these keys on every span, so they are produced from
        one place rather than set ad hoc at each call site.
        """
        attrs = {
            "basis.tenant_id": self.tenant_id,
            "basis.run_id": self.run_id,
            "basis.principal_kind": self.principal.kind,
        }
        for key, value in (
            ("basis.project_id", self.project_id),
            ("basis.session_id", self.session_id),
            ("basis.job_id", self.job_id),
            ("basis.persona_id", self.persona_id),
            ("basis.task_key", self.task_key),
            ("enduser.id", self.principal.subject_id),
        ):
            if value:
                attrs[key] = str(value)
        attrs.update({"basis." + k: str(v) for k, v in self.attributes.items()})
        return attrs


_current: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "basis_run_context", default=None
)


def current_context() -> RunContext | None:
    """The RunContext bound to this async context, or None."""
    return _current.get()


def require_context() -> RunContext:
    """The bound RunContext, or raise.

    Fail-closed: code that needs tenancy must not proceed without it.
    """
    ctx = _current.get()
    if ctx is None:
        raise TenantIsolationError(
            "no RunContext bound; wrap this call in `with basis.context.bind(ctx):`"
        )
    return ctx


@contextmanager
def bind(ctx: RunContext) -> Iterator[RunContext]:
    """Bind a RunContext for the duration of the block.

    Generalizes config.set_delegated_token / reset_delegated_token, which had
    the same reset-token discipline but carried only a bearer string.
    """
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)
