"""Exception taxonomy for the basis substrate.

Lifted from AILedSDLC/agent/config.py, where RbacDenied / DelegationExpired /
FsdPreflightError were declared alongside settings and contextvars in one module.
They are errors, not configuration, so they live here.

Every error that a caller is expected to surface to a human carries a
``marker()`` — a stable, machine-readable prefix written into a job's
error_message so a UI can key a specific banner off it. That convention is
load-bearing in AILedSDLC (AiReviewPage keys on ``rbac_denied:`` and
``delegation_expired:``), so it is preserved verbatim.
"""
from __future__ import annotations


class BasisError(Exception):
    """Base for everything raised by this package."""

    def marker(self) -> str:
        return str(self)


class ConfigurationError(BasisError):
    """A required setting is missing or malformed.

    Raised at *use* time, never at import time. The lifted modules raised
    RuntimeError from module scope on a missing DATABASE_URL, which made them
    impossible to import in a test or in a consumer that does not use the DB.
    """


class AuthorizationDenied(BasisError):
    """A downstream authorization check refused the acting principal.

    Generalizes AILedSDLC's RbacDenied: same resource/action/message payload,
    but no longer specific to a Fastify 403 body.
    """

    def __init__(self, resource: str, action: str, message: str):
        self.resource = resource
        self.action = action
        self.message = message
        super().__init__(message)

    def marker(self) -> str:
        return f"rbac_denied: {self.message}"


class DelegationExpired(BasisError):
    """A delegated (per-run, user-attributed) credential expired mid-run and
    could not be refreshed.

    Fail-closed by contract: catching this MUST NOT fall back to a service
    credential for a user-attributable write. A long run that outruns its token
    TTL has to surface as a distinct failure, not as a silent zero-result
    success.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def marker(self) -> str:
        return f"delegation_expired: {self.message}"


class ToolDenied(BasisError):
    """A tool invocation was refused by the tool policy.

    New in basis. AILedSDLC filtered disallowed tools out of the catalog before
    the model ever saw them, which is the right primary defence but leaves no
    signal if a tool is reached by another path. The registry raises this so a
    policy breach is loud rather than absent.
    """

    def __init__(self, tool_name: str, reason: str):
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"tool {tool_name!r} denied: {reason}")

    def marker(self) -> str:
        return f"tool_denied: {self}"


class TenantIsolationError(BasisError):
    """A scope crossed a tenant boundary, or was built without a tenant.

    New in basis, and the reason RunContext exists. Nothing in the lifted code
    had a tenant concept, so nothing could detect this.
    """


class ModelUnavailable(BasisError):
    """Every model in a task's fallback chain failed with a retryable error."""

    def __init__(self, task_key: str, attempts: int, last_error: BaseException | None = None):
        self.task_key = task_key
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"no model succeeded for task_key={task_key!r} after {attempts} attempt(s)"
            + (f": {last_error}" if last_error else "")
        )
