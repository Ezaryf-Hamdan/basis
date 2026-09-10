"""basis - shared platform substrate for AI-led delivery.

Lifted and refactored out of AILedSDLC/agent, packaged so ai-core (and any
other consumer) can depend on it rather than re-implementing it.

Import the light core freely; the model, embedding and tracing layers pull
their SDKs lazily, so a consumer that only needs tool governance or run
context does not need boto3, Strands or an OTel exporter installed.

    from basis import Principal, RunContext, bind

    ctx = RunContext(
        principal=Principal(subject_id=user_id, tenant_id=tenant_id, token=jwt),
        project_id=project_id,
        task_key="classify",
    )
    with bind(ctx):
        ...
"""
from .context import (
    Principal,
    RunContext,
    bind,
    current_context,
    require_context,
)
from .errors import (
    AuthorizationDenied,
    BasisError,
    ConcurrentModificationError,
    ConfigurationError,
    DelegationExpired,
    ModelUnavailable,
    TenantIsolationError,
    ToolDenied,
)
from .settings import Settings, settings

__version__ = "0.1.0"

__all__ = [
    "AuthorizationDenied",
    "BasisError",
    "ConcurrentModificationError",
    "ConfigurationError",
    "DelegationExpired",
    "ModelUnavailable",
    "Principal",
    "RunContext",
    "Settings",
    "TenantIsolationError",
    "ToolDenied",
    "__version__",
    "bind",
    "current_context",
    "require_context",
    "settings",
]
