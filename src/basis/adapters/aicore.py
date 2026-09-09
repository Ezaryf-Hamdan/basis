"""Adapter: basis onto ai-core's existing schema.

basis speaks ``tenant_id`` and owns its own tables. ai-core speaks
``workspace_id`` and already has a schema with a stronger isolation mechanism
than basis's own. This module is the translation layer, so neither has to
change.

**What ai-core already has, and basis therefore must not duplicate:**

| basis concept        | ai-core table                              |
|----------------------|--------------------------------------------|
| tenant               | ``workspaces`` (`workspace_id`)            |
| task model config    | ``model_roles`` (workspace_id, role)       |
| persona              ``pipeline_agents`` + ``pipeline_stage_assignments`` |
| conversation memory  | ``conversations`` / ``messages``           |
| audit                | ``audit_log`` (append-only)                |
| provider keys        | ``provider_credentials`` (per workspace)   |
| chunks / retrieval   | ``chunks`` (pgvector + generated tsvector) |
| workflow runs        | ``pipeline_runs`` (`stage_results` jsonb)  |

**The critical part is `bind_workspace`.** ai-core enforces isolation with
Postgres row-level security: ``FORCE ROW LEVEL SECURITY``, a non-owner
``rag_app`` role, and policies of the form
``USING (workspace_id = current_setting('app.workspace_id'))``. An unset
setting yields NULL, which matches nothing - correct fail-closed design, and it
means any client that does not set the GUC reads **zero rows with no error**.

Without `bind_workspace`, every basis query against ai-core's database silently
returns nothing. It is not optional.
"""
from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from .. import db
from ..context import Principal, RunContext, bind, require_context
from ..models.catalog import accepts_sampling
from ..models.resolver import TaskModel

__all__ = [
    "PROVIDER_MODEL_PREFIXES",
    "WORKSPACE_GUC",
    "AiCoreAuditSink",
    "AiCorePersonaStore",
    "AiCoreResolver",
    "bind_workspace",
    "context_for_workspace",
]

log = logging.getLogger(__name__)

#: The session variable ai-core's RLS policies read.
#: See ai-core `web/src/lib/db/index.ts` and `web/drizzle/0001_rls.sql`.
WORKSPACE_GUC = "app.workspace_id"

#: ai-core stores provider and model separately (`model_roles.provider` +
#: `.model`); basis resolves a single model id. For Bedrock-routed Anthropic
#: models the id needs the cross-region inference-profile prefix that
#: `strands.models.BedrockModel` expects.
PROVIDER_MODEL_PREFIXES: Mapping[str, str] = {
    "bedrock": "us.anthropic.",
    "anthropic": "",
    "openai": "",
    "google": "",
    "ollama": "",
}


@contextmanager
def bind_workspace(ctx: RunContext) -> Iterator[RunContext]:
    """Bind a RunContext *and* the workspace GUC that ai-core's RLS requires.

    Use this instead of `basis.bind` for every ai-core call path::

        with bind_workspace(ctx):
            result = await gateway.invoke("chat", user_prompt=text)

    The GUC is applied per transaction with ``is_local => true``, matching
    ai-core's own `set_config(..., true)` usage, so it cannot leak to the next
    borrower of a pooled connection.
    """
    with bind(ctx), db.session_vars(**{WORKSPACE_GUC: ctx.tenant_id}):
        yield ctx


def context_for_workspace(
    workspace_id: str,
    user_id: str,
    *,
    token: str | None = None,
    roles: Sequence[str] = (),
    project_id: str | None = None,
    task_key: str | None = None,
    **kwargs: Any,
) -> RunContext:
    """Build a RunContext from ai-core's identity vocabulary.

    ``workspace_id`` becomes basis's ``tenant_id``. That is the entire mapping,
    and keeping it in one function means the equivalence is stated once rather
    than assumed in fifty call sites.
    """
    return RunContext(
        principal=Principal(
            subject_id=user_id,
            tenant_id=workspace_id,
            roles=frozenset(roles),
            token=token,
        ),
        project_id=project_id,
        task_key=task_key,
        **kwargs,
    )


class AiCoreResolver:
    """Resolves a basis task key against ai-core's `model_roles`.

    ai-core's model config is (workspace_id, role) -> (provider, model,
    fallbacks jsonb). basis's task key maps onto ``role``.

    This adapter also **fixes a live bug in ai-core**. `pipeline_agents`
    carries ``temperature real NOT NULL DEFAULT 0.4`` and
    `web/src/lib/agents/run-model.ts` passes it to `generateText`
    unconditionally. Sampling parameters were removed on Claude Sonnet 5 and
    the Opus 4.7/4.8/5 family - they return a 400. Any workspace that assigns
    one of those models to a pipeline stage fails, and ai-core's fallback
    guards *empty text* (`if (res.text)`) rather than exceptions, so the
    rejection propagates and fails the stage rather than falling back.

    Requests routed through basis drop the parameter for models that reject it
    (see `models.catalog.accepts_sampling` and `providers.bedrock`), so this
    path is safe even where the TypeScript path is not.
    """

    def __init__(self, *, dsn: str | None = None, default_provider: str = "bedrock"):
        self._dsn = dsn
        self._default_provider = default_provider

    @staticmethod
    def qualify(provider: str, model: str) -> str:
        """Turn ai-core's (provider, model) pair into a basis model id."""
        prefix = PROVIDER_MODEL_PREFIXES.get(provider, "")
        if prefix and not model.startswith(prefix):
            return prefix + model
        return model

    def resolve(self, task_key: str, *, tenant_id: str) -> TaskModel:
        """Read `model_roles`, falling back to the 'chat' role then the default.

        The 'chat' fallback mirrors ai-core's own behaviour in `run-model.ts`,
        which falls back to the workspace's chat role when a stage's assigned
        model produces nothing.
        """
        row = db.query_one(
            """
            SELECT role, provider, model, fallbacks
            FROM model_roles
            WHERE workspace_id = %s AND role = %s
            """,
            (tenant_id, task_key),
            dsn=self._dsn,
        )

        if row is None and task_key != "chat":
            row = db.query_one(
                """
                SELECT role, provider, model, fallbacks
                FROM model_roles
                WHERE workspace_id = %s AND role = 'chat'
                """,
                (tenant_id,),
                dsn=self._dsn,
            )
            if row is not None:
                log.debug(
                    "task_key %r has no model_roles row; using workspace chat role",
                    task_key,
                )

        if row is None:
            from ..settings import settings

            return TaskModel(task_key=task_key, model_id=settings().default_model_id)

        provider = row["provider"] or self._default_provider
        model_id = self.qualify(provider, row["model"])

        fallbacks = row.get("fallbacks") or []
        if isinstance(fallbacks, str):
            fallbacks = json.loads(fallbacks)
        fallback_ids = tuple(
            self.qualify(
                (f.get("provider") if isinstance(f, dict) else provider) or provider,
                str(f.get("model")) if isinstance(f, dict) else str(f),
            )
            for f in fallbacks
            # An entry with no model is unusable; skipping it beats sending an
            # empty model id to a provider and getting an opaque 400 back.
            if not isinstance(f, dict) or f.get("model")
        )

        return TaskModel(
            task_key=task_key,
            model_id=model_id,
            model_config={},
            fallback_model_ids=fallback_ids,
            source="aicore.model_roles",
        )


class AiCorePersonaStore:
    """Reads ai-core's `pipeline_agents` as basis personas.

    Maps ai-core's columns onto the persona shape basis expects
    (``role``/``system_prompt``/``model_id``/``model_config``), so
    `agents.AgentContext.persona` and `Agent.system_prompt` work unchanged.

    ``temperature`` is carried into ``model_config`` *only* when the resolved
    model accepts it - the same fix described on `AiCoreResolver`. Dropping it
    here rather than at the provider means a caller inspecting the persona sees
    the effective configuration, not one that would 400.
    """

    def __init__(self, *, dsn: str | None = None):
        self._dsn = dsn

    def _to_persona(self, row: Mapping[str, Any]) -> dict[str, Any]:
        model_id = AiCoreResolver.qualify(row["provider"], row["model"])
        model_config: dict[str, Any] = {
            "max_tokens": row["max_output_tokens"],
        }
        temperature = row.get("temperature")
        if temperature is not None:
            if accepts_sampling(model_id):
                model_config["temperature"] = float(temperature)
            else:
                log.debug(
                    "dropping temperature=%s for %s: sampling parameters are "
                    "rejected by this model",
                    temperature,
                    model_id,
                )
        return {
            "id": str(row["id"]),
            "name": row["name"],
            "role": row["name"],
            "workstream": None,
            "system_prompt": row.get("system_prompt") or None,
            "model_id": model_id,
            "model_config": model_config,
            "is_active": True,
            "description": row.get("description"),
        }

    def all(self, *, tenant_id: str) -> list[dict[str, Any]]:
        rows = db.query_all(
            """
            SELECT id, name, description, provider, model, system_prompt,
                   temperature, max_output_tokens
            FROM pipeline_agents
            WHERE workspace_id = %s
            ORDER BY created_at DESC
            """,
            (tenant_id,),
            dsn=self._dsn,
        )
        return [self._to_persona(r) for r in rows]

    def by_id(self, persona_id: str, *, tenant_id: str) -> dict[str, Any]:
        row = db.query_one(
            """
            SELECT id, name, description, provider, model, system_prompt,
                   temperature, max_output_tokens
            FROM pipeline_agents
            WHERE id = %s AND workspace_id = %s
            """,
            (persona_id, tenant_id),
            dsn=self._dsn,
        )
        if not row:
            from ..errors import ConfigurationError

            raise ConfigurationError(
                "no pipeline_agent %s in workspace %s" % (persona_id, tenant_id)
            )
        return self._to_persona(row)

    def for_stage(self, stage_key: int, *, tenant_id: str) -> dict[str, Any] | None:
        """The agent assigned to a pipeline stage, via `pipeline_stage_assignments`."""
        row = db.query_one(
            """
            SELECT a.id, a.name, a.description, a.provider, a.model,
                   a.system_prompt, a.temperature, a.max_output_tokens
            FROM pipeline_stage_assignments sa
            JOIN pipeline_agents a ON a.id = sa.agent_id
            WHERE sa.workspace_id = %s AND sa.stage_key = %s
            """,
            (tenant_id, stage_key),
            dsn=self._dsn,
        )
        return self._to_persona(row) if row else None


class AiCoreAuditSink:
    """Writes basis tool invocations into ai-core's `audit_log`.

    ai-core's audit table is deliberately append-only - `0001_rls.sql` revokes
    UPDATE and DELETE from `rag_app`. basis's own retention pruner must
    therefore not be pointed at it; ai-core owns that lifecycle.

    Its columns are generic (`action`, `target`, `detail jsonb`), so the
    basis-specific fields go into `detail` rather than forcing a schema change.
    """

    def __init__(self, *, dsn: str | None = None):
        self._dsn = dsn

    def record_invocation(
        self,
        tool_name: str,
        *,
        ctx: RunContext | None = None,
        allowed: bool = True,
        reason: str | None = None,
        duration_ms: int | None = None,
        error: str | None = None,
        effect: str | None = None,
    ) -> None:
        run = ctx if ctx is not None else require_context()
        try:
            db.execute(
                """
                INSERT INTO audit_log
                  (workspace_id, actor_id, action, target, detail, created_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                """,
                (
                    run.tenant_id,
                    run.user_id if _is_uuid(run.user_id) else None,
                    "tool.invoke" if allowed else "tool.denied",
                    tool_name,
                    json.dumps(
                        {
                            "run_id": run.run_id,
                            "project_id": run.project_id,
                            "job_id": run.job_id,
                            "persona_id": run.persona_id,
                            "principal_kind": run.principal.kind,
                            "principal_id": run.user_id,
                            "effect": effect,
                            "allowed": allowed,
                            "denied_reason": reason,
                            "duration_ms": duration_ms,
                            "error": error,
                        }
                    ),
                ),
                dsn=self._dsn,
            )
        except Exception as exc:
            log.error("audit_log write failed for tool %s: %s", tool_name, exc)


def _is_uuid(value: str | None) -> bool:
    """ai-core's `audit_log.actor_id` is a uuid column; basis subject ids are
    free-form strings. A non-uuid subject (a service account name, say) is
    written into `detail.principal_id` instead of failing the insert."""
    if not value:
        return False
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False
