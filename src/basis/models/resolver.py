"""Resolve a task key to a model, its parameters, and its fallback chain.

Lifted from ``model_config.get_model`` / ``get_system_prompt`` /
``get_fallback_chain``, which share one resolution rule:

    agent_task_models[task_key]
      -> agent_personas[role, workstream]   (when the key contains ':')
      -> agent_personas[role]
      -> absolute default

That rule is good and is kept exactly. What is fixed is how it was executed.

**Four cascades became one query.** Each of the three functions ran its own
copy of the cascade, and each cascade issued up to two ``_query_one`` calls,
each of which opened and closed its own connection. Resolving one task key
fully - model, prompt, and fallbacks - cost up to six connections and six
round trips for configuration that changes approximately never. The original
says so out loud: "Reads DB on every call (no cache)". Here one query returns
every column, and the result is cached for ``model_config_ttl_seconds``.

**Resolution is tenant-scoped.** The lifted queries filtered on ``task_key``
and ``role`` alone, so one tenant's model configuration applied to every
tenant. That is the concrete shape of the Multi-Client Readiness gap: a client
who tunes a task onto a cheaper model changes it for everyone.

**Validation is corrected.** ``validate_model_config`` capped ``max_tokens`` at
64000; the current Claude 5 family supports 128000 with streaming, which the
provider defaults to. It also accepted ``temperature`` and ``top_p`` for every
model - see ``catalog.accepts_sampling``; those are now rejected for models
that will 400 on them, at configuration time rather than at request time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..settings import settings
from ..storage import TaskModelRepository
from .catalog import DEFAULT_MODEL_ID, accepts_sampling, max_tokens_cap

__all__ = ["TaskModel", "clear_cache", "resolve", "validate_model_config"]


@dataclass(frozen=True)
class TaskModel:
    """Everything needed to invoke a task, resolved in one lookup."""

    task_key: str
    model_id: str
    model_config: dict[str, Any] = field(default_factory=dict)
    system_prompt: str | None = None
    fallback_model_ids: tuple[str, ...] = ()
    fallback_model_config: dict[str, Any] = field(default_factory=dict)
    source: str = "default"

    def chain(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        """(model_id, config) pairs in attempt order: primary then fallbacks."""
        return ((self.model_id, self.model_config), *tuple((mid, self.fallback_model_config) for mid in self.fallback_model_ids))


# The resolution SQL - one query replacing the six the source could issue -
# now lives in `storage.postgres.PgTaskModelRepository.resolve`. The
# cascade *rule* (task_models -> persona[role, workstream] -> persona[role]
# -> default) is policy and stays here; executing it is storage.

_cache: dict[tuple[str, str], tuple[float, TaskModel]] = {}


def clear_cache() -> None:
    """Drop the resolution cache. For tests and for config-change hooks."""
    _cache.clear()


def _split_task_key(task_key: str) -> tuple[str, str | None]:
    """``role:workstream`` -> (role, workstream); otherwise (key, None).

    Same convention as the original, which used the presence of ':' to decide
    whether to match a persona's workstream column.
    """
    if ":" in task_key:
        role, workstream = task_key.split(":", 1)
        return role, workstream
    return task_key, None


def resolve(
    task_key: str,
    *,
    tenant_id: str,
    dsn: str | None = None,
    use_cache: bool = True,
    repo: TaskModelRepository | None = None,
) -> TaskModel:
    """Resolve a task key for a tenant. Never raises on a miss - returns the default."""
    cfg = settings()
    cache_key = (tenant_id, task_key)
    ttl = cfg.model_config_ttl_seconds

    if use_cache and ttl > 0:
        hit = _cache.get(cache_key)
        if hit and (time.monotonic() - hit[0]) < ttl:
            return hit[1]

    if repo is None:
        from ..storage.postgres import PgTaskModelRepository

        repo = PgTaskModelRepository(dsn=dsn)

    role, workstream = _split_task_key(task_key)
    row = repo.resolve(
        tenant_id=tenant_id, task_key=task_key, role=role, workstream=workstream
    )

    if row:
        resolved = TaskModel(
            task_key=task_key,
            model_id=row["model_id"] or cfg.default_model_id,
            model_config=dict(row.get("model_config") or {}),
            system_prompt=row.get("system_prompt"),
            fallback_model_ids=tuple(row.get("fallback_model_ids") or ()),
            fallback_model_config=dict(row.get("fallback_model_config") or {}),
            source=row["source"],
        )
    else:
        resolved = TaskModel(
            task_key=task_key,
            model_id=cfg.default_model_id or DEFAULT_MODEL_ID,
            source="default",
        )

    if use_cache and ttl > 0:
        _cache[cache_key] = (time.monotonic(), resolved)
    return resolved


def validate_model_config(cfg: dict[str, Any], model_id: str | None = None) -> list[str]:
    """Validate a ``model_config`` payload. Returns error strings; empty is valid.

    Lifted from ``model_config.validate_model_config`` with the bounds
    corrected and one check added. When ``model_id`` is supplied, sampling
    parameters are rejected for models that do not accept them - catching at
    configuration time what would otherwise be a 400 at request time.
    """
    errors: list[str] = []

    cap = max_tokens_cap(model_id) if model_id else 128_000
    if cfg.get("max_tokens") is not None:
        v = cfg["max_tokens"]
        if not isinstance(v, int) or isinstance(v, bool) or v < 1 or v > cap:
            errors.append("max_tokens must be int in 1..%d (got %r)" % (cap, v))

    sampling_ok = accepts_sampling(model_id) if model_id else True

    for key, lo, hi in (("temperature", 0.0, 1.0), ("top_p", 0.0, 1.0)):
        if cfg.get(key) is None:
            continue
        v = cfg[key]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < lo or v > hi:
            errors.append("%s must be number in %s..%s (got %r)" % (key, lo, hi, v))
        elif not sampling_ok:
            errors.append(
                "%s is not supported by %s (sampling parameters were removed on "
                "this model and return a 400)" % (key, model_id)
            )

    if cfg.get("top_k") is not None:
        v = cfg["top_k"]
        if not isinstance(v, int) or isinstance(v, bool) or v < 1 or v > 500:
            errors.append("top_k must be int in 1..500 (got %r)" % (v,))
        elif not sampling_ok:
            errors.append(
                "top_k is not supported by %s (sampling parameters were removed "
                "on this model and return a 400)" % (model_id,)
            )

    if cfg.get("stop_sequences") is not None:
        v = cfg["stop_sequences"]
        if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
            errors.append("stop_sequences must be array of strings")

    for flag in ("enable_prompt_caching", "streaming"):
        if cfg.get(flag) is not None and not isinstance(cfg[flag], bool):
            errors.append("%s must be boolean" % flag)

    return errors
