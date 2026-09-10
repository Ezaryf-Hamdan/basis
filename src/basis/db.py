"""Pooled Postgres access.

Consolidates the four hand-rolled connection helpers in AILedSDLC
(``memory_service._connection``, ``tool_access._connection``,
``model_config._query_one``, and the bare ``psycopg2.connect`` in memory.py /
personas.py / long_term_memory.py / tool_call_capture.py).

Two changes from the lifted code:

1. **psycopg 3 instead of psycopg2.** ai-core/parser runs Python 3.14, where
   ``psycopg2-binary`` has no wheels. psycopg 3 keeps ``%s`` placeholders, so
   the lifted SQL is unchanged; ``RealDictCursor`` becomes ``row_factory=dict_row``.

2. **A real pool instead of connect-per-call.** The lifted code opened and
   closed a socket for every query. ``model_config`` did up to four for one
   task_key lookup. Both #200 and #367 in that repo are the same bug - psycopg2's
   ``with connection`` commits but does not close, so a missed explicit close
   leaks a backend. A pool removes the whole failure class, and the
   ``transaction()`` helper below preserves the commit-on-success /
   rollback-on-error / always-release discipline those helpers documented.

The pool is keyed by DSN so a process talking to two databases gets two pools,
and is created lazily so importing this module never opens a socket.
"""
from __future__ import annotations

import atexit
import contextlib
import contextvars
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from .errors import ConfigurationError
from .settings import settings

__all__ = [
    "close_pools",
    "cursor",
    "execute",
    "query_all",
    "query_one",
    "session_vars",
    "transaction",
]

#: Query parameters. A sequence binds ``%s`` placeholders positionally; a
#: mapping binds ``%(name)s`` ones. basis uses both - the resolution cascade
#: and the lineage CTE are far clearer with named parameters.
Params = Sequence[Any] | Mapping[str, Any] | None

_pools: dict[str, Any] = {}
_pools_lock = threading.Lock()


def _get_pool(dsn: str) -> Any:
    """Lazily create (and memoize) a ConnectionPool for this DSN."""
    pool = _pools.get(dsn)
    if pool is not None:
        return pool

    try:
        from psycopg_pool import ConnectionPool
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ConfigurationError(
            "psycopg[pool] is required for database access; install basis with "
            "its base dependencies"
        ) from exc

    cfg = settings()
    with _pools_lock:
        pool = _pools.get(dsn)
        if pool is None:
            pool = ConnectionPool(
                conninfo=dsn,
                min_size=cfg.pool_min_size,
                max_size=cfg.pool_max_size,
                timeout=cfg.pool_timeout,
                # Reconnect after a DB restart rather than returning broken
                # connections. Without this, a pool open before a restart silently
                # returns connections that fail on first use with no retry.
                reconnect_timeout=cfg.pool_reconnect_timeout,
                # open=False + explicit open() keeps import side-effect-free and
                # surfaces a bad DSN at first use rather than at construction.
                open=False,
                name="basis",
            )
            pool.open()
            _pools[dsn] = pool
    return pool


def _resolve_dsn(dsn: str | None) -> str:
    return dsn or settings().require_database_url()


_session_vars: contextvars.ContextVar[Mapping[str, str] | None] = contextvars.ContextVar(
    "basis_db_session_vars", default=None
)


@contextmanager
def session_vars(**values: str) -> Iterator[None]:
    """Bind Postgres session variables for every transaction in this block.

    Applied with ``set_config(key, value, is_local => true)``, so they are
    scoped to the transaction and cannot leak to the next borrower of a pooled
    connection. That ``is_local`` flag is the whole reason this exists as a
    dedicated mechanism rather than a plain ``SET``.

    This is what makes basis usable against a consumer that enforces isolation
    with row-level security rather than application predicates. A schema using
    ``USING (workspace_id = current_setting('app.workspace_id'))`` returns
    **zero rows** when the setting is unset - the correct fail-closed
    behaviour, and an invisible failure for any client that does not set it.

        with db.session_vars(**{"app.workspace_id": ws_id}):
            ...                       # every basis query now sees this workspace

    Nesting merges, so an inner block can add a variable without dropping the
    outer one.
    """
    current = _session_vars.get() or {}
    merged = {**current, **values}
    token = _session_vars.set(merged)
    try:
        yield
    finally:
        _session_vars.reset(token)


@contextmanager
def transaction(dsn: str | None = None) -> Iterator[Any]:
    """A pooled connection wrapped in a transaction.

    Commits on clean exit, rolls back on exception, always returns the
    connection to the pool. This is the behaviour ``memory_service._connection``
    and ``tool_access._connection`` implemented by hand, minus the close bug
    they were both written to work around.

    Any variables bound by ``session_vars`` are applied first, inside the same
    transaction, so RLS policies see them.
    """
    pool = _get_pool(_resolve_dsn(dsn))
    with pool.connection() as conn:
        # psycopg 3 connections are transactional by default and the
        # ``pool.connection()`` context manager commits on success and rolls
        # back on exception, so no manual commit/rollback is needed here.
        bound = _session_vars.get()
        if bound:
            with conn.cursor() as cur:
                for key, value in bound.items():
                    # Parameterized: a session-variable name is an identifier,
                    # but set_config takes it as a value, so no interpolation.
                    cur.execute(
                        "SELECT set_config(%s, %s, true)", (key, str(value))
                    )
        yield conn


@contextmanager
def cursor(dsn: str | None = None, *, dict_rows: bool = True) -> Iterator[Any]:
    """A cursor inside a transaction.

    ``dict_rows=True`` reproduces psycopg2's ``RealDictCursor``, which the
    lifted code used for every read.
    """
    from psycopg.rows import dict_row

    with transaction(dsn) as conn, (
        conn.cursor(row_factory=dict_row) if dict_rows else conn.cursor()
    ) as cur:
        yield cur


def _normalize(row: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce UUID columns to ``str``.

    psycopg 3 returns ``uuid.UUID`` objects for uuid columns, but basis's
    domain works in strings throughout - ``RunContext.tenant_id``,
    ``ArtifactVersion.id`` and every port signature are ``str``. Mixing the two
    is a silent correctness bug rather than a cosmetic one: a caller comparing
    ``impact_of()`` ids against an ``ArtifactVersion.id`` gets an empty
    intersection, with no error and no obvious cause.

    Normalizing here rather than in each repository means every query is
    consistent, including the adapters'.
    """
    out = {}
    for key, value in row.items():
        out[key] = str(value) if isinstance(value, uuid.UUID) else value
    return out


def query_one(
    sql: str, params: Params = None, *, dsn: str | None = None
) -> dict[str, Any] | None:
    """Single row as a dict, or None.

    Direct replacement for ``model_config._query_one``.
    """
    with cursor(dsn) as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return _normalize(row) if row else None


def query_all(
    sql: str, params: Params = None, *, dsn: str | None = None
) -> list[dict[str, Any]]:
    """All rows as dicts."""
    with cursor(dsn) as cur:
        cur.execute(sql, params)
        return [_normalize(r) for r in cur.fetchall()]


def execute(
    sql: str, params: Params = None, *, dsn: str | None = None
) -> int:
    """Run a write; return the affected row count."""
    with cursor(dsn, dict_rows=False) as cur:
        cur.execute(sql, params)
        return cur.rowcount


def close_pools() -> None:
    """Close every pool. Registered at exit; also useful between tests."""
    with _pools_lock:
        for pool in _pools.values():
            # Best effort: a pool that fails to close during interpreter
            # shutdown must not mask the real exit path.
            with contextlib.suppress(Exception):
                pool.close()
        _pools.clear()


atexit.register(close_pools)
