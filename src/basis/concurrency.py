"""Context-preserving executor dispatch.

Lifted verbatim in behaviour from ``config.run_in_executor_ctx``. The original
docstring records exactly why it exists, and it is worth keeping because the
bug it fixes is subtle and expensive:

    "#198: run `fn(*args)` in the default executor INSIDE a copy of the current
    context, so a contextvar set in the job coroutine (the per-job token
    tracker) is visible to the worker thread and resolves to THIS job's tracker.
    Pooled executor threads don't otherwise inherit contextvars, which both
    loses tokens and - via thread reuse - re-introduces cross-job attribution.
    Every run_in_executor dispatch that (transitively) records tokens MUST route
    through this."

That last sentence is the important one, and it is why this belongs in the
substrate rather than in an application module: cross-job token attribution is
a correctness property of billing, and a plain ``loop.run_in_executor`` silently
breaks it. In basis the leaked value is the RunContext, so a plain dispatch
would lose the tenant - the same bug with a larger blast radius.

The sync-in-async pattern this supports is itself inherited: psycopg and boto3
are blocking, so DB and Bedrock calls are offloaded rather than awaited.
"""
from __future__ import annotations

import asyncio
import contextvars
import functools
from collections.abc import Callable
from concurrent.futures import Executor
from typing import Any, TypeVar

__all__ = ["gather_bounded", "maybe_offload", "offload", "run_in_context"]

T = TypeVar("T")


async def run_in_context(
    fn: Callable[..., T],
    *args: Any,
    executor: Executor | None = None,
) -> T:
    """Run a blocking callable in an executor, preserving contextvars.

    Replaces ``config.run_in_executor_ctx(loop, fn, *args)``. The loop argument
    is gone - it is always the running loop, and passing it in only created an
    opportunity to pass the wrong one.
    """
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(executor, lambda: ctx.run(fn, *args))


async def offload(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking callable off the event loop, preserving contextvars.

    ``run_in_context`` with keyword support. Use it around any synchronous I/O
    called from a coroutine.
    """
    if kwargs:
        return await run_in_context(functools.partial(fn, **kwargs), *args)
    return await run_in_context(fn, *args)


async def maybe_offload(
    target: Any, fn: Callable[..., T], *args: Any, **kwargs: Any
) -> T:
    """Offload unless ``target`` declares itself non-blocking.

    A store or repository may set ``blocking = False`` to say "my methods do no
    I/O" - the in-memory backends do. For those, a thread hop per call is pure
    overhead, and a workflow checkpointing twice per step would pay it on every
    step for nothing.

    The default is ``True``, so an implementation that forgets to declare
    itself gets the safe behaviour rather than the fast one.
    """
    if getattr(target, "blocking", True) is False:
        if kwargs:
            return fn(*args, **kwargs)
        return fn(*args)
    return await offload(fn, *args, **kwargs)


async def gather_bounded(
    coros: list[Any], *, limit: int
) -> list[Any]:
    """``asyncio.gather`` with a concurrency ceiling.

    New, but addresses a real pattern in the lifted code: job_executor fans out
    bulk work (bulk_generate_requirements, bulk_classify_fitgap, ...) across an
    unbounded gather, which is how a large project exhausts both the connection
    pool and the Bedrock throttle budget at once.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    sem = asyncio.Semaphore(limit)

    async def _guard(coro: Any) -> Any:
        async with sem:
            return await coro

    return await asyncio.gather(*(_guard(c) for c in coros))
