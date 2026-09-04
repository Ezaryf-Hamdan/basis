"""Tracer setup and span helpers.

Lifted from ``otel_setup.py``. Three things change.

1. **No import-time side effect.** The original ran ``tracer = init_otel()`` at
   module scope, so importing anything that touched otel_setup configured a
   global TracerProvider and installed a propagator. In a shared package that
   is hostile: a consumer with its own OTel wiring would have it silently
   replaced at import. Here ``init_tracing()`` is explicit and idempotent, and
   ``tracer()`` works without it (returning a no-op tracer if nothing is
   configured).

2. **OTel is optional.** If the ``otel`` extra is not installed, every function
   here degrades to a no-op instead of raising on import. A consumer that wants
   tool governance should not need an exporter.

3. **Spans carry tenancy.** ``with_job_span`` took flat kwargs and nothing
   ensured tenant_id / project_id / run_id were among them. ``span()`` merges
   ``RunContext.span_attributes()`` in automatically. This is the concrete fix
   for the FinOps gap: spans without a tenant cannot attribute cost per client,
   and the previous code had no tenant to attribute to.

The AWS X-Ray id generator and propagator are preserved, including the
``OTEL_PYTHON_DISTRO == "aws_distro"`` early return - when ADOT is active via
``opentelemetry-instrument`` it owns the provider and we must not fight it.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..context import RunContext, current_context
from ..settings import settings

__all__ = [
    "Tracer",
    "current_trace_ids",
    "init_tracing",
    "otel_available",
    "span",
    "tracer",
]

_lock = threading.Lock()
_initialized = False
_tracer: Any = None


def otel_available() -> bool:
    try:
        import opentelemetry.trace  # noqa: F401
    except ImportError:
        return False
    return True


class _NoopSpan:
    """Stands in for a span when OTel is not installed.

    Implements only what this package calls, and swallows the rest, so call
    sites do not need ``if tracer:`` guards.
    """

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def add_event(self, name: str, attributes: Any = None) -> None:
        return None

    def record_exception(self, exc: BaseException) -> None:
        return None

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        return None

    def get_span_context(self) -> None:
        return None


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Iterator[_NoopSpan]:
        yield _NoopSpan()


def init_tracing(force: bool = False) -> Any:
    """Configure a TracerProvider and return a tracer. Idempotent.

    Returns a no-op tracer when the ``otel`` extra is absent.
    """
    global _initialized, _tracer

    if _initialized and not force:
        return _tracer

    with _lock:
        if _initialized and not force:
            return _tracer

        if not otel_available():
            _tracer = _NoopTracer()
            _initialized = True
            return _tracer

        from opentelemetry import trace

        cfg = settings()

        # ADOT auto-configures provider, exporters and propagators when active.
        # Setting our own on top of it produces duplicate pipelines.
        if os.environ.get("OTEL_PYTHON_DISTRO") == "aws_distro":
            _tracer = trace.get_tracer(cfg.service_name)
            _initialized = True
            return _tracer

        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create(
            {
                "service.name": cfg.service_name,
                "service.version": cfg.service_version,
            }
        )

        id_generator = None
        try:
            from opentelemetry.sdk.extension.aws.trace import AwsXRayIdGenerator

            id_generator = AwsXRayIdGenerator()
        except ImportError:
            pass

        provider = (
            TracerProvider(resource=resource, id_generator=id_generator)
            if id_generator
            else TracerProvider(resource=resource)
        )

        # Exporter selection. The lifted code fell back to ConsoleSpanExporter
        # when no endpoint was set, which is right for an application and wrong
        # for a library: a consumer that installs the `otel` extra without
        # configuring a collector would get every span - prompts included -
        # printed to stdout, and a "I/O operation on closed file" traceback when
        # the batch processor flushes after stdout is gone.
        #
        # So: OTLP when configured, console only when explicitly asked for, and
        # otherwise a provider with no processor. Spans are still created and
        # still carry their attributes, they are simply dropped - which keeps
        # `span()` and the trace-id correlation working with no output.
        if cfg.otel_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter: Any = OTLPSpanExporter(
                endpoint=cfg.otel_endpoint.rstrip("/") + "/v1/traces"
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
        elif cfg.otel_console:
            from opentelemetry.sdk.trace.export import (
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )

            # Simple, not Batch: console output is for a human watching a dev
            # run, and batching means they see nothing until a flush.
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)

        try:
            from opentelemetry.propagate import set_global_textmap
            from opentelemetry.propagators.aws import AwsXRayPropagator

            set_global_textmap(AwsXRayPropagator())
        except ImportError:
            pass

        _tracer = trace.get_tracer(cfg.service_name)
        _initialized = True
        return _tracer


def tracer() -> Any:
    """The configured tracer, initialising on first use."""
    if _tracer is None:
        return init_tracing()
    return _tracer


@contextmanager
def span(
    name: str,
    ctx: RunContext | None = None,
    **attributes: Any,
) -> Iterator[Any]:
    """Start a span carrying the run's tenancy attributes.

    Replaces ``with_job_span``. The RunContext is taken from the argument, else
    from the bound context, so ordinary call sites pass nothing and still get
    tenant_id / project_id / run_id on the span.
    """
    run = ctx if ctx is not None else current_context()
    merged: dict[str, Any] = {}
    if run is not None:
        merged.update(run.span_attributes())
    merged.update({k: v for k, v in attributes.items() if v is not None})

    with tracer().start_as_current_span(name) as sp:
        for key, value in merged.items():
            sp.set_attribute(key, value)
        try:
            yield sp
        except Exception as exc:
            # The original swallowed nothing but also recorded nothing; an
            # un-flagged error span is invisible in a latency dashboard.
            try:
                sp.record_exception(exc)
                from opentelemetry.trace import Status, StatusCode

                sp.set_status(Status(StatusCode.ERROR, str(exc)))
            except Exception:
                pass
            raise


class Tracer:
    """Object form of ``span``, satisfying ``ports.TracerPort``.

    The module-level ``span`` function is the ergonomic entry point for
    application code. This class exists so another basis function can *hold* a
    tracer without importing this module - which is what removes the
    models -> observability edge. Swap in ``ports.NullTracer`` to disable
    tracing without touching a call site.
    """

    def span(
        self, name: str, ctx: RunContext | None = None, **attributes: Any
    ) -> Any:
        return span(name, ctx, **attributes)


def current_trace_ids() -> tuple[str | None, str | None]:
    """(trace_id, span_id) as lowercase hex, or (None, None).

    Same formatting as ActivityLogger used inline (``format(id, '032x')`` /
    ``'016x'``), pulled out so the activity writer and any other correlation
    sink agree.
    """
    if not otel_available():
        return None, None
    from opentelemetry import trace

    sp = trace.get_current_span()
    sc = sp.get_span_context() if sp else None
    if not sc or not sc.trace_id:
        return None, None
    return format(sc.trace_id, "032x"), format(sc.span_id, "016x")
