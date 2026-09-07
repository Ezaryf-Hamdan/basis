"""Message routing to a persona.

Lifted from ``router.py`` (keyword routing) and ``llm_router.py`` (LLM
classification with keyword fallback). The *mechanism* in both is reusable; the
*content* is not, and separating them is the whole job here.

What was hardcoded in the source and is now configuration:

  * ``router.py``'s ``SA_KEYWORDS`` / ``SECURITY_KEYWORDS`` /
    ``CROSS_MODULE_SIGNALS`` and the literal workstream tuple
    ``("finance", "scm", "hcm", "eam", "risk", "sustainability")``.
  * ``llm_router.py``'s ``VALID_CATEGORIES`` and the ten-line SAP routing rules
    embedded in ``ROUTER_PROMPT``.

Both are SAP delivery taxonomy. A shared package that ships them would route
ai-core's traffic into workstreams that do not exist for it.

The design worth keeping is the two-tier one: try the LLM classifier, validate
its answer against a closed category set, and fall back to deterministic
keyword routing on an invalid answer or any exception. That is the right
shape - it means a router outage degrades rather than fails, and the closed-set
validation means a hallucinated category cannot propagate. ``llm_router`` got
this right and it is preserved exactly.

One correction: ``llm_router.route_with_llm`` caught bare ``Exception`` and
silently fell back, so a misconfigured router model looked identical to an
ambiguous message and every request quietly paid for a failed LLM call first.
The fallback now logs, and ``RoutingResult`` reports which tier decided.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..context import RunContext, require_context
from ..ports import ModelClient

__all__ = ["KeywordRouter", "LlmRouter", "RoutingResult", "RoutingRules"]

log = logging.getLogger(__name__)

ROUTER_TASK_KEY = "message_router"


@dataclass(frozen=True)
class RoutingRules:
    """The application's routing taxonomy.

    ``categories`` is the closed set a route must land in. ``keywords`` maps a
    category to the terms that select it, and is evaluated in ``priority``
    order - the original checked security before architecture before
    workstream, and that ordering was load-bearing, so it stays explicit rather
    than depending on dict iteration order.
    """

    categories: frozenset[str]
    keywords: Mapping[str, Sequence[str]] = field(default_factory=dict)
    priority: Sequence[str] = ()
    default_category: str = ""
    # Categories that a context value (e.g. the current workstream) may select
    # directly when no keyword matches.
    context_selectable: frozenset[str] = field(default_factory=frozenset)
    context_key: str = "workstream"

    def __post_init__(self) -> None:
        if not self.categories:
            raise ValueError("RoutingRules requires at least one category")
        if self.default_category and self.default_category not in self.categories:
            raise ValueError(
                "default_category %r is not in categories" % self.default_category
            )
        unknown = set(self.keywords) - set(self.categories)
        if unknown:
            raise ValueError("keywords reference unknown categories: %s" % sorted(unknown))
        unknown_priority = set(self.priority) - set(self.categories)
        if unknown_priority:
            raise ValueError(
                "priority references unknown categories: %s" % sorted(unknown_priority)
            )

    def ordered_categories(self) -> list[str]:
        """Priority first, then everything else with keywords."""
        seen = list(self.priority)
        rest = [c for c in self.keywords if c not in seen]
        return seen + rest


@dataclass(frozen=True)
class RoutingResult:
    """Where a message was routed, and by what."""

    category: str
    decided_by: str  # "llm" | "keyword" | "context" | "default"
    raw_model_answer: str | None = None


class KeywordRouter:
    """Deterministic routing. Lifted from ``router.route_message``."""

    def __init__(self, rules: RoutingRules):
        self.rules = rules

    def route(self, message: str, context: Mapping[str, object] | None = None) -> RoutingResult:
        context = context or {}
        lowered = (message or "").lower()

        for category in self.rules.ordered_categories():
            terms = self.rules.keywords.get(category, ())
            if any(term in lowered for term in terms):
                return RoutingResult(category=category, decided_by="keyword")

        # Fall back to a category named by the context, as the original did
        # with ``context.get("workstream")``.
        ctx_value = context.get(self.rules.context_key)
        if isinstance(ctx_value, str) and ctx_value in self.rules.context_selectable:
            return RoutingResult(category=ctx_value, decided_by="context")

        if self.rules.default_category:
            return RoutingResult(
                category=self.rules.default_category, decided_by="default"
            )
        raise ValueError("message did not route and no default_category is set")


class LlmRouter:
    """LLM classification with a validated closed set and keyword fallback."""

    def __init__(
        self,
        rules: RoutingRules,
        gateway: ModelClient,
        *,
        task_key: str = ROUTER_TASK_KEY,
        instructions: str | None = None,
    ):
        self.rules = rules
        self._gateway = gateway
        self._task_key = task_key
        self._fallback = KeywordRouter(rules)
        self._instructions = instructions or self._default_instructions()

    def _default_instructions(self) -> str:
        """A taxonomy-neutral classifier prompt.

        The original inlined ten SAP-specific routing rules. Here the category
        list is generated from the rules and the application supplies any extra
        guidance via ``instructions``.
        """
        categories = ", ".join(sorted(self.rules.categories))
        return (
            "You are a message router. Classify the user's message into exactly "
            "one category.\n"
            "Valid categories: %s\n\n"
            "Respond with ONLY the category name, nothing else." % categories
        )

    async def route(
        self,
        message: str,
        context: Mapping[str, object] | None = None,
        *,
        ctx: RunContext | None = None,
    ) -> RoutingResult:
        run = ctx if ctx is not None else require_context()
        context = context or {}

        rendered_context = ", ".join(
            "%s=%s" % (k, v) for k, v in sorted(context.items())
        )
        user_prompt = "Context: %s\nMessage: %s" % (
            rendered_context or "none",
            message,
        )

        try:
            response = await self._gateway.invoke(
                self._task_key,
                system_prompt=self._instructions,
                user_prompt=user_prompt,
                ctx=run,
            )
        except Exception as exc:
            # Logged, unlike the original's silent `except Exception: return
            # route_message(...)`. A router model that is failing every call
            # should be visible, not just slow.
            log.warning(
                "router model call failed for run %s, using keyword fallback: %s",
                run.run_id,
                exc,
            )
            return self._fallback.route(message, context)

        answer = response.text.strip().lower().replace('"', "").replace("'", "")
        if answer in self.rules.categories:
            return RoutingResult(
                category=answer, decided_by="llm", raw_model_answer=response.text
            )

        log.info(
            "router returned out-of-set category %r; using keyword fallback", answer
        )
        result = self._fallback.route(message, context)
        return RoutingResult(
            category=result.category,
            decided_by=result.decided_by,
            raw_model_answer=response.text,
        )
