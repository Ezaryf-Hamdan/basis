"""Tests for the DB-free half of basis.

Everything here runs without Postgres, AWS or an OTel exporter - which is
itself one of the points of the refactor. The lifted modules could not be
imported at all without a DATABASE_URL in the environment.
"""
from __future__ import annotations

import pytest

from basis import Principal, RunContext, bind, current_context, require_context
from basis.errors import TenantIsolationError, ToolDenied
from basis.models import catalog
from basis.models.resolver import validate_model_config
from basis.observability.redaction import redact_pii, truncate
from basis.personas.routing import KeywordRouter, RoutingRules
from basis.tools.capture import extract_tool_calls
from basis.tools.policy import ToolEffect, ToolPolicy

# ── context ────────────────────────────────────────────────────────────────

def _principal(**kw):
    base = {"subject_id": "u1", "tenant_id": "t1"}
    base.update(kw)
    return Principal(**base)


def test_principal_requires_tenant():
    with pytest.raises(TenantIsolationError):
        Principal(subject_id="u1", tenant_id="")


def test_principal_requires_subject():
    with pytest.raises(TenantIsolationError):
        Principal(subject_id="", tenant_id="t1")


def test_run_id_is_generated():
    a = RunContext(principal=_principal())
    b = RunContext(principal=_principal())
    assert a.run_id and b.run_id and a.run_id != b.run_id


def test_bind_and_require():
    ctx = RunContext(principal=_principal(), project_id="p1")
    assert current_context() is None
    with bind(ctx):
        assert require_context() is ctx
    assert current_context() is None


def test_require_context_without_bind_raises():
    with pytest.raises(TenantIsolationError):
        require_context()


def test_bind_restores_previous_on_exception():
    outer = RunContext(principal=_principal())
    inner = RunContext(principal=_principal())
    with bind(outer):
        with pytest.raises(RuntimeError):
            with bind(inner):
                raise RuntimeError("boom")
        assert require_context() is outer


def test_require_project_raises_when_unscoped():
    ctx = RunContext(principal=_principal())
    with pytest.raises(TenantIsolationError):
        ctx.require_project()


def test_child_gets_new_run_id_and_keeps_tenant():
    parent = RunContext(principal=_principal(), project_id="p1")
    child = parent.child(task_key="sub")
    assert child.run_id != parent.run_id
    assert child.tenant_id == parent.tenant_id
    assert child.project_id == "p1"
    assert child.task_key == "sub"


def test_child_cannot_change_tenant():
    parent = RunContext(principal=_principal())
    other = Principal(subject_id="u2", tenant_id="t2")
    with pytest.raises(TenantIsolationError):
        parent.child(principal=other)


def test_span_attributes_carry_tenancy():
    ctx = RunContext(
        principal=_principal(), project_id="p1", job_id="j1", task_key="classify"
    )
    attrs = ctx.span_attributes()
    assert attrs["basis.tenant_id"] == "t1"
    assert attrs["basis.project_id"] == "p1"
    assert attrs["basis.run_id"] == ctx.run_id
    assert attrs["basis.task_key"] == "classify"
    # Unset optionals must be absent, not None-valued - OTel rejects None.
    assert "basis.session_id" not in attrs
    assert all(v is not None for v in attrs.values())


def test_principal_from_claims():
    p = Principal.from_claims(
        {"sub": "u9", "tenant_id": "t9", "email": "a@b.com", "roles": ["admin"]},
        token="jwt",
    )
    assert p.subject_id == "u9"
    assert p.tenant_id == "t9"
    assert p.has_role("admin")
    assert p.token == "jwt"
    assert not p.is_service


def test_service_principal():
    p = Principal.service("svc", "t1", token="k")
    assert p.is_service


# ── tool policy ────────────────────────────────────────────────────────────

POLICY = ToolPolicy(
    allow_exact=frozenset({"propose_change", "trigger_build"}),
    known_scopes=frozenset({"authoring"}),
    signal_names=frozenset({"propose_change", "trigger_build"}),
)


def test_reads_always_allowed():
    for name in ("get_thing", "list_things", "search_things", "browse_things"):
        assert POLICY.allows(name)


def test_writers_never_allowed_even_in_known_scope():
    for name in (
        "create_thing",
        "update_thing",
        "delete_thing",
        "link_thing",
        "set_project_owner",
        "send_notification",
    ):
        assert not POLICY.allows(name, scope="authoring")


def test_unknown_tool_excluded_by_default():
    # The core property: a tool nobody classified is refused, so a future
    # writer added upstream is not silently reachable.
    assert not POLICY.allows("frobnicate_everything", scope="authoring")


def test_allow_exact_requires_known_scope():
    assert POLICY.allows("propose_change", scope="authoring")
    assert not POLICY.allows("propose_change", scope="something_else")
    assert not POLICY.allows("propose_change", scope=None)


def test_grants_can_only_narrow():
    assert POLICY.allows("get_thing")
    assert not POLICY.allows("get_thing", granted=frozenset({"list_thing"}))
    # A grant cannot widen past the policy.
    assert not POLICY.allows(
        "delete_thing", scope="authoring", granted=frozenset({"delete_thing"})
    )


def test_empty_grant_set_denies_everything():
    # None means "no persona narrowing"; an empty set means "granted nothing".
    assert POLICY.allows("get_thing", granted=None)
    assert not POLICY.allows("get_thing", granted=frozenset())


def test_classify():
    assert POLICY.classify("get_x") is ToolEffect.READ
    assert POLICY.classify("create_x") is ToolEffect.WRITE
    assert POLICY.classify("propose_change") is ToolEffect.SIGNAL
    assert POLICY.classify("mystery") is ToolEffect.UNKNOWN


def test_default_policy_is_read_only():
    bare = ToolPolicy()
    assert bare.allows("get_x")
    assert not bare.allows("propose_change", scope="anything")


def test_filter_normalizes_tool_shapes():
    tools = [
        {"name": "get_a"},
        {"tool_spec": {"name": "delete_b"}},
        {"tool_spec": {"name": "list_c"}},
    ]
    kept = list(POLICY.filter(tools))
    assert len(kept) == 2


def test_denied_reason_is_actionable():
    assert POLICY.denied_reason("get_x") is None
    assert "write-effect" in POLICY.denied_reason("delete_x", scope="authoring")
    assert "unknown scope" in POLICY.denied_reason("propose_change", scope="nope")


# ── transcript capture ─────────────────────────────────────────────────────

def test_extract_pairs_by_tool_use_id_not_position():
    # Parallel tool use: results arrive in the opposite order to the calls.
    messages = [
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "1", "name": "get_a", "input": {"x": 1}}},
                {"toolUse": {"toolUseId": "2", "name": "get_b", "input": {"y": 2}}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "2", "content": [{"text": "B"}]}},
                {"toolResult": {"toolUseId": "1", "content": [{"text": "A"}]}},
            ],
        },
    ]
    calls = extract_tool_calls(messages)
    assert [c.name for c in calls] == ["get_a", "get_b"]
    assert [c.result_text for c in calls] == ["A", "B"]


def test_extract_joins_multiple_result_blocks():
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "f"}}]},
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "1",
                        "content": [{"text": "one"}, {"text": "two"}],
                    }
                }
            ],
        },
    ]
    calls = extract_tool_calls(messages)
    assert calls[0].result_text == "one\ntwo"


def test_extract_skips_malformed_without_losing_the_rest():
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"name": "missing_id"}}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "2", "name": "ok"}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "2", "content": []}}]},
        "not a dict",
        {"role": "user", "content": "not a list"},
    ]
    calls = extract_tool_calls(messages)
    assert [c.name for c in calls] == ["ok"]


def test_unanswered_call_is_omitted():
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "f"}}]}
    ]
    assert extract_tool_calls(messages) == []


def test_truncated_reports_the_flag():
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "f"}}]},
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "1", "content": [{"text": "x" * 50}]}}
            ],
        },
    ]
    call = extract_tool_calls(messages)[0]
    text, was_truncated = call.truncated(10)
    assert was_truncated and len(text) == 10
    assert call.truncated(100) == ("x" * 50, False)


# ── redaction ──────────────────────────────────────────────────────────────

def test_redact_none_returns_empty_string():
    assert redact_pii(None) == ""
    assert redact_pii("") == ""


def test_redact_email_and_phone():
    out = redact_pii("mail bob.smith@example.com or call +1 555 123 4567")
    assert "bob.smith@example.com" not in out
    assert "<EMAIL>" in out
    assert "<PHONE>" in out


def test_redact_credentials():
    assert "<AWS_KEY>" in redact_pii("key AKIAIOSFODNN7EXAMPLE here")
    assert "AKIAIOSFODNN7EXAMPLE" not in redact_pii("key AKIAIOSFODNN7EXAMPLE here")
    out = redact_pii("api_key: sk-abc123secret")
    assert "sk-abc123secret" not in out


def test_redact_private_key_block():
    blob = "-----BEGIN RSA PRIVATE KEY-----\nabc\ndef\n-----END RSA PRIVATE KEY-----"
    assert redact_pii(blob) == "<PRIVATE_KEY>"


def test_truncate_marks_truncation():
    assert truncate("abc", 10) == "abc"
    out = truncate("a" * 100, 10)
    assert out.startswith("a" * 10)
    assert out.endswith("...[TRUNCATED]")


# ── model catalog ──────────────────────────────────────────────────────────

def test_claude_5_family_rejects_sampling():
    # The bug this catalog exists to prevent: the source passed temperature to
    # these models unconditionally, which returns a 400.
    assert not catalog.accepts_sampling("us.anthropic.claude-sonnet-5")
    assert not catalog.accepts_sampling("us.anthropic.claude-opus-5")
    assert not catalog.accepts_sampling("us.anthropic.claude-opus-4-8")


def test_haiku_and_nova_accept_sampling():
    assert catalog.accepts_sampling("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert catalog.accepts_sampling("us.amazon.nova-pro-v1:0")


def test_unknown_claude_5_id_still_refuses_sampling():
    # Substring fallback, so an id absent from the catalog is handled safely.
    assert not catalog.accepts_sampling("eu.anthropic.claude-opus-5-something")


def test_caching_support():
    assert catalog.supports_caching("us.anthropic.claude-opus-5")
    assert not catalog.supports_caching("us.amazon.nova-pro-v1:0")


def test_prices_present_for_claude_models():
    assert catalog.price_for("us.anthropic.claude-opus-5") == (5.00, 25.00)
    assert catalog.price_for("us.anthropic.claude-sonnet-5") == (2.00, 10.00)
    assert catalog.price_for("us.amazon.nova-pro-v1:0") is None


def test_canonical_id_lookup_works():
    assert catalog.spec_for("claude-opus-5") is not None


def test_available_models_shape_is_backward_compatible():
    rows = catalog.available_models()
    assert rows and {"id", "name", "max_tokens_cap", "supports_caching"} <= set(rows[0])


# ── config validation ──────────────────────────────────────────────────────

def test_valid_config_passes():
    assert validate_model_config({"max_tokens": 4096}) == []


def test_max_tokens_bounds_use_model_cap():
    assert validate_model_config({"max_tokens": 100_000}) == []
    errs = validate_model_config(
        {"max_tokens": 100_000}, "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    assert errs and "max_tokens" in errs[0]


def test_temperature_rejected_for_sonnet_5():
    errs = validate_model_config(
        {"temperature": 0.5}, "us.anthropic.claude-sonnet-5"
    )
    assert errs and "not supported" in errs[0]


def test_temperature_allowed_for_haiku():
    assert (
        validate_model_config(
            {"temperature": 0.5}, "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        == []
    )


def test_bool_is_not_a_valid_int():
    # isinstance(True, int) is True in Python, so this needs an explicit guard.
    assert validate_model_config({"max_tokens": True}) != []


def test_bad_stop_sequences():
    assert validate_model_config({"stop_sequences": "nope"}) != []
    assert validate_model_config({"stop_sequences": [1, 2]}) != []
    assert validate_model_config({"stop_sequences": ["ok"]}) == []


# ── routing ────────────────────────────────────────────────────────────────

RULES = RoutingRules(
    categories=frozenset({"security", "architect", "finance"}),
    keywords={
        "security": ["authorization", "role"],
        "architect": ["architecture", "integration"],
    },
    priority=["security", "architect"],
    default_category="architect",
    context_selectable=frozenset({"finance"}),
)


def test_keyword_priority_order_respected():
    r = KeywordRouter(RULES)
    # Contains both an architecture and a security term; security wins.
    result = r.route("integration of the authorization model")
    assert result.category == "security"
    assert result.decided_by == "keyword"


def test_context_selects_when_no_keyword_matches():
    r = KeywordRouter(RULES)
    result = r.route("what about the numbers", {"workstream": "finance"})
    assert result.category == "finance"
    assert result.decided_by == "context"


def test_falls_back_to_default():
    r = KeywordRouter(RULES)
    result = r.route("hello there")
    assert result.category == "architect"
    assert result.decided_by == "default"


def test_context_cannot_select_a_non_selectable_category():
    r = KeywordRouter(RULES)
    result = r.route("hello", {"workstream": "security"})
    assert result.decided_by == "default"


def test_rules_reject_unknown_category_references():
    with pytest.raises(ValueError):
        RoutingRules(categories=frozenset({"a"}), keywords={"b": ["x"]})
    with pytest.raises(ValueError):
        RoutingRules(categories=frozenset({"a"}), default_category="z")


# ── consolidation parsing ──────────────────────────────────────────────────

def test_parse_fenced_json_block():
    from basis.memory.consolidator import parse_consolidation_response

    raw = 'prose\n```json\n[{"memory_type":"fact","content":"x","importance":0.9}]\n```\nmore'
    out = parse_consolidation_response(raw)
    assert out == [{"memory_type": "fact", "content": "x", "importance": 0.9}]


def test_parse_bare_array_fallback():
    from basis.memory.consolidator import parse_consolidation_response

    out = parse_consolidation_response('[{"memory_type":"decision","content":"y"}]')
    assert out[0]["importance"] == 0.5


def test_parse_rejects_unknown_memory_type():
    from basis.memory.consolidator import parse_consolidation_response

    out = parse_consolidation_response('[{"memory_type":"invented","content":"z"}]')
    assert out == []


def test_parse_clamps_importance():
    from basis.memory.consolidator import parse_consolidation_response

    out = parse_consolidation_response(
        '[{"memory_type":"fact","content":"a","importance":9}]'
    )
    assert out[0]["importance"] == 1.0


def test_parse_handles_garbage():
    from basis.memory.consolidator import parse_consolidation_response

    for raw in ("", "no json here", "{}", "[", '{"not":"a list"}'):
        assert parse_consolidation_response(raw) == []


# ── errors ─────────────────────────────────────────────────────────────────

def test_error_markers_are_stable():
    from basis.errors import AuthorizationDenied, DelegationExpired

    assert AuthorizationDenied("fsd", "write", "nope").marker().startswith(
        "rbac_denied: "
    )
    assert DelegationExpired("ttl").marker().startswith("delegation_expired: ")
    assert ToolDenied("delete_x", "write").marker().startswith("tool_denied: ")
