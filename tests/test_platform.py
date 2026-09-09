"""Tests for the four functions built from scratch.

All DB-free: the workflow engine runs against InMemoryCheckpointStore, agents
against a stub gateway, chunking and RRF are pure. Artifact and retrieval SQL
needs Postgres and is covered by `integration`-marked tests elsewhere.
"""
from __future__ import annotations

import pytest

from basis import Principal, RunContext, bind
from basis.agents import (
    AgentContext,
    AgentInput,
    AgentRegistry,
    AgentResult,
    AgentRunner,
    CallableAgent,
    Delegation,
    DelegationLimitExceeded,
)
from basis.artifacts import ArtifactState, LineageKind, content_hash
from basis.artifacts.store import _TRANSITIONS
from basis.errors import AuthorizationDenied
from basis.knowledge import (
    CHUNK_SIZE,
    Corpus,
    RetrievalQuery,
    RetrievedChunk,
    RRFReranker,
    chunk_text,
)
from basis.workflow import (
    InMemoryCheckpointStore,
    RetryPolicy,
    RunStatus,
    Step,
    StepKind,
    StepStatus,
    Workflow,
    WorkflowDefinitionError,
    WorkflowEngine,
    WorkflowRunError,
)


def _ctx(**kw):
    base = {"subject_id": "u1", "tenant_id": "t1"}
    roles = kw.pop("roles", ())
    kind = kw.pop("kind", "user")
    return RunContext(
        principal=Principal(
            subject_id=base["subject_id"],
            tenant_id=base["tenant_id"],
            kind=kind,
            roles=frozenset(roles),
        ),
        project_id=kw.pop("project_id", "p1"),
        **kw,
    )


# ═══════════════════════════════════════════════════════════════════════════
# AGENTS
# ═══════════════════════════════════════════════════════════════════════════

def _agent_ctx():
    return AgentContext(run=_ctx(), gateway=None)  # type: ignore[arg-type]


async def test_callable_agent_wraps_plain_return():
    agent = CallableAgent("echo", lambda ctx, task: task.payload.get("x"))

    async def fn(ctx, task):
        return task.payload.get("x")

    agent = CallableAgent("echo", fn)
    runner = AgentRunner(AgentRegistry([agent]))
    result = await runner.run("echo", _agent_ctx(), AgentInput(payload={"x": 42}))
    assert result.output == 42


async def test_agent_context_exposes_no_way_to_run_another_agent():
    """The structural half of invariant 1.

    If AgentContext ever gains a registry, runner or engine attribute, an agent
    can sequence work and the invariant becomes advisory again.
    """
    ctx = _agent_ctx()
    for forbidden in ("registry", "runner", "engine", "workflow", "memory", "db"):
        assert not hasattr(ctx, forbidden), (
            "AgentContext.%s would let an agent orchestrate" % forbidden
        )


async def test_delegation_is_satisfied_by_the_runner():
    async def needs_help(ctx, task):
        if "helper" not in task.upstream:
            return AgentResult(
                complete=False,
                delegations=[
                    Delegation(
                        agent_name="helper",
                        input=AgentInput(payload={"n": 5}),
                        result_key="helper",
                    )
                ],
            )
        return AgentResult(output=task.upstream["helper"] * 2)

    async def helper(ctx, task):
        return task.payload["n"]

    runner = AgentRunner(
        AgentRegistry([CallableAgent("main", needs_help), CallableAgent("helper", helper)])
    )
    result = await runner.run("main", _agent_ctx(), AgentInput())
    assert result.output == 10


async def test_delegation_round_limit():
    async def never_satisfied(ctx, task):
        return AgentResult(
            complete=False,
            delegations=[Delegation(agent_name="h", input=AgentInput())],
        )

    async def helper(ctx, task):
        return "x"

    runner = AgentRunner(
        AgentRegistry(
            [CallableAgent("main", never_satisfied), CallableAgent("helper", helper)]
        ),
        max_rounds=2,
    )
    # 'h' is not registered; register under the name the delegation uses.
    runner.registry.register(CallableAgent("h", helper))
    with pytest.raises(DelegationLimitExceeded):
        await runner.run("main", _agent_ctx(), AgentInput())


async def test_delegation_cycle_is_detected():
    async def a(ctx, task):
        return AgentResult(
            complete=False,
            delegations=[Delegation(agent_name="b", input=AgentInput())],
        )

    async def b(ctx, task):
        return AgentResult(
            complete=False,
            delegations=[Delegation(agent_name="a", input=AgentInput())],
        )

    runner = AgentRunner(
        AgentRegistry([CallableAgent("a", a), CallableAgent("b", b)])
    )
    with pytest.raises(DelegationLimitExceeded) as exc:
        await runner.run("a", _agent_ctx(), AgentInput())
    assert "cycle" in str(exc.value) or "depth" in str(exc.value)


def test_registry_rejects_duplicate_names():
    async def fn(ctx, task):
        return None

    reg = AgentRegistry([CallableAgent("dup", fn)])
    with pytest.raises(ValueError):
        reg.register(CallableAgent("dup", fn))


def test_agent_requires_a_name():
    from basis.agents.base import Agent

    class Nameless(Agent):
        async def run(self, ctx, task):
            return AgentResult()

    with pytest.raises(ValueError):
        Nameless()


def test_agent_input_with_upstream_is_immutable():
    a = AgentInput(payload={"x": 1})
    b = a.with_upstream("k", "v")
    assert a.upstream == {}
    assert b.upstream == {"k": "v"}
    assert b.payload == {"x": 1}


# ═══════════════════════════════════════════════════════════════════════════
# WORKFLOW — definition
# ═══════════════════════════════════════════════════════════════════════════

async def _noop(ctx, task):
    return "ok"


def _task_step(sid, deps=(), **kw):
    return Step(id=sid, kind=StepKind.TASK, task=_noop, depends_on=tuple(deps), **kw)


def test_layers_group_independent_steps():
    wf = Workflow(
        "w",
        [
            _task_step("a"),
            _task_step("b"),
            _task_step("c", ["a", "b"]),
        ],
    )
    assert wf.layers == (("a", "b"), ("c",))


def test_cycle_is_rejected_at_definition_time():
    with pytest.raises(WorkflowDefinitionError) as exc:
        Workflow("w", [_task_step("a", ["b"]), _task_step("b", ["a"])])
    assert "cycle" in str(exc.value)


def test_dangling_dependency_rejected():
    with pytest.raises(WorkflowDefinitionError) as exc:
        Workflow("w", [_task_step("a", ["nope"])])
    assert "unknown step" in str(exc.value)


def test_duplicate_step_id_rejected():
    with pytest.raises(WorkflowDefinitionError):
        Workflow("w", [_task_step("a"), _task_step("a")])


def test_self_dependency_rejected():
    with pytest.raises(WorkflowDefinitionError):
        _task_step("a", ["a"])


def test_empty_workflow_rejected():
    with pytest.raises(WorkflowDefinitionError):
        Workflow("w", [])


def test_agent_step_needs_an_agent():
    with pytest.raises(WorkflowDefinitionError):
        Step(id="a", kind=StepKind.AGENT)


def test_approval_step_must_not_carry_work():
    with pytest.raises(WorkflowDefinitionError):
        Step(id="a", kind=StepKind.APPROVAL, task=_noop)


def test_missing_agents_reported():
    wf = Workflow("w", [Step(id="s", kind=StepKind.AGENT, agent="ghost")])
    assert wf.missing_agents(["real"]) == frozenset({"ghost"})


# ═══════════════════════════════════════════════════════════════════════════
# WORKFLOW — execution
# ═══════════════════════════════════════════════════════════════════════════

async def test_run_completes_and_passes_upstream():
    seen = {}

    async def capture(ctx, task):
        seen.update(task.upstream)
        return "second"

    wf = Workflow(
        "w",
        [
            _task_step("first"),
            Step(id="second", kind=StepKind.TASK, task=capture, depends_on=("first",)),
        ],
    )
    engine = WorkflowEngine(store=InMemoryCheckpointStore())
    ctx = _ctx()
    with bind(ctx):
        record = await engine.start(wf, ctx)
    assert record.status is RunStatus.COMPLETE
    assert seen == {"first": "ok"}


async def test_failure_fails_run_and_skips_dependents():
    async def boom(ctx, task):
        raise RuntimeError("nope")

    wf = Workflow(
        "w",
        [
            Step(id="bad", kind=StepKind.TASK, task=boom),
            _task_step("after", ["bad"]),
        ],
    )
    store = InMemoryCheckpointStore()
    engine = WorkflowEngine(store=store)
    ctx = _ctx()
    with bind(ctx):
        with pytest.raises(WorkflowRunError):
            await engine.start(wf, ctx)

    record = store.load(
        next(iter(store._runs))[1], tenant_id="t1"
    )
    assert record is not None
    assert record.status is RunStatus.FAILED
    assert record.step("bad").status is StepStatus.FAILED
    # The distinction that matters: 'after' is SKIPPED, not left PENDING, so
    # "blocked by an upstream failure" is readable from the record.
    assert record.step("after").status is StepStatus.SKIPPED


async def test_retry_policy_retries_then_succeeds():
    calls = {"n": 0}

    async def flaky(ctx, task):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "recovered"

    wf = Workflow(
        "w",
        [
            Step(
                id="s",
                kind=StepKind.TASK,
                task=flaky,
                retry=RetryPolicy(max_attempts=3, base_delay=0.0),
            )
        ],
    )
    engine = WorkflowEngine(store=InMemoryCheckpointStore())
    ctx = _ctx()
    with bind(ctx):
        record = await engine.start(wf, ctx)
    assert record.status is RunStatus.COMPLETE
    assert calls["n"] == 3
    assert record.step("s").attempts == 3


def test_retry_policy_respects_exception_types():
    policy = RetryPolicy(max_attempts=5, retry_on=(TimeoutError,))
    assert policy.should_retry(TimeoutError(), 1)
    assert not policy.should_retry(ValueError(), 1)
    assert not policy.should_retry(TimeoutError(), 5)


def test_retry_delay_is_exponential_and_capped():
    policy = RetryPolicy(max_attempts=10, base_delay=1.0, max_delay=4.0)
    assert [policy.delay_for(i) for i in (1, 2, 3, 4, 9)] == [1.0, 2.0, 4.0, 4.0, 4.0]


async def test_approval_gate_pauses_then_resumes():
    wf = Workflow(
        "w",
        [
            _task_step("prepare"),
            Step(id="gate", kind=StepKind.APPROVAL, depends_on=("prepare",)),
            _task_step("publish", ["gate"]),
        ],
    )
    store = InMemoryCheckpointStore()
    engine = WorkflowEngine(store=store)
    ctx = _ctx()

    with bind(ctx):
        record = await engine.start(wf, ctx)
        assert record.status is RunStatus.WAITING_APPROVAL
        assert record.pending_approval() == "gate"
        assert record.step("publish").status is StepStatus.PENDING

        resumed = await engine.approve(wf, ctx, record.run_id, "gate")

    assert resumed.status is RunStatus.COMPLETE
    assert resumed.step("gate").approved_by == "u1"
    assert resumed.step("publish").status is StepStatus.COMPLETE


async def test_rejection_is_a_distinct_terminal_state():
    wf = Workflow("w", [Step(id="gate", kind=StepKind.APPROVAL), _task_step("after", ["gate"])])
    engine = WorkflowEngine(store=InMemoryCheckpointStore())
    ctx = _ctx()
    with bind(ctx):
        record = await engine.start(wf, ctx)
        rejected = await engine.reject(wf, ctx, record.run_id, "gate", reason="not ready")
    assert rejected.status is RunStatus.REJECTED
    assert rejected.status is not RunStatus.FAILED
    assert "not ready" in (rejected.error or "")


async def test_service_principal_cannot_approve():
    wf = Workflow("w", [Step(id="gate", kind=StepKind.APPROVAL)])
    engine = WorkflowEngine(store=InMemoryCheckpointStore())
    human = _ctx()
    with bind(human):
        record = await engine.start(wf, human)

    robot = _ctx(kind="service")
    with bind(robot):
        with pytest.raises(AuthorizationDenied):
            await engine.approve(wf, robot, record.run_id, "gate")


async def test_approver_role_is_enforced():
    wf = Workflow(
        "w",
        [Step(id="gate", kind=StepKind.APPROVAL, approver_roles=frozenset({"lead"}))],
    )
    engine = WorkflowEngine(store=InMemoryCheckpointStore())
    ctx = _ctx()
    with bind(ctx):
        record = await engine.start(wf, ctx)
        with pytest.raises(AuthorizationDenied):
            await engine.approve(wf, ctx, record.run_id, "gate")

    lead = _ctx(roles=["lead"])
    with bind(lead):
        done = await engine.approve(wf, lead, record.run_id, "gate")
    assert done.status is RunStatus.COMPLETE


async def test_resume_retries_an_interrupted_step():
    store = InMemoryCheckpointStore()
    wf = Workflow("w", [_task_step("s")])
    engine = WorkflowEngine(store=store)
    ctx = _ctx()

    with bind(ctx):
        record = await engine.start(wf, ctx)
        # Simulate a crash mid-step: RUNNING with no output.
        record.step("s").status = StepStatus.RUNNING
        record.step("s").output = None
        record.status = RunStatus.RUNNING
        store.save_run(record)

        resumed = await engine.resume(wf, ctx, record.run_id)

    assert resumed.status is RunStatus.COMPLETE
    assert resumed.step("s").output == "ok"


async def test_resume_of_terminal_run_is_a_noop():
    wf = Workflow("w", [_task_step("s")])
    store = InMemoryCheckpointStore()
    engine = WorkflowEngine(store=store)
    ctx = _ctx()
    with bind(ctx):
        record = await engine.start(wf, ctx)
        again = await engine.resume(wf, ctx, record.run_id)
    assert again.status is RunStatus.COMPLETE
    assert again.step("s").attempts == 1


async def test_completed_steps_are_not_rerun_on_resume():
    calls = {"n": 0}

    async def once(ctx, task):
        calls["n"] += 1
        return "done"

    wf = Workflow(
        "w",
        [
            Step(id="a", kind=StepKind.TASK, task=once),
            Step(id="gate", kind=StepKind.APPROVAL, depends_on=("a",)),
        ],
    )
    engine = WorkflowEngine(store=InMemoryCheckpointStore())
    ctx = _ctx()
    with bind(ctx):
        record = await engine.start(wf, ctx)
        await engine.approve(wf, ctx, record.run_id, "gate")
    # The paid step ran exactly once across the pause/resume boundary.
    assert calls["n"] == 1


def test_run_status_terminality():
    assert RunStatus.COMPLETE.is_terminal
    assert RunStatus.REJECTED.is_terminal
    assert not RunStatus.WAITING_APPROVAL.is_terminal


# ═══════════════════════════════════════════════════════════════════════════
# ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════

def test_content_hash_is_key_order_independent():
    # Without sort_keys, every re-ingestion of an identical payload would look
    # like an external edit.
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_content_hash_detects_change():
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_content_hash_handles_text_and_bytes():
    assert content_hash("x") == content_hash(b"x")


def test_approval_transitions_are_constrained():
    assert ArtifactState.IN_REVIEW in _TRANSITIONS[ArtifactState.DRAFT]
    assert ArtifactState.APPROVED in _TRANSITIONS[ArtifactState.IN_REVIEW]
    # Cannot jump straight from draft to approved - the review step is the control.
    assert ArtifactState.APPROVED not in _TRANSITIONS[ArtifactState.DRAFT]
    # Superseded is terminal.
    assert _TRANSITIONS[ArtifactState.SUPERSEDED] == frozenset()


def test_lineage_kinds_distinguish_revision_from_derivation():
    assert LineageKind.REVISES != LineageKind.DERIVED_FROM


def test_artifact_state_finality():
    assert ArtifactState.APPROVED.is_final
    assert not ArtifactState.DRAFT.is_final


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE
# ═══════════════════════════════════════════════════════════════════════════

def test_short_text_is_one_chunk():
    chunks = chunk_text("hello world")
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].start == 0


def test_empty_text_yields_nothing():
    assert chunk_text("") == []


def test_chunking_overlaps_and_covers():
    text = "word " * 20_000  # 100k chars
    chunks = chunk_text(text, chunk_size=10_000, overlap=1_000, max_chunks=None)
    assert len(chunks) > 1
    # Consecutive windows overlap rather than abut.
    assert chunks[1].start < chunks[0].end
    # Coverage reaches the end of the input.
    assert chunks[-1].end == len(text)


def test_chunking_breaks_on_paragraph_boundary():
    body = "A" * 6_000 + "\n\n" + "B" * 6_000
    chunks = chunk_text(body, chunk_size=8_000, overlap=100, max_chunks=None)
    # The first chunk should end at the paragraph break, not mid-A.
    assert chunks[0].text.endswith("\n\n")


def test_chunking_respects_max_chunks():
    text = "x" * 500_000
    chunks = chunk_text(text, chunk_size=1_000, overlap=100, max_chunks=5)
    assert len(chunks) == 5


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_text("abc" * 1000, chunk_size=100, overlap=100)


def test_lifted_defaults_preserved():
    assert CHUNK_SIZE == 24_000


def test_corpus_scoping_flags():
    assert Corpus.CLIENT_PROJECT.is_project_scoped
    assert Corpus.CLIENT_PROJECT.is_tenant_scoped
    assert Corpus.ORG_ASSETS.is_tenant_scoped
    assert not Corpus.ORG_ASSETS.is_project_scoped
    # The only corpus readable across tenants.
    assert not Corpus.PUBLIC_DOMAIN.is_tenant_scoped


def test_query_requires_a_corpus():
    with pytest.raises(ValueError):
        RetrievalQuery(text="x", corpora=())


def test_rrf_rewards_agreement_between_arms():
    # 'both' is mid-ranked in each arm but appears in both; 'vec_only' tops the
    # vector arm alone. RRF should still favour the chunk both arms found.
    both = RetrievedChunk(id="both", content="", corpus=Corpus.ORG_ASSETS,
                          distance=0.20, text_rank=0.50)
    vec_only = RetrievedChunk(id="vec", content="", corpus=Corpus.ORG_ASSETS,
                              distance=0.10)
    txt_only = RetrievedChunk(id="txt", content="", corpus=Corpus.ORG_ASSETS,
                              text_rank=0.90)

    ranked = RRFReranker(k=1).rerank(
        RetrievalQuery(text="q", limit=3), [both, vec_only, txt_only]
    )
    assert ranked[0].id == "both"
    assert ranked[0].score > ranked[1].score


def test_rrf_respects_limit():
    candidates = [
        RetrievedChunk(id=str(i), content="", corpus=Corpus.ORG_ASSETS, distance=i / 10)
        for i in range(10)
    ]
    ranked = RRFReranker().rerank(RetrievalQuery(text="q", limit=3), candidates)
    assert len(ranked) == 3


def test_rrf_handles_empty():
    assert RRFReranker().rerank(RetrievalQuery(text="q"), []) == []


def test_citation_format_matches_existing_convention():
    chunk = RetrievedChunk(
        id="c1", content="", corpus=Corpus.ORG_ASSETS, title="Exchange Rate Config"
    )
    assert chunk.citation() == "[Source: Exchange Rate Config]"


# ═══════════════════════════════════════════════════════════════════════════
# ADAPTER
# ═══════════════════════════════════════════════════════════════════════════

def test_aicore_context_maps_workspace_to_tenant():
    from basis.adapters.aicore import context_for_workspace

    ctx = context_for_workspace("ws-1", "user-9", roles=["admin"], project_id="p")
    assert ctx.tenant_id == "ws-1"
    assert ctx.user_id == "user-9"
    assert ctx.principal.has_role("admin")


def test_aicore_qualifies_bedrock_model_ids():
    from basis.adapters.aicore import AiCoreResolver

    assert (
        AiCoreResolver.qualify("bedrock", "claude-opus-5")
        == "us.anthropic.claude-opus-5"
    )
    # Already-prefixed ids are left alone.
    assert (
        AiCoreResolver.qualify("bedrock", "us.anthropic.claude-opus-5")
        == "us.anthropic.claude-opus-5"
    )
    # Non-Bedrock providers use the bare id.
    assert AiCoreResolver.qualify("ollama", "llama3") == "llama3"


def test_aicore_persona_drops_temperature_for_claude_5():
    """The live ai-core bug, fixed on the basis path.

    pipeline_agents.temperature defaults to 0.4 NOT NULL and run-model.ts sends
    it unconditionally; Claude 5-family models reject sampling params with a 400.
    """
    from basis.adapters.aicore import AiCorePersonaStore

    store = AiCorePersonaStore()
    row = {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "analyst",
        "description": None,
        "provider": "bedrock",
        "model": "claude-sonnet-5",
        "system_prompt": "you analyse",
        "temperature": 0.4,
        "max_output_tokens": 2000,
    }
    persona = store._to_persona(row)
    assert persona["model_id"] == "us.anthropic.claude-sonnet-5"
    assert "temperature" not in persona["model_config"]
    assert persona["model_config"]["max_tokens"] == 2000


def test_aicore_persona_keeps_temperature_for_haiku():
    from basis.adapters.aicore import AiCorePersonaStore

    row = {
        "id": "00000000-0000-0000-0000-000000000002",
        "name": "quick",
        "description": None,
        "provider": "bedrock",
        "model": "claude-haiku-4-5-20251001-v1:0",
        "system_prompt": "",
        "temperature": 0.4,
        "max_output_tokens": 1000,
    }
    persona = AiCorePersonaStore()._to_persona(row)
    assert persona["model_config"]["temperature"] == 0.4


def test_session_vars_nest_and_merge():
    from basis import db

    with db.session_vars(**{"app.workspace_id": "w1"}):
        assert db._session_vars.get() == {"app.workspace_id": "w1"}
        with db.session_vars(**{"app.other": "x"}):
            bound = db._session_vars.get()
            assert bound == {"app.workspace_id": "w1", "app.other": "x"}
        assert db._session_vars.get() == {"app.workspace_id": "w1"}
    assert db._session_vars.get() is None
