"""Whole-turn runtime dispatch behavior contracts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from agent.runtime_api import (
    RuntimeBackgroundOutcome,
    RuntimeBackgroundResult,
    RuntimeApprovalRequestEvent,
    RuntimeCancelledEvent,
    RuntimeCompactionEvent,
    RuntimeCompactionPhase,
    CompactionOwnership,
    RuntimeCompletedEvent,
    RuntimeEventKind,
    RuntimeFailedEvent,
    RuntimeFailure,
    RuntimeFailurePhase,
    RuntimeDescriptor,
    RuntimeSelection,
    RuntimeStateEnvelope,
    RuntimeStateEvent,
    RuntimeStatusEvent,
    RuntimeToolRequestEvent,
    RuntimeUsageEvent,
    RuntimeUsageReceipt,
    RuntimeRegistration,
    runtime_api_manifest,
)
from agent.runtime_dispatch import (
    HermesRuntimeHostServices,
    RuntimeExecutionError,
    build_runtime_turn_request,
    close_runtime_session,
    get_runtime_session,
    make_builtin_codex_registration,
    run_runtime_sync,
)
from model_tools import _run_async


class _HostServices:
    def __init__(self):
        self.statuses = []
        self.states = []
        self.receipts = []
        self.compactions = []

    async def execute_tool(self, name, arguments):
        raise AssertionError("not used")

    async def request_approval(self, action, details):
        raise AssertionError("not used")

    async def emit_status(self, message):
        self.statuses.append(message)

    async def persist_state(self, state):
        self.states.append(state)

    async def persist_usage(self, receipt):
        self.receipts.append(receipt)

    async def emit_compaction(self, event):
        self.compactions.append(event)

    def cancellation_requested(self):
        return False


def _request():
    return build_runtime_turn_request(
        provider="example",
        model="example-large",
        api_mode="example_runtime",
        messages=({"role": "user", "content": "hello"},),
        prompt_snapshot="stable prompt",
        tool_schemas=(),
    )


def test_runtime_turn_request_deep_freezes_state_and_host_inputs():
    messages = [{"role": "user", "content": {"parts": ["hello"]}}]
    tools = [{"type": "function", "function": {"name": "pwd"}}]
    state_data = {"resume": {"external": "synthetic"}}

    request = build_runtime_turn_request(
        provider="example",
        model="example-large",
        api_mode="example_runtime",
        messages=messages,
        prompt_snapshot="stable prompt",
        tool_schemas=tools,
        session_state=RuntimeStateEnvelope(
            runtime_id="example-runtime",
            schema_version=1,
            state=state_data,
        ),
    )
    messages[0]["content"]["parts"].append("late mutation")
    tools[0]["function"]["name"] = "terminal"
    state_data["resume"]["external"] = "late mutation"

    assert request.messages[0]["content"]["parts"] == ("hello",)
    assert request.tool_schemas[0]["function"]["name"] == "pwd"
    assert request.session_state.state["resume"]["external"] == "synthetic"
    with pytest.raises(TypeError):
        request.session_state.state["resume"]["external"] = "blocked"


def test_public_runtime_request_events_are_typed_and_frozen():
    tool = RuntimeToolRequestEvent(
        request_id="tool-1",
        name="pwd",
        arguments={"path": "."},
    )
    approval = RuntimeApprovalRequestEvent(
        request_id="approval-1",
        action="terminal",
        details={"reason": "synthetic test"},
    )
    compaction = RuntimeCompactionEvent(
        phase=RuntimeCompactionPhase.STARTED,
        details={"watchdog_seconds": 60},
    )

    assert tool.kind is RuntimeEventKind.TOOL_REQUEST
    assert approval.kind is RuntimeEventKind.APPROVAL_REQUEST
    assert compaction.kind is RuntimeEventKind.COMPACTION
    with pytest.raises(Exception):
        tool.name = "terminal"


class _UnknownEventRuntime:
    def __init__(self):
        self.close_calls = 0

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield object()
        yield RuntimeCompletedEvent(result={"final_response": "done"})

    async def close(self):
        self.close_calls += 1


def test_dispatch_rejects_unknown_event_types_without_closing_session_runtime():
    runtime = _UnknownEventRuntime()

    with pytest.raises(RuntimeExecutionError, match="unsupported event type"):
        run_runtime_sync(runtime, _request(), _HostServices())

    assert runtime.close_calls == 0


class _RuntimeDatabase:
    def __init__(self):
        self.states = []
        self.receipts = []
        self.aggregate_receipts = []
        self.inserted = True

    def update_runtime_state(self, session_id, state):
        self.states.append((session_id, state))

    def record_runtime_usage_receipt(self, session_id, receipt):
        self.receipts.append((session_id, receipt))
        return self.inserted

    def queue_token_counts(self, session_id, **kwargs):
        self.aggregate_receipts.append((session_id, kwargs))


class _RuntimeAgent:
    valid_tool_names = frozenset()
    tools = ()
    session_id = "synthetic-session"
    _interrupt_requested = False

    def __init__(self):
        self._session_db = _RuntimeDatabase()


def test_host_persists_runtime_state_and_idempotent_usage_for_selected_runtime():
    agent = _RuntimeAgent()
    host = HermesRuntimeHostServices(
        agent,
        task_id="synthetic-task",
        runtime_id="example-runtime",
    )
    state = RuntimeStateEnvelope(
        runtime_id="example-runtime",
        schema_version=1,
        state={"external": "synthetic"},
    )
    receipt = RuntimeUsageReceipt(
        runtime_id="example-runtime",
        provider="example",
        model="example-large",
        billing_mode="subscription_included",
        cost_status="included",
        correlation_id="synthetic-turn",
        fallback_used=True,
        failure_phase=RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT,
    )

    _run_async(host.persist_state(state))
    _run_async(host.persist_usage(receipt))
    assert agent._session_db.states == [("synthetic-session", state)]
    assert agent._session_db.receipts == [("synthetic-session", receipt)]
    assert agent._session_db.receipts[0][1].fallback_used is True
    assert (
        agent._session_db.receipts[0][1].failure_phase
        is RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT
    )
    assert len(agent._session_db.aggregate_receipts) == 1

    agent._session_db.inserted = False
    _run_async(host.persist_usage(receipt))
    assert len(agent._session_db.receipts) == 2
    assert len(agent._session_db.aggregate_receipts) == 1


def test_host_rejects_state_and_usage_for_a_different_runtime():
    host = HermesRuntimeHostServices(
        _RuntimeAgent(),
        task_id="synthetic-task",
        runtime_id="example-runtime",
    )
    wrong_state = RuntimeStateEnvelope(
        runtime_id="other-runtime",
        schema_version=1,
        state={},
    )
    wrong_receipt = RuntimeUsageReceipt(
        runtime_id="other-runtime",
        provider="example",
        model="example-large",
        billing_mode="subscription_included",
        cost_status="included",
    )

    with pytest.raises(RuntimeExecutionError, match="identity does not match"):
        _run_async(host.persist_state(wrong_state))
    with pytest.raises(RuntimeExecutionError, match="identity does not match"):
        _run_async(host.persist_usage(wrong_receipt))


class _PostTerminalRuntime:
    def __init__(self):
        self.close_calls = 0

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeCompletedEvent(result={"final_response": "done"})
        yield RuntimeStatusEvent(message="too late")

    async def close(self):
        self.close_calls += 1


def test_dispatch_rejects_events_after_terminal_without_closing_session_runtime():
    runtime = _PostTerminalRuntime()

    with pytest.raises(RuntimeExecutionError, match="after its terminal event"):
        run_runtime_sync(runtime, _request(), _HostServices())

    assert runtime.close_calls == 0


class _SuccessfulRuntime:
    def __init__(self):
        self.close_calls = 0

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeStatusEvent(message="working")
        yield RuntimeStateEvent(
            state=RuntimeStateEnvelope(
                runtime_id="example-runtime",
                schema_version=1,
                state={"external_session": "synthetic"},
            )
        )
        yield RuntimeUsageEvent(
            receipt=RuntimeUsageReceipt(
                runtime_id="example-runtime",
                provider="example",
                model="example-large",
                billing_mode="subscription_included",
                cost_status="included",
            )
        )
        yield RuntimeCompletedEvent(result={"final_response": "done"})

    async def close(self):
        self.close_calls += 1


def test_dispatch_keeps_runtime_open_after_success():
    runtime = _SuccessfulRuntime()
    host = _HostServices()

    result = run_runtime_sync(runtime, _request(), host)

    assert result.response == {"final_response": "done"}
    assert host.statuses == ["working"]
    assert [state.runtime_id for state in host.states] == ["example-runtime"]
    assert [receipt.billing_mode for receipt in host.receipts] == [
        "subscription_included"
    ]
    assert runtime.close_calls == 0


class _FailedRuntime:
    def __init__(self):
        self.close_calls = 0

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeFailedEvent(
            failure=RuntimeFailure(
                code="synthetic_failure",
                message="synthetic failure",
                phase=RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT,
                replay_safe=False,
                retryable=True,
            )
        )

    async def close(self):
        self.close_calls += 1


def test_dispatch_returns_classified_failure_without_authorizing_fallback():
    runtime = _FailedRuntime()

    result = run_runtime_sync(runtime, _request(), _HostServices())

    assert result.failure is not None
    assert result.failure.phase is RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT
    assert result.failure.replay_safe is False
    assert result.replay_safe is False
    assert isinstance(result.terminal, RuntimeFailedEvent)
    assert runtime.close_calls == 0


class _ExplodingRuntime:
    def __init__(self):
        self.close_calls = 0

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        raise RuntimeError("synthetic transport failure")
        yield  # pragma: no cover

    async def close(self):
        self.close_calls += 1


def test_unclassified_runtime_exception_is_fail_closed_without_per_turn_close():
    runtime = _ExplodingRuntime()

    result = run_runtime_sync(runtime, _request(), _HostServices())

    assert result.failure is not None
    assert result.failure.code == "runtime_exception"
    assert result.failure.phase is RuntimeFailurePhase.BEFORE_VISIBLE_OUTPUT
    assert result.failure.replay_safe is False
    assert runtime.close_calls == 0


class _ExplodingAfterStateRuntime:
    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeStateEvent(
            state=RuntimeStateEnvelope(
                runtime_id="example-runtime",
                schema_version=1,
                state={"external_session": "synthetic"},
            )
        )
        raise RuntimeError("synthetic failure after persistence")

    async def close(self):
        return None


def test_unclassified_exception_after_persistence_is_classified_after_side_effects():
    result = run_runtime_sync(
        _ExplodingAfterStateRuntime(),
        _request(),
        _HostServices(),
    )

    assert result.failure is not None
    assert result.failure.phase is RuntimeFailurePhase.AFTER_SIDE_EFFECTS
    assert result.failure.replay_safe is False


class _ExplodingAfterStatusRuntime:
    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeStatusEvent(message="working")
        raise RuntimeError("synthetic failure after visible status")

    async def close(self):
        return None


def test_unclassified_exception_after_visible_status_is_not_preflight_safe():
    result = run_runtime_sync(
        _ExplodingAfterStatusRuntime(),
        _request(),
        _HostServices(),
    )

    assert result.failure is not None
    assert result.failure.phase is RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT
    assert result.failure.replay_safe is False


class _CompactingRuntime:
    def __init__(self):
        self.close_calls = 0

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeCompactionEvent(
            phase=RuntimeCompactionPhase.STARTED,
            details={"watchdog_seconds": 30},
        )
        yield RuntimeCompactionEvent(phase=RuntimeCompactionPhase.COMPLETED)
        yield RuntimeCompletedEvent(result={"final_response": "done"})

    async def close(self):
        self.close_calls += 1


def test_runtime_compaction_events_are_projected_and_recorded_before_completion():
    runtime = _CompactingRuntime()
    host = _HostServices()

    result = run_runtime_sync(runtime, _request(), host)

    assert [event.phase for event in host.compactions] == [
        RuntimeCompactionPhase.STARTED,
        RuntimeCompactionPhase.COMPLETED,
    ]
    assert [event.phase for event in result.events if isinstance(event, RuntimeCompactionEvent)] == [
        RuntimeCompactionPhase.STARTED,
        RuntimeCompactionPhase.COMPLETED,
    ]
    assert runtime.close_calls == 0


def test_host_owned_compaction_rejects_runtime_compaction_event():
    runtime = _CompactingRuntime()
    descriptor = RuntimeDescriptor(
        runtime_id="host-owned-runtime",
        plugin_version="0.1.0",
        runtime_api_min=1,
        runtime_api_max=1,
        required_host_capabilities=frozenset(),
        provider_ids=frozenset({"example"}),
        api_modes=frozenset({"example_runtime"}),
        session_state_schema_version=1,
        compaction_ownership=CompactionOwnership.HOST,
    )

    with pytest.raises(RuntimeExecutionError, match="host owns compaction"):
        run_runtime_sync(
            runtime,
            _request(),
            _HostServices(),
            descriptor=descriptor,
        )

    assert runtime.close_calls == 0


class _CancelledRuntime:
    def __init__(self):
        self.close_calls = 0

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeCancelledEvent(reason="synthetic cancellation")

    async def close(self):
        self.close_calls += 1


def test_runtime_cancellation_is_one_terminal_outcome_without_per_turn_close():
    runtime = _CancelledRuntime()

    result = run_runtime_sync(runtime, _request(), _HostServices())

    assert result.cancelled is True
    assert isinstance(result.terminal, RuntimeCancelledEvent)
    assert sum(
        isinstance(event, (RuntimeCompletedEvent, RuntimeCancelledEvent, RuntimeFailedEvent))
        for event in result.events
    ) == 1
    assert runtime.close_calls == 0


class _CancelledAfterTerminalRuntime:
    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeCompletedEvent(result={"final_response": "done"})
        raise asyncio.CancelledError

    async def close(self):
        return None


def test_cancellation_after_terminal_preserves_exactly_one_terminal_event():
    result = run_runtime_sync(
        _CancelledAfterTerminalRuntime(),
        _request(),
        _HostServices(),
    )

    assert result.completed is True
    assert result.response == {"final_response": "done"}
    assert sum(
        isinstance(event, (RuntimeCompletedEvent, RuntimeCancelledEvent, RuntimeFailedEvent))
        for event in result.events
    ) == 1


def test_background_result_is_bounded_provider_neutral_and_immutable():
    result = RuntimeBackgroundResult(
        content="synthetic background result",
        outcome=RuntimeBackgroundOutcome.COMPLETED,
    )

    assert result.content == "synthetic background result"
    assert set(result.__dataclass_fields__) == {"content", "outcome"}
    with pytest.raises(Exception):
        result.content = "late mutation"
    with pytest.raises(ValueError, match="content exceeds"):
        RuntimeBackgroundResult(content="x" * 16_385)


def test_background_delivery_capability_has_a_host_consumer():
    assert "background_delivery_v1" in runtime_api_manifest()["host_capabilities"]
    assert callable(HermesRuntimeHostServices.emit_background_result)


def test_host_queues_background_result_for_exact_bound_parent_and_rejects_after_close(
    monkeypatch,
):
    from queue import SimpleQueue

    from gateway import session_context
    from tools.process_registry import format_process_notification, process_registry

    route = {
        "HERMES_SESSION_KEY": "telegram:direct:synthetic-chat",
        "HERMES_UI_SESSION_ID": "synthetic-ui",
        "HERMES_SESSION_PLATFORM": "telegram",
        "HERMES_SESSION_CHAT_TYPE": "direct",
        "HERMES_SESSION_CHAT_ID": "synthetic-chat",
        "HERMES_SESSION_THREAD_ID": "synthetic-thread",
        "HERMES_SESSION_USER_ID": "synthetic-user",
        "HERMES_SESSION_SCOPE_ID": "synthetic-scope",
    }
    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda name, default="": route.get(name, default),
    )
    queue = SimpleQueue()
    monkeypatch.setattr(process_registry, "completion_queue", queue)
    host = HermesRuntimeHostServices(
        _RuntimeAgent(),
        task_id="synthetic-task",
        runtime_id="example-runtime",
    )

    _run_async(
        host.emit_background_result(
            RuntimeBackgroundResult(content="background complete")
        )
    )

    event = queue.get_nowait()
    assert event["parent_session_id"] == "synthetic-session"
    assert event["session_key"] == route["HERMES_SESSION_KEY"]
    assert event["origin_ui_session_id"] == route["HERMES_UI_SESSION_ID"]
    assert event["chat_id"] == route["HERMES_SESSION_CHAT_ID"]
    assert event["summary"] == "background complete"
    assert event["type"] == "async_delegation"
    assert "background complete" in format_process_notification(event)

    _run_async(host.close())
    with pytest.raises(RuntimeExecutionError, match="closed"):
        _run_async(
            host.emit_background_result(RuntimeBackgroundResult(content="too late"))
        )


def test_runtime_and_host_binding_are_reused_until_session_close():
    instances = []

    def factory():
        runtime = _SuccessfulRuntime()
        instances.append(runtime)
        return runtime

    registration = RuntimeRegistration(
        descriptor=RuntimeDescriptor(
            runtime_id="example-runtime",
            plugin_version="0.1.0",
            runtime_api_min=1,
            runtime_api_max=1,
            required_host_capabilities=frozenset({"background_delivery_v1"}),
            provider_ids=frozenset({"example"}),
            api_modes=frozenset({"example_runtime"}),
            session_state_schema_version=1,
        ),
        factory=factory,
        plugin_id="synthetic-plugin",
    )
    agent = _RuntimeAgent()

    first = get_runtime_session(
        agent,
        registration,
        task_id="synthetic-turn-1",
    )
    second = get_runtime_session(
        agent,
        registration,
        task_id="synthetic-turn-2",
    )

    assert first is second
    assert first.runtime is instances[0]
    assert len(instances) == 1
    close_runtime_session(agent)
    close_runtime_session(agent)
    assert instances[0].close_calls == 1


def test_session_change_closes_old_runtime_before_rebinding():
    instances = []

    def factory():
        runtime = _SuccessfulRuntime()
        instances.append(runtime)
        return runtime

    registration = RuntimeRegistration(
        descriptor=RuntimeDescriptor(
            runtime_id="example-runtime",
            plugin_version="0.1.0",
            runtime_api_min=1,
            runtime_api_max=1,
            required_host_capabilities=frozenset({"background_delivery_v1"}),
            provider_ids=frozenset({"example"}),
            api_modes=frozenset({"example_runtime"}),
            session_state_schema_version=1,
        ),
        factory=factory,
        plugin_id="synthetic-plugin",
    )
    agent = _RuntimeAgent()
    first = get_runtime_session(agent, registration, task_id="turn-1")

    agent.session_id = "synthetic-session-2"
    second = get_runtime_session(agent, registration, task_id="turn-2")

    assert first is not second
    assert instances[0].close_calls == 1
    assert instances[1].close_calls == 0
    close_runtime_session(agent)


def test_builtin_codex_session_refreshes_its_per_turn_runner():
    agent = _RuntimeAgent()
    first = get_runtime_session(
        agent,
        make_builtin_codex_registration(lambda: {"final_response": "first"}),
        task_id="turn-1",
    )
    first_result = run_runtime_sync(first.runtime, _request(), first.host)
    second = get_runtime_session(
        agent,
        make_builtin_codex_registration(lambda: {"final_response": "second"}),
        task_id="turn-2",
    )
    second_result = run_runtime_sync(second.runtime, _request(), second.host)

    assert first is second
    assert first_result.response == {"final_response": "first"}
    assert second_result.response == {"final_response": "second"}
    close_runtime_session(agent)
