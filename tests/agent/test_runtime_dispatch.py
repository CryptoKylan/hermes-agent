"""Whole-turn runtime dispatch behavior contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agent.runtime_api import (
    RuntimeApprovalRequestEvent,
    RuntimeCompactionEvent,
    RuntimeCompactionPhase,
    RuntimeCompletedEvent,
    RuntimeEventKind,
    RuntimeSelection,
    RuntimeStateEnvelope,
    RuntimeStateEvent,
    RuntimeStatusEvent,
    RuntimeToolRequestEvent,
    RuntimeUsageEvent,
    RuntimeUsageReceipt,
)
from agent.runtime_dispatch import (
    HermesRuntimeHostServices,
    RuntimeExecutionError,
    build_runtime_turn_request,
    run_runtime_sync,
)
from model_tools import _run_async


class _HostServices:
    def __init__(self):
        self.statuses = []
        self.states = []
        self.receipts = []

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


def test_dispatch_rejects_unknown_event_types_and_closes_runtime_once():
    runtime = _UnknownEventRuntime()

    with pytest.raises(RuntimeExecutionError, match="unsupported event type"):
        run_runtime_sync(runtime, _request(), _HostServices())

    assert runtime.close_calls == 1


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
    )

    _run_async(host.persist_state(state))
    _run_async(host.persist_usage(receipt))
    assert agent._session_db.states == [("synthetic-session", state)]
    assert agent._session_db.receipts == [("synthetic-session", receipt)]
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


def test_dispatch_rejects_events_after_terminal_and_closes_runtime_once():
    runtime = _PostTerminalRuntime()

    with pytest.raises(RuntimeExecutionError, match="after its terminal event"):
        run_runtime_sync(runtime, _request(), _HostServices())

    assert runtime.close_calls == 1


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


def test_dispatch_closes_runtime_once_after_success():
    runtime = _SuccessfulRuntime()
    host = _HostServices()

    result = run_runtime_sync(runtime, _request(), host)

    assert result.response == {"final_response": "done"}
    assert host.statuses == ["working"]
    assert [state.runtime_id for state in host.states] == ["example-runtime"]
    assert [receipt.billing_mode for receipt in host.receipts] == [
        "subscription_included"
    ]
    assert runtime.close_calls == 1
