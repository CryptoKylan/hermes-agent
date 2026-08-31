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
    RuntimeExecutionError,
    build_runtime_turn_request,
    run_runtime_sync,
)


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
