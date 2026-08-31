"""Host-owned dispatch for built-in and plugin whole-turn runtimes."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from agent.runtime_api import (
    AgentRuntime,
    CompactionOwnership,
    RuntimeApprovalRequestEvent,
    RuntimeCancelledEvent,
    RuntimeCompactionEvent,
    RuntimeCompletedEvent,
    RuntimeContentEvent,
    RuntimeEvent,
    RuntimeFailedEvent,
    RuntimeFailure,
    RuntimeFailurePhase,
    RuntimeHostServices,
    RuntimeDescriptor,
    RuntimeRegistration,
    RuntimeSelection,
    RuntimeStateEvent,
    RuntimeStateEnvelope,
    RuntimeStatusEvent,
    RuntimeToolRequestEvent,
    RuntimeTurnRequest,
    RuntimeUsageEvent,
    RuntimeUsageReceipt,
)


_RUNTIME_EVENT_TYPES = (
    RuntimeContentEvent,
    RuntimeStatusEvent,
    RuntimeToolRequestEvent,
    RuntimeApprovalRequestEvent,
    RuntimeCompactionEvent,
    RuntimeStateEvent,
    RuntimeUsageEvent,
    RuntimeCompletedEvent,
    RuntimeCancelledEvent,
    RuntimeFailedEvent,
)

_RUNTIME_SIDE_EFFECT_EVENT_TYPES = (
    RuntimeCompactionEvent,
    RuntimeStateEvent,
    RuntimeUsageEvent,
)

_RUNTIME_VISIBLE_EVENT_TYPES = (
    RuntimeContentEvent,
    RuntimeStatusEvent,
    RuntimeToolRequestEvent,
    RuntimeApprovalRequestEvent,
)


class RuntimeExecutionError(RuntimeError):
    """A runtime failed its preflight or terminal event contract."""

    def __init__(self, message: str, *, failure: RuntimeFailure | None = None):
        super().__init__(message)
        self.failure = failure


@dataclass(frozen=True)
class RuntimeDispatchResult:
    response: Mapping[str, Any]
    events: tuple[RuntimeEvent, ...]
    terminal: RuntimeCompletedEvent | RuntimeCancelledEvent | RuntimeFailedEvent | None = None
    failure: RuntimeFailure | None = None
    cancelled: bool = False

    @property
    def completed(self) -> bool:
        """Whether the runtime produced a successful terminal event."""
        return isinstance(self.terminal, RuntimeCompletedEvent)

    @property
    def replay_safe(self) -> bool:
        """Expose the runtime's explicit replay classification to the host."""
        return bool(self.failure is not None and self.failure.replay_safe)


def _unclassified_failure_phase(events: Sequence[RuntimeEvent]) -> RuntimeFailurePhase:
    """Return the most conservative phase proved by already-emitted events."""

    if any(isinstance(event, _RUNTIME_SIDE_EFFECT_EVENT_TYPES) for event in events):
        return RuntimeFailurePhase.AFTER_SIDE_EFFECTS
    if any(isinstance(event, _RUNTIME_VISIBLE_EVENT_TYPES) for event in events):
        return RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT
    return RuntimeFailurePhase.BEFORE_VISIBLE_OUTPUT


def _freeze_value(value: Any) -> Any:
    """Deep-copy host-owned input into immutable public-contract values."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in copy.deepcopy(dict(value)).items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in copy.deepcopy(value))
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in copy.deepcopy(value))
    return copy.deepcopy(value)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy host-owned turn input before exposing it to a plugin."""
    return _freeze_value(value)


def build_runtime_turn_request(
    *,
    provider: str,
    model: str,
    api_mode: str,
    messages: Sequence[Mapping[str, Any]],
    prompt_snapshot: str,
    tool_schemas: Sequence[Mapping[str, Any]],
    session_state: RuntimeStateEnvelope | None = None,
    attachments: Sequence[Mapping[str, Any]] = (),
    correlation_id: str | None = None,
) -> RuntimeTurnRequest:
    canonical_tool_schemas = json.dumps(
        list(tool_schemas),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    frozen_session_state = None
    if session_state is not None:
        frozen_session_state = RuntimeStateEnvelope(
            runtime_id=str(session_state.runtime_id),
            schema_version=int(session_state.schema_version),
            state=_freeze_mapping(session_state.state),
        )
    return RuntimeTurnRequest(
        selection=RuntimeSelection(
            provider=provider,
            model=model,
            api_mode=api_mode,
        ),
        messages=tuple(_freeze_mapping(item) for item in messages),
        prompt_snapshot=str(prompt_snapshot),
        tool_schemas=tuple(_freeze_mapping(item) for item in tool_schemas),
        tool_schema_hash=hashlib.sha256(canonical_tool_schemas).hexdigest(),
        session_state=frozen_session_state,
        attachments=tuple(_freeze_mapping(item) for item in attachments),
        correlation_id=correlation_id,
    )


async def _collect_runtime_turn(
    runtime: AgentRuntime,
    request: RuntimeTurnRequest,
    host: RuntimeHostServices,
    descriptor: RuntimeDescriptor | None = None,
) -> RuntimeDispatchResult:
    events: list[RuntimeEvent] = []
    terminal: RuntimeCompletedEvent | RuntimeCancelledEvent | RuntimeFailedEvent | None = None
    try:
        failure = runtime.preflight(request)
        if failure is not None:
            terminal = RuntimeFailedEvent(failure=failure)
            events.append(terminal)
            return RuntimeDispatchResult(
                response={},
                events=tuple(events),
                terminal=terminal,
                failure=failure,
            )

        async for event in runtime.run_turn(request, host):
            if not isinstance(event, _RUNTIME_EVENT_TYPES):
                raise RuntimeExecutionError(
                    f"runtime emitted unsupported event type: {type(event).__name__}"
                )
            if terminal is not None:
                raise RuntimeExecutionError(
                    "runtime emitted an event after its terminal event"
                )
            events.append(event)
            if isinstance(event, RuntimeStatusEvent):
                await host.emit_status(event.message)
            elif isinstance(event, RuntimeCompactionEvent):
                # Compaction is an observable lifecycle event, not a signal to
                # invoke the host compressor. Runtime-native implementations
                # own the actual operation; the host only projects the event.
                if (
                    descriptor is not None
                    and descriptor.compaction_ownership
                    is not CompactionOwnership.RUNTIME_NATIVE
                ):
                    raise RuntimeExecutionError(
                        "runtime emitted compaction event while host owns compaction"
                    )
                projector = getattr(host, "emit_compaction", None)
                if callable(projector):
                    await projector(event)
            elif isinstance(event, RuntimeStateEvent):
                await host.persist_state(event.state)
            elif isinstance(event, RuntimeUsageEvent):
                await host.persist_usage(event.receipt)
            if isinstance(
                event,
                (RuntimeCompletedEvent, RuntimeCancelledEvent, RuntimeFailedEvent),
            ):
                terminal = event

        if terminal is None:
            raise RuntimeExecutionError("runtime ended without a terminal event")
        if isinstance(terminal, RuntimeFailedEvent):
            return RuntimeDispatchResult(
                response={},
                events=tuple(events),
                terminal=terminal,
                failure=terminal.failure,
            )
        if isinstance(terminal, RuntimeCancelledEvent):
            return RuntimeDispatchResult(
                response={},
                events=tuple(events),
                terminal=terminal,
                cancelled=True,
            )
        return RuntimeDispatchResult(
            response=terminal.result or {},
            events=tuple(events),
            terminal=terminal,
        )
    except asyncio.CancelledError:
        # A task cancellation is a terminal runtime outcome. It never implies
        # replay safety: no provider/runtime exception is used to authorize a
        # host fallback.
        terminal = RuntimeCancelledEvent(reason="runtime task cancelled")
        events.append(terminal)
        return RuntimeDispatchResult(
            response={},
            events=tuple(events),
            terminal=terminal,
            cancelled=True,
        )
    except RuntimeExecutionError:
        # Contract violations (unknown event, duplicate terminal, or missing
        # terminal) remain hard errors. They are not classified as replayable
        # runtime failures and therefore cannot silently enter fallback.
        raise
    except Exception:
        # An unclassified runtime exception is fail-closed. Do not infer
        # replay safety from its type or message; expose only a bounded,
        # conservative result for host policy.
        failure = RuntimeFailure(
            code="runtime_exception",
            message="runtime execution failed",
            phase=_unclassified_failure_phase(events),
            replay_safe=False,
            retryable=False,
        )
        terminal = RuntimeFailedEvent(failure=failure)
        events.append(terminal)
        return RuntimeDispatchResult(
            response={},
            events=tuple(events),
            terminal=terminal,
            failure=failure,
        )
    finally:
        # The collector is the lifecycle owner: one dispatch means exactly one
        # close attempt, regardless of preflight, cancellation, failure, or a
        # malformed event stream. A close error must not erase the classified
        # turn outcome already returned above.
        try:
            await runtime.close()
        except asyncio.CancelledError:
            # Cancellation of cleanup itself must not erase the turn's
            # already-classified terminal outcome.
            pass
        except Exception:
            # Closing is best-effort at this boundary. Runtime implementations
            # must remain disposable even when their transport has already
            # failed, and a cleanup exception must not trigger a second close.
            pass


def run_runtime_sync(
    runtime: AgentRuntime,
    request: RuntimeTurnRequest,
    host: RuntimeHostServices,
    *,
    descriptor: RuntimeDescriptor | None = None,
) -> RuntimeDispatchResult:
    """Run the async contract from Hermes' existing synchronous turn loop."""
    from model_tools import _run_async

    return _run_async(_collect_runtime_turn(runtime, request, host, descriptor))


class BuiltInCodexRuntime:
    """Codex whole-turn consumer of AgentRuntime v1.

    The callback is host-owned and captures the legacy Codex adapter while it
    is incrementally migrated.  Third-party runtimes never receive this
    callback or the private AIAgent object it closes over.
    """

    def __init__(self, runner: Callable[[], Mapping[str, Any]]):
        self._runner = runner

    def preflight(self, request: RuntimeTurnRequest) -> RuntimeFailure | None:
        return None

    async def run_turn(self, request, host):
        if host.cancellation_requested():
            yield RuntimeCancelledEvent(reason="cancelled before runtime start")
            return
        yield RuntimeCompletedEvent(result=self._runner())

    async def close(self) -> None:
        return None


def make_builtin_codex_registration(
    runner: Callable[[], Mapping[str, Any]],
) -> RuntimeRegistration:
    """Return the host-owned Codex consumer for the shared runtime resolver."""
    return RuntimeRegistration(
        descriptor=RuntimeDescriptor(
            runtime_id="hermes-codex-app-server",
            plugin_version="builtin",
            runtime_api_min=1,
            runtime_api_max=1,
            required_host_capabilities=frozenset({"cancellation_v1"}),
            provider_ids=frozenset(),
            api_modes=frozenset({"codex_app_server"}),
            session_state_schema_version=1,
            compaction_ownership=CompactionOwnership.RUNTIME_NATIVE,
        ),
        factory=lambda: BuiltInCodexRuntime(runner),
        plugin_id="hermes-core",
    )


class HermesRuntimeHostServices:
    """The only stateful Hermes surface available to runtime plugins."""

    def __init__(self, agent: Any, *, task_id: str, runtime_id: str):
        self._agent = agent
        self._task_id = str(task_id)
        self._runtime_id = str(runtime_id)
        self._tool_call_count = 0
        self._compaction_events: list[dict[str, Any]] = []
        try:
            # A fresh host is created for each whole turn, so the lifecycle
            # projection cannot accidentally bleed into a later turn.
            self._agent._runtime_compaction_events = self._compaction_events
        except Exception:
            pass
        allowed = set(getattr(agent, "valid_tool_names", ()) or ())
        for schema in getattr(agent, "tools", ()) or ():
            if not isinstance(schema, Mapping):
                continue
            function = schema.get("function")
            if isinstance(function, Mapping) and function.get("name"):
                allowed.add(str(function["name"]))
        self._allowed_tool_names = frozenset(allowed)

    async def execute_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        """Execute one runtime-requested tool through Hermes' canonical funnel.

        The synthetic assistant/tool-call objects are host-private adapters for
        the existing single-call executor.  That executor owns scope checks,
        plugin middleware and approval, guardrails, progress, persistence, and
        terminal result normalization.  The plugin receives only the canonical
        tool result content.
        """
        normalized_name = str(name or "").strip()
        if not normalized_name or normalized_name not in self._allowed_tool_names:
            raise RuntimeExecutionError(
                f"tool '{normalized_name or '<empty>'}' is not available in this session"
            )
        if not isinstance(arguments, Mapping):
            raise RuntimeExecutionError("runtime tool arguments must be a mapping")

        executor = getattr(self._agent, "_execute_tool_calls", None)
        if not callable(executor):
            raise RuntimeExecutionError("Hermes tool executor is unavailable")

        self._tool_call_count += 1
        tool_call_id = f"runtime-tool-{self._tool_call_count:04d}"
        tool_call = SimpleNamespace(
            id=tool_call_id,
            type="function",
            function=SimpleNamespace(
                name=normalized_name,
                arguments=json.dumps(dict(arguments), ensure_ascii=False),
            ),
        )
        assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
        tool_messages: list[Mapping[str, Any]] = []
        executor(assistant_message, tool_messages, self._task_id)

        matches = [
            message
            for message in tool_messages
            if message.get("role") == "tool"
            and message.get("tool_call_id") == tool_call_id
        ]
        if len(matches) != 1:
            raise RuntimeExecutionError(
                "Hermes tool executor did not produce exactly one canonical result"
            )

        return matches[0].get("content")

    async def request_approval(
        self,
        action: str,
        details: Mapping[str, Any],
    ) -> bool:
        from tools.approval import request_tool_approval

        try:
            from tools.terminal_tool import _get_approval_callback

            callback = _get_approval_callback()
        except Exception:
            callback = None
        decision = request_tool_approval(
            action,
            str(details.get("reason") or f"Runtime requested approval for {action}"),
            rule_key=str(details.get("rule_key") or ""),
            approval_callback=callback,
        )
        return bool(decision.get("approved"))

    async def emit_status(self, message: str) -> None:
        touch = getattr(self._agent, "_touch_activity", None)
        if callable(touch):
            touch(message)

    async def persist_state(self, state: RuntimeStateEnvelope) -> None:
        if state.runtime_id != self._runtime_id:
            raise RuntimeExecutionError(
                "runtime state identity does not match the selected runtime"
            )
        database = getattr(self._agent, "_session_db", None)
        session_id = getattr(self._agent, "session_id", None)
        if database is None or not session_id:
            raise RuntimeExecutionError(
                "runtime state persistence requires an active Hermes session"
            )
        database.update_runtime_state(session_id, state)

    async def persist_usage(self, receipt: RuntimeUsageReceipt) -> None:
        if receipt.runtime_id != self._runtime_id:
            raise RuntimeExecutionError(
                "runtime usage identity does not match the selected runtime"
            )
        database = getattr(self._agent, "_session_db", None)
        session_id = getattr(self._agent, "session_id", None)
        if database is None or not session_id:
            raise RuntimeExecutionError(
                "runtime usage persistence requires an active Hermes session"
            )
        inserted = database.record_runtime_usage_receipt(session_id, receipt)
        if not inserted:
            return
        database.queue_token_counts(
            session_id,
            input_tokens=receipt.input_tokens,
            output_tokens=receipt.output_tokens,
            cache_read_tokens=receipt.cache_read_tokens,
            cache_write_tokens=receipt.cache_write_tokens,
            reasoning_tokens=receipt.reasoning_tokens,
            billing_provider=receipt.provider,
            billing_mode=receipt.billing_mode,
            cost_status=receipt.cost_status,
            model=receipt.model,
            api_call_count=1,
        )

    async def emit_compaction(self, event: RuntimeCompactionEvent) -> None:
        """Project runtime-native compaction into the host lifecycle stream.

        The event remains runtime-owned: this method never invokes Hermes'
        compressor.  Only its typed phase and a bounded set of scalar details
        are retained on the agent for lifecycle observers; arbitrary runtime
        payloads are intentionally not persisted or surfaced.
        """
        if not isinstance(event, RuntimeCompactionEvent):
            raise RuntimeExecutionError("runtime compaction event has an unsupported type")
        details = {
            key: value
            for key, value in event.details.items()
            if key in {"watchdog_seconds"}
            and isinstance(value, (str, int, float, bool))
        }
        record = {
            "runtime_id": self._runtime_id,
            "phase": event.phase.value,
            "details": details,
        }
        self._compaction_events.append(record)

        # Keep existing host status delivery as the lifecycle projection. The
        # message is derived solely from the typed phase and carries no runtime
        # details or provider-specific policy.
        await self.emit_status(f"Runtime compaction {event.phase.value}")

    def cancellation_requested(self) -> bool:
        return bool(getattr(self._agent, "_interrupt_requested", False))
