"""Versioned public contract for independently packaged whole-turn runtimes.

The types in this module are intentionally provider-neutral and dependency
free.  Runtime plugins receive immutable request data plus a host-services
facade; they never receive an :class:`agent.model.AIAgent` or another private
host object.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    FrozenSet,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
    runtime_checkable,
)


RUNTIME_API_VERSION = 1

# Capabilities are concrete host promises, not provider feature marketing.
# A plugin must declare every service that it needs at registration time.
HOST_RUNTIME_CAPABILITIES: FrozenSet[str] = frozenset(
    {
        "cancellation_v1",
        "compaction_events_v1",
        "host_approval_v1",
        "host_status_v1",
        "host_tool_execution_v1",
        "runtime_state_v1",
        "usage_receipts_v1",
    }
)

_RUNTIME_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class RuntimeCompatibilityError(RuntimeError):
    """A runtime cannot safely register against this host contract."""


class RuntimeRegistrationError(RuntimeError):
    """A runtime registration conflicts with an existing registration."""


class CompactionOwnership(str, Enum):
    HOST = "host"
    RUNTIME_NATIVE = "runtime_native"


class RuntimeCompactionPhase(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    WATCHDOG = "watchdog"


class RuntimeFailurePhase(str, Enum):
    PREFLIGHT = "preflight"
    BEFORE_VISIBLE_OUTPUT = "before_visible_output"
    AFTER_VISIBLE_OUTPUT = "after_visible_output"
    AFTER_SIDE_EFFECTS = "after_side_effects"


class RuntimeEventKind(str, Enum):
    CONTENT = "content"
    STATUS = "status"
    TOOL_REQUEST = "tool_request"
    APPROVAL_REQUEST = "approval_request"
    SESSION_STATE = "session_state"
    COMPACTION = "compaction"
    USAGE = "usage"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class RuntimeSelection:
    """Normalized provider/model selection used before runtime creation."""

    provider: str
    model: str
    api_mode: str


@dataclass(frozen=True)
class RuntimeDescriptor:
    """Machine-readable compatibility and routing handshake."""

    runtime_id: str
    plugin_version: str
    runtime_api_min: int
    runtime_api_max: int
    required_host_capabilities: FrozenSet[str]
    provider_ids: FrozenSet[str]
    api_modes: FrozenSet[str]
    session_state_schema_version: int
    model_prefixes: tuple[str, ...] = ()
    compaction_ownership: CompactionOwnership = CompactionOwnership.HOST
    feature_flags: FrozenSet[str] = field(default_factory=frozenset)

    def supports(self, selection: RuntimeSelection) -> bool:
        """Return the descriptor-only routing decision without side effects."""
        if self.provider_ids and selection.provider not in self.provider_ids:
            return False
        if self.api_modes and selection.api_mode not in self.api_modes:
            return False
        if self.model_prefixes and not any(
            selection.model.startswith(prefix) for prefix in self.model_prefixes
        ):
            return False
        return True

    def to_manifest(self) -> dict[str, Any]:
        """Return stable JSON-compatible handshake metadata."""
        return {
            "runtime_id": self.runtime_id,
            "plugin_version": self.plugin_version,
            "runtime_api_min": self.runtime_api_min,
            "runtime_api_max": self.runtime_api_max,
            "required_host_capabilities": sorted(self.required_host_capabilities),
            "provider_ids": sorted(self.provider_ids),
            "api_modes": sorted(self.api_modes),
            "model_prefixes": list(self.model_prefixes),
            "session_state_schema_version": self.session_state_schema_version,
            "compaction_ownership": self.compaction_ownership.value,
            "feature_flags": sorted(self.feature_flags),
        }


@dataclass(frozen=True)
class RuntimeStateEnvelope:
    runtime_id: str
    schema_version: int
    state: Mapping[str, Any]


@dataclass(frozen=True)
class RuntimeUsageReceipt:
    runtime_id: str
    provider: str
    model: str
    billing_mode: str
    cost_status: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    replay_safe: bool = False
    correlation_id: str | None = None


@dataclass(frozen=True)
class RuntimeFailure:
    code: str
    message: str
    phase: RuntimeFailurePhase
    replay_safe: bool
    retryable: bool = False


@dataclass(frozen=True)
class RuntimeTurnRequest:
    """Immutable normalized input for one whole turn."""

    selection: RuntimeSelection
    messages: Sequence[Mapping[str, Any]]
    prompt_snapshot: str
    tool_schemas: Sequence[Mapping[str, Any]]
    tool_schema_hash: str
    session_state: RuntimeStateEnvelope | None = None
    attachments: Sequence[Mapping[str, Any]] = ()
    correlation_id: str | None = None


@dataclass(frozen=True)
class RuntimeContentEvent:
    kind: RuntimeEventKind = field(default=RuntimeEventKind.CONTENT, init=False)
    text: str = ""


@dataclass(frozen=True)
class RuntimeStatusEvent:
    message: str
    kind: RuntimeEventKind = field(default=RuntimeEventKind.STATUS, init=False)


@dataclass(frozen=True)
class RuntimeToolRequestEvent:
    request_id: str
    name: str
    arguments: Mapping[str, Any]
    kind: RuntimeEventKind = field(default=RuntimeEventKind.TOOL_REQUEST, init=False)


@dataclass(frozen=True)
class RuntimeApprovalRequestEvent:
    request_id: str
    action: str
    details: Mapping[str, Any]
    kind: RuntimeEventKind = field(
        default=RuntimeEventKind.APPROVAL_REQUEST,
        init=False,
    )


@dataclass(frozen=True)
class RuntimeCompactionEvent:
    phase: RuntimeCompactionPhase
    details: Mapping[str, Any] = field(default_factory=dict)
    kind: RuntimeEventKind = field(default=RuntimeEventKind.COMPACTION, init=False)


@dataclass(frozen=True)
class RuntimeStateEvent:
    state: RuntimeStateEnvelope
    kind: RuntimeEventKind = field(default=RuntimeEventKind.SESSION_STATE, init=False)


@dataclass(frozen=True)
class RuntimeUsageEvent:
    receipt: RuntimeUsageReceipt
    kind: RuntimeEventKind = field(default=RuntimeEventKind.USAGE, init=False)


@dataclass(frozen=True)
class RuntimeCompletedEvent:
    result: Mapping[str, Any] | None = None
    kind: RuntimeEventKind = field(default=RuntimeEventKind.COMPLETED, init=False)


@dataclass(frozen=True)
class RuntimeCancelledEvent:
    reason: str = "cancelled"
    kind: RuntimeEventKind = field(default=RuntimeEventKind.CANCELLED, init=False)


@dataclass(frozen=True)
class RuntimeFailedEvent:
    failure: RuntimeFailure
    kind: RuntimeEventKind = field(default=RuntimeEventKind.FAILED, init=False)


RuntimeEvent: TypeAlias = (
    RuntimeContentEvent
    | RuntimeStatusEvent
    | RuntimeToolRequestEvent
    | RuntimeApprovalRequestEvent
    | RuntimeCompactionEvent
    | RuntimeStateEvent
    | RuntimeUsageEvent
    | RuntimeCompletedEvent
    | RuntimeCancelledEvent
    | RuntimeFailedEvent
)


@runtime_checkable
class RuntimeHostServices(Protocol):
    """Stable host-owned security, state, status, and lifecycle boundary."""

    async def execute_tool(self, name: str, arguments: Mapping[str, Any]) -> Any: ...

    async def request_approval(
        self, action: str, details: Mapping[str, Any]
    ) -> bool: ...

    async def emit_status(self, message: str) -> None: ...

    async def persist_state(self, state: RuntimeStateEnvelope) -> None: ...

    async def persist_usage(self, receipt: RuntimeUsageReceipt) -> None: ...

    def cancellation_requested(self) -> bool: ...


@runtime_checkable
class AgentRuntime(Protocol):
    """Whole-turn runtime implemented by built-ins or third-party plugins."""

    def preflight(self, request: RuntimeTurnRequest) -> RuntimeFailure | None: ...

    def run_turn(
        self,
        request: RuntimeTurnRequest,
        host: RuntimeHostServices,
    ) -> AsyncIterator[RuntimeEvent]: ...

    async def close(self) -> None: ...


RuntimeFactory: TypeAlias = Callable[[], AgentRuntime]


@dataclass(frozen=True)
class RuntimeRegistration:
    descriptor: RuntimeDescriptor
    factory: RuntimeFactory
    plugin_id: str


def resolve_runtime_registration(
    selection: RuntimeSelection,
    registrations: Sequence[RuntimeRegistration],
) -> RuntimeRegistration | None:
    """Resolve built-in and plugin runtimes through one pure routing path."""
    matches = [
        registration
        for registration in registrations
        if registration.descriptor.supports(selection)
    ]
    if not matches:
        return None
    if len(matches) > 1:
        runtime_ids = sorted(item.descriptor.runtime_id for item in matches)
        raise RuntimeRegistrationError(
            "multiple runtimes support the same selection: "
            + ", ".join(runtime_ids)
        )
    return matches[0]


def validate_runtime_descriptor(descriptor: RuntimeDescriptor) -> None:
    """Fail closed before a runtime factory, credential, or SDK is touched."""
    if not isinstance(descriptor, RuntimeDescriptor):
        raise RuntimeCompatibilityError("runtime descriptor has an unsupported type")
    if not _RUNTIME_ID_RE.fullmatch(descriptor.runtime_id):
        raise RuntimeCompatibilityError(
            "runtime_id must use lowercase letters, digits, '.', '_' or '-'"
        )
    if not descriptor.plugin_version:
        raise RuntimeCompatibilityError("plugin_version must not be empty")
    if (
        descriptor.runtime_api_min < 1
        or descriptor.runtime_api_max < descriptor.runtime_api_min
    ):
        raise RuntimeCompatibilityError("runtime API range is invalid")
    if not (
        descriptor.runtime_api_min
        <= RUNTIME_API_VERSION
        <= descriptor.runtime_api_max
    ):
        raise RuntimeCompatibilityError(
            f"runtime API {descriptor.runtime_api_min}..{descriptor.runtime_api_max} "
            f"is incompatible with host API {RUNTIME_API_VERSION}"
        )
    missing = descriptor.required_host_capabilities - HOST_RUNTIME_CAPABILITIES
    if missing:
        raise RuntimeCompatibilityError(
            "runtime requires unsupported host capabilities: "
            + ", ".join(sorted(missing))
        )
    if descriptor.session_state_schema_version < 1:
        raise RuntimeCompatibilityError(
            "session_state_schema_version must be at least 1"
        )
    if not descriptor.provider_ids and not descriptor.api_modes:
        raise RuntimeCompatibilityError(
            "runtime must declare at least one provider_id or api_mode"
        )


def runtime_api_manifest() -> dict[str, Any]:
    """Return the host half of the compatibility handshake."""
    return {
        "runtime_api_version": RUNTIME_API_VERSION,
        "host_capabilities": sorted(HOST_RUNTIME_CAPABILITIES),
        "event_kinds": [kind.value for kind in RuntimeEventKind],
        "failure_phases": [phase.value for phase in RuntimeFailurePhase],
        "compaction_phases": [phase.value for phase in RuntimeCompactionPhase],
    }
