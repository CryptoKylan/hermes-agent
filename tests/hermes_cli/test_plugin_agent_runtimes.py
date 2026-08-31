"""Public behavior tests for whole-turn runtime plugin registration."""

from __future__ import annotations

import pytest

from agent.runtime_api import (
    HOST_RUNTIME_CAPABILITIES,
    RUNTIME_API_VERSION,
    RuntimeCompatibilityError,
    RuntimeDescriptor,
    RuntimeSelection,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _make_context(name: str = "runtime-plugin") -> tuple[PluginContext, PluginManager]:
    manager = PluginManager()
    manager._discovered = True
    context = PluginContext(PluginManifest(name=name), manager)
    return context, manager


def _descriptor(**overrides) -> RuntimeDescriptor:
    values = {
        "runtime_id": "example-runtime",
        "plugin_version": "0.1.0",
        "runtime_api_min": RUNTIME_API_VERSION,
        "runtime_api_max": RUNTIME_API_VERSION,
        "required_host_capabilities": frozenset({"host_tool_execution"}),
        "provider_ids": frozenset({"example"}),
        "api_modes": frozenset({"example_runtime"}),
        "session_state_schema_version": 1,
    }
    values.update(overrides)
    return RuntimeDescriptor(**values)


def test_incompatible_api_is_rejected_before_factory_runs():
    context, manager = _make_context()
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return object()

    descriptor = _descriptor(
        runtime_api_min=RUNTIME_API_VERSION + 1,
        runtime_api_max=RUNTIME_API_VERSION + 1,
    )

    with pytest.raises(RuntimeCompatibilityError, match="runtime API"):
        context.register_agent_runtime(descriptor=descriptor, factory=factory)

    assert factory_calls == 0
    assert manager.get_agent_runtime("example-runtime") is None


def test_missing_host_capability_is_rejected_before_factory_runs():
    context, manager = _make_context()
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return object()

    missing = "capability_that_this_host_does_not_export"
    assert missing not in HOST_RUNTIME_CAPABILITIES

    with pytest.raises(RuntimeCompatibilityError, match=missing):
        context.register_agent_runtime(
            descriptor=_descriptor(
                required_host_capabilities=frozenset({missing}),
            ),
            factory=factory,
        )

    assert factory_calls == 0
    assert manager.get_agent_runtime("example-runtime") is None


def test_compatible_runtime_is_selected_without_instantiating_it():
    context, manager = _make_context()
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return object()

    descriptor = _descriptor()
    context.register_agent_runtime(descriptor=descriptor, factory=factory)

    registration = manager.select_agent_runtime(
        RuntimeSelection(
            provider="example",
            model="example-large",
            api_mode="example_runtime",
        )
    )

    assert registration is not None
    assert registration.descriptor == descriptor
    assert registration.plugin_id == "runtime-plugin"
    assert factory_calls == 0


def test_runtime_registration_is_removed_when_plugin_unloads():
    context, manager = _make_context()
    context.register_agent_runtime(descriptor=_descriptor(), factory=object)

    assert manager.get_agent_runtime("example-runtime") is not None
    assert manager.unload("runtime-plugin") is True
    assert manager.get_agent_runtime("example-runtime") is None

