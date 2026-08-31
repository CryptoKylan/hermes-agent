"""Public behavior tests for whole-turn runtime plugin registration."""

from __future__ import annotations

import pytest

from agent.runtime_api import (
    HOST_RUNTIME_CAPABILITIES,
    RUNTIME_API_VERSION,
    RuntimeCompatibilityError,
    RuntimeDescriptor,
    RuntimeRegistration,
    RuntimeRegistrationError,
    RuntimeSelection,
    resolve_runtime_registration,
    runtime_api_manifest,
)
from agent.runtime_dispatch import (
    build_runtime_turn_request,
    make_builtin_codex_registration,
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
        "required_host_capabilities": frozenset({"host_tool_execution_v1"}),
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


def test_provider_profile_registration_is_removed_when_plugin_unloads():
    from providers import get_provider_profile, register_provider
    from providers.base import ProviderProfile

    context, manager = _make_context()
    profile = ProviderProfile(
        name="example-runtime-provider",
        aliases=("example-runtime-alias",),
        display_name="Example runtime",
        description="Whole-turn runtime test profile",
        api_mode="agent_runtime",
        auth_type="oauth_external",
    )

    # Match module-shaped entry-point discovery: import-time registration of
    # the same object is adopted by register(ctx), then owned by the ledger.
    register_provider(profile)
    context.register_provider_profile(profile)
    assert get_provider_profile(profile.name) is profile
    assert get_provider_profile("example-runtime-alias") is profile

    assert manager.unload("runtime-plugin") is True
    assert get_provider_profile(profile.name) is None
    assert get_provider_profile("example-runtime-alias") is None


def test_host_manifest_exports_only_versioned_concrete_capabilities():
    manifest = runtime_api_manifest()

    assert manifest["runtime_api_version"] == RUNTIME_API_VERSION
    assert manifest["host_capabilities"] == sorted(HOST_RUNTIME_CAPABILITIES)
    assert "host_tool_execution_v1" in manifest["host_capabilities"]
    assert "provider_profile_registration_v1" in manifest["host_capabilities"]
    assert all(capability.endswith("_v1") for capability in manifest["host_capabilities"])


def test_runtime_turn_request_is_deeply_immutable():
    source_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        }
    ]
    source_tools = [
        {
            "name": "pwd",
            "parameters": {"type": "object", "required": []},
        }
    ]

    request = build_runtime_turn_request(
        provider="example",
        model="example-large",
        api_mode="example_runtime",
        messages=source_messages,
        prompt_snapshot="stable prompt",
        tool_schemas=source_tools,
    )

    source_messages[0]["content"][0]["text"] = "mutated outside"
    source_tools[0]["parameters"]["required"].append("path")

    assert request.messages[0]["content"][0]["text"] == "hello"
    assert request.tool_schemas[0]["parameters"]["required"] == ()
    same_schema = build_runtime_turn_request(
        provider="example",
        model="example-large",
        api_mode="example_runtime",
        messages=(),
        prompt_snapshot="stable prompt",
        tool_schemas=(
            {
                "parameters": {"required": [], "type": "object"},
                "name": "pwd",
            },
        ),
    )
    assert request.tool_schema_hash == same_schema.tool_schema_hash
    assert len(request.tool_schema_hash) == 64
    with pytest.raises(TypeError):
        request.messages[0]["content"][0]["text"] = "mutated inside"
    with pytest.raises(AttributeError):
        request.tool_schemas[0]["parameters"]["required"].append("path")


def test_builtin_and_plugin_registrations_use_one_resolver():
    builtin = RuntimeRegistration(
        descriptor=_descriptor(
            runtime_id="hermes-codex",
            provider_ids=frozenset(),
            api_modes=frozenset({"codex_app_server"}),
        ),
        factory=object,
        plugin_id="hermes-core",
    )
    plugin = RuntimeRegistration(
        descriptor=_descriptor(),
        factory=object,
        plugin_id="runtime-plugin",
    )

    assert resolve_runtime_registration(
        RuntimeSelection(
            provider="codex",
            model="gpt-5.6-sol",
            api_mode="codex_app_server",
        ),
        (builtin, plugin),
    ) is builtin
    assert resolve_runtime_registration(
        RuntimeSelection(
            provider="example",
            model="example-large",
            api_mode="example_runtime",
        ),
        (builtin, plugin),
    ) is plugin


def test_shared_resolver_rejects_ambiguous_runtime_selection():
    first = RuntimeRegistration(
        descriptor=_descriptor(runtime_id="first-runtime"),
        factory=object,
        plugin_id="first-plugin",
    )
    second = RuntimeRegistration(
        descriptor=_descriptor(runtime_id="second-runtime"),
        factory=object,
        plugin_id="second-plugin",
    )

    with pytest.raises(RuntimeRegistrationError, match="multiple runtimes"):
        resolve_runtime_registration(
            RuntimeSelection(
                provider="example",
                model="example-large",
                api_mode="example_runtime",
            ),
            (first, second),
        )


def test_builtin_codex_registration_is_resolved_with_plugin_registrations():
    context, manager = _make_context()
    context.register_agent_runtime(descriptor=_descriptor(), factory=object)
    builtin = make_builtin_codex_registration(lambda: {"final_response": "done"})

    registration = resolve_runtime_registration(
        RuntimeSelection(
            provider="codex",
            model="gpt-5.6-sol",
            api_mode="codex_app_server",
        ),
        (builtin, *manager.iter_agent_runtime_registrations()),
    )

    assert registration is builtin
    assert registration.plugin_id == "hermes-core"
    assert registration.descriptor.required_host_capabilities == frozenset(
        {"cancellation_v1"}
    )
