"""External whole-turn runtime integration through the real AIAgent seam."""

from __future__ import annotations

import run_agent

from agent.runtime_api import (
    RUNTIME_API_VERSION,
    RuntimeCompletedEvent,
    RuntimeDescriptor,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


class _ExternalRuntime:
    def __init__(self, counters):
        self._counters = counters

    def preflight(self, request):
        self._counters["preflight"] += 1
        return None

    async def run_turn(self, request, host):
        self._counters["turn"] += 1
        self._counters["prompt_snapshot"] = request.prompt_snapshot
        yield RuntimeCompletedEvent(
            result={
                "final_response": "external runtime reply",
                "messages": list(request.messages),
                "completed": True,
                "partial": False,
                "error": None,
            }
        )

    async def close(self):
        self._counters["close"] += 1


def test_external_plugin_runtime_is_selected_before_the_ordinary_model_loop(
    monkeypatch,
):
    manager = PluginManager()
    manager._discovered = True
    context = PluginContext(PluginManifest(name="external-runtime"), manager)
    counters = {
        "factory": 0,
        "preflight": 0,
        "turn": 0,
        "close": 0,
        "prompt_snapshot": None,
    }

    def factory():
        counters["factory"] += 1
        return _ExternalRuntime(counters)

    context.register_agent_runtime(
        descriptor=RuntimeDescriptor(
            runtime_id="external-test-runtime",
            plugin_version="0.1.0",
            runtime_api_min=RUNTIME_API_VERSION,
            runtime_api_max=RUNTIME_API_VERSION,
            required_host_capabilities=frozenset({"cancellation_v1"}),
            provider_ids=frozenset({"openai"}),
            api_modes=frozenset({"chat_completions"}),
            session_state_schema_version=1,
        ),
        factory=factory,
    )

    import hermes_cli.plugins as plugins_module

    monkeypatch.setattr(plugins_module, "_plugin_manager", manager)
    agent = run_agent.AIAgent(
        api_key="synthetic-test-value",
        base_url="https://test.invalid",
        provider="openai",
        model="synthetic-model",
        api_mode="chat_completions",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._cached_system_prompt = "composed synthetic prompt"

    result = agent.run_conversation("hello")

    assert result["final_response"] == "external runtime reply"
    assert counters == {
        "factory": 1,
        "preflight": 1,
        "turn": 1,
        "close": 1,
        "prompt_snapshot": "composed synthetic prompt",
    }
