# ADR: AgentRuntime v1 host boundary

Status: accepted

AgentRuntime v1 is a provider-neutral whole-turn interface. Plugins receive an
immutable turn request and a bounded `RuntimeHostServices` facade. The host
continues to own tools, approval, persistence, usage, cancellation, routing,
delivery, and lifecycle. A plugin never receives an `AIAgent`, Hermes session
identifier, gateway route, credential, or provider-specific core policy.

## Background delivery

`background_delivery_v1` adds one public value and one host operation:

- `RuntimeBackgroundResult` is frozen and contains only normalized UTF-8 text
  bounded to 16 KiB plus a `completed` or `failed` outcome.
- `RuntimeHostServices.emit_background_result(result)` accepts that value after
  a turn terminal event. Background results are outside the `run_turn` event
  stream, so every turn still emits exactly one terminal event.

The Hermes host binding captures the exact parent session and live return route.
It translates a background result onto the existing completion consumer. That
consumer owns idle wake-up, busy-session requeue, exact-parent preflight,
transcript re-entry, outbound adapter delivery, and retry. No runtime plugin
performs a latest-session lookup or supplies routing metadata.

## Lifecycle

One runtime instance and one host-services binding are cached on the parent
`AIAgent` and reused across its turns. Turn correlation and the live route are
refreshed only while the same parent session remains bound. A selection,
descriptor, plugin, or parent-session change closes the old binding before a
replacement is created.

Per-turn dispatch does not close the runtime. The binding closes exactly once
when the agent is evicted from the cache or hard-closed. Closing seals the host
first, so any later `emit_background_result` call is rejected, then closes the
runtime. This keeps a detached result from escaping a retired parent.

## Consequences

- Provider SDK readers and background task policy remain plugin-owned.
- Hermes core gains no provider model, auth, subscription, or dependency rule.
- Delivery remains process-local unless the existing host consumer provides a
  stronger guarantee; v1 adds no daemon, queue, datastore, or auth boundary.
- `background_delivery_v1` is declared in
  `agent/runtime_capabilities.json` and in the runtime API manifest.
