# AgentRuntime v1 coupling map

| Boundary | Owner | Public data | Private data retained by owner |
| --- | --- | --- | --- |
| Turn selection | Hermes | `RuntimeSelection` | provider client and credentials |
| Turn execution | Runtime plugin | `RuntimeTurnRequest`, typed events | SDK session and reader |
| Tool and approval | Hermes host binding | bounded names, arguments, decisions | `AIAgent`, approval callbacks |
| Runtime state and usage | Hermes host binding | typed envelopes and receipts | Hermes session ID and database |
| Background completion | Hermes host binding | `RuntimeBackgroundResult` | parent session, UI/session key, gateway route, delivery identity |
| Session lifecycle | Hermes `AIAgent` | none | cached runtime/binding and close state |

## Background flow

1. The plugin completes `run_turn` with exactly one terminal event.
2. Detached plugin work later calls `emit_background_result` on the same host
   binding with bounded normalized content.
3. The binding adds its host-private exact-parent route and a host-generated
   delivery identity.
4. The existing completion consumer wakes the exact parent, requeues while it
   is busy, projects the result into a new transcript turn, and retries failed
   adapter acceptance.
5. Agent eviction or shutdown seals the binding and closes the runtime once;
   post-close emission fails.

The plugin has no gateway import, route fields, latest-session fallback,
arbitrary metadata channel, or host/provider session identifier. Core has no
plugin-private import and no provider-specific policy.
