# AgentRuntime Plugin API v1

Status: Candidate contract on `codex/agent-runtime-plugin-api-v1`

Host base: `NousResearch/hermes-agent@64b96bb5d2755f1d34347e1fb15924a97d652f31`

Decision owner: Hermes core maintainers

Source issue: https://github.com/NousResearch/hermes-agent/issues/25267

Source implementation: https://github.com/NousResearch/hermes-agent/pull/65982

## Desired function

A clean Hermes host can discover and run independently released whole-turn
agent runtimes. A third-party runtime is installed through the existing native
plugin entry point and can be removed without leaving provider-specific code or
dependencies in Hermes core.

## Current gate

Prove one small provider-neutral host interface supports the built-in Codex
app-server runtime and an external Claude Agent SDK runtime while permissions,
tool execution, session persistence, usage accounting, cancellation, and
fallback policy remain host-owned.

## First-principles review

- **Hard constraints:** no duplicated runtime implementation; no Claude/Fable
  selection or subscription policy in generic core; no Claude SDK dependency in
  core; host-owned tools and approval; fail-closed billing; no monkeypatch or
  MCP-server substitution; clean operation without the plugin; independent
  plugin release, removal, and rollback; contributor attribution preserved.
- **Soft assumptions challenged:** the model-provider profile must own a whole
  turn; provider-named loop branches are permanent; the first release must drop
  `claude_sdk_session_id`; multiple new plugin loaders are needed; all of PR
  #65982 belongs upstream.
- **Magic-wand floor:** one versioned `AgentRuntime` protocol, one registry, one
  dispatch point, one host-services facade, bounded generic request/event/state/
  usage/failure envelopes, one built-in Codex consumer, and one external Claude
  consumer through the existing plugin loader.
- **Measured current cost:** the downstream candidate changes 116 files with
  26,615 insertions and 335 deletions relative to its merge base with current
  main. Fifty-five test files account for 16,051 inserted lines. Ten
  Claude-specific source/docs paths account for 6,770 inserted lines; 47 other
  host-core paths account for 3,789 insertions and 305 deletions. The six main
  Claude runtime/transport modules total 6,219 lines. Claude-specific behavior
  is named directly in conversation dispatch, runtime/provider resolution,
  lifecycle, compaction, state, usage, fallback, gateway delivery, approvals,
  lazy dependencies, configuration, diagnostics, and packaging.
- **Software Idiot Index (estimate):** high, about 6x on host coupling surface
  (47 provider-coupled host files versus roughly eight host files needed for a
  registry, dispatch adapter, generic state/receipts, lifecycle, plugin context,
  and tests). This does not imply the Claude implementation is waste; most of it
  moves intact behind the plugin boundary.
- **80/20 move:** generalize the existing Codex whole-turn short-circuit, adapt
  Codex as the first provider-neutral consumer, then move Claude behavior behind
  the same interface.
- **Recommendation:** SIMPLIFY. Do not add another orchestrator, service,
  datastore, auth system, or model router.
- **Proof needed:** clean host without plugin; real entry-point discovery;
  compatibility rejection before factory activation; built-in Codex regression;
  subscription-only Claude text/tool/resume slice; unchanged frozen parity
  contract; plugin removal restoring built-in-only operation.
- **Negative-risk notes:** retain additive state migration; never expose host
  internals or credentials; never let runtime code bypass approval; never hide
  missing host compatibility with a monkeypatch.
- **Confidence:** 90% in the seam and ownership boundary; implementation and
  runtime safety remain gated by the named contract, CI, and live matrix.

## Decision

Hermes exports `RUNTIME_API_VERSION = 1` and a concrete set of host capability
identifiers. The existing `hermes_agent.plugins` loader receives one additive
`PluginContext.register_agent_runtime()` method. No second loader is added.

Registration accepts a frozen `RuntimeDescriptor` and a zero-argument runtime
factory. Registration validates the descriptor before the factory is retained
or invoked. An incompatible API range or missing host capability raises a typed,
actionable compatibility error. Credential resolution, dependency installation,
SDK process creation, and model calls are forbidden during descriptor creation
and registration.

The plugin runtime registry is profile-scoped through the existing
plugin-manager scope. The built-in Codex runtime is supplied by host bootstrap
as a `RuntimeRegistration` and resolved together with plugin registrations by
the same pure resolver. Unloading a plugin disposes its runtime registration.
A clean host with no matching third-party runtime continues through the
ordinary Hermes conversation loop unchanged.

### Descriptor and capability handshake

`RuntimeDescriptor` contains:

- `runtime_id` and `plugin_version`;
- inclusive `runtime_api_min` and `runtime_api_max`;
- a frozen set of concrete `required_host_capabilities`;
- supported provider/model selectors;
- `session_state_schema_version`;
- only capability flags consumed by Codex or the Claude Agent SDK runtime.

The machine-readable capability identifiers and descriptor schema live beside
the public Python protocol in `agent/runtime_api.py`; `runtime_api_manifest()`
is the canonical JSON-compatible handshake. Adding a capability requires a
concrete built-in or external consumer and a contract test.

### Whole-turn protocol

`AgentRuntime.preflight(request)` is a pure, side-effect-free support and
compatibility decision.
`AgentRuntime.run_turn(request, host)` is async and returns an async stream of
typed `RuntimeEvent` values. The host owns event validation and terminal-state
enforcement.

`RuntimeTurnRequest` is immutable and contains normalized content, attachments,
prompt snapshot, stable tool schemas and their SHA-256 hash, provider/model
selection, correlation ids, and a generic runtime-state envelope. Cancellation
is observed through the host facade. The request contains no `AIAgent`,
`SessionDB`, plugin manager, credential object, or mutable host internals.

`RuntimeHostServices` is a stable facade for:

- host tool execution and permission/approval;
- status and bounded content emission;
- safe runtime-state and usage-receipt persistence;
- cancellation and lifecycle observation;
- optional hybrid tool and delegation bridges exposed as host capabilities.

Runtimes can request those operations but cannot substitute their own policy or
obtain underlying host objects. Tool calls must traverse the canonical Hermes
executor funnel: availability and toolset scope, request/execution middleware,
pre-tool policy and approval, guardrails, activity/progress, underlying dispatch,
post-tool and transform hooks, result normalization, persistence, and terminal
accounting. Direct `tools.registry.dispatch()` is not a valid host facade.

### Events and terminal state

The v1 event union is deliberately small: content delta, status, tool request,
approval request, state update, compaction lifecycle, usage receipt, classified
failure, cancellation, and completion. Exactly one completion, failure, or
cancellation event terminates a turn; host dispatch rejects duplicate or
post-terminal events.

### Failure and fallback

`RuntimeFailure` distinguishes preflight failure, failure before visible output
or side effects, failure after visible output, and failure after side effects.
It carries an explicit replay classification. The host may fall back only when
host policy allows it and the runtime reports replay-safe before any output or
effect. Exception type alone never authorizes replay. The Claude subscription
plugin declares fallback disabled for explicit Fable sessions.

### State and usage

The host stores a versioned `RuntimeStateEnvelope` keyed by Hermes session and
`runtime_id`. The payload is safe opaque JSON data and must not contain
credentials. Current upstream main has no Claude-specific session column. The
frozen downstream candidate does; the first compatibility release may import
that candidate's `claude_sdk_session_id` when present and never drops it.

`RuntimeUsageReceipt` records runtime/provider/model, billing mode, cost status,
available token/cache fields, replay/fallback classification, and safe
correlation identifiers. Host persistence remains the source of truth.

### Compaction and lifecycle

Compaction ownership is `HOST` or `RUNTIME_NATIVE`. A native runtime emits
start/completed/failed/watchdog events and the host bypasses its compressor by
capability, never by provider name. Close, cancel, interrupt, transport failure,
gateway displacement, and resume use generic lifecycle methods and one terminal
state per turn.

## Ownership boundary

Hermes owns the protocol, registry, dispatch, generic lifecycle/replay rules,
generic state and receipt persistence, compaction ownership/events, and stable
host facades.

The external Claude plugin owns the `claude-agent-sdk` dependency and version
policy, SDK process/session management, Claude/Fable selection, subscription and
OAuth detection, fail-closed billing, content conversion, native compaction,
prompt/memory/skills/project context, Claude resume state, diagnostics, setup,
documentation, tests, and packaging.

No Claude model, billing, dependency, configuration, or session type is added to
the generic protocol. No final release may retain duplicate Claude
implementations. A temporary migration reader is allowed only for additive
compatibility.

## Compatibility and rollback

The host API is additive within v1. New descriptor fields are optional with
defaults; removed or renamed public fields require a new API version. Plugin
removal unregisters the runtime and leaves generic persisted state inert.
Rolling the host back to the exact upstream base restores the prior built-in
paths. The external plugin documents the exact host SHA/API capability matrix
until the core change is released upstream.

## Proof boundary

Acceptance proves only exact host/plugin SHAs on the frozen local contract. It
does not prove upstream merge, future-main compatibility, arbitrary provider or
channel parity, shared Eva cutover, fleet/customer readiness, metered routes, or
identical model prose.
