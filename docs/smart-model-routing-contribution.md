# Smart Model Routing — Contribution Notes

This document records the complete design and implementation context for the smart model-routing work in this fork. It is intentionally written as a future pull-request/contribution handoff so the work can be reviewed, tested, revised, or split into smaller upstream PRs without reconstructing the reasoning from chat history.

## Goal

Add an opt-in, task-aware model router to Hermes Agent without creating a second provider/client/retry framework.

For each user turn, the router should:

1. classify the task;
2. discover models through Hermes' existing provider catalog machinery;
3. read rich model metadata from `agent.models_dev`;
4. enforce an explicit cost policy;
5. filter for required capabilities;
6. rank candidates using task fit, quality priors, reliability, and observed latency;
7. activate the selected model through Hermes' canonical runtime switch;
8. expose the ranked alternatives to Hermes' existing fallback loop for the duration of that turn;
9. restore the original runtime after the turn.

The feature is disabled unless `smart_model_routing.enabled` is explicitly true.

## Architectural rule

The router is an orchestration layer, not a replacement for Hermes infrastructure.

Hermes remains the owner of:

- provider credentials and credential pools;
- model discovery and provider catalogs;
- client construction and transport selection;
- retry and error classification;
- native provider fallback behavior;
- prompt/context/compression lifecycle;
- streaming and tool execution.

The router must not duplicate those systems.

## Integration seam

The final integration point is `agent/turn_facade.py`, not the giant `agent/conversation_loop.py` implementation.

`TurnFacadeMixin.run_conversation()` is already the once-per-user-turn admission boundary. It performs turn admission and ContextVar setup and then calls `agent.conversation_loop.run_conversation()`. The smart-routing context manager wraps only that call.

This is important because routing must happen **before** `build_turn_context()` constructs the model-specific system prompt, cache plan, and context limits. Routing the already-built loop would risk preparing prompt state for the wrong model.

The integration therefore has this shape:

```text
AIAgent.run_conversation()
    |
    +-- turn admission / relay / accounting
    |
    +-- smart_route_turn()
    |       |
    |       +-- discover Hermes provider catalogs
    |       +-- read models.dev metadata
    |       +-- classify/filter/rank
    |       +-- switch_model() to selected runtime
    |       +-- install transient native fallback chain
    |       |
    |       +-- conversation_loop.run_conversation()
    |               |
    |               +-- build_turn_context()
    |               +-- normal Hermes API/tool/retry loop
    |               +-- native fallback execution
    |
    +-- restore original runtime
    +-- normal turn finalization
```

## Why not patch `conversation_loop.py` directly?

`conversation_loop.py` has undergone an active god-file decomposition. Its prologue is already extracted into `agent/turn_context.py`, its finalization is extracted into `agent/turn_finalizer.py`, and iteration phases are split into multiple `agent/turn_*.py` modules. A wrapper around the exported `run_conversation()` would work mechanically, but it would make the integration depend on the exported function's implementation boundary and could obscure the actual once-per-turn lifecycle.

The turn facade is a cleaner semantic seam and preserves the public method signature.

## Cost safety

`FREE_ONLY` is deliberately strict.

A model is considered free only when it is explicitly listed in `smart_model_routing.free_models`. Missing pricing metadata is **not** treated as free, and a zero/unknown price from a catalog is not sufficient entitlement evidence.

This prevents a metadata outage or stale catalog from silently turning a free-only session into a paid session.

`FREE_PREFERRED` keeps paid candidates eligible but gives explicitly free candidates a deterministic ranking bonus.

`ANY` makes no cost preference.

## Configuration

Example:

```yaml
smart_model_routing:
  enabled: true
  cost_policy: free_only

  # Cross-provider discovery is opt-in. If omitted, only the current provider is considered.
  providers:
    - nvidia
    - zai
    - gemini
    - ollama-cloud

  # Explicit entitlement. Use provider-local model IDs here.
  free_models:
    nvidia:
      - deepseek-v4-pro-0813
      - deepseek-v4-flash-0731
    zai:
      - <verified-free-model-id>

  # Optional minimum model context requirement.
  # min_context_length: 100000

  # Optional catalog refresh. Hermes already owns provider-model caching.
  # force_refresh: false
```

The example intentionally does not claim that every model offered by a provider is free. Free entitlement changes over time and must be verified against the provider's current terms/catalog before being placed in `free_models`.

## Provider/model discovery

The router uses Hermes' existing `provider_model_ids()` machinery. That machinery already knows about:

- curated static catalogs;
- provider-specific live `/models` discovery;
- provider model caches;
- models.dev integration;
- provider-specific fallback behavior.

Do not introduce a second persistent model catalog in the router.

## Metadata

`agent.models_dev.ModelInfo` supplies context size, reasoning, tool-call support, modalities, pricing fields, release/status information, and related metadata.

The router accepts both ModelInfo-like objects and mappings so its pure functions remain easy to unit test.

When models.dev does not provide task-specific quality/coding/research scores, the router uses conservative name/family priors. These are ranking priors, not benchmark claims. They should eventually be replaced or supplemented by measured evaluation data if Hermes gains a formal model-quality benchmark registry.

## Health

Health is process-local and deliberately lightweight:

- successful calls increment a success count;
- failures increment a failure count;
- successful latency observations contribute to a latency score.

The router does not create a second retry/cooldown system. Provider failures, credential rotation, retry policy, and fallback execution remain Hermes responsibilities.

A future enhancement can persist router observations, but that should be designed separately from the initial integration.

## Fallback behavior

The routing decision contains a ranked chain. During a routed turn, its alternatives are installed into Hermes' existing `_fallback_chain` only after the primary `switch_model()` call succeeds.

This ordering is intentional: `switch_model()` owns runtime/fallback-state normalization, so installing the temporary chain before the switch could cause Hermes' native switch logic to rewrite it.

For `FREE_ONLY`, the transient chain contains only explicitly free router candidates. Existing paid fallback entries are not appended for that turn.

For `FREE_PREFERRED` and `ANY`, router candidates are placed before the pre-existing fallback chain, with duplicate provider/model pairs removed.

The previous fallback state is restored after the turn.

## Runtime restoration

Do not restore the old client object directly.

A model switch can retire/rebuild clients, and provider changes can alter credential-pool and transport state. The integration therefore snapshots the original runtime identity and reconstructs it through `AIAgent.switch_model()` after the routed turn.

This is intentionally aligned with Hermes' existing runtime ownership instead of maintaining a parallel client lifecycle.

## Fail-open behavior

If smart routing cannot discover models, resolve metadata, parse configuration, satisfy capabilities, or activate a destination, it yields control back to the normal Hermes runtime without changing the turn.

The feature is therefore an optimization layer, not a new single point of failure.

## Explicit exclusions

- No changes to the user's live `~/.hermes/config.yaml` are part of this contribution.
- No API keys, tokens, or credentials belong in the repository.
- The local llama.cpp/Qwen3.6 runtime is intentionally outside this provider router.
- Web/search/browser/STT providers are tools/capabilities, not chat-model candidates.
- Do not install a second A2A implementation for this feature.
- Do not create a second credential/retry/client framework.

## Tests

Current unit coverage is split between:

- `tests/agent/test_smart_router.py`
  - strict FREE_ONLY behavior;
  - explicit free entitlement;
  - FREE_PREFERRED vs ANY ranking;
  - coding task preference;
  - health observations;
  - free-model normalization;
  - method-based and field-based metadata capability handling.

- `tests/agent/test_smart_routing_runtime.py`
  - turn-scoped activation;
  - restoration of model/provider/fallback state;
  - FREE_ONLY exclusion of paid fallback entries;
  - disabled-routing no-op;
  - fail-open discovery failure.

Before proposing the change upstream, run the focused tests and then the full Hermes test suite in the actual Hermes development environment.

Recommended commands from the repository root:

```bash
python -m pytest -q tests/agent/test_smart_router.py tests/agent/test_smart_routing_runtime.py
python -m pytest -q
```

If the project uses a specific maintained test command in the current checkout, prefer that command over the examples above.

## Manual smoke test

1. Keep `smart_model_routing.enabled: false` and confirm Hermes behaves exactly as before.
2. Enable routing with a single provider and one explicitly free model.
3. Send a short ordinary message and verify the selected model in the turn log.
4. Send a coding/tool-use task and verify a tool-capable model is selected.
5. Force a provider failure and verify Hermes' normal fallback machinery continues to work.
6. Send the next turn and verify the automatic route does not remain as a permanent `/model` switch.
7. Verify `/model` still behaves as a persistent/session/one-turn user command according to its existing semantics.
8. Test a multimodal turn and verify non-vision candidates are filtered.
9. Test a long-context request with `min_context_length` and verify smaller candidates are filtered.

## Review checklist for an upstream PR

- [ ] Feature remains opt-in and disabled by default.
- [ ] No credentials or secrets are committed.
- [ ] No duplicate provider/client/retry implementation was introduced.
- [ ] FREE_ONLY never infers entitlement from missing/unknown pricing.
- [ ] Router runs once per user turn, not once per API retry/tool iteration.
- [ ] System prompt/context are built after the target model is selected.
- [ ] Codex app-server lifecycle remains untouched.
- [ ] Existing `/model` semantics remain unchanged.
- [ ] Native Hermes fallback execution remains the only retry/fallback executor.
- [ ] Original runtime is reconstructed through canonical runtime switching.
- [ ] Temporary router fallback state cannot leak into the next turn.
- [ ] Provider/model catalog caching remains Hermes-owned.
- [ ] Focused tests pass.
- [ ] Full test suite passes.
- [ ] Formatting/lint/type checks required by the current project pass.

## Suggested PR title

`feat(agent): add opt-in task-aware smart model routing`

## Suggested PR summary

This contribution adds an opt-in smart model-routing layer that selects an already-configured Hermes provider/model per user turn using task classification, capability filtering, explicit cost policy, model metadata, and lightweight runtime health observations. It integrates at the existing turn facade and delegates provider resolution, client construction, retries, credential handling, tool execution, and fallback execution back to Hermes. Routing is transient: the original runtime is reconstructed after the turn, and the router's ranked alternatives are exposed to Hermes' native fallback chain only for that turn.

The implementation intentionally reuses Hermes' existing model discovery and models.dev metadata systems rather than introducing a second model registry or provider framework.
