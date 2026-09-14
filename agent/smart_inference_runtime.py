"""Turn-scoped runtime binding for automatic Smart Inference routing.

This module deliberately does *not* use ``AIAgent.switch_model()``. That method
represents a durable user/runtime selection and updates ``_primary_runtime`` and
billing state. Automatic routing is different: it is a temporary execution
choice for one user turn.

Hermes still owns provider resolution, credential pools, clients, context
compressor setup, prompt-cache policy, reasoning configuration, and restoration.
The helper below reuses those existing runtime primitives while leaving the
user-selected primary runtime untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def temporary_model_runtime(
    agent: Any,
    *,
    model: str,
    provider: str,
    api_key: str = "",
    base_url: str = "",
    api_mode: str = "",
    capabilities: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Temporarily bind ``agent`` to a resolved model/provider for one turn.

    The implementation mirrors the *runtime* portion of Hermes' native model
    switch, but intentionally omits the durable parts: ``_primary_runtime`` is
    never rewritten, fallback state is not persisted, and billing-route
    persistence is not performed.

    The full pre-bind runtime snapshot is restored even when the turn raises or
    is interrupted. This lets the normal Hermes loop continue to own retries,
    provider fallback, streaming, tools, and all other turn mechanics.
    """

    from agent import agent_runtime_helpers as runtime

    old_provider = str(getattr(agent, "provider", "") or "")
    old_norm = old_provider.strip().lower()
    new_norm = str(provider or "").strip().lower()

    snapshot = runtime._snapshot_switch_state(agent)

    try:
        (
            resolved_api_mode,
            resolved_base_url,
            destination_capabilities,
        ) = runtime._resolve_switch_destination(
            agent,
            model,
            provider,
            base_url,
            api_mode,
            capabilities,
            old_norm,
            new_norm,
        )

        runtime._swap_switch_runtime(
            agent,
            model,
            provider,
            api_key,
            resolved_base_url,
            resolved_api_mode,
            old_provider,
            old_norm,
            new_norm,
        )

        custom_providers, effective_context_length = runtime._resolve_switch_context_length(
            agent, snapshot
        )
        if custom_providers is not None:
            agent._custom_providers = custom_providers

        agent._use_prompt_caching, agent._use_native_cache_layout = (
            agent._anthropic_prompt_cache_policy(
                provider=provider,
                base_url=agent.base_url,
                api_mode=resolved_api_mode,
                model=model,
            )
        )

        if getattr(agent, "context_compressor", None):
            runtime._update_switch_compressor(
                agent,
                custom_providers,
                effective_context_length,
                snapshot,
            )

        # Same model-resolution semantics as Hermes' native switch. These values
        # are transient because the complete pre-turn snapshot is restored below.
        try:
            from hermes_constants import resolve_reasoning_config
            from hermes_cli.config import load_config

            agent.reasoning_config = resolve_reasoning_config(load_config() or {}, agent.model)
        except Exception:
            # A missing optional reasoning override must not make automatic
            # routing unusable; the normal runtime already has a valid config.
            pass

        agent._cached_system_prompt = None
        agent.runtime_capabilities = destination_capabilities

        # Reuse the normal post-switch fallback/request-override preparation, but
        # do not call _persist_switch_billing_route() and do not rebuild
        # _primary_runtime. Fallbacks remain Hermes-owned within this turn.
        runtime._finish_switch(agent, provider, old_norm, new_norm)

        yield
    finally:
        runtime._restore_switch_snapshot(agent, snapshot)


__all__ = ["temporary_model_runtime"]
