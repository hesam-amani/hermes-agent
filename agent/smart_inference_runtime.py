"""Turn-scoped runtime binding for automatic Smart Inference routing.

Automatic routing is deliberately different from Hermes' durable ``switch_model``
operation. This module reuses Hermes' destination-resolution/client-building
primitives, but never rewrites ``_primary_runtime`` or billing state.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


_RUNTIME_FIELDS = (
    "model", "provider", "requested_provider", "base_url", "api_mode", "api_key",
    "client", "_anthropic_client", "_anthropic_api_key", "_anthropic_base_url",
    "_is_anthropic_oauth", "_config_context_length", "_reasoning_echo_flag",
    "runtime_capabilities", "_credential_pool", "_credential_pool_entry_id",
    "_client_kwargs", "_use_prompt_caching", "_use_native_cache_layout",
    "_cached_system_prompt", "reasoning_config", "request_overrides", "_custom_providers",
    "_fallback_chain", "_fallback_model", "_fallback_index", "_fallback_activated",
    "_provider_fallback_active", "_provider_fallback_route", "_primary_runtime",
)

_COMPRESSOR_FIELDS = (
    "model", "context_length", "base_url", "api_key", "provider", "api_mode",
    "threshold_tokens",
)

_MISSING = object()


def _copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return [dict(item) if isinstance(item, dict) else item for item in value]
    return value


def _snapshot(agent: Any) -> dict[str, Any]:
    state = {
        name: _copy_value(getattr(agent, name, _MISSING))
        for name in _RUNTIME_FIELDS
    }
    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        state["__compressor__"] = {
            name: _copy_value(getattr(compressor, name, _MISSING))
            for name in _COMPRESSOR_FIELDS
        }
    transport_cache = getattr(agent, "_transport_cache", _MISSING)
    state["__transport_cache__"] = (
        dict(transport_cache) if isinstance(transport_cache, dict) else transport_cache
    )
    return state


def _restore(agent: Any, state: dict[str, Any]) -> None:
    for name in _RUNTIME_FIELDS:
        value = state.get(name, _MISSING)
        if value is _MISSING:
            continue
        try:
            setattr(agent, name, value)
        except Exception:
            # Restoration must be best-effort for optional/private attributes;
            # Hermes' existing restore machinery follows the same principle.
            pass

    transport_cache = state.get("__transport_cache__", _MISSING)
    if transport_cache is not _MISSING:
        try:
            agent._transport_cache = transport_cache
        except Exception:
            pass

    compressor_state = state.get("__compressor__")
    compressor = getattr(agent, "context_compressor", None)
    if isinstance(compressor_state, dict) and compressor is not None:
        for name, value in compressor_state.items():
            if value is _MISSING:
                continue
            try:
                setattr(compressor, name, value)
            except Exception:
                pass


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
    """Bind ``agent`` to one model for the lifetime of a user turn.

    The normal Hermes conversation loop remains in charge while this context is
    active: request assembly, tools, retries, streaming and native fallback all
    see the selected runtime. On exit the original primary runtime and the
    transient routing state are restored, including the context-compressor
    destination.
    """
    from agent import agent_runtime_helpers as runtime

    old_provider = str(getattr(agent, "provider", "") or "")
    old_norm = old_provider.strip().lower()
    new_norm = str(provider or "").strip().lower()
    state = _snapshot(agent)
    switch_snapshot = runtime._snapshot_switch_state(agent)

    try:
        resolved_api_mode, resolved_base_url, destination_capabilities = (
            runtime._resolve_switch_destination(
                agent, model, provider, base_url, api_mode, capabilities,
                old_norm, new_norm,
            )
        )

        runtime._swap_switch_runtime(
            agent, model, provider, api_key, resolved_base_url,
            resolved_api_mode, old_provider, old_norm, new_norm,
        )

        custom_providers, effective_context_length = runtime._resolve_switch_context_length(
            agent, switch_snapshot
        )
        if custom_providers is not None:
            agent._custom_providers = custom_providers

        agent._use_prompt_caching, agent._use_native_cache_layout = (
            agent._anthropic_prompt_cache_policy(
                provider=provider, base_url=agent.base_url,
                api_mode=resolved_api_mode, model=model,
            )
        )

        if getattr(agent, "context_compressor", None):
            runtime._update_switch_compressor(
                agent, custom_providers, effective_context_length, switch_snapshot,
            )

        try:
            from hermes_constants import resolve_reasoning_config
            from hermes_cli.config import load_config
            agent.reasoning_config = resolve_reasoning_config(load_config() or {}, agent.model)
        except Exception:
            pass

        agent._cached_system_prompt = None
        agent.runtime_capabilities = destination_capabilities

        # This prepares per-destination request overrides/fallback bookkeeping,
        # but the complete state above is restored when the turn ends. Crucially,
        # we never call _finish_switch's billing persistence or rebuild
        # _primary_runtime.
        runtime._finish_switch(agent, provider, old_norm, new_norm)

        yield
    finally:
        _restore(agent, state)


__all__ = ["temporary_model_runtime"]
