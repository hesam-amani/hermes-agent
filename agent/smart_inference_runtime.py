"""Temporary, turn-scoped Hermes runtime binding for Smart Inference.

Smart Inference selects a route; Hermes continues to own the actual runtime.

This module intentionally does NOT call ``switch_model()`` and does NOT update
``_primary_runtime``. The selected model exists only for the lifetime of the
current turn.

Hermes continues to own:
    - provider resolution
    - credentials
    - clients
    - transport
    - request construction
    - tools
    - retries
    - fallback
    - streaming
    - persistence
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Iterator


# These are the runtime fields that the temporary switch may touch.
#
# Keep this public-to-the-module (rather than hiding it inside the snapshot
# implementation) because the runtime tests use it to verify complete
# restoration.
_RUNTIME_FIELDS = (
    "model",
    "provider",
    "requested_provider",
    "base_url",
    "api_mode",
    "api_key",
    "client",
    "_anthropic_client",
    "_anthropic_api_key",
    "_anthropic_base_url",
    "_is_anthropic_oauth",
    "_config_context_length",
    "_reasoning_echo_flag",
    "runtime_capabilities",
    "_credential_pool",
    "_credential_pool_entry_id",
    "_client_kwargs",
    "_use_prompt_caching",
    "_use_native_cache_layout",
    "_cached_system_prompt",
    "reasoning_config",
    "request_overrides",
    "_custom_providers",
    "_transport_cache",
)

_COMPRESSOR_FIELDS = (
    "model",
    "context_length",
    "base_url",
    "api_key",
    "provider",
    "api_mode",
    "threshold_tokens",
)

_MISSING = object()


def _safe_copy(value: Any) -> Any:
    """Copy runtime state while preserving opaque sentinel objects."""

    if type(value) is object:
        return value

    if isinstance(value, dict):
        return {
            _safe_copy(key): _safe_copy(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_safe_copy(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_safe_copy(item) for item in value)

    if isinstance(value, set):
        return {_safe_copy(item) for item in value}

    try:
        return deepcopy(value)
    except Exception:
        return value


def _snapshot_agent(agent: Any) -> dict[str, Any]:
    """Create a snapshot of the agent's runtime fields and context compressor."""
    snapshot: dict[str, Any] = {}

    for name in _RUNTIME_FIELDS:
        if hasattr(agent, name):
            snapshot[name] = _safe_copy(getattr(agent, name))

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        snapshot["_context_compressor"] = {
            name: _safe_copy(getattr(compressor, name))
            for name in _COMPRESSOR_FIELDS
            if hasattr(compressor, name)
        }

    return snapshot


def _restore_agent(agent: Any, snapshot: dict[str, Any]) -> None:
    """Restore the agent's runtime fields and context compressor from a snapshot."""
    for name in _RUNTIME_FIELDS:
        if name not in snapshot:
            continue

        try:
            setattr(agent, name, snapshot[name])
        except Exception:
            pass

    compressor_snapshot = snapshot.get("_context_compressor")
    compressor = getattr(agent, "context_compressor", None)

    if compressor is not None and compressor_snapshot:
        for name, value in compressor_snapshot.items():
            try:
                setattr(compressor, name, value)
            except Exception:
                pass


def _resolve_destination(
    agent: Any,
    model: str,
    provider: str,
) -> tuple[str, str, dict[str, Any]]:
    """Resolve a Hermes switch destination.

    Hermes' current private helper returns:

        (api_mode, base_url, destination_capabilities)
    """

    from agent import agent_runtime_helpers as runtime

    old_provider = getattr(
        agent,
        "provider",
        "",
    ) or ""

    old_norm = old_provider.strip().lower()
    new_norm = (provider or "").strip().lower()

    result = runtime._resolve_switch_destination(
        agent,
        model,
        provider,
        "",
        "",
        None,
        old_norm,
        new_norm,
    )

    if not isinstance(result, tuple) or len(result) != 3:
        raise TypeError(
            "Hermes _resolve_switch_destination returned "
            f"an unexpected value: {result!r}"
        )

    api_mode, base_url, capabilities = result

    return (
        api_mode,
        base_url,
        capabilities,
    )


def _activate(
    agent: Any,
    model: str,
    provider: str,
    api_key: str = "",
) -> None:
    """Activate the selected model using Hermes' native switch primitives."""

    from agent import agent_runtime_helpers as runtime

    old_provider = getattr(
        agent,
        "provider",
        "",
    ) or ""

    old_norm = old_provider.strip().lower()
    new_norm = (provider or "").strip().lower()

    api_mode, base_url, capabilities = (
        runtime._resolve_switch_destination(
            agent,
            model,
            provider,
            "",
            "",
            None,
            old_norm,
            new_norm,
        )
    )

    runtime._swap_switch_runtime(
        agent,
        model,
        provider,
        api_key,
        base_url,
        api_mode,
        old_provider,
        old_norm,
        new_norm,
    )

    custom_providers, effective_context_length = (
        runtime._resolve_switch_context_length(
            agent,
            None,
        )
    )

    if custom_providers is not None:
        agent._custom_providers = custom_providers

    if (
        hasattr(agent, "context_compressor")
        and agent.context_compressor
    ):
        runtime._update_switch_compressor(
            agent,
            custom_providers,
            effective_context_length,
            None,
        )

    # This is deliberately NOT _finish_switch().
    #
    # _finish_switch() resets fallback state, prunes the fallback chain and
    # therefore represents durable model-switch semantics. Smart Inference
    # must not do that.
    runtime._apply_switched_provider_request_overrides(
        agent,
        provider,
    )

    # Publish destination capabilities only for the duration of this context.
    agent.runtime_capabilities = (
        dict(capabilities)
        if isinstance(capabilities, dict)
        else capabilities
    )


def _retire_temporary_client(
    agent: Any,
    snapshot: dict[str, Any],
) -> None:
    """Best-effort cleanup of a temporary shared client."""

    try:
        from agent import agent_runtime_helpers as runtime

        current_client = getattr(
            agent,
            "client",
            None,
        )

        previous_client = snapshot.get(
            "client",
            _MISSING,
        )

        if (
            current_client is not None
            and previous_client is not _MISSING
            and current_client is not previous_client
        ):
            retire = getattr(
                runtime,
                "_retire_shared_openai_client",
                None,
            )

            if retire is not None:
                try:
                    retire(
                        agent,
                        current_client,
                    )
                except TypeError:
                    try:
                        retire(current_client)
                    except Exception:
                        pass
                except Exception:
                    pass

    except Exception:
        # Cleanup must never hide the actual turn result/error.
        pass


@contextmanager
def temporary_model_runtime(
    agent: Any,
    model: str,
    provider: str,
    api_key: str = "",
) -> Iterator[None]:
    """Temporarily route one turn through ``provider/model``.

    The original Hermes runtime is restored unconditionally when the context
    exits, including when the turn raises.
    """

    snapshot = _snapshot_agent(agent)

    try:
        _activate(
            agent,
            model,
            provider,
            api_key,
        )
        yield

    except BaseException:
        _retire_temporary_client(
            agent,
            snapshot,
        )
        _restore_agent(
            agent,
            snapshot,
        )
        raise

    else:
        _retire_temporary_client(
            agent,
            snapshot,
        )
        _restore_agent(
            agent,
            snapshot,
        )
