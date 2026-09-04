"""Turn-scoped integration between SmartRouter and the Hermes runtime.

This module deliberately does not construct clients or implement retries. It asks
Hermes to resolve and activate a destination through the existing runtime path,
then restores the original runtime when the turn finishes. Router fallbacks are
installed only for the duration of this turn; Hermes still owns actual fallback
execution, credentials, transports, and retry behavior.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from time import monotonic
from typing import Any, Iterator

from agent.smart_router import (
    CostPolicy,
    ModelCandidate,
    RoutingDecision,
    RoutingRequest,
    SmartRouter,
    discover_candidates,
    normalize_free_models,
)

logger = logging.getLogger(__name__)
_ROUTER = SmartRouter()


def _config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        value = cfg.get("smart_model_routing", {})
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _providers(cfg: dict[str, Any], agent: Any) -> list[str]:
    configured = cfg.get("providers")
    if isinstance(configured, str):
        configured = [x.strip() for x in configured.split(",") if x.strip()]
    if isinstance(configured, list):
        return list(dict.fromkeys(str(x).strip().lower() for x in configured if str(x).strip()))
    # Safe default: never probe every provider in the catalog. The user opts
    # into cross-provider routing by naming the providers in config.yaml.
    current = str(getattr(agent, "provider", "") or "").strip().lower()
    return [current] if current else []


def _prompt_text(user_message: Any) -> str:
    if isinstance(user_message, str):
        return user_message
    if isinstance(user_message, dict):
        content = user_message.get("content") or user_message.get("text") or ""
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
            return "\n".join(parts)
        return str(content)
    return str(user_message or "")


def _is_multimodal_message(user_message: Any) -> bool:
    if isinstance(user_message, dict):
        content = user_message.get("content")
    else:
        content = user_message
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict)
        and str(item.get("type") or "").lower() in {"image", "image_url", "input_image"}
        for item in content
    )


def _requirements(prompt: str) -> tuple[bool, bool, bool]:
    text = prompt.lower()
    tools = any(x in text for x in (
        "use the terminal", "run this", "execute", "shell", "command", "file", "repo",
        "repository", "browser", "tool",
    ))
    vision = any(x in text for x in ("image", "photo", "screenshot", "picture"))
    reasoning = any(x in text for x in ("prove", "derive", "analyze", "reason", "solve", "calculate"))
    return tools, vision, reasoning


def _runtime_kwargs(candidate: ModelCandidate) -> dict[str, Any]:
    from hermes_cli.runtime_provider import resolve_runtime_provider

    resolved = resolve_runtime_provider(
        requested=candidate.provider,
        target_model=candidate.model,
        explicit_base_url=(candidate.extra or {}).get("base_url") or None,
        explicit_api_key=(candidate.extra or {}).get("api_key") or None,
    )
    return {
        "new_model": candidate.model,
        "new_provider": str(resolved.get("provider") or candidate.provider),
        "api_key": resolved.get("api_key") or "",
        "base_url": resolved.get("base_url") or "",
        "api_mode": resolved.get("api_mode") or "",
    }


def _original_runtime(agent: Any) -> dict[str, Any]:
    return {
        "new_model": agent.model,
        "new_provider": agent.provider,
        "api_key": agent.api_key if isinstance(agent.api_key, str) else "",
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
    }


def _fallback_entry(candidate: ModelCandidate) -> dict[str, str]:
    return {"provider": candidate.provider, "model": candidate.model}


def _install_router_fallbacks(
    agent: Any,
    decision: RoutingDecision,
    policy: CostPolicy,
) -> tuple[Any, Any, Any]:
    """Temporarily expose ranked candidates to Hermes' native fallback loop."""
    old_chain = getattr(agent, "_fallback_chain", None)
    old_index = getattr(agent, "_fallback_index", 0)
    old_unavailable = getattr(agent, "_unavailable_fallback_keys", None)

    router_entries = [_fallback_entry(candidate) for candidate in decision.fallbacks]
    if policy is CostPolicy.FREE_ONLY:
        # FREE_ONLY is a hard spending boundary: do not append any user-configured
        # paid fallback entries to this turn.
        new_chain = router_entries
    else:
        existing = list(old_chain or [])
        seen = {
            (str(item.get("provider", "")).lower(), str(item.get("model", "")).lower())
            for item in router_entries
        }
        new_chain = router_entries + [
            item for item in existing
            if isinstance(item, dict)
            and (str(item.get("provider", "")).lower(), str(item.get("model", "")).lower()) not in seen
        ]

    agent._fallback_chain = new_chain
    agent._fallback_index = 0
    agent._unavailable_fallback_keys = set()
    return old_chain, old_index, old_unavailable


def _restore_router_fallbacks(agent: Any, snapshot: tuple[Any, Any, Any]) -> None:
    old_chain, old_index, old_unavailable = snapshot
    agent._fallback_chain = old_chain
    agent._fallback_index = old_index
    agent._unavailable_fallback_keys = old_unavailable


def _runtime_differs(agent: Any, original: dict[str, Any]) -> bool:
    return (
        getattr(agent, "model", None) != original["new_model"]
        or str(getattr(agent, "provider", "") or "").lower() != str(original["new_provider"] or "").lower()
        or getattr(agent, "base_url", None) != original["base_url"]
        or getattr(agent, "api_mode", None) != original["api_mode"]
    )


@contextmanager
def smart_route_turn(agent: Any, user_message: Any) -> Iterator[RoutingDecision | None]:
    """Select and temporarily activate a route for one user turn.

    All failures are fail-open: Hermes simply continues with its existing
    primary runtime. Automatic routing is disabled unless explicitly enabled.
    """
    cfg = _config()
    if not cfg.get("enabled", False):
        yield None
        return

    if getattr(agent, "_smart_routing_locked", False):
        yield None
        return

    # The Codex app-server has its own subprocess lifecycle; do not redirect it.
    if getattr(agent, "api_mode", "") == "codex_app_server":
        yield None
        return

    prompt = _prompt_text(user_message)
    if not prompt.strip():
        yield None
        return

    providers = _providers(cfg, agent)
    if not providers:
        yield None
        return

    try:
        free_models = normalize_free_models(cfg.get("free_models", []))
        candidates = discover_candidates(
            _ROUTER,
            providers,
            free_models,
            force_refresh=bool(cfg.get("force_refresh", False)),
        )
        requires_tools, requires_vision, requires_reasoning = _requirements(prompt)
        # Hermes is an agent, so if tools are actually registered, the selected
        # model must advertise tool calling even when the user's wording does not
        # explicitly mention a tool. Multimodal payloads likewise require vision.
        requires_tools = requires_tools or bool(getattr(agent, "tools", None))
        requires_vision = requires_vision or _is_multimodal_message(user_message)
        policy_name = str(cfg.get("cost_policy", CostPolicy.FREE_ONLY.value)).lower()
        try:
            policy = CostPolicy(policy_name)
        except ValueError:
            policy = CostPolicy.FREE_ONLY
        request = RoutingRequest(
            prompt=prompt,
            cost_policy=policy,
            requires_tools=requires_tools,
            requires_vision=requires_vision,
            requires_reasoning=requires_reasoning,
            min_context_length=cfg.get("min_context_length"),
        )
        decision = _ROUTER.route(request, candidates)
    except Exception as exc:
        logger.debug("Smart routing skipped: %s", exc, exc_info=True)
        yield None
        return

    original = _original_runtime(agent)
    started = monotonic()
    fallback_snapshot = None
    try:
        if not (
            str(decision.primary.provider).lower() == str(agent.provider).lower()
            and decision.primary.model == agent.model
        ):
            agent.switch_model(**_runtime_kwargs(decision.primary))

        # Install the ranked fallback chain only AFTER switch_model() has completed.
        # switch_model() owns fallback-state normalization, so installing before it
        # could cause the native switch code to filter or rewrite our transient chain.
        fallback_snapshot = _install_router_fallbacks(agent, decision, policy)
        yield decision
        _ROUTER.observe(
            decision.primary,
            success=True,
            latency_seconds=monotonic() - started,
        )
    except Exception:
        _ROUTER.observe(decision.primary, success=False)
        raise
    finally:
        try:
            # The native fallback loop may have moved the agent away from both the
            # router primary and the original model. Automatic routing is turn-scoped,
            # so restore the original runtime whenever the effective identity changed.
            if _runtime_differs(agent, original):
                agent.switch_model(**original)
        except Exception:
            logger.exception("Smart routing could not restore original runtime")
        finally:
            if fallback_snapshot is not None:
                _restore_router_fallbacks(agent, fallback_snapshot)


__all__ = ["smart_route_turn"]
