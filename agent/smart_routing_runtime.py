"""Turn-scoped integration between SmartRouter and the Hermes runtime.

This module deliberately does not construct clients or implement retries. It
asks Hermes to resolve and activate a destination through the existing
AIAgent.switch_model() path, then switches back to a freshly-built original
runtime when the turn finishes. This avoids restoring a client object that may
have been retired during the temporary switch.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from time import monotonic
from typing import Any, Iterator

from agent.smart_router import (
    CostPolicy, ModelCandidate, RoutingDecision, RoutingRequest, SmartRouter,
    discover_candidates, normalize_free_models,
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
        return str(user_message.get("content") or user_message.get("text") or "")
    return str(user_message or "")


def _requirements(prompt: str) -> tuple[bool, bool, bool]:
    text = prompt.lower()
    tools = any(x in text for x in (
        "use the terminal", "run this", "execute", "shell", "command", "file", "repo", "repository", "browser", "tool",
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

    # Explicitly opt out when a caller/command marks this turn as locked. The
    # marker is intentionally generic so CLI/gateway/ACP can share it later.
    if getattr(agent, "_smart_routing_locked", False):
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
        candidates = discover_candidates(_ROUTER, providers, free_models)
        requires_tools, requires_vision, requires_reasoning = _requirements(prompt)
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
    switched = False
    try:
        if not (
            str(decision.primary.provider).lower() == str(agent.provider).lower()
            and decision.primary.model == agent.model
        ):
            agent.switch_model(**_runtime_kwargs(decision.primary))
            switched = True
        yield decision
        _ROUTER.observe(decision.primary, success=True, latency_seconds=monotonic() - started)
    except Exception:
        _ROUTER.observe(decision.primary, success=False)
        raise
    finally:
        if switched:
            try:
                # Rebuild the original client through Hermes' canonical switch
                # path. Never restore a possibly-closed client object.
                agent.switch_model(**original)
            except Exception:
                logger.exception("Smart routing could not restore original runtime")


__all__ = ["smart_route_turn"]
