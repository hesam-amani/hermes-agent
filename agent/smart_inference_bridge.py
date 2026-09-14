"""Small Hermes-facing bridge for per-turn Smart Inference routing.

This module owns no transport logic. It only turns Hermes' already-configured
primary/fallback routes into Smart Inference candidates and returns a decision.
Runtime activation is handled separately by ``smart_inference_runtime``.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)


def _configured_routes(agent: Any) -> list[tuple[str, str]]:
    """Return Hermes routes already present in the live runtime, deduplicated."""
    routes: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(provider: Any, model: Any) -> None:
        provider = str(provider or "").strip()
        model = str(model or "").strip()
        if not provider or not model:
            return
        key = (provider.lower(), model.lower())
        if key not in seen:
            seen.add(key)
            routes.append((provider, model))

    add(getattr(agent, "provider", ""), getattr(agent, "model", ""))
    for entry in getattr(agent, "_fallback_chain", []) or []:
        if isinstance(entry, dict):
            add(entry.get("provider"), entry.get("model"))

    fallback = getattr(agent, "_fallback_model", None)
    if isinstance(fallback, dict):
        add(fallback.get("provider"), fallback.get("model"))

    return routes


def _is_free(metadata: Any) -> bool:
    """Infer zero-cost status only from explicit Hermes metadata."""
    for input_name, output_name in (
        ("input_cost", "output_cost"),
        ("input_price", "output_price"),
    ):
        try:
            input_cost = getattr(metadata, input_name, None)
            output_cost = getattr(metadata, output_name, None)
            if input_cost is not None and output_cost is not None:
                return float(input_cost) == 0.0 and float(output_cost) == 0.0
        except (TypeError, ValueError):
            continue
    return False


def discover_candidates(agent: Any) -> list[Any]:
    """Build Smart Inference candidates from Hermes-known routes only."""
    try:
        from smart_inference import candidate_from_metadata
    except ImportError:
        logger.debug("Smart Inference package is not installed")
        return []

    candidates = []
    for provider, model in _configured_routes(agent):
        try:
            metadata = agent.models_dev.get_model_info(provider, model, allow_network=False)
            if metadata is None:
                continue
            candidates.append(
                candidate_from_metadata(
                    provider,
                    model,
                    metadata,
                    free=_is_free(metadata),
                )
            )
        except Exception:
            logger.debug("Could not inspect Smart Inference route %s/%s", provider, model, exc_info=True)
    return candidates


def choose_for_turn(agent: Any, prompt: str, *, cost_policy: str = "free_preferred") -> Any:
    """Return a decision for a turn, or ``None`` when routing should fail open."""
    try:
        from smart_inference import CostPolicy, InferenceRequest, choose

        policy = CostPolicy(cost_policy)
        candidates = discover_candidates(agent)
        if not candidates:
            return None
        return choose(InferenceRequest(prompt=prompt, cost_policy=policy), candidates)
    except Exception:
        logger.debug("Smart Inference decision failed; keeping Hermes primary", exc_info=True)
        return None


def route_turn(agent: Any, prompt: str, *, cost_policy: str = "free_preferred"):
    """Context manager for one automatic route; explicit runtime remains authoritative.

    The caller must opt in by setting ``agent._smart_inference_enabled``. A turn that
    already has an explicit model selection is never overridden.
    """
    from contextlib import nullcontext

    if not getattr(agent, "_smart_inference_enabled", False):
        return nullcontext()
    if getattr(agent, "_smart_inference_explicit", False):
        return nullcontext()
    if getattr(agent, "api_mode", "") == "codex_app_server":
        return nullcontext()

    decision = choose_for_turn(agent, prompt, cost_policy=cost_policy)
    if decision is None:
        return nullcontext()

    current = (str(getattr(agent, "provider", "")), str(getattr(agent, "model", "")))
    if (decision.primary.provider, decision.primary.model) == current:
        return nullcontext()

    from agent.smart_inference_runtime import temporary_model_runtime

    logger.info(
        "Smart Inference: %s -> %s (%s)",
        f"{current[0]}/{current[1]}",
        decision.primary.key,
        decision.rationale,
    )
    return temporary_model_runtime(
        agent,
        model=decision.primary.model,
        provider=decision.primary.provider,
    )


__all__ = ["discover_candidates", "choose_for_turn", "route_turn"]
