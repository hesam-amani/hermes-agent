"""Hermes integration bridge for Smart Inference.

This module is intentionally thin.

Smart Inference decides which available model is best for the current turn.
Hermes remains responsible for all runtime/provider behavior.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

from smart_inference.adapter import ( # pyright: ignore[reportMissingImports]
    candidate_from_metadata,
    metadata_is_free,
)
from smart_inference.router import ( # pyright: ignore[reportMissingImports]
    CostPolicy,
    InferenceRequest,
    choose,
)

from agent import models_dev

from .smart_inference_runtime import temporary_model_runtime


logger = logging.getLogger(__name__)


_POLICY_MAP = {
    "free_only": CostPolicy.FREE_ONLY,
    "free_preferred": CostPolicy.FREE_PREFERRED,
    "any": CostPolicy.ANY,
}


def _get_metadata(
    provider: str,
    model: str,
    metadata_source: Any | None = None,
) -> Any | None:
    try:
        source = metadata_source or models_dev
        return source.get_model_info(
            provider,
            model,
            allow_network=False,
        )
    except Exception:
        logger.debug(
            "Smart Inference could not obtain metadata for %s/%s",
            provider,
            model,
            exc_info=True,
        )
        return None


def _candidate(
    provider: str,
    model: str,
    metadata_source: Any | None = None,
) -> Any | None:
    metadata = _get_metadata(
        provider,
        model,
        metadata_source,
    )

    if metadata is None:
        return None

    return candidate_from_metadata(
        provider,
        model,
        metadata,
        free=metadata_is_free(metadata),
    )


def discover_candidates(agent: Any) -> list[Any]:
    """Discover only models Hermes already knows how to route to."""

    routes: list[tuple[str, str]] = []

    provider = getattr(agent, "provider", None)
    model = getattr(agent, "model", None)
    metadata_source = getattr(agent, "models_dev", None)

    if provider and model:
        routes.append((str(provider), str(model)))

    fallback_chain = getattr(agent, "_fallback_chain", None)

    if fallback_chain:
        for item in fallback_chain:
            fallback_provider: str | None = None
            fallback_model: str | None = None

            if isinstance(item, dict):
                fallback_provider = item.get("provider")
                fallback_model = item.get("model")
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                fallback_provider = item[0]
                fallback_model = item[1]
            elif isinstance(item, str):
                # A plain model string can use the current provider.
                fallback_provider = provider
                fallback_model = item

            if fallback_provider and fallback_model:
                routes.append(
                    (
                        str(fallback_provider),
                        str(fallback_model),
                    )
                )

    fallback_model = getattr(agent, "_fallback_model", None)

    if isinstance(fallback_model, dict):
        fallback_provider = fallback_model.get("provider")
        fallback_model_id = fallback_model.get("model")

        if fallback_provider and fallback_model_id:
            routes.append(
                (
                    str(fallback_provider),
                    str(fallback_model_id),
                )
            )

    candidates: list[Any] = []
    seen: set[tuple[str, str]] = set()

    for route_provider, route_model in routes:
        key = (route_provider, route_model)

        if key in seen:
            continue

        seen.add(key)

        candidate = _candidate(
            route_provider,
            route_model,
            metadata_source,
        )

        if candidate is not None:
            candidates.append(candidate)

    return candidates


def _policy(agent: Any) -> CostPolicy:
    value = getattr(
        agent,
        "_smart_inference_policy",
        "free_preferred",
    )

    if isinstance(value, CostPolicy):
        return value

    return _POLICY_MAP.get(
        str(value).lower(),
        CostPolicy.FREE_PREFERRED,
    )


def choose_for_turn(
    agent: Any,
    user_message: str,
) -> Any | None:
    """Return a Smart Inference decision for this turn."""

    candidates = discover_candidates(agent)

    if not candidates:
        return None

    request = InferenceRequest(
        prompt=user_message,
        candidates=candidates,
        cost_policy=_policy(agent),
    )

    return choose(request)


def routing_enabled(agent: Any) -> bool:
    return bool(
        getattr(
            agent,
            "_smart_inference_enabled",
            False,
        )
    )


def explicit_model_selected(agent: Any) -> bool:
    return bool(
        getattr(
            agent,
            "_smart_inference_explicit",
            False,
        )
    )


def route_turn(agent: Any, user_message: str):
    """Return a context manager for the selected route.

    If Smart Inference should not act, this returns a no-op context manager.
    """

    if not routing_enabled(agent):
        return nullcontext()

    if explicit_model_selected(agent):
        return nullcontext()

    if getattr(agent, "api_mode", None) == "codex_app_server":
        return nullcontext()

    try:
        decision = choose_for_turn(
            agent,
            user_message,
        )
    except Exception:
        logger.exception(
            "Smart Inference failed; continuing with Hermes' current route"
        )
        return nullcontext()

    if decision is None or decision.primary is None:
        return nullcontext()

    current_provider = str(
        getattr(agent, "provider", "")
    )
    current_model = str(
        getattr(agent, "model", "")
    )

    selected_provider = str(
        decision.primary.ref.provider
    )
    selected_model = str(
        decision.primary.ref.model
    )

    if (
        selected_provider == current_provider
        and selected_model == current_model
    ):
        return nullcontext()

    logger.info(
        "Smart Inference selected %s/%s for task=%s",
        selected_provider,
        selected_model,
        getattr(decision.task, "value", decision.task),
    )

    return temporary_model_runtime(
        agent,
        selected_provider,
        selected_model,
    )
