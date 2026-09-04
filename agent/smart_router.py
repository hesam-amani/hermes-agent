"""Task-aware smart model router for Hermes Agent.

The router decides which already-configured provider/model should handle a turn.
Hermes remains the owner of credentials, clients, transports, retries, and
fallback execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, Iterable, Mapping


class CostPolicy(str, Enum):
    FREE_ONLY = "free_only"
    FREE_PREFERRED = "free_preferred"
    ANY = "any"


class TaskClass(str, Enum):
    CODING = "coding"
    REASONING = "reasoning"
    RESEARCH = "research"
    LONG_CONTEXT = "long_context"
    VISION = "vision"
    CASUAL = "casual"
    GENERAL = "general"


@dataclass(frozen=True)
class ModelCandidate:
    provider: str
    model: str
    free: bool = False
    context_length: int | None = None
    supports_tools: bool = False
    supports_vision: bool = False
    supports_reasoning: bool = False
    quality: float = 0.5
    coding: float = 0.5
    research: float = 0.5
    latency_score: float = 0.5
    reliability_score: float = 0.5
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingRequest:
    prompt: str
    cost_policy: CostPolicy = CostPolicy.FREE_ONLY
    requires_tools: bool = False
    requires_vision: bool = False
    requires_reasoning: bool = False
    min_context_length: int | None = None
    task_class: TaskClass | None = None


@dataclass(frozen=True)
class RoutingDecision:
    task_class: TaskClass
    primary: ModelCandidate
    fallbacks: tuple[ModelCandidate, ...]
    scores: dict[str, float]

    @property
    def chain(self) -> tuple[ModelCandidate, ...]:
        return (self.primary, *self.fallbacks)


@dataclass
class _Health:
    successes: int = 0
    failures: int = 0
    latency_total: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total else 0.5

    @property
    def latency_score(self) -> float:
        if not self.successes:
            return 0.5
        avg = self.latency_total / self.successes
        return 1.0 / (1.0 + max(0.0, avg))


class SmartRouter:
    """Deterministic task/capability/cost/health ranking layer."""

    def __init__(self) -> None:
        self._health: dict[tuple[str, str], _Health] = {}

    def observe(self, candidate: ModelCandidate, *, success: bool, latency_seconds: float | None = None) -> None:
        health = self._health.setdefault((candidate.provider, candidate.model), _Health())
        if success:
            health.successes += 1
            if latency_seconds is not None:
                health.latency_total += max(0.0, latency_seconds)
        else:
            health.failures += 1

    def classify(self, prompt: str) -> TaskClass:
        text = str(prompt or "").lower()
        coding = "```" in text or any(x in text for x in (
            "code", "python", "typescript", "javascript", "bug", "compile", "function", "api",
            "pytest", "git", "refactor", "stack trace", "traceback", "repository", "repo",
        ))
        research = any(x in text for x in ("research", "sources", "cite", "compare", "investigate", "latest", "paper"))
        vision = any(x in text for x in ("image", "photo", "screenshot", "picture", "vision"))
        long_context = any(x in text for x in ("long document", "entire repository", "whole codebase", "large context", "all files"))
        reasoning = any(x in text for x in ("prove", "derive", "analyze", "reason", "why", "solve", "calculate"))
        if vision:
            return TaskClass.VISION
        if coding:
            return TaskClass.CODING
        if research:
            return TaskClass.RESEARCH
        if long_context:
            return TaskClass.LONG_CONTEXT
        if reasoning:
            return TaskClass.REASONING
        if len(text.split()) < 20:
            return TaskClass.CASUAL
        return TaskClass.GENERAL

    def filter_candidates(self, candidates: Iterable[ModelCandidate], request: RoutingRequest) -> list[ModelCandidate]:
        result: list[ModelCandidate] = []
        for candidate in candidates:
            if request.cost_policy is CostPolicy.FREE_ONLY and not candidate.free:
                continue
            if request.requires_tools and not candidate.supports_tools:
                continue
            if request.requires_vision and not candidate.supports_vision:
                continue
            if request.requires_reasoning and not candidate.supports_reasoning:
                continue
            if request.min_context_length is not None and (candidate.context_length or 0) < request.min_context_length:
                continue
            result.append(candidate)
        return result

    def score(self, candidate: ModelCandidate, task: TaskClass) -> float:
        health = self._health.get((candidate.provider, candidate.model))
        reliability = health.success_rate if health else candidate.reliability_score
        latency = health.latency_score if health else candidate.latency_score
        task_fit = {
            TaskClass.CODING: candidate.coding,
            TaskClass.RESEARCH: candidate.research,
            TaskClass.REASONING: 0.5 * float(candidate.supports_reasoning) + 0.5 * candidate.quality,
            TaskClass.VISION: float(candidate.supports_vision),
            TaskClass.LONG_CONTEXT: min((candidate.context_length or 0) / 200_000, 1.0),
            TaskClass.CASUAL: candidate.quality,
            TaskClass.GENERAL: candidate.quality,
        }[task]
        return 0.40 * task_fit + 0.25 * candidate.quality + 0.20 * reliability + 0.15 * latency

    def route(self, request: RoutingRequest, candidates: Iterable[ModelCandidate]) -> RoutingDecision:
        task = request.task_class or self.classify(request.prompt)
        filtered = self.filter_candidates(candidates, request)
        if not filtered:
            raise LookupError("No model satisfies the smart-routing policy and capabilities")
        ranked = sorted(filtered, key=lambda c: self.score(c, task), reverse=True)
        return RoutingDecision(task, ranked[0], tuple(ranked[1:]), {
            f"{c.provider}/{c.model}": self.score(c, task) for c in ranked
        })

    def discover_models(self, provider: str, *, force_refresh: bool = False) -> list[str]:
        from hermes_cli.models import provider_model_ids
        return provider_model_ids(provider, force_refresh=force_refresh)


def normalize_free_models(value: Any) -> set[tuple[str, str]]:
    """Parse explicit free entitlements from config; never infer free from price."""
    result: set[tuple[str, str]] = set()
    if isinstance(value, Mapping):
        for provider, models in value.items():
            if isinstance(models, str):
                models = [models]
            if isinstance(models, Iterable):
                for model in models:
                    if str(model).strip():
                        result.add((str(provider).strip().lower(), str(model).strip().lower()))
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for item in value:
            text = str(item).strip()
            if "/" in text:
                provider, model = text.split("/", 1)
                if provider and model:
                    result.add((provider.lower(), model.lower()))
    return result


def candidate_from_metadata(provider: str, model: str, metadata: Any, free_models: set[tuple[str, str]]) -> ModelCandidate:
    def value(name: str, default: Any = None) -> Any:
        if isinstance(metadata, Mapping):
            return metadata.get(name, default)
        return getattr(metadata, name, default)
    context = value("context_window") or 0
    try:
        context = int(context)
    except (TypeError, ValueError):
        context = 0
    key = (provider.strip().lower(), model.strip().lower())
    return ModelCandidate(
        provider=provider, model=model, free=key in free_models,
        context_length=context or None,
        supports_tools=bool(value("tool_call", False)),
        supports_vision=bool(value("attachment", False) or "image" in tuple(value("input_modalities", ()) or ())),
        supports_reasoning=bool(value("reasoning", False)),
        quality=float(value("quality", 0.5) or 0.5),
        coding=float(value("coding", 0.5) or 0.5),
        research=float(value("research", 0.5) or 0.5),
        extra={
            "family": value("family", ""),
            "status": value("status", ""),
            "cost_input": value("cost_input", 0.0),
            "cost_output": value("cost_output", 0.0),
        },
    )


def discover_candidates(router: SmartRouter, providers: Iterable[str], free_models: set[tuple[str, str]], *, force_refresh: bool = False) -> list[ModelCandidate]:
    from agent.models_dev import get_model_info
    candidates: list[ModelCandidate] = []
    for provider in providers:
        try:
            models = router.discover_models(provider, force_refresh=force_refresh)
        except Exception:
            continue
        for model in models:
            try:
                metadata = get_model_info(provider, model)
            except Exception:
                metadata = None
            candidates.append(candidate_from_metadata(provider, model, metadata, free_models))
    return candidates


__all__ = [
    "CostPolicy", "ModelCandidate", "RoutingDecision", "RoutingRequest", "SmartRouter", "TaskClass",
    "candidate_from_metadata", "discover_candidates", "normalize_free_models",
]
