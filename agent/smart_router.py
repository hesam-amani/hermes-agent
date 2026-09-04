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

    def observe(
        self,
        candidate: ModelCandidate,
        *,
        success: bool,
        latency_seconds: float | None = None,
    ) -> None:
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
        research = any(x in text for x in (
            "research", "sources", "cite", "compare", "investigate", "latest", "paper",
        ))
        vision = any(x in text for x in (
            "image", "photo", "screenshot", "picture", "vision",
        ))
        long_context = any(x in text for x in (
            "long document", "entire repository", "whole codebase", "large context", "all files",
        ))
        reasoning = any(x in text for x in (
            "prove", "derive", "analyze", "reason", "why", "solve", "calculate",
        ))
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

    def filter_candidates(
        self,
        candidates: Iterable[ModelCandidate],
        request: RoutingRequest,
    ) -> list[ModelCandidate]:
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
            if request.min_context_length is not None and (
                candidate.context_length or 0
            ) < request.min_context_length:
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

    def route(
        self,
        request: RoutingRequest,
        candidates: Iterable[ModelCandidate],
    ) -> RoutingDecision:
        task = request.task_class or self.classify(request.prompt)
        filtered = self.filter_candidates(candidates, request)
        if not filtered:
            raise LookupError("No model satisfies the smart-routing policy and capabilities")

        ranked = sorted(
            filtered,
            key=lambda c: (
                self.score(c, task),
                1 if c.free else 0,
                c.provider.lower(),
                c.model.lower(),
            ),
            reverse=True,
        )

        # FREE_PREFERRED keeps every eligible model available but gives explicitly
        # entitled free models a meaningful, deterministic advantage. FREE_ONLY was
        # already filtered above; ANY intentionally makes no cost preference.
        if request.cost_policy is CostPolicy.FREE_PREFERRED:
            ranked = sorted(
                ranked,
                key=lambda c: (
                    self.score(c, task) + (0.12 if c.free else 0.0),
                    1 if c.free else 0,
                    c.provider.lower(),
                    c.model.lower(),
                ),
                reverse=True,
            )

        return RoutingDecision(
            task,
            ranked[0],
            tuple(ranked[1:]),
            {f"{c.provider}/{c.model}": self.score(c, task) for c in ranked},
        )

    def discover_models(self, provider: str, *, force_refresh: bool = False) -> list[str]:
        from hermes_cli.models import provider_model_ids
        return provider_model_ids(provider, force_refresh=force_refresh)


def normalize_free_models(value: Any) -> set[tuple[str, str]]:
    """Parse explicit free entitlements from config; never infer free from price.

    Both ``provider/model`` strings and provider-local model IDs are accepted in
    mappings. A provider prefix is stripped when it matches the mapping key, so
    ``nvidia: [nvidia/foo]`` and ``nvidia: [foo]`` mean the same model.
    """
    result: set[tuple[str, str]] = set()
    if isinstance(value, Mapping):
        for provider, models in value.items():
            provider_name = str(provider).strip().lower()
            if isinstance(models, str):
                models = [models]
            if isinstance(models, Iterable) and not isinstance(models, (bytes, str)):
                for model in models:
                    model_name = str(model).strip()
                    if not model_name or not provider_name:
                        continue
                    prefix = provider_name + "/"
                    if model_name.lower().startswith(prefix):
                        model_name = model_name[len(prefix):]
                    result.add((provider_name, model_name.lower()))
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for item in value:
            text = str(item).strip()
            if "/" in text:
                provider, model = text.split("/", 1)
                if provider and model:
                    result.add((provider.lower(), model.lower()))
    return result


def _metadata_value(metadata: Any, name: str, default: Any = None) -> Any:
    if isinstance(metadata, Mapping):
        return metadata.get(name, default)
    return getattr(metadata, name, default)


def _metadata_bool(metadata: Any, method_name: str, field_name: str, default: bool = False) -> bool:
    method = getattr(metadata, method_name, None)
    if callable(method):
        try:
            return bool(method())
        except Exception:
            pass
    return bool(_metadata_value(metadata, field_name, default))


def _clamp01(value: Any, default: float = 0.5) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _infer_task_scores(provider: str, model: str, metadata: Any) -> tuple[float, float, float, float]:
    """Derive conservative task priors when models.dev has no quality taxonomy.

    These are priors, not claims of benchmark superiority. Explicit metadata fields
    still win when present.
    """
    text = f"{provider} {model} {_metadata_value(metadata, 'family', '')}".lower()
    coding = 0.65 if any(x in text for x in ("coder", "code", "coding", "devstral", "qwen")) else 0.5
    research = 0.60 if any(x in text for x in ("research", "perplexity", "search")) else 0.5
    quality = 0.55 if any(x in text for x in ("pro", "ultra", "opus", "sonnet", "reasoning", "think")) else 0.5
    if any(x in text for x in ("mini", "nano", "flash", "lite", "small")):
        quality = min(quality, 0.52)
    return quality, coding, research, 0.5


def candidate_from_metadata(
    provider: str,
    model: str,
    metadata: Any,
    free_models: set[tuple[str, str]],
) -> ModelCandidate:
    context = _metadata_value(metadata, "context_window") or 0
    try:
        context = int(context)
    except (TypeError, ValueError):
        context = 0

    provider_name = provider.strip().lower()
    model_name = model.strip()
    key = (provider_name, model_name.lower())
    quality, coding, research, latency = _infer_task_scores(provider_name, model_name, metadata)

    explicit_quality = _metadata_value(metadata, "quality", None)
    explicit_coding = _metadata_value(metadata, "coding", None)
    explicit_research = _metadata_value(metadata, "research", None)
    if explicit_quality is not None:
        quality = _clamp01(explicit_quality, quality)
    if explicit_coding is not None:
        coding = _clamp01(explicit_coding, coding)
    if explicit_research is not None:
        research = _clamp01(explicit_research, research)

    modalities = _metadata_value(metadata, "input_modalities", ()) or ()
    try:
        modalities = tuple(str(x).lower() for x in modalities)
    except TypeError:
        modalities = ()

    return ModelCandidate(
        provider=provider,
        model=model,
        free=key in free_models,
        context_length=context or None,
        supports_tools=_metadata_bool(metadata, "supports_tools", "tool_call", False),
        supports_vision=(
            _metadata_bool(metadata, "supports_vision", "attachment", False)
            or any("image" in item for item in modalities)
        ),
        supports_reasoning=_metadata_bool(metadata, "supports_reasoning", "reasoning", False),
        quality=quality,
        coding=coding,
        research=research,
        latency_score=latency,
        extra={
            "family": _metadata_value(metadata, "family", ""),
            "status": _metadata_value(metadata, "status", ""),
            "cost_input": _metadata_value(metadata, "cost_input", 0.0),
            "cost_output": _metadata_value(metadata, "cost_output", 0.0),
        },
    )


def discover_candidates(
    router: SmartRouter,
    providers: Iterable[str],
    free_models: set[tuple[str, str]],
    *,
    force_refresh: bool = False,
) -> list[ModelCandidate]:
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
    "CostPolicy",
    "ModelCandidate",
    "RoutingDecision",
    "RoutingRequest",
    "SmartRouter",
    "TaskClass",
    "candidate_from_metadata",
    "discover_candidates",
    "normalize_free_models",
]
