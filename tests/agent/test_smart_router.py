from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.smart_router import (
    CostPolicy,
    ModelCandidate,
    RoutingRequest,
    SmartRouter,
    TaskClass,
    candidate_from_metadata,
    normalize_free_models,
)


def test_free_only_never_infers_free_from_missing_metadata():
    router = SmartRouter()
    candidates = [ModelCandidate("demo", "model", free=False)]
    with pytest.raises(LookupError):
        router.route(RoutingRequest("hello", cost_policy=CostPolicy.FREE_ONLY), candidates)


def test_free_only_accepts_explicit_entitlement():
    router = SmartRouter()
    candidates = [ModelCandidate("demo", "model", free=True)]
    decision = router.route(RoutingRequest("hello", cost_policy=CostPolicy.FREE_ONLY), candidates)
    assert decision.primary.model == "model"


def test_free_preferred_favors_free_when_quality_is_close():
    router = SmartRouter()
    paid = ModelCandidate("demo", "paid", free=False, quality=0.60)
    free = ModelCandidate("demo", "free", free=True, quality=0.51)
    decision = router.route(
        RoutingRequest("hello", cost_policy=CostPolicy.FREE_PREFERRED),
        [paid, free],
    )
    assert decision.primary.model == "free"


def test_any_does_not_apply_free_preference():
    router = SmartRouter()
    paid = ModelCandidate("demo", "paid", free=False, quality=0.60)
    free = ModelCandidate("demo", "free", free=True, quality=0.51)
    decision = router.route(
        RoutingRequest("hello", cost_policy=CostPolicy.ANY),
        [paid, free],
    )
    assert decision.primary.model == "paid"


def test_coding_is_preferred_for_coding_prompt():
    router = SmartRouter()
    candidates = [
        ModelCandidate("demo", "general", free=True, quality=0.9, coding=0.1),
        ModelCandidate("demo", "coder", free=True, quality=0.7, coding=1.0),
    ]
    decision = router.route(
        RoutingRequest("please debug this Python code", task_class=TaskClass.CODING),
        candidates,
    )
    assert decision.primary.model == "coder"


def test_health_observations_change_ranking():
    router = SmartRouter()
    slow = ModelCandidate("demo", "slow", free=True, quality=0.9)
    fast = ModelCandidate("demo", "fast", free=True, quality=0.9)
    router.observe(slow, success=False)
    router.observe(fast, success=True, latency_seconds=0.01)
    decision = router.route(RoutingRequest("hello"), [slow, fast])
    assert decision.primary.model == "fast"


def test_normalize_free_models_supports_mapping_and_provider_model_strings():
    assert normalize_free_models({"nvidia": ["nvidia/model", "model-2"]}) == {
        ("nvidia", "model"),
        ("nvidia", "model-2"),
    }
    assert normalize_free_models(["nvidia/model", "zai/glm-5.2"]) == {
        ("nvidia", "model"),
        ("zai", "glm-5.2"),
    }


def test_modelinfo_method_based_capabilities_are_supported():
    info = SimpleNamespace(
        context_window=128000,
        family="test",
        input_modalities=("text", "image"),
    )
    info.supports_tools = lambda: True
    info.supports_vision = lambda: True
    info.supports_reasoning = lambda: True
    candidate = candidate_from_metadata("demo", "model", info, {("demo", "model")})
    assert candidate.free is True
    assert candidate.context_length == 128000
    assert candidate.supports_tools is True
    assert candidate.supports_vision is True
    assert candidate.supports_reasoning is True
