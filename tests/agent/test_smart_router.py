from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.smart_router import (
    CostPolicy,
    ModelCandidate,
    RoutingRequest,
    SmartRouter,
    TaskClass,
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


def test_coding_is_preferred_for_coding_prompt():
    router = SmartRouter()
    candidates = [
        ModelCandidate("demo", "general", free=True, quality=0.9, coding=0.1),
        ModelCandidate("demo", "coder", free=True, quality=0.7, coding=1.0),
    ]
    decision = router.route(RoutingRequest("please debug this Python code", task_class=TaskClass.CODING), candidates)
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
        ("nvidia", "nvidia/model"),
        ("nvidia", "model-2"),
    }
    assert normalize_free_models(["nvidia/model", "zai/glm-5.2"]) == {
        ("nvidia", "model"),
        ("zai", "glm-5.2"),
    }


def test_router_accepts_modelinfo_like_metadata():
    from agent.smart_router import candidate_from_metadata

    info = SimpleNamespace(
        context_window=128000,
        tool_call=True,
        attachment=True,
        reasoning=True,
        family="test",
    )
    candidate = candidate_from_metadata("demo", "model", info, {("demo", "model")})
    assert candidate.free is True
    assert candidate.context_length == 128000
    assert candidate.supports_tools is True
    assert candidate.supports_vision is True
    assert candidate.supports_reasoning is True
