from __future__ import annotations

from types import SimpleNamespace

from agent import smart_routing_runtime as runtime
from agent.smart_router import CostPolicy, ModelCandidate, RoutingDecision, TaskClass


class FakeAgent:
    def __init__(self):
        self.provider = "demo"
        self.model = "old"
        self.api_key = "key"
        self.base_url = "https://demo.invalid/v1"
        self.api_mode = "chat_completions"
        self._fallback_chain = [{"provider": "demo", "model": "old-fallback"}]
        self._fallback_index = 2
        self._unavailable_fallback_keys = {("demo", "old-fallback")}
        self.switches = []

    def switch_model(self, **kwargs):
        self.switches.append(dict(kwargs))
        self.model = kwargs["new_model"]
        self.provider = kwargs["new_provider"]
        self.api_key = kwargs.get("api_key", "")
        self.base_url = kwargs.get("base_url", "")
        self.api_mode = kwargs.get("api_mode", "")


def _decision():
    primary = ModelCandidate("demo", "new", free=True, quality=1.0)
    fallback = ModelCandidate("demo", "backup", free=True, quality=0.9)
    return RoutingDecision(TaskClass.GENERAL, primary, (fallback,), {"demo/new": 1.0})


def test_smart_route_turn_activates_and_restores(monkeypatch):
    agent = FakeAgent()
    decision = _decision()

    monkeypatch.setattr(runtime, "_config", lambda: {
        "enabled": True,
        "providers": ["demo"],
        "free_models": ["demo/new", "demo/backup"],
        "cost_policy": "free_only",
    })
    monkeypatch.setattr(runtime, "discover_candidates", lambda *args, **kwargs: list(decision.chain))
    monkeypatch.setattr(runtime._ROUTER, "route", lambda request, candidates: decision)
    monkeypatch.setattr(runtime, "_runtime_kwargs", lambda candidate: {
        "new_model": candidate.model,
        "new_provider": candidate.provider,
        "api_key": "new-key",
        "base_url": "https://new.invalid/v1",
        "api_mode": "chat_completions",
    })

    with runtime.smart_route_turn(agent, "hello") as selected:
        assert selected is decision
        assert agent.provider == "demo"
        assert agent.model == "new"
        assert agent._fallback_chain == [{"provider": "demo", "model": "backup"}]
        assert agent._fallback_index == 0

    assert agent.model == "old"
    assert agent.provider == "demo"
    assert agent._fallback_chain == [{"provider": "demo", "model": "old-fallback"}]
    assert agent._fallback_index == 2
    assert agent._unavailable_fallback_keys == {("demo", "old-fallback")}
    assert [call["new_model"] for call in agent.switches] == ["new", "old"]


def test_free_only_does_not_restore_paid_fallbacks(monkeypatch):
    agent = FakeAgent()
    agent._fallback_chain = [{"provider": "paid", "model": "expensive"}]
    decision = _decision()

    monkeypatch.setattr(runtime, "_config", lambda: {
        "enabled": True,
        "providers": ["demo"],
        "free_models": ["demo/new", "demo/backup"],
        "cost_policy": CostPolicy.FREE_ONLY.value,
    })
    monkeypatch.setattr(runtime, "discover_candidates", lambda *args, **kwargs: list(decision.chain))
    monkeypatch.setattr(runtime._ROUTER, "route", lambda request, candidates: decision)
    monkeypatch.setattr(runtime, "_runtime_kwargs", lambda candidate: {
        "new_model": candidate.model,
        "new_provider": candidate.provider,
        "api_key": "key",
        "base_url": "https://demo.invalid/v1",
        "api_mode": "chat_completions",
    })

    with runtime.smart_route_turn(agent, "hello"):
        assert agent._fallback_chain == [{"provider": "demo", "model": "backup"}]

    assert agent._fallback_chain == [{"provider": "paid", "model": "expensive"}]


def test_disabled_routing_is_noop(monkeypatch):
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "_config", lambda: {"enabled": False})

    with runtime.smart_route_turn(agent, "please do something") as decision:
        assert decision is None

    assert agent.switches == []
    assert agent.model == "old"


def test_routing_failure_is_fail_open(monkeypatch):
    agent = FakeAgent()
    monkeypatch.setattr(runtime, "_config", lambda: {
        "enabled": True,
        "providers": ["demo"],
        "free_models": ["demo/new"],
    })

    def explode(*args, **kwargs):
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(runtime, "discover_candidates", explode)

    with runtime.smart_route_turn(agent, "hello") as decision:
        assert decision is None

    assert agent.switches == []
    assert agent.model == "old"
