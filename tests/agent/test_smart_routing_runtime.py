from __future__ import annotations

from agent import smart_routing_runtime as runtime
from agent.smart_router import CostPolicy, ModelCandidate, RoutingDecision, TaskClass


class FakeAgent:
    def __init__(self):
        self.provider = "demo"
        self.model = "old"
        self.api_key = "key"
        self.base_url = "https://demo.invalid/v1"
        self.api_mode = "chat_completions"
        self.tools = []
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


def _decision(primary_model="new"):
    primary = ModelCandidate("demo", primary_model, free=True, quality=1.0)
    fallback = ModelCandidate("demo", "backup", free=True, quality=0.9)
    return RoutingDecision(TaskClass.GENERAL, primary, (fallback,), {f"demo/{primary_model}": 1.0})


def _configure(monkeypatch, decision):
    monkeypatch.setattr(runtime, "_config", lambda: {
        "enabled": True,
        "providers": ["demo"],
        "free_models": ["demo/new", "demo/backup", "demo/old"],
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


def test_smart_route_turn_activates_and_restores(monkeypatch):
    agent = FakeAgent()
    decision = _decision()
    _configure(monkeypatch, decision)

    with runtime.smart_route_turn(agent, "hello") as selected:
        assert selected is decision
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
    _configure(monkeypatch, decision)

    with runtime.smart_route_turn(agent, "hello"):
        assert agent._fallback_chain == [{"provider": "demo", "model": "backup"}]

    assert agent._fallback_chain == [{"provider": "paid", "model": "expensive"}]


def test_same_primary_is_restored_if_native_fallback_changes_runtime(monkeypatch):
    agent = FakeAgent()
    decision = _decision(primary_model="old")
    _configure(monkeypatch, decision)

    with runtime.smart_route_turn(agent, "hello"):
        # Simulate Hermes' native fallback activating during the turn. The router
        # did not need to switch the primary, but the automatic route is still
        # turn-scoped and must restore the original model.
        agent.model = "backup"
        agent.provider = "demo"
        agent.base_url = "https://backup.invalid/v1"
        agent.api_mode = "chat_completions"

    assert agent.model == "old"
    assert agent.base_url == "https://demo.invalid/v1"
    assert agent.switches[-1]["new_model"] == "old"


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
