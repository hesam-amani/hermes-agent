from contextlib import contextmanager
from types import SimpleNamespace

import pytest


def _turn_context():
    return SimpleNamespace(
        user_message="solve this",
        original_user_message="solve this",
        conversation_history=[],
        effective_task_id=None,
        turn_id="turn-test",
        should_review_memory=False,
        plugin_user_context=None,
        ext_prefetch_cache=None,
        messages=[],
        active_system_prompt="",
        current_turn_user_idx=0,
        preflight_compression_blocked=False,
    )


def _agent():
    return SimpleNamespace(
        _last_compaction_in_place=False,
        _last_compression_attempt_recorded=False,
        _last_compression_attempt_in_place=None,
        _delivered_interim_texts=set(),
        _incremental_persistence_failed=False,
        _last_persistence_error_cause=None,
        _compression_adoption_failed=False,
        _ephemeral_reasoning_off=False,
        _auth_pool_refresh_counts={},
        _last_turn_usage=None,
        max_compression_attempts=3,
        api_mode="codex_app_server",
        model="original-model",
        provider="original-provider",
        max_iterations=1,
        iteration_budget=SimpleNamespace(remaining=1),
        _budget_grace_call=False,
    )


def _patch_turn_setup(monkeypatch, conversation_loop):
    monkeypatch.setattr(
        conversation_loop,
        "begin_fast_mode_turn",
        lambda agent, history: None,
    )
    monkeypatch.setattr(
        conversation_loop,
        "build_turn_context",
        lambda *args, **kwargs: _turn_context(),
    )

    def refresh():
        return None

    return refresh


def test_conversation_turn_uses_temporary_smart_inference_runtime(monkeypatch):
    from agent import conversation_loop
    import agent.smart_inference_bridge as bridge

    agent = _agent()
    observed = []

    _patch_turn_setup(monkeypatch, conversation_loop)
    monkeypatch.setattr(
        agent,
        "_try_refresh_env_client_credentials",
        lambda: None,
        raising=False,
    )

    @contextmanager
    def fake_runtime(agent, user_message):
        observed.append(("enter", agent.model, agent.provider, user_message))
        old_model = agent.model
        old_provider = agent.provider

        agent.model = "selected-model"
        agent.provider = "selected-provider"

        try:
            yield
        finally:
            observed.append(("exit", agent.model, agent.provider))
            agent.model = old_model
            agent.provider = old_provider

    monkeypatch.setattr(bridge, "route_turn", fake_runtime)

    def codex_turn(**kwargs):
        observed.append(("codex", agent.model, agent.provider))
        return {"response": "ok"}

    agent._run_codex_app_server_turn = codex_turn

    result = conversation_loop._run_conversation_turn(
        agent,
        "solve this",
    )

    assert result == {"response": "ok"}
    assert observed == [
        ("enter", "original-model", "original-provider", "solve this"),
        ("codex", "selected-model", "selected-provider"),
        ("exit", "selected-model", "selected-provider"),
    ]
    assert agent.model == "original-model"
    assert agent.provider == "original-provider"


def test_conversation_turn_restores_runtime_after_turn_exception(monkeypatch):
    from agent import conversation_loop
    import agent.smart_inference_bridge as bridge

    agent = _agent()
    observed = []

    _patch_turn_setup(monkeypatch, conversation_loop)
    monkeypatch.setattr(
        agent,
        "_try_refresh_env_client_credentials",
        lambda: None,
        raising=False,
    )

    @contextmanager
    def fake_runtime(agent, user_message):
        old_model = agent.model
        old_provider = agent.provider

        agent.model = "selected-model"
        agent.provider = "selected-provider"
        observed.append(("enter", agent.model, agent.provider))

        try:
            yield
        finally:
            observed.append(("restore", agent.model, agent.provider))
            agent.model = old_model
            agent.provider = old_provider

    monkeypatch.setattr(bridge, "route_turn", fake_runtime)

    def codex_turn(**kwargs):
        observed.append(("codex", agent.model, agent.provider))
        raise RuntimeError("simulated turn failure")

    agent._run_codex_app_server_turn = codex_turn

    with pytest.raises(RuntimeError, match="simulated turn failure"):
        conversation_loop._run_conversation_turn(
            agent,
            "solve this",
        )

    assert observed == [
        ("enter", "selected-model", "selected-provider"),
        ("codex", "selected-model", "selected-provider"),
        ("restore", "selected-model", "selected-provider"),
    ]
    assert agent.model == "original-model"
    assert agent.provider == "original-provider"
