from types import SimpleNamespace

import pytest

from agent import smart_inference_runtime as bridge


@pytest.fixture
def runtime_stubs(monkeypatch):
    import agent.agent_runtime_helpers as runtime

    calls = []

    def snapshot(agent):
        return {"model": agent.model, "provider": agent.provider, "api_key": agent.api_key}

    def resolve_destination(agent, model, provider, base_url, api_mode, capabilities, old_norm, new_norm):
        calls.append(("resolve", model, provider))
        return api_mode or "chat_completions", base_url or "https://example.test/v1", capabilities or {"tools": True}

    def swap(agent, model, provider, api_key, base_url, api_mode, old_provider, old_norm, new_norm):
        calls.append(("swap", model, provider))
        agent.model = model
        agent.provider = provider
        agent.api_key = api_key or agent.api_key
        agent.base_url = base_url
        agent.api_mode = api_mode

    def resolve_context(agent, snapshot):
        return None, 8192

    def update_compressor(agent, custom_providers, context_length, snapshot):
        calls.append(("compressor", context_length))
        agent.context_compressor.model = agent.model
        agent.context_compressor.context_length = context_length

    def finish(agent, provider, old_norm, new_norm):
        calls.append(("finish", provider))
        agent._fallback_chain = [{"provider": provider, "model": agent.model}]
        agent.request_overrides = {"extra_body": {"route": provider}}

    monkeypatch.setattr(runtime, "_snapshot_switch_state", snapshot)
    monkeypatch.setattr(runtime, "_resolve_switch_destination", resolve_destination)
    monkeypatch.setattr(runtime, "_swap_switch_runtime", swap)
    monkeypatch.setattr(runtime, "_resolve_switch_context_length", resolve_context)
    monkeypatch.setattr(runtime, "_update_switch_compressor", update_compressor)
    monkeypatch.setattr(runtime, "_finish_switch", finish)
    return calls


def make_agent():
    return SimpleNamespace(
        model="primary-model",
        provider="primary-provider",
        requested_provider="primary-provider",
        api_key="primary-key",
        base_url="https://primary.test/v1",
        api_mode="chat_completions",
        client=object(),
        _anthropic_client=None,
        _anthropic_api_key="",
        _anthropic_base_url="",
        _is_anthropic_oauth=False,
        _config_context_length=None,
        _reasoning_echo_flag=False,
        runtime_capabilities={"tools": True},
        _credential_pool="primary-pool",
        _credential_pool_entry_id="primary-entry",
        _client_kwargs={"api_key": "primary-key"},
        _use_prompt_caching=True,
        _use_native_cache_layout=False,
        _cached_system_prompt="primary-prompt",
        reasoning_config={"effort": "medium"},
        request_overrides={"extra_body": {"route": "primary"}},
        _custom_providers=[{"name": "primary"}],
        _fallback_chain=[{"provider": "fallback", "model": "fallback-model"}],
        _fallback_model={"provider": "fallback", "model": "fallback-model"},
        _fallback_index=2,
        _fallback_activated=True,
        _provider_fallback_active=True,
        _provider_fallback_route=("primary-model", "primary-provider"),
        _primary_runtime={"model": "primary-model", "provider": "primary-provider"},
        _transport_cache={"primary": object()},
        context_compressor=SimpleNamespace(
            model="primary-model",
            context_length=4096,
            base_url="https://primary.test/v1",
            api_key="primary-key",
            provider="primary-provider",
            api_mode="chat_completions",
            threshold_tokens=1000,
        ),
    )


def test_runtime_is_temporary_and_restores_everything(runtime_stubs):
    agent = make_agent()
    before = {
        name: getattr(agent, name)
        for name in bridge._RUNTIME_FIELDS
    }
    compressor_before = vars(agent.context_compressor).copy()

    with bridge.temporary_model_runtime(
        agent,
        model="smart-model",
        provider="smart-provider",
        api_key="smart-key",
    ):
        assert agent.model == "smart-model"
        assert agent.provider == "smart-provider"
        assert agent.request_overrides == {"extra_body": {"route": "smart-provider"}}
        assert agent.context_compressor.model == "smart-model"

    for name, expected in before.items():
        assert getattr(agent, name) == expected
    assert vars(agent.context_compressor) == compressor_before


def test_runtime_restores_after_exception(runtime_stubs):
    agent = make_agent()
    original = (agent.model, agent.provider, agent.api_key)

    with pytest.raises(RuntimeError):
        with bridge.temporary_model_runtime(agent, model="smart", provider="provider"):
            assert agent.model == "smart"
            raise RuntimeError("turn failed")

    assert (agent.model, agent.provider, agent.api_key) == original
