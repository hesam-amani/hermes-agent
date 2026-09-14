from types import SimpleNamespace


def _metadata(*, free=False, reasoning=False, tools=True, vision=False, context=128_000):
    return SimpleNamespace(
        input_cost=0 if free else 1,
        output_cost=0 if free else 1,
        reasoning=reasoning,
        tool_call=tools,
        attachment=vision,
        input_modalities=("text", "image") if vision else ("text",),
        structured_output=True,
        context_window=context,
        family="reasoning" if reasoning else "general",
    )


def test_discover_candidates_uses_only_hermes_routes(monkeypatch):
    from agent import smart_inference_bridge as bridge

    metadata = {
        ("primary", "model-a"): _metadata(free=True, reasoning=True),
        ("fallback", "model-b"): _metadata(free=True, reasoning=False),
    }

    class ModelsDev:
        @staticmethod
        def get_model_info(provider, model, allow_network=False):
            return metadata.get((provider, model))

    agent = SimpleNamespace(
        provider="primary",
        model="model-a",
        _fallback_chain=[{"provider": "fallback", "model": "model-b"}],
        _fallback_model=None,
        models_dev=ModelsDev(),
    )

    candidates = bridge.discover_candidates(agent)
    assert [candidate.ref.key for candidate in candidates] == [
        "primary/model-a",
        "fallback/model-b",
    ]


def test_route_turn_is_fail_open_when_disabled():
    from agent.smart_inference_bridge import route_turn

    agent = SimpleNamespace(_smart_inference_enabled=False)
    with route_turn(agent, "solve this"):
        pass


def test_route_turn_does_not_override_explicit_selection():
    from agent.smart_inference_bridge import route_turn

    agent = SimpleNamespace(
        _smart_inference_enabled=True,
        _smart_inference_explicit=True,
    )
    with route_turn(agent, "solve this"):
        pass
