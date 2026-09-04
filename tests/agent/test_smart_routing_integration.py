from __future__ import annotations

import inspect

from agent.turn_facade import TurnFacadeMixin


def test_turn_facade_routes_before_conversation_loop_call():
    source = inspect.getsource(TurnFacadeMixin.run_conversation)
    route_pos = source.index("with smart_route_turn(self, user_message):")
    loop_pos = source.index("result = run_conversation(")
    assert route_pos < loop_pos


def test_turn_facade_preserves_public_run_conversation_signature():
    signature = inspect.signature(TurnFacadeMixin.run_conversation)
    assert list(signature.parameters)[:2] == ["self", "user_message"]
    assert "conversation_history" in signature.parameters
    assert "stream_callback" in signature.parameters
    assert "moa_config" in signature.parameters
