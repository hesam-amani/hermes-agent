#!/usr/bin/env python3
"""Apply the SmartRouter turn hook to a Hermes checkout.

Run from the Hermes repository root. The script is intentionally idempotent and
fails closed if the expected conversation-loop seam has changed upstream.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "agent" / "conversation_loop.py"

MARKER = '__all__ = ["run_conversation"]'
OLD_DEF = "def run_conversation(\n"
NEW_DEF = "def _run_conversation_impl(\n"
WRAPPER = '''def run_conversation(*args, **kwargs):
    """Run one Hermes turn through the optional SmartRouter.

    Routing happens once around the complete turn. Hermes continues to own
    retries, credential rotation, tool loops, and provider fallback behavior.
    """
    agent = args[0] if args else kwargs.get("agent")
    user_message = args[1] if len(args) > 1 else kwargs.get("user_message", "")
    if agent is None:
        return _run_conversation_impl(*args, **kwargs)

    from agent.smart_routing_runtime import smart_route_turn
    with smart_route_turn(agent, user_message):
        return _run_conversation_impl(*args, **kwargs)


'''


def main() -> None:
    text = LOOP.read_text()

    if "def _run_conversation_impl(\n" in text:
        print("SmartRouter turn hook is already applied.")
        return

    if text.count(OLD_DEF) != 1:
        raise SystemExit(
            f"Refusing to patch: expected exactly one {OLD_DEF!r}, found {text.count(OLD_DEF)}."
        )
    if MARKER not in text:
        raise SystemExit("Refusing to patch: conversation_loop export marker is missing.")

    text = text.replace(OLD_DEF, NEW_DEF, 1)
    text = text.replace(MARKER, WRAPPER + MARKER, 1)
    LOOP.write_text(text)
    print(f"Patched {LOOP}")


if __name__ == "__main__":
    main()
