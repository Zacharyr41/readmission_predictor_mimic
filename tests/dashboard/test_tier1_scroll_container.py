"""Tier 1 (cheap, ungated) — chat history lives in a scrollable container.

Regression guard for the "scroll isn't on, conversation glitches past two
messages" bug. The fix wraps the chat transcript in a fixed-pixel-height
``st.container(height=...)`` so the message area scrolls *within itself*
(auto-anchoring to the latest turn) instead of growing the page and making
each rerun jump the viewport.

In Streamlit's ``AppTest`` element tree a fixed-height container surfaces as
a ``Block`` whose ``proto.height_config`` selects ``pixel_height`` (a default
container selects ``use_content``/``use_stretch`` instead). We assert that
such a container exists *and* contains the chat messages — no pipeline, no
Anthropic API, no DuckDB, so it runs in the default invocation.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.conversational.models import AnswerResult

_APP = str(
    Path(__file__).resolve().parents[2] / "src" / "conversational" / "app.py"
)


def _seed_messages() -> list[dict]:
    """A three-turn transcript — past the two-message glitch threshold."""
    return [
        {"role": "user", "content": "How many sepsis patients?"},
        {"role": "assistant", "content": AnswerResult(text_summary="412 patients.")},
        {"role": "user", "content": "What is their mean lactate?"},
        {"role": "assistant", "content": AnswerResult(text_summary="2.4 mmol/L.")},
        {"role": "user", "content": "And their readmission rate?"},
        {"role": "assistant", "content": AnswerResult(text_summary="18%.")},
    ]


def _run_seeded() -> AppTest:
    at = AppTest.from_file(_APP, default_timeout=30)
    at.session_state["messages"] = _seed_messages()
    return at.run()


def _scroll_containers(node, acc=None):
    """Blocks with an explicit positive ``pixel_height`` (i.e. scrollable)."""
    if acc is None:
        acc = []
    proto = getattr(node, "proto", None)
    hc = getattr(proto, "height_config", None)
    if hc is not None and hc.WhichOneof("height_spec") == "pixel_height":
        if hc.pixel_height > 0:
            acc.append(node)
    for child in getattr(node, "children", {}).values():
        _scroll_containers(child, acc)
    return acc


def _contains_chat_message(node) -> bool:
    if getattr(node, "type", None) == "chat_message":
        return True
    return any(
        _contains_chat_message(c) for c in getattr(node, "children", {}).values()
    )


def test_chat_history_renders_in_fixed_height_scroll_container():
    at = _run_seeded()
    assert not at.exception, at.exception

    scrollers = _scroll_containers(at.main)
    assert scrollers, (
        "No fixed-pixel-height container found — chat transcript is not "
        "scroll-bounded, so the page jumps once the conversation exceeds the "
        "fold. Wrap the chat history in st.container(height=...)."
    )

    # The scroll container must actually hold the chat turns (not some
    # unrelated bounded element elsewhere on the page).
    assert any(_contains_chat_message(c) for c in scrollers), (
        "Found a fixed-height container, but it does not contain the chat "
        "messages — the transcript must render *inside* the scroll box."
    )


def test_chat_input_still_present_outside_scroll_container():
    """Regression: the bottom-pinned chat input must remain (it cannot live
    inside the height container) so the user can still ask questions."""
    at = _run_seeded()
    assert not at.exception, at.exception
    assert len(at.chat_input) >= 1, "chat input widget disappeared"
