"""Shared fixtures and helpers for the conversational analytics test suite.

Phase 0 scaffolding. The philosophy here: tests exercise behaviour, not mock shape.
Adding a new behavioural case is a matter of dropping a JSON file into one of the
fixture directories — not editing a ``.py`` test.

Fixture directories
-------------------
``fixtures/decomposer_cases/*.json``
    Hand-authored cases: ``{question, llm_response, expected_cq, tags}``.
    Each file is one case. ``llm_response`` is the exact string the mock Anthropic
    client will return; ``expected_cq`` is the CompetencyQuestion dict we expect
    after decomposer post-processing. ``tags`` are free-form labels.

``fixtures/prompt_examples/*.json``
    Mirror of the canonical examples under
    ``src/conversational/prompt_examples/single_cq/``. Regenerated at test-
    session start so the test fixture tracks the source. The source-of-truth
    layout inverted in Phase 3: examples are now authored as JSON files under
    ``src/`` and the prompt is *built from* them, rather than extracted from
    a prompt string. The drift-proof guarantee is the same — tests that
    parametrize over these fixtures see whatever the prompt builder currently
    embeds.

``fixtures/malformed_json/*.txt``
    Raw strings that simulate LLM responses which are not clean JSON. Used by
    extraction-fuzz tests: ``_extract_json`` must never raise on these.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.conversational.prompts import (
    DECOMPOSITION_SYSTEM_PROMPT,
    PROMPT_EXAMPLES_DIR as _SRC_PROMPT_EXAMPLES_DIR,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"
DECOMPOSER_CASES_DIR = FIXTURES_DIR / "decomposer_cases"
PROMPT_EXAMPLES_DIR = FIXTURES_DIR / "prompt_examples"
MALFORMED_JSON_DIR = FIXTURES_DIR / "malformed_json"


# ---------------------------------------------------------------------------
# Mock Anthropic client factory
# ---------------------------------------------------------------------------


def mock_anthropic(responses: list) -> MagicMock:
    """Return a ``MagicMock`` client that yields each response in turn.

    The mock mimics ``anthropic.Anthropic``: ``client.messages.create(...)``
    returns the next prebuilt response object. A test can inspect
    ``client.messages.create.call_args_list`` to verify multi-call
    behaviour (retry sequences, tool-use loops, etc.) without matching
    exact prompt substrings.

    Each item in ``responses`` is one of:

    * ``str`` (legacy): treated as ``{"text": <str>, "stop_reason":
      "end_turn"}``. The response carries a single text content block;
      ``response.content[0].text`` is the string. ``stop_reason`` is set
      explicitly to ``"end_turn"`` so tool-use loops correctly identify
      the final turn (without this, MagicMock's default ``.stop_reason``
      is itself a MagicMock and never equals ``"end_turn"``).
    * ``dict`` with ``{"text": str, "stop_reason": "end_turn"}``: same
      as the legacy str shape, just explicit. Optionally also carries
      ``"tool_use": [...]`` for mixed-content responses.
    * ``dict`` with ``{"tool_use": [{"id", "name", "input"}, ...],
      "stop_reason": "tool_use"}``: the model wants to call tools.
      Each tool_use block becomes a content block with ``.type ==
      "tool_use"``, ``.id``, ``.name``, ``.input``. Optional
      ``"text"`` adds a leading text block (e.g. model's reasoning).

    Raises
    ------
    StopIteration (via MagicMock side_effect) if the code under test makes
    more calls than there are responses.
    """
    client = MagicMock()
    response_objects = [_build_response(item) for item in responses]
    client.messages.create.side_effect = response_objects
    return client


def _build_response(item) -> MagicMock:
    """Build one response MagicMock from a list-item spec.

    See ``mock_anthropic`` docstring for accepted shapes.
    """
    resp = MagicMock()
    if isinstance(item, str):
        # Legacy: text-only response, end_turn.
        text_block = MagicMock(type="text", text=item)
        resp.content = [text_block]
        resp.stop_reason = "end_turn"
        return resp
    if not isinstance(item, dict):
        raise TypeError(
            f"mock_anthropic items must be str or dict; got {type(item).__name__}"
        )
    blocks = []
    text = item.get("text")
    if text is not None:
        text_block = MagicMock(type="text")
        text_block.text = text
        blocks.append(text_block)
    for tu in item.get("tool_use", []) or []:
        # NOTE: ``MagicMock(name=...)`` is a footgun — ``name`` is the
        # mock's *display* name for repr, not a regular attribute. Set
        # ``.name`` explicitly after construction so ``getattr(block,
        # "name")`` returns the tool name string.
        block = MagicMock(type="tool_use")
        block.id = tu.get("id", "tu_anonymous")
        block.name = tu["name"]
        block.input = tu.get("input", {})
        blocks.append(block)
    resp.content = blocks
    resp.stop_reason = item.get("stop_reason", "end_turn")
    return resp


# ---------------------------------------------------------------------------
# Prompt-example extraction (auto-regenerated each session)
# ---------------------------------------------------------------------------


def _sync_prompt_examples() -> list[Path]:
    """Mirror ``src/conversational/prompt_examples/single_cq/`` into the test
    fixture directory.

    Called once per test session (see ``_prompt_examples_synced`` autouse
    fixture). Overwrites any existing files so the fixture can never lag
    the canonical source. Filenames are preserved so test IDs stay stable.
    """
    PROMPT_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Clear out the old snapshot so removed source files don't linger.
    for existing in PROMPT_EXAMPLES_DIR.glob("*.json"):
        existing.unlink()

    src_single = _SRC_PROMPT_EXAMPLES_DIR / "single_cq"
    written: list[Path] = []
    if src_single.exists():
        for src_path in sorted(src_single.glob("*.json")):
            dest = PROMPT_EXAMPLES_DIR / src_path.name
            dest.write_text(src_path.read_text())
            written.append(dest)
    return written


@pytest.fixture(scope="session", autouse=True)
def _prompt_examples_synced() -> None:
    """Regenerate prompt-example fixtures once per session.

    Autouse + session-scoped so every test collecting from
    ``PROMPT_EXAMPLES_DIR`` sees an up-to-date snapshot of the live prompt.
    This is what makes drift between prompt and fixtures impossible.
    """
    _sync_prompt_examples()


# ---------------------------------------------------------------------------
# Fixture loaders
# ---------------------------------------------------------------------------


def load_decomposer_cases() -> list[pytest.param]:
    """Yield pytest params for every file in ``fixtures/decomposer_cases/``.

    Each case file is a JSON object with at minimum:
        {
            "name": str,
            "question": str,
            "llm_response": str,         # raw LLM response the mock will return
            "expected_cq": dict,         # expected post-processed CQ
            "tags": list[str]            # optional
        }

    Missing directories yield an empty list so the runner still collects
    (useful when fixtures haven't been seeded yet).
    """
    if not DECOMPOSER_CASES_DIR.exists():
        return []
    params = []
    for path in sorted(DECOMPOSER_CASES_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        params.append(pytest.param(data, id=data.get("name", path.stem)))
    return params


def load_prompt_examples() -> list[pytest.param]:
    """Yield pytest params for every block extracted from the live prompt.

    Depends on ``_prompt_examples_synced``; callers should not worry about
    regeneration ordering because the autouse session fixture runs first.
    """
    if not PROMPT_EXAMPLES_DIR.exists():
        return []
    params = []
    for path in sorted(PROMPT_EXAMPLES_DIR.glob("*.json")):
        raw = path.read_text()
        try:
            data = json.loads(raw)
            params.append(pytest.param(data, id=path.stem))
        except json.JSONDecodeError:
            # Unparseable blocks are still parametrized so the validity test
            # can fail on them with a clear message.
            params.append(
                pytest.param(
                    {"__raw__": raw, "__parse_error__": True},
                    id=path.stem,
                )
            )
    return params


def load_malformed_json() -> list[pytest.param]:
    """Yield pytest params for every malformed-response fixture."""
    if not MALFORMED_JSON_DIR.exists():
        return []
    params = []
    for path in sorted(MALFORMED_JSON_DIR.glob("*.txt")):
        params.append(pytest.param(path.read_text(), id=path.stem))
    return params
