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
    Auto-extracted at test-session start from DECOMPOSITION_SYSTEM_PROMPT. Every
    ```json``` block inside the live prompt is written to a file here, named by
    a hash of the block's question. Because these are regenerated on every run,
    they cannot drift from the live prompt — if the prompt changes, so do the
    files, and so do any tests that parametrize over them.

``fixtures/malformed_json/*.txt``
    Raw strings that simulate LLM responses which are not clean JSON. Used by
    extraction-fuzz tests: ``_extract_json`` must never raise on these.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.conversational.prompts import DECOMPOSITION_SYSTEM_PROMPT


FIXTURES_DIR = Path(__file__).parent / "fixtures"
DECOMPOSER_CASES_DIR = FIXTURES_DIR / "decomposer_cases"
PROMPT_EXAMPLES_DIR = FIXTURES_DIR / "prompt_examples"
MALFORMED_JSON_DIR = FIXTURES_DIR / "malformed_json"


# ---------------------------------------------------------------------------
# Mock Anthropic client factory
# ---------------------------------------------------------------------------


def mock_anthropic(responses: list[str]) -> MagicMock:
    """Return a ``MagicMock`` client that yields each response text in turn.

    The mock mimics ``anthropic.Anthropic``: ``client.messages.create(...)``
    returns an object whose ``content[0].text`` is the next string in
    ``responses``. A test can inspect ``client.messages.create.call_args_list``
    to verify retry behaviour (message roles, counts, etc.) without matching
    exact prompt substrings.

    Parameters
    ----------
    responses:
        Strings to return in order, one per ``messages.create`` call.

    Raises
    ------
    StopIteration (via MagicMock side_effect) if the code under test makes
    more calls than there are responses — callers expecting a single call
    should pass a single-element list and the mock will raise if a second
    call is made.
    """
    client = MagicMock()
    response_objects = []
    for text in responses:
        resp = MagicMock()
        resp.content = [MagicMock(text=text)]
        response_objects.append(resp)
    client.messages.create.side_effect = response_objects
    return client


# ---------------------------------------------------------------------------
# Prompt-example extraction (auto-regenerated each session)
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def _extract_prompt_json_blocks(prompt: str) -> list[dict[str, Any]]:
    """Parse every ```json``` block in *prompt* and return them as dicts.

    Blocks that fail to parse are skipped silently — the prompt-example-
    validity test will catch that separately (by parametrizing over the
    raw block source). For now we only return successfully-parsed blocks
    so callers can use them as CompetencyQuestion payloads.
    """
    blocks: list[dict[str, Any]] = []
    for match in _JSON_BLOCK_RE.finditer(prompt):
        raw = match.group(1).strip()
        try:
            blocks.append(json.loads(raw))
        except json.JSONDecodeError:
            # Still capture the raw text so the validity test can fail on it
            blocks.append({"__raw__": raw, "__parse_error__": True})
    return blocks


def _sync_prompt_examples() -> list[Path]:
    """Regenerate ``fixtures/prompt_examples/`` from the live prompt.

    Called once per test session (see ``_prompt_examples_synced`` autouse
    fixture). Overwrites any existing files so no drift is possible. Each
    block is stored under a filename derived from the ``original_question``
    field (stable, human-readable) or a content hash (fallback).
    """
    PROMPT_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Clear out any previous snapshot so removed prompt blocks don't linger.
    for existing in PROMPT_EXAMPLES_DIR.glob("*.json"):
        existing.unlink()

    written: list[Path] = []
    for block in _extract_prompt_json_blocks(DECOMPOSITION_SYSTEM_PROMPT):
        if block.get("__parse_error__"):
            # Preserve the raw text for the validity test to trip on.
            raw = block["__raw__"]
            digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
            path = PROMPT_EXAMPLES_DIR / f"_unparseable_{digest}.json"
            path.write_text(raw)
            written.append(path)
            continue

        question = block.get("original_question", "")
        if question:
            slug = _slugify(question)[:60]
        else:
            slug = hashlib.sha1(
                json.dumps(block, sort_keys=True).encode("utf-8")
            ).hexdigest()[:10]
        path = PROMPT_EXAMPLES_DIR / f"{slug}.json"
        path.write_text(json.dumps(block, indent=2))
        written.append(path)

    return written


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "unnamed"


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
