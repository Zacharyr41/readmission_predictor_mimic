"""Contract-level tests for the decomposer.

These tests verify structural invariants and introspective properties that do
NOT depend on a specific LLM response. They are intentionally small in number
and generic in form — each test is a loop over fixtures, not a hand-written
per-case assertion. Adding a new behavioural case means dropping a file into
``fixtures/``, not editing this file.

Phase 0 scope:
  - Every ``json`` block inside the live prompt validates as CompetencyQuestion.
  - Prompt ↔ supported-fields enumeration is consistent (stubbed today, real in Phase 1).
  - ``_validate_return_type`` is idempotent and never raises.
  - Conversation-history truncation is bounded at the last 5 turns and ordered correctly.
  - ``_extract_json`` never raises on arbitrary malformed text.
  - Retry-on-validation-failure produces a correctly shaped follow-up message array.
"""

from __future__ import annotations

import json
import re

import pytest

from src.conversational.decomposer import (
    _extract_json,
    _validate_return_type,
    decompose,
)
from src.conversational.models import (
    AnswerResult,
    CompetencyQuestion,
    ClinicalConcept,
)
from src.conversational.prompts import (
    DECOMPOSITION_SYSTEM_PROMPT,
    build_decomposition_messages,
)

from tests.test_conversational.conftest import (
    load_decomposer_cases,
    load_malformed_json,
    load_prompt_examples,
    mock_anthropic,
)


# Supported filter fields per the current decomposer implementation.
# Phase 1 will replace this literal with ``registry.supported_names("filter")``.
_CURRENT_SUPPORTED_FILTERS = frozenset({
    "age", "gender", "diagnosis", "admission_type",
    "subject_id", "readmitted_30d", "readmitted_60d",
})


# ---------------------------------------------------------------------------
# 1. Prompt examples are valid CompetencyQuestions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block", load_prompt_examples())
def test_every_prompt_example_validates_as_competency_question(block: dict):
    """Each ```json``` block inside the live prompt must parse and validate.

    This catches prompt edits that introduce malformed JSON or schema drift.
    Runs against the regenerated ``fixtures/prompt_examples/`` snapshot, so the
    live prompt is the source of truth.
    """
    assert not block.get("__parse_error__"), (
        f"Prompt contains an unparseable JSON block:\n{block.get('__raw__')!r}"
    )
    cq = CompetencyQuestion.model_validate(block)
    assert isinstance(cq, CompetencyQuestion)


# ---------------------------------------------------------------------------
# 2. Prompt ↔ supported-fields round-trip
# ---------------------------------------------------------------------------


def test_every_supported_filter_field_appears_in_prompt():
    """Every filter field the decomposer enforces must be named in the prompt.

    Stub today: reads from ``_CURRENT_SUPPORTED_FILTERS``. Phase 1 swaps the
    source to ``OperationRegistry.supported_names('filter')`` so the test
    becomes self-updating.
    """
    for field in _CURRENT_SUPPORTED_FILTERS:
        assert field in DECOMPOSITION_SYSTEM_PROMPT, (
            f"Supported filter field {field!r} is missing from the prompt — "
            "the LLM will never be told it's an option."
        )


def test_prompt_does_not_advertise_unsupported_filter_fields():
    """Catch the reverse drift: a field listed in the prompt but not enforced.

    Extracts the ``field`` enumeration from the prompt schema block and checks
    it against ``_CURRENT_SUPPORTED_FILTERS``. The whole block is a single
    Literal like ``<age|gender|diagnosis|...>`` — we parse that.
    """
    match = re.search(
        r'"field":\s*"<([^">]+)>"',
        DECOMPOSITION_SYSTEM_PROMPT,
    )
    assert match, "Could not locate patient_filters field enumeration in prompt"
    advertised = set(match.group(1).split("|"))
    extra = advertised - _CURRENT_SUPPORTED_FILTERS
    assert not extra, (
        f"Prompt advertises filter fields that the decomposer will reject: {extra}"
    )


# ---------------------------------------------------------------------------
# 3. _validate_return_type invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", load_decomposer_cases())
def test_validate_return_type_is_idempotent(case: dict):
    """Applying ``_validate_return_type`` twice yields the same result as once.

    Any post-processing rule that is not idempotent is a latent bug: rerunning
    the validator (which can happen in retry loops) must not keep mutating state.
    """
    cq = CompetencyQuestion.model_validate(case["expected_cq"])
    once = _validate_return_type(cq.model_copy(deep=True))
    twice = _validate_return_type(once.model_copy(deep=True))
    assert twice.model_dump() == once.model_dump()


@pytest.mark.parametrize("case", load_decomposer_cases())
def test_validate_return_type_never_raises(case: dict):
    """``_validate_return_type`` must never raise — it's a best-effort cleanup."""
    cq = CompetencyQuestion.model_validate(case["expected_cq"])
    _validate_return_type(cq)


# ---------------------------------------------------------------------------
# 4. Conversation history: truncation and ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_turns", [0, 1, 5, 10])
def test_conversation_history_truncation_and_ordering(n_turns: int):
    """The messages array contains the last 5 history turns (user+assistant pairs) + the new question.

    Parametrized over four sizes to exercise (a) empty history, (b) a single
    turn, (c) exactly the truncation boundary, (d) well over the boundary.
    We assert only on roles, counts, and ordering — never on substrings of
    prompt content, so that prompt-text changes don't break the test.
    """
    history = []
    for i in range(n_turns):
        cq = CompetencyQuestion(original_question=f"prev question {i}")
        answer = AnswerResult(text_summary=f"prev answer {i}")
        history.append((cq, answer))

    messages = build_decomposition_messages(
        "new question",
        conversation_history=history or None,
    )

    # At most the last 5 turns (user+assistant = 2 messages each) + the new user message.
    expected_history_turns = min(n_turns, 5)
    assert len(messages) == (expected_history_turns * 2) + 1

    # Roles alternate user/assistant across the history, then a trailing user.
    for idx in range(expected_history_turns):
        assert messages[idx * 2]["role"] == "user"
        assert messages[idx * 2 + 1]["role"] == "assistant"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "new question"

    # History is the tail of the provided list, not the head — verify ordering.
    if n_turns > expected_history_turns:
        # The first included user message should be the (n_turns - 5)-th original.
        expected_first = f"prev question {n_turns - expected_history_turns}"
        assert messages[0]["content"] == expected_first


# ---------------------------------------------------------------------------
# 5. JSON extraction is total (never raises)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", load_malformed_json())
def test_extract_json_never_raises(raw: str):
    """``_extract_json`` is a pure text utility and must be total.

    Callers downstream may still fail to parse the result — that's fine —
    but the extractor itself must never raise. This guarantees the decomposer
    can always fall back to its retry path.
    """
    result = _extract_json(raw)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 6. Retry-on-malformed-JSON: structural assertions
# ---------------------------------------------------------------------------


def test_retry_on_malformed_json_sends_corrective_follow_up():
    """After a malformed first response, the second API call must include
    (a) the prior assistant turn verbatim and (b) a corrective user turn.

    We assert on message roles, counts, and the presence of the prior
    assistant text — NOT on specific wording of the corrective prompt, so
    rewording the retry message doesn't break this test.
    """
    valid = json.dumps({
        "original_question": "avg creatinine",
        "clinical_concepts": [
            {"name": "creatinine", "concept_type": "biomarker"}
        ],
        "return_type": "text",
        "scope": "cohort",
    })
    bad_first_response = "not valid json at all {{{"
    client = mock_anthropic([bad_first_response, valid])

    decompose(client, "test question")

    assert client.messages.create.call_count == 2

    second_call = client.messages.create.call_args_list[1]
    messages = second_call.kwargs["messages"]

    # First call had only the user question. Second call must extend that
    # with the prior assistant turn + a corrective user turn.
    roles = [m["role"] for m in messages]
    assert roles[0] == "user"
    assert "assistant" in roles, "Retry must include the prior assistant turn"

    assistant_turn = next(m for m in messages if m["role"] == "assistant")
    assert assistant_turn["content"] == bad_first_response, (
        "The assistant's failing response must be replayed verbatim so the "
        "LLM can see what it produced and correct itself."
    )

    # The final message is the corrective user prompt.
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] != "test question", (
        "The corrective user turn must differ from the original question."
    )


def test_retry_on_unsupported_filter_sends_corrective_follow_up():
    """When the LLM returns a valid CQ that uses a rejected filter field, the
    decomposer retries with a corrective message naming the unsupported fields.

    We check the retry shape, not its exact wording.
    """
    bad = json.dumps({
        "original_question": "neuro ICU lactate",
        "clinical_concepts": [
            {"name": "lactate", "concept_type": "biomarker"}
        ],
        "patient_filters": [
            {"field": "icu_type", "operator": "=", "value": "neuro"}
        ],
        "return_type": "text",
        "scope": "cohort",
    })
    good = json.dumps({
        "original_question": "neuro ICU lactate",
        "clinical_concepts": [
            {"name": "lactate", "concept_type": "biomarker"}
        ],
        "patient_filters": [
            {"field": "diagnosis", "operator": "contains", "value": "neuro"}
        ],
        "return_type": "text",
        "scope": "cohort",
    })
    client = mock_anthropic([bad, good])

    result = decompose(client, "lactate for neuro ICU patients")

    assert client.messages.create.call_count == 2
    # Final result must only reference supported fields.
    for f in result.patient_filters:
        assert f.field in _CURRENT_SUPPORTED_FILTERS


# ---------------------------------------------------------------------------
# 7. Prompt examples round-trip through CompetencyQuestion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block", load_prompt_examples())
def test_prompt_examples_validate_return_type_idempotent(block: dict):
    """Combine (1) and (3): every prompt example, when parsed, survives
    ``_validate_return_type`` idempotently. Catches the case where a prompt
    example itself would be 'corrected' on first run — a sign the example
    teaches the LLM something the validator will then undo.
    """
    if block.get("__parse_error__"):
        pytest.skip("Unparseable prompt block — covered by validity test")
    cq = CompetencyQuestion.model_validate(block)
    once = _validate_return_type(cq.model_copy(deep=True))
    twice = _validate_return_type(once.model_copy(deep=True))
    assert twice.model_dump() == once.model_dump()
