"""Fixture-driven end-to-end decomposer tests.

Each case in ``fixtures/decomposer_cases/`` describes:
    question      — the clinical natural-language input
    llm_response  — the raw string the mock Anthropic client will return
    expected_cq   — the post-processed CompetencyQuestion we expect
    tags          — free-form labels (for filtering / documentation)

This file contains exactly one test body. Every new behavioural scenario is
added by dropping a JSON file into the fixture directory — never by editing
this file. The body is deliberately generic so that mock-mimicry cannot creep
back in: we compare the decomposer's output (post-process) to the
fixture's ``expected_cq``, dict-for-dict, with tolerant equality on
ordering-insensitive lists.

Phase 6 will extend the fixture set to 30+ cases (including ambiguous /
clarifying / temporal / comparison edges). The runner here does not change.
"""

from __future__ import annotations

import pytest

from src.conversational.decomposer import decompose
from src.conversational.models import CompetencyQuestion

from tests.test_conversational.conftest import (
    load_decomposer_cases,
    mock_anthropic,
)


def _normalise_cq_dict(cq: CompetencyQuestion) -> dict:
    """Produce a comparison-stable dict representation.

    - Filters are sorted by ``(field, operator, value)`` so input order
      from the LLM doesn't matter.
    - Temporal constraints are sorted by ``(reference_event, relation, time_window)``.
    - Clinical concepts are sorted by ``(name, concept_type)``.
    - ``attributes`` inside concepts are sorted.
    """
    d = cq.model_dump(mode="json")
    d["patient_filters"] = sorted(
        d.get("patient_filters", []),
        key=lambda f: (f.get("field", ""), f.get("operator", ""), str(f.get("value", ""))),
    )
    d["temporal_constraints"] = sorted(
        d.get("temporal_constraints", []),
        key=lambda t: (
            t.get("reference_event", ""),
            t.get("relation", ""),
            t.get("time_window") or "",
        ),
    )
    concepts = d.get("clinical_concepts", [])
    for c in concepts:
        c["attributes"] = sorted(c.get("attributes", []))
    d["clinical_concepts"] = sorted(
        concepts,
        key=lambda c: (c.get("name", ""), c.get("concept_type", "")),
    )
    return d


# Phase 4 added synthesised / optional fields to CompetencyQuestion. Fixtures
# authored before Phase 4 don't pin these values, so the runner drops them
# from both sides of the comparison unless the fixture's ``expected_cq``
# explicitly sets them. A fixture that pins e.g. a specific
# ``clarifying_question`` string still gets strict equality; a fixture that
# omits them accepts whatever the decomposer synthesised.
_OPTIONAL_PHASE4_FIELDS = ("interpretation_summary", "clarifying_question")


def _drop_unpinned_optional_fields(actual: dict, expected: dict) -> tuple[dict, dict]:
    """Return (actual, expected) with optional Phase-4 fields stripped from
    the actual side when the expected fixture didn't pin them.

    Both dicts are returned as fresh copies — never mutate the inputs.
    """
    actual = dict(actual)
    expected = dict(expected)
    for field in _OPTIONAL_PHASE4_FIELDS:
        if field not in expected or expected.get(field) is None:
            actual.pop(field, None)
            expected.pop(field, None)
    return actual, expected


@pytest.mark.parametrize("case", load_decomposer_cases())
def test_decomposer_case(case: dict):
    """Run the case's ``llm_response`` through ``decompose`` and compare.

    Tolerant to ordering of lists; strict on field values and types.
    """
    client = mock_anthropic([case["llm_response"]])
    result = decompose(client, case["question"])

    expected = _normalise_cq_dict(CompetencyQuestion.model_validate(case["expected_cq"]))
    actual = _normalise_cq_dict(result)
    actual, expected = _drop_unpinned_optional_fields(actual, expected)

    assert actual == expected, (
        f"Case {case['name']!r} tagged {case.get('tags', [])}:\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}"
    )
