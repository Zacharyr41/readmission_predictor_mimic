"""Property-based fuzz tests for the decomposer.

Phase 6: hypothesis-driven tests that complement the fixture suite. The
fixture-based tests pin EXACT inputs and expected outputs; the fuzz tests
assert INVARIANTS that must hold for ANY input of the right shape.

Invariants covered:
  - ``_validate_return_type`` is total (never raises) for any
    structurally-valid CompetencyQuestion.
  - ``_validate_return_type`` is idempotent.
  - ``_synthesise_interpretation`` is total and produces a non-empty
    string for every well-formed CQ.
  - ``_extract_json`` is total on arbitrary input strings — it never
    raises, regardless of what malformed text an LLM emits.
  - JSON round-trip through ``_extract_json`` + ``json.loads`` +
    ``CompetencyQuestion.model_validate`` is stable: if we generate a
    CQ, serialise it, extract it, and re-parse it, we get the same CQ.

Hypothesis auto-generates edge cases we would never author by hand
(empty strings, unicode boundaries, deeply nested empty lists, etc.).
When a test fails it prints a minimal reproducer; add that case as a
regular fixture under ``decomposer_cases/`` so it's captured in
regression forever after.
"""

from __future__ import annotations

import json

import pytest

# Hypothesis is declared as a dev dependency; skip-cleanly if it's somehow
# absent so CI doesn't hard-fail on environments that skipped dev installs.
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from src.conversational.decomposer import (  # noqa: E402
    _extract_json,
    _synthesise_interpretation,
    _validate_return_type,
)
from src.conversational.models import (  # noqa: E402
    ClinicalConcept,
    CompetencyQuestion,
    PatientFilter,
    ReturnType,
    TemporalConstraint,
)


# ---------------------------------------------------------------------------
# Strategies — constructing valid CompetencyQuestion objects
# ---------------------------------------------------------------------------


# Concept-type set matches the Literal in ClinicalConcept. Hypothesis picks
# one at random so every generated CQ targets a valid concept_type.
_concept_types = st.sampled_from([
    "biomarker", "vital", "drug", "diagnosis", "microbiology", "outcome",
])


_filter_fields = st.sampled_from([
    "age", "gender", "diagnosis", "admission_type", "subject_id",
    "readmitted_30d", "readmitted_60d",
])


_filter_operators = st.sampled_from([">", "<", "=", ">=", "<=", "contains"])


_scope_values = st.sampled_from(["single_patient", "cohort", "comparison"])


_return_types = st.sampled_from([
    "text", "table", "text_and_table", "visualization",
])


_aggregations = st.one_of(
    st.none(),
    st.sampled_from(["mean", "avg", "median", "max", "min", "count", "sum", "exists"]),
)


# Comparison field can be None or one of the registered axes.
_comparison_fields = st.one_of(
    st.none(),
    st.sampled_from([
        "gender", "age", "readmitted_30d", "readmitted_60d",
        "admission_type", "discharge_location",
    ]),
)


_short_text = st.text(min_size=1, max_size=50).filter(lambda s: s.strip())


@st.composite
def _clinical_concepts(draw) -> list[ClinicalConcept]:
    n = draw(st.integers(min_value=0, max_value=4))
    concepts = []
    for _ in range(n):
        concepts.append(ClinicalConcept(
            name=draw(_short_text),
            concept_type=draw(_concept_types),
            attributes=draw(st.lists(_short_text, max_size=3)),
        ))
    return concepts


@st.composite
def _patient_filters(draw) -> list[PatientFilter]:
    n = draw(st.integers(min_value=0, max_value=3))
    filters = []
    for _ in range(n):
        filters.append(PatientFilter(
            field=draw(_filter_fields),
            operator=draw(_filter_operators),
            value=draw(_short_text),
        ))
    return filters


@st.composite
def _temporal_constraints(draw) -> list[TemporalConstraint]:
    n = draw(st.integers(min_value=0, max_value=2))
    constraints = []
    for _ in range(n):
        constraints.append(TemporalConstraint(
            relation=draw(st.sampled_from(["before", "after", "during", "within"])),
            reference_event=draw(_short_text),
            time_window=draw(st.one_of(st.none(), st.sampled_from(["24h", "48h", "7d", "30m"]))),
        ))
    return constraints


@st.composite
def _competency_questions(draw) -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question=draw(_short_text),
        clinical_concepts=draw(_clinical_concepts()),
        temporal_constraints=draw(_temporal_constraints()),
        patient_filters=draw(_patient_filters()),
        aggregation=draw(_aggregations),
        return_type=draw(_return_types),
        scope=draw(_scope_values),
        comparison_field=draw(_comparison_fields),
        interpretation_summary=draw(st.one_of(st.none(), _short_text)),
        clarifying_question=draw(st.one_of(st.none(), _short_text)),
    )


# ---------------------------------------------------------------------------
# 1. _validate_return_type invariants
# ---------------------------------------------------------------------------


class TestValidateReturnTypeInvariants:
    """For every structurally-valid CompetencyQuestion, the post-processor
    must be total and idempotent. If either invariant breaks, some retry or
    replay path downstream will behave non-deterministically."""

    @given(_competency_questions())
    @settings(max_examples=100, deadline=None)
    def test_never_raises(self, cq: CompetencyQuestion):
        _validate_return_type(cq.model_copy(deep=True))

    @given(_competency_questions())
    @settings(max_examples=100, deadline=None)
    def test_idempotent(self, cq: CompetencyQuestion):
        once = _validate_return_type(cq.model_copy(deep=True))
        twice = _validate_return_type(once.model_copy(deep=True))
        assert twice.model_dump() == once.model_dump()

    @given(_competency_questions())
    @settings(max_examples=100, deadline=None)
    def test_return_type_always_in_enum(self, cq: CompetencyQuestion):
        """``_validate_return_type`` may rewrite the field but must always
        leave it as a valid ReturnType value."""
        result = _validate_return_type(cq.model_copy(deep=True))
        assert isinstance(result.return_type, ReturnType)


# ---------------------------------------------------------------------------
# 2. _synthesise_interpretation invariants
# ---------------------------------------------------------------------------


class TestSynthesiseInterpretationInvariants:
    """The synthesiser is the fallback when the LLM omits an echo. It must
    never raise and must never produce an empty string — otherwise the UI's
    echo block ends up blank."""

    @given(_competency_questions())
    @settings(max_examples=100, deadline=None)
    def test_never_raises(self, cq: CompetencyQuestion):
        _synthesise_interpretation(cq)

    @given(_competency_questions())
    @settings(max_examples=100, deadline=None)
    def test_produces_nonempty_string(self, cq: CompetencyQuestion):
        result = _synthesise_interpretation(cq)
        assert isinstance(result, str)
        assert result.strip()


# ---------------------------------------------------------------------------
# 3. _extract_json is total on arbitrary text
# ---------------------------------------------------------------------------


class TestExtractJsonFuzz:
    """The LLM can emit literally any bytes. ``_extract_json`` is the first
    line of defence — it must never raise, regardless of input. Downstream
    ``json.loads`` is allowed to raise, but extraction itself must not."""

    @given(st.text(max_size=2000))
    @settings(max_examples=200, deadline=None)
    def test_never_raises_on_arbitrary_text(self, text: str):
        result = _extract_json(text)
        assert isinstance(result, str)

    @given(st.binary(max_size=500).map(
        lambda b: b.decode("utf-8", errors="replace")
    ))
    @settings(max_examples=100, deadline=None)
    def test_never_raises_on_mojibake(self, text: str):
        _extract_json(text)


# ---------------------------------------------------------------------------
# 4. JSON round-trip through the extraction pipeline
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    """A CQ serialised to JSON, wrapped in a code fence, run through
    ``_extract_json``, and re-parsed must equal the original. This is the
    contract the mock-anthropic flow relies on."""

    @given(_competency_questions())
    @settings(max_examples=50, deadline=None)
    def test_round_trip_through_extract_json(self, cq: CompetencyQuestion):
        payload = cq.model_dump_json()
        wrapped = f"```json\n{payload}\n```"
        extracted = _extract_json(wrapped)
        restored = CompetencyQuestion.model_validate_json(extracted)
        assert restored == cq

    @given(_competency_questions())
    @settings(max_examples=50, deadline=None)
    def test_round_trip_without_code_fence(self, cq: CompetencyQuestion):
        payload = cq.model_dump_json()
        extracted = _extract_json(payload)
        restored = CompetencyQuestion.model_validate_json(extracted)
        assert restored == cq

    @given(_competency_questions(), st.text(max_size=100), st.text(max_size=100))
    @settings(max_examples=50, deadline=None)
    def test_round_trip_with_prose_around_json(
        self, cq: CompetencyQuestion, prefix: str, suffix: str,
    ):
        """LLMs often wrap JSON with "Here is the result:" / "Hope that helps".
        Extraction must recover the JSON regardless of surrounding prose."""
        payload = cq.model_dump_json()
        # Avoid prefix/suffix containing braces that would confuse extraction.
        prefix = prefix.replace("{", "(").replace("}", ")")
        suffix = suffix.replace("{", "(").replace("}", ")")
        wrapped = f"{prefix}\n{payload}\n{suffix}"
        extracted = _extract_json(wrapped)
        restored = CompetencyQuestion.model_validate(json.loads(extracted))
        # The round-trip may strip default-valued fields via
        # model_dump_json, so compare dumps rather than model equality.
        assert restored.model_dump() == cq.model_dump()