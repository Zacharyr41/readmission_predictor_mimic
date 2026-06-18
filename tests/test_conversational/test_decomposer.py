"""Tests for the LLM decomposer module."""

import json
from unittest.mock import MagicMock

import pytest

from src.conversational.decomposer import (
    _extract_json,
    _validate_return_type,
    decompose,
)
from src.conversational.models import (
    AnswerResult,
    ClinicalConcept,
    CompetencyQuestion,
    ReturnType,
    TemporalConstraint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cq_json(**overrides) -> str:
    """Return a valid CompetencyQuestion JSON string with optional overrides."""
    base = {
        "original_question": "test question",
        "clinical_concepts": [{"name": "lactate", "concept_type": "biomarker"}],
        "return_type": "text_and_table",
        "scope": "cohort",
    }
    base.update(overrides)
    return json.dumps(base)


def _mock_client(*response_texts: str) -> MagicMock:
    """Create a mock Anthropic client returning *response_texts* in sequence."""
    client = MagicMock()
    responses = []
    for text in response_texts:
        resp = MagicMock()
        resp.content = [MagicMock(text=text)]
        responses.append(resp)
    client.messages.create.side_effect = responses
    return client


# ---------------------------------------------------------------------------
# TestExtractJson
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_raw_json(self):
        raw = '{"key": "value"}'
        assert _extract_json(raw) == raw

    def test_markdown_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        assert _extract_json(text) == '{"key": "value"}'

    def test_markdown_without_language_tag(self):
        text = '```\n{"key": "value"}\n```'
        assert _extract_json(text) == '{"key": "value"}'

    def test_text_with_embedded_json(self):
        text = 'Here is the result: {"key": "value"} hope that helps!'
        assert _extract_json(text) == '{"key": "value"}'

    def test_no_json_returns_input(self):
        text = "no json here"
        assert _extract_json(text) == text


# ---------------------------------------------------------------------------
# TestValidateReturnType
# ---------------------------------------------------------------------------


class TestValidateReturnType:
    def test_visualization_without_temporal_downgraded(self):
        cq = CompetencyQuestion(
            original_question="Show creatinine values",
            return_type=ReturnType.VISUALIZATION,
        )
        result = _validate_return_type(cq)
        assert result.return_type == ReturnType.TEXT_AND_TABLE

    def test_visualization_with_temporal_preserved(self):
        cq = CompetencyQuestion(
            original_question="Show creatinine over time",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker")
            ],
            temporal_constraints=[
                TemporalConstraint(relation="during", reference_event="ICU stay")
            ],
            return_type=ReturnType.VISUALIZATION,
        )
        result = _validate_return_type(cq)
        assert result.return_type == ReturnType.VISUALIZATION

    def test_visualization_with_explicit_keyword_preserved(self):
        cq = CompetencyQuestion(
            original_question="Plot the creatinine distribution",
            return_type=ReturnType.VISUALIZATION,
        )
        result = _validate_return_type(cq)
        assert result.return_type == ReturnType.VISUALIZATION

    def test_text_cohort_upgraded(self):
        cq = CompetencyQuestion(
            original_question="What is the mean lactate in sepsis patients?",
            clinical_concepts=[
                ClinicalConcept(name="lactate", concept_type="biomarker")
            ],
            aggregation="mean",
            return_type=ReturnType.TEXT,
            scope="cohort",
        )
        result = _validate_return_type(cq)
        assert result.return_type == ReturnType.TEXT_AND_TABLE

    def test_text_count_stays_text(self):
        cq = CompetencyQuestion(
            original_question="How many patients had sepsis?",
            clinical_concepts=[
                ClinicalConcept(name="sepsis", concept_type="diagnosis")
            ],
            aggregation="count",
            return_type=ReturnType.TEXT,
            scope="cohort",
        )
        result = _validate_return_type(cq)
        assert result.return_type == ReturnType.TEXT

    def test_text_and_table_unchanged(self):
        cq = CompetencyQuestion(
            original_question="Show lactate for all patients",
            return_type=ReturnType.TEXT_AND_TABLE,
            scope="cohort",
        )
        result = _validate_return_type(cq)
        assert result.return_type == ReturnType.TEXT_AND_TABLE


# ---------------------------------------------------------------------------
# TestDecompose
# ---------------------------------------------------------------------------


class TestDecompose:
    def test_successful_decomposition(self):
        cq_json = _make_cq_json(
            original_question="What is the average creatinine?",
            clinical_concepts=[
                {"name": "creatinine", "concept_type": "biomarker"}
            ],
            aggregation="mean",
            return_type="text_and_table",
            scope="cohort",
        )
        client = _mock_client(cq_json)

        result = decompose(client, "What is the average creatinine?")

        assert isinstance(result, CompetencyQuestion)
        assert result.clinical_concepts[0].name == "creatinine"
        assert result.aggregation == "mean"
        assert result.scope == "cohort"

    def test_follow_up_receives_history(self):
        cq_json = _make_cq_json()
        client = _mock_client(cq_json)

        prev_cq = CompetencyQuestion(
            original_question="What is the lactate?",
            clinical_concepts=[
                ClinicalConcept(name="lactate", concept_type="biomarker")
            ],
        )
        prev_answer = AnswerResult(text_summary="Lactate was 2.1 mmol/L.")

        decompose(
            client,
            "Now show me sodium",
            conversation_history=[(prev_cq, prev_answer)],
        )

        call_kwargs = client.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1]["messages"]
        # History: user + assistant turn, then the new question = 3 messages
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is the lactate?"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["content"] == "Now show me sodium"

    def test_json_in_code_block(self):
        cq_json = _make_cq_json()
        wrapped = f"```json\n{cq_json}\n```"
        client = _mock_client(wrapped)

        result = decompose(client, "test question")
        assert isinstance(result, CompetencyQuestion)

    def test_validation_retry_on_malformed_json(self):
        valid_json = _make_cq_json()
        client = _mock_client("not valid json {{{", valid_json)

        result = decompose(client, "test question")

        assert isinstance(result, CompetencyQuestion)
        assert client.messages.create.call_count == 2

    def test_return_type_text_upgraded_for_cohort(self):
        cq_json = _make_cq_json(
            return_type="text",
            scope="cohort",
            aggregation="mean",
        )
        client = _mock_client(cq_json)

        result = decompose(client, "test question")
        assert result.return_type == ReturnType.TEXT_AND_TABLE

    def test_visualization_preserved_with_trend_signal(self):
        cq_json = _make_cq_json(
            return_type="visualization",
            temporal_constraints=[
                {"relation": "during", "reference_event": "ICU stay"}
            ],
        )
        client = _mock_client(cq_json)

        result = decompose(
            client, "How did lactate change over time?"
        )
        assert result.return_type == ReturnType.VISUALIZATION

    def test_original_question_always_set_from_input(self):
        cq_json = _make_cq_json(
            original_question="LLM paraphrased this differently",
        )
        client = _mock_client(cq_json)

        actual_question = "What is the creatinine?"
        result = decompose(client, actual_question)
        assert result.original_question == actual_question


# ---------------------------------------------------------------------------
# Prompt content tests
# ---------------------------------------------------------------------------


class TestFilterValidation:
    def test_unsupported_filter_triggers_retry(self):
        """If LLM returns unsupported filter field, decomposer retries."""
        bad_json = _make_cq_json(
            patient_filters=[{"field": "icu_type", "operator": "=", "value": "neuro"}],
        )
        corrected_json = _make_cq_json(
            patient_filters=[
                {"field": "diagnosis", "operator": "contains", "value": "neuro"},
            ],
        )
        client = _mock_client(bad_json, corrected_json)
        result = decompose(client, "lactate for neuro ICU patients")

        # Should have retried (2 API calls)
        assert client.messages.create.call_count == 2
        # Result should have valid filter fields
        assert all(
            f.field in {"age", "gender", "diagnosis", "admission_type",
                        "subject_id", "readmitted_30d", "readmitted_60d"}
            for f in result.patient_filters
        )

    def test_supported_filters_no_retry(self):
        """Valid filter fields don't trigger a retry."""
        good_json = _make_cq_json(
            patient_filters=[{"field": "age", "operator": ">", "value": "65"}],
        )
        client = _mock_client(good_json)
        result = decompose(client, "creatinine for patients over 65")
        assert client.messages.create.call_count == 1
        assert result.patient_filters[0].field == "age"


class TestSupportedFilterFields:
    def test_prompt_enumerates_filter_fields(self):
        """The system prompt lists all supported filter field names."""
        from src.conversational.prompts import DECOMPOSITION_SYSTEM_PROMPT

        for field in [
            "age", "gender", "diagnosis", "admission_type",
            "subject_id", "readmitted_30d", "readmitted_60d",
        ]:
            assert field in DECOMPOSITION_SYSTEM_PROMPT, (
                f"Supported filter field '{field}' missing from decomposition prompt"
            )


class TestComputedIndexRouting:
    """A condition qualified by a CALCULATED clinical index ("high anion gap
    metabolic acidosis", "elevated osmolal gap", "low P/F ratio") must decompose
    into the base condition + a value-threshold filter on the *computed* index —
    NOT a single diagnosis whose long_title is matched by LIKE, and NOT an
    assumption that the index exists as one stored value. The fix is a GENERAL
    principle (any computable index), not an anion-gap special case."""

    def test_prompt_teaches_general_computed_index_principle(self):
        from src.conversational.prompts import DECOMPOSITION_SYSTEM_PROMPT as P
        low = P.lower()
        # A measured/computed qualifier is a VALUE-THRESHOLD filter, never a
        # diagnosis — and both routes are offered.
        assert "lab_value" in P and "derived_value" in P
        assert "anion gap" in low
        # PREFER the stored MIMIC measurement (reliable ground truth) over
        # recomputation — this is the key correction: anion gap IS a stored lab.
        assert "ground truth" in low or "stored value" in low or "reliable" in low
        # Keep the user's threshold DIRECTION so high vs non-high both work
        # (derived_value's single literature threshold can't express "non-high").
        assert "non-high" in low or "direction" in low
        # General — applies to quantities NOT explicitly listed (not overfit).
        assert "not listed here" in low or "any such quantity" in low


@pytest.mark.parametrize(
    "question,index_term",
    [
        ("Among ICU admissions, how many had a high anion gap metabolic acidosis?", "anion gap"),
        ("How many ICU admissions had an elevated osmolal gap?", "osmolal gap"),
    ],
)
def test_live_computed_index_condition_routes_to_value_threshold(question, index_term):
    """LIVE (gated): the real decomposer must route a computable-index-qualified
    condition to a value-threshold measurement, not a diagnosis named after the
    index. Two distinct indices guard against overfitting to anion gap. Skipped
    unless ANTHROPIC_API_KEY is set."""
    import os
    from pathlib import Path

    if not os.environ.get("ANTHROPIC_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    import anthropic

    from src.conversational.decomposer import decompose_question

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    cqs = decompose_question(client, question).competency_questions

    diag_names = [
        c.name.lower()
        for cq in cqs
        for c in cq.clinical_concepts
        if c.concept_type == "diagnosis"
    ]
    assert not any(index_term in n for n in diag_names), (
        f"{index_term!r} was modeled as a diagnosis concept: {diag_names}"
    )

    measurements = [
        (f.measurement or "").lower()
        for cq in cqs
        for f in cq.patient_filters
        if getattr(f, "measurement", None)
    ]
    assert any(index_term in m for m in measurements), (
        f"{index_term!r} not routed to a value-threshold measurement; "
        f"measurements={measurements}"
    )
