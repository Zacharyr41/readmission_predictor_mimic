"""Decomposer recognises similarity phrasings (Phase 9).

The real decomposer calls ``client.messages.create`` on an
``anthropic.Anthropic``-compatible client. These tests build a mock
client that returns a canned JSON payload shaped like the LLM's
output, then assert the parsed ``CompetencyQuestion`` carries the
expected ``similarity_spec``.

The mocks cover two decomposer-visible shapes:

* Shape A — single CQ with ``scope="patient_similarity"``.
* Shape A — causal CQ + ``similarity_spec`` (cohort-narrowing).

The production decomposer's ``DECOMPOSITION_SYSTEM_PROMPT`` is
extended in commit 7 so the LLM knows when to emit these shapes;
see ``src/conversational/prompts.py`` + the few-shot examples at
``src/conversational/prompt_examples/similarity_cq/``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.conversational.decomposer import decompose_question


def _mock_anthropic_client(json_payload: dict) -> MagicMock:
    """Build a MagicMock whose ``.messages.create(...)`` returns a
    response object with ``.content[0].text`` set to
    ``json.dumps(json_payload)`` — the shape the decomposer reads."""
    client = MagicMock()
    response = MagicMock()
    content_item = MagicMock()
    content_item.text = json.dumps(json_payload)
    response.content = [content_item]
    client.messages.create.return_value = response
    return client


class TestSimilarityDecomposition:
    def test_similar_to_hadm_emits_similarity_spec(self):
        """'Show me patients similar to hadm 101' → scope='patient_similarity'
        with anchor_hadm_id populated."""
        payload = {
            "original_question": "Show me patients similar to hadm 101",
            "scope": "patient_similarity",
            "similarity_spec": {"anchor_hadm_id": 101, "top_k": 30},
        }
        client = _mock_anthropic_client(payload)
        result = decompose_question(
            client, "Show me patients similar to hadm 101",
        )
        assert len(result.competency_questions) == 1
        cq = result.competency_questions[0]
        assert cq.scope == "patient_similarity"
        assert cq.similarity_spec is not None
        assert cq.similarity_spec.anchor_hadm_id == 101

    def test_template_anchor_from_natural_language(self):
        """'Find patients like a 68yo F with afib' → template anchor."""
        payload = {
            "original_question": "Find patients like a 68-year-old female with atrial fibrillation",
            "scope": "patient_similarity",
            "similarity_spec": {
                "anchor_template": {"age": 68, "gender_F": 1, "snomed_group_I48": 1},
                "top_k": 30,
            },
        }
        client = _mock_anthropic_client(payload)
        result = decompose_question(
            client,
            "Find patients like a 68-year-old female with atrial fibrillation",
        )
        cq = result.competency_questions[0]
        assert cq.scope == "patient_similarity"
        assert cq.similarity_spec.anchor_template is not None
        assert cq.similarity_spec.anchor_template["age"] == 68


class TestCausalPlusSimilarityDecomposition:
    def test_intervention_question_similar_to_patient_stays_causal(self):
        """'Compare tPA vs no tPA among patients similar to hadm 101' →
        scope='causal_effect' AND similarity_spec populated."""
        payload = {
            "original_question": "Among patients similar to hadm 101, compare tPA vs no tPA on 30-day readmission",
            "scope": "causal_effect",
            "similarity_spec": {"anchor_hadm_id": 101, "top_k": 30},
            "intervention_set": [
                {"label": "tPA", "kind": "drug", "rxnorm_ingredient": "8410"},
                {"label": "no_tPA", "kind": "drug", "rxnorm_ingredient": "8410", "is_control": True},
            ],
            "outcome_vector": [
                {"name": "readmitted_30d", "outcome_type": "binary",
                 "extractor_key": "readmitted_30d"},
            ],
            "aggregation_spec": {"kind": "identity"},
        }
        client = _mock_anthropic_client(payload)
        result = decompose_question(
            client,
            "Among patients similar to hadm 101, compare tPA vs no tPA on 30-day readmission",
        )
        cq = result.competency_questions[0]
        assert cq.scope == "causal_effect"
        assert cq.similarity_spec is not None
        assert cq.similarity_spec.anchor_hadm_id == 101
