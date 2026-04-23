"""Decomposer recognizes similarity phrasings (Phase 9).

The decomposer must emit ``similarity_spec`` for questions that
involve comparing to a reference patient / profile. Uses the existing
mock-LLM test pattern — no real LLM call.
"""

from __future__ import annotations

from unittest.mock import patch

from src.conversational.decomposer import decompose_question
from src.conversational.models import CompetencyQuestion


def _mock_llm_response(payload: dict) -> str:
    """Returns a JSON string the decomposer parses."""
    import json

    return json.dumps(payload)


class TestSimilarityDecomposition:
    def test_similar_to_hadm_emits_similarity_spec(self):
        """'Show me patients similar to hadm 101' → CQ with
        similarity_spec.anchor_hadm_id=101 and scope='patient_similarity'."""
        fake_payload = {
            "original_question": "Show me patients similar to hadm 101",
            "scope": "patient_similarity",
            "similarity_spec": {"anchor_hadm_id": 101, "top_k": 30},
        }
        with patch("src.conversational.decomposer._call_llm") as m:
            m.return_value = _mock_llm_response(fake_payload)
            cq: CompetencyQuestion = decompose_question(
                "Show me patients similar to hadm 101",
            )
        assert cq.scope == "patient_similarity"
        assert cq.similarity_spec is not None
        assert cq.similarity_spec.anchor_hadm_id == 101

    def test_template_anchor_from_natural_language(self):
        """'Find patients like a 68yo F with afib' → template anchor."""
        fake_payload = {
            "original_question": "Find patients like a 68-year-old female with atrial fibrillation",
            "scope": "patient_similarity",
            "similarity_spec": {
                "anchor_template": {"age": 68, "gender_F": 1, "snomed_group_I48": 1},
                "top_k": 30,
            },
        }
        with patch("src.conversational.decomposer._call_llm") as m:
            m.return_value = _mock_llm_response(fake_payload)
            cq = decompose_question(
                "Find patients like a 68-year-old female with atrial fibrillation",
            )
        assert cq.scope == "patient_similarity"
        assert cq.similarity_spec.anchor_template is not None
        assert cq.similarity_spec.anchor_template["age"] == 68


class TestCausalPlusSimilarityDecomposition:
    def test_intervention_question_similar_to_patient_stays_causal(self):
        """'Compare tPA vs no tPA among patients similar to hadm 101' →
        scope='causal_effect' AND similarity_spec populated."""
        fake_payload = {
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
        with patch("src.conversational.decomposer._call_llm") as m:
            m.return_value = _mock_llm_response(fake_payload)
            cq = decompose_question(
                "Among patients similar to hadm 101, compare tPA vs no tPA on 30-day readmission",
            )
        assert cq.scope == "causal_effect"
        assert cq.similarity_spec is not None
        assert cq.similarity_spec.anchor_hadm_id == 101
