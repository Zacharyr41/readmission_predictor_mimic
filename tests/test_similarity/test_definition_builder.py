"""TDD for the free-text → ``CohortDefinition`` builder (plan II-D).

The builder is the missing link that makes the anchorless cohort path
reachable: it translates a clinician's free-text description into a
schema-validated :class:`CohortDefinition`, applying the clause→kernel
taxonomy (point→symmetric, "worse/rising"→one-sided, "has X"→asymmetric
binary + prefilter). The LLM is *mocked* in every test here — what we
assert is that the builder faithfully parses, validates, repairs, and
resolves the model's JSON, never that the (mocked) model exercised
judgment. Taxonomy enforcement lives in the system prompt, which is
asserted structurally by ``test_system_prompt_*``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.pygower import Direction, Kind
from src.similarity.definition_builder import (
    build_definition,
    build_definition_system_prompt,
)
from src.similarity.feature_catalog import (
    catalog_feature_names,
    cohort_feature_catalog,
)
from src.similarity.models import CohortDefinition


def _mock_client(responses: list[str]) -> MagicMock:
    """A stand-in anthropic client whose ``messages.create`` yields each
    response's text in turn (``response.content[0].text``)."""
    client = MagicMock()
    objs = []
    for text in responses:
        resp = MagicMock()
        block = MagicMock(type="text")
        block.text = text
        resp.content = [block]
        resp.stop_reason = "end_turn"
        objs.append(resp)
    client.messages.create.side_effect = objs
    return client


def _defn_json(**overrides) -> str:
    base = {
        "prefilters": [
            {"field": "admission_type", "operator": "=", "value": "EMERGENCY"},
        ],
        "traits": [
            {
                "name": "age", "source": "sql", "kind": "quantitative",
                "reference_value": 68, "direction": "symmetric", "weight": 0.6,
            },
            {
                "name": "gender", "source": "sql", "kind": "nominal",
                "reference_value": "F", "weight": 1.0,
            },
            {
                "name": "creatinine_max", "source": "sql", "kind": "quantitative",
                "reference_value": None, "direction": "higher_more_similar",
                "weight": 2.0,
            },
        ],
        "distance_threshold": 0.35,
        "top_k": 30,
    }
    base.update(overrides)
    return json.dumps(base)


def _graph_defn_json(**overrides) -> str:
    """A definition whose single trait is a scored graph_temporal trait."""
    base = {
        "traits": [
            {
                "name": "lactate_slope_48h", "source": "graph_temporal",
                "kind": "quantitative", "reference_value": 1.5,
                "direction": "higher_more_similar", "weight": 2.0,
                "template": "sim_series_by_admission", "concept": "lactate",
                "concept_type": "biomarker",
                "graph_params": {"agg": "slope", "window_hours": 48},
            },
        ],
        "distance_threshold": 0.35,
        "top_k": 30,
    }
    base.update(overrides)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# Feature catalog — the SQL-extractable trait vocabulary the builder may emit.
# ---------------------------------------------------------------------------


class TestFeatureCatalog:
    def test_catalog_kinds(self):
        cat = cohort_feature_catalog()
        assert cat["age"].kind == Kind.QUANTITATIVE
        assert cat["creatinine_max"].kind == Kind.QUANTITATIVE
        assert cat["gender"].kind == Kind.NOMINAL
        assert cat["admission_type"].kind == Kind.NOMINAL

    def test_nominal_features_carry_categories(self):
        cat = cohort_feature_catalog()
        assert "F" in cat["gender"].categories
        assert "M" in cat["gender"].categories
        assert "EMERGENCY" in cat["admission_type"].categories

    def test_catalog_names_are_extractor_columns(self):
        # Every catalog name must be a column produced by
        # _fetch_admission_features, or run_cohort's missing-column guard
        # rejects it. These five quantitative + two nominal are exactly that set.
        expected = {
            "age", "icu_los_hours", "creatinine_max", "sodium_mean",
            "platelet_min", "gender", "admission_type",
        }
        assert expected <= catalog_feature_names()


# ---------------------------------------------------------------------------
# Parsing / validation.
# ---------------------------------------------------------------------------


class TestBuildDefinition:
    def test_parses_traits_and_prefilters(self):
        client = _mock_client([_defn_json()])
        defn = build_definition(
            client, "Find emergency patients like a 68yo woman with rising creatinine",
        )
        assert isinstance(defn, CohortDefinition)
        assert {t.name for t in defn.traits} == {"age", "gender", "creatinine_max"}
        age = next(t for t in defn.traits if t.name == "age")
        assert age.direction == Direction.SYMMETRIC
        assert age.reference_value == 68
        cr = next(t for t in defn.traits if t.name == "creatinine_max")
        assert cr.direction == Direction.HIGHER_MORE_SIMILAR
        assert [f.field for f in defn.prefilters] == ["admission_type"]
        assert defn.distance_threshold == 0.35

    def test_strips_markdown_code_fence(self):
        fenced = "```json\n" + _defn_json() + "\n```"
        client = _mock_client([fenced])
        defn = build_definition(client, "…")
        assert isinstance(defn, CohortDefinition)

    def test_directional_reference_resolved_from_frozen_high_end(self):
        client = _mock_client([_defn_json()])
        frozen = {"creatinine_max": (0.5, 9.8), "age": (19.0, 91.0)}
        defn = build_definition(client, "…rising creatinine…", reference_ranges=frozen)
        cr = next(t for t in defn.traits if t.name == "creatinine_max")
        # higher_more_similar + null reference → the "bad-enough" high end.
        assert cr.reference_value == 9.8
        # symmetric trait with an explicit value is left untouched.
        age = next(t for t in defn.traits if t.name == "age")
        assert age.reference_value == 68

    def test_directional_reference_resolved_from_frozen_low_end(self):
        j = json.dumps({
            "traits": [{
                "name": "platelet_min", "source": "sql", "kind": "quantitative",
                "reference_value": None, "direction": "lower_more_similar",
                "weight": 1.0,
            }],
        })
        client = _mock_client([j])
        frozen = {"platelet_min": (27.0, 502.0)}
        defn = build_definition(client, "…dropping platelets…", reference_ranges=frozen)
        assert defn.traits[0].reference_value == 27.0

    def test_decomposer_cq_hints_are_passed_to_the_model(self):
        # The builder consumes the decomposer's structured CQ (filters /
        # concepts) as guidance, so those hints must reach the model's turn.
        from src.conversational.models import CompetencyQuestion, PatientFilter

        cq = CompetencyQuestion(
            original_question="Find septic ICU patients like a 68yo woman",
            scope="patient_similarity",
            patient_filters=[
                PatientFilter(field="diagnosis", operator="contains", value="sepsis"),
            ],
        )
        client = _mock_client([_defn_json()])
        build_definition(client, "Find septic ICU patients…", cq=cq)
        msgs = client.messages.create.call_args.kwargs["messages"]
        assert "sepsis" in json.dumps(msgs)

    def test_graph_temporal_trait_preserved_and_not_catalog_checked(self):
        # graph_temporal trait names are graph-derived, NOT extractor columns,
        # so they are exempt from the SQL catalog guard. Post-III-A they carry a
        # feature-extractor template (validated by the graph guard instead).
        j = json.dumps({
            "traits": [{
                "name": "lactate_slope_48h", "source": "graph_temporal",
                "kind": "quantitative", "reference_value": 1.0,
                "direction": "higher_more_similar", "weight": 2.0,
                "template": "sim_series_by_admission", "concept": "lactate",
                "graph_params": {"agg": "slope", "window_hours": 48},
            }],
        })
        client = _mock_client([j])
        defn = build_definition(client, "…worsening lactate…")
        assert defn.traits[0].source == "graph_temporal"
        assert defn.traits[0].template == "sim_series_by_admission"
        assert defn.traits[0].direction == Direction.HIGHER_MORE_SIMILAR
        # Only one LLM call: a valid graph_temporal trait is not an "offender".
        assert client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Self-repair retries (mirrors the decomposer's retry discipline).
# ---------------------------------------------------------------------------


class TestRetryRepair:
    def test_invalid_json_retries_once_then_succeeds(self):
        client = _mock_client(["not json at all", _defn_json()])
        defn = build_definition(client, "…")
        assert client.messages.create.call_count == 2
        assert isinstance(defn, CohortDefinition)

    def test_unknown_sql_trait_name_triggers_corrective_retry(self):
        bad = json.dumps({
            "traits": [{
                "name": "blood_pressure", "source": "sql", "kind": "quantitative",
                "reference_value": 120,
            }],
        })
        client = _mock_client([bad, _defn_json()])
        defn = build_definition(client, "…")
        assert client.messages.create.call_count == 2
        # The corrective turn must name the offender + show the allowed vocab.
        second = client.messages.create.call_args_list[1]
        convo = json.dumps(second.kwargs["messages"])
        assert "blood_pressure" in convo
        assert "creatinine_max" in convo  # allowed-names list echoed
        assert {t.name for t in defn.traits} == {"age", "gender", "creatinine_max"}

    def test_persistent_unknown_trait_name_raises(self):
        bad = json.dumps({
            "traits": [{
                "name": "blood_pressure", "source": "sql", "kind": "quantitative",
                "reference_value": 120,
            }],
        })
        client = _mock_client([bad, bad])
        with pytest.raises(ValueError, match="blood_pressure"):
            build_definition(client, "…")

    def test_persistent_invalid_json_raises(self):
        client = _mock_client(["garbage", "still garbage"])
        with pytest.raises(Exception):
            build_definition(client, "…")


# ---------------------------------------------------------------------------
# System prompt — structural guardrail (the taxonomy lives here).
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_prompt_encodes_directional_taxonomy(self):
        p = build_definition_system_prompt().lower()
        for cue in ("worse", "rising", "higher_more_similar",
                    "lower_more_similar", "symmetric"):
            assert cue in p

    def test_prompt_lists_feature_catalog_and_prefilter_fields(self):
        p = build_definition_system_prompt()
        for name in ("age", "creatinine_max", "sodium_mean", "gender",
                     "admission_type"):
            assert name in p
        # prefilter vocabulary (the PatientFilter fields) is surfaced too.
        assert "diagnosis" in p


# ---------------------------------------------------------------------------
# III-D: graph_temporal guardrails — the prompt teaches the feature-extractor
# template vocabulary, and the builder validates/self-repairs against the live
# registry so the non-deterministic layer stays inside the template rails.
# ---------------------------------------------------------------------------


class TestGraphTemporalGuardrails:
    def test_prompt_teaches_graph_temporal_templates(self):
        p = build_definition_system_prompt()
        # Real template names are surfaced so the model picks an extractable one.
        for name in ("sim_series_by_admission", "sim_dose_series",
                     "sim_precedence_count"):
            assert name in p
        # The graph-trait fields are part of the structured-output contract.
        for key in ("graph_temporal", "template", "concept", "graph_params"):
            assert key in p

    def test_graph_template_guidance_matches_registry(self):
        # The prompt guidance must not drift from the live extractor registry:
        # every documented template exists, and every registered template is
        # documented.
        from src.similarity.definition_builder import _GRAPH_TEMPLATE_GUIDANCE
        from src.similarity.graph_features import TEMPLATES

        assert set(_GRAPH_TEMPLATE_GUIDANCE) == set(TEMPLATES)

    def test_valid_graph_temporal_trait_passes_in_one_call(self):
        client = _mock_client([_graph_defn_json()])
        defn = build_definition(client, "…worsening lactate over the first 48h…")
        assert client.messages.create.call_count == 1
        t = defn.traits[0]
        assert t.source == "graph_temporal"
        assert t.template == "sim_series_by_admission"
        assert t.concept == "lactate"
        assert t.direction == Direction.HIGHER_MORE_SIMILAR

    def test_missing_template_triggers_corrective_retry(self):
        bad = json.dumps({
            "traits": [{
                "name": "lactate_slope_48h", "source": "graph_temporal",
                "kind": "quantitative", "reference_value": 1.5,
                "direction": "higher_more_similar", "weight": 2.0,
            }],
        })
        client = _mock_client([bad, _graph_defn_json()])
        defn = build_definition(client, "…")
        assert client.messages.create.call_count == 2
        # The corrective turn names the offender + echoes the legal templates.
        convo = json.dumps(client.messages.create.call_args_list[1].kwargs["messages"])
        assert "lactate_slope_48h" in convo
        assert "sim_series_by_admission" in convo
        assert defn.traits[0].template == "sim_series_by_admission"

    def test_unknown_template_triggers_corrective_retry(self):
        bad = json.dumps({
            "traits": [{
                "name": "lactate_slope_48h", "source": "graph_temporal",
                "kind": "quantitative", "reference_value": 1.5,
                "direction": "higher_more_similar", "weight": 2.0,
                "template": "sim_made_up", "concept": "lactate",
            }],
        })
        client = _mock_client([bad, _graph_defn_json()])
        defn = build_definition(client, "…")
        assert client.messages.create.call_count == 2
        assert defn.traits[0].template == "sim_series_by_admission"

    def test_persistent_invalid_template_raises(self):
        bad = json.dumps({
            "traits": [{
                "name": "lactate_slope_48h", "source": "graph_temporal",
                "kind": "quantitative", "reference_value": 1.5,
                "direction": "higher_more_similar", "template": "sim_made_up",
            }],
        })
        client = _mock_client([bad, bad])
        with pytest.raises(ValueError, match="template"):
            build_definition(client, "…")

    def test_extract_json_tolerates_backticks_in_payload(self):
        # Same fence-anchoring fix as the decomposer: a payload value that
        # itself contains a triple-backtick must not truncate extraction.
        from src.similarity.definition_builder import _extract_json

        payload = '{"x": "```", "y": 1}'
        wrapped = f"```json\n{payload}\n```"
        assert json.loads(_extract_json(wrapped)) == {"x": "```", "y": 1}
