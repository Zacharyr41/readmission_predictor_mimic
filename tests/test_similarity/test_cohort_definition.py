"""Schema tests for the cohort-by-similarity definition (plan task II-C).

``CohortDefinition`` is the formal, schema-validated translation of a free-text
cohort request: a Boolean ``prefilters`` gate, a list of typed ``TraitSpec``
Gower columns, an LLM-proposed ``distance_threshold``, and a ``top_k`` cap.

These tests pin the contract the definition builder (II-D) must emit and the
cohort runner (III-B) consumes — including the bridge to pygower's ``ColumnSpec``
and the JSON round-trip the LLM output must survive.
"""

from __future__ import annotations

import pytest

from src.conversational.models import PatientFilter
from src.pygower import ColumnSpec, Direction, Kind, Missing
from src.pygower.kernels import one_sided_kernel, quantitative_kernel, select_kernel
from src.similarity.models import CohortDefinition, SimilaritySpec, TraitSpec


# ---------------------------------------------------------------------------
# TraitSpec
# ---------------------------------------------------------------------------


class TestTraitSpec:
    def test_quantitative_symmetric_trait_ok(self):
        t = TraitSpec(name="age", source="sql", kind="quantitative",
                      reference_value=68, weight=0.6)
        assert t.kind == Kind.QUANTITATIVE
        assert t.direction == Direction.SYMMETRIC
        assert t.reference_value == 68
        assert t.weight == 0.6

    def test_directional_quantitative_trait_ok(self):
        t = TraitSpec(name="lactate_slope_48h", source="graph_temporal",
                      kind="quantitative", direction="higher_more_similar",
                      reference_value=1.5, weight=2.0)
        assert t.direction == Direction.HIGHER_MORE_SIMILAR

    def test_binary_presence_trait_ok(self):
        t = TraitSpec(name="sepsis", source="sql", kind="binary",
                      asymmetric=True, present_value=True, reference_value=True,
                      weight=1.5)
        assert t.kind == Kind.BINARY
        assert t.asymmetric is True

    def test_directional_non_quantitative_rejected(self):
        # The one-sided kernel is only defined for quantitative columns.
        with pytest.raises(ValueError, match="directional|quantitative"):
            TraitSpec(name="x", source="sql", kind="binary",
                      direction="higher_more_similar", reference_value=True)

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="weight"):
            TraitSpec(name="x", source="sql", kind="quantitative",
                      reference_value=1, weight=-0.5)

    def test_missing_policy_default_and_override(self):
        default = TraitSpec(name="x", source="sql", kind="quantitative",
                            reference_value=1)
        assert default.missing == Missing.EXCLUDE
        zero = TraitSpec(name="x", source="sql", kind="quantitative",
                         reference_value=1, missing="zero_similarity")
        assert zero.missing == Missing.ZERO_SIMILARITY

    def test_unknown_field_forbidden(self):
        with pytest.raises(ValueError):
            TraitSpec(name="x", source="sql", kind="quantitative",
                      reference_value=1, bogus=123)

    def test_to_column_spec_maps_fields(self):
        t = TraitSpec(name="los", source="sql", kind="quantitative",
                      direction="lower_more_similar", reference_value=15,
                      range_=(0.0, 30.0), weight=0.5, missing="zero_similarity")
        spec = t.to_column_spec()
        assert isinstance(spec, ColumnSpec)
        assert spec.kind == Kind.QUANTITATIVE
        assert spec.direction == Direction.LOWER_MORE_SIMILAR
        assert spec.range_ == (0.0, 30.0)
        assert spec.weight == 0.5
        assert spec.missing == Missing.ZERO_SIMILARITY

    def test_to_column_spec_selects_one_sided_kernel_for_directional(self):
        directional = TraitSpec(name="s", source="graph_temporal",
                                kind="quantitative",
                                direction="higher_more_similar",
                                reference_value=1.0).to_column_spec()
        assert select_kernel(directional) is one_sided_kernel
        symmetric = TraitSpec(name="age", source="sql", kind="quantitative",
                              reference_value=68).to_column_spec()
        assert select_kernel(symmetric) is quantitative_kernel

    def test_binary_to_column_spec_is_asymmetric(self):
        spec = TraitSpec(name="sepsis", source="sql", kind="binary",
                         asymmetric=True, reference_value=True).to_column_spec()
        assert spec.kind == Kind.BINARY
        assert spec.asymmetric is True

    def test_json_round_trip_preserves_fields(self):
        t = TraitSpec(name="lactate_slope_48h", source="graph_temporal",
                      kind="quantitative", direction="higher_more_similar",
                      reference_value=1.5, range_=(0.0, 5.0), weight=2.0,
                      missing="zero_similarity")
        back = TraitSpec.model_validate_json(t.model_dump_json())
        assert back == t


# ---------------------------------------------------------------------------
# CohortDefinition
# ---------------------------------------------------------------------------


def _trait(name="age", **kw):
    base = dict(source="sql", kind="quantitative", reference_value=68)
    base.update(kw)
    return TraitSpec(name=name, **base)


class TestCohortDefinition:
    def test_minimal_valid_definition(self):
        d = CohortDefinition(traits=[_trait()])
        assert d.distance_threshold == pytest.approx(0.35)
        # Default is NO cap: once a candidate is within the Gower distance it
        # belongs in the cohort — top_k is opt-in ("top N"), not the default.
        assert d.top_k is None
        assert d.prefilters == []

    def test_definition_with_prefilters(self):
        d = CohortDefinition(
            prefilters=[
                PatientFilter(field="in_icu", operator="=", value="true"),
                PatientFilter(field="hospital_los_days", operator="<=", value="15"),
            ],
            traits=[_trait(), _trait(name="sepsis", kind="binary",
                                     asymmetric=True, reference_value=True)],
            distance_threshold=0.4,
            top_k=50,
        )
        assert len(d.prefilters) == 2
        assert len(d.traits) == 2

    def test_empty_traits_rejected(self):
        with pytest.raises(ValueError, match="trait"):
            CohortDefinition(traits=[])

    def test_threshold_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="threshold"):
            CohortDefinition(traits=[_trait()], distance_threshold=1.5)
        with pytest.raises(ValueError, match="threshold"):
            CohortDefinition(traits=[_trait()], distance_threshold=-0.1)

    def test_top_k_positive_or_none(self):
        with pytest.raises(ValueError, match="top_k"):
            CohortDefinition(traits=[_trait()], top_k=0)
        assert CohortDefinition(traits=[_trait()], top_k=None).top_k is None

    def test_duplicate_trait_names_rejected(self):
        with pytest.raises(ValueError, match="unique|duplicate"):
            CohortDefinition(traits=[_trait(name="age"), _trait(name="age")])

    def test_json_round_trip(self):
        d = CohortDefinition(
            prefilters=[PatientFilter(field="in_icu", operator="=", value="true")],
            traits=[_trait(), _trait(name="los", direction="lower_more_similar",
                                     reference_value=15)],
            distance_threshold=0.3,
        )
        back = CohortDefinition.model_validate_json(d.model_dump_json())
        assert back == d


# ---------------------------------------------------------------------------
# SimilaritySpec.cohort_definition (additive; anchorless cohort mode)
# ---------------------------------------------------------------------------


class TestSimilaritySpecCohortMode:
    def test_cohort_definition_without_anchor_ok(self):
        spec = SimilaritySpec(cohort_definition=CohortDefinition(traits=[_trait()]))
        assert spec.cohort_definition is not None
        assert spec.anchor_hadm_id is None

    def test_cohort_definition_with_anchor_rejected(self):
        with pytest.raises(ValueError, match="both|cohort_definition"):
            SimilaritySpec(anchor_hadm_id=101,
                           cohort_definition=CohortDefinition(traits=[_trait()]))

    def test_no_anchor_no_cohort_still_rejected(self):
        # Regression: the original "exactly one anchor" rule still fires when
        # neither an anchor nor a cohort_definition is supplied.
        with pytest.raises(ValueError, match="exactly one"):
            SimilaritySpec()

    def test_anchor_only_unchanged(self):
        spec = SimilaritySpec(anchor_hadm_id=101)
        assert spec.cohort_definition is None
