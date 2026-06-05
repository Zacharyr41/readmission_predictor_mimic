"""End-to-end tests for the contextual-only cohort runner (plan task III-B).

``run_cohort(definition, backend)`` is the anchorless one-vs-many path: a free-
text request has already been translated into a schema-validated
``CohortDefinition`` (prefilters + typed Gower ``TraitSpec`` columns + a distance
threshold + a top_k cap). The runner narrows the candidate pool with the
prefilters, pulls a typed feature matrix, synthesizes the reference *profile*
vector from each trait's ``reference_value``, and scores every candidate's
Gower distance to that profile — cohort = ``distance <= threshold`` (top_k cap).

This file pins the *contextual-only* contract (no ``graph_temporal`` traits):

  * membership / ordering / threshold / top_k against the synthetic DuckDB,
  * the load-bearing directional (one-sided) kernel — a candidate *more extreme*
    than the reference is **not** penalized (locked decision #2),
  * frozen ranges (locked decision #6) — the runner refuses to learn a
    normalization range from the query batch,
  * per-trait signed contributions surfaced for the explanation layer,
  * prefilters narrowing the pool via the shared filter registry,
  * ``graph_temporal`` traits raising a clear "deferred to III-A" error.

Synthetic cohort (``synthetic_duckdb_with_events``), one row per admission::

    hadm  subj  age  gender  admission_type  icu_los_hours  creatinine_max
    101   1     65   M       EMERGENCY       69.6           1.2
    102   1     65   M       EMERGENCY        0.0           1.2*  (median-filled)
    103   2     72   F       ELECTIVE        148.8          0.9
    104   3     58   M       EMERGENCY        0.0           1.2*
    105   4     45   F       URGENT           0.0           1.2*
    106   5     80   M       EMERGENCY       174.0          1.5
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.conversational.models import PatientFilter
from src.pygower import Direction, Kind
from src.similarity.models import (
    CohortDefinition,
    CohortMember,
    CohortResult,
    TraitContribution,
    TraitSpec,
)
from src.similarity.run import run_cohort

ONTOLOGY_DIR = Path(__file__).parent.parent.parent / "ontology" / "definition"


# ---------------------------------------------------------------------------
# Backend fixture — the real rich DuckDB backend (so prefilters compile via the
# shared OperationRegistry, exactly as in production). The thin
# ``similarity_backend`` adapter only exposes ``.execute`` and can't run the
# filter compiler, so we wrap the same synthetic connection in a
# ``_DuckDBBackend``.
# ---------------------------------------------------------------------------


@pytest.fixture
def rich_backend(synthetic_duckdb_with_events):
    from src.conversational.extractor import _DuckDBBackend

    backend = _DuckDBBackend.__new__(_DuckDBBackend)
    backend._conn = synthetic_duckdb_with_events
    return backend


def _age_trait(reference_value=68, weight=1.0, range_=(18.0, 68.0)):
    # range_ spans 50 years so distances are |age - ref| / 50 — round numbers.
    return TraitSpec(name="age", source="sql", kind="quantitative",
                     reference_value=reference_value, weight=weight, range_=range_)


# ---------------------------------------------------------------------------
# Core membership / ordering / threshold
# ---------------------------------------------------------------------------


class TestRunCohortContextual:
    def test_returns_cohort_result(self, rich_backend):
        result = run_cohort(CohortDefinition(traits=[_age_trait()]), rich_backend)
        assert isinstance(result, CohortResult)
        assert result.n_pool == 6
        assert all(isinstance(m, CohortMember) for m in result.members)

    def test_membership_and_distances(self, rich_backend):
        # Single symmetric age trait, ref=68, range=50 ⇒ distance = |age-68|/50:
        #   101/102→0.06, 103→0.08, 104→0.20, 106→0.24, 105→0.46.
        # threshold 0.35 keeps everyone but 105.
        result = run_cohort(
            CohortDefinition(traits=[_age_trait()], distance_threshold=0.35),
            rich_backend,
        )
        by_hadm = {m.hadm_id: m.distance for m in result.members}
        assert set(by_hadm) == {101, 102, 103, 104, 106}
        assert by_hadm[101] == pytest.approx(0.06)
        assert by_hadm[103] == pytest.approx(0.08)
        assert by_hadm[104] == pytest.approx(0.20)
        assert by_hadm[106] == pytest.approx(0.24)
        assert 105 not in by_hadm  # 0.46 > 0.35

    def test_members_sorted_by_ascending_distance(self, rich_backend):
        result = run_cohort(
            CohortDefinition(traits=[_age_trait()], distance_threshold=1.0),
            rich_backend,
        )
        dists = [m.distance for m in result.members]
        assert dists == sorted(dists)
        assert result.members[0].hadm_id in (101, 102)  # nearest age to 68

    def test_top_k_caps_returned_members(self, rich_backend):
        result = run_cohort(
            CohortDefinition(traits=[_age_trait()], distance_threshold=1.0, top_k=2),
            rich_backend,
        )
        assert result.n_returned == 2
        assert len(result.members) == 2
        # The two nearest by age are the 65yo admissions 101 + 102.
        assert {m.hadm_id for m in result.members} == {101, 102}

    def test_no_cap_returns_all_within_distance(self, rich_backend):
        # top_k=None (the default): no artificial cap — every candidate within
        # the Gower distance is returned. Threshold 1.0 admits the whole pool,
        # so all 6 come back even though the previous test capped the same pool
        # at 2.
        result = run_cohort(
            CohortDefinition(traits=[_age_trait()], distance_threshold=1.0, top_k=None),
            rich_backend,
        )
        assert result.n_pool == 6
        assert result.n_returned == 6
        assert len(result.members) == 6

    def test_tight_threshold_excludes(self, rich_backend):
        result = run_cohort(
            CohortDefinition(traits=[_age_trait()], distance_threshold=0.07),
            rich_backend,
        )
        # only 101/102 (0.06) clear 0.07; 103 (0.08) does not.
        assert {m.hadm_id for m in result.members} == {101, 102}

    def test_subject_ids_populated(self, rich_backend):
        result = run_cohort(
            CohortDefinition(traits=[_age_trait()], distance_threshold=1.0),
            rich_backend,
        )
        subj = {m.hadm_id: m.subject_id for m in result.members}
        assert subj[101] == 1 and subj[103] == 2 and subj[106] == 5


# ---------------------------------------------------------------------------
# Per-trait contributions
# ---------------------------------------------------------------------------


class TestContributions:
    def test_each_member_has_one_contribution_per_trait(self, rich_backend):
        defn = CohortDefinition(
            traits=[
                _age_trait(),
                TraitSpec(name="creatinine_max", source="sql", kind="quantitative",
                          reference_value=1.2, range_=(0.0, 5.0)),
            ],
            distance_threshold=1.0,
        )
        result = run_cohort(defn, rich_backend)
        for m in result.members:
            assert [c.name for c in m.contributions] == ["age", "creatinine_max"]
            assert all(isinstance(c, TraitContribution) for c in m.contributions)

    def test_signed_contribution_sign(self, rich_backend):
        # 101 is age 65 vs ref 68 — a near match ⇒ similarity > 0.5 ⇒ signed > 0.
        result = run_cohort(
            CohortDefinition(traits=[_age_trait()], distance_threshold=1.0),
            rich_backend,
        )
        m101 = next(m for m in result.members if m.hadm_id == 101)
        c = m101.contributions[0]
        assert c.name == "age"
        assert c.similarity == pytest.approx(0.94)  # 1 - 0.06
        assert c.signed > 0  # w*(2*0.94-1) = 0.88
        assert c.included is True


# ---------------------------------------------------------------------------
# Directional (one-sided) kernel — the load-bearing correctness decision
# ---------------------------------------------------------------------------


class TestDirectionalKernel:
    def _creatinine_defn(self, direction):
        return CohortDefinition(
            traits=[TraitSpec(
                name="creatinine_max", source="sql", kind="quantitative",
                direction=direction, reference_value=1.0, range_=(0.0, 2.0),
            )],
            distance_threshold=1.0,
        )

    def test_higher_more_similar_does_not_penalize_exceeding(self, rich_backend):
        # ref=1.0; hadm 106 has creatinine 1.5 (> ref).
        directional = run_cohort(
            self._creatinine_defn(Direction.HIGHER_MORE_SIMILAR), rich_backend
        )
        symmetric = run_cohort(
            self._creatinine_defn(Direction.SYMMETRIC), rich_backend
        )
        d106 = next(m for m in directional.members if m.hadm_id == 106).distance
        s106 = next(m for m in symmetric.members if m.hadm_id == 106).distance
        # one-sided: exceeding the reference is fully similar ⇒ distance 0.
        assert d106 == pytest.approx(0.0)
        # symmetric penalizes the same candidate: |1.5-1.0|/2 = 0.25.
        assert s106 == pytest.approx(0.25)
        assert d106 < s106

    def test_one_sided_still_penalizes_shortfall(self, rich_backend):
        # hadm 103 has creatinine 0.9 (< ref 1.0) ⇒ penalized either way.
        directional = run_cohort(
            self._creatinine_defn(Direction.HIGHER_MORE_SIMILAR), rich_backend
        )
        d103 = next(m for m in directional.members if m.hadm_id == 103).distance
        assert d103 == pytest.approx(0.05)  # (1.0-0.9)/2


# ---------------------------------------------------------------------------
# Frozen ranges (locked decision #6)
# ---------------------------------------------------------------------------


class TestFrozenRanges:
    def test_quantitative_trait_without_range_raises(self, rich_backend):
        defn = CohortDefinition(traits=[
            TraitSpec(name="age", source="sql", kind="quantitative",
                      reference_value=68),  # no range_
        ])
        with pytest.raises(ValueError, match="range|frozen"):
            run_cohort(defn, rich_backend)

    def test_reference_ranges_supplies_frozen_range(self, rich_backend):
        defn = CohortDefinition(
            traits=[TraitSpec(name="age", source="sql", kind="quantitative",
                              reference_value=68)],  # no range_ on the trait
            distance_threshold=1.0,
        )
        result = run_cohort(defn, rich_backend, reference_ranges={"age": (18.0, 68.0)})
        by_hadm = {m.hadm_id: m.distance for m in result.members}
        assert by_hadm[101] == pytest.approx(0.06)  # |65-68|/50


# ---------------------------------------------------------------------------
# Prefilters narrow the pool via the shared registry
# ---------------------------------------------------------------------------


class TestPrefilters:
    def test_gender_prefilter_narrows_pool(self, rich_backend):
        defn = CohortDefinition(
            prefilters=[PatientFilter(field="gender", operator="=", value="F")],
            traits=[_age_trait()],
            distance_threshold=1.0,
        )
        result = run_cohort(defn, rich_backend)
        # only the two female admissions (103 subj2, 105 subj4) survive the gate.
        assert result.n_pool == 2
        assert {m.hadm_id for m in result.members} <= {103, 105}

    def test_admission_type_prefilter(self, rich_backend):
        defn = CohortDefinition(
            prefilters=[PatientFilter(field="admission_type", operator="=",
                                      value="EMERGENCY")],
            traits=[_age_trait()],
            distance_threshold=1.0,
        )
        result = run_cohort(defn, rich_backend)
        # EMERGENCY admissions: 101, 102, 104, 106.
        assert result.n_pool == 4
        assert {m.hadm_id for m in result.members} <= {101, 102, 104, 106}


# ---------------------------------------------------------------------------
# graph_temporal traits build the per-question RDF graph over the pool (III-A)
# ---------------------------------------------------------------------------


def _install_graph_spy(monkeypatch):
    """Replace ``build_query_graph`` with a spy that records its call kwargs.

    Returns the call-record list. The spy returns an empty graph, so the graph
    feature columns come back all-NaN — fine for routing assertions (we only
    care THAT the graph was built and with which ``skip_allen_relations``).
    """
    from rdflib import Graph

    calls: list[dict] = []

    def _spy(ontology_dir, extraction, *, skip_allen_relations=False,
             max_workers=1, drug_category_resolver=None):
        calls.append({
            "skip_allen_relations": skip_allen_relations,
            "ontology_dir": ontology_dir,
            "max_workers": max_workers,
            "n_patients": len(extraction.patients),
        })
        return Graph(), {}

    monkeypatch.setattr(
        "src.conversational.graph_builder.build_query_graph", _spy,
    )
    return calls


def _lactate_slope_trait():
    return TraitSpec(
        name="lactate_slope_48h", source="graph_temporal", kind="quantitative",
        direction="higher_more_similar", template="sim_series_by_admission",
        concept="lactate", reference_value=1.5, range_=(0.0, 5.0),
        graph_params={"agg": "slope", "window_hours": 48},
    )


class TestGraphTemporalWiring:
    def test_missing_ontology_dir_raises_valueerror(self, rich_backend):
        # The old contract raised NotImplementedError; now a graph_temporal trait
        # without an ontology_dir is a clear ValueError (the graph path is real,
        # it just needs the ontology to build the per-question graph).
        defn = CohortDefinition(traits=[_age_trait(), _lactate_slope_trait()])
        with pytest.raises(ValueError, match="ontology_dir|graph_temporal"):
            run_cohort(defn, rich_backend)  # no ontology_dir supplied

    def test_graph_temporal_without_template_raises(self, rich_backend, tmp_path):
        defn = CohortDefinition(traits=[
            TraitSpec(name="lactate_slope_48h", source="graph_temporal",
                      kind="quantitative", reference_value=1.5, range_=(0.0, 5.0)),
        ])
        with pytest.raises(ValueError, match="template"):
            run_cohort(defn, rich_backend, ontology_dir=tmp_path)

    def test_builds_graph_for_temporal_trait(self, rich_backend, tmp_path, monkeypatch):
        calls = _install_graph_spy(monkeypatch)
        defn = CohortDefinition(
            traits=[_lactate_slope_trait()], distance_threshold=1.0,
        )
        result = run_cohort(defn, rich_backend, ontology_dir=tmp_path)
        assert isinstance(result, CohortResult)
        assert len(calls) == 1
        # No precedence trait ⇒ the (expensive) Allen pass is skipped.
        assert calls[0]["skip_allen_relations"] is True
        assert result.provenance["graph_built"] is True
        assert "lactate_slope_48h" in result.provenance["graph_traits"]

    def test_precedence_trait_requests_allen(self, rich_backend, tmp_path, monkeypatch):
        # A precedence trait needs Allen relations, so skip_allen_relations=False
        # (locked decision I-D: only pay for Allen when precedence is asked for).
        calls = _install_graph_spy(monkeypatch)
        defn = CohortDefinition(
            traits=[TraitSpec(
                name="lactate_before_pressor", source="graph_temporal",
                kind="binary", template="sim_precedence_count", concept="lactate",
                reference_value=True,
                graph_params={"concept_b": "norepinephrine", "concept_b_type": "drug",
                              "relation": "meets", "as_bool": True},
            )],
            distance_threshold=1.0,
        )
        run_cohort(defn, rich_backend, ontology_dir=tmp_path)
        assert len(calls) == 1
        assert calls[0]["skip_allen_relations"] is False

    def test_sql_only_skips_graph_build(self, rich_backend, tmp_path, monkeypatch):
        calls = _install_graph_spy(monkeypatch)
        run_cohort(
            CohortDefinition(traits=[_age_trait()], distance_threshold=1.0),
            rich_backend, ontology_dir=tmp_path,
        )
        assert calls == []  # contextual-only ⇒ no graph build


class TestGraphTemporalEndToEnd:
    """Real graph build over the synthetic DB (no mocking) — proves the full
    extract → build_query_graph → extract_graph_features → merge → score path.

    Uses ``sim_icu_los`` (a concept-free duration template) because the synthetic
    labs are single-reading, so a slope is undefined; ICU LOS is real data:
    stay 1001 spans 70h (101), 1002 148h (103), 1003 174h (106); 102/104/105 have
    no ICU stay so their graph LOS is NaN.
    """

    def _icu_los_defn(self, reference_value=70.0, threshold=1.0):
        return CohortDefinition(
            traits=[TraitSpec(
                name="icu_stay_hours", source="graph_temporal", kind="quantitative",
                template="sim_icu_los", reference_value=reference_value,
            )],
            distance_threshold=threshold,
        )

    def test_icu_los_graph_trait_scored(self, rich_backend):
        result = run_cohort(
            self._icu_los_defn(), rich_backend, ontology_dir=ONTOLOGY_DIR,
            reference_ranges={"icu_stay_hours": (0.0, 240.0)},
        )
        by_hadm = {m.hadm_id: m.distance for m in result.members}
        # Only admissions with an ICU stay get a graph-derived LOS.
        assert set(by_hadm) <= {101, 103, 106}
        assert 101 in by_hadm
        assert by_hadm[101] == pytest.approx(0.0)  # ref 70h == stay 1001's LOS
        # 102/104/105 have no ICU stay ⇒ NaN LOS ⇒ excluded.
        assert {102, 104, 105}.isdisjoint(by_hadm)

    def test_graph_provenance_recorded(self, rich_backend):
        result = run_cohort(
            self._icu_los_defn(), rich_backend, ontology_dir=ONTOLOGY_DIR,
            reference_ranges={"icu_stay_hours": (0.0, 240.0)},
        )
        assert result.provenance["graph_built"] is True
        assert result.provenance["graph_skip_allen_relations"] is True
        assert result.provenance["graph_traits"] == ["icu_stay_hours"]


# ---------------------------------------------------------------------------
# Provenance carries the criteria (for the orchestrator to log — II-D / I-E)
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_provenance_records_criteria(self, rich_backend):
        defn = CohortDefinition(
            prefilters=[PatientFilter(field="gender", operator="=", value="F")],
            traits=[_age_trait()],
            distance_threshold=0.4,
            top_k=10,
        )
        result = run_cohort(defn, rich_backend)
        prov = result.provenance
        assert prov["distance_threshold"] == 0.4
        assert prov["top_k"] == 10
        assert any(t["name"] == "age" for t in prov["traits"])
