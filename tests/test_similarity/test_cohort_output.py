"""III-C: cohort output — quantities of interest + downloadable CSV.

``CohortResult`` owns its presentation-layer serialization: cohort-level
quantities of interest (size, distance distribution, per-trait cohort means vs
the reference profile) and a ready-to-download CSV of ``(subject_id, hadm_id)``
pairs. These are pure model methods — no backend, no LLM, no graph — so they
are tested in isolation here. The cohort is admission-keyed (the candidate pool
is hadm-scoped), so the encounter identifier is ``hadm_id``; MIMIC's
ICU-specific ``stay_id`` is intentionally not fabricated.
"""

from __future__ import annotations

import csv
import io

import pytest

from src.similarity.models import (
    CohortDefinition,
    CohortMember,
    CohortResult,
    TraitContribution,
    TraitSpec,
)


def _defn(threshold: float = 0.35) -> CohortDefinition:
    return CohortDefinition(
        traits=[
            TraitSpec(
                name="age", source="sql", kind="quantitative",
                reference_value=68, range_=(18.0, 90.0), weight=0.6,
            ),
            TraitSpec(
                name="sepsis", source="sql", kind="binary",
                reference_value=True, asymmetric=True, weight=1.5,
            ),
        ],
        distance_threshold=threshold,
        top_k=30,
    )


def _member(
    hadm: int, subject: int, dist: float, *,
    age_sim: float, age_signed: float,
    sepsis_sim: float, sepsis_signed: float, sepsis_incl: bool = True,
) -> CohortMember:
    return CohortMember(
        hadm_id=hadm, subject_id=subject, distance=dist,
        contributions=[
            TraitContribution(
                name="age", similarity=age_sim, signed=age_signed,
                weight=0.6, included=True,
            ),
            TraitContribution(
                name="sepsis", similarity=sepsis_sim, signed=sepsis_signed,
                weight=1.5, included=sepsis_incl,
            ),
        ],
    )


class TestQuantitiesOfInterest:
    def test_empty_cohort_has_no_distance_block(self):
        result = CohortResult(
            definition=_defn(), members=[], n_pool=10, n_returned=0,
        )
        qoi = result.quantities_of_interest()
        assert qoi["n_returned"] == 0
        assert qoi["n_pool"] == 10
        assert qoi["distance"] is None
        # Every trait still appears even with no members contributing.
        names = {pt["name"] for pt in qoi["per_trait"]}
        assert names == {"age", "sepsis"}
        for pt in qoi["per_trait"]:
            assert pt["included_count"] == 0
            assert pt["mean_similarity"] is None
            assert pt["mean_signed"] is None

    def test_distance_distribution(self):
        members = [
            _member(101, 1, 0.10, age_sim=0.9, age_signed=0.48,
                    sepsis_sim=1.0, sepsis_signed=1.5),
            _member(102, 2, 0.20, age_sim=0.7, age_signed=0.24,
                    sepsis_sim=1.0, sepsis_signed=1.5),
            _member(103, 3, 0.30, age_sim=0.5, age_signed=0.0,
                    sepsis_sim=1.0, sepsis_signed=1.5),
        ]
        result = CohortResult(
            definition=_defn(), members=members, n_pool=50, n_returned=3,
        )
        d = result.quantities_of_interest()["distance"]
        assert d["min"] == pytest.approx(0.10)
        assert d["max"] == pytest.approx(0.30)
        assert d["median"] == pytest.approx(0.20)
        assert d["mean"] == pytest.approx(0.20)

    def test_per_trait_means_vs_reference(self):
        members = [
            _member(101, 1, 0.10, age_sim=0.9, age_signed=0.48,
                    sepsis_sim=1.0, sepsis_signed=1.5),
            _member(102, 2, 0.20, age_sim=0.7, age_signed=0.24,
                    sepsis_sim=1.0, sepsis_signed=1.5),
        ]
        result = CohortResult(
            definition=_defn(), members=members, n_pool=50, n_returned=2,
        )
        by_name = {
            pt["name"]: pt
            for pt in result.quantities_of_interest()["per_trait"]
        }
        assert by_name["age"]["reference_value"] == 68
        assert by_name["age"]["weight"] == pytest.approx(0.6)
        assert by_name["age"]["mean_similarity"] == pytest.approx(0.8)
        assert by_name["age"]["mean_signed"] == pytest.approx(0.36)
        assert by_name["age"]["included_count"] == 2
        assert by_name["sepsis"]["mean_similarity"] == pytest.approx(1.0)

    def test_excluded_trait_not_counted_in_similarity_mean(self):
        # One member's sepsis trait was excluded (NaN under the exclude policy);
        # it must not dilute the cohort mean similarity for that trait.
        members = [
            _member(101, 1, 0.10, age_sim=0.9, age_signed=0.48,
                    sepsis_sim=1.0, sepsis_signed=1.5),
            _member(102, 2, 0.20, age_sim=0.7, age_signed=0.24,
                    sepsis_sim=0.0, sepsis_signed=0.0, sepsis_incl=False),
        ]
        result = CohortResult(
            definition=_defn(), members=members, n_pool=50, n_returned=2,
        )
        by_name = {
            pt["name"]: pt
            for pt in result.quantities_of_interest()["per_trait"]
        }
        assert by_name["sepsis"]["included_count"] == 1
        assert by_name["sepsis"]["mean_similarity"] == pytest.approx(1.0)


class TestToCsv:
    def test_header_and_rows(self):
        members = [
            CohortMember(hadm_id=101, subject_id=1, distance=0.06),
            CohortMember(hadm_id=103, subject_id=2, distance=0.08),
        ]
        result = CohortResult(
            definition=_defn(), members=members, n_pool=6, n_returned=2,
        )
        rows = list(csv.reader(io.StringIO(result.to_csv())))
        assert rows[0] == ["rank", "subject_id", "hadm_id", "distance"]
        assert rows[1] == ["1", "1", "101", "0.06"]
        assert rows[2] == ["2", "2", "103", "0.08"]

    def test_empty_cohort_is_header_only(self):
        result = CohortResult(
            definition=_defn(), members=[], n_pool=6, n_returned=0,
        )
        rows = list(csv.reader(io.StringIO(result.to_csv())))
        assert rows == [["rank", "subject_id", "hadm_id", "distance"]]

    def test_rank_follows_member_order(self):
        # run_cohort pre-sorts members nearest-first; the CSV preserves it.
        members = [
            CohortMember(hadm_id=5, subject_id=50, distance=0.01),
            CohortMember(hadm_id=9, subject_id=90, distance=0.50),
        ]
        result = CohortResult(
            definition=_defn(), members=members, n_pool=2, n_returned=2,
        )
        rows = list(csv.reader(io.StringIO(result.to_csv())))
        assert [r[0] for r in rows[1:]] == ["1", "2"]
        assert [r[2] for r in rows[1:]] == ["5", "9"]
