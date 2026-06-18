"""Tests for the SQL fast-path compiler.

Phase 7a: the fast-path turns aggregate/comparison/list CQs into a single
SQL statement, bypassing extract+graph+reason entirely. The core invariant:
**fast-path result rows match graph-path result rows** (within float
tolerance). If they don't, the clinician gets a different answer depending
on which path the planner chose — unacceptable.

Tests are parametrized over a fixture set of CQs so adding coverage is
a one-case append. Each parity case runs both paths against a shared
synthetic DuckDB fixture and asserts equal result sets.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from src.conversational.extractor import _DuckDBBackend
from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    PatientFilter,
)


# ---------------------------------------------------------------------------
# Conn-sharing backend adapter (same trick as test_operations.py — DuckDB
# rejects mixed read/write handles on the same file, so we wrap the
# already-open connection from the synthetic_duckdb fixture).
# ---------------------------------------------------------------------------


class _ConnBackend(_DuckDBBackend):
    def __init__(self, conn) -> None:
        self._conn = conn

    def close(self) -> None:
        pass


@pytest.fixture
def backend(synthetic_duckdb_with_events):
    return _ConnBackend(synthetic_duckdb_with_events)


@pytest.fixture
def backend_with_creatinine_variants(synthetic_duckdb_with_events):
    """Adds urine (51082) and 24-hr (51067) creatinine to the base fixture
    so the LIKE-pooling bug is observable end-to-end. The base fixture has
    only itemid 50912 (serum), so without these extra rows the LIKE bug is
    invisible at the synthetic level.

    Values chosen to make the pollution unmistakable: serum is ~1.2 mg/dL,
    urine creatinine is typically 20-300 mg/dL, 24-hr collections are
    measured in mg/24hr (hundreds to thousands). A LIKE-pooled mean differs
    from a serum-restricted mean by orders of magnitude, so the assertion
    is trivially distinguishable.
    """
    conn = synthetic_duckdb_with_events
    conn.execute(
        "INSERT INTO d_labitems VALUES "
        "(51082, 'Urine Creatinine', 'Urine', 'Chemistry'), "
        "(51067, 'Creatinine 24-Hour', 'Urine', 'Chemistry')"
    )
    conn.execute(
        "INSERT INTO labevents VALUES "
        "(5, 1, 101, 1001, 51082, '2150-01-16 12:00:00', 100.0, 'mg/dL', NULL, NULL), "
        "(6, 2, 103, 1002, 51067, '2151-03-04 10:00:00', 1200.0, 'mg/24hr', NULL, NULL)"
    )
    return _ConnBackend(conn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cq(
    *,
    concepts: list[tuple] | None = None,
    filters: list[tuple[str, str, str]] | None = None,
    aggregation: str | None = None,
    scope: str = "cohort",
    comparison_field: str | None = None,
    return_type: str = "text_and_table",
) -> CompetencyQuestion:
    """Build a CompetencyQuestion for tests.

    ``concepts`` accepts 2-tuples ``(name, concept_type)`` or 3-tuples
    ``(name, concept_type, loinc_code)`` — the third element exercises
    the LOINC-grounded biomarker resolution path. Mixing both shapes in
    one list is allowed.
    """
    def _make_concept(c: tuple) -> ClinicalConcept:
        name, ctype, *rest = c
        return ClinicalConcept(
            name=name,
            concept_type=ctype,
            loinc_code=rest[0] if rest else None,
        )

    return CompetencyQuestion(
        original_question="test",
        clinical_concepts=[_make_concept(c) for c in (concepts or [])],
        patient_filters=[
            PatientFilter(field=f, operator=o, value=v)
            for f, o, v in (filters or [])
        ],
        aggregation=aggregation,
        scope=scope,
        comparison_field=comparison_field,
        return_type=return_type,
    )


def _rows_equal(a: list[dict], b: list[dict], *, tol: float = 1e-6) -> bool:
    """Order-insensitive row equality with float tolerance.

    Rows compared on their value sets — column names on both sides must
    match the reasoner's SPARQL shape, which is what the parity test
    enforces overall.
    """
    if len(a) != len(b):
        return False

    def _norm(r: dict) -> tuple:
        return tuple(
            (k, round(v, 6) if isinstance(v, float) else v)
            for k, v in sorted(r.items())
        )

    return sorted(map(_norm, a)) == sorted(map(_norm, b))


# ---------------------------------------------------------------------------
# 1. Module surface
# ---------------------------------------------------------------------------


class TestSqlFastpathModule:
    def test_module_exports_compile_and_result_type(self):
        from src.conversational import sql_fastpath

        assert hasattr(sql_fastpath, "compile_sql")
        assert hasattr(sql_fastpath, "SqlFastpathQuery")

    def test_compiled_query_has_sql_params_columns(self, backend):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("creatinine", "biomarker")],
            aggregation="mean",
        )
        q = compile_sql(cq, backend, get_default_registry())
        assert isinstance(q.sql, str) and q.sql.strip()
        assert isinstance(q.params, list)
        assert isinstance(q.columns, list) and q.columns


# ---------------------------------------------------------------------------
# 2. Column-shape contract — columns match reasoner SPARQL output
# ---------------------------------------------------------------------------


class TestColumnShape:
    """Every registered SQL-fast aggregate must emit a column name that
    downstream consumers (answerer._COLUMN_MAP + _camel_to_title) already
    understand. Column mismatches would silently produce empty/generic
    answers."""

    @pytest.mark.parametrize("agg,expected_col", [
        ("mean", "mean_value"),
        ("avg", "mean_value"),  # alias
        ("max", "max_value"),
        ("min", "min_value"),
        ("count", "count_value"),
    ])
    def test_biomarker_aggregate_column(self, backend, agg, expected_col):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(concepts=[("creatinine", "biomarker")], aggregation=agg)
        q = compile_sql(cq, backend, get_default_registry())
        assert expected_col in q.columns

    @pytest.mark.parametrize("axis", [
        "gender", "admission_type", "readmitted_30d",
        "readmitted_60d", "discharge_location",
    ])
    def test_comparison_axis_emits_group_value_column(self, backend, axis):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("creatinine", "biomarker")],
            aggregation="mean",
            scope="comparison",
            comparison_field=axis,
        )
        q = compile_sql(cq, backend, get_default_registry())
        assert "group_value" in q.columns
        assert "avg_value" in q.columns
        assert "count" in q.columns


# ---------------------------------------------------------------------------
# 3. End-to-end SQL parity with the synthetic DuckDB fixture
# ---------------------------------------------------------------------------


def _run_fastpath(backend, cq) -> list[dict]:
    """Mirrors the orchestrator's fast-path wiring: for biomarker concepts
    that carry a LOINC code, resolve to MIMIC itemids before compiling so
    the WHERE clause uses ``itemid IN`` instead of ``LIKE``."""
    from pathlib import Path

    from src.conversational.concept_resolver import ConceptResolver
    from src.conversational.operations import get_default_registry
    from src.conversational.sql_fastpath import compile_sql

    resolved_itemids: list[int] | None = None
    if cq.clinical_concepts and cq.clinical_concepts[0].concept_type == "biomarker":
        concept = cq.clinical_concepts[0]
        if concept.loinc_code:
            resolver = ConceptResolver(
                mappings_dir=Path(__file__).parent.parent.parent / "data" / "mappings",
            )
            biom = resolver.resolve_biomarker(concept)
            resolved_itemids = biom.itemids

    q = compile_sql(
        cq, backend, get_default_registry(),
        resolved_itemids=resolved_itemids,
    )
    rows = backend.execute(q.sql, q.params)
    return [dict(zip(q.columns, r)) for r in rows]


class TestMicrobiologyOrganismQualifier:
    """A microbiology concept that names BOTH a specimen and an organism must
    AND the two — the answer is the *intersection* (a blood culture that grew
    *this* organism), not the union of "any blood culture" and "any isolate of
    the organism".

    iter10 bug: ``_compile_microbiology_aggregate`` read only the concept
    ``name`` (OR-matched against spec_type_desc/org_name) and silently dropped
    ``attributes``, where the decomposer carries the second culture dimension.
    So "blood culture that grew E. coli" counted *every* positive blood culture
    (organism ignored — a ~7x over-count in sepsis), and a question whose
    organism never matched ``org_name`` collapsed to zero. The fix conjoins
    each attribute term as an additional ``(spec_type_desc OR org_name)`` ILIKE
    clause, each term still matched against either column because the decomposer
    may place specimen or organism in either slot.

    Synthetic fixture: hadm 101 = BLOOD CULTURE / STAPHYLOCOCCUS AUREUS,
    hadm 103 = URINE / ESCHERICHIA COLI — so a blood culture grew S. aureus but
    *no* blood culture grew E. coli (E. coli is in urine only).
    """

    @staticmethod
    def _micro_cq(*, name, attributes, culture_status="positive"):
        return CompetencyQuestion(
            original_question="test",
            clinical_concepts=[ClinicalConcept(
                name=name,
                concept_type="microbiology",
                attributes=list(attributes),
                culture_status=culture_status,
            )],
            aggregation="count",
            scope="cohort",
            return_type="text",
        )

    @staticmethod
    def _count(backend, cq):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        q = compile_sql(cq, backend, get_default_registry())
        rows = backend.execute(q.sql, q.params)
        return rows[0][0]

    def test_specimen_alone_counts_every_positive_blood_culture(self, backend):
        # No organism qualifier → every positive blood culture (hadm 101).
        cq = self._micro_cq(name="blood culture", attributes=[])
        assert self._count(backend, cq) == 1

    def test_specimen_and_matching_organism_intersect(self, backend):
        # blood AND staph aureus → hadm 101 (the one blood culture that grew it).
        cq = self._micro_cq(
            name="blood culture", attributes=["Staphylococcus aureus"])
        assert self._count(backend, cq) == 1

    def test_specimen_and_nonmatching_organism_is_empty(self, backend):
        # E. coli grew only in URINE here, so a *blood* culture growing E. coli
        # matches nothing. Pre-fix (attributes dropped) this wrongly returned 1
        # (the blood culture, organism ignored) — the core iter10 over-count.
        cq = self._micro_cq(
            name="blood culture", attributes=["Escherichia coli"])
        assert self._count(backend, cq) == 0

    def test_organism_as_name_specimen_as_attribute_is_symmetric(self, backend):
        # The decomposer may place organism in ``name`` and specimen in
        # ``attributes``; each term is matched against either column, so the
        # intersection is identical regardless of slot assignment.
        assert self._count(
            backend, self._micro_cq(name="Escherichia coli", attributes=["urine"])
        ) == 1
        assert self._count(
            backend,
            self._micro_cq(name="Escherichia coli", attributes=["blood culture"]),
        ) == 0

    def test_specimen_culture_suffix_is_stripped_to_source(self, backend):
        # MIMIC records non-blood specimens by anatomic source ('URINE',
        # 'SPUTUM'), never '<source> culture'. So "urine culture" verbatim hits
        # 0 rows; stripping the modality token matches the URINE row (hadm 103).
        assert self._count(
            backend, self._micro_cq(name="urine culture", attributes=[])
        ) == 1
        # Blood is the one specimen MIMIC labels '... CULTURE'; the stem 'blood'
        # still matches it (hadm 101), so stripping never regresses blood.
        assert self._count(
            backend, self._micro_cq(name="blood culture", attributes=[])
        ) == 1
        # Same normalization when the specimen rides in attributes.
        assert self._count(
            backend,
            self._micro_cq(name="Escherichia coli", attributes=["urine culture"]),
        ) == 1


class TestOutcomeMortalityComparison:
    """An outcome (mortality) question with a comparison axis must split by that
    axis and report each group's OWN rate, not the pooled cohort rate.

    iter14 bug: ``_compile_outcome_rate`` ignored ``comparison_field`` and
    always grouped only by ``hospital_expire_flag``, so "compare mortality
    between men and women" silently collapsed to the overall mortality.

    Synthetic fixture (whole cohort, no filter): Male admissions = 101, 102,
    104, 106 (4 distinct; only 106 expired) → male mortality 1/4 = 0.25; Female
    = 103, 105 (2 distinct; none expired) → 0.0. The pooled overall rate is
    1/6 ≈ 0.167 — distinct from both, so the split is observable.
    """

    @staticmethod
    def _outcome_cq(*, comparison_field=None):
        return _cq(
            concepts=[("mortality", "outcome")],
            aggregation="count",
            scope="comparison" if comparison_field else "cohort",
            comparison_field=comparison_field,
        )

    @staticmethod
    def _rows(backend, cq):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        q = compile_sql(cq, backend, get_default_registry())
        rows = [dict(zip(q.columns, r)) for r in backend.execute(q.sql, q.params)]
        return rows, q.columns

    def test_ungrouped_outcome_reports_pooled_rate(self, backend):
        rows, cols = self._rows(backend, self._outcome_cq())
        assert "group_value" not in cols
        died = [r for r in rows if r["expired"] == 1][0]
        assert math.isclose(died["fraction"], 1 / 6, rel_tol=1e-6)

    def test_comparison_outcome_splits_by_axis_with_per_group_rate(self, backend):
        rows, cols = self._rows(
            backend, self._outcome_cq(comparison_field="gender")
        )
        assert "group_value" in cols
        assert {r["group_value"] for r in rows} == {"M", "F"}
        # Male mortality = 1 death / 4 male admissions = 0.25 — its OWN rate,
        # not the pooled 1/6 the bug produced.
        male_died = [
            r for r in rows if r["group_value"] == "M" and r["expired"] == 1
        ][0]
        assert math.isclose(male_died["fraction"], 0.25, rel_tol=1e-6)
        # Female group present, zero deaths → survival fraction 1.0 (its own
        # denominator, the 2 female admissions).
        fem_survived = [
            r for r in rows if r["group_value"] == "F" and r["expired"] == 0
        ][0]
        assert math.isclose(fem_survived["fraction"], 1.0, rel_tol=1e-6)


class TestOutcomeReadmissionRate:
    """Readmission is a first-class outcome, like mortality: "30-day readmission
    rate" must compute the RATE (readmitted / cohort) via SQL, joining the
    readmission-labels CTE — not route to a concept-less graph build that
    returns a count or LOS.

    Synthetic readmission labels: subject 1's two admissions are 101 (disch
    2150-01-20) then 102 (admit 2150-02-10) — inside the 30-day window — so 101
    is ``readmitted_30d = 1``; the other five admissions are not. Overall 30-day
    readmission rate = 1/6 ≈ 0.167.
    """

    @staticmethod
    def _rows(backend, cq):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        q = compile_sql(cq, backend, get_default_registry())
        rows = [dict(zip(q.columns, r)) for r in backend.execute(q.sql, q.params)]
        return rows, q.columns

    def test_30d_readmission_outcome_reports_rate(self, backend):
        cq = _cq(
            concepts=[("30-day readmission", "outcome")], aggregation="count"
        )
        rows, cols = self._rows(backend, cq)
        # The flag column is the readmission label, not the mortality flag.
        assert "readmitted30" in cols and "expired" not in cols
        readmit = [r for r in rows if r["readmitted30"] == 1][0]
        assert math.isclose(readmit["fraction"], 1 / 6, rel_tol=1e-6)

    def test_60d_readmission_window_selected_from_name(self, backend):
        cq = _cq(
            concepts=[("60-day readmission", "outcome")], aggregation="count"
        )
        _, cols = self._rows(backend, cq)
        assert "readmitted60" in cols


class TestSplitByConditionComparison:
    """``comparison_field='condition'`` + a ``split_condition`` groups the cohort
    by presence/absence of a sub-condition, emitting
    ``CASE WHEN EXISTS(<sub-condition>) THEN 'yes' ELSE 'no' END`` as the GROUP BY
    — so demo prompts like "mortality split by whether [a condition]" compile in
    one SQL pass instead of being declined for an unsupported axis."""

    @staticmethod
    def _split_cq(*, concepts, split_condition, aggregation="count"):
        return CompetencyQuestion(
            original_question="test",
            clinical_concepts=[
                ClinicalConcept(name=n, concept_type=t) for n, t in concepts
            ],
            aggregation=aggregation,
            scope="comparison",
            comparison_field="condition",
            split_condition=split_condition,
        )

    @staticmethod
    def _compile(backend, cq):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        return compile_sql(cq, backend, get_default_registry())

    # -- SQL shape -----------------------------------------------------------

    def test_diagnosis_split_emits_case_when_exists_with_prefix_param(self, backend):
        cq = self._split_cq(
            concepts=[("in-hospital mortality", "outcome")],
            split_condition=PatientFilter(
                field="diagnosis", operator="contains",
                value="chronic anticoagulant use",
                resolved_icd_codes=["Z79.01", "Z79.02"],
            ),
        )
        q = self._compile(backend, cq)
        assert "CASE WHEN EXISTS" in q.sql
        assert "di2.icd_code LIKE ?" in q.sql
        # Codes are normalized (dots stripped, uppercased) and prefix-matched.
        assert "Z7901%" in q.params and "Z7902%" in q.params
        # The grouped outcome shape carries the group_value + flag columns.
        assert q.columns[0] == "group_value"

    def test_ventilation_split_emits_procedureevents_itemids(self, backend):
        cq = self._split_cq(
            concepts=[("in-hospital mortality", "outcome")],
            split_condition=PatientFilter(
                field="ventilation", operator="=", value="1",
            ),
        )
        q = self._compile(backend, cq)
        assert "CASE WHEN EXISTS" in q.sql
        assert "procedureevents" in q.sql
        # Invasive Ventilation (225792), Intubation (224385), Non-invasive
        # Ventilation (225794) — verified-live procedureevents itemids.
        assert "225792" in q.sql and "224385" in q.sql and "225794" in q.sql

    def test_ventilation_detected_by_value_keyword(self, backend):
        # The decomposer routinely mislabels ventilation field="diagnosis"
        # value="mechanical ventilation"; the keyword path still routes it to the
        # procedureevents EXISTS, not an (empty) diagnosis title-LIKE.
        cq = self._split_cq(
            concepts=[("in-hospital mortality", "outcome")],
            split_condition=PatientFilter(
                field="diagnosis", operator="contains",
                value="mechanical ventilation",
            ),
        )
        q = self._compile(backend, cq)
        assert "procedureevents" in q.sql and "225792" in q.sql
        assert "%mechanical ventilation%" not in q.params  # not a diagnosis LIKE

    def test_or_any_split_ors_each_grounded_leaf(self, backend):
        # "anticoagulants OR antiplatelets" → or_any of two diagnosis leaves;
        # each grounds to its codes and the EXISTS clauses are OR-ed together.
        cq = self._split_cq(
            concepts=[("in-hospital mortality", "outcome")],
            split_condition=PatientFilter(
                field="or_any", operator="in", value="any",
                sub_filters=[
                    PatientFilter(
                        field="diagnosis", operator="contains",
                        value="chronic anticoagulant use",
                        resolved_icd_codes=["Z79.01"],
                    ),
                    PatientFilter(
                        field="diagnosis", operator="contains",
                        value="chronic antiplatelet use",
                        resolved_icd_codes=["Z79.02"],
                    ),
                ],
            ),
        )
        q = self._compile(backend, cq)
        assert q.sql.count("EXISTS") >= 2  # one per leaf, OR-ed
        assert "Z7901%" in q.params and "Z7902%" in q.params

    def test_diagnosis_split_without_codes_falls_back_to_title_like(self, backend):
        cq = self._split_cq(
            concepts=[("in-hospital mortality", "outcome")],
            split_condition=PatientFilter(
                field="diagnosis", operator="contains",
                value="atrial fibrillation",
            ),
        )
        q = self._compile(backend, cq)
        assert "CASE WHEN EXISTS" in q.sql
        # Title-LIKE fallback joins the dictionary on the typed phrase.
        assert "d_icd_diagnoses" in q.sql
        assert "%atrial fibrillation%" in q.params

    # -- Executable parity on the synthetic fixture --------------------------

    def test_split_by_diagnosis_groups_cohort_yes_no(self, backend):
        # Split the whole cohort's in-hospital mortality by whether the
        # admission carries an ischemic-stroke (I63x) diagnosis. Fixture: 101,
        # 102, 103, 106 are I63x (yes → 4 admissions, only 106 expired); 104,
        # 105 are not (no → 2 admissions, none expired). So the 'yes' group's
        # mortality is its OWN rate 1/4, distinct from the pooled 1/6.
        cq = self._split_cq(
            concepts=[("in-hospital mortality", "outcome")],
            split_condition=PatientFilter(
                field="diagnosis", operator="contains",
                value="ischemic stroke", resolved_icd_codes=["I63"],
            ),
        )
        q = self._compile(backend, cq)
        rows = [dict(zip(q.columns, r)) for r in backend.execute(q.sql, q.params)]
        assert {r["group_value"] for r in rows} == {"yes", "no"}
        yes_died = [
            r for r in rows if r["group_value"] == "yes" and r["expired"] == 1
        ][0]
        assert math.isclose(yes_died["fraction"], 0.25, rel_tol=1e-6)
        # 'no' group: 2 admissions, none expired → survival fraction 1.0.
        no_survived = [
            r for r in rows if r["group_value"] == "no" and r["expired"] == 0
        ][0]
        assert math.isclose(no_survived["fraction"], 1.0, rel_tol=1e-6)

    def test_split_by_diagnosis_on_count_aggregate(self, backend):
        # The split axis also works on a plain diagnosis COUNT. Count epilepsy
        # admissions (G40x: only hadm 104), split by whether they ALSO carry a
        # stroke (I63x) diagnosis. hadm 104 has no I63x, so it lands in the 'no'
        # group → count 1; the 'yes' group is empty (absent from the result).
        cq = self._split_cq(
            concepts=[("epilepsy", "diagnosis")],
            split_condition=PatientFilter(
                field="diagnosis", operator="contains",
                value="ischemic stroke", resolved_icd_codes=["I63"],
            ),
        )
        # Ground the main concept's codes via the orchestrator-supplied kwarg.
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        q = compile_sql(
            cq, backend, get_default_registry(), resolved_icd_codes=["G40"],
        )
        rows = [dict(zip(q.columns, r)) for r in backend.execute(q.sql, q.params)]
        by_group = {r["group_value"]: r["count"] for r in rows}
        assert by_group.get("no") == 1
        assert "yes" not in by_group


class TestEventOrderingPostProcessor:
    """``_event_ordering_post_processor`` turns per-patient FIRST-event-time rows
    into the most-common temporal order + per-first-event fractions + median
    inter-event gap."""

    @staticmethod
    def _post(rows):
        from src.conversational.operations import get_default_registry

        op = get_default_registry().require("aggregate", "event_ordering")
        return op.post_processor(rows)

    def test_most_common_sequence_and_gcs_first_fraction(self):
        from datetime import datetime

        # patient1: intubation @ t0, GCS drop @ t1 → order (intubation, GCS drop)
        # patient2: GCS drop @ t0, intubation @ t1 → order (GCS drop, intubation)
        # patient3: GCS drop @ t0, intubation @ t1 → same as patient2
        # So the most common sequence is "GCS drop → intubation" (2 of 3), and
        # GCS drop is FIRST for 2 of 3 patients.
        rows = [
            {"hadm_id": 1, "event_name": "intubation", "event_time": datetime(2020, 1, 1, 0, 0)},
            {"hadm_id": 1, "event_name": "GCS drop", "event_time": datetime(2020, 1, 1, 1, 0)},
            {"hadm_id": 2, "event_name": "GCS drop", "event_time": datetime(2020, 1, 2, 0, 0)},
            {"hadm_id": 2, "event_name": "intubation", "event_time": datetime(2020, 1, 2, 3, 0)},
            {"hadm_id": 3, "event_name": "GCS drop", "event_time": datetime(2020, 1, 3, 0, 0)},
            {"hadm_id": 3, "event_name": "intubation", "event_time": datetime(2020, 1, 3, 5, 0)},
        ]
        out, cols = self._post(rows)
        row = out[0]
        assert row["most_common_sequence"] == "GCS drop → intubation"
        assert row["n_patients"] == 2
        assert math.isclose(row["pct"], 2 / 3, rel_tol=1e-6)
        # GCS drop is first for patients 2 and 3 → 2/3.
        assert math.isclose(row["gcs_drop_first_fraction"], 2 / 3, rel_tol=1e-6)
        assert "gcs_drop_first_fraction" in cols
        # Median first→last gap: patient1 1h, patient2 3h, patient3 5h → median 3h.
        assert math.isclose(row["median_first_to_last_hours"], 3.0, rel_tol=1e-6)

    def test_null_event_times_are_dropped(self):
        from datetime import datetime

        # patient1 only ever had intubation (GCS drop time is NULL) → a 1-event
        # sequence; patient2 had both. The NULL row must not count.
        rows = [
            {"hadm_id": 1, "event_name": "intubation", "event_time": datetime(2020, 1, 1, 0, 0)},
            {"hadm_id": 1, "event_name": "GCS drop", "event_time": None},
            {"hadm_id": 2, "event_name": "intubation", "event_time": datetime(2020, 1, 2, 0, 0)},
            {"hadm_id": 2, "event_name": "GCS drop", "event_time": datetime(2020, 1, 2, 2, 0)},
        ]
        out, _ = self._post(rows)
        # patient1's gap is undefined (only 1 event); patient2's is 2h → median 2h.
        assert math.isclose(out[0]["median_first_to_last_hours"], 2.0, rel_tol=1e-6)

    def test_empty_rows_returns_null_summary(self):
        out, cols = self._post([])
        assert out[0]["most_common_sequence"] is None
        assert out[0]["n_patients"] == 0
        assert "median_first_to_last_hours" in cols


class TestEventOrderingCompile:
    """``_compile_event_ordering`` emits a UNION ALL of per-event MIN(time)
    sub-queries over the cohort, one per event concept."""

    @staticmethod
    def _ordering_cq(*, concepts, filters=None):
        return CompetencyQuestion(
            original_question="test",
            clinical_concepts=[
                ClinicalConcept(name=n, concept_type=t) for n, t in concepts
            ],
            patient_filters=[
                PatientFilter(field=f, operator=o, value=v)
                for f, o, v in (filters or [])
            ],
            aggregation="event_ordering",
            scope="cohort",
        )

    @staticmethod
    def _compile(backend, cq):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        return compile_sql(cq, backend, get_default_registry())

    def test_union_all_with_min_time_per_event(self, backend):
        cq = self._ordering_cq(
            concepts=[
                ("intubation", "procedure"),
                ("hyperosmolar therapy", "drug"),
            ],
        )
        q = self._compile(backend, cq)
        assert q.columns == ["hadm_id", "event_name", "event_time"]
        assert "UNION ALL" in q.sql
        # Intubation → procedureevents itemids (Invasive Ventilation 225792 +
        # Intubation 224385); hyperosmolar → mannitol/hypertonic.
        assert "procedureevents" in q.sql
        assert "225792" in q.sql and "224385" in q.sql
        assert "%mannitol%" in q.sql and "%hypertonic%" in q.sql
        assert q.sql.count("MIN(") == 2
        assert "AS event_time" in q.sql

    def test_cohort_filter_applied_in_each_subquery(self, backend):
        # The ICH diagnosis cohort filter must appear in BOTH sub-queries so all
        # event times are within-cohort. It is applied as a correlated EXISTS
        # (not a JOIN) so two diagnosis filters can't collide on alias ``di``.
        cq = self._ordering_cq(
            concepts=[
                ("intubation", "procedure"),
                ("hyperosmolar therapy", "drug"),
            ],
            filters=[("diagnosis", "contains", "intracerebral hemorrhage")],
        )
        cq.patient_filters[0].resolved_icd_codes = ["I61"]
        q = self._compile(backend, cq)
        # Each sub-query's WHERE carries a correlated EXISTS for the cohort
        # filter — one per sub-query — not a cohort table JOIN.
        assert q.sql.count("EXISTS (") == 2
        assert "di2.icd_code LIKE ?" in q.sql
        # With resolved codes the EXISTS uses a prefix LIKE; applied in each of
        # the 2 sub-queries, the LIKE param repeats.
        assert q.params.count("I61%") == 2

    def test_two_diagnosis_filters_no_alias_collision(self, backend):
        # Regression for "Duplicate table alias di": a cohort with TWO diagnosis
        # filters (ICH + ICP-monitoring proxy) must compile without colliding on
        # the hardcoded ``di`` alias. EXISTS sub-conditions self-contain their
        # aliases, so the UNION compiles + executes.
        cq = self._ordering_cq(
            concepts=[
                ("intubation", "procedure"),
                ("hyperosmolar therapy", "drug"),
            ],
            filters=[
                ("diagnosis", "contains", "intracerebral hemorrhage"),
                ("diagnosis", "contains", "intracranial pressure monitoring"),
            ],
        )
        cq.patient_filters[0].resolved_icd_codes = ["I61"]
        cq.patient_filters[1].resolved_icd_codes = ["I62"]
        q = self._compile(backend, cq)
        # Two cohort filters × two sub-queries = four EXISTS predicates.
        assert q.sql.count("EXISTS (") == 4
        # No cohort table JOIN means no hardcoded ``di``/``dd`` alias — so the
        # two diagnosis filters can't collide. Each sub-query only JOINs its own
        # event table (procedureevents); the cohort filters are all EXISTS.
        assert " di " not in f" {q.sql} " and " di." not in q.sql
        assert " dd " not in f" {q.sql} " and " dd." not in q.sql

    def test_gcs_event_skipped_offline_but_others_remain(self, backend):
        # The DuckDB/offline backend has no derived GCS table, so the GCS event
        # is dropped — but intubation + hyperosmolar still form a valid UNION.
        cq = self._ordering_cq(
            concepts=[
                ("intubation", "procedure"),
                ("hyperosmolar therapy", "drug"),
                ("GCS drop", "vital"),
            ],
        )
        q = self._compile(backend, cq)
        assert "UNION ALL" in q.sql
        assert q.sql.count("MIN(") == 2  # GCS sub-query omitted offline
        assert "mimiciv_3_1_derived" not in q.sql

    def test_executable_union_orders_events_on_fixture(self, backend):
        # Add a procedureevents table + mannitol prescriptions so a 2-event
        # ordering executes end-to-end. Two patients (101, 106) get intubation
        # BEFORE mannitol; one patient (103) gets mannitol only. So the
        # most-common sequence is "intubation → hyperosmolar therapy" (2 of 3).
        conn = backend._conn
        conn.execute(
            "CREATE TABLE procedureevents ("
            "subject_id INTEGER, hadm_id INTEGER, stay_id INTEGER, "
            "itemid INTEGER, starttime TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO procedureevents VALUES "
            "(1, 101, 1001, 224385, '2150-01-15 09:00:00'), "
            "(5, 106, 1003, 225792, '2151-04-11 06:00:00')"
        )
        conn.execute(
            "INSERT INTO prescriptions VALUES "
            "(1, 101, '2150-01-15 12:00:00', '2150-01-16 12:00:00', "
            "'Mannitol 20%', 50.0, 'g', 'IV'), "
            "(5, 106, '2151-04-11 10:00:00', '2151-04-12 10:00:00', "
            "'Mannitol 20%', 50.0, 'g', 'IV'), "
            "(2, 103, '2151-03-02 08:00:00', '2151-03-03 08:00:00', "
            "'Mannitol 20%', 50.0, 'g', 'IV')"
        )
        cq = self._ordering_cq(
            concepts=[
                ("intubation", "procedure"),
                ("hyperosmolar therapy", "drug"),
            ],
        )
        q = self._compile(backend, cq)
        raw = backend.execute(q.sql, q.params)
        rows = [dict(zip(q.columns, r)) for r in raw]
        from src.conversational.operations import get_default_registry

        op = get_default_registry().require("aggregate", "event_ordering")
        out, _ = op.post_processor(rows)
        # 101 and 106 both go intubation → mannitol; 103 has mannitol only. So
        # the 2-event order wins (2 of 3 patients).
        assert out[0]["most_common_sequence"] == "intubation → hyperosmolar therapy"
        assert out[0]["n_patients"] == 2


class TestDiagnosisGroundedCodePrefixMatch:
    """OMOPHub's ``icd_autocode`` returns DOTTED, often category-level codes
    ('I63', 'I63.9'); MIMIC stores them UNDOTTED and BILLABLE ('I6300', 'I639').
    Grounded codes must be normalized (dots stripped) and PREFIX-matched, or an
    exact ``IN`` on the dotted/category code matches nothing — the iter18
    grounding bug that silently returned 0 for ischemic stroke / DKA / etc.

    Synthetic diagnoses: hadm 101=I639, 102=I634, 103=I630, 106=I639 — all
    ischemic stroke (I63x); 104=G409, 105=I251 are not. So I63x covers 4 distinct
    admissions, and the fixture titles read "Cerebral infarction" (no "ischemic
    stroke"), so the title-LIKE fallback can't mask the prefix match.
    """

    @staticmethod
    def _count(backend, codes):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(concepts=[("ischemic stroke", "diagnosis")], aggregation="count")
        q = compile_sql(
            cq, backend, get_default_registry(), resolved_icd_codes=codes
        )
        return backend.execute(q.sql, q.params)[0][0]

    def test_category_code_prefix_matches_billable_descendants(self, backend):
        # 'I63' is exactly what icd_autocode returns for "ischemic stroke"; it
        # must catch the billable I639/I634/I630 rows (4 admissions). An exact
        # IN ('I63') matched zero — the bug.
        assert self._count(backend, ["I63"]) == 4

    def test_dotted_code_is_normalized(self, backend):
        # 'I63.9' → 'I639' → the two I639 admissions (101, 106).
        assert self._count(backend, ["I63.9"]) == 2

    def test_unmatched_code_yields_zero_not_error(self, backend):
        # A grounded code absent from MIMIC compiles and returns 0 cleanly.
        assert self._count(backend, ["I67.82"]) == 0


class TestPrimaryDiagnosisQualifier:
    """A "primary diagnosis of X" count must restrict to MIMIC's principal
    diagnosis (``diagnoses_icd.seq_num = 1``); higher seq_nums are secondary
    diagnoses / comorbidities. The decomposer sets ``primary_only=True`` and the
    compiler adds the ``seq_num = 1`` predicate — general for any condition
    (e.g. acute MI: ~8.6k primary vs ~16.5k any-position)."""

    @staticmethod
    def _sql(backend, *, primary_only):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = CompetencyQuestion(
            original_question="test",
            clinical_concepts=[ClinicalConcept(
                name="acute myocardial infarction",
                concept_type="diagnosis",
                primary_only=primary_only,
            )],
            aggregation="count",
            scope="cohort",
            return_type="text",
        )
        return compile_sql(
            cq, backend, get_default_registry(), resolved_icd_codes=["I21"]
        ).sql

    def test_primary_only_emits_seq_num_filter(self, backend):
        assert "di.seq_num = 1" in self._sql(backend, primary_only=True)

    def test_default_any_position_has_no_seq_num_filter(self, backend):
        assert "di.seq_num = 1" not in self._sql(backend, primary_only=False)


class TestBiomarkerAggregateCorrectness:
    """Against synthetic_duckdb_with_events the biomarker AVG/MAX/MIN/COUNT
    for creatinine should match what a straight DuckDB query against the
    fixture would return."""

    def _direct_creatinine(self, backend, fn: str) -> float | int | None:
        sql = (
            f"SELECT {fn}(l.valuenum) FROM labevents l "
            f"JOIN d_labitems d ON l.itemid = d.itemid "
            f"WHERE d.label ILIKE ? AND l.valuenum IS NOT NULL"
        )
        return backend.execute(sql, ["%creatinine%"])[0][0]

    @pytest.mark.parametrize("agg,fn,col", [
        ("mean", "AVG", "mean_value"),
        ("max", "MAX", "max_value"),
        ("min", "MIN", "min_value"),
        ("count", "COUNT", "count_value"),
    ])
    def test_biomarker_aggregate_matches_direct_query(
        self, backend, agg, fn, col,
    ):
        cq = _cq(concepts=[("creatinine", "biomarker")], aggregation=agg)
        rows = _run_fastpath(backend, cq)
        assert len(rows) == 1
        expected = self._direct_creatinine(backend, fn)
        actual = rows[0][col]
        if isinstance(expected, float):
            assert math.isclose(actual, expected, rel_tol=1e-6)
        else:
            assert actual == expected

    # -- LOINC-grounded biomarker resolution (lab-resolver fix) ----------
    # The three tests below pin the contract for the LOINC-grounding fix.
    # See docs/... and the lab-resolver project memory for the full bug
    # narrative; in short: ``LIKE '%creatinine%'`` pools serum / urine /
    # 24-hr / ratio variants into one AVG, producing clinically wrong
    # numbers (~4.95 mg/dL instead of ~1.34). The fix threads a
    # decomposer-supplied LOINC code through the resolver into a
    # ``WHERE l.itemid IN (...)`` clause, restricting to serum.
    #
    # Test 1.1 documents the BUGGY behaviour and is deleted once the fix
    # lands. 1.2 is the contract for the fixed behaviour. 1.3 is the
    # regression guard for the LIKE fallback when no LOINC is supplied.

    def test_creatinine_with_loinc_restricts_to_serum(
        self, backend_with_creatinine_variants,
    ):
        """Contract for the fix: when the decomposer emits a LOINC code,
        the compiler restricts to that LOINC's MIMIC itemids only — urine
        and 24-hr variants are excluded.

        Expected after fix: AVG = (1.2 + 0.9 + 1.5) / 3 = 1.2 mg/dL.
        Expected before fix: AVG ≈ 260 mg/dL (LIKE still pools).
        Currently fails to even import because ClinicalConcept lacks
        ``loinc_code`` and ``_cq`` doesn't unpack a 3-tuple — those are
        Phase 2's first production change.
        """
        cq = _cq(
            concepts=[("creatinine", "biomarker", "2160-0")],
            aggregation="mean",
        )
        rows = _run_fastpath(backend_with_creatinine_variants, cq)
        serum_only_mean = (1.2 + 0.9 + 1.5) / 3
        assert math.isclose(rows[0]["mean_value"], serum_only_mean, rel_tol=1e-2)

    def test_creatinine_without_loinc_falls_back_to_like(self, backend):
        """Backward-compat regression guard: when no LOINC is supplied,
        the compiler must still emit the LIKE-based query, matching the
        existing behavior. The base fixture has only itemid 50912 (serum)
        so LIKE happens to give the right answer — this test will keep
        passing both before and after the fix.
        """
        cq = _cq(concepts=[("creatinine", "biomarker")], aggregation="mean")
        rows = _run_fastpath(backend, cq)
        expected = (1.2 + 0.9 + 1.5) / 3
        assert math.isclose(rows[0]["mean_value"], expected, rel_tol=1e-6)

    # -- Compiler-direct tests (Phase 3) --------------------------------
    # These test compile_sql at the unit-test layer: when given an
    # itemid list, it emits ``WHERE l.itemid IN (?, ?, ...)``; when given
    # only names, it preserves the existing LIKE behavior. The orchestrator
    # is responsible for calling resolve_biomarker and threading through.

    def test_compile_emits_itemid_in_when_itemids_provided(self, backend):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("creatinine", "biomarker", "2160-0")],
            aggregation="mean",
        )
        query = compile_sql(
            cq, backend, get_default_registry(),
            resolved_names=["creatinine"],
            resolved_itemids=[50912, 51081],
        )
        assert "itemid IN (" in query.sql
        # No label-substring filter when itemids are used (backend-agnostic:
        # DuckDB emits ``ILIKE``, BigQuery emits ``LOWER(...) LIKE LOWER(...)``).
        assert "ILIKE" not in query.sql
        assert "LIKE" not in query.sql
        assert 50912 in query.params
        assert 51081 in query.params

    def test_compile_emits_like_when_no_itemids(self, backend):
        """Backward compat at the compiler layer: no resolved_itemids → LIKE.
        Backend-agnostic: DuckDB emits ``ILIKE``, BigQuery ``LOWER LIKE LOWER``.
        Either form means the label-substring path fired."""
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(concepts=[("creatinine", "biomarker")], aggregation="mean")
        query = compile_sql(
            cq, backend, get_default_registry(),
            resolved_names=["creatinine"],
        )
        sql_upper = query.sql.upper()
        assert "ILIKE" in sql_upper or "LIKE" in sql_upper
        assert "%creatinine%" in query.params
        assert "itemid IN (" not in query.sql


class TestComparisonCorrectness:
    """GROUP BY on a registered axis should produce the same per-group
    statistics as a direct DuckDB GROUP BY."""

    def test_creatinine_by_gender_matches_direct_query(self, backend):
        direct = backend.execute(
            """
            SELECT p.gender, AVG(l.valuenum), COUNT(l.valuenum)
            FROM labevents l
            JOIN d_labitems d ON l.itemid = d.itemid
            JOIN admissions a ON l.hadm_id = a.hadm_id
            JOIN patients p ON a.subject_id = p.subject_id
            WHERE d.label ILIKE ? AND l.valuenum IS NOT NULL
            GROUP BY p.gender
            """,
            ["%creatinine%"],
        )
        expected = [
            {"group_value": g, "avg_value": v, "count": c}
            for g, v, c in direct
        ]

        cq = _cq(
            concepts=[("creatinine", "biomarker")],
            aggregation="mean",
            scope="comparison",
            comparison_field="gender",
        )
        actual = _run_fastpath(backend, cq)
        assert _rows_equal(actual, expected)


class TestFilterCompilationReused:
    """Cohort filters emitted by OperationRegistry.compile_filters must flow
    into the fast-path's WHERE clause exactly as they do in the graph-path
    cohort query. Otherwise fast-path results leak out of the cohort."""

    def test_age_filter_restricts_fastpath(self, backend):
        # Fixture has patients with ages 45, 58, 65, 72, 80. Age > 70 keeps
        # only patients 2 (72) and 5 (80) — of whom patient 5 has a
        # creatinine reading (1.5 mg/dL) and patient 2 also does (0.9).
        cq = _cq(
            concepts=[("creatinine", "biomarker")],
            filters=[("age", ">", "70")],
            aggregation="mean",
        )
        rows = _run_fastpath(backend, cq)
        assert len(rows) == 1
        # Verify against direct query with the same filter.
        expected = backend.execute(
            """
            SELECT AVG(l.valuenum) FROM labevents l
            JOIN d_labitems d ON l.itemid = d.itemid
            JOIN admissions a ON l.hadm_id = a.hadm_id
            JOIN patients p ON a.subject_id = p.subject_id
            WHERE d.label ILIKE ?
              AND l.valuenum IS NOT NULL
              AND p.anchor_age > ?
            """,
            ["%creatinine%", 70],
        )[0][0]
        assert math.isclose(rows[0]["mean_value"], expected, rel_tol=1e-6)

    def test_diagnosis_filter_restricts_fastpath(self, backend):
        cq = _cq(
            concepts=[("creatinine", "biomarker")],
            filters=[("diagnosis", "contains", "cerebral")],
            aggregation="count",
        )
        rows = _run_fastpath(backend, cq)
        assert len(rows) == 1
        assert isinstance(rows[0]["count_value"], int)


class _FakeBQBackend:
    """BigQuery-shaped backend whose ``table()`` returns a backtick-quoted FQN,
    so ``_derived_table`` resolves ``mimiciv_3_1_derived.gcs``. Not executable —
    used to assert the GCS-via-fast-path SQL shape only."""

    def table(self, name: str) -> str:
        return f"`physionet-data.x.{name}`"

    @staticmethod
    def ilike(column: str) -> str:
        return f"{column} ILIKE ?"

    def readmission_labels_expr(self) -> str:
        return "`physionet-data.x.readmission_labels`"


class TestGcsThresholdFilterFastpath:
    """A GCS ``vital_value`` threshold compiles, through the fast-path, to an
    EXISTS over the derived ``gcs`` TOTAL column — never the component
    label-LIKE. Guards the gate: a non-GCS vital is byte-identical to before."""

    def test_gcs_filter_emits_derived_total_exists_on_bigquery(self):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("in-hospital mortality", "outcome")],
            aggregation="count",
        )
        cq.patient_filters = [
            PatientFilter(
                field="vital_value", operator="<=", value="8", measurement="GCS",
            )
        ]
        q = compile_sql(cq, _FakeBQBackend(), get_default_registry())
        # The cohort restriction is an EXISTS over the derived gcs TOTAL column.
        assert "mimiciv_3_1_derived.gcs" in q.sql
        assert "g.gcs <= ?" in q.sql
        assert "icu.stay_id = g.stay_id" in q.sql
        # Never the meaningless component label-LIKE.
        assert "d_items" not in q.sql
        assert 8.0 in q.params

    def test_gcs_filter_offline_drops_cohort_no_derived_table(self, backend):
        # Offline DuckDB: the derived gcs table is unavailable, so the GCS
        # filter degrades to an empty cohort (1 = 0) and the query still
        # compiles + executes without referencing a missing table.
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("creatinine", "biomarker")],
            aggregation="count",
        )
        cq.patient_filters = [
            PatientFilter(
                field="vital_value", operator="<=", value="8", measurement="GCS",
            )
        ]
        q = compile_sql(cq, backend, get_default_registry())
        assert "mimiciv_3_1_derived" not in q.sql
        assert "1 = 0" in q.sql
        # Executes (count is an int, here 0 — no admission can satisfy 1 = 0).
        rows = [dict(zip(q.columns, r)) for r in backend.execute(q.sql, q.params)]
        assert rows[0]["count_value"] == 0

    def test_nongcs_vital_value_filter_unchanged(self, backend):
        # MAP threshold still compiles to the chartevents label-LIKE EXISTS —
        # the GCS gate must not perturb other vitals.
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(concepts=[("creatinine", "biomarker")], aggregation="count")
        cq.patient_filters = [
            PatientFilter(
                field="vital_value", operator="<", value="65",
                measurement="mean arterial pressure",
            )
        ]
        q = compile_sql(cq, backend, get_default_registry())
        assert "chartevents" in q.sql and "d_items" in q.sql
        assert "mimiciv_3_1_derived" not in q.sql
        assert "%mean arterial pressure%" in q.params


class TestCompileSqlMcpFlagPlumbing:
    """Inc 9.3 — compile_sql accepts ``enable_mcp_grounding`` kwarg and
    threads it through ``_filter_fragment`` to ``FilterCompileContext``.
    The diagnosis-filter compiler then consults the flag to decide
    whether to ground via icd_autocode."""

    def test_default_mcp_flag_is_false(self, backend, monkeypatch):
        """When compile_sql is called without enable_mcp_grounding,
        FilterCompileContext is constructed with the default False —
        no MCP call attempted even for diagnosis-typed filters."""
        from src.conversational import concept_resolver as cr
        from src.conversational.models import PatientFilter
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        call_count = [0]
        def fake_autocode(text, **kwargs):
            call_count[0] += 1
            return {"status": "ok", "results": []}
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        cq = _cq(
            concepts=[("creatinine", "biomarker")],
            filters=[("diagnosis", "contains", "sepsis")],
            aggregation="mean",
        )
        compile_sql(
            cq, backend, get_default_registry(),
            resolved_names=["creatinine"],
        )
        assert call_count[0] == 0

    def test_mcp_flag_true_triggers_filter_grounding(
        self, backend, monkeypatch,
    ):
        """compile_sql(enable_mcp_grounding=True) → diagnosis filter
        compiles with a grounded IN-list (autocode fallback path).

        Uses 'carcinoid syndrome' so Inc 10's registry-first lookup
        misses and the icd_autocode mock actually fires; for registered
        cohort names the registry path emits prefix LIKE clauses
        instead of an IN-list."""
        from src.conversational import concept_resolver as cr
        from src.conversational.models import PatientFilter
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cr._cached_icd_autocode.cache_clear()

        def fake_autocode(text, **kwargs):
            return {
                "status": "ok",
                "results": [
                    {"code": "E34.0", "title": "Carcinoid syndrome", "confidence": 0.92},
                ],
            }
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        cq = _cq(
            concepts=[("creatinine", "biomarker")],
            filters=[("diagnosis", "contains", "carcinoid syndrome")],
            aggregation="mean",
        )
        query = compile_sql(
            cq, backend, get_default_registry(),
            resolved_names=["creatinine"],
            enable_mcp_grounding=True,
        )
        # Grounded codes are normalized (dots stripped) and prefix-matched.
        assert "di.icd_code LIKE ?" in query.sql
        assert "E340%" in query.params

    def test_mcp_flag_does_not_affect_biomarker_path_without_diagnosis_filter(
        self, backend, monkeypatch,
    ):
        """A biomarker query with no diagnosis filter doesn't trigger
        any MCP call even when the flag is True. Belt-and-suspenders
        guard against accidental fan-out."""
        from src.conversational import concept_resolver as cr
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        call_count = [0]
        def fake_autocode(text, **kwargs):
            call_count[0] += 1
            return {"status": "ok", "results": []}
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        cq = _cq(concepts=[("creatinine", "biomarker")], aggregation="mean")
        compile_sql(
            cq, backend, get_default_registry(),
            resolved_names=["creatinine"],
            enable_mcp_grounding=True,
        )
        assert call_count[0] == 0


class TestDiagnosisCountIcdGrounded:
    """Front-half OMOPHub grounding (Inc 4): when ``resolved_icd_codes``
    is supplied, the diagnosis-count compile branch emits normalized
    PREFIX-LIKE clauses as a parallel OR with the existing title-LIKE
    clause. Defaults to LIKE-only when codes are not supplied (back-compat).

    Grounded codes are dotted/category-level ('A41.9', 'I63'); MIMIC stores
    them undotted+billable, so they are dot-stripped and prefix-matched (an
    exact IN matched nothing). The parallel-OR with title-LIKE still catches
    ICD-9 admissions outside OMOPHub's ICD10CM-only coverage.
    """

    def test_emits_icd_prefix_when_codes_supplied(self, backend):
        """Resolved ICD codes → normalized ``di.icd_code LIKE ?`` prefix clauses."""
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("sepsis", "diagnosis")],
            aggregation="count",
            scope="cohort",
        )
        query = compile_sql(
            cq, backend, get_default_registry(),
            resolved_names=["sepsis"],
            resolved_icd_codes=["A41.9", "R65.21"],
        )
        # Prefix-LIKE clauses with the normalized (undotted) codes as params.
        assert "di.icd_code LIKE ?" in query.sql
        assert "A419%" in query.params
        assert "R6521%" in query.params

    def test_emits_codes_only_when_grounded(self, backend):
        """When ICD codes are supplied the cohort clause is CODES-ONLY — the
        broad title-LIKE is dropped so a precise grounding can't be re-polluted
        by title matches (the diagnosis analogue of the biomarker→LOINC fix)."""
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("sepsis", "diagnosis")],
            aggregation="count",
            scope="cohort",
        )
        query = compile_sql(
            cq, backend, get_default_registry(),
            resolved_names=["sepsis"],
            resolved_icd_codes=["A41.9"],
        )
        # ICD-prefix clause present...
        assert "di.icd_code LIKE ?" in query.sql
        assert "A419%" in query.params
        # ...and the broad title-LIKE substring is NOT bound (codes-only).
        assert "%sepsis%" not in query.params

    def test_falls_back_to_like_only_when_codes_none(self, backend):
        """Default behavior preserved: no resolved_icd_codes → LIKE-only.
        Critical for back-compat — existing tests + production paths
        without grounding must produce byte-identical SQL."""
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("sepsis", "diagnosis")],
            aggregation="count",
            scope="cohort",
        )
        query = compile_sql(
            cq, backend, get_default_registry(),
            resolved_names=["sepsis"],
            # resolved_icd_codes intentionally omitted (default None).
        )
        # No IN-list clause. LIKE clause present.
        assert "di.icd_code IN (" not in query.sql
        assert "%sepsis%" in query.params

    def test_falls_back_to_like_only_when_codes_empty_disallowed(self, backend):
        """Empty list shouldn't reach compile_sql — the validator on
        ClinicalConcept.icd_codes rejects empty lists. But compile_sql
        treats `[]` defensively as 'no grounding' rather than emitting
        ``IN ()`` (which most dialects reject). Guards against future
        callers passing `[]` directly."""
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("sepsis", "diagnosis")],
            aggregation="count",
            scope="cohort",
        )
        query = compile_sql(
            cq, backend, get_default_registry(),
            resolved_names=["sepsis"],
            resolved_icd_codes=[],
        )
        assert "di.icd_code IN (" not in query.sql

    def test_diagnosis_list_unchanged_by_resolved_icd_codes(self, backend):
        """Out-of-scope path: diagnosis-list (no aggregation) still uses
        LIKE-only, doesn't pick up resolved_icd_codes. Same parallel-OR
        pattern can be added later as a follow-up if/when needed."""
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("cerebral", "diagnosis")],
            return_type="table",
            scope="cohort",
        )
        query = compile_sql(
            cq, backend, get_default_registry(),
            resolved_names=["cerebral"],
            resolved_icd_codes=["I63.9"],  # ignored
        )
        # IN-list NOT in the diagnosis-list path.
        assert "di.icd_code IN (" not in query.sql

    def test_executes_against_real_duckdb_with_grounded_codes(self, backend):
        """Sanity: the IN-list SQL is actually executable against the test
        DuckDB fixture. Uses 'cerebral' from the fixture (3 hadms) and a
        code that overlaps to make sure SQL is valid even when the IN-list
        finds no rows itself (LIKE branch carries the count)."""
        from src.conversational.sql_fastpath import compile_sql
        from src.conversational.operations import get_default_registry

        cq = _cq(
            concepts=[("cerebral", "diagnosis")],
            aggregation="count",
            scope="cohort",
        )
        query = compile_sql(
            cq, backend, get_default_registry(),
            resolved_names=["cerebral"],
            resolved_icd_codes=["I63.9", "I63.0"],
        )
        rows = backend.execute(query.sql, query.params)
        # Should execute without error and return one count row.
        assert len(rows) == 1


class TestDiagnosisList:
    """patient_list_by_diagnosis shape: a plain SELECT over diagnoses_icd."""

    def test_diagnosis_list_returns_patient_rows(self, backend):
        cq = _cq(
            concepts=[("cerebral", "diagnosis")],
            return_type="table",
            scope="cohort",
        )
        rows = _run_fastpath(backend, cq)
        # Fixture has 3 cerebral admissions: 101, 102, 103 (and 106 shares
        # code I639). The fast-path returns one row per (hadm, diagnosis).
        assert rows
        # Column shape matches the SPARQL template.
        for r in rows:
            assert set(r.keys()) >= {"subjectId", "hadmId", "icdCode"}


# ---------------------------------------------------------------------------
# 4. Negative / guard tests
# ---------------------------------------------------------------------------


class TestCompileRefusesUnsupportedShapes:
    """The planner is supposed to route these to GRAPH, but the compiler
    should still refuse them defensively so a misrouted CQ fails loudly
    rather than emitting malformed SQL."""

    def test_median_raises(self, backend):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(concepts=[("creatinine", "biomarker")], aggregation="median")
        with pytest.raises(ValueError, match="(?i)median|sql_fn|fast.?path"):
            compile_sql(cq, backend, get_default_registry())

    def test_temporal_constraint_raises(self, backend):
        from src.conversational.models import TemporalConstraint
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(concepts=[("creatinine", "biomarker")], aggregation="mean")
        cq.temporal_constraints = [
            TemporalConstraint(relation="during", reference_event="ICU stay")
        ]
        with pytest.raises(ValueError, match="(?i)temporal|graph"):
            compile_sql(cq, backend, get_default_registry())

    def test_multiple_concepts_raises(self, backend):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("creatinine", "biomarker"), ("lactate", "biomarker")],
            aggregation="mean",
        )
        with pytest.raises(ValueError, match="(?i)concept"):
            compile_sql(cq, backend, get_default_registry())


# ===========================================================================
# Measurement-value cohort filters (lab / vital / derived) + composition
# ===========================================================================

from pathlib import Path as _Path  # noqa: E402
from src.conversational.concept_resolver import ConceptResolver  # noqa: E402
from src.conversational.derived_formula import (  # noqa: E402
    DerivedFormula, Operand,
)

_MAPPINGS = _Path(__file__).parent.parent.parent / "data" / "mappings"


def _outcome_cq(filters):
    return CompetencyQuestion(
        original_question="test",
        clinical_concepts=[ClinicalConcept(
            name="in-hospital mortality", concept_type="outcome")],
        patient_filters=filters,
        aggregation="count", scope="cohort", return_type="text_and_table",
    )


class TestMeasurementValueFilters:
    @pytest.fixture
    def resolver(self):
        return ConceptResolver(mappings_dir=_MAPPINGS)

    def test_lab_value_grounds_to_itemid_and_executes(self, backend, resolver):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql
        cq = _outcome_cq([PatientFilter(
            field="lab_value", operator="<", value="50",
            measurement="platelet count", loinc_code="777-3")])
        q = compile_sql(cq, backend, get_default_registry(), resolver=resolver)
        assert "EXISTS" in q.sql and "labevents" in q.sql
        assert 51265 in q.params  # platelet itemid, grounded via LOINC
        assert ".valuenum <" in q.sql
        backend.execute(q.sql, q.params)  # executes without error

    def test_lab_value_label_fallback_without_resolver(self, backend):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql
        cq = _outcome_cq([PatientFilter(
            field="lab_value", operator="<", value="50",
            measurement="platelet count")])
        q = compile_sql(cq, backend, get_default_registry())  # no resolver
        assert "EXISTS" in q.sql and "d_labitems" in q.sql  # label-LIKE fallback
        assert "%platelet count%" in q.params
        backend.execute(q.sql, q.params)

    def test_vital_value_grounds_to_chartevents(self, backend, resolver):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql
        cq = _outcome_cq([PatientFilter(
            field="vital_value", operator=">", value="90",
            measurement="heart rate", loinc_code="8867-4")])
        q = compile_sql(cq, backend, get_default_registry(), resolver=resolver)
        assert "EXISTS" in q.sql and "chartevents" in q.sql
        assert 220045 in q.params and ".valuenum >" in q.sql
        backend.execute(q.sql, q.params)

    def test_icu_stay_filter_exists_icustays(self, backend):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql
        cq = _outcome_cq([PatientFilter(field="icu_stay", operator="=", value="1")])
        q = compile_sql(cq, backend, get_default_registry())
        assert "EXISTS" in q.sql and "icustays" in q.sql
        backend.execute(q.sql, q.params)

    def test_or_any_unions_child_exists_clauses(self, backend, resolver):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql
        cq = _outcome_cq([PatientFilter(
            field="or_any", operator="in", value="any", sub_filters=[
                PatientFilter(field="lab_value", operator="<", value="50",
                              measurement="platelet count", loinc_code="777-3"),
                PatientFilter(field="vital_value", operator=">", value="90",
                              measurement="heart rate", loinc_code="8867-4"),
                PatientFilter(field="icu_stay", operator="=", value="1"),
            ])])
        q = compile_sql(cq, backend, get_default_registry(), resolver=resolver)
        assert " OR " in q.sql and q.sql.count("EXISTS") >= 3
        backend.execute(q.sql, q.params)


class TestDerivedValueFilter:
    @pytest.fixture
    def shock_index(self):
        # threshold 0.5 so the fixture's co-charted HR 78 / SBP 120 = 0.65 matches
        return DerivedFormula(
            operands=(
                Operand(ref="hr", itemids=(220045,), table="chartevents",
                        guard_low=10, guard_high=300),
                Operand(ref="sbp", itemids=(220179,), table="chartevents",
                        guard_low=30, guard_high=300),
            ),
            expression={"op": "/", "args": ["hr", "sbp"]},
            operator=">=", threshold=0.5,
            time_semantics="per_instant", stay_aggregate="max",
        )

    def test_per_instant_time_matched_self_join(self, backend, shock_index):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql
        cq = _outcome_cq([PatientFilter(
            field="derived_value", operator=">=", value="0.5",
            measurement="shock index")])
        q = compile_sql(cq, backend, get_default_registry(),
                        derived_formulas={"shock index": shock_index})
        # Row-level "ever crosses" form (de-correlatable EXISTS, no HAVING).
        # (The outer outcome query has its own GROUP BY hospital_expire_flag —
        # only the derived EXISTS must be aggregate-free.)
        assert ".stay_id = " in q.sql and ".charttime = " in q.sql
        assert "NULLIF(" in q.sql and ") >= ?" in q.sql
        assert "HAVING" not in q.sql
        backend.execute(q.sql, q.params)  # co-charted HR/SBP → ratio computable

    def test_per_stay_three_operand_subqueries(self, backend):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql
        # anion-gap-style 3-operand per_stay formula (all labevents)
        formula = DerivedFormula(
            operands=(
                Operand(ref="na", itemids=(50983,), table="labevents", aggregate="max"),
                Operand(ref="cl", itemids=(50902,), table="labevents", aggregate="max"),
                Operand(ref="hco3", itemids=(50882,), table="labevents", aggregate="max"),
            ),
            expression={"op": "-", "args": ["na", {"op": "+", "args": ["cl", "hco3"]}]},
            operator=">", threshold=12, time_semantics="per_stay",
        )
        cq = _outcome_cq([PatientFilter(
            field="derived_value", operator=">", value="12",
            measurement="anion gap")])
        q = compile_sql(cq, backend, get_default_registry(),
                        derived_formulas={"anion gap": formula})
        assert q.sql.count("SELECT MAX(mv.valuenum)") == 3  # one subquery per operand
        backend.execute(q.sql, q.params)

    def test_missing_formula_drops_arm_to_empty(self, backend):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql
        cq = _outcome_cq([PatientFilter(
            field="derived_value", operator=">=", value="0.9",
            measurement="shock index")])
        q = compile_sql(cq, backend, get_default_registry())  # no derived_formulas
        assert "1 = 0" in q.sql
        backend.execute(q.sql, q.params)


class TestMultiDiagnosisFilterCompileSql:
    """End-to-end guard for the duplicate-``di``-alias crash. A cohort aggregate
    filtered by TWO diagnoses compiled to SQL with ``di``/``dd`` repeated in one
    FROM clause; BigQuery (and DuckDB) reject that as a duplicate table alias —
    the ``ValidatorBlockedQueryError`` → "Analysis failed" crash on
    *"... high anion gap metabolic acidosis vs. a non high anion gap ..."*."""

    def test_two_diagnosis_filters_compile_and_execute(self, backend):
        import re

        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("creatinine", "biomarker")],
            filters=[
                ("diagnosis", "contains", "metabolic acidosis"),
                ("diagnosis", "contains", "diabetes"),
            ],
            aggregation="mean",
        )
        q = compile_sql(cq, backend, get_default_registry())

        # Each diagnosis filter joins diagnoses_icd under a DISTINCT alias.
        di_aliases = [
            a for a in re.findall(r"diagnoses_icd`?\s+(\w+)\b", q.sql)
            if a.startswith("di")
        ]
        assert len(di_aliases) == 2, q.sql
        assert len(set(di_aliases)) == 2, f"duplicate alias in {q.sql}"

        # Executes without a "duplicate table alias" binder error (the crash).
        backend.execute(q.sql, q.params)

    def test_diagnosis_concept_count_with_diagnosis_filter(self, backend):
        """The actual reported crash: a diagnosis CONCEPT counted
        (``FROM diagnoses_icd di``) within a cohort that ALSO carries a diagnosis
        FILTER (which joins diagnoses_icd again). The count path reserves
        ``di``/``dd``; the filter must continue at ``di1``/``dd1``. This is what
        *"... high anion gap metabolic acidosis among ICU admissions with a
        metabolic acidosis"* decomposed to."""
        import re

        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("high anion gap metabolic acidosis", "diagnosis")],
            filters=[("diagnosis", "contains", "metabolic acidosis")],
            aggregation="count",
        )
        q = compile_sql(cq, backend, get_default_registry())

        di_aliases = [
            a for a in re.findall(r"diagnoses_icd`?\s+(\w+)\b", q.sql)
            if a.startswith("di")
        ]
        assert len(di_aliases) == 2, q.sql
        assert len(set(di_aliases)) == 2, f"duplicate alias in {q.sql}"
        # The base concept keeps di; the filter is pushed to di1.
        assert "di" in di_aliases and "di1" in di_aliases, di_aliases

        backend.execute(q.sql, q.params)


class TestStoredMeasurementThreshold:
    """A computed-but-STORED quantity (anion gap → MIMIC labevents itemid 50868)
    must route to a lab_value threshold on the STORED value, using the FILTER's
    operator — so 'high' (>) and 'non-high' (<=) both work and the stored value
    is the source of truth (no flaky derived recomputation, no diagnosis LIKE,
    no `1 = 0`). Guards the anion-gap-by-anion-gap-type fix."""

    @pytest.mark.parametrize("op,val", [(">", "12"), ("<=", "12")])
    def test_anion_gap_lab_value_uses_stored_value_and_filter_operator(
        self, backend, op, val,
    ):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = CompetencyQuestion(
            original_question="count",
            clinical_concepts=[
                ClinicalConcept(name="metabolic acidosis", concept_type="diagnosis"),
            ],
            patient_filters=[
                PatientFilter(field="diagnosis", operator="contains", value="metabolic acidosis"),
                PatientFilter(field="lab_value", operator=op, value=val, measurement="anion gap"),
            ],
            aggregation="count", scope="cohort",
        )
        q = compile_sql(cq, backend, get_default_registry())
        sql = q.sql.lower()

        # Grounds the STORED anion-gap lab via its label (which matches itemid
        # 50868 "Anion Gap"), and thresholds the stored valuenum with the
        # FILTER's operator (so the high/non-high split is correct).
        assert "labevents" in sql
        assert "%anion gap%" in str(q.params).lower()
        assert f"valuenum {op} ?" in sql
        # NOT a flaky derived recomputation and NOT a tautological empty arm.
        assert "1 = 0" not in sql
        backend.execute(q.sql, q.params)
