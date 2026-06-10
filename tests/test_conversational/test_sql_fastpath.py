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

    def test_combines_icd_prefix_with_title_like_as_or(self, backend):
        """Final WHERE shape: ``((di.icd_code LIKE ?) OR (<existing LIKE>))``.
        ICD-9 admissions whose codes aren't in OMOPHub still match via
        the LIKE branch on dd.long_title."""
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
        # Both clauses present.
        assert "di.icd_code LIKE ?" in query.sql
        assert "A419%" in query.params
        # The LIKE clause is backend-dependent (DuckDB ILIKE; BigQuery
        # LOWER(...) LIKE LOWER(...)). Either form is acceptable.
        sql_upper = query.sql.upper()
        assert "ILIKE" in sql_upper or "LIKE" in sql_upper
        # The label-substring param ("%sepsis%") is still emitted.
        assert "%sepsis%" in query.params

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
