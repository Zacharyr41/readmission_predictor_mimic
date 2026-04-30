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
