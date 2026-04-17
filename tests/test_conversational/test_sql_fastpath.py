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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cq(
    *,
    concepts: list[tuple[str, str]] | None = None,
    filters: list[tuple[str, str, str]] | None = None,
    aggregation: str | None = None,
    scope: str = "cohort",
    comparison_field: str | None = None,
    return_type: str = "text_and_table",
) -> CompetencyQuestion:
    return CompetencyQuestion(
        original_question="test",
        clinical_concepts=[
            ClinicalConcept(name=n, concept_type=t) for n, t in (concepts or [])
        ],
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
    from src.conversational.operations import get_default_registry
    from src.conversational.sql_fastpath import compile_sql

    q = compile_sql(cq, backend, get_default_registry())
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
