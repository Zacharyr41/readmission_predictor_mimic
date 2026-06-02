"""Tests for pre-aggregation biological-impossibility outlier screening.

Originating bug: a clinician asked for the average lactate among sepsis
patients aged >= 65 and got a mean > 8 mmol/L. The cause was a single
``labevents`` row whose lactate ``valuenum`` had been entered as
1,000,000 (a data-entry error). ``AVG()`` over that poison value polluted
the result. The existing plausibility critic only ever sees the scalar
aggregate, never the individual rows, so it could not catch this.

The fix screens out *biologically impossible* values **in SQL, before the
aggregate**, using a constant ``BETWEEN`` envelope resolved once per analyte.
High-but-possible values (sepsis lactate 12 mmol/L) are kept; only impossible
values (1e6) are removed. The screen is uniform across AVG/MAX/MIN/COUNT and
GROUP BY so the reported ``n`` stays consistent with the screened mean.

These tests are written TDD-first: they describe the target API
(``OutlierScreen``, ``compile_sql(..., outlier_screen=...)``,
``BiologicalLimitsResolver``, ``OutlierReport``) before it exists, so they
fail RED until the implementation lands.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# anthropic is not installed in the test environment; stub it so the
# orchestrator-wiring test below can construct ConversationalPipeline
# (its __init__ does ``import anthropic``). Mirrors test_orchestrator.py.
sys.modules.setdefault("anthropic", MagicMock())

from src.conversational.models import ClinicalConcept

# Reuse the conn-sharing backend adapter and the CQ builder from the
# fast-path suite so this file stays in lockstep with the compiler's
# test conventions (same fixtures, same _cq tuple shapes).
from tests.test_conversational.test_sql_fastpath import _ConnBackend, _cq


# ---------------------------------------------------------------------------
# Constants + fixture
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
BIO_LIMITS_PATH = REPO_ROOT / "data" / "ontology_cache" / "biological_limits.json"

# Lactate biological-possibility envelope (mmol/L). Far wider than the
# normal reference range (0.5-2.0): sepsis can drive lactate to ~20, and we
# only want to remove the physically-impossible. 12 is a real sepsis value
# (kept); 1e6 is a data-entry error (removed).
LACTATE_LOW = 0.0
LACTATE_HIGH = 40.0

# Real MIMIC-IV itemid for blood lactate. Passed as ``resolved_itemids`` so
# the WHERE clause uses ``itemid IN (...)`` and the test is independent of
# the LOINC -> itemid mappings file.
LACTATE_ITEMID = 50813


@pytest.fixture
def backend_with_lactate_poison(synthetic_duckdb_with_events):
    """Base synthetic events + a lactate analyte with one poison row.

    Four lactate rows for the age >= 65 cohort (patients 1, 2, 5):
      - 1.2 mmol/L  (patient 1, normal)
      - 2.5 mmol/L  (patient 2, mildly elevated)
      - 12.0 mmol/L (patient 5, real severe-sepsis value -- KEEP)
      - 1e6 mmol/L  (patient 1, data-entry error -- REMOVE)

    Clean mean = (1.2 + 2.5 + 12.0) / 3 = 5.2333...
    Polluted mean = (1.2 + 2.5 + 12.0 + 1e6) / 4 = 250003.925
    """
    conn = synthetic_duckdb_with_events
    conn.execute(
        "INSERT INTO d_labitems VALUES (50813, 'Lactate', 'Blood', 'Chemistry')"
    )
    conn.execute(
        "INSERT INTO labevents VALUES "
        "(50, 1, 101, 1001, 50813, '2150-01-16 06:00:00', 1.2, 'mmol/L', 0.5, 2.0), "
        "(51, 2, 103, 1002, 50813, '2151-03-03 08:00:00', 2.5, 'mmol/L', 0.5, 2.0), "
        "(52, 5, 106, 1003, 50813, '2151-04-12 10:00:00', 12.0, 'mmol/L', 0.5, 2.0), "
        "(53, 1, 101, 1001, 50813, '2150-01-16 07:00:00', 1000000.0, 'mmol/L', 0.5, 2.0)"
    )
    return _ConnBackend(conn)


@pytest.fixture
def backend_with_lactate_unit_mismatch(synthetic_duckdb_with_events):
    """Standard lactate rows plus one row recorded in a different unit.

    On top of the four mmol/L rows (incl. the 1e6 poison) there is one
    extra lactate row in **mg/dL**:
      - 90.0 mg/dL (patient 2): ~10 mmol/L after conversion -- a real
        severe-sepsis value, but 90 falls *outside* the mmol/L envelope
        [0, 40]. The bound is expressed in mmol/L, so it must NOT apply
        to a mg/dL row -- the units guard keeps it.

    With the guard (``screen.units='mmol/L'``) only the matching-unit
    1e6 row is removed (n_outliers == 1). Without it, both 1e6 and 90
    are shed (n_outliers == 2) -- the false removal the guard prevents.
    """
    conn = synthetic_duckdb_with_events
    conn.execute(
        "INSERT INTO d_labitems VALUES (50813, 'Lactate', 'Blood', 'Chemistry')"
    )
    conn.execute(
        "INSERT INTO labevents VALUES "
        "(50, 1, 101, 1001, 50813, '2150-01-16 06:00:00', 1.2, 'mmol/L', 0.5, 2.0), "
        "(51, 2, 103, 1002, 50813, '2151-03-03 08:00:00', 2.5, 'mmol/L', 0.5, 2.0), "
        "(52, 5, 106, 1003, 50813, '2151-04-12 10:00:00', 12.0, 'mmol/L', 0.5, 2.0), "
        "(53, 1, 101, 1001, 50813, '2150-01-16 07:00:00', 1000000.0, 'mmol/L', 0.5, 2.0), "
        "(54, 2, 103, 1002, 50813, '2151-03-03 09:00:00', 90.0, 'mg/dL', 0.5, 2.0)"
    )
    return _ConnBackend(conn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _screen(low: float = LACTATE_LOW, high: float = LACTATE_HIGH, **kw):
    from src.conversational.sql_fastpath import OutlierScreen

    return OutlierScreen(low=low, high=high, **kw)


def _run_screened(backend, cq, screen, *, resolved_itemids=None):
    """Compile with an outlier screen, execute, and zip rows with the full
    column list (``outlier_agg_columns`` includes the with-outliers + count
    columns the orchestrator consumes for the report)."""
    from src.conversational.operations import get_default_registry
    from src.conversational.sql_fastpath import compile_sql

    q = compile_sql(
        cq,
        backend,
        get_default_registry(),
        resolved_itemids=resolved_itemids,
        outlier_screen=screen,
    )
    cols = q.outlier_agg_columns or q.columns
    rows = backend.execute(q.sql, q.params)
    return [dict(zip(cols, r)) for r in rows], q


# ---------------------------------------------------------------------------
# 1. Bug replication + the screened fix (scalar aggregates)
# ---------------------------------------------------------------------------


class TestOutlierScreenAggregate:
    def test_no_screen_baseline_is_polluted(self, backend_with_lactate_poison):
        """Documents the bug: without a screen, the poison row pollutes the
        mean. This is today's behavior and must stay unchanged for the
        no-screen path."""
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("lactate", "biomarker")],
            aggregation="mean",
            filters=[("age", ">=", "65")],
        )
        q = compile_sql(
            cq,
            backend_with_lactate_poison,
            get_default_registry(),
            resolved_itemids=[LACTATE_ITEMID],
        )
        rows = backend_with_lactate_poison.execute(q.sql, q.params)
        mean = dict(zip(q.columns, rows[0]))["mean_value"]
        assert mean > 1000.0  # ~250004 -- clinically nonsensical

    def test_screened_mean_excludes_poison(self, backend_with_lactate_poison):
        cq = _cq(
            concepts=[("lactate", "biomarker")],
            aggregation="mean",
            filters=[("age", ">=", "65")],
        )
        rows, _q = _run_screened(
            backend_with_lactate_poison, cq, _screen(),
            resolved_itemids=[LACTATE_ITEMID],
        )
        row = rows[0]
        assert math.isclose(row["mean_value"], (1.2 + 2.5 + 12.0) / 3, rel_tol=1e-6)
        assert row["mean_value_with_outliers"] > 1000.0
        assert row["n_outliers"] == 1
        assert row["n_total"] == 4

    def test_max_keeps_high_but_real_value(self, backend_with_lactate_poison):
        """No false removal: the real sepsis value (12) is the clean MAX;
        only the impossible 1e6 is shed to the with-outliers column."""
        cq = _cq(
            concepts=[("lactate", "biomarker")],
            aggregation="max",
            filters=[("age", ">=", "65")],
        )
        rows, _q = _run_screened(
            backend_with_lactate_poison, cq, _screen(),
            resolved_itemids=[LACTATE_ITEMID],
        )
        row = rows[0]
        assert math.isclose(row["max_value"], 12.0)
        assert math.isclose(row["max_value_with_outliers"], 1_000_000.0)
        assert row["n_outliers"] == 1

    def test_count_screened_and_consistent(self, backend_with_lactate_poison):
        """COUNT is screened like any other aggregate so the reported ``n``
        matches the screened mean's denominator (n_total - n_outliers)."""
        cq = _cq(
            concepts=[("lactate", "biomarker")],
            aggregation="count",
            filters=[("age", ">=", "65")],
        )
        rows, _q = _run_screened(
            backend_with_lactate_poison, cq, _screen(),
            resolved_itemids=[LACTATE_ITEMID],
        )
        row = rows[0]
        assert row["count_value"] == 3
        assert row["count_value_with_outliers"] == 4
        assert row["n_outliers"] == 1
        assert row["n_total"] == 4
        assert row["count_value"] == row["n_total"] - row["n_outliers"]

    def test_no_screen_compiles_unchanged(self, backend_with_lactate_poison):
        """outlier_screen=None must reproduce today's SQL exactly: no
        with-outliers columns, no companion query, no BETWEEN clause."""
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("lactate", "biomarker")],
            aggregation="mean",
            filters=[("age", ">=", "65")],
        )
        q = compile_sql(
            cq,
            backend_with_lactate_poison,
            get_default_registry(),
            resolved_itemids=[LACTATE_ITEMID],
            outlier_screen=None,
        )
        assert q.outlier_agg_columns is None
        assert q.outlier_rows_sql is None
        assert "BETWEEN" not in q.sql.upper()


# ---------------------------------------------------------------------------
# 2. Logging -- the companion rows query surfaces the removed rows
# ---------------------------------------------------------------------------


class TestOutlierRowsCompanion:
    def test_companion_query_logs_poison_with_context(self, backend_with_lactate_poison):
        cq = _cq(
            concepts=[("lactate", "biomarker")],
            aggregation="mean",
            filters=[("age", ">=", "65")],
        )
        _rows, q = _run_screened(
            backend_with_lactate_poison, cq, _screen(),
            resolved_itemids=[LACTATE_ITEMID],
        )
        assert q.outlier_rows_sql is not None
        raw = backend_with_lactate_poison.execute(
            q.outlier_rows_sql, q.outlier_rows_params
        )
        logged = [dict(zip(q.outlier_rows_columns, r)) for r in raw]
        assert len(logged) == 1
        rec = logged[0]
        assert math.isclose(rec["valuenum"], 1_000_000.0)
        assert rec["subject_id"] == 1
        assert rec["hadm_id"] == 101
        assert "lactate" in str(rec["label"]).lower()
        assert rec["valueuom"] == "mmol/L"


# ---------------------------------------------------------------------------
# 2b. Units guard -- the bound only screens rows in the bound's own unit
# ---------------------------------------------------------------------------


class TestUnitsGuard:
    """The impossibility bound is expressed in one unit; it must not
    screen a row recorded in a *different* unit (a value that is legit in
    its own unit can look impossible in the bound's). itemid-grounding
    keeps units homogeneous in practice, so this is belt-and-suspenders,
    but a stray mixed-unit row must never be silently dropped."""

    @staticmethod
    def _mean_cq():
        return _cq(
            concepts=[("lactate", "biomarker")],
            aggregation="mean",
            filters=[("age", ">=", "65")],
        )

    def test_mismatched_unit_row_is_kept(self, backend_with_lactate_unit_mismatch):
        rows, _q = _run_screened(
            backend_with_lactate_unit_mismatch, self._mean_cq(),
            _screen(units="mmol/L"),
            resolved_itemids=[LACTATE_ITEMID],
        )
        row = rows[0]
        # Only the matching-unit (mmol/L) 1e6 poison is removed; the
        # 90 mg/dL row is outside [0, 40] but in a different unit, so the
        # bound does not apply and the row is kept.
        assert row["n_outliers"] == 1
        # Every candidate row (5) is part of the screened denominator --
        # the kept mg/dL row included.
        assert row["n_total"] == 5

    def test_mismatched_unit_row_absent_from_companion(
        self, backend_with_lactate_unit_mismatch,
    ):
        _rows, q = _run_screened(
            backend_with_lactate_unit_mismatch, self._mean_cq(),
            _screen(units="mmol/L"),
            resolved_itemids=[LACTATE_ITEMID],
        )
        raw = backend_with_lactate_unit_mismatch.execute(
            q.outlier_rows_sql, q.outlier_rows_params
        )
        logged = [dict(zip(q.outlier_rows_columns, r)) for r in raw]
        # The companion logs only the mmol/L poison, never the mg/dL row.
        assert len(logged) == 1
        assert math.isclose(logged[0]["valuenum"], 1_000_000.0)
        assert logged[0]["valueuom"] == "mmol/L"

    def test_without_units_guard_mismatched_row_is_falsely_removed(
        self, backend_with_lactate_unit_mismatch,
    ):
        """Documents WHY the guard exists: with no units on the screen,
        the mg/dL value (90) is wrongly screened against the mmol/L
        envelope and removed alongside the genuine poison."""
        rows, _q = _run_screened(
            backend_with_lactate_unit_mismatch, self._mean_cq(),
            _screen(),  # units=None -> no guard, today's behavior
            resolved_itemids=[LACTATE_ITEMID],
        )
        assert rows[0]["n_outliers"] == 2

    def test_units_guard_emitted_in_sql(self, backend_with_lactate_unit_mismatch):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        q = compile_sql(
            self._mean_cq(),
            backend_with_lactate_unit_mismatch,
            get_default_registry(),
            resolved_itemids=[LACTATE_ITEMID],
            outlier_screen=_screen(units="mmol/L"),
        )
        sql = q.sql.upper()
        # The aggregate carries the per-row unit comparison alongside the
        # BETWEEN envelope.
        assert "VALUEUOM" in sql
        assert "TRIM(" in sql
        assert "BETWEEN" in sql
        # The companion rows query carries the same guard.
        assert "VALUEUOM" in q.outlier_rows_sql.upper()


# ---------------------------------------------------------------------------
# 3. SQL emission (Tier-3 style: assert on the compiled SQL text)
# ---------------------------------------------------------------------------


class TestOutlierSqlEmission:
    def test_aggregate_sql_contains_case_between_screen(self, backend_with_lactate_poison):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("lactate", "biomarker")],
            aggregation="mean",
            filters=[("age", ">=", "65")],
        )
        q = compile_sql(
            cq,
            backend_with_lactate_poison,
            get_default_registry(),
            resolved_itemids=[LACTATE_ITEMID],
            outlier_screen=_screen(),
        )
        sql = q.sql.upper()
        assert "CASE WHEN" in sql
        assert "BETWEEN" in sql
        assert "MEAN_VALUE_WITH_OUTLIERS" in sql
        assert "N_OUTLIERS" in sql

    def test_companion_sql_contains_not_between_and_limit(self, backend_with_lactate_poison):
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(
            concepts=[("lactate", "biomarker")],
            aggregation="mean",
            filters=[("age", ">=", "65")],
        )
        q = compile_sql(
            cq,
            backend_with_lactate_poison,
            get_default_registry(),
            resolved_itemids=[LACTATE_ITEMID],
            outlier_screen=_screen(),
        )
        assert q.outlier_rows_sql is not None
        rsql = q.outlier_rows_sql.upper()
        assert "NOT (" in rsql
        assert "BETWEEN" in rsql
        assert "LIMIT" in rsql


# ---------------------------------------------------------------------------
# 4. GROUP BY / comparison -- the screen applies per group
# ---------------------------------------------------------------------------


class TestGroupByScreen:
    def test_comparison_screens_each_group(self, backend_with_lactate_poison):
        cq = _cq(
            concepts=[("lactate", "biomarker")],
            aggregation="mean",
            scope="comparison",
            comparison_field="gender",
            filters=[("age", ">=", "65")],
        )
        rows, q = _run_screened(
            backend_with_lactate_poison, cq, _screen(),
            resolved_itemids=[LACTATE_ITEMID],
        )
        by_group = {r["group_value"]: r for r in rows}
        # M = patients 1 & 5: clean rows 1.2 and 12.0 (poison 1e6 removed).
        assert math.isclose(by_group["M"]["avg_value"], (1.2 + 12.0) / 2, rel_tol=1e-6)
        # F = patient 2: single row 2.5, no outliers.
        assert math.isclose(by_group["F"]["avg_value"], 2.5, rel_tol=1e-6)
        # The companion query is not grouped -- it lists the one poison row.
        raw = backend_with_lactate_poison.execute(
            q.outlier_rows_sql, q.outlier_rows_params
        )
        assert len(raw) == 1


# ---------------------------------------------------------------------------
# 5. Categorical count -- the screen is a harmless no-op (no numeric bound)
# ---------------------------------------------------------------------------


class TestCategoricalNoScreen:
    def test_diagnosis_count_unaffected_by_screen(self, backend_with_lactate_poison):
        """A diagnosis COUNT carries no numeric measurement, so even when a
        screen is passed the compiler finds nothing to bound: today's SQL is
        emitted verbatim, with no with-outliers columns and no companion."""
        from src.conversational.operations import get_default_registry
        from src.conversational.sql_fastpath import compile_sql

        cq = _cq(concepts=[("cerebral", "diagnosis")], aggregation="count")
        q = compile_sql(
            cq,
            backend_with_lactate_poison,
            get_default_registry(),
            outlier_screen=_screen(),
        )
        assert q.outlier_agg_columns is None
        assert q.outlier_rows_sql is None
        assert "BETWEEN" not in q.sql.upper()


# ---------------------------------------------------------------------------
# 6. Biological-limits resolver -- bounds from seed + grounded fallback
# ---------------------------------------------------------------------------


class TestBiologicalLimitsResolver:
    def test_seed_resolves_lactate_envelope(self):
        from src.conversational.outliers import BiologicalLimitsResolver

        resolver = BiologicalLimitsResolver(
            cache_path=BIO_LIMITS_PATH, enable_derivation=False,
        )
        limits = resolver.resolve(
            ClinicalConcept(name="lactate", concept_type="biomarker")
        )
        assert limits is not None
        # Sepsis-real value kept inside the envelope; poison far outside it.
        assert limits.low <= 12.0 <= limits.high
        assert limits.high < 1_000_000.0

    def test_grounded_derivation_caches_and_uses_judgment_model(
        self, tmp_path, monkeypatch
    ):
        """On a cache miss with derivation enabled, the resolver consults an
        EvidenceAgent (Sonnet/Opus -- never Haiku), validates the bounds, and
        writes them back to the cache. The agent is faked so the test is
        offline."""
        import src.conversational.outliers as outliers_mod
        from src.conversational.outliers import BiologicalLimitsResolver

        captured: dict = {}

        class _FakeEvidenceAgent:
            def __init__(self, client, *, model="claude-sonnet-4-6", **kw):
                captured["model"] = model
                self.calls = 0

            def consult(self, system_prompt, user_prompt):
                from types import SimpleNamespace

                self.calls += 1
                captured["calls"] = self.calls
                return SimpleNamespace(
                    parsed_json={"low": 0.0, "high": 50.0, "units": "ng/mL"},
                    final_text="",
                    observed_citations=[],
                    tool_calls=[],
                )

        monkeypatch.setattr(outliers_mod, "EvidenceAgent", _FakeEvidenceAgent)

        cache = tmp_path / "bl.json"  # missing -> empty index -> forces derivation
        resolver = BiologicalLimitsResolver(
            cache_path=cache, client=object(), enable_derivation=True,
        )
        limits = resolver.resolve(
            ClinicalConcept(name="obscure_analyte_xyz", concept_type="biomarker")
        )
        assert limits is not None
        assert math.isclose(limits.high, 50.0)
        assert captured.get("calls") == 1
        model = captured["model"].lower()
        assert "haiku" not in model
        assert ("sonnet" in model) or ("opus" in model)
        assert cache.exists()  # written back for next time

    def test_graceful_skip_returns_none_when_no_bound(self, tmp_path):
        """No seed entry + derivation disabled => return None so the caller
        skips screening (never invents a bound, never false-removes)."""
        from src.conversational.outliers import BiologicalLimitsResolver

        resolver = BiologicalLimitsResolver(
            cache_path=tmp_path / "empty.json", enable_derivation=False,
        )
        limits = resolver.resolve(
            ClinicalConcept(name="obscure_analyte_xyz", concept_type="biomarker")
        )
        assert limits is None


# ---------------------------------------------------------------------------
# 7. OutlierReport model -- carries the report onto AnswerResult for the UI
# ---------------------------------------------------------------------------


class TestOutlierReportModel:
    def test_report_attaches_to_answer_result(self):
        from src.conversational.models import AnswerResult, OutlierReport

        report = OutlierReport(
            analyte="lactate",
            bound_low=0.0,
            bound_high=40.0,
            units="mmol/L",
            source="seed:literature",
            method="biological_limits",
            n_removed=1,
            n_total=4,
            removed_rows=[{"valuenum": 1_000_000.0, "subject_id": 1, "hadm_id": 101}],
            value_with_outliers=250003.925,
            data_table_with_outliers=[{"mean_value": 250003.925}],
        )
        answer = AnswerResult(text_summary="mean lactate is 5.23 mmol/L", outlier_report=report)
        assert answer.outlier_report is report
        assert answer.outlier_report.n_removed == 1
        assert answer.outlier_report.bound_high == 40.0


# ---------------------------------------------------------------------------
# 8. Orchestrator wiring -- the live path that the UI consumes
# ---------------------------------------------------------------------------


class TestOrchestratorOutlierWiring:
    """End-to-end through ``ConversationalPipeline._run_sql_fastpath``: the
    screened aggregate runs against a real backend and the assembled
    ``OutlierReport`` is attached to the ``AnswerResult``. This exercises the
    report-assembly logic (``_build_outlier_report``) the unit tests above
    don't reach -- so the wiring isn't shipped unverified."""

    def test_run_sql_fastpath_attaches_outlier_report(
        self, backend_with_lactate_poison,
    ):
        from types import SimpleNamespace
        from unittest.mock import patch

        from src.conversational.models import AnswerResult
        from src.conversational.orchestrator import ConversationalPipeline
        from src.conversational.outliers import BiologicalLimits
        from src.conversational.sql_fastpath import OutlierScreen

        pipeline = ConversationalPipeline(
            Path("/tmp/test.duckdb"), Path("/tmp/ontology"), "test-key",
        )
        # Deterministic LOINC->itemid grounding so the WHERE uses
        # ``itemid IN (50813)`` independently of the mappings file.
        pipeline._resolver.resolve_biomarker = MagicMock(
            return_value=SimpleNamespace(
                itemids=[LACTATE_ITEMID], fallback_reason=None,
            )
        )

        cq = _cq(
            concepts=[("lactate", "biomarker", "2524-7")],
            aggregation="mean",
            filters=[("age", ">=", "65")],
        )
        screen = OutlierScreen(low=LACTATE_LOW, high=LACTATE_HIGH)
        limits = BiologicalLimits(
            low=LACTATE_LOW, high=LACTATE_HIGH,
            units="mmol/L", source="seed:literature",
        )

        captured: dict = {}

        def _fake_generate_answer(client, cq_, rows, graph_stats, sql):
            captured["rows"] = rows
            return AnswerResult(text_summary="mean lactate")

        with patch(
            "src.conversational.orchestrator.generate_answer",
            _fake_generate_answer,
        ):
            answer, _sql_list, _fb = pipeline._run_sql_fastpath(
                cq, backend_with_lactate_poison, resolved_names=[],
                outlier_screen=screen, outlier_limits=limits,
            )

        # The answerer sees the *clean* mean (poison excluded), same shape as
        # the unscreened path -- a single ``mean_value`` column.
        assert math.isclose(
            captured["rows"][0]["mean_value"],
            (1.2 + 2.5 + 12.0) / 3, rel_tol=1e-6,
        )

        rep = answer.outlier_report
        assert rep is not None
        assert rep.analyte == "lactate"
        assert rep.n_removed == 1
        assert rep.n_total == 4
        assert math.isclose(rep.bound_low, LACTATE_LOW)
        assert math.isclose(rep.bound_high, LACTATE_HIGH)
        assert rep.units == "mmol/L"
        # The removed poison row is logged with its context columns.
        assert len(rep.removed_rows) == 1
        assert math.isclose(rep.removed_rows[0]["valuenum"], 1_000_000.0)
        assert rep.removed_rows[0]["subject_id"] == 1
        # The with-outliers value is precomputed (the polluted mean) so the UI
        # toggle needs no backend round-trip.
        assert rep.value_with_outliers is not None
        assert rep.value_with_outliers > 1000.0

    def test_no_report_when_screen_resolves_nothing(
        self, backend_with_lactate_poison,
    ):
        """When no bound resolves (screen=None), the path is byte-for-byte
        today's: clean answer, no ``outlier_report``."""
        from unittest.mock import patch

        from src.conversational.models import AnswerResult
        from src.conversational.orchestrator import ConversationalPipeline

        pipeline = ConversationalPipeline(
            Path("/tmp/test.duckdb"), Path("/tmp/ontology"), "test-key",
        )
        pipeline._resolver.resolve_biomarker = MagicMock(
            return_value=__import__("types").SimpleNamespace(
                itemids=[LACTATE_ITEMID], fallback_reason=None,
            )
        )
        cq = _cq(
            concepts=[("lactate", "biomarker", "2524-7")],
            aggregation="mean",
            filters=[("age", ">=", "65")],
        )
        with patch(
            "src.conversational.orchestrator.generate_answer",
            lambda *a, **k: AnswerResult(text_summary="mean lactate"),
        ):
            answer, _sql, _fb = pipeline._run_sql_fastpath(
                cq, backend_with_lactate_poison, resolved_names=[],
                outlier_screen=None, outlier_limits=None,
            )
        assert answer.outlier_report is None
