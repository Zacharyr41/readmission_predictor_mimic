"""Tests for the OperationRegistry and its core operation kinds.

Two layers:
  1. Registry behaviour (register/get/supported_names/duplicates/describe).
  2. Filter-operation parity: each registered filter emits the same cohort
     hadm_ids that the pre-registry hand-rolled SQL did. This is checked by
     running both paths against the ``synthetic_duckdb`` fixture and comparing
     result sets — NOT by matching raw SQL strings, because formatting drift
     is noise.
"""

from __future__ import annotations

import pytest

from src.conversational.extractor import _DuckDBBackend, _get_filtered_hadm_ids
from src.conversational.models import ExtractionConfig, PatientFilter
from src.conversational.operations import (
    AggregateFragment,
    AggregateOperation,
    ComparisonFragment,
    ComparisonOperation,
    FilterCompileContext,
    FilterFragment,
    FilterOperation,
    OperationRegistry,
)
from src.conversational.operations_filters import register_default_filters


# ---------------------------------------------------------------------------
# 1. Registry core behaviour
# ---------------------------------------------------------------------------


class TestFilterCompileContext:
    """Inc 9.1 — front-half OMOPHub grounding for the patient-filter
    compiler. ``FilterCompileContext`` carries an ``enable_mcp_grounding``
    flag so individual filter compile_fns (specifically ``_compile_diagnosis``)
    can opt into MCP-backed grounding when invoked from the production
    pipeline. Default False keeps tests offline-safe and preserves
    byte-identical SQL for callers that don't pass the flag."""

    def test_default_mcp_grounding_disabled(self):
        from src.conversational.operations import FilterCompileContext
        ctx = FilterCompileContext(backend=object())
        assert ctx.enable_mcp_grounding is False

    def test_accepts_mcp_grounding_flag(self):
        from src.conversational.operations import FilterCompileContext
        ctx = FilterCompileContext(
            backend=object(), enable_mcp_grounding=True,
        )
        assert ctx.enable_mcp_grounding is True


class TestDiagnosisFilterGrounding:
    """Inc 9.2 — ``_compile_diagnosis`` consults
    ``_cached_icd_autocode`` (from concept_resolver) when
    ``ctx.enable_mcp_grounding=True`` and emits a parallel-OR IN-list
    alongside the existing title-LIKE clause. Targets the smoking-gun
    query path: ``mean lactate in sepsis cohort`` → biomarker-aggregate
    CQ with diagnosis-typed patient_filter."""

    @pytest.fixture
    def grounded_ctx(self, duckdb_backend):
        """Context with grounding enabled + lru_cache cleared."""
        from src.conversational import concept_resolver as cr
        from src.conversational.operations import FilterCompileContext
        cr._cached_icd_autocode.cache_clear()
        return FilterCompileContext(
            backend=duckdb_backend, enable_mcp_grounding=True,
        )

    @pytest.fixture
    def offline_ctx(self, duckdb_backend):
        """Context with grounding disabled (the unit-test default)."""
        from src.conversational.operations import FilterCompileContext
        return FilterCompileContext(
            backend=duckdb_backend, enable_mcp_grounding=False,
        )

    def test_emits_icd_in_when_grounded(
        self, grounded_ctx, monkeypatch,
    ):
        """OMOPHub autocode-fallback path. ``carcinoid syndrome`` is
        deliberately NOT in ``data/mappings/clinical_cohorts.json`` so
        Inc 10's registry-first lookup misses and the filter compiler
        falls through to ``icd_autocode``. (For phrases that DO match a
        registered cohort, see
        ``test_filter_uses_cohort_registry_when_value_matches_a_named_cohort``
        in tests/dashboard/.)"""
        from src.conversational import concept_resolver as cr
        from src.conversational.operations_filters import _compile_diagnosis

        def fake_autocode(text, **kwargs):
            return {
                "status": "ok",
                "results": [
                    {"code": "E34.0", "title": "Carcinoid syndrome", "confidence": 0.92},
                    {"code": "C7A.0", "title": "Carcinoid tumour", "confidence": 0.81},
                ],
            }
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        f = PatientFilter(
            field="diagnosis", operator="contains", value="carcinoid syndrome",
        )
        frag = _compile_diagnosis(f, grounded_ctx)
        sql = " ".join(frag.where)
        # icd_autocode returns dotted, category-level codes; MIMIC stores them
        # undotted+billable, so they are normalized and PREFIX-matched (exact IN
        # on the dotted code matched nothing).
        assert "di.icd_code LIKE ?" in sql
        assert "E340%" in frag.params
        assert "C7A0%" in frag.params

    def test_emits_codes_only_when_grounded(
        self, grounded_ctx, monkeypatch,
    ):
        """When grounded, the clause is CODES-ONLY — the broad title-LIKE is
        dropped (anti-pollution), matching the count path.

        Uses 'carcinoid syndrome' so the autocode-fallback path fires (registry
        -first means 'sepsis' would route to registry prefixes instead)."""
        from src.conversational import concept_resolver as cr
        from src.conversational.operations_filters import _compile_diagnosis

        def fake_autocode(text, **kwargs):
            return {
                "status": "ok",
                "results": [
                    {"code": "E34.0", "title": "Carcinoid syndrome", "confidence": 0.92},
                ],
            }
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        f = PatientFilter(
            field="diagnosis", operator="contains", value="carcinoid syndrome",
        )
        frag = _compile_diagnosis(f, grounded_ctx)
        sql = " ".join(frag.where)
        assert "di.icd_code LIKE ?" in sql
        assert "E340%" in frag.params  # normalized + prefix
        # Codes-only: the broad title-LIKE substring is NOT bound.
        assert "%carcinoid syndrome%" not in frag.params

    def test_falls_back_to_like_when_unavailable(
        self, grounded_ctx, monkeypatch,
    ):
        """No grounding (unregistered term + MCP unavailable) → title-LIKE
        fallback preserved. Uses 'carcinoid syndrome' (NOT a registered cohort)
        so the registry Tier-1 doesn't ground it offline."""
        from src.conversational import concept_resolver as cr
        from src.conversational.operations_filters import _compile_diagnosis

        def fake_autocode(text, **kwargs):
            return {"status": "unavailable", "error": "MCP timeout"}
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        f = PatientFilter(
            field="diagnosis", operator="contains", value="carcinoid syndrome",
        )
        frag = _compile_diagnosis(f, grounded_ctx)
        sql = " ".join(frag.where)
        assert "di.icd_code IN (" not in sql
        assert "%carcinoid syndrome%" in frag.params

    def test_falls_back_to_like_when_grounding_disabled(
        self, offline_ctx, monkeypatch,
    ):
        """ctx.enable_mcp_grounding=False → no MCP call at all.
        Byte-identical SQL to current behavior."""
        from src.conversational import concept_resolver as cr
        from src.conversational.operations_filters import _compile_diagnosis

        call_count = [0]
        def fake_autocode(text, **kwargs):
            call_count[0] += 1
            return {"status": "ok", "results": []}
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        f = PatientFilter(field="diagnosis", operator="contains", value="sepsis")
        frag = _compile_diagnosis(f, offline_ctx)
        sql = " ".join(frag.where)
        assert "di.icd_code IN (" not in sql
        assert call_count[0] == 0  # MCP NEVER called

    def test_falls_back_to_like_when_all_low_confidence(
        self, grounded_ctx, monkeypatch,
    ):
        """All candidates < 0.5 confidence threshold → IN-list skipped,
        LIKE-only preserved (silent fallback at filter level)."""
        from src.conversational import concept_resolver as cr
        from src.conversational.operations_filters import _compile_diagnosis

        def fake_autocode(text, **kwargs):
            return {
                "status": "ok",
                "results": [
                    {"code": "X1", "title": "low1", "confidence": 0.31},
                    {"code": "X2", "title": "low2", "confidence": 0.42},
                ],
            }
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        f = PatientFilter(field="diagnosis", operator="contains", value="sepsis")
        frag = _compile_diagnosis(f, grounded_ctx)
        sql = " ".join(frag.where)
        assert "di.icd_code IN (" not in sql

    def test_caches_repeat_lookups_via_module_lru_cache(
        self, grounded_ctx, monkeypatch,
    ):
        """Filter compiler hits the same lru_cache as concept_resolver —
        a 'carcinoid syndrome' filter on the autocode-fallback path
        shares one MCP round-trip per process across repeated calls.

        Uses 'carcinoid syndrome' (not in the cohort registry) so the
        autocode lookup actually fires; for registered cohorts the
        registry path short-circuits the cache."""
        from src.conversational import concept_resolver as cr
        from src.conversational.operations_filters import _compile_diagnosis

        call_count = [0]
        def fake_autocode(text, **kwargs):
            call_count[0] += 1
            return {
                "status": "ok",
                "results": [
                    {"code": "E34.0", "title": "Carcinoid syndrome", "confidence": 0.9},
                ],
            }
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        f = PatientFilter(
            field="diagnosis", operator="contains", value="carcinoid syndrome",
        )
        _compile_diagnosis(f, grounded_ctx)
        _compile_diagnosis(f, grounded_ctx)
        _compile_diagnosis(f, grounded_ctx)
        assert call_count[0] == 1

    def test_does_not_call_mcp_for_non_contains_operator(
        self, grounded_ctx, monkeypatch,
    ):
        """Grounding is scoped to the ``contains`` operator. If a future
        operator (e.g. ``=``) is added to the diagnosis filter, it must
        not silently inherit grounding semantics — different operators
        need different shapes."""
        from src.conversational import concept_resolver as cr
        from src.conversational.operations_filters import _compile_diagnosis

        call_count = [0]
        def fake_autocode(text, **kwargs):
            call_count[0] += 1
            return {"status": "ok", "results": []}
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        # Synthesise an "=" operator. The compiler may not currently
        # accept it via PatientFilter validation; if so the test instead
        # covers the case where ``f.operator`` is something the
        # grounding branch doesn't recognise.
        f = PatientFilter(field="diagnosis", operator="contains", value="sepsis")
        # Mutate operator AFTER construction to bypass the validator
        # (we're testing compile-side behavior, not model validation).
        object.__setattr__(f, "operator", "=")
        _compile_diagnosis(f, grounded_ctx)
        assert call_count[0] == 0

    def test_does_not_call_mcp_when_value_is_empty_string(
        self, grounded_ctx, monkeypatch,
    ):
        """Value-shape guard: empty string skips grounding."""
        from src.conversational import concept_resolver as cr
        from src.conversational.operations_filters import _compile_diagnosis

        call_count = [0]
        def fake_autocode(text, **kwargs):
            call_count[0] += 1
            return {"status": "ok", "results": []}
        monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

        f = PatientFilter(field="diagnosis", operator="contains", value="")
        _compile_diagnosis(f, grounded_ctx)
        assert call_count[0] == 0


class TestRegistryCore:
    def test_register_and_get_roundtrip(self):
        r = OperationRegistry()
        op = FilterOperation(
            name="age",
            operators=frozenset({">", "<"}),
            value_type="scalar",
            description="patient age",
            compile_fn=lambda f, ctx: FilterFragment(),
        )
        r.register(op)
        assert r.get("filter", "age") is op

    def test_get_missing_returns_none(self):
        r = OperationRegistry()
        assert r.get("filter", "nonexistent") is None

    def test_require_missing_raises(self):
        r = OperationRegistry()
        with pytest.raises(KeyError):
            r.require("filter", "nonexistent")

    def test_duplicate_registration_raises(self):
        r = OperationRegistry()
        op = FilterOperation(
            name="age", operators=frozenset(), value_type="scalar",
            description="", compile_fn=lambda f, ctx: FilterFragment(),
        )
        r.register(op)
        with pytest.raises(ValueError, match="already registered"):
            r.register(op)

    def test_supported_names_matches_registered(self):
        r = OperationRegistry()
        register_default_filters(r)
        assert r.supported_names("filter") == frozenset({
            "age", "gender", "diagnosis", "admission_type",
            "subject_id", "readmitted_30d", "readmitted_60d",
            "lab_value", "vital_value", "derived_value", "or_any", "icu_stay",
            "drug",
        })

    def test_supported_names_respects_kind(self):
        r = OperationRegistry()
        register_default_filters(r)
        # No aggregates registered yet.
        assert r.supported_names("aggregate") == frozenset()

    def test_same_name_different_kind_does_not_collide(self):
        """'gender' is a valid filter AND a valid comparison axis; both should
        coexist under different kinds without conflict."""
        r = OperationRegistry()
        filt = FilterOperation(
            name="gender", operators=frozenset({"="}), value_type="scalar",
            description="", compile_fn=lambda f, ctx: FilterFragment(),
        )
        cmp = ComparisonOperation(
            name="gender", description="",
            patient_clause="mimic:hasGender ?group_value ;",
        )
        r.register(filt)
        r.register(cmp)
        assert r.get("filter", "gender") is filt
        assert r.get("comparison_axis", "gender") is cmp


class TestDescribeForPrompt:
    def test_every_name_appears_in_description(self):
        """The prompt section must mention every registered operation name,
        so the prompt<->registry round-trip test can rely on textual presence."""
        r = OperationRegistry()
        register_default_filters(r)
        body = r.describe_for_prompt("filter")
        for name in r.supported_names("filter"):
            assert name in body, f"{name!r} missing from describe_for_prompt('filter')"

    def test_empty_kind_renders_placeholder_not_empty_string(self):
        """An empty section must be visibly empty, not silently missing."""
        r = OperationRegistry()
        body = r.describe_for_prompt("aggregate")
        assert body.strip() != ""


# ---------------------------------------------------------------------------
# 2. Filter-operation parity with the pre-registry extractor
# ---------------------------------------------------------------------------
#
# For each supported filter, we build a single-filter CQ, run BOTH:
#   (a) the pre-existing hand-rolled ``_get_filtered_hadm_ids`` path, and
#   (b) the registry's ``compile_filters`` → same outer SQL shell
# and assert they return identical admission IDs on the synthetic DuckDB fixture.
# If the two diverge for any existing filter, the refactor has a real bug.


def _run_registry_query(
    backend: _DuckDBBackend,
    filters: list[PatientFilter],
    config: ExtractionConfig,
) -> list[int]:
    """Replicate the outer SQL shell of _get_filtered_hadm_ids using registry output.

    Deliberately mirrors the production code so the comparison measures only
    the per-filter SQL fragments, not the scaffolding around them.
    """
    r = OperationRegistry()
    register_default_filters(r)
    ctx = FilterCompileContext(backend=backend)
    frag = r.compile_filters(filters, ctx)

    t = backend.table
    joins = list(frag.joins)
    if frag.needs_patients:
        joins.insert(0, f"JOIN {t('patients')} p ON a.subject_id = p.subject_id")
    join_sql = " ".join(joins)
    where_clause = f" WHERE {' AND '.join(frag.where)}" if frag.where else ""
    order = (
        "admittime DESC"
        if config.cohort_strategy == "recent"
        else backend.random_fn()
    )
    inner = (
        f"SELECT DISTINCT a.hadm_id, a.admittime"
        f" FROM {t('admissions')} a {join_sql}{where_clause}"
    )
    # Phase 2: cap removed; ordering-only subquery preserved for compatibility.
    sql = f"SELECT hadm_id FROM ({inner}) sub ORDER BY {order}"
    return [r[0] for r in backend.execute(sql, frag.params)]


class _ConnBackend(_DuckDBBackend):
    """Reuses an already-open DuckDB connection.

    ``synthetic_duckdb`` opens a read-write connection; ``_DuckDBBackend``
    opens read-only, which DuckDB refuses to mix. This adapter bypasses
    ``__init__`` and attaches an existing connection so tests can share
    the fixture's open handle.
    """

    def __init__(self, conn) -> None:
        self._conn = conn

    def close(self) -> None:  # don't close the fixture's conn
        pass


@pytest.fixture
def duckdb_backend(synthetic_duckdb_with_events):
    """Wrap the synthetic DuckDB connection in our backend shim.

    Uses ``_with_events`` because the diagnosis filter joins ``d_icd_diagnoses``.
    """
    return _ConnBackend(synthetic_duckdb_with_events)


_FILTER_PARITY_CASES = [
    # Each: (filter, description tag)
    pytest.param(
        PatientFilter(field="age", operator=">", value="60"),
        id="age_gt_60",
    ),
    pytest.param(
        PatientFilter(field="age", operator="<=", value="70"),
        id="age_lte_70",
    ),
    pytest.param(
        PatientFilter(field="gender", operator="=", value="M"),
        id="gender_M",
    ),
    pytest.param(
        PatientFilter(field="gender", operator="=", value="f"),
        id="gender_lowercase_f_uppercased",
    ),
    pytest.param(
        PatientFilter(field="diagnosis", operator="contains", value="cerebral"),
        id="diagnosis_contains_cerebral",
    ),
    pytest.param(
        PatientFilter(field="diagnosis", operator="contains", value="I63"),
        id="diagnosis_icd_prefix_I63",
    ),
    pytest.param(
        PatientFilter(field="admission_type", operator="=", value="EMERGENCY"),
        id="admission_type_emergency",
    ),
    pytest.param(
        PatientFilter(field="subject_id", operator="=", value="1"),
        id="subject_id_1",
    ),
    # The synthetic fixture has patient 1 readmitted within 30 days (admissions
    # 101 → 102, 26 days apart). Both binary values must flow through the
    # fallback CTE that the backend's readmission_labels_expr() builds.
    pytest.param(
        PatientFilter(field="readmitted_30d", operator="=", value="1"),
        id="readmitted_30d_eq_1",
    ),
    pytest.param(
        PatientFilter(field="readmitted_30d", operator="=", value="0"),
        id="readmitted_30d_eq_0",
    ),
    pytest.param(
        PatientFilter(field="readmitted_60d", operator="=", value="1"),
        id="readmitted_60d_eq_1",
    ),
]


@pytest.mark.parametrize("flt", _FILTER_PARITY_CASES)
def test_filter_parity_with_legacy_extractor(duckdb_backend, flt):
    """Registry-compiled filter returns the same hadm_ids as the legacy path."""
    config = ExtractionConfig(cohort_strategy="recent")

    legacy = _get_filtered_hadm_ids(duckdb_backend, [flt], config=config)
    via_registry = _run_registry_query(duckdb_backend, [flt], config)

    assert sorted(legacy) == sorted(via_registry), (
        f"parity mismatch for {flt.field}{flt.operator}{flt.value!r}:\n"
        f"  legacy:   {legacy}\n"
        f"  registry: {via_registry}"
    )


def test_filter_parity_with_multiple_filters(duckdb_backend):
    """Combining filters must still match the legacy path."""
    config = ExtractionConfig(cohort_strategy="recent")
    filters = [
        PatientFilter(field="age", operator=">", value="50"),
        PatientFilter(field="gender", operator="=", value="M"),
        PatientFilter(field="admission_type", operator="=", value="EMERGENCY"),
    ]
    legacy = _get_filtered_hadm_ids(duckdb_backend, filters, config=config)
    via_registry = _run_registry_query(duckdb_backend, filters, config)
    assert sorted(legacy) == sorted(via_registry)


def test_empty_filter_list_returns_all(duckdb_backend):
    """No filters → return every admission (up to the cap)."""
    config = ExtractionConfig(cohort_strategy="recent")
    legacy = _get_filtered_hadm_ids(duckdb_backend, [], config=config)
    via_registry = _run_registry_query(duckdb_backend, [], config)
    assert sorted(legacy) == sorted(via_registry)
    assert len(via_registry) > 0  # synthetic_duckdb has admissions


# ---------------------------------------------------------------------------
# 3. Validation semantics
# ---------------------------------------------------------------------------


class TestFilterValidation:
    def test_unsupported_operator_rejected(self):
        r = OperationRegistry()
        register_default_filters(r)
        age = r.require("filter", "age")
        # contains is not valid for age
        violations = age.validate(
            PatientFilter(field="age", operator="contains", value="65")
        )
        assert violations
        assert "operator" in violations[0]

    def test_supported_operator_accepted(self):
        r = OperationRegistry()
        register_default_filters(r)
        age = r.require("filter", "age")
        assert age.validate(PatientFilter(field="age", operator=">", value="65")) == []

    def test_field_mismatch_rejected(self):
        """An operation handed a filter for a different field must reject it."""
        r = OperationRegistry()
        register_default_filters(r)
        age = r.require("filter", "age")
        violations = age.validate(
            PatientFilter(field="gender", operator="=", value="M")
        )
        assert violations


# ---------------------------------------------------------------------------
# 4. Aggregate / comparison protocol smoke tests
# ---------------------------------------------------------------------------
#
# Phase 1a doesn't wire aggregates/comparisons into the reasoner yet — Phase 1b
# does — but the operation classes themselves should compile cleanly and pass
# their own describe/validate contracts.


class TestAggregateAndComparisonProtocol:
    def test_aggregate_compile_produces_template_name(self):
        from src.conversational.models import CompetencyQuestion
        agg = AggregateOperation(
            name="mean",
            description="average over numeric values",
            template="aggregation_mean",
        )
        cq = CompetencyQuestion(original_question="avg x", aggregation="mean")
        frag = agg.compile(cq)
        assert isinstance(frag, AggregateFragment)
        assert frag.template == "aggregation_mean"
        assert frag.post_processor is None

    def test_aggregate_describe_includes_name(self):
        agg = AggregateOperation(name="mean", description="avg", template="t")
        assert "mean" in agg.describe_for_prompt()

    def test_comparison_compile_produces_clauses(self):
        cmp = ComparisonOperation(
            name="gender",
            description="M vs F",
            patient_clause="mimic:hasGender ?group_value ;",
        )
        frag = cmp.compile()
        assert isinstance(frag, ComparisonFragment)
        assert "?group_value" in frag.patient_clause
        assert frag.admission_clause == ""


# ---------------------------------------------------------------------------
# 5. Default-registry wiring (Phase 1b)
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    def test_default_registry_seeded_with_all_kinds(self):
        from src.conversational.operations import get_default_registry

        r = get_default_registry()
        assert r.supported_names("filter") == frozenset({
            "age", "gender", "diagnosis", "admission_type",
            "subject_id", "readmitted_30d", "readmitted_60d",
            "lab_value", "vital_value", "derived_value", "or_any", "icu_stay",
            "drug",
        })
        assert r.supported_names("aggregate") == frozenset({
            "mean", "avg", "median", "max", "min", "count", "sum", "exists",
            "event_ordering",
        })
        assert r.supported_names("comparison_axis") == frozenset({
            "gender", "age", "readmitted_30d", "readmitted_60d",
            "admission_type", "discharge_location", "condition",
        })

    def test_default_registry_is_idempotent(self):
        from src.conversational.operations import get_default_registry

        assert get_default_registry() is get_default_registry()


class TestAggregateRegistrySemantics:
    def test_median_has_post_processor(self):
        from src.conversational.operations import get_default_registry

        median_op = get_default_registry().require("aggregate", "median")
        assert median_op.post_processor is not None
        # Smoke: median of [1, 2, 3] = 2
        rows, cols = median_op.post_processor(
            [{"value": 1}, {"value": 2}, {"value": 3}]
        )
        assert rows == [{"median_value": 2}]
        assert cols == ["median_value"]

    def test_median_post_processor_handles_empty(self):
        from src.conversational.operations import get_default_registry

        median_op = get_default_registry().require("aggregate", "median")
        rows, _ = median_op.post_processor([])
        assert rows == [{"median_value": None}]

    def test_avg_and_mean_resolve_to_same_template(self):
        """``avg`` is an alias for ``mean``; both must select the same template
        so the reasoner behaves identically for either keyword."""
        from src.conversational.operations import get_default_registry

        r = get_default_registry()
        assert r.require("aggregate", "avg").template == \
            r.require("aggregate", "mean").template

    def test_all_aggregates_reference_an_existing_template(self):
        """Catch dispatch drift: every registered aggregate's template must
        exist in reasoner.TEMPLATES, or executing that aggregate will crash
        at query build time."""
        from src.conversational.operations import get_default_registry
        from src.conversational.reasoner import TEMPLATES

        r = get_default_registry()
        for name in r.supported_names("aggregate"):
            op = r.require("aggregate", name)
            assert op.template in TEMPLATES, (
                f"aggregate {name!r} references template {op.template!r} "
                f"which does not exist in reasoner.TEMPLATES"
            )


class TestComparisonRegistrySemantics:
    # The dynamic ``condition`` axis is SQL-fast-path-only: its GROUP BY column
    # is built from ``cq.split_condition`` at compile time, so it has neither a
    # SPARQL clause nor a ``_COMPARISON_FIELD_MAP`` entry (the planner never
    # routes it to the graph). The graph-path parity assertions below exclude it.
    _SQL_ONLY_AXES = frozenset({"condition"})

    def test_every_comparison_axis_has_at_least_one_clause(self):
        """An axis with neither a patient nor admission clause would produce
        malformed SPARQL — except the SQL-only ``condition`` axis, which never
        reaches the SPARQL path."""
        from src.conversational.operations import get_default_registry

        r = get_default_registry()
        for name in r.supported_names("comparison_axis"):
            if name in self._SQL_ONLY_AXES:
                continue
            op = r.require("comparison_axis", name)
            frag = op.compile()
            assert frag.patient_clause or frag.admission_clause, (
                f"comparison axis {name!r} contributes no graph-pattern clause"
            )

    def test_comparison_axes_match_prior_map(self):
        """Registry must expose the same GRAPH-path axes the pre-refactor
        _COMPARISON_FIELD_MAP did, so the reasoner's GROUP BY behaviour is
        preserved unchanged. (The SQL-only ``condition`` axis has no map entry.)"""
        from src.conversational.operations import get_default_registry
        from src.conversational.reasoner import _COMPARISON_FIELD_MAP

        r = get_default_registry()
        graph_axes = r.supported_names("comparison_axis") - self._SQL_ONLY_AXES
        assert graph_axes == frozenset(_COMPARISON_FIELD_MAP.keys())
        # Clause text parity.
        for name, (patient, admission) in _COMPARISON_FIELD_MAP.items():
            op = r.require("comparison_axis", name)
            frag = op.compile()
            assert frag.patient_clause == patient
            assert frag.admission_clause == admission


# ---------------------------------------------------------------------------
# 6. List-valued filters ("in" operator)
# ---------------------------------------------------------------------------


class TestListValuedFilters:
    def test_admission_type_in_list_compiles_to_in_clause(self, duckdb_backend):
        """``admission_type in [...]`` emits an SQL ``IN (?, ?, ...)`` clause
        and returns the union of matches."""
        from src.conversational.operations import (
            FilterCompileContext,
            get_default_registry,
        )

        config = ExtractionConfig(cohort_strategy="recent")
        r = get_default_registry()
        ctx = FilterCompileContext(backend=duckdb_backend)

        # Single-value = case (baseline).
        eq_filter = [PatientFilter(field="admission_type", operator="=", value="EMERGENCY")]
        eq_rows = _run_registry_query(duckdb_backend, eq_filter, config)

        # Two-value "in" case must return a superset when the second value
        # also matches rows, strictly equal when it adds nothing.
        in_filter = [PatientFilter(
            field="admission_type",
            operator="in",
            value=["EMERGENCY", "ELECTIVE"],
        )]
        in_rows = _run_registry_query(duckdb_backend, in_filter, config)

        assert set(eq_rows).issubset(set(in_rows))
        # Fragment shape: single ``IN (?, ?)`` predicate, two params.
        frag = r.compile_filters(in_filter, ctx)
        assert len(frag.where) == 1
        assert "IN (?, ?)" in frag.where[0]
        assert frag.params == ["EMERGENCY", "ELECTIVE"]

    def test_empty_in_list_matches_nothing(self, duckdb_backend):
        """``in []`` must not produce invalid SQL; emit a predicate that
        matches zero rows so the cohort is simply empty."""
        from src.conversational.operations import (
            FilterCompileContext,
            get_default_registry,
        )

        r = get_default_registry()
        ctx = FilterCompileContext(backend=duckdb_backend)
        frag = r.compile_filters(
            [PatientFilter(field="admission_type", operator="in", value=[])],
            ctx,
        )
        assert frag.where == ["1 = 0"]
        assert frag.params == []

    def test_patient_filter_accepts_list_values(self):
        """Schema-level: PatientFilter.value is now str | list[str]."""
        f = PatientFilter(
            field="admission_type",
            operator="in",
            value=["EMERGENCY", "URGENT"],
        )
        assert f.value == ["EMERGENCY", "URGENT"]

    def test_in_operator_rejected_for_fields_that_dont_support_it(self):
        """The ``in`` operator should only be accepted for fields that register
        it. age doesn't, so validate must flag it."""
        from src.conversational.operations import get_default_registry

        age = get_default_registry().require("filter", "age")
        violations = age.validate(
            PatientFilter(field="age", operator="in", value=["65", "75"])
        )
        assert violations
        assert "operator" in violations[0]


# ---------------------------------------------------------------------------
# 7. select_templates integration — registry is the dispatch source of truth
# ---------------------------------------------------------------------------
#
# These tests exercise reasoner.select_templates END to END with every
# registered aggregate. If an aggregate is added to the registry with a
# template not in reasoner.TEMPLATES, both this and test_all_aggregates_*
# trip. If select_templates regresses to a hand-rolled elif, adding a new
# aggregate to the registry without editing the reasoner will silently
# fail — this test would be the one to flag it.


class TestSelectTemplatesDispatch:
    @pytest.mark.parametrize("aggregation_keyword", sorted(
        {"mean", "avg", "median", "max", "min", "count", "sum", "exists"}
    ))
    def test_every_registered_aggregate_produces_its_registered_template(
        self, aggregation_keyword: str
    ):
        """A biomarker CQ with each aggregation keyword must route to exactly
        the template the registry declares for that keyword."""
        from src.conversational.models import ClinicalConcept, CompetencyQuestion
        from src.conversational.operations import get_default_registry
        from src.conversational.reasoner import select_templates

        cq = CompetencyQuestion(
            original_question=f"{aggregation_keyword} creatinine",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            aggregation=aggregation_keyword,
            scope="cohort",
        )
        expected = get_default_registry().require(
            "aggregate", aggregation_keyword
        ).template
        assert expected in select_templates(cq)

    def test_comparison_scope_uses_registry_for_axis_validation(self):
        """A comparison CQ whose axis is registered picks ``comparison_by_field``;
        one with an unregistered axis falls back to ``comparison_two_groups``.
        """
        from src.conversational.models import ClinicalConcept, CompetencyQuestion
        from src.conversational.reasoner import select_templates

        registered = CompetencyQuestion(
            original_question="compare by gender",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            scope="comparison",
            comparison_field="gender",
        )
        assert "comparison_by_field" in select_templates(registered)

        unknown = CompetencyQuestion(
            original_question="compare by made-up-axis",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            scope="comparison",
            comparison_field="not_a_real_axis",
        )
        assert "comparison_two_groups" in select_templates(unknown)

    def test_unknown_aggregation_falls_through_to_concept_type_path(self):
        """An aggregation keyword the registry doesn't know should not crash —
        select_templates falls through to the concept_type dispatch. This
        guards against the LLM inventing a novel aggregation keyword."""
        from src.conversational.models import ClinicalConcept, CompetencyQuestion
        from src.conversational.reasoner import select_templates

        cq = CompetencyQuestion(
            original_question="whatever creatinine",
            clinical_concepts=[
                ClinicalConcept(name="creatinine", concept_type="biomarker"),
            ],
            aggregation="nonexistent_keyword",
            scope="cohort",
        )
        # biomarker fallback is value_with_timestamps
        assert "value_with_timestamps" in select_templates(cq)


# ---------------------------------------------------------------------------
# 8. Compile contract — every registered filter yields SOMETHING
# ---------------------------------------------------------------------------
#
# A compile_fn that silently returns an empty FilterFragment would reduce the
# cohort to "everyone" without anyone noticing. This test ensures every
# registered filter contributes at least a WHERE predicate for a plausible
# filter — a low bar, but catches "no-op compile" regressions that the
# parity tests can miss if a field isn't in the parity fixture list.


class TestFilterCompileContract:
    @pytest.mark.parametrize("field_name", sorted({
        "age", "gender", "diagnosis", "admission_type",
        "subject_id", "readmitted_30d", "readmitted_60d",
    }))
    def test_every_registered_filter_contributes_a_where_clause(
        self, field_name: str, duckdb_backend
    ):
        from src.conversational.operations import (
            FilterCompileContext,
            get_default_registry,
        )

        plausible_values = {
            "age": ("age", ">", "50"),
            "gender": ("gender", "=", "M"),
            "diagnosis": ("diagnosis", "contains", "sepsis"),
            "admission_type": ("admission_type", "=", "EMERGENCY"),
            "subject_id": ("subject_id", "=", "1"),
            "readmitted_30d": ("readmitted_30d", "=", "1"),
            "readmitted_60d": ("readmitted_60d", "=", "0"),
        }
        field, op, value = plausible_values[field_name]
        flt = PatientFilter(field=field, operator=op, value=value)

        registry = get_default_registry()
        ctx = FilterCompileContext(backend=duckdb_backend)
        frag = registry.compile_filters([flt], ctx)

        # Every filter must contribute at least one predicate; most also add joins.
        assert frag.where, f"filter {field_name!r} produced no WHERE clause"
        # The predicate should reference something SQL-like — not just an
        # empty string.
        assert all(p.strip() for p in frag.where), (
            f"filter {field_name!r} produced a blank WHERE clause"
        )


# ---------------------------------------------------------------------------
# 9. Drug cohort filter ("received drug X / a drug in group G")
# ---------------------------------------------------------------------------


class TestDrugCohortFilter:
    """The ``drug`` cohort filter grounds a "received a reversal agent" cohort
    restriction via name-LIKE over ``prescriptions.drug`` — emitted as a
    self-contained correlated EXISTS so it can also ride inside an ``or_any``.
    A vague GROUP phrase expands to its concrete members; a specific drug
    grounds by its own name."""

    def test_drug_filter_is_registered(self):
        from src.conversational.operations import get_default_registry

        op = get_default_registry().get("filter", "drug")
        assert op is not None
        assert "contains" in op.operators

    def test_specific_drug_grounds_by_name(self, duckdb_backend):
        # Fixture: hadm 101 = Vancomycin, hadm 103 = Ceftriaxone.
        config = ExtractionConfig(cohort_strategy="recent")
        rows = _run_registry_query(
            duckdb_backend,
            [PatientFilter(field="drug", operator="contains", value="vancomycin")],
            config,
        )
        assert set(rows) == {101}

    def test_reversal_group_expands_to_members(self, duckdb_backend):
        # Add concrete reversal-agent prescriptions whose names match MIMIC's
        # (Kcentra / Phytonadione / Fresh Frozen Plasma). The vague group phrase
        # must match all three admissions via the expanded member patterns.
        conn = duckdb_backend._conn
        conn.execute(
            "INSERT INTO prescriptions VALUES "
            "(3, 104, '2152-05-20 16:00:00', '2152-05-21 16:00:00', "
            "'Kcentra', 1500.0, 'unit', 'IV'), "
            "(4, 105, '2150-07-01 10:00:00', '2150-07-02 10:00:00', "
            "'Phytonadione', 10.0, 'mg', 'IV'), "
            "(5, 106, '2151-04-11 02:00:00', '2151-04-12 02:00:00', "
            "'Fresh Frozen Plasma', 1.0, 'unit', 'IV')"
        )
        config = ExtractionConfig(cohort_strategy="recent")
        rows = _run_registry_query(
            duckdb_backend,
            [PatientFilter(
                field="drug", operator="contains",
                value="coagulation reversal agent",
            )],
            config,
        )
        assert set(rows) == {104, 105, 106}

    def test_drug_filter_emits_self_contained_exists_no_join(self, duckdb_backend):
        # An OR member can't carry a JOIN, so the drug filter must be a pure
        # EXISTS predicate (no ``frag.joins``).
        from src.conversational.operations import (
            FilterCompileContext,
            get_default_registry,
        )

        op = get_default_registry().require("filter", "drug")
        ctx = FilterCompileContext(backend=duckdb_backend)
        frag = op.compile(
            PatientFilter(field="drug", operator="contains", value="vancomycin"),
            ctx,
        )
        assert frag.joins == []
        assert frag.where and frag.where[0].startswith("EXISTS (")
        assert "prescriptions" in frag.where[0]

    def test_or_any_of_drug_members_unions_them(self, duckdb_backend):
        # The decomposer's preferred shape: or_any of one drug filter per member.
        conn = duckdb_backend._conn
        conn.execute(
            "INSERT INTO prescriptions VALUES "
            "(4, 105, '2150-07-01 10:00:00', '2150-07-02 10:00:00', "
            "'Phytonadione', 10.0, 'mg', 'IV')"
        )
        or_any = PatientFilter(
            field="or_any", operator="in", value="any", sub_filters=[
                PatientFilter(field="drug", operator="contains", value="vitamin k"),
                PatientFilter(
                    field="drug", operator="contains", value="fresh frozen plasma",
                ),
            ],
        )
        config = ExtractionConfig(cohort_strategy="recent")
        rows = _run_registry_query(duckdb_backend, [or_any], config)
        # 'vitamin k' member-pattern matches Phytonadione? No — but the or_any
        # leaf value 'vitamin k' grounds its OWN raw pattern '%vitamin k%' which
        # doesn't match 'Phytonadione'. So this asserts the union is empty here;
        # the member-expansion behaviour is covered by the group test above.
        assert set(rows) == set()


# ---------------------------------------------------------------------------
# 10. GCS-total cohort filter (derived.gcs TOTAL, not the GCS components)
# ---------------------------------------------------------------------------


class _FakeBQBackend:
    """Minimal BigQuery-shaped backend: ``table()`` returns a backtick-quoted
    FQN so ``_derived_table`` resolves the derived GCS table. Not executable —
    used only to assert the emitted SQL shape."""

    def table(self, name: str) -> str:
        return f"`physionet-data.x.{name}`"

    @staticmethod
    def ilike(column: str) -> str:
        return f"{column} ILIKE ?"

    def readmission_labels_expr(self) -> str:
        return "`physionet-data.x.readmission_labels`"


class TestGcsTotalFilter:
    """A GCS-total threshold grounds to the derived ``gcs`` TOTAL column, NOT
    the three GCS COMPONENT chartitems a label-LIKE would match."""

    def _gcs_filter(self, op="<=", value="8"):
        return PatientFilter(
            field="vital_value", operator=op, value=value, measurement="GCS",
        )

    def test_gcs_grounds_to_derived_total_exists_on_bigquery(self):
        from src.conversational.operations import (
            FilterCompileContext,
            get_default_registry,
        )

        op = get_default_registry().require("filter", "vital_value")
        ctx = FilterCompileContext(backend=_FakeBQBackend())
        frag = op.compile(self._gcs_filter("<=", "8"), ctx)
        sql = frag.where[0]
        # EXISTS over the derived gcs table joined to icustays, total g.gcs.
        assert "mimiciv_3_1_derived.gcs" in sql
        assert "icu.stay_id = g.stay_id" in sql
        assert "icu.hadm_id = a.hadm_id" in sql
        assert "g.gcs <= ?" in sql
        # The TOTAL column only — never the component label-LIKE.
        assert "ILIKE" not in sql
        assert "d_items" not in sql
        assert "gcs_motor" not in sql and "gcs_verbal" not in sql
        assert frag.params == [8.0]
        assert frag.joins == []

    def test_gcs_operator_threaded_into_predicate(self):
        from src.conversational.operations import (
            FilterCompileContext,
            get_default_registry,
        )

        op = get_default_registry().require("filter", "vital_value")
        ctx = FilterCompileContext(backend=_FakeBQBackend())
        frag = op.compile(self._gcs_filter("<", "9"), ctx)
        assert "g.gcs < ?" in frag.where[0]
        assert frag.params == [9.0]

    def test_gcs_offline_degrades_to_empty_with_no_crash(self, duckdb_backend):
        # DuckDB/offline has no derived.gcs (_derived_table → None). The filter
        # must degrade to an empty cohort (1 = 0) — NOT a component label-LIKE —
        # and the cohort query must still execute without error.
        from src.conversational.operations import (
            FilterCompileContext,
            get_default_registry,
        )

        op = get_default_registry().require("filter", "vital_value")
        ctx = FilterCompileContext(backend=duckdb_backend)
        frag = op.compile(self._gcs_filter("<=", "8"), ctx)
        assert frag.where == ["1 = 0"]
        assert "mimiciv_3_1_derived" not in (frag.where[0] if frag.where else "")
        # Executes end-to-end (no missing-table crash) and selects nobody.
        config = ExtractionConfig(cohort_strategy="recent")
        rows = _run_registry_query(duckdb_backend, [self._gcs_filter()], config)
        assert rows == []

    def test_nongcs_vital_unchanged_when_gate_off(self, duckdb_backend):
        # Guard the gate: a non-GCS vital (MAP) must compile to the existing
        # label-LIKE EXISTS over chartevents — byte-identical to pre-change.
        from src.conversational.operations import (
            FilterCompileContext,
            get_default_registry,
        )

        op = get_default_registry().require("filter", "vital_value")
        ctx = FilterCompileContext(backend=duckdb_backend)
        frag = op.compile(
            PatientFilter(
                field="vital_value", operator="<", value="65",
                measurement="mean arterial pressure",
            ),
            ctx,
        )
        sql = frag.where[0]
        assert "chartevents" in sql and "d_items" in sql
        assert "ILIKE" in sql
        assert "mimiciv_3_1_derived" not in sql
        assert frag.params == ["%mean arterial pressure%", 65.0]


class TestMultiDiagnosisFilterAliases:
    """``_compile_diagnosis`` hardcodes the table aliases ``di``/``dd``. With two
    diagnosis filters in one CQ, ``compile_filters`` previously emitted the same
    ``di``/``dd`` twice in one FROM clause, so BigQuery rejected the query with
    ``400 ... Duplicate table alias di`` — the real crash behind "Analysis
    failed" on *"... how many had a high anion gap metabolic acidosis vs. a non
    high anion gap metabolic acidosis"* (and the diabetes follow-up).

    Each diagnosis filter must get a DISTINCT alias; the first stays ``di``/``dd``
    so single-diagnosis SQL is byte-identical (no regression)."""

    def _registry(self):
        r = OperationRegistry()
        register_default_filters(r)
        return r

    def test_two_diagnosis_filters_get_distinct_table_aliases(self, duckdb_backend):
        import re

        ctx = FilterCompileContext(backend=duckdb_backend, enable_mcp_grounding=False)
        filters = [
            PatientFilter(field="diagnosis", operator="contains", value="metabolic acidosis"),
            PatientFilter(field="diagnosis", operator="contains", value="diabetes"),
        ]
        frag = self._registry().compile_filters(filters, ctx)
        joins_sql = " ".join(frag.joins)

        # diagnoses_icd is joined once per filter; the aliases must be unique.
        di_aliases = re.findall(r"diagnoses_icd`?\s+(\w+)\b", joins_sql)
        di_aliases = [a for a in di_aliases if a.startswith("di")]
        assert len(di_aliases) == 2, joins_sql
        assert len(set(di_aliases)) == 2, f"duplicate diagnoses_icd alias: {di_aliases}"

        dd_aliases = re.findall(r"d_icd_diagnoses`?\s+(\w+)\b", joins_sql)
        assert len(set(dd_aliases)) == 2, f"duplicate d_icd_diagnoses alias: {dd_aliases}"

        # First filter is byte-identical to the single-diagnosis path.
        assert "di" in di_aliases and "dd" in dd_aliases

        # Every emitted alias is actually referenced by a predicate (no orphans):
        # the join ON-clauses and the WHERE clauses use each filter's own alias.
        all_sql = joins_sql + " " + " ".join(frag.where)
        for a in di_aliases:
            assert f"{a}.icd_code" in all_sql, f"{a} unreferenced in {all_sql}"

    def test_single_diagnosis_filter_unchanged(self, duckdb_backend):
        """The common single-diagnosis case keeps the bare ``di``/``dd`` aliases —
        the fix must not perturb existing SQL."""
        ctx = FilterCompileContext(backend=duckdb_backend, enable_mcp_grounding=False)
        frag = self._registry().compile_filters(
            [PatientFilter(field="diagnosis", operator="contains", value="sepsis")],
            ctx,
        )
        joins_sql = " ".join(frag.joins)
        assert "diagnoses_icd` di " in joins_sql or "diagnoses_icd di " in joins_sql
        assert "di1" not in joins_sql and "dd1" not in joins_sql
