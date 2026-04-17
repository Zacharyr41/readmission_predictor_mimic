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
    sql = (
        f"SELECT hadm_id FROM ({inner}) sub"
        f" ORDER BY {order} LIMIT {config.max_cohort_size}"
    )
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
]


@pytest.mark.parametrize("flt", _FILTER_PARITY_CASES)
def test_filter_parity_with_legacy_extractor(duckdb_backend, flt):
    """Registry-compiled filter returns the same hadm_ids as the legacy path."""
    config = ExtractionConfig(max_cohort_size=100, cohort_strategy="recent")

    legacy = _get_filtered_hadm_ids(duckdb_backend, [flt], config=config)
    via_registry = _run_registry_query(duckdb_backend, [flt], config)

    assert sorted(legacy) == sorted(via_registry), (
        f"parity mismatch for {flt.field}{flt.operator}{flt.value!r}:\n"
        f"  legacy:   {legacy}\n"
        f"  registry: {via_registry}"
    )


def test_filter_parity_with_multiple_filters(duckdb_backend):
    """Combining filters must still match the legacy path."""
    config = ExtractionConfig(max_cohort_size=100, cohort_strategy="recent")
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
    config = ExtractionConfig(max_cohort_size=100, cohort_strategy="recent")
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
