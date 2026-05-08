"""Tier 3 — SQL emission regression tests.

Fast (<1s/test), no LLM, no MCP network. Constructs CompetencyQuestions
directly, monkeypatches ``concept_resolver.icd_autocode`` /
``mimic_itemid_search`` for the grounded paths, and asserts the SQL
shape that ``compile_sql()`` emits.

These are the regression guards for Inc 4 (diagnosis-count IN-list),
Inc 7 (biomarker MCP fallback), and Inc 9 (filter-side IN-list).
Designed to fire instantly if any of those wirings are accidentally
unthreaded.
"""

from __future__ import annotations

import duckdb
import pytest

from src.conversational import concept_resolver as cr
from src.conversational.models import (
    ClinicalConcept, CompetencyQuestion, PatientFilter,
)
from src.conversational.operations import get_default_registry
from src.conversational.sql_fastpath import compile_sql

from tests.dashboard.lib.scenarios import (
    MOCK_MIMIC_ITEMID_LACTATE_RESPONSE,
    MOCK_OMOPHUB_SEPSIS_RESPONSE,
    SEPSIS_FAMILY_PREFIXES,
)
from tests.test_conversational.test_sql_fastpath import _ConnBackend


@pytest.fixture
def backend():
    """Minimal DuckDB-backed backend shim — only used for SQL emission
    inspection (we never execute the SQL). In-memory connection is fine."""
    return _ConnBackend(duckdb.connect(":memory:"))


def _record_sql(reporter, query, name: str) -> None:
    """Stash the emitted SQL onto the reporter for the markdown artifact."""
    reporter.add_question(name)
    reporter.add_sql(query.sql, query.params)


def test_diagnosis_count_path_emits_in_list_parallel_or(backend, reporter):
    """Inc 4 regression: diagnosis-typed CQ with grounded ICD codes
    emits ``((di.icd_code IN (?, ?)) OR (<existing LIKE>))`` so ICD-10
    grounded admissions match precisely while ICD-9 admissions stay
    matchable via the LIKE branch."""
    cq = CompetencyQuestion(
        original_question="how many sepsis patients?",
        clinical_concepts=[
            ClinicalConcept(name="sepsis", concept_type="diagnosis"),
        ],
        aggregation="count", scope="cohort",
    )
    query = compile_sql(
        cq, backend, get_default_registry(),
        resolved_names=["sepsis"],
        resolved_icd_codes=["A41.9", "R65.21"],
    )
    _record_sql(reporter, query, "How many sepsis patients? (Inc 4)")

    in_clause_present = "di.icd_code IN (" in query.sql
    reporter.add_assertion(
        "SQL contains 'di.icd_code IN (' parallel-OR clause",
        in_clause_present,
        detail=f"sql: {query.sql[:200]!r}",
    )
    assert in_clause_present

    a41_in_params = "A41.9" in query.params
    reporter.add_assertion(
        "Params include grounded sepsis code 'A41.9'",
        a41_in_params,
        detail=f"params: {query.params!r}",
    )
    assert a41_in_params


def test_filter_side_emits_in_list_for_lactate_in_sepsis(
    backend, monkeypatch, reporter,
):
    """Inc 9 smoking-gun regression. The original failing query
    decomposes to a biomarker-aggregate CQ with diagnosis as a
    patient_filter. With ``enable_mcp_grounding=True``, the diagnosis
    filter compiler must call icd_autocode and emit a parallel-OR
    IN-list — the same pattern as Inc 4 but on the filter side."""
    monkeypatch.setattr(
        cr, "icd_autocode",
        lambda *a, **kw: MOCK_OMOPHUB_SEPSIS_RESPONSE,
        raising=False,
    )

    cq = CompetencyQuestion(
        original_question="What is the mean lactate in our sepsis cohort?",
        clinical_concepts=[
            ClinicalConcept(name="lactate", concept_type="biomarker"),
        ],
        patient_filters=[
            PatientFilter(field="diagnosis", operator="contains", value="sepsis"),
        ],
        aggregation="mean", scope="cohort",
    )
    query = compile_sql(
        cq, backend, get_default_registry(),
        resolved_names=["lactate"],
        enable_mcp_grounding=True,
    )
    _record_sql(reporter, query, "Mean lactate in sepsis cohort (Inc 9 smoking-gun)")

    in_present = "di.icd_code IN (" in query.sql
    reporter.add_assertion(
        "Filter side emits 'di.icd_code IN (' (Inc 9 wiring)",
        in_present,
    )
    assert in_present

    has_sepsis_code = any(
        isinstance(p, str) and p.startswith(SEPSIS_FAMILY_PREFIXES)
        for p in query.params
    )
    reporter.add_assertion(
        "Params contain sepsis-family ICD code (A41/R65/A40/A42)",
        has_sepsis_code,
        detail=f"params: {query.params!r}",
    )
    assert has_sepsis_code


def test_filter_falls_back_to_like_only_when_omophub_unavailable(
    backend, monkeypatch, reporter,
):
    """Graceful degradation: when OMOPHub returns ``unavailable``, the
    filter compiler skips the IN-list and emits LIKE-only — same as the
    pre-Inc-9 behavior. The pipeline must keep answering even when the
    MCP is unreachable."""
    monkeypatch.setattr(
        cr, "icd_autocode",
        lambda *a, **kw: {"status": "unavailable", "error": "MCP timeout"},
        raising=False,
    )

    cq = CompetencyQuestion(
        original_question="Mean lactate in sepsis (MCP down)",
        clinical_concepts=[
            ClinicalConcept(name="lactate", concept_type="biomarker"),
        ],
        patient_filters=[
            PatientFilter(field="diagnosis", operator="contains", value="sepsis"),
        ],
        aggregation="mean", scope="cohort",
    )
    query = compile_sql(
        cq, backend, get_default_registry(),
        resolved_names=["lactate"],
        enable_mcp_grounding=True,
    )
    _record_sql(reporter, query, "Mean lactate in sepsis (OMOPHub unavailable — graceful degradation)")

    no_in_clause = "di.icd_code IN (" not in query.sql
    reporter.add_assertion(
        "SQL emits LIKE-only (no IN-list) when OMOPHub unavailable",
        no_in_clause,
    )
    assert no_in_clause

    has_like_param = "%sepsis%" in query.params
    reporter.add_assertion(
        "LIKE branch params still present (back-compat)",
        has_like_param,
    )
    assert has_like_param


def test_biomarker_local_loinc_grounding_unchanged(
    backend, monkeypatch, reporter,
):
    """Back-compat: biomarker CQ with a LOINC code that's in the local
    index should resolve via the local mapping — ``mimic_itemid_search``
    must NOT fire. This catches if Inc 7 accidentally bypassed the
    fast-path local lookup."""
    call_count = [0]
    def fake_search(*a, **kw):
        call_count[0] += 1
        return {"status": "ok", "results": []}
    monkeypatch.setattr(cr, "mimic_itemid_search", fake_search, raising=False)

    # Use creatinine (LOINC 2160-0 is in the local mapping).
    cq = CompetencyQuestion(
        original_question="mean creatinine",
        clinical_concepts=[
            ClinicalConcept(
                name="creatinine", concept_type="biomarker",
                loinc_code="2160-0",
            ),
        ],
        aggregation="mean", scope="cohort",
    )
    # Resolve via the live resolver so we exercise the actual code path.
    resolver = cr.ConceptResolver(
        mappings_dir=__import__("pathlib").Path("data/mappings"),
        enable_mcp_grounding=True,
    )
    biom = resolver.resolve_biomarker(cq.clinical_concepts[0])
    query = compile_sql(
        cq, backend, get_default_registry(),
        resolved_names=["creatinine"],
        resolved_itemids=biom.itemids,
        enable_mcp_grounding=True,
    )
    _record_sql(reporter, query, "Mean creatinine (LOINC in local index — fast path)")

    has_itemid_in = "itemid IN (" in query.sql
    reporter.add_assertion(
        "Local-LOINC path emits 'itemid IN (...)' (no MCP needed)",
        has_itemid_in,
    )
    assert has_itemid_in

    no_mcp_call = call_count[0] == 0
    reporter.add_assertion(
        "mimic_itemid_search NOT called (fast-path preserved)",
        no_mcp_call,
        detail=f"call_count={call_count[0]}",
    )
    assert no_mcp_call


def test_biomarker_mimic_itemid_search_fallback_when_loinc_misses(
    backend, monkeypatch, reporter,
):
    """Inc 7 regression. When the LOINC isn't in the local index AND
    grounding is enabled, the resolver falls back to
    ``mimic_itemid_search`` and recovers labevents itemids."""
    monkeypatch.setattr(
        cr, "mimic_itemid_search",
        lambda *a, **kw: MOCK_MIMIC_ITEMID_LACTATE_RESPONSE,
        raising=False,
    )

    resolver = cr.ConceptResolver(
        mappings_dir=__import__("pathlib").Path("data/mappings"),
        enable_mcp_grounding=True,
    )
    concept = ClinicalConcept(
        name="lactate", concept_type="biomarker",
        loinc_code="99999-9",  # not in local index
    )
    biom = resolver.resolve_biomarker(concept)

    cq = CompetencyQuestion(
        original_question="mean lactate (unknown LOINC)",
        clinical_concepts=[concept],
        aggregation="mean", scope="cohort",
    )
    query = compile_sql(
        cq, backend, get_default_registry(),
        resolved_names=["lactate"],
        resolved_itemids=biom.itemids,
        enable_mcp_grounding=True,
    )
    _record_sql(reporter, query, "Mean lactate (LOINC unknown — Inc 7 fallback)")

    grounded = biom.itemids is not None and 50813 in biom.itemids
    reporter.add_assertion(
        "Resolver recovered grounded itemids via mimic_itemid_search",
        grounded,
        detail=f"itemids={biom.itemids!r}",
    )
    assert grounded

    has_itemid_in = "itemid IN (" in query.sql
    reporter.add_assertion(
        "SQL emits 'itemid IN (...)' for MCP-grounded itemids",
        has_itemid_in,
    )
    assert has_itemid_in
