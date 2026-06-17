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
from src.conversational.sql_render import render_sql_with_params

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

    in_clause_present = "di.icd_code LIKE ?" in query.sql
    reporter.add_assertion(
        "SQL contains 'di.icd_code LIKE ?' prefix-match parallel-OR clause",
        in_clause_present,
        detail=f"sql: {query.sql[:200]!r}",
    )
    assert in_clause_present

    a41_in_params = "A419%" in query.params
    reporter.add_assertion(
        "Params include normalized prefix for sepsis code ('A419%')",
        a41_in_params,
        detail=f"params: {query.params!r}",
    )
    assert a41_in_params


def test_rendered_sql_inlines_params_diagnosis_count(backend, reporter):
    """Query-details feature: ``SqlFastpathQuery.rendered_sql`` inlines the
    bound parameter values so the expander shows what actually ran — no ``?``
    placeholders survive. Mirrors the Inc 4 diagnosis-count emission case;
    rendering is additive (the parameterized ``query.sql`` is unchanged)."""
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
    rendered = query.rendered_sql
    _record_sql(reporter, query, "Rendered SQL inlines params (Query Details)")

    no_placeholders = "?" not in rendered
    reporter.add_assertion(
        "rendered_sql has no leftover '?' placeholders",
        no_placeholders,
        detail=f"rendered: {rendered[:300]!r}",
    )
    assert no_placeholders

    value_inlined = "'A419%'" in rendered
    reporter.add_assertion(
        "rendered_sql inlines the grounded ICD prefix as a quoted literal "
        "('A419%')",
        value_inlined,
        detail=f"rendered: {rendered[:300]!r}",
    )
    assert value_inlined

    # Structural tokens survive — only the ``?`` placeholders change.
    assert "di.icd_code LIKE" in rendered

    # The property is exactly the pure renderer applied to (sql, params),
    # and is idempotent across accesses.
    assert rendered == render_sql_with_params(query.sql, query.params)
    assert query.rendered_sql == rendered


def test_filter_side_emits_in_list_for_unregistered_phrase(
    backend, monkeypatch, reporter,
):
    """Inc 9 OMOPHub autocode-fallback path. Inc 10 made the registry
    win for known cohort names, but for arbitrary phrases that don't
    resolve via ``resolve_cohort_name`` the filter should still fall
    back to OMOPHub's ``icd_autocode`` and emit a parallel-OR IN-list.

    This test uses 'carcinoid syndrome' (deliberately not in
    ``data/mappings/clinical_cohorts.json``) so the autocode path
    fires."""
    monkeypatch.setattr(
        cr, "icd_autocode",
        lambda *a, **kw: {
            "status": "ok",
            "results": [
                {"code": "E34.0", "title": "Carcinoid syndrome", "confidence": 0.93},
            ],
        },
        raising=False,
    )

    cq = CompetencyQuestion(
        original_question="What is the mean serotonin in our carcinoid cohort?",
        clinical_concepts=[
            ClinicalConcept(name="serotonin", concept_type="biomarker"),
        ],
        patient_filters=[
            PatientFilter(
                field="diagnosis", operator="contains",
                value="carcinoid syndrome",
            ),
        ],
        aggregation="mean", scope="cohort",
    )
    query = compile_sql(
        cq, backend, get_default_registry(),
        resolved_names=["serotonin"],
        enable_mcp_grounding=True,
    )
    _record_sql(reporter, query, "Mean serotonin in carcinoid cohort (Inc 9 autocode fallback)")

    in_present = "di.icd_code LIKE ?" in query.sql
    reporter.add_assertion(
        "Filter emits 'di.icd_code LIKE ?' for non-registry phrase (autocode fallback)",
        in_present,
    )
    assert in_present

    has_grounded_code = "E340%" in query.params
    reporter.add_assertion(
        "Params contain the normalized autocode ICD prefix (E340%)",
        has_grounded_code,
        detail=f"params: {query.params!r}",
    )
    assert has_grounded_code


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


def test_filter_uses_cohort_registry_when_value_matches_a_named_cohort(
    backend, monkeypatch, reporter,
):
    """Inc 10 — registry-first cohort grounding.

    Previously (Inc 9), a "sepsis" patient_filter routed through
    ``icd_autocode``, which returns confidence-ranked specific codes
    (e.g. A41.9, R65.21, A40.9). That set skews toward severe sepsis
    with shock — exactly the patients with the highest lactate values
    (clinical criterion, not a bug) — so the resulting SQL gives a
    cohort mean ~3× higher than the broader sepsis cohort the catalog
    uses for its reference distribution.

    The fix: when the filter value resolves to a registered cohort name
    via ``resolve_cohort_name``, use the registry's ICD prefixes
    (``A41.``, ``R65.20``, ``R65.21``, ICD-9 ``995.91``/``995.92``) as
    LIKE patterns instead of autocode's narrow IN-list. The registry's
    definition is what ``mimic_distribution_lookup`` already uses, so
    after Inc 10 the live SQL and the catalog reference query the SAME
    cohort.

    OMOPHub remains the fallback for arbitrary phrases that don't
    match any registered cohort (rare conditions, etc.).
    """
    # Stub icd_autocode so we can prove it ISN'T called when the
    # registry resolves the cohort name.
    autocode_calls = [0]
    def fake_autocode(*a, **kw):
        autocode_calls[0] += 1
        return MOCK_OMOPHUB_SEPSIS_RESPONSE
    monkeypatch.setattr(cr, "icd_autocode", fake_autocode, raising=False)

    cq = CompetencyQuestion(
        original_question="Mean lactate in sepsis cohort",
        clinical_concepts=[
            ClinicalConcept(name="lactate", concept_type="biomarker"),
        ],
        patient_filters=[
            # "sepsis" matches the registered cohort name in
            # data/mappings/clinical_cohorts.json.
            PatientFilter(field="diagnosis", operator="contains", value="sepsis"),
        ],
        aggregation="mean", scope="cohort",
    )
    query = compile_sql(
        cq, backend, get_default_registry(),
        resolved_names=["lactate"],
        enable_mcp_grounding=True,
    )
    _record_sql(reporter, query, "Mean lactate in sepsis cohort (Inc 10 registry-first)")

    # The registry's sepsis prefixes (dot-stripped, % suffixed for LIKE):
    #   ICD-10: A41%, R6520%, R6521%
    #   ICD-9:  99591%, 99592%
    expected_like_prefixes = {"A41%", "R6520%", "R6521%", "99591%", "99592%"}
    actual_prefixes = {p for p in query.params if isinstance(p, str) and "%" in p}
    matches = expected_like_prefixes & actual_prefixes
    has_registry_prefixes = len(matches) >= 3  # at least the ICD-10 set
    reporter.add_assertion(
        f"Params contain registry sepsis LIKE prefixes (got {sorted(matches)!r}; "
        f"need ≥3 of {sorted(expected_like_prefixes)!r})",
        has_registry_prefixes,
        detail=f"all params: {query.params!r}",
    )
    assert has_registry_prefixes, (
        f"expected registry-defined sepsis prefixes; got params={query.params!r}"
    )

    # ICD prefix LIKE clauses should appear in the SQL (any of the
    # canonical patterns).
    has_prefix_like = (
        "icd_code LIKE ?" in query.sql
        or "di.icd_code LIKE ?" in query.sql
    )
    reporter.add_assertion(
        "SQL contains 'icd_code LIKE ?' for prefix matching",
        has_prefix_like,
        detail=f"sql: {query.sql[:300]!r}",
    )
    assert has_prefix_like

    # icd_autocode should NOT have been called — registry hit short-
    # circuited the MCP path.
    autocode_skipped = autocode_calls[0] == 0
    reporter.add_assertion(
        "icd_autocode skipped (registry hit takes precedence)",
        autocode_skipped,
        detail=f"autocode call count: {autocode_calls[0]}",
    )
    assert autocode_skipped, (
        f"expected registry to win, but icd_autocode was called "
        f"{autocode_calls[0]} times"
    )


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
