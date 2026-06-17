"""SQL fast-path compiler for the conversational pipeline.

Phase 7a: for a CompetencyQuestion classified as ``QueryPlan.SQL_FAST``,
this module produces a single SQL statement that answers the question
directly — skipping extraction, graph construction, and SPARQL. The
output row shape is identical to what the equivalent SPARQL template
would return, so the downstream answerer sees no difference.

Supported shapes today (``QueryPlanner.classify`` gates entry):
  - Single-concept aggregate with a portable ``sql_fn`` (AVG/MAX/MIN/COUNT),
    no temporal constraints, over biomarker/vital/drug/microbiology/outcome.
  - Comparison (``scope == "comparison"``) on a registered axis whose
    ``sql_group_by`` is set. Emits GROUP BY in one query.
  - Diagnosis list (no aggregation, ``concept_type == "diagnosis"``).

The compiler reuses:
  - ``OperationRegistry.compile_filters`` for patient-filter WHERE clauses.
  - ``backend.ilike``, ``backend.table``, ``backend.readmission_labels_expr``
    for dialect-agnostic SQL emission.

Unsupported CQ shapes raise ``ValueError`` so a misrouted CQ fails loudly.
"""

from __future__ import annotations

import contextvars
import re
from dataclasses import dataclass
from typing import Any

from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    PatientFilter,
)
from src.conversational.sql_render import render_sql_with_params

# Ambient grounding context for measurement-value filters, set by ``compile_sql``
# and read by ``_filter_fragment`` — avoids threading ``resolver`` /
# ``derived_formulas`` through all six concept-type compile branches. Holds
# ``(resolver, derived_formulas)``: ``resolver`` (a ConceptResolver) lets the
# lab_value/vital_value filters ground their analyte LOINC → itemids;
# ``derived_formulas`` maps a derived-index name → a pre-resolved DerivedFormula
# for the derived_value filter. Defaults to ``(None, None)`` so the filters fall
# back to label-LIKE / drop gracefully when grounding wasn't supplied.
_FILTER_GROUNDING_CTX: contextvars.ContextVar = contextvars.ContextVar(
    "sql_fastpath_filter_grounding", default=(None, None),
)
from src.conversational.operations import (
    FilterCompileContext,
    OperationRegistry,
)


@dataclass
class OutlierScreen:
    """Absolute biological-possibility envelope for a numeric analyte.

    Values outside ``[low, high]`` are treated as data-entry errors /
    physically-impossible measurements and removed *before* aggregation, in
    SQL, so the screen scales to millions of rows. The bounds come from
    ``outliers.BiologicalLimitsResolver`` (LOINC + literature grounded), never
    from the cohort distribution — high-but-possible values (e.g. a sepsis
    lactate of 12 mmol/L) are kept; only impossible values are removed.

    ``max_rows_logged`` caps the companion outlier-rows query so a pathological
    column cannot drag millions of rows back for the UI report.

    ``units`` is the unit the bound is expressed in (e.g. ``"mmol/L"``). When
    set, a per-row units guard is emitted (biomarker/labevents only): the bound
    screens a row *unless* that row records a different unit, so a value that is
    legitimate in its own unit is never removed against a wrong-unit envelope.
    itemid-grounding keeps units homogeneous in practice, so this is a
    safety net; ``None`` disables the guard (the bound applies to every row).
    """

    low: float
    high: float
    max_rows_logged: int = 100
    units: str | None = None


@dataclass
class SqlFastpathQuery:
    """A compiled SQL statement ready for ``backend.execute``.

    ``columns`` lists the result-set column names in the same order they
    appear in the SELECT. The caller zips them with each row tuple to
    build the dicts the answerer consumes.
    """

    sql: str
    params: list[Any]
    columns: list[str]
    # Outlier-screening companion fields. All None unless an OutlierScreen was
    # applied (biomarker/vital aggregates only). ``columns`` stays the
    # answerer-facing prefix (e.g. ["mean_value"]); ``outlier_agg_columns`` is
    # the full ordered SELECT list the orchestrator zips rows against — it adds
    # the with-outliers value plus n_outliers/n_total so the report and the
    # "include outliers" toggle are precomputed in the same single pass.
    outlier_agg_columns: list[str] | None = None
    outlier_rows_sql: str | None = None
    outlier_rows_params: list[Any] | None = None
    outlier_rows_columns: list[str] | None = None
    outlier_low: float | None = None
    outlier_high: float | None = None

    @property
    def rendered_sql(self) -> str:
        """The ``sql`` template with ``params`` inlined as literals — the
        values-filled statement shown in the "Query Details" expander.

        Display only: ``sql`` + ``params`` remain what actually execute (the
        parameterized form preserves DuckDB/BigQuery binding and typing). A
        plain property (not cached) because the dataclass is rebuilt via
        ``dataclasses.replace`` mid-pipeline, so it always reflects the
        current ``sql``/``params``.

        Never raises: a real compiled query always has matching placeholders
        and params, but a *display* string must not crash a live answer — so
        if the two ever drift (the pure renderer's strict guard fires) we fall
        back to the parameterized template.
        """
        try:
            return render_sql_with_params(self.sql, self.params)
        except ValueError:
            return self.sql


# Aggregate → SELECT column name. Matches the SPARQL template's
# ``?mean_value`` / ``?max_value`` etc. so the answerer's _COLUMN_MAP
# lookup works unchanged.
_AGG_COLUMN_NAME: dict[str, str] = {
    "AVG": "mean_value",
    "MAX": "max_value",
    "MIN": "min_value",
    "COUNT": "count_value",
}


def compile_sql(
    cq: CompetencyQuestion,
    backend: Any,
    registry: OperationRegistry,
    *,
    resolved_names: list[str] | None = None,
    resolved_itemids: list[int] | None = None,
    resolved_icd_codes: list[str] | None = None,
    enable_mcp_grounding: bool = False,
    outlier_screen: "OutlierScreen | None" = None,
    resolver: Any = None,
    derived_formulas: dict[str, Any] | None = None,
) -> SqlFastpathQuery:
    """Compile a CQ to a SQL fast-path query.

    Thin wrapper that publishes the measurement-value-filter grounding context
    (``resolver`` + ``derived_formulas``) for ``_filter_fragment`` via a
    contextvar, then delegates to ``_compile_sql_dispatch``. The try/finally
    reset keeps the ambient context from leaking across compile calls. See
    ``_compile_sql_dispatch`` for the full parameter documentation.
    """
    token = _FILTER_GROUNDING_CTX.set((resolver, derived_formulas))
    try:
        return _compile_sql_dispatch(
            cq, backend, registry,
            resolved_names=resolved_names,
            resolved_itemids=resolved_itemids,
            resolved_icd_codes=resolved_icd_codes,
            enable_mcp_grounding=enable_mcp_grounding,
            outlier_screen=outlier_screen,
        )
    finally:
        _FILTER_GROUNDING_CTX.reset(token)


def _compile_sql_dispatch(
    cq: CompetencyQuestion,
    backend: Any,
    registry: OperationRegistry,
    *,
    resolved_names: list[str] | None = None,
    resolved_itemids: list[int] | None = None,
    resolved_icd_codes: list[str] | None = None,
    enable_mcp_grounding: bool = False,
    outlier_screen: "OutlierScreen | None" = None,
) -> SqlFastpathQuery:
    """Dispatch to the right compile branch based on CQ shape.

    Parameters
    ----------
    cq:
        The CompetencyQuestion. The planner has already decided this is
        fast-path-eligible; the compiler re-checks minimal invariants and
        raises ``ValueError`` if violated.
    backend:
        Duck-typed DuckDB/BigQuery backend (``table``, ``ilike``, etc.).
    registry:
        The OperationRegistry used for aggregate/axis metadata and
        filter-WHERE-clause compilation.
    resolved_names:
        Optional list of concrete concept names to OR-match in the WHERE
        clause. Used when the orchestrator's ConceptResolver expands a
        category (e.g. "antibiotics" → ["vancomycin", "ceftriaxone", …])
        so the fast-path covers the same membership as the graph path.
        Defaults to ``[cq.clinical_concepts[0].name]``.
    resolved_itemids:
        Optional list of MIMIC ``itemid`` values for biomarker concepts
        whose LOINC code was resolved successfully (see
        ``ConceptResolver.resolve_biomarker``). When provided, the
        biomarker compile branch emits ``WHERE l.itemid IN (?, ?, …)``
        instead of a ``LIKE`` filter on ``d.label`` — avoids pooling
        unit-incompatible variants (e.g. serum vs urine creatinine).
        Ignored for non-biomarker concept types.
    resolved_icd_codes:
        Optional list of ICD codes for diagnosis concepts whose ``name``
        was grounded via OMOPHub's ``icd_autocode`` (see
        ``ConceptResolver.resolve_diagnosis``). When non-empty, the
        diagnosis-count compile branch emits ``WHERE (di.icd_code IN
        (?, …)) OR (<existing LIKE clause>)`` — IN-list as a parallel OR
        with the title-LIKE fallback so ICD-9 admissions still match.
        Ignored for non-diagnosis concept types and for the
        diagnosis-list (no-aggregation) path.
    """
    # Defensive re-check: planner should already have gated these out,
    # but an explicit raise here surfaces misrouting instead of emitting
    # broken SQL.
    #
    # event_ordering is a MULTI-event operation (it deliberately carries ≥2
    # clinical_concepts and asks "which event came first"), so it is dispatched
    # BEFORE the single-concept / temporal guards below — those would otherwise
    # reject it. It returns per-patient FIRST-event-time rows; the aggregate's
    # ``_event_ordering_post_processor`` turns them into the ordering summary.
    if cq.aggregation == "event_ordering":
        return _compile_event_ordering(cq, backend, registry,
                                       enable_mcp_grounding=enable_mcp_grounding)

    if cq.temporal_constraints:
        raise ValueError(
            "sql_fastpath cannot compile a CQ with temporal_constraints — "
            "route to the graph path"
        )
    if len(cq.clinical_concepts) > 1:
        raise ValueError(
            "sql_fastpath cannot compile a CQ with multiple clinical_concepts"
        )
    if not cq.clinical_concepts:
        raise ValueError(
            "sql_fastpath requires exactly one clinical_concept; "
            "metadata-only CQs aren't supported in 7a"
        )

    concept = cq.clinical_concepts[0]
    names = list(resolved_names) if resolved_names else [concept.name]
    if not names:
        raise ValueError("resolved_names must be non-empty when provided")

    # Plain diagnosis list: no aggregation, return (subjectId, hadmId, …).
    if cq.aggregation is None and concept.concept_type == "diagnosis":
        return _compile_diagnosis_list(
            cq, concept, names, backend, registry,
            enable_mcp_grounding=enable_mcp_grounding,
        )

    if cq.aggregation is None:
        raise ValueError(
            "sql_fastpath needs an aggregation for non-diagnosis concepts"
        )

    agg_op = registry.get("aggregate", cq.aggregation)
    if agg_op is None or getattr(agg_op, "sql_fn", None) is None:
        raise ValueError(
            f"aggregation {cq.aggregation!r} is not SQL-fast-path-compilable "
            "(sql_fn is None; likely needs Python post-processing)"
        )

    sql_fn: str = agg_op.sql_fn  # type: ignore[assignment]

    # Route by concept type.
    if concept.concept_type in {"biomarker", "vital"}:
        return _compile_event_aggregate(
            cq, concept, names, sql_fn, backend, registry,
            itemids=resolved_itemids if concept.concept_type == "biomarker" else None,
            enable_mcp_grounding=enable_mcp_grounding,
            outlier_screen=outlier_screen,
        )
    if concept.concept_type == "drug":
        return _compile_drug_aggregate(
            cq, concept, names, sql_fn, backend, registry,
            enable_mcp_grounding=enable_mcp_grounding,
        )
    if concept.concept_type == "microbiology":
        return _compile_microbiology_aggregate(
            cq, concept, names, sql_fn, backend, registry,
            enable_mcp_grounding=enable_mcp_grounding,
        )
    if concept.concept_type == "diagnosis":
        return _compile_diagnosis_count(
            cq, concept, names, sql_fn, backend, registry,
            icd_codes=resolved_icd_codes,
            enable_mcp_grounding=enable_mcp_grounding,
        )
    if concept.concept_type == "outcome":
        return _compile_outcome_rate(
            cq, concept, sql_fn, backend, registry,
            enable_mcp_grounding=enable_mcp_grounding,
        )

    raise ValueError(
        f"sql_fastpath does not support concept_type={concept.concept_type!r}"
    )


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def _comparison_group_by_col(
    cq: CompetencyQuestion,
    registry: OperationRegistry,
    backend: Any | None = None,
    *,
    enable_mcp_grounding: bool = False,
) -> tuple[str | None, list[Any]]:
    """Return ``(group_by_col, group_by_params)`` for a comparison CQ.

    For the FIXED-column axes (gender/age/admission_type/…) the column is the
    axis's ``sql_group_by`` and there are no params — ``(col, [])``.

    For the dynamic ``condition`` axis (``comparison_field == "condition"``
    plus a ``split_condition``) the column is a correlated
    ``CASE WHEN EXISTS(<sub-condition>) THEN 'yes' ELSE 'no' END`` and the
    params are the sub-condition's. The same CASE/EXISTS text + params appear
    in both the SELECT and the GROUP BY; callers reference the column by its
    SELECT alias in the GROUP BY (``GROUP BY group_value``) so the params are
    supplied exactly once.

    Returns ``(None, [])`` when the CQ isn't a comparison scope.
    """
    if cq.scope != "comparison":
        return None, []
    if not cq.comparison_field:
        raise ValueError("comparison scope requires comparison_field")
    # The dynamic split-by-condition axis is checked BEFORE the sql_group_by
    # guard because its registered axis carries ``sql_group_by=None`` (the
    # column is built here from the sub-condition, not a fixed table column).
    if cq.comparison_field == "condition":
        if cq.split_condition is None:
            raise ValueError(
                "comparison_field='condition' requires split_condition"
            )
        if backend is None:
            raise ValueError(
                "split-by-condition comparison requires a backend to build "
                "the EXISTS sub-condition"
            )
        ctx = _split_condition_ctx(backend, enable_mcp_grounding)
        exists_sql, params = _split_condition_exists(cq.split_condition, ctx)
        col = f"CASE WHEN {exists_sql} THEN 'yes' ELSE 'no' END"
        return col, list(params)
    axis_op = registry.get("comparison_axis", cq.comparison_field)
    if axis_op is None or getattr(axis_op, "sql_group_by", None) is None:
        raise ValueError(
            f"comparison axis {cq.comparison_field!r} has no sql_group_by"
        )
    return axis_op.sql_group_by, []


def _split_condition_ctx(backend: Any, enable_mcp_grounding: bool) -> FilterCompileContext:
    """Build a FilterCompileContext for compiling a split_condition's EXISTS.

    Threads the same measurement-grounding context (resolver / derived
    formulas) the cohort filters use so a ``lab_value`` / ``vital_value``
    split_condition grounds its analyte → itemids identically. The registry
    is left ``None`` here; ``_split_condition_exists`` injects it on the rare
    composite path that needs it.
    """
    resolver, derived_formulas = _FILTER_GROUNDING_CTX.get()
    return FilterCompileContext(
        backend=backend,
        enable_mcp_grounding=enable_mcp_grounding,
        resolver=resolver,
        derived_formulas=derived_formulas,
    )


def _split_condition_exists(
    f: "PatientFilter", ctx: FilterCompileContext,
) -> tuple[str, list[Any]]:
    """Build a correlated ``EXISTS(...)`` for a split-by-condition sub-condition.

    The outer query's admissions alias is ``a`` (every fast-path FROM/JOIN
    introduces ``admissions a``), so the inner queries correlate on
    ``a.hadm_id``. Returns ``(exists_sql, params)``.

    Grounding per field — mirrors the cohort-filter EXISTS style in
    ``operations_filters.py``:

    * ``diagnosis`` — ICD-grounded via ``f.resolved_icd_codes`` (set by the
      orchestrator's ``_disambiguate_diagnoses``); each code is normalised and
      prefix-matched (``norm + '%'``), exactly like the diagnosis cohort filter.
      When no codes are grounded, fall back to a title-LIKE EXISTS joining
      ``d_icd_diagnoses`` on ``f.value``.
    * ``ventilation`` — invasive/non-invasive ventilation + intubation
      procedure events (``procedureevents`` itemids 225792 'Invasive
      Ventilation', 224385 'Intubation', 225794 'Non-invasive Ventilation').
      No params.
    * anything else (``lab_value`` / ``vital_value`` / ``icu_stay`` / …) —
      compiled through the registered filter op, whose ``FilterFragment.where[0]``
      is already an ``EXISTS(...)``; that fragment is reused verbatim.
    """
    t = ctx.backend.table
    field = getattr(f, "field", None)
    value = f.value if isinstance(f.value, str) else ""

    # Mechanical ventilation is a PROCEDURE, but the decomposer routinely
    # mislabels it field="diagnosis" value="mechanical ventilation". Detect it by
    # marker so either shape lands on the procedureevents EXISTS (invasive vent /
    # intubation / non-invasive vent). Markers are specific enough to skip
    # diagnoses that merely mention a ventilator (e.g. ventilator-assoc pneumonia).
    if field == "ventilation" or any(
        m in value.lower() for m in ("mechanical vent", "intubat", "invasive vent")
    ):
        inner = (
            f"SELECT 1 FROM {t('procedureevents')} pe "
            f"WHERE pe.hadm_id = a.hadm_id "
            f"AND pe.itemid IN (225792, 224385, 225794)"
        )
        return f"EXISTS ({inner})", []

    # ``or_any`` composite — match if ANY sub-condition holds (e.g. chronic
    # anticoagulant OR antiplatelet use). Recurse so each leaf grounds via its own
    # path; diagnosis leaves carry resolved_icd_codes set by the orchestrator.
    if field == "or_any":
        parts: list[str] = []
        or_params: list[Any] = []
        for sub in (getattr(f, "sub_filters", None) or []):
            sub_sql, sub_p = _split_condition_exists(sub, ctx)
            parts.append(sub_sql)
            or_params.extend(sub_p)
        if parts:
            return "(" + " OR ".join(parts) + ")", or_params
        return "EXISTS (SELECT 1 WHERE 1 = 0)", []

    if field == "diagnosis":
        resolved = getattr(f, "resolved_icd_codes", None)
        if resolved:
            from src.conversational.health_evidence.cohorts import (
                normalize_icd_prefix,
            )
            clauses: list[str] = []
            params: list[Any] = []
            for code in resolved:
                norm = normalize_icd_prefix(code)
                if not norm:
                    continue
                clauses.append("di2.icd_code LIKE ?")
                params.append(norm + "%")
            if clauses:
                inner = (
                    f"SELECT 1 FROM {t('diagnoses_icd')} di2 "
                    f"WHERE di2.hadm_id = a.hadm_id AND ({' OR '.join(clauses)})"
                )
                return f"EXISTS ({inner})", params
        # No grounded codes → title-LIKE fallback (ICD-9 / ungrounded terms).
        value = f.value if isinstance(f.value, str) else ""
        inner = (
            f"SELECT 1 FROM {t('diagnoses_icd')} di2 "
            f"JOIN {t('d_icd_diagnoses')} dd2 ON di2.icd_code = dd2.icd_code "
            f"AND di2.icd_version = dd2.icd_version "
            f"WHERE di2.hadm_id = a.hadm_id "
            f"AND ({ctx.backend.ilike('dd2.long_title')} OR di2.icd_code LIKE ?)"
        )
        return f"EXISTS ({inner})", [f"%{value}%", f"{value}%"]

    # Generic path: reuse the registered filter's EXISTS fragment. The filter
    # compile_fns correlate on ``ctx.admission_alias`` ("a" by default), which
    # is exactly the outer alias here.
    registry = ctx.registry or _default_operation_registry()
    ctx.registry = registry
    op = registry.get("filter", field)
    if op is None:
        # Unknown field — emit a never-true sub-condition so the split degrades
        # to a single "no" group rather than crashing.
        return "EXISTS (SELECT 1 WHERE 1 = 0)", []
    frag = op.compile(f, ctx)
    if frag.joins or not frag.where:
        # A filter that needs an unconditional JOIN can't be a self-contained
        # correlated EXISTS; degrade safely.
        return "EXISTS (SELECT 1 WHERE 1 = 0)", []
    where_sql = " AND ".join(frag.where)
    return f"({where_sql})", list(frag.params)


def _default_operation_registry() -> OperationRegistry:
    """Lazy accessor for the process-wide default registry.

    Used by ``_split_condition_exists`` when no registry is threaded through
    the compile context (e.g. a direct ``_comparison_group_by_col`` call in a
    unit test). Kept out of the module top-level import to avoid the
    operations ↔ operations_filters circular-import dance at module load.
    """
    from src.conversational.operations import get_default_registry

    return get_default_registry()


def _filter_fragment(
    cq: CompetencyQuestion,
    backend: Any,
    registry: OperationRegistry,
    *,
    enable_mcp_grounding: bool = False,
) -> tuple[list[str], list[str], list[Any], bool, bool]:
    """Compile patient_filters to SQL fragments.

    Returns ``(joins, where, params, needs_patients, needs_readmission)``
    where ``needs_readmission`` tracks whether any filter already joined
    the readmission_labels CTE — so the GROUP BY path doesn't double-join.

    ``enable_mcp_grounding`` (Inc 9.3) is threaded into the
    ``FilterCompileContext`` so the diagnosis filter can ground via
    OMOPHub-backed icd_autocode in the production pipeline. Default
    False keeps unit tests offline-safe.
    """
    resolver, derived_formulas = _FILTER_GROUNDING_CTX.get()
    ctx = FilterCompileContext(
        backend=backend, enable_mcp_grounding=enable_mcp_grounding,
        resolver=resolver, derived_formulas=derived_formulas,
    )
    frag = registry.compile_filters(cq.patient_filters, ctx)
    needs_readmission = any(
        "rl." in w for w in frag.where
    ) or any("rl " in j or " rl " in j for j in frag.joins)
    return (
        list(frag.joins),
        list(frag.where),
        list(frag.params),
        frag.needs_patients,
        needs_readmission,
    )


def _needs_readmission_axis(group_by_col: str | None) -> bool:
    """Does the comparison axis reference the readmission_labels CTE?"""
    return bool(group_by_col) and group_by_col.startswith("rl.")


def _needs_patients_axis(group_by_col: str | None) -> bool:
    """Does the comparison axis reference the patients table?"""
    return bool(group_by_col) and group_by_col.startswith("p.")


def _ilike_any(backend: Any, column: str, names: list[str]) -> tuple[str, list[str]]:
    """Build ``(col ILIKE ? OR col ILIKE ? …)`` for a list of names.

    Used when a category resolves to multiple concrete names — mirrors the
    graph path, which issues one extract call per resolved name.
    """
    clauses = [backend.ilike(column) for _ in names]
    return "(" + " OR ".join(clauses) + ")", [f"%{n}%" for n in names]


def _itemid_in(alias: str, itemids: list[int]) -> tuple[str, list[int]]:
    """Build ``alias.itemid IN (?, ?, …)`` for a list of MIMIC itemids.

    Used when a biomarker concept carries a LOINC code that's been resolved
    to specific MIMIC labitem identifiers — preferred over ``_ilike_any``
    because itemid filtering avoids the unit-pollution problem that
    ``LIKE '%creatinine%'`` causes (which pools serum + urine + ratios).
    """
    placeholders = ",".join(["?"] * len(itemids))
    return f"{alias}.itemid IN ({placeholders})", list(itemids)


# ---------------------------------------------------------------------------
# Biomarker / vital event aggregation
# ---------------------------------------------------------------------------


def _compile_event_aggregate(
    cq: CompetencyQuestion,
    concept: ClinicalConcept,
    names: list[str],
    sql_fn: str,
    backend: Any,
    registry: OperationRegistry,
    *,
    itemids: list[int] | None = None,
    enable_mcp_grounding: bool = False,
    outlier_screen: "OutlierScreen | None" = None,
) -> SqlFastpathQuery:
    """AVG/MAX/MIN/COUNT over biomarker (labevents) or vital (chartevents).

    When ``itemids`` is supplied (biomarker concepts only, populated by the
    orchestrator from ``ConceptResolver.resolve_biomarker``), the WHERE
    clause filters on ``itemid IN (...)`` instead of the label-substring
    LIKE — avoids pooling unit-incompatible variants of the same lab.

    When ``outlier_screen`` is supplied, the aggregate is emitted twice in one
    pass — a clean value that excludes rows outside the biological-possibility
    envelope and a with-outliers value that keeps them — plus per-query
    n_outliers/n_total counts and a companion query returning the removed rows.
    The same ``CASE WHEN`` template screens COUNT too, so the reported ``n``
    stays consistent with the screened mean.
    """
    t = backend.table
    if concept.concept_type == "biomarker":
        event_table = f"{t('labevents')} l"
        event_alias = "l"
        dict_table = f"{t('d_labitems')} d"
        name_col = "d.label"
        join_dict = f"JOIN {dict_table} ON l.itemid = d.itemid"
    else:  # vital
        event_table = f"{t('chartevents')} c"
        event_alias = "c"
        dict_table = f"{t('d_items')} d"
        name_col = "d.label"
        join_dict = f"JOIN {dict_table} ON c.itemid = d.itemid"

    group_by_col, group_by_params = _comparison_group_by_col(
        cq, registry, backend, enable_mcp_grounding=enable_mcp_grounding,
    )
    filter_joins, filter_where, filter_params, filter_needs_patients, filter_has_readmission = \
        _filter_fragment(cq, backend, registry, enable_mcp_grounding=enable_mcp_grounding)

    # Decide what extra structural joins we need:
    #  - admissions: always, so cohort filters on `a.*` work.
    #  - patients: if any filter or the group-by axis needs it.
    #  - readmission_labels: if the axis is rl.* and filters didn't already join it.
    needs_patients = filter_needs_patients or _needs_patients_axis(group_by_col)
    needs_readmission = (
        _needs_readmission_axis(group_by_col) and not filter_has_readmission
    )

    joins: list[str] = [f"JOIN {t('admissions')} a ON {event_alias}.hadm_id = a.hadm_id"]
    if needs_patients:
        joins.append(f"JOIN {t('patients')} p ON a.subject_id = p.subject_id")
    if needs_readmission:
        joins.append(
            f"JOIN {backend.readmission_labels_expr()} rl ON a.hadm_id = rl.hadm_id"
        )
    joins.extend(filter_joins)

    # WHERE: concept match + non-null value + cohort filters.
    # When the resolver supplied itemids (LOINC-grounded biomarker), filter
    # on those directly; otherwise fall back to the label-substring LIKE.
    if itemids:
        name_clause, name_params = _itemid_in(event_alias, itemids)
    else:
        name_clause, name_params = _ilike_any(backend, name_col, names)
    where_clauses: list[str] = [
        name_clause,
        f"{event_alias}.valuenum IS NOT NULL",
    ]
    where_params: list[Any] = list(name_params)
    where_clauses.extend(filter_where)
    where_params.extend(filter_params)
    where_sql = " AND ".join(where_clauses)
    from_sql = f"FROM {event_table} {join_dict} {' '.join(joins)}"

    val = f"{event_alias}.valuenum"

    # Screen predicates, built once so the clean aggregate, the n_outliers
    # count, and the companion rows query stay in lockstep. ``clean_cond`` is
    # the keep-this-value condition; ``outlier_cond`` is its negation (the
    # row is a removable outlier). Both carry the same params in the same
    # order (``[lo, hi]`` plus, when the units guard is active, the bound
    # unit), so callers can reuse ``screen_cond_params`` per occurrence.
    #
    # Units guard (biomarker/labevents only — chartevents has no per-row unit
    # here): the impossibility bound is expressed in one unit, so apply it to a
    # row UNLESS that row records a *different* unit. A genuinely mixed-unit
    # value (e.g. a mg/dL reading among mmol/L rows) is then never screened
    # against the wrong-unit envelope. NULL or matching units → bound applies,
    # so data-entry errors (which often drop the unit) are still caught.
    clean_cond = ""
    outlier_cond = ""
    screen_cond_params: list[Any] = []
    if outlier_screen is not None:
        lo, hi = outlier_screen.low, outlier_screen.high
        if outlier_screen.units and concept.concept_type == "biomarker":
            applies = (
                f"({event_alias}.valueuom IS NULL "
                f"OR LOWER(TRIM({event_alias}.valueuom)) = LOWER(TRIM(?)))"
            )
            clean_cond = f"(({val} BETWEEN ? AND ?) OR NOT {applies})"
            outlier_cond = f"(NOT ({val} BETWEEN ? AND ?) AND {applies})"
            screen_cond_params = [lo, hi, outlier_screen.units]
        else:
            clean_cond = f"{val} BETWEEN ? AND ?"
            outlier_cond = f"NOT ({val} BETWEEN ? AND ?)"
            screen_cond_params = [lo, hi]

    # SELECT and GROUP BY shape. With an OutlierScreen, every value-bearing
    # aggregate is emitted twice in one pass — clean (CASE WHEN kept) and
    # with-outliers — plus n_outliers/n_total, using portable CASE WHEN
    # (FILTER(WHERE ...) is not BigQuery-safe). The SELECT-clause screen params
    # precede the WHERE params positionally.
    select_params: list[Any] = []
    outlier_agg_columns: list[str] | None = None
    if group_by_col is None:
        if outlier_screen is None:
            select_cols = f"{sql_fn}({val}) AS {_AGG_COLUMN_NAME[sql_fn]}"
            columns = [_AGG_COLUMN_NAME[sql_fn]]
        else:
            clean_col = _AGG_COLUMN_NAME[sql_fn]
            with_col = f"{clean_col}_with_outliers"
            select_cols = (
                f"{sql_fn}(CASE WHEN {clean_cond} THEN {val} END) AS {clean_col}, "
                f"{sql_fn}({val}) AS {with_col}, "
                f"SUM(CASE WHEN {outlier_cond} THEN 1 ELSE 0 END) AS n_outliers, "
                f"COUNT({val}) AS n_total"
            )
            columns = [clean_col]
            outlier_agg_columns = [clean_col, with_col, "n_outliers", "n_total"]
            select_params = list(screen_cond_params) + list(screen_cond_params)
        group_by_clause = ""
    else:
        # Comparison: always AVG + COUNT, matching comparison_by_field SPARQL.
        if outlier_screen is None:
            select_cols = (
                f"{group_by_col} AS group_value, "
                f"AVG({val}) AS avg_value, "
                f"COUNT({val}) AS count"
            )
            columns = ["group_value", "avg_value", "count"]
        else:
            select_cols = (
                f"{group_by_col} AS group_value, "
                f"AVG(CASE WHEN {clean_cond} THEN {val} END) AS avg_value, "
                f"AVG({val}) AS avg_value_with_outliers, "
                f"COUNT(CASE WHEN {clean_cond} THEN {val} END) AS count, "
                f"SUM(CASE WHEN {outlier_cond} THEN 1 ELSE 0 END) AS n_outliers"
            )
            columns = ["group_value", "avg_value", "count"]
            outlier_agg_columns = [
                "group_value", "avg_value", "avg_value_with_outliers",
                "count", "n_outliers",
            ]
            select_params = list(screen_cond_params) * 3
        # GROUP BY the SELECT alias rather than re-emitting the axis expression,
        # so a parameterized axis (the ``condition`` CASE/EXISTS) binds its
        # params exactly once (in the SELECT). Both DuckDB and BigQuery permit
        # grouping by output alias.
        group_by_clause = "GROUP BY group_value"

    # ``group_by_col`` is the first SELECT column for the comparison shape, so
    # its params lead the positional list.
    params: list[Any] = list(group_by_params) + select_params + where_params

    sql = (
        f"SELECT {select_cols} "
        f"{from_sql} "
        f"WHERE {where_sql} "
        f"{group_by_clause}"
    ).strip()

    # Companion outlier-rows query: same FROM/JOIN/WHERE plus the same
    # ``outlier_cond`` the aggregate uses (envelope + optional units guard),
    # capped by LIMIT, so the orchestrator logs/shows exactly the rows the
    # screen removed — and only those (a mismatched-unit row the guard keeps
    # never appears here either). labevents carries valueuom; chartevents does
    # not, so emit NULL there for the displayed provenance column.
    outlier_rows_sql: str | None = None
    outlier_rows_params: list[Any] | None = None
    outlier_rows_columns: list[str] | None = None
    outlier_low: float | None = None
    outlier_high: float | None = None
    if outlier_screen is not None:
        outlier_low, outlier_high = outlier_screen.low, outlier_screen.high
        uom_expr = (
            f"{event_alias}.valueuom"
            if concept.concept_type == "biomarker"
            else "NULL"
        )
        rows_select = (
            f"{val} AS valuenum, "
            f"{event_alias}.subject_id AS subject_id, "
            f"{event_alias}.hadm_id AS hadm_id, "
            f"{event_alias}.charttime AS charttime, "
            f"d.label AS label, "
            f"{uom_expr} AS valueuom"
        )
        outlier_rows_columns = [
            "valuenum", "subject_id", "hadm_id", "charttime", "label", "valueuom",
        ]
        outlier_rows_sql = (
            f"SELECT {rows_select} "
            f"{from_sql} "
            f"WHERE {where_sql} AND {outlier_cond} "
            f"ORDER BY {val} DESC "
            f"LIMIT ?"
        ).strip()
        outlier_rows_params = (
            list(where_params)
            + list(screen_cond_params)
            + [outlier_screen.max_rows_logged]
        )

    return SqlFastpathQuery(
        sql=sql,
        params=params,
        columns=columns,
        outlier_agg_columns=outlier_agg_columns,
        outlier_rows_sql=outlier_rows_sql,
        outlier_rows_params=outlier_rows_params,
        outlier_rows_columns=outlier_rows_columns,
        outlier_low=outlier_low,
        outlier_high=outlier_high,
    )


# ---------------------------------------------------------------------------
# Drug aggregation (prescriptions table)
# ---------------------------------------------------------------------------


def _compile_drug_aggregate(
    cq: CompetencyQuestion,
    concept: ClinicalConcept,
    names: list[str],
    sql_fn: str,
    backend: Any,
    registry: OperationRegistry,
    *,
    enable_mcp_grounding: bool = False,
) -> SqlFastpathQuery:
    """Drug events have no numeric value to AVG/MAX/MIN over — only COUNT
    is meaningful. Reject other aggregates so the planner doesn't produce
    nonsense."""
    if sql_fn not in {"COUNT"}:
        raise ValueError(
            f"sql_fastpath supports only COUNT for drug concepts; got {sql_fn}"
        )

    t = backend.table
    group_by_col, group_by_params = _comparison_group_by_col(
        cq, registry, backend, enable_mcp_grounding=enable_mcp_grounding,
    )
    filter_joins, filter_where, filter_params, filter_needs_patients, filter_has_readmission = \
        _filter_fragment(cq, backend, registry, enable_mcp_grounding=enable_mcp_grounding)

    needs_patients = filter_needs_patients or _needs_patients_axis(group_by_col)
    needs_readmission = (
        _needs_readmission_axis(group_by_col) and not filter_has_readmission
    )

    joins: list[str] = [f"JOIN {t('admissions')} a ON pr.hadm_id = a.hadm_id"]
    if needs_patients:
        joins.append(f"JOIN {t('patients')} p ON a.subject_id = p.subject_id")
    if needs_readmission:
        joins.append(
            f"JOIN {backend.readmission_labels_expr()} rl ON a.hadm_id = rl.hadm_id"
        )
    joins.extend(filter_joins)

    # Distinct admissions, not raw prescription rows: a single admission
    # carries many orders for the same drug (often one per administration), so
    # COUNT(*) inflates a "how many patients/admissions" count by an order of
    # magnitude. Mirrors the diagnosis/outcome count grain (COUNT(DISTINCT
    # hadm_id)).
    if group_by_col is None:
        select_cols = "COUNT(DISTINCT a.hadm_id) AS count_value"
        columns = ["count_value"]
        group_by_clause = ""
    else:
        select_cols = (
            f"{group_by_col} AS group_value, "
            f"COUNT(DISTINCT a.hadm_id) AS avg_value, "
            f"COUNT(DISTINCT a.hadm_id) AS count"
        )
        columns = ["group_value", "avg_value", "count"]
        group_by_clause = "GROUP BY group_value"

    name_clause, name_params = _ilike_any(backend, "pr.drug", names)
    where_clauses: list[str] = [name_clause]
    # ``group_by_col`` (when present) leads the SELECT, so its params precede
    # the WHERE-clause params positionally.
    params: list[Any] = list(group_by_params) + list(name_params)
    where_clauses.extend(filter_where)
    params.extend(filter_params)

    sql = (
        f"SELECT {select_cols} "
        f"FROM {t('prescriptions')} pr {' '.join(joins)} "
        f"WHERE {' AND '.join(where_clauses)} "
        f"{group_by_clause}"
    ).strip()

    return SqlFastpathQuery(sql=sql, params=params, columns=columns)


# ---------------------------------------------------------------------------
# Microbiology
# ---------------------------------------------------------------------------


# A trailing "culture"/"cultures"/"cx" is the *test modality*, not the
# specimen, and MIMIC's ``spec_type_desc`` records the specimen by anatomic
# source — ``URINE``, ``SPUTUM``, ``STOOL``, ``SWAB`` — with the literal word
# "culture" kept only for blood (``BLOOD CULTURE``). So matching "sputum
# culture"/"urine culture" verbatim hits 0 rows. Dropping the modality token
# lets the source match the schema vocabulary ("sputum culture" → ``SPUTUM``)
# while blood still matches (``BLOOD CULTURE`` contains "blood"). This is a
# morphological normalization, not a curated specimen table.
_CULTURE_SUFFIX_RE = re.compile(r"\s+(?:cultures?|cx)\s*$", re.IGNORECASE)


def _microbiology_match_term(term: str) -> str:
    """Strip a trailing "culture"/"cultures"/"cx" modality token from a
    microbiology term so the specimen source matches ``spec_type_desc``.

    Only strips when a non-empty stem remains (so a bare "culture" is left
    untouched). Organism names never carry the suffix, so they pass through
    unchanged."""
    stripped = _CULTURE_SUFFIX_RE.sub("", term).strip()
    return stripped or term


def _compile_microbiology_aggregate(
    cq: CompetencyQuestion,
    concept: ClinicalConcept,
    names: list[str],
    sql_fn: str,
    backend: Any,
    registry: OperationRegistry,
    *,
    enable_mcp_grounding: bool = False,
) -> SqlFastpathQuery:
    if sql_fn != "COUNT":
        raise ValueError(
            f"sql_fastpath supports only COUNT for microbiology; got {sql_fn}"
        )

    t = backend.table
    group_by_col, group_by_params = _comparison_group_by_col(
        cq, registry, backend, enable_mcp_grounding=enable_mcp_grounding,
    )
    filter_joins, filter_where, filter_params, filter_needs_patients, filter_has_readmission = \
        _filter_fragment(cq, backend, registry, enable_mcp_grounding=enable_mcp_grounding)

    needs_patients = filter_needs_patients or _needs_patients_axis(group_by_col)
    needs_readmission = (
        _needs_readmission_axis(group_by_col) and not filter_has_readmission
    )

    joins: list[str] = [f"JOIN {t('admissions')} a ON m.hadm_id = a.hadm_id"]
    if needs_patients:
        joins.append(f"JOIN {t('patients')} p ON a.subject_id = p.subject_id")
    if needs_readmission:
        joins.append(
            f"JOIN {backend.readmission_labels_expr()} rl ON a.hadm_id = rl.hadm_id"
        )
    joins.extend(filter_joins)

    # Distinct admissions, not raw culture-event rows: one admission can have
    # many cultures matching the term. Mirrors the diagnosis/outcome grain.
    if group_by_col is None:
        select_cols = "COUNT(DISTINCT a.hadm_id) AS count_value"
        columns = ["count_value"]
        group_by_clause = ""
    else:
        select_cols = (
            f"{group_by_col} AS group_value, "
            f"COUNT(DISTINCT a.hadm_id) AS avg_value, "
            f"COUNT(DISTINCT a.hadm_id) AS count"
        )
        columns = ["group_value", "avg_value", "count"]
        group_by_clause = "GROUP BY group_value"

    # Microbiology matches both specimen type and organism name, so each
    # resolved name contributes two ILIKE clauses OR-combined.
    per_name_clauses: list[str] = []
    per_name_params: list[str] = []
    for n in names:
        nt = _microbiology_match_term(n)
        per_name_clauses.append(
            f"({backend.ilike('m.spec_type_desc')} OR {backend.ilike('m.org_name')})"
        )
        per_name_params.extend([f"%{nt}%", f"%{nt}%"])
    where_clauses: list[str] = ["(" + " OR ".join(per_name_clauses) + ")"]
    # ``group_by_col`` (when present) leads the SELECT, so its params precede
    # the WHERE-clause params positionally.
    params: list[Any] = list(group_by_params) + list(per_name_params)

    # Qualifier attributes narrow the SAME culture, so they *AND* with the
    # name match. A microbiology question often names two dimensions — a
    # specimen and an organism ("blood culture that grew E. coli") — and the
    # decomposer puts one in ``name`` and the other in ``attributes``. Each
    # term is still matched against *either* column (spec_type_desc OR
    # org_name) because the decomposer may place specimen or organism in
    # either slot, but the terms are conjoined: the result must match the
    # specimen AND the organism, not their union. This is what gives
    # "blood culture that grew E. coli" its true cohort (specimen∈blood ∧
    # organism∈E. coli) instead of every blood culture *or* every E. coli
    # isolate. General — no per-organism or per-specimen table; the organism
    # is grounded to MIMIC's scientific-binomial ``org_name`` upstream by the
    # decomposer (see prompts.py microbiology organism-grounding rule).
    for attr in (concept.attributes or []):
        term = attr.strip()
        if not term:
            continue
        term = _microbiology_match_term(term)
        where_clauses.append(
            f"({backend.ilike('m.spec_type_desc')} OR {backend.ilike('m.org_name')})"
        )
        params.extend([f"%{term}%", f"%{term}%"])

    # Result-status qualifier: a culture is *positive* iff an organism was
    # isolated (org_name IS NOT NULL); *negative* / no-growth is org_name IS
    # NULL. Grounded in the MIMIC schema, so "positive blood culture" counts
    # cultures that grew, not cultures merely drawn (a ~4x narrowing in sepsis).
    if concept.culture_status == "positive":
        where_clauses.append("m.org_name IS NOT NULL")
    elif concept.culture_status == "negative":
        where_clauses.append("m.org_name IS NULL")

    where_clauses.extend(filter_where)
    params.extend(filter_params)

    sql = (
        f"SELECT {select_cols} "
        f"FROM {t('microbiologyevents')} m {' '.join(joins)} "
        f"WHERE {' AND '.join(where_clauses)} "
        f"{group_by_clause}"
    ).strip()

    return SqlFastpathQuery(sql=sql, params=params, columns=columns)


# ---------------------------------------------------------------------------
# Diagnosis (count + list)
# ---------------------------------------------------------------------------


def _compile_diagnosis_count(
    cq: CompetencyQuestion,
    concept: ClinicalConcept,
    names: list[str],
    sql_fn: str,
    backend: Any,
    registry: OperationRegistry,
    *,
    icd_codes: list[str] | None = None,
    enable_mcp_grounding: bool = False,
) -> SqlFastpathQuery:
    if sql_fn != "COUNT":
        raise ValueError(
            f"sql_fastpath supports only COUNT for diagnosis; got {sql_fn}"
        )

    t = backend.table
    group_by_col, group_by_params = _comparison_group_by_col(
        cq, registry, backend, enable_mcp_grounding=enable_mcp_grounding,
    )
    filter_joins, filter_where, filter_params, filter_needs_patients, filter_has_readmission = \
        _filter_fragment(cq, backend, registry, enable_mcp_grounding=enable_mcp_grounding)

    needs_patients = filter_needs_patients or _needs_patients_axis(group_by_col)
    needs_readmission = (
        _needs_readmission_axis(group_by_col) and not filter_has_readmission
    )

    joins: list[str] = [
        f"JOIN {t('d_icd_diagnoses')} dd "
        f"ON di.icd_code = dd.icd_code AND di.icd_version = dd.icd_version",
        f"JOIN {t('admissions')} a ON di.hadm_id = a.hadm_id",
    ]
    if needs_patients:
        joins.append(f"JOIN {t('patients')} p ON a.subject_id = p.subject_id")
    if needs_readmission:
        joins.append(
            f"JOIN {backend.readmission_labels_expr()} rl ON a.hadm_id = rl.hadm_id"
        )
    joins.extend(filter_joins)

    if group_by_col is None:
        select_cols = "COUNT(DISTINCT di.hadm_id) AS count_value"
        columns = ["count_value"]
        group_by_clause = ""
    else:
        select_cols = (
            f"{group_by_col} AS group_value, "
            f"COUNT(DISTINCT di.hadm_id) AS avg_value, "
            f"COUNT(DISTINCT di.hadm_id) AS count"
        )
        columns = ["group_value", "avg_value", "count"]
        group_by_clause = "GROUP BY group_value"

    # Each resolved name contributes (title ILIKE OR icd_code LIKE), all OR-combined.
    per_name_clauses: list[str] = []
    per_name_params: list[str] = []
    for n in names:
        per_name_clauses.append(
            f"({backend.ilike('dd.long_title')} OR di.icd_code LIKE ?)"
        )
        per_name_params.extend([f"%{n}%", f"{n}%"])
    like_or = "(" + " OR ".join(per_name_clauses) + ")"

    # When OMOPHub-grounded ICD codes are supplied, OR them in alongside
    # the LIKE clause. Net WHERE shape:
    #   ((<code prefix matches>) OR <like_or>) AND <filters>
    # ``icd_autocode`` returns DOTTED, often category-level codes ('I63',
    # 'E11.1'); MIMIC stores them UNDOTTED and BILLABLE ('I6300', 'E1110'). An
    # exact ``IN`` on the dotted/category code therefore matched *nothing* — so
    # each grounded code is normalized (dots stripped, uppercased) and matched
    # as a PREFIX, so the grounded category catches its billable descendants
    # (mirrors the cohort-registry Tier-1 prefix match). The LIKE branch still
    # catches ICD-9 admissions outside OMOPHub's ICD10CM-only coverage. Empty
    # list defensively treated as "no grounding".
    # ``group_by_col`` (when present) leads the SELECT, so its params precede
    # the WHERE-clause params positionally.
    params: list[Any] = list(group_by_params)
    code_prefix_clauses: list[str] = []
    code_prefix_params: list[str] = []
    if icd_codes:
        from src.conversational.health_evidence.cohorts import normalize_icd_prefix
        for c in icd_codes:
            norm = normalize_icd_prefix(c)
            if not norm:
                continue
            code_prefix_clauses.append("di.icd_code LIKE ?")
            code_prefix_params.append(norm + "%")
    if code_prefix_clauses:
        # Codes-only when grounded: drop the broad title-LIKE so a precise ICD
        # grounding isn't re-polluted by title matches (e.g. nontraumatic SAH
        # I60+430 must not re-admit traumatic S06.6). The title-LIKE stays the
        # fallback ONLY when nothing grounds.
        cohort_clause = f"({' OR '.join(code_prefix_clauses)})"
        params.extend(code_prefix_params)
    else:
        cohort_clause = like_or
        params.extend(per_name_params)
    where_clauses: list[str] = [cohort_clause]
    # "Primary diagnosis of X" → restrict to the principal diagnosis
    # (``seq_num = 1``); higher seq_nums are secondary diagnoses / comorbidities.
    # General for any condition (no per-condition logic).
    if concept.primary_only:
        where_clauses.append("di.seq_num = 1")
    where_clauses.extend(filter_where)
    params.extend(filter_params)

    sql = (
        f"SELECT {select_cols} "
        f"FROM {t('diagnoses_icd')} di {' '.join(joins)} "
        f"WHERE {' AND '.join(where_clauses)} "
        f"{group_by_clause}"
    ).strip()

    return SqlFastpathQuery(sql=sql, params=params, columns=columns)


def _compile_diagnosis_list(
    cq: CompetencyQuestion,
    concept: ClinicalConcept,
    names: list[str],
    backend: Any,
    registry: OperationRegistry,
    *,
    enable_mcp_grounding: bool = False,
) -> SqlFastpathQuery:
    """Patient-list-by-diagnosis matching the SPARQL template output shape."""
    if cq.scope == "comparison":
        raise ValueError(
            "sql_fastpath diagnosis-list does not support comparison scope"
        )

    t = backend.table
    filter_joins, filter_where, filter_params, filter_needs_patients, _ = \
        _filter_fragment(cq, backend, registry, enable_mcp_grounding=enable_mcp_grounding)

    joins: list[str] = [
        f"LEFT JOIN {t('d_icd_diagnoses')} dd "
        f"ON di.icd_code = dd.icd_code AND di.icd_version = dd.icd_version",
        f"JOIN {t('admissions')} a ON di.hadm_id = a.hadm_id",
    ]
    if filter_needs_patients:
        joins.append(f"JOIN {t('patients')} p ON a.subject_id = p.subject_id")
    joins.extend(filter_joins)

    columns = ["subjectId", "hadmId", "icdCode", "longTitle"]
    select_cols = (
        "a.subject_id AS subjectId, "
        "di.hadm_id AS hadmId, "
        "di.icd_code AS icdCode, "
        "dd.long_title AS longTitle"
    )

    per_name_clauses: list[str] = []
    per_name_params: list[str] = []
    for n in names:
        per_name_clauses.append(
            f"({backend.ilike('dd.long_title')} OR di.icd_code LIKE ?)"
        )
        per_name_params.extend([f"%{n}%", f"{n}%"])
    where_clauses: list[str] = ["(" + " OR ".join(per_name_clauses) + ")"]
    params: list[Any] = list(per_name_params)
    where_clauses.extend(filter_where)
    params.extend(filter_params)

    sql = (
        f"SELECT {select_cols} "
        f"FROM {t('diagnoses_icd')} di {' '.join(joins)} "
        f"WHERE {' AND '.join(where_clauses)} "
        f"ORDER BY a.subject_id, di.hadm_id"
    ).strip()

    return SqlFastpathQuery(sql=sql, params=params, columns=columns)


# ---------------------------------------------------------------------------
# Outcome (mortality)
# ---------------------------------------------------------------------------


def _outcome_flag(concept: ClinicalConcept) -> tuple[str, str, bool]:
    """Ground an ``outcome`` concept to the binary admission-level flag whose
    rate the question asks for: returns ``(flag_sql, flag_alias, needs_rl)``.

    Two outcome families map onto MIMIC's admission schema. In-hospital
    *mortality* is ``admissions.hospital_expire_flag``. *Readmission* is the
    30-/60-day label the readmission-labels CTE computes (so it needs the ``rl``
    join); the window is read from the concept name and defaults to 30-day, so
    "30-day readmission rate" and "60-day readmission" both resolve. Anything
    else falls back to mortality — the historical default. This is the schema
    grounding of an outcome, the direct analogue of the comparison-axis →
    ``sql_group_by`` mapping, not a curated synonym list."""
    name = (concept.name or "").lower()
    if "readmiss" in name or "readmit" in name:
        window = "60" if "60" in name else "30"
        return f"rl.readmitted_{window}d", f"readmitted{window}", True
    return "a.hospital_expire_flag", "expired", False


def _compile_outcome_rate(
    cq: CompetencyQuestion,
    concept: ClinicalConcept,
    sql_fn: str,
    backend: Any,
    registry: OperationRegistry,
    *,
    enable_mcp_grounding: bool = False,
) -> SqlFastpathQuery:
    """Binary-outcome rate over the cohort — the ``(<flag>, count, fraction)``
    shape. Covers in-hospital *mortality* (``hospital_expire_flag``) and hospital
    *readmission* (30-/60-day labels via the readmission-labels CTE); the flag is
    selected by ``_outcome_flag`` from the outcome concept. We ignore the
    caller's aggregation keyword and always emit COUNT DISTINCT per flag value
    PLUS an in-query proportion, so a "... RATE" question is answered with an
    actual rate — the ``flag = 1`` row's ``fraction`` — rather than a raw count
    or a division left to the answerer LLM. The window ``SUM`` over the grouped
    counts is the cohort total; ``NULLIF`` guards the empty-cohort divide.

    Comparison scope: when the CQ carries a ``comparison_field`` the outcome is
    grouped by that axis too and the ``fraction`` window is *partitioned by the
    axis* — so each group's ``flag = 1`` fraction is that group's own rate, not
    its share of the whole cohort."""
    t = backend.table
    flag_expr, flag_alias, flag_needs_rl = _outcome_flag(concept)
    group_by_col, group_by_params = _comparison_group_by_col(
        cq, registry, backend, enable_mcp_grounding=enable_mcp_grounding,
    )
    filter_joins, filter_where, filter_params, filter_needs_patients, filter_has_readmission = \
        _filter_fragment(cq, backend, registry, enable_mcp_grounding=enable_mcp_grounding)

    needs_patients = filter_needs_patients or _needs_patients_axis(group_by_col)
    # rl is needed when the OUTCOME flag is a readmission label, the comparison
    # axis is a readmission flag, or a filter referenced ``rl.`` without having
    # joined it — but never twice.
    needs_readmission = (
        flag_needs_rl
        or _needs_readmission_axis(group_by_col)
        or any("rl." in w for w in filter_where)
    ) and not filter_has_readmission

    joins: list[str] = []
    if needs_patients:
        joins.append(f"JOIN {t('patients')} p ON a.subject_id = p.subject_id")
    if needs_readmission:
        joins.append(
            f"JOIN {backend.readmission_labels_expr()} rl ON a.hadm_id = rl.hadm_id"
        )
    joins.extend(filter_joins)

    where_clauses = list(filter_where)
    where_part = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    if group_by_col is None:
        params = list(filter_params)
        sql = (
            f"SELECT {flag_expr} AS {flag_alias}, "
            f"COUNT(DISTINCT a.hadm_id) AS count, "
            f"COUNT(DISTINCT a.hadm_id) / "
            f"NULLIF(SUM(COUNT(DISTINCT a.hadm_id)) OVER (), 0) AS fraction "
            f"FROM {t('admissions')} a {' '.join(joins)} "
            f"{where_part} "
            f"GROUP BY {flag_expr}"
        ).strip()
        columns = [flag_alias, "count", "fraction"]
    else:
        # Partition the share window by the comparison axis so each group's
        # fraction is its OWN rate, not its slice of the whole cohort.
        #
        # The axis is computed ONCE per admission in an inner subquery (as the
        # ``group_value`` column), then the outer query groups + windows over
        # that plain column. This keeps the axis expression — and its params —
        # in a single position, which matters for the dynamic ``condition`` axis
        # whose ``CASE WHEN EXISTS(...)`` correlates on ``a.hadm_id``: re-emitting
        # a correlated subquery in GROUP BY/PARTITION BY is rejected by DuckDB
        # ("column must appear in the GROUP BY"). For the fixed-column axes this
        # is equivalent to the prior direct GROUP BY (grouping by a renamed
        # column), so their results are unchanged.
        params = list(group_by_params) + list(filter_params)
        inner = (
            f"SELECT a.hadm_id AS hadm_id, "
            f"{group_by_col} AS group_value, "
            f"{flag_expr} AS flag "
            f"FROM {t('admissions')} a {' '.join(joins)} "
            f"{where_part}"
        ).strip()
        sql = (
            f"SELECT group_value, flag AS {flag_alias}, "
            f"COUNT(DISTINCT hadm_id) AS count, "
            f"COUNT(DISTINCT hadm_id) / "
            f"NULLIF(SUM(COUNT(DISTINCT hadm_id)) OVER (PARTITION BY group_value), 0) "
            f"AS fraction "
            f"FROM ({inner}) sub "
            f"GROUP BY group_value, flag"
        ).strip()
        columns = ["group_value", flag_alias, "count", "fraction"]

    return SqlFastpathQuery(sql=sql, params=params, columns=columns)


# ---------------------------------------------------------------------------
# Event ordering (most-common temporal order of N events)
# ---------------------------------------------------------------------------


def _is_bigquery_backend(backend: Any) -> bool:
    """Heuristic: BigQuery's ``table()`` returns a backtick-quoted FQN
    (``` `physionet-data.…` ```); DuckDB returns a bare table name. Used to
    decide whether the GCS-derived table FQN is available."""
    try:
        return backend.table("admissions").startswith("`")
    except Exception:  # noqa: BLE001
        return False


def _derived_table(backend: Any, name: str) -> str | None:
    """Fully-qualified name for a ``mimiciv_3_1_derived`` table on BigQuery.

    Returns ``None`` on backends that don't carry the derived dataset (DuckDB /
    the synthetic offline fixture), so the caller can skip the GCS sub-query
    rather than emit SQL against a table that doesn't exist. There's no derived
    entry in ``_BQ_TABLES`` (those are raw hosp/icu sources), so the FQN is
    composed here against the same project the raw tables live in."""
    if _is_bigquery_backend(backend):
        return f"`physionet-data.mimiciv_3_1_derived.{name}`"
    return None


def _event_ordering_event_sql(
    concept: ClinicalConcept, backend: Any,
) -> tuple[str, str, str, str] | None:
    """Ground one event concept to ``(join, time_expr, where, label)`` for an
    event_ordering sub-query, or ``None`` if it can't be grounded.

    Each tuple plugs into ``SELECT a.hadm_id, '<label>' AS event_name,
    MIN(<time_expr>) AS event_time FROM admissions a <join> WHERE <where> …``.
    Grounding is by lower-cased substring of the concept name (sufficient for
    the demo); ``where`` is the event-specific predicate (cohort filters are
    AND-ed on by the caller), and ``label`` is the human event name surfaced in
    the post-processed ordering.
    """
    t = backend.table
    name = (concept.name or "").lower()

    if "intubation" in name or "mechanical ventilation" in name or "intubat" in name:
        join = f"JOIN {t('procedureevents')} pe ON pe.hadm_id = a.hadm_id"
        return join, "pe.starttime", "pe.itemid IN (225792, 224385)", concept.name

    if "mannitol" in name or "hypertonic" in name or "hyperosmolar" in name or "osmotic" in name:
        join = f"JOIN {t('prescriptions')} pr ON pr.hadm_id = a.hadm_id"
        where = "(LOWER(pr.drug) LIKE '%mannitol%' OR LOWER(pr.drug) LIKE '%hypertonic%')"
        return join, "pr.starttime", where, concept.name

    if "gcs" in name or "glasgow" in name or "coma scale" in name:
        gcs = _derived_table(backend, "gcs")
        if gcs is None:
            # Offline / DuckDB: the derived GCS table isn't available, so this
            # event can't be grounded. Returning None drops it from the UNION;
            # the post-processor still works on the remaining events.
            return None
        # First GCS DROP of ≥2 points per ICU stay → its charttime, mapped to
        # hadm_id via icustays. The LAG window finds the first step-down.
        drop = (
            f"(SELECT icu.hadm_id AS hadm_id, g.charttime AS charttime FROM ("
            f"  SELECT stay_id, charttime, gcs - LAG(gcs) OVER ("
            f"    PARTITION BY stay_id ORDER BY charttime) AS gcs_delta"
            f"  FROM {gcs}"
            f") g JOIN {t('icustays')} icu ON g.stay_id = icu.stay_id "
            f"WHERE g.gcs_delta <= -2)"
        )
        join = f"JOIN {drop} gcsdrop ON gcsdrop.hadm_id = a.hadm_id"
        return join, "gcsdrop.charttime", "1 = 1", concept.name

    return None


def _compile_event_ordering(
    cq: CompetencyQuestion,
    backend: Any,
    registry: OperationRegistry,
    *,
    enable_mcp_grounding: bool = False,
) -> SqlFastpathQuery:
    """Per-patient FIRST time of each named event, as a UNION ALL the
    ``event_ordering`` post-processor turns into the most-common order.

    One sub-query per event concept in ``cq.clinical_concepts``:

        SELECT a.hadm_id, '<event label>' AS event_name, MIN(<time>) AS event_time
        FROM admissions a <event join> [cohort-filter joins]
        WHERE <event predicate> [AND cohort-filter predicates]
        GROUP BY a.hadm_id

    The cohort ``patient_filters`` (e.g. ICH diagnosis + ICP monitoring) are
    applied in EVERY sub-query as correlated ``EXISTS`` predicates (via
    ``_split_condition_exists``) rather than as JOINs, so every event time is
    within the same cohort WITHOUT introducing per-filter table aliases. The
    JOIN form hardcodes ``di``/``dd`` inside ``_compile_diagnosis``, so two
    diagnosis filters (e.g. intracerebral hemorrhage + ICP monitoring) collided
    on alias ``di``; the EXISTS form has no such collision (each sub-condition
    self-contains its inner aliases). Events that can't be grounded on this
    backend (e.g. GCS on the offline fixture with no derived dataset) are
    skipped; the remaining events still produce an ordering.

    Columns are ``["hadm_id", "event_name", "event_time"]`` — the post-processor
    is keyed on exactly these.
    """
    if len(cq.clinical_concepts) < 2:
        raise ValueError(
            "event_ordering needs at least 2 event concepts to order"
        )

    # Compile each cohort filter to a correlated EXISTS on ``a.hadm_id`` (the
    # outer admissions alias), reusing ``_split_condition_exists``. Build the
    # ctx exactly like ``_filter_fragment`` does — reading the resolver /
    # derived-formula grounding from the ambient contextvar — and inject the
    # registry so the generic EXISTS-reuse path can look up child filter ops.
    resolver, derived_formulas = _FILTER_GROUNDING_CTX.get()
    filter_ctx = FilterCompileContext(
        backend=backend, enable_mcp_grounding=enable_mcp_grounding,
        resolver=resolver, derived_formulas=derived_formulas,
        registry=registry,
    )
    cohort_exists: list[str] = []
    cohort_exists_params: list[Any] = []
    for f in (cq.patient_filters or []):
        exists_sql, exists_params = _split_condition_exists(f, filter_ctx)
        cohort_exists.append(exists_sql)
        cohort_exists_params.extend(exists_params)

    sub_selects: list[str] = []
    params: list[Any] = []
    for concept in cq.clinical_concepts:
        grounded = _event_ordering_event_sql(concept, backend)
        if grounded is None:
            continue
        join, time_expr, ev_where, label = grounded
        where_clauses = [ev_where] + cohort_exists
        # The event-grounding ``label`` is a fixed string literal — safe to
        # interpolate (it's an event NAME from the decomposer, single-quoted).
        safe_label = label.replace("'", "''")
        sub = (
            f"SELECT a.hadm_id AS hadm_id, '{safe_label}' AS event_name, "
            f"MIN({time_expr}) AS event_time "
            f"FROM {backend.table('admissions')} a {join} "
            f"WHERE {' AND '.join(where_clauses)} "
            f"GROUP BY a.hadm_id"
        ).strip()
        sub_selects.append(sub)
        # The cohort-filter EXISTS params repeat once per sub-query (the same
        # EXISTS predicates are applied in each), in sub-query order.
        params.extend(cohort_exists_params)

    if not sub_selects:
        raise ValueError(
            "event_ordering could not ground any of the event concepts "
            f"{[c.name for c in cq.clinical_concepts]!r} on this backend"
        )

    sql = "\nUNION ALL\n".join(sub_selects)
    return SqlFastpathQuery(
        sql=sql,
        params=params,
        columns=["hadm_id", "event_name", "event_time"],
    )
