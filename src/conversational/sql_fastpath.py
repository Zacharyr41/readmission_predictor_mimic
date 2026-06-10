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

import re
from dataclasses import dataclass
from typing import Any

from src.conversational.models import ClinicalConcept, CompetencyQuestion
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
        return _compile_outcome_mortality(
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
    cq: CompetencyQuestion, registry: OperationRegistry,
) -> str | None:
    """Return the SQL GROUP BY column for a comparison CQ, or None if the
    CQ isn't a comparison scope."""
    if cq.scope != "comparison":
        return None
    if not cq.comparison_field:
        raise ValueError("comparison scope requires comparison_field")
    axis_op = registry.get("comparison_axis", cq.comparison_field)
    if axis_op is None or getattr(axis_op, "sql_group_by", None) is None:
        raise ValueError(
            f"comparison axis {cq.comparison_field!r} has no sql_group_by"
        )
    return axis_op.sql_group_by


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
    ctx = FilterCompileContext(
        backend=backend, enable_mcp_grounding=enable_mcp_grounding,
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

    group_by_col = _comparison_group_by_col(cq, registry)
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
        group_by_clause = f"GROUP BY {group_by_col}"

    params: list[Any] = select_params + where_params

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
    group_by_col = _comparison_group_by_col(cq, registry)
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
        group_by_clause = f"GROUP BY {group_by_col}"

    name_clause, name_params = _ilike_any(backend, "pr.drug", names)
    where_clauses: list[str] = [name_clause]
    params: list[Any] = list(name_params)
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
    group_by_col = _comparison_group_by_col(cq, registry)
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
        group_by_clause = f"GROUP BY {group_by_col}"

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
    params: list[Any] = list(per_name_params)

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
    group_by_col = _comparison_group_by_col(cq, registry)
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
        group_by_clause = f"GROUP BY {group_by_col}"

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
    #   ((di.icd_code IN (?,?,...)) OR <like_or>) AND <filters>
    # The IN-list catches grounded ICD-10 admissions precisely; the LIKE
    # branch catches ICD-9 admissions whose codes aren't in OMOPHub's
    # ICD10CM-only coverage. Empty list defensively treated as "no
    # grounding" rather than emitting ``IN ()``.
    params: list[Any] = []
    if icd_codes:
        in_placeholders = ", ".join(["?"] * len(icd_codes))
        cohort_clause = f"((di.icd_code IN ({in_placeholders})) OR {like_or})"
        params.extend(icd_codes)
    else:
        cohort_clause = like_or
    where_clauses: list[str] = [cohort_clause]
    params.extend(per_name_params)
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


def _compile_outcome_mortality(
    cq: CompetencyQuestion,
    concept: ClinicalConcept,
    sql_fn: str,
    backend: Any,
    registry: OperationRegistry,
    *,
    enable_mcp_grounding: bool = False,
) -> SqlFastpathQuery:
    """Mortality breakdown — the SPARQL ``mortality_count`` template's
    (``expired``, ``count``) shape, enriched with each bucket's ``fraction`` of
    the cohort. We ignore the caller's aggregation keyword and always emit
    COUNT DISTINCT per expired flag (the clinically meaningful split), PLUS an
    in-query proportion so a "mortality / survival RATE" question is answered
    with an actual rate — the ``expired = 1`` row's ``fraction`` — rather than
    leaving the division to the answerer LLM. The window ``SUM`` over the grouped
    counts is the cohort total; ``NULLIF`` guards the (unreachable-when-any-row-
    exists) empty-cohort divide."""
    t = backend.table
    filter_joins, filter_where, filter_params, filter_needs_patients, filter_has_readmission = \
        _filter_fragment(cq, backend, registry, enable_mcp_grounding=enable_mcp_grounding)

    joins: list[str] = []
    if filter_needs_patients:
        joins.append(f"JOIN {t('patients')} p ON a.subject_id = p.subject_id")
    if filter_has_readmission is False and any(
        "rl." in w for w in filter_where
    ):
        joins.append(
            f"JOIN {backend.readmission_labels_expr()} rl ON a.hadm_id = rl.hadm_id"
        )
    joins.extend(filter_joins)

    where_clauses = list(filter_where)
    params = list(filter_params)
    where_part = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    sql = (
        f"SELECT a.hospital_expire_flag AS expired, "
        f"COUNT(DISTINCT a.hadm_id) AS count, "
        f"COUNT(DISTINCT a.hadm_id) / "
        f"NULLIF(SUM(COUNT(DISTINCT a.hadm_id)) OVER (), 0) AS fraction "
        f"FROM {t('admissions')} a {' '.join(joins)} "
        f"{where_part} "
        f"GROUP BY a.hospital_expire_flag"
    ).strip()

    return SqlFastpathQuery(
        sql=sql, params=params, columns=["expired", "count", "fraction"],
    )
