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

from dataclasses import dataclass
from typing import Any

from src.conversational.models import ClinicalConcept, CompetencyQuestion
from src.conversational.operations import (
    FilterCompileContext,
    OperationRegistry,
)


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
        return _compile_diagnosis_list(cq, concept, names, backend, registry)

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
        return _compile_event_aggregate(cq, concept, names, sql_fn, backend, registry)
    if concept.concept_type == "drug":
        return _compile_drug_aggregate(cq, concept, names, sql_fn, backend, registry)
    if concept.concept_type == "microbiology":
        return _compile_microbiology_aggregate(cq, concept, names, sql_fn, backend, registry)
    if concept.concept_type == "diagnosis":
        return _compile_diagnosis_count(cq, concept, names, sql_fn, backend, registry)
    if concept.concept_type == "outcome":
        return _compile_outcome_mortality(cq, concept, sql_fn, backend, registry)

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
    cq: CompetencyQuestion, backend: Any, registry: OperationRegistry,
) -> tuple[list[str], list[str], list[Any], bool, bool]:
    """Compile patient_filters to SQL fragments.

    Returns ``(joins, where, params, needs_patients, needs_readmission)``
    where ``needs_readmission`` tracks whether any filter already joined
    the readmission_labels CTE — so the GROUP BY path doesn't double-join.
    """
    ctx = FilterCompileContext(backend=backend)
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
) -> SqlFastpathQuery:
    """AVG/MAX/MIN/COUNT over biomarker (labevents) or vital (chartevents)."""
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
        _filter_fragment(cq, backend, registry)

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

    # SELECT and GROUP BY shape.
    if group_by_col is None:
        select_cols = f"{sql_fn}({event_alias}.valuenum) AS {_AGG_COLUMN_NAME[sql_fn]}"
        columns = [_AGG_COLUMN_NAME[sql_fn]]
        group_by_clause = ""
    else:
        # Comparison: always AVG + COUNT, matching comparison_by_field SPARQL.
        select_cols = (
            f"{group_by_col} AS group_value, "
            f"AVG({event_alias}.valuenum) AS avg_value, "
            f"COUNT({event_alias}.valuenum) AS count"
        )
        columns = ["group_value", "avg_value", "count"]
        group_by_clause = f"GROUP BY {group_by_col}"

    # WHERE: concept match + non-null value + cohort filters.
    name_clause, name_params = _ilike_any(backend, name_col, names)
    where_clauses: list[str] = [
        name_clause,
        f"{event_alias}.valuenum IS NOT NULL",
    ]
    params: list[Any] = list(name_params)
    where_clauses.extend(filter_where)
    params.extend(filter_params)

    sql = (
        f"SELECT {select_cols} "
        f"FROM {event_table} {join_dict} {' '.join(joins)} "
        f"WHERE {' AND '.join(where_clauses)} "
        f"{group_by_clause}"
    ).strip()

    return SqlFastpathQuery(sql=sql, params=params, columns=columns)


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
        _filter_fragment(cq, backend, registry)

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

    if group_by_col is None:
        select_cols = f"COUNT(*) AS count_value"
        columns = ["count_value"]
        group_by_clause = ""
    else:
        select_cols = (
            f"{group_by_col} AS group_value, "
            f"COUNT(*) AS avg_value, "
            f"COUNT(*) AS count"
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


def _compile_microbiology_aggregate(
    cq: CompetencyQuestion,
    concept: ClinicalConcept,
    names: list[str],
    sql_fn: str,
    backend: Any,
    registry: OperationRegistry,
) -> SqlFastpathQuery:
    if sql_fn != "COUNT":
        raise ValueError(
            f"sql_fastpath supports only COUNT for microbiology; got {sql_fn}"
        )

    t = backend.table
    group_by_col = _comparison_group_by_col(cq, registry)
    filter_joins, filter_where, filter_params, filter_needs_patients, filter_has_readmission = \
        _filter_fragment(cq, backend, registry)

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

    if group_by_col is None:
        select_cols = "COUNT(*) AS count_value"
        columns = ["count_value"]
        group_by_clause = ""
    else:
        select_cols = (
            f"{group_by_col} AS group_value, "
            f"COUNT(*) AS avg_value, "
            f"COUNT(*) AS count"
        )
        columns = ["group_value", "avg_value", "count"]
        group_by_clause = f"GROUP BY {group_by_col}"

    # Microbiology matches both specimen type and organism name, so each
    # resolved name contributes two ILIKE clauses OR-combined.
    per_name_clauses: list[str] = []
    per_name_params: list[str] = []
    for n in names:
        per_name_clauses.append(
            f"({backend.ilike('m.spec_type_desc')} OR {backend.ilike('m.org_name')})"
        )
        per_name_params.extend([f"%{n}%", f"%{n}%"])
    where_clauses: list[str] = ["(" + " OR ".join(per_name_clauses) + ")"]
    params: list[Any] = list(per_name_params)
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
) -> SqlFastpathQuery:
    if sql_fn != "COUNT":
        raise ValueError(
            f"sql_fastpath supports only COUNT for diagnosis; got {sql_fn}"
        )

    t = backend.table
    group_by_col = _comparison_group_by_col(cq, registry)
    filter_joins, filter_where, filter_params, filter_needs_patients, filter_has_readmission = \
        _filter_fragment(cq, backend, registry)

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
    where_clauses: list[str] = ["(" + " OR ".join(per_name_clauses) + ")"]
    params: list[Any] = list(per_name_params)
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
) -> SqlFastpathQuery:
    """Patient-list-by-diagnosis matching the SPARQL template output shape."""
    if cq.scope == "comparison":
        raise ValueError(
            "sql_fastpath diagnosis-list does not support comparison scope"
        )

    t = backend.table
    filter_joins, filter_where, filter_params, filter_needs_patients, _ = \
        _filter_fragment(cq, backend, registry)

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
) -> SqlFastpathQuery:
    """Mortality count — matches the SPARQL ``mortality_count`` template's
    (``expired``, ``count``) shape. We ignore the caller's aggregation keyword
    and always emit COUNT DISTINCT per expired flag, because that's the
    clinically meaningful answer."""
    t = backend.table
    filter_joins, filter_where, filter_params, filter_needs_patients, filter_has_readmission = \
        _filter_fragment(cq, backend, registry)

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
        f"COUNT(DISTINCT a.hadm_id) AS count "
        f"FROM {t('admissions')} a {' '.join(joins)} "
        f"{where_part} "
        f"GROUP BY a.hospital_expire_flag"
    ).strip()

    return SqlFastpathQuery(
        sql=sql, params=params, columns=["expired", "count"],
    )
