"""Filter operations — the one place to declare a supported cohort filter field.

Each ``register_filter_*`` function here writes a single filter operation to
the given ``OperationRegistry``. Adding a new filter field to the pipeline is
a matter of adding one function here and calling it from ``register_all``.

The SQL produced by each filter's ``compile`` intentionally matches the exact
predicates previously written by hand in ``extractor._get_filtered_hadm_ids``.
Phase 1 parity tests compare the registry output against the pre-existing
hand-rolled SQL on a synthetic DuckDB to catch any accidental semantic drift.
"""

from __future__ import annotations

from src.conversational.models import PatientFilter
from src.conversational.operations import (
    COMPARISON_OPERATORS,
    FilterCompileContext,
    FilterFragment,
    FilterOperation,
    OperationRegistry,
)


# ---------------------------------------------------------------------------
# Existing (pre-registry) filter fields — must produce identical SQL
# ---------------------------------------------------------------------------


def _compile_age(f: PatientFilter, ctx: FilterCompileContext) -> FilterFragment:
    # Existing: needs patients JOIN, predicate "p.anchor_age {op} ?", value int-coerced.
    # Operator is constrained by the registered ``operators`` set, so interpolation
    # into the SQL string is safe.
    return FilterFragment(
        where=[f"{ctx.patient_alias}.anchor_age {f.operator} ?"],
        params=[int(f.value)],
        needs_patients=True,
    )


def _compile_gender(f: PatientFilter, ctx: FilterCompileContext) -> FilterFragment:
    # Existing: needs patients JOIN, predicate "p.gender = ?", value uppercased.
    return FilterFragment(
        where=[f"{ctx.patient_alias}.gender = ?"],
        params=[f.value.upper()],
        needs_patients=True,
    )


def _compile_diagnosis(f: PatientFilter, ctx: FilterCompileContext) -> FilterFragment:
    """Existing: joins diagnoses_icd + d_icd_diagnoses, predicate matches
    either the long_title (case-insensitive) OR the ICD code prefix.

    Inc 9.2 — when ``ctx.enable_mcp_grounding`` is True AND the filter
    is a non-empty ``contains`` clause, ground the natural-language
    phrase to ICD-10 admissions and emit a parallel-OR clause alongside
    the existing title-LIKE branch.

    Inc 10 — grounding consults two sources in priority order:

    1. **Cohort registry** (``data/mappings/clinical_cohorts.json`` via
       :func:`resolve_cohort_name`). When the value matches a
       registered cohort, use the registry's ICD prefixes as ``LIKE``
       patterns. The registry is what
       :func:`mimic_distribution_lookup` already uses, so live SQL and
       the catalog reference query the SAME cohort definition.
    2. **OMOPHub ``icd_autocode``** (semantic search). Used only when
       the value doesn't resolve to any registered cohort name.
       Returns confidence-ranked specific codes. Useful for arbitrary
       phrases (rare conditions, narrow diagnoses) the registry
       doesn't cover.

    The LIKE-on-title branch is always retained as a parallel-OR
    sibling so ICD-9 admissions whose codes aren't in OMOPHub's
    ICD10CM-only coverage still match.

    Falls back to LIKE-only when grounding is disabled, the value is
    empty/non-string, or both the registry and OMOPHub return nothing.
    """
    t = ctx.backend.table
    joins = [
        f"JOIN {t('diagnoses_icd')} di ON {ctx.admission_alias}.hadm_id = di.hadm_id "
        f"JOIN {t('d_icd_diagnoses')} dd ON di.icd_code = dd.icd_code "
        f"AND di.icd_version = dd.icd_version"
    ]

    # Existing LIKE clause + params — always present so ICD-9
    # admissions stay matchable via title text.
    like_clause = f"({ctx.backend.ilike('dd.long_title')} OR di.icd_code LIKE ?)"
    like_params = [f"%{f.value}%", f"{f.value}%"]

    grounded = _maybe_ground_diagnosis_filter_clause(f, ctx)
    if grounded is not None:
        clause_sql, clause_params = grounded
        clause = f"({clause_sql} OR {like_clause})"
        params = list(clause_params) + like_params
    else:
        clause = like_clause
        params = like_params

    return FilterFragment(joins=joins, where=[clause], params=params)


def _maybe_ground_diagnosis_filter_clause(
    f: PatientFilter, ctx: FilterCompileContext,
) -> tuple[str, list] | None:
    """Build a SQL clause + params for an ICD-grounded diagnosis filter.

    Two-tier resolution (Inc 10):

    1. Cohort registry: ``resolve_cohort_name`` matches the filter's
       value against ``data/mappings/clinical_cohorts.json``. On hit,
       emit ``(di.icd_version = 10 AND di.icd_code LIKE 'A41%') OR …``
       across all ICD-10 + ICD-9 prefixes — this is the same definition
       the catalog's ``mimic_distribution_lookup`` uses.
    2. OMOPHub ``icd_autocode``: when the registry doesn't match, fall
       back to confidence-filtered specific codes from semantic search.
       Emits ``di.icd_code IN (?, ?, ?)``.

    Returns ``None`` to signal the caller to fall back to LIKE-only.
    Never raises.
    """
    if not ctx.enable_mcp_grounding:
        return None
    if f.operator != "contains":
        return None
    value = f.value
    if not isinstance(value, str) or not value:
        return None

    # Tier 1 — cohort registry. The lazy import keeps a hard dependency
    # off operations_filters' module-load path.
    from src.conversational.health_evidence.cohorts import (
        load_cohorts, normalize_icd_prefix, resolve_cohort_name,
    )
    cohort_name = resolve_cohort_name(value)
    if cohort_name is not None:
        defn = load_cohorts().get(cohort_name) or {}
        icd10 = list(defn.get("icd10_prefixes") or [])
        icd9 = list(defn.get("icd9_prefixes") or [])
        clauses: list[str] = []
        params: list[str] = []
        for p in icd10:
            clauses.append("(di.icd_version = 10 AND di.icd_code LIKE ?)")
            params.append(normalize_icd_prefix(p) + "%")
        for p in icd9:
            clauses.append("(di.icd_version = 9 AND di.icd_code LIKE ?)")
            params.append(normalize_icd_prefix(p) + "%")
        if clauses:
            return f"({' OR '.join(clauses)})", params
        # Registry entry has no prefixes — fall through to OMOPHub
        # rather than emit no clause at all. Defensive only; current
        # registry has prefixes for every cohort.

    # Tier 2 — OMOPHub icd_autocode (semantic search).
    from src.conversational.concept_resolver import (
        _ICD_CONFIDENCE_THRESHOLD, _cached_icd_autocode,
    )
    try:
        cached = _cached_icd_autocode(value.lower(), "10")
    except LookupError:
        return None
    accepted_codes: list[str] = []
    for code, conf in cached:
        if conf is None or conf >= _ICD_CONFIDENCE_THRESHOLD:
            accepted_codes.append(code)
    if not accepted_codes:
        return None
    # icd_autocode returns DOTTED, often category-level codes ('I63', 'E11.1');
    # MIMIC stores them UNDOTTED and BILLABLE ('I6300', 'E1110'). Normalize and
    # PREFIX-match each (mirroring the Tier-1 registry path) — an exact IN on the
    # dotted/category code matched nothing, silently emptying the cohort.
    clauses: list[str] = []
    params: list[str] = []
    for code in accepted_codes:
        norm = normalize_icd_prefix(code)
        if not norm:
            continue
        clauses.append("di.icd_code LIKE ?")
        params.append(norm + "%")
    if not clauses:
        return None
    return f"({' OR '.join(clauses)})", params


def _admission_type_description() -> str:
    """Filter-field description naming the REAL MIMIC-IV admission types.

    The example values are pulled from the frozen schema-grounded categorical-
    domain artifact (not hardcoded MIMIC-III literals), so the decomposer is
    taught the vocabulary the data actually uses — ``EW EMER.``/``DIRECT EMER.``
    rather than a stale ``EMERGENCY`` that matches no rows. Falls back to a
    generic description if the artifact is unavailable.
    """
    from src.similarity.categorical_domains import describe_domain

    examples = describe_domain("admission_type")
    lead = f"admission type (e.g. {examples})" if examples else "admission type"
    return f'{lead}; use operator "in" with a list value to match multiple'


def _compile_admission_type(f: PatientFilter, ctx: FilterCompileContext) -> FilterFragment:
    # Existing: single predicate on admissions.admission_type.
    # New: ``in`` operator accepts a list of allowed values and emits ``IN (?, ?, ...)``.
    # The ``in`` path is the template for how list-valued filters compile —
    # more fields can adopt it by adding "in" to their operator set.
    if f.operator == "in":
        values = f.value if isinstance(f.value, list) else [f.value]
        if not values:
            # Empty IN list would be invalid SQL; emit a predicate that
            # matches nothing so the cohort ends up empty rather than erroring.
            return FilterFragment(where=["1 = 0"])
        placeholders = ", ".join(["?"] * len(values))
        return FilterFragment(
            where=[f"{ctx.admission_alias}.admission_type IN ({placeholders})"],
            params=list(values),
        )
    return FilterFragment(
        where=[f"{ctx.admission_alias}.admission_type = ?"],
        params=[f.value],
    )


def _compile_subject_id(f: PatientFilter, ctx: FilterCompileContext) -> FilterFragment:
    # Existing: single predicate on admissions.subject_id, value int-coerced.
    return FilterFragment(
        where=[f"{ctx.admission_alias}.subject_id = ?"],
        params=[int(f.value)],
    )


def _compile_readmitted(field_name: str):
    """Factory for readmitted_30d / readmitted_60d — identical SQL modulo column."""

    def compile_fn(f: PatientFilter, ctx: FilterCompileContext) -> FilterFragment:
        rl_expr = ctx.backend.readmission_labels_expr()
        return FilterFragment(
            joins=[f"JOIN {rl_expr} rl ON {ctx.admission_alias}.hadm_id = rl.hadm_id"],
            where=[f"rl.{field_name} {f.operator} ?"],
            params=[int(f.value)],
        )

    return compile_fn


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_default_filters(registry: OperationRegistry) -> None:
    """Register the seven pre-existing filter fields.

    Each registration mirrors the exact operators and value shape the old
    hand-rolled dispatch accepted. Phase 1 parity tests verify row-for-row
    equivalence on a synthetic DuckDB fixture.
    """
    registry.register(FilterOperation(
        name="age",
        operators=COMPARISON_OPERATORS,
        value_type="scalar",
        description="patient anchor age in years (integer)",
        compile_fn=_compile_age,
    ))
    registry.register(FilterOperation(
        name="gender",
        operators=frozenset({"="}),
        value_type="scalar",
        description='patient gender ("M"|"F"; case-insensitive on input)',
        compile_fn=_compile_gender,
    ))
    registry.register(FilterOperation(
        name="diagnosis",
        operators=frozenset({"contains"}),
        value_type="scalar",
        description="substring match against ICD-10 long title OR ICD code prefix",
        compile_fn=_compile_diagnosis,
    ))
    registry.register(FilterOperation(
        name="admission_type",
        operators=frozenset({"=", "in"}),
        value_type="scalar_or_list",
        description=_admission_type_description(),
        compile_fn=_compile_admission_type,
    ))
    registry.register(FilterOperation(
        name="subject_id",
        operators=frozenset({"="}),
        value_type="scalar",
        description="single patient MIMIC subject_id (integer)",
        compile_fn=_compile_subject_id,
    ))
    registry.register(FilterOperation(
        name="readmitted_30d",
        operators=COMPARISON_OPERATORS,
        value_type="scalar",
        description="binary readmission-within-30-days flag (0 or 1)",
        compile_fn=_compile_readmitted("readmitted_30d"),
    ))
    registry.register(FilterOperation(
        name="readmitted_60d",
        operators=COMPARISON_OPERATORS,
        value_type="scalar",
        description="binary readmission-within-60-days flag (0 or 1)",
        compile_fn=_compile_readmitted("readmitted_60d"),
    ))
