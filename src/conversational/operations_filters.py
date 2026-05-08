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
    is a non-empty ``contains`` clause, consult OMOPHub-backed
    ``icd_autocode`` (via the same module-level ``_cached_icd_autocode``
    used by ``ConceptResolver.resolve_diagnosis``) to ground the
    natural-language phrase to a list of ICD-10-CM codes. When grounded,
    emit an additional ``di.icd_code IN (?, ?, ...)`` clause as a
    parallel OR with the existing LIKE branch — IN-list catches
    grounded ICD-10 admissions precisely; LIKE catches the long tail of
    ICD-9 admissions whose codes aren't in OMOPHub's ICD10CM-only
    coverage.

    Falls back to LIKE-only when grounding is disabled, MCP returns
    unavailable, no candidates pass the confidence threshold, the
    operator isn't ``contains``, or the value isn't a non-empty string.
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

    grounded_codes = _maybe_ground_diagnosis_filter(f, ctx)
    if grounded_codes:
        in_placeholders = ", ".join(["?"] * len(grounded_codes))
        clause = f"((di.icd_code IN ({in_placeholders})) OR {like_clause})"
        params = list(grounded_codes) + like_params
    else:
        clause = like_clause
        params = like_params

    return FilterFragment(joins=joins, where=[clause], params=params)


def _maybe_ground_diagnosis_filter(
    f: PatientFilter, ctx: FilterCompileContext,
) -> list[str] | None:
    """Attempt OMOPHub-backed grounding of the diagnosis filter's value.

    Returns a non-empty list of ICD codes on success, or ``None`` to
    signal the caller to fall back to LIKE-only. Never raises.
    """
    if not ctx.enable_mcp_grounding:
        return None
    if f.operator != "contains":
        return None
    value = f.value
    if not isinstance(value, str) or not value:
        return None
    # Lazy import to avoid a circular dependency between operations_filters
    # and concept_resolver at module load. The cache lives on the
    # concept_resolver module so it's shared with the resolver path.
    from src.conversational.concept_resolver import (
        _ICD_CONFIDENCE_THRESHOLD, _cached_icd_autocode,
    )
    try:
        cached = _cached_icd_autocode(value.lower(), "10")
    except LookupError:
        return None
    accepted: list[str] = []
    for code, conf in cached:
        if conf is None or conf >= _ICD_CONFIDENCE_THRESHOLD:
            accepted.append(code)
    return accepted or None


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
        description=(
            'admission type (e.g. "EMERGENCY", "ELECTIVE", "URGENT"); '
            'use operator "in" with a list value to match multiple'
        ),
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
