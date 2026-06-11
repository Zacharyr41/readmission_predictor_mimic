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

import logging

from src.conversational.models import PatientFilter
from src.conversational.operations import (
    COMPARISON_OPERATORS,
    FilterCompileContext,
    FilterFragment,
    FilterOperation,
    OperationRegistry,
)

logger = logging.getLogger(__name__)


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
        # Codes-only when grounded (drop the broad title-LIKE) — the same anti-
        # pollution rule as the count path, so count and filter resolve to the
        # SAME cohort instead of 1,576-vs-9,362.
        clause = clause_sql
        params = list(clause_params)
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
    # Tier 0 — codes the orchestrator already resolved for THIS filter via
    # candidate disambiguation (``_disambiguate_diagnoses``). Used first and
    # unconditionally so the filter path shares the count path's exact grounding.
    resolved = getattr(f, "resolved_icd_codes", None)
    if resolved:
        from src.conversational.health_evidence.cohorts import normalize_icd_prefix
        clauses0: list[str] = []
        params0: list[str] = []
        for code in resolved:
            norm = normalize_icd_prefix(code)
            if not norm:
                continue
            clauses0.append("di.icd_code LIKE ?")
            params0.append(norm + "%")
        if clauses0:
            return f"({' OR '.join(clauses0)})", params0

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
# Measurement-value cohort filters (lab / vital / derived) + composition
# ---------------------------------------------------------------------------


def _ground_measurement_itemids(
    ctx: FilterCompileContext, measurement: str | None, loinc_code: str | None,
    concept_type: str,
) -> list[int] | None:
    """Ground an analyte name + LOINC → MIMIC itemids via the resolver.

    Returns ``None`` (→ caller falls back to label-LIKE) when no resolver is
    available (unit tests), no measurement name was supplied, or grounding
    misses. Reuses ``resolve_biomarker`` (labevents) / ``resolve_vital``
    (chartevents) — no new grounding logic.
    """
    if ctx.resolver is None or not measurement:
        return None
    from src.conversational.models import ClinicalConcept

    concept = ClinicalConcept(
        name=measurement, concept_type=concept_type, loinc_code=loinc_code,
    )
    if concept_type == "biomarker":
        res = ctx.resolver.resolve_biomarker(concept)
    else:
        res = ctx.resolver.resolve_vital(concept)
    return res.itemids  # None on no-LOINC / no-coverage → label-LIKE fallback


def _is_gcs_measurement(measurement: str | None, value: object) -> bool:
    """True when this measurement filter is a Glasgow Coma Scale TOTAL threshold.

    GCS is NOT a single-LOINC labitem — ``d_items`` only carries the three
    COMPONENT rows (Eye 1–4 / Verbal 1–5 / Motor 1–6), so the generic label-LIKE
    fallback (``d_items.label ILIKE '%GCS%'``) matches the components and a
    ``valuenum <= 8`` predicate against them is meaningless. We detect a GCS
    threshold by name and route it to the derived TOTAL ``gcs`` column instead.

    Matched against the analyte name (``measurement``); also the literal value
    when it carries the analyte (some decompositions put the term there).
    """
    haystacks = [measurement]
    if isinstance(value, str):
        haystacks.append(value)
    for h in haystacks:
        if not h:
            continue
        low = h.lower()
        if "gcs" in low or "glasgow coma" in low:
            return True
    return False


def _compile_gcs_total_exists(
    f: PatientFilter, ctx: FilterCompileContext,
) -> FilterFragment:
    """Ground a GCS-total threshold to the derived ``gcs`` table's TOTAL column.

    Emits a correlated ``EXISTS`` over ``mimiciv_3_1_derived.gcs`` joined to
    ``icustays`` so the per-ICU-stay GCS rows map back to the admission:

        EXISTS (SELECT 1 FROM <derived.gcs> g
                JOIN <icustays> icu ON icu.stay_id = g.stay_id
                WHERE icu.hadm_id = a.hadm_id AND g.gcs <op> ?)

    Uses the TOTAL ``g.gcs`` (3–15), never the components. The operator comes
    from the registered comparison-operator set (safe to interpolate); the
    threshold is the single bound param.

    The derived ``gcs`` table only exists on the BigQuery backend; on DuckDB /
    the offline fixture ``_derived_table`` returns ``None``. Rather than emit
    SQL against a table that doesn't exist (which would crash on execute), we
    degrade to ``1 = 0`` (empty cohort) and log a warning — matching how
    ``_compile_derived_value`` degrades when its formula is unavailable, and
    deliberately NOT falling back to the meaningless component label-LIKE.
    """
    # Lazy import keeps the operations ↔ sql_fastpath edge off module load.
    from src.conversational.sql_fastpath import _derived_table

    gcs = _derived_table(ctx.backend, "gcs")
    if gcs is None:
        logger.warning(
            "GCS-threshold cohort filter requested but the derived GCS table "
            "(mimiciv_3_1_derived.gcs) is unavailable on this backend; "
            "grounding to an empty cohort rather than the meaningless GCS "
            "component label-match. Run against BigQuery for a real GCS cohort.",
        )
        return FilterFragment(where=["1 = 0"])
    op = f.operator
    try:
        thr = float(f.value)
    except (TypeError, ValueError):
        return FilterFragment(where=["1 = 0"])  # malformed threshold → empty
    t = ctx.backend.table
    a = ctx.admission_alias
    inner = (
        f"SELECT 1 FROM {gcs} g "
        f"JOIN {t('icustays')} icu ON icu.stay_id = g.stay_id "
        f"WHERE icu.hadm_id = {a}.hadm_id AND g.gcs {op} ?"
    )
    return FilterFragment(where=[f"EXISTS ({inner})"], params=[thr])


def _compile_measurement_value(
    f: PatientFilter, ctx: FilterCompileContext, *, concept_type: str,
) -> FilterFragment:
    """Cohort = admissions with ≥1 reading of an analyte crossing a threshold.

    Emits a correlated ``EXISTS`` on the event table (labevents for biomarker,
    chartevents for vital): itemid-grounded when possible, else a label-LIKE
    fallback on the dictionary table. The operator comes from the registered
    operator set (safe to interpolate); the threshold is parameterized.

    GCS is special-cased BEFORE the generic itemid/LIKE logic: a "GCS ≤ 8"
    threshold grounds to the derived TOTAL ``gcs`` column (3–15), not the three
    GCS COMPONENT chartitems a label-LIKE would match. See
    ``_compile_gcs_total_exists``. The gate is keyed on the analyte name so
    every other vital/lab is byte-identical to before.
    """
    if concept_type == "vital" and _is_gcs_measurement(f.measurement, f.value):
        return _compile_gcs_total_exists(f, ctx)
    op = f.operator
    try:
        thr = float(f.value)
    except (TypeError, ValueError):
        return FilterFragment(where=["1 = 0"])  # malformed threshold → empty
    t = ctx.backend.table
    a = ctx.admission_alias
    if concept_type == "biomarker":
        ev, dic, table_name, dict_name = "lev", "dlab", "labevents", "d_labitems"
    else:
        ev, dic, table_name, dict_name = "cev", "dit", "chartevents", "d_items"
    itemids = _ground_measurement_itemids(
        ctx, f.measurement, f.loinc_code, concept_type,
    )
    correlate = f"{ev}.hadm_id = {a}.hadm_id"
    if itemids:
        placeholders = ",".join(["?"] * len(itemids))
        inner = (
            f"SELECT 1 FROM {t(table_name)} {ev} "
            f"WHERE {correlate} AND {ev}.itemid IN ({placeholders}) "
            f"AND {ev}.valuenum IS NOT NULL AND {ev}.valuenum {op} ?"
        )
        params: list = list(itemids) + [thr]
    else:
        analyte = f.measurement or (f.value if isinstance(f.value, str) else "")
        inner = (
            f"SELECT 1 FROM {t(table_name)} {ev} "
            f"JOIN {t(dict_name)} {dic} ON {ev}.itemid = {dic}.itemid "
            f"WHERE {correlate} AND {ctx.backend.ilike(f'{dic}.label')} "
            f"AND {ev}.valuenum IS NOT NULL AND {ev}.valuenum {op} ?"
        )
        params = [f"%{analyte}%", thr]
    return FilterFragment(where=[f"EXISTS ({inner})"], params=params)


def _compile_lab_value(f: PatientFilter, ctx: FilterCompileContext) -> FilterFragment:
    return _compile_measurement_value(f, ctx, concept_type="biomarker")


def _compile_vital_value(f: PatientFilter, ctx: FilterCompileContext) -> FilterFragment:
    return _compile_measurement_value(f, ctx, concept_type="vital")


def _compile_derived_value(
    f: PatientFilter, ctx: FilterCompileContext,
) -> FilterFragment:
    """Cohort = admissions where a derived clinical index crosses its threshold.

    The :class:`DerivedFormula` (operands + arithmetic AST + operator/threshold)
    is pre-resolved by the orchestrator from a PubMed lookup and supplied via
    ``ctx.derived_formulas`` (keyed by the lower-cased index name). Compiled
    through the safe AST→SQL emitter in ``per_instant`` (time-matched self-join)
    or ``per_stay`` (per-operand scalar subqueries) mode. When no formula is
    available (lookup failed / offline) the arm drops to ``1 = 0`` so the rest
    of the query still returns.
    """
    name = (f.measurement or "").strip().lower()
    formula = (ctx.derived_formulas or {}).get(name)
    if formula is None:
        return FilterFragment(where=["1 = 0"])
    from src.conversational.derived_formula import (
        FormulaError, validate_formula,
    )
    try:
        validate_formula(formula)
        if formula.time_semantics == "per_instant":
            sql, params = _compile_derived_per_instant(formula, ctx)
        else:
            sql, params = _compile_derived_per_stay(formula, ctx)
    except FormulaError:
        return FilterFragment(where=["1 = 0"])
    return FilterFragment(where=[sql], params=params)


def _compile_derived_per_instant(formula, ctx: FilterCompileContext) -> tuple[str, list]:
    """N-alias same-charttime self-join; the expression is evaluated per instant
    and thresholded as a ROW-LEVEL predicate inside ``EXISTS``.

    Cohort membership = the index crosses its threshold at *some* instant
    ("ever elevated / ever low") — which is both the natural semantics for an
    "elevated <index>" cohort and a de-correlatable semi-join. (A GROUP BY +
    ``HAVING MAX(expr)`` would be a correlated aggregate subquery that BigQuery
    refuses to de-correlate; the EXISTS quantifier already supplies the "ever".)
    Operands share one event table so the charttime self-join is valid.
    """
    from src.conversational.derived_formula import emit_expression

    t = ctx.backend.table
    a = ctx.admission_alias
    ops = formula.operands
    table = ops[0].table
    aliases = [f"dop{i}" for i in range(len(ops))]
    base = aliases[0]
    from_parts = [f"{t(table)} {base}"]
    where_parts = [f"{base}.hadm_id = {a}.hadm_id"]
    params: list = []
    for i, (al, o) in enumerate(zip(aliases, ops)):
        if i > 0:
            from_parts.append(
                f"JOIN {t(table)} {al} ON {al}.stay_id = {base}.stay_id "
                f"AND {al}.charttime = {base}.charttime"
            )
        placeholders = ",".join(["?"] * len(o.itemids))
        where_parts.append(f"{al}.itemid IN ({placeholders})")
        params.extend(o.itemids)
        where_parts.append(f"{al}.valuenum BETWEEN ? AND ?")
        params.extend([o.guard_low, o.guard_high])
    ref_col = {o.ref: f"{al}.valuenum" for al, o in zip(aliases, ops)}
    expr_sql, expr_params = emit_expression(
        formula.expression, lambda r: (ref_col[r], []),
    )
    where_parts.append(f"({expr_sql}) {formula.operator} ?")
    params.extend(expr_params)
    params.append(float(formula.threshold))
    inner = (
        f"SELECT 1 FROM {' '.join(from_parts)} "
        f"WHERE {' AND '.join(where_parts)}"
    )
    return f"EXISTS ({inner})", params


def _compile_derived_per_stay(formula, ctx: FilterCompileContext) -> tuple[str, list]:
    """Each operand is an independent per-stay scalar subquery; the expression
    combines the scalars and is thresholded (a scalar WHERE predicate)."""
    from src.conversational.derived_formula import emit_expression

    t = ctx.backend.table
    a = ctx.admission_alias
    subq: dict[str, str] = {}
    subq_params: dict[str, list] = {}
    for o in formula.operands:
        placeholders = ",".join(["?"] * len(o.itemids))
        subq[o.ref] = (
            f"(SELECT {o.aggregate.upper()}(mv.valuenum) FROM {t(o.table)} mv "
            f"WHERE mv.hadm_id = {a}.hadm_id AND mv.itemid IN ({placeholders}) "
            f"AND mv.valuenum BETWEEN ? AND ?)"
        )
        subq_params[o.ref] = list(o.itemids) + [o.guard_low, o.guard_high]
    expr_sql, expr_params = emit_expression(
        formula.expression, lambda r: (subq[r], subq_params[r]),
    )
    return f"({expr_sql}) {formula.operator} ?", expr_params + [float(formula.threshold)]


def _compile_or_any(f: PatientFilter, ctx: FilterCompileContext) -> FilterFragment:
    """Union cohort — OR-combine the predicates of the child filters in
    ``f.sub_filters`` into one parenthesized clause. Each child is compiled
    through the registry, so any registered filter is a valid member. Children
    are expected to be self-contained predicates (EXISTS-style measurement /
    icu filters); a child that emits table JOINs is skipped, since an
    unconditional JOIN can't participate in an OR.
    """
    subs = f.sub_filters or []
    registry = ctx.registry
    if not subs or registry is None:
        return FilterFragment(where=["1 = 0"])
    sub_wheres: list[str] = []
    params: list = []
    for sub in subs:
        op = registry.get("filter", sub.field)
        if op is None:
            continue
        frag = op.compile(sub, ctx)
        if frag.joins or not frag.where:
            continue  # OR can't carry an unconditional JOIN
        sub_wheres.append("(" + " AND ".join(frag.where) + ")")
        params.extend(frag.params)
    if not sub_wheres:
        return FilterFragment(where=["1 = 0"])
    return FilterFragment(where=["(" + " OR ".join(sub_wheres) + ")"], params=params)


def _compile_icu_stay(f: PatientFilter, ctx: FilterCompileContext) -> FilterFragment:
    """Restrict to admissions with at least one ICU stay (``value`` ignored)."""
    t = ctx.backend.table
    a = ctx.admission_alias
    inner = f"SELECT 1 FROM {t('icustays')} icu WHERE icu.hadm_id = {a}.hadm_id"
    return FilterFragment(where=[f"EXISTS ({inner})"], params=[])


# ---------------------------------------------------------------------------
# Drug cohort filter ("admissions that received drug X / a drug in group G")
# ---------------------------------------------------------------------------

import json as _json  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_DRUG_GROUPS_PATH = (
    _Path(__file__).resolve().parents[2]
    / "data" / "mappings" / "drug_groups.json"
)


def _load_drug_groups() -> dict[str, dict]:
    """Load + cache the drug-group registry (``data/mappings/drug_groups.json``).

    Returns ``{group_name: {patterns: [...], aliases: [...]}}`` with the
    ``_metadata`` block filtered out. Missing/unreadable file degrades to an
    empty registry (the drug filter then grounds purely by the raw drug name),
    so the registry is a convenience layer and never a hard requirement.
    """
    cached = getattr(_load_drug_groups, "_cache", None)
    if cached is not None:
        return cached
    try:
        raw = _json.loads(_DRUG_GROUPS_PATH.read_text())
        groups = {k: v for k, v in raw.items() if not k.startswith("_")}
    except Exception as exc:  # noqa: BLE001
        logger.warning("drug_groups.json unreadable (%s); ignoring", exc)
        groups = {}
    _load_drug_groups._cache = groups  # type: ignore[attr-defined]
    return groups


def _drug_like_patterns(value: str) -> list[str]:
    """Resolve a drug filter value to the set of name-LIKE patterns to OR.

    A vague GROUP phrase ("coagulation reversal agent") expands to its concrete
    member patterns from the registry (``%prothrombin%``, ``%kcentra%``,
    ``%phytonadione%``, ``%vitamin k%``, ``%fresh frozen plasma%``, ``%ffp%``);
    a specific drug name ("vancomycin") grounds by itself. The raw phrase is
    ALWAYS kept as an additional pattern (the escape hatch) so an unrecognised
    drug still matches on its own name even when it also resolves to a group.
    Matching is case-insensitive substring (``ILIKE '%pat%'``) against
    MIMIC ``prescriptions.drug``. De-duplicated, order-stable.
    """
    norm = (value or "").strip().lower()
    patterns: list[str] = []
    if not norm:
        return patterns
    for name, defn in _load_drug_groups().items():
        candidates = [name] + list(defn.get("aliases") or [])
        if any((c or "").strip().lower() == norm for c in candidates):
            patterns.extend(p for p in (defn.get("patterns") or []) if p)
            break
    # Always retain the raw phrase so a specific drug grounds by its own name
    # (and as a safety net if a group's member list is incomplete).
    patterns.append(norm)
    seen: set[str] = set()
    deduped: list[str] = []
    for p in patterns:
        key = p.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _compile_drug(f: PatientFilter, ctx: FilterCompileContext) -> FilterFragment:
    """Cohort = admissions with ≥1 prescription matching the drug (or group).

    Emits a correlated ``EXISTS`` over ``prescriptions`` so the filter is a
    self-contained predicate (no JOIN) — it can therefore participate in an
    ``or_any`` UNION and in the event_ordering EXISTS reuse path, exactly like
    the measurement-value filters. A vague group phrase expands to its concrete
    member name-LIKE patterns via the drug-group registry; a specific drug
    grounds by its own name. The patterns are OR-combined inside the EXISTS.

    Mirrors the SQL-fast-path drug aggregate's ``pr.drug`` name-LIKE grounding,
    so a "received a coagulation reversal agent" cohort restriction grounds the
    SAME concrete drug names whether it lands in the graph cohort query or the
    fast-path.
    """
    a = ctx.admission_alias
    t = ctx.backend.table
    value = f.value if isinstance(f.value, str) else ""
    patterns = _drug_like_patterns(value)
    if not patterns:
        return FilterFragment(where=["1 = 0"])  # no drug named → empty cohort
    name_clauses = " OR ".join(ctx.backend.ilike("pr.drug") for _ in patterns)
    inner = (
        f"SELECT 1 FROM {t('prescriptions')} pr "
        f"WHERE pr.hadm_id = {a}.hadm_id AND ({name_clauses})"
    )
    params = [f"%{p}%" for p in patterns]
    return FilterFragment(where=[f"EXISTS ({inner})"], params=params)


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
    # --- measurement-value cohort filters (lab / vital / derived) + composition ---
    registry.register(FilterOperation(
        name="lab_value",
        operators=COMPARISON_OPERATORS,
        value_type="scalar",
        description=(
            "lab-value threshold cohort: set measurement to the analyte (e.g. "
            "'platelet count') and loinc_code to its LOINC; value is the numeric "
            "threshold (e.g. operator '<' value '50' for platelets < 50 K/uL)"
        ),
        compile_fn=_compile_lab_value,
    ))
    registry.register(FilterOperation(
        name="vital_value",
        operators=COMPARISON_OPERATORS,
        value_type="scalar",
        description=(
            "vital-sign threshold cohort: set measurement to the vital (e.g. "
            "'mean arterial pressure') and loinc_code to its LOINC; value is the "
            "numeric threshold (e.g. operator '<' value '65' for MAP < 65 mmHg)"
        ),
        compile_fn=_compile_vital_value,
    ))
    registry.register(FilterOperation(
        name="derived_value",
        operators=COMPARISON_OPERATORS,
        value_type="scalar",
        description=(
            "derived clinical index/score cohort (e.g. 'shock index'): set "
            "measurement to the index name — its formula and abnormal threshold "
            "are looked up from the medical literature; operator/value are "
            "placeholders"
        ),
        compile_fn=_compile_derived_value,
    ))
    registry.register(FilterOperation(
        name="or_any",
        operators=frozenset({"in"}),
        value_type="list",
        description=(
            "UNION of cohort sub-filters (an 'A or B or C' cohort): put the "
            "child filters in sub_filters (each a lab_value / vital_value / "
            "derived_value / etc.); operator 'in', value can be 'any'"
        ),
        compile_fn=_compile_or_any,
    ))
    registry.register(FilterOperation(
        name="icu_stay",
        operators=frozenset({"="}),
        value_type="scalar",
        description=(
            "restrict to admissions with at least one ICU stay (operator '=' "
            "value '1' for 'ICU patients')"
        ),
        compile_fn=_compile_icu_stay,
    ))
    registry.register(FilterOperation(
        name="drug",
        operators=frozenset({"contains"}),
        value_type="scalar",
        description=(
            "restrict to admissions that received a drug (operator 'contains', "
            "value a drug name like 'vancomycin' OR a recognised group phrase "
            "like 'coagulation reversal agent', which expands to its member "
            "drugs). For an 'A or B or C' drug cohort, prefer one drug filter "
            "per member inside an or_any"
        ),
        compile_fn=_compile_drug,
    ))
