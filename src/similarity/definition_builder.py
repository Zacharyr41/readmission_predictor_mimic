"""Free-text → :class:`CohortDefinition` builder (plan II-D).

This is the link that makes the anchorless cohort path reachable: a clinician
describes a *set of traits* ("emergency patients like a 68-year-old woman whose
creatinine ran high and whose platelets dropped, on a short stay") and this
module turns it into a schema-validated ``CohortDefinition`` the cohort runner
can score. The translation is a judgment task, so it uses **Sonnet** (never
Haiku) governed by a system prompt that encodes the clause→kernel taxonomy:

* a *point/level* clause → quantitative + ``symmetric`` (distance to the value);
* an *identity/category* clause → ``nominal`` (match-or-not);
* a *directional/monotonic* clause ("worse", "rising", "declining", "severe",
  "high/low") → quantitative + a **one-sided** kernel, with the "bad-enough"
  reference value resolved from the frozen reference-population stats so that a
  candidate *more* extreme than the reference is never penalized;
* a *crisp gate* ("ICU", "has sepsis", "stay under 15 days") → a Boolean
  prefilter that narrows the pool, plus (for presence/identity) a high-weight
  trait.

The (mocked, in tests) model does the linguistic judgment; this module owns the
deterministic half — strict Pydantic validation, a self-repair retry that names
any feature the model invented outside the extractable :mod:`feature catalog
<src.similarity.feature_catalog>`, and resolution of directional reference
values against the frozen ranges. ``graph_temporal`` traits are passed through
untouched (they are graph-derived, exempt from the catalog guard, and wired in
plan III-A); the cohort runner rejects them until then.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from src.pygower import Direction, Kind
from src.similarity.feature_catalog import (
    catalog_feature_names,
    cohort_feature_catalog,
)
from src.similarity.models import CohortDefinition

if TYPE_CHECKING:  # pragma: no cover
    import anthropic

    from src.conversational.models import CompetencyQuestion

# Judgment/translation task → Sonnet, never Haiku (see memory:
# feedback_critic_model_choice). Same model id the question decomposer uses.
_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 2048
_MAX_ATTEMPTS = 2


# The graph-derived feature-extractor templates the builder may emit for a
# ``source="graph_temporal"`` trait. Keys MUST stay in lock-step with
# :data:`src.similarity.graph_features.TEMPLATES` (enforced by
# ``test_graph_template_guidance_matches_registry``); the values are the human
# guidance surfaced in the system prompt so the model picks an *extractable*
# template instead of inventing one. Kept as a literal (not derived from the
# registry) so importing this module never drags rdflib in at load time.
_GRAPH_TEMPLATE_GUIDANCE: dict[str, str] = {
    "sim_series_by_admission": (
        "biomarker / vital trajectory per admission — concept = the lab/vital "
        'label (e.g. "lactate"); graph_params: agg="slope"|"delta", window_hours.'
    ),
    "sim_dose_series": (
        "per-administration drug-dose trajectory — concept = a drug name or "
        'category (e.g. "vasopressors"); graph_params: agg, window_hours.'
    ),
    "sim_distinct_drug_count": (
        "number of distinct prescribed drugs per admission — concept-free."
    ),
    "sim_icu_los": (
        "total ICU length of stay in hours, summed over the admission's stays "
        "— concept-free."
    ),
    "sim_time_to_first_event": (
        "hours from ICU intime to the first event of a concept — concept = the "
        "biomarker/drug label."
    ),
    "sim_precedence_count": (
        "count of Allen temporal-precedence edges from concept-A events to "
        "concept-B events — concept = A; graph_params: concept_b = B, as_bool."
    ),
}


# ---------------------------------------------------------------------------
# Prompt construction.
# ---------------------------------------------------------------------------


def _prefilter_fields() -> list[str]:
    """The legal ``PatientFilter.field`` names (the prefilter vocabulary).

    Imported lazily — same caution the decomposer takes — so importing this
    module never drags the whole operations registry in at module load.
    """
    from src.conversational.operations import get_default_registry

    return sorted(get_default_registry().supported_names("filter"))


def _feature_catalog_block() -> str:
    lines: list[str] = []
    for info in cohort_feature_catalog().values():
        suffix = ""
        if info.categories:
            suffix = f"  categories: {list(info.categories)}"
        lines.append(f"- {info.name} ({info.kind.value}) — {info.description}{suffix}")
    return "\n".join(lines)


def _graph_template_block() -> str:
    """The legal ``graph_temporal`` templates, rendered for the system prompt."""
    return "\n".join(
        f"- {name} — {desc}" for name, desc in _GRAPH_TEMPLATE_GUIDANCE.items()
    )


def build_definition_system_prompt() -> str:
    """The system prompt: the clause→kernel taxonomy + the legal vocabularies.

    Built from the live feature catalog and prefilter registry so it never
    drifts from what the runner can actually pull.
    """
    return f"""You translate a clinician's free-text description of a patient \
COHORT into a single strict JSON object — a CohortDefinition. Return ONLY that \
JSON object, no prose, no markdown fence.

HOW COHORT MATCHING WORKS
Every candidate admission is scored by Gower distance to a synthesized \
reference *profile* assembled from the traits below. The cohort is every \
candidate with distance <= distance_threshold, ranked nearest-first and capped \
at top_k. Prefilters cheaply narrow the candidate pool BEFORE scoring.

CHOOSE EACH TRAIT'S KERNEL FROM THE CLAUSE'S LINGUISTIC FORM (a general rule, \
not a per-lab lookup):
- Point / level ("68 years old", "sodium around 140"): kind="quantitative", \
direction="symmetric", reference_value = the stated number.
- Category / identity ("female", "emergency admission"): kind="nominal", \
reference_value = the stated category string.
- Directional / monotonic ("worse", "rising", "escalating", "climbing", \
"declining", "dropping", "severe", "high", "low", "at least"): \
kind="quantitative", direction="higher_more_similar" when MORE is the \
described/worse end, or "lower_more_similar" when LESS is. Leave \
reference_value = null — it is filled from frozen population stats. A candidate \
MORE extreme than the reference is never penalized.
- Trend over a window ("lactate trend over the first 48h", "severity slope", \
"escalating vasopressors", "lactate cleared before pressors started"): this is \
graph-derived — set source="graph_temporal", choose a "template" from the LEGAL \
graph_temporal templates below, put the clinical concept in "concept" \
(+ "concept_type"), put aggregation options in "graph_params", and set \
direction from the wording. Name it after what it measures (e.g. \
"lactate_slope_48h"). Leave reference_value=null for a directional trend.
- Crisp gate ("ICU patients", "has sepsis", "stay under 15 days"): add a \
prefilter to narrow the pool; if it is a presence/identity, ALSO add it as a \
high-weight trait.

LEGAL sql FEATURE NAMES (for source="sql", trait.name MUST be one of these):
{_feature_catalog_block()}

LEGAL graph_temporal templates (for source="graph_temporal", trait.template MUST \
be one of these):
{_graph_template_block()}

LEGAL prefilter fields (PatientFilter.field MUST be one of these):
{_prefilter_fields()}
PatientFilter operators: > < = >= <= contains in

OUTPUT JSON SCHEMA (emit ONLY these keys):
{{
  "prefilters": [{{"field": <prefilter field>, "operator": <op>, "value": <str>}}],
  "traits": [{{
    "name": <feature name>,
    "source": "sql" | "graph_temporal",
    "kind": "quantitative" | "nominal" | "binary",
    "reference_value": <number | string | true | null>,
    "direction": "symmetric" | "higher_more_similar" | "lower_more_similar",
    "weight": <number, default 1.0; give crisp/presence traits 1.5-2.0>,
    "template": <graph_temporal ONLY: one of the LEGAL graph_temporal templates>,
    "concept": <graph_temporal: the lab/drug/category label, or null>,
    "concept_type": <graph_temporal: "biomarker"|"drug"|"drug_category"|null>,
    "graph_params": <graph_temporal: object, e.g. {{"agg": "slope", "window_hours": 48}}>
  }}],
  "distance_threshold": <float in [0,1], typically 0.30-0.40>,
  "top_k": <int, e.g. 30>
}}
Do NOT emit a "range_" for quantitative traits — the normalization scale comes \
from frozen population stats. The template/concept/concept_type/graph_params \
keys apply ONLY to source="graph_temporal" traits; omit them for sql traits. \
Emit no keys other than those shown.

WORKED EXAMPLE
Request: "Find emergency ICU patients similar to a 68-year-old woman whose \
creatinine ran high and whose platelets dropped, hospital stay on the shorter \
side."
{{
  "prefilters": [{{"field": "admission_type", "operator": "=", "value": "EMERGENCY"}}],
  "traits": [
    {{"name": "age", "source": "sql", "kind": "quantitative", "reference_value": 68, "direction": "symmetric", "weight": 0.8}},
    {{"name": "gender", "source": "sql", "kind": "nominal", "reference_value": "F", "weight": 1.0}},
    {{"name": "creatinine_max", "source": "sql", "kind": "quantitative", "reference_value": null, "direction": "higher_more_similar", "weight": 1.5}},
    {{"name": "platelet_min", "source": "sql", "kind": "quantitative", "reference_value": null, "direction": "lower_more_similar", "weight": 1.5}},
    {{"name": "icu_los_hours", "source": "sql", "kind": "quantitative", "reference_value": null, "direction": "lower_more_similar", "weight": 0.5}}
  ],
  "distance_threshold": 0.35,
  "top_k": 30
}}

WORKED EXAMPLE (with temporal trends)
Request: "ICU patients like a 68-year-old with sepsis whose lactate got \
progressively worse over the first 48 hours and who needed escalating \
vasopressors."
{{
  "prefilters": [{{"field": "diagnosis", "operator": "contains", "value": "sepsis"}}],
  "traits": [
    {{"name": "age", "source": "sql", "kind": "quantitative", "reference_value": 68, "direction": "symmetric", "weight": 0.6}},
    {{"name": "lactate_slope_48h", "source": "graph_temporal", "kind": "quantitative", "reference_value": null, "direction": "higher_more_similar", "weight": 2.0, "template": "sim_series_by_admission", "concept": "lactate", "concept_type": "biomarker", "graph_params": {{"agg": "slope", "window_hours": 48}}}},
    {{"name": "vasopressor_dose_slope", "source": "graph_temporal", "kind": "quantitative", "reference_value": null, "direction": "higher_more_similar", "weight": 1.5, "template": "sim_dose_series", "concept": "vasopressors", "concept_type": "drug_category", "graph_params": {{"agg": "slope"}}}}
  ],
  "distance_threshold": 0.35,
  "top_k": 30
}}"""


def _build_user_message(question: str, cq: "CompetencyQuestion | None") -> str:
    """The turn message: the request plus any structured hints the decomposer
    already extracted (filters / concepts / its interpretation)."""
    parts = [f"Clinician request:\n{question}"]
    if cq is not None:
        hints: dict[str, Any] = {}
        if getattr(cq, "patient_filters", None):
            hints["filters"] = [f.model_dump() for f in cq.patient_filters]
        if getattr(cq, "clinical_concepts", None):
            hints["clinical_concepts"] = [c.name for c in cq.clinical_concepts]
        summary = getattr(cq, "interpretation_summary", None)
        if summary:
            hints["interpretation"] = summary
        if hints:
            parts.append(
                "Structured hints from the question decomposer "
                "(guidance, not gospel):\n" + json.dumps(hints, indent=2, default=str)
            )
    parts.append("Return ONLY the CohortDefinition JSON.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Parse / validate / repair / resolve.
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> str:
    """Pull a JSON object out of *text*, tolerating a markdown code fence.

    Fence delimiters are anchored to line boundaries so a triple backtick that
    appears *inside* a JSON string value cannot be mistaken for the closing
    fence and truncate the payload (same fix as the decomposer's extractor).
    """
    match = re.search(
        r"^[ \t]*```(?:json)?[ \t]*\n(.*?)\n[ \t]*```[ \t]*$",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if match:
        return match.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _offending_sql_trait_names(defn: CohortDefinition) -> set[str]:
    """``source="sql"`` trait names that are not extractable features.

    ``graph_temporal`` traits are exempt — they are graph-derived (III-A), not
    columns of the contextual extractor.
    """
    allowed = catalog_feature_names()
    return {t.name for t in defn.traits if t.source == "sql"} - allowed


def _offending_graph_traits(defn: CohortDefinition) -> dict[str, str]:
    """``source="graph_temporal"`` traits whose ``template`` is missing/unknown.

    Returns ``trait name → the offending template`` (``""`` when none was set)
    so the corrective turn can name both the trait and what was wrong. This is
    the graph counterpart to :func:`_offending_sql_trait_names`: it keeps the
    non-deterministic layer inside the live extractor rails — a ``graph_temporal``
    trait whose template is not one the runner can actually run would silently
    drop the trait at feature-extraction time. ``TEMPLATES`` is imported lazily
    so this module never drags rdflib in at load time.
    """
    from src.similarity.graph_features import TEMPLATES

    legal = set(TEMPLATES)
    offenders: dict[str, str] = {}
    for t in defn.traits:
        if t.source != "graph_temporal":
            continue
        template = getattr(t, "template", None)
        if not template or template not in legal:
            offenders[t.name] = template or ""
    return offenders


def _resolve_reference_values(
    defn: CohortDefinition,
    reference_ranges: dict[str, tuple[float, float]] | None,
) -> CohortDefinition:
    """Fill a directional trait's null ``reference_value`` from frozen stats.

    A one-sided clause ("rising creatinine") states a *direction*, not a number;
    the "bad-enough" point is the population extreme in that direction — the p99
    high for ``higher_more_similar``, the p1 low for ``lower_more_similar`` (plan
    II-E ranges). Symmetric traits and traits the model already pinned are left
    untouched, as are features with no frozen range.
    """
    if not reference_ranges:
        return defn
    new_traits = []
    for t in defn.traits:
        needs = (
            t.kind == Kind.QUANTITATIVE
            and t.reference_value is None
            and t.direction != Direction.SYMMETRIC
            and t.name in reference_ranges
        )
        if needs:
            low, high = reference_ranges[t.name]
            value = high if t.direction == Direction.HIGHER_MORE_SIMILAR else low
            t = t.model_copy(update={"reference_value": float(value)})
        new_traits.append(t)
    return defn.model_copy(update={"traits": new_traits})


def build_definition(
    client: "anthropic.Anthropic",
    question: str,
    *,
    cq: "CompetencyQuestion | None" = None,
    reference_ranges: dict[str, tuple[float, float]] | None = None,
    model: str = _MODEL,
    max_tokens: int = _MAX_TOKENS,
) -> CohortDefinition:
    """Translate *question* into a validated :class:`CohortDefinition`.

    Mirrors the decomposer's retry discipline: on a parse/validation failure or
    an invented (non-extractable) ``sql`` trait name, the error is fed back once
    so the model self-corrects; a second failure raises. ``reference_ranges``
    (the frozen II-E stats) resolves directional traits' "bad-enough" reference
    values. The LLM is the only non-deterministic step; everything after the
    ``messages.create`` call is deterministic and unit-testable.
    """
    system = build_definition_system_prompt()
    messages = [{"role": "user", "content": _build_user_message(question, cq)}]

    for attempt in range(_MAX_ATTEMPTS):
        last = attempt == _MAX_ATTEMPTS - 1
        response = client.messages.create(
            model=model, max_tokens=max_tokens, system=system, messages=messages,
        )
        raw_text = response.content[0].text
        try:
            data = json.loads(_extract_json(raw_text))
            defn = CohortDefinition.model_validate(data)
        except Exception as exc:
            if last:
                raise
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({
                "role": "user",
                "content": (
                    f"Your response was not valid JSON or failed schema "
                    f"validation: {exc}. Return ONLY the corrected "
                    "CohortDefinition JSON object."
                ),
            })
            continue

        offenders = _offending_sql_trait_names(defn)
        if offenders:
            if last:
                raise ValueError(
                    f"cohort definition references unknown sql trait(s) "
                    f"{sorted(offenders)}; allowed sql feature names are "
                    f"{sorted(catalog_feature_names())}"
                )
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({
                "role": "user",
                "content": (
                    f"These trait names are not extractable sql features: "
                    f"{sorted(offenders)}. Allowed sql feature names are: "
                    f"{sorted(catalog_feature_names())}. Rephrase using only "
                    "those, or mark a temporal-trend trait with "
                    "source='graph_temporal'. Return ONLY the corrected JSON."
                ),
            })
            continue

        graph_offenders = _offending_graph_traits(defn)
        if graph_offenders:
            legal = sorted(_GRAPH_TEMPLATE_GUIDANCE)
            if last:
                raise ValueError(
                    f"cohort definition has graph_temporal trait(s) with a "
                    f"missing or unknown feature-extractor template "
                    f"{sorted(graph_offenders)}; legal templates are {legal}"
                )
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({
                "role": "user",
                "content": (
                    f"These graph_temporal traits have a missing or unknown "
                    f'feature-extractor "template": {sorted(graph_offenders)}. '
                    f"Every graph_temporal trait MUST set \"template\" to one "
                    f"of: {legal}. Return ONLY the corrected JSON."
                ),
            })
            continue

        return _resolve_reference_values(defn, reference_ranges)

    raise AssertionError("unreachable: retry loop exited without return")  # pragma: no cover


__all__ = ["build_definition", "build_definition_system_prompt"]
