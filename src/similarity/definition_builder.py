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
- Trend over a window ("lactate trend over the first 48h", "severity slope"): \
this is graph-derived — set source="graph_temporal" and a name ending in \
"_slope" or "_trend", direction from the wording. (These are not yet scored, \
but emit them faithfully.)
- Crisp gate ("ICU patients", "has sepsis", "stay under 15 days"): add a \
prefilter to narrow the pool; if it is a presence/identity, ALSO add it as a \
high-weight trait.

LEGAL sql FEATURE NAMES (for source="sql", trait.name MUST be one of these):
{_feature_catalog_block()}

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
    "weight": <number, default 1.0; give crisp/presence traits 1.5-2.0>
  }}],
  "distance_threshold": <float in [0,1], typically 0.30-0.40>,
  "top_k": <int, e.g. 30>
}}
Do NOT emit a "range_" for quantitative traits — the normalization scale comes \
from frozen population stats. Emit no keys other than those shown.

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
    """Pull a JSON object out of *text*, tolerating a markdown code fence."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
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

        return _resolve_reference_values(defn, reference_ranges)

    raise AssertionError("unreachable: retry loop exited without return")  # pragma: no cover


__all__ = ["build_definition", "build_definition_system_prompt"]
