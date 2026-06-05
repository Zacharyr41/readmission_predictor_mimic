"""Graph-derived similarity feature extractors (plan III-A).

The graph is the temporal feature extractor: severity slopes, trend-over-window
deltas, dose trajectories, durations, time-to-first-event and event precedence
become *columns* in the candidate feature matrix that pygower scores. Each
template is ``hadm_id``-scoped, runs one SPARQL query over an in-memory rdflib
``Graph`` (pulling time-ordered rows), and aggregates in Python — the
pull-then-aggregate idiom, never a SPARQL regression.

A template returns ``dict[int, float]`` keyed by ``hadm_id``; an admission with
no defined value (concept absent, or a single reading so no trend) is simply
*omitted*, which surfaces downstream as ``NaN`` so the trait's missing policy
decides — never a spurious flat ``0``. ``extract_graph_features`` assembles the
requested columns into a frame with a stable column set (globally-absent
concepts still get an all-``NaN`` column) so the Gower fit/transform column set
never shifts between cohorts.

The PREFIX block is inlined in every query so the templates run on *any* graph,
including one whose namespaces were never bound.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from rdflib import Graph

from src.graph_construction.temporal.allen_relations import ALLEN_PREDICATES
from src.pygower import Kind

# Declared inline (not relying on graph.bind) so queries work on any graph.
_PREFIXES = """\
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
PREFIX time: <http://www.w3.org/2006/time#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

# Predicates a free-text concept can match: a biomarker/vital *type* name, a
# drug *name*, or a drug *category* ("vasopressors"). One pattern, reused.
_CONCEPT_PREDICATES = (
    "mimic:hasBiomarkerType",
    "mimic:hasClinicalSignName",
    "mimic:hasDrugName",
    "mimic:hasDrugCategory",
)


# ---------------------------------------------------------------------------
# Request + template descriptors
# ---------------------------------------------------------------------------


@dataclass
class GraphFeatureRequest:
    """One graph-derived column to extract.

    Attributes:
        column: Output column name (becomes a trait/feature name).
        template: Key into :data:`TEMPLATES`.
        concept: The clinical concept the template keys on (biomarker label,
            drug name, or drug category). ``None`` for concept-free templates
            (e.g. distinct-drug-count, ICU LOS).
        params: Template-specific options (``agg``, ``window_hours``,
            ``concept_b``, ``relation``, ``as_bool``).
    """

    column: str
    template: str
    concept: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Template:
    """A feature extractor and the pygower :class:`Kind` of its output."""

    fn: Callable[[Graph, GraphFeatureRequest], dict[int, float]]
    kind: Callable[[GraphFeatureRequest], Kind]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _sanitize(value: str) -> str:
    """Escape characters that could break a SPARQL string literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _parse_ts(node) -> datetime:
    """Parse a dateTimeStamp literal (``...Z``) into a naive ``datetime``."""
    s = str(node)
    if s.endswith("Z"):
        s = s[:-1]
    return datetime.fromisoformat(s)


def _num(node) -> float:
    """Coerce a *guaranteed-numeric* rdflib literal (e.g. a COUNT) to ``float``.

    For readings that come from free-text clinical columns and may not parse,
    use :func:`_maybe_num` instead — this one is for aggregates the query itself
    guarantees numeric.
    """
    try:
        return float(node.toPython())
    except (AttributeError, TypeError, ValueError):
        return float(str(node))


def _maybe_num(node) -> float | None:
    """Best-effort coerce an rdflib literal to a *finite* ``float``, else ``None``.

    MIMIC's ``labevents.value`` and dose columns are free text: a range result
    (``"2.4-3.6"``), a qualified result (``"<0.1"``), or an error token can
    reach the graph even when the numeric column is null. Such a reading has no
    usable value for a slope/delta, so it is *omitted* (the module's
    missing-value policy) rather than aborting the whole cohort with a raw
    ``could not convert string to float`` ``ValueError``. Non-finite values
    (NaN/inf) are screened too, so they never poison ``np.polyfit``.
    """
    try:
        v = float(node.toPython())
    except (AttributeError, TypeError, ValueError):
        try:
            v = float(str(node))
        except (TypeError, ValueError):
            return None
    return v if np.isfinite(v) else None


def _concept_match_block(var: str, cname: str) -> str:
    """A UNION that binds ``?{cname}`` from any concept-naming predicate of ``?{var}``."""
    arms = " UNION ".join(
        f"{{ ?{var} {pred} ?{cname} . }}" for pred in _CONCEPT_PREDICATES
    )
    return f"{{ {arms} }}"


def _run(graph: Graph, query: str):
    """Run a SELECT with the shared prefixes prepended."""
    return graph.query(_PREFIXES + query)


def _aggregate_series(
    rows: list[tuple[datetime, float]], agg: str, window_hours: float | None
) -> float | None:
    """Reduce a ``(timestamp, value)`` trajectory to a scalar trend.

    A trajectory needs >=2 readings; otherwise there is no defined trend and we
    return ``None`` (absent), so a single reading never masquerades as a flat 0.
    """
    rows = sorted(rows, key=lambda r: r[0])
    if window_hours is not None and rows:
        first = rows[0][0]
        rows = [
            r
            for r in rows
            if (r[0] - first).total_seconds() / 3600.0 <= window_hours + 1e-9
        ]
    if len(rows) < 2:
        return None

    ts0 = rows[0][0]
    xs = [(r[0] - ts0).total_seconds() / 3600.0 for r in rows]
    ys = [r[1] for r in rows]

    if agg == "delta":
        return ys[-1] - ys[0]
    # slope (per hour); undefined if every reading shares one timestamp.
    if np.ptp(xs) == 0:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def _series_by_admission(graph: Graph, req: GraphFeatureRequest) -> dict[int, float]:
    """Biomarker / vital trajectory aggregated per admission (slope or delta)."""
    concept = _sanitize(req.concept or "")
    query = f"""
SELECT DISTINCT ?hadm ?e ?value ?ts WHERE {{
    ?adm a mimic:HospitalAdmission ;
         mimic:hasAdmissionId ?hadm ;
         mimic:containsICUStay ?stay .
    ?e mimic:associatedWithICUStay ?stay ;
       mimic:hasValue ?value ;
       time:inXSDDateTimeStamp ?ts .
    {{ {{ ?e mimic:hasBiomarkerType ?cname . }}
       UNION {{ ?e mimic:hasClinicalSignName ?cname . }} }}
    FILTER(LCASE(STR(?cname)) = LCASE("{concept}"))
}}
"""
    # Dedupe by event URI so a UNION never double-weights one reading. A
    # non-numeric reading (range/qualified/error text) is omitted, not fatal.
    per_admission: dict[int, dict[Any, tuple[datetime, float]]] = defaultdict(dict)
    for hadm, event, value, ts in _run(graph, query):
        v = _maybe_num(value)
        if v is None:
            continue
        per_admission[int(hadm.toPython())][event] = (_parse_ts(ts), v)

    agg = req.params.get("agg", "slope")
    window = req.params.get("window_hours")
    out: dict[int, float] = {}
    for hadm, events in per_admission.items():
        scalar = _aggregate_series(list(events.values()), agg, window)
        if scalar is not None:
            out[hadm] = scalar
    return out


def _dose_series(graph: Graph, req: GraphFeatureRequest) -> dict[int, float]:
    """Per-administration dose trajectory for a drug name or category."""
    concept = _sanitize(req.concept or "")
    query = f"""
SELECT DISTINCT ?hadm ?e ?dose ?ts WHERE {{
    ?adm a mimic:HospitalAdmission ;
         mimic:hasAdmissionId ?hadm ;
         mimic:containsICUStay ?stay .
    ?e mimic:associatedWithICUStay ?stay ;
       mimic:hasDoseValue ?dose ;
       time:hasBeginning ?b .
    ?b time:inXSDDateTimeStamp ?ts .
    {{ {{ ?e mimic:hasDrugName ?cname . }}
       UNION {{ ?e mimic:hasDrugCategory ?cname . }} }}
    FILTER(LCASE(STR(?cname)) = LCASE("{concept}"))
}}
"""
    per_admission: dict[int, dict[Any, tuple[datetime, float]]] = defaultdict(dict)
    for hadm, event, dose, ts in _run(graph, query):
        v = _maybe_num(dose)
        if v is None:
            continue
        per_admission[int(hadm.toPython())][event] = (_parse_ts(ts), v)

    agg = req.params.get("agg", "slope")
    window = req.params.get("window_hours")
    out: dict[int, float] = {}
    for hadm, events in per_admission.items():
        scalar = _aggregate_series(list(events.values()), agg, window)
        if scalar is not None:
            out[hadm] = scalar
    return out


def _distinct_drug_count(graph: Graph, req: GraphFeatureRequest) -> dict[int, float]:
    """Number of distinct prescribed drugs per admission."""
    query = """
SELECT ?hadm (COUNT(DISTINCT ?drug) AS ?n) WHERE {
    ?adm a mimic:HospitalAdmission ;
         mimic:hasAdmissionId ?hadm ;
         mimic:containsICUStay ?stay .
    ?e mimic:associatedWithICUStay ?stay ;
       mimic:hasDrugName ?drug .
} GROUP BY ?hadm
"""
    out: dict[int, float] = {}
    for hadm, n in _run(graph, query):
        out[int(hadm.toPython())] = _num(n)
    return out


def _icu_los(graph: Graph, req: GraphFeatureRequest) -> dict[int, float]:
    """Total ICU length of stay (hours) summed over an admission's stays."""
    query = """
SELECT ?hadm ?stay ?intime ?outtime WHERE {
    ?adm a mimic:HospitalAdmission ;
         mimic:hasAdmissionId ?hadm ;
         mimic:containsICUStay ?stay .
    ?stay time:hasBeginning ?bi .
    ?bi time:inXSDDateTimeStamp ?intime .
    OPTIONAL { ?stay time:hasEnd ?eo . ?eo time:inXSDDateTimeStamp ?outtime . }
}
"""
    per_stay: dict[int, dict[Any, float]] = defaultdict(dict)
    for hadm, stay, intime, outtime in _run(graph, query):
        if outtime is None:
            continue
        hours = (_parse_ts(outtime) - _parse_ts(intime)).total_seconds() / 3600.0
        per_stay[int(hadm.toPython())][stay] = hours
    return {hadm: sum(stays.values()) for hadm, stays in per_stay.items()}


def _time_to_first_event(graph: Graph, req: GraphFeatureRequest) -> dict[int, float]:
    """Hours from ICU intime to the first event of a concept."""
    concept = _sanitize(req.concept or "")
    query = f"""
SELECT ?hadm ?intime ?ts WHERE {{
    ?adm a mimic:HospitalAdmission ;
         mimic:hasAdmissionId ?hadm ;
         mimic:containsICUStay ?stay .
    ?stay time:hasBeginning ?bi .
    ?bi time:inXSDDateTimeStamp ?intime .
    ?e mimic:associatedWithICUStay ?stay .
    {_concept_match_block("e", "cname")}
    FILTER(LCASE(STR(?cname)) = LCASE("{concept}"))
    {{ {{ ?e time:inXSDDateTimeStamp ?ts . }}
       UNION {{ ?e time:hasBeginning ?bb . ?bb time:inXSDDateTimeStamp ?ts . }} }}
}}
"""
    out: dict[int, float] = {}
    for hadm, intime, ts in _run(graph, query):
        hours = (_parse_ts(ts) - _parse_ts(intime)).total_seconds() / 3600.0
        key = int(hadm.toPython())
        if key not in out or hours < out[key]:
            out[key] = hours
    return out


def _precedence_count(graph: Graph, req: GraphFeatureRequest) -> dict[int, float]:
    """Count Allen-relation edges from concept-A events to concept-B events."""
    concept_a = _sanitize(req.concept or "")
    concept_b = _sanitize(req.params.get("concept_b") or "")
    relation = req.params.get("relation", "before")
    rel_uri = ALLEN_PREDICATES[relation]
    query = f"""
SELECT DISTINCT ?hadm ?a ?b WHERE {{
    ?adm a mimic:HospitalAdmission ;
         mimic:hasAdmissionId ?hadm ;
         mimic:containsICUStay ?stay .
    ?a mimic:associatedWithICUStay ?stay .
    ?b mimic:associatedWithICUStay ?stay .
    ?a <{rel_uri}> ?b .
    {_concept_match_block("a", "an")}
    FILTER(LCASE(STR(?an)) = LCASE("{concept_a}"))
    {_concept_match_block("b", "bn")}
    FILTER(LCASE(STR(?bn)) = LCASE("{concept_b}"))
}}
"""
    counts: dict[int, int] = defaultdict(int)
    for hadm, _a, _b in _run(graph, query):
        counts[int(hadm.toPython())] += 1

    as_bool = bool(req.params.get("as_bool", False))
    return {
        hadm: (1.0 if as_bool else float(n)) for hadm, n in counts.items() if n > 0
    }


def _kind_quantitative(req: GraphFeatureRequest) -> Kind:
    return Kind.QUANTITATIVE


def _kind_precedence(req: GraphFeatureRequest) -> Kind:
    return Kind.BINARY if req.params.get("as_bool") else Kind.QUANTITATIVE


TEMPLATES: dict[str, _Template] = {
    "sim_series_by_admission": _Template(_series_by_admission, _kind_quantitative),
    "sim_dose_series": _Template(_dose_series, _kind_quantitative),
    "sim_distinct_drug_count": _Template(_distinct_drug_count, _kind_quantitative),
    "sim_icu_los": _Template(_icu_los, _kind_quantitative),
    "sim_time_to_first_event": _Template(_time_to_first_event, _kind_quantitative),
    "sim_precedence_count": _Template(_precedence_count, _kind_precedence),
}


# ---------------------------------------------------------------------------
# Public assembly API
# ---------------------------------------------------------------------------


def extract_graph_features(
    graph: Graph, requests: list[GraphFeatureRequest]
) -> pd.DataFrame:
    """Assemble requested graph features into a ``hadm_id``-indexed frame.

    The column set is stable: a globally-absent concept still yields an
    all-``NaN`` column, so the Gower fit/transform column set never shifts
    between cohorts. Admissions with no value for a column carry ``NaN`` there.
    """
    if not requests:
        return pd.DataFrame()

    columns: dict[str, dict[int, float]] = {}
    all_hadms: set[int] = set()
    for req in requests:
        template = TEMPLATES[req.template]  # KeyError on unknown template.
        values = template.fn(graph, req)
        columns[req.column] = values
        all_hadms.update(values.keys())

    index = sorted(all_hadms)
    data = {
        req.column: [columns[req.column].get(hadm, np.nan) for hadm in index]
        for req in requests
    }
    df = pd.DataFrame(data, index=index, columns=[req.column for req in requests])
    df.index.name = "hadm_id"
    return df


def feature_kinds(requests: list[GraphFeatureRequest]) -> dict[str, Kind]:
    """Map each requested column to its pygower :class:`Kind`."""
    return {req.column: TEMPLATES[req.template].kind(req) for req in requests}


# ---------------------------------------------------------------------------
# Semantic signature — a name-independent key for a graph feature's scale.
# ---------------------------------------------------------------------------


# The params that change a template's numeric *scale* (and so must select a
# different frozen normalization range), in canonical order. Anything not listed
# does not move the scale and is excluded so cosmetic param differences don't
# fragment the key. Templates absent from this map are scale-determined by their
# (template, concept) alone (ICU LOS, distinct-drug count, time-to-first-event).
_SIGNATURE_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "sim_series_by_admission": ("agg", "window_hours"),
    "sim_dose_series": ("agg", "window_hours"),
    "sim_precedence_count": ("relation", "concept_b"),
}

# Effective defaults the templates apply when a param is omitted, so a trait that
# states ``agg="slope"`` and one that relies on the default hash identically.
_SIGNATURE_PARAM_DEFAULTS: dict[str, Any] = {"agg": "slope", "relation": "before"}


def _norm_param(value: Any) -> str:
    """Canonical token for a param value (so ``48`` and ``48.0`` and ``"48"`` agree)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, str):
        return value.strip().lower()
    return str(value)


def graph_feature_signature(
    template: str, concept: str | None, params: Mapping[str, Any] | None
) -> str:
    """A name-independent key for a graph feature's *normalization scale*.

    Two traits that pull the same ``template`` on the same ``concept`` with the
    same scale-relevant aggregation (slope vs delta, window, precedence partner)
    share a signature regardless of the LLM-chosen column ``name``. That is what
    lets a frozen population range — fit once on a fixed reference cohort (locked
    decision #6) — be looked up for a trait whose name was invented per query
    (``lactate_slope_48h``, ``vasopressor_dose_slope``, …). The concept match in
    the SPARQL templates is case-insensitive, so the concept token is lower-cased
    here to match; absent scale params fall back to the template's effective
    default so an explicit and an implicit value collide on the same key.
    """
    params = params or {}
    parts = [template, f"concept={(concept or '').strip().lower()}"]
    for key in _SIGNATURE_PARAM_KEYS.get(template, ()):
        if key in params:
            value = params[key]
        elif key in _SIGNATURE_PARAM_DEFAULTS:
            value = _SIGNATURE_PARAM_DEFAULTS[key]
        else:
            continue  # absent and no default (e.g. window_hours) → omit
        if value is None:
            continue
        parts.append(f"{key}={_norm_param(value)}")
    return "|".join(parts)


def request_signature(req: GraphFeatureRequest) -> str:
    """The :func:`graph_feature_signature` of a :class:`GraphFeatureRequest`."""
    return graph_feature_signature(req.template, req.concept, req.params)


# ---------------------------------------------------------------------------
# Shared per-question graph build — the one place that turns a backend + a
# concept set + a request set into a hadm-indexed feature frame. The cohort
# runner (scoring the query pool) and the reference-range builder (fitting the
# frozen scale on a fixed cohort) both go through here, so the column a range
# normalizes is produced by exactly the same code that produces the column it
# is later compared against (no SQL/graph or build-path divergence).
# ---------------------------------------------------------------------------


def build_graph_feature_frame(
    backend: Any,
    *,
    prefilters: list,
    concepts: list,
    requests: list[GraphFeatureRequest],
    ontology_dir,
    resolver: Any = None,
    extraction_config: Any = None,
    drug_category_resolver: Any = None,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Extract → build the RDF graph over the cohort → pull the feature frame.

    ``prefilters`` gates the cohort (the query's pool, or a fixed reference
    population); ``concepts`` are the clinical concepts to pull into the graph;
    ``requests`` name the columns to compute. The (expensive) Allen-relation
    pass runs only when a precedence request needs it (locked decision I-D).
    Imports the conversational extractor / graph builder lazily so importing
    this module never drags those (or rdflib's heavier deps) in at load time.
    """
    from src.conversational.extractor import _extract
    from src.conversational.graph_builder import build_query_graph
    from src.conversational.models import CompetencyQuestion

    cq = CompetencyQuestion(
        original_question="cohort graph-feature extraction",
        patient_filters=list(prefilters),
        clinical_concepts=list(concepts),
        scope="cohort",
    )
    extraction = _extract(backend, cq, config=extraction_config, resolver=resolver)
    has_precedence = any(r.template == "sim_precedence_count" for r in requests)
    graph, _stats = build_query_graph(
        ontology_dir, extraction,
        skip_allen_relations=not has_precedence,
        max_workers=max_workers,
        drug_category_resolver=drug_category_resolver,
    )
    return extract_graph_features(graph, requests)
