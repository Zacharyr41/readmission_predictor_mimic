"""Top-level entry point for the patient-similarity engine (Phase 9).

``run_similarity(spec, backend)`` wires the three compute modules
(contextual + temporal + combined) together and produces a ranked
``SimilarityResult``.

**Pipeline**:

1. Resolve the anchor — real patient ⇒ SQL fetch of features + events;
   template ⇒ covariate dict + no trajectory.
2. Pull the candidate pool — all ``hadm_id`` rows except the anchor,
   narrowed by ``spec.candidate_filters`` (future extension point;
   pass-through in 8d).
3. Build the minimal contextual feature matrix for the pool via the
   lightweight inline SQL extractor in this module. (Commit 7 will
   optionally upgrade to ``src.feature_extraction.feature_builder``
   for richer features; the shape of this module's output is a
   subset of that schema so the swap is drop-in.)
4. Build bucketed event sets per admission via
   ``src.similarity.bucketing.assign_buckets``.
5. Score contextual + temporal + combined.
6. Filter by ``min_similarity``, cap at ``top_k``, package into
   a ``SimilarityResult`` with provenance.

**Feature extractor scope (8d shipment)**: demographics (age,
gender, admission_type), ICU length of stay, a minimal-severity
placeholder triplet (creatinine_max / sodium_mean / platelet_min —
pulled from labevents where available, otherwise population-median
defaults), and zero-initialised Charlson + SNOMED flags. The zero-
fill means comorbidity groups contribute only the baseline "both
zero ⇒ matched" signal across all candidates; ranking is driven by
demographics + severity + temporal in the end-to-end smoke. The
full comorbidity extraction plugs in here in a follow-on commit
that also wires ``build_feature_matrix``.

Plan: /Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pygower import ColumnSpec, Kind, gower_distances
from src.similarity.bucketing import assign_buckets
from src.similarity.combined import combine_scores
from src.similarity.contextual import (
    CHARLSON_WEIGHTS,
    compute_contextual_similarity,
)
from src.similarity.models import (
    CohortDefinition,
    CohortMember,
    CohortResult,
    SimilarityResult,
    SimilarityScore,
    SimilaritySpec,
    TraitContribution,
)
from src.similarity.temporal import compute_temporal_similarity


# ---------------------------------------------------------------------------
# Feature extraction (inline, minimal).
# ---------------------------------------------------------------------------


# SNOMED-group flag columns used by ``compute_contextual_similarity``.
# Zero-filled for 8d; the extractor upgrade will populate these from the
# ontology mapping that already lives in ``src/causal/interventions.py``.
_DEFAULT_SNOMED_GROUPS = (
    "snomed_group_I48",   # afib
    "snomed_group_I63",   # stroke
    "snomed_group_N18",   # CKD
    "snomed_group_E11",   # diabetes
    "snomed_group_J44",   # COPD
)


def _apply_defaults(row: dict) -> dict:
    """Fill missing columns with sensible defaults so contextual
    scoring doesn't crash on NaN. Zero for flags / indices; clinical
    midpoints for severity; 0 for LOS / social."""
    defaults = {
        "age": 0, "gender_M": 0, "gender_F": 0, "gender_unknown": 0,
        # Raw admission_type label (nominal). Missing → None so the contextual
        # nominal match treats an unknown admission type as dissimilar (the same
        # missing-policy the gender flags follow), instead of a dead one-hot.
        "admission_type": None,
        "charlson_index": 0,
        "creatinine_max": 1.0, "sodium_mean": 140.0, "platelet_min": 220.0,
        "icu_los_hours": 0.0,
        "language_barrier": 0, "is_neuro_service": 0,
    }
    for flag in CHARLSON_WEIGHTS:
        defaults[flag] = 0
    for g in _DEFAULT_SNOMED_GROUPS:
        defaults[g] = 0
    for k, v in defaults.items():
        row.setdefault(k, v)
    return row


def _fetch_admission_features(backend: Any, hadm_ids: list[int]) -> pd.DataFrame:
    """Pull demographic + ICU-LOS + minimal-severity features for a
    list of hadm_ids. Returns a DataFrame keyed by ``hadm_id`` + all
    feature columns expected by ``compute_contextual_similarity``.
    """
    if not hadm_ids:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(hadm_ids))
    rows = backend.execute(
        f"""
        SELECT a.hadm_id, a.subject_id, p.anchor_age, p.gender,
               a.admission_type
        FROM admissions a
        JOIN patients p ON a.subject_id = p.subject_id
        WHERE a.hadm_id IN ({placeholders})
        """,
        list(hadm_ids),
    )
    base = pd.DataFrame(
        rows, columns=["hadm_id", "subject_id", "age", "gender", "admission_type"],
    )
    base["gender_M"] = (base["gender"] == "M").astype(int)
    base["gender_F"] = (base["gender"] == "F").astype(int)
    base["gender_unknown"] = (~base["gender"].isin(["M", "F"])).astype(int)
    # admission_type is kept as its RAW schema label (a nominal feature scored
    # by an identity match), not one-hot encoded. The old EMERGENCY/ELECTIVE/
    # URGENT one-hot was a MIMIC-III vocabulary that mapped every MIMIC-IV row to
    # "_other", so it contributed zero discriminating signal; the raw compare
    # needs no hardcoded value list and works for any schema vocabulary.

    # ICU LOS aggregated across stays.
    icu_rows = backend.execute(
        f"""
        SELECT hadm_id, SUM(los) * 24.0 AS icu_los_hours
        FROM icustays
        WHERE hadm_id IN ({placeholders})
        GROUP BY hadm_id
        """,
        list(hadm_ids),
    )
    icu_df = pd.DataFrame(icu_rows, columns=["hadm_id", "icu_los_hours"])
    base = base.merge(icu_df, on="hadm_id", how="left")
    base["icu_los_hours"] = base["icu_los_hours"].fillna(0.0)

    # Minimal severity features from labevents — keep extractor small.
    try:
        lab_rows = backend.execute(
            f"""
            SELECT l.hadm_id,
                   MAX(CASE WHEN l.itemid = 50912 THEN l.valuenum END) AS creatinine_max,
                   AVG(CASE WHEN l.itemid = 50983 THEN l.valuenum END) AS sodium_mean,
                   MIN(CASE WHEN l.itemid = 51265 THEN l.valuenum END) AS platelet_min
            FROM labevents l
            WHERE l.hadm_id IN ({placeholders})
            GROUP BY l.hadm_id
            """,
            list(hadm_ids),
        )
        lab_df = pd.DataFrame(
            lab_rows, columns=["hadm_id", "creatinine_max", "sodium_mean", "platelet_min"],
        )
        base = base.merge(lab_df, on="hadm_id", how="left")
    except Exception:
        # labevents may not exist on all backends; fall back to defaults.
        pass

    # Zero-fill Charlson + SNOMED flags (commit 7+ will populate these).
    for flag in CHARLSON_WEIGHTS:
        base[flag] = 0
    base["charlson_index"] = 0
    for g in _DEFAULT_SNOMED_GROUPS:
        base[g] = 0
    base["language_barrier"] = 0
    base["is_neuro_service"] = 0

    # Default-fill any remaining NaN on the defined schema.
    base = base.apply(
        lambda col: col.fillna(col.median() if col.dtype.kind in "if" else 0)
        if col.isna().any() else col
    )
    return base


def _fetch_admission_events(
    backend: Any, hadm_id: int,
) -> tuple[list[dict], datetime | None, datetime | None, list[dict]]:
    """Pull the event stream + admission window + ICU stays for one
    admission. Events are currently just prescriptions — commit 7
    extends to labs / procedures / vitals."""
    adm = backend.execute(
        "SELECT admittime, dischtime FROM admissions WHERE hadm_id = ?",
        [hadm_id],
    )
    if not adm:
        return [], None, None, []
    adm_start = pd.to_datetime(adm[0][0])
    adm_end = pd.to_datetime(adm[0][1])

    stay_rows = backend.execute(
        "SELECT intime, outtime FROM icustays WHERE hadm_id = ?",
        [hadm_id],
    )
    icu_stays = [
        {"intime": pd.to_datetime(r[0]), "outtime": pd.to_datetime(r[1])}
        for r in stay_rows
        if r[0] is not None and r[1] is not None
    ]

    rx_rows = backend.execute(
        """
        SELECT drug, starttime FROM prescriptions WHERE hadm_id = ?
        """,
        [hadm_id],
    )
    events: list[dict] = []
    for drug, ts in rx_rows:
        if ts is None or drug is None:
            continue
        events.append({
            "code": f"snomed_drug:{str(drug).strip().lower()}",
            "timestamp": pd.to_datetime(ts),
        })

    return events, adm_start.to_pydatetime(), adm_end.to_pydatetime(), icu_stays


# ---------------------------------------------------------------------------
# Anchor resolution.
# ---------------------------------------------------------------------------


def _resolve_anchor(
    spec: SimilaritySpec, backend: Any,
) -> tuple[dict, int | None, dict[str, set[str]], int | None, str]:
    """Return ``(anchor_features, anchor_hadm_id, anchor_buckets,
    anchor_subject_id, anchor_description)``.

    For template anchors, ``anchor_buckets`` is empty and
    ``anchor_hadm_id`` is None — downstream temporal scoring degrades
    to contextual-only.
    """
    if spec.anchor_template is not None:
        features = _apply_defaults(dict(spec.anchor_template))
        keys = ", ".join(
            f"{k}={v}" for k, v in sorted(spec.anchor_template.items())
        )
        return features, None, {}, None, f"template anchor: {keys}"

    if spec.anchor_hadm_id is not None:
        hadm = spec.anchor_hadm_id
        df = _fetch_admission_features(backend, [hadm])
        if df.empty:
            raise ValueError(
                f"SimilaritySpec.anchor_hadm_id={hadm} resolved to no admission"
            )
        anchor_row = df.iloc[0].to_dict()
        events, adm_start, adm_end, icu_stays = _fetch_admission_events(backend, hadm)
        buckets: dict[str, set[str]] = {}
        if adm_start is not None and adm_end is not None:
            buckets, _ = assign_buckets(events, adm_start, adm_end, icu_stays)
        subj = int(anchor_row["subject_id"])
        desc = (
            f"hadm_id={hadm} (subject {subj}, "
            f"{int(anchor_row['age'])}yo {anchor_row['gender']})"
        )
        return _apply_defaults(anchor_row), hadm, buckets, subj, desc

    # spec validator guarantees exactly one — anchor_subject_id branch.
    subject_id = spec.anchor_subject_id
    rows = backend.execute(
        "SELECT hadm_id FROM admissions WHERE subject_id = ? ORDER BY admittime",
        [subject_id],
    )
    if not rows:
        raise ValueError(
            f"SimilaritySpec.anchor_subject_id={subject_id} resolved to no admissions"
        )
    # Use the most recent admission as representative.
    hadm = int(rows[-1][0])
    df = _fetch_admission_features(backend, [hadm])
    anchor_row = df.iloc[0].to_dict()
    events, adm_start, adm_end, icu_stays = _fetch_admission_events(backend, hadm)
    buckets = {}
    if adm_start is not None and adm_end is not None:
        buckets, _ = assign_buckets(events, adm_start, adm_end, icu_stays)
    desc = (
        f"subject_id={subject_id} "
        f"(representative hadm {hadm}, {int(anchor_row['age'])}yo {anchor_row['gender']})"
    )
    return _apply_defaults(anchor_row), hadm, buckets, subject_id, desc


# ---------------------------------------------------------------------------
# Top-level runner.
# ---------------------------------------------------------------------------


def _fetch_candidate_hadm_ids(backend: Any, exclude: int | None) -> list[int]:
    """All admissions in the DB, minus the anchor (if any).

    8d doesn't yet honor ``spec.candidate_filters`` (extension point
    for commit 7+); this returns the full pool.
    """
    if exclude is None:
        rows = backend.execute("SELECT hadm_id FROM admissions", [])
    else:
        rows = backend.execute(
            "SELECT hadm_id FROM admissions WHERE hadm_id != ?", [exclude],
        )
    return [int(r[0]) for r in rows]


def run_similarity(spec: SimilaritySpec, backend: Any) -> SimilarityResult:
    """Compute a ranked similarity result against ``backend``."""
    anchor_features, anchor_hadm_id, anchor_buckets, _anchor_subj, anchor_description = (
        _resolve_anchor(spec, backend)
    )

    candidate_hadm_ids = _fetch_candidate_hadm_ids(backend, anchor_hadm_id)
    n_pool = len(candidate_hadm_ids)

    if n_pool == 0:
        return SimilarityResult(
            anchor_description=anchor_description,
            n_pool=0,
            n_returned=0,
            scores=[],
            spec=spec,
            provenance={"note": "no candidates found"},
        )

    candidate_df = _fetch_admission_features(backend, candidate_hadm_ids)

    ctx_scores = compute_contextual_similarity(
        anchor_features=anchor_features,
        candidate_features_df=candidate_df,
        weights=spec.contextual_weights,
    )

    anchor_has_trajectory = bool(anchor_buckets)
    candidate_buckets_by_hadm: dict[int, dict[str, set[str]]] = {}
    if anchor_has_trajectory:
        for hadm in candidate_hadm_ids:
            events, adm_start, adm_end, icu_stays = _fetch_admission_events(backend, hadm)
            if adm_start is None or adm_end is None:
                candidate_buckets_by_hadm[hadm] = {}
                continue
            buckets, _ = assign_buckets(events, adm_start, adm_end, icu_stays)
            candidate_buckets_by_hadm[hadm] = buckets

    temp_scores = (
        compute_temporal_similarity(
            anchor_buckets=anchor_buckets,
            candidate_buckets_by_hadm=candidate_buckets_by_hadm,
            decay=spec.temporal_decay,
        )
        if anchor_has_trajectory
        else {}
    )

    # Map hadm_id → subject_id for the score object.
    subject_by_hadm = dict(zip(candidate_df["hadm_id"], candidate_df["subject_id"]))

    similarity_scores: list[SimilarityScore] = []
    for hadm in candidate_hadm_ids:
        ctx = ctx_scores.get(hadm)
        if ctx is None:
            continue
        temp = temp_scores.get(hadm) if anchor_has_trajectory else None
        combined = combine_scores(ctx, temp, spec.temporal_weight)
        temp_value = (
            temp.overall_score
            if temp is not None and temp.temporal_available
            else None
        )
        similarity_scores.append(
            SimilarityScore(
                hadm_id=hadm,
                subject_id=int(subject_by_hadm.get(hadm, 0)),
                combined=combined,
                contextual=ctx.overall_score,
                temporal=temp_value,
                contextual_explanation=ctx,
                temporal_explanation=temp,
            )
        )

    similarity_scores.sort(key=lambda s: -s.combined)

    if spec.min_similarity is not None:
        similarity_scores = [
            s for s in similarity_scores if s.combined >= spec.min_similarity
        ]
    if spec.top_k is not None:
        similarity_scores = similarity_scores[: spec.top_k]

    provenance = {
        "n_pool": n_pool,
        "anchor_resolved": (
            "hadm_id" if spec.anchor_hadm_id is not None
            else "subject_id" if spec.anchor_subject_id is not None
            else "template"
        ),
        "anchor_has_trajectory": anchor_has_trajectory,
        "temporal_weight": spec.temporal_weight,
        "temporal_decay": spec.temporal_decay,
        "feature_extractor_version": "phase9-commit5-minimal",
    }

    return SimilarityResult(
        anchor_description=anchor_description,
        n_pool=n_pool,
        n_returned=len(similarity_scores),
        scores=similarity_scores,
        spec=spec,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Anchorless cohort runner (plan III-B) — one-vs-many distance to a synthesized
# reference profile, thresholded. ``sql`` traits are pulled by the contextual
# extractor; ``graph_temporal`` traits (plan III-A) are pulled from a per-question
# RDF graph built over the prefiltered pool (the graph IS the temporal feature
# extractor: slopes, dose trajectories, durations, precedence become columns).
# ---------------------------------------------------------------------------


def _cohort_candidate_pool(definition: CohortDefinition, backend: Any) -> list[int]:
    """Hadm pool after the definition's Boolean prefilters.

    With no prefilters the pool is every admission. Prefilters compile through
    the shared ``OperationRegistry`` (``src.conversational``), so any registered
    filter field (age, gender, admission_type, diagnosis, …) narrows the pool
    here for free — there is exactly one place that knows how to turn a
    ``PatientFilter`` into SQL, and the cohort path reuses it.
    """
    if not definition.prefilters:
        return _fetch_candidate_hadm_ids(backend, exclude=None)
    from src.conversational.extractor import _get_filtered_hadm_ids

    return _get_filtered_hadm_ids(backend, definition.prefilters)


def _build_column_specs(
    definition: CohortDefinition,
    reference_ranges: dict[str, tuple[float, float]] | None,
) -> dict[str, ColumnSpec]:
    """Project each trait onto a frozen-range pygower ``ColumnSpec``.

    A quantitative trait MUST carry a normalization range — either on the trait
    (``range_``) or via ``reference_ranges`` (the frozen reference-population
    stats from plan II-E). Refusing to fall back to a batch-learned range is
    locked decision #6 (fit/transform separation): a single-row profile would
    otherwise yield a degenerate range and meaningless distances.
    """
    ranges = reference_ranges or {}
    spec: dict[str, ColumnSpec] = {}
    for t in definition.traits:
        cs = t.to_column_spec()
        if cs.kind == Kind.QUANTITATIVE and cs.range_ is None:
            rr = ranges.get(t.name)
            if rr is None:
                raise ValueError(
                    f"quantitative trait {t.name!r} has no frozen range_ and "
                    "none was supplied via reference_ranges; run_cohort refuses "
                    "to learn a normalization range from the query batch "
                    "(fit/transform separation, locked decision #6)"
                )
            cs = replace(cs, range_=(float(rr[0]), float(rr[1])))
        spec[t.name] = cs
    return spec


def _cohort_provenance(definition: CohortDefinition, n_pool: int) -> dict:
    """The logged criteria: prefilters + every trait's kernel-relevant config."""
    return {
        "n_pool": n_pool,
        "distance_threshold": definition.distance_threshold,
        "top_k": definition.top_k,
        "prefilters": [f.model_dump() for f in definition.prefilters],
        "traits": [
            {
                "name": t.name,
                "source": t.source,
                "kind": t.kind.value,
                "direction": t.direction.value,
                "weight": t.weight,
                "reference_value": t.reference_value,
            }
            for t in definition.traits
        ],
        "feature_extractor_version": "phase-cohort-contextual-v1",
    }


def _trait_concepts(trait: Any) -> list[tuple[str, str]]:
    """The ``(concept_name, concept_type)`` pairs a graph trait needs extracted.

    The primary ``concept`` (defaulting to a biomarker type) plus, for a
    precedence trait, the ``concept_b`` partner (defaulting to a drug). Concept-
    free templates (ICU LOS, distinct-drug count) contribute no pairs — the
    extractor always pulls the structural rows (patients / admissions / stays)
    those templates read.
    """
    pairs: list[tuple[str, str]] = []
    if trait.concept:
        pairs.append((trait.concept, trait.concept_type or "biomarker"))
    cb = trait.graph_params.get("concept_b")
    if cb:
        pairs.append((cb, trait.graph_params.get("concept_b_type", "drug")))
    return pairs


def _merge_graph_traits(
    candidate_df: pd.DataFrame,
    definition: CohortDefinition,
    graph_traits: list,
    *,
    backend: Any,
    ontology_dir: Path,
    resolver: Any,
    extraction_config: Any,
    drug_category_resolver: Any,
    max_workers: int,
) -> dict:
    """Build the per-question RDF graph over the pool and merge graph columns.

    The graph IS the temporal feature extractor (plan III-A): each ``graph_temporal``
    trait names a feature-extractor ``template`` keyed on a clinical ``concept``.
    We synthesize a ``CompetencyQuestion`` from the definition's prefilters + those
    concepts, run the shared extractor, build the graph, and pull a hadm-indexed
    feature frame — then write each trait's value back onto ``candidate_df`` as a
    column (NaN where the admission has no value, so the column set stays stable
    and the trait's missing policy decides). A precedence trait needs Allen
    relations, so we only pay the (expensive) Allen pass when one is present
    (locked decision I-D). Mutates ``candidate_df`` in place; returns the graph
    provenance to fold into the logged criteria.
    """
    from src.conversational.extractor import _extract
    from src.conversational.graph_builder import build_query_graph
    from src.conversational.models import ClinicalConcept, CompetencyQuestion
    from src.similarity.graph_features import (
        GraphFeatureRequest,
        extract_graph_features,
    )

    concepts: list[ClinicalConcept] = []
    seen: set[tuple[str, str]] = set()
    for t in graph_traits:
        for name, ctype in _trait_concepts(t):
            if (name, ctype) not in seen:
                seen.add((name, ctype))
                concepts.append(ClinicalConcept(name=name, concept_type=ctype))

    cq = CompetencyQuestion(
        original_question="cohort graph-feature extraction",
        patient_filters=list(definition.prefilters),
        clinical_concepts=concepts,
        scope="cohort",
    )
    extraction = _extract(backend, cq, config=extraction_config, resolver=resolver)

    has_precedence = any(t.template == "sim_precedence_count" for t in graph_traits)
    graph, _stats = build_query_graph(
        ontology_dir, extraction,
        skip_allen_relations=not has_precedence,
        max_workers=max_workers,
        drug_category_resolver=drug_category_resolver,
    )

    requests = [
        GraphFeatureRequest(
            column=t.name, template=t.template, concept=t.concept,
            params=dict(t.graph_params or {}),
        )
        for t in graph_traits
    ]
    gdf = extract_graph_features(graph, requests)

    # Explicit per-hadm lookup (robust to an empty / 0-row feature frame): every
    # candidate gets the column, NaN where the graph yielded no value.
    for t in graph_traits:
        mapping = gdf[t.name].to_dict() if t.name in gdf.columns else {}
        candidate_df[t.name] = [
            mapping.get(int(h), np.nan) for h in candidate_df["hadm_id"]
        ]

    return {
        "graph_built": True,
        "graph_skip_allen_relations": not has_precedence,
        "graph_traits": [t.name for t in graph_traits],
    }


def run_cohort(
    definition: CohortDefinition,
    backend: Any,
    *,
    reference_ranges: dict[str, tuple[float, float]] | None = None,
    ontology_dir: Path | None = None,
    resolver: Any = None,
    extraction_config: Any = None,
    drug_category_resolver: Any = None,
    max_workers: int = 1,
) -> CohortResult:
    """Compute an anchorless cohort by Gower distance to a synthesized profile.

    Steps:

    1. Narrow the candidate pool with the definition's Boolean prefilters.
    2. Pull the typed feature matrix for the pool — ``sql`` traits via the
       contextual extractor; ``graph_temporal`` traits (plan III-A) from a
       per-question RDF graph built over the pool (requires ``ontology_dir``).
    3. Synthesize the reference *profile* vector from each trait's
       ``reference_value`` (the X side of the one-vs-many distance).
    4. Score every candidate's Gower distance to the profile, with FROZEN ranges
       and the per-trait kernel (symmetric / one-sided / asymmetric-binary)
       selected from the trait's ``kind`` + ``direction``.
    5. Cohort = ``distance <= distance_threshold`` ranked nearest-first, capped
       at ``top_k``; each member carries per-trait signed contributions.

    A definition with ≥1 ``graph_temporal`` trait requires ``ontology_dir`` (so
    the graph can be built); without it a clear ``ValueError`` is raised.
    """
    graph_traits = [t for t in definition.traits if t.source == "graph_temporal"]
    if graph_traits:
        no_template = [t.name for t in graph_traits if not t.template]
        if no_template:
            raise ValueError(
                "graph_temporal trait(s) carry no feature-extractor `template`: "
                f"{no_template}; set TraitSpec.template (e.g. "
                "'sim_series_by_admission')"
            )
        if ontology_dir is None:
            raise ValueError(
                "cohort definition has graph_temporal trait(s) "
                f"{[t.name for t in graph_traits]} but no ontology_dir was "
                "supplied; the graph feature path needs the ontology to build "
                "the per-question RDF graph over the candidate pool"
            )

    pool = _cohort_candidate_pool(definition, backend)
    candidate_df = _fetch_admission_features(backend, pool)
    n_pool = len(candidate_df)
    provenance = _cohort_provenance(definition, n_pool)

    if n_pool == 0:
        return CohortResult(
            definition=definition, members=[], n_pool=0, n_returned=0,
            provenance={**provenance, "note": "no candidates after prefilters"},
        )

    if graph_traits:
        provenance.update(
            _merge_graph_traits(
                candidate_df, definition, graph_traits,
                backend=backend, ontology_dir=ontology_dir, resolver=resolver,
                extraction_config=extraction_config,
                drug_category_resolver=drug_category_resolver,
                max_workers=max_workers,
            )
        )

    trait_names = [t.name for t in definition.traits]
    missing_cols = [n for n in trait_names if n not in candidate_df.columns]
    if missing_cols:
        raise ValueError(
            "cohort traits reference feature columns not produced by the "
            f"contextual / graph extractors: {missing_cols}"
        )

    spec = _build_column_specs(definition, reference_ranges)
    profile = pd.DataFrame(
        [{t.name: t.reference_value for t in definition.traits}],
        columns=trait_names,
    )
    candidates = candidate_df[trait_names]

    distance_matrix, contribs = gower_distances(
        profile, candidates, spec=spec, return_contributions=True,
    )
    distances = distance_matrix[0]
    signed = [c.signed()[0] for c in contribs]  # one (m,) array per trait

    subject_by_hadm = dict(zip(candidate_df["hadm_id"], candidate_df["subject_id"]))
    hadms = candidate_df["hadm_id"].tolist()

    members: list[CohortMember] = []
    for j, hadm in enumerate(hadms):
        dist = distances[j]
        if np.isnan(dist) or dist > definition.distance_threshold:
            continue
        contributions = [
            TraitContribution(
                name=contribs[c].name,
                similarity=float(contribs[c].similarity[0, j]),
                signed=float(signed[c][j]),
                weight=float(contribs[c].weight),
                included=bool(contribs[c].delta[0, j] > 0),
            )
            for c in range(len(definition.traits))
        ]
        members.append(
            CohortMember(
                hadm_id=int(hadm),
                subject_id=int(subject_by_hadm.get(hadm, 0)),
                distance=float(dist),
                contributions=contributions,
            )
        )

    members.sort(key=lambda mbr: mbr.distance)
    if definition.top_k is not None:
        members = members[: definition.top_k]

    return CohortResult(
        definition=definition,
        members=members,
        n_pool=n_pool,
        n_returned=len(members),
        provenance=provenance,
    )


__all__ = ["run_similarity", "run_cohort"]
