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

from datetime import datetime
from typing import Any

import pandas as pd

from src.similarity.bucketing import assign_buckets
from src.similarity.combined import combine_scores
from src.similarity.contextual import (
    CHARLSON_WEIGHTS,
    compute_contextual_similarity,
)
from src.similarity.models import (
    SimilarityResult,
    SimilarityScore,
    SimilaritySpec,
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
        "admission_type_EMERGENCY": 0, "admission_type_ELECTIVE": 0,
        "admission_type_URGENT": 0, "admission_type_other": 0,
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
    for typ in ("EMERGENCY", "ELECTIVE", "URGENT"):
        base[f"admission_type_{typ}"] = (base["admission_type"] == typ).astype(int)
    base["admission_type_other"] = (
        ~base["admission_type"].isin(["EMERGENCY", "ELECTIVE", "URGENT"])
    ).astype(int)

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
                   AVG(CASE WHEN l.itemid = 50971 THEN l.valuenum END) AS sodium_mean,
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


__all__ = ["run_similarity"]
