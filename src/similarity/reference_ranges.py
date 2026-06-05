"""Frozen Gower normalization ranges for the cohort-by-similarity feature set.

Locked decision #6 (fit/transform separation): the ranges that normalize a
quantitative trait's Gower distance ``1 − |Δ| / R`` are **fit once** on a fixed
reference population and frozen into a committed artifact —
``data/mappings/similarity_reference_ranges.json`` — never re-learned from the
candidate batch of a single query. A one-vs-many cohort query scores against a
1-row profile, so a batch-learned range would be degenerate anyway; freezing is
both the principled and the only workable choice.

Two halves:

* :func:`compute_reference_ranges` (builder) runs robust p1/p99 percentiles over
  the **same per-admission aggregates** the cohort feature extractor pulls
  (``src/similarity/run.py:_fetch_admission_features``) — per-hadm ``MAX``
  creatinine, summed ICU hours, etc. — so the frozen range matches the column it
  normalizes. Features whose reference population has no spread (a single value
  or no measurements → ``R = 0``) are dropped rather than emitted as a
  divide-by-zero trap.
* :func:`load_reference_ranges` (loader) reads the committed JSON into the
  ``{feature: (low, high)}`` mapping the cohort runner injects as
  ``run_cohort(..., reference_ranges=...)``. It never raises on a missing or
  malformed file (returns ``{}``); a quantitative trait that then finds no
  frozen range raises a clear error in the runner instead.

The per-feature SQL mirrors ``_fetch_admission_features`` exactly; keep the two
in sync if the extractor's aggregates change.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Repo-root-relative default. ``__file__`` is ``src/similarity/reference_ranges.py``
# so three ``.parent`` hops land at the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REFERENCE_RANGES_PATH = (
    _REPO_ROOT / "data" / "mappings" / "similarity_reference_ranges.json"
)

# Module-level so tests can monkeypatch the artifact location.
REFERENCE_RANGES_PATH: Path = _DEFAULT_REFERENCE_RANGES_PATH


# Each entry is a SQL block yielding one ``(hadm_id, v)`` row per admission,
# where ``v`` is the quantitative feature value (NULL when unmeasured). These
# MUST stay aligned with ``_fetch_admission_features`` so the frozen range
# normalizes the same quantity the runner scores.
_FEATURE_QUERIES: dict[str, str] = {
    "age": (
        "SELECT a.hadm_id AS hadm_id, p.anchor_age AS v "
        "FROM admissions a JOIN patients p ON a.subject_id = p.subject_id"
    ),
    "icu_los_hours": (
        "SELECT hadm_id, SUM(los) * 24.0 AS v FROM icustays GROUP BY hadm_id"
    ),
    "creatinine_max": (
        "SELECT hadm_id, MAX(CASE WHEN itemid = 50912 THEN valuenum END) AS v "
        "FROM labevents GROUP BY hadm_id"
    ),
    "sodium_mean": (
        "SELECT hadm_id, AVG(CASE WHEN itemid = 50983 THEN valuenum END) AS v "
        "FROM labevents GROUP BY hadm_id"
    ),
    "platelet_min": (
        "SELECT hadm_id, MIN(CASE WHEN itemid = 51265 THEN valuenum END) AS v "
        "FROM labevents GROUP BY hadm_id"
    ),
}

_DEFAULT_PERCENTILES: tuple[float, float] = (0.01, 0.99)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def compute_reference_ranges(
    backend: Any,
    features: list[str] | None = None,
    *,
    percentiles: tuple[float, float] = _DEFAULT_PERCENTILES,
    cohort_hadm_ids: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute frozen ``[low, high]`` ranges for each quantitative feature.

    ``backend`` is any object exposing ``execute(sql, params) -> list[tuple]``
    (the production ``_DuckDBBackend`` / ``_BigQueryBackend`` and the test
    wrappers all satisfy this). ``percentiles`` is the robust ``(low, high)``
    pair — defaulting to the 1st/99th — that trims measurement outliers.
    ``cohort_hadm_ids`` restricts the reference population to a fixed admission
    set; ``None`` means the whole population (the default "all admissions"
    frozen reference).

    Returns ``{feature: {"low", "high", "n"}}``. Features with no spread
    (``high <= low``) or no measurements are dropped, since a zero-width range
    is a divide-by-zero trap for the Gower kernel rather than a usable scale.
    """
    p_low, p_high = float(percentiles[0]), float(percentiles[1])
    names = list(features) if features is not None else list(_FEATURE_QUERIES)

    cohort_clause = ""
    if cohort_hadm_ids:
        id_list = ",".join(str(int(h)) for h in cohort_hadm_ids)
        cohort_clause = f" AND hadm_id IN ({id_list})"

    ranges: dict[str, dict[str, float]] = {}
    for name in names:
        inner = _FEATURE_QUERIES.get(name)
        if inner is None:
            logger.warning("no reference-range query registered for %r; skipping", name)
            continue
        # Percentiles are code-controlled floats (not user input), so inlining
        # them is injection-safe and sidesteps DuckDB's constant-arg requirement
        # for quantile_cont's fraction argument.
        sql = (
            f"SELECT quantile_cont(v, {p_low:.6f}) AS low, "
            f"quantile_cont(v, {p_high:.6f}) AS high, "
            f"COUNT(*) AS n "
            f"FROM ({inner}) sub WHERE v IS NOT NULL{cohort_clause}"
        )
        try:
            rows = backend.execute(sql, [])
        except Exception as exc:  # pragma: no cover - backend-specific failures
            logger.warning("reference-range query for %r failed: %s", name, exc)
            continue
        if not rows:
            continue
        low, high, n = rows[0]
        if low is None or high is None:
            logger.debug("feature %r has no measurements; skipping", name)
            continue
        low_f, high_f = float(low), float(high)
        if high_f <= low_f:
            logger.debug(
                "feature %r has zero-width range [%s, %s]; skipping",
                name, low_f, high_f,
            )
            continue
        ranges[name] = {"low": low_f, "high": high_f, "n": int(n)}
    return ranges


def build_artifact(
    backend: Any,
    *,
    cohort: str = "all_admissions",
    features: list[str] | None = None,
    percentiles: tuple[float, float] = _DEFAULT_PERCENTILES,
    cohort_hadm_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Assemble the full frozen-ranges artifact (metadata + ranges)."""
    ranges = compute_reference_ranges(
        backend,
        features=features,
        percentiles=percentiles,
        cohort_hadm_ids=cohort_hadm_ids,
    )
    return {
        "version": "1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cohort": cohort,
        "percentiles": {"low": float(percentiles[0]), "high": float(percentiles[1])},
        "ranges": ranges,
    }


def write_artifact(artifact: dict[str, Any], path: Path | None = None) -> Path:
    """Write the artifact JSON to ``path`` (default ``REFERENCE_RANGES_PATH``)."""
    target = Path(path) if path is not None else REFERENCE_RANGES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    return target


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_ranges_section(
    section: str, path: Path | None = None
) -> dict[str, tuple[float, float]]:
    """Read one ``{key: {low, high}}`` section of the artifact into ``{key: (low, high)}``.

    Never raises: a missing / unreadable / malformed artifact (or a missing
    section) yields ``{}`` so the caller degrades gracefully. Shared by the SQL
    (``ranges``) and graph (``graph_ranges``) loaders.
    """
    target = Path(path) if path is not None else REFERENCE_RANGES_PATH
    if not target.exists():
        logger.debug("reference-ranges artifact not found at %s", target)
        return {}
    try:
        doc = json.loads(target.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("reference-ranges artifact unreadable (%s): %s", target, exc)
        return {}
    if not isinstance(doc, dict):
        return {}
    ranges = doc.get(section)
    if not isinstance(ranges, dict):
        return {}
    out: dict[str, tuple[float, float]] = {}
    for name, rec in ranges.items():
        if not isinstance(rec, dict):
            continue
        low, high = rec.get("low"), rec.get("high")
        if low is None or high is None:
            continue
        try:
            out[name] = (float(low), float(high))
        except (TypeError, ValueError):
            continue
    return out


def load_reference_ranges(path: Path | None = None) -> dict[str, tuple[float, float]]:
    """Load the frozen SQL-feature ranges into ``{feature: (low, high)}``.

    Never raises: a missing / unreadable / malformed artifact yields ``{}`` so
    the caller degrades gracefully (a quantitative trait with no frozen range
    then raises a clear error in the cohort runner, rather than crashing here).
    """
    return _load_ranges_section("ranges", path)


def load_graph_reference_ranges(
    path: Path | None = None,
) -> dict[str, tuple[float, float]]:
    """Load the frozen graph-feature ranges into ``{signature: (low, high)}``.

    Reads the ``graph_ranges`` section (sibling to ``ranges``); keys are
    :func:`~src.similarity.graph_features.graph_feature_signature` strings, so the
    cohort runner can resolve a dynamically-named ``graph_temporal`` trait by its
    scale instead of its (per-query) name. Never raises, like its SQL sibling.
    """
    return _load_ranges_section("graph_ranges", path)


# ---------------------------------------------------------------------------
# Graph-feature builder — fits the frozen scale for graph_temporal columns on a
# FIXED reference cohort, keyed by semantic signature so the LLM-chosen trait
# name is irrelevant. The graph IS the feature extractor, so unlike the SQL
# features there is no static query: we build the per-question RDF graph over the
# reference cohort and read the same columns the cohort runner reads.
# ---------------------------------------------------------------------------


def compute_graph_reference_ranges(
    backend: Any,
    *,
    concepts: list,
    requests: list,
    ontology_dir: Any,
    percentiles: tuple[float, float] = _DEFAULT_PERCENTILES,
    prefilters: list | None = None,
    resolver: Any = None,
    extraction_config: Any = None,
    drug_category_resolver: Any = None,
    max_workers: int = 1,
) -> dict[str, dict[str, float]]:
    """Fit frozen ``[low, high]`` graph-feature ranges on a FIXED reference cohort.

    The graph counterpart to :func:`compute_reference_ranges`: build the
    per-question RDF graph over the reference cohort, pull each request's feature
    column, and take robust ``(p_low, p_high)`` percentiles — but keyed by the
    feature's :func:`~src.similarity.graph_features.request_signature`, not the
    (dynamic) column name, so the frozen range survives the LLM renaming the
    trait. Columns with fewer than two finite reference values, or no spread, are
    dropped (a zero-width range is a divide-by-zero trap for the Gower kernel).

    The reference cohort is whatever ``prefilters`` + ``extraction_config`` select
    — typically no prefilters with an ``extraction_config.max_admissions`` cap, so
    the build is a bounded, fixed sample of the population. This is fit on a fixed
    reference, never on the query batch (locked decision #6).
    """
    import numpy as np

    from src.similarity.graph_features import (
        build_graph_feature_frame,
        request_signature,
    )

    if not requests:
        return {}
    p_low, p_high = float(percentiles[0]) * 100.0, float(percentiles[1]) * 100.0
    frame = build_graph_feature_frame(
        backend,
        prefilters=list(prefilters or []),
        concepts=list(concepts),
        requests=list(requests),
        ontology_dir=ontology_dir,
        resolver=resolver,
        extraction_config=extraction_config,
        drug_category_resolver=drug_category_resolver,
        max_workers=max_workers,
    )
    ranges: dict[str, dict[str, float]] = {}
    for req in requests:
        sig = request_signature(req)
        if sig in ranges or req.column not in frame.columns:
            continue
        values = frame[req.column].to_numpy(dtype="float64")
        values = values[np.isfinite(values)]
        if values.size < 2:
            logger.debug("graph feature %r has <2 reference values; skipping", sig)
            continue
        low = float(np.percentile(values, p_low))
        high = float(np.percentile(values, p_high))
        if high <= low:
            logger.debug("graph feature %r has zero-width range; skipping", sig)
            continue
        ranges[sig] = {"low": low, "high": high, "n": int(values.size)}
    return ranges
