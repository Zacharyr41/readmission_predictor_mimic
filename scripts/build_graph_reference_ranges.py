"""Freeze the Gower normalization ranges for ``graph_temporal`` cohort traits.

The graph counterpart to ``build_similarity_reference_ranges.py``. Where the SQL
script freezes ranges for name-keyed aggregates (age, creatinine_max, ...), a
``graph_temporal`` trait carries a per-query, LLM-chosen *name*
(``lactate_slope_48h``) that never matches a name key. So these ranges are keyed
by *semantic signature* instead — ``template|concept=...|<scale-params>`` — which
the cohort runner resolves regardless of what the model named the trait
(``src/similarity/graph_features.py:graph_feature_signature``).

Output (merged into the SAME artifact as the SQL ranges, never clobbering them):
- data/mappings/similarity_reference_ranges.json
    a sibling ``graph_ranges`` section: ``{signature: {low, high, n}}``, robust
    p1/p99 percentiles computed over a per-question RDF graph built on a BOUNDED,
    FIXED reference cohort (``--max-admissions`` random sample). Fitting on a
    fixed reference — never the query batch — is locked decision #6.

This is a PERFORMANCE optimization, not a correctness gate: a signature absent
from the artifact is fit on-demand at query time over the same fixed reference
cohort (``run_cohort``'s escape hatch) and cached per process. Pre-freezing the
demo signatures just moves that first-query cost offline.

The default request set is the demo's two trend signatures (lactate slope over
48h, vasopressor dose slope), mirroring ``definition_builder``'s worked example
so the frozen signatures match what the model emits at query time. Concepts that
need a drug-category resolver (``vasopressors``) are only resolvable here if one
is wired; unresolved signatures are simply skipped (the escape hatch covers them).

Usage:
    # BigQuery (the demo backend) — bounded reference sample:
    .venv/bin/python scripts/build_graph_reference_ranges.py \
        --backend bigquery --project mimic-485500 --max-admissions 4000

    # Local DuckDB (note: partial labs make biomarker slopes sparse):
    .venv/bin/python scripts/build_graph_reference_ranges.py \
        --backend duckdb --db data/processed/mimiciv.duckdb

Idempotent. Re-running recomputes ``graph_ranges`` and preserves ``ranges``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.conversational.models import ClinicalConcept, ExtractionConfig
from src.similarity.graph_features import GraphFeatureRequest, request_signature
from src.similarity.reference_ranges import compute_graph_reference_ranges

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO / "data" / "processed" / "mimiciv.duckdb"
DEFAULT_OUT = REPO / "data" / "mappings" / "similarity_reference_ranges.json"
DEFAULT_ONTOLOGY_DIR = REPO / "ontology" / "definition"
DEFAULT_PROJECT = "mimic-485500"
# A bounded, fixed reference sample — matches ``run.py:_REFERENCE_COHORT_CAP`` so
# the offline-frozen scale equals the one the on-demand escape hatch would fit.
DEFAULT_MAX_ADMISSIONS = 4000


# The demo's graph_temporal signatures, copied from definition_builder's worked
# example so the frozen signature equals the query-time one. The signature key is
# (template, concept, scale-params) — ``concept_type`` is NOT part of it, it only
# steers which events the graph extracts. ``vasopressors`` is a drug *category*;
# without a drug-category resolver wired here it may not expand to member drugs,
# so its signature can come back SKIPPED — the query-time escape hatch (which has
# the resolver) still fits it. We pass ``drug`` (the closest type ClinicalConcept
# accepts) so the build runs; the lactate biomarker signature builds regardless.
_DEMO_CONCEPTS = [
    ClinicalConcept(name="lactate", concept_type="biomarker"),
    ClinicalConcept(name="vasopressors", concept_type="drug"),
]
_DEMO_REQUESTS = [
    GraphFeatureRequest(
        column="lactate_slope_48h", template="sim_series_by_admission",
        concept="lactate", params={"agg": "slope", "window_hours": 48},
    ),
    GraphFeatureRequest(
        column="vasopressor_dose_slope", template="sim_dose_series",
        concept="vasopressors", params={"agg": "slope"},
    ),
]


def _make_backend(args: argparse.Namespace):
    if args.backend == "bigquery":
        from src.conversational.extractor import _BigQueryBackend

        return _BigQueryBackend(project=args.project)
    from src.conversational.extractor import _DuckDBBackend

    return _DuckDBBackend(args.db)


def _load_or_init_artifact(path: Path) -> dict:
    """The existing artifact (to preserve its SQL ``ranges``), or a fresh shell."""
    if path.exists():
        try:
            doc = json.loads(path.read_text())
            if isinstance(doc, dict):
                return doc
        except (OSError, ValueError):
            pass
    return {"version": "1", "ranges": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("bigquery", "duckdb"),
                        default="bigquery",
                        help="Reference backend (default bigquery, the demo data).")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="BigQuery project (ignored for --backend duckdb).")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="DuckDB path (ignored for --backend bigquery).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Artifact to merge the graph_ranges section into.")
    parser.add_argument("--ontology-dir", type=Path, default=DEFAULT_ONTOLOGY_DIR,
                        help="Ontology definition dir used to build the RDF graph.")
    parser.add_argument("--max-admissions", type=int, default=DEFAULT_MAX_ADMISSIONS,
                        help="Cap the bounded random reference cohort (decision #6).")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="Concurrent extraction batches.")
    parser.add_argument("--p-low", type=float, default=0.01,
                        help="Lower robust percentile (default 0.01).")
    parser.add_argument("--p-high", type=float, default=0.99,
                        help="Upper robust percentile (default 0.99).")
    args = parser.parse_args()

    ref_cfg = ExtractionConfig(
        max_admissions=args.max_admissions, cohort_strategy="random",
        max_concurrent_batches=args.max_workers,
    )

    backend = _make_backend(args)
    try:
        graph_ranges = compute_graph_reference_ranges(
            backend,
            concepts=_DEMO_CONCEPTS,
            requests=_DEMO_REQUESTS,
            ontology_dir=args.ontology_dir,
            percentiles=(args.p_low, args.p_high),
            prefilters=None,  # whole-population sample, capped — never the query batch
            extraction_config=ref_cfg,
            max_workers=args.max_workers,
        )
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    artifact = _load_or_init_artifact(args.out)
    artifact["graph_ranges"] = graph_ranges
    artifact["graph_generated_at"] = datetime.now(timezone.utc).isoformat()
    artifact["graph_percentiles"] = {"low": float(args.p_low), "high": float(args.p_high)}
    artifact["graph_reference_cohort"] = {
        "backend": args.backend,
        "max_admissions": args.max_admissions,
        "strategy": "random",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True))

    # Report which of the requested signatures actually got a usable range, so a
    # silently-skipped one (constant / unresolved concept) is visible.
    print(f"Wrote {len(graph_ranges)} graph-feature range(s) to {args.out}")
    requested = {request_signature(r): r.column for r in _DEMO_REQUESTS}
    for sig, col in sorted(requested.items()):
        rec = graph_ranges.get(sig)
        if rec is None:
            print(f"  SKIPPED  {col:22s} {sig}  (no spread / unresolved concept)")
        else:
            print(f"  {col:22s} [{rec['low']:.4g}, {rec['high']:.4g}]  n={rec['n']}")
            print(f"             {sig}")


if __name__ == "__main__":
    main()
