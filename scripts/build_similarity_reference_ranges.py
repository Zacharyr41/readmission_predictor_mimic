"""Build the frozen Gower normalization ranges for cohort-by-similarity.

Output:
- data/mappings/similarity_reference_ranges.json
    ``{feature: {low, high, n}}`` plus metadata. ``low``/``high`` are robust
    p1/p99 percentiles of each quantitative cohort feature, computed over the
    SAME per-admission aggregates the cohort runner pulls
    (``src/similarity/run.py:_fetch_admission_features``). These freeze the
    normalization scale of the Gower kernel (locked decision #6): a one-vs-many
    cohort query scores against a 1-row profile, so the range must come from a
    fixed reference population, never the query batch.

The computed ranges are de-identified aggregate summary statistics (population
percentiles), the same class of artifact as ``data/processed/lab_distributions``.

Usage:
    .venv/bin/python scripts/build_similarity_reference_ranges.py
    .venv/bin/python scripts/build_similarity_reference_ranges.py \
        --db data/processed/mimiciv.duckdb \
        --out data/mappings/similarity_reference_ranges.json

Idempotent. Re-running overwrites the output file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.conversational.extractor import _DuckDBBackend
from src.similarity.reference_ranges import build_artifact, write_artifact

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO / "data" / "processed" / "mimiciv.duckdb"
DEFAULT_OUT = REPO / "data" / "mappings" / "similarity_reference_ranges.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="Path to the MIMIC DuckDB (read-only).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Path for the frozen-ranges JSON artifact.")
    parser.add_argument("--cohort", default="all_admissions",
                        help="Label recorded in the artifact metadata.")
    parser.add_argument("--p-low", type=float, default=0.01,
                        help="Lower robust percentile (default 0.01).")
    parser.add_argument("--p-high", type=float, default=0.99,
                        help="Upper robust percentile (default 0.99).")
    args = parser.parse_args()

    backend = _DuckDBBackend(args.db)
    try:
        artifact = build_artifact(
            backend,
            cohort=args.cohort,
            percentiles=(args.p_low, args.p_high),
        )
    finally:
        backend.close()

    path = write_artifact(artifact, args.out)
    ranges = artifact["ranges"]
    print(f"Wrote {len(ranges)} frozen feature range(s) to {path}")
    for name, rec in sorted(ranges.items()):
        print(f"  {name:18s} [{rec['low']:.4g}, {rec['high']:.4g}]  n={rec['n']}")


if __name__ == "__main__":
    main()
