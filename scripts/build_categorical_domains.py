"""Build the frozen categorical-domain artifact for cohort-by-similarity.

Output:
- data/mappings/similarity_categorical_domains.json
    ``{column: {values, counts, n}}`` plus metadata. ``values`` is the distinct
    non-null literal set of each categorical column (``admission_type``,
    ``gender``), most-common first, derived directly from the data. This is the
    single source of truth for the MIMIC-IV categorical vocabulary: the cohort
    prompt teaches it, and the validate-and-repair guard enforces it, so a stale
    MIMIC-III literal (``EMERGENCY``) can no longer reach a prefilter and match
    nothing.

The computed domains are de-identified aggregate summary statistics (distinct
values + counts), the same class of artifact as ``similarity_reference_ranges``.

DuckDB suffices: ``admission_type``/``gender`` are complete in the local DuckDB
(only *labs* are partially loaded), so the committed artifact is correct.
Regenerating against BigQuery is possible but unnecessary for these columns.

Usage:
    .venv/bin/python scripts/build_categorical_domains.py
    .venv/bin/python scripts/build_categorical_domains.py \
        --db data/processed/mimiciv.duckdb \
        --out data/mappings/similarity_categorical_domains.json

Idempotent. Re-running overwrites the output file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.conversational.extractor import _DuckDBBackend
from src.similarity.categorical_domains import build_artifact, write_artifact

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO / "data" / "processed" / "mimiciv.duckdb"
DEFAULT_OUT = REPO / "data" / "mappings" / "similarity_categorical_domains.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help="Path to the MIMIC DuckDB (read-only).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Path for the frozen categorical-domain JSON artifact.")
    parser.add_argument("--source", default="all_admissions",
                        help="Label recorded in the artifact metadata.")
    args = parser.parse_args()

    backend = _DuckDBBackend(args.db)
    try:
        artifact = build_artifact(backend, source=args.source)
    finally:
        backend.close()

    path = write_artifact(artifact, args.out)
    domains = artifact["domains"]
    print(f"Wrote {len(domains)} categorical domain(s) to {path}")
    for name, rec in sorted(domains.items()):
        preview = ", ".join(
            f"{v} ({rec['counts'][v]})" for v in rec["values"][:6]
        )
        more = "" if len(rec["values"]) <= 6 else f", +{len(rec['values']) - 6} more"
        print(f"  {name:16s} n={rec['n']}: {preview}{more}")


if __name__ == "__main__":
    main()
