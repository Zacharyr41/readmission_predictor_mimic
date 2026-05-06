"""Generate the two offline catalogs the Phase H critic reads from.

Outputs:
- data/ontology_cache/loinc_reference_ranges.json
    LOINC-keyed reference ranges. Each entry: ``{low, high, units}``.
    Source: modal ``ref_range_lower`` / ``ref_range_upper`` /
    ``valueuom`` recorded by clinicians in MIMIC's ``labevents`` for
    each itemid, joined to LOINC via ``data/mappings/labitem_to_snomed.json``.

- data/processed/lab_distributions.json
    Itemid-keyed empirical distributions. Each entry:
    ``{n, mean, p50, p95, units}``. Source: ``valuenum`` over
    ``labevents`` for each itemid, top-N by row count.

The LOINC ranges are MIMIC-clinician-recorded, NOT IFCC / national
reference values. They reflect what the lab reported as "normal" at
the time of the measurement — which is exactly what an external
critic should compare against when reasoning about a MIMIC answer.

Usage:
    .venv/bin/python scripts/build_phase_h_catalogs.py

Idempotent. Re-running overwrites the output files.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parents[1]
DUCKDB_PATH = REPO / "data" / "processed" / "mimiciv.duckdb"
LABITEM_MAP_PATH = REPO / "data" / "mappings" / "labitem_to_snomed.json"

LOINC_OUT = REPO / "data" / "ontology_cache" / "loinc_reference_ranges.json"
DIST_OUT = REPO / "data" / "processed" / "lab_distributions.json"


def _modal(rows: list, key) -> object | None:
    """Return the most-frequent value for ``key`` across ``rows``.
    ``rows`` are duckdb tuples; ``key`` is the column index."""
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def _load_itemid_to_loinc() -> dict[int, str]:
    """Build itemid→LOINC dict from the SNOMED mapping file."""
    raw = json.loads(LABITEM_MAP_PATH.read_text())
    out: dict[int, str] = {}
    for itemid_str, rec in raw.items():
        if itemid_str == "_metadata":
            continue
        if not isinstance(rec, dict):
            continue
        loinc = rec.get("loinc")
        if not loinc:
            continue
        try:
            out[int(itemid_str)] = str(loinc)
        except ValueError:
            continue
    return out


def _select_eligible_itemids(
    con: duckdb.DuckDBPyConnection, *, min_rows: int, top_n: int,
) -> list[tuple[int, int]]:
    """Top ``top_n`` itemids by valuenum-bearing row count, with at
    least ``min_rows`` rows. Returns (itemid, n_rows)."""
    rows = con.execute(f"""
        SELECT itemid, COUNT(*) AS n
        FROM labevents
        WHERE valuenum IS NOT NULL
        GROUP BY itemid
        HAVING n >= {int(min_rows)}
        ORDER BY n DESC
        LIMIT {int(top_n)}
    """).fetchall()
    return [(int(r[0]), int(r[1])) for r in rows]


def _stats_for_itemid(
    con: duckdb.DuckDBPyConnection, itemid: int,
) -> dict[str, object] | None:
    """Compute distribution stats + modal units + modal ref-range for one itemid.
    Returns None if no valuenum data."""
    # One pass: grab a sample of (valuenum, valueuom, ref_lo, ref_hi). For
    # large itemids the FULL pass is fine (millions of rows hash fast).
    rows = con.execute("""
        SELECT valuenum, valueuom, ref_range_lower, ref_range_upper
        FROM labevents
        WHERE itemid = ? AND valuenum IS NOT NULL
    """, [itemid]).fetchall()
    if not rows:
        return None

    nums = sorted(r[0] for r in rows)
    n = len(nums)
    mean = sum(nums) / n
    p50 = nums[n // 2]
    p95_idx = min(n - 1, int(0.95 * n))
    p95 = nums[p95_idx]

    units = _modal(rows, 1)
    ref_lo = _modal(rows, 2)
    ref_hi = _modal(rows, 3)

    return {
        "n": n,
        "mean": round(mean, 4),
        "p50": round(p50, 4),
        "p95": round(p95, 4),
        "units": str(units) if units is not None else "",
        "ref_lo": float(ref_lo) if ref_lo is not None else None,
        "ref_hi": float(ref_hi) if ref_hi is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-rows", type=int, default=1000,
        help="Skip itemids with fewer than N valuenum rows (default 1000)",
    )
    parser.add_argument(
        "--top-n", type=int, default=300,
        help="Compute distributions for top-N itemids by row count (default 300)",
    )
    args = parser.parse_args()

    if not DUCKDB_PATH.exists():
        print(f"ERROR: {DUCKDB_PATH} does not exist")
        return 2
    if not LABITEM_MAP_PATH.exists():
        print(f"ERROR: {LABITEM_MAP_PATH} does not exist")
        return 2

    LOINC_OUT.parent.mkdir(parents=True, exist_ok=True)
    DIST_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {DUCKDB_PATH}...")
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    itemid_to_loinc = _load_itemid_to_loinc()
    print(f"Loaded {len(itemid_to_loinc)} itemid→LOINC mappings")

    eligible = _select_eligible_itemids(
        con, min_rows=args.min_rows, top_n=args.top_n,
    )
    print(f"Selected {len(eligible)} itemids with ≥{args.min_rows} valuenum rows")

    distributions: dict[str, dict] = {}
    loinc_ranges: dict[str, dict] = {}

    for i, (itemid, n_rows) in enumerate(eligible, 1):
        stats = _stats_for_itemid(con, itemid)
        if stats is None:
            continue
        if i % 25 == 0:
            print(f"  [{i}/{len(eligible)}] itemid {itemid} (n={stats['n']:,})")

        distributions[str(itemid)] = {
            "n": stats["n"],
            "mean": stats["mean"],
            "p50": stats["p50"],
            "p95": stats["p95"],
            "units": stats["units"],
        }

        loinc = itemid_to_loinc.get(itemid)
        if loinc and stats["ref_lo"] is not None and stats["ref_hi"] is not None:
            # If multiple itemids map to the same LOINC, the last one wins —
            # but they should agree by construction. Keep the entry with
            # the largest n_rows when conflicts exist.
            existing = loinc_ranges.get(loinc)
            if existing is None or stats["n"] > existing.get("_n", 0):
                loinc_ranges[loinc] = {
                    "low": stats["ref_lo"],
                    "high": stats["ref_hi"],
                    "units": stats["units"],
                    "_n": stats["n"],  # tie-breaker; stripped below
                }

    # Strip the _n tie-breaker from final output.
    for entry in loinc_ranges.values():
        entry.pop("_n", None)

    print(
        f"\nWriting {len(distributions)} distributions to {DIST_OUT}"
    )
    DIST_OUT.write_text(
        json.dumps(distributions, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"Writing {len(loinc_ranges)} LOINC ranges to {LOINC_OUT}"
    )
    LOINC_OUT.write_text(
        json.dumps(loinc_ranges, indent=2, sort_keys=True) + "\n",
    )

    print()
    print(f"Distribution file size: {DIST_OUT.stat().st_size:,} bytes")
    print(f"LOINC catalog size:     {LOINC_OUT.stat().st_size:,} bytes")
    print(
        f"\nDistributions cover {len(distributions)} itemids "
        f"out of {len(eligible)} eligible."
    )
    print(
        f"LOINC ranges cover {len(loinc_ranges)} codes out of "
        f"{sum(1 for x in eligible if itemid_to_loinc.get(x[0]))} "
        f"itemids with a LOINC mapping."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
