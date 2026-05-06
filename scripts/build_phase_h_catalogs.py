"""Generate the two offline catalogs the Phase H critic reads from.

Outputs:
- data/ontology_cache/loinc_reference_ranges.json
    LOINC-keyed reference ranges. Each entry: ``{low, high, units}``.
    Source: modal ``ref_range_lower`` / ``ref_range_upper`` /
    ``valueuom`` recorded by clinicians in MIMIC's ``labevents`` for
    each itemid, joined to LOINC via ``data/mappings/labitem_to_snomed.json``.
    Chartevents has no ref_range columns, so the LOINC catalog stays
    labevents-only by construction.

- data/processed/lab_distributions.json
    Itemid-keyed empirical distributions. Each entry:
    ``{n, mean, p50, p95, units}``. Source: ``valuenum`` over both
    ``labevents`` AND ``chartevents`` (Tier B+ — covers vitals like
    HR, SpO₂, BP, RR, GCS in addition to labs). Top-N per table by
    row count. Inc 7 will switch this from a flat per-itemid record
    to a nested ``{itemid: {cohort: stats}}`` schema.

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

# Cohort-stratified bucket emission threshold. Cohort × itemid pairs
# with fewer than this many rows are omitted from the catalog (the
# stats would be too noisy to be useful). Same value the on-the-fly
# compute path uses, so cache hits and compute hits are comparable.
_MIN_COHORT_N = 30
# Reserved cohort name for the unstratified bucket.
_ALL_COHORT_KEY = "all"


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
    con: duckdb.DuckDBPyConnection, *,
    min_rows: int, top_n: int, table: str,
) -> list[tuple[int, int]]:
    """Top ``top_n`` itemids in ``table`` by valuenum-bearing row count,
    with at least ``min_rows`` rows. Returns (itemid, n_rows)."""
    if table not in ("labevents", "chartevents"):
        raise ValueError(f"unsupported table {table!r}")
    rows = con.execute(f"""
        SELECT itemid, COUNT(*) AS n
        FROM {table}
        WHERE valuenum IS NOT NULL
        GROUP BY itemid
        HAVING n >= {int(min_rows)}
        ORDER BY n DESC
        LIMIT {int(top_n)}
    """).fetchall()
    return [(int(r[0]), int(r[1])) for r in rows]


def _stats_for_itemid(
    con: duckdb.DuckDBPyConnection, itemid: int,
    *, table: str, has_ref_range: bool,
) -> dict[str, object] | None:
    """Compute distribution stats + modal units (+ modal ref-range when
    the table has ref_range columns — labevents has them, chartevents
    doesn't). Returns None if no valuenum data."""
    if table not in ("labevents", "chartevents"):
        raise ValueError(f"unsupported table {table!r}")

    if has_ref_range:
        rows = con.execute(f"""
            SELECT valuenum, valueuom, ref_range_lower, ref_range_upper
            FROM {table}
            WHERE itemid = ? AND valuenum IS NOT NULL
        """, [itemid]).fetchall()
    else:
        # No ref_range columns — pad NULLs to keep the column-index
        # scheme stable for ``_modal``.
        rows = con.execute(f"""
            SELECT valuenum, valueuom, CAST(NULL AS DOUBLE), CAST(NULL AS DOUBLE)
            FROM {table}
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
    ref_lo = _modal(rows, 2) if has_ref_range else None
    ref_hi = _modal(rows, 3) if has_ref_range else None

    return {
        "n": n,
        "mean": round(mean, 4),
        "p50": round(p50, 4),
        "p95": round(p95, 4),
        "units": str(units) if units is not None else "",
        "ref_lo": float(ref_lo) if ref_lo is not None else None,
        "ref_hi": float(ref_hi) if ref_hi is not None else None,
    }


# Tables we scan for distributions. Each entry: (table_name, has_ref_range).
# labevents has ref_range_lower / ref_range_upper (used to populate the
# LOINC catalog); chartevents does not, so its entries don't contribute
# to the LOINC catalog but they DO populate the distribution file.
_DIST_TABLES: list[tuple[str, bool]] = [
    ("labevents", True),
    ("chartevents", False),
]


def _build_hadm_cohorts_table(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Materialise a temporary ``hadm_cohorts(hadm_id, cohort)`` table —
    one row per (hadm, cohort) where the hadm carries a diagnosis
    matching that cohort's ICD prefixes. Returns ``{cohort: hadm_count}``
    for logging.

    Done ONCE per generator run (one scan of diagnoses_icd × cohort
    count). The downstream per-itemid stats query joins to this table.
    """
    # Insert REPO into sys.path so the cohorts module imports cleanly
    # when the generator is invoked from a different cwd.
    import sys
    sys.path.insert(0, str(REPO))
    from src.conversational.health_evidence.cohorts import (
        cohort_filter_sql, load_cohorts,
    )

    con.execute("DROP TABLE IF EXISTS hadm_cohorts")
    con.execute(
        "CREATE TEMPORARY TABLE hadm_cohorts ("
        "  hadm_id INTEGER, cohort VARCHAR"
        ")"
    )
    counts: dict[str, int] = {}
    for cohort_name in load_cohorts():
        try:
            subquery = cohort_filter_sql(cohort_name)
            cohort_lit = cohort_name.replace("'", "''")
            con.execute(
                f"INSERT INTO hadm_cohorts "
                f"SELECT hadm_id, '{cohort_lit}' FROM ({subquery})"
            )
            n = con.execute(
                "SELECT COUNT(*) FROM hadm_cohorts WHERE cohort = ?",
                [cohort_name],
            ).fetchone()[0]
            counts[cohort_name] = int(n)
        except Exception as exc:  # noqa: BLE001 — never crash the run
            print(
                f"  WARN: cohort {cohort_name!r} skipped during "
                f"materialisation: {exc}"
            )
    return counts


def _stratified_stats_for_itemid(
    con: duckdb.DuckDBPyConnection,
    itemid: int,
    *,
    table: str,
    has_ref_range: bool,
    min_cohort_n: int = _MIN_COHORT_N,
) -> dict[str, dict] | None:
    """Compute per-cohort stats for one itemid in one table. Returns
    ``{cohort: {n, mean, p50, p95, units}}`` with the ``"all"`` bucket
    always present (computed unstratified) plus any cohort with at
    least ``min_cohort_n`` rows for this itemid. Returns None when
    the all-bucket has no data.

    Single duckdb scan per itemid: pulls (valuenum, valueuom, cohort)
    where rows joined to ``hadm_cohorts`` carry their cohort label,
    and rows that don't match any cohort just don't appear in the
    cohort buckets. The 'all' bucket is computed by aggregating the
    raw table without the cohort join.
    """
    # 'all' bucket: unstratified
    all_stats = _stats_for_itemid(
        con, itemid, table=table, has_ref_range=has_ref_range,
    )
    if all_stats is None:
        return None
    out: dict[str, dict] = {
        _ALL_COHORT_KEY: {
            "n": all_stats["n"],
            "mean": all_stats["mean"],
            "p50": all_stats["p50"],
            "p95": all_stats["p95"],
            "units": all_stats["units"],
        }
    }

    # Per-cohort: pull rows joined to hadm_cohorts. A single hadm_id can
    # appear in multiple cohorts (e.g. sepsis + heart_failure), so each
    # row potentially feeds multiple buckets.
    rows = con.execute(
        f"""
        SELECT t.valuenum, t.valueuom, hc.cohort
        FROM {table} t
        JOIN hadm_cohorts hc ON t.hadm_id = hc.hadm_id
        WHERE t.itemid = ? AND t.valuenum IS NOT NULL
        """,
        [itemid],
    ).fetchall()

    by_cohort: dict[str, list] = {}
    for valuenum, valueuom, cohort in rows:
        by_cohort.setdefault(cohort, []).append((valuenum, valueuom))

    for cohort, samples in by_cohort.items():
        if len(samples) < min_cohort_n:
            continue
        nums = sorted(s[0] for s in samples if s[0] is not None)
        if not nums:
            continue
        n = len(nums)
        mean = sum(nums) / n
        p50 = nums[n // 2]
        p95 = nums[min(n - 1, int(0.95 * n))]
        units_counter = Counter(s[1] for s in samples if s[1] is not None)
        units = units_counter.most_common(1)[0][0] if units_counter else ""
        out[cohort] = {
            "n": n,
            "mean": round(mean, 4),
            "p50": round(p50, 4),
            "p95": round(p95, 4),
            "units": str(units),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-rows", type=int, default=100,
        help=(
            "Skip itemids with fewer than N valuenum rows (default 100). "
            "Tier D dropped this from 1000 to 100 to extend the long tail."
        ),
    )
    parser.add_argument(
        "--top-n", type=int, default=2000,
        help=(
            "Compute distributions for top-N itemids per table by row "
            "count (default 2000 — effectively no cap). Per-table so "
            "labevents and chartevents both get coverage even though "
            "chartevents has 4× more rows."
        ),
    )
    parser.add_argument(
        "--stratify-by-cohort", action="store_true",
        help=(
            "Emit cohort-stratified buckets per itemid (Tier D nested "
            "schema). Output becomes "
            "{itemid: {cohort_name: {n, mean, p50, p95, units}}}. "
            "When omitted, the output is the flat schema (Tier B)."
        ),
    )
    parser.add_argument(
        "--min-cohort-n", type=int, default=_MIN_COHORT_N,
        help=(
            f"When --stratify-by-cohort is on, omit cohort buckets with "
            f"fewer than N rows for the itemid (default {_MIN_COHORT_N})."
        ),
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

    distributions: dict[str, dict] = {}
    loinc_ranges: dict[str, dict] = {}
    total_eligible = 0
    total_loinc_eligible = 0

    # When stratifying, materialise the hadm→cohort mapping ONCE so
    # the per-itemid query is a single join, not one query per cohort.
    if args.stratify_by_cohort:
        print("\nMaterialising hadm_cohorts table for stratification...")
        cohort_counts = _build_hadm_cohorts_table(con)
        for cohort_name, n in sorted(cohort_counts.items()):
            print(f"  {cohort_name}: {n:,} hadm_ids")

    for table, has_ref_range in _DIST_TABLES:
        eligible = _select_eligible_itemids(
            con, min_rows=args.min_rows, top_n=args.top_n, table=table,
        )
        total_eligible += len(eligible)
        print(
            f"\n[{table}] Selected {len(eligible)} itemids with "
            f"≥{args.min_rows} valuenum rows"
        )

        for i, (itemid, _n_rows) in enumerate(eligible, 1):
            stats = _stats_for_itemid(
                con, itemid,
                table=table, has_ref_range=has_ref_range,
            )
            if stats is None:
                continue
            if i % 50 == 0:
                print(
                    f"  [{i}/{len(eligible)}] {table} itemid {itemid} "
                    f"(n={stats['n']:,})"
                )

            # Build the distribution entry — flat or nested depending
            # on the --stratify-by-cohort flag.
            if args.stratify_by_cohort:
                stratified = _stratified_stats_for_itemid(
                    con, itemid,
                    table=table, has_ref_range=has_ref_range,
                    min_cohort_n=args.min_cohort_n,
                )
                if stratified is None:
                    continue
                # When the same itemid exists in both tables, keep the
                # one with the larger 'all' n.
                existing = distributions.get(str(itemid))
                if existing is None or int(stratified[_ALL_COHORT_KEY]["n"]) > int(
                    (existing.get(_ALL_COHORT_KEY) or {}).get("n", 0)
                ):
                    distributions[str(itemid)] = stratified
            else:
                existing = distributions.get(str(itemid))
                if existing is None or int(stats["n"]) > int(
                    existing.get("n", 0)
                ):
                    distributions[str(itemid)] = {
                        "n": stats["n"],
                        "mean": stats["mean"],
                        "p50": stats["p50"],
                        "p95": stats["p95"],
                        "units": stats["units"],
                    }

            # LOINC catalog entry — only when this itemid has both a
            # mapped LOINC code AND a ref_range from labevents (chart
            # itemids don't carry ref_range).
            if not has_ref_range:
                continue
            loinc = itemid_to_loinc.get(itemid)
            if loinc is None:
                continue
            total_loinc_eligible += 1
            if stats["ref_lo"] is None or stats["ref_hi"] is None:
                continue
            existing_loinc = loinc_ranges.get(loinc)
            if existing_loinc is None or stats["n"] > existing_loinc.get("_n", 0):
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
        f"\nDistributions cover {len(distributions)} unique itemids "
        f"out of {total_eligible} eligible (across both tables)."
    )
    print(
        f"LOINC ranges cover {len(loinc_ranges)} codes out of "
        f"{total_loinc_eligible} labevents itemids with a LOINC mapping."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
