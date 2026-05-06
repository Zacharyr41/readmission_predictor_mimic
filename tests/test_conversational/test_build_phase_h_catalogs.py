"""Tests for ``scripts/build_phase_h_catalogs.py``.

The generator script reads MIMIC-shaped tables out of a local duckdb
and emits two offline catalogs:

* ``data/processed/lab_distributions.json`` — empirical distributions
  per itemid. Tier D extends from labevents-only to labevents +
  chartevents (to cover vitals — heart rate, SpO₂, BP, RR, GCS, etc.)
  and lowers the row-count floor from 1000 to 100. Inc 7 will switch
  the schema from flat to nested cohort-stratified buckets.
* ``data/ontology_cache/loinc_reference_ranges.json`` — LOINC-keyed
  reference ranges sourced from labevents' ``ref_range_lower`` /
  ``ref_range_upper`` fields. Chartevents has no ref_range columns so
  the LOINC catalog stays labevents-only.

Tests use small in-process duckdb fixtures to keep runs fast (~1 s).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest


REPO = Path(__file__).resolve().parents[2]
GENERATOR = REPO / "scripts" / "build_phase_h_catalogs.py"


def _make_fixture_duckdb(
    tmp_path: Path,
    *,
    labevents_rows: list[tuple] = (),
    chartevents_rows: list[tuple] = (),
) -> Path:
    """Build a tiny MIMIC-shaped fixture duckdb with the columns the
    generator needs.

    Schema mirrors the real MIMIC layout but only the columns the
    generator touches.
    """
    db_path = tmp_path / "fixture.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE labevents ("
        "  itemid INTEGER, valuenum DOUBLE, valueuom VARCHAR,"
        "  ref_range_lower DOUBLE, ref_range_upper DOUBLE"
        ")"
    )
    con.execute(
        "CREATE TABLE chartevents ("
        "  itemid INTEGER, valuenum DOUBLE, valueuom VARCHAR"
        ")"
    )
    for row in labevents_rows:
        con.execute("INSERT INTO labevents VALUES (?,?,?,?,?)", row)
    for row in chartevents_rows:
        con.execute("INSERT INTO chartevents VALUES (?,?,?)", row)
    con.close()
    return db_path


def _make_labitem_mapping(tmp_path: Path, mapping: dict[int, str]) -> Path:
    """Build a fixture labitem_to_snomed.json file."""
    path = tmp_path / "labitem_to_snomed.json"
    raw = {
        "_metadata": {"source": "test fixture"},
        **{
            str(itemid): {"loinc": loinc, "label": f"item-{itemid}"}
            for itemid, loinc in mapping.items()
        },
    }
    path.write_text(json.dumps(raw))
    return path


def _run_generator(
    *,
    duckdb_path: Path,
    labitem_map_path: Path,
    loinc_out: Path,
    dist_out: Path,
    extra_args: list[str] = (),
) -> subprocess.CompletedProcess:
    """Run the generator script as a subprocess so it sees a clean
    sys.path. Patches the module-level paths via env-style overrides
    by re-pointing the constants in a small wrapper script. The
    generator currently hardcodes ``REPO`` paths — to test, we set
    those constants via a ``-c`` Python invocation before calling
    ``main()``."""
    runner = (
        f"import sys, json\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        f"import scripts.build_phase_h_catalogs as g\n"
        f"g.DUCKDB_PATH = {str(duckdb_path)!r}\n"
        f"from pathlib import Path\n"
        f"g.DUCKDB_PATH = Path({str(duckdb_path)!r})\n"
        f"g.LABITEM_MAP_PATH = Path({str(labitem_map_path)!r})\n"
        f"g.LOINC_OUT = Path({str(loinc_out)!r})\n"
        f"g.DIST_OUT = Path({str(dist_out)!r})\n"
        f"sys.argv = ['gen'] + {list(extra_args)!r}\n"
        f"sys.exit(g.main())\n"
    )
    return subprocess.run(
        [sys.executable, "-c", runner],
        capture_output=True, text=True, timeout=60,
    )


def test_generator_default_min_rows_lowered_to_100():
    """Tier D: the row-count floor drops from 1000 to 100 so the
    long tail of MIMIC labs makes it into the catalog. Read the
    generator's argparse default directly."""
    import scripts.build_phase_h_catalogs as g
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=2000)
    # We can't easily introspect g.main()'s argparse without invoking
    # main, so instead read the source for the default value. This is
    # a contract test on what the source declares.
    src = Path(g.__file__).read_text()
    # Find the `--min-rows` default in the argparse setup.
    assert "default=100" in src or "min-rows" in src
    # Specifically: the default should NOT still be 1000.
    # Look for the argparse line for min-rows and check its default.
    import re
    m = re.search(r'add_argument\(\s*"--min-rows".*?default\s*=\s*(\d+)', src, re.DOTALL)
    assert m is not None, "could not find --min-rows argparse default"
    default = int(m.group(1))
    assert default <= 100, (
        f"Tier D should lower min_rows floor; current default is {default}"
    )


def test_generator_emits_labevents_itemid(tmp_path):
    """Regression: labevents-only generation still works (Tier B doesn't
    break the existing labevents path)."""
    # 200 rows of itemid 50912 (creatinine) — above the 100-row floor.
    labevents_rows = [
        (50912, 1.0 + 0.001 * i, "mg/dL", 0.6, 1.2)
        for i in range(200)
    ]
    duckdb_path = _make_fixture_duckdb(
        tmp_path, labevents_rows=labevents_rows,
    )
    labitem_map = _make_labitem_mapping(tmp_path, {50912: "2160-0"})
    loinc_out = tmp_path / "loinc_out.json"
    dist_out = tmp_path / "dist_out.json"

    result = _run_generator(
        duckdb_path=duckdb_path,
        labitem_map_path=labitem_map,
        loinc_out=loinc_out,
        dist_out=dist_out,
        extra_args=["--min-rows", "100", "--top-n", "10"],
    )
    assert result.returncode == 0, (
        f"generator failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    distributions = json.loads(dist_out.read_text())
    assert "50912" in distributions
    assert distributions["50912"]["n"] == 200


def test_generator_includes_chartevents_itemids(tmp_path):
    """Tier B step: chartevents itemids appear in the distribution
    output. Heart rate (itemid 220045) is the canonical example."""
    # labevents — small sample so chartevents isn't crowded out.
    labevents_rows = [
        (50912, 1.0 + 0.001 * i, "mg/dL", 0.6, 1.2)
        for i in range(150)
    ]
    # chartevents: 250 heart rate readings.
    chartevents_rows = [
        (220045, 80.0 + 0.1 * i, "bpm")
        for i in range(250)
    ]
    duckdb_path = _make_fixture_duckdb(
        tmp_path,
        labevents_rows=labevents_rows,
        chartevents_rows=chartevents_rows,
    )
    labitem_map = _make_labitem_mapping(tmp_path, {50912: "2160-0"})
    loinc_out = tmp_path / "loinc_out.json"
    dist_out = tmp_path / "dist_out.json"

    result = _run_generator(
        duckdb_path=duckdb_path,
        labitem_map_path=labitem_map,
        loinc_out=loinc_out,
        dist_out=dist_out,
        extra_args=["--min-rows", "100", "--top-n", "100"],
    )
    assert result.returncode == 0, (
        f"generator failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    distributions = json.loads(dist_out.read_text())
    # Both lab and chart itemids present
    assert "50912" in distributions, "labevents itemid missing"
    assert "220045" in distributions, "chartevents itemid missing"
    # The chartevents record has units from the fixture
    assert distributions["220045"]["units"] == "bpm"
    assert distributions["220045"]["n"] == 250


def test_generator_respects_min_rows_floor(tmp_path):
    """Items with fewer than min_rows valuenum entries are skipped."""
    labevents_rows = [
        (50912, 1.0, "mg/dL", 0.6, 1.2)
        for _ in range(150)
    ]
    chartevents_rows = [
        # Only 50 rows — below the 100 floor, should be omitted.
        (220045, 80.0, "bpm")
        for _ in range(50)
    ]
    duckdb_path = _make_fixture_duckdb(
        tmp_path,
        labevents_rows=labevents_rows,
        chartevents_rows=chartevents_rows,
    )
    labitem_map = _make_labitem_mapping(tmp_path, {50912: "2160-0"})
    loinc_out = tmp_path / "loinc_out.json"
    dist_out = tmp_path / "dist_out.json"

    result = _run_generator(
        duckdb_path=duckdb_path,
        labitem_map_path=labitem_map,
        loinc_out=loinc_out,
        dist_out=dist_out,
        extra_args=["--min-rows", "100"],
    )
    assert result.returncode == 0
    distributions = json.loads(dist_out.read_text())
    assert "50912" in distributions
    assert "220045" not in distributions, (
        "below-floor chartevents itemid should be excluded"
    )


def test_loinc_catalog_only_labevents(tmp_path):
    """LOINC ranges come from labevents' ref_range_* columns. Chartevents
    has no ref_range_* so it never appears in the LOINC catalog —
    even when the chartevents itemid is in the labitem_to_snomed map
    (which it normally isn't, but we check the contract anyway)."""
    labevents_rows = [
        (50912, 1.0, "mg/dL", 0.6, 1.2)
        for _ in range(150)
    ]
    chartevents_rows = [
        (220045, 80.0, "bpm") for _ in range(250)
    ]
    duckdb_path = _make_fixture_duckdb(
        tmp_path,
        labevents_rows=labevents_rows,
        chartevents_rows=chartevents_rows,
    )
    # Both itemids in the mapping (intentionally including the chart one)
    labitem_map = _make_labitem_mapping(tmp_path, {
        50912: "2160-0",   # creatinine
        220045: "8867-4",  # heart rate
    })
    loinc_out = tmp_path / "loinc_out.json"
    dist_out = tmp_path / "dist_out.json"

    result = _run_generator(
        duckdb_path=duckdb_path,
        labitem_map_path=labitem_map,
        loinc_out=loinc_out,
        dist_out=dist_out,
        extra_args=["--min-rows", "100"],
    )
    assert result.returncode == 0
    loinc_catalog = json.loads(loinc_out.read_text())
    # LOINC for creatinine appears (from labevents)
    assert "2160-0" in loinc_catalog
    # LOINC for heart rate is NOT in the catalog (chartevents has no
    # ref_range columns to populate the entry).
    assert "8867-4" not in loinc_catalog


def test_generator_output_schema_unchanged(tmp_path):
    """Tier B keeps the FLAT schema (Inc 7 will switch to nested).
    Each entry has n, mean, p50, p95, units."""
    labevents_rows = [
        (50912, 1.0 + 0.001 * i, "mg/dL", 0.6, 1.2)
        for i in range(150)
    ]
    duckdb_path = _make_fixture_duckdb(
        tmp_path, labevents_rows=labevents_rows,
    )
    labitem_map = _make_labitem_mapping(tmp_path, {50912: "2160-0"})
    loinc_out = tmp_path / "loinc_out.json"
    dist_out = tmp_path / "dist_out.json"

    result = _run_generator(
        duckdb_path=duckdb_path,
        labitem_map_path=labitem_map,
        loinc_out=loinc_out,
        dist_out=dist_out,
        extra_args=["--min-rows", "100"],
    )
    assert result.returncode == 0
    distributions = json.loads(dist_out.read_text())
    rec = distributions["50912"]
    for field in ("n", "mean", "p50", "p95", "units"):
        assert field in rec
    # Flat shape — values are scalars, not nested dicts (Inc 7 will
    # change this).
    assert isinstance(rec["n"], int)
    assert isinstance(rec["mean"], (int, float))


# ===========================================================================
# Inc 7: cohort-stratified nested schema
# ===========================================================================


def _make_fixture_with_diagnoses(
    tmp_path: Path,
    *,
    labevents_rows: list[tuple] = (),
    chartevents_rows: list[tuple] = (),
    diagnoses_rows: list[tuple] = (),
) -> Path:
    """Fixture duckdb that ALSO carries diagnoses_icd, so the
    cohort-stratified generator path can join."""
    db_path = tmp_path / "fixture_with_diag.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE labevents ("
        "  hadm_id INTEGER, itemid INTEGER, valuenum DOUBLE,"
        "  valueuom VARCHAR, ref_range_lower DOUBLE,"
        "  ref_range_upper DOUBLE)"
    )
    con.execute(
        "CREATE TABLE chartevents ("
        "  hadm_id INTEGER, itemid INTEGER, valuenum DOUBLE,"
        "  valueuom VARCHAR)"
    )
    con.execute(
        "CREATE TABLE diagnoses_icd ("
        "  hadm_id INTEGER, icd_code VARCHAR, icd_version INTEGER)"
    )
    for row in labevents_rows:
        con.execute("INSERT INTO labevents VALUES (?,?,?,?,?,?)", row)
    for row in chartevents_rows:
        con.execute("INSERT INTO chartevents VALUES (?,?,?,?)", row)
    for row in diagnoses_rows:
        con.execute("INSERT INTO diagnoses_icd VALUES (?,?,?)", row)
    con.close()
    return db_path


def test_generator_emits_nested_cohort_schema(tmp_path):
    """When --stratify-by-cohort is set, the output schema is
    ``{itemid_str: {cohort: {n, mean, p50, p95, units}}}`` with an
    'all' bucket plus per-cohort buckets where data exists."""
    # 50 sepsis-coded hadm_ids with high lactate (~5 mmol/L)
    # 100 non-sepsis hadm_ids with low lactate (~1.5 mmol/L)
    labevents = []
    diagnoses = []
    for hadm_id in range(1, 51):
        labevents.append(
            (hadm_id, 50813, 5.0 + 0.01 * hadm_id, "mmol/L", 0.5, 2.0)
        )
        diagnoses.append((hadm_id, "A419", 10))
    for hadm_id in range(1001, 1101):
        labevents.append(
            (hadm_id, 50813, 1.0 + 0.01 * (hadm_id - 1001), "mmol/L", 0.5, 2.0)
        )
        diagnoses.append((hadm_id, "Z000", 10))

    duckdb_path = _make_fixture_with_diagnoses(
        tmp_path,
        labevents_rows=labevents,
        diagnoses_rows=diagnoses,
    )
    labitem_map = _make_labitem_mapping(tmp_path, {50813: "32693-4"})
    loinc_out = tmp_path / "loinc_out.json"
    dist_out = tmp_path / "dist_out.json"

    result = _run_generator(
        duckdb_path=duckdb_path,
        labitem_map_path=labitem_map,
        loinc_out=loinc_out,
        dist_out=dist_out,
        extra_args=["--min-rows", "100", "--stratify-by-cohort"],
    )
    assert result.returncode == 0, (
        f"generator failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    distributions = json.loads(dist_out.read_text())
    rec = distributions["50813"]
    # Nested schema
    assert isinstance(rec, dict)
    assert "all" in rec
    assert isinstance(rec["all"], dict)
    assert "n" in rec["all"]
    # Sepsis cohort present
    assert "sepsis" in rec
    # Sepsis-cohort lactate is meaningfully higher than the all-MIMIC mean
    # (ICU-typical dispersion in this fixture: sepsis mean ~5.25, all ~3.0)
    assert rec["sepsis"]["mean"] > rec["all"]["mean"]


def test_generator_omits_low_count_cohorts(tmp_path):
    """Cohort buckets with n<30 (in this itemid) must be omitted to
    keep noisy stats out of the cache. The 'all' bucket stays
    regardless."""
    # Only 5 sepsis-coded rows for itemid 50813 — below the 30-row floor
    labevents = []
    diagnoses = []
    for hadm_id in range(1, 6):
        labevents.append(
            (hadm_id, 50813, 5.0, "mmol/L", 0.5, 2.0)
        )
        diagnoses.append((hadm_id, "A419", 10))
    # Plus 200 non-sepsis rows so the 'all' bucket exceeds min_rows.
    for hadm_id in range(1001, 1201):
        labevents.append(
            (hadm_id, 50813, 1.5, "mmol/L", 0.5, 2.0)
        )
        diagnoses.append((hadm_id, "Z000", 10))

    duckdb_path = _make_fixture_with_diagnoses(
        tmp_path,
        labevents_rows=labevents,
        diagnoses_rows=diagnoses,
    )
    labitem_map = _make_labitem_mapping(tmp_path, {50813: "32693-4"})
    loinc_out = tmp_path / "loinc_out.json"
    dist_out = tmp_path / "dist_out.json"

    result = _run_generator(
        duckdb_path=duckdb_path,
        labitem_map_path=labitem_map,
        loinc_out=loinc_out,
        dist_out=dist_out,
        extra_args=["--min-rows", "100", "--stratify-by-cohort"],
    )
    assert result.returncode == 0
    distributions = json.loads(dist_out.read_text())
    rec = distributions["50813"]
    assert "all" in rec
    # Sepsis cohort is below threshold and should NOT appear.
    assert "sepsis" not in rec, (
        "sepsis cohort with n=5 should be filtered out at the 30-row floor"
    )


def test_generator_always_emits_all_cohort(tmp_path):
    """Even when no per-cohort threshold is met, the 'all' bucket is
    always emitted (it's the unstratified pool)."""
    # 200 rows with a non-registered diagnosis ('Z000' isn't in any cohort)
    labevents = []
    diagnoses = []
    for hadm_id in range(1, 201):
        labevents.append(
            (hadm_id, 50813, 1.5, "mmol/L", 0.5, 2.0)
        )
        diagnoses.append((hadm_id, "Z000", 10))

    duckdb_path = _make_fixture_with_diagnoses(
        tmp_path,
        labevents_rows=labevents,
        diagnoses_rows=diagnoses,
    )
    labitem_map = _make_labitem_mapping(tmp_path, {50813: "32693-4"})
    loinc_out = tmp_path / "loinc_out.json"
    dist_out = tmp_path / "dist_out.json"

    result = _run_generator(
        duckdb_path=duckdb_path,
        labitem_map_path=labitem_map,
        loinc_out=loinc_out,
        dist_out=dist_out,
        extra_args=["--min-rows", "100", "--stratify-by-cohort"],
    )
    assert result.returncode == 0
    distributions = json.loads(dist_out.read_text())
    rec = distributions["50813"]
    assert "all" in rec
    # No registered cohort matches Z000, so only 'all' is present.
    assert set(rec.keys()) == {"all"}


def test_stratify_flag_is_opt_in(tmp_path):
    """Without --stratify-by-cohort the output stays flat (Tier B
    behaviour). The flag is opt-in so existing callers don't break."""
    labevents_rows = [
        (i, 50912, 1.0 + 0.001 * i, "mg/dL", 0.6, 1.2)
        for i in range(150)
    ]
    duckdb_path = _make_fixture_with_diagnoses(
        tmp_path, labevents_rows=labevents_rows,
    )
    labitem_map = _make_labitem_mapping(tmp_path, {50912: "2160-0"})
    loinc_out = tmp_path / "loinc_out.json"
    dist_out = tmp_path / "dist_out.json"

    result = _run_generator(
        duckdb_path=duckdb_path,
        labitem_map_path=labitem_map,
        loinc_out=loinc_out,
        dist_out=dist_out,
        extra_args=["--min-rows", "100"],  # no --stratify-by-cohort
    )
    assert result.returncode == 0
    distributions = json.loads(dist_out.read_text())
    rec = distributions["50912"]
    # Flat shape preserved when flag is off.
    assert "n" in rec
    assert "mean" in rec
    assert isinstance(rec["n"], int)
    assert "all" not in rec  # no nested 'all' key when flat
