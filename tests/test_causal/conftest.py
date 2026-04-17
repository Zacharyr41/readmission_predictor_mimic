"""Fixtures for ``src.causal`` tests.

Extends ``tests/conftest.py::synthetic_duckdb_with_events`` with the
additional prescriptions / procedures_icd rows needed to exercise all
four intervention-resolution paths (RxNorm, SNOMED, ICD-10-PCS, LOINC)
plus the mutual-exclusivity edge cases in
``src.causal.treatment_assignment``.

The layout is intentional — each admission in the base fixture is
given a distinct exposure profile so a single assignment run can
observe: clean single-arm matches (n=1), no exposure (n=0), and
overlap (n≥2).
"""

from __future__ import annotations

import duckdb
import pytest


@pytest.fixture
def synthetic_duckdb_for_causal(
    synthetic_duckdb_with_events: duckdb.DuckDBPyConnection,
) -> duckdb.DuckDBPyConnection:
    """Extend the base synthetic fixture with causal-inference test data.

    Exposure profile by hadm_id:

      * 101 (subject 1) — Alteplase (tPA)  [+ Vancomycin already present]
      * 102 (subject 1) — Warfarin
      * 103 (subject 2) — (Ceftriaxone already present, no new drug)
      * 104 (subject 3) — NO drug exposure (tests n_matching=0)
      * 105 (subject 4) — Alteplase AND Warfarin (tests n_matching=2 overlap)
      * 106 (subject 5) — (Unchanged)

    Procedures added: 101, 102 carry an ICD-10-PCS thrombolytic code
    (3E03317) so the ICD-10-PCS resolver path has hits.
    """
    conn = synthetic_duckdb_with_events

    # Augment prescriptions. The base fixture inserted 2 rows (vanc→101,
    # ceftriaxone→103). We add: tPA→101, Warfarin→102, tPA+Warfarin→105.
    conn.execute("""
        INSERT INTO prescriptions VALUES
        (1, 101, '2150-01-15 09:00:00', '2150-01-15 11:00:00', 'Alteplase', 90.0, 'mg', 'IV'),
        (1, 102, '2150-02-10 12:00:00', '2150-02-15 08:00:00', 'Warfarin', 5.0, 'mg', 'PO'),
        (4, 105, '2150-07-01 10:00:00', '2150-07-01 10:30:00', 'Alteplase', 90.0, 'mg', 'IV'),
        (4, 105, '2150-07-02 08:00:00', '2150-07-05 08:00:00', 'Warfarin', 5.0, 'mg', 'PO')
    """)

    # procedures_icd is not created by synthetic_duckdb_with_events — create + populate.
    conn.execute("""
        CREATE TABLE procedures_icd (
            subject_id INTEGER,
            hadm_id INTEGER,
            seq_num INTEGER,
            icd_code VARCHAR,
            icd_version INTEGER
        )
    """)
    # 3E03317 = "Introduction of Other Thrombolytic into Peripheral Vein,
    # Percutaneous Approach" (canonical ICD-10-PCS for IV tPA).
    conn.execute("""
        INSERT INTO procedures_icd VALUES
        (1, 101, 1, '3E03317', 10),
        (1, 102, 1, '3E03317', 10),
        (4, 105, 1, '3E03317', 10)
    """)

    return conn


class DuckDBAdapter:
    """Thin ``backend`` adapter exposing just ``execute(sql, params)``.

    The real ``_DuckDBBackend`` in ``src.conversational.extractor`` is
    wired to a path + connection-per-thread. For causal tests we only
    need the ``execute(sql, params) -> list[tuple]`` contract, so this
    wraps the raw DuckDB connection from the fixture.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def execute(self, sql: str, params: list) -> list[tuple]:
        return self._conn.execute(sql, params).fetchall()


@pytest.fixture
def duckdb_backend(synthetic_duckdb_for_causal: duckdb.DuckDBPyConnection) -> DuckDBAdapter:
    return DuckDBAdapter(synthetic_duckdb_for_causal)
