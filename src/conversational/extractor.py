"""Template-based SQL extraction from DuckDB driven by CompetencyQuestion."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb

from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    ExtractionResult,
    PatientFilter,
    TemporalConstraint,
)

_SAFE_COMPARISON_OPS = frozenset({">", "<", "=", ">=", "<="})


def extract(db_path: Path, cq: CompetencyQuestion) -> ExtractionResult:
    """Extract MIMIC-IV data from DuckDB based on a CompetencyQuestion.

    Opens a read-only connection, builds template SQL from the structured
    question fields, and returns patients/admissions/ICU stays/events.
    """
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        hadm_ids = _get_filtered_hadm_ids(conn, cq.patient_filters)
        if not hadm_ids:
            return ExtractionResult()

        events: dict[str, list[dict]] = {}
        for concept in cq.clinical_concepts:
            rows = _extract_concept(conn, concept, hadm_ids, cq.temporal_constraints)
            if rows:
                events.setdefault(concept.concept_type, []).extend(rows)

        return ExtractionResult(
            patients=_fetch_patients(conn, hadm_ids),
            admissions=_fetch_admissions(conn, hadm_ids),
            icu_stays=_fetch_icu_stays(conn, hadm_ids),
            events=events,
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Patient filtering
# ---------------------------------------------------------------------------


def _get_filtered_hadm_ids(
    conn: duckdb.DuckDBPyConnection,
    filters: list[PatientFilter],
) -> list[int]:
    """Return admission IDs matching all patient filters."""
    if not filters:
        return [r[0] for r in conn.execute("SELECT hadm_id FROM admissions").fetchall()]

    joins: list[str] = []
    conditions: list[str] = []
    params: list = []
    needs_patients = False

    for f in filters:
        if f.field == "age":
            needs_patients = True
            if f.operator not in _SAFE_COMPARISON_OPS:
                raise ValueError(f"Unsupported operator for age: {f.operator}")
            conditions.append(f"p.anchor_age {f.operator} ?")
            params.append(int(f.value))

        elif f.field == "gender":
            needs_patients = True
            conditions.append("p.gender = ?")
            params.append(f.value.upper())

        elif f.field == "diagnosis":
            joins.append(
                "JOIN diagnoses_icd di ON a.hadm_id = di.hadm_id "
                "JOIN d_icd_diagnoses dd ON di.icd_code = dd.icd_code "
                "AND di.icd_version = dd.icd_version"
            )
            conditions.append("(dd.long_title ILIKE ? OR di.icd_code LIKE ?)")
            params.extend([f"%{f.value}%", f"{f.value}%"])

        elif f.field == "admission_type":
            conditions.append("a.admission_type = ?")
            params.append(f.value)

        elif f.field == "subject_id":
            conditions.append("a.subject_id = ?")
            params.append(int(f.value))

    if needs_patients:
        joins.insert(0, "JOIN patients p ON a.subject_id = p.subject_id")

    join_sql = " ".join(joins)
    where_sql = " AND ".join(conditions)

    sql = f"SELECT DISTINCT a.hadm_id FROM admissions a {join_sql} WHERE {where_sql}"
    return [r[0] for r in conn.execute(sql, params).fetchall()]


# ---------------------------------------------------------------------------
# Temporal constraint helpers
# ---------------------------------------------------------------------------


def _parse_time_window(window: str) -> str:
    """Convert '48h', '7d', '30m' etc. to a DuckDB INTERVAL literal."""
    match = re.match(
        r"(\d+)\s*(h(?:ours?)?|d(?:ays?)?|m(?:in(?:utes?)?)?)",
        window.lower().strip(),
    )
    if not match:
        raise ValueError(f"Cannot parse time window: {window}")
    value = match.group(1)
    unit_char = match.group(2)[0]
    unit_map = {"h": "HOUR", "d": "DAY", "m": "MINUTE"}
    return f"INTERVAL '{value}' {unit_map[unit_char]}"


def _temporal_sql(
    constraints: list[TemporalConstraint],
    time_col: str,
    hadm_col: str,
) -> str:
    """Build SQL fragment for temporal constraints (empty string if none)."""
    parts: list[str] = []
    for tc in constraints:
        ref = tc.reference_event.lower()
        if "icu" not in ref:
            continue

        exists_prefix = (
            f"AND EXISTS (SELECT 1 FROM icustays _icu "
            f"WHERE _icu.hadm_id = {hadm_col}"
        )

        if tc.relation == "during":
            parts.append(
                f"{exists_prefix} "
                f"AND {time_col} >= _icu.intime "
                f"AND {time_col} <= _icu.outtime)"
            )
        elif tc.relation == "within" and tc.time_window:
            interval = _parse_time_window(tc.time_window)
            parts.append(
                f"{exists_prefix} "
                f"AND {time_col} >= _icu.intime "
                f"AND {time_col} <= _icu.intime + {interval})"
            )
        elif tc.relation == "before":
            parts.append(f"{exists_prefix} AND {time_col} < _icu.intime)")
        elif tc.relation == "after":
            parts.append(f"{exists_prefix} AND {time_col} > _icu.outtime)")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------


def _in_clause(values: list[int]) -> tuple[str, list[int]]:
    """Return (placeholders, params) for an IN clause."""
    return ", ".join(["?"] * len(values)), list(values)


# ---------------------------------------------------------------------------
# Concept extraction dispatch
# ---------------------------------------------------------------------------


def _extract_concept(
    conn: duckdb.DuckDBPyConnection,
    concept: ClinicalConcept,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint],
) -> list[dict]:
    """Dispatch to the appropriate handler by concept_type."""
    handlers = {
        "biomarker": _extract_biomarkers,
        "vital": _extract_vitals,
        "drug": _extract_drugs,
        "diagnosis": _extract_diagnoses,
        "microbiology": _extract_microbiology,
    }
    handler = handlers.get(concept.concept_type)
    if handler is None:
        return []
    return handler(conn, concept.name, hadm_ids, temporal)


# ---------------------------------------------------------------------------
# Per-concept-type extractors
# ---------------------------------------------------------------------------


def _extract_biomarkers(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint],
) -> list[dict]:
    ph, params = _in_clause(hadm_ids)
    tc_sql = _temporal_sql(temporal, "l.charttime", "l.hadm_id")
    sql = f"""
        SELECT l.labevent_id, l.subject_id, l.hadm_id, l.itemid,
               l.charttime, d.label, d.fluid, d.category,
               l.valuenum, l.valueuom, l.ref_range_lower, l.ref_range_upper
        FROM labevents l
        JOIN d_labitems d ON l.itemid = d.itemid
        WHERE d.label ILIKE ?
          AND l.valuenum IS NOT NULL
          AND l.hadm_id IN ({ph})
          {tc_sql}
        ORDER BY l.charttime
    """
    cols = [
        "labevent_id", "subject_id", "hadm_id", "itemid", "charttime",
        "label", "fluid", "category", "valuenum", "valueuom",
        "ref_range_lower", "ref_range_upper",
    ]
    return [dict(zip(cols, r)) for r in conn.execute(sql, [f"%{name}%"] + params).fetchall()]


def _extract_vitals(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint],
) -> list[dict]:
    ph, params = _in_clause(hadm_ids)
    tc_sql = _temporal_sql(temporal, "c.charttime", "c.hadm_id")
    sql = f"""
        SELECT c.stay_id, c.subject_id, c.hadm_id, c.itemid,
               c.charttime, d.label, d.category, c.valuenum
        FROM chartevents c
        JOIN d_items d ON c.itemid = d.itemid
        WHERE d.label ILIKE ?
          AND c.valuenum IS NOT NULL
          AND c.hadm_id IN ({ph})
          {tc_sql}
        ORDER BY c.charttime
    """
    cols = [
        "stay_id", "subject_id", "hadm_id", "itemid",
        "charttime", "label", "category", "valuenum",
    ]
    return [dict(zip(cols, r)) for r in conn.execute(sql, [f"%{name}%"] + params).fetchall()]


def _extract_drugs(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint],
) -> list[dict]:
    ph, params = _in_clause(hadm_ids)
    tc_sql = _temporal_sql(temporal, "pr.starttime", "pr.hadm_id")
    sql = f"""
        SELECT pr.hadm_id, pr.subject_id, pr.drug, pr.starttime,
               pr.stoptime, pr.dose_val_rx, pr.dose_unit_rx, pr.route
        FROM prescriptions pr
        WHERE pr.drug ILIKE ?
          AND pr.hadm_id IN ({ph})
          {tc_sql}
        ORDER BY pr.starttime
    """
    cols = [
        "hadm_id", "subject_id", "drug", "starttime",
        "stoptime", "dose_val_rx", "dose_unit_rx", "route",
    ]
    return [dict(zip(cols, r)) for r in conn.execute(sql, [f"%{name}%"] + params).fetchall()]


def _extract_diagnoses(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint] | None = None,
) -> list[dict]:
    ph, params = _in_clause(hadm_ids)
    sql = f"""
        SELECT di.hadm_id, di.subject_id, di.seq_num, di.icd_code,
               di.icd_version, dd.long_title
        FROM diagnoses_icd di
        LEFT JOIN d_icd_diagnoses dd
          ON di.icd_code = dd.icd_code AND di.icd_version = dd.icd_version
        WHERE (dd.long_title ILIKE ? OR di.icd_code LIKE ?)
          AND di.hadm_id IN ({ph})
        ORDER BY di.seq_num
    """
    cols = [
        "hadm_id", "subject_id", "seq_num", "icd_code",
        "icd_version", "long_title",
    ]
    return [
        dict(zip(cols, r))
        for r in conn.execute(sql, [f"%{name}%", f"{name}%"] + params).fetchall()
    ]


def _extract_microbiology(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint],
) -> list[dict]:
    ph, params = _in_clause(hadm_ids)
    tc_sql = _temporal_sql(temporal, "m.charttime", "m.hadm_id")
    sql = f"""
        SELECT m.microevent_id, m.subject_id, m.hadm_id, m.charttime,
               m.spec_type_desc, m.org_name
        FROM microbiologyevents m
        WHERE (m.spec_type_desc ILIKE ? OR m.org_name ILIKE ?)
          AND m.hadm_id IN ({ph})
          {tc_sql}
        ORDER BY m.charttime
    """
    cols = [
        "microevent_id", "subject_id", "hadm_id", "charttime",
        "spec_type_desc", "org_name",
    ]
    return [
        dict(zip(cols, r))
        for r in conn.execute(sql, [f"%{name}%", f"%{name}%"] + params).fetchall()
    ]


# ---------------------------------------------------------------------------
# Supporting context fetchers
# ---------------------------------------------------------------------------


def _fetch_patients(conn: duckdb.DuckDBPyConnection, hadm_ids: list[int]) -> list[dict]:
    ph, params = _in_clause(hadm_ids)
    sql = f"""
        SELECT DISTINCT p.subject_id, p.gender, p.anchor_age
        FROM patients p
        JOIN admissions a ON p.subject_id = a.subject_id
        WHERE a.hadm_id IN ({ph})
        ORDER BY p.subject_id
    """
    cols = ["subject_id", "gender", "anchor_age"]
    return [dict(zip(cols, r)) for r in conn.execute(sql, params).fetchall()]


def _fetch_admissions(conn: duckdb.DuckDBPyConnection, hadm_ids: list[int]) -> list[dict]:
    ph, params = _in_clause(hadm_ids)
    sql = f"""
        SELECT hadm_id, subject_id, admittime, dischtime,
               admission_type, discharge_location
        FROM admissions
        WHERE hadm_id IN ({ph})
        ORDER BY admittime
    """
    cols = [
        "hadm_id", "subject_id", "admittime", "dischtime",
        "admission_type", "discharge_location",
    ]
    return [dict(zip(cols, r)) for r in conn.execute(sql, params).fetchall()]


def _fetch_icu_stays(conn: duckdb.DuckDBPyConnection, hadm_ids: list[int]) -> list[dict]:
    ph, params = _in_clause(hadm_ids)
    sql = f"""
        SELECT stay_id, hadm_id, subject_id, intime, outtime, los
        FROM icustays
        WHERE hadm_id IN ({ph})
        ORDER BY intime
    """
    cols = ["stay_id", "hadm_id", "subject_id", "intime", "outtime", "los"]
    return [dict(zip(cols, r)) for r in conn.execute(sql, params).fetchall()]
