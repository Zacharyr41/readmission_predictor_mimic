"""Template-based SQL extraction driven by CompetencyQuestion.

Supports two backends:
- DuckDB (local): ``extract(db_path, cq)``
- BigQuery (remote): ``extract_bigquery(cq, project=...)``
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from src.conversational.models import (
    ClinicalConcept,
    CompetencyQuestion,
    ExtractionConfig,
    ExtractionResult,
    PatientFilter,
    TemporalConstraint,
)

if TYPE_CHECKING:
    from google.cloud import bigquery

logger = logging.getLogger(__name__)

_SAFE_COMPARISON_OPS = frozenset({">", "<", "=", ">=", "<="})

# BigQuery fully-qualified table mapping (physionet-data MIMIC-IV v3.1)
_BQ_SOURCE = "physionet-data"
_BQ_TABLES = {
    "patients": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.patients",
    "admissions": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.admissions",
    "icustays": f"{_BQ_SOURCE}.mimiciv_3_1_icu.icustays",
    "labevents": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.labevents",
    "d_labitems": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.d_labitems",
    "chartevents": f"{_BQ_SOURCE}.mimiciv_3_1_icu.chartevents",
    "d_items": f"{_BQ_SOURCE}.mimiciv_3_1_icu.d_items",
    "prescriptions": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.prescriptions",
    "diagnoses_icd": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.diagnoses_icd",
    "d_icd_diagnoses": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.d_icd_diagnoses",
    "microbiologyevents": f"{_BQ_SOURCE}.mimiciv_3_1_hosp.microbiologyevents",
}


def _get_bigquery_module():
    """Lazy import of google.cloud.bigquery (patchable for tests)."""
    from google.cloud import bigquery as bq

    return bq


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


class _DuckDBBackend:
    """Thin wrapper around a read-only DuckDB connection."""

    def __init__(self, db_path: Path) -> None:
        self._conn = duckdb.connect(str(db_path), read_only=True)

    def table(self, name: str) -> str:
        return name

    def execute(self, sql: str, params: list) -> list[tuple]:
        return self._conn.execute(sql, params).fetchall()

    @staticmethod
    def ilike(column: str) -> str:
        """Case-insensitive LIKE expression (DuckDB supports ILIKE)."""
        return f"{column} ILIKE ?"

    @staticmethod
    def random_fn() -> str:
        return "RANDOM()"

    def close(self) -> None:
        self._conn.close()


class _BigQueryBackend:
    """Queries physionet-data MIMIC-IV datasets on BigQuery directly."""

    def __init__(self, project: str | None = None) -> None:
        bq = _get_bigquery_module()
        self._client: bigquery.Client = bq.Client(project=project)
        self._bq = bq

    def table(self, name: str) -> str:
        return f"`{_BQ_TABLES[name]}`"

    def execute(self, sql: str, params: list) -> list[tuple]:
        bq_sql, bq_params = self._convert_params(sql, params)
        config = self._bq.QueryJobConfig(query_parameters=bq_params)
        rows = self._client.query(bq_sql, job_config=config).result()
        return [tuple(row.values()) for row in rows]

    @staticmethod
    def ilike(column: str) -> str:
        """Case-insensitive LIKE for BigQuery (no ILIKE support)."""
        return f"LOWER({column}) LIKE LOWER(?)"

    @staticmethod
    def random_fn() -> str:
        return "RAND()"

    def _convert_params(
        self, sql: str, params: list
    ) -> tuple[str, list]:
        """Replace ``?`` with ``@p0, @p1, ...`` and build typed parameters."""
        bq_params = []
        for i, val in enumerate(params):
            name = f"p{i}"
            sql = sql.replace("?", f"@{name}", 1)
            if isinstance(val, int):
                bq_params.append(self._bq.ScalarQueryParameter(name, "INT64", val))
            elif isinstance(val, float):
                bq_params.append(self._bq.ScalarQueryParameter(name, "FLOAT64", val))
            else:
                bq_params.append(
                    self._bq.ScalarQueryParameter(name, "STRING", str(val))
                )
        return sql, bq_params

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract(
    db_path: Path,
    cq: CompetencyQuestion,
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    """Extract MIMIC-IV data from a local DuckDB database.

    Opens a read-only connection, builds template SQL from the structured
    question fields, and returns patients/admissions/ICU stays/events.
    """
    backend = _DuckDBBackend(db_path)
    try:
        return _extract(backend, cq, config=config)
    finally:
        backend.close()


def extract_bigquery(
    cq: CompetencyQuestion,
    project: str | None = None,
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    """Extract MIMIC-IV data directly from BigQuery.

    Queries the ``physionet-data`` MIMIC-IV datasets. Requires
    Application Default Credentials (``gcloud auth application-default login``).
    """
    backend = _BigQueryBackend(project)
    try:
        return _extract(backend, cq, config=config)
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Core extraction logic (backend-agnostic)
# ---------------------------------------------------------------------------


def _extract(
    backend: _DuckDBBackend | _BigQueryBackend,
    cq: CompetencyQuestion,
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    hadm_ids = _get_filtered_hadm_ids(backend, cq.patient_filters, config=config)
    if not hadm_ids:
        return ExtractionResult()

    events: dict[str, list[dict]] = {}
    for concept in cq.clinical_concepts:
        rows = _extract_concept(backend, concept, hadm_ids, cq.temporal_constraints)
        if rows:
            events.setdefault(concept.concept_type, []).extend(rows)

    return ExtractionResult(
        patients=_fetch_patients(backend, hadm_ids),
        admissions=_fetch_admissions(backend, hadm_ids),
        icu_stays=_fetch_icu_stays(backend, hadm_ids),
        events=events,
    )


# ---------------------------------------------------------------------------
# Patient filtering
# ---------------------------------------------------------------------------


def _get_filtered_hadm_ids(
    backend: _DuckDBBackend | _BigQueryBackend,
    filters: list[PatientFilter],
    config: ExtractionConfig | None = None,
) -> list[int]:
    """Return admission IDs matching all patient filters."""
    cfg = config or ExtractionConfig()
    t = backend.table

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
                f"JOIN {t('diagnoses_icd')} di ON a.hadm_id = di.hadm_id "
                f"JOIN {t('d_icd_diagnoses')} dd ON di.icd_code = dd.icd_code "
                f"AND di.icd_version = dd.icd_version"
            )
            conditions.append(
                f"({backend.ilike('dd.long_title')} OR di.icd_code LIKE ?)"
            )
            params.extend([f"%{f.value}%", f"{f.value}%"])

        elif f.field == "admission_type":
            conditions.append("a.admission_type = ?")
            params.append(f.value)

        elif f.field == "subject_id":
            conditions.append("a.subject_id = ?")
            params.append(int(f.value))

        elif f.field in ("readmitted_30d", "readmitted_60d"):
            joins.append(
                f"JOIN {t('readmission_labels')} rl ON a.hadm_id = rl.hadm_id"
            )
            if f.operator not in _SAFE_COMPARISON_OPS:
                raise ValueError(
                    f"Unsupported operator for {f.field}: {f.operator}"
                )
            conditions.append(f"rl.{f.field} {f.operator} ?")
            params.append(int(f.value))

        else:
            logger.warning("Ignoring unsupported patient filter field: %s", f.field)

    if needs_patients:
        joins.insert(0, f"JOIN {t('patients')} p ON a.subject_id = p.subject_id")

    join_sql = " ".join(joins)
    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

    order = "admittime DESC" if cfg.cohort_strategy == "recent" else backend.random_fn()
    # Use a subquery so ORDER BY references a column visible after SELECT DISTINCT
    # (BigQuery rejects ORDER BY on columns not in the DISTINCT select list).
    inner = (
        f"SELECT DISTINCT a.hadm_id, a.admittime"
        f" FROM {t('admissions')} a {join_sql}{where_clause}"
    )
    sql = (
        f"SELECT hadm_id FROM ({inner}) sub"
        f" ORDER BY {order} LIMIT {cfg.max_cohort_size}"
    )
    hadm_ids = [r[0] for r in backend.execute(sql, params)]
    if len(hadm_ids) == cfg.max_cohort_size:
        logger.warning(
            "Cohort capped at %d admissions (limit reached).",
            cfg.max_cohort_size,
        )
    return hadm_ids


# ---------------------------------------------------------------------------
# Temporal constraint helpers
# ---------------------------------------------------------------------------


def _parse_time_window(window: str) -> str:
    """Convert '48h', '7d', '30m' etc. to an INTERVAL literal."""
    match = re.match(
        r"(\d+)\s*(h(?:ours?)?|d(?:ays?)?|m(?:in(?:utes?)?)?)",
        window.lower().strip(),
    )
    if not match:
        raise ValueError(f"Cannot parse time window: {window}")
    value = match.group(1)
    unit_char = match.group(2)[0]
    unit_map = {"h": "HOUR", "d": "DAY", "m": "MINUTE"}
    return f"INTERVAL {value} {unit_map[unit_char]}"


def _temporal_sql(
    constraints: list[TemporalConstraint],
    time_col: str,
    hadm_col: str,
    backend: _DuckDBBackend | _BigQueryBackend,
) -> str:
    """Build SQL fragment for temporal constraints (empty string if none)."""
    icu_tbl = backend.table("icustays")
    parts: list[str] = []
    for tc in constraints:
        ref = tc.reference_event.lower()
        if "icu" not in ref:
            continue

        exists_prefix = (
            f"AND EXISTS (SELECT 1 FROM {icu_tbl} _icu "
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
    backend: _DuckDBBackend | _BigQueryBackend,
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
    if concept.concept_type == "biomarker":
        return handler(backend, concept, hadm_ids, temporal)
    return handler(backend, concept.name, hadm_ids, temporal)


# ---------------------------------------------------------------------------
# Per-concept-type extractors
# ---------------------------------------------------------------------------


def _extract_biomarkers(
    backend: _DuckDBBackend | _BigQueryBackend,
    concept: ClinicalConcept,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint],
) -> list[dict]:
    t = backend.table
    name = concept.name
    ph, params = _in_clause(hadm_ids)
    tc_sql = _temporal_sql(temporal, "l.charttime", "l.hadm_id", backend)

    # Optional attribute filtering (e.g. fluid="Blood" for serum)
    attr_sql = ""
    attr_params: list[str] = []
    if concept.attributes:
        attr_clauses = [backend.ilike("d.fluid") for _ in concept.attributes]
        attr_sql = f"AND ({' OR '.join(attr_clauses)})"
        attr_params = [f"%{a}%" for a in concept.attributes]

    sql = f"""
        SELECT l.labevent_id, l.subject_id, l.hadm_id, l.itemid,
               l.charttime, d.label, d.fluid, d.category,
               l.valuenum, l.valueuom, l.ref_range_lower, l.ref_range_upper
        FROM {t('labevents')} l
        JOIN {t('d_labitems')} d ON l.itemid = d.itemid
        WHERE {backend.ilike('d.label')}
          AND l.valuenum IS NOT NULL
          AND l.hadm_id IN ({ph})
          {attr_sql}
          {tc_sql}
        ORDER BY l.charttime
    """
    cols = [
        "labevent_id", "subject_id", "hadm_id", "itemid", "charttime",
        "label", "fluid", "category", "valuenum", "valueuom",
        "ref_range_lower", "ref_range_upper",
    ]
    return [dict(zip(cols, r)) for r in backend.execute(
        sql, [f"%{name}%"] + params + attr_params,
    )]


def _extract_vitals(
    backend: _DuckDBBackend | _BigQueryBackend,
    name: str,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint],
) -> list[dict]:
    t = backend.table
    ph, params = _in_clause(hadm_ids)
    tc_sql = _temporal_sql(temporal, "c.charttime", "c.hadm_id", backend)
    sql = f"""
        SELECT c.stay_id, c.subject_id, c.hadm_id, c.itemid,
               c.charttime, d.label, d.category, c.valuenum
        FROM {t('chartevents')} c
        JOIN {t('d_items')} d ON c.itemid = d.itemid
        WHERE {backend.ilike('d.label')}
          AND c.valuenum IS NOT NULL
          AND c.hadm_id IN ({ph})
          {tc_sql}
        ORDER BY c.charttime
    """
    cols = [
        "stay_id", "subject_id", "hadm_id", "itemid",
        "charttime", "label", "category", "valuenum",
    ]
    return [dict(zip(cols, r)) for r in backend.execute(sql, [f"%{name}%"] + params)]


def _extract_drugs(
    backend: _DuckDBBackend | _BigQueryBackend,
    name: str,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint],
) -> list[dict]:
    t = backend.table
    ph, params = _in_clause(hadm_ids)
    tc_sql = _temporal_sql(temporal, "pr.starttime", "pr.hadm_id", backend)
    sql = f"""
        SELECT pr.hadm_id, pr.subject_id, pr.drug, pr.starttime,
               pr.stoptime, pr.dose_val_rx, pr.dose_unit_rx, pr.route
        FROM {t('prescriptions')} pr
        WHERE {backend.ilike('pr.drug')}
          AND pr.hadm_id IN ({ph})
          {tc_sql}
        ORDER BY pr.starttime
    """
    cols = [
        "hadm_id", "subject_id", "drug", "starttime",
        "stoptime", "dose_val_rx", "dose_unit_rx", "route",
    ]
    return [dict(zip(cols, r)) for r in backend.execute(sql, [f"%{name}%"] + params)]


def _extract_diagnoses(
    backend: _DuckDBBackend | _BigQueryBackend,
    name: str,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint] | None = None,
) -> list[dict]:
    t = backend.table
    ph, params = _in_clause(hadm_ids)
    sql = f"""
        SELECT di.hadm_id, di.subject_id, di.seq_num, di.icd_code,
               di.icd_version, dd.long_title
        FROM {t('diagnoses_icd')} di
        LEFT JOIN {t('d_icd_diagnoses')} dd
          ON di.icd_code = dd.icd_code AND di.icd_version = dd.icd_version
        WHERE ({backend.ilike('dd.long_title')} OR di.icd_code LIKE ?)
          AND di.hadm_id IN ({ph})
        ORDER BY di.seq_num
    """
    cols = [
        "hadm_id", "subject_id", "seq_num", "icd_code",
        "icd_version", "long_title",
    ]
    return [
        dict(zip(cols, r))
        for r in backend.execute(sql, [f"%{name}%", f"{name}%"] + params)
    ]


def _extract_microbiology(
    backend: _DuckDBBackend | _BigQueryBackend,
    name: str,
    hadm_ids: list[int],
    temporal: list[TemporalConstraint],
) -> list[dict]:
    t = backend.table
    ph, params = _in_clause(hadm_ids)
    tc_sql = _temporal_sql(temporal, "m.charttime", "m.hadm_id", backend)
    sql = f"""
        SELECT m.microevent_id, m.subject_id, m.hadm_id, m.charttime,
               m.spec_type_desc, m.org_name
        FROM {t('microbiologyevents')} m
        WHERE ({backend.ilike('m.spec_type_desc')} OR {backend.ilike('m.org_name')})
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
        for r in backend.execute(sql, [f"%{name}%", f"%{name}%"] + params)
    ]


# ---------------------------------------------------------------------------
# Supporting context fetchers
# ---------------------------------------------------------------------------


def _fetch_patients(
    backend: _DuckDBBackend | _BigQueryBackend, hadm_ids: list[int]
) -> list[dict]:
    t = backend.table
    ph, params = _in_clause(hadm_ids)
    sql = f"""
        SELECT DISTINCT p.subject_id, p.gender, p.anchor_age
        FROM {t('patients')} p
        JOIN {t('admissions')} a ON p.subject_id = a.subject_id
        WHERE a.hadm_id IN ({ph})
        ORDER BY p.subject_id
    """
    cols = ["subject_id", "gender", "anchor_age"]
    return [dict(zip(cols, r)) for r in backend.execute(sql, params)]


def _fetch_admissions(
    backend: _DuckDBBackend | _BigQueryBackend, hadm_ids: list[int]
) -> list[dict]:
    t = backend.table
    ph, params = _in_clause(hadm_ids)

    # Try to join readmission_labels for readmitted_30d/60d.
    # Fall back to defaults (0) if the table doesn't exist.
    try:
        sql = f"""
            SELECT a.hadm_id, a.subject_id, a.admittime, a.dischtime,
                   a.admission_type, a.discharge_location,
                   a.hospital_expire_flag,
                   COALESCE(rl.readmitted_30d, 0) AS readmitted_30d,
                   COALESCE(rl.readmitted_60d, 0) AS readmitted_60d
            FROM {t('admissions')} a
            LEFT JOIN {t('readmission_labels')} rl ON a.hadm_id = rl.hadm_id
            WHERE a.hadm_id IN ({ph})
            ORDER BY a.admittime
        """
        cols = [
            "hadm_id", "subject_id", "admittime", "dischtime",
            "admission_type", "discharge_location", "hospital_expire_flag",
            "readmitted_30d", "readmitted_60d",
        ]
        return [dict(zip(cols, r)) for r in backend.execute(sql, params)]
    except Exception:
        # readmission_labels table may not exist
        sql = f"""
            SELECT hadm_id, subject_id, admittime, dischtime,
                   admission_type, discharge_location, hospital_expire_flag
            FROM {t('admissions')}
            WHERE hadm_id IN ({ph})
            ORDER BY admittime
        """
        cols = [
            "hadm_id", "subject_id", "admittime", "dischtime",
            "admission_type", "discharge_location", "hospital_expire_flag",
        ]
        return [dict(zip(cols, r)) for r in backend.execute(sql, params)]


def _fetch_icu_stays(
    backend: _DuckDBBackend | _BigQueryBackend, hadm_ids: list[int]
) -> list[dict]:
    t = backend.table
    ph, params = _in_clause(hadm_ids)
    sql = f"""
        SELECT stay_id, hadm_id, subject_id, intime, outtime, los
        FROM {t('icustays')}
        WHERE hadm_id IN ({ph})
        ORDER BY intime
    """
    cols = ["stay_id", "hadm_id", "subject_id", "intime", "outtime", "los"]
    return [dict(zip(cols, r)) for r in backend.execute(sql, params)]
