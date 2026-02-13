"""Derived tables for MIMIC-IV analysis.

Creates derived tables for age calculation, cohort selection, and readmission labels.
"""

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# ICD-10 stroke prefixes → ICD-9 equivalents (LIKE patterns).
# 433_1 / 434_1: the trailing "1" means "with infarction" (acute stroke).
# Codes without "1" (e.g. 43300) are chronic stenosis and deliberately excluded.
_STROKE_ICD10_TO_ICD9: dict[str, list[str]] = {
    "I63": ["433_1", "434_1", "436"],  # Ischemic stroke
    "I60": ["430"],                     # Subarachnoid hemorrhage
    "I61": ["431"],                     # Intracerebral hemorrhage
}


def create_age_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create derived age table with age at admission.

    Age is computed as: anchor_age + (YEAR(admittime) - anchor_year)

    Args:
        conn: DuckDB connection with patients and admissions tables loaded.
    """
    conn.execute("""
        CREATE OR REPLACE TABLE age AS
        SELECT
            p.subject_id,
            a.hadm_id,
            p.anchor_age + (YEAR(a.admittime) - p.anchor_year) AS age
        FROM patients p
        JOIN admissions a ON p.subject_id = a.subject_id
    """)

    row_count = conn.execute("SELECT COUNT(*) FROM age").fetchone()[0]
    logger.info(f"Created age table with {row_count:,} rows")


def select_neurology_cohort(
    conn: duckdb.DuckDBPyConnection,
    icd_prefixes: list[str],
) -> pd.DataFrame:
    """Select cohort based on ICD diagnosis codes and inclusion criteria.

    Inclusion criteria (from reference repo):
    - Age 18-89
    - ICU stay > 24 hours and < 100 days (2400 hours)
    - First ICU stay per admission
    - Has diagnosis matching one of the ICD prefixes

    Args:
        conn: DuckDB connection with required tables loaded.
        icd_prefixes: List of ICD-10 code prefixes (e.g., ["I63", "I61"]).
            ICD-9 equivalents are matched automatically for known stroke codes.

    Returns:
        DataFrame with columns: subject_id, hadm_id, stay_id
    """
    # Build version-aware ICD filter (ICD-10 + ICD-9 equivalents)
    icd10_parts = [f"d.icd_code LIKE '{p}%'" for p in icd_prefixes]
    icd10_clause = f"(d.icd_version = 10 AND ({' OR '.join(icd10_parts)}))"

    icd9_parts = []
    for prefix in icd_prefixes:
        for pattern in _STROKE_ICD10_TO_ICD9.get(prefix, []):
            icd9_parts.append(f"d.icd_code LIKE '{pattern}'")

    if icd9_parts:
        icd9_clause = f"(d.icd_version = 9 AND ({' OR '.join(icd9_parts)}))"
        version_filter = f"({icd10_clause} OR {icd9_clause})"
    else:
        version_filter = icd10_clause

    query = f"""
        WITH ranked_icu_stays AS (
            -- Rank ICU stays per admission to get first stay only
            SELECT
                i.subject_id,
                i.hadm_id,
                i.stay_id,
                i.intime,
                i.outtime,
                DATE_DIFF('hour', i.intime, i.outtime) AS icu_hours,
                ROW_NUMBER() OVER (
                    PARTITION BY i.hadm_id
                    ORDER BY i.intime
                ) AS stay_rank
            FROM icustays i
        ),
        eligible_stays AS (
            -- Apply ICU stay duration criteria and first stay filter
            SELECT *
            FROM ranked_icu_stays
            WHERE stay_rank = 1
              AND icu_hours > 24
              AND icu_hours < 2400
        ),
        diagnosis_filter AS (
            -- Get unique hadm_ids with matching *principal* diagnosis
            SELECT DISTINCT hadm_id
            FROM diagnoses_icd d
            WHERE d.seq_num = 1
              AND {version_filter}
        )
        SELECT
            es.subject_id,
            es.hadm_id,
            es.stay_id
        FROM eligible_stays es
        JOIN age a ON es.subject_id = a.subject_id AND es.hadm_id = a.hadm_id
        JOIN diagnosis_filter df ON es.hadm_id = df.hadm_id
        WHERE a.age BETWEEN 18 AND 89
        ORDER BY es.subject_id, es.hadm_id
    """

    df = conn.execute(query).fetchdf()
    logger.info(f"Selected neurology cohort with {len(df):,} patients")
    return df


def create_readmission_labels(
    conn: duckdb.DuckDBPyConnection,
    windows: list[int] | None = None,
) -> None:
    """Create readmission labels table.

    Uses LEAD() window function to find next admission for each patient.
    Excludes patients who died in hospital (hospital_expire_flag=1) and
    patients discharged to hospice or against medical advice.

    Args:
        conn: DuckDB connection with admissions table loaded.
        windows: List of readmission windows in days (default: [30, 60]).
    """
    if windows is None:
        windows = [30, 60]

    # Build dynamic columns for each window
    window_columns = ", ".join(
        [f"CASE WHEN days_to_readmission IS NOT NULL AND days_to_readmission <= {w} THEN 1 ELSE 0 END AS readmitted_{w}d"
         for w in windows]
    )

    query = f"""
        CREATE OR REPLACE TABLE readmission_labels AS
        WITH ordered_admissions AS (
            SELECT
                subject_id,
                hadm_id,
                dischtime,
                discharge_location,
                hospital_expire_flag,
                LEAD(admittime) OVER (
                    PARTITION BY subject_id
                    ORDER BY admittime
                ) AS next_admittime
            FROM admissions
        ),
        with_days AS (
            SELECT
                subject_id,
                hadm_id,
                dischtime,
                discharge_location,
                next_admittime,
                DATE_DIFF('day', dischtime, next_admittime) AS days_to_readmission,
                hospital_expire_flag
            FROM ordered_admissions
        )
        SELECT
            subject_id,
            hadm_id,
            dischtime,
            next_admittime,
            days_to_readmission,
            {window_columns}
        FROM with_days
        WHERE hospital_expire_flag = 0
          AND discharge_location NOT IN ('HOSPICE', 'AGAINST ADVICE')
    """

    conn.execute(query)

    row_count = conn.execute("SELECT COUNT(*) FROM readmission_labels").fetchone()[0]
    logger.info(f"Created readmission_labels table with {row_count:,} rows")
