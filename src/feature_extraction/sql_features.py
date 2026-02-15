"""DuckDB SQL feature extractors for hospital readmission prediction.

These extractors replace the equivalent SPARQL-based extractors for tabular
features that don't need the RDF graph (demographics, stay, labs, vitals,
medications, diagnoses). They query DuckDB directly, avoiding the costly
round-trip through rdflib's in-memory SPARQL engine.

All functions expect a ``cohort`` temp table to be registered on the connection
with columns: subject_id, hadm_id, stay_id.
"""

import logging

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_labels_sql(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Extract readmission labels for cohort admissions.

    Returns:
        DataFrame with columns: hadm_id, readmitted_30d, readmitted_60d
    """
    df = conn.execute("""
        SELECT r.hadm_id, r.readmitted_30d, r.readmitted_60d
        FROM readmission_labels r
        WHERE r.hadm_id IN (SELECT hadm_id FROM cohort)
    """).fetchdf()
    return df


def extract_subject_ids_sql(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Extract subject_id for each cohort admission.

    Returns:
        DataFrame with columns: hadm_id, subject_id
    """
    df = conn.execute("""
        SELECT a.hadm_id, a.subject_id
        FROM admissions a
        WHERE a.hadm_id IN (SELECT hadm_id FROM cohort)
    """).fetchdf()
    return df


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------


def extract_demographics_sql(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Extract demographic features for cohort admissions.

    Returns:
        DataFrame with columns: hadm_id, age, gender_M, gender_F
    """
    df = conn.execute("""
        SELECT
            a.hadm_id,
            ag.age,
            CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS gender_M,
            CASE WHEN p.gender = 'F' THEN 1 ELSE 0 END AS gender_F
        FROM admissions a
        JOIN patients p ON a.subject_id = p.subject_id
        JOIN age ag ON a.subject_id = ag.subject_id AND a.hadm_id = ag.hadm_id
        WHERE a.hadm_id IN (SELECT hadm_id FROM cohort)
    """).fetchdf()
    return df


def extract_stay_features_sql(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Extract ICU stay features for cohort admissions.

    Returns:
        DataFrame with columns: hadm_id, icu_los_hours, num_icu_days,
                                admission_type_* (one-hot encoded)
    """
    df = conn.execute("""
        SELECT
            a.hadm_id,
            a.admission_type,
            i.los * 24.0 AS icu_los_hours,
            DATE_DIFF('day', CAST(i.intime AS DATE), CAST(i.outtime AS DATE)) + 1
                AS num_icu_days
        FROM admissions a
        JOIN icustays i ON a.hadm_id = i.hadm_id
        WHERE a.hadm_id IN (SELECT hadm_id FROM cohort)
    """).fetchdf()

    if df.empty:
        return pd.DataFrame(columns=["hadm_id", "icu_los_hours", "num_icu_days"])

    # One-hot encode admission type
    if "admission_type" in df.columns:
        dummies = pd.get_dummies(df["admission_type"], prefix="admission_type")
        df = pd.concat([df.drop("admission_type", axis=1), dummies], axis=1)

    return df


def extract_lab_summary_sql(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Extract lab (biomarker) summary statistics for cohort admissions.

    Scoped to the ICU stay window (charttime between intime and outtime).

    Returns:
        DataFrame with columns: hadm_id, {biomarker}_{mean,min,max,std,count,
                                first,last,abnormal_rate}
    """
    long_df = conn.execute("""
        WITH lab_with_order AS (
            SELECT
                i.hadm_id,
                d.label AS biomarker,
                l.valuenum,
                l.ref_range_lower,
                l.ref_range_upper,
                ROW_NUMBER() OVER (
                    PARTITION BY i.hadm_id, d.label ORDER BY l.charttime ASC
                ) AS rn_asc,
                ROW_NUMBER() OVER (
                    PARTITION BY i.hadm_id, d.label ORDER BY l.charttime DESC
                ) AS rn_desc
            FROM labevents l
            JOIN d_labitems d ON l.itemid = d.itemid
            JOIN icustays i ON l.hadm_id = i.hadm_id
            WHERE l.valuenum IS NOT NULL
              AND l.charttime >= i.intime
              AND l.charttime <= i.outtime
              AND i.hadm_id IN (SELECT hadm_id FROM cohort)
        ),
        agg AS (
            SELECT
                hadm_id,
                biomarker,
                AVG(valuenum) AS mean_val,
                MIN(valuenum) AS min_val,
                MAX(valuenum) AS max_val,
                CASE WHEN COUNT(*) > 1 THEN STDDEV_POP(valuenum) ELSE 0.0 END AS std_val,
                COUNT(*) AS count_val,
                AVG(CASE
                    WHEN ref_range_lower IS NOT NULL AND valuenum < ref_range_lower THEN 1.0
                    WHEN ref_range_upper IS NOT NULL AND valuenum > ref_range_upper THEN 1.0
                    ELSE 0.0
                END) AS abnormal_rate
            FROM lab_with_order
            GROUP BY hadm_id, biomarker
        ),
        first_last AS (
            SELECT hadm_id, biomarker, valuenum AS first_val
            FROM lab_with_order WHERE rn_asc = 1
        ),
        last_vals AS (
            SELECT hadm_id, biomarker, valuenum AS last_val
            FROM lab_with_order WHERE rn_desc = 1
        )
        SELECT
            a.hadm_id, a.biomarker,
            a.mean_val, a.min_val, a.max_val, a.std_val,
            a.count_val, a.abnormal_rate,
            f.first_val, lv.last_val
        FROM agg a
        JOIN first_last f ON a.hadm_id = f.hadm_id AND a.biomarker = f.biomarker
        JOIN last_vals lv ON a.hadm_id = lv.hadm_id AND a.biomarker = lv.biomarker
    """).fetchdf()

    if long_df.empty:
        return pd.DataFrame(columns=["hadm_id"])

    # Pivot long → wide
    rows: dict[int, dict] = {}
    for _, r in long_df.iterrows():
        hadm_id = int(r["hadm_id"])
        bio = r["biomarker"]
        if hadm_id not in rows:
            rows[hadm_id] = {"hadm_id": hadm_id}
        rows[hadm_id][f"{bio}_mean"] = r["mean_val"]
        rows[hadm_id][f"{bio}_min"] = r["min_val"]
        rows[hadm_id][f"{bio}_max"] = r["max_val"]
        rows[hadm_id][f"{bio}_std"] = r["std_val"]
        rows[hadm_id][f"{bio}_count"] = int(r["count_val"])
        rows[hadm_id][f"{bio}_first"] = r["first_val"]
        rows[hadm_id][f"{bio}_last"] = r["last_val"]
        rows[hadm_id][f"{bio}_abnormal_rate"] = r["abnormal_rate"]

    return pd.DataFrame(list(rows.values()))


def extract_vital_summary_sql(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Extract vital sign summary statistics for cohort admissions.

    Returns:
        DataFrame with columns: hadm_id, {vital}_{mean,min,max,std,cv,count}
    """
    long_df = conn.execute("""
        SELECT
            i.hadm_id,
            d.label AS vital_name,
            AVG(c.valuenum) AS mean_val,
            MIN(c.valuenum) AS min_val,
            MAX(c.valuenum) AS max_val,
            CASE WHEN COUNT(*) > 1 THEN STDDEV_POP(c.valuenum) ELSE 0.0 END AS std_val,
            COUNT(*) AS count_val
        FROM chartevents c
        JOIN d_items d ON c.itemid = d.itemid
        JOIN icustays i ON c.stay_id = i.stay_id
        WHERE c.valuenum IS NOT NULL
          AND i.hadm_id IN (SELECT hadm_id FROM cohort)
        GROUP BY i.hadm_id, d.label
    """).fetchdf()

    if long_df.empty:
        return pd.DataFrame(columns=["hadm_id"])

    # Pivot long → wide, compute CV
    rows: dict[int, dict] = {}
    for _, r in long_df.iterrows():
        hadm_id = int(r["hadm_id"])
        vital = r["vital_name"]
        if hadm_id not in rows:
            rows[hadm_id] = {"hadm_id": hadm_id}

        mean_val = r["mean_val"]
        std_val = r["std_val"]

        rows[hadm_id][f"{vital}_mean"] = mean_val
        rows[hadm_id][f"{vital}_min"] = r["min_val"]
        rows[hadm_id][f"{vital}_max"] = r["max_val"]
        rows[hadm_id][f"{vital}_std"] = std_val
        rows[hadm_id][f"{vital}_cv"] = std_val / mean_val if mean_val != 0 else 0.0
        rows[hadm_id][f"{vital}_count"] = int(r["count_val"])

    return pd.DataFrame(list(rows.values()))


def extract_medication_features_sql(
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Extract medication features for cohort admissions.

    Returns:
        DataFrame with columns: hadm_id, num_distinct_meds,
                                total_prescription_days, has_prescription
    """
    df = conn.execute("""
        SELECT
            c.hadm_id,
            COALESCE(p.num_distinct_meds, 0) AS num_distinct_meds,
            COALESCE(p.total_prescription_days, 0.0) AS total_prescription_days,
            CASE WHEN p.num_distinct_meds IS NOT NULL THEN 1 ELSE 0 END
                AS has_prescription
        FROM cohort c
        LEFT JOIN (
            SELECT
                hadm_id,
                COUNT(DISTINCT drug) AS num_distinct_meds,
                SUM(
                    DATE_DIFF('second', starttime, stoptime) / 86400.0
                ) AS total_prescription_days
            FROM prescriptions
            WHERE hadm_id IN (SELECT hadm_id FROM cohort)
              AND starttime IS NOT NULL
              AND stoptime IS NOT NULL
            GROUP BY hadm_id
        ) p ON c.hadm_id = p.hadm_id
    """).fetchdf()
    return df


def extract_diagnosis_features_sql(
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Extract diagnosis features for cohort admissions.

    Returns:
        DataFrame with columns: hadm_id, num_diagnoses, icd_chapter_*
    """
    df = conn.execute("""
        SELECT
            c.hadm_id,
            COALESCE(dx.num_diagnoses, 0) AS num_diagnoses,
            dx.primary_chapter
        FROM cohort c
        LEFT JOIN (
            SELECT
                d.hadm_id,
                COUNT(*) AS num_diagnoses,
                UPPER(LEFT(
                    MIN(CASE WHEN d.seq_num = 1 THEN d.icd_code END), 1
                )) AS primary_chapter
            FROM diagnoses_icd d
            WHERE d.hadm_id IN (SELECT hadm_id FROM cohort)
            GROUP BY d.hadm_id
        ) dx ON c.hadm_id = dx.hadm_id
    """).fetchdf()

    if df.empty:
        return pd.DataFrame(columns=["hadm_id", "num_diagnoses"])

    # Fill missing chapters
    df["primary_chapter"] = df["primary_chapter"].fillna("Unknown")

    # One-hot encode primary ICD chapter
    if len(df) > 0:
        dummies = pd.get_dummies(df["primary_chapter"], prefix="icd_chapter")
        df = pd.concat([df.drop("primary_chapter", axis=1), dummies], axis=1)

    return df
