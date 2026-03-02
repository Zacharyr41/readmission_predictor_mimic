"""TBI cohort selection and WLST label derivation.

Selects severe TBI patients (GCS <= 8) from MIMIC-IV and derives
WLST labels from code status changes and discharge disposition.
"""

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def select_tbi_cohort(
    conn: duckdb.DuckDBPyConnection,
    icd_prefixes: list[str] | None = None,
    gcs_threshold: int = 8,
    icu_types: list[str] | None = None,
    observation_window_hours: int = 48,
    patients_limit: int = 0,
) -> pd.DataFrame:
    """Select severe TBI cohort from MIMIC-IV.

    Inclusion criteria:
    - ICD-10 S06.x TBI diagnosis (any seq_num)
    - First ICU stay per admission
    - ICU stay > 24 hours
    - Admitted to neuro/trauma ICU
    - Age 18-89
    - Admission GCS <= threshold (within first 24h)

    Args:
        conn: DuckDB connection with MIMIC tables loaded.
        icd_prefixes: ICD-10 prefixes for TBI (default: ["S06"]).
        gcs_threshold: Maximum GCS total for inclusion (default: 8).
        icu_types: ICU care unit types to include.
        observation_window_hours: Feature window in hours (stored for reference).
        patients_limit: Max patients (0 = no limit).

    Returns:
        DataFrame with columns: subject_id, hadm_id, stay_id, intime, outtime,
        first_careunit, gcs_total, gcs_eye, gcs_verbal, gcs_motor.
    """
    if icd_prefixes is None:
        icd_prefixes = ["S06"]
    if icu_types is None:
        icu_types = [
            "Neuro Stepdown",
            "Neuro Surgical Intensive Care Unit (Neuro SICU)",
            "Trauma SICU (TSICU)",
        ]

    # Build ICD filter
    icd_parts = [f"d.icd_code LIKE '{p}%'" for p in icd_prefixes]
    icd_filter = " OR ".join(icd_parts)

    # Build ICU type filter
    icu_type_list = ", ".join(f"'{t}'" for t in icu_types)

    limit_clause = f"LIMIT {patients_limit}" if patients_limit > 0 else ""

    query = f"""
        WITH tbi_diagnoses AS (
            SELECT DISTINCT hadm_id
            FROM diagnoses_icd d
            WHERE d.icd_version = 10
              AND d.seq_num = 1
              AND ({icd_filter})
        ),
        ranked_icu_stays AS (
            SELECT
                i.subject_id, i.hadm_id, i.stay_id,
                i.intime, i.outtime, i.first_careunit,
                DATE_DIFF('hour', i.intime, i.outtime) AS icu_hours,
                ROW_NUMBER() OVER (
                    PARTITION BY i.hadm_id ORDER BY i.intime
                ) AS stay_rank
            FROM icustays i
        ),
        eligible_stays AS (
            SELECT *
            FROM ranked_icu_stays
            WHERE stay_rank = 1
              AND icu_hours > 24
              AND first_careunit IN ({icu_type_list})
        ),
        age_filtered AS (
            SELECT es.*
            FROM eligible_stays es
            JOIN tbi_diagnoses td ON es.hadm_id = td.hadm_id
            JOIN age a ON es.subject_id = a.subject_id AND es.hadm_id = a.hadm_id
            WHERE a.age BETWEEN 18 AND 89
        ),
        admission_gcs AS (
            SELECT
                ce.stay_id,
                MIN(CASE WHEN ce.itemid = 220739 THEN ce.valuenum END) AS gcs_eye,
                MIN(CASE WHEN ce.itemid = 223900 THEN ce.valuenum END) AS gcs_verbal,
                MIN(CASE WHEN ce.itemid = 223901 THEN ce.valuenum END) AS gcs_motor,
                COALESCE(MIN(CASE WHEN ce.itemid = 220739 THEN ce.valuenum END), 0)
                + COALESCE(MIN(CASE WHEN ce.itemid = 223900 THEN ce.valuenum END), 0)
                + COALESCE(MIN(CASE WHEN ce.itemid = 223901 THEN ce.valuenum END), 0)
                    AS gcs_total
            FROM chartevents ce
            JOIN age_filtered af ON ce.stay_id = af.stay_id
            WHERE ce.itemid IN (220739, 223900, 223901)
              AND ce.charttime BETWEEN af.intime AND af.intime + INTERVAL '24' HOUR
            GROUP BY ce.stay_id
        )
        SELECT
            af.subject_id, af.hadm_id, af.stay_id,
            af.intime, af.outtime, af.first_careunit,
            g.gcs_total, g.gcs_eye, g.gcs_verbal, g.gcs_motor
        FROM age_filtered af
        JOIN admission_gcs g ON af.stay_id = g.stay_id
        WHERE g.gcs_total <= {gcs_threshold}
          AND g.gcs_total > 0
        ORDER BY af.subject_id, af.hadm_id
        {limit_clause}
    """

    df = conn.execute(query).fetchdf()
    logger.info(f"Selected TBI cohort: {len(df)} patients (GCS <= {gcs_threshold})")
    return df


def create_wlst_labels(
    conn: duckdb.DuckDBPyConnection,
    cohort_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive WLST labels from code status changes and discharge disposition.

    WLST label = 1 if:
    - Code status changed from "Full code" to any limited code (DNR, DNI, CMO, etc.)
      at any point during the ICU stay, OR
    - Discharge disposition is HOSPICE

    Features are restricted to 48h but the label reflects the entire stay.

    Args:
        conn: DuckDB connection with chartevents and admissions tables.
        cohort_df: DataFrame from select_tbi_cohort().

    Returns:
        DataFrame with WLST labels and outcome categories.
    """
    # Register cohort as temp table
    conn.execute(
        "CREATE OR REPLACE TEMP TABLE wlst_cohort AS SELECT * FROM cohort_df"
    )

    query = """
        WITH code_status_events AS (
            SELECT
                ce.subject_id, ce.hadm_id, ce.stay_id,
                ce.charttime AS code_status_time,
                ce.value AS code_status_value,
                ROW_NUMBER() OVER (
                    PARTITION BY ce.stay_id ORDER BY ce.charttime
                ) AS event_rank
            FROM chartevents ce
            JOIN wlst_cohort c ON ce.stay_id = c.stay_id
            WHERE ce.itemid = 223758
        ),
        first_non_full_code AS (
            SELECT DISTINCT ON (stay_id)
                stay_id, code_status_time, code_status_value
            FROM code_status_events
            WHERE code_status_value NOT IN ('Full code', 'Full Code')
            ORDER BY stay_id, event_rank
        ),
        death_info AS (
            SELECT hadm_id, hospital_expire_flag, deathtime, discharge_location
            FROM admissions
        )
        SELECT
            c.subject_id, c.hadm_id, c.stay_id, c.intime,
            c.gcs_total, c.gcs_eye, c.gcs_verbal, c.gcs_motor,
            fnfc.code_status_time,
            fnfc.code_status_value,
            d.hospital_expire_flag,
            d.deathtime,
            d.discharge_location,
            DATE_DIFF('hour', c.intime, fnfc.code_status_time)
                AS hours_to_code_change,
            CASE
                WHEN fnfc.code_status_value IS NOT NULL THEN 1
                WHEN d.discharge_location = 'HOSPICE' THEN 1
                ELSE 0
            END AS wlst_label,
            CASE
                WHEN fnfc.code_status_value IN ('Comfort measures only', 'CMO')
                     AND d.hospital_expire_flag = 1 THEN 'CMO_death'
                WHEN fnfc.code_status_value IN ('DNR', 'DNI', 'DNR / DNI')
                     AND d.hospital_expire_flag = 1 THEN 'DNR_death'
                WHEN fnfc.code_status_value IS NOT NULL
                     AND d.hospital_expire_flag = 0 THEN 'limited_code_survived'
                WHEN fnfc.code_status_value IS NULL
                     AND d.hospital_expire_flag = 1 THEN 'full_code_death'
                WHEN fnfc.code_status_value IS NULL
                     AND d.hospital_expire_flag = 0 THEN 'full_code_survived'
                WHEN d.discharge_location = 'HOSPICE' THEN 'hospice'
                ELSE 'other'
            END AS outcome_category
        FROM wlst_cohort c
        LEFT JOIN first_non_full_code fnfc ON c.stay_id = fnfc.stay_id
        LEFT JOIN death_info d ON c.hadm_id = d.hadm_id
        ORDER BY c.subject_id, c.hadm_id
    """

    df = conn.execute(query).fetchdf()

    # Log label distribution
    label_counts = df["wlst_label"].value_counts()
    logger.info(f"WLST label distribution:\n{label_counts.to_string()}")

    outcome_counts = df["outcome_category"].value_counts()
    logger.info(f"Outcome categories:\n{outcome_counts.to_string()}")

    if len(df) > 0:
        wlst_rate = df["wlst_label"].mean()
        logger.info(f"WLST rate: {wlst_rate:.1%} ({label_counts.get(1, 0)}/{len(df)})")

    conn.execute("DROP TABLE IF EXISTS wlst_cohort")
    return df


def generate_cohort_summary(labels_df: pd.DataFrame) -> str:
    """Generate a markdown summary of the WLST cohort.

    Args:
        labels_df: DataFrame from create_wlst_labels().

    Returns:
        Markdown-formatted summary string.
    """
    n = len(labels_df)
    if n == 0:
        return "# WLST Cohort Summary\n\nNo patients in cohort.\n"

    wlst_pos = labels_df["wlst_label"].sum()
    wlst_neg = n - wlst_pos
    mortality = labels_df["hospital_expire_flag"].sum() if "hospital_expire_flag" in labels_df.columns else "N/A"

    lines = [
        "# WLST Cohort Summary\n",
        f"## Cohort Size: {n} patients\n",
        "## WLST Label Distribution",
        f"- WLST positive (label=1): {wlst_pos} ({wlst_pos/n:.1%})",
        f"- WLST negative (label=0): {wlst_neg} ({wlst_neg/n:.1%})\n",
        f"## Mortality: {mortality} ({mortality/n:.1%})\n" if isinstance(mortality, (int, float)) else "",
        "## Outcome Categories",
    ]

    outcome_counts = labels_df["outcome_category"].value_counts()
    for cat, count in outcome_counts.items():
        lines.append(f"- {cat}: {count} ({count/n:.1%})")

    lines.append("\n## GCS Distribution")
    if "gcs_total" in labels_df.columns:
        gcs_stats = labels_df["gcs_total"].describe()
        lines.append(f"- Mean: {gcs_stats['mean']:.1f}")
        lines.append(f"- Median: {gcs_stats['50%']:.1f}")
        lines.append(f"- Range: {gcs_stats['min']:.0f} - {gcs_stats['max']:.0f}")

    if "hours_to_code_change" in labels_df.columns:
        code_changes = labels_df["hours_to_code_change"].dropna()
        if len(code_changes) > 0:
            lines.append("\n## Time to Code Status Change")
            lines.append(f"- Median: {code_changes.median():.1f} hours")
            lines.append(f"- Within 48h: {(code_changes <= 48).sum()} ({(code_changes <= 48).mean():.1%})")
            lines.append(f"- After 48h: {(code_changes > 48).sum()} ({(code_changes > 48).mean():.1%})")

    return "\n".join(lines) + "\n"
