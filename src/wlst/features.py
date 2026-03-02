"""48-hour feature extraction for WLST prediction.

Extracts tabular features from the first 48 hours of ICU data for use
in classical ML baselines (Logistic Regression, XGBoost).
"""

import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# GCS component itemids
GCS_EYE_ITEMID = 220739
GCS_VERBAL_ITEMID = 223900
GCS_MOTOR_ITEMID = 223901

# MAP itemids
MAP_ARTERIAL_ITEMID = 220052
MAP_CUFF_ITEMID = 220181

# Code status itemid (EXCLUDED from features)
CODE_STATUS_ITEMID = 223758

# Vasopressor drug names
VASOPRESSOR_DRUGS = {
    "norepinephrine", "phenylephrine", "vasopressin",
    "epinephrine", "dopamine",
}

# ICP medication keywords
ICP_MED_KEYWORDS = {"mannitol", "hypertonic saline", "levetiracetam"}

# Lab itemids
LAB_ITEMS = {
    50983: "sodium",
    50813: "lactate",
    50931: "glucose",
    51237: "inr",
    50912: "creatinine",
}

# Neurosurgery procedure itemids
NEUROSURGERY_ITEMIDS = {225752, 228114, 227190}  # Craniectomy, ICP monitor, EVD

# Ventilation itemid
VENTILATION_ITEMID = 225792

# O2 flow itemid (chartevents)
O2_FLOW_ITEMID = 223834


def extract_wlst_features(
    conn: duckdb.DuckDBPyConnection,
    labels_df: pd.DataFrame,
    observation_window_hours: int = 48,
    stage: str = "stage1",
    mappings_dir: Path | None = None,
) -> pd.DataFrame:
    """Extract 48-hour tabular features for WLST prediction.

    Args:
        conn: DuckDB connection with MIMIC tables loaded.
        labels_df: DataFrame from create_wlst_labels().
        observation_window_hours: Feature window in hours.
        stage: "stage1" or "stage2" (adds non-clinical features).
        mappings_dir: Path to data/mappings/ for ICD-to-AIS/Charlson.

    Returns:
        DataFrame with features, labels, and identifiers.
    """
    if len(labels_df) == 0:
        return pd.DataFrame()

    # Register cohort
    conn.execute(
        "CREATE OR REPLACE TEMP TABLE feat_cohort AS SELECT * FROM labels_df"
    )

    # Extract feature groups
    features = labels_df[["subject_id", "hadm_id", "stay_id", "wlst_label", "outcome_category"]].copy()

    demographics = _extract_demographics(conn)
    gcs_features = _extract_gcs_features(conn, observation_window_hours)
    gcs_bins = _extract_gcs_hourly_bins(conn, observation_window_hours)
    hemodynamics = _extract_hemodynamic_features(conn, observation_window_hours)
    lab_features = _extract_lab_features(conn, observation_window_hours)
    ventilation = _extract_ventilation_features(conn, observation_window_hours)
    o2_flow = _extract_o2_flow_features(conn, observation_window_hours)
    neurosurgery = _extract_neurosurgery_features(conn, observation_window_hours)
    icp_meds = _extract_icp_medication_features(conn, observation_window_hours)

    # Merge all feature groups
    for df in [demographics, gcs_features, gcs_bins, hemodynamics, lab_features,
               ventilation, o2_flow, neurosurgery, icp_meds]:
        if df is not None and len(df) > 0:
            features = features.merge(df, on="stay_id", how="left")

    # Stage 2: add non-clinical confounders
    if stage == "stage2":
        stage2 = _extract_stage2_features(conn, mappings_dir)
        if stage2 is not None and len(stage2) > 0:
            features = features.merge(stage2, on="hadm_id", how="left")

    # ICD-to-AIS head injury severity
    ais_features = _extract_ais_features(conn, mappings_dir)
    if ais_features is not None and len(ais_features) > 0:
        features = features.merge(ais_features, on="hadm_id", how="left")

    conn.execute("DROP TABLE IF EXISTS feat_cohort")

    logger.info(f"Extracted {features.shape[1]} features for {len(features)} patients")
    return features


def _extract_demographics(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    base = conn.execute("""
        SELECT
            c.stay_id,
            a.age,
            CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS is_male
        FROM feat_cohort c
        JOIN age a ON c.subject_id = a.subject_id AND c.hadm_id = a.hadm_id
        JOIN patients p ON c.subject_id = p.subject_id
    """).fetchdf()

    # BMI from OMR table (most recent before admission)
    try:
        bmi = conn.execute("""
            SELECT c.stay_id, AVG(CAST(o.result_value AS DOUBLE)) AS bmi
            FROM feat_cohort c
            JOIN admissions adm ON c.hadm_id = adm.hadm_id
            JOIN omr o ON c.subject_id = o.subject_id
            WHERE o.result_name LIKE '%BMI%'
              AND o.chartdate <= CAST(adm.admittime AS DATE)
              AND o.result_value IS NOT NULL
              AND TRY_CAST(o.result_value AS DOUBLE) IS NOT NULL
            GROUP BY c.stay_id
        """).fetchdf()
        if len(bmi) > 0:
            base = base.merge(bmi, on="stay_id", how="left")
    except duckdb.CatalogException:
        pass

    return base


def _extract_gcs_features(
    conn: duckdb.DuckDBPyConnection, window_hours: int,
) -> pd.DataFrame:
    """Extract GCS trajectory features from first 48h."""
    return conn.execute(f"""
        WITH gcs_raw AS (
            SELECT
                ce.stay_id, ce.charttime,
                MAX(CASE WHEN ce.itemid = {GCS_EYE_ITEMID} THEN ce.valuenum END) AS gcs_eye,
                MAX(CASE WHEN ce.itemid = {GCS_VERBAL_ITEMID} THEN ce.valuenum END) AS gcs_verbal,
                MAX(CASE WHEN ce.itemid = {GCS_MOTOR_ITEMID} THEN ce.valuenum END) AS gcs_motor
            FROM chartevents ce
            JOIN feat_cohort c ON ce.stay_id = c.stay_id
            WHERE ce.itemid IN ({GCS_EYE_ITEMID}, {GCS_VERBAL_ITEMID}, {GCS_MOTOR_ITEMID})
              AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '{window_hours}' HOUR
            GROUP BY ce.stay_id, ce.charttime
        ),
        gcs_with_total AS (
            SELECT *,
                COALESCE(gcs_eye, 0) + COALESCE(gcs_verbal, 0) + COALESCE(gcs_motor, 0) AS gcs_total
            FROM gcs_raw
        ),
        first_last AS (
            SELECT stay_id,
                FIRST_VALUE(gcs_motor) OVER w AS gcs_motor_first,
                LAST_VALUE(gcs_motor) OVER w AS gcs_motor_last,
                FIRST_VALUE(gcs_total) OVER w AS gcs_total_first
            FROM gcs_with_total
            WINDOW w AS (PARTITION BY stay_id ORDER BY charttime
                         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
        )
        SELECT
            g.stay_id,
            MIN(g.gcs_total) AS gcs_total_min,
            MAX(g.gcs_total) AS gcs_total_max,
            AVG(g.gcs_total) AS gcs_total_mean,
            MIN(g.gcs_motor) AS gcs_motor_min,
            MAX(g.gcs_motor) AS gcs_motor_max,
            ANY_VALUE(fl.gcs_motor_last) - ANY_VALUE(fl.gcs_motor_first) AS gcs_motor_delta,
            CASE WHEN ANY_VALUE(fl.gcs_motor_last) > ANY_VALUE(fl.gcs_motor_first) THEN 1 ELSE 0 END AS gcs_improving,
            ANY_VALUE(fl.gcs_total_first) AS gcs_admission_total,
            COUNT(*) AS gcs_measurement_count
        FROM gcs_with_total g
        JOIN (SELECT DISTINCT stay_id, gcs_motor_first, gcs_motor_last, gcs_total_first FROM first_last) fl
            ON g.stay_id = fl.stay_id
        GROUP BY g.stay_id
    """).fetchdf()


def _extract_gcs_hourly_bins(
    conn: duckdb.DuckDBPyConnection, window_hours: int,
) -> pd.DataFrame:
    """Extract GCS motor trajectory in 12-hour bins (0-12h, 12-24h, 24-36h, 36-48h)."""
    return conn.execute(f"""
        WITH gcs_binned AS (
            SELECT
                ce.stay_id,
                CASE
                    WHEN DATE_DIFF('hour', c.intime, ce.charttime) < 12 THEN '0_12h'
                    WHEN DATE_DIFF('hour', c.intime, ce.charttime) < 24 THEN '12_24h'
                    WHEN DATE_DIFF('hour', c.intime, ce.charttime) < 36 THEN '24_36h'
                    ELSE '36_48h'
                END AS time_bin,
                ce.valuenum AS gcs_motor
            FROM chartevents ce
            JOIN feat_cohort c ON ce.stay_id = c.stay_id
            WHERE ce.itemid = {GCS_MOTOR_ITEMID}
              AND ce.valuenum IS NOT NULL
              AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '{window_hours}' HOUR
        )
        SELECT
            stay_id,
            AVG(CASE WHEN time_bin = '0_12h' THEN gcs_motor END) AS gcs_motor_0_12h,
            AVG(CASE WHEN time_bin = '12_24h' THEN gcs_motor END) AS gcs_motor_12_24h,
            AVG(CASE WHEN time_bin = '24_36h' THEN gcs_motor END) AS gcs_motor_24_36h,
            AVG(CASE WHEN time_bin = '36_48h' THEN gcs_motor END) AS gcs_motor_36_48h
        FROM gcs_binned
        GROUP BY stay_id
    """).fetchdf()


def _extract_o2_flow_features(
    conn: duckdb.DuckDBPyConnection, window_hours: int,
) -> pd.DataFrame:
    """Extract O2 flow features (itemid 223834) from first 48h."""
    return conn.execute(f"""
        SELECT
            c.stay_id,
            CASE WHEN COUNT(ce.stay_id) > 0 THEN 1 ELSE 0 END AS o2_flow_any,
            AVG(ce.valuenum) AS o2_flow_mean,
            MAX(ce.valuenum) AS o2_flow_max
        FROM feat_cohort c
        LEFT JOIN chartevents ce
            ON c.stay_id = ce.stay_id
            AND ce.itemid = {O2_FLOW_ITEMID}
            AND ce.valuenum IS NOT NULL
            AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '{window_hours}' HOUR
        GROUP BY c.stay_id
    """).fetchdf()


def _extract_hemodynamic_features(
    conn: duckdb.DuckDBPyConnection, window_hours: int,
) -> pd.DataFrame:
    """Extract MAP and vasopressor features from first 48h."""
    map_features = conn.execute(f"""
        SELECT
            c.stay_id,
            AVG(ce.valuenum) AS map_mean,
            MIN(ce.valuenum) AS map_min,
            MAX(ce.valuenum) AS map_max,
            STDDEV(ce.valuenum) AS map_std
        FROM chartevents ce
        JOIN feat_cohort c ON ce.stay_id = c.stay_id
        WHERE ce.itemid IN ({MAP_ARTERIAL_ITEMID}, {MAP_CUFF_ITEMID})
          AND ce.valuenum IS NOT NULL
          AND ce.charttime BETWEEN c.intime AND c.intime + INTERVAL '{window_hours}' HOUR
        GROUP BY c.stay_id
    """).fetchdf()

    # Vasopressor features
    try:
        conn.execute("SELECT 1 FROM inputevents LIMIT 0")
        drug_list = ", ".join(f"'{d}'" for d in VASOPRESSOR_DRUGS)
        vaso_features = conn.execute(f"""
            SELECT
                c.stay_id,
                1 AS vasopressor_any,
                COUNT(DISTINCT d.label) AS vasopressor_drug_count,
                SUM(DATE_DIFF('minute', ie.starttime,
                    LEAST(ie.endtime, c.intime + INTERVAL '{window_hours}' HOUR)))
                    / 60.0 AS vasopressor_hours,
                MAX(ie.rate) AS vasopressor_max_rate
            FROM inputevents ie
            JOIN d_items d ON ie.itemid = d.itemid
            JOIN feat_cohort c ON ie.stay_id = c.stay_id
            WHERE LOWER(d.label) IN ({drug_list})
              AND ie.starttime BETWEEN c.intime AND c.intime + INTERVAL '{window_hours}' HOUR
            GROUP BY c.stay_id
        """).fetchdf()
    except duckdb.CatalogException:
        vaso_features = pd.DataFrame(columns=["stay_id", "vasopressor_any"])

    result = map_features
    if len(vaso_features) > 0:
        result = result.merge(vaso_features, on="stay_id", how="left")
    if "vasopressor_any" in result.columns:
        result["vasopressor_any"] = result["vasopressor_any"].fillna(0).astype(int)
    else:
        result["vasopressor_any"] = 0

    return result


def _extract_lab_features(
    conn: duckdb.DuckDBPyConnection, window_hours: int,
) -> pd.DataFrame:
    """Extract lab features from first 48h for WLST-relevant labs."""
    itemid_list = ", ".join(str(i) for i in LAB_ITEMS.keys())

    raw = conn.execute(f"""
        SELECT
            c.stay_id, l.itemid, l.valuenum,
            ROW_NUMBER() OVER (PARTITION BY c.stay_id, l.itemid ORDER BY l.charttime) AS rn_first,
            ROW_NUMBER() OVER (PARTITION BY c.stay_id, l.itemid ORDER BY l.charttime DESC) AS rn_last
        FROM labevents l
        JOIN feat_cohort c ON l.hadm_id = c.hadm_id
        WHERE l.itemid IN ({itemid_list})
          AND l.valuenum IS NOT NULL
          AND l.charttime BETWEEN c.intime AND c.intime + INTERVAL '{window_hours}' HOUR
    """).fetchdf()

    if len(raw) == 0:
        return pd.DataFrame(columns=["stay_id"])

    # Aggregate per stay x lab
    agg = raw.groupby(["stay_id", "itemid"]).agg(
        lab_min=("valuenum", "min"),
        lab_max=("valuenum", "max"),
        lab_mean=("valuenum", "mean"),
    ).reset_index()

    # First and last values
    first_vals = raw[raw["rn_first"] == 1][["stay_id", "itemid", "valuenum"]].rename(
        columns={"valuenum": "lab_first"}
    )
    last_vals = raw[raw["rn_last"] == 1][["stay_id", "itemid", "valuenum"]].rename(
        columns={"valuenum": "lab_last"}
    )
    agg = agg.merge(first_vals, on=["stay_id", "itemid"], how="left")
    agg = agg.merge(last_vals, on=["stay_id", "itemid"], how="left")

    # Pivot to wide format
    item_name_map = {v: k for k, v in LAB_ITEMS.items()}
    agg["lab_name"] = agg["itemid"].map(LAB_ITEMS)

    result = pd.DataFrame({"stay_id": raw["stay_id"].unique()})
    for itemid, name in LAB_ITEMS.items():
        sub = agg[agg["itemid"] == itemid]
        for stat in ["min", "max", "mean", "first", "last"]:
            col = f"lab_{stat}"
            pivot = sub[["stay_id", col]].rename(columns={col: f"{name}_{stat}"})
            result = result.merge(pivot, on="stay_id", how="left")

    return result


def _extract_ventilation_features(
    conn: duckdb.DuckDBPyConnection, window_hours: int,
) -> pd.DataFrame:
    """Extract ventilation features from first 48h."""
    try:
        conn.execute("SELECT 1 FROM procedureevents LIMIT 0")
    except duckdb.CatalogException:
        return pd.DataFrame(columns=["stay_id", "vent_initiated"])

    return conn.execute(f"""
        SELECT
            c.stay_id,
            CASE WHEN COUNT(pe.stay_id) > 0 THEN 1 ELSE 0 END AS vent_initiated,
            MIN(DATE_DIFF('hour', c.intime, pe.starttime)) AS hours_to_vent,
            SUM(DATE_DIFF('minute', pe.starttime,
                LEAST(pe.endtime, c.intime + INTERVAL '{window_hours}' HOUR)))
                / 60.0 AS vent_duration_48h
        FROM feat_cohort c
        LEFT JOIN procedureevents pe
            ON c.stay_id = pe.stay_id
            AND pe.itemid = {VENTILATION_ITEMID}
            AND pe.starttime BETWEEN c.intime AND c.intime + INTERVAL '{window_hours}' HOUR
        GROUP BY c.stay_id
    """).fetchdf()


def _extract_neurosurgery_features(
    conn: duckdb.DuckDBPyConnection, window_hours: int,
) -> pd.DataFrame:
    """Extract neurosurgery procedure features from first 48h."""
    try:
        conn.execute("SELECT 1 FROM procedureevents LIMIT 0")
    except duckdb.CatalogException:
        return pd.DataFrame(columns=["stay_id", "any_neurosurgery"])

    itemid_list = ", ".join(str(i) for i in NEUROSURGERY_ITEMIDS)
    return conn.execute(f"""
        SELECT
            c.stay_id,
            CASE WHEN COUNT(pe.stay_id) > 0 THEN 1 ELSE 0 END AS any_neurosurgery,
            MAX(CASE WHEN pe.itemid = 225752 THEN 1 ELSE 0 END) AS craniectomy,
            MAX(CASE WHEN pe.itemid = 228114 THEN 1 ELSE 0 END) AS icp_monitor,
            MAX(CASE WHEN pe.itemid = 227190 THEN 1 ELSE 0 END) AS evd_placed
        FROM feat_cohort c
        LEFT JOIN procedureevents pe
            ON c.stay_id = pe.stay_id
            AND pe.itemid IN ({itemid_list})
            AND pe.starttime BETWEEN c.intime AND c.intime + INTERVAL '{window_hours}' HOUR
        GROUP BY c.stay_id
    """).fetchdf()


def _extract_icp_medication_features(
    conn: duckdb.DuckDBPyConnection, window_hours: int,
) -> pd.DataFrame:
    """Extract ICP medication features from first 48h."""
    try:
        conn.execute("SELECT 1 FROM inputevents LIMIT 0")
    except duckdb.CatalogException:
        return pd.DataFrame(columns=["stay_id", "icp_med_count"])

    like_clauses = " OR ".join(
        f"LOWER(d.label) LIKE '%{kw}%'" for kw in ICP_MED_KEYWORDS
    )
    result = conn.execute(f"""
        SELECT
            c.stay_id,
            COUNT(matched.itemid) AS icp_med_count,
            MAX(CASE WHEN LOWER(matched.label) LIKE '%mannitol%' THEN 1 ELSE 0 END) AS mannitol_given,
            MAX(CASE WHEN LOWER(matched.label) LIKE '%hypertonic saline%' THEN 1 ELSE 0 END) AS hypertonic_saline_given,
            MAX(CASE WHEN LOWER(matched.label) LIKE '%levetiracetam%' THEN 1 ELSE 0 END) AS levetiracetam_given
        FROM feat_cohort c
        LEFT JOIN (
            SELECT ie.stay_id, ie.starttime, d.label, d.itemid
            FROM inputevents ie
            JOIN d_items d ON ie.itemid = d.itemid
            WHERE ({like_clauses})
        ) matched ON c.stay_id = matched.stay_id
            AND matched.starttime BETWEEN c.intime AND c.intime + INTERVAL '{window_hours}' HOUR
        GROUP BY c.stay_id
    """).fetchdf()

    # Also check prescriptions table for ICP medications
    try:
        conn.execute("SELECT 1 FROM prescriptions LIMIT 0")
        rx_like = " OR ".join(
            f"LOWER(rx.drug) LIKE '%{kw}%'" for kw in ICP_MED_KEYWORDS
        )
        rx_df = conn.execute(f"""
            SELECT
                c.stay_id,
                MAX(CASE WHEN LOWER(rx.drug) LIKE '%mannitol%' THEN 1 ELSE 0 END) AS rx_mannitol,
                MAX(CASE WHEN LOWER(rx.drug) LIKE '%hypertonic saline%' THEN 1 ELSE 0 END) AS rx_hypertonic_saline,
                MAX(CASE WHEN LOWER(rx.drug) LIKE '%levetiracetam%' THEN 1 ELSE 0 END) AS rx_levetiracetam
            FROM feat_cohort c
            JOIN prescriptions rx ON c.hadm_id = rx.hadm_id
            WHERE ({rx_like})
              AND rx.starttime BETWEEN c.intime AND c.intime + INTERVAL '{window_hours}' HOUR
            GROUP BY c.stay_id
        """).fetchdf()
        if len(rx_df) > 0:
            result = result.merge(rx_df, on="stay_id", how="left")
            # Combine: med given from either inputevents or prescriptions
            for med in ["mannitol", "hypertonic_saline", "levetiracetam"]:
                rx_col = f"rx_{med}"
                ie_col = f"{med}_given"
                if rx_col in result.columns and ie_col in result.columns:
                    result[ie_col] = (result[ie_col].fillna(0) + result[rx_col].fillna(0)).clip(upper=1).astype(int)
                    result.drop(columns=[rx_col], inplace=True)
    except duckdb.CatalogException:
        pass

    return result


def _extract_stage2_features(
    conn: duckdb.DuckDBPyConnection, mappings_dir: Path | None,
) -> pd.DataFrame:
    """Extract Stage 2 non-clinical confounder features."""
    # Language
    lang_df = conn.execute("""
        SELECT c.hadm_id,
            CASE WHEN adm.language != 'ENGLISH' THEN 1 ELSE 0 END AS language_barrier
        FROM feat_cohort c
        JOIN admissions adm ON c.hadm_id = adm.hadm_id
    """).fetchdf()

    # Hospital service (encode as is_neuro_service)
    try:
        service_df = conn.execute("""
            SELECT DISTINCT ON (c.hadm_id)
                c.hadm_id,
                CASE WHEN s.curr_service IN ('NSURG', 'NB', 'NMED') THEN 1 ELSE 0 END AS is_neuro_service
            FROM feat_cohort c
            JOIN services s ON c.hadm_id = s.hadm_id
            ORDER BY c.hadm_id, s.transfertime
        """).fetchdf()
        lang_df = lang_df.merge(service_df, on="hadm_id", how="left")
    except duckdb.CatalogException:
        pass

    # Transfer count
    try:
        transfer_df = conn.execute("""
            SELECT c.hadm_id, COUNT(*) AS transfer_count
            FROM feat_cohort c
            JOIN transfers t ON c.hadm_id = t.hadm_id
            GROUP BY c.hadm_id
        """).fetchdf()
        lang_df = lang_df.merge(transfer_df, on="hadm_id", how="left")
    except duckdb.CatalogException:
        pass

    # Charlson Comorbidity Index
    charlson_df = _compute_charlson_index(conn, mappings_dir)
    if charlson_df is not None and len(charlson_df) > 0:
        lang_df = lang_df.merge(charlson_df, on="hadm_id", how="left")

    return lang_df


def _extract_ais_features(
    conn: duckdb.DuckDBPyConnection, mappings_dir: Path | None,
) -> pd.DataFrame | None:
    """Map ICD-10 TBI codes to AIS head region scores."""
    if mappings_dir is None:
        return None

    ais_path = mappings_dir / "icd10_to_ais_head.json"
    if not ais_path.exists():
        logger.warning(f"AIS mapping file not found: {ais_path}")
        return None

    with open(ais_path) as f:
        ais_map = json.load(f)

    # Get TBI diagnoses for cohort
    dx_df = conn.execute("""
        SELECT c.hadm_id, di.icd_code
        FROM feat_cohort c
        JOIN diagnoses_icd di ON c.hadm_id = di.hadm_id
        WHERE di.icd_version = 10 AND di.icd_code LIKE 'S06%'
    """).fetchdf()

    if len(dx_df) == 0:
        return None

    # Map ICD codes to AIS scores (use prefix matching)
    def _get_ais(icd_code: str) -> int:
        # Try exact match first, then progressively shorter prefixes
        for length in range(len(icd_code), 2, -1):
            prefix = icd_code[:length]
            if prefix in ais_map:
                score = ais_map[prefix]
                # Handle range values like "4-5" by taking the max
                if isinstance(score, str) and "-" in score:
                    return int(score.split("-")[-1])
                return int(score)
        return 0

    dx_df["ais_score"] = dx_df["icd_code"].apply(_get_ais)

    # Take max AIS score per admission
    result = dx_df.groupby("hadm_id").agg(
        head_ais_score=("ais_score", "max"),
    ).reset_index()

    return result


def _compute_charlson_index(
    conn: duckdb.DuckDBPyConnection, mappings_dir: Path | None,
) -> pd.DataFrame | None:
    """Compute Charlson Comorbidity Index from ICD-10 codes."""
    if mappings_dir is None:
        return None

    charlson_path = mappings_dir / "icd10_to_charlson.json"
    if not charlson_path.exists():
        logger.warning(f"Charlson mapping file not found: {charlson_path}")
        return None

    with open(charlson_path) as f:
        charlson_map = json.load(f)

    # Get all diagnoses for cohort
    dx_df = conn.execute("""
        SELECT c.hadm_id, di.icd_code
        FROM feat_cohort c
        JOIN diagnoses_icd di ON c.hadm_id = di.hadm_id
        WHERE di.icd_version = 10
    """).fetchdf()

    if len(dx_df) == 0:
        return None

    # Map each diagnosis to Charlson components
    results = []
    for hadm_id, group in dx_df.groupby("hadm_id"):
        components = {}
        for icd_code in group["icd_code"]:
            for category, info in charlson_map.items():
                prefixes = info.get("icd10_prefixes", [])
                weight = info.get("weight", 1)
                for prefix in prefixes:
                    if icd_code.startswith(prefix):
                        components[category] = weight
                        break
        charlson_score = sum(components.values())
        row = {"hadm_id": hadm_id, "charlson_index": charlson_score}
        # Add individual component flags
        for category in charlson_map:
            row[f"charlson_{category}"] = 1 if category in components else 0
        results.append(row)

    return pd.DataFrame(results)
