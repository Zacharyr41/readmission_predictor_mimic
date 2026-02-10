"""Tabular feature extraction from RDF graphs.

This module extracts tabular features from the clinical knowledge graph,
including demographics, stay characteristics, lab summaries, vitals,
medications, and diagnoses.

Note on potential data leakage:
- Diagnosis features are extracted from discharge diagnoses (diagnoses_icd).
  These are coded at discharge and may contain information about the full
  hospital course. Consider excluding or carefully validating diagnosis
  features in predictive models.
"""

import pandas as pd
import numpy as np
from rdflib import Graph

from src.graph_construction.ontology import MIMIC_NS, TIME_NS


def extract_demographics(graph: Graph) -> pd.DataFrame:
    """Extract demographic features for each admission.

    Args:
        graph: RDF graph containing patient and admission data.

    Returns:
        DataFrame with columns: hadm_id, age, gender_M, gender_F
    """
    query = """
    SELECT ?hadmId ?age ?gender
    WHERE {
        ?patient rdf:type mimic:Patient ;
                 mimic:hasAge ?age ;
                 mimic:hasGender ?gender ;
                 mimic:hasAdmission ?admission .
        ?admission mimic:hasAdmissionId ?hadmId .
    }
    """

    results = list(graph.query(query))

    data = []
    for row in results:
        hadm_id = int(row[0])
        age = int(row[1])
        gender = str(row[2])

        data.append({
            "hadm_id": hadm_id,
            "age": age,
            "gender_M": 1 if gender == "M" else 0,
            "gender_F": 1 if gender == "F" else 0,
        })

    return pd.DataFrame(data)


def extract_stay_features(graph: Graph) -> pd.DataFrame:
    """Extract ICU stay features for each admission.

    Args:
        graph: RDF graph containing admission and ICU stay data.

    Returns:
        DataFrame with columns: hadm_id, icu_los_hours, num_icu_days,
                                admission_type_* (one-hot encoded)
    """
    # Query ICU stay data
    query = """
    SELECT ?hadmId ?admissionType ?losDays
    WHERE {
        ?admission rdf:type mimic:HospitalAdmission ;
                   mimic:hasAdmissionId ?hadmId ;
                   mimic:hasAdmissionType ?admissionType ;
                   mimic:containsICUStay ?icuStay .
        ?icuStay time:hasDuration ?duration .
        ?duration time:numericDuration ?losDays .
    }
    """

    results = list(graph.query(query))

    data = []
    for row in results:
        hadm_id = int(row[0])
        admission_type = str(row[1])
        los_days = float(row[2])

        data.append({
            "hadm_id": hadm_id,
            "admission_type": admission_type,
            "icu_los_hours": los_days * 24.0,
        })

    df = pd.DataFrame(data)

    if df.empty:
        return pd.DataFrame(columns=["hadm_id", "icu_los_hours", "num_icu_days"])

    # Count ICU days per admission
    day_query = """
    SELECT ?hadmId (COUNT(?icuDay) AS ?numDays)
    WHERE {
        ?admission rdf:type mimic:HospitalAdmission ;
                   mimic:hasAdmissionId ?hadmId ;
                   mimic:containsICUStay ?icuStay .
        ?icuStay mimic:hasICUDay ?icuDay .
    }
    GROUP BY ?hadmId
    """

    day_results = list(graph.query(day_query))
    day_counts = {int(row[0]): int(row[1]) for row in day_results}

    df["num_icu_days"] = df["hadm_id"].map(day_counts).fillna(0).astype(int)

    # One-hot encode admission type
    if "admission_type" in df.columns:
        admission_dummies = pd.get_dummies(df["admission_type"], prefix="admission_type")
        df = pd.concat([df.drop("admission_type", axis=1), admission_dummies], axis=1)

    return df


def extract_lab_summary(graph: Graph) -> pd.DataFrame:
    """Extract lab (biomarker) summary statistics for each admission.

    Args:
        graph: RDF graph containing biomarker event data.

    Returns:
        DataFrame with columns: hadm_id, {biomarker}_mean, {biomarker}_min,
                                {biomarker}_max, {biomarker}_std, {biomarker}_count,
                                {biomarker}_first, {biomarker}_last, {biomarker}_abnormal_rate
    """
    # SPARQL query to retrieve all biomarker events with their values and reference ranges.
    # The query joins:
    # - HospitalAdmission to get hadm_id
    # - ICUStay to link admission to events
    # - BioMarkerEvent to get lab test results
    # OPTIONAL clauses handle missing reference ranges (not all labs have them).
    # Results are ordered chronologically within each admission/biomarker for
    # first/last value extraction.
    query = """
    SELECT ?hadmId ?biomarkerType ?value ?refLower ?refUpper ?charttime
    WHERE {
        ?admission rdf:type mimic:HospitalAdmission ;
                   mimic:hasAdmissionId ?hadmId ;
                   mimic:containsICUStay ?icuStay .
        ?event rdf:type mimic:BioMarkerEvent ;
               mimic:associatedWithICUStay ?icuStay ;
               mimic:hasBiomarkerType ?biomarkerType ;
               mimic:hasValue ?value ;
               time:inXSDDateTimeStamp ?charttime .
        OPTIONAL { ?event mimic:hasRefRangeLower ?refLower . }
        OPTIONAL { ?event mimic:hasRefRangeUpper ?refUpper . }
    }
    ORDER BY ?hadmId ?biomarkerType ?charttime
    """

    results = list(graph.query(query))

    if not results:
        return pd.DataFrame(columns=["hadm_id"])

    # Group results by hadm_id and biomarker type
    data = {}
    for row in results:
        hadm_id = int(row[0])
        biomarker_type = str(row[1])
        value = float(row[2])
        ref_lower = float(row[3]) if row[3] is not None else None
        ref_upper = float(row[4]) if row[4] is not None else None

        key = (hadm_id, biomarker_type)
        if key not in data:
            data[key] = {"values": [], "abnormal": []}

        data[key]["values"].append(value)

        # Check if abnormal
        is_abnormal = 0
        if ref_lower is not None and value < ref_lower:
            is_abnormal = 1
        elif ref_upper is not None and value > ref_upper:
            is_abnormal = 1
        data[key]["abnormal"].append(is_abnormal)

    # Compute aggregates
    rows = {}
    for (hadm_id, biomarker_type), vals in data.items():
        if hadm_id not in rows:
            rows[hadm_id] = {"hadm_id": hadm_id}

        values = vals["values"]
        abnormal = vals["abnormal"]

        rows[hadm_id][f"{biomarker_type}_mean"] = np.mean(values)
        rows[hadm_id][f"{biomarker_type}_min"] = np.min(values)
        rows[hadm_id][f"{biomarker_type}_max"] = np.max(values)
        rows[hadm_id][f"{biomarker_type}_std"] = np.std(values) if len(values) > 1 else 0.0
        rows[hadm_id][f"{biomarker_type}_count"] = len(values)
        rows[hadm_id][f"{biomarker_type}_first"] = values[0]
        rows[hadm_id][f"{biomarker_type}_last"] = values[-1]
        rows[hadm_id][f"{biomarker_type}_abnormal_rate"] = np.mean(abnormal)

    return pd.DataFrame(list(rows.values()))


def extract_vital_summary(graph: Graph) -> pd.DataFrame:
    """Extract vital sign summary statistics for each admission.

    Args:
        graph: RDF graph containing clinical sign event data.

    Returns:
        DataFrame with columns: hadm_id, {vital}_mean, {vital}_min,
                                {vital}_max, {vital}_std, {vital}_cv, {vital}_count
    """
    query = """
    SELECT ?hadmId ?vitalName ?value ?charttime
    WHERE {
        ?admission rdf:type mimic:HospitalAdmission ;
                   mimic:hasAdmissionId ?hadmId ;
                   mimic:containsICUStay ?icuStay .
        ?event rdf:type mimic:ClinicalSignEvent ;
               mimic:associatedWithICUStay ?icuStay ;
               mimic:hasClinicalSignName ?vitalName ;
               mimic:hasValue ?value ;
               time:inXSDDateTimeStamp ?charttime .
    }
    ORDER BY ?hadmId ?vitalName ?charttime
    """

    results = list(graph.query(query))

    if not results:
        return pd.DataFrame(columns=["hadm_id"])

    # Group results by hadm_id and vital type
    data = {}
    for row in results:
        hadm_id = int(row[0])
        vital_name = str(row[1])
        value = float(row[2])

        key = (hadm_id, vital_name)
        if key not in data:
            data[key] = []
        data[key].append(value)

    # Compute aggregates
    rows = {}
    for (hadm_id, vital_name), values in data.items():
        if hadm_id not in rows:
            rows[hadm_id] = {"hadm_id": hadm_id}

        mean_val = np.mean(values)
        std_val = np.std(values) if len(values) > 1 else 0.0

        rows[hadm_id][f"{vital_name}_mean"] = mean_val
        rows[hadm_id][f"{vital_name}_min"] = np.min(values)
        rows[hadm_id][f"{vital_name}_max"] = np.max(values)
        rows[hadm_id][f"{vital_name}_std"] = std_val
        rows[hadm_id][f"{vital_name}_cv"] = std_val / mean_val if mean_val != 0 else 0.0
        rows[hadm_id][f"{vital_name}_count"] = len(values)

    return pd.DataFrame(list(rows.values()))


def extract_medication_features(graph: Graph) -> pd.DataFrame:
    """Extract medication features for each admission.

    Args:
        graph: RDF graph containing prescription event data.

    Returns:
        DataFrame with columns: hadm_id, num_distinct_meds, total_prescription_days, has_prescription
    """
    # First get all admissions
    all_admissions_query = """
    SELECT DISTINCT ?hadmId
    WHERE {
        ?admission rdf:type mimic:HospitalAdmission ;
                   mimic:hasAdmissionId ?hadmId .
    }
    """

    all_admissions = list(graph.query(all_admissions_query))
    hadm_ids = [int(row[0]) for row in all_admissions]

    # Query prescription events
    query = """
    SELECT ?hadmId ?drugName ?startTime ?endTime
    WHERE {
        ?admission rdf:type mimic:HospitalAdmission ;
                   mimic:hasAdmissionId ?hadmId ;
                   mimic:containsICUStay ?icuStay .
        ?event rdf:type mimic:PrescriptionEvent ;
               mimic:associatedWithICUStay ?icuStay ;
               mimic:hasDrugName ?drugName ;
               time:hasBeginning ?begin ;
               time:hasEnd ?end .
        ?begin time:inXSDDateTimeStamp ?startTime .
        ?end time:inXSDDateTimeStamp ?endTime .
    }
    """

    results = list(graph.query(query))

    # Process prescription data
    abx_data = {}
    for row in results:
        hadm_id = int(row[0])
        drug_name = str(row[1])
        start_str = str(row[2])
        end_str = str(row[3])

        # Parse timestamps
        if start_str.endswith("Z"):
            start_str = start_str[:-1]
        if end_str.endswith("Z"):
            end_str = end_str[:-1]

        from datetime import datetime
        start_dt = datetime.fromisoformat(start_str)
        end_dt = datetime.fromisoformat(end_str)
        duration_days = (end_dt - start_dt).total_seconds() / (24 * 3600)

        if hadm_id not in abx_data:
            abx_data[hadm_id] = {"drugs": set(), "total_days": 0.0}

        abx_data[hadm_id]["drugs"].add(drug_name)
        abx_data[hadm_id]["total_days"] += duration_days

    # Build result DataFrame
    data = []
    for hadm_id in hadm_ids:
        if hadm_id in abx_data:
            data.append({
                "hadm_id": hadm_id,
                "num_distinct_meds": len(abx_data[hadm_id]["drugs"]),
                "total_prescription_days": abx_data[hadm_id]["total_days"],
                "has_prescription": 1,
            })
        else:
            data.append({
                "hadm_id": hadm_id,
                "num_distinct_meds": 0,
                "total_prescription_days": 0.0,
                "has_prescription": 0,
            })

    return pd.DataFrame(data)


def extract_diagnosis_features(graph: Graph) -> pd.DataFrame:
    """Extract diagnosis features for each admission.

    Note: Diagnosis codes are from discharge diagnoses and may contain
    information about the full hospital course. Use with caution in
    predictive models to avoid data leakage.

    Args:
        graph: RDF graph containing diagnosis event data.

    Returns:
        DataFrame with columns: hadm_id, num_diagnoses, primary_icd_chapter
    """
    # Get all admissions first
    all_admissions_query = """
    SELECT DISTINCT ?hadmId
    WHERE {
        ?admission rdf:type mimic:HospitalAdmission ;
                   mimic:hasAdmissionId ?hadmId .
    }
    """

    all_admissions = list(graph.query(all_admissions_query))
    hadm_ids = [int(row[0]) for row in all_admissions]

    # Query diagnosis data
    query = """
    SELECT ?hadmId ?icdCode ?seqNum
    WHERE {
        ?admission rdf:type mimic:HospitalAdmission ;
                   mimic:hasAdmissionId ?hadmId ;
                   mimic:hasDiagnosis ?diagnosis .
        ?diagnosis mimic:hasIcdCode ?icdCode ;
                   mimic:hasSequenceNumber ?seqNum .
    }
    ORDER BY ?hadmId ?seqNum
    """

    results = list(graph.query(query))

    # Group by admission
    dx_data = {}
    for row in results:
        hadm_id = int(row[0])
        icd_code = str(row[1])
        seq_num = int(row[2])

        if hadm_id not in dx_data:
            dx_data[hadm_id] = {"count": 0, "primary_code": None}

        dx_data[hadm_id]["count"] += 1

        if seq_num == 1:
            dx_data[hadm_id]["primary_code"] = icd_code

    # Extract ICD chapter from primary code
    def get_icd_chapter(icd_code: str) -> str:
        """Extract chapter letter from ICD-10 code."""
        if icd_code and len(icd_code) > 0:
            return icd_code[0].upper()
        return "Unknown"

    # Build result DataFrame
    data = []
    for hadm_id in hadm_ids:
        if hadm_id in dx_data:
            primary_code = dx_data[hadm_id]["primary_code"]
            data.append({
                "hadm_id": hadm_id,
                "num_diagnoses": dx_data[hadm_id]["count"],
                "primary_icd_chapter": get_icd_chapter(primary_code) if primary_code else "Unknown",
            })
        else:
            data.append({
                "hadm_id": hadm_id,
                "num_diagnoses": 0,
                "primary_icd_chapter": "Unknown",
            })

    df = pd.DataFrame(data)

    # One-hot encode primary ICD chapter
    if "primary_icd_chapter" in df.columns and len(df) > 0:
        chapter_dummies = pd.get_dummies(df["primary_icd_chapter"], prefix="icd_chapter")
        df = pd.concat([df.drop("primary_icd_chapter", axis=1), chapter_dummies], axis=1)

    return df
