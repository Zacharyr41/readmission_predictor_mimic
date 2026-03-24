"""SPARQL reasoning engine for the conversational analytics pipeline.

Selects and executes parameterized SPARQL queries against an RDF knowledge
graph built by the graph_builder module.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel
from rdflib import Graph

from src.conversational.models import CompetencyQuestion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class ReasoningResult(BaseModel):
    """Structured output from the SPARQL reasoning engine."""

    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    sparql_queries: list[str] = []
    template_names: list[str] = []


# ---------------------------------------------------------------------------
# SPARQL prefix block
# ---------------------------------------------------------------------------

_PREFIXES = """\
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX mimic: <http://www.cnam.fr/MIMIC4-ICU-BSI/V1#>
PREFIX time: <http://www.w3.org/2006/time#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(value: str) -> str:
    """Escape characters that could break SPARQL string literals."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _result_to_dicts(result) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert an rdflib SPARQL SELECT result to a list of dicts."""
    if not hasattr(result, "vars") or result.vars is None:
        return [], []

    columns = [str(v) for v in result.vars]
    rows: list[dict[str, Any]] = []
    for row in result:
        d: dict[str, Any] = {}
        for i, col in enumerate(columns):
            val = row[i]
            if val is not None:
                try:
                    d[col] = val.toPython()
                except AttributeError:
                    d[col] = str(val)
            else:
                d[col] = None
        rows.append(d)
    return rows, columns


def _compute_median(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Compute median from value_lookup rows."""
    from statistics import median

    values = [float(r["value"]) for r in rows if r.get("value") is not None]
    if not values:
        return [{"median_value": None}], ["median_value"]
    return [{"median_value": median(values)}], ["median_value"]


# ---------------------------------------------------------------------------
# SPARQL template library
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, str] = {
    "value_lookup": """
SELECT ?value ?unit
WHERE {{
    {{ ?e rdf:type mimic:BioMarkerEvent ;
         mimic:hasBiomarkerType ?label ;
         mimic:hasValue ?value .
       OPTIONAL {{ ?e mimic:hasUnit ?unit . }}
       FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
    UNION
    {{ ?e rdf:type mimic:ClinicalSignEvent ;
         mimic:hasClinicalSignName ?label ;
         mimic:hasValue ?value .
       BIND("" AS ?unit)
       FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
}}
""",
    "value_with_timestamps": """
SELECT ?value ?unit ?timestamp
WHERE {{
    {{ ?e rdf:type mimic:BioMarkerEvent ;
         mimic:hasBiomarkerType ?label ;
         mimic:hasValue ?value ;
         time:inXSDDateTimeStamp ?timestamp .
       OPTIONAL {{ ?e mimic:hasUnit ?unit . }}
       FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
    UNION
    {{ ?e rdf:type mimic:ClinicalSignEvent ;
         mimic:hasClinicalSignName ?label ;
         mimic:hasValue ?value ;
         time:inXSDDateTimeStamp ?timestamp .
       BIND("" AS ?unit)
       FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
}}
ORDER BY ?timestamp
""",
    "aggregation_mean": """
SELECT (AVG(?value) AS ?mean_value)
WHERE {{
    {{ ?e rdf:type mimic:BioMarkerEvent ;
         mimic:hasBiomarkerType ?label ;
         mimic:hasValue ?value .
       FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
    UNION
    {{ ?e rdf:type mimic:ClinicalSignEvent ;
         mimic:hasClinicalSignName ?label ;
         mimic:hasValue ?value .
       FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
}}
""",
    "aggregation_max": """
SELECT (MAX(?value) AS ?max_value)
WHERE {{
    {{ ?e rdf:type mimic:BioMarkerEvent ;
         mimic:hasBiomarkerType ?label ;
         mimic:hasValue ?value .
       FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
    UNION
    {{ ?e rdf:type mimic:ClinicalSignEvent ;
         mimic:hasClinicalSignName ?label ;
         mimic:hasValue ?value .
       FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
}}
""",
    "aggregation_min": """
SELECT (MIN(?value) AS ?min_value)
WHERE {{
    {{ ?e rdf:type mimic:BioMarkerEvent ;
         mimic:hasBiomarkerType ?label ;
         mimic:hasValue ?value .
       FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
    UNION
    {{ ?e rdf:type mimic:ClinicalSignEvent ;
         mimic:hasClinicalSignName ?label ;
         mimic:hasValue ?value .
       FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
}}
""",
    "aggregation_count": """
SELECT (COUNT(?value) AS ?count_value)
WHERE {{
    {{ ?e rdf:type mimic:BioMarkerEvent ;
         mimic:hasBiomarkerType ?label ;
         mimic:hasValue ?value .
       FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
    UNION
    {{ ?e rdf:type mimic:ClinicalSignEvent ;
         mimic:hasClinicalSignName ?label ;
         mimic:hasValue ?value .
       FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
}}
""",
    "patient_list_by_diagnosis": """
SELECT ?subjectId ?hadmId ?icdCode ?longTitle
WHERE {{
    ?patient rdf:type mimic:Patient ;
             mimic:hasSubjectId ?subjectId ;
             mimic:hasAdmission ?admission .
    ?admission mimic:hasAdmissionId ?hadmId ;
               mimic:hasDiagnosis ?dx .
    ?dx mimic:hasIcdCode ?icdCode .
    OPTIONAL {{ ?dx mimic:hasLongTitle ?longTitle . }}
    FILTER(
        STRSTARTS(STR(?icdCode), "{concept_name}")
        || CONTAINS(LCASE(STR(?longTitle)), LCASE("{concept_name}"))
    )
}}
ORDER BY ?subjectId ?hadmId
""",
    "drug_lookup": """
SELECT ?drugName ?startTime ?endTime ?dose ?doseUnit ?route
WHERE {{
    ?e rdf:type mimic:PrescriptionEvent ;
       mimic:hasDrugName ?drugName .
    OPTIONAL {{
        ?e time:hasBeginning ?begin .
        ?begin time:inXSDDateTimeStamp ?startTime .
    }}
    OPTIONAL {{
        ?e time:hasEnd ?end .
        ?end time:inXSDDateTimeStamp ?endTime .
    }}
    OPTIONAL {{ ?e mimic:hasDoseValue ?dose . }}
    OPTIONAL {{ ?e mimic:hasDoseUnit ?doseUnit . }}
    OPTIONAL {{ ?e mimic:hasRoute ?route . }}
    FILTER(CONTAINS(LCASE(STR(?drugName)), LCASE("{concept_name}")))
}}
ORDER BY ?startTime
""",
    "event_count_by_type": """
SELECT ?type (COUNT(?event) AS ?count)
WHERE {{
    ?event rdf:type ?type .
    FILTER(?type IN (
        mimic:BioMarkerEvent, mimic:ClinicalSignEvent,
        mimic:PrescriptionEvent, mimic:DiagnosisEvent,
        mimic:MicrobiologyEvent
    ))
}}
GROUP BY ?type
ORDER BY DESC(?count)
""",
    "temporal_before": """
SELECT ?eventType ?eventLabel ?timestamp
WHERE {{
    ?eventA time:before ?eventB .
    ?eventA rdf:type ?eventType ;
            time:inXSDDateTimeStamp ?timestamp .
    {{ ?eventA mimic:hasBiomarkerType ?eventLabel }}
    UNION
    {{ ?eventA mimic:hasClinicalSignName ?eventLabel }}
    UNION
    {{ ?eventA mimic:hasDrugName ?eventLabel }}
    FILTER(CONTAINS(LCASE(STR(?eventLabel)), LCASE("{concept_name}")))
}}
ORDER BY ?timestamp
""",
    "temporal_during": """
SELECT ?eventType ?eventLabel ?value ?timestamp
WHERE {{
    ?eventA time:inside ?interval .
    ?eventA rdf:type ?eventType ;
            time:inXSDDateTimeStamp ?timestamp .
    OPTIONAL {{ ?eventA mimic:hasValue ?value . }}
    {{ ?eventA mimic:hasBiomarkerType ?eventLabel }}
    UNION
    {{ ?eventA mimic:hasClinicalSignName ?eventLabel }}
    UNION
    {{ ?eventA mimic:hasDrugName ?eventLabel }}
}}
ORDER BY ?timestamp
""",
    "icu_length_of_stay": """
SELECT ?hadmId ?stayId ?losDays
WHERE {{
    ?admission rdf:type mimic:HospitalAdmission ;
               mimic:hasAdmissionId ?hadmId ;
               mimic:containsICUStay ?icuStay .
    ?icuStay mimic:hasStayId ?stayId ;
             time:hasDuration ?duration .
    ?duration time:numericDuration ?losDays .
}}
""",
    "admission_details": """
SELECT ?hadmId ?admissionType ?dischargeLocation ?readmitted30 ?readmitted60
WHERE {{
    ?admission rdf:type mimic:HospitalAdmission ;
               mimic:hasAdmissionId ?hadmId ;
               mimic:hasAdmissionType ?admissionType ;
               mimic:hasDischargeLocation ?dischargeLocation ;
               mimic:readmittedWithin30Days ?readmitted30 ;
               mimic:readmittedWithin60Days ?readmitted60 .
}}
""",
    "patient_demographics": """
SELECT ?subjectId ?age ?gender
WHERE {{
    ?patient rdf:type mimic:Patient ;
             mimic:hasSubjectId ?subjectId ;
             mimic:hasAge ?age ;
             mimic:hasGender ?gender .
}}
ORDER BY ?subjectId
""",
    "microbiology_results": """
SELECT ?specimenType ?organismName ?timestamp
WHERE {{
    ?e rdf:type mimic:MicrobiologyEvent ;
       time:inXSDDateTimeStamp ?timestamp .
    OPTIONAL {{ ?e mimic:hasSpecimenType ?specimenType . }}
    OPTIONAL {{ ?e mimic:hasOrganismName ?organismName . }}
    FILTER(
        CONTAINS(LCASE(STR(?specimenType)), LCASE("{concept_name}"))
        || CONTAINS(LCASE(STR(?organismName)), LCASE("{concept_name}"))
    )
}}
ORDER BY ?timestamp
""",
    "all_events_for_stay": """
SELECT ?eventType ?label ?value ?timestamp
WHERE {{
    ?stay rdf:type mimic:ICUStay ;
          mimic:hasStayId ?stayId .
    ?event mimic:associatedWithICUStay ?stay ;
           rdf:type ?eventType ;
           time:inXSDDateTimeStamp ?timestamp .
    OPTIONAL {{ ?event mimic:hasValue ?value . }}
    OPTIONAL {{
        {{ ?event mimic:hasBiomarkerType ?label }}
        UNION
        {{ ?event mimic:hasClinicalSignName ?label }}
        UNION
        {{ ?event mimic:hasDrugName ?label }}
        UNION
        {{ ?event mimic:hasSpecimenType ?label }}
    }}
    FILTER(?eventType IN (
        mimic:BioMarkerEvent, mimic:ClinicalSignEvent,
        mimic:PrescriptionEvent, mimic:MicrobiologyEvent
    ))
}}
ORDER BY ?timestamp
""",
    "comparison_two_groups": """
SELECT ?readmitted30 (AVG(?value) AS ?avg_value) (COUNT(?value) AS ?count)
WHERE {{
    ?patient rdf:type mimic:Patient ;
             mimic:hasAdmission ?admission .
    ?admission mimic:readmittedWithin30Days ?readmitted30 ;
               mimic:containsICUStay ?stay .
    ?event mimic:associatedWithICUStay ?stay ;
           mimic:hasValue ?value .
    {{
        ?event mimic:hasBiomarkerType ?label .
        FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
    UNION
    {{
        ?event mimic:hasClinicalSignName ?label .
        FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
}}
GROUP BY ?readmitted30
""",
}

# Alias: trend_over_time uses the same template as value_with_timestamps
TEMPLATES["trend_over_time"] = TEMPLATES["value_with_timestamps"]

# Parameterized comparison: {group_property} is substituted at build time
TEMPLATES["comparison_by_field"] = """
SELECT ?group_value (AVG(?value) AS ?avg_value) (COUNT(?value) AS ?count)
WHERE {{
    ?patient rdf:type mimic:Patient ;
             {group_join}
             mimic:hasAdmission ?admission .
    ?admission {admission_group_clause}
               mimic:containsICUStay ?stay .
    ?event mimic:associatedWithICUStay ?stay ;
           mimic:hasValue ?value .
    {{
        ?event mimic:hasBiomarkerType ?label .
        FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
    UNION
    {{
        ?event mimic:hasClinicalSignName ?label .
        FILTER(LCASE(STR(?label)) = LCASE("{concept_name}"))
    }}
}}
GROUP BY ?group_value
"""

# Mapping from comparison_field values to SPARQL fragments
TEMPLATES["mortality_count"] = """
SELECT ?expired (COUNT(DISTINCT ?admission) AS ?count)
WHERE {{
    ?admission rdf:type mimic:HospitalAdmission ;
               mimic:hasHospitalExpireFlag ?expired .
}}
GROUP BY ?expired
"""

_COMPARISON_FIELD_MAP: dict[str, tuple[str, str]] = {
    # field_name → (group_join on Patient, admission_group_clause)
    "gender": (
        "mimic:hasGender ?group_value ;",
        "",
    ),
    "age": (
        "mimic:hasAge ?group_value ;",
        "",
    ),
    "readmitted_30d": (
        "",
        "mimic:readmittedWithin30Days ?group_value ;",
    ),
    "readmitted_60d": (
        "",
        "mimic:readmittedWithin60Days ?group_value ;",
    ),
    "admission_type": (
        "",
        "mimic:hasAdmissionType ?group_value ;",
    ),
    "discharge_location": (
        "",
        "mimic:hasDischargeLocation ?group_value ;",
    ),
}


# ---------------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------------


def select_templates(cq: CompetencyQuestion) -> list[str]:
    """Map a CompetencyQuestion to the SPARQL template names to execute."""
    templates: list[str] = []

    if not cq.clinical_concepts:
        if any(f.field in ("age", "gender") for f in cq.patient_filters):
            templates.append("patient_demographics")
        elif cq.aggregation:
            # Aggregation without concepts → likely LOS or admission property
            templates.append("icu_length_of_stay")
        else:
            templates.append("admission_details")
        return templates

    for concept in cq.clinical_concepts:
        # Outcome has its own template regardless of aggregation
        if concept.concept_type == "outcome":
            templates.append("mortality_count")
        elif cq.scope == "comparison":
            if cq.comparison_field and cq.comparison_field in _COMPARISON_FIELD_MAP:
                templates.append("comparison_by_field")
            else:
                templates.append("comparison_two_groups")
        elif cq.aggregation == "median":
            # No SPARQL MEDIAN — fetch raw values for Python post-processing
            templates.append("value_lookup")
        elif cq.aggregation in ("mean", "avg"):
            templates.append("aggregation_mean")
        elif cq.aggregation == "max":
            templates.append("aggregation_max")
        elif cq.aggregation == "min":
            templates.append("aggregation_min")
        elif cq.aggregation == "count":
            templates.append("aggregation_count")
        elif concept.concept_type in ("biomarker", "vital"):
            templates.append("value_with_timestamps")
        elif concept.concept_type == "drug":
            templates.append("drug_lookup")
        elif concept.concept_type == "diagnosis":
            templates.append("patient_list_by_diagnosis")
        elif concept.concept_type == "microbiology":
            templates.append("microbiology_results")

    # Temporal constraint overlays
    for tc in cq.temporal_constraints:
        if tc.relation == "before" and "temporal_before" not in templates:
            templates.append("temporal_before")
        elif tc.relation in ("during", "within") and "temporal_during" not in templates:
            templates.append("temporal_during")

    return templates


# ---------------------------------------------------------------------------
# SPARQL building
# ---------------------------------------------------------------------------


def build_sparql(
    template_name: str,
    cq: CompetencyQuestion,
    concept_index: int = 0,
) -> str:
    """Fill a SPARQL template with parameters from the CompetencyQuestion."""
    template = TEMPLATES[template_name]

    concept_name = ""
    if cq.clinical_concepts and concept_index < len(cq.clinical_concepts):
        concept_name = _sanitize(cq.clinical_concepts[concept_index].name)

    format_kwargs: dict[str, str] = {"concept_name": concept_name}

    if template_name == "comparison_by_field" and cq.comparison_field:
        group_join, admission_clause = _COMPARISON_FIELD_MAP.get(
            cq.comparison_field, ("", ""),
        )
        format_kwargs["group_join"] = group_join
        format_kwargs["admission_group_clause"] = admission_clause

    return _PREFIXES + template.format(**format_kwargs)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def reason(graph: Graph, cq: CompetencyQuestion) -> ReasoningResult:
    """Execute SPARQL queries against an RDF graph for a CompetencyQuestion.

    Parameters
    ----------
    graph:
        An rdflib Graph (built by ``build_query_graph``).
    cq:
        The structured competency question.

    Returns
    -------
    ReasoningResult
        Rows, columns, and the SPARQL queries executed.
    """
    template_names = select_templates(cq)
    all_rows: list[dict[str, Any]] = []
    all_columns: list[str] = []
    sparql_queries: list[str] = []

    if cq.clinical_concepts:
        for i, _concept in enumerate(cq.clinical_concepts):
            for tname in template_names:
                sparql = build_sparql(tname, cq, concept_index=i)
                sparql_queries.append(sparql)

                try:
                    result = graph.query(sparql)
                except Exception:
                    logger.warning(
                        "SPARQL query failed for template %s", tname, exc_info=True,
                    )
                    continue

                rows, columns = _result_to_dicts(result)

                # Median post-processing
                if cq.aggregation == "median" and tname == "value_lookup" and rows:
                    rows, columns = _compute_median(rows)

                if columns and not all_columns:
                    all_columns = columns
                all_rows.extend(rows)
    else:
        for tname in template_names:
            sparql = build_sparql(tname, cq)
            sparql_queries.append(sparql)

            try:
                result = graph.query(sparql)
            except Exception:
                logger.warning(
                    "SPARQL query failed for template %s", tname, exc_info=True,
                )
                continue

            rows, columns = _result_to_dicts(result)
            if columns and not all_columns:
                all_columns = columns
            all_rows.extend(rows)

    return ReasoningResult(
        rows=all_rows,
        columns=all_columns,
        sparql_queries=sparql_queries,
        template_names=template_names,
    )
