"""Pydantic data models for the conversational analytics pipeline."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class ClinicalConcept(BaseModel):
    name: str
    concept_type: Literal[
        "biomarker", "vital", "drug", "diagnosis", "microbiology", "outcome"
    ]
    attributes: list[str] = []
    resolved_from_category: bool = False


class TemporalConstraint(BaseModel):
    relation: Literal["before", "after", "during", "within"]
    reference_event: str
    time_window: str | None = None


class PatientFilter(BaseModel):
    field: str
    operator: Literal[">", "<", "=", ">=", "<=", "contains", "in"]
    value: str | list[str]


class ReturnType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    TEXT_AND_TABLE = "text_and_table"
    VISUALIZATION = "visualization"


class CompetencyQuestion(BaseModel):
    original_question: str
    clinical_concepts: list[ClinicalConcept] = []
    temporal_constraints: list[TemporalConstraint] = []
    patient_filters: list[PatientFilter] = []
    aggregation: str | None = None
    return_type: ReturnType = ReturnType.TEXT_AND_TABLE
    scope: Literal["single_patient", "cohort", "comparison"] = "single_patient"
    comparison_field: str | None = None


class ExtractionConfig(BaseModel):
    """Configurable settings for data extraction.

    Phase 2 removed the artificial ``max_cohort_size`` cap; the cohort query
    now returns every matching admission. ``batch_size`` bounds the width of
    the ``hadm_id IN (...)`` clauses downstream fetchers send to the
    database — a performance knob, not a semantic one.
    """

    model_config = {"extra": "forbid"}
    batch_size: int = 2000
    cohort_strategy: Literal["recent", "random"] = "recent"


class ExtractionResult(BaseModel):
    patients: list[dict] = []
    admissions: list[dict] = []
    icu_stays: list[dict] = []
    events: dict[str, list[dict]] = {}


class AnswerResult(BaseModel):
    text_summary: str
    data_table: list[dict] | None = None
    table_columns: list[str] | None = None
    visualization_spec: dict | None = None
    graph_stats: dict = {}
    sparql_queries_used: list[str] = []
