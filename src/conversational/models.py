"""Pydantic data models for the conversational analytics pipeline."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, field_validator


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
    # Phase 4: clinician-facing echo + optional clarifying-question short-circuit.
    # ``interpretation_summary`` is always populated before the CQ reaches the
    # orchestrator — synthesised by the decomposer from the structured fields
    # when the LLM omits it — so the UI can show "this is what I'm answering"
    # on every turn. ``clarifying_question`` is only set when the LLM detects
    # ambiguity it cannot resolve; the orchestrator then short-circuits the
    # downstream pipeline and surfaces the question back to the user.
    interpretation_summary: str | None = None
    clarifying_question: str | None = None


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
    # Phase 4: carry the decomposer's interpretation into the UI so the
    # clinician can verify "this is what I'm actually answering" before
    # reading the summary. Propagated by the orchestrator from
    # ``CompetencyQuestion.interpretation_summary``.
    interpretation_summary: str | None = None
    # Truthy when the pipeline short-circuited on a clarifying question; the
    # UI renders this as a follow-up prompt to the user instead of an answer.
    clarifying_question: str | None = None
    # Phase 4.5: when a big question decomposes into multiple CompetencyQuestions,
    # each sub-CQ's per-answer AnswerResult is carried here so the UI can show
    # the breakdown under one synthesized top-level summary. None for single-CQ
    # turns.
    sub_answers: list["AnswerResult"] | None = None


class DecompositionResult(BaseModel):
    """Output of the decomposer.

    Phase 4.5: the decomposer now returns a wrapper around a list of
    CompetencyQuestions plus an optional narrative. The common case is still
    a single CQ with no narrative; "big questions" yield a narrative and
    several sub-CQs that all share one downstream graph.

    The ``competency_questions`` list is never empty — ambiguous questions
    are represented as a single CQ with its ``clarifying_question`` field
    set, not an empty list.
    """

    narrative: str | None = None
    competency_questions: list[CompetencyQuestion]

    @field_validator("competency_questions")
    @classmethod
    def _at_least_one_cq(cls, v: list[CompetencyQuestion]) -> list[CompetencyQuestion]:
        if not v:
            raise ValueError("DecompositionResult must contain at least one CompetencyQuestion")
        return v

    @property
    def is_multi(self) -> bool:
        """True when the decomposer produced more than one CQ for this turn."""
        return len(self.competency_questions) > 1
