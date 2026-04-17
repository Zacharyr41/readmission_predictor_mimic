"""Pydantic data models for the conversational analytics pipeline."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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


# ---------------------------------------------------------------------------
# Phase 8a — causal-inference input-side specs.
#
# These describe the shape of a CompetencyQuestion whose scope is
# ``"causal_effect"``. The compute-side result types live in
# ``src.causal.models`` so that module can evolve its internals without
# reaching back into the conversational schema.
#
# Per the 2026-04-17 decision log: interventions MUST be derived from
# structured ontologies (RxNorm / SNOMED / ICD-10-PCS / LOINC). No
# manual synonym lists. Every ``InterventionSpec`` carries a
# ``provenance`` payload that an investigator can reproduce row-for-row.
# ---------------------------------------------------------------------------


class InterventionSpec(BaseModel):
    """One element of the intervention set I = {I_1, …, I_C} (spec §1).

    An intervention predicate is a Boolean function over (patient,
    admission) grounded in at least one ontology code. Exactly one of
    ``rxnorm_ingredient``, ``snomed_concept_id``, ``icd10pcs_code``, or
    ``loinc_code`` must be set — a string-only "drug name" is not
    sufficient (the no-curation rule; see
    ``memory/feedback_correctness_no_curation.md``).

    ``label`` is the human-readable identifier used in result keys
    (``mu_c[label]``) and the UI. ``provenance`` records where the
    intervention definition came from — ontology source, version, and
    the resolution trace — so the treatment-assignment step in 8b
    produces an auditable cohort.
    """

    model_config = {"extra": "forbid"}

    label: str = Field(description="human-readable intervention identifier, used as a result key")
    kind: Literal["drug", "procedure", "measurement", "other"]
    # Exactly one ontology code must be provided. Enforced by the
    # validator below; 8b's resolution pipeline fans these out into the
    # concrete predicates ingested by the treatment-assignment step.
    rxnorm_ingredient: str | None = None
    snomed_concept_id: str | None = None
    icd10pcs_code: str | None = None
    loinc_code: str | None = None
    # Optional predicate-narrowing (e.g. "within 24h of admission").
    # Phase 8c uses this to implement early-vs-late vasopressor
    # distinctions; 8a just threads it through.
    timing_window: str | None = None
    # Reserved for the "no exposure" / control arm. When true, the
    # intervention resolver must invert the ontology predicate.
    is_control: bool = False
    # Auditable provenance: {"ontology": "RxNorm", "version": "2024-07",
    # "resolved_via": "rxnav/REST", "descendants_expanded": N, "notes": "…"}.
    provenance: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_ontology_code(self) -> "InterventionSpec":
        codes = [
            self.rxnorm_ingredient,
            self.snomed_concept_id,
            self.icd10pcs_code,
            self.loinc_code,
        ]
        n_set = sum(1 for c in codes if c)
        if n_set != 1:
            raise ValueError(
                "InterventionSpec must carry exactly one ontology code "
                "(rxnorm_ingredient | snomed_concept_id | icd10pcs_code | "
                f"loinc_code); got {n_set}. See memory/feedback_correctness_no_curation.md "
                "for why hand-maintained synonym lists are not accepted."
            )
        return self


class OutcomeSpec(BaseModel):
    """One element of the outcome vector (f_1, …, f_n) (spec §1, §3).

    ``outcome_type`` drives estimator / diagnostic dispatch: a binary
    outcome fits a logistic learner; continuous fits a regression
    learner; ordinal uses a proportional-odds variant; time_to_event
    uses the survival branch landing in 8c.
    ``extractor_key`` is the lookup key into the forthcoming
    OutcomeRegistry (phase 8c); 8a only stores it.

    ``censoring_clock`` is consulted by the survival extractor to decide
    t=0 (admission vs. discharge vs. ICU-out). Default ``"admission"``
    follows the standard epidemiology convention (time since admission);
    the user can override per outcome.
    """

    model_config = {"extra": "forbid"}

    name: str
    outcome_type: Literal["continuous", "binary", "ordinal", "time_to_event"]
    extractor_key: str
    higher_is_better: bool = Field(
        default=False,
        description="if False (default), lower values are clinically preferred; drives ranking direction",
    )
    censoring_clock: Literal["admission", "discharge", "icu_out"] = "admission"
    censoring_horizon_days: int | None = None


class AggregationSpec(BaseModel):
    """The aggregation function g: Y_1 × … × Y_n → ℝ (spec §3).

    ``kind="identity"`` is the default for n=1 — no composition.
    ``kind="weighted_sum"`` takes ``weights`` mapping outcome names to
    real weights (negative weights encode "lower is better" flipping).
    ``kind="dominant"`` takes ``priority`` — a list of outcome names
    ordered best-first; the first with a clinically meaningful
    between-arm difference dominates the ranking.
    ``kind="utility"`` reserves space for a caller-provided callable at
    runtime; 8a rejects it (requires 8f registration).
    """

    model_config = {"extra": "forbid"}

    kind: Literal["identity", "weighted_sum", "dominant", "utility"] = "identity"
    weights: dict[str, float] | None = None
    priority: list[str] | None = None

    @model_validator(mode="after")
    def _kind_consistent_with_params(self) -> "AggregationSpec":
        if self.kind == "weighted_sum" and not self.weights:
            raise ValueError("weighted_sum AggregationSpec requires non-empty weights")
        if self.kind == "dominant" and not self.priority:
            raise ValueError("dominant AggregationSpec requires non-empty priority")
        if self.kind == "identity" and (self.weights or self.priority):
            raise ValueError("identity AggregationSpec must not carry weights or priority")
        return self


class CompetencyQuestion(BaseModel):
    original_question: str
    clinical_concepts: list[ClinicalConcept] = []
    temporal_constraints: list[TemporalConstraint] = []
    patient_filters: list[PatientFilter] = []
    aggregation: str | None = None
    return_type: ReturnType = ReturnType.TEXT_AND_TABLE
    scope: Literal["single_patient", "cohort", "comparison", "causal_effect"] = (
        "single_patient"
    )
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
    # ---- Phase 8a — causal-inference fields ------------------------------
    # Populated only when ``scope == "causal_effect"``. Empty / defaulted
    # for every other scope, so existing callers / tests are unaffected.
    # The planner branch at QueryPlan.CAUSAL reads these; non-causal
    # plans ignore them entirely.
    intervention_set: list[InterventionSpec] | None = None
    outcome_vector: list[OutcomeSpec] | None = None
    aggregation_spec: AggregationSpec | None = None
    # The query point x (spec §6 input 4): covariate dict for personalized
    # estimands. None ⇒ compute population-average estimands only.
    query_patient_covariates: dict | None = None
    alpha: float = 0.05
    mode: Literal["causal", "associative"] = "associative"
    # Registry-lookup key for the estimator family (T/S/X-learner, AIPW,
    # TMLE, …). 8d registers "t_learner"; 8e/8g add the rest. 8a doesn't
    # dispatch on this yet — the stub ignores it — but carrying the field
    # now lets the schema tests for 8d-h work without a second migration.
    estimator_family: str = "t_learner"
    uncertainty_method: Literal["asymptotic", "bootstrap", "bayesian"] = "bootstrap"

    @model_validator(mode="after")
    def _causal_scope_requires_intervention_set(self) -> "CompetencyQuestion":
        """Phase 8a guard. For causal_effect scope, the planner's
        CAUSAL route requires |I| ≥ 2. We do NOT raise here on |I| < 2 —
        the planner handles the degenerate case by falling back to a
        non-causal plan. But we DO reject negative α and a claimed
        ``mode="causal"`` without any assumption acknowledgement once
        the 8h assumption-ledger plumbing lands; for 8a we only check α.
        """
        if self.alpha <= 0.0 or self.alpha >= 1.0:
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha}")
        return self


class ExtractionConfig(BaseModel):
    """Configurable settings for data extraction.

    Phase 2 removed the artificial ``max_cohort_size`` cap; the cohort query
    now returns every matching admission. ``batch_size`` bounds the width of
    the ``hadm_id IN (...)`` clauses downstream fetchers send to the
    database — a performance knob, not a semantic one.

    Phase 7b: ``max_concurrent_batches`` controls how many batch fetches run
    in parallel. BigQuery and DuckDB are both fine with modest concurrency;
    each batch fires 3-N independent queries so 8 workers overlap ~30-40
    queries at once. Drop to 1 to run sequentially (legacy behaviour);
    raise to 16 if the database isn't your bottleneck.
    """

    model_config = {"extra": "forbid"}
    batch_size: int = 2000
    cohort_strategy: Literal["recent", "random"] = "recent"
    max_concurrent_batches: int = 8

    @field_validator("max_concurrent_batches")
    @classmethod
    def _positive_concurrency(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"max_concurrent_batches must be >= 1 (got {v}); use 1 for "
                "sequential execution"
            )
        return v


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
