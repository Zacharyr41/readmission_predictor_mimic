"""Pydantic data models for the conversational analytics pipeline."""

from enum import Enum
from typing import Literal

import re

from pydantic import BaseModel, Field, field_validator, model_validator

_LOINC_PATTERN = re.compile(r"^\d{1,7}-\d$")


class ClinicalConcept(BaseModel):
    name: str
    concept_type: Literal[
        "biomarker", "vital", "drug", "diagnosis", "microbiology", "outcome"
    ]
    attributes: list[str] = []
    resolved_from_category: bool = False
    loinc_code: str | None = None


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
    ``extractor_key`` is the lookup key into the ``OutcomeRegistry``
    (phase 8c); ``extractor_params`` carries the per-call configuration
    for parametric extractors (e.g. ``{"icd_prefixes": ["I60", "I61"]}``
    for a bleeding-diagnosis outcome). Parameters live on the spec so
    the outcome definition travels with the CQ — reviewers don't have
    to chase a side-channel registry to see how "major bleeding" was
    defined for a given study.

    ``censoring_clock`` is consulted by the survival extractor to decide
    t=0 (admission vs. discharge vs. ICU-out). Default ``"admission"``
    follows the standard epidemiology convention (time since admission);
    the user can override per outcome.
    """

    model_config = {"extra": "forbid"}

    name: str
    outcome_type: Literal["continuous", "binary", "ordinal", "time_to_event"]
    extractor_key: str
    extractor_params: dict = Field(default_factory=dict)
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
    scope: Literal[
        "single_patient", "cohort", "comparison", "causal_effect",
        "patient_similarity",
    ] = "single_patient"
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
    # Phase 8d — bootstrap configuration.
    # Additive fields; non-causal CQs ignore both. ``uncertainty_reps``
    # is the B in the plan (decision context); ``random_state`` threads
    # through the BootstrapRunner to every estimator fit for
    # reproducibility. Tests override with small B for speed.
    uncertainty_reps: int = 200
    random_state: int = 0
    # Phase 9 — similarity spec for both standalone similarity CQs
    # (``scope="patient_similarity"``) and cohort-narrowing on causal
    # CQs (``scope="causal_effect"`` + ``similarity_spec``). The
    # annotation is a forward-ref string because
    # ``src.similarity.models.SimilaritySpec`` imports ``PatientFilter``
    # from this module — a direct import here would be circular. Instead
    # we trigger resolution at the bottom of this module (see the
    # late import + ``model_rebuild`` below), which works because by
    # that point ``PatientFilter`` is already defined so the circular
    # import unwinds cleanly.
    similarity_spec: "SimilaritySpec | None" = None

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


class CriticVerdict(BaseModel):
    """Output of the second-pass plausibility critic (`src/conversational/critic.py`).

    Populated when the orchestrator runs a critic pass over an AnswerResult.
    ``severity`` drives UI rendering: ``"info"`` is silent (default state for
    legitimate answers), ``"warn"`` surfaces a visible note, ``"block"`` should
    suppress the answer or render it behind an explicit override.

    The optional ``suggested_loinc`` + ``correction_rationale`` fields carry
    a self-healing recommendation: when the critic flags a LOINC-grounding
    failure, the orchestrator can re-run the SQL fast-path with the proposed
    code (see ``orchestrator._run_with_critic_retry``).
    """

    plausible: bool
    severity: Literal["info", "warn", "block"]
    concern: str | None = None
    reference_used: str | None = None
    raw_response: str | None = None  # truncated; debug-only
    # Self-healing fields. Populated only when the critic catches a
    # LOINC-grounding failure AND has a confident alternate code. The
    # validator silently coerces malformed strings to ``None`` so a noisy
    # critic response never crashes a turn.
    suggested_loinc: str | None = None
    correction_rationale: str | None = None
    # Externally-grounded evidence: when the critic invokes ``pubmed_search``
    # (or other future tools), it can cite the records it consulted. Each
    # entry: ``{"type": "pubmed", "pmid": str, "title": str, "url": str}``.
    # The validator does shape-only checks (drops malformed entries); the
    # critic loop is responsible for filtering against actually-observed
    # PMIDs to prevent the model from fabricating citations.
    cited_sources: list[dict] | None = None

    @field_validator("suggested_loinc", mode="before")
    @classmethod
    def _coerce_invalid_loinc_to_none(cls, v):
        if v is None or v == "":
            return None
        if not isinstance(v, str) or not _LOINC_PATTERN.match(v):
            return None
        return v

    @field_validator("cited_sources", mode="before")
    @classmethod
    def _validate_cited_sources_shape(cls, v):
        """Drop malformed entries; coerce empty list to None.

        Accepts entries that are dicts with a string-coercible ``pmid``
        consisting of digits. Other entries are silently dropped. If the
        list ends up empty after filtering, returns None to keep the
        no-citations state distinguishable from the empty-list state."""
        if v is None or v == "":
            return None
        if not isinstance(v, list):
            return None
        cleaned: list[dict] = []
        for item in v:
            if not isinstance(item, dict):
                continue
            pmid = item.get("pmid")
            if pmid is None:
                continue
            pmid_str = str(pmid)
            if not pmid_str.isdigit():
                continue
            cleaned.append({**item, "pmid": pmid_str})
        return cleaned or None


class SqlValidationVerdict(BaseModel):
    """Output of the pre-execution SQL validator (`src/conversational/sql_validator.py`).

    The validator is an LLM judge that runs AFTER ``compile_sql`` and BEFORE
    ``backend.execute`` so confidently-broken SQL can be blocked before
    BigQuery cost is paid. Verdict semantics:

    - ``pass``: the SQL is logically consistent with the CompetencyQuestion;
      proceed to execution as usual.
    - ``warn``: the SQL has a soft concern (e.g. unit-pooling on a LIKE-
      fallback) that the post-execution critic should also weigh; the
      orchestrator threads ``concern`` into the critic's ``fallback_warning``.
    - ``block``: the SQL is high-confidence broken (e.g. aggregating the
      wrong column, referencing an unjoined table); the orchestrator
      short-circuits — no execute, no answerer, no critic.

    The validator NEVER blocks a LIKE-fallback path even on suspected unit
    pooling — that case is escalated to the critic's self-healing retry,
    which can mutate the LOINC code. Blocking would remove that path.
    """

    verdict: Literal["pass", "warn", "block"]
    concern: str | None = None
    suggested_fix: str | None = None
    reference_used: str | None = None
    raw_response: str | None = None  # truncated; debug-only
    # Phase E (deterministic validator): when the validator runs the BigQuery
    # dry-run, populate the cost-tracking fields. None on the v1 LLM-judge
    # code path (which doesn't know byte counts) or when the validator
    # didn't reach the dry_run stage.
    bytes_processed: int | None = None
    estimated_usd: float | None = None


class Disambiguation(BaseModel):
    """Output of :func:`clinical_consult.disambiguate`.

    The disambiguator takes an ambiguous concept name (e.g. "lactate") and
    consults literature/registry tools to decide whether it can be
    canonicalized. ``confidence="high"`` is what allows the orchestrator to
    drop the decomposer's clarifying_question and proceed with the resolved
    code; ``"medium"``/``"low"`` flow to the clarify-enrichment path so
    the user still sees the alternates.
    """

    input_name: str
    canonical_name: str
    alternates: list[str] = []
    resolved_code: str | None = None
    code_system: Literal["loinc", "snomed", "rxnorm"] | None = None
    confidence: Literal["low", "medium", "high"]
    reasoning: str | None = None
    citations: list[dict] | None = None


class ClarifyingMessage(BaseModel):
    """Output of :func:`clinical_consult.clarify`.

    Replaces the decomposer's raw ``clarifying_question`` with a
    literature-grounded message. ``alternates_offered`` lists the
    candidates surfaced to the user; ``citations`` carries the evidence
    used to draft the message (anti-hallucination filtered upstream).
    """

    text: str
    alternates_offered: list[str] = []
    citations: list[dict] | None = None


class ContextualNote(BaseModel):
    """Output of :func:`clinical_consult.contextualize`.

    Appended to a successful answer's ``text_summary`` (with citations
    surfaced separately on ``AnswerResult.contextual_citations``) when
    the contextualization flag is enabled and the critic's verdict is
    "info"-or-absent. Never used to override a critic warning.
    """

    text: str
    citations: list[dict] | None = None


# ---------------------------------------------------------------------------
# Phase F: source-of-truth sub-agent output schema.
#
# Mirrors the JSON shape from the user's research:
# {"query": "...", "answer_summary": "≤3 sentences",
#  "findings": [{"claim": "...", "evidence": [...],
#                "confidence": "high|medium|low",
#                "status": "verified|unverified|conflicting"}],
#  "unresolved": [...],
#  "tools_called": [{"name": "...", "count": int}]}
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """One piece of evidence backing a HealthFinding claim."""

    source: Literal["pubmed", "snomed", "rxnorm", "loinc", "mimic_distribution",
                    "openfda", "clinicaltrials", "icd"]
    id: str
    tool: str | None = None
    snippet: str | None = None


class HealthFinding(BaseModel):
    """One claim + its supporting evidence and confidence."""

    claim: str
    evidence: list[Evidence] = []
    confidence: Literal["low", "medium", "high"]
    status: Literal["verified", "unverified", "conflicting"]


class HealthAnswer(BaseModel):
    """Full output of :class:`HealthSourceOfTruthAgent.consult`.

    The orchestrator (or any caller) consumes this when it needs
    cross-MCP biomedical grounding. ``answer_summary`` is for surfacing
    to the user; ``findings`` carries the structured per-claim evidence
    for citation rendering or downstream reasoning."""

    query: str
    answer_summary: str
    findings: list[HealthFinding] = []
    unresolved: list[str] = []
    tools_called: list[dict] = []  # [{"name": str, "count": int, "status": str}]


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
    # Second-pass plausibility verdict (see CriticVerdict). None when the
    # critic was disabled or failed silently.
    critic_verdict: CriticVerdict | None = None
    # Self-healing trace: when the orchestrator runs the SQL fast-path more
    # than once because a critic-suggested LOINC correction triggered a
    # retry, each attempt is recorded here. ``None`` means no retry happened
    # (the common case). Each entry is a dict with: ``attempt`` (int),
    # ``loinc_used`` (str|None), ``text_summary`` (str), ``fallback_warning``
    # (str|None), ``critic_verdict`` (dict|None).
    correction_trace: list[dict] | None = None
    # Phase C: contextual citations carry the literature/registry refs that
    # informed an appended ``ContextualNote``. None when contextualization
    # is disabled or didn't fire. Same shape as ``CriticVerdict.cited_sources``
    # so UI rendering can be shared.
    contextual_citations: list[dict] | None = None


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


# ---------------------------------------------------------------------------
# Phase 9: resolve the forward reference on
# ``CompetencyQuestion.similarity_spec`` so this module can be imported
# standalone. At this point in the file, all local classes (in particular
# ``PatientFilter``) are defined, so ``src.similarity.models`` can safely
# import from here without the partial-import cycle breaking.
# ---------------------------------------------------------------------------

from src.similarity.models import SimilaritySpec  # noqa: E402

CompetencyQuestion.model_rebuild(
    _types_namespace={"SimilaritySpec": SimilaritySpec}, force=True,
)
