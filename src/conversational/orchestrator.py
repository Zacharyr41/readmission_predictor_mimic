"""Pipeline orchestrator for the conversational analytics layer.

Wires together decomposer, extractor, graph_builder, reasoner, and answerer
into a single ``ask(question) -> AnswerResult`` interface with conversation
history management.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from src.conversational.answerer import generate_answer
from src.conversational.clinical_consult import (
    clarify,
    contextualize,
    disambiguate,
)
from src.conversational.concept_resolver import ConceptResolver
from src.conversational.critic import critique
from src.conversational.decomposer import decompose_question
from src.conversational.extractor import (
    _BigQueryBackend,
    _DuckDBBackend,
    _extract,
    extract,
    extract_bigquery,
    merge_extractions,
)
from src.conversational.graph_builder import build_query_graph
from src.conversational.models import (
    AnswerResult,
    CompetencyQuestion,
    DecompositionResult,
    ExtractionConfig,
)
from src.conversational.operations import get_default_registry
from src.conversational.planner import QueryPlan, QueryPlanner
from src.conversational.health_evidence.mcp_client import (
    McpClient,
    McpServerConfig,
)
from src.conversational.health_evidence.sub_agent import (
    HealthSourceOfTruthAgent,
)
from src.conversational.reasoner import reason
from src.conversational.sql_fastpath import compile_sql
from src.conversational.sql_validator_dry_run import (
    validate_sql_deterministic,
)

if TYPE_CHECKING:
    import anthropic

logger = logging.getLogger(__name__)


class ConversationalPipeline:
    """Chains decomposer -> extractor -> graph_builder -> reasoner -> answerer."""

    def __init__(
        self,
        db_path: Path,
        ontology_dir: Path,
        api_key: str,
        *,
        data_source: str = "local",
        bigquery_project: str | None = None,
        extraction_config: ExtractionConfig | None = None,
        max_workers: int = 1,
        enable_critic: bool = True,
        critic_max_retries: int = 1,
        enable_pre_validator: bool = True,
        pre_validator_timeout: float = 5.0,
        pre_validator_max_usd: float = 0.50,
        pre_validator_max_bytes: int = 10 * 1024**3,
        bq_validator_mcp_config: McpServerConfig | None = None,
        enable_disambiguation: bool = True,
        enable_clarify_enrichment: bool = True,
        enable_contextualization: bool = False,
        enable_sub_agent_in_contextualize: bool = False,
    ) -> None:
        import anthropic as _anthropic

        self._db_path = db_path
        self._ontology_dir = ontology_dir
        self._data_source = data_source
        self._bigquery_project = bigquery_project
        self._extraction_config = extraction_config
        self._max_workers = max_workers
        # Second-pass plausibility critic. ON by default in production;
        # tests opt out via ``enable_critic=False`` to avoid extending
        # every mock LLM response list.
        self._enable_critic = enable_critic
        # Self-healing retry budget for the SQL fast-path. 1 retry is
        # enough to catch the lactate-class LOINC misidentification case
        # without unbounded latency. Configurable for tests that exercise
        # multi-retry behavior or boundary conditions.
        self._critic_max_retries = critic_max_retries
        # Pre-execution SQL validator (Phase B → E: now deterministic via
        # the bq-validator MCP, replacing the v1 LLM judge). Default-ON in
        # production; tests opt out via ``enable_pre_validator=False`` to
        # avoid spawning the MCP subprocess. Counters track outcomes for
        # cost telemetry.
        self._enable_pre_validator = enable_pre_validator
        self._pre_validator_timeout = pre_validator_timeout
        self._pre_validator_max_usd = pre_validator_max_usd
        self._pre_validator_max_bytes = pre_validator_max_bytes
        self._bq_validator_mcp_config = bq_validator_mcp_config
        self._bq_validator_client: McpClient | None = None
        self._pre_validator_blocks: int = 0
        self._pre_validator_warns: int = 0
        self._pre_validator_passes: int = 0
        # Phase E: graph-extraction blocks. Drained from
        # ``backend.blocked_queries`` after each ask().
        self._extractor_blocks: int = 0
        # Phase C: clinical-consult flags + counters. Disambiguation and
        # clarify-enrichment are default-ON because they only fire on
        # already-ambiguous turns (the decomposer flagged them) and the
        # clarify path doesn't touch BigQuery, so the cost ceiling is
        # one extra LLM call per ambiguous concept. Contextualization is
        # default-OFF — pure UX additive feature, opting in is a deliberate
        # cost choice.
        self._enable_disambiguation = enable_disambiguation
        self._enable_clarify_enrichment = enable_clarify_enrichment
        self._enable_contextualization = enable_contextualization
        # Phase F: when contextualization is on, optionally route the
        # literature lookup through the HealthSourceOfTruthAgent sub-agent
        # for cross-MCP grounding (PubMed + LOINC + MIMIC distribution).
        # Default OFF — opt-in costs ~2x the LLM calls of plain
        # contextualize (sub-agent has its own context window).
        self._enable_sub_agent_in_contextualize = enable_sub_agent_in_contextualize
        self._disambiguations_attempted: int = 0
        self._disambiguations_resolved: int = 0
        self._clarify_enrichments: int = 0
        self._contextualizations: int = 0
        self._sub_agent_consultations: int = 0
        # Per-turn cache: maps a CQ identity (id() at decomp time) to the
        # list of partial Disambiguation objects produced for it. The
        # clarify short-circuit reads this so it can pass partials to the
        # clarify() formatter. Reset at the top of each ask().
        self._last_disambiguations: dict[int, list] = {}
        repo_root = Path(__file__).parent.parent.parent
        # Phase 5: wire SNOMED hierarchy if the cached JSON is present.
        # The SnomedHierarchy class itself degrades gracefully on missing
        # files, but constructing one when the file is absent produces
        # spurious warnings — so we only instantiate when the file exists.
        hierarchy = None
        hierarchy_path = repo_root / "data" / "ontology_cache" / "snomed_hierarchy.json"
        if hierarchy_path.exists():
            from src.graph_construction.terminology.snomed_hierarchy import (
                SnomedHierarchy,
            )
            hierarchy = SnomedHierarchy(hierarchy_path)
        self._resolver = ConceptResolver(
            mappings_dir=repo_root / "data" / "mappings",
            hierarchy=hierarchy,
        )
        self._registry = get_default_registry()
        self._planner = QueryPlanner(registry=self._registry)
        self._client: anthropic.Anthropic = _anthropic.Anthropic(api_key=api_key)
        self.conversation_history: list[tuple[CompetencyQuestion, AnswerResult]] = []
        self.max_history: int = 10

    def ask(self, question: str) -> AnswerResult:
        """Run the full pipeline for a natural-language question.

        Phase 4.5 + Phase 7a: the decomposer returns one or more
        CompetencyQuestions. Each sub-CQ is independently classified by
        the planner:

          - **SQL fast-path** (``QueryPlan.SQL_FAST``) — single-concept
            aggregate / comparison / diagnosis list / mortality. Skips
            extract + graph + SPARQL; one direct SQL call.
          - **Graph path** (``QueryPlan.GRAPH``) — anything needing the
            RDF knowledge graph (temporal Allen relations, time-series
            viz, median, multi-concept).

        Graph-path sub-CQs share ONE graph for the turn (Phase 4.5
        contract preserved). SQL-fast sub-CQs skip merge+build+reason
        entirely but still contribute to the final multi-CQ AnswerResult.

        Clarify short-circuit: if ANY sub-CQ has ``clarifying_question``,
        no downstream stages run.
        """
        try:
            decomp = decompose_question(
                self._client, question,
                conversation_history=list(self.conversation_history) or None,
            )

            # Phase C: literature-based disambiguation BEFORE the existing
            # clarify short-circuit. CQs whose ambiguity resolves with high
            # confidence get their clarifying_question cleared and their
            # concept codes mutated, then proceed to the normal pipeline.
            self._last_disambiguations = {}
            if self._enable_disambiguation:
                self._try_disambiguate(decomp, original_question=question)

            # Clarify short-circuit: any sub-CQ with a non-empty
            # clarifying_question wins; the whole turn becomes a clarify.
            clarifying_cq = next(
                (
                    cq for cq in decomp.competency_questions
                    if cq.clarifying_question and cq.clarifying_question.strip()
                ),
                None,
            )
            if clarifying_cq is not None:
                raw_text = clarifying_cq.clarifying_question
                final_text = raw_text
                # Phase C: clarify-enrichment formats a literature-grounded
                # message. On any failure (None) we fall through to the raw
                # decomposer text — perfect-fidelity regression guard.
                if self._enable_clarify_enrichment:
                    partials = self._last_disambiguations.get(
                        id(clarifying_cq), [],
                    )
                    msg = clarify(
                        self._client, question, raw_text, partials,
                    )
                    if msg is not None and msg.text:
                        self._clarify_enrichments += 1
                        final_text = msg.text
                answer = AnswerResult(
                    text_summary=final_text,
                    interpretation_summary=clarifying_cq.interpretation_summary,
                    clarifying_question=final_text,
                )
                # Record the first CQ in history so conversation context is preserved.
                self._remember(decomp.competency_questions[0], answer)
                return answer

            # Pre-resolve concepts once per CQ. SQL fast-path needs the
            # resolved name list for OR-matching (category → concrete names);
            # graph path's ``extract`` re-resolves internally but benefits
            # from the ``resolved_from_category`` marker being set here.
            per_cq_resolved: list[list[list[str]]] = []  # per CQ, per concept
            for cq in decomp.competency_questions:
                per_concept: list[list[str]] = []
                for concept in cq.clinical_concepts:
                    resolved = self._resolver.resolve(concept)
                    if len(resolved) > 1 or (
                        len(resolved) == 1 and resolved[0] != concept.name
                    ):
                        concept.resolved_from_category = True
                    per_concept.append(resolved)
                per_cq_resolved.append(per_concept)

            # Open one backend for the whole turn so SQL fast-path and
            # graph-path extractions share the same connection/client.
            with self._open_backend() as backend:
                sub_answers: list[AnswerResult | None] = [None] * len(
                    decomp.competency_questions
                )
                graph_cqs: list[tuple[int, CompetencyQuestion]] = []
                graph_extractions: list = []
                fastpath_sparql: list[str] = []

                for idx, cq in enumerate(decomp.competency_questions):
                    plan = self._planner.classify(cq)
                    if plan == QueryPlan.SQL_FAST:
                        sub, sql_used, _fb = self._run_with_critic_retry(
                            cq, backend, resolved_names=per_cq_resolved[idx][0],
                        )
                        # ``interpretation_summary`` and ``critic_verdict`` are
                        # already attached inside ``_run_with_critic_retry``.
                        sub = self._contextualize(sub, cq)
                        sub_answers[idx] = sub
                        fastpath_sparql.extend(sql_used)  # stash for aggregation
                    elif plan == QueryPlan.CAUSAL:
                        # Phase 8a: wired to a stub that returns a
                        # well-shaped but NaN-valued CausalEffectResult so
                        # the UI can be built against the real contract
                        # before estimators land in 8d. When the stub fires
                        # in a live session, the summary text makes it
                        # obvious we're not returning a real estimate.
                        sub = self._run_causal(cq)
                        sub.interpretation_summary = cq.interpretation_summary
                        sub = self._critique(sub, cq)
                        sub = self._contextualize(sub, cq)
                        sub_answers[idx] = sub
                    elif plan == QueryPlan.SIMILARITY:
                        # Phase 9: standalone similarity ranking. The
                        # decomposer marks scope='patient_similarity' and
                        # populates ``similarity_spec``; ``_run_similarity``
                        # calls into ``src.similarity.run.run_similarity``
                        # and wraps the ranked output as a 6-column
                        # ``data_table``. Causal CQs that *also* carry a
                        # similarity_spec stay on QueryPlan.CAUSAL — the
                        # spec is consumed there as a cohort-narrowing
                        # directive (src/causal/run.py:187-219).
                        sub = self._run_similarity(cq, backend)
                        sub.interpretation_summary = cq.interpretation_summary
                        sub = self._critique(sub, cq)
                        sub = self._contextualize(sub, cq)
                        sub_answers[idx] = sub
                    else:
                        graph_cqs.append((idx, cq))
                        graph_extractions.append(
                            _extract(
                                backend, cq,
                                config=self._extraction_config,
                                resolver=self._resolver,
                            )
                        )

                graph_stats: dict = {}
                graph_sparql: list[str] = []
                if graph_cqs:
                    merged = (
                        graph_extractions[0] if len(graph_extractions) == 1
                        else merge_extractions(graph_extractions)
                    )
                    any_temporal = any(
                        bool(cq.temporal_constraints) for _, cq in graph_cqs
                    )
                    graph, graph_stats = build_query_graph(
                        self._ontology_dir, merged,
                        skip_allen_relations=not any_temporal,
                        max_workers=self._max_workers,
                    )
                    for idx, cq in graph_cqs:
                        reasoning = reason(graph, cq)
                        graph_sparql.extend(reasoning.sparql_queries)
                        sub = generate_answer(
                            self._client, cq, reasoning.rows,
                            graph_stats, reasoning.sparql_queries,
                        )
                        sub.interpretation_summary = cq.interpretation_summary
                        sub = self._critique(sub, cq)
                        sub = self._contextualize(sub, cq)
                        sub_answers[idx] = sub

            # All slots filled now (planner produces exactly one plan per CQ).
            completed: list[AnswerResult] = [a for a in sub_answers if a is not None]

            if not decomp.is_multi:
                answer = completed[0]
            else:
                answer = AnswerResult(
                    text_summary=decomp.narrative or "Multi-part answer:",
                    interpretation_summary=decomp.narrative,
                    graph_stats=graph_stats,
                    sparql_queries_used=list(fastpath_sparql) + list(graph_sparql),
                    sub_answers=completed,
                )

            self._remember(decomp.competency_questions[0], answer)
            return answer

        except Exception:
            logger.exception("Pipeline failed for question: %s", question)
            return AnswerResult(
                text_summary=(
                    "An error occurred while processing your question. "
                    "Please try rephrasing."
                ),
            )

    # -- internal helpers --------------------------------------------------

    @contextmanager
    def _open_backend(self):
        """Open the right backend for this pipeline's data source.

        Used once per ``ask()`` call; both SQL fast-path and graph-path
        extractions share the connection/client to avoid repeated setup
        cost. Close is guaranteed by the contextmanager.

        Phase E: when running against BigQuery with the pre-validator
        enabled, inject the deterministic validator at the backend
        boundary. ``execute_tolerant`` (used by all 9 extractor.py call
        sites) catches block verdicts and logs to ``backend.blocked_queries``
        so the orchestrator can surface a structured warning.
        """
        if self._data_source == "bigquery":
            validator = self._make_extractor_validator()
            backend = _BigQueryBackend(
                self._bigquery_project, validator=validator,
                max_bytes_billed=self._pre_validator_max_bytes,
            )
        else:
            backend = _DuckDBBackend(self._db_path)
        try:
            yield backend
        finally:
            try:
                # Drain extractor blocks into the per-pipeline counter so
                # the chat UI / telemetry can see them later.
                self._extractor_blocks += len(
                    getattr(backend, "blocked_queries", []) or []
                )
            except Exception:  # noqa: BLE001
                pass
            backend.close()

    def _make_extractor_validator(self):
        """Build the validator callable injected into ``_BigQueryBackend``.

        Returns ``None`` when the pre-validator is disabled OR the MCP
        client is unavailable. The backend handles ``None`` cleanly
        (skips validation).
        """
        if not self._enable_pre_validator:
            return None
        client = self._get_bq_validator_client()
        if client is None:
            return None
        max_usd = self._pre_validator_max_usd
        max_bytes = self._pre_validator_max_bytes
        timeout = self._pre_validator_timeout

        def validator(sql: str, params: list):
            from src.conversational.sql_fastpath import SqlFastpathQuery

            wrapped = SqlFastpathQuery(sql=sql, params=list(params), columns=[])
            return validate_sql_deterministic(
                wrapped, mcp_client=client,
                fallback_warning=None,
                resolved_itemids=[1],  # treat extractor SQL as "grounded";
                                       # never downgrade to warn on the
                                       # tolerant extractor path
                max_usd=max_usd, max_bytes=max_bytes, timeout=timeout,
            )

        return validator

    def _try_disambiguate(
        self,
        decomp,
        *,
        original_question: str,
    ) -> None:
        """For each CQ with a clarifying_question, attempt literature-based
        disambiguation of its clinical_concepts.

        Side effects:
          * Mutates concepts in place — when a concept's disambiguation is
            high-confidence and produces a resolved code, the matching
            ontology field on the concept is set and ``resolved_from_category``
            is marked True.
          * Clears ``cq.clarifying_question`` when ALL of the CQ's concepts
            disambiguate with high confidence — letting the pipeline proceed
            without a user round-trip.
          * Records the per-CQ Disambiguation list under
            ``self._last_disambiguations`` keyed by ``id(cq)`` so the clarify
            short-circuit can pass partials to ``clarify()``.

        Never raises — every disambiguate() call has its own failure
        handling, returning None on any error; we treat None as
        "couldn't resolve" without crashing the pipeline.
        """
        for cq in decomp.competency_questions:
            if not (cq.clarifying_question and cq.clarifying_question.strip()):
                continue
            if not cq.clinical_concepts:
                continue
            per_cq: list = []
            all_high = True
            for concept in cq.clinical_concepts:
                self._disambiguations_attempted += 1
                d = disambiguate(
                    self._client, concept,
                    original_question=original_question,
                )
                if d is None:
                    all_high = False
                    continue
                per_cq.append(d)
                if d.confidence == "high" and d.resolved_code:
                    # Mutate the concept with the resolved code on the right
                    # ontology field. Only LOINC is wired into the SQL fast
                    # path today; other code systems get parked on the
                    # concept for downstream use but do NOT change routing.
                    if d.code_system == "loinc":
                        concept.loinc_code = d.resolved_code
                    concept.resolved_from_category = True
                else:
                    all_high = False
            if all_high and per_cq:
                self._disambiguations_resolved += 1
                cq.clarifying_question = None
            self._last_disambiguations[id(cq)] = per_cq

    def _critique(
        self,
        sub: AnswerResult,
        cq: CompetencyQuestion,
        *,
        fallback_warning: str | None = None,
    ) -> AnswerResult:
        """Apply the second-pass plausibility critic to a sub-answer.

        Returns the same AnswerResult mutated with ``critic_verdict`` set
        when the critic returns a verdict; ``critic_verdict`` stays None
        when the critic is disabled or fails (failure is logged in
        :func:`critic.critique`). Never raises.
        """
        if not self._enable_critic:
            return sub
        verdict = critique(self._client, cq, sub, fallback_warning=fallback_warning)
        if verdict is not None:
            sub.critic_verdict = verdict
        return sub

    def _contextualize(
        self,
        sub: AnswerResult,
        cq: CompetencyQuestion,
    ) -> AnswerResult:
        """Optionally append a literature-grounded context note to the answer.

        Only fires when ``enable_contextualization`` is True AND the critic
        verdict is None (no critic ran) OR severity == "info" (no warning
        to surface). Never overrides a critic warn/block — the critic's
        concern is more important to the user than a literature note.

        Phase F: when ``enable_sub_agent_in_contextualize`` is True, route
        the lookup through the HealthSourceOfTruthAgent sub-agent for
        richer cross-MCP grounding (PubMed + LOINC + MIMIC distribution).
        Otherwise use the simpler ``clinical_consult.contextualize``
        EvidenceAgent path.

        On any failure (contextualize() / sub-agent returns None), the
        answer is returned unchanged. Pure additive feature.
        """
        if not self._enable_contextualization:
            return sub
        v = sub.critic_verdict
        if v is not None and v.severity != "info":
            return sub  # don't drown a warn/block with a literature note

        if self._enable_sub_agent_in_contextualize:
            note = self._contextualize_via_sub_agent(sub, cq)
        else:
            note = contextualize(self._client, sub, cq)
        if note is None:
            return sub
        self._contextualizations += 1
        sep = "\n\n— Context —\n"
        sub.text_summary = (sub.text_summary or "") + sep + note.text
        if note.citations:
            sub.contextual_citations = list(note.citations)
        return sub

    def _contextualize_via_sub_agent(
        self,
        sub: AnswerResult,
        cq: CompetencyQuestion,
    ):
        """Use HealthSourceOfTruthAgent for contextualization.

        Returns a ContextualNote (or None) — same contract as
        ``clinical_consult.contextualize``.

        PHI invariant: only ``cq.original_question`` and the decomposer's
        ``interpretation_summary`` are forwarded to the sub-agent. The
        answer's ``data_table`` (which CAN contain MIMIC row data) is
        NEVER passed. Only the bare ``text_summary`` (which the answerer
        already sanitised for user display) is passed as a hint string.
        """
        from src.conversational.models import ContextualNote

        result = self._consult_health_source_of_truth(
            question=cq.original_question,
            context={
                "interpretation": cq.interpretation_summary,
                "answer_summary_excerpt": (sub.text_summary or "")[:500],
            },
        )
        if result is None or not result.answer_summary:
            return None
        # Map HealthAnswer findings into the ContextualNote.citations
        # shape (list of dicts mirroring CriticVerdict.cited_sources).
        citations: list[dict] = []
        for f in result.findings or []:
            for ev in f.evidence or []:
                citations.append({
                    "type": ev.source,
                    "id": ev.id,
                    "tool": ev.tool,
                })
        return ContextualNote(
            text=result.answer_summary,
            citations=citations or None,
        )

    def _consult_health_source_of_truth(
        self,
        question: str,
        context: dict | None = None,
    ):
        """Delegate a focused biomedical question to the source-of-truth
        sub-agent. Returns ``HealthAnswer | None``.

        Callers (currently: ``_contextualize_via_sub_agent``; future:
        critic, disambiguate) use this when they need cross-MCP grounding
        (PubMed + LOINC + MIMIC distribution + future SNOMED/RxNorm/etc).

        PHI safety: ``question`` and values in ``context`` MUST be PHI-
        free. The sub-agent NEVER receives data_table, hadm_ids, or
        subject_ids. The static signature check in
        ``test_sub_agent.py::TestPhiCompartmentalization`` enforces
        this at CI time.

        Never raises. Returns ``None`` on any failure. Increments
        ``self._sub_agent_consultations`` on success.
        """
        try:
            agent = HealthSourceOfTruthAgent(self._client)
            result = agent.consult(question, context=context)
            if result is not None:
                self._sub_agent_consultations += 1
            return result
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "_consult_health_source_of_truth failed: %s (%s)",
                exc, type(exc).__name__,
            )
            return None

    def _should_retry(
        self,
        sub: AnswerResult,
        cq: CompetencyQuestion,
        attempt: int,
        max_retries: int,
    ) -> bool:
        """Decide whether the SQL fast-path should retry with a corrected LOINC.

        Conservative: every condition must hold. Specifically the critic
        must have flagged a non-info severity AND emitted a non-null
        ``suggested_loinc`` AND the concept must be a biomarker AND the
        suggestion must differ from the LOINC already on the CQ (idempotent
        guard against degenerate loops).
        """
        if attempt >= max_retries:
            return False
        v = sub.critic_verdict
        if v is None or v.severity not in {"warn", "block"}:
            return False
        if not v.suggested_loinc:
            return False
        if not cq.clinical_concepts:
            return False
        c = cq.clinical_concepts[0]
        if c.concept_type != "biomarker":
            return False
        if v.suggested_loinc == c.loinc_code:
            return False  # idempotent guard
        return True

    def _run_with_critic_retry(
        self,
        cq: CompetencyQuestion,
        backend,
        *,
        resolved_names: list[str],
        max_retries: int | None = None,
    ) -> tuple[AnswerResult, list[str], str | None]:
        """SQL fast-path with one round of critic-driven LOINC correction.

        The flow:
          1. Run ``_run_sql_fastpath`` with the current CQ.
          2. Run the critic on the resulting answer.
          3. If ``_should_retry`` says yes, mutate
             ``cq.clinical_concepts[0].loinc_code`` to the suggested code
             and loop. Otherwise break.
          4. If more than one attempt happened, attach the trace to
             ``answer.correction_trace`` so the UI can render the journey.

        Bound by ``max_retries`` (default 1; configurable via constructor
        ``self._critic_max_retries``). The retry never fires when
        ``_enable_critic`` is False (no verdict to drive the decision)
        or on non-biomarker concepts (no LOINC mutation seam).

        Causal/similarity/graph branches do NOT use this method — they
        keep today's single-attempt behavior. The seam to extend self-
        healing to those branches is v2 (each has different failure
        modes and mutation points).
        """
        cap = self._critic_max_retries if max_retries is None else max_retries
        trace: list[dict] = []
        sub: AnswerResult | None = None
        sql_used: list[str] = []
        fb: str | None = None
        for attempt in range(cap + 1):
            loinc_used = (
                cq.clinical_concepts[0].loinc_code
                if cq.clinical_concepts else None
            )

            # Pre-execution validation (Phase B). Compile the SQL once,
            # ask the validator whether to proceed, then reuse the
            # compiled query for execution to avoid double-compile.
            validator_verdict = None
            preview_query = None
            preview_itemids: list[int] | None = None
            preview_fb: str | None = None
            if self._enable_pre_validator:
                preview_query, preview_itemids, preview_fb = (
                    self._compile_fastpath_preview(
                        cq, backend, resolved_names=resolved_names,
                    )
                )
                if preview_query is not None:
                    client = self._get_bq_validator_client()
                    if client is not None:
                        validator_verdict = validate_sql_deterministic(
                            preview_query,
                            mcp_client=client,
                            fallback_warning=preview_fb,
                            resolved_itemids=preview_itemids,
                            max_usd=self._pre_validator_max_usd,
                            max_bytes=self._pre_validator_max_bytes,
                            timeout=self._pre_validator_timeout,
                        )

            # Block path: short-circuit. No execute / answer / critic.
            if validator_verdict is not None and validator_verdict.verdict == "block":
                self._pre_validator_blocks += 1
                sub = AnswerResult(
                    text_summary=_format_validator_block_message(
                        validator_verdict,
                    ),
                )
                sub.interpretation_summary = cq.interpretation_summary
                fb = preview_fb
                sql_used = [preview_query.sql] if preview_query is not None else []
                trace.append({
                    "attempt": attempt,
                    "loinc_used": loinc_used,
                    "text_summary": sub.text_summary,
                    "fallback_warning": fb,
                    "blocked_pre_execution": True,
                    "validator_concern": validator_verdict.concern,
                    "validator_reference": validator_verdict.reference_used,
                    "critic_verdict": None,
                })
                break  # do not retry on block

            # Execute + answer (existing path, possibly mocked in tests).
            # Reuse the pre-compiled query when available to avoid
            # recompiling. When the validator path was skipped (or
            # compilation failed), _run_sql_fastpath compiles internally.
            sub, sql_used, fb = self._run_sql_fastpath(
                cq, backend, resolved_names=resolved_names,
                precompiled_query=preview_query,
                precompiled_fallback_warning=preview_fb,
            )
            sub.interpretation_summary = cq.interpretation_summary

            # Thread a warn verdict into the critic's fallback_warning so
            # the critic knows about the pre-flight concern.
            critic_fb = fb
            if validator_verdict is not None and validator_verdict.verdict == "warn":
                self._pre_validator_warns += 1
                note = (
                    f"pre-execution validator warned: {validator_verdict.concern}"
                )
                critic_fb = (fb + "\n" + note) if fb else note
            elif validator_verdict is not None and validator_verdict.verdict == "pass":
                self._pre_validator_passes += 1

            sub = self._critique(sub, cq, fallback_warning=critic_fb)
            trace.append({
                "attempt": attempt,
                "loinc_used": loinc_used,
                "text_summary": sub.text_summary,
                "fallback_warning": fb,
                "validator_verdict": (
                    validator_verdict.verdict
                    if validator_verdict is not None else None
                ),
                "critic_verdict": (
                    sub.critic_verdict.model_dump()
                    if sub.critic_verdict is not None else None
                ),
            })
            if not self._should_retry(sub, cq, attempt, cap):
                break
            try:
                cq.clinical_concepts[0].loinc_code = (
                    sub.critic_verdict.suggested_loinc
                )
            except Exception:
                logger.exception(
                    "self-healing critic: LOINC mutation failed; aborting retry"
                )
                break
        if sub is not None and len(trace) > 1:
            sub.correction_trace = trace
        return sub, sql_used, fb

    def _compile_fastpath_preview(
        self,
        cq: CompetencyQuestion,
        backend,
        *,
        resolved_names: list[str],
    ):
        """Compile the SQL fast-path query for pre-execution validation.

        Mirrors the resolution/compile steps that ``_run_sql_fastpath``
        runs internally, returning ``(query, resolved_itemids,
        fallback_warning)``. Returns ``(None, None, None)`` if compilation
        raises — in that case the validator is skipped and the orchestrator
        proceeds; the underlying error will surface from ``_run_sql_fastpath``
        when called.
        """
        try:
            resolved_itemids: list[int] | None = None
            fallback_warning: str | None = None
            if cq.clinical_concepts:
                concept = cq.clinical_concepts[0]
                if concept.concept_type == "biomarker" and concept.loinc_code:
                    biom = self._resolver.resolve_biomarker(concept)
                    resolved_itemids = biom.itemids
                    fallback_warning = biom.fallback_reason

            query = compile_sql(
                cq, backend, self._registry,
                resolved_names=resolved_names,
                resolved_itemids=resolved_itemids,
            )
            return query, resolved_itemids, fallback_warning
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pre-validator preview compile failed; skipping validation: %s",
                exc,
            )
            return None, None, None

    def _run_sql_fastpath(
        self,
        cq: CompetencyQuestion,
        backend,
        *,
        resolved_names: list[str],
        precompiled_query=None,
        precompiled_fallback_warning: str | None = None,
    ) -> tuple[AnswerResult, list[str], str | None]:
        """Compile and execute a SQL fast-path CQ; wrap in AnswerResult.

        Returns (answer, [sql_text], fallback_warning):
          - answer: the AnswerResult with text_summary etc.
          - [sql_text]: SQL we ran (surfaced in Query Details).
          - fallback_warning: structured note when LOINC grounding failed,
            or None. Returned alongside (rather than only embedded in
            text_summary) so the orchestrator can pass it to the critic
            for second-pass plausibility review.

        Biomarker concepts that carry a LOINC code go through
        ``ConceptResolver.resolve_biomarker`` before compilation so the
        WHERE clause can use ``itemid IN (...)`` rather than a
        unit-pooling ``LIKE`` filter. When grounding fails (LOINC absent
        from the mapping table or no MIMIC labitem coverage), we fall
        back to LIKE and surface a visible warning to the user via
        ``AnswerResult.text_summary`` AND to the critic via the third
        return value.

        ``precompiled_query`` / ``precompiled_fallback_warning`` are an
        optimization seam used by ``_run_with_critic_retry``: when the
        pre-validator already compiled the query for its preview, we pass
        the result through to avoid recompiling. When None (the legacy
        external-caller path), this method does its own compile.
        """
        if precompiled_query is None:
            resolved_itemids: list[int] | None = None
            fallback_warning: str | None = None
            if cq.clinical_concepts:
                concept = cq.clinical_concepts[0]
                if concept.concept_type == "biomarker" and concept.loinc_code:
                    biom = self._resolver.resolve_biomarker(concept)
                    resolved_itemids = biom.itemids
                    fallback_warning = biom.fallback_reason

            query = compile_sql(
                cq, backend, self._registry,
                resolved_names=resolved_names,
                resolved_itemids=resolved_itemids,
            )
        else:
            query = precompiled_query
            fallback_warning = precompiled_fallback_warning
        raw_rows = backend.execute(query.sql, query.params)
        rows = [dict(zip(query.columns, r)) for r in raw_rows]
        answer = generate_answer(
            self._client, cq, rows,
            {},  # no graph_stats on the fast-path
            [query.sql],  # surface the SQL alongside any SPARQL
        )
        if fallback_warning:
            # Append a user-visible note explaining the answer may pool
            # variants. Phase 9b proper warning surface lives behind
            # AnswerResult; for now we co-locate with text_summary so the
            # warning is visible in chat without a model change.
            answer.text_summary = (
                (answer.text_summary or "")
                + f"\n\n⚠️ Note: {fallback_warning}"
            ).strip()
        return answer, [query.sql], fallback_warning

    def _run_causal(self, cq: CompetencyQuestion) -> AnswerResult:
        """Phase 8a: wrap a ``CausalEffectResult`` into an ``AnswerResult``.

        The wrapping is deliberately minimal — just enough so the existing
        Streamlit UI renders *something* recognisable on a CAUSAL plan.
        Phase 8i replaces this with a proper causal-effect panel (pairwise
        τ heatmap, per-outcome breakdown, diagnostic block, mode badge).
        The stub-vs-real distinction is surfaced to the user via the
        ``is_stub`` flag bleeding into the text summary so no one mistakes
        8a output for a real estimate.
        """
        from src.causal.run import run_causal

        result = run_causal(cq)
        if result.is_stub:
            summary = (
                "Causal-inference pipeline is wired (Phase 8a). Schema and "
                "routing are in place, but the estimator itself lands in "
                "Phase 8d — this result is a shape-only placeholder with "
                "NaN point estimates. The final system will return μ_c "
                "point estimates + uncertainty intervals + pairwise τ "
                "contrasts + diagnostics here."
            )
        else:
            # Phase 8d+ summary; 8a never reaches this branch.
            summary = (
                f"Causal-effect estimates for {len(result.mu_c)} interventions "
                f"(mode={result.mode}). See the causal panel for pairwise "
                f"contrasts and diagnostics."
            )

        # Flatten mu_c into a data_table the existing UI can render.
        data_table = [
            {
                "intervention": label,
                "mu_point": ui.point,
                "mu_lower": ui.lower,
                "mu_upper": ui.upper,
            }
            for label, ui in result.mu_c.items()
        ]
        return AnswerResult(
            text_summary=summary,
            data_table=data_table,
            table_columns=["intervention", "mu_point", "mu_lower", "mu_upper"],
        )

    def _run_similarity(self, cq: CompetencyQuestion, backend) -> AnswerResult:
        """Phase 9: wrap a ``SimilarityResult`` into an ``AnswerResult``.

        Lazy import of ``run_similarity`` mirrors the ``_run_causal``
        pattern above — keeps the heavy ``src.similarity`` graph out of
        the orchestrator import path until a similarity CQ actually
        arrives. The flat 6-column data_table is what the existing
        Streamlit dataframe renderer in ``src/conversational/app.py``
        already knows how to display; richer per-bucket breakdowns
        ride along on ``SimilarityScore.contextual_explanation`` and
        ``temporal_explanation`` for a future explanation panel
        (Phase 9b).
        """
        from src.similarity.run import run_similarity

        if cq.similarity_spec is None:
            # Defensive: scope='patient_similarity' but no spec means the
            # decomposer produced an underspecified CQ. Surface a plain
            # message rather than crashing or silently mis-routing.
            return AnswerResult(
                text_summary=(
                    "This question is scoped to patient similarity but no "
                    "similarity specification was produced. Please rephrase "
                    "with a clearer anchor (a specific patient or a "
                    "covariate template)."
                ),
            )

        result = run_similarity(cq.similarity_spec, backend)
        data_table = [
            {
                "rank": i + 1,
                "hadm_id": s.hadm_id,
                "subject_id": s.subject_id,
                "combined": s.combined,
                "contextual": s.contextual,
                "temporal": s.temporal,
            }
            for i, s in enumerate(result.scores)
        ]
        summary = (
            f"Found {result.n_returned} of {result.n_pool} candidates "
            f"similar to {result.anchor_description}. "
            f"See the table for ranked combined / contextual / temporal "
            f"scores."
        )
        return AnswerResult(
            text_summary=summary,
            data_table=data_table,
            table_columns=[
                "rank", "hadm_id", "subject_id",
                "combined", "contextual", "temporal",
            ],
        )

    def _extract_one(self, cq: CompetencyQuestion):
        """Legacy single-CQ extractor kept for external callers. Modern
        code inside ``ask()`` uses ``_extract(backend, cq, ...)`` directly
        so the backend is shared across sub-CQs."""
        if self._data_source == "bigquery":
            return extract_bigquery(
                cq, project=self._bigquery_project,
                config=self._extraction_config,
                resolver=self._resolver,
            )
        return extract(
            self._db_path, cq, config=self._extraction_config,
            resolver=self._resolver,
        )

    def _remember(self, cq: CompetencyQuestion, answer: AnswerResult) -> None:
        """Append to conversation_history, enforcing ``max_history``."""
        self.conversation_history.append((cq, answer))
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def _get_bq_validator_client(self) -> McpClient | None:
        """Lazy bq-validator MCP client.

        Returns None if no config was supplied AND the default server
        path is unavailable — the orchestrator gracefully proceeds
        without validation in that case rather than crashing the turn.

        The client is reused across all calls in this pipeline's lifetime
        (subprocess respawn cost is high). It's cleaned up via
        ``ConversationalPipeline.close()`` and the McpClient atexit hook.
        """
        if self._bq_validator_client is not None:
            return self._bq_validator_client

        config = self._bq_validator_mcp_config
        if config is None:
            from pathlib import Path
            import sys

            repo_root = Path(__file__).parent.parent.parent
            server_path = repo_root / "mcp_servers" / "bq_validator" / "server.py"
            if not server_path.exists():
                logger.info(
                    "bq-validator server not found at %s; skipping validation",
                    server_path,
                )
                return None
            config = McpServerConfig(
                name="bq-validator",
                transport="stdio",
                command=sys.executable,
                args=[str(server_path)],
            )

        try:
            self._bq_validator_client = McpClient(config)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to create bq-validator McpClient: %s", exc,
            )
            return None
        return self._bq_validator_client

    def close(self) -> None:
        """Tear down any owned MCP client connections.

        Safe to call multiple times. Called automatically at process exit
        via the McpClient atexit hook, but callers may invoke directly
        for cleaner test isolation."""
        if self._bq_validator_client is not None:
            try:
                self._bq_validator_client.close()
            except Exception:  # noqa: BLE001
                pass
            self._bq_validator_client = None

    def reset(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()


def _format_validator_block_message(verdict) -> str:
    """User-facing text when the pre-execution validator blocks a query.

    The text explains why the query was not run, which gives the user a
    concrete next step (rephrase, supply a more specific concept, etc.).
    """
    parts = [
        "Query was blocked by the pre-execution validator before running.",
    ]
    if verdict.concern:
        parts.append(f"\nReason: {verdict.concern}")
    if verdict.suggested_fix:
        parts.append(f"\nSuggested correction: {verdict.suggested_fix}")
    return "\n".join(parts)


def create_pipeline_from_settings() -> ConversationalPipeline:
    """Create a pipeline from config/settings.py defaults."""
    from config.settings import Settings

    settings = Settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set in .env or environment.")

    ontology_dir = Path(__file__).parent.parent.parent / "ontology" / "definition"

    return ConversationalPipeline(
        db_path=settings.duckdb_path,
        ontology_dir=ontology_dir,
        api_key=settings.anthropic_api_key,
    )
