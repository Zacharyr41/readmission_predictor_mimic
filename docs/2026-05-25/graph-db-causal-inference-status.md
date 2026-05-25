# Graph-DB-Backed Causal / Association Pipeline: Status & Path Forward

_Drafted 2026-05-25. Re-frames the earlier same-day snapshot around the actual
envisioned end-to-end pipeline, rather than asking the narrower "is Neo4j
wired?" question in isolation._

## Context

This repo is meant to support two lines of usage:

1. **Tabular SQL queries** — mostly implemented today via the SQL fast-path
   (`src/conversational/sql_fastpath.py`) and the SPARQL `GRAPH` plan.
2. **Graph-DB construction + causal / association queries** — the line this
   document addresses. The envisioned flow is:

   1. User asks a complex / causal / association question in chat. Example:
      *"Among sepsis patients who received antibiotics within 24 hours of
      admission, what factors were associated with readmission within 30
      days?"*
   2. The system constructs a graph from MIMIC data per the
      **Vannieuwenhuyze et al. (2025)** algorithm
      (`README.md:408`, `README.md:425-427`), scoped to the cohort the
      question implies.
   3. The causal-inference layer queries that graph — using temporal and
      contextual edges — and structures the queries so answers are causally
      meaningful, not merely descriptive.
   4. The result is rendered back in the chat UI.

This document is the status snapshot for line 2, with a "prior-to-wiring"
architectural recommendation. It is intentionally before any code change —
the point is to align on the seams.

---

## Algorithm reference (what the graph is supposed to be)

- **Citation**: `README.md:408` —
  *Vannieuwenhuyze, A., Mimouni, N., & Du Mouza, C. (2025). "A Conceptual
  Model for Discovering Implicit Temporal Knowledge in Clinical Data". In
  Advances in Conceptual Modeling (ER 2025), LNCS vol. 16190. Springer.
  DOI: 10.1007/978-3-032-08620-4_6.*
  Original implementation: `README.md:410`
  (github.com/avannieuwenhuyze/clinical-tkg-cmls2025).
- **What it specifies**: OWL-Time-compliant RDF knowledge graph; clinical
  events modeled as temporal instants or proper intervals; Allen interval
  relations (before / meets / overlaps / during / starts / finishes) computed
  between pairwise events within an ICU stay. Summarized at
  `docs/architecture.md:5-9` and `docs/ontology.md:1-5`.

The codebase **does implement** this algorithm. Node writers live in
`src/graph_construction/event_writers.py` (BioMarkerEvent at lines 212–273,
ClinicalSignEvent at 276–329, MicrobiologyEvent at 332–383, PrescriptionEvent
at 389–455, DiagnosisEvent at 461–499). Allen relations are computed in
`src/graph_construction/temporal/allen_relations.py:25-102` with the six
OWL-Time predicates mapped at lines 15–22.

---

## Current state of the 4-step pipeline

Legend: **WIRED** runs end-to-end · **PARTIAL** runs but narrower than the
vision · **DEAD** code exists but unreachable from chat · **ABSENT** nothing
matches.

### Step 1 — Question intake & classification

- **WIRED**: Chat input — `src/conversational/app.py:331`
  (`st.chat_input("Ask a clinical question...")`).
- **WIRED**: LLM decomposition into a structured `CompetencyQuestion` —
  `src/conversational/decomposer.py:169` (`decompose_question`), called from
  `src/conversational/orchestrator.py:203`.
- **WIRED**: Plan routing — `src/conversational/planner.py:84-151`. Four
  plans: `SQL_FAST`, `GRAPH`, `CAUSAL`, `SIMILARITY`
  (`planner.py:43-65`).
- **PARTIAL / mis-routed for association questions**: The decomposer
  recognizes only four scopes (`single_patient`, `cohort`, `comparison`,
  `causal_effect` — `src/conversational/prompts.py:260-289`). The example
  *"what factors were associated with"* maps to none of them cleanly. The
  `causal_effect` scope additionally requires `intervention_set` with
  `|I| ≥ 2` (`src/conversational/models.py:227`, validator at
  `models.py:261`; enforced at `planner.py:102-105`). Association-discovery
  questions have no explicit intervention set and fall through to
  `QueryPlan.GRAPH` (`planner.py:110-151`).

**Gap**: no `ASSOCIATION` / `EXPLORATORY` scope or plan; the decomposer
prompt has no template for *"find factors predicting outcome Y in cohort C"*.

### Step 2 — Per-question graph construction

- **WIRED**: `build_query_graph(ontology_dir, extraction, ...)` —
  `src/conversational/graph_builder.py:39`, invoked once per question at
  `src/conversational/orchestrator.py:334`. Materializes an in-memory
  rdflib graph from an `ExtractionResult`, then is discarded after the turn.
- **PARTIAL**: cohort extraction — `src/conversational/extractor.py` (called
  via `orchestrator.py:316-322`). Filters by **clinical concepts**
  (diagnosis = sepsis, drug = antibiotic, etc.) but does **not** filter
  admissions by multi-event predicates such as *"sepsis AND antibiotic
  starttime within 24h of admittime"*. The 24-hour window is currently
  expressible only as a SPARQL post-filter on the materialized graph, not
  as a cohort-inclusion criterion at extraction time.
- **PARTIAL**: edge coverage —
  - Structural / containment edges fully present:
    `hasAdmission` (`patient_writer.py:103`), `containsICUStay`
    (`event_writers.py:136`), `hasICUDay` / `partOf`
    (`event_writers.py:197-198`), `hasICUStayEvent` /
    `hasICUDayEvent` (e.g. `event_writers.py:265, 271`), `hasDiagnosis` /
    `diagnosisOf` (`event_writers.py:496-497`), `followedBy`
    (`patient_writer.py:109-119`).
  - Six Allen relations present
    (`temporal/allen_relations.py:15-22, 25-102`).
  - `readmittedWithin30Days` / `readmittedWithin60Days` already attached to
    each admission node (`patient_writer.py:93`).
- **DEAD**: comorbidities. `write_comorbidity(...)` is defined at
  `src/graph_construction/event_writers.py:502` but is **not imported by
  `graph_builder.py`** (see imports at `graph_builder.py:19-27` — seven
  writers, no comorbidity) and not called from
  `src/graph_construction/pipeline.py` either. So comorbidity nodes never
  enter the graph today.
- **ABSENT**: medication categorization. Drug names live as free-text
  literals on `PrescriptionEvent.hasDrugName`
  (`event_writers.py:389-455`); there is no RxNorm-class / ATC mapping that
  would resolve *"antibiotic"* to a set of ingredients at query time. The
  SNOMED mapper (`terminology/snomed_mapper.py`) exposes lookups but is
  only used by the dead `write_comorbidity` path.

### Step 3 — Causal / association query against the graph

This is the load-bearing gap.

- **DEAD for this question shape**: `src/causal/run.py:run_causal` (line
  ~102) is fully implemented for *treatment-effect* questions: it assembles
  a cohort DataFrame via `src/causal/cohort.py:build_cohort_frame`, fits
  S/T/X-learner metalearners (`src/causal/estimators/metalearners.py`), and
  returns a `CausalEffectResult` with point estimates + bootstrap CIs. It
  is invoked at `src/conversational/orchestrator.py:962-1007`
  (`_run_causal`). But the planner only routes here when
  `scope=="causal_effect"` AND `|interventions| ≥ 2`
  (`planner.py:102-105`). The example association question fails both
  gates.
- **ABSENT — operates on the graph**: even when `run_causal` *is* invoked,
  it queries the **SQL backend** (DuckDB / BigQuery), not the per-question
  RDF graph from step 2. A grep across `src/causal/` finds no `neo4j`, no
  `Cypher`, no SPARQL, no rdflib usage, no DAG / backdoor / adjustment-set
  / d-separation logic. The only `confound` hit is a forward-looking
  comment at `src/causal/estimators/base.py:333` ("8h adds
  unmeasured-confounder sensitivity").
- **ABSENT — SPARQL association templates**:
  `src/conversational/reasoner.py` is the SPARQL template engine. Its
  templates handle value lookups, patient lists, comparisons, and Allen
  temporal patterns — but a grep for `association`, `factor.*associated`,
  `risk.*factor`, `odds.*ratio`, `relative.*risk` across `reasoner.py` and
  `prompts.py` returns **no matches**. There is no template that computes
  bivariate or multivariable association measures between graph-derived
  features and an outcome node.

### Step 4 — Display

- **WIRED**: result rendering — `src/conversational/app.py:195-312`
  (`_render_answer`) handles `AnswerResult` with `text_summary`,
  `data_table`, optional `visualization_spec`, and a query-details
  expander.
- **PARTIAL**: result schemas. Today the UI assumes one of two shapes:
  causal-intervention table
  (`[intervention, mu_point, mu_lower, mu_upper]` — produced by
  `orchestrator.py:_run_causal`), or generic SPARQL rows
  (`reasoner.reason()` output). There is no schema for *"ranked factors
  with effect sizes and CIs"*, so even if step 3 produced one, step 4
  would render it as a generic table without the right framing.

---

## The single root cause

The example clinician question is an **association-discovery** question
(*"what factors are associated with outcome Y in cohort C?"*), not a
treatment-effect question (*"what is the effect of intervention I on
outcome Y?"*). The repo has the latter (`src/causal/`) but not the former.
Closing the loop the user describes is mostly about adding the
association-discovery path **and** wiring it to consume the per-question
graph that already exists. The graph itself is closer to ready than the
causal / association layer that should consume it.

---

## Recommended approach (prior to any wiring)

These are the minimum architectural decisions to lock in before writing
code. They are *not* yet an implementation plan — they are the design seams
to align on so the eventual wiring is coherent.

### 1. Add an `ASSOCIATION` scope and plan, distinct from `CAUSAL`

- Extend `prompts.py:260-289` decomposer scopes with `"association"` for
  questions of the form *"what factors are associated with / predict
  outcome Y in cohort C?"*.
- Add `QueryPlan.ASSOCIATION` to `planner.py:43-65`. Its classifier
  signature: a `CompetencyQuestion` with `scope=="association"`, a
  non-trivial cohort filter, and an outcome field (no required
  intervention_set).
- Keep `QueryPlan.CAUSAL` strictly for `|I| ≥ 2` treatment-effect
  questions — do not overload it.

### 2. Make the graph the substrate for `ASSOCIATION` and (eventually) `CAUSAL`

- The per-question graph already exists
  (`graph_builder.build_query_graph`, called at `orchestrator.py:334`). The
  decision is whether `ASSOCIATION` / future `CAUSAL` queries should:
  - **(a)** consume the same in-memory rdflib graph via SPARQL, or
  - **(b)** push that graph into Neo4j and use Cypher
    (`src/graph_analysis/neo4j_import.py:36-111` is ready), or
  - **(c)** stay on SQL with the graph only providing the cohort.
- **Recommendation: (a) first.** The rdflib path is in-process, has no
  service dependency, and already runs per-question. Defer Neo4j until
  there is a concrete reason (scale, interactive exploration, etc.).
  Document the decision so future contributors don't relitigate it.

### 3. Define a graph-backed cohort + outcome contract

The graph already encodes admissions, prescriptions, diagnoses, ICU stays,
biomarkers, vitals, microbiology, Allen relations, and the
`readmittedWithin30Days` outcome flag (`patient_writer.py:93`). To support
the example question end-to-end the contract needs:

- **Cohort filter**: multi-event SPARQL pattern *"admission A has a
  DiagnosisEvent with ICD ∈ sepsis-set, AND has a PrescriptionEvent P with
  hasDrugName ∈ antibiotic-set such that P.starttime is within 24h of
  A.admittime"*. The 24h check uses Allen `intervalStarts` /
  `intervalMeets` or an xsd:dateTime arithmetic filter. This needs to land
  as either (i) a reusable SPARQL fragment in `reasoner.py`'s template
  library, or (ii) a new `cohort_predicate` field on `CompetencyQuestion`
  that decomposes into such a fragment.
- **Outcome accessor**: SPARQL that selects each admission with its
  `readmittedWithin30Days` boolean.
- **Feature enumerator**: SPARQL that, for each admission in the cohort,
  emits feature values — biomarkers, vitals, diagnosis flags, drug flags,
  demographics — keyed by `hadm_id`. This is the equivalent of
  `src/causal/covariates.py:build_covariate_matrix` but reading from the
  graph rather than DuckDB.

### 4. Fill the two cheap graph gaps before designing further

These reduce uncertainty in the contract above:

- **Wire `write_comorbidity`** into `graph_builder.py` and
  `graph_construction/pipeline.py`. Currently dead at
  `event_writers.py:502`; one import + one call site each. Without
  comorbidities in the graph, *"factors associated with readmission"* is
  missing the most clinically obvious feature family.
- **Resolve drug categories at query time**. Add a step (in the decomposer
  or extractor) that expands *"antibiotic"* into a concrete set of
  ingredient names / RxNorm CUIs / generic-name patterns before the cohort
  SPARQL runs. Hook into the existing
  `src/conversational/concept_resolver.py` rather than introducing a new
  resolution path.

### 5. Define the association estimator surface (consume X, T-or-cohort, Y)

The estimators in `src/causal/estimators/` already operate on numpy arrays
(`metalearners.py` consumes X, T, Y arrays from
`cohort.build_cohort_frame`). To reuse them for association discovery:

- Add an association-mode shim in `src/causal/run.py` that, given a CQ
  with `scope=="association"`, calls a new
  `build_cohort_frame_from_graph(cq, graph)` (parallel to the existing SQL
  `build_cohort_frame`) and runs bivariate + multivariable models per
  candidate feature, ranking by effect size with bootstrap CIs.
- The estimators themselves are backend-agnostic; only the cohort assembly
  is SQL-bound today (`cohort.py`). Keep the estimator code untouched;
  introduce a parallel cohort module rather than refactoring the existing
  protocol.

### 6. Result schema + rendering for ranked-factor output

- New result type (extension of `AnswerResult` in `models.py`) with a
  `factors: list[FactorRow]` field where each row has `name`,
  `effect_size`, `ci_lower`, `ci_upper`, `n_exposed`, `n_unexposed`,
  optional `p_value`.
- Render in `app.py:_render_answer` as a sorted table with explicit
  framing language: "Factors *associated with* readmission (not proven
  causal)."

### 7. Explicit non-goals for this phase

- **Not yet**: Neo4j wiring. Treat `src/graph_analysis/neo4j_import.py` as
  future infrastructure; defer until the rdflib path is exercised.
- **Not yet**: DAG discovery / backdoor-criterion adjustment-set
  derivation. The Phase 8 spec (`memory/project_phase8_causal_spec.md`)
  locks the Neyman–Rubin framework with hand-supplied covariates. Honor
  that for `CAUSAL`; association-discovery does not need a DAG yet.

---

## Critical files (read these before any wiring)

- `README.md:408, 425-427` — Vannieuwenhuyze citation and BibTeX.
- `docs/architecture.md:5-9`, `docs/ontology.md:1-5` — algorithm
  summaries.
- `src/conversational/graph_builder.py:39-100` — `build_query_graph` entry
  point (the seam to reuse).
- `src/conversational/orchestrator.py:181-350` — chat → CQ → extraction →
  graph → reasoner pipeline.
- `src/conversational/planner.py:43-151` — plan routing (extension point
  for `ASSOCIATION`).
- `src/conversational/decomposer.py:169`,
  `src/conversational/prompts.py:260-289` — scope inference and prompt
  templates.
- `src/conversational/reasoner.py` — SPARQL templates (where association
  templates would live).
- `src/conversational/models.py:227, 261` — `CompetencyQuestion` schema
  and the intervention_set validator.
- `src/causal/run.py:102-279`, `src/causal/cohort.py`,
  `src/causal/estimators/metalearners.py` — existing treatment-effect
  path; estimator code is reusable for association mode.
- `src/graph_construction/event_writers.py:212-499, 502-532` — node
  writers; line 502 is the dead `write_comorbidity`.
- `src/graph_construction/temporal/allen_relations.py:15-22, 25-102` —
  six-relation Allen computer.
- `src/graph_construction/patient_writer.py:93` —
  `readmittedWithin30Days` predicate already on the graph.

---

## Verification (when implementation begins)

The bar for *"the 4-step pipeline works"* is a single integration test of
the example question end-to-end. Suggested staging:

1. **Algorithm fidelity unit tests** — `tests/test_graph_construction/`
   already covers Allen relations and event writers. Add tests confirming
   `write_comorbidity` is invoked by both `build_query_graph` and
   `pipeline.build_graph` once wired.
2. **Cohort SPARQL test** — given a fixture extraction containing 5
   sepsis admissions (3 with antibiotic-within-24h, 2 without), the new
   cohort fragment must return exactly the 3.
3. **Feature enumerator test** — for those 3 admissions, the enumerator
   must return rows keyed by hadm_id with biomarker / vital / diagnosis /
   drug / demographic columns populated where present in the graph.
4. **Association estimator test** — feed a synthetic X, Y where the true
   associations are known (e.g., feature 1 OR ≈ 2.0, feature 2 OR ≈ 1.0);
   confirm the estimator ranks feature 1 above feature 2 with bootstrap
   CIs of the right sign.
5. **End-to-end dashboard test** — extend the Tier 1 AppTest harness
   (`tests/dashboard/`, per recent commits `5b29c0b`, `ff5b3a8`) with the
   example question; assert that the rendered answer contains a
   ranked-factor table and the "associated, not proven causal" framing.
6. **Live demo on the WLST or neuro cohort** — use the dataset already
   covered by `wlst_e2e_report.md` / `wlst_full_cohort_report.md` to
   confirm the pipeline produces clinically sensible factor rankings
   (subjective check, but necessary).

Run order: `pytest tests/test_graph_construction tests/test_conversational
tests/test_causal -x`, then the dashboard suite with `RUN_LIVE_DASHBOARD=1`
once that gate exists.
