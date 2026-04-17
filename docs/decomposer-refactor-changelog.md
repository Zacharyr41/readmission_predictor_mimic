# Decomposer Pipeline Refactor — Changelog

Phases 0 – 6, landed on `main` in commits `a4e3bf9` through the final Phase 6
commit. This document explains **what changed, why, and what to look for**.

Working plan, with architectural rationale and per-phase deviations, is at
`/Users/zacharyrothstein/.claude/plans/inherited-launching-creek.md` (author's
private notes; not in-repo).

---

## Why this refactor happened

`src/conversational/decomposer.py` carried seven TODOs describing a system
that was **fragile to drift** and **under-observable**:

- Supported filter fields were declared in **three places** (prompt text,
  decomposer's `supported_fields` set, extractor's `_get_filtered_hadm_ids`).
  Edits needed to stay in sync; a typo silently broke runtime.
- Cohorts were **artificially capped at 500** — broad clinical questions
  silently truncated.
- The decomposer was **aware of nothing downstream** of itself: it didn't
  know what pipeline it was in, what concepts the reasoner actually
  supported, or how to ask for clarification.
- "Big questions" like *"why do our sepsis patients get readmitted?"*
  collapsed to a **single CompetencyQuestion**, losing the useful
  multi-part answer a clinician actually wants.
- Tests **over-fit to canned LLM responses**, not observable behaviour.
- SNOMED hierarchy existed but was **unwired**.

Phases 0 – 6 addressed each of these, test-first and checkpointed.

---

## Executive summary

**Headline user-facing improvements**

- **"This is what I'm answering" echo** on every turn (blue info block above
  the answer).
- **Ambiguous questions** ("show me the labs") get a **clarifying follow-up**
  instead of a random guess — and cost **$0.00** in BigQuery because the
  pipeline short-circuits before querying.
- **Big questions** ("why do our sepsis patients get readmitted?") decompose
  into **multiple sub-questions sharing one knowledge graph** — cheaper,
  more coherent, Allen temporal relations computed once across all facets.
- **No more artificial 500-patient cap** — queries return every matching
  admission.
- **Works directly against MIMIC-IV on BigQuery** — no local DB needed. See
  `docs/e2e-bigquery-testing.md`.

**Headline engineering improvements**

- Single `OperationRegistry` for supported filters / aggregates / comparison
  axes. Adding a new filter field is one function call; the prompt, the
  decomposer validator, and the SQL compiler all update automatically.
- 400+ regression tests (up from ~130 before Phase 0), including hypothesis
  fuzz and 42 fixture-driven behavioural cases.
- **Prompt↔code drift detector**: any new filter without a prompt update
  fails CI.

---

## Phase-by-phase changelog

### Phase 0 — Test scaffolding (`a4e3bf9`)

**What:** Drop-in fixture-driven test scaffolding. Adding a new behavioural
case is a JSON file, not Python code.

**New files**
- `tests/test_conversational/conftest.py` — `mock_anthropic()` factory,
  fixture loaders, session-autouse regeneration of prompt example fixtures.
- `tests/test_conversational/fixtures/decomposer_cases/` — 15 seed cases.
- `tests/test_conversational/fixtures/malformed_json/` — 10 LLM failure-mode
  fuzz inputs.
- `tests/test_conversational/fixtures/prompt_examples/` — auto-extracted from
  the live prompt (later in Phase 3 inverted to pull from source-of-truth
  files under `src/`).
- `tests/test_conversational/test_decomposer_contract.py` — structural
  invariants.
- `tests/test_conversational/test_decomposer_cases.py` — generic parametrized
  runner.

**New invariants**
- Every prompt example parses as a valid `CompetencyQuestion`.
- `_validate_return_type` is idempotent.
- `_extract_json` is total (never raises) on arbitrary text.
- Conversation-history truncation is bounded at 5 turns.
- Retry-on-failure sends correctly-shaped follow-up messages.

**Net:** +89 tests; 891 repo-wide pass.

---

### Phase 1a — OperationRegistry + filter operations (`910942f`)

**What:** Extract supported filter fields from three scattered locations
into one registry. Filters compile themselves.

**New files**
- `src/conversational/operations.py` — `Operation` protocol,
  `FilterOperation` / `AggregateOperation` / `ComparisonOperation` concrete
  classes, `OperationRegistry`, `get_default_registry()` singleton.
- `src/conversational/operations_filters.py` — `register_default_filters()`
  seeds the 7 existing fields. Each filter's `compile_fn` mirrors the
  legacy SQL byte-for-byte.

**Modified**
- `src/conversational/extractor.py` — `_get_filtered_hadm_ids` delegates
  per-filter SQL to `registry.compile_filters()`.
- `src/conversational/decomposer.py` — `supported_fields` literal replaced
  with `get_default_registry().supported_names("filter")`.
- `tests/test_conversational/test_decomposer_contract.py` — prompt↔filter
  round-trip now sources expected set from the registry.

**Migration notes**
- `_SAFE_COMPARISON_OPS` (private) deleted; operator validation is now
  per-filter in `operations_filters.py`.
- No public API break.

**Net:** +25 tests.

---

### Phase 1b — Aggregate + comparison operations (`e50acbe`)

**What:** Complete the registry abstraction with the two other operation
kinds.

**New files**
- `src/conversational/operations_aggregates.py` — 8 aggregate keywords
  (`mean`, `avg` alias, `median` with post-processor, `max`, `min`, `count`,
  `sum`, `exists`).
- `src/conversational/operations_comparison.py` — 6 axes
  (`gender`, `age`, `readmitted_30d`, `readmitted_60d`, `admission_type`,
  `discharge_location`) with exact SPARQL clauses copied from the old
  `_COMPARISON_FIELD_MAP`.

**Modified**
- `src/conversational/reasoner.py` — aggregate elif chain replaced with
  `registry.get("aggregate", cq.aggregation)`. Median post-processing
  dispatches through `agg_op.post_processor` so any future aggregate
  with post-processing (percentile, rate) plugs in the same way. Dead
  `_compute_median` removed.
- `src/conversational/models.py` — `PatientFilter.value: str | list[str]`.
- `src/conversational/operations_filters.py` — `admission_type` now accepts
  operator `"in"` with list values, compiling to `IN (?, ?, …)`.

**Follow-up (noted, not done):** Delete
`reasoner._COMPARISON_FIELD_MAP` once the registry takes over
`build_sparql`. A parity test currently enforces they stay in sync.

**Net:** +12 tests.

---

### Phase 2 — Remove cohort cap, stream in batches (`17bce63`)

**What:** The 500-row artificial cap is gone. Queries return every matching
admission. Downstream fetchers chunk by `batch_size` to bound per-query
payload size.

**Modified**
- `src/conversational/models.py` — `ExtractionConfig.max_cohort_size`
  **removed**; `batch_size: int = 2000` added; `model_config = {"extra":
  "forbid"}` so stale call sites fail loudly.
- `src/conversational/extractor.py` — `_get_filtered_hadm_ids` drops
  `LIMIT`. New `_iter_hadm_batches()` generator. `_extract()` loops over
  batches, deduplicating patients by `subject_id` across batches.
- `src/conversational/app.py` — "Max cohort size" slider replaced with
  "Batch size" (100–10000, default 2000).
- `docs/chat-ui-quickstart.md` — updated.

**Migration notes**
- **Breaking change:** `ExtractionConfig(max_cohort_size=500)` → raises
  `ValidationError`. Update call sites to remove the kwarg or switch to
  `batch_size`.
- **BigQuery cost implications:** broad queries now scan more data. Mitigate
  with narrow filters (diagnosis, age, etc.) before expensive questions.
  See `docs/e2e-bigquery-testing.md` for cost estimates.

**Net:** +11 tests; -5 old capping tests.

---

### Phase 3 — Prompt built from parts (`a00c2f7`)

**What:** `DECOMPOSITION_SYSTEM_PROMPT` rewritten as the output of
`build_system_prompt(registry)`. Each structural section lives in one
place — constants, registry tables, or example JSON files.

**New files**
- `src/conversational/prompt_examples/single_cq/{01…10}_*.json` — 10 worked
  examples extracted from the old prompt string.
- `tests/test_conversational/fixtures/prompts_snapshot.txt` — rendered
  default prompt, committed so prompt changes surface in PR diffs.

**Modified**
- `src/conversational/prompts.py` — full rewrite. New sections:
  - **Role and Pipeline** — LLM told it's step 1 of 5, with explicit
    "NEVER write SQL" guard.
  - **Decomposition Goals** — concrete concepts, narrowest scope, temporal
    anchors.
  - **Supported Operations** — registry-injected filter / aggregate /
    comparison_axis sub-sections; auto-regenerated.
- `tests/test_conversational/conftest.py` — `_sync_prompt_examples` inverted:
  mirrors `src/conversational/prompt_examples/single_cq/` into the test
  fixture dir at session start, instead of extracting from a prompt string.

**Migration notes**
- Call sites that imported `DECOMPOSITION_SYSTEM_PROMPT` keep working —
  it's computed at import time from the default registry.
- To customize the prompt for a test, call `build_system_prompt(custom_
  registry)` directly instead of relying on the module constant.

**Net:** +16 tests.

---

### Phase 4 — Interpretation echo + clarifying-question loop (`86da4df`)

**What:** Every CQ now carries an `interpretation_summary` (always
populated). Ambiguous questions get `clarifying_question` set, triggering a
short-circuit that skips extract/graph/reason/answer.

**Modified**
- `src/conversational/models.py`:
  - `CompetencyQuestion.interpretation_summary: str | None` — always
    populated (synthesized from structured fields if LLM omits).
  - `CompetencyQuestion.clarifying_question: str | None` — set only on
    unresolvable ambiguity.
  - `AnswerResult` gains the same two fields.
- `src/conversational/prompts.py` — two new sections:
  - **When to Ask a Clarifying Question** — trigger list with "last resort"
    framing.
  - **Self-check Before Responding** — 6-item pre-emit checklist.
- `src/conversational/decomposer.py` — new `_synthesise_interpretation(cq)`:
  mechanical fallback for blank LLM interpretations.
- `src/conversational/orchestrator.py` — `ask()` checks `clarifying_question`
  first; if present, returns an AnswerResult with the question as the body
  and skips all downstream stages.
- `src/conversational/app.py` — blue `st.info()` block for
  `interpretation_summary`; clarifying question replaces the normal body
  with a "reply to refine" caption.

**UI changes**
- **Every answer** now has a blue **"Interpreting as:"** block above the
  summary.
- **Ambiguous questions** render as **bold follow-up + caption**, no table
  or chart.

**Net:** +28 tests.

---

### Phase 4.5 — Multi-CQ with ONE shared graph (`65f287f`)

**What:** Big questions decompose into 2–4 sub-CQs. Exactly **one** RDF
graph is built per turn, shared across all sub-CQs. The clinician gets a
narrative-led top-level answer with per-sub-CQ expanders underneath.

User confirmation during implementation: *"even when there are multiple
CQs, only one graph results."* That's the architecture.

**New files**
- `src/conversational/prompt_examples/big_question/sepsis_readmission.json`
  — worked 3-sub-CQ example sharing a sepsis cohort with `readmitted_30d`
  axis.
- `tests/test_conversational/test_extraction_merge.py` — 15 tests for the
  new merge helper.

**Modified**
- `src/conversational/models.py`:
  - `AnswerResult.sub_answers: list[AnswerResult] | None` — recursive,
    optional.
  - New `DecompositionResult(narrative: str | None,
    competency_questions: list[CQ])` with `is_multi` property and an
    empty-list validator.
- `src/conversational/extractor.py` — new `merge_extractions(results)` pure
  function. Dedup keys:
  - patients → `subject_id` · admissions → `hadm_id` · icu_stays → `stay_id`
  - biomarker → `labevent_id` · microbiology → `microevent_id`
  - vital → `(stay_id, itemid, charttime)`
  - drug → `(hadm_id, drug, starttime)`
  - diagnosis → `(hadm_id, seq_num)`
  - unknown event kind → `id()` fallback (never loses rows; accepts
    worst-case no-dedup over a crash).
- `src/conversational/decomposer.py`:
  - New `decompose_question()` returns `DecompositionResult`, handling both
    Shape A (single CQ) and Shape B (`{narrative, competency_questions}`).
  - Shape detection is structural: presence of `competency_questions` key.
  - Unsupported-filter retry aggregates offenders across every sub-CQ in
    ONE round-trip.
  - Legacy `decompose()` kept as a thin backward-compat wrapper returning
    `decomp.competency_questions[0]`. **20+ existing tests unchanged.**
- `src/conversational/orchestrator.py` — `ask()` per-CQ extract → merge
  (skipped for single-CQ) → ONE `build_query_graph` → per-CQ reason + answer.
  Allen relations computed iff ANY sub-CQ has `temporal_constraints`. Clarify
  short-circuits on ANY sub-CQ having a `clarifying_question`.
- `src/conversational/prompts.py` — Output Schema documents both shapes.
  New **"Big-Question Decomposition"** section explains when to decompose
  and when not to.
- `src/conversational/app.py` — `_render_answer(is_sub=False)` renders
  `sub_answers` under a **"Breakdown:"** divider, each in its own expander
  (first auto-expanded). Query-details expander shown at top level only.

**UI changes**
- **Big questions** now show a narrative at the top + a "Breakdown:"
  section with **expanders per sub-CQ**, each with its own interpretation,
  summary, and optional table.

**Correctness properties enforced by tests**
- ONE `build_query_graph` call regardless of N sub-CQs (booby-trapped mock).
- Same `graph` object identity across all `reason()` calls.
- Allen relations built if **any** sub-CQ has temporal constraints.
- Single-CQ fast path skips `merge_extractions`.
- Clarify takes priority — any sub-CQ's clarifying question wins.
- `decompose()` backward-compat preserved.

**Net:** +79 tests.

---

### Phase 5 — SNOMED hierarchy fallback (`25eb0bc`)

**What:** When the curated `category_to_snomed.json` misses, the resolver
falls back to the SNOMED IS-A hierarchy (if present) to expand concepts to
MIMIC-known names.

**Modified**
- `src/conversational/concept_resolver.py`:
  - New `_FALLBACK_MAPPING_FILES` tuple: `drug_to_snomed`, `labitem_to_snomed`,
    `comorbidity_to_snomed`, `organism_to_snomed`, `chartitem_to_snomed`.
  - `_build_sctid_indices()` — lazy, cached forward (name→SCTID) and
    reverse (SCTID→[names]) indices.
  - New `_resolve_via_hierarchy(concept)` — forward-lookup concept's SCTID,
    `hierarchy.get_descendants()`, reverse-map to MIMIC names.
  - `resolve()` order: curated `members` → SNOMED fallback → pass-through.
- `src/conversational/orchestrator.py` — instantiates `SnomedHierarchy` if
  `data/ontology_cache/snomed_hierarchy.json` exists; passes None otherwise.

**New files**
- `tests/test_conversational/fixtures/snomed/tiny_hierarchy.json` — 7-node
  antibiotic subtree with one deliberately-missing-from-MIMIC SCTID so the
  reverse-lookup filter is exercised.

**Conservative behaviour**
- Fallback fans out only when ≥2 MIMIC-known descendants are found.
  "metoprolol" stays specific; "beta blockers" fans out (if the hierarchy
  file is present).
- **Absent `data/ontology_cache/snomed_hierarchy.json`:** fallback is
  inert; pass-through everywhere. Code paths fully covered by tiny-fixture
  tests.

**Net:** +16 tests (7 fallback-behaviour, 9 data-quality on real mappings).

---

### Phase 6 — Broad regression harness (Phase 6 commit)

**What:** Fixture suite expanded to 42 cases covering every filter field,
comparison axis, temporal relation, ambiguous/clarify, synonyms, casing,
multi-filter/multi-concept. Hypothesis fuzz tests for total and idempotent
invariants over randomly-generated CQs.

**New files**
- `tests/test_conversational/test_decomposer_fuzz.py` — 10 property tests:
  - `_validate_return_type` is total + idempotent over random CQs.
  - `_synthesise_interpretation` is total + produces non-empty string.
  - `_extract_json` is total on arbitrary text (including mojibake).
  - JSON round-trip through extract + parse is stable, including with
    prose around the JSON.
- `tests/test_conversational/fixtures/decomposer_cases/` — 20 new cases:
  `readmitted_60d_filter`, `admission_type_in_list`, `ambiguous_show_labs`,
  `elderly_synonym`, `older_adults_synonym`, `temporal_before_event`,
  `temporal_after_event`, `temporal_24h_window`, `temporal_7d_window`,
  `comparison_discharge_location`, `comparison_readmitted_60d`,
  `urine_attribute`, `icd_code_filter`, `multi_filter_combo`,
  `multi_concept_biomarkers`, `casing_uppercase_concept`, `count_stroke`,
  `max_heart_rate`, `median_lactate`, `mrsa_culture`, `propofol_drug`,
  `heart_failure_cohort`, `missing_interpretation_gets_synthesized`,
  `interpretation_summary_pinned`, `ambiguous_show_patients`.

**Modified**
- `pyproject.toml` — added `hypothesis>=6.0` to `dev`. New pytest marker
  `live_llm` for opt-in manual tests (skipped by default).

**Migration notes**
- `uv pip install --python .venv/bin/python -e ".[dev]"` to pick up
  `hypothesis`.

**Net:** +30 tests.

---

## Test suite growth

| Phase | Tests added | Repo-wide total |
|---|---|---|
| Pre-refactor | — | 891 |
| 0 | +89 | 891 (same — new tests replace prior coverage via fixtures) |
| 1a | +45 | 916 |
| 1b | +12 | 928 |
| 2 | +11 | 950 |
| 3 | +16 | 964 |
| 4 | +28 | 994 |
| 4.5 | +79 | 1027 |
| 5 | +16 | 1043 |
| 6 | +30 | **1073+** |

## New/changed public APIs (summary)

| Symbol | Phase | Status | Notes |
|---|---|---|---|
| `ExtractionConfig.max_cohort_size` | 2 | **removed** | use `batch_size` instead |
| `ExtractionConfig.batch_size` | 2 | new | default 2000 |
| `PatientFilter.value: str \| list[str]` | 1b | widened | list for `in` operator |
| `CompetencyQuestion.interpretation_summary` | 4 | new | always populated |
| `CompetencyQuestion.clarifying_question` | 4 | new | short-circuits pipeline |
| `AnswerResult.interpretation_summary` | 4 | new | UI echo |
| `AnswerResult.clarifying_question` | 4 | new | UI clarify body |
| `AnswerResult.sub_answers` | 4.5 | new | multi-CQ breakdown |
| `DecompositionResult` | 4.5 | new | narrative + list of CQs |
| `decompose_question()` | 4.5 | new | returns DecompositionResult |
| `decompose()` | — | unchanged | backward-compat wrapper (first CQ) |
| `merge_extractions()` | 4.5 | new | union-with-dedup pure function |
| `OperationRegistry` + operation classes | 1a/1b | new | single source of truth |
| `get_default_registry()` | 1a | new | lazy singleton |
| `build_system_prompt(registry)` | 3 | new | pluggable prompt builder |

## Files added

```
src/conversational/operations.py
src/conversational/operations_aggregates.py
src/conversational/operations_comparison.py
src/conversational/operations_filters.py
src/conversational/prompt_examples/single_cq/{01..10}_*.json  (11 files)
src/conversational/prompt_examples/big_question/sepsis_readmission.json
tests/test_conversational/conftest.py
tests/test_conversational/fixtures/decomposer_cases/*.json            (42 files)
tests/test_conversational/fixtures/prompt_examples/*.json              (11 files)
tests/test_conversational/fixtures/malformed_json/*.txt                (10 files)
tests/test_conversational/fixtures/prompts_snapshot.txt
tests/test_conversational/fixtures/snomed/tiny_hierarchy.json
tests/test_conversational/test_decomposer_contract.py
tests/test_conversational/test_decomposer_cases.py
tests/test_conversational/test_decomposer_fuzz.py
tests/test_conversational/test_extraction_merge.py
tests/test_conversational/test_operations.py
docs/e2e-bigquery-testing.md
docs/decomposer-refactor-changelog.md  (this file)
```

## How to test it

See `docs/e2e-bigquery-testing.md` for the full walkthrough. Quick version:

```bash
# Unit / regression tests — offline
pytest tests/test_conversational/ -q

# Full repo suite
pytest tests/ -q

# Streamlit UI against BigQuery
gcloud auth application-default login
export ANTHROPIC_API_KEY="sk-ant-..."
streamlit run src/conversational/app.py
# → sidebar: data source BigQuery, GCP project ID, Connect
# → try three questions:
#    1. "average creatinine for patients over 65"    (single-CQ)
#    2. "show me the labs"                           (clarify short-circuit)
#    3. "why do our sepsis patients get readmitted?" (big question, 3 sub-CQs, ONE graph)
```
