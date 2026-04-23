# Phase 9: Patient Similarity Engine

Phase 9 delivers an interpretable, deterministic two-axis similarity
engine over MIMIC-IV admissions. Both a standalone chat query
("show me the 30 patients most similar to hadm 101") and a cohort-
narrowing directive on causal questions ("GLP-1 effect on
thromboembolism among patients similar to this one") are first-class
inputs.

- Plan file: `/Users/zacharyrothstein/.claude/plans/vivid-knitting-forest.md`
- Phase 9 commits: `69dbca2` (RED) → `26e030c` (scaffold) →
  `83330e1` (contextual) → `3448ac5` (temporal + bucketing) →
  `832b5b5` (combined + explanations + run_similarity) →
  `92345a9` (planner + CQ + causal) → `b61a78a` (decomposer + prompt) →
  this commit (docs)

## Overview

Before Phase 9 the LLM decided what "similar" meant on a per-query
basis. That was unprincipled and non-reproducible. Phase 9 replaces
the LLM's ad-hoc judgement with a deterministic scoring pipeline
that a clinician can audit — every rank decision has a traceable
contribution from specific features (contextual) and specific
events (temporal).

```
CompetencyQuestion (scope="patient_similarity" | "causal_effect" + similarity_spec)
   │
   │  src/conversational/planner.py::QueryPlanner.classify   (line 76+)
   ▼
QueryPlan.SIMILARITY                   QueryPlan.CAUSAL
   │                                      │
   ▼                                      ▼
run_similarity                       run_causal
  src/similarity/run.py                src/causal/run.py
   │                                      │
   │  _resolve_anchor  (run.py:154)       │  similarity_spec branch (run.py:~150)
   │  (real patient or template)          │  → run_similarity → top-K hadm_ids
   ▼                                      │  → note into DiagnosticReport.notes
 candidate pool                           ▼
   │                              build_cohort_frame (narrowed)
   │  contextual + temporal scoring       │
   ▼                                      ▼
compute_contextual_similarity        BootstrapRunner + estimators
  src/similarity/contextual.py           (8d layer — unchanged)
compute_temporal_similarity
  src/similarity/temporal.py
assign_buckets
  src/similarity/bucketing.py
   │
   ▼
combine_scores  (α · s_temp + (1-α) · s_ctx)
  src/similarity/combined.py
   │
   ▼
SimilarityResult (ranked, with ContextualExplanation +
 TemporalExplanation per candidate)
  src/similarity/models.py
```

## The four locked decisions

All four are recorded in the plan file. Implementation pointers follow.

### 1. Anchor = real patient OR template

`SimilaritySpec` accepts exactly one of `anchor_hadm_id`,
`anchor_subject_id`, or `anchor_template` (covariate dict). The
schema validator at `src/similarity/models.py::_validate_spec`
enforces the "exactly one" invariant. Real-patient anchors produce
a full contextual + temporal score. Template anchors skip temporal
— `run_similarity._resolve_anchor` returns empty anchor_buckets,
`compute_temporal_similarity` sets `temporal_available=False` on
every candidate, and `combine_scores` degrades to contextual-only
regardless of the caller's `temporal_weight`.

### 2. Hybrid temporal bucketing

Events are bucketed by ICU calendar day for ICU admissions, by
admission-relative windows for floor-only admissions.
`src/similarity/bucketing.py::get_bucketing_mode` dispatches on
whether any ICU stay is present; `assign_buckets` emits the
per-mode labels:

| Mode | Labels | When |
|---|---|---|
| `icu_day` | `icu_day_N` (N = 0-indexed day offset from `intime`); `pre_icu_0_24`, `pre_icu_24_plus` for pre-ICU events | Any ICU stay present |
| `admission_relative` | `h_0_24`, `h_24_48`, then `day_N` for N ≥ 2 | Floor-only admissions |

Post-ICU events are deliberately skipped in Phase 9 scope. The
bucket labels preserve insertion order so
`compute_temporal_similarity` can iterate time-correctly without
parsing the label strings.

### 3. Both cohort-narrowing AND standalone CQ scope

The same `SimilaritySpec` is reused:

- **Standalone similarity CQ** — `scope="patient_similarity"`, no
  `intervention_set`. Planner routes to `QueryPlan.SIMILARITY`
  (`src/conversational/planner.py:60`), orchestrator branch
  TBD in the Streamlit follow-on. `run_similarity` returns a
  ranked `SimilarityResult` directly.
- **Causal narrowing** — `scope="causal_effect"` AND `similarity_spec`
  set. Planner keeps the CQ on `QueryPlan.CAUSAL`. `run_causal`
  detects the spec at `src/causal/run.py:~175`, runs
  `run_similarity(spec, backend)`, takes the top-K hadm_ids, and
  passes them as `cohort_hadm_ids` to `build_cohort_frame`. The
  narrowing details (anchor description, top_k, n_pool,
  n_returned) land in `DiagnosticReport.notes` so investigators
  can audit cohort selection.

### 4. Full chat/LLM integration (Streamlit deferred)

- **Decomposer**: the LLM prompt at
  `src/conversational/prompts.py::_SCOPE_INFERENCE` (lines 231+)
  now documents the `patient_similarity` scope + the three anchor
  alternatives + the similarity + causal combination rule. Three
  few-shot examples live under
  `src/conversational/prompt_examples/single_cq/` as
  `06_similarity_to_hadm.json`, `07_similarity_template_anchor.json`,
  `08_similarity_causal_narrowing.json`.
- **CompetencyQuestion schema**: `similarity_spec` is a forward-ref
  annotation (avoids a circular import between
  `conversational.models` and `similarity.models`); the rebuild at
  the bottom of `conversational.models` ensures standalone imports
  still work.
- **Streamlit UI**: explicitly deferred to a follow-on commit. The
  backend (`SimilarityResult` + `format_similarity_text` helpers at
  `src/similarity/explanations.py`) is ready; the `app.py` panel +
  cohort-narrowed-by-similarity card can be added without touching
  any of the Phase 9 compute layer.

## Contextual similarity (5-group Gower-like)

`src/similarity/contextual.py::compute_contextual_similarity` scores
each candidate across five groups and computes a weighted mean as
the overall contextual score. Default group weights:

| Group | Default weight | Features | Distance |
|---|---|---|---|
| demographics | 0.15 | age, gender one-hot, admission_type one-hot | age `|Δ|/50` clipped, flag match |
| comorbidity_burden | 0.35 | `charlson_index` + 17 Charlson flags | 50/50 split: normalized `|Δ|` on index + weighted Jaccard on flags (Charlson weights 1/2/3/6) |
| comorbidity_set | 0.25 | all `snomed_group_*` columns | plain Jaccard |
| severity | 0.15 | `creatinine_max`, `sodium_mean`, `platelet_min`, `icu_los_hours`, `admission_type_EMERGENCY` | `|Δ|/scale` with fixed clinical scales (3.0 / 20.0 / 100.0 / 72.0) |
| social | 0.10 | `language_barrier`, `is_neuro_service` | flag match |

Weights sum to 1.0. Callers override via
`SimilaritySpec.contextual_weights` — the caller-supplied dict is
validated to sum to 1.0 across provided keys; missing groups get 0
weight (caller explicitly excluded).

**Per-feature contributions** are computed as
`weight × (2 × feature_similarity − 1)` so a perfect match produces
+weight, a perfect mismatch produces −weight, and the neutral
midpoint produces 0. `top_contributors` / `top_detractors` are the
top-5 of each sign — they drive the chat explanation.

## Temporal similarity (bucketed weighted Jaccard)

`src/similarity/temporal.py::compute_temporal_similarity` runs
per-bucket Jaccard with empty-set conventions:

- both buckets empty → 1.0 (trivially aligned)
- one bucket empty → 0.0 (no overlap possible)
- else → `|A ∩ C| / |A ∪ C|`

Aggregated as a decay-weighted mean: `w_b = decay^bucket_index`.
Default `decay=0.9` gives earlier buckets ~10% more weight per
position (matches the clinical intuition that first-hour decisions
signal more than late-stay routine care). `decay=1.0` yields
equal-weighted per-bucket scoring.

**Explanation partition**: for every bucket the implementation emits
`(bucket_label, event_code)` tuples into three lists:

- `shared_events` — events in both sides' bucket at this position
- `anchor_only` — events only in the anchor's bucket
- `candidate_only` — events only in the candidate's bucket

The chat-layer formatter
(`format_temporal_text` at `src/similarity/explanations.py`) surfaces
the first 5 shared + first 3 candidate-only events by default.

**LOS gap**: `los_gap_days = |n_buckets_anchor - n_buckets_candidate|`
is reported separately so the UI can flag mismatched-length stays
(the per-bucket Jaccard already penalises the mismatch via
one-side-empty buckets; the LOS gap is purely informational).

## Combined similarity

`src/similarity/combined.py::combine_scores` returns
`α · s_temp + (1 − α) · s_ctx` clipped to `[0, 1]`, where `α =
temporal_weight` from the spec (default 0.5). Template-anchor
fallback: if `temporal` is `None` OR `temporal.temporal_available`
is `False`, returns the contextual score verbatim regardless of
`temporal_weight`.

## What Phase 9 does *not* do

These land in follow-on commits:

- **Streamlit UI panel** — the structured `SimilarityResult` + the
  `format_similarity_text` / `format_contextual_text` /
  `format_temporal_text` helpers are ready; the `app.py`
  rendering is the missing piece.
- **Orchestrator branch for `QueryPlan.SIMILARITY`** — the planner
  routes correctly but the orchestrator's dispatch needs a
  `elif plan == QueryPlan.SIMILARITY:` clause to call
  `run_similarity` end-to-end. Backend + models + CQ schema are
  already in place.
- **Full comorbidity extraction for real-patient anchors** —
  `run.py::_fetch_admission_features` zero-fills Charlson + SNOMED
  flags in Phase 9 scope. Upgrading to the full
  `build_feature_matrix` from `src/feature_extraction/feature_builder.py:175`
  is a drop-in replacement (same column names).
- **Candidate-filter plumbing** — `SimilaritySpec.candidate_filters`
  is carried on the spec but not yet honored by `run_similarity`;
  narrows the pool via the existing `PatientFilter` compilation
  stack when wired.
- **Approximate-NN indexing** (FAISS / Annoy) — only needed if
  production pools exceed ~50K admissions. Current O(N × F) + O(N × B)
  is sub-second on ≤10K candidates.

## Example: end-to-end research query

Query: *"Among patients similar to hadm 101, does tPA reduce 30-day
readmission?"*

1. **Decomposer** recognises the "similar to X" + intervention
   pattern (see `_SCOPE_INFERENCE` guidance and example 08). Emits:
   ```json
   {
     "scope": "causal_effect",
     "similarity_spec": {"anchor_hadm_id": 101, "top_k": 30},
     "intervention_set": [
       {"label": "tPA", "kind": "drug", "rxnorm_ingredient": "8410"},
       {"label": "no_tPA", "kind": "drug", "rxnorm_ingredient": "8410", "is_control": true}
     ],
     "outcome_vector": [{"name": "readmitted_30d", "outcome_type": "binary", "extractor_key": "readmitted_30d"}],
     "aggregation_spec": {"kind": "identity"}
   }
   ```

2. **Planner** routes to `QueryPlan.CAUSAL` (causal takes priority
   over similarity when both are present).

3. **`run_causal`** sees `cq.similarity_spec` is set. It calls
   `run_similarity(spec, backend)` which:
   - resolves anchor hadm 101 → fetches features + bucketed events
   - scores every other admission in the DB via contextual + temporal
   - ranks, applies top_k=30, returns `SimilarityResult`

4. **`run_causal`** extracts the top-30 `hadm_id`s from
   `sim_result.scores`, passes them as `cohort_hadm_ids` to
   `build_cohort_frame`, prepends a narrowing note to
   `DiagnosticReport.notes`:
   > "Phase 9 — cohort narrowed by similarity to hadm_id=101
   > (subject 1, 65yo M) (top_k=30, n_pool=~300000, n_returned=30)."

5. **Rest of the causal pipeline** (T-learner + bootstrap + CI)
   runs unchanged on the 30-admission narrowed cohort.

6. **Response** surfaces both the causal point estimate + bootstrap
   CI AND the similarity explanation (per-group contextual
   breakdown, shared events across ICU days). A reviewer can see
   both "the effect is X" and "these are the 30 patients the
   effect was measured across."

## Test matrix

12 new test files across three directories.

| File | Covers |
|---|---|
| `tests/test_similarity/test_spec_schema.py` | SimilaritySpec exactly-one-anchor + weight bounds + explanation shapes (17 tests) |
| `tests/test_similarity/test_contextual.py` | 5-group distance; programmed ranking 2001 > 2003 > 2002 > 2004; weight override; contributors/detractors signed correctly (9 tests) |
| `tests/test_similarity/test_temporal.py` | Identical → 1.0, disjoint → 0.0, partial intermediate; decay ordering; explanation partitioning; LOS gap; template anchor sets `temporal_available=False` (11 tests) |
| `tests/test_similarity/test_bucketing.py` | Mode dispatch; ICU-day vs admission-relative labels; pre-ICU bucket; empty inputs (9 tests) |
| `tests/test_similarity/test_combined.py` | α ∈ {0, 0.5, 1} semantics; template-anchor fallback; output clipped to [0,1] (6 tests) |
| `tests/test_similarity/test_explanations.py` | Plain-text formatters mention groups / shared events / LOS gap / overall (7 tests) |
| `tests/test_similarity/test_run_similarity_end_to_end.py` | Full pipeline against 6-admission DB: SimilarityResult shape, anchor exclusion, top_k cap, descending sort, template anchor, provenance (7 tests) |
| `tests/test_conversational/test_similarity_planner.py` | scope=patient_similarity → SIMILARITY; causal + similarity_spec → CAUSAL (3 tests) |
| `tests/test_causal/test_run_causal_similarity_narrowing.py` | Audit note on pre-built cohort; note carries anchor + top_k; live narrowing note carries n_pool + n_returned (3 tests) |
| `tests/test_conversational/test_decomposer_similarity_examples.py` | Anchor_hadm_id emission; template anchor emission; causal + similarity_spec combined (3 tests) |

Full suite counts:

| | passed | skipped | xfailed |
|---|---|---|---|
| Before Phase 9 | 1343 | 3 | 2 |
| After Phase 9 | 1424 | 3 | 2 |

Net +81 passed, zero regressions, two xfails unchanged (both waiting
on later Phase 8 sub-phases).

## Running the suite

```
# Phase 9 new tests
.venv/bin/python -m pytest tests/test_similarity -v

# Integration — planner + causal narrowing + decomposer
.venv/bin/python -m pytest \
  tests/test_conversational/test_similarity_planner.py \
  tests/test_conversational/test_decomposer_similarity_examples.py \
  tests/test_causal/test_run_causal_similarity_narrowing.py -v

# Full repo
.venv/bin/python -m pytest tests/
```
