# Phase 8d + Phase 9 — Mentor Demo Guide

> **What you'll demonstrate.** Two end-to-end capabilities just landed:
> *Phase 8d* — formal causal inference (T/S/X-learner via econml,
> stratified bootstrap CIs, propensity-overlap diagnostics) — and
> *Phase 9* — interpretable patient similarity (5-group contextual
> distance + hybrid-bucketed temporal Jaccard, with chat integration
> and similarity-narrowed causal cohorts).
>
> This guide is **Streamlit chat only** by design. Every prompt below
> is something the mentor can read, type into the chat box, and watch
> render. The demo runs ~45 minutes and exercises 16 distinct prompts
> across 14 code paths.

---

## Section 0 — Coverage map (chat-only)

Honest map of which surfaces are exercised by the prompts in this guide.

| Capability | Chat? | Code path |
|---|---|---|
| Causal `effect_of`, binary T-learner | ✅ | C1 |
| Causal `effect_of`, continuous T-learner | ✅ | C2 |
| Multi-arm causal (n ≥ 3 interventions) | ✅ | C3, C4 |
| S-learner registry dispatch | ✅ | C4 |
| X-learner + propensity-overlap diagnostic | ✅ | C5 |
| Multi-outcome (k ≥ 2) with identity aggregation | ✅ | C6 |
| ICD-10-PCS procedure intervention | ✅ | C7 |
| Survival outcome refusal | ✅ (loud refusal) | C8 |
| Non-identity aggregation refusal | ✅ (loud refusal) | C9 |
| Similarity, template anchor (no temporal) | ✅ | S1, S2, S5 |
| Similarity, `hadm_id` anchor (full temporal Jaccard) | ✅ | S3 |
| Similarity, `subject_id` anchor | ✅ | S4 |
| Custom `contextual_weights` override | ✅ | S2 |
| `min_similarity` threshold | ✅ | S5 |
| Similarity-narrowed causal, template anchor | ✅ | Capstone-1 |
| Similarity-narrowed causal, hadm anchor | ✅ | Capstone-2 |

**Known visual gap (called out so the mentor isn't surprised).** The
chat renders the flat 6-column similarity table but does *not* yet
render the rich Phase-9 explanation panels (per-bucket events, top
contributors / detractors, LOS-gap pill). The structured payloads ride
along on `SimilarityScore.contextual_explanation` and
`temporal_explanation` and the formatters at
`src/similarity/explanations.py` are ready — the Streamlit panel itself
is queued as Phase 9b.

**Phrasing caveat.** S3 and Capstone-2 both ask for similarity to a
specific admission via `hadm_id`. That's clinically unnatural phrasing;
a real clinician would describe the patient. We use it to demonstrate
that the second anchor mode works end-to-end, since the temporal-Jaccard
pipeline only runs with a real admission.

---

## Section 1 — Pre-demo setup (5 min)

### 1.1 Local prerequisites

Run each line and confirm the expected output.

```bash
# 1. DuckDB present (~16 GB)
ls -lh data/processed/mimiciv.duckdb

# 2. Mappings populated
ls data/mappings/ | grep -E '(drug|icd10cm|labitem|loinc)_to_snomed.json'

# 3. The new SIMILARITY branch + the existing causal/similarity tests pass
.venv/bin/python -m pytest \
  tests/test_conversational/test_orchestrator.py::TestSimilarityBranch \
  tests/test_similarity \
  tests/test_causal -q
```

If any of those three fail, fix before the mentor arrives — none of the
demo prompts will work otherwise.

### 1.2 Pick real `hadm_id` / `subject_id` ahead of time

Three prompts (S3, S4, Capstone-2) reference a specific admission or
subject. Pick them once before the demo so the chat doesn't stall on
"can't resolve". The query below picks ICU admissions with ≥ 5
prescription events — these have enough trajectory for the temporal-
Jaccard story to be interesting.

```bash
.venv/bin/python -c "
import duckdb
con = duckdb.connect('data/processed/mimiciv.duckdb')
# hadm candidates with ICU + meaningful prescription history
print('Candidate hadm_ids:')
for hadm, subj, n in con.execute('''
  SELECT a.hadm_id, a.subject_id, COUNT(p.drug) AS n_rx
  FROM admissions a
  JOIN icustays i ON a.hadm_id = i.hadm_id
  JOIN prescriptions p ON a.hadm_id = p.hadm_id
  GROUP BY a.hadm_id, a.subject_id
  HAVING COUNT(p.drug) > 5
  LIMIT 5
''').fetchall():
    print(f'  hadm={hadm}, subject={subj}, rx_events={n}')
print()
# subject candidates with multiple admissions (for S4 subject_id mode)
print('Candidate subject_ids (≥ 2 admissions):')
for subj, n_adm in con.execute('''
  SELECT subject_id, COUNT(*) AS n_adm
  FROM admissions
  GROUP BY subject_id
  HAVING COUNT(*) >= 2
  LIMIT 5
''').fetchall():
    print(f'  subject={subj}, admissions={n_adm}')
"
```

Substitute the chosen IDs into the three prompts marked
`<HADM_ID>` / `<SUBJECT_ID>` below.

### 1.3 Launch the chat

```bash
bash scripts/run_chat.sh
# OR equivalently
.venv/bin/streamlit run src/conversational/app.py
```

Browser opens at `http://localhost:8501`. The orchestrator opens
`data/processed/mimiciv.duckdb` per `config/settings.py:17`.

### 1.4 Pre-flight smoke prompt

Type into the chat:

> *"What is the average creatinine for ICU patients over 65?"*

This routes through the SQL fast-path (`_run_sql_fastpath` in
`orchestrator.py:272-295`) and returns a single-cell answer. If the
chat errors here, the rest of the demo will too — restart Streamlit
and check the API key.

---

## Section 2 — Arc A: Causal inference (~16 min, 9 prompts)

Every prompt lands in `ConversationalPipeline._run_causal`
(`src/conversational/orchestrator.py:297-342`) → `run_causal`
(`src/causal/run.py`). Each renders a 4-column table:
`(intervention, mu_point, mu_lower, mu_upper)`. The diagnostics — bootstrap
rep count, propensity-overlap score, audit-trail notes — attach to
`CausalEffectResult.diagnostics.notes`.

### C1 — vanilla binary T-learner

> *"Compare 30-day readmission between patients who received tPA vs
> those who didn't, among ischemic-stroke admissions."*

**Code path.** Decomposer → `QueryPlanner.classify` returns
`QueryPlan.CAUSAL` → `run_causal` (`src/causal/run.py:93+`) → cohort
builder (`src/causal/cohort.py:107`) → `TLearnerAdapter`
(`metalearners.py`, RandomForest/LogReg base for binary) →
`BootstrapRunner` (`estimators/base.py:120-307`) → flatten to table.

**What the mentor sees.** Two rows: `intervention=tPA`,
`intervention=no_tPA`, each with point estimate + 95% CI on the
readmission-probability scale.

### C2 — continuous outcome, T-learner

> *"What's the effect of warfarin vs no warfarin on ICU length of stay
> among atrial-fibrillation patients?"*

**Code path.** Same dispatch as C1; the outcome's `outcome_type` flips
to `continuous`, so `_base_learners.make_base_learner()` selects XGBoost
as the per-arm regressor instead of LogReg.

**What the mentor sees.** Same 4-column shape, but `mu_point` is on the
LOS scale (days). Bootstrap re-fits XGBoost B times — slower than C1.

### C3 — 3-arm T-learner

> *"Compare ceftriaxone, vancomycin, and piperacillin-tazobactam on
> 30-day readmission among sepsis admissions."*

**Code path.** N-ary intervention set (|I|=3). `run_causal` builds the
lexicographic-keyed τ matrix via the pairwise-contrast assembly; the
table flattens μ_c (3 entries). The pairwise τ matrix isn't yet surfaced
in the flat table (Phase 8i), but `CausalEffectResult.tau_matrix` is
populated.

**What the mentor sees.** Three rows in lexicographic intervention order
(`ceftriaxone`, `piperacillin-tazobactam`, `vancomycin`), per-arm sample
sizes in the diagnostics block.

### C4 — same N-ary scenario, S-learner

> *"Estimate that same 3-way comparison using an S-learner instead."*

**Code path.** Decomposer's `estimator_family` = `s_learner`. Registry
dispatch in `src/causal/estimators/registry.py` selects the S-learner
subclass: one model with treatment one-hot appended, shared capacity
across arms.

**What the mentor sees.** Three rows again, but the point estimates may
shift relative to C3. This is the moment to discuss bias / variance
trade-offs (S-learner shares capacity across arms; T-learner fits per
arm with no information sharing).

### C5 — X-learner with overlap diagnostic

> *"Estimate the effect of dexmedetomidine vs propofol on ICU length of
> stay using an X-learner, and tell me about overlap."*

**Code path.** X-learner branch (`metalearners.py::XLearnerAdapter`,
econml-backed cross-fitting + pseudo-outcome weighting) +
propensity-overlap computation in
`src/causal/estimators/_propensity.py`. `DiagnosticReport.overlap_score`
is populated; the chat summary surfaces the value.

**What the mentor sees.** A 2-row table plus a diagnostics line
("propensity overlap = 0.78 — sufficient" or similar). This is the most
research-flavored example: the system isn't just bootstrapping CIs, it's
flagging when the identification assumption (overlap / positivity) is at
risk.

### C6 — multi-outcome (k=2), identity aggregation [NEW]

> *"Compare warfarin vs no warfarin on both 30-day readmission and ICU
> length of stay among atrial-fibrillation patients."*

**Code path.** Decomposer emits `outcome_vector` with two `OutcomeSpec`
entries (`readmitted_30d` binary, `icu_los_days` continuous) +
`AggregationSpec(kind="identity")`. `run_causal` does *not* trigger the
aggregation refusal (kind="identity" is the always-supported case at
`src/causal/run.py:160-167`); both extractors from
`OutcomeRegistry.get_default_registry()` populate the cohort frame.

**What the mentor sees.** Per-arm μ for both outcomes. The Phase 8d
ranking key is the first outcome only — the diagnostics block notes
this honestly ("multi-outcome ranking reflects first outcome;
multi-outcome composition lands in 8f"). This is the multi-outcome
plumbing working without papering over its own limitations.

### C7 — ICD-10-PCS procedure intervention [NEW]

> *"Compare 30-day readmission between patients who received cardiac
> catheterization vs those who didn't."*

**Code path.** Decomposer emits `InterventionSpec(kind="procedure",
icd10pcs_code=...)` rather than the RxNorm-drug shape every other
example uses. `InterventionResolver` in `src/causal/interventions.py`
dispatches to its ICD-10-PCS branch — direct prefix match in the
`procedures_icd` table — instead of the drug → SNOMED → drug-name
expansion path that tPA / warfarin / antibiotics all take.

**What the mentor sees.** Same 4-column shape as C1, but the audit
trail in the diagnostics block names the procedure-table source rather
than the drug-name expansion. The mentor sees that the intervention
resolver isn't drug-only — Phase 8b's ontology grounding genuinely
covers all four code types (RxNorm, SNOMED, ICD-10-PCS, LOINC).

### C8 — survival guard rail (intentional refusal)

> *"What's the effect of tPA on time to readmission?"*

**Code path.** Decomposer marks `outcome_type="time_to_event"`;
`run_causal`'s pre-flight raises `SurvivalNotYetSupported`
(`src/causal/run.py:170-178`). The orchestrator catches the exception
and surfaces a plain-English error.

**What the mentor sees.** Plain-English error explaining that survival
lands in Phase 8g, with a pointer to the cohort builder (which already
assembles `(time, event)` columns correctly — only the estimator side
is missing). This is a *good* failure: the system refuses an
out-of-scope question loudly rather than silently mis-treating censored
data as a binary outcome.

### C9 — aggregation guard rail (intentional refusal)

> *"Compare aspirin vs no aspirin on a weighted sum of readmission and
> mortality."*

**Code path.** Decomposer emits `aggregation_spec=AggregationSpec(kind=
"weighted_sum", ...)`; `run_causal` raises `AggregationNotYetSupported`
(`src/causal/run.py:160-167`). Multi-outcome composition arrives in
Phase 8f.

**What the mentor sees.** Same loud-refusal pattern. Use this to
discuss what *should* happen (Holm–Bonferroni multi-outcome adjustment
or utility-weighted composite) and that the system is explicit about
the gap rather than papering over it.

---

## Section 3 — Arc B: Standalone similarity (~12 min, 5 prompts)

Every prompt lands in `ConversationalPipeline._run_similarity`
(`src/conversational/orchestrator.py:344-397`) → `run_similarity`
(`src/similarity/run.py:311`) and renders a 6-column ranked table:
`(rank, hadm_id, subject_id, combined, contextual, temporal)`.

Note: this dispatch branch was added today. The booby-trap test
`tests/test_conversational/test_orchestrator.py::TestSimilarityBranch`
guards it — if a future change accidentally bypasses the branch, the
test fails with `AssertionError` rather than silently routing to the
graph path.

### S1 — template anchor, contextual-only

> *"Find admissions similar to a 68-year-old woman with atrial
> fibrillation, CKD stage 3, and a history of embolic stroke. Rank the
> top 10."*

**Code path.** Decomposer maps the clinical description to
`SimilaritySpec.anchor_template={"age": 68, "gender_F": 1,
"snomed_group_I48": 1, "snomed_group_N18": 1, ...}`. The template branch
of `_resolve_anchor` (`src/similarity/run.py:240-245`) returns empty
`anchor_buckets`. `compute_temporal_similarity` therefore doesn't run;
`combine_scores` returns the contextual score verbatim
(`src/similarity/combined.py`).

**What the mentor sees.** 10 rows sorted descending by `combined`;
`temporal=None` for every row (the contextual-only fallback is the
expected behaviour for template anchors).

### S2 — severity-weighted template

> *"Show me the 10 admissions most similar to a 75-year-old man
> admitted for septic shock with acute kidney injury, weighting severity
> heavily."*

**Code path.** Decomposer fills
`contextual_weights={"severity": 0.6, "demographics": 0.10, ...}`.
`feature_groups.py` validates the override (must sum to 1.0 across
provided keys; missing groups get 0 weight) and passes the dict into
`compute_contextual_similarity`. The ranking shifts vs the default
weights — an empirical demonstration that the weighting machinery is
live, not boilerplate.

**What the mentor sees.** A 10-row ranked table with the severity group
dominating the ordering. The diagnostic-block phrasing
(`contextual_weights override applied`) confirms the override fired.

### S3 — `hadm_id` anchor, full temporal Jaccard [NEW]

> *"Show me the 10 admissions most similar to admission `<HADM_ID>`."*
> (Substitute the integer chosen in §1.2.)

**Code path.** Decomposer extracts the integer per the
`_SCOPE_INFERENCE` guidance + few-shot example
`06_similarity_to_hadm.json`. The `hadm_id` branch of `_resolve_anchor`
(`src/similarity/run.py:247-264`) fetches features + bucketed events for
the anchor admission, then `_fetch_admission_events` runs again per
candidate. `compute_temporal_similarity` runs over per-bucket Jaccards
(`src/similarity/temporal.py`); buckets come from
`assign_buckets` in ICU-day mode (`src/similarity/bucketing.py`).
`combine_scores` returns `0.5 · s_temp + 0.5 · s_ctx` (default α).

**What the mentor sees.** A 10-row ranked table where `temporal` is
populated for every row — the first time in this demo that the
temporal-Jaccard pipeline runs end-to-end. Compare ordering to S1: same
machinery, but cohort selection now reflects shared trajectories rather
than just covariates.

### S4 — `subject_id` anchor [NEW]

> *"Find patients similar to subject `<SUBJECT_ID>` across their
> admissions, top 10."* (Substitute the integer chosen in §1.2.)

**Code path.** The `anchor_subject_id` branch of `_resolve_anchor`
(`src/similarity/run.py:267-288`) lists all admissions for the subject
and picks the most recent as representative — that admission's features
+ buckets become the anchor. Downstream pipeline is identical to S3.

**What the mentor sees.** Same 6-column shape as S3 with the temporal
column populated. Surface the "most recent admission" choice when
walking the result so the mentor sees it's a deliberate design
decision, not a tiebreak — and a candidate question for follow-on work
(should it be the first admission? An aggregate? A user choice?).

### S5 — `min_similarity` threshold [NEW]

> *"Find patients similar to a 72-year-old woman with COPD exacerbation
> and pneumonia, but only show me ones with similarity above 0.7."*

**Code path.** Decomposer emits `SimilaritySpec.min_similarity=0.7`
alongside the template anchor. `run_similarity` filters the ranked list
post-sort, pre-`top_k` (`src/similarity/run.py:388-391`). Template
anchor → contextual-only fallback as in S1.

**What the mentor sees.** A potentially-shorter table than top_k=10
would have produced. If 0.7 is above the natural distribution this can
return very few rows (or zero) — useful talking point for "what does
0.7 similarity *mean*?" The answer: a weighted mean of five group
scores, not a probability or a metric-space distance — it's calibrated
relative to the group weights, not to any external scale.

---

## Section 4 — Arc C: Capstone — similarity-narrowed causal (~9 min)

The integration moment. Phase 9 isn't a side-show — it's a Phase 8d
*cohort source*. A causal CQ that carries `similarity_spec` stays on
`QueryPlan.CAUSAL`; `run_causal` calls `run_similarity` first, takes
the top-K hadm_ids, threads them as `cohort_hadm_ids` into the cohort
builder, and emits an audit note into `DiagnosticReport.notes`
(`src/causal/run.py:187-219`). The mentor sees the cohort provenance
verbatim in the diagnostics block.

### Capstone-1 — template-narrowed causal

> *"Among admissions similar to a 68-year-old woman with atrial
> fibrillation and CKD, what's the effect of GLP-1 exposure vs no
> exposure on 30-day readmission?"*

**Code path.** Decomposer emits both `scope="causal_effect"` AND
`similarity_spec={anchor_template: {...}, top_k: 30}`. Planner keeps
the CQ on `QueryPlan.CAUSAL`. `run_causal` detects the spec, calls
`run_similarity`, takes top-30 hadm_ids, narrows the cohort. Template
anchor → contextual-only similarity → cohort selection driven by
demographics + comorbidity + severity.

**What the mentor sees.** The familiar 4-column causal table —
`intervention=GLP1`, `intervention=no_GLP1` — but the diagnostics block
now contains a Phase-9 audit note like:

```
Phase 9 — cohort narrowed by similarity to template anchor
(age=68, snomed_group_I48=1, snomed_group_N18=1)
(top_k=30, n_pool=12483, n_returned=30).
```

A specific clinical profile drives cohort selection; that cohort flows
into the formal causal estimator; the estimate carries the provenance.

### Capstone-2 — `hadm_id`-narrowed causal [NEW]

> *"Among admissions similar to admission `<HADM_ID>`, compare alteplase
> (tPA) vs no tPA on 30-day readmission."* (Substitute the integer
> chosen in §1.2.)

**Code path.** Same dispatch as Capstone-1, but `similarity_spec` now
carries `anchor_hadm_id` rather than `anchor_template`. The narrowing
uses real-anchor mode with full temporal Jaccard (S3's pipeline). The
audit note references the real hadm anchor and carries `n_pool` +
`n_returned`.

**What the mentor sees.** Same 4-column shape as Capstone-1, but the
audit note now references the hadm_id and the cohort selection
mechanism is genuinely different: candidates ranked by *trajectory*
similarity rather than just *profile* similarity. Compare against
Capstone-1: same downstream estimator, two different cohort selection
mechanisms, the system is honest about which it used.

---

## Section 5 — Troubleshooting

| Symptom | Likely cause | Pointer |
|---|---|---|
| Bootstrap stalls on a small cohort | Tiny arm size hits LR class-imbalance per replicate | Pre-narrow the cohort, or ask the question with a broader filter |
| Decomposer drops the integer in S3 / Capstone-2 | `<HADM_ID>` typed without surrounding context | Few-shot examples `06_similarity_to_hadm.json` + `08_similarity_causal_narrowing.json` show the expected phrasing — keep "admission" as the noun |
| Decomposer can't resolve "similar to ..." | Anchor description too vague | Decomposer should emit `clarifying_question` (`src/conversational/models.py:194`); rephrase with at least one concrete clinical attribute |
| Streamlit shows raw JSON instead of a table | Renderer fallback when `data_table` shape is unexpected | Inspect `_render_answer` at `src/conversational/app.py:195+` |
| Template anchor returns `temporal=None` for all rows | Expected — no trajectory for a synthetic profile | Documented contextual-only fallback in `combine_scores` |
| Causal table shows `NaN` point estimates | Phase-8a stub fired (cohort empty / shape rejected) | Check `result.is_stub`; widen the cohort filter and re-ask |
| `run_similarity` is slow on the full DB | `_fetch_admission_events` issues per-candidate SQL | Mention as a known cost; live runs against the full pool take O(N × F) + O(N × B) — sub-second on ≤10K candidates, slower on the full MIMIC-IV cohort. Approximate-NN indexing is queued behind this demo. |

---

## Section 6 — 45-minute runbook

Concrete schedule. Adjust if the mentor interrupts with deep questions
on a single example — the negative tests (C8, C9) are the easiest to
drop if you run long, and S4 can be cut to S3 + S5 if needed.

| Time | Activity | Prompts |
|---|---|---|
| 0–5 | Setup + smoke check | §1.4 pre-flight smoke; verify pre-picked IDs |
| 5–10 | Causal: tightest path | C1 (binary tPA), C2 (continuous LOS) |
| 10–14 | Causal: multi-arm + learner switch | C3 (3-arm T-learner), C4 (same scenario, S-learner) |
| 14–18 | Causal: identification + intervention breadth | C5 (X-learner + overlap), C7 (ICD-PCS procedure) |
| 18–22 | Causal: multi-outcome + guard rails | C6 (multi-outcome), C8 (survival refusal), C9 (aggregation refusal) |
| 22–28 | Similarity: anchor breadth | S1 (template), S3 (hadm_id), S4 (subject_id) |
| 28–32 | Similarity: weights + threshold | S2 (severity-weighted), S5 (min_similarity) |
| 32–40 | Capstone | Capstone-1 (template-narrowed), Capstone-2 (hadm-narrowed) |
| 40–45 | Q&A buffer | open |

---

## Section 7 — Day-before verification

Run this end-to-end the day before. If any step fails, fix before the
mentor arrives.

1. Working tree clean after the demo-prep commit (`git status --short`
   shows only the GNN-track artefacts, which are unrelated).
2. `.venv/bin/python -m pytest
   tests/test_conversational/test_orchestrator.py::TestSimilarityBranch
   tests/test_similarity tests/test_causal -q` passes.
3. Streamlit launches; pre-flight smoke (§1.4) returns a single-cell
   answer.
4. Run **C1, C5, S1, S3, Capstone-2** end-to-end in chat. These five
   collectively touch: binary T-learner, X-learner + overlap diagnostic,
   template-anchor similarity, hadm-anchor similarity (full temporal
   Jaccard), and similarity-narrowed causal. If all five render
   correctly, every other prompt in the guide is a permutation of code
   paths these already exercise.
5. Confirm C8 + C9 surface plain-English errors rather than stack
   traces (the orchestrator's outer try/except in
   `orchestrator.py:244-251` catches; the user-facing text comes from
   the underlying exception's message).
6. If the chosen `<HADM_ID>` or `<SUBJECT_ID>` from §1.2 fails to
   resolve at demo time (ICU stay missing, prescriptions sparse), pick
   a different one from the candidate list — the §1.2 query returns
   five candidates per query specifically so you have alternates.

---

## Appendix — Files and code paths

* Orchestrator dispatch — `src/conversational/orchestrator.py:160-200`
  (causal + new similarity branch)
* Causal `_run_causal` wrapper — `src/conversational/orchestrator.py:297-342`
* Similarity `_run_similarity` wrapper — `src/conversational/orchestrator.py:344-397`
* Causal entry point — `src/causal/run.py:93+`
* Causal guard rails — `src/causal/run.py:160-178` (aggregation,
  survival)
* Phase-9 narrowing branch — `src/causal/run.py:187-219`
* Cohort builder — `src/causal/cohort.py:107`
* Estimator registry — `src/causal/estimators/registry.py`
* Base learner selection — `src/causal/estimators/_base_learners.py`
* Bootstrap runner — `src/causal/estimators/base.py:120-307`
* Propensity overlap — `src/causal/estimators/_propensity.py`
* Intervention resolver (RxNorm / SNOMED / ICD-10-PCS / LOINC) —
  `src/causal/interventions.py`
* Outcome extractors (registry of binary / continuous / time_to_event /
  diagnosis-within-horizon) — `src/causal/outcomes.py`
* Similarity entry point — `src/similarity/run.py:311`
* Anchor resolution (template / hadm / subject) — `src/similarity/run.py:230-288`
* Combined-score logic + template fallback — `src/similarity/combined.py`
* Contextual scoring (5-group Gower-like) — `src/similarity/contextual.py`
* Temporal scoring (bucketed weighted Jaccard) — `src/similarity/temporal.py`
* Hybrid bucketing (ICU-day / admission-relative) — `src/similarity/bucketing.py`
* Explanation formatters — `src/similarity/explanations.py`
* Decomposer scope-inference guidance — `src/conversational/prompts.py:231-261`
* Few-shot examples (similarity) —
  `tests/test_conversational/fixtures/prompt_examples/06_similarity_to_hadm.json`,
  `07_similarity_template_anchor.json`,
  `08_similarity_causal_narrowing.json`
* Booby-trap test for the new SIMILARITY branch —
  `tests/test_conversational/test_orchestrator.py::TestSimilarityBranch`

Spec docs you can reference if asked:

* `docs/phase8d-first-estimators.md` — the formal Phase-8d spec
* `docs/phase9-patient-similarity.md` — the formal Phase-9 spec
